"""Card D8a — the diagnostic data model + recorder.

The bars this file holds (one test class each):

1. TWO CLOCKS on every record, and latency measured in WALL time from the frame's ARRIVAL.
2. TracePolicy bounds retention WITHOUT ever weakening the counters.
3. Recording never blocks capture, and a dropped record is NEVER silent.
4. The funnel is derived ONCE — including evidence-window UNION seconds (not count x interval)
   and the savings identity (which is None, not fabricated, when there are no judge calls).
5. It works from a direct ``perceive_judged()`` driver, not only from the runtime worker.
"""

import asyncio
import json
import unittest

from percept.adapters import Box, Goal, Observation, Verdict
from percept.agent import Percept
from percept.backends.fake import FakeVision
from percept.config import GateSettings, Settings
from percept.observability import TraceEvent
from percept.tracing import (
    ADMISSION_STAGE,
    OBSERVATION_STAGE,
    STAGE_GATE_CLASSIFIED,
    STAGE_INCIDENT_CREATED,
    STAGE_JUDGE_FINISHED,
    STAGE_JUDGE_STARTED,
    STAGE_TRACE_DROPPED,
    BaselineAssumption,
    ComponentIdentity,
    PlanIdentity,
    RunTraceManifest,
    TracePolicy,
    TraceRecorder,
    TraceSummary,
    load_trace,
    union_seconds,
)


class FakeClock:
    """A wall clock we control, so latency assertions are exact rather than flaky."""

    def __init__(self, start: float = 1000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, dt: float) -> None:
        self.now += dt


def manifest(**kw) -> RunTraceManifest:
    kw.setdefault("policy", TracePolicy(level="full"))
    return RunTraceManifest.create(wall_clock=lambda: 1000.0, run_id="run-test", **kw)


def recorder(clock=None, **kw) -> TraceRecorder:
    m = kw.pop("manifest", None) or manifest(**{k: kw.pop(k) for k in list(kw) if k in
                                                ("policy", "baseline", "evidence_window_s", "source")})
    return TraceRecorder(m, wall_clock=clock or FakeClock(), **kw)


# ---------------------------------------------------------------------------
# 1. Two clocks
# ---------------------------------------------------------------------------
class TestTwoClocks(unittest.TestCase):
    def test_every_record_carries_both_clocks(self):
        clock = FakeClock()
        rec = recorder(clock)
        r = rec.record(OBSERVATION_STAGE, source_ts=26.2, escalate=True)
        self.assertEqual(r.source_ts, 26.2)         # MEDIA time — what an incident is reported at
        self.assertEqual(r.observed_wall_ts, 1000.0)
        self.assertEqual(r.emitted_wall_ts, 1000.0)

    def test_source_time_and_wall_time_are_independent(self):
        """A file replay runs source time far faster than wall time; both must stay truthful."""
        clock = FakeClock()
        rec = recorder(clock)
        for i in range(5):
            clock.advance(0.01)          # 10ms of wall per frame
            rec.record(OBSERVATION_STAGE, source_ts=float(i))  # 1s of source per frame
        self.assertAlmostEqual(rec.watch_duration_s(), 4.0)     # SOURCE span, not wall span
        self.assertAlmostEqual(rec.records[-1].emitted_wall_ts - rec.records[0].emitted_wall_ts, 0.04, places=6)

    def test_latency_is_measured_from_the_frames_arrival(self):
        """A downstream stage must measure from when the runtime SAW the frame, not from itself —
        otherwise a 900ms judge shows as 0ms of latency."""
        clock = FakeClock()
        rec = recorder(clock)
        rec.record(OBSERVATION_STAGE, trace_id="camera@26.2", source_ts=26.2)
        clock.advance(0.9)               # the judge call
        gate = rec.record(STAGE_GATE_CLASSIFIED, trace_id="camera@26.2", source_ts=26.2,
                          truth="yes", action="fire")
        self.assertAlmostEqual(gate.latency_s, 0.9, places=6)
        self.assertEqual(gate.source_ts, 26.2)   # the incident's event time is UNCHANGED by latency

    def test_explicit_clocks_win_over_defaults(self):
        rec = recorder()
        r = rec.record(STAGE_JUDGE_FINISHED, source_ts=3.0, observed_wall_ts=10.0, emitted_wall_ts=12.5)
        self.assertEqual(r.latency_s, 2.5)

    def test_latency_is_none_when_a_clock_is_missing(self):
        from percept.tracing import TraceRecord

        self.assertIsNone(TraceRecord(seq=0, trace_id="t", stage="x", observed_wall_ts=1.0).latency_s)


# ---------------------------------------------------------------------------
# 2. TracePolicy
# ---------------------------------------------------------------------------
class TestTracePolicy(unittest.TestCase):
    def test_levels_retain_progressively_more(self):
        summary = TracePolicy(level="summary")
        decisions = TracePolicy(level="decisions")
        full = TracePolicy(level="full")
        self.assertFalse(summary.retains(OBSERVATION_STAGE))
        self.assertFalse(summary.retains(STAGE_GATE_CLASSIFIED))
        self.assertTrue(summary.retains(STAGE_INCIDENT_CREATED))
        self.assertFalse(decisions.retains(OBSERVATION_STAGE))   # per-frame is NOT decision level
        self.assertTrue(decisions.retains(STAGE_GATE_CLASSIFIED))
        self.assertTrue(full.retains(OBSERVATION_STAGE))

    def test_invalid_policies_are_rejected(self):
        with self.assertRaises(ValueError):
            TracePolicy(level="verbose")  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            TracePolicy(max_bytes=0)
        with self.assertRaises(ValueError):
            TracePolicy(max_duration_s=0)

    def test_summary_level_keeps_counters_exact(self):
        """THE bar: retention is a STORAGE decision, never a measurement one."""
        rec = recorder(policy=TracePolicy(level="summary", incident_window_s=0.0))
        for i in range(50):
            rec.record(OBSERVATION_STAGE, trace_id=f"c@{i}", source_ts=float(i), escalate=i % 5 == 0)
            if i % 5 == 0:
                rec.record(ADMISSION_STAGE, trace_id=f"c@{i}", source_ts=float(i))
                rec.record(STAGE_JUDGE_STARTED, trace_id=f"c@{i}", source_ts=float(i), goal_id="g1")
                rec.record(STAGE_JUDGE_FINISHED, trace_id=f"c@{i}", source_ts=float(i), goal_id="g1")
        s = rec.summary()
        self.assertEqual(s.observed, 50)          # counted...
        self.assertEqual(s.admitted, 10)
        self.assertEqual(s.judge_completed, 10)
        self.assertEqual(len(rec.records), 0)     # ...but nothing retained at summary level
        # filtering is not LOSS: it must not be reported as a dropped (incomplete-trace) record
        self.assertEqual(s.records_dropped, 0)
        self.assertGreater(rec.drop_reasons["policy_level"], 0)

    def test_incident_window_retains_full_detail_around_an_incident(self):
        rec = recorder(policy=TracePolicy(level="summary", incident_window_s=2.0))
        rec.record(OBSERVATION_STAGE, trace_id="c@1", source_ts=1.0)          # far from any incident
        rec.record(STAGE_INCIDENT_CREATED, trace_id="c@10", source_ts=10.0, incident_id="i1")
        rec.record(OBSERVATION_STAGE, trace_id="c@11", source_ts=11.0)        # inside the window
        rec.record(OBSERVATION_STAGE, trace_id="c@20", source_ts=20.0)        # outside again
        kept = [(r.stage, r.source_ts) for r in rec.records]
        self.assertIn((STAGE_INCIDENT_CREATED, 10.0), kept)
        self.assertIn((OBSERVATION_STAGE, 11.0), kept)
        self.assertNotIn((OBSERVATION_STAGE, 1.0), kept)
        self.assertNotIn((OBSERVATION_STAGE, 20.0), kept)

    def test_max_duration_bounds_retention_by_source_time(self):
        rec = recorder(policy=TracePolicy(level="full", max_duration_s=5.0))
        for i in range(12):
            rec.record(OBSERVATION_STAGE, trace_id=f"c@{i}", source_ts=float(i))
        self.assertEqual(rec.summary().observed, 12)                     # all counted
        self.assertTrue(all(r.source_ts <= 5.0 for r in rec.records if r.stage == OBSERVATION_STAGE))

    def test_media_is_stripped_unless_include_media(self):
        rec = recorder(policy=TracePolicy(level="full"))
        r = rec.record(ADMISSION_STAGE, source_ts=1.0, frame="<bytes>", roi=True)
        self.assertNotIn("frame", r.data)
        self.assertIn("roi", r.data)
        rec2 = recorder(policy=TracePolicy(level="full", include_media=True))
        self.assertIn("frame", rec2.record(ADMISSION_STAGE, source_ts=1.0, frame="<bytes>").data)

    def test_named_postures(self):
        self.assertEqual(TracePolicy.production().level, "summary")
        self.assertEqual(TracePolicy.evaluation().level, "decisions")
        self.assertEqual(TracePolicy.diagnostic().level, "full")
        self.assertIsNotNone(TracePolicy.diagnostic().max_duration_s)  # per-frame MUST be bounded


# ---------------------------------------------------------------------------
# 3. Never block, never lose silently
# ---------------------------------------------------------------------------
class TestDropsAreLoud(unittest.TestCase):
    def test_ring_overflow_surfaces_a_trace_dropped_record(self):
        rec = recorder(max_records=10, policy=TracePolicy(level="full"))
        for i in range(100):
            rec.record(OBSERVATION_STAGE, trace_id=f"c@{i}", source_ts=float(i))
        self.assertGreater(rec.lost_records, 0)
        markers = [r for r in rec.records if r.stage == STAGE_TRACE_DROPPED]
        self.assertEqual(len(markers), 1, "the loss must be visible in the retained stream itself")
        self.assertEqual(markers[0].data["dropped"], rec.lost_records)
        self.assertLessEqual(len(rec.records), 10)
        # and the FUNNEL is untouched by the loss
        self.assertEqual(rec.summary().observed, 100)
        self.assertEqual(rec.summary().records_dropped, rec.lost_records)

    def test_byte_bound_also_drops_loudly(self):
        rec = recorder(policy=TracePolicy(level="full", max_bytes=600))
        for i in range(200):
            rec.record(OBSERVATION_STAGE, trace_id=f"c@{i}", source_ts=float(i))
        self.assertGreater(rec.lost_records, 0)
        self.assertIn("max_bytes", rec.drop_reasons)
        self.assertTrue(any(r.stage == STAGE_TRACE_DROPPED for r in rec.records))

    def test_a_broken_tracer_payload_never_raises_into_capture(self):
        rec = recorder()
        cb = rec.tracer()

        class Exploding:
            stage = "edge.decision"
            trace_id = "c@1"
            t = 1.0
            goal_id = None

            @property
            def data(self):  # a tracer that blows up mid-ingest
                raise RuntimeError("boom")

        cb(Exploding())                       # must not raise
        self.assertEqual(rec.drop_reasons.get("recorder_error"), 1)

    def test_recording_does_not_await_or_do_io(self):
        """Ingest is synchronous by construction: calling it from inside a running event loop must
        not yield control (that is what would let a slow recorder stall capture)."""
        rec = recorder()

        async def main():
            for i in range(500):
                rec.record(OBSERVATION_STAGE, trace_id=f"c@{i}", source_ts=float(i))
            return rec.summary().observed

        self.assertEqual(asyncio.run(main()), 500)


# ---------------------------------------------------------------------------
# 4. The funnel + savings
# ---------------------------------------------------------------------------
class TestFunnelDerivation(unittest.TestCase):
    def _run(self, *, baseline=None, frames=20, admit_every=5, usage=None):
        rec = recorder(policy=TracePolicy(level="full"),
                       baseline=baseline or BaselineAssumption(),
                       evidence_window_s=0.0)
        for i in range(frames):
            tid = f"camera@{i}"
            admitted = i % admit_every == 0
            rec.record(OBSERVATION_STAGE, trace_id=tid, source_ts=float(i), escalate=admitted)
            if not admitted:
                continue
            rec.record(ADMISSION_STAGE, trace_id=tid, source_ts=float(i),
                       window_start_s=float(i) - 2.0, window_end_s=float(i))
            rec.record(STAGE_JUDGE_STARTED, trace_id=tid, source_ts=float(i), goal_id="g1")
            rec.record(STAGE_JUDGE_FINISHED, trace_id=tid, source_ts=float(i), goal_id="g1",
                       satisfied=True, confidence=0.9)
            if usage:
                rec.record_usage(trace_id=tid, source_ts=float(i), **usage)
            rec.record(STAGE_GATE_CLASSIFIED, trace_id=tid, source_ts=float(i), goal_id="g1",
                       truth="yes", action="fire" if i == 0 else "silent")
        rec.record(STAGE_INCIDENT_CREATED, trace_id="camera@0", source_ts=0.0, incident_id="i1")
        rec.close()
        return rec

    def test_funnel_counts(self):
        s = self._run().summary()
        self.assertEqual(s.observed, 20)
        self.assertEqual(s.admitted, 4)
        self.assertEqual(s.judge_started, 4)
        self.assertEqual(s.judge_completed, 4)
        self.assertEqual(s.gate_outcomes["fire"], 1)
        self.assertEqual(s.gate_outcomes["silent"], 3)
        self.assertEqual(s.incidents, 1)
        self.assertAlmostEqual(s.watch_duration_s, 19.0)

    def test_admitted_is_a_union_not_count_times_interval(self):
        """4 admissions x a 2s window is 8s only if the windows are disjoint. Here they overlap the
        source span partially — the union is what the number must report."""
        s = self._run(admit_every=1, frames=5).summary()
        self.assertEqual(s.admitted, 5)
        # windows [-2,0],[-1,1],[0,2],[1,3],[2,4] -> union [-2,4] = 6s, NOT 5 x 2 = 10s
        self.assertAlmostEqual(s.admitted_evidence_union_s, 6.0)

    def test_union_seconds_merges_and_ignores_gaps(self):
        self.assertAlmostEqual(union_seconds([(0, 2), (1, 3), (10, 11)]), 4.0)
        self.assertAlmostEqual(union_seconds([]), 0.0)
        self.assertAlmostEqual(union_seconds([(5, 5)]), 0.0)

    def test_zero_judge_calls_means_savings_is_none_not_a_number(self):
        rec = recorder(policy=TracePolicy(level="full"),
                       baseline=BaselineAssumption.every_observation(100))
        for i in range(100):
            rec.record(OBSERVATION_STAGE, trace_id=f"c@{i}", source_ts=float(i))
        s = rec.summary()
        self.assertEqual(s.judge_completed, 0)
        self.assertEqual(s.estimated_baseline_calls, 100)
        self.assertEqual(s.estimated_saved_calls, 100)
        self.assertIsNone(s.estimated_saved_tokens)   # NEVER fabricated
        self.assertEqual(s.savings_text(), "—")

    def test_savings_uses_mean_input_tokens_of_actual_calls(self):
        rec = self._run(frames=20, admit_every=5,
                        baseline=BaselineAssumption.every_observation(20),
                        usage={"input_tokens": 1000.0, "output_tokens": 50.0, "cost_usd": 0.01})
        s = rec.summary()
        self.assertEqual(s.judge_completed, 4)
        self.assertEqual(s.usage_reported_calls, 4)
        self.assertEqual(s.mean_input_tokens_per_call, 1000.0)
        self.assertEqual(s.estimated_baseline_calls, 20)
        self.assertEqual(s.estimated_saved_calls, 16)          # 20 - 4
        self.assertEqual(s.estimated_saved_tokens, 16 * 1000.0)
        self.assertAlmostEqual(s.cost_usd, 0.04)
        self.assertIn("Baseline", s.savings_text())            # the assumption is CARRIED, always

    def test_no_declared_baseline_means_no_savings_at_all(self):
        s = self._run(usage={"input_tokens": 1000.0}).summary()
        self.assertFalse(s.baseline.declared)
        self.assertIsNone(s.estimated_baseline_calls)
        self.assertIsNone(s.estimated_saved_calls)
        self.assertIsNone(s.estimated_saved_tokens)

    def test_savings_never_goes_negative(self):
        rec = self._run(frames=10, admit_every=1,
                        baseline=BaselineAssumption(kind="fixed", opportunities=2, description="two"),
                        usage={"input_tokens": 100.0})
        s = rec.summary()
        self.assertEqual(s.judge_completed, 10)
        self.assertEqual(s.estimated_saved_calls, 0)           # max(0, 2 - 10)
        self.assertEqual(s.estimated_saved_tokens, 0.0)

    def test_baseline_helpers(self):
        b = BaselineAssumption.fixed_interval(duration_s=60.0, interval_s=5.0)
        self.assertEqual(b.opportunities, 12)
        self.assertTrue(b.declared)
        self.assertFalse(BaselineAssumption().declared)

    def test_source_healthy_time_excludes_a_latched_degradation(self):
        rec = recorder(policy=TracePolicy(level="full"))
        rec.record(OBSERVATION_STAGE, trace_id="c@0", source_ts=0.0)
        rec.health_sink()(_Health("judge_scheduler_stalled", ts=10.0))
        rec.health_sink()(_Health("judge_scheduler_recovered", ts=16.0))
        rec.record(OBSERVATION_STAGE, trace_id="c@30", source_ts=30.0)
        rec.close()
        s = rec.summary()
        self.assertAlmostEqual(s.watch_duration_s, 30.0)
        self.assertAlmostEqual(s.source_healthy_s, 24.0)

    def test_unrecovered_degradation_runs_to_end_of_run(self):
        rec = recorder(policy=TracePolicy(level="full"))
        rec.record(OBSERVATION_STAGE, trace_id="c@0", source_ts=0.0)
        rec.health_sink()(_Health("capture_plan_unmet", ts=10.0))
        rec.record(OBSERVATION_STAGE, trace_id="c@40", source_ts=40.0)
        rec.close()
        self.assertAlmostEqual(rec.summary().source_healthy_s, 10.0)

    def test_scheduler_drops_land_in_the_funnel(self):
        rec = recorder(policy=TracePolicy(level="full"))
        sink = rec.health_sink()
        sink(_Health("judge_superseded", ts=2.0, data={"reason": "superseded_by_newer_admission"}))
        sink(_Health("judge_revision_dropped", ts=3.0, data={"reason": "goals_revision_changed"}))
        sink(_Health("gate_suppressed", ts=4.0, data={"reason": "refractory"}))
        s = rec.summary()
        self.assertEqual(s.judge_dropped, 2)
        self.assertEqual(s.judge_superseded, 1)
        self.assertEqual(s.gate_outcomes["suppressed"], 1)

    def test_unknown_health_kind_is_recorded_not_lost(self):
        rec = recorder(policy=TracePolicy(level="full"))
        rec.health_sink()(_Health("some_future_event", ts=1.0, data={"x": 1}))
        kinds = [r.data.get("kind") for r in rec.records]
        self.assertIn("some_future_event", kinds)


class _Health:
    """Stands in for percept.runtime.worker.RuntimeHealthEvent (duck-typed by the recorder)."""

    def __init__(self, kind, ts=None, data=None):
        self.kind, self.ts, self.data = kind, ts, data or {}


# ---------------------------------------------------------------------------
# 5. Identity, round-trip, and both drivers
# ---------------------------------------------------------------------------
class TestManifestAndRoundTrip(unittest.TestCase):
    def test_plan_identity_is_content_addressed(self):
        g1 = [Goal(id="g1", condition="the milk is boiling over", say="boiling!")]
        g2 = [Goal(id="g1", condition="the milk is boiling over", say="boiling!")]
        g3 = [Goal(id="g1", condition="the kettle is on", say="boiling!")]
        self.assertEqual(PlanIdentity.from_goals(g1).plan_hash, PlanIdentity.from_goals(g2).plan_hash)
        self.assertNotEqual(PlanIdentity.from_goals(g1).plan_hash, PlanIdentity.from_goals(g3).plan_hash)
        self.assertEqual(PlanIdentity.from_goals(g1).goal_ids, ("g1",))

    # -- Card X3: the identity covers the RESOLVED policy, not only the goals ------------
    def test_plan_hash_separates_arms_that_differ_only_in_a_threshold(self):
        """A threshold A/B has identical goals by construction; the hash must still tell them apart."""
        goals = [Goal(id="g1", condition="the milk is boiling over", say="boiling!")]
        lo = PlanIdentity.from_goals(goals, policy={"motion_gate": 0.10, "salience_stat": "mean"})
        hi = PlanIdentity.from_goals(goals, policy={"motion_gate": 0.20, "salience_stat": "mean"})
        self.assertEqual(lo.goal_ids, hi.goal_ids)
        self.assertNotEqual(lo.plan_hash, hi.plan_hash)

    def test_plan_hash_is_order_independent(self):
        """Sorted-key canonicalization, the same contract profile_versioning.content_hash holds."""
        goals = [Goal(id="g1", condition="the milk is boiling over", say="boiling!")]
        fwd = {"motion_gate": 0.10, "refractory_s": 0.5, "require_false_first": False}
        bwd = {"require_false_first": False, "refractory_s": 0.5, "motion_gate": 0.10}
        self.assertNotEqual(list(fwd), list(bwd))
        self.assertEqual(
            PlanIdentity.from_goals(goals, policy=fwd).plan_hash,
            PlanIdentity.from_goals(goals, policy=bwd).plan_hash,
        )

    def test_absent_and_explicitly_none_policy_knobs_are_the_same_configuration(self):
        goals = [Goal(id="g1", condition="c", say="s")]
        a = PlanIdentity.from_goals(goals, policy={"motion_gate": 0.1})
        b = PlanIdentity.from_goals(goals, policy={"motion_gate": 0.1, "dwell": None})
        self.assertEqual(a.plan_hash, b.plan_hash)
        # ...and an int cutoff equals the same float cutoff.
        self.assertEqual(
            PlanIdentity.from_goals(goals, policy={"refractory_s": 1}).plan_hash,
            PlanIdentity.from_goals(goals, policy={"refractory_s": 1.0}).plan_hash,
        )

    def test_plan_mode_is_part_of_the_identity_not_a_label_beside_it(self):
        goals = [Goal(id="g1", condition="c", say="s")]
        self.assertNotEqual(
            PlanIdentity.from_goals(goals, plan_mode="strict").plan_hash,
            PlanIdentity.from_goals(goals, plan_mode="relaxed").plan_hash,
        )

    def test_policy_round_trips_through_to_dict_without_moving_the_hash(self):
        goals = [Goal(id="g1", condition="c", say="s")]
        p = PlanIdentity.from_goals(
            goals, plan_mode="strict",
            policy={"motion_gate": 0.1, "require_false_first": True, "salience_stat": "p95"},
        )
        back = PlanIdentity.from_dict(p.to_dict())
        self.assertEqual(back, p)
        self.assertEqual(back.policy_dict["motion_gate"], 0.1)
        self.assertIsInstance(hash(p), int)   # frozen + hashable: policy is a tuple, not a dict

    def test_nested_policy_blocks_flatten_to_dotted_keys(self):
        from percept.tracing import canonical_policy

        self.assertEqual(
            canonical_policy({"gate": {"refractory_s": 0.5, "require_false_first": True}}),
            {"gate.refractory_s": 0.5, "gate.require_false_first": True},
        )

    def test_manifest_carries_versions_and_identities(self):
        import percept
        from percept.watch import WATCH_VERSION

        m = RunTraceManifest.create(
            source="camera",
            goals=[Goal(id="g1", condition="x", say="x")],
            components=[ComponentIdentity(role="judge", name="anthropic", model="claude-sonnet-4-5",
                                          version="1.0")],
            baseline=BaselineAssumption.every_observation(10),
        )
        self.assertEqual(m.percept_version, percept.__version__)
        self.assertEqual(m.watch_version, WATCH_VERSION)
        self.assertIsNotNone(m.contract_version)
        self.assertEqual(m.components[0].model, "claude-sonnet-4-5")
        self.assertTrue(m.run_id)

    def test_artifact_round_trips_through_json(self):
        rec = recorder(policy=TracePolicy(level="full"),
                       baseline=BaselineAssumption.every_observation(8))
        for i in range(8):
            rec.record(OBSERVATION_STAGE, trace_id=f"c@{i}", source_ts=float(i), escalate=i < 2)
            if i < 2:
                rec.record(ADMISSION_STAGE, trace_id=f"c@{i}", source_ts=float(i))
                rec.record(STAGE_JUDGE_FINISHED, trace_id=f"c@{i}", source_ts=float(i), goal_id="g1")
                rec.record_usage(trace_id=f"c@{i}", input_tokens=500.0)
        rec.close()
        text = rec.to_json()
        m2, records, s2 = load_trace(text)
        self.assertEqual(m2.run_id, rec.manifest.run_id)
        self.assertEqual(len(records), len(rec.records))
        self.assertEqual(s2.observed, rec.summary().observed)
        self.assertEqual(s2.estimated_saved_calls, 6)
        self.assertEqual(s2.estimated_saved_tokens, 3000.0)
        self.assertEqual(json.loads(text)["schema_version"], m2.schema_version)

    def test_write_json(self):
        import tempfile
        import pathlib

        rec = recorder(policy=TracePolicy(level="full"))
        rec.record(OBSERVATION_STAGE, trace_id="c@0", source_ts=0.0)
        with tempfile.TemporaryDirectory() as d:
            p = rec.write_json(pathlib.Path(d) / "nested" / "trace.json")
            self.assertTrue(p.exists())
            m2, records, s2 = load_trace(p.read_text())
            self.assertEqual(s2.observed, 1)

    def test_recount_from_records_alone(self):
        """A trace file taken at level=full re-derives the same funnel with no counters handed in."""
        rec = recorder(policy=TracePolicy(level="full"),
                       baseline=BaselineAssumption.every_observation(6))
        for i in range(6):
            rec.record(OBSERVATION_STAGE, trace_id=f"c@{i}", source_ts=float(i))
            if i < 3:
                rec.record(ADMISSION_STAGE, trace_id=f"c@{i}", source_ts=float(i))
                rec.record(STAGE_JUDGE_FINISHED, trace_id=f"c@{i}", source_ts=float(i))
        live = rec.summary()
        recount = TraceSummary.derive(rec.manifest, list(rec.records))
        self.assertEqual(recount.observed, live.observed)
        self.assertEqual(recount.admitted, live.admitted)
        self.assertEqual(recount.judge_completed, live.judge_completed)
        self.assertAlmostEqual(recount.watch_duration_s, live.watch_duration_s)

    def test_tracer_consumes_the_existing_TraceEvent_stream_unchanged(self):
        rec = recorder(policy=TracePolicy(level="full"))
        cb = rec.tracer()
        cb(TraceEvent(trace_id="camera@12.0", stage=OBSERVATION_STAGE, t=12.0,
                      data={"motion": 0.4, "escalate": True, "coast_admit": False}))
        cb(TraceEvent(trace_id="camera@12.0", stage=ADMISSION_STAGE, t=12.0))
        cb(TraceEvent(trace_id="camera@12.0", stage=STAGE_GATE_CLASSIFIED, t=12.0, goal_id="g1",
                      data={"truth": "yes", "action": "fire", "reason": "rising_edge"}))
        s = rec.summary()
        self.assertEqual((s.observed, s.admitted, s.gate_outcomes["fire"]), (1, 1, 1))
        self.assertEqual(rec.records[0].data["motion"], 0.4)

    def test_observed_falls_back_to_admissions_for_a_direct_driver(self):
        """A bare perceive_judged() driver emits no edge.decision at all — observed must not be 0."""
        rec = recorder(policy=TracePolicy(level="full"))
        for i in range(3):
            rec.record(ADMISSION_STAGE, trace_id=f"c@{i}", source_ts=float(i))
        s = rec.summary()
        self.assertEqual(s.observed, 3)
        self.assertEqual(s.admitted, 3)


# ---------------------------------------------------------------------------
# 6. End-to-end against the REAL perceive_judged path (no runtime worker)
# ---------------------------------------------------------------------------
CUP = Box(0.4, 0.4, 0.2, 0.2)


class TestPerceiveJudgedDriver(unittest.TestCase):
    """Requirement 6: a client that bypasses the runtime worker still gets a full trace surface."""

    def _core(self):
        vision = FakeVision(
            script={"cup": Verdict(True, 0.95, "a cup of coffee")},
            detect_script={"cup": [Observation("cup", CUP, 0.9)]},
        )
        core = Percept.create(
            vision=vision,
            settings=Settings(gate=GateSettings(threshold=0.7, validity_window_s=10.0)),
            discover_plugins=False,
        )
        core.add_goal(Goal(id="g1", condition="the user is drinking coffee", say="stay off caffeine"))
        return core

    def test_direct_driver_produces_a_full_funnel(self):
        core = self._core()
        rec = TraceRecorder(
            RunTraceManifest.create(
                source="camera", driver="perceive_judged", goals=list(core.goals.active()),
                policy=TracePolicy(level="full"),
                baseline=BaselineAssumption.every_observation(3),
                components=[ComponentIdentity(role="vision", name="fake")],
            ),
            wall_clock=FakeClock(),
        )
        core.on_trace = rec.tracer()

        async def drive():
            for t in (1.0, 2.0, 3.0):
                await core.perceive_judged("cup", t=t)

        asyncio.run(drive())
        rec.close()
        s = rec.summary()

        self.assertEqual(rec.manifest.driver, "perceive_judged")
        self.assertEqual(s.admitted, 3)         # source.frame per call
        self.assertEqual(s.observed, 3)         # no edge gate on this path -> fallback
        self.assertEqual(s.judge_started, 3)
        self.assertEqual(s.judge_completed, 3)
        self.assertEqual(s.gate_outcomes["fire"], 1)          # rising edge fires exactly once
        self.assertGreaterEqual(s.gate_outcomes["silent"], 1)
        self.assertEqual(s.estimated_saved_calls, 0)          # baseline 3, actual 3
        self.assertIsNotNone(rec.manifest.plan.plan_hash)
        # every retained record carries both clocks
        for r in rec.records:
            self.assertIsNotNone(r.observed_wall_ts)
            self.assertIsNotNone(r.emitted_wall_ts)
            self.assertIsNotNone(r.source_ts)

    def test_recorder_faults_never_break_perception(self):
        core = self._core()
        rec = TraceRecorder(RunTraceManifest.create(policy=TracePolicy(level="full")))
        core.on_trace = rec.tracer()

        async def drive():
            return await core.perceive_judged("cup", t=1.0)

        rec.close()                       # a closed recorder rejects everything
        fires = asyncio.run(drive())
        self.assertEqual(len(fires), 1)   # perception is UNAFFECTED
        self.assertEqual(len(rec.records), 0)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
