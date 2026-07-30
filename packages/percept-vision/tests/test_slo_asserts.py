"""M4 -- SLO CI asserts: reflex p95, $/stream-hour, fovea call-reduction (percept/slo.py).

Offline/keyless: synthetic frames + a scripted :class:`~percept.contracts.Verdict`'s ``usage``
dict, no network, no API keys. Each assertion's threshold is a pre-registered constant in
``percept/slo.py`` with its own rationale -- see that module's docstring for the full "why this
number" (this file only exercises the assertions, it does not re-derive the thresholds).

SLO 4 (``percept/slo.py`` section 6 -- the per-goal judge round-trip p99 launch gate) is the one
assertion here that is RELEASE-TIME rather than per-merge, and the one that does not pass. Its
mechanics are asserted unconditionally like everything else in this file; the gate itself runs only
when ``PERCEPT_RELEASE_GATES=1`` is set, which is how a release run asks for it. It is expected to
FAIL there. That is deliberate and the ceiling is not moved to hide it -- see section 6's header.
"""

from __future__ import annotations

import asyncio
import os
import unittest

from percept.adapters import Goal
from percept.agent import Percept
from percept.contracts import Verdict
from percept.harness.tier0 import Frame, GatedWatchLoop, MotionGate, compile_watchspec, gray_frame
from percept.observability import judge_usage
from percept.slo import (
    DOLLARS_PER_STREAM_HOUR_CEILING_USD,
    FOVEA_CALL_REDUCTION_FLOOR,
    JUDGE_ROUND_TRIP_P99_CEILING_S,
    LAUNCH_GATE_JUDGE_P99,
    MEASURED_JUDGE_ROUND_TRIP,
    REFLEX_P95_THRESHOLD_S,
    JudgeCallSample,
    dollars_per_stream_hour,
    fovea_call_reduction,
    judge_p99_gate_from_distribution,
    judge_p99_gate_from_samples,
    judge_provider_spans,
    judge_round_trip_p99,
    judge_stage_latencies,
    measure_reflex_latencies,
    price_usage,
    reflex_p95,
)
from percept.tracing import (
    STAGE_EDGE_DECISION,
    STAGE_JUDGE_DEGRADED,
    STAGE_JUDGE_FINISHED,
    STAGE_JUDGE_STARTED,
    RunTraceManifest,
    TraceRecord,
    TraceSummary,
)

GRID_W, GRID_H = 64, 48

#: Set to ``1`` to run the release-time gates in this file. Absent (the default) skips them, so a
#: per-merge build is not permanently red on a gap that is a release decision.
RELEASE_GATES_ENV = "PERCEPT_RELEASE_GATES"


def _still() -> Frame:
    return gray_frame(GRID_W, GRID_H, fill=0)


def _motion() -> Frame:
    """A scattered, deterministic 'moving' pattern -- large enough of a change to clear a
    permissive gate threshold, same shape as bench_gating.py's fixture."""
    pixels = [0] * (GRID_W * GRID_H)
    for i in range(0, len(pixels), 7):
        pixels[i] = 255
    return Frame(w=GRID_W, h=GRID_H, pixels=tuple(pixels))


def _mostly_still_stream(n: int) -> list[Frame]:
    """A deterministic mostly-still sequence with motion roughly every 10th frame -- the same
    'still scene + occasional event' shape production tier-0 sees (percept/harness/bench_gating.py),
    sized up here for a stable p95 sample / a representative call-reduction fixture."""
    motion_idx = frozenset(i for i in range(n) if i % 10 == 0)
    return [_motion() if i in motion_idx else _still() for i in range(n)]


# ---------------------------------------------------------------------------
# SLO 1 -- REFLEX p95: the tier-0 gate's own decision latency
# ---------------------------------------------------------------------------


class TestReflexP95(unittest.TestCase):
    def test_reflex_p95_under_still_scene_budget(self):
        goals = [Goal(id="g1", condition="a person entered the frame", say="hi")]
        spec = compile_watchspec(goals, threshold=0.2)
        gate = MotionGate(spec)
        frames = _mostly_still_stream(300)

        # Warm-up: the first few calls pay one-time import/allocator costs unrelated to the
        # steady-state per-frame latency a long-running stream actually pays; exclude them from
        # the measured p95 sample (amortized once per process, not once per frame in production).
        warm, measured = frames[:10], frames[10:]
        for f in warm:
            gate.evaluate(f, ts=0.0)

        latencies = measure_reflex_latencies(gate, measured, ts_start=0.0)
        p95 = reflex_p95(latencies)

        self.assertLess(
            p95,
            REFLEX_P95_THRESHOLD_S,
            f"reflex p95 {p95 * 1000:.4f}ms exceeds the {REFLEX_P95_THRESHOLD_S * 1000:.1f}ms "
            f"still-scene budget (15% of the max_fps=15.0 frame period, percept.sampling."
            f"SensePolicy) -- the tier-0 gate is no longer 'effectively free'.",
        )


# ---------------------------------------------------------------------------
# SLO 2 -- $/stream-hour: judge_usage() token counts x PRICE_TABLE
# ---------------------------------------------------------------------------


class TestDollarsPerStreamHour(unittest.TestCase):
    def test_dollars_per_stream_hour_under_ceiling(self):
        # A representative mostly-still deployment: tier-0 suppresses most frames, so the judge
        # fires at a modest, sparse rate -- 30 judged calls/hour (about once every 2 minutes),
        # each a single-frame vision judgment (~2000 input / ~150 output tokens is a plausible
        # single-image-plus-prompt judge call) against the default vision_model (percept/config.py
        # Models.vision_model = "claude-opus-4-8").
        verdict = Verdict(
            satisfied=True,
            confidence=0.9,
            evidence="a person is visible",
            usage={"input_tokens": 2000.0, "output_tokens": 150.0},
        )
        usage = judge_usage(verdict)  # the already-landed hook (percept/observability.py)
        calls = [JudgeCallSample(usage=usage, model="claude-opus-4-8") for _ in range(30)]

        dph = dollars_per_stream_hour(calls, elapsed_s=3600.0, stream_hours=1.0)

        self.assertLess(
            dph,
            DOLLARS_PER_STREAM_HOUR_CEILING_USD,
            f"${dph:.4f}/stream-hour exceeds the ${DOLLARS_PER_STREAM_HOUR_CEILING_USD:.2f}/"
            f"stream-hour ceiling (tests/test_p_econ.py's the release criteria placeholder, reused here).",
        )

    def test_price_usage_prefers_backend_reported_cost_over_token_estimate(self):
        """A backend that reports its own billed cost_usd wins over a token x price estimate."""
        usage = {"cost_usd": 0.5, "input_tokens": 999_999.0}
        self.assertEqual(price_usage(usage, model="claude-opus-4-8"), 0.5)

    def test_price_usage_falls_back_to_default_price_for_uncatalogued_model(self):
        """An unrecognized model prices at the most expensive known tier (fail conservative)."""
        usage = {"input_tokens": 1_000_000.0, "output_tokens": 0.0}
        self.assertEqual(price_usage(usage, model="some-future-model"), 15.00)

    def test_price_usage_is_zero_for_empty_usage(self):
        self.assertEqual(price_usage({}, model="claude-opus-4-8"), 0.0)


# ---------------------------------------------------------------------------
# SLO 3 -- fovea call-reduction: the fraction of frames tier-0 suppresses
# ---------------------------------------------------------------------------


class TestFoveaCallReduction(unittest.TestCase):
    def test_fovea_call_reduction_over_floor(self):
        core = Percept.create(mode="offline")  # offline fakes -- no keys, deterministic
        goals = [
            Goal(id="g_coffee", condition="a coffee cup is on the desk", say="that looks like coffee"),
            Goal(id="g_person", condition="a person entered the frame", say="someone's here"),
        ]
        for g in goals:
            core.add_goal(g)

        loop = GatedWatchLoop(core, goals)
        report = asyncio.run(loop.run(_mostly_still_stream(200), clock_start=0.0, dt=1.0))
        reduction = fovea_call_reduction(report)

        self.assertGreaterEqual(
            reduction,
            FOVEA_CALL_REDUCTION_FLOOR,
            f"fovea call-reduction {reduction:.1%} fell below the {FOVEA_CALL_REDUCTION_FLOOR:.0%} "
            f"floor -- tier-0 gating is no longer clearing its minimum economic bar (at least half "
            f"the naive per-frame VLM spend suppressed).",
        )


# ---------------------------------------------------------------------------
# SLO 4 -- per-goal judge round-trip p99: the RELEASE-TIME launch gate
# ---------------------------------------------------------------------------


def _judge_span(
    trace_id: str,
    goal_id: str,
    *,
    observed: float,
    started: float,
    finished: float,
    end_stage: str = STAGE_JUDGE_FINISHED,
) -> list[TraceRecord]:
    """One goal's judge span as the recorder writes it: a start record and a closing record.

    Both inherit the frame's ``observed_wall_ts``, exactly as ``TraceRecorder`` stamps them, so the
    closing record's ``latency_s`` is admitted-frame -> this goal's verdict.
    """
    return [
        TraceRecord(
            seq=0,
            trace_id=trace_id,
            stage=STAGE_JUDGE_STARTED,
            source_ts=observed,
            observed_wall_ts=observed,
            emitted_wall_ts=started,
            goal_id=goal_id,
        ),
        TraceRecord(
            seq=0,
            trace_id=trace_id,
            stage=end_stage,
            source_ts=observed,
            observed_wall_ts=observed,
            emitted_wall_ts=finished,
            goal_id=goal_id,
        ),
    ]


def _edge_decisions(n: int) -> list[TraceRecord]:
    """``n`` tier-0 gate decisions: one per frame, each emitted the instant it was observed.

    This is ~99% of a real run's records, and it is why a trace summary's blended latency
    percentiles cannot answer a question about judge latency.
    """
    return [
        TraceRecord(
            seq=0,
            trace_id=f"camera@{i}",
            stage=STAGE_EDGE_DECISION,
            source_ts=float(i),
            observed_wall_ts=float(i),
            emitted_wall_ts=float(i),
        )
        for i in range(n)
    ]


class TestJudgeRoundTripP99GateMechanics(unittest.TestCase):
    """Per-merge: the gate's arithmetic and, above all, WHICH records it is allowed to read.

    These assert the mechanics on synthetic records, not any real provider's number -- the same
    scaffold-then-measure split the other release criteria in this tree use.
    """

    def test_latencies_come_from_judge_stages_and_never_from_the_edge_population(self):
        """The two-population artifact, made concrete.

        999 tier-0 decisions at 0.00s and one 8s judgment. The trace summary's blended
        ``latency.p95_s`` reads 0.00s -- it is dominated by the cheap population -- while the judge
        stages read 8s. A gate sourced from that summary field would be green with a judge eight
        times over its budget, which is precisely why ``percept/slo.py`` section 6 forbids it.
        """
        records = _edge_decisions(999) + _judge_span(
            "camera@999", "g1", observed=999.0, started=999.5, finished=1007.0
        )
        summary = TraceSummary.derive(RunTraceManifest(run_id="r"), records)

        self.assertEqual(summary.latency_p95_s, 0.0)
        self.assertEqual(summary.latency_p50_s, 0.0)
        self.assertEqual(judge_stage_latencies(records), [8.0])
        self.assertEqual(judge_round_trip_p99(judge_stage_latencies(records)), 8.0)

    def test_a_degraded_judgment_is_a_completed_wait_and_stays_in_the_sample(self):
        """Dropping degraded judgments would censor the slow tail -- section 5's own mistake."""
        records = _judge_span(
            "camera@1",
            "g1",
            observed=1.0,
            started=1.1,
            finished=13.0,
            end_stage=STAGE_JUDGE_DEGRADED,
        )
        self.assertEqual(judge_stage_latencies(records), [12.0])

    def test_every_goal_in_one_batch_contributes_its_own_wait(self):
        """One admitted frame, one trace id, four goals: the gate is PER GOAL, so four samples."""
        records: list[TraceRecord] = []
        for index, finish in enumerate((3.0, 6.0, 9.0, 12.0)):
            records += _judge_span(
                "camera@1", f"g{index}", observed=0.0, started=0.1, finished=finish
            )
        self.assertEqual(judge_stage_latencies(records), [3.0, 6.0, 9.0, 12.0])

    def test_provider_spans_separate_the_call_from_the_queueing(self):
        """The diagnostic half: a 9s wait made of a 1s call plus 8s of queueing reports both, so a
        failing gate names a cause instead of only a colour."""
        records = _judge_span(
            "camera@1", "g1", observed=0.0, started=8.0, finished=9.0
        )
        self.assertEqual(judge_stage_latencies(records), [9.0])
        self.assertEqual(judge_provider_spans(records), [1.0])

    def test_the_gate_fires_above_the_ceiling_and_clears_below_it(self):
        # Nearest-rank p99 of 100 samples is the 99th sorted value, so the breach has to occupy the
        # top 2% to move it -- a single outlier in 100 is a p100 event and the gate correctly ignores
        # it. Pinned explicitly so the fixture cannot be read as "one slow call fails the gate".
        over = judge_p99_gate_from_samples([2.4] * 98 + [9.0, 9.0])
        self.assertFalse(over.passed)
        self.assertTrue(over.is_decided)
        self.assertEqual(over.measured_s, 9.0)
        self.assertIn("FAIL", over.describe())
        self.assertEqual(judge_p99_gate_from_samples([2.4] * 99 + [9.0]).measured_s, 2.4)

        under = judge_p99_gate_from_samples([0.4] * 98 + [1.2, 1.2])
        self.assertTrue(under.passed)
        self.assertLess(under.measured_s, JUDGE_ROUND_TRIP_P99_CEILING_S)
        self.assertIn("PASS", under.describe())

    def test_a_floor_under_the_ceiling_is_undecided_and_never_a_pass(self):
        """A median below the ceiling says nothing about the tail, so it cannot buy a green."""
        from percept.slo import LatencyConditions, LatencyDistribution

        modest = LatencyDistribution(
            p25_s=None,
            p50_s=1.0,
            p95_s=None,
            max_s=None,
            n=None,
            conditions=LatencyConditions(
                model="m", transport="t", frames=None, corpus="c", build="b"
            ),
        )
        gate = judge_p99_gate_from_distribution(modest)
        self.assertTrue(gate.is_floor)
        self.assertFalse(gate.passed)
        self.assertFalse(gate.is_decided)
        self.assertIn("UNDECIDED", gate.describe())

    def test_the_ceiling_is_the_same_figure_the_p95_bar_already_pre_registered(self):
        """One "how long may a judgment take" number in the tree, not two that can drift apart."""
        from percept.slo import JUDGE_ROUND_TRIP_P95_CEILING_S

        self.assertEqual(JUDGE_ROUND_TRIP_P99_CEILING_S, JUDGE_ROUND_TRIP_P95_CEILING_S)

    def test_the_recorded_measurement_is_uncensored_and_reports_no_p99(self):
        """What the launch gate is judged against, pinned: a real tail, and an honest gap where a
        p99 was never reported (so the standing verdict must read as a FLOOR)."""
        self.assertFalse(MEASURED_JUDGE_ROUND_TRIP.is_censored)
        self.assertIsNone(MEASURED_JUDGE_ROUND_TRIP.p95_s)
        self.assertEqual(MEASURED_JUDGE_ROUND_TRIP.p50_s, 7.34)
        self.assertEqual(MEASURED_JUDGE_ROUND_TRIP.max_s, 16.93)
        self.assertTrue(LAUNCH_GATE_JUDGE_P99.is_floor)
        self.assertTrue(LAUNCH_GATE_JUDGE_P99.is_decided)


@unittest.skipUnless(
    os.environ.get(RELEASE_GATES_ENV) == "1",
    f"release-time gate; set {RELEASE_GATES_ENV}=1 to run it",
)
class TestJudgeRoundTripP99LaunchGate(unittest.TestCase):
    """RELEASE-TIME. This gate does not pass, and it is not supposed to be made to.

    The measured per-goal round trip has a median of 7.34s against a 2.5s ceiling. The two honest
    responses are to make the pipeline faster or to decide at release that the gap is acceptable and
    say so out loud. Raising the ceiling, or marking this expected-to-fail, would be the third
    response -- the one that makes the gate stop measuring anything.
    """

    def test_per_goal_judge_round_trip_p99_under_ceiling(self):
        gate = LAUNCH_GATE_JUDGE_P99
        self.assertTrue(
            gate.passed,
            f"launch gate NOT MET: {gate.describe()}. The ceiling "
            f"({JUDGE_ROUND_TRIP_P99_CEILING_S:g}s) is the figure already pre-registered for judge "
            f"round-trip latency and is not to be raised to clear this; the measured span includes "
            f"the queueing a goal does behind the concurrency bound and behind its batch, which is "
            f"where the gap lives.",
        )


if __name__ == "__main__":
    unittest.main()
