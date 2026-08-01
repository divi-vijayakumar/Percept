"""percept.eval.stage_attribution — the stage-attribution helper.

The bars this file holds, one class each:

1. COMPOSITION — every count and every percentile is the one the owning module already derived.
   If this module ever starts computing its own, these tests go red.
2. DISCRIMINATION (mutation) — two traces that differ ONLY in which stage consumed the time must
   produce a DIFFERENT attribution, while the whole-pipe figure stays byte-identical. A helper that
   reports the same table for both is worse than none, so the whole-pipe equality is asserted too:
   it is the proof that the per-segment split is carrying the signal.
3. DISCRIMINATION (mutation) — two traces with identical timing but a different admission ratio must
   move the FUNNEL while leaving every latency percentile untouched.
4. REFUSAL — a censored distribution must refuse to yield a percentile, on a declared deadline and
   on trace-derived truncation alike; and removing the cause must flip the refusal back, so the
   refusal is shown to be conditional rather than unconditional.
5. HONESTY — not-measured is absent rather than zero, a zero denominator has no ratio, and admitted
   evidence is a UNION of windows rather than count x interval.
"""

from __future__ import annotations

import unittest

from percept.eval.stage_attribution import (
    SEGMENT_NAMES,
    CensoredLatencyError,
    StageAttribution,
    UnmeasuredSegmentError,
    attribute,
    conditions_from_manifest,
    stage_funnel,
    stage_latencies,
)
from percept.slo import LatencyConditions, judge_provider_spans, measure_judge_latency
from percept.tracing import (
    ADMISSION_STAGE,
    OBSERVATION_STAGE,
    STAGE_GATE_CLASSIFIED,
    STAGE_INCIDENT_CREATED,
    STAGE_JUDGE_DEGRADED,
    STAGE_JUDGE_FINISHED,
    STAGE_JUDGE_STARTED,
    ComponentIdentity,
    RunTraceManifest,
    TracePolicy,
    TraceRecorder,
    TraceSummary,
    union_seconds,
)

CONDITIONS = LatencyConditions(
    model="fake-offline",
    transport="synthetic trace",
    frames=None,
    corpus="hand-built trace (not a corpus)",
    build="test",
)


class FakeClock:
    """A wall clock we control, so every latency assertion is exact rather than flaky."""

    def __init__(self, start: float = 1000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, dt: float) -> None:
        self.now += dt


def _manifest(**kw) -> RunTraceManifest:
    kw.setdefault("policy", TracePolicy(level="full"))
    return RunTraceManifest.create(wall_clock=lambda: 1000.0, run_id="run-attr", **kw)


def _cost(costs, name: str, index: int) -> float:
    """A per-segment cost: a scalar applies to every unit, a sequence is indexed per unit."""
    value = costs.get(name, 0.0)
    if isinstance(value, (int, float)):
        return float(value)
    return float(value[index])


def build_trace(
    costs,
    *,
    frames: int = 9,
    admit_every: int = 3,
    close_judge: bool = True,
    degrade_unclosed: bool = False,
    window_s: float = 0.0,
    stop_after: str = "",
    undelivered: tuple[int, ...] = (),
):
    """Record a trace whose every stage cost is dictated by ``costs``.

    ``observed_wall_ts`` is stamped explicitly on every record of a trace so ARRIVAL is a controlled
    input rather than whenever the recorder first saw the id — that is what makes ``intake`` and
    ``end_to_end`` measurable at all.

    ``close_judge=False`` opens a judge span and never closes it (a call that could not finish);
    ``degrade_unclosed`` closes those with ``judge.degraded`` instead, which is the shipped rule for
    keeping a slow outcome IN the sample. ``stop_after`` truncates each admitted unit's spine after
    the named stage, so a segment can be made unmeasurable on purpose. ``undelivered`` names frame
    indices whose gate FIRES but whose incident never appears — a fire that did not become an event.
    """
    clock = FakeClock()
    rec = TraceRecorder(_manifest(), wall_clock=clock)
    admitted_index = 0
    for i in range(frames):
        tid = f"camera@{i}"
        arrival = clock.now
        admitted = i % admit_every == 0
        clock.advance(_cost(costs, "intake", i))
        rec.record(
            OBSERVATION_STAGE,
            trace_id=tid,
            source_ts=float(i),
            observed_wall_ts=arrival,
            escalate=admitted,
        )
        if not admitted:
            clock.advance(0.001)
            continue
        k = admitted_index
        admitted_index += 1
        clock.advance(_cost(costs, "admission", k))
        window = (
            {"window_start_s": float(i) - window_s, "window_end_s": float(i)} if window_s else {}
        )
        rec.record(
            ADMISSION_STAGE,
            trace_id=tid,
            source_ts=float(i),
            observed_wall_ts=arrival,
            **window,
        )
        if stop_after == "admission":
            clock.advance(0.001)
            continue
        clock.advance(_cost(costs, "dispatch", k))
        rec.record(
            STAGE_JUDGE_STARTED,
            trace_id=tid,
            source_ts=float(i),
            goal_id="watch",
            observed_wall_ts=arrival,
        )
        clock.advance(_cost(costs, "judge", k))
        if not close_judge:
            if degrade_unclosed:
                rec.record(
                    STAGE_JUDGE_DEGRADED,
                    trace_id=tid,
                    source_ts=float(i),
                    goal_id="watch",
                    observed_wall_ts=arrival,
                    cause="provider timeout",
                )
            clock.advance(0.001)
            continue
        rec.record(
            STAGE_JUDGE_FINISHED,
            trace_id=tid,
            source_ts=float(i),
            goal_id="watch",
            observed_wall_ts=arrival,
            satisfied=True,
            confidence=0.9,
        )
        clock.advance(_cost(costs, "classification", k))
        rec.record(
            STAGE_GATE_CLASSIFIED,
            trace_id=tid,
            source_ts=float(i),
            goal_id="watch",
            observed_wall_ts=arrival,
            truth="yes",
            action="fire",
        )
        clock.advance(_cost(costs, "delivery", k))
        if i not in undelivered:
            rec.record(
                STAGE_INCIDENT_CREATED,
                trace_id=tid,
                source_ts=float(i),
                observed_wall_ts=arrival,
                incident_id=f"inc-{i}",
            )
        clock.advance(0.001)
    rec.close()
    return rec.records


def _attr(records, **kw) -> StageAttribution:
    kw.setdefault("conditions", CONDITIONS)
    return attribute(records, **kw)


# The two traces at the heart of the mutation test. Per admitted unit the TOTAL is identical —
# judge and delivery simply trade the same three costs — so any whole-pipe figure is blind to the
# difference and only a per-stage split can see it.
SLOW_JUDGE_COSTS = {
    "intake": 0.004,
    "admission": 0.002,
    "dispatch": 0.010,
    "judge": [0.100, 0.200, 1.000],
    "classification": 0.003,
    "delivery": [0.010, 0.010, 0.010],
}
SLOW_DELIVERY_COSTS = dict(
    SLOW_JUDGE_COSTS,
    judge=[0.010, 0.010, 0.010],
    delivery=[0.100, 0.200, 1.000],
)

#: Same spine, one scalar cost per stage — usable for any number of admitted units.
FLAT_COSTS = {
    "intake": 0.004,
    "admission": 0.002,
    "dispatch": 0.010,
    "judge": 0.500,
    "classification": 0.003,
    "delivery": 0.010,
}


# ---------------------------------------------------------------------------
# 1. Composition — the numbers have exactly one owner
# ---------------------------------------------------------------------------
class TestComposition(unittest.TestCase):
    def setUp(self):
        self.records = build_trace(SLOW_JUDGE_COSTS)

    def test_funnel_counts_are_the_trace_summary_fields_not_a_recount(self):
        """Every funnel count must BE a TraceSummary field. Two modules with two ideas of what
        ``observed`` means is the drift this composition exists to stop."""
        summary = TraceSummary.derive(_manifest(), list(self.records))
        funnel = stage_funnel(self.records, manifest=_manifest())
        self.assertEqual(funnel.counts["observed"], summary.observed)
        self.assertEqual(funnel.counts["admitted"], summary.admitted)
        self.assertEqual(funnel.counts["judge_started"], summary.judge_started)
        self.assertEqual(funnel.counts["judge_completed"], summary.judge_completed)
        self.assertEqual(funnel.counts["fired"], summary.incidents)
        self.assertEqual(funnel.counts["observed"], 9)
        self.assertEqual(funnel.counts["admitted"], 3)
        self.assertEqual(funnel.counts["fired"], 3)

    def test_fired_counts_delivered_events_not_gate_fires(self):
        """``fired`` is the OUT point of the funnel — ``incident.created`` — and NOT the gate's fire
        bucket. On this trace three units fire and only two become events; reading the gate bucket
        would report a delivery failure as a success and leave the last funnel ratio at a flat 1.0,
        which is the one place a broken sink would have been visible."""
        records = build_trace(SLOW_JUDGE_COSTS, undelivered=(6,))
        funnel = stage_funnel(records)
        self.assertEqual(funnel.gate_outcomes["fire"], 3)
        self.assertEqual(funnel.counts["judge_completed"], 3)
        self.assertEqual(funnel.counts["fired"], 2)
        self.assertAlmostEqual(funnel.ratio("outcome"), 2 / 3)
        self.assertEqual(funnel.step("outcome").dropped, 1)

    def test_each_funnel_point_reads_its_own_stage(self):
        """A funnel whose points are wired to the wrong summary fields still looks monotone. This
        trace makes all five counts distinct, so any crossed wire changes a number."""
        records = build_trace(FLAT_COSTS, frames=12, admit_every=3, undelivered=(6, 9))
        records = [
            r
            for r in records
            if not (r.stage == STAGE_JUDGE_STARTED and r.trace_id == "camera@9")
        ]
        funnel = stage_funnel(records)
        self.assertEqual(funnel.counts["observed"], 12)
        self.assertEqual(funnel.counts["admitted"], 4)
        self.assertEqual(funnel.counts["judge_started"], 3)
        self.assertEqual(funnel.counts["judge_completed"], 4)
        self.assertEqual(funnel.counts["fired"], 2)
        self.assertEqual(len(set(funnel.counts.values())), 4)

    def test_judge_segment_agrees_with_the_shipped_judge_span_definition(self):
        """DRIFT GUARD. ``percept.slo.judge_provider_spans`` already defines the judge span, down to
        the rule that a DEGRADED judgment closes it. The generalized pairing here must reproduce it
        exactly on a well-formed trace; if it ever stops doing so, the two definitions have forked
        and this test is the alarm."""
        shipped = measure_judge_latency(
            judge_provider_spans(self.records), conditions=CONDITIONS
        )
        mine = stage_latencies(self.records, conditions=CONDITIONS)["judge"]
        self.assertEqual(mine.n, shipped.n)
        for field in ("p25_s", "p50_s", "p95_s", "max_s"):
            self.assertAlmostEqual(getattr(mine, field), getattr(shipped, field), places=9)

    def test_judge_segment_still_agrees_when_a_call_degrades(self):
        """The degraded outcome is a completed wait from the client's side. Both definitions must
        keep it IN the sample, otherwise the slow tail is censored by the measurement itself."""
        records = build_trace(SLOW_JUDGE_COSTS, close_judge=False, degrade_unclosed=True)
        shipped = judge_provider_spans(records)
        mine = stage_latencies(records, conditions=CONDITIONS)["judge"]
        self.assertEqual(mine.n, len(shipped))
        self.assertAlmostEqual(mine.max_s, max(shipped), places=9)
        self.assertAlmostEqual(mine.max_s, 1.0, places=9)

    def test_admitted_evidence_is_a_union_of_windows_not_count_times_interval(self):
        """Admissions at t=0,3,6 each carrying a 4s window overlap. 3 x 4 = 12s is the wrong answer;
        the union of [-4,0],[-1,3],[2,6] is 10s. This is inherited from union_seconds, not redone."""
        records = build_trace(SLOW_JUDGE_COSTS, window_s=4.0)
        funnel = stage_funnel(records)
        self.assertEqual(funnel.counts["admitted"], 3)
        self.assertAlmostEqual(funnel.admitted_evidence_union_s, 10.0)
        self.assertAlmostEqual(
            funnel.admitted_evidence_union_s,
            union_seconds([(-4.0, 0.0), (-1.0, 3.0), (2.0, 6.0)]),
        )
        self.assertNotAlmostEqual(funnel.admitted_evidence_union_s, 3 * 4.0)

    def test_conditions_come_from_the_manifest_and_are_never_invented(self):
        manifest = _manifest(
            components=[ComponentIdentity(role="vision", name="fake", model="fake-model-x")]
        )
        conds = conditions_from_manifest(manifest)
        self.assertEqual(conds.model, "fake-model-x")
        self.assertIn("run-attr", conds.corpus)
        self.assertIsNone(conds.frames)

    def test_conditions_read_unstated_rather_than_a_plausible_guess(self):
        conds = conditions_from_manifest(None)
        self.assertEqual(conds.model, "unstated")
        self.assertEqual(conds.corpus, "unstated")


# ---------------------------------------------------------------------------
# 2. MUTATION — move the time between stages, the attribution must move
# ---------------------------------------------------------------------------
class TestLatencyAttributionDiscriminates(unittest.TestCase):
    """The mutation: two traces identical in every count and in total time, differing ONLY in which
    stage consumed it."""

    def setUp(self):
        self.slow_judge = _attr(build_trace(SLOW_JUDGE_COSTS))
        self.slow_delivery = _attr(build_trace(SLOW_DELIVERY_COSTS))

    def test_the_two_traces_are_indistinguishable_on_the_funnel(self):
        """The control. If the funnels differed, a moved number could be explained by unit counts
        rather than by where the time went, and the discrimination below would prove nothing."""
        self.assertEqual(self.slow_judge.funnel.counts, self.slow_delivery.funnel.counts)
        for step in ("admission", "dispatch", "judge", "outcome"):
            self.assertEqual(
                self.slow_judge.funnel.ratio(step), self.slow_delivery.funnel.ratio(step)
            )

    def test_the_whole_pipe_figure_cannot_tell_them_apart(self):
        """The reason this module exists. End-to-end latency is byte-identical across a trace whose
        model call ate the budget and one whose delivery did. A pass/fail on the pipe is not a
        diagnosis."""
        for p in (25.0, 50.0, 95.0, 100.0):
            self.assertAlmostEqual(
                self.slow_judge.percentile("end_to_end", p),
                self.slow_delivery.percentile("end_to_end", p),
                places=9,
            )
        # 0.004 intake + 0.002 admission + 0.010 dispatch + 1.000 judge + 0.003 classification
        # + 0.010 delivery — the same total whichever of the two stages the second spent it on.
        self.assertAlmostEqual(self.slow_judge.percentile("end_to_end", 100.0), 1.029, places=9)

    def test_the_attribution_moves_to_the_stage_that_ate_the_time(self):
        self.assertEqual(self.slow_judge.slowest_segment(95.0), "judge")
        self.assertEqual(self.slow_delivery.slowest_segment(95.0), "delivery")

    def test_the_moved_percentiles_swap_exactly(self):
        """Not merely different — the two segments trade their distributions, which is the strongest
        available statement that the split is reading the right spans."""
        self.assertAlmostEqual(self.slow_judge.percentile("judge", 95.0), 1.0, places=9)
        self.assertAlmostEqual(self.slow_judge.percentile("delivery", 95.0), 0.01, places=9)
        self.assertAlmostEqual(self.slow_delivery.percentile("judge", 95.0), 0.01, places=9)
        self.assertAlmostEqual(self.slow_delivery.percentile("delivery", 95.0), 1.0, places=9)
        self.assertGreater(
            self.slow_judge.percentile("judge", 95.0),
            self.slow_delivery.percentile("judge", 95.0),
        )
        self.assertLess(
            self.slow_judge.percentile("delivery", 95.0),
            self.slow_delivery.percentile("delivery", 95.0),
        )

    def test_the_untouched_segments_do_not_move(self):
        """A helper that smeared the mutation across every segment would also 'discriminate'. The
        stages that did not change must read identically in both traces."""
        for name in ("intake", "admission", "dispatch", "classification"):
            self.assertAlmostEqual(
                self.slow_judge.percentile(name, 95.0),
                self.slow_delivery.percentile(name, 95.0),
                places=9,
            )
        self.assertAlmostEqual(self.slow_judge.percentile("dispatch", 95.0), 0.010, places=9)
        self.assertAlmostEqual(self.slow_judge.percentile("intake", 95.0), 0.004, places=9)

    def test_describe_names_the_stage_a_reader_should_look_at(self):
        text = self.slow_judge.describe()
        self.assertIn("judge", text)
        self.assertIn("publishable: True", text)


# ---------------------------------------------------------------------------
# 3. MUTATION — move the admission ratio, the funnel must move and latency must not
# ---------------------------------------------------------------------------
class TestFunnelAttributionDiscriminates(unittest.TestCase):
    """The complementary mutation. Identical per-unit timing, different admission ratio: the funnel
    half must carry the whole signal, and every latency percentile must stay put."""

    def setUp(self):
        self.thrifty = _attr(build_trace(FLAT_COSTS, frames=9, admit_every=9))
        self.spendy = _attr(build_trace(FLAT_COSTS, frames=9, admit_every=3))

    def test_the_funnel_moves(self):
        self.assertEqual(self.thrifty.funnel.counts["admitted"], 1)
        self.assertEqual(self.spendy.funnel.counts["admitted"], 3)
        self.assertAlmostEqual(self.thrifty.funnel.ratio("admission"), 1 / 9)
        self.assertAlmostEqual(self.spendy.funnel.ratio("admission"), 3 / 9)
        self.assertAlmostEqual(self.thrifty.funnel.admission_economy, 8 / 9)
        self.assertAlmostEqual(self.spendy.funnel.admission_economy, 6 / 9)
        self.assertGreater(
            self.thrifty.funnel.admission_economy, self.spendy.funnel.admission_economy
        )

    def test_latency_does_not_move(self):
        for name in self.thrifty.measured_segments:
            for p in (50.0, 95.0, 100.0):
                self.assertAlmostEqual(
                    self.thrifty.percentile(name, p), self.spendy.percentile(name, p), places=9
                )

    def test_only_the_sample_count_records_the_difference(self):
        self.assertEqual(self.thrifty.latencies["judge"].n, 1)
        self.assertEqual(self.spendy.latencies["judge"].n, 3)


# ---------------------------------------------------------------------------
# 4. REFUSAL — a censored distribution yields no percentile
# ---------------------------------------------------------------------------
class TestCensoringRefusal(unittest.TestCase):
    def setUp(self):
        self.records = build_trace(SLOW_JUDGE_COSTS)

    def test_a_declared_deadline_makes_the_segment_unquotable(self):
        attribution = _attr(self.records, deadlines={"judge": 1.5})
        self.assertFalse(attribution.is_publishable("judge"))
        self.assertFalse(attribution.publishable)
        self.assertEqual(attribution.unpublishable_segments, ("judge",))
        with self.assertRaises(CensoredLatencyError):
            attribution.percentile("judge", 95.0)
        with self.assertRaises(CensoredLatencyError):
            attribution.percentile("judge", 50.0)

    def test_removing_the_deadline_flips_the_refusal_back(self):
        """MUTATION on the refusal itself. A guard that refuses unconditionally detects nothing —
        the same records with no deadline in force must yield the number."""
        censored = _attr(self.records, deadlines={"judge": 1.5})
        clean = _attr(self.records)
        with self.assertRaises(CensoredLatencyError):
            censored.percentile("judge", 95.0)
        self.assertAlmostEqual(clean.percentile("judge", 95.0), 1.0, places=9)
        self.assertTrue(clean.publishable)

    def test_a_censored_segment_blocks_ranking_rather_than_being_skipped(self):
        """Silently ranking the segments that ARE quotable would name the wrong stage: the censored
        one could be the slowest and nothing in the sample can say."""
        attribution = _attr(self.records, deadlines={"judge": 1.5})
        with self.assertRaises(CensoredLatencyError):
            attribution.slowest_segment(95.0)
        self.assertEqual(_attr(self.records).slowest_segment(95.0), "judge")

    def test_censoring_one_segment_poisons_the_whole_report(self):
        self.assertFalse(_attr(self.records, deadlines={"delivery": 0.05}).publishable)
        self.assertTrue(_attr(self.records).publishable)

    def test_the_censored_distribution_is_retained_and_says_so(self):
        """The censored run is the evidence for why a deadline must change, so it is kept — it just
        may not be quoted as a measurement."""
        attribution = _attr(self.records, deadlines={"judge": 1.5})
        described = attribution.distribution("judge").describe()
        self.assertIn("CENSORED", described)
        self.assertIn("NOT A MEASUREMENT", described)
        self.assertIn("[NOT QUOTABLE]", attribution.describe())

    def test_an_unknown_segment_name_in_deadlines_is_rejected(self):
        """A typo'd deadline that silently applied to nothing would report a censored run as clean."""
        with self.assertRaises(KeyError):
            stage_latencies(self.records, conditions=CONDITIONS, deadlines={"judgment": 1.5})


class TestTruncationRefusalDerivedFromTheTrace(unittest.TestCase):
    """The refusal that needs no declaration: the records themselves show calls that never finished,
    so the sample excludes exactly the slow half a percentile is supposed to describe."""

    def test_judge_spans_that_never_closed_make_the_segment_unquotable(self):
        records = build_trace(SLOW_JUDGE_COSTS, close_judge=False)
        attribution = _attr(records)
        self.assertEqual(attribution.truncation["judge"].starts, 3)
        self.assertEqual(attribution.truncation["judge"].unclosed, 3)
        self.assertTrue(attribution.truncation["judge"].is_truncated)
        self.assertNotIn("judge", attribution.latencies)  # nothing closed, so nothing measured
        with self.assertRaises(UnmeasuredSegmentError):
            attribution.percentile("judge", 95.0)

    def test_a_partly_closed_judge_segment_refuses_without_any_declared_deadline(self):
        records = list(build_trace(SLOW_JUDGE_COSTS))
        # Drop the LAST unit's judge.finished: the slowest call is the one that vanishes, which is
        # exactly how a truncated sample flatters the median.
        survivors = [
            r
            for r in records
            if not (r.stage == STAGE_JUDGE_FINISHED and r.trace_id == "camera@6")
        ]
        attribution = _attr(survivors)
        self.assertEqual(attribution.truncation["judge"].unclosed, 1)
        self.assertTrue(attribution.truncation["judge"].is_truncated)
        self.assertFalse(attribution.is_publishable("judge"))
        with self.assertRaises(CensoredLatencyError):
            attribution.percentile("judge", 95.0)
        self.assertIn("never closed", attribution.latencies["judge"].note)

    def test_closing_the_span_with_a_degraded_outcome_flips_the_refusal_back(self):
        """MUTATION on the derived refusal. A degraded judgment IS a closed span by the shipped
        definition, so a trace that degrades instead of vanishing must be quotable again."""
        truncated = _attr(build_trace(SLOW_JUDGE_COSTS, close_judge=False, degrade_unclosed=False))
        degraded = _attr(build_trace(SLOW_JUDGE_COSTS, close_judge=False, degrade_unclosed=True))
        self.assertTrue(truncated.truncation["judge"].is_truncated)
        self.assertFalse(degraded.truncation["judge"].is_truncated)
        self.assertTrue(degraded.is_publishable("judge"))
        self.assertAlmostEqual(degraded.percentile("judge", 95.0), 1.0, places=9)

    def test_ordinary_funnel_attrition_is_not_mistaken_for_truncation(self):
        """The anti-false-positive. An admitted frame that never reached a model is the funnel doing
        its job, not a truncated sample — the dispatch segment must stay quotable and the funnel is
        where that drop is reported."""
        records = build_trace(SLOW_JUDGE_COSTS, stop_after="admission")
        attribution = _attr(records)
        self.assertEqual(attribution.funnel.counts["admitted"], 3)
        self.assertEqual(attribution.funnel.counts["judge_started"], 0)
        self.assertEqual(attribution.truncation["dispatch"].unclosed, 3)
        self.assertFalse(attribution.truncation["dispatch"].is_truncated)
        self.assertNotIn("dispatch", attribution.latencies)
        self.assertTrue(attribution.is_publishable("admission"))
        self.assertEqual(attribution.funnel.ratio("dispatch"), 0.0)


# ---------------------------------------------------------------------------
# 5. HONESTY — absent, not zero; no ratio, not a plausible one
# ---------------------------------------------------------------------------
class TestNoFalseNumbers(unittest.TestCase):
    def test_an_unmeasured_segment_is_absent_rather_than_zero(self):
        attribution = _attr(build_trace(SLOW_JUDGE_COSTS, stop_after="admission"))
        self.assertIn("dispatch", attribution.unmeasured_segments)
        self.assertIn("judge", attribution.unmeasured_segments)
        with self.assertRaises(UnmeasuredSegmentError):
            attribution.percentile("judge", 95.0)
        self.assertIn("not measured", attribution.describe())

    def test_a_zero_denominator_has_no_ratio(self):
        """Not 0.0 and not 1.0 — both are numbers a bar would pass."""
        attribution = _attr(())
        self.assertIsNone(attribution.funnel.ratio("admission"))
        self.assertIsNone(attribution.funnel.admission_economy)
        self.assertEqual(attribution.measured_segments, ())
        self.assertIsNone(attribution.slowest_segment())
        self.assertTrue(attribution.publishable)  # nothing measured, nothing to be wrong about

    def test_a_negative_drop_is_reported_rather_than_clamped(self):
        """A downstream point outnumbering an upstream one COUNTED IN THE SAME UNIT means records
        were lost between them. Clamping that at zero would make a trace with a hole in it read
        exactly like a clean one, which is the single state a stage attribution must never hide: the
        missing records are on the segment you are about to blame.

        The unit qualifier is load-bearing and was once missing. ``dispatch`` is the one step whose
        two points count different populations — evidence in, (evidence x concern) out — so it is
        compared through :attr:`FunnelStep.upstream_dispatches`; ordinary fan-out is not a loss and
        must not read as one. ``judge`` below has no such crossing: both of its points are per-goal,
        so a shortfall there is exactly what this says it is."""
        records = [r for r in build_trace(SLOW_JUDGE_COSTS) if r.stage != STAGE_JUDGE_STARTED]
        funnel = stage_funnel(records)
        self.assertEqual(funnel.counts["judge_started"], 0)
        self.assertEqual(funnel.counts["judge_completed"], 3)
        self.assertEqual(funnel.step("judge").dropped, -3)
        self.assertIsNone(funnel.ratio("judge"))
        self.assertEqual(_attr(records).truncation["judge"].orphan_closes, 3)

    def test_a_percentile_the_distribution_does_not_carry_is_refused(self):
        attribution = _attr(build_trace(SLOW_JUDGE_COSTS))
        with self.assertRaises(ValueError):
            attribution.percentile("judge", 99.0)

    def test_a_backwards_clock_is_clamped_but_counted(self):
        """A silently clamped negative span is a lie about a real clock fault."""
        records = list(build_trace(SLOW_JUDGE_COSTS))
        for index, record in enumerate(records):
            if record.stage == STAGE_GATE_CLASSIFIED and record.trace_id == "camera@0":
                records[index] = type(record)(
                    **{**record.to_dict(), "emitted_wall_ts": (record.emitted_wall_ts or 0.0) - 5.0}
                )
        attribution = _attr(records)
        self.assertEqual(attribution.truncation["classification"].clamped, 1)
        self.assertAlmostEqual(attribution.percentile("classification", 25.0), 0.0, places=9)
        self.assertIn("clamped", attribution.latencies["classification"].note)

    def test_every_declared_segment_is_reported_on(self):
        attribution = _attr(build_trace(SLOW_JUDGE_COSTS))
        self.assertEqual(
            tuple(sorted(attribution.measured_segments + attribution.unmeasured_segments)),
            tuple(sorted(SEGMENT_NAMES)),
        )
        self.assertEqual(attribution.measured_segments, SEGMENT_NAMES)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
