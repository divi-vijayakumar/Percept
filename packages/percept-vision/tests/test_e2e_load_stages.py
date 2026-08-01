"""End-to-end capability — SUSTAINED LOAD, with stage attribution.

One product function, not one mechanism: drive the probe harness's load driver at a full
streams/watches/episodes/fan-out profile, hold it, and ask the five questions a load gate is
actually supposed to answer.

    1. Is the latency figure QUOTABLE at all (uncensored) before any percentile is reported?
    2. When a budget is missed, does the report NAME THE STAGE rather than the pipe?
    3. Is the queue bounded, and is every drop declared rather than silent?
    4. Can one watch consume another's reserved capacity?
    5. Does the funnel stay well-formed under load?

WHAT THE NUMBERS IN THIS FILE ARE
---------------------------------
Every latency here is a ``LatencyProfile`` this test chose and handed to
:class:`~percept.backends.fake_judge.FakeJudge`. **It is not provider latency and is never
presented as one** — real provider qualification is a separate lane and needs keys this suite does
not have. What a chosen latency CAN prove is wiring: that the stage the time was spent in is the
stage the attribution names, and that a censored sample refuses to be quoted. Those are properties
of the code, and they are what this file asserts.

For the same reason this file asserts **no absolute latency ceiling**. A ceiling asserted against a
latency this file picked would pass forever whatever the ceiling was; the launch-gate pair is
handled in ``TestTheLaunchGatePercentilePair`` below, which shows the arithmetic instead.

Reused, never reimplemented:

* ``percept.harness.load_driver`` — the streams/watches/episodes/fan-out shape and its ramp.
* ``percept.backends.fake_judge`` — the seeded latency and the key-free judge double.
* ``percept.eval.stage_attribution`` — the funnel table and the per-segment percentiles. Nothing
  in this file parses a trace itself or takes a percentile itself.
* ``percept.backends.judge_pool`` — the real bounded-occupancy router, for the queue and isolation
  lanes. A test-local queue simulation would prove only that the simulation is bounded.
* ``percept.tracing.TraceRecorder`` — the trace the attribution reads.

THE PROJECTION, STATED OUT LOUD
-------------------------------
The load driver counts frames and admissions; it does not emit a trace. ``_project_trace`` turns one
driver run into a ``.ptrace`` by giving every frame the driver emitted an ``edge.decision`` and every
episode it admitted the full admitted spine, with each stage's wall cost declared in
:class:`StageCosts` (the judge leg alone comes from the driver's own ``FakeJudge`` samples). The
projection invents no unit: the counts are the driver's, and the builder REFUSES to run rather than
manufacture an observation it did not have (see ``_project_trace``'s per-tick check).
"""

from __future__ import annotations

import asyncio
import unittest
from dataclasses import dataclass

from percept import Goal, Percept
from percept.backends.fake_judge import FakeJudge, LatencyProfile
from percept.backends.judge_pool import (
    JudgeCapability,
    JudgePool,
    JudgeRoute,
)
from percept.clock import FakeClock
from percept.config import CacheSettings, GateSettings, Settings
from percept.contracts import Verdict
from percept.eval.stage_attribution import (
    CensoredLatencyError,
    UnmeasuredSegmentError,
    attribute,
    stage_funnel,
)
from percept.harness.load_driver import LoadDriver, LoadProfile, RampSchedule
from percept.harness.stream_gen import parametric_motion_clip
from percept.slo import (
    LatencyConditions,
    judge_provider_spans,
    measure_judge_latency,
)
from percept.tracing import (
    ADMISSION_STAGE,
    OBSERVATION_STAGE,
    STAGE_GATE_CLASSIFIED,
    STAGE_INCIDENT_CREATED,
    STAGE_JUDGE_FINISHED,
    STAGE_JUDGE_STARTED,
    BaselineAssumption,
    RunTraceManifest,
    TracePolicy,
    TraceRecorder,
    TraceSummary,
    judge_call_was_served,
    judge_served_from,
)

# ---------------------------------------------------------------------------
# 0. The load shape, the chosen latencies, and the projection
# ---------------------------------------------------------------------------

#: The scripted frame token the load driver hands its judge on every admitted episode.
LOAD_TOKEN = "load-driver-frame"

DECISIVE = Verdict(satisfied=True, confidence=0.95, evidence="scripted load-probe verdict")

#: Conditions that travel with every percentile taken here. ``model`` names the DOUBLE on purpose:
#: a figure whose conditions said "gemini" would be a lie about where the number came from.
CONDITIONS = LatencyConditions(
    model="fake_judge double (a latency profile this test chose)",
    transport="projected trace from a load-driver run",
    frames=None,
    corpus="one synthetic run — a run is not a corpus",
    build="offline test",
)

#: The held load shape. Concurrent streams x watches per stream, an admitted-episode rate per
#: stream-minute, and the judge fan-out per admission. Fan-out is 1 here so one admitted unit of
#: evidence maps to one judged unit; the fan-out > 1 shape is exercised separately, in
#: ``TestFanOutIsNotADrop``, because it is the one step where the two points count different things.
HELD_PROFILE = LoadProfile(n=3, m=2, e_per_stream_min=120.0, f=1)
HELD_SCHEDULE = RampSchedule(ramp_s=2.0, steady_s=8.0)
TICK_S = 0.25


@dataclass(frozen=True)
class StageCosts:
    """The wall cost of every non-judge stage, in seconds. Numbers this test chose.

    The judge leg is deliberately absent: it is the one segment whose duration comes from the
    driver's own ``FakeJudge`` samples rather than from this table.
    """

    intake: float = 0.004
    admission: float = 0.002
    dispatch: float = 0.010
    classification: float = 0.003
    delivery: float = 0.010


#: The two costings at the heart of the attribution mutation. Per admitted unit the TOTAL is
#: identical — the judge leg and the dispatch (queue-wait) leg simply trade the same 1.19s — so a
#: whole-pipe figure is blind to the difference and only a per-stage split can see it.
JUDGE_HEAVY_COSTS = StageCosts(dispatch=0.010)
JUDGE_HEAVY_LATENCY = LatencyProfile(kind="fixed", value_s=1.200)
DISPATCH_HEAVY_COSTS = StageCosts(dispatch=1.200)
DISPATCH_HEAVY_LATENCY = LatencyProfile(kind="fixed", value_s=0.010)


def _timeline():
    """A dense synthetic clip — 30s at 20fps — so the driver's frame stream always outnumbers its
    admissions and the projection never has to invent an observation."""
    return parametric_motion_clip(w=4, h=4, n=600, motion_frames=set(range(0, 600, 5)), dt=0.05)


@dataclass
class LoadRun:
    """One held load run: the driver's own telemetry, plus the trace projected from it."""

    stats: object
    judge: FakeJudge
    records: tuple
    manifest: RunTraceManifest
    recorder: TraceRecorder


def _drive(
    *,
    profile: LoadProfile = HELD_PROFILE,
    schedule: RampSchedule = HELD_SCHEDULE,
    latency: LatencyProfile = JUDGE_HEAVY_LATENCY,
    seed: int = 11,
    step_s: float = TICK_S,
):
    """Drive the load driver to completion under a fake clock, keeping per-tick admission times.

    ``simulate_sleep=False``: the sampled latency is recorded without being slept, exactly as the
    already-landed overload and latency probes drive it, so no test pays real wall time. Time moves
    only through ``clock.advance``.
    """
    clock = FakeClock()
    judge = FakeJudge(
        seed=seed,
        latency=latency,
        simulate_sleep=False,
        script={LOAD_TOKEN: DECISIVE},
    )
    driver = LoadDriver(profile, schedule, _timeline(), clock=clock, judge=judge)

    ticks: list[tuple[float, int, int]] = []  # (tick wall time, new frames, new judge calls)
    seen_frames = 0
    seen_calls = 0
    while True:
        asyncio.run(driver.tick())
        now = clock()
        new_frames = driver.stats.frames_emitted - seen_frames
        new_calls = len(judge.calls) - seen_calls
        seen_frames = driver.stats.frames_emitted
        seen_calls = len(judge.calls)
        ticks.append((now, new_frames, new_calls))
        if driver.done():
            break
        clock.advance(step_s * 1000.0)
    return driver, judge, ticks


def _project_trace(
    driver,
    judge: FakeJudge,
    ticks,
    *,
    costs: StageCosts,
    fan_out: int,
    baseline: BaselineAssumption | None = None,
    unfinished_every: int = 0,
    undelivered_every: int = 0,
    usage_model: str | None = None,
) -> LoadRun:
    """Turn one driver run into a ``.ptrace``.

    Every frame the driver emitted becomes an ``edge.decision``; every episode it admitted becomes
    the admitted spine (``source.frame`` -> ``judge.started`` -> ``judge.finished`` ->
    ``gate.classified`` -> ``incident.created``), one judge/gate pair per fan-out goal. Both wall
    clocks are stamped explicitly, so no assertion below depends on how fast this machine runs.

    ``unfinished_every`` leaves every n-th admission's judge span OPEN (a call that could not
    finish); ``undelivered_every`` fires the gate on every n-th admission without an incident ever
    appearing; ``usage_model`` meters every completed call through the recorder's own usage seam.
    All three exist so the refusal, funnel and cost assertions can be shown to fail.

    A tick that admitted more episodes than it emitted frames RAISES. The alternative — padding the
    observation count so the funnel looks well-formed — would be a fabricated number, and the whole
    point of this file is that the counts are the driver's.
    """
    manifest = RunTraceManifest.create(
        run_id="held-load",
        source="load-driver",
        driver="load_driver",
        policy=TracePolicy(level="full"),
        baseline=baseline or BaselineAssumption(),
        wall_clock=lambda: 0.0,
    )
    rec = TraceRecorder(manifest, max_records=200_000, wall_clock=lambda: 0.0)

    frame_no = 0
    admission_no = 0
    for now, new_frames, new_calls in ticks:
        new_admissions = new_calls // fan_out
        if new_admissions > new_frames:
            raise AssertionError(
                f"tick at {now:g}s admitted {new_admissions} episode(s) but the driver emitted only "
                f"{new_frames} frame(s); widen the timeline rather than inventing observations"
            )
        for i in range(new_frames):
            tid = f"load-driver@{frame_no}"
            frame_no += 1
            admitted = i < new_admissions
            rec.record(
                OBSERVATION_STAGE,
                trace_id=tid,
                source_ts=now,
                observed_wall_ts=now,
                emitted_wall_ts=now + costs.intake,
                escalate=admitted,
            )
            if not admitted:
                continue
            k = admission_no
            admission_no += 1
            t_admitted = now + costs.intake + costs.admission
            rec.record(
                ADMISSION_STAGE,
                trace_id=tid,
                source_ts=now,
                observed_wall_ts=now,
                emitted_wall_ts=t_admitted,
            )
            t_started = t_admitted + costs.dispatch
            for g in range(fan_out):
                goal_id = f"watch-{g}"
                rec.record(
                    STAGE_JUDGE_STARTED,
                    trace_id=tid,
                    source_ts=now,
                    goal_id=goal_id,
                    observed_wall_ts=now,
                    emitted_wall_ts=t_started,
                )
                if unfinished_every and k % unfinished_every == 0:
                    continue  # a call that could not finish: the span is left open on purpose
                judge_s = judge.calls[k * fan_out + g].latency_s
                t_finished = t_started + judge_s
                rec.record(
                    STAGE_JUDGE_FINISHED,
                    trace_id=tid,
                    source_ts=now,
                    goal_id=goal_id,
                    observed_wall_ts=now,
                    emitted_wall_ts=t_finished,
                    satisfied=True,
                    confidence=0.95,
                )
                if usage_model is not None:
                    rec.record_usage(
                        trace_id=tid,
                        source_ts=now,
                        goal_id=goal_id,
                        input_tokens=1200.0,
                        output_tokens=40.0,
                        model=usage_model,
                    )
                t_classified = t_finished + costs.classification
                rec.record(
                    STAGE_GATE_CLASSIFIED,
                    trace_id=tid,
                    source_ts=now,
                    goal_id=goal_id,
                    observed_wall_ts=now,
                    emitted_wall_ts=t_classified,
                    truth="yes",
                    action="fire",
                )
                if undelivered_every and k % undelivered_every == 0:
                    continue  # the gate fired and no event ever existed
                rec.record(
                    STAGE_INCIDENT_CREATED,
                    trace_id=tid,
                    source_ts=now,
                    observed_wall_ts=now,
                    emitted_wall_ts=t_classified + costs.delivery,
                    incident_id=f"inc-{k}-{g}",
                )
    rec.close()
    return LoadRun(
        stats=driver.stats,
        judge=judge,
        records=rec.records,
        manifest=manifest,
        recorder=rec,
    )


def held_run(
    *,
    costs: StageCosts = JUDGE_HEAVY_COSTS,
    latency: LatencyProfile = JUDGE_HEAVY_LATENCY,
    profile: LoadProfile = HELD_PROFILE,
    fan_out: int | None = None,
    seed: int = 11,
    baseline: BaselineAssumption | None = None,
    unfinished_every: int = 0,
    undelivered_every: int = 0,
    usage_model: str | None = None,
) -> LoadRun:
    driver, judge, ticks = _drive(profile=profile, latency=latency, seed=seed)
    return _project_trace(
        driver,
        judge,
        ticks,
        costs=costs,
        fan_out=profile.f if fan_out is None else fan_out,
        baseline=baseline,
        unfinished_every=unfinished_every,
        undelivered_every=undelivered_every,
        usage_model=usage_model,
    )


def _attr(run: LoadRun, **kw):
    kw.setdefault("conditions", CONDITIONS)
    kw.setdefault("manifest", run.manifest)
    return attribute(run.records, **kw)


# ---------------------------------------------------------------------------
# 1. The load actually happened, and the funnel it produced is well-formed
# ---------------------------------------------------------------------------
class TestTheLoadWasReallyHeld(unittest.TestCase):
    """A load assertion over a run that did not load anything is the emptiest kind of green."""

    def setUp(self):
        self.run = held_run()

    def test_the_full_shape_was_produced_not_just_concurrency(self):
        stats = self.run.stats
        self.assertGreater(stats.episodes_admitted, 30)
        self.assertEqual(stats.judge_calls_issued, stats.episodes_admitted * HELD_PROFILE.f)
        self.assertEqual(stats.judge_calls_failed, 0)
        self.assertEqual(stats.streams_active, HELD_PROFILE.n)
        # The ramp is real: episodes land in BOTH phases, so the steady phase was actually held
        # rather than the whole run being a ramp.
        self.assertGreater(stats.episodes_by_phase["ramp"], 0)
        self.assertGreater(stats.episodes_by_phase["steady"], stats.episodes_by_phase["ramp"])
        self.assertAlmostEqual(stats.elapsed_s, HELD_SCHEDULE.total_s, places=6)

    def test_the_projection_reports_the_drivers_own_counts(self):
        funnel = stage_funnel(self.run.records, manifest=self.run.manifest)
        self.assertEqual(funnel.counts["observed"], self.run.stats.frames_emitted)
        self.assertEqual(funnel.counts["admitted"], self.run.stats.episodes_admitted)
        self.assertEqual(funnel.counts["judge_started"], self.run.stats.judge_calls_issued)
        self.assertEqual(self.run.recorder.lost_records, 0)

    def test_the_funnel_identity_holds_under_load(self):
        """observed >= admitted >= judge_started >= judge_completed >= fired, at fan-out 1.

        This is the spine every stage metric hangs off. A step whose downstream point OUTNUMBERS its
        upstream one is reported as a NEGATIVE ``dropped``, which is what makes the violation
        visible rather than a silently impossible ratio.
        """
        funnel = stage_funnel(self.run.records, manifest=self.run.manifest)
        counts = [funnel.counts[p] for p in ("observed", "admitted", "judge_started",
                                             "judge_completed", "fired")]
        self.assertEqual(counts, sorted(counts, reverse=True), funnel.describe())
        for step in funnel.steps:
            with self.subTest(step=step.name):
                self.assertGreaterEqual(step.dropped, 0, step.describe())
                self.assertIsNotNone(step.ratio)
                self.assertLessEqual(step.ratio, 1.0, step.describe())

    def test_a_fire_that_never_became_an_event_moves_the_last_funnel_step(self):
        """MUTATION for the identity assertion above. Withhold every third incident and the outcome
        step must lose exactly those units — if it does not, the identity check is reading a
        constant rather than the trace."""
        clean = stage_funnel(held_run().records)
        broken = stage_funnel(held_run(undelivered_every=3).records)
        self.assertEqual(clean.step("outcome").dropped, 0)
        self.assertGreater(broken.step("outcome").dropped, 0)
        self.assertLess(broken.counts["fired"], clean.counts["fired"])
        self.assertLess(broken.step("outcome").ratio, 1.0)
        # and every OTHER step is untouched, so the miss is attributed to delivery alone
        for name in ("admission", "dispatch", "judge"):
            self.assertEqual(clean.step(name).dropped, broken.step(name).dropped)


# ---------------------------------------------------------------------------
# 2. Uncensored before quoted
# ---------------------------------------------------------------------------
class TestLatencyIsUncensoredBeforeAnyPercentileIsReported(unittest.TestCase):
    """A percentile taken from a truncated sample describes the truncation, not the system.

    The rule is wired, not re-derived: ``StageAttribution.percentile`` refuses, and this class
    proves the refusal is CONDITIONAL — it fires when a cause exists and stops when the cause is
    removed. An unconditional refusal would be as useless as no refusal.
    """

    def test_a_clean_held_run_is_publishable_and_yields_percentiles(self):
        att = _attr(held_run())
        self.assertTrue(att.publishable, att.describe())
        self.assertEqual(att.unpublishable_segments, ())
        for segment in att.measured_segments:
            with self.subTest(segment=segment):
                self.assertTrue(att.is_publishable(segment))
                self.assertGreaterEqual(att.percentile(segment, 95.0), 0.0)

    def test_a_declared_deadline_censors_the_segment_it_bounded(self):
        """MUTATION. Same trace, one declared deadline — the judge percentile must stop being
        quotable, and the whole report must stop being publishable with it."""
        run = held_run()
        clean = _attr(run)
        self.assertTrue(clean.publishable)
        censored = _attr(run, deadlines={"judge": 1.5})
        self.assertFalse(censored.publishable)
        self.assertIn("judge", censored.unpublishable_segments)
        with self.assertRaises(CensoredLatencyError):
            censored.percentile("judge", 95.0)
        # the censored numbers are RETAINED as the evidence for why a deadline must move
        self.assertIsNotNone(censored.distribution("judge").p95_s)
        self.assertTrue(censored.distribution("judge").is_censored)

    def test_a_span_that_never_closed_censors_without_any_declaration(self):
        """The second, independent ground for refusal: the trace itself shows calls that could not
        finish, so the sample silently excludes exactly the slow ones a p95 is meant to describe.
        No deadline is declared here — the evidence is in the records."""
        att = _attr(held_run(unfinished_every=4))
        self.assertTrue(att.truncation["judge"].is_truncated)
        self.assertGreater(att.truncation["judge"].unclosed, 0)
        self.assertFalse(att.is_publishable("judge"))
        with self.assertRaises(CensoredLatencyError):
            att.percentile("judge", 50.0)

    def test_removing_the_cause_restores_the_percentile(self):
        """The refusal must be conditional. Same construction, no unfinished spans, and the judge
        percentile is quotable again."""
        att = _attr(held_run(unfinished_every=0))
        self.assertFalse(att.truncation["judge"].is_truncated)
        self.assertEqual(att.truncation["judge"].unclosed, 0)
        self.assertTrue(att.is_publishable("judge"))
        self.assertAlmostEqual(att.percentile("judge", 95.0), JUDGE_HEAVY_LATENCY.value_s, places=6)

    def test_a_censored_segment_blocks_the_ranking_too(self):
        """Naming the slowest stage from the segments that are still quotable would point a fix at
        the wrong stage — the censored one could be the slowest and there is no way to know."""
        att = _attr(held_run(), deadlines={"judge": 1.5})
        with self.assertRaises(CensoredLatencyError):
            att.slowest_segment(95.0)

    def test_an_unmeasured_segment_has_no_number_rather_than_a_zero(self):
        """Not measured is not zero. The projection never emits ``source.audio`` or a health
        transition, so segments with no sample must be ABSENT."""
        att = _attr(held_run(unfinished_every=1))  # every judge span left open
        self.assertIn("judge", att.unmeasured_segments)
        self.assertNotIn("judge", att.latencies)
        with self.assertRaises(UnmeasuredSegmentError):
            att.percentile("judge", 95.0)


# ---------------------------------------------------------------------------
# 3. A miss names its stage
# ---------------------------------------------------------------------------
class TestAMissNamesItsStage(unittest.TestCase):
    """Two runs with the SAME end-to-end cost and a different stage eating it.

    This is the mutation that decides whether the attribution is worth having: if both runs produce
    the same table, the split is decorative. The whole-pipe equality is asserted alongside, because
    it is the proof that the per-segment split — not some incidental difference between the runs —
    is carrying the signal.
    """

    def setUp(self):
        self.slow_judge = _attr(held_run(costs=JUDGE_HEAVY_COSTS, latency=JUDGE_HEAVY_LATENCY))
        self.slow_queue = _attr(held_run(costs=DISPATCH_HEAVY_COSTS, latency=DISPATCH_HEAVY_LATENCY))

    def test_the_whole_pipe_figure_is_blind_to_the_difference(self):
        a = self.slow_judge.percentile("end_to_end", 95.0)
        b = self.slow_queue.percentile("end_to_end", 95.0)
        self.assertAlmostEqual(a, b, places=6)
        self.assertAlmostEqual(a, 1.229, places=6)

    def test_the_per_segment_split_is_not_blind(self):
        self.assertEqual(self.slow_judge.slowest_segment(95.0), "judge")
        self.assertEqual(self.slow_queue.slowest_segment(95.0), "dispatch")

    def test_every_segment_carries_its_own_declared_cost(self):
        """Each segment's p95 is the cost that stage was actually given — so a report that names a
        stage is naming it for a reason a reader can check."""
        att = self.slow_judge
        expected = {
            "intake": JUDGE_HEAVY_COSTS.intake,
            "admission": JUDGE_HEAVY_COSTS.admission,
            "dispatch": JUDGE_HEAVY_COSTS.dispatch,
            "judge": JUDGE_HEAVY_LATENCY.value_s,
            "classification": JUDGE_HEAVY_COSTS.classification,
            "delivery": JUDGE_HEAVY_COSTS.delivery,
        }
        for name, value in expected.items():
            with self.subTest(segment=name):
                self.assertAlmostEqual(att.percentile(name, 95.0), value, places=6)

    def test_the_segments_are_separate_populations_and_do_not_add_up(self):
        """A guard against reading the table as a per-incident decomposition. The intake sample has
        one entry per FRAME and the judge sample one per JUDGED GOAL; they are different
        populations, and their sizes say so."""
        att = self.slow_judge
        n_intake = att.distribution("intake").n
        n_judge = att.distribution("judge").n
        self.assertGreater(n_intake, n_judge)
        self.assertEqual(n_intake, self.slow_judge.funnel.counts["observed"])
        self.assertEqual(n_judge, self.slow_judge.funnel.counts["judge_completed"])


# ---------------------------------------------------------------------------
# 4. The launch-gate percentile pair
# ---------------------------------------------------------------------------
class TestTheLaunchGatePercentilePair(unittest.TestCase):
    """The pair "median at or under 2.5s AND 95th at or under 2.0s" cannot both bind.

    Percentiles are monotone, so any distribution clearing the 95th-percentile ceiling of 2.0s has
    already cleared a median ceiling of 2.5s. The median leg can therefore never be the reason a run
    fails: it detects nothing the tail leg has not already caught. That is a property of the two
    NUMBERS, not of any run, so no amount of load can make it bind — it is reported as a
    calibration item rather than encoded here as a bar this file could pass.
    """

    TAIL_CEILING_S = 2.0
    MEDIAN_CEILING_S = 2.5

    def _distribution(self, value_s: float):
        att = _attr(held_run(latency=LatencyProfile(kind="fixed", value_s=value_s)))
        return att.distribution("end_to_end")

    def test_the_median_leg_cannot_fail_while_the_tail_leg_passes(self):
        for value_s in (0.01, 0.5, 1.2, 1.9, 2.4, 3.0):
            with self.subTest(judge_latency_s=value_s):
                dist = self._distribution(value_s)
                self.assertLessEqual(dist.p50_s, dist.p95_s)  # monotone, by construction
                tail_ok = dist.p95_s <= self.TAIL_CEILING_S
                median_ok = dist.p50_s <= self.MEDIAN_CEILING_S
                self.assertTrue(
                    median_ok or not tail_ok,
                    "a run cleared the tail ceiling and missed the median ceiling — that would "
                    "make the median leg binding, which monotone percentiles forbid",
                )

    def test_the_check_is_not_vacuous_a_tighter_median_ceiling_does_bind(self):
        """MUTATION. Move the median ceiling BELOW the tail ceiling and the same walk produces a run
        that passes the tail leg and fails the median leg — so the assertion above is a statement
        about these two numbers, not a tautology about any two numbers."""
        binding = [
            value_s
            for value_s in (0.01, 0.5, 1.2, 1.9)
            if self._distribution(value_s).p95_s <= self.TAIL_CEILING_S
            and self._distribution(value_s).p50_s > 1.0
        ]
        self.assertTrue(binding, "no run separates a 1.0s median ceiling from the 2.0s tail ceiling")

    def test_a_percentile_carries_the_conditions_it_was_measured_under(self):
        """A number with no conditions attached is not a fact about a system. In particular the
        model field must name the DOUBLE, so nobody reads this as provider latency."""
        dist = _attr(held_run()).distribution("end_to_end")
        self.assertIn("fake_judge", dist.conditions.model)
        self.assertIn("run is not a corpus", dist.conditions.corpus)


# ---------------------------------------------------------------------------
# 5. The queue is bounded, and every drop is declared
# ---------------------------------------------------------------------------
class _CountingAdapter:
    """Wraps the judge double and records peak simultaneous occupancy. Instrumentation only —
    every latency and verdict still comes from ``FakeJudge``."""

    def __init__(self, judge: FakeJudge) -> None:
        self._judge = judge
        self.inflight = 0
        self.peak = 0
        self.entered = 0

    async def judge(self, condition, frame, **kwargs):
        self.inflight += 1
        self.entered += 1
        self.peak = max(self.peak, self.inflight)
        try:
            return await self._judge.judge(condition, frame, **kwargs)
        finally:
            self.inflight -= 1


def _pool_judge(latency_s: float = 0.05, seed: int = 3) -> FakeJudge:
    return FakeJudge(
        seed=seed,
        latency=LatencyProfile(kind="fixed", value_s=latency_s),
        simulate_sleep=True,  # a real await, so concurrent calls really do overlap
        script={"tok": DECISIVE, "tok-b": DECISIVE},
    )


def _capability(shapes, *, max_inflight: int) -> JudgeCapability:
    return JudgeCapability(
        evidence_shapes=frozenset(shapes),
        latency_slo_s=1.0,
        modes=frozenset({"reflex", "deliberate"}),
        max_inflight=max_inflight,
    )


class TestQueueBehaviourIsBoundedAndItsDropsAreDeclared(unittest.TestCase):
    """Occupancy is bounded and a refusal is inspectable — the real router, not a simulation.

    The router has no queue: an over-eager caller does not get slower, it gets a capacity-shaped
    refusal. That is a legitimate design, and it is only safe because the refusal is DECLARED. These
    tests hold both halves.
    """

    CAPACITY = 2
    OFFERED = 8

    def _run(self, *, receipt_limit: int = 256):
        judge = _pool_judge()
        adapter = _CountingAdapter(judge)
        pool = JudgePool(
            [JudgeRoute("only", adapter, _capability({"single_image"}, max_inflight=self.CAPACITY))],
            mode="reflex",
            deadline_s=5.0,
            hedge_after_s=0.0,
            receipt_limit=receipt_limit,
        )

        async def drive():
            return await asyncio.gather(
                *(pool.judge("is it so", "tok") for _ in range(self.OFFERED))
            )

        verdicts = asyncio.run(drive())
        return pool, adapter, verdicts

    def test_occupancy_never_exceeds_the_declared_bound(self):
        pool, adapter, _verdicts = self._run()
        self.assertEqual(adapter.peak, self.CAPACITY)
        self.assertEqual(pool.max_inflight, self.CAPACITY)
        self.assertLess(adapter.entered, self.OFFERED)  # the surplus really was refused

    def test_every_refusal_is_declared_with_a_named_outcome(self):
        pool, adapter, verdicts = self._run()
        receipts = pool.receipts()
        refused = [r for r in receipts if r.outcome == "capacity"]
        served = [r for r in receipts if r.outcome in ("decisive", "uncertain")]
        self.assertEqual(len(served), adapter.entered)
        self.assertEqual(len(refused), self.OFFERED - adapter.entered)
        self.assertEqual(len(receipts), self.OFFERED)
        # nothing landed in an unnamed bucket
        self.assertEqual({r.outcome for r in receipts}, {"decisive", "capacity"})

    def test_a_refusal_never_becomes_a_fire(self):
        """Bounded is not enough on its own: the refused calls must fail CLOSED, never return a
        satisfied verdict the gate would act on."""
        _pool, adapter, verdicts = self._run()
        unknowns = [v for v in verdicts if v.confidence == 0.0]
        self.assertEqual(len(unknowns), self.OFFERED - adapter.entered)
        for v in unknowns:
            self.assertFalse(v.satisfied)
            self.assertIn("judge pool unavailable", v.evidence)

    def test_raising_the_bound_moves_both_the_occupancy_and_the_refusals(self):
        """MUTATION. If the occupancy assertion is reading the bound rather than the run, widening
        the bound changes nothing. It must change both numbers."""
        self.CAPACITY = 4
        try:
            pool, adapter, _verdicts = self._run()
        finally:
            self.CAPACITY = 2
        self.assertEqual(adapter.peak, 4)
        self.assertEqual(len([r for r in pool.receipts() if r.outcome == "capacity"]),
                         self.OFFERED - adapter.entered)

    def test_the_receipt_ring_says_how_many_declarations_it_dropped(self):
        """REWRITTEN (was ``test_the_receipt_ring_drops_the_declarations_without_saying_so``).

        It characterized the defect: the ring evicted silently, so under sustained load — this
        file's whole subject — the evidence that a drop was declared was itself dropped, and nothing
        on the pool could tell a reader how many refusals it no longer had. The ring is still
        bounded (that half was never wrong); what it now also does is ACCOUNT for the bound, the way
        ADR-001 §9 requires and the trace recorder's ``trace_dropped`` marker already does.

        The companion assertion in ``test_every_refusal_is_declared_with_a_named_outcome`` only
        holds because that run offers fewer decisions than the ring holds — which is exactly the
        distinction the drop count now makes readable.
        """
        pool, adapter, _verdicts = self._run(receipt_limit=3)
        receipts = pool.receipts()
        self.assertEqual(len(receipts), 3)  # this run made 8 decisions

        # the refusals that really happened still outnumber the refusals on record ...
        really_refused = self.OFFERED - adapter.entered
        self.assertGreater(really_refused, len([r for r in receipts if r.outcome == "capacity"]))

        # ... and the difference is now accounted for rather than lost. Retained + dropped is every
        # decision the pool made, so a reader can reconstruct what the ring no longer holds.
        snapshot = pool.snapshot()
        self.assertEqual(snapshot["receipts_dropped"], self.OFFERED - len(receipts))
        self.assertEqual(pool.receipts_dropped(), snapshot["receipts_dropped"])
        self.assertEqual(len(receipts) + snapshot["receipts_dropped"], self.OFFERED)

        # the drop carries a REASON, not just a number (ADR-001 §9: never a silent counter bump)
        self.assertEqual(snapshot["receipts_dropped_reason"], "receipt_limit")
        self.assertEqual(snapshot["receipt_limit"], 3)

        # and it is attributed to the route whose declarations were lost
        self.assertEqual(
            sum(state["receipts_dropped"] for state in snapshot["routes"].values()),
            snapshot["receipts_dropped"],
        )
        self.assertEqual(snapshot["routes"]["only"]["receipts_dropped"], snapshot["receipts_dropped"])

    def test_a_ring_that_held_everything_reports_no_drop(self):
        """MUTATION. If the count above were reading the offered total rather than the evictions,
        a ring wide enough to hold every decision would still report drops. It must report none —
        otherwise 'this pool lost declarations' is a constant, not a measurement."""
        pool, _adapter, _verdicts = self._run(receipt_limit=64)
        snapshot = pool.snapshot()
        self.assertEqual(len(pool.receipts()), self.OFFERED)
        self.assertEqual(snapshot["receipts_dropped"], 0)
        self.assertIsNone(snapshot["receipts_dropped_reason"])
        self.assertEqual(snapshot["routes"]["only"]["receipts_dropped"], 0)


# ---------------------------------------------------------------------------
# 6. Isolation
# ---------------------------------------------------------------------------
class TestOneWatchCannotConsumeAnothersReservedCapacity(unittest.TestCase):
    """The unit of reserved capacity that EXISTS is the route, not the watch.

    A watch whose evidence only one route can serve is isolated from a watch saturating a different
    route: that route's declared occupancy is reserved to it. Two watches sharing a route are NOT
    isolated — there is no per-watch reservation anywhere in the tree — and the second test states
    that plainly rather than leaving it implied.
    """

    def _two_route_pool(self, *, share_shape: bool):
        """``share_shape`` widens route B to also accept route A's evidence shape — the mutation
        that should destroy the isolation the first test asserts."""
        judge_a = _pool_judge(latency_s=0.08, seed=1)
        judge_b = _pool_judge(latency_s=0.01, seed=2)
        adapter_a = _CountingAdapter(judge_a)
        adapter_b = _CountingAdapter(judge_b)
        shapes_b = {"ordered_images", "single_image"} if share_shape else {"ordered_images"}
        pool = JudgePool(
            [
                JudgeRoute("route-a", adapter_a, _capability({"single_image"}, max_inflight=1)),
                JudgeRoute("route-b", adapter_b, _capability(shapes_b, max_inflight=1)),
            ],
            mode="reflex",
            deadline_s=5.0,
            hedge_after_s=0.0,
            max_parallel=4,
        )
        return pool, adapter_a, adapter_b

    def _drive(self, pool, *, saturating: int = 6):
        async def go():
            noisy = [pool.judge("watch a", "tok") for _ in range(saturating)]
            quiet = [pool.judge("watch b", "tok-b", frames=["f0", "f1"])]
            return await asyncio.gather(*noisy, *quiet)

        return asyncio.run(go())

    def test_a_saturating_watch_cannot_take_the_other_routes_slot(self):
        pool, adapter_a, adapter_b = self._two_route_pool(share_shape=False)
        verdicts = self._drive(pool)
        quiet_verdict = verdicts[-1]
        self.assertTrue(quiet_verdict.satisfied)
        self.assertGreaterEqual(quiet_verdict.confidence, 0.7)
        self.assertEqual(adapter_b.entered, 1)
        self.assertEqual(adapter_b.peak, 1)
        # every capacity refusal in the run belongs to the saturated route
        refused = [r for r in pool.receipts() if r.outcome == "capacity"]
        self.assertGreater(len(refused), 0)
        self.assertEqual({r.route for r in refused}, {"route-a"})

    def test_widening_the_second_route_destroys_that_isolation(self):
        """MUTATION. Let route B also serve route A's evidence shape and the noisy watch's
        decisions start scheduling route B too — B's reserved slot is consumed by A's traffic, and
        the quiet watch that owns it is refused. If the assertion above survived this, it was not
        testing isolation."""
        isolated_pool, _a1, _b1 = self._two_route_pool(share_shape=False)
        isolated = self._drive(isolated_pool)[-1]
        shared_pool, _a2, _b2 = self._two_route_pool(share_shape=True)
        shared = self._drive(shared_pool)[-1]

        # the quiet watch's OWN answer flips from a real verdict to a fail-closed unknown
        self.assertTrue(isolated.satisfied)
        self.assertFalse(shared.satisfied)
        self.assertEqual(shared.confidence, 0.0)

        # and the refusals move onto the route that used to be reserved
        def refused_on(pool, route):
            return len([r for r in pool.receipts() if r.outcome == "capacity" and r.route == route])

        self.assertEqual(refused_on(isolated_pool, "route-b"), 0)
        self.assertGreater(refused_on(shared_pool, "route-b"), 0)

    def test_two_watches_on_one_route_contend_and_the_loser_is_told(self):
        """The honest negative. Reserved capacity is per ROUTE; nothing in the tree reserves it per
        WATCH. Two watches sharing a route do consume each other's capacity — what the router
        guarantees is that the loser is refused loudly, not that it is protected."""
        judge = _pool_judge(latency_s=0.05)
        adapter = _CountingAdapter(judge)
        pool = JudgePool(
            [JudgeRoute("shared", adapter, _capability({"single_image"}, max_inflight=1))],
            mode="reflex",
            deadline_s=5.0,
            hedge_after_s=0.0,
        )

        async def go():
            return await asyncio.gather(
                pool.judge("watch a", "tok"),
                pool.judge("watch b", "tok"),
            )

        verdicts = asyncio.run(go())
        self.assertEqual(adapter.peak, 1)
        losers = [v for v in verdicts if v.confidence == 0.0]
        self.assertEqual(len(losers), 1)
        self.assertIn("capacity", {r.outcome for r in pool.receipts()})


# ---------------------------------------------------------------------------
# 7. Economy — what a load run can and cannot say about cost
# ---------------------------------------------------------------------------
class TestEconomyRefusesToInventANumber(unittest.TestCase):
    """A cost bar must not be passable by a zero that was never measured.

    This file cannot state a cost RATIO against any provider baseline: the judge double reports no
    usage and there is no matched-quality comparison in the tree. What it can hold is that the
    absence is typed as unknown rather than as free, which is the property that stops a load run
    from passing a cost ceiling it never measured.
    """

    def test_calls_were_made_and_the_cost_reads_unknown_not_zero(self):
        run = held_run()
        summary = stage_funnel(run.records, manifest=run.manifest).summary
        self.assertGreater(summary.judge_completed, 0)
        self.assertIsNone(summary.cost_usd)
        self.assertIsNone(summary.input_tokens)
        self.assertIsNone(summary.dollars_per_stream_hour)
        self.assertEqual(summary.cost_text(), "—")

    def test_metering_the_same_run_makes_the_cost_a_real_number(self):
        """MUTATION for the assertion above. If ``cost_usd`` read ``None`` unconditionally, the
        no-false-zero check would be decorative. Meter the same run through the recorder's own
        usage seam against a catalogued model and the figure appears — so the ``None`` is a
        statement about THIS run, not about the field."""
        run = held_run(usage_model="claude-haiku-4-5")
        summary = stage_funnel(run.records, manifest=run.manifest).summary
        self.assertGreater(summary.usage_reported_calls, 0)
        self.assertIsNotNone(summary.cost_usd)
        self.assertGreater(summary.cost_usd, 0.0)
        self.assertIsNotNone(summary.input_tokens)
        self.assertIsNotNone(summary.dollars_per_stream_hour)
        self.assertNotEqual(summary.cost_text(), "—")

    def test_an_uncatalogued_model_is_priced_as_unknown_not_as_free(self):
        """Tokens are known and the price is not. The run reports what it knows and nothing it
        doesn't — a guessed rate here is what would let a cost ceiling pass on a fiction."""
        run = held_run(usage_model="a-model-nobody-catalogued")
        summary = stage_funnel(run.records, manifest=run.manifest).summary
        self.assertGreater(summary.input_tokens, 0.0)
        self.assertIsNone(summary.cost_usd)
        self.assertEqual(summary.priced_calls, 0)

    def test_a_partial_aggregate_supply_is_refused_rather_than_discarded(self):
        """FIND-LOADSTAGE-07. ``derive`` recounts ``counters``/``usage``/``evidence_windows``/
        ``unhealthy_s`` as ONE group when ``counters`` is omitted, so supplying any of the other
        three alone used to have no effect at all — the value was rebound before it was read, with
        no error and no warning.

        A harness with its own usage totals therefore got ``cost_usd=None`` back, byte-identical to
        the same call with no ``usage`` argument, and read it as "this run could not be priced". In
        a module whose central rule is that an unknown must never be reported as a measured zero,
        silently answering a MEASURED question with "unknown" is the same failure in a quieter
        register. Refusing names the mistake; discarding hides it.
        """
        run = held_run()
        metered = {
            "calls": 10, "input_tokens": 1000.0, "output_tokens": 100.0,
            "priced_calls": 10, "cost_usd": 0.42,
        }
        with self.assertRaises(ValueError) as caught:
            TraceSummary.derive(run.manifest, list(run.records), usage=metered)
        self.assertIn("counters", str(caught.exception))
        self.assertIn("usage", str(caught.exception))

        # The same refusal for the other two members of the group, so the rule is the group's and
        # not one argument's.
        with self.assertRaises(ValueError):
            TraceSummary.derive(run.manifest, list(run.records), evidence_windows=[(0.0, 1.0)])
        with self.assertRaises(ValueError):
            TraceSummary.derive(run.manifest, list(run.records), unhealthy_s=3.0)

    def test_the_whole_group_together_is_honoured_and_the_plain_recount_still_works(self):
        """The control that proves the refusal above is about the PARTIAL supply, not about the
        arguments being unusable: hand ``derive`` the complete group and it uses every part of it,
        and omit the group entirely and it recounts as it always has."""
        run = held_run()
        recounted = TraceSummary.derive(run.manifest, list(run.records))
        self.assertIsNone(recounted.cost_usd)  # this run reported no usage — unknown, not free

        supplied = TraceSummary.derive(
            run.manifest,
            list(run.records),
            counters={"judge_completed": 10, "admitted": 10, "observed": 10},
            usage={
                "calls": 10, "input_tokens": 1000.0, "output_tokens": 100.0,
                "priced_calls": 10, "cost_usd": 0.42,
            },
        )
        self.assertAlmostEqual(supplied.cost_usd, 0.42)
        self.assertAlmostEqual(supplied.input_tokens, 1000.0)
        self.assertEqual(supplied.usage_reported_calls, 10)

    def test_savings_is_none_without_a_declared_baseline(self):
        run = held_run()
        summary = stage_funnel(run.records, manifest=run.manifest).summary
        self.assertIsNone(summary.estimated_baseline_calls)
        self.assertIsNone(summary.estimated_saved_calls)
        self.assertEqual(summary.savings_text(), "—")

    def test_a_declared_baseline_moves_the_call_count_and_still_refuses_the_tokens(self):
        """MUTATION for the assertion above: declaring a baseline must change the call figure. The
        token figure stays unknown because no call reported usage — there is no measured mean to
        project onto the avoided calls, and inventing one is the fabrication this rule forbids."""
        run = held_run(baseline=BaselineAssumption.every_observation(5000))
        summary = stage_funnel(run.records, manifest=run.manifest).summary
        self.assertEqual(summary.estimated_baseline_calls, 5000)
        self.assertEqual(summary.estimated_saved_calls, 5000 - summary.judge_completed)
        self.assertIsNone(summary.estimated_saved_tokens)
        self.assertEqual(summary.savings_text(), "—")

    def test_admission_economy_is_the_drivers_own_ratio(self):
        """The share of observed units that never reached a model. Reported as a fraction; no floor
        is asserted, because on a synthetic stream the admission rate is a parameter this file set
        and a floor asserted against it would pass forever."""
        run = held_run()
        funnel = stage_funnel(run.records, manifest=run.manifest)
        expected = 1.0 - (run.stats.episodes_admitted / run.stats.frames_emitted)
        self.assertAlmostEqual(funnel.admission_economy, expected, places=9)
        self.assertGreater(funnel.admission_economy, 0.0)

    def test_a_zero_denominator_has_no_ratio_at_all(self):
        """``0.0`` and ``1.0`` are both numbers an economy bar would happily pass. With nothing
        observed there is no ratio, and the funnel says so."""
        manifest = RunTraceManifest.create(
            run_id="empty", policy=TracePolicy(level="full"), wall_clock=lambda: 0.0
        )
        funnel = stage_funnel((), manifest=manifest)
        self.assertIsNone(funnel.step("admission").ratio)
        self.assertIsNone(funnel.admission_economy)


# ---------------------------------------------------------------------------
# 8. Cross-cutting: the cache must not silently rescue a latency miss
# ---------------------------------------------------------------------------
YES = Verdict(satisfied=True, confidence=0.95, evidence="scripted")


class _SlowVision:
    """A judge that takes a real, chosen amount of wall time — and counts how often it is spent."""

    def __init__(self, delay_s: float) -> None:
        self.delay_s = delay_s
        self.calls = 0

    async def judge(self, condition, frame, **kwargs):
        self.calls += 1
        await asyncio.sleep(self.delay_s)
        return YES

    async def detect(self, frame, goals):
        return []


def _cached_run(*, delay_s: float = 0.15, ticks: int = 6):
    """Judge the SAME frame repeatedly with the perception cache on, tracing every tick."""
    import numpy as np

    settings = Settings(
        gate=GateSettings(threshold=0.7, validity_window_s=1000.0),
        cache=CacheSettings(enabled=True, threshold=0.0, ttl_s=1000.0, audit_rate=0.0),
    )
    vision = _SlowVision(delay_s)
    agent = Percept.create(vision=vision, settings=settings, discover_plugins=False)
    agent.add_goal(Goal(id="cup", condition="a cup is on the table", say="cup"))
    manifest = RunTraceManifest.create(
        run_id="cache-load", policy=TracePolicy(level="full"), wall_clock=lambda: 0.0
    )
    rec = TraceRecorder(manifest)
    agent.on_trace = rec.tracer()
    frame = np.zeros((16, 16, 3), dtype=np.uint8)

    async def go():
        for i in range(ticks):
            await agent.perceive_judged(frame, t=float(i))

    asyncio.run(go())
    return agent, vision, rec, manifest


class TestTheCacheMustNotSilentlyRescueALatencyMiss(unittest.TestCase):
    """A warm cache made a slow judge look fast, and nothing in the trace said which was which.

    The runtime always distinguished the two: it records judge latency only for a tick that actually
    spent a provider call. The TRACE did not — a cache-served tick emits the same
    ``judge.started``/``judge.finished`` pair as a real call — so every consumer reading the judge
    span out of a trace mixed sub-millisecond cache serves into a provider-latency percentile. Under
    sustained load with a warm cache that is exactly a latency miss being rescued silently.
    FIND-LOADSTAGE-01, fixed: the closing record now carries the shared zero-spend marker
    (``percept.tracing.judge_call_was_served``) and :func:`judge_provider_spans` drops those pairs.
    """

    def test_the_runtime_knows_which_ticks_actually_spent_a_call(self):
        agent, vision, _rec, _manifest = _cached_run()
        metrics = agent.metrics.snapshot()
        self.assertEqual(vision.calls, 1)
        self.assertEqual(metrics["cache.hit"], 5)
        self.assertEqual(metrics["judge.latency.count"], 1)
        self.assertEqual(metrics["funnel.judged"], 6)

    def test_the_judge_span_counts_only_the_calls_that_were_spent(self):
        """The judge span is documented as "the model call itself". Five of these six ticks are
        cache serves that spent no model call, and the trace now separates them."""
        _agent, vision, rec, _manifest = _cached_run()
        self.assertEqual(len(judge_provider_spans(rec.records)), vision.calls)

    def test_the_reported_judge_median_is_the_providers_not_the_caches(self):
        """The distortion this class was filed for, measured after the fix.

        Was a CHARACTERIZATION recording the size of the error: six spans, ``p50`` 0.000144 s
        against a real provider call of 0.1516 s — a ~1000x understatement that ``is_publishable``
        happily quoted. The population is now the one provider call that was actually made, so the
        reported median IS the provider's.
        """
        _agent, vision, rec, _manifest = _cached_run(delay_s=0.15)
        spans = judge_provider_spans(rec.records)
        dist = measure_judge_latency(spans, conditions=CONDITIONS)
        self.assertEqual(len(spans), vision.calls)
        self.assertGreater(min(spans), 0.1)  # every retained span is the real call
        self.assertGreater(dist.p50_s, 0.1)  # ...so the median cannot be dragged to the cache
        self.assertTrue(dist.is_publishable)  # and it is honest to quote

    def test_the_serves_are_marked_rather_than_missing(self):
        """The spans are dropped because the trace SAYS they were served, not because the records
        were withheld. All six ticks are still in the funnel — a served tick is a real observation
        of the stream — and the marker names which mechanism served each one."""
        _agent, vision, rec, _manifest = _cached_run()
        closes = [r for r in rec.records if r.stage == STAGE_JUDGE_FINISHED]
        self.assertEqual(len(closes), 6)
        served = [r for r in closes if judge_call_was_served(r.data)]
        self.assertEqual(len(served), 6 - vision.calls)
        self.assertEqual({judge_served_from(r.data) for r in served}, {"cache"})
        self.assertEqual(rec.summary().judge_completed, 6)
        self.assertEqual(rec.summary().judge_cache_hits, 5)


# ---------------------------------------------------------------------------
# 9. Cross-cutting: the funnel identity and the fan-out shape
# ---------------------------------------------------------------------------
class TestFanOutIsNotADrop(unittest.TestCase):
    """More than one watch on one frame makes the judged COUNT outnumber the admitted count.

    ``source.frame`` is emitted once per admitted unit of evidence; ``judge.started`` once per goal.
    A load profile with two watches per stream — the ordinary case, and the shape this whole file
    drives — therefore produces a trace whose ``judge_started`` count is twice its ``admitted``
    count. That is fan-out, not loss, and the two must not read the same: the dispatch step restates
    the admitted count in the judged unit before comparing, so ordinary multi-watch traffic reports
    a drop of zero and a real record loss still reports a positive one.
    """

    def _two_watch_records(self):
        vision = _SlowVision(0.0)
        settings = Settings(gate=GateSettings(threshold=0.7, validity_window_s=100.0))
        agent = Percept.create(vision=vision, settings=settings, discover_plugins=False)
        agent.add_goal(Goal(id="cup", condition="a cup is on the table", say="cup"))
        agent.add_goal(Goal(id="person", condition="a person is present", say="person"))
        manifest = RunTraceManifest.create(
            run_id="two-watch", policy=TracePolicy(level="full"), wall_clock=lambda: 0.0
        )
        rec = TraceRecorder(manifest)
        agent.on_trace = rec.tracer()

        async def go():
            for i in range(3):
                await agent.perceive_judged(f"frame-{i}", t=float(i))

        asyncio.run(go())
        return rec.records, manifest

    def test_the_trace_that_does_it_is_well_formed(self):
        """The precondition that makes the next assertion mean something: this recorder lost
        nothing, so the count really is what the runtime emitted."""
        records, manifest = self._two_watch_records()
        funnel = stage_funnel(records, manifest=manifest)
        self.assertEqual(funnel.counts["admitted"], 3)
        self.assertEqual(funnel.counts["judge_started"], 6)
        self.assertEqual(funnel.summary.records_dropped, 0)
        self.assertEqual(funnel.concerns, 2, "the trace names both watches")

    def test_a_well_formed_trace_never_reports_a_negative_drop(self):
        """A negative drop means records were lost. Two watches on one source are not a loss."""
        records, manifest = self._two_watch_records()
        funnel = stage_funnel(records, manifest=manifest)
        for step in funnel.steps:
            self.assertGreaterEqual(step.dropped, 0, step.describe())
        dispatch = funnel.step("dispatch")
        self.assertEqual(dispatch.dropped, 0)
        self.assertEqual(dispatch.ratio, 1.0)
        self.assertEqual(dispatch.upstream_count, 3, "the raw admitted count is not restated away")
        self.assertEqual(dispatch.upstream_dispatches, 6, "3 units offered to 2 concerns")

        # and ONLY that step. The other three compare two counts already in the same unit, so
        # multiplying them by the concern count would invent a drop where the trace shows none —
        # the mirror-image error of the one being fixed.
        for step in funnel.steps:
            if step.name == "dispatch":
                continue
            with self.subTest(step=step.name):
                self.assertEqual(step.fan_out, 1, step.describe())
                self.assertEqual(step.upstream_dispatches, step.upstream_count)

    def test_a_real_record_loss_on_the_same_step_is_still_reported(self):
        """MUTATION, and the reason the fix is a restatement rather than a clamp. Drop one
        ``judge.started`` from the same two-watch trace and the dispatch step must show the miss —
        if fan-out and loss both read zero, the signal has been thrown away rather than corrected."""
        records, manifest = self._two_watch_records()
        lossy = [r for r in records if not (r.stage == STAGE_JUDGE_STARTED and r.goal_id == "cup")]
        funnel = stage_funnel(lossy, manifest=manifest)
        self.assertEqual(funnel.concerns, 2, "the lost stage is still named by the goal's others")
        self.assertEqual(funnel.step("dispatch").dropped, 3)
        self.assertAlmostEqual(funnel.step("dispatch").ratio, 0.5, places=9)

    def test_the_fanned_out_load_profile_agrees(self):
        """REWRITTEN (was ``test_the_fanned_out_load_profile_shows_the_same_shape``, a
        CHARACTERIZATION of the break). The same fan-out reached through the load driver's knob
        rather than through two watches, so the correction is visible as a load-profile property and
        not only as an agent one: the raw counts still stand in a 2:1 ratio and the step reports no
        drop."""
        run = held_run(profile=LoadProfile(n=2, m=2, e_per_stream_min=120.0, f=2))
        funnel = stage_funnel(run.records, manifest=run.manifest)
        self.assertEqual(funnel.counts["judge_started"], funnel.counts["admitted"] * 2)
        self.assertEqual(funnel.concerns, 2)
        self.assertEqual(funnel.step("dispatch").dropped, 0)
        self.assertAlmostEqual(funnel.step("dispatch").ratio, 1.0, places=9)


if __name__ == "__main__":
    unittest.main()
