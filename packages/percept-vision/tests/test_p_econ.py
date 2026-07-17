"""Probe P-ECON-1 scaffold (Viability §7, board card A24) — G2 gate economy at held recall.

SCAFFOLD-ONLY: wires the probe's own sweep + pass/fail logic against fakes/synthetic fixtures and
asserts it. Does NOT run a for-record benchmark (that is gated on `bars-locked` —
``eval/viability/reference_config.yaml`` still reads ``bars_locked: false`` with every numeric bar
as a ``TODO`` placeholder at the time this scaffold was written). Property under test: can tier-0
(VID) cut judge volume enough to clear the cost ceiling WHILE HOLDING RECALL?

Reuses (imported, never edited):
  * ``percept.harness.golden_eval.{GroundTruthEpisode, build_golden_timeline, score}`` (card A14) —
    answers the RECALL half of the sweep: at a given admit policy, which ground-truth episodes get
    a predicted ``JudgmentEvent`` at all (an episode tier-0 filters out below its admit threshold
    never gets one -> a miss), scored via the frozen §5a event-level matching rule.
  * ``percept.harness.metrics.{MetricsCollector, CostSample}`` (card A13) — answers the COST/VOLUME
    half: ``cost_summary()`` for judgments-per-hour + $/stream-hour, ``fanout_summary()`` for the
    observed fan-out ``F`` per admission.
  * ``percept.harness.load_driver.{LoadDriver, LoadProfile, RampSchedule}`` (card A11) +
    ``percept.backends.fake_judge.FakeJudge`` (card A12) — drive the admit-policy sweep's judge-call
    VOLUME deterministically: at each swept admit threshold, the number of candidates that clear the
    threshold becomes that policy point's admitted-episodes-per-stream-minute (``E``), and
    ``LoadDriver`` issues exactly ``F`` judge calls per admission through ``FakeJudge`` so
    judgments/hour and $/stream-hour come from REUSED harness machinery, not hand-computed numbers.
  * ``percept.clock.FakeClock`` — the single injected time source; zero wall-clock waits anywhere in
    this file.

**What "admit policy" means in this scaffold.** Real tier-0 (VID) admission is a visual-change/ROI
detector (``percept/harness/tier0.py``) — NOT in this card's reuse list, so it is not imported or
reimplemented here. Instead, each synthetic candidate episode below carries a fixed, deterministic
"VID salience" score in ``[0, 1]``, and "the admit policy" is a single swept threshold: a candidate
is admitted to the judge iff ``salience >= threshold``. This is the minimum stand-in that lets the
sweep exercise the real trade-off tier-0 gating creates (stricter threshold -> fewer judge calls,
cheaper, but risks dropping a true-event candidate whose salience falls below it -> recall loss) —
proving the SWEEP MECHANICS and the PASS/FAIL DECISION LOGIC, never a claim about real VID recall.

**Bars-locked inputs — where they plug in.** ``recall_min`` / ``cost_ceiling_usd_per_stream_hour`` /
``naive_vlm_baseline_fps`` are Viability §2 numbers that come from
``eval/viability/reference_config.yaml``'s ``numeric_bars`` ONLY once ``bars_locked: true`` is set
(committed BEFORE results are inspected — CURRENT_PHASE's run-validity gate). They are NOT invented
here. ``evaluate_p_econ_1`` below takes them as keyword parameters defaulted to clearly-named
``PLACEHOLDER_*`` module constants (see the block right below this docstring) purely so this
scaffold's pass/fail LOGIC is exercisable and deterministic today; a real for-record P-ECON-1 run
MUST pass the actual locked values from ``reference_config.yaml`` explicitly and must never rely on
these defaults.
"""

from __future__ import annotations

import asyncio
import unittest
from dataclasses import dataclass

from percept.backends.fake_judge import FakeJudge
from percept.clock import FakeClock
from percept.contracts import GroundTruth
from percept.harness.golden_eval import GroundTruthEpisode, build_golden_timeline, score
from percept.harness.load_driver import LoadDriver, LoadProfile, RampSchedule
from percept.harness.metrics import CostSample, MetricsCollector
from percept.harness.stream_gen import parametric_motion_clip

# ---------------------------------------------------------------------------
# PLACEHOLDER bars-locked inputs — NOT real numbers. Real values come ONLY from
# eval/viability/reference_config.yaml's `numeric_bars` once `bars_locked: true` (see module
# docstring). These exist solely so the pass/fail LOGIC below is callable/testable today.
# ---------------------------------------------------------------------------

#: PLACEHOLDER for `numeric_bars.recall_min.<archetype>` (reference_config.yaml: currently `TODO`).
PLACEHOLDER_RECALL_MIN = 0.85

#: PLACEHOLDER for `numeric_bars.cost_ceiling_usd_per_stream_hour` (currently `TODO`).
PLACEHOLDER_COST_CEILING_USD_PER_STREAM_HOUR = 5.0

#: PLACEHOLDER for `numeric_bars.naive_vlm_baseline_fps` (currently `TODO`).
PLACEHOLDER_NAIVE_VLM_BASELINE_FPS = 1.0

#: PLACEHOLDER for a judge-pool throughput ceiling beyond which queueing would push latency past
#: budget. Not itself a named `numeric_bars` field yet — a real run derives it from the queue-depth
#: / starvation-bound tolerances (`starvation_bound_seconds`, `queue_age_slope_tolerance_...`) once
#: those are filled; here it is just another clearly-labeled placeholder.
PLACEHOLDER_MAX_JUDGMENTS_PER_HOUR_FOR_LATENCY_BUDGET = 20_000.0

#: SYNTHETIC unit judge-call cost used only to make this scaffold's dollars arithmetic concrete and
#: deterministic. Real per-call cost comes from the provider's actual billed rate at run time — not
#: invented here, and NOT a `reference_config.yaml` field itself (it is folded into the vendor's
#: real $/call, which the for-record run reads off actual billing).
SYNTHETIC_COST_PER_JUDGE_CALL_USD = 0.02

#: Constant fan-out (F, judge calls issued per admission) used across the whole sweep in this
#: scaffold — kept fixed so the sweep isolates the ADMIT-THRESHOLD variable cleanly.
SYNTHETIC_FANOUT_F = 2

#: One admitted-episodes-per-stream-minute unit of LoadDriver's `e_per_stream_min` maps to exactly
#: one admitted candidate over this many steady-state seconds (60s => `e_per_stream_min` numerically
#: equals the admitted candidate COUNT for this sweep point, no extra scaling needed).
STEADY_S = 60.0


def _naive_baseline_dollars_per_stream_hour(
    baseline_fps: float, cost_per_judge_call_usd: float = SYNTHETIC_COST_PER_JUDGE_CALL_USD
) -> float:
    """The naive-VLM baseline's $/stream-hour: sample-and-judge EVERY frame at `baseline_fps`, no
    tier-0 gating at all, continuously for a full stream-hour (3600s)."""
    calls_per_hour = baseline_fps * 3600.0
    return calls_per_hour * cost_per_judge_call_usd


# ---------------------------------------------------------------------------
# Synthetic dataset — known recall/cost so the sweep + pass/fail decision is deterministic.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SyntheticCandidate:
    """One synthetic tier-0 candidate: a fixed VID salience score, and (for a genuine event) the
    paired :class:`GroundTruthEpisode` it must fire against. ``ground_truth_episode is None`` marks
    a NOISE candidate — never scored for recall, but still costs a judge call if admitted (exactly
    the volume tier-0 gating is meant to cut)."""

    candidate_id: str
    salience: float
    ground_truth_episode: GroundTruthEpisode | None


def _true_event_candidates(n: int = 10) -> tuple[SyntheticCandidate, ...]:
    """`n` real events with salience spread across [0.50, 0.90] (deterministic, no randomness) —
    high enough that a lenient admit policy catches all of them, low enough that a sufficiently
    strict policy starts missing the weakest ones (the trade-off this probe sweeps)."""
    out = []
    for i in range(n):
        salience = 0.50 + 0.40 * (i / max(1, n - 1))
        start = 100.0 * i
        gt = GroundTruth.by_construction(
            event_start_ts=start,
            eligible_after_s=1.0,
            fixture_id=f"p-econ-1-true-{i}",
        )
        episode = GroundTruthEpisode(
            episode_id=f"true-{i}",
            ground_truth=gt,
            episode_end_ts=start + 5.0,
            allowed_lateness_s=2.0,
        )
        out.append(SyntheticCandidate(f"true-{i}", salience, episode))
    return tuple(out)


def _noise_candidates(n: int = 40) -> tuple[SyntheticCandidate, ...]:
    """`n` non-event candidates (no ground truth) with salience spread across the FULL [0, 1]
    range — these are what a lenient admit policy wastes judge calls on, and what a stricter policy
    correctly cuts without any recall cost."""
    out = []
    for i in range(n):
        salience = (i * 7 % n) / max(1, n - 1)  # deterministic scramble, no randomness
        out.append(SyntheticCandidate(f"noise-{i}", salience, None))
    return tuple(out)


SYNTHETIC_TRUE_EVENTS: tuple[SyntheticCandidate, ...] = _true_event_candidates()
SYNTHETIC_NOISE: tuple[SyntheticCandidate, ...] = _noise_candidates()
SYNTHETIC_CANDIDATES: tuple[SyntheticCandidate, ...] = SYNTHETIC_TRUE_EVENTS + SYNTHETIC_NOISE


def _timeline():
    """A short synthetic frame timeline for LoadDriver's `fan_out` plumbing. Content is irrelevant
    to this probe's cost/recall math (FakeJudge's calls carry a fixed synthetic condition string,
    never real frame content) — StreamGen.poll() simply returns no more frames once exhausted, so a
    short timeline is safe even though `STEADY_S` may exceed its span."""
    return parametric_motion_clip(w=4, h=4, n=50, motion_frames=set(range(0, 50, 5)), dt=1.0)


# ---------------------------------------------------------------------------
# The sweep — one point per admit threshold.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EconSweepPoint:
    """One admit-policy sweep point: the §7/G2 numbers a real P-ECON-1 run would plot."""

    admit_threshold: float
    episodes_admitted: int
    recall: float
    judgments_per_hour: float
    dollars_per_stream_hour: float
    fanout_f: float


def sweep_admit_policy(
    thresholds: "list[float]",
    candidates: "tuple[SyntheticCandidate, ...]" = SYNTHETIC_CANDIDATES,
    *,
    fanout_f: int = SYNTHETIC_FANOUT_F,
    cost_per_judge_call_usd: float = SYNTHETIC_COST_PER_JUDGE_CALL_USD,
    steady_s: float = STEADY_S,
    judge_seed: int = 0,
) -> tuple[EconSweepPoint, ...]:
    """Sweep ``thresholds`` (a VID admit-salience cutoff each) and, for each, compute
    ``(recall, judgments_per_hour, dollars_per_stream_hour, fanout_f)``.

    RECALL comes from :func:`~percept.harness.golden_eval.score` over ALL true-event ground-truth
    episodes vs. a golden-style predicted timeline built ONLY for the true events that cleared this
    threshold (an episode below threshold gets no predicted event at all -> a miss, exactly modeling
    "tier-0 never even admitted it to the judge").

    COST/VOLUME comes from driving :class:`~percept.harness.load_driver.LoadDriver` +
    :class:`~percept.backends.fake_judge.FakeJudge` for exactly as many admissions as cleared this
    threshold (both true-event AND noise candidates cost a judge call if admitted), then reading
    :class:`~percept.harness.metrics.MetricsCollector`'s ``cost_summary()``/``fanout_summary()``.
    """
    all_true_episodes = tuple(c.ground_truth_episode for c in candidates if c.ground_truth_episode)

    points = []
    for threshold in thresholds:
        admitted = [c for c in candidates if c.salience >= threshold]
        admitted_true_episodes = tuple(
            c.ground_truth_episode for c in admitted if c.ground_truth_episode is not None
        )
        total_admitted = len(admitted)

        # -- recall (golden_eval, card A14) --------------------------------------------------
        predicted_events = (
            build_golden_timeline(admitted_true_episodes) if admitted_true_episodes else ()
        )
        result = score(all_true_episodes, predicted_events)
        recall = result.recall

        # -- cost / volume (load_driver + fake_judge + metrics, cards A11/A12/A13) ------------
        clock = FakeClock()
        collector = MetricsCollector(clock)
        if total_admitted > 0:
            profile = LoadProfile(n=1, m=1, e_per_stream_min=float(total_admitted), f=fanout_f)
            schedule = RampSchedule(ramp_s=0.0, steady_s=steady_s)
            judge = FakeJudge(seed=judge_seed, simulate_sleep=False)
            driver = LoadDriver(profile, schedule, _timeline(), clock=clock, judge=judge)
            stats = asyncio.run(driver.run_to_completion(step_s=steady_s))

            collector.record_load_driver_stats(stats)
            cost_usd = stats.judge_calls_issued * cost_per_judge_call_usd
            collector.record_cost_sample(
                CostSample(
                    judgments=stats.judge_calls_issued,
                    cost_usd=cost_usd,
                    elapsed_s=stats.elapsed_s,
                    stream_hours=profile.n * stats.elapsed_s / 3600.0,
                )
            )
            fanout = collector.fanout_summary()
            cost = collector.cost_summary()
            episodes_admitted_observed = stats.episodes_admitted
            fanout_f_observed = fanout.mean_observed_f
            jph = cost.judgments_per_hour
            dph = cost.dollars_per_stream_hour
        else:
            episodes_admitted_observed = 0
            fanout_f_observed = 0.0
            jph = 0.0
            dph = 0.0

        points.append(
            EconSweepPoint(
                admit_threshold=threshold,
                episodes_admitted=episodes_admitted_observed,
                recall=recall,
                judgments_per_hour=jph,
                dollars_per_stream_hour=dph,
                fanout_f=fanout_f_observed,
            )
        )
    return tuple(points)


# ---------------------------------------------------------------------------
# Pass/fail decision — Viability §7 P-ECON-1 pass condition, evaluated at ONE sweep point.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EconGateResult:
    """The full pass/fail breakdown for one :class:`EconSweepPoint` against the bars — every
    sub-condition is exposed (not just the overall bool) so a failing decision is diagnosable."""

    point: EconSweepPoint
    recall_holds: bool
    under_cost_ceiling: bool
    cheaper_than_naive_baseline: bool
    within_latency_budget: bool
    naive_baseline_dollars_per_stream_hour: float

    @property
    def passed(self) -> bool:
        return (
            self.recall_holds
            and self.under_cost_ceiling
            and self.cheaper_than_naive_baseline
            and self.within_latency_budget
        )


def evaluate_p_econ_1(
    point: EconSweepPoint,
    *,
    recall_min: float = PLACEHOLDER_RECALL_MIN,
    cost_ceiling_usd_per_stream_hour: float = PLACEHOLDER_COST_CEILING_USD_PER_STREAM_HOUR,
    naive_vlm_baseline_fps: float = PLACEHOLDER_NAIVE_VLM_BASELINE_FPS,
    max_judgments_per_hour_for_latency_budget: float = (
        PLACEHOLDER_MAX_JUDGMENTS_PER_HOUR_FOR_LATENCY_BUDGET
    ),
    cost_per_judge_call_usd: float = SYNTHETIC_COST_PER_JUDGE_CALL_USD,
) -> EconGateResult:
    """The Viability §7 P-ECON-1 pass condition at one admit-policy sweep point:

        at recall_min: $/stream-hour <= ceiling AND admitted rate keeps latency within budget
        AND cheaper than the naive-VLM baseline (the whole point of tier-0 cutting judge volume).

    ``recall_min`` / ``cost_ceiling_usd_per_stream_hour`` / ``naive_vlm_baseline_fps`` are
    BARS-LOCKED inputs (see module docstring) — the defaults here are clearly-named
    ``PLACEHOLDER_*`` constants, never a claim about the real bars. A real for-record run MUST pass
    the actual values read from ``eval/viability/reference_config.yaml`` once ``bars_locked: true``.
    """
    naive_dph = _naive_baseline_dollars_per_stream_hour(
        naive_vlm_baseline_fps, cost_per_judge_call_usd
    )
    return EconGateResult(
        point=point,
        recall_holds=point.recall >= recall_min,
        under_cost_ceiling=point.dollars_per_stream_hour <= cost_ceiling_usd_per_stream_hour,
        cheaper_than_naive_baseline=point.dollars_per_stream_hour < naive_dph,
        within_latency_budget=(
            point.judgments_per_hour <= max_judgments_per_hour_for_latency_budget
        ),
        naive_baseline_dollars_per_stream_hour=naive_dph,
    )


def cheapest_point_holding_recall(
    points: "tuple[EconSweepPoint, ...]", *, recall_min: float
) -> "EconSweepPoint | None":
    """The operating point a real tier-0 admit policy would choose: among sweep points that HOLD
    recall (``recall >= recall_min``), the cheapest one. ``None`` if no swept point holds recall."""
    holding = [p for p in points if p.recall >= recall_min]
    if not holding:
        return None
    return min(holding, key=lambda p: p.dollars_per_stream_hour)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPEcon1Sweep(unittest.TestCase):
    """P-ECON-1 (Viability §7, G2): the sweep mechanics over a deterministic synthetic dataset."""

    THRESHOLDS = [0.0, 0.3, 0.5, 0.55, 0.65, 0.75, 0.85, 0.95, 1.01]

    def test_sweep_covers_admit_everything_to_admit_nothing(self):
        points = sweep_admit_policy(self.THRESHOLDS)
        self.assertEqual(len(points), len(self.THRESHOLDS))
        # threshold 0.0 admits every candidate (all salience >= 0.0).
        self.assertEqual(points[0].episodes_admitted, len(SYNTHETIC_CANDIDATES))
        # threshold > 1.0 (max possible salience) admits nothing.
        self.assertEqual(points[-1].episodes_admitted, 0)
        self.assertEqual(points[-1].recall, 0.0)
        self.assertEqual(points[-1].dollars_per_stream_hour, 0.0)
        self.assertEqual(points[-1].judgments_per_hour, 0.0)

    def test_recall_is_monotonically_non_increasing_as_threshold_rises(self):
        points = sweep_admit_policy(self.THRESHOLDS)
        for earlier, later in zip(points, points[1:]):
            self.assertLessEqual(later.recall, earlier.recall + 1e-9)

    def test_cost_and_volume_are_monotonically_non_increasing_as_threshold_rises(self):
        points = sweep_admit_policy(self.THRESHOLDS)
        for earlier, later in zip(points, points[1:]):
            self.assertLessEqual(later.dollars_per_stream_hour, earlier.dollars_per_stream_hour + 1e-9)
            self.assertLessEqual(later.judgments_per_hour, earlier.judgments_per_hour + 1e-9)
            self.assertLessEqual(later.episodes_admitted, earlier.episodes_admitted)

    def test_fanout_f_matches_configured_fanout_at_every_nonzero_admission_point(self):
        """LoadDriver issues exactly F judge calls per admission (its own docstring's exactness
        guarantee) — this probe's fanout_summary() must read that back exactly, at every threshold
        that admits at least one candidate."""
        points = sweep_admit_policy(self.THRESHOLDS)
        for p in points:
            if p.episodes_admitted > 0:
                self.assertEqual(p.fanout_f, float(SYNTHETIC_FANOUT_F))

    def test_recall_exactly_matches_the_fraction_of_true_events_admitted(self):
        """Sanity: in this synthetic dataset, recall at a threshold is EXACTLY (true events with
        salience >= threshold) / (total true events) — proves golden_eval.score is doing the
        expected thing here, not some other accounting."""
        points = sweep_admit_policy(self.THRESHOLDS)
        for threshold, point in zip(self.THRESHOLDS, points):
            expected_admitted_true = sum(
                1 for c in SYNTHETIC_TRUE_EVENTS if c.salience >= threshold
            )
            expected_recall = expected_admitted_true / len(SYNTHETIC_TRUE_EVENTS)
            self.assertAlmostEqual(point.recall, expected_recall, places=9)

    def test_admitting_everything_holds_full_recall(self):
        points = sweep_admit_policy([0.0])
        self.assertEqual(points[0].recall, 1.0)

    def test_a_threshold_above_the_weakest_true_event_drops_recall(self):
        """The weakest true event has salience 0.50 (see `_true_event_candidates`); a threshold
        strictly above it must miss that event and show recall < 1.0."""
        weakest_salience = min(c.salience for c in SYNTHETIC_TRUE_EVENTS)
        points = sweep_admit_policy([weakest_salience + 0.01])
        self.assertLess(points[0].recall, 1.0)


class TestPEcon1PassFailDecision(unittest.TestCase):
    """P-ECON-1 pass/fail logic (Viability §7 pass condition), asserted deterministically against
    the synthetic sweep — NOT a real bars-locked verdict (see module docstring)."""

    THRESHOLDS = [0.0, 0.3, 0.5, 0.55, 0.65, 0.75, 0.85, 0.95, 1.01]

    def setUp(self):
        self.points = sweep_admit_policy(self.THRESHOLDS)
        self.recall_min = 1.0  # require EVERY true event caught, for a crisp deterministic test

    def test_cheapest_point_holding_recall_is_a_stricter_admit_policy_than_admit_everything(self):
        chosen = cheapest_point_holding_recall(self.points, recall_min=self.recall_min)
        self.assertIsNotNone(chosen)
        admit_everything = self.points[0]
        self.assertEqual(chosen.recall, 1.0)
        # The whole point of a working admit policy: strictly less admission (and so strictly
        # cheaper) than the naive "admit everything" point, while still holding full recall.
        self.assertLessEqual(chosen.episodes_admitted, admit_everything.episodes_admitted)
        self.assertLessEqual(chosen.dollars_per_stream_hour, admit_everything.dollars_per_stream_hour)

    def test_gate_passes_at_the_cheapest_recall_holding_point_under_a_generous_ceiling(self):
        chosen = cheapest_point_holding_recall(self.points, recall_min=self.recall_min)
        assert chosen is not None
        result = evaluate_p_econ_1(
            chosen,
            recall_min=self.recall_min,
            # A ceiling comfortably above what the cheapest-recall-holding synthetic point costs,
            # and a naive baseline generous enough that the admit-policy point undercuts it.
            cost_ceiling_usd_per_stream_hour=1000.0,
            naive_vlm_baseline_fps=5.0,
            max_judgments_per_hour_for_latency_budget=1_000_000.0,
        )
        self.assertTrue(result.recall_holds)
        self.assertTrue(result.under_cost_ceiling)
        self.assertTrue(result.cheaper_than_naive_baseline)
        self.assertTrue(result.within_latency_budget)
        self.assertTrue(result.passed)

    def test_gate_fails_on_recall_when_admit_policy_is_too_strict_even_though_cheap(self):
        """A threshold that drops recall must fail the gate EVEN IF cost/latency look great — proves
        this isn't just a cost check wearing a recall label."""
        too_strict = next(p for p in self.points if p.admit_threshold == 0.95)
        self.assertLess(too_strict.recall, 1.0)  # sanity: this point really does miss events
        result = evaluate_p_econ_1(
            too_strict,
            recall_min=self.recall_min,
            cost_ceiling_usd_per_stream_hour=1000.0,
            naive_vlm_baseline_fps=5.0,
            max_judgments_per_hour_for_latency_budget=1_000_000.0,
        )
        self.assertFalse(result.recall_holds)
        self.assertFalse(result.passed)

    def test_gate_fails_on_cost_ceiling_when_admit_everything_is_too_expensive(self):
        """`admit-everything` (threshold 0.0) holds full recall but should fail against a ceiling set
        below its own cost — proves the ceiling check is load-bearing, not vacuous."""
        admit_everything = self.points[0]
        self.assertEqual(admit_everything.recall, 1.0)
        self.assertGreater(admit_everything.dollars_per_stream_hour, 0.0)
        stingy_ceiling = admit_everything.dollars_per_stream_hour / 2.0
        result = evaluate_p_econ_1(
            admit_everything,
            recall_min=self.recall_min,
            cost_ceiling_usd_per_stream_hour=stingy_ceiling,
            naive_vlm_baseline_fps=5.0,
            max_judgments_per_hour_for_latency_budget=1_000_000.0,
        )
        self.assertFalse(result.under_cost_ceiling)
        self.assertFalse(result.passed)

    def test_gate_fails_when_naive_baseline_is_cheaper_than_the_admit_policy(self):
        """Even a recall-holding, under-ceiling point must fail if it is NOT actually cheaper than
        the naive-VLM baseline — the entire premise of Viability §7/G2 is "does tier-0 gating beat
        naive sampling", not just "is it under some absolute ceiling"."""
        chosen = cheapest_point_holding_recall(self.points, recall_min=self.recall_min)
        assert chosen is not None
        self.assertGreater(chosen.dollars_per_stream_hour, 0.0)
        # An absurdly low naive baseline FPS makes the naive baseline cheaper than ANY judge-backed
        # admit policy -> cheaper_than_naive_baseline must flip false.
        result = evaluate_p_econ_1(
            chosen,
            recall_min=self.recall_min,
            cost_ceiling_usd_per_stream_hour=1000.0,
            naive_vlm_baseline_fps=1e-6,
            max_judgments_per_hour_for_latency_budget=1_000_000.0,
        )
        self.assertFalse(result.cheaper_than_naive_baseline)
        self.assertFalse(result.passed)

    def test_gate_fails_on_latency_budget_when_admitted_rate_is_too_high(self):
        """`admit-everything` drives more judgments/hour than a tight latency-budget ceiling allows
        -> within_latency_budget must flip false even with recall/cost/baseline all satisfied."""
        admit_everything = self.points[0]
        self.assertGreater(admit_everything.judgments_per_hour, 0.0)
        tight_budget = admit_everything.judgments_per_hour / 2.0
        result = evaluate_p_econ_1(
            admit_everything,
            recall_min=self.recall_min,
            cost_ceiling_usd_per_stream_hour=1000.0,
            naive_vlm_baseline_fps=5.0,
            max_judgments_per_hour_for_latency_budget=tight_budget,
        )
        self.assertFalse(result.within_latency_budget)
        self.assertFalse(result.passed)

    def test_placeholder_defaults_are_directly_callable_and_clearly_not_real_bars(self):
        """The function must be callable with ZERO explicit bars (placeholders only) — this is the
        "defaulted to placeholders a real run overrides" contract from the card. Not asserting a
        particular pass/fail outcome here (the placeholders are not real bars); only that the call
        succeeds and returns a well-formed, fully-populated result."""
        chosen = cheapest_point_holding_recall(self.points, recall_min=PLACEHOLDER_RECALL_MIN)
        assert chosen is not None
        result = evaluate_p_econ_1(chosen)
        self.assertIsInstance(result.passed, bool)
        self.assertGreater(result.naive_baseline_dollars_per_stream_hour, 0.0)


if __name__ == "__main__":
    unittest.main()
