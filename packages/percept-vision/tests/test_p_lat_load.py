"""PROBE SCAFFOLD — P-LAT-LOAD-1 (Viability §7, board card A28, the real G3 gate — DUAL, under load).

**Scaffold only — gated on bars-locked, no for-record run.** This proves the WIRING a future
for-record P-LAT-LOAD-1 run needs, not a real latency verdict about Percept itself: as the
``N/M/E/F`` load profile scales, can this harness compute BOTH G3-A (``admitted_ts ->
decision_emitted_ts``, ``JudgmentTiming.admitted_to_decision_s``) AND G3-B (``decision_eligible_ts
-> decision_emitted_ts``, the P-LAT-ACTIONABLE-1 ``ingest_to_decision_s`` substitution) p95s at
each load level, combine them into a single DUAL band, find the load level where that band first
degrades off GREEN (the "knee"), and confirm a fixed-load run shows no p95 drift between its first
and second half?

Per Viability §7, Reflex requires BOTH G3-A and G3-B < 2s (GREEN); 2-3s on either leg is YELLOW
(CONDITIONAL-GO / Reflex beta); >= 3s on either leg is RED (NO-GO for the Reflex claim, not for
Percept as a whole).

Reuses, never edits, the harness pieces named in the card: ``load_driver.py``
(``LoadDriver``/``LoadProfile``/``RampSchedule``), ``fake_judge.py``
(``FakeJudge``/``LatencyProfile``), ``metrics.py`` (``MetricsCollector``),
``percept.contracts.JudgmentTiming``, and ``percept.clock.FakeClock``.
``percept.harness.stream_gen.parametric_motion_clip`` is imported only because ``LoadDriver``
requires a ``Timeline`` to build its ``fan_out`` streams (same pattern as ``tests/test_load_driver.py``
and ``tests/test_p_lat_runtime.py``). This file follows the G3-A synthetic-timing pattern from
``tests/test_p_lat_runtime.py`` and the G3-B eligible-latency substitution from
``tests/test_p_lat_actionable.py`` (both read for pattern only — neither test module is imported).

**Why judge/admission latency are hand-set PARAMETERS, not measured.** ``eval/viability/
reference_config.yaml`` has ``bars_locked: false`` and every N/M/E/F and latency-threshold field is
still ``TODO`` — there is no locked reference load profile or provider latency to replay yet. The
``LOAD_LEVELS`` below are placeholder PARAMETERS (N/M/E/F magnitudes AND the per-level judge-call /
admission-delay latencies layered onto each admission) chosen by hand so this scaffold has a
non-trivial, hand-verifiable GREEN -> GREEN -> YELLOW -> RED walk to exercise the knee-finder and
banding logic against. A for-record run replaces these placeholders with the bars-locked values and
a real provider-latency distribution; it does not change this file's wiring.

**Single-instant funnel, same simplification as P-LAT-RUNTIME-1/P-LAT-ACTIONABLE-1.** Every
admission's ``admitted_ts`` is pinned to ``0.0`` (zero judge-wait, ``judge_started_ts ==
admitted_ts``) and ``ingest_observed_ts`` stands in for ``decision_eligible_ts`` (P-LAT-ACTIONABLE-1's
substitution — ``JudgmentTiming`` has no ``decision_eligible_ts`` field by design). This isolates the
judge-funnel + admission-delay legs the property cares about; queueing/backpressure under contention
is P-BOUND-1's concern and is not duplicated here.
"""

from __future__ import annotations

import asyncio
import unittest
from dataclasses import dataclass

from percept.backends.fake_judge import FakeJudge, LatencyProfile
from percept.clock import FakeClock
from percept.contracts import JudgmentTiming
from percept.harness.load_driver import LoadDriver, LoadDriverStats, LoadProfile, RampSchedule
from percept.harness.metrics import LatencyReport, MetricsCollector
from percept.harness.stream_gen import parametric_motion_clip

# ---------------------------------------------------------------------------
# Viability §7 per-leg band thresholds (same GREEN/YELLOW/RED convention as P-LAT-RUNTIME-1 /
# P-LAT-ACTIONABLE-1) + the DUAL combined gate this probe adds on top of them.
# ---------------------------------------------------------------------------

GREEN_MAX_S = 2.0  # exclusive upper bound of GREEN
YELLOW_MAX_S = 3.0  # exclusive upper bound of YELLOW; >= this is RED

# Fixed, tiny gate/reducer overhead layered onto judge_completed_ts -> decision_emitted_ts, same
# constant P-LAT-RUNTIME-1 uses, so every synthetic JudgmentTiming here has a nonzero (but
# negligible relative to the 2s/3s band thresholds) decision_overhead_s.
_DECISION_OVERHEAD_S = 0.05


def classify_latency_band(p95_s: float) -> str:
    """Viability §7 single-leg banding: GREEN < 2s, YELLOW 2s<=x<3s, RED >= 3s."""
    if p95_s < GREEN_MAX_S:
        return "GREEN"
    if p95_s < YELLOW_MAX_S:
        return "YELLOW"
    return "RED"


def combined_band(g3a_p95_s: float, g3b_p95_s: float) -> str:
    """The real G3 gate — DUAL: GREEN only if BOTH G3-A and G3-B are < 2s; RED if EITHER is >= 3s
    (checked first, so a RED leg always dominates a YELLOW one on the other leg); otherwise
    YELLOW. Reflex requires GREEN on this combined band, not just on either leg alone."""
    band_a = classify_latency_band(g3a_p95_s)
    band_b = classify_latency_band(g3b_p95_s)
    if "RED" in (band_a, band_b):
        return "RED"
    if "YELLOW" in (band_a, band_b):
        return "YELLOW"
    return "GREEN"


def find_knee(levels_with_bands: list[tuple[str, str]]) -> str | None:
    """Return the name of the first load level (in the given, presumably load-ascending, order)
    whose combined band leaves GREEN — the point where the DUAL gate first degrades under load.
    ``None`` if every level in the input stayed GREEN (no knee found in the tested range)."""
    for name, band in levels_with_bands:
        if band != "GREEN":
            return name
    return None


def latency_drift_within_tolerance(
    first_half_p95_s: float, second_half_p95_s: float, *, tolerance_s: float = 0.2
) -> bool:
    """Viability §7 'no time-drift at fixed load': True iff a fixed-load run's second-half p95 is
    within ``tolerance_s`` of its first-half p95, in either direction."""
    return abs(second_half_p95_s - first_half_p95_s) <= tolerance_s


def _timeline(n: int = 8, dt: float = 1.0):
    return parametric_motion_clip(w=4, h=4, n=n, motion_frames={1, 3}, dt=dt)


# ---------------------------------------------------------------------------
# LoadLevel — one point on the load-scaling curve: a real N/M/E/F profile (driven through
# LoadDriver) plus the two PARAMETER latencies (judge-call latency, synthetic admission delay)
# layered onto every admission it produces.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LoadLevel:
    name: str
    profile: LoadProfile
    schedule: RampSchedule
    judge_latency_s: float  # PARAMETER — grows the G3-A leg
    admission_delay_s: float  # PARAMETER — grows the G3-B leg on top of G3-A (late tier-0 admission)
    step_s: float = 0.25


# Four placeholder load levels, N/M/E/F scaling up together with the layered latency PARAMETERS,
# hand-chosen to walk GREEN -> GREEN -> YELLOW -> RED (a non-trivial knee for the tests below to
# find). Real N/M/E/F magnitudes + latency distributions are bars-locked inputs pending
# eval/viability/reference_config.yaml (currently bars_locked: false, all-TODO).
LOAD_LEVELS: tuple[LoadLevel, ...] = (
    LoadLevel(
        name="light",
        profile=LoadProfile(n=1, m=1, e_per_stream_min=60.0, f=1),
        schedule=RampSchedule(ramp_s=0.0, steady_s=5.0),
        judge_latency_s=0.3,
        admission_delay_s=0.05,
    ),
    LoadLevel(
        name="moderate",
        profile=LoadProfile(n=2, m=2, e_per_stream_min=90.0, f=2),
        schedule=RampSchedule(ramp_s=0.5, steady_s=4.0),
        judge_latency_s=1.0,
        admission_delay_s=0.3,
    ),
    LoadLevel(
        name="heavy",
        profile=LoadProfile(n=4, m=2, e_per_stream_min=120.0, f=3),
        schedule=RampSchedule(ramp_s=1.0, steady_s=4.0),
        judge_latency_s=1.7,
        admission_delay_s=0.5,
    ),
    LoadLevel(
        name="peak",
        profile=LoadProfile(n=8, m=3, e_per_stream_min=180.0, f=4),
        schedule=RampSchedule(ramp_s=1.0, steady_s=6.0),
        judge_latency_s=3.2,
        admission_delay_s=1.0,
    ),
)

# A dedicated, longer-steady-phase level for the no-drift check — same shape as "moderate" but held
# steady long enough to admit plenty of episodes to split cleanly into a first and second half.
DRIFT_TEST_LEVEL = LoadLevel(
    name="fixed-load-drift-check",
    profile=LoadProfile(n=2, m=2, e_per_stream_min=90.0, f=2),
    schedule=RampSchedule(ramp_s=0.5, steady_s=20.0),
    judge_latency_s=1.0,
    admission_delay_s=0.3,
)


def _run_level(level: LoadLevel, *, seed: int) -> tuple[list[JudgmentTiming], LoadDriverStats]:
    """Drive LoadDriver+FakeJudge to completion at `level`'s N/M/E/F + schedule (fixed judge
    latency, ``simulate_sleep=False``, matching P-LAT-RUNTIME-1's pattern so no test pays real wall
    time), then fold each successful judge call into a synthetic JudgmentTiming carrying BOTH legs:
    ``admitted_to_decision_s()`` reads as G3-A, ``ingest_to_decision_s()`` (``ingest_observed_ts``
    standing in for ``decision_eligible_ts``, the P-LAT-ACTIONABLE-1 trick) reads as G3-B.
    """
    clock = FakeClock()
    judge = FakeJudge(
        seed=seed,
        latency=LatencyProfile(kind="fixed", value_s=level.judge_latency_s),
        simulate_sleep=False,
    )
    driver = LoadDriver(level.profile, level.schedule, _timeline(), clock=clock, judge=judge)
    stats = asyncio.run(driver.run_to_completion(step_s=level.step_s))
    assert stats.judge_calls_failed == 0

    timings: list[JudgmentTiming] = []
    admitted_ts = 0.0  # single-instant funnel, same simplification as P-LAT-RUNTIME-1
    for call in judge.calls:
        if call.injected is not None:
            continue
        eligible_ts = admitted_ts - level.admission_delay_s  # decision_eligible_ts substitution
        judge_completed_ts = admitted_ts + call.latency_s
        decision_emitted_ts = judge_completed_ts + _DECISION_OVERHEAD_S
        timings.append(
            JudgmentTiming(
                ingest_observed_ts=eligible_ts,
                admitted_ts=admitted_ts,
                judge_started_ts=admitted_ts,
                judge_completed_ts=judge_completed_ts,
                decision_emitted_ts=decision_emitted_ts,
            )
        )
    return timings, stats


def _report_for(timings: list[JudgmentTiming]) -> LatencyReport:
    collector = MetricsCollector(FakeClock())
    for t in timings:
        collector.record_judgment(t)
    return collector.latency_summary()


def _levels_report(level: LoadLevel, *, seed: int) -> tuple[LatencyReport, LoadDriverStats]:
    timings, stats = _run_level(level, seed=seed)
    return _report_for(timings), stats


# ---------------------------------------------------------------------------
# Banding logic, hand-verified against known synthetic values, independent of the load driver.
# ---------------------------------------------------------------------------


class TestBandingLogicOnKnownSyntheticSet(unittest.TestCase):
    def test_single_leg_boundary_values(self):
        self.assertEqual(classify_latency_band(0.0), "GREEN")
        self.assertEqual(classify_latency_band(1.999), "GREEN")
        self.assertEqual(classify_latency_band(2.0), "YELLOW")  # GREEN is a strict "< 2s"
        self.assertEqual(classify_latency_band(2.999), "YELLOW")
        self.assertEqual(classify_latency_band(3.0), "RED")  # YELLOW is a strict "< 3s"
        self.assertEqual(classify_latency_band(10.0), "RED")

    def test_dual_gate_requires_both_legs_green(self):
        self.assertEqual(combined_band(0.5, 0.8), "GREEN")
        self.assertEqual(combined_band(2.0, 0.5), "YELLOW")  # G3-A alone crosses into YELLOW
        self.assertEqual(combined_band(0.5, 2.5), "YELLOW")  # G3-B alone crosses into YELLOW
        self.assertEqual(combined_band(2.5, 2.5), "YELLOW")  # both YELLOW -> still YELLOW
        self.assertEqual(combined_band(3.0, 0.5), "RED")  # G3-A alone hits RED -> dominates
        self.assertEqual(combined_band(0.5, 3.0), "RED")  # G3-B alone hits RED -> dominates
        self.assertEqual(combined_band(2.5, 3.5), "RED")  # RED beats YELLOW
        self.assertEqual(combined_band(3.0, 3.0), "RED")


# ---------------------------------------------------------------------------
# The load-scaling scenario itself: BOTH G3-A and G3-B p95 at each LOAD_LEVELS point, plus the knee.
# ---------------------------------------------------------------------------


class TestLoadScalingDualGate(unittest.TestCase):
    """Runs LoadDriver at each LOAD_LEVELS point, computes BOTH G3-A and G3-B p95 through the real
    MetricsCollector aggregation path, and confirms each lands in its hand-computed combined band —
    walking GREEN -> GREEN -> YELLOW -> RED as N/M/E/F (and the layered latency PARAMETERS) scale
    up, exactly the property Viability §7 asks P-LAT-LOAD-1 to probe."""

    # name -> (expected G3-A p95, expected G3-B p95, expected combined band), hand-computed from
    # each LoadLevel's judge_latency_s / admission_delay_s / _DECISION_OVERHEAD_S:
    #   G3-A = judge_latency_s + _DECISION_OVERHEAD_S
    #   G3-B = admission_delay_s + judge_latency_s + _DECISION_OVERHEAD_S
    _EXPECTED = {
        "light": (0.35, 0.40, "GREEN"),
        "moderate": (1.05, 1.35, "GREEN"),
        "heavy": (1.75, 2.25, "YELLOW"),
        "peak": (3.25, 4.25, "RED"),
    }

    def test_each_level_matches_hand_computed_dual_p95_and_band(self):
        for idx, level in enumerate(LOAD_LEVELS):
            with self.subTest(level=level.name):
                report, stats = _levels_report(level, seed=idx)

                # Sanity: the driver actually produced admissions, and the full N/M/E/F shape (not
                # just N x M) was exercised — same check as P-LAT-RUNTIME-1's load scenario.
                self.assertGreater(stats.episodes_admitted, 0)
                self.assertEqual(stats.judge_calls_issued, stats.episodes_admitted * level.profile.f)

                g3a = report.admitted_to_decision.p95
                g3b = report.ingest_to_decision.p95
                expected_g3a, expected_g3b, expected_band = self._EXPECTED[level.name]

                self.assertAlmostEqual(g3a, expected_g3a)
                self.assertAlmostEqual(g3b, expected_g3b)
                self.assertEqual(combined_band(g3a, g3b), expected_band)

    def test_knee_is_the_first_level_that_leaves_green(self):
        levels_with_bands = [
            (level.name, self._EXPECTED[level.name][2]) for level in LOAD_LEVELS
        ]
        self.assertEqual(
            levels_with_bands,
            [("light", "GREEN"), ("moderate", "GREEN"), ("heavy", "YELLOW"), ("peak", "RED")],
        )
        self.assertEqual(find_knee(levels_with_bands), "heavy")

    def test_knee_recomputed_live_from_driver_runs_matches_hand_computed_knee(self):
        """Same knee, but derived end-to-end from live LoadDriver runs (not the hand-computed
        table) — proving the wiring, not just the arithmetic."""
        levels_with_bands = []
        for idx, level in enumerate(LOAD_LEVELS):
            report, _stats = _levels_report(level, seed=idx)
            band = combined_band(
                report.admitted_to_decision.p95, report.ingest_to_decision.p95
            )
            levels_with_bands.append((level.name, band))
        self.assertEqual(find_knee(levels_with_bands), "heavy")

    def test_knee_is_none_when_every_level_stays_green(self):
        self.assertIsNone(find_knee([("a", "GREEN"), ("b", "GREEN")]))

    def test_knee_is_none_for_empty_input(self):
        self.assertIsNone(find_knee([]))


# ---------------------------------------------------------------------------
# No-time-drift-at-fixed-load: p95 in the second half of a fixed-load run ~= the first half.
# ---------------------------------------------------------------------------


class TestNoTimeDriftAtFixedLoad(unittest.TestCase):
    def test_drift_helper_hand_values(self):
        self.assertTrue(latency_drift_within_tolerance(1.50, 1.55, tolerance_s=0.2))
        self.assertTrue(latency_drift_within_tolerance(1.50, 1.50, tolerance_s=0.0))
        self.assertFalse(latency_drift_within_tolerance(1.50, 2.00, tolerance_s=0.2))
        self.assertFalse(latency_drift_within_tolerance(2.00, 1.50, tolerance_s=0.2))  # direction-agnostic

    def test_fixed_load_run_shows_no_drift_between_first_and_second_half(self):
        """Runs one load level's steady phase long enough to admit plenty of episodes, splits the
        resulting timings (in admission/call order, i.e. chronologically) into a first and second
        half, and confirms BOTH legs' p95 hold steady across the split — the property's explicit
        'no time-drift at fixed load' ask."""
        level = DRIFT_TEST_LEVEL
        timings, stats = _run_level(level, seed=7)
        self.assertGreater(stats.episodes_admitted, 0)
        self.assertGreaterEqual(len(timings), 20)  # enough samples for a meaningful split

        half = len(timings) // 2
        first_half, second_half = timings[:half], timings[half:]
        self.assertGreater(len(first_half), 0)
        self.assertGreater(len(second_half), 0)

        first_report = _report_for(first_half)
        second_report = _report_for(second_half)

        # Fixed judge latency + fixed admission delay -> every sample is identical -> exact,
        # hand-verifiable match in BOTH halves, zero drift.
        expected_g3a = level.judge_latency_s + _DECISION_OVERHEAD_S
        expected_g3b = level.admission_delay_s + level.judge_latency_s + _DECISION_OVERHEAD_S
        for report in (first_report, second_report):
            self.assertAlmostEqual(report.admitted_to_decision.p95, expected_g3a)
            self.assertAlmostEqual(report.ingest_to_decision.p95, expected_g3b)

        self.assertTrue(
            latency_drift_within_tolerance(
                first_report.admitted_to_decision.p95,
                second_report.admitted_to_decision.p95,
                tolerance_s=0.05,
            )
        )
        self.assertTrue(
            latency_drift_within_tolerance(
                first_report.ingest_to_decision.p95,
                second_report.ingest_to_decision.p95,
                tolerance_s=0.05,
            )
        )


if __name__ == "__main__":
    unittest.main()
