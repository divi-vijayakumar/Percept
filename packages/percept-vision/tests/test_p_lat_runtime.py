"""PROBE SCAFFOLD — P-LAT-RUNTIME-1 (the release criteria, gate G3-A).

**Scaffold only — gated on bars-locked, no for-record run.** This proves the WIRING a future
for-record P-LAT-RUNTIME-1 run needs, not a real latency verdict about Percept itself:

* the ``admitted_ts -> decision_emitted_ts`` latency (``JudgmentTiming.admitted_to_decision_s``,
  ADR-001 §5's G3-A number) can be measured both AT REST (a single light stream/watch) and UNDER
  LOAD (the full N/M/E/F profile via :mod:`percept.harness.load_driver`), aggregated through
  :class:`~percept.harness.metrics.MetricsCollector`, and its p95 computed deterministically.
* that p95 classifies correctly into the the release criteria band: GREEN ``< 2s`` / YELLOW ``2-3s`` /
  RED ``>= 3s``.

Reuses, never edits, the already-landed harness pieces named in the card: ``load_driver.py``
(``LoadDriver``/``LoadProfile``/``RampSchedule``), ``fake_judge.py`` (``FakeJudge``/
``LatencyProfile``), ``metrics.py`` (``MetricsCollector``), ``percept.contracts.JudgmentTiming``,
and ``percept.clock.FakeClock``. ``percept.harness.stream_gen.parametric_motion_clip`` is imported
only because :class:`LoadDriver` requires a :class:`~percept.harness.stream_gen.Timeline` to
construct its ``fan_out`` streams (same pattern as the existing ``tests/test_load_driver.py``) —
no source file is edited by this test.

**Why timing is hand-assembled here, not read off the driver.** ``LoadDriver`` exercises the
N/M/E/F admission *shape* and routes each admission's F fan-out calls through ``FakeJudge`` (which
records real per-call sampled latency in ``FakeJudge.calls``), but it does not itself emit
``JudgmentTiming`` records — that stitching is exactly what this probe scaffold is for. Each
driver run is executed under ``simulate_sleep=False`` (matching ``tests/test_load_driver.py``) so
:class:`~percept.clock.FakeClock` stays the single source of "now" and no test pays real wall
time; the observed call latencies are then folded into synthetic (but load-driver-produced)
``JudgmentTiming`` records with a fixed, tiny decision/gate overhead layered on top of
``judge_completed_ts``.

Both scenarios (rest, load) use a single-instant admission funnel with zero judge-wait
(``judge_started_ts == admitted_ts``) — this scaffold isolates the *judge/model* leg of G3-A;
queueing/backpressure under contention is P-BOUND-1's concern (the implementation's own docstring), not
duplicated here.
"""

from __future__ import annotations

import asyncio
import unittest

from percept.backends.fake_judge import FakeJudge, LatencyProfile
from percept.clock import FakeClock
from percept.contracts import JudgmentTiming
from percept.harness.load_driver import LoadDriver, LoadProfile, RampSchedule
from percept.harness.metrics import MetricsCollector
from percept.harness.stream_gen import parametric_motion_clip

# ---------------------------------------------------------------------------
# the release criteria G3-A band thresholds — GREEN < 2s, YELLOW 2-3s, RED >= 3s.
# ---------------------------------------------------------------------------

GREEN_MAX_S = 2.0  # exclusive upper bound of GREEN
YELLOW_MAX_S = 3.0  # exclusive upper bound of YELLOW; >= this is RED

# Fixed, tiny gate/reducer overhead layered onto judge_completed_ts -> decision_emitted_ts, so the
# synthetic JudgmentTiming records below have a nonzero (but negligible relative to the band
# thresholds) decision_overhead_s, same as a real funnel would.
_DECISION_OVERHEAD_S = 0.05


def classify_latency_band(p95_s: float) -> str:
    """the release criteria G3-A banding: GREEN < 2s, YELLOW 2s<=x<3s, RED >= 3s."""
    if p95_s < GREEN_MAX_S:
        return "GREEN"
    if p95_s < YELLOW_MAX_S:
        return "YELLOW"
    return "RED"


def _timeline(n: int = 8, dt: float = 1.0):
    return parametric_motion_clip(w=4, h=4, n=n, motion_frames={1, 3}, dt=dt)


def _run_driver(
    profile: LoadProfile,
    schedule: RampSchedule,
    *,
    latency: LatencyProfile,
    seed: int,
    step_s: float = 0.5,
):
    """Drive a LoadDriver to completion under a FakeClock, zero wall-clock time paid (matches
    ``tests/test_load_driver.py``'s ``simulate_sleep=False`` pattern)."""
    clock = FakeClock()
    judge = FakeJudge(seed=seed, latency=latency, simulate_sleep=False)
    driver = LoadDriver(profile, schedule, _timeline(), clock=clock, judge=judge)
    stats = asyncio.run(driver.run_to_completion(step_s=step_s))
    return driver, stats, judge


def _timings_from_judge_calls(judge: FakeJudge) -> list[JudgmentTiming]:
    """Fold each recorded (successful) FakeJudge call's sampled latency into a synthetic
    JudgmentTiming: zero admission/queue wait, judge_duration == the driver-produced sampled
    latency, plus a fixed decision overhead. Injected-failure calls are skipped — a failed judge
    call never reaches decision_emitted (ADR-001 fail-closed path), so it has no admitted_to_decision
    latency to report here."""
    timings = []
    for call in judge.calls:
        if call.injected is not None:
            continue
        judge_completed = call.latency_s
        decision_emitted = judge_completed + _DECISION_OVERHEAD_S
        timings.append(
            JudgmentTiming(
                ingest_observed_ts=0.0,
                admitted_ts=0.0,
                judge_started_ts=0.0,
                judge_completed_ts=judge_completed,
                decision_emitted_ts=decision_emitted,
            )
        )
    return timings


class TestBandingLogicOnKnownSyntheticSet(unittest.TestCase):
    """Banding logic asserted directly against hand-authored, known latency values — independent
    of the load driver, per the card's explicit ask."""

    def test_boundary_values(self):
        self.assertEqual(classify_latency_band(0.0), "GREEN")
        self.assertEqual(classify_latency_band(1.999), "GREEN")
        self.assertEqual(classify_latency_band(2.0), "YELLOW")  # GREEN is a strict "< 2s"
        self.assertEqual(classify_latency_band(2.5), "YELLOW")
        self.assertEqual(classify_latency_band(2.999), "YELLOW")
        self.assertEqual(classify_latency_band(3.0), "RED")  # YELLOW is a strict "< 3s"
        self.assertEqual(classify_latency_band(10.0), "RED")

    def test_known_synthetic_latency_sets_via_metrics_collector(self):
        """Feed three known, hand-authored admitted->decision latency sets through
        MetricsCollector.latency_summary() (nearest-rank p95, see percept.harness.metrics) and
        confirm each lands in the expected band — GREEN, YELLOW, RED, one set each."""

        def _summary_p95(latencies_s: list[float]) -> float:
            collector = MetricsCollector(FakeClock())
            for lat in latencies_s:
                collector.record_judgment(
                    JudgmentTiming(
                        ingest_observed_ts=0.0,
                        admitted_ts=0.0,
                        judge_started_ts=0.0,
                        judge_completed_ts=lat,
                        decision_emitted_ts=lat,
                    )
                )
            return collector.latency_summary().admitted_to_decision.p95

        # nearest-rank p95 of n=5 values (rank = ceil(0.95*5) = 5, 1-indexed) is the max.
        green_p95 = _summary_p95([0.5, 1.0, 1.2, 1.5, 1.9])
        self.assertAlmostEqual(green_p95, 1.9)
        self.assertEqual(classify_latency_band(green_p95), "GREEN")

        yellow_p95 = _summary_p95([1.0, 1.5, 2.0, 2.2, 2.8])
        self.assertAlmostEqual(yellow_p95, 2.8)
        self.assertEqual(classify_latency_band(yellow_p95), "YELLOW")

        red_p95 = _summary_p95([1.0, 2.0, 3.0, 3.5, 4.0])
        self.assertAlmostEqual(red_p95, 4.0)
        self.assertEqual(classify_latency_band(red_p95), "RED")


class TestRestScenario(unittest.TestCase):
    """A single light stream/watch (N=1, M=1) at a low, fixed judge latency — the "at rest" half
    of the property. Expect a GREEN admitted->decision p95."""

    def test_rest_p95_is_computed_and_classified_green(self):
        profile = LoadProfile(n=1, m=1, e_per_stream_min=60.0, f=1)
        schedule = RampSchedule(ramp_s=0.0, steady_s=5.0)
        driver, stats, judge = _run_driver(
            profile,
            schedule,
            latency=LatencyProfile(kind="fixed", value_s=0.3),
            seed=0,
        )

        # Sanity: the driver actually produced admissions/judge calls to build timing from.
        self.assertGreater(stats.episodes_admitted, 0)
        self.assertEqual(stats.judge_calls_failed, 0)

        timings = _timings_from_judge_calls(judge)
        self.assertEqual(len(timings), stats.judge_calls_issued)

        collector = MetricsCollector(FakeClock())
        for t in timings:
            collector.record_judgment(t)

        report = collector.latency_summary()
        p95 = report.admitted_to_decision.p95
        # fixed 0.3s latency + fixed 0.05s overhead -> every sample is exactly 0.35s.
        self.assertAlmostEqual(p95, 0.35)
        self.assertEqual(classify_latency_band(p95), "GREEN")


class TestLoadScenario(unittest.TestCase):
    """The full N/M/E/F load profile (via LoadDriver) at a heavier, fixed judge latency — the
    "under load" half of the property. Expect a RED admitted->decision p95, illustrating the band
    the same funnel leg can degrade into once contention/slow judge calls stack up."""

    def test_load_p95_is_computed_and_classified_red(self):
        profile = LoadProfile(n=4, m=2, e_per_stream_min=120.0, f=3)
        schedule = RampSchedule(ramp_s=1.0, steady_s=4.0)
        driver, stats, judge = _run_driver(
            profile,
            schedule,
            latency=LatencyProfile(kind="fixed", value_s=3.2),
            seed=1,
            step_s=0.25,
        )

        self.assertGreater(stats.episodes_admitted, 0)
        self.assertEqual(stats.judge_calls_failed, 0)
        # Confirms the full N/M/E/F shape (not just N x M) was actually exercised.
        self.assertEqual(stats.judge_calls_issued, stats.episodes_admitted * profile.f)

        timings = _timings_from_judge_calls(judge)
        self.assertEqual(len(timings), stats.judge_calls_issued)

        collector = MetricsCollector(FakeClock())
        for t in timings:
            collector.record_judgment(t)

        report = collector.latency_summary()
        p95 = report.admitted_to_decision.p95
        # fixed 3.2s latency + fixed 0.05s overhead -> every sample is exactly 3.25s.
        self.assertAlmostEqual(p95, 3.25)
        self.assertEqual(classify_latency_band(p95), "RED")

    def test_rest_vs_load_bands_differ(self):
        """The property's whole point: the SAME funnel leg, measured at rest vs under load, can
        land in different bands. Re-derive both p95s here and assert they diverge."""
        rest_profile = LoadProfile(n=1, m=1, e_per_stream_min=60.0, f=1)
        rest_schedule = RampSchedule(ramp_s=0.0, steady_s=5.0)
        _, _, rest_judge = _run_driver(
            rest_profile,
            rest_schedule,
            latency=LatencyProfile(kind="fixed", value_s=0.3),
            seed=0,
        )
        rest_collector = MetricsCollector(FakeClock())
        for t in _timings_from_judge_calls(rest_judge):
            rest_collector.record_judgment(t)
        rest_p95 = rest_collector.latency_summary().admitted_to_decision.p95

        load_profile = LoadProfile(n=4, m=2, e_per_stream_min=120.0, f=3)
        load_schedule = RampSchedule(ramp_s=1.0, steady_s=4.0)
        _, _, load_judge = _run_driver(
            load_profile,
            load_schedule,
            latency=LatencyProfile(kind="fixed", value_s=3.2),
            seed=1,
            step_s=0.25,
        )
        load_collector = MetricsCollector(FakeClock())
        for t in _timings_from_judge_calls(load_judge):
            load_collector.record_judgment(t)
        load_p95 = load_collector.latency_summary().admitted_to_decision.p95

        self.assertEqual(classify_latency_band(rest_p95), "GREEN")
        self.assertEqual(classify_latency_band(load_p95), "RED")
        self.assertNotEqual(classify_latency_band(rest_p95), classify_latency_band(load_p95))


if __name__ == "__main__":
    unittest.main()
