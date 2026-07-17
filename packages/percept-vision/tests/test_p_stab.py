"""PROBE SCAFFOLD — P-STAB-1 (Viability §7, board card A20, gate G1 endurance/soak).

**Scaffold only — gated on bars-locked, no for-record run.** This proves the pass/fail WIRING a
future for-record P-STAB-1 run needs, not a real endurance verdict about Percept itself:
``eval/viability/reference_config.yaml`` still reads ``bars_locked: false`` with every numeric bar
(including this probe's three tolerances, under ``numeric_bars``) as a ``TODO`` placeholder at the
time this scaffold was written.

Property under test (Viability §7): over a long run the runtime survives without leaks/drift/crash.
The HONEST claim a soak can support is bounded, not absolute — a soak run of any finite length
cannot prove "indefinitely" stable, only that OVER THE SAMPLED HORIZON:

  * queue-age does not trend upward beyond ``ε_age`` (no positive slope past tolerance),
  * memory does not trend upward beyond ``ε_mem`` (no material slope past tolerance),
  * p95 latency does not drift beyond its own tolerance, hour-N vs hour-1,
  * state stays bounded by construction (the same latest-pending-supersession mechanism
    ``percept.harness.metrics.MetricsCollector`` already provides — see ``test_p_bound.py``,
    not re-proven here),
  * an injected provider-failure window shows CLEAN recovery (failures confined exactly to the
    injected window, zero residual failures either side of it).

This probe therefore asserts SLOPES AND TOLERANCES, never permanence.

Reuses (imported, never edited — no source file in this repo is touched by this test):
  * ``percept.harness.soak_runner.{SoakConfig, SoakRunner, compute_trend, assert_clean_recovery}``
    (card A15) — the hourly-snapshot soak driver and its hour-N-vs-hour-1 trend/recovery helpers.
    This probe adds ONLY the tolerance-comparison (pass/fail) layer on top; it does not reimplement
    any of the driving, snapshotting, or trend math.
  * ``percept.harness.load_driver.LoadProfile`` (card A11) — the N/M/E/F load shape passed into
    ``SoakConfig``.
  * ``percept.backends.fake_judge.{LatencyProfile, ErrorInjectionConfig}`` (card A12) — per-hour
    latency/error injection, via ``SoakConfig.latency_fn`` / ``errors_fn``.
  * ``percept.harness.metrics`` — reached only indirectly, through ``SoakRunner``/``compute_trend``.
  * ``percept.clock.FakeClock`` — the sole time source; every scenario below runs under a fresh
    ``FakeClock`` with ``simulate_sleep=False`` throughout the harness, so nothing here pays real
    wall-clock time and every run is bit-for-bit reproducible given the same seed/config.

**Bars-locked inputs — where they plug in.** ``eval/viability/reference_config.yaml``'s
``numeric_bars`` block names these three P-STAB-1 tolerances explicitly (all currently ``TODO``):

  * ``queue_age_slope_tolerance_percent_per_hour`` — ε_age, a RELATIVE (percent-of-hour-1-baseline)
    tolerance on ``SoakTrend.queue_age_slope_s_per_hour``.
  * ``memory_slope_tolerance_mb_per_hour`` — ε_mem, an ABSOLUTE tolerance (MB/hour) on
    ``SoakTrend.memory_slope_bytes_per_hour``.
  * ``latency_drift_tolerance_ms_p95`` — an ABSOLUTE tolerance (ms) on ``SoakTrend.latency_p95_drift_s``.

They are NOT invented here. ``evaluate_p_stab_1`` below takes them as keyword parameters defaulted
to clearly-named ``PLACEHOLDER_*`` module constants (block right below this docstring) purely so
this scaffold's pass/fail LOGIC is exercisable and deterministic today; a real for-record P-STAB-1
run MUST pass the actual locked values from ``reference_config.yaml`` explicitly and must never rely
on these defaults.

Not wired into ``percept/registry.py``. Downstream: a for-record P-STAB-1 run (this probe's logic,
against a real multi-hour run with the actual bars-locked tolerances) is follow-up work gated on
``bars_locked: true`` — out of scope for this scaffold.
"""

from __future__ import annotations

import asyncio
import unittest
from dataclasses import dataclass
from typing import Sequence

from percept.backends.fake_judge import ErrorInjectionConfig, LatencyProfile
from percept.clock import FakeClock
from percept.harness.load_driver import LoadProfile
from percept.harness.soak_runner import (
    HourSnapshot,
    SoakConfig,
    SoakRunner,
    SoakTrend,
    assert_clean_recovery,
    compute_trend,
)
from percept.harness.stream_gen import parametric_motion_clip

# ---------------------------------------------------------------------------
# PLACEHOLDER bars-locked inputs — NOT real numbers. Real values come ONLY from
# eval/viability/reference_config.yaml's `numeric_bars` once `bars_locked: true` (see module
# docstring). These exist solely so the pass/fail LOGIC below is callable/testable today.
# ---------------------------------------------------------------------------

#: PLACEHOLDER for `numeric_bars.queue_age_slope_tolerance_percent_per_hour` (currently `TODO`).
#: A RELATIVE tolerance: max allowed queue-age growth, expressed as a percent of the hour-1
#: baseline queue-age, per simulated hour.
PLACEHOLDER_QUEUE_AGE_SLOPE_TOLERANCE_PERCENT_PER_HOUR = 2.5

#: PLACEHOLDER for `numeric_bars.memory_slope_tolerance_mb_per_hour` (currently `TODO`).
PLACEHOLDER_MEMORY_SLOPE_TOLERANCE_MB_PER_HOUR = 50.0

#: PLACEHOLDER for `numeric_bars.latency_drift_tolerance_ms_p95` (currently `TODO`).
PLACEHOLDER_LATENCY_DRIFT_TOLERANCE_MS_P95 = 200.0

_BYTES_PER_MB = 1_000_000.0
_MS_PER_S = 1000.0

#: Floor used only when the hour-1 baseline queue-age is ~0 (a percent-of-baseline tolerance is
#: undefined/infinite against a zero baseline). In that case ANY material positive slope is
#: disallowed; this tiny absolute floor exists purely to avoid failing a truly-flat run on
#: floating-point noise. Not itself a `reference_config.yaml` field.
_ZERO_BASELINE_AGE_SLOPE_FLOOR_S_PER_HOUR = 1e-9


def _timeline(n: int = 8, dt: float = 1.0):
    return parametric_motion_clip(w=4, h=4, n=n, motion_frames={1, 3}, dt=dt)


def _base_profile() -> LoadProfile:
    # n=1 stream, m=1 watch, 30 admitted episodes/stream-minute (=> 0.5/sec/watch), f=2 fan-out.
    return LoadProfile(n=1, m=1, e_per_stream_min=30.0, f=2)


def _run_soak(config: SoakConfig) -> list[HourSnapshot]:
    clock = FakeClock()
    runner = SoakRunner(config, clock=clock)
    return asyncio.run(runner.run())


# ---------------------------------------------------------------------------
# Pass/fail decision — Viability §7 P-STAB-1 pass condition, evaluated over one soak run.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StabGateResult:
    """The full pass/fail breakdown for one soak run against the bars — every sub-condition is
    exposed (not just the overall bool) so a failing decision is diagnosable."""

    trend: SoakTrend
    allowed_queue_age_slope_s_per_hour: float
    allowed_memory_slope_bytes_per_hour: float
    allowed_latency_p95_drift_s: float
    queue_age_ok: bool
    memory_ok: bool
    latency_ok: bool
    recovery_ok: bool

    @property
    def passed(self) -> bool:
        return self.queue_age_ok and self.memory_ok and self.latency_ok and self.recovery_ok


def evaluate_p_stab_1(
    snapshots: Sequence[HourSnapshot],
    *,
    failure_hours: frozenset[int] = frozenset(),
    queue_age_slope_tolerance_percent_per_hour: float = (
        PLACEHOLDER_QUEUE_AGE_SLOPE_TOLERANCE_PERCENT_PER_HOUR
    ),
    memory_slope_tolerance_mb_per_hour: float = PLACEHOLDER_MEMORY_SLOPE_TOLERANCE_MB_PER_HOUR,
    latency_drift_tolerance_ms_p95: float = PLACEHOLDER_LATENCY_DRIFT_TOLERANCE_MS_P95,
) -> StabGateResult:
    """The Viability §7 P-STAB-1 pass condition over a whole soak run:

        no positive queue-age slope beyond ε_age, no material memory slope beyond ε_mem, no p95
        latency drift beyond tolerance (hour-N vs hour-1, all via ``compute_trend``), AND clean
        recovery from any injected provider-failure window (``assert_clean_recovery``).

    ``queue_age_slope_tolerance_percent_per_hour`` / ``memory_slope_tolerance_mb_per_hour`` /
    ``latency_drift_tolerance_ms_p95`` are BARS-LOCKED inputs (see module docstring) — the defaults
    here are clearly-named ``PLACEHOLDER_*`` constants, never a claim about the real bars. A real
    for-record run MUST pass the actual values read from ``eval/viability/reference_config.yaml``
    once ``bars_locked: true``.
    """
    trend = compute_trend(snapshots)

    baseline_age_s = snapshots[0].max_oldest_pending_age_s
    if baseline_age_s > 0:
        allowed_age_slope = (queue_age_slope_tolerance_percent_per_hour / 100.0) * baseline_age_s
    else:
        allowed_age_slope = _ZERO_BASELINE_AGE_SLOPE_FLOOR_S_PER_HOUR
    queue_age_ok = trend.queue_age_slope_s_per_hour <= allowed_age_slope

    allowed_mem_slope = memory_slope_tolerance_mb_per_hour * _BYTES_PER_MB
    memory_ok = trend.memory_slope_bytes_per_hour <= allowed_mem_slope

    allowed_latency_drift = latency_drift_tolerance_ms_p95 / _MS_PER_S
    latency_ok = trend.latency_p95_drift_s <= allowed_latency_drift

    recovery_ok = assert_clean_recovery(snapshots, failure_hours=failure_hours)

    return StabGateResult(
        trend=trend,
        allowed_queue_age_slope_s_per_hour=allowed_age_slope,
        allowed_memory_slope_bytes_per_hour=allowed_mem_slope,
        allowed_latency_p95_drift_s=allowed_latency_drift,
        queue_age_ok=queue_age_ok,
        memory_ok=memory_ok,
        latency_ok=latency_ok,
        recovery_ok=recovery_ok,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestStableScenarioPasses(unittest.TestCase):
    """(1) A STABLE multi-hour scenario — flat latency, flat injected RSS, no provider failures —
    must clear all three tolerances and report clean recovery (vacuously, no failure window)."""

    def test_flat_latency_flat_rss_no_failures_passes_the_gate(self):
        config = SoakConfig(
            hours=6,
            profile=_base_profile(),
            timeline=_timeline(),
            hour_s=20.0,
            tick_s=5.0,
            seed=0,
            latency_fn=lambda hour: LatencyProfile(kind="fixed", value_s=0.05),
            rss_fn=lambda hour: 100_000_000,
        )
        snapshots = _run_soak(config)
        self.assertEqual(len(snapshots), 6)
        # Sanity: the run actually did work every hour.
        self.assertTrue(all(s.episodes_admitted > 0 for s in snapshots))

        result = evaluate_p_stab_1(snapshots)
        self.assertTrue(result.queue_age_ok, result)
        self.assertTrue(result.memory_ok, result)
        self.assertTrue(result.latency_ok, result)
        self.assertTrue(result.recovery_ok, result)
        self.assertTrue(result.passed, result)

        # Exactly-flat scenario -> exactly-zero slopes/drift, well inside any positive tolerance.
        self.assertAlmostEqual(result.trend.queue_age_slope_s_per_hour, 0.0, delta=1e-9)
        self.assertAlmostEqual(result.trend.memory_slope_bytes_per_hour, 0.0, delta=1e-9)
        self.assertAlmostEqual(result.trend.latency_p95_drift_s, 0.0, delta=1e-9)

    def test_stable_run_is_bit_for_bit_deterministic_under_fake_clock(self):
        """Same config/seed, replayed twice under two independent FakeClocks, must produce
        identical snapshots — no real-time or unseeded-randomness leakage anywhere in the chain."""

        def _make_config() -> SoakConfig:
            return SoakConfig(
                hours=3,
                profile=_base_profile(),
                timeline=_timeline(),
                hour_s=20.0,
                tick_s=5.0,
                seed=0,
                latency_fn=lambda hour: LatencyProfile(kind="fixed", value_s=0.05),
                rss_fn=lambda hour: 100_000_000,
            )

        snapshots_a = _run_soak(_make_config())
        snapshots_b = _run_soak(_make_config())
        self.assertEqual(snapshots_a, snapshots_b)


class TestLeakDriftScenarioIsDetected(unittest.TestCase):
    """(2) A LEAK/DRIFT scenario — RSS and judge latency both growing hour-over-hour — must be
    DETECTED as a fail (so the probe has teeth, not just a rubber stamp)."""

    def test_growing_rss_and_latency_fails_the_gate(self):
        config = SoakConfig(
            hours=4,
            profile=_base_profile(),
            timeline=_timeline(),
            hour_s=20.0,
            tick_s=5.0,
            seed=0,
            # Judge latency grows every hour -> queue-age (oldest-pending-age at resolution, which
            # equals that hour's fixed latency in soak_runner's model) grows in lockstep. Hour 1 is
            # 0.1s, hour 4 is 0.4s -> a 0.3s p95 drift, well past the 200ms placeholder tolerance,
            # and a queue-age slope far past any percent-of-baseline tolerance.
            latency_fn=lambda hour: LatencyProfile(kind="fixed", value_s=0.1 * hour),
            # Synthetic RSS grows 200MB/hour -> a textbook leak signature, well past the 50MB/hour
            # placeholder tolerance.
            rss_fn=lambda hour: 100_000_000 + hour * 200_000_000,
        )
        snapshots = _run_soak(config)

        result = evaluate_p_stab_1(snapshots)
        self.assertFalse(result.queue_age_ok, result)
        self.assertFalse(result.memory_ok, result)
        self.assertFalse(result.latency_ok, result)
        self.assertFalse(result.passed, result)

        # Detected: well past tolerance, not just nonzero-by-floating-point-noise.
        self.assertGreater(result.trend.memory_slope_bytes_per_hour, 100_000_000)
        self.assertGreater(result.trend.latency_p95_drift_s, 0.2)
        self.assertGreater(
            result.trend.queue_age_slope_s_per_hour, result.allowed_queue_age_slope_s_per_hour
        )

    def test_a_scenario_that_only_leaks_memory_fails_only_the_memory_condition(self):
        """A more surgical leak — flat latency, growing RSS only — must fail memory_ok specifically
        while queue_age_ok/latency_ok stay clean, proving the three conditions are independent, not
        one aggregate check standing in for all three."""
        config = SoakConfig(
            hours=4,
            profile=_base_profile(),
            timeline=_timeline(),
            hour_s=20.0,
            tick_s=5.0,
            seed=0,
            latency_fn=lambda hour: LatencyProfile(kind="fixed", value_s=0.05),
            rss_fn=lambda hour: 100_000_000 + hour * 200_000_000,
        )
        snapshots = _run_soak(config)
        result = evaluate_p_stab_1(snapshots)

        self.assertTrue(result.queue_age_ok, result)
        self.assertTrue(result.latency_ok, result)
        self.assertFalse(result.memory_ok, result)
        self.assertFalse(result.passed, result)


class TestInjectedFailureWindowRecoversCleanly(unittest.TestCase):
    """(3) An injected provider-failure window must show CLEAN recovery: failures confined exactly
    to the configured hours, zero residual failures either side of it."""

    def test_failure_confined_to_injected_hour_recovers_cleanly_and_gate_passes(self):
        failure_hours = frozenset({2})

        def _errors_fn(hour: int) -> ErrorInjectionConfig:
            if hour in failure_hours:
                return ErrorInjectionConfig(rate=1.0, modes=("provider_error",))
            return ErrorInjectionConfig(rate=0.0)

        config = SoakConfig(
            hours=3,
            profile=_base_profile(),
            timeline=_timeline(),
            hour_s=20.0,
            tick_s=5.0,
            seed=0,
            # Flat latency/RSS elsewhere -> the OTHER three conditions stay clean, isolating this
            # test to the recovery signal specifically.
            latency_fn=lambda hour: LatencyProfile(kind="fixed", value_s=0.05),
            rss_fn=lambda hour: 100_000_000,
            errors_fn=_errors_fn,
        )
        snapshots = _run_soak(config)
        by_hour = {s.hour: s for s in snapshots}

        # Hour 2: every issued judge call was injected with a failure (rate=1.0).
        self.assertGreater(by_hour[2].judge_calls_issued, 0)
        self.assertEqual(by_hour[2].judge_calls_failed, by_hour[2].judge_calls_issued)
        # Hours 1 and 3: completely clean, zero residual failures either side of the window.
        self.assertEqual(by_hour[1].judge_calls_failed, 0)
        self.assertEqual(by_hour[3].judge_calls_failed, 0)

        result = evaluate_p_stab_1(snapshots, failure_hours=failure_hours)
        self.assertTrue(result.recovery_ok, result)
        self.assertTrue(result.passed, result)

    def test_a_mis_scoped_failure_window_fails_the_recovery_condition(self):
        """The same run, checked against the WRONG failure window, must fail recovery_ok (and
        therefore the overall gate) — proves this isn't a check that always passes."""
        actual_failure_hours = frozenset({2})

        def _errors_fn(hour: int) -> ErrorInjectionConfig:
            if hour in actual_failure_hours:
                return ErrorInjectionConfig(rate=1.0, modes=("provider_error",))
            return ErrorInjectionConfig(rate=0.0)

        config = SoakConfig(
            hours=3,
            profile=_base_profile(),
            timeline=_timeline(),
            hour_s=20.0,
            tick_s=5.0,
            seed=0,
            latency_fn=lambda hour: LatencyProfile(kind="fixed", value_s=0.05),
            rss_fn=lambda hour: 100_000_000,
            errors_fn=_errors_fn,
        )
        snapshots = _run_soak(config)

        result_wrong_window = evaluate_p_stab_1(snapshots, failure_hours=frozenset({1}))
        self.assertFalse(result_wrong_window.recovery_ok, result_wrong_window)
        self.assertFalse(result_wrong_window.passed, result_wrong_window)

        result_no_window = evaluate_p_stab_1(snapshots, failure_hours=frozenset())
        self.assertFalse(result_no_window.recovery_ok, result_no_window)
        self.assertFalse(result_no_window.passed, result_no_window)


if __name__ == "__main__":
    unittest.main()
