"""Probe-harness ``soak_runner`` (the benchmark metrics contract, P-STAB-1).

Proves, under a :class:`~percept.clock.FakeClock` only (no real wall-clock waits anywhere), that
:class:`~percept.harness.soak_runner.SoakRunner`:

  * runs a steady load over N simulated hours, taking one
    :class:`~percept.harness.metrics.MetricsCollector`-backed hourly snapshot each hour;
  * a STABLE synthetic load (flat latency profile, flat injected RSS, no provider failures) yields
    hour-N-vs-hour-1 slopes of ~zero for queue-age and memory, and zero latency drift;
  * a LEAK/DRIFT scenario (RSS growing hour-over-hour, judge latency growing hour-over-hour) is
    DETECTED — all three slopes/drift come back clearly nonzero, well past a tight tolerance;
  * an injected provider-failure window is confined exactly to the configured hours, and the run
    recovers cleanly (zero residual failures) in every hour outside that window.
"""

from __future__ import annotations

import unittest

from percept.backends.fake_judge import ErrorInjectionConfig, LatencyProfile
from percept.clock import FakeClock
from percept.harness.load_driver import LoadProfile
from percept.harness.soak_runner import (
    SoakConfig,
    SoakRunner,
    assert_clean_recovery,
    compute_trend,
)
from percept.harness.stream_gen import parametric_motion_clip

_TOL = 1e-9


def _timeline(n: int = 8, dt: float = 1.0):
    return parametric_motion_clip(w=4, h=4, n=n, motion_frames={1, 3}, dt=dt)


def _base_profile() -> LoadProfile:
    # n=1 stream, m=1 watch, 30 admitted episodes/stream-minute (=> 0.5/sec/watch), f=2 fan-out.
    return LoadProfile(n=1, m=1, e_per_stream_min=30.0, f=2)


class TestStableLoadYieldsZeroSlope(unittest.TestCase):
    def test_flat_latency_flat_rss_no_failures_gives_zero_slopes(self):
        config = SoakConfig(
            hours=4,
            profile=_base_profile(),
            timeline=_timeline(),
            hour_s=20.0,
            tick_s=5.0,
            seed=0,
            latency_fn=lambda hour: LatencyProfile(kind="fixed", value_s=0.05),
            rss_fn=lambda hour: 100_000_000,
        )
        clock = FakeClock()
        runner = SoakRunner(config, clock=clock)

        import asyncio

        snapshots = asyncio.run(runner.run())

        self.assertEqual(len(snapshots), 4)
        self.assertEqual([s.hour for s in snapshots], [1, 2, 3, 4])
        # Same steady profile/schedule every hour -> identical admission counts hour over hour.
        self.assertTrue(all(s.episodes_admitted == snapshots[0].episodes_admitted for s in snapshots))
        self.assertGreater(snapshots[0].episodes_admitted, 0)

        trend = compute_trend(snapshots)
        self.assertAlmostEqual(trend.queue_age_slope_s_per_hour, 0.0, delta=_TOL)
        self.assertAlmostEqual(trend.memory_slope_bytes_per_hour, 0.0, delta=_TOL)
        self.assertAlmostEqual(trend.latency_p95_drift_s, 0.0, delta=_TOL)

        # No failures were configured anywhere.
        self.assertTrue(assert_clean_recovery(snapshots, failure_hours=frozenset()))


class TestLeakScenarioIsDetected(unittest.TestCase):
    def test_growing_rss_and_latency_produce_clearly_nonzero_slopes(self):
        config = SoakConfig(
            hours=4,
            profile=_base_profile(),
            timeline=_timeline(),
            hour_s=20.0,
            tick_s=5.0,
            seed=0,
            # Judge latency grows every hour -> queue-age (oldest-pending-age at resolution,
            # which equals that hour's fixed latency in this model) grows in lockstep.
            latency_fn=lambda hour: LatencyProfile(kind="fixed", value_s=0.05 * hour),
            # Synthetic RSS grows every hour -> a textbook leak signature.
            rss_fn=lambda hour: 100_000_000 + hour * 50_000_000,
        )
        clock = FakeClock()
        runner = SoakRunner(config, clock=clock)

        import asyncio

        snapshots = asyncio.run(runner.run())
        trend = compute_trend(snapshots)

        # Detected: well past a tight tolerance, not just nonzero-by-floating-point-noise.
        self.assertGreater(trend.memory_slope_bytes_per_hour, 1_000_000)
        self.assertGreater(trend.queue_age_slope_s_per_hour, 0.01)
        self.assertGreater(trend.latency_p95_drift_s, 0.01)

        # Sanity: the stable-scenario thresholds above would have failed this run.
        self.assertFalse(abs(trend.memory_slope_bytes_per_hour) < _TOL)
        self.assertFalse(abs(trend.queue_age_slope_s_per_hour) < _TOL)


class TestProviderFailureWindowRecoversCleanly(unittest.TestCase):
    def test_failures_confined_to_injected_hour_and_clean_after(self):
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
            errors_fn=_errors_fn,
        )
        clock = FakeClock()
        runner = SoakRunner(config, clock=clock)

        import asyncio

        snapshots = asyncio.run(runner.run())
        by_hour = {s.hour: s for s in snapshots}

        # Hour 2: every issued judge call was injected with a failure (rate=1.0).
        self.assertGreater(by_hour[2].judge_calls_issued, 0)
        self.assertEqual(by_hour[2].judge_calls_failed, by_hour[2].judge_calls_issued)
        self.assertEqual(by_hour[2].judge_calls_ok, 0)

        # Hours 1 and 3: completely clean, zero residual failures either side of the window.
        self.assertEqual(by_hour[1].judge_calls_failed, 0)
        self.assertEqual(by_hour[3].judge_calls_failed, 0)
        self.assertGreater(by_hour[1].judge_calls_ok, 0)
        self.assertGreater(by_hour[3].judge_calls_ok, 0)

        self.assertTrue(assert_clean_recovery(snapshots, failure_hours=failure_hours))
        # A mis-scoped failure window must be rejected by the same check.
        self.assertFalse(assert_clean_recovery(snapshots, failure_hours=frozenset({1})))
        self.assertFalse(assert_clean_recovery(snapshots, failure_hours=frozenset()))


class TestSoakConfigValidation(unittest.TestCase):
    def test_rejects_bad_values(self):
        with self.assertRaises(ValueError):
            SoakConfig(hours=0, profile=_base_profile(), timeline=_timeline())
        with self.assertRaises(ValueError):
            SoakConfig(hours=1, profile=_base_profile(), timeline=_timeline(), hour_s=0)
        with self.assertRaises(ValueError):
            SoakConfig(hours=1, profile=_base_profile(), timeline=_timeline(), tick_s=0)


class TestComputeTrendValidation(unittest.TestCase):
    def test_requires_at_least_two_snapshots(self):
        config = SoakConfig(hours=1, profile=_base_profile(), timeline=_timeline(), hour_s=20.0, tick_s=5.0)
        clock = FakeClock()
        runner = SoakRunner(config, clock=clock)

        import asyncio

        snapshots = asyncio.run(runner.run())
        self.assertEqual(len(snapshots), 1)
        with self.assertRaises(ValueError):
            compute_trend(snapshots)


if __name__ == "__main__":
    unittest.main()
