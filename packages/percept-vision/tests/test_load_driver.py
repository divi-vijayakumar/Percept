"""Probe-harness ``load_driver`` (Viability A11) — full N/M/E/F ramp + steady load, L1 no-key.

Proves, under a :class:`~percept.clock.FakeClock`, the things downstream probes (P-BOUND-1,
P-LAT-LOAD-1) need to trust before they can assert "the intended N/M/E/F load was actually
produced":

  * a small (N, M, E, F) config drives exactly the expected number of streams (via ``fan_out``),
    admitted episodes, and fan-out judge calls (via ``FakeJudge``) — computed via the closed-form
    ramp/steady rate integral, not approximated.
  * the ramp -> steady transition happens at the right time and produces the right split.
  * the observed load is independent of how finely ``tick()``/``run_to_completion`` is driven —
    zero wall-clock waits, fully deterministic given the same clock trajectory.
"""

from __future__ import annotations

import asyncio
import unittest

from percept.backends.fake_judge import FakeJudge
from percept.clock import FakeClock
from percept.harness.load_driver import LoadDriver, LoadDriverStats, LoadProfile, RampSchedule
from percept.harness.stream_gen import parametric_motion_clip


def _timeline(n: int = 8, dt: float = 1.0):
    return parametric_motion_clip(w=4, h=4, n=n, motion_frames={1, 3}, dt=dt)


class TestLoadProfileValidation(unittest.TestCase):
    def test_rejects_bad_values(self):
        with self.assertRaises(ValueError):
            LoadProfile(n=0, m=1, e_per_stream_min=60, f=1)
        with self.assertRaises(ValueError):
            LoadProfile(n=1, m=0, e_per_stream_min=60, f=1)
        with self.assertRaises(ValueError):
            LoadProfile(n=1, m=1, e_per_stream_min=0, f=1)
        with self.assertRaises(ValueError):
            LoadProfile(n=1, m=1, e_per_stream_min=60, f=0)

    def test_full_rate_per_watch(self):
        profile = LoadProfile(n=2, m=2, e_per_stream_min=60, f=1)
        # 60 admitted episodes/stream-minute split across 2 watches -> 0.5/sec per watch.
        self.assertAlmostEqual(profile.full_rate_per_watch, 0.5)


class TestRampScheduleValidation(unittest.TestCase):
    def test_rejects_bad_values(self):
        with self.assertRaises(ValueError):
            RampSchedule(ramp_s=-1, steady_s=1)
        with self.assertRaises(ValueError):
            RampSchedule(ramp_s=0, steady_s=0)

    def test_total_and_phase_at(self):
        sched = RampSchedule(ramp_s=4, steady_s=3)
        self.assertEqual(sched.total_s, 7)
        self.assertEqual(sched.phase_at(0.0), "ramp")
        self.assertEqual(sched.phase_at(4.0), "ramp")  # boundary counts as ramp
        self.assertEqual(sched.phase_at(4.0001), "steady")
        self.assertEqual(sched.phase_at(7.0), "steady")


class TestLoadDriverExactCounts(unittest.TestCase):
    """N=2, M=1, E=60/stream-min (=> 1.0 ep/sec/watch), F=2, ramp_s=4, steady_s=3.

    Closed-form ramp integral per watch: full/ramp_s * (ramp_s**2)/2 = 1/4 * 16/2 = 2.0.
    Steady integral per watch: full * steady_s = 1.0 * 3 = 3.0.
    So per watch: 2 ramp admissions + 3 steady admissions = 5 total.
    With N=2 streams, M=1 watch/stream: 4 ramp + 6 steady = 10 total episodes.
    F=2 -> 20 judge calls, all clean (no error injection configured) -> 20 ok, 0 failed.
    """

    def _build(self):
        profile = LoadProfile(n=2, m=1, e_per_stream_min=60.0, f=2)
        schedule = RampSchedule(ramp_s=4.0, steady_s=3.0)
        clock = FakeClock()
        judge = FakeJudge(seed=0, simulate_sleep=False)
        driver = LoadDriver(profile, schedule, _timeline(n=8, dt=1.0), clock=clock, judge=judge)
        return driver, clock, judge

    def test_two_tick_run_produces_exact_expected_counts(self):
        driver, clock, _judge = self._build()

        asyncio.run(driver.tick())  # t=0: no elapsed time yet, nothing admitted
        self.assertEqual(driver.stats.episodes_admitted, 0)

        clock.advance(4000)  # -> t=4s, end of ramp
        asyncio.run(driver.tick())
        self.assertEqual(driver.stats.episodes_by_phase["ramp"], 4)
        self.assertEqual(driver.stats.episodes_by_phase["steady"], 0)

        clock.advance(3000)  # -> t=7s, end of steady phase
        asyncio.run(driver.tick())

        self.assertEqual(driver.stats.streams_active, 2)
        self.assertEqual(driver.stats.episodes_admitted, 10)
        self.assertEqual(driver.stats.episodes_by_phase, {"ramp": 4, "steady": 6})
        self.assertEqual(driver.stats.judge_calls_issued, 20)
        self.assertEqual(driver.stats.judge_calls_ok, 20)
        self.assertEqual(driver.stats.judge_calls_failed, 0)
        self.assertEqual(driver.stats.observed_f, 2.0)  # exact — F is a multiplier, not sampled
        self.assertTrue(driver.done())

        # All 8 frames/stream became due by t=7 (fan_out actually used for the N streams).
        self.assertEqual(driver.stats.frames_emitted, 16)

    def test_single_jump_to_end_yields_identical_counts_as_two_ticks(self):
        """Determinism: the admitted-episode count for a given clock trajectory does not depend
        on tick granularity — one big jump to the end matches the two-tick run above exactly."""
        driver, clock, _judge = self._build()

        clock.advance(7000)  # straight to t=7s in one jump
        asyncio.run(driver.tick())

        self.assertEqual(driver.stats.episodes_admitted, 10)
        self.assertEqual(driver.stats.episodes_by_phase, {"ramp": 4, "steady": 6})
        self.assertEqual(driver.stats.judge_calls_issued, 20)

    def test_run_to_completion_matches_manual_ticks(self):
        profile = LoadProfile(n=2, m=1, e_per_stream_min=60.0, f=2)
        schedule = RampSchedule(ramp_s=4.0, steady_s=3.0)
        clock = FakeClock()
        judge = FakeJudge(seed=0, simulate_sleep=False)
        driver = LoadDriver(profile, schedule, _timeline(n=8, dt=1.0), clock=clock, judge=judge)

        stats = asyncio.run(driver.run_to_completion(step_s=0.5))

        self.assertIs(stats, driver.stats)
        self.assertEqual(stats.episodes_admitted, 10)
        self.assertEqual(stats.episodes_by_phase, {"ramp": 4, "steady": 6})
        self.assertEqual(stats.judge_calls_issued, 20)
        self.assertEqual(stats.elapsed_s, 7.0)
        self.assertTrue(driver.done())

    def test_fine_and_coarse_step_granularity_agree(self):
        def _run(step_s: float) -> LoadDriverStats:
            profile = LoadProfile(n=2, m=1, e_per_stream_min=60.0, f=2)
            schedule = RampSchedule(ramp_s=4.0, steady_s=3.0)
            clock = FakeClock()
            judge = FakeJudge(seed=0, simulate_sleep=False)
            driver = LoadDriver(profile, schedule, _timeline(n=8, dt=1.0), clock=clock, judge=judge)
            return asyncio.run(driver.run_to_completion(step_s=step_s))

        fine = _run(0.25)
        coarse = _run(9.0)  # overshoots total_s (7s) in a single jump

        self.assertEqual(fine.episodes_admitted, coarse.episodes_admitted)
        self.assertEqual(fine.episodes_by_phase, coarse.episodes_by_phase)
        self.assertEqual(fine.judge_calls_issued, coarse.judge_calls_issued)
        self.assertEqual(fine.frames_emitted, coarse.frames_emitted)
        self.assertEqual(fine.elapsed_s, coarse.elapsed_s)


class TestNoRampPureSteadyAndMultiWatch(unittest.TestCase):
    def test_zero_ramp_and_two_watches_spread_admissions_but_total_is_exact(self):
        # N=1, M=2, E=60/stream-min -> 0.5 ep/sec/watch, no ramp, steady_s=10 -> 5.0/watch exactly.
        profile = LoadProfile(n=1, m=2, e_per_stream_min=60.0, f=1)
        schedule = RampSchedule(ramp_s=0.0, steady_s=10.0)
        clock = FakeClock()
        judge = FakeJudge(seed=1, simulate_sleep=False)
        driver = LoadDriver(profile, schedule, _timeline(n=1, dt=1.0), clock=clock, judge=judge)

        clock.advance(10_000)
        asyncio.run(driver.tick())

        self.assertEqual(driver.stats.episodes_admitted, 10)  # 2 watches * 5.0 each
        self.assertEqual(driver.stats.episodes_by_phase, {"ramp": 0, "steady": 10})
        self.assertEqual(driver.stats.judge_calls_issued, 10)

    def test_observed_e_per_stream_min_reads_below_target_during_ramp(self):
        profile = LoadProfile(n=2, m=1, e_per_stream_min=60.0, f=1)
        schedule = RampSchedule(ramp_s=4.0, steady_s=3.0)
        clock = FakeClock()
        judge = FakeJudge(seed=0, simulate_sleep=False)
        driver = LoadDriver(profile, schedule, _timeline(n=8, dt=1.0), clock=clock, judge=judge)

        clock.advance(4000)
        asyncio.run(driver.tick())
        # 4 episodes / 2 streams / (4s / 60) minutes = 4/2/(1/15) = 30, well below target E=60.
        self.assertAlmostEqual(driver.stats.observed_e_per_stream_min(profile.n), 30.0)


class TestErrorInjectionSurfacesAsFailedJudgeCalls(unittest.TestCase):
    def test_injected_errors_are_recorded_as_failed_not_raised(self):
        from percept.backends.fake_judge import ErrorInjectionConfig

        profile = LoadProfile(n=1, m=1, e_per_stream_min=60.0, f=3)
        schedule = RampSchedule(ramp_s=0.0, steady_s=5.0)
        clock = FakeClock()
        judge = FakeJudge(
            seed=2,
            errors=ErrorInjectionConfig(rate=1.0, modes=("timeout",)),
            simulate_sleep=False,
        )
        driver = LoadDriver(profile, schedule, _timeline(n=1, dt=1.0), clock=clock, judge=judge)

        clock.advance(5000)
        asyncio.run(driver.tick())  # must not raise even though every judge call is injected

        self.assertEqual(driver.stats.judge_calls_issued, driver.stats.episodes_admitted * 3)
        self.assertEqual(driver.stats.judge_calls_failed, driver.stats.judge_calls_issued)
        self.assertEqual(driver.stats.judge_calls_ok, 0)


if __name__ == "__main__":
    unittest.main()
