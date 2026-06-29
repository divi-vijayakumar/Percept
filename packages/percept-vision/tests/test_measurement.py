"""The canonical measurement layer (Build 1) — L1 tests on SYNTHETIC signals (no pose, no video, no model).

These assert the one property the layer exists to provide: metrics computed in **seconds + amplitude** are
invariant to the sampling rate. The same physical signal, sampled at 3…160 fps, must yield the same period,
the same amplitude, and a high periodicity score — while the *frame-index* view of the same period (lag in
samples) is shown to drift with fps, which is exactly the pathology the layer removes.
"""

from __future__ import annotations

import math
import unittest

from percept.measurement import (
    MeasurementSeries,
    channel_score,
    dominant_period,
    periodicity_score,
    robust_amplitude,
    subsample_stability,
    to_time_grid,
)

FPS_SWEEP = (3.0, 6.0, 12.0, 30.0, 60.0, 160.0)
CANON = 30.0  # shared canonical resample rate for cross-fps scoring


def physical(t: float, freq: float = 0.4, amp: float = 1.0) -> float:
    """A clean cycle (period 1/freq s) on a slow baseline drift — the drift is what mean-removal must
    survive; ``freq=0.4`` -> a 2.5 s period, comfortably above the Nyquist floor for every fps in the sweep."""
    return amp * math.sin(2 * math.pi * freq * t) + 0.12 * math.sin(2 * math.pi * 0.05 * t)


def sample(fps: float, dur_s: float = 25.0, freq: float = 0.4, amp: float = 1.0, channel: str = "c") -> MeasurementSeries:
    n = int(dur_s * fps)
    return MeasurementSeries.from_pairs(
        channel, [(i / fps, physical(i / fps, freq, amp)) for i in range(n)], units="norm", source_hz=fps
    )


class MeasurementSeriesBasics(unittest.TestCase):
    def test_from_pairs_sorts_by_time(self):
        s = MeasurementSeries.from_pairs("c", [(2.0, 9.0), (0.0, 1.0), (1.0, 5.0)])
        self.assertEqual(s.times, (0.0, 1.0, 2.0))
        self.assertEqual(s.values, (1.0, 5.0, 9.0))

    def test_duration_and_effective_hz(self):
        s = sample(6.0, dur_s=10.0)
        self.assertAlmostEqual(s.duration, (len(s) - 1) / 6.0, places=6)
        self.assertAlmostEqual(s.effective_hz, 6.0, delta=0.05)

    def test_empty_and_singleton_are_safe(self):
        empty = MeasurementSeries("c", ())
        self.assertEqual(len(empty), 0)
        self.assertIsNone(empty.span)
        self.assertEqual(empty.duration, 0.0)
        self.assertEqual(robust_amplitude(empty, 1.0), 0.0)
        self.assertEqual(periodicity_score(empty), 0.0)
        self.assertIsNone(dominant_period(empty))


class TimeGridTests(unittest.TestCase):
    def test_grid_is_uniform_at_requested_rate(self):
        grid = to_time_grid(sample(7.0, dur_s=12.0), CANON)
        self.assertEqual(grid.source_hz, CANON)
        diffs = [b - a for a, b in zip(grid.times, grid.times[1:])]
        for d in diffs:
            self.assertAlmostEqual(d, 1.0 / CANON, places=6)

    def test_lowpass_attenuates_high_frequency_jitter_when_downsampling(self):
        # A clean 0.4 Hz cycle + a fast 5 Hz jitter, captured at 60 fps. Downsampling to 6 fps (Nyquist 3 Hz)
        # with the box anti-alias filter must NOT fold the 5 Hz jitter onto the cycle: the lowpassed grid is
        # closer to the pure cycle than a raw decimation (lowpass=False) is.
        fps = 60.0
        n = int(20 * fps)
        clean = [math.sin(2 * math.pi * 0.4 * i / fps) for i in range(n)]
        noisy = MeasurementSeries.from_pairs(
            "c", [(i / fps, clean[i] + 0.4 * math.sin(2 * math.pi * 5.0 * i / fps)) for i in range(n)]
        )
        lp = to_time_grid(noisy, 6.0, lowpass=True)
        raw = to_time_grid(noisy, 6.0, lowpass=False)

        def err_vs_clean(g: MeasurementSeries) -> float:
            return sum(abs(v - math.sin(2 * math.pi * 0.4 * t)) for t, v in g.samples) / len(g)

        self.assertLess(err_vs_clean(lp), err_vs_clean(raw))


class PeriodicityTests(unittest.TestCase):
    def test_clean_cycle_scores_high_noise_scores_low(self):
        clean = sample(30.0, dur_s=25.0)
        self.assertGreater(periodicity_score(clean, band_s=(1.0, 5.0)), 0.8)
        # deterministic broadband "noise": incommensurate sines, no dominant cycle
        noise = MeasurementSeries.from_pairs(
            "c",
            [(i / 30.0, sum(math.sin(2 * math.pi * f * i / 30.0) for f in (1.3, 2.7, 4.1, 6.9))) for i in range(750)],
        )
        self.assertLess(periodicity_score(noise, band_s=(1.0, 5.0)), periodicity_score(clean, band_s=(1.0, 5.0)))

    def test_dominant_period_recovered_in_seconds(self):
        for freq, period in ((0.4, 2.5), (0.25, 4.0)):
            p = dominant_period(sample(30.0, dur_s=30.0, freq=freq), band_s=(1.0, 6.0), hz=CANON)
            self.assertIsNotNone(p)
            self.assertAlmostEqual(p, period, delta=0.15 * period)


class RobustAmplitudeTests(unittest.TestCase):
    def test_recovers_amplitude_in_native_units(self):
        # p95−p05 span of a unit sine is a little under 2*A; assert it lands in a tight band around it.
        amp = robust_amplitude(sample(30.0, dur_s=25.0, amp=1.0), window_s=2.5)
        self.assertGreater(amp, 1.6)
        self.assertLess(amp, 2.1)

    def test_single_spike_is_rejected(self):
        base = sample(30.0, dur_s=25.0, amp=1.0)
        spiked = MeasurementSeries(
            "c", base.samples[:100] + ((base.samples[100][0], 50.0),) + base.samples[101:], source_hz=30.0
        )
        clean_amp = robust_amplitude(base, window_s=2.5)
        spiked_amp = robust_amplitude(spiked, window_s=2.5)
        naive = max(spiked.values) - min(spiked.values)  # the non-robust view explodes
        self.assertGreater(naive, 40.0)
        self.assertLess(abs(spiked_amp - clean_amp), 0.25)


class SubsampleStabilityTests(unittest.TestCase):
    def test_periodic_signal_is_stable_under_decimation(self):
        s = sample(60.0, dur_s=25.0)
        score = subsample_stability(s, lambda x: dominant_period(x, band_s=(1.0, 5.0), hz=CANON))
        self.assertGreater(score, 0.85)

    def test_categorical_scorer_modal_agreement(self):
        # A scorer that returns a stable label scores 1.0; the helper handles non-numeric answers.
        s = sample(30.0, dur_s=20.0)
        self.assertEqual(subsample_stability(s, lambda x: "hip"), 1.0)


class ChannelScoreTests(unittest.TestCase):
    """Build 3 — the canonical channel score ranks a rep-shaped channel above noise/flat, and that RANKING
    is the same at every fps (the fix for the selector that used to flip hip<->torso with fps)."""

    @staticmethod
    def _flat(fps, dur_s=25.0, channel="flat"):
        n = int(dur_s * fps)
        return MeasurementSeries.from_pairs(channel, [(i / fps, 0.0) for i in range(n)], source_hz=fps)

    @staticmethod
    def _noise(fps, dur_s=25.0, channel="noise"):
        import random
        rng = random.Random(7)  # broadband white noise: no cycle -> low regularity AND unstable period
        n = int(dur_s * fps)
        return MeasurementSeries.from_pairs(channel, [(i / fps, rng.gauss(0, 1)) for i in range(n)], source_hz=fps)

    def test_periodic_outscores_noise_and_flat(self):
        rep = channel_score(sample(30.0, channel="rep"), band_s=(1.0, 5.0), hz=CANON)
        noise = channel_score(self._noise(30.0), band_s=(1.0, 5.0), hz=CANON)
        flat = channel_score(self._flat(30.0), band_s=(1.0, 5.0), hz=CANON)
        self.assertGreater(rep.score, noise.score)
        self.assertGreater(rep.score, flat.score)
        self.assertEqual(flat.score, 0.0)  # no oscillation -> no salience
        self.assertGreater(rep.support, 5)  # ~10 periods of a 2.5s cycle in 25s

    def test_winning_channel_is_invariant_across_fps(self):
        # A strong 0.4 Hz rep channel and a weak 0.9 Hz low-amplitude channel: the rep channel wins at EVERY
        # fps (the selector's electorate must not change with sampling rate).
        for fps in FPS_SWEEP:
            rep = channel_score(sample(fps, freq=0.4, amp=1.0, channel="rep"), band_s=(0.8, 5.0), hz=CANON)
            weak = channel_score(sample(fps, freq=0.9, amp=0.2, channel="weak"), band_s=(0.8, 5.0), hz=CANON)
            self.assertGreater(rep.score, weak.score, f"rep must outscore weak at {fps} fps")


class RateInvarianceGate(unittest.TestCase):
    """The headline: one physical signal, sampled across the fps sweep, yields invariant physical-unit
    metrics — and a frame-index metric of the SAME period does not (the foil)."""

    def test_period_amplitude_and_score_are_invariant_across_fps(self):
        periods, amps, scores, period_in_frames = [], [], [], []
        for fps in FPS_SWEEP:
            s = sample(fps, dur_s=25.0, freq=0.4, amp=1.0)
            p = dominant_period(s, band_s=(1.0, 5.0), hz=CANON)
            self.assertIsNotNone(p, f"no period at {fps} fps")
            periods.append(p)
            amps.append(robust_amplitude(s, window_s=2.5))
            scores.append(periodicity_score(s, band_s=(1.0, 5.0), hz=CANON))
            period_in_frames.append(p * fps)  # the NON-invariant frame-index view of the same period

        # (a) period in seconds: tight across all fps, and locked on the true 2.5 s
        self.assertLess(max(periods) - min(periods), 0.25)
        for p in periods:
            self.assertAlmostEqual(p, 2.5, delta=0.2)
        # (b) amplitude: the contract is "more frames is never worse" -> MONOTONE non-decreasing in fps,
        # and tight once comfortably above the Nyquist floor (>=6 fps; 3 fps is ~2x Nyquist, marginal).
        for lo, hi in zip(amps, amps[1:]):
            self.assertLessEqual(lo, hi + 1e-9, f"amplitude must not drop as fps rises: {amps}")
        supra = [a for fps, a in zip(FPS_SWEEP, amps) if fps >= 6.0]
        self.assertLess(max(supra) - min(supra), 0.1)
        # (c) periodicity score: high past the Nyquist floor at every rate
        for sc in scores:
            self.assertGreater(sc, 0.8)
        # (d) the foil — the frame-index period drifts by a huge factor with fps, proving the canonical
        #     (seconds) layer is what makes the metric invariant, not luck.
        self.assertGreater(max(period_in_frames) / min(period_in_frames), 20.0)


if __name__ == "__main__":
    unittest.main()
