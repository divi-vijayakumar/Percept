"""Probe-harness ``stream_gen`` (Viability A10) — determinism + fan-out, L1 no-key.

Two things this proves, both required before A11's ``load_driver`` (and downstream Phase-A
probes) can trust this generator:

  * replay driven by an injected clock is fully deterministic — the same timeline + the same
    schedule of clock advances yields byte-identical emitted sequences, with zero wall-clock waits
    (no ``time.sleep`` anywhere in the seam).
  * fan-out to N concurrent streams produces N genuinely independent consumers: polling one never
    perturbs another's cursor, whether they share a clock or are driven by separate clocks.
"""

from __future__ import annotations

import unittest

from percept.clock import FakeClock
from percept.harness.stream_gen import (
    StreamGen,
    TimedFrame,
    fan_out,
    fixture_clip,
    parametric_motion_clip,
    random_motion_clip,
    synthetic_clip,
    timed_clip,
)
from percept.harness.tier0 import Frame, gray_frame


class TestTimelineConstruction(unittest.TestCase):
    def test_fixture_clip_spaces_frames_at_fixed_dt(self):
        frames = [gray_frame(2, 2, fill=i) for i in range(4)]
        tl = fixture_clip(frames, dt=0.5, start=1.0)
        self.assertEqual([tf.t for tf in tl], [1.0, 1.5, 2.0, 2.5])
        self.assertEqual([tf.frame for tf in tl], frames)

    def test_fixture_clip_rejects_nonpositive_dt(self):
        with self.assertRaises(ValueError):
            fixture_clip([gray_frame(1, 1)], dt=0.0)

    def test_timed_clip_uses_explicit_timestamps(self):
        frames = [gray_frame(1, 1, fill=i) for i in range(3)]
        tl = timed_clip(frames, [0.0, 0.2, 0.9])
        self.assertEqual([tf.t for tf in tl], [0.0, 0.2, 0.9])

    def test_timed_clip_rejects_length_mismatch(self):
        with self.assertRaises(ValueError):
            timed_clip([gray_frame(1, 1)], [0.0, 1.0])

    def test_timed_frame_rejects_negative_t(self):
        with self.assertRaises(ValueError):
            TimedFrame(t=-0.1, frame=gray_frame(1, 1))

    def test_synthetic_clip_calls_frame_fn_by_index(self):
        seen = []

        def _fn(i: int) -> Frame:
            seen.append(i)
            return gray_frame(1, 1, fill=i)

        tl = synthetic_clip(_fn, 5, dt=1.0)
        self.assertEqual(seen, [0, 1, 2, 3, 4])
        self.assertEqual([tf.t for tf in tl], [0.0, 1.0, 2.0, 3.0, 4.0])

    def test_parametric_motion_clip_marks_only_given_frames(self):
        tl = parametric_motion_clip(w=4, h=4, n=6, motion_frames={2, 4}, motion_pixels=3, motion_value=200)
        still = gray_frame(4, 4, fill=0)
        for i, tf in enumerate(tl):
            if i in (2, 4):
                self.assertNotEqual(tf.frame, still)
                self.assertEqual(tf.frame.pixels[0], 200)
            else:
                self.assertEqual(tf.frame, still)


class TestRandomMotionClipDeterminism(unittest.TestCase):
    def test_same_seed_yields_identical_timeline(self):
        tl1 = random_motion_clip(seed=7, n=40, motion_prob=0.2)
        tl2 = random_motion_clip(seed=7, n=40, motion_prob=0.2)
        self.assertEqual(tl1, tl2)

    def test_different_seed_yields_different_timeline(self):
        tl1 = random_motion_clip(seed=1, n=40, motion_prob=0.2)
        tl2 = random_motion_clip(seed=2, n=40, motion_prob=0.2)
        self.assertNotEqual(tl1, tl2)


class TestStreamGenReplayDeterminism(unittest.TestCase):
    def _timeline(self):
        return parametric_motion_clip(w=4, h=4, n=10, motion_frames={2, 5, 8}, dt=1.0)

    def test_same_clock_schedule_yields_identical_emitted_sequence(self):
        tl = self._timeline()
        clock1 = FakeClock()
        clock2 = FakeClock()
        s1 = StreamGen(tl, clock=clock1)
        s2 = StreamGen(tl, clock=clock2)

        advances_ms = [0, 500, 500, 1000, 1500, 2000, 3000, 4000]
        emitted1: list[TimedFrame] = []
        emitted2: list[TimedFrame] = []
        for ms in advances_ms:
            clock1.advance(ms)
            clock2.advance(ms)
            emitted1.extend(s1.poll())
            emitted2.extend(s2.poll())

        self.assertEqual(emitted1, emitted2)
        self.assertTrue(emitted1)  # sanity: the schedule actually emitted something

    def test_poll_is_nonblocking_and_only_returns_due_frames(self):
        tl = self._timeline()
        clock = FakeClock()
        s = StreamGen(tl, clock=clock)

        first = s.poll()  # t=0 frame is due immediately, nothing later is
        self.assertEqual([tf.t for tf in first], [0.0])

        again = s.poll()  # no clock advance -> nothing new due
        self.assertEqual(again, [])

        clock.advance(2500)  # 2.5s elapsed -> t=1.0 and t=2.0 now due (t=0 already emitted)
        due = s.poll()
        self.assertEqual([tf.t for tf in due], [1.0, 2.0])

    def test_stream_exhausts_and_reports_done(self):
        tl = self._timeline()
        clock = FakeClock()
        s = StreamGen(tl, clock=clock)
        clock.advance(60_000)  # advance far past the last frame
        emitted = s.poll()
        self.assertEqual(len(emitted), len(tl))
        self.assertTrue(s.done())
        self.assertEqual(s.emitted_count, len(tl))
        self.assertEqual(s.poll(), [])  # exhausted -> no more, still non-blocking

    def test_reset_reanchors_to_current_clock(self):
        tl = self._timeline()
        clock = FakeClock()
        s = StreamGen(tl, clock=clock)
        clock.advance(5000)
        s.poll()
        self.assertGreater(s.emitted_count, 0)

        clock.advance(1000)
        s.reset()
        self.assertEqual(s.emitted_count, 0)
        self.assertEqual(s.poll(), [tl[0]])  # t=0 frame due again immediately post-reset


class TestFanOut(unittest.TestCase):
    def _timeline(self):
        return parametric_motion_clip(w=4, h=4, n=6, motion_frames={1, 3}, dt=1.0)

    def test_fan_out_requires_exactly_one_clock_arg(self):
        tl = self._timeline()
        with self.assertRaises(ValueError):
            fan_out(tl, 2)  # neither given
        with self.assertRaises(ValueError):
            fan_out(tl, 2, clock=FakeClock(), clocks=[FakeClock(), FakeClock()])  # both given

    def test_fan_out_rejects_nonpositive_n(self):
        with self.assertRaises(ValueError):
            fan_out(self._timeline(), 0, clock=FakeClock())

    def test_fan_out_clocks_length_must_match_n(self):
        with self.assertRaises(ValueError):
            fan_out(self._timeline(), 3, clocks=[FakeClock(), FakeClock()])

    def test_fan_out_shared_clock_produces_n_independent_consumers(self):
        tl = self._timeline()
        clock = FakeClock()
        streams = fan_out(tl, 3, clock=clock)
        self.assertEqual(len(streams), 3)

        clock.advance(2500)
        # Fully drain stream 0 only.
        due0 = streams[0].poll()
        self.assertEqual([tf.t for tf in due0], [0.0, 1.0, 2.0])
        self.assertEqual(streams[0].emitted_count, 3)

        # Streams 1 and 2 are untouched by stream 0's poll — same clock, independent cursors.
        self.assertEqual(streams[1].emitted_count, 0)
        self.assertEqual(streams[2].emitted_count, 0)
        due1 = streams[1].poll()
        due2 = streams[2].poll()
        self.assertEqual(due0, due1)
        self.assertEqual(due1, due2)

    def test_fan_out_per_stream_clocks_desynchronize_consumers(self):
        tl = self._timeline()
        clock_a = FakeClock()
        clock_b = FakeClock()
        s_a, s_b = fan_out(tl, 2, clocks=[clock_a, clock_b])

        clock_a.advance(3000)  # stream A races ahead
        due_a = s_a.poll()
        due_b = s_b.poll()  # clock_b hasn't moved past its own t=0 start

        self.assertEqual([tf.t for tf in due_a], [0.0, 1.0, 2.0, 3.0])
        self.assertEqual([tf.t for tf in due_b], [0.0])  # only the t=0 frame is due at clock_b's time

    def test_fan_out_same_source_timeline_object_is_shared_immutably(self):
        tl = self._timeline()
        streams = fan_out(tl, 4, clock=FakeClock())
        for s in streams:
            self.assertEqual(s.timeline, tl)


if __name__ == "__main__":
    unittest.main()
