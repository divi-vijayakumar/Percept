"""A-clock switch (NO-OP) + A-4 decimation fix.

These lock three things:
  * ``timebase="media"`` reproduces today's ``t = i/src_fps`` (video) / ``(i+step)/sr`` (audio)
    arithmetic BYTE-FOR-BYTE — the regression net a later PR flips against.
  * ``timebase="wall"`` stamps ``t = now()`` per kept frame / emitted chunk, while the
    ``AudioFrame.t_start/t_end`` PCM-slice offsets stay MEDIA under both modes.
  * A source reporting one fps but DELIVERING another surfaces ``fps_misreported`` (A-4) —
    the previously-silent half-rate decimation.
"""
from __future__ import annotations

import os
import queue
import sys
import tempfile
import threading
import time
import unittest
import wave
from unittest.mock import patch

from percept.runtime.audio import AudioSourceConfig, _capture_wav
from percept.runtime.source import SourceConfig, _capture


# --- fakes ---------------------------------------------------------------------------------------
class _FakeGray:
    def __init__(self, n):
        self.n = n

    def flatten(self):
        return [0] * self.n


class _FakeBuffer:
    def tobytes(self):
        return b"jpeg"


class _FakeCv2:
    CAP_FFMPEG = 1900
    CAP_PROP_FPS = 5
    COLOR_BGR2GRAY = 6

    def __init__(self, cap):
        self.cap = cap

    def VideoCapture(self, *args):
        return self.cap

    def cvtColor(self, bgr, code):
        return bgr

    def resize(self, gray, size):
        return _FakeGray(size[0] * size[1])

    def imencode(self, ext, bgr):
        return True, _FakeBuffer()


class _PacedCap:
    """Reports ``reported_fps`` but DELIVERS a frame every ``1/delivered_fps`` of a fake clock.

    ``clock`` is a 1-elem list advanced on every ``read()`` — so the injected ``now`` sees the frame
    cadence a real (blocking) ``cap.read()`` would impose, with zero real sleeping."""

    def __init__(self, n, reported_fps, delivered_fps, clock):
        self.left = n
        self.reported = reported_fps
        self.dt = 1.0 / delivered_fps
        self.clock = clock
        self.released = False

    def isOpened(self):
        return True

    def get(self, prop):
        return self.reported

    def read(self):
        if self.left <= 0:
            return (False, None)
        self.left -= 1
        self.clock[0] += self.dt
        return (True, object())

    def release(self):
        self.released = True


def _drain(q):
    out = []
    while True:
        item = q.get_nowait()
        if item is None:
            break
        out.append(item)
    return out


# --- video: media lane is byte-for-byte today ----------------------------------------------------
class VideoTimebaseTests(unittest.TestCase):
    def test_media_timebase_matches_pre_change_t_sequence(self):
        clock = [0.0]
        cap = _PacedCap(n=24, reported_fps=30.0, delivered_fps=30.0, clock=clock)
        cv2 = _FakeCv2(cap)
        q: queue.Queue = queue.Queue()
        cfg = SourceConfig(fps=4.0, timebase="media")
        with patch.dict(sys.modules, {"cv2": cv2}):
            _capture("v.mp4", cfg, q, threading.Event(), now=lambda: clock[0])
        ts = [it[0] for it in _drain(q)]
        step = 8  # round(30 / 4) — the unclamped reported fps drives the decimation
        expected = [i / 30.0 for i in range(24) if i % step == 0]
        self.assertEqual(ts, expected)  # 0/30, 8/30, 16/30 — exact media offsets

    def test_wall_timebase_stamps_now_not_media(self):
        clock = [100.0]
        cap = _PacedCap(n=24, reported_fps=30.0, delivered_fps=30.0, clock=clock)
        cv2 = _FakeCv2(cap)
        q: queue.Queue = queue.Queue()
        cfg = SourceConfig(fps=4.0, timebase="wall")
        with patch.dict(sys.modules, {"cv2": cv2}):
            _capture("v.mp4", cfg, q, threading.Event(), now=lambda: clock[0])
        ts = [it[0] for it in _drain(q)]
        self.assertTrue(all(t >= 100.0 for t in ts))       # wall clock, not media ~1e0
        self.assertEqual(ts, sorted(ts))                    # monotonic
        self.assertNotEqual(ts[0], 0.0)                     # NOT the media offset i/src_fps


# --- A-4: delivered-fps misreport ----------------------------------------------------------------
class FpsMisreportTests(unittest.TestCase):
    def test_fps_misreported_fires_when_delivered_differs(self):
        clock = [0.0]
        cap = _PacedCap(n=120, reported_fps=30.0, delivered_fps=15.0, clock=clock)
        cv2 = _FakeCv2(cap)
        q: queue.Queue = queue.Queue()
        health: list = []
        cfg = SourceConfig(fps=4.0, timebase="media")
        with patch.dict(sys.modules, {"cv2": cv2}):
            _capture("v.mp4", cfg, q, threading.Event(), health=health, now=lambda: clock[0])
        events = [d for k, d in health if k == "fps_misreported"]
        self.assertEqual(len(events), 1)  # fires once, not per-frame spam
        self.assertEqual(events[0]["reported"], 30.0)
        self.assertAlmostEqual(events[0]["measured"], 15.0, delta=1.5)

    def test_no_misreport_when_honest(self):
        clock = [0.0]
        cap = _PacedCap(n=120, reported_fps=30.0, delivered_fps=30.0, clock=clock)
        cv2 = _FakeCv2(cap)
        q: queue.Queue = queue.Queue()
        health: list = []
        cfg = SourceConfig(fps=4.0, timebase="media")
        with patch.dict(sys.modules, {"cv2": cv2}):
            _capture("v.mp4", cfg, q, threading.Event(), health=health, now=lambda: clock[0])
        self.assertEqual([k for k, _ in health if k == "fps_misreported"], [])

    def test_instant_offline_reads_do_not_false_fire(self):
        # A pre-decoded/instant fixture (real monotonic clock, ~0 elapsed span) must NOT trip the
        # misreport heuristic — otherwise every offline replay would false-fire.
        clock = [0.0]
        cap = _PacedCap(n=200, reported_fps=30.0, delivered_fps=15.0, clock=clock)
        cv2 = _FakeCv2(cap)
        q: queue.Queue = queue.Queue()
        health: list = []
        cfg = SourceConfig(fps=4.0, timebase="media")
        with patch.dict(sys.modules, {"cv2": cv2}):
            _capture("v.mp4", cfg, q, threading.Event(), health=health, now=time.monotonic)
        self.assertEqual([k for k, _ in health if k == "fps_misreported"], [])


# --- audio: media lane byte-for-byte; wall stamps now; PCM offsets stay media ---------------------
class AudioTimebaseTests(unittest.TestCase):
    def _wav(self, sr=16_000, secs=1):
        path = tempfile.mktemp(suffix=".wav")
        with wave.open(path, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sr)
            w.writeframes(b"\x00\x00" * (sr * secs))
        return path

    def test_media_timebase_event_t_equals_media_offsets(self):
        path = self._wav()
        q: queue.Queue = queue.Queue()
        cfg = AudioSourceConfig(chunk_s=0.25, timebase="media")
        try:
            _capture_wav(path, cfg, q, threading.Event(), None)
        finally:
            os.unlink(path)
        rows = _drain(q)
        sr, step = 16_000, 4_000
        for i, (t_event, frame, _e) in enumerate(rows):
            self.assertEqual(t_event, (i * step + step) / sr)   # (i+step)/sr media event t
            self.assertEqual(frame.t_start, (i * step) / sr)
            self.assertEqual(frame.t_end, (i * step + step) / sr)

    def test_wall_timebase_stamps_now_but_frame_offsets_stay_media(self):
        path = self._wav()
        q: queue.Queue = queue.Queue()
        counter = [0]

        def fake_now():
            counter[0] += 1
            return 500.0 + counter[0]

        cfg = AudioSourceConfig(chunk_s=0.25, timebase="wall")
        try:
            _capture_wav(path, cfg, q, threading.Event(), None, now=fake_now)
        finally:
            os.unlink(path)
        rows = _drain(q)
        sr, step = 16_000, 4_000
        for i, (t_event, frame, _e) in enumerate(rows):
            self.assertGreaterEqual(t_event, 500.0)             # wall now(), not media
            # t_start/t_end are PCM-slice offsets — MEDIA under wall too (two jobs, two fields)
            self.assertEqual(frame.t_start, (i * step) / sr)
            self.assertEqual(frame.t_end, (i * step + step) / sr)


# --- worker wiring: fps_misreported must reach on_health, not sit in an unread list ---------------
class WorkerHealthWiringTests(unittest.TestCase):
    def test_fps_misreported_reaches_on_health(self):
        import asyncio

        from percept import Goal, Percept
        from percept.adapters import Verdict
        from percept.backends.fake import FakeVision
        from percept.edge.tier0 import gray_frame
        from percept.runtime import run

        STILL = gray_frame(8, 8, 0)

        async def fake_open_source(uri, *, fps=4.0, timebase="media", dropped=None, health=None, **_kw):
            if health is not None:
                health.append(("fps_misreported", {"reported": 30.0, "measured": 15.0}))
            for i in range(3):
                yield (float(i), STILL, ("image/jpeg", b"x"))

        vision = FakeVision(script={"quiet": Verdict(satisfied=False, confidence=0.9, evidence="n")})
        core = Percept.create(vision=vision, discover_plugins=False)
        core.add_goal(Goal(id="cry", condition="the baby is crying", say="awake"))

        events: list = []

        async def go():
            with patch("percept.runtime.source.open_source", fake_open_source):
                async for _ in run(core, source="webcam:0", on_health=lambda e: events.append(e)):
                    pass

        asyncio.run(go())
        kinds = [getattr(e, "kind", None) for e in events]
        self.assertIn("fps_misreported", kinds)
        ev = next(e for e in events if getattr(e, "kind", None) == "fps_misreported")
        self.assertEqual(ev.data, {"reported": 30.0, "measured": 15.0})


if __name__ == "__main__":
    unittest.main()
