"""Runtime source helpers: URI normalization, capture opening, and backpressure."""

from __future__ import annotations

import asyncio
import queue
import sys
import threading
import unittest
from unittest.mock import patch

from percept.runtime import SourceTracker
from percept.runtime.source import (
    SourceConfig,
    _capture,
    _open_capture,
    _put_latest,
    _put_terminal,
)


class _FakeCv2:
    CAP_FFMPEG = 1900
    CAP_PROP_FPS = 5
    CAP_PROP_POS_MSEC = 0
    COLOR_BGR2GRAY = 6

    def __init__(self, cap=None):
        self.calls = []
        self.cap = cap

    def VideoCapture(self, *args):
        self.calls.append(args)
        return self.cap if self.cap is not None else object()

    def cvtColor(self, bgr, code):
        return bgr

    def resize(self, gray, size):
        return _FakeGray(size[0] * size[1])

    def imencode(self, ext, bgr):
        return True, _FakeBuffer()


class _ReconnectCv2(_FakeCv2):
    def __init__(self, caps):
        super().__init__()
        self.caps = list(caps)

    def VideoCapture(self, *args):
        self.calls.append(args)
        return self.caps.pop(0)


class _FakeGray:
    def __init__(self, n):
        self.n = n

    def flatten(self):
        return [0] * self.n


class _FakeBuffer:
    def tobytes(self):
        return b"\xff\xd8jpeg\xff\xd9"


class _FakeCap:
    def __init__(self, opened=True, reads=None, fps=30.0):
        self.opened = opened
        self.reads = list(reads or [])
        self.fps = fps
        self.position_ms = 0.0
        self.released = False

    def isOpened(self):
        return self.opened

    def get(self, prop):
        return self.position_ms if prop == _FakeCv2.CAP_PROP_POS_MSEC else self.fps

    def read(self):
        return self.reads.pop(0) if self.reads else (False, None)

    def release(self):
        self.released = True


class RuntimeSourceTests(unittest.TestCase):
    def test_rtsp_uses_ffmpeg_capture_when_tcp_enabled(self):
        cv2 = _FakeCv2()
        cap = _open_capture(cv2, "rtsp://camera/stream", SourceConfig(rtsp_tcp=True))
        self.assertIsNotNone(cap)
        self.assertEqual(cv2.calls, [("rtsp://camera/stream", _FakeCv2.CAP_FFMPEG)])

    def test_rtsp_can_use_plain_capture(self):
        cv2 = _FakeCv2()
        _open_capture(cv2, "rtsp://camera/stream", SourceConfig(rtsp_tcp=False))
        self.assertEqual(cv2.calls, [("rtsp://camera/stream",)])

    def test_put_latest_drops_oldest_when_full(self):
        q = queue.Queue(maxsize=1)
        _put_latest(q, "old")
        _put_latest(q, "new")
        self.assertEqual(q.get_nowait(), "new")

    def test_put_latest_counts_drops(self):
        q = queue.Queue(maxsize=1)
        dropped = [0]
        _put_latest(q, "old", dropped)
        _put_latest(q, "new", dropped)  # full -> discard "old", bump the counter
        self.assertEqual(dropped[0], 1)
        self.assertEqual(q.get_nowait(), "new")

    def test_terminal_waits_behind_full_evidence_and_stops_on_cancel(self):
        q = queue.Queue(maxsize=1)
        q.put_nowait("evidence")
        stop = threading.Event()
        done = threading.Event()

        def publish():
            _put_terminal(q, None, stop)
            done.set()

        thread = threading.Thread(target=publish, daemon=True)
        thread.start()
        self.assertFalse(done.wait(0.15))
        self.assertEqual(q.get_nowait(), "evidence")
        self.assertTrue(done.wait(1.0))
        self.assertIsNone(q.get_nowait())

        q.put_nowait("evidence")
        done.clear()
        stop.clear()
        thread = threading.Thread(target=publish, daemon=True)
        thread.start()
        self.assertFalse(done.wait(0.15))
        stop.set()
        self.assertTrue(done.wait(1.0))
        self.assertEqual(q.get_nowait(), "evidence")

    def test_capture_reports_open_failure(self):
        cap = _FakeCap(opened=False)
        cv2 = _FakeCv2(cap)
        q = queue.Queue()
        with patch.dict(sys.modules, {"cv2": cv2}):
            _capture("video.mp4", SourceConfig(), q, threading.Event())
        first = q.get_nowait()
        self.assertIsInstance(first, RuntimeError)
        self.assertIsNone(q.get_nowait())
        self.assertTrue(cap.released)

    def test_capture_yields_finite_video_frames(self):
        cap = _FakeCap(reads=[(True, object()), (False, None)], fps=4.0)
        cv2 = _FakeCv2(cap)
        q = queue.Queue()
        with patch.dict(sys.modules, {"cv2": cv2}):
            _capture("video.mp4", SourceConfig(fps=4.0), q, threading.Event())
        item = q.get_nowait()
        self.assertEqual(item.source_time, 0.0)
        self.assertIsNotNone(item.observed_time)
        self.assertEqual(item[0], 0.0)
        self.assertEqual(item[2], ("image/jpeg", b"\xff\xd8jpeg\xff\xd9"))
        self.assertEqual(item.source_id, "video:video.mp4")
        self.assertIsNone(q.get_nowait())
        self.assertTrue(cap.released)

    def test_nonadvancing_opencv_position_is_not_relabelled_as_source_time(self):
        cap = _FakeCap(
            reads=[(True, object()), (True, object()), (False, None)],
            fps=4.0,
        )
        cv2 = _FakeCv2(cap)  # this fake mirrors an unsupported POS_MSEC: permanent 0.0
        q = queue.Queue()
        with patch.dict(sys.modules, {"cv2": cv2}):
            _capture("video.mp4", SourceConfig(fps=4.0), q, threading.Event())
        first, second = q.get_nowait(), q.get_nowait()
        self.assertEqual(first.source_time, 0.0)
        self.assertIsNone(second.source_time)
        self.assertIsNone(q.get_nowait())

    def test_cancel_while_video_producer_stalled_does_not_strand_to_thread_get(self):
        import percept.runtime.source as source_mod

        def stalled_capture(uri, config, q, stop, dropped, *, health=None):
            # Deliberately ignores cancellation long enough to reproduce an empty capture queue.
            stop.wait(30.0)

        async def consume():
            async for _ in source_mod.open_source("fake.mp4"):
                pass

        async def exercise():
            with patch.object(source_mod, "_capture", stalled_capture):
                task = asyncio.create_task(consume())
                await asyncio.sleep(0.1)
                task.cancel()
                await asyncio.wait_for(asyncio.shield(task), timeout=5.0)

        with self.assertRaises(asyncio.CancelledError):
            asyncio.run(exercise())

    def test_reconnect_starts_a_new_epoch_at_sequence_zero(self):
        caps = [
            _FakeCap(reads=[(True, object()), (False, None)], fps=4.0),
            _FakeCap(reads=[(True, object()), (False, None)], fps=4.0),
        ]
        cv2 = _ReconnectCv2(caps)
        q = queue.Queue()
        with patch.dict(sys.modules, {"cv2": cv2}), patch("time.sleep", return_value=None):
            _capture(
                "video.mp4",
                SourceConfig(fps=4.0, reconnects=1),
                q,
                threading.Event(),
            )
        first, second = q.get_nowait(), q.get_nowait()
        self.assertEqual((first.epoch, first.sequence), (0, 0))
        self.assertEqual((second.epoch, second.sequence), (1, 0))
        tracker = SourceTracker()
        self.assertTrue(tracker.inspect(first).accepted)
        reconnect = tracker.inspect(second)
        self.assertTrue(reconnect.accepted)
        self.assertEqual(reconnect.events, ("source_reconnect",))
        self.assertIsNone(q.get_nowait())

    def test_queue_eviction_becomes_a_sequence_gap(self):
        tracker = SourceTracker()
        q = queue.Queue(maxsize=1)
        first = self._captured_envelope(0)
        self.assertTrue(tracker.inspect(first).accepted)
        _put_latest(q, self._captured_envelope(1))
        _put_latest(q, self._captured_envelope(2))
        gap = tracker.inspect(q.get_nowait())
        self.assertEqual(gap.events, ("source_gap",))
        self.assertEqual(gap.coverage.missing_sequences, 1)

    @staticmethod
    def _captured_envelope(sequence):
        from percept.runtime import SourceEnvelope, SourceFormat
        from percept.edge.tier0 import gray_frame

        return SourceEnvelope(
            source_id="video",
            modality="video",
            epoch=0,
            sequence=sequence,
            observed_time=float(sequence),
            source_time=float(sequence),
            format=SourceFormat(media_type="image/jpeg"),
            payload=("image/jpeg", b"\xff\xd8jpeg\xff\xd9"),
            signal=gray_frame(2, 2, 0),
        )


if __name__ == "__main__":
    unittest.main()
