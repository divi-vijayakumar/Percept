"""Runtime source helpers: URI normalization, capture opening, and backpressure."""

from __future__ import annotations

import queue
import sys
import threading
import unittest
from unittest.mock import patch

from percept.runtime.source import SourceConfig, _capture, _open_capture, _put_latest


class _FakeCv2:
    CAP_FFMPEG = 1900
    CAP_PROP_FPS = 5
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


class _FakeGray:
    def __init__(self, n):
        self.n = n

    def flatten(self):
        return [0] * self.n


class _FakeBuffer:
    def tobytes(self):
        return b"jpeg"


class _FakeCap:
    def __init__(self, opened=True, reads=None, fps=30.0):
        self.opened = opened
        self.reads = list(reads or [])
        self.fps = fps
        self.released = False

    def isOpened(self):
        return self.opened

    def get(self, prop):
        return self.fps

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
        self.assertEqual(item[0], 0.0)
        self.assertEqual(item[2], ("image/jpeg", b"jpeg"))
        self.assertIsNone(q.get_nowait())
        self.assertTrue(cap.released)


if __name__ == "__main__":
    unittest.main()
