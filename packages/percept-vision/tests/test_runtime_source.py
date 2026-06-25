"""Runtime source helpers: URI normalization, capture opening, and backpressure."""

from __future__ import annotations

import queue
import unittest

from percept.runtime.source import SourceConfig, _open_capture, _put_latest


class _FakeCv2:
    CAP_FFMPEG = 1900

    def __init__(self):
        self.calls = []

    def VideoCapture(self, *args):
        self.calls.append(args)
        return object()


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


if __name__ == "__main__":
    unittest.main()
