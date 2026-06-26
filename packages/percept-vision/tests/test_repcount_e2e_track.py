from __future__ import annotations

import unittest

from eval.bench import repcount
from percept_harness.detectors import Boundary, DetectorManifest, register
from percept_harness.tier0 import gray_frame


class _EveryFrameBoundary:
    signal = "motion"
    name = "test-every-frame-boundary"

    def reset(self) -> None:
        pass

    def push(self, t, _v):
        return [Boundary(self.name, "peak", t, 1.0)]


register(
    "test-every-frame-boundary",
    _EveryFrameBoundary,
    DetectorManifest(
        "test-every-frame-boundary",
        "synthetic motion",
        "benchmark wiring test",
        "one boundary per frame",
    ),
)


class RepCountE2ETests(unittest.TestCase):
    def test_edge_boundaries_flow_into_core_count(self):
        import eval.count_video as count_video
        import numpy as np

        original = count_video.decode
        try:
            count_video.decode = lambda _path, _fps: (
                [(0.0, gray_frame(2, 2, 0)), (1.0, gray_frame(2, 2, 255)), (2.0, gray_frame(2, 2, 0))],
                [(0.0, np.zeros((2, 2, 3), dtype=np.uint8))] * 3,
            )
            result = repcount.percept_e2e_rep_count(
                "synthetic.mp4",
                action="jump_jack",
                detectors=["test-every-frame-boundary"],
            )
        finally:
            count_video.decode = original

        self.assertEqual(result["count"], 3)
        self.assertEqual(len(result["boundaries"]), 3)
        self.assertEqual(result["goal"], "a completed jump jack repetition")


if __name__ == "__main__":
    unittest.main()
