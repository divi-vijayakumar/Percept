from __future__ import annotations

import math
import unittest

import percept.harness.detectors as harness_detectors
from eval.bench import repcount
from percept.harness.detectors import Boundary, DetectorManifest, register
from percept.harness.tier0 import gray_frame


class _EveryFrameBoundary:
    signal = "motion"
    name = "test-every-frame-boundary"

    def reset(self) -> None:
        pass

    def push(self, t, _v):
        return [Boundary(self.name, "peak", t, 1.0)]


class RepCountE2ETests(unittest.TestCase):
    def setUp(self):
        # register in setUp (not module level) so another suite's registry reset
        # (test_registration's tearDown -> clear_detectors) can't leave us unregistered.
        if "test-every-frame-boundary" not in harness_detectors.registered():
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

    def test_rhythm_path_counts_full_cycles_not_every_frame(self):
        # The default (pose-channels) path runs the rhythm sense: the eye MEASURES a channel and the brain
        # counts COMPLETE cycles. Patch the eye to emit a clean 5-cycle oscillation; expect 5, not 60.
        import eval.count_video as count_video
        import numpy as np

        N, T = 60, 12  # 60 frames, 5 oscillations of period 12

        class _FakeEye:
            def __init__(self):
                self.i = 0

            def measure(self, _frame):
                v = math.sin(2 * math.pi * self.i / T)
                self.i += 1
                return {"limb-vertical": v}

        orig_decode, orig_make = count_video.decode, harness_detectors.make_detector
        try:
            count_video.decode = lambda _p, _f: (
                [(i / 6.0, gray_frame(2, 2, 0)) for i in range(N)],
                [(i / 6.0, np.zeros((2, 2, 3), dtype=np.uint8)) for i in range(N)],
            )
            harness_detectors.make_detector = lambda name: _FakeEye()
            result = repcount.percept_e2e_rep_count("synthetic.mp4", action="jump_jack")  # default eye
        finally:
            count_video.decode, harness_detectors.make_detector = orig_decode, orig_make

        self.assertEqual(result["detectors"], ["pose-channels"])
        self.assertEqual(result["count"], 5)  # full cycles, NOT 60 frames
        self.assertEqual(len(result["boundaries"]), 5)
        self.assertEqual(result["boundaries"][0]["kind"], "rep")
        self.assertIn("span", result["boundaries"][0])  # first-class start/peak/stop


if __name__ == "__main__":
    unittest.main()
