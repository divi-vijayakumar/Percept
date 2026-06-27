"""The fovea — SamplingController (perceive much at capture rate, process per-concern density).

Deterministic: a synthetic ``(t, motion)`` capture stream in, processing decisions out. No model.
"""

from __future__ import annotations

import unittest

from percept.behavior import coerce_authored_sense
from percept.sampling import SamplingController, SensePolicy


def _feed(ctrl, cid, fps, dur, motion_fn):
    """Drive a ``fps``-captured stream of length ``dur`` and return the timestamps it chose to PROCESS."""
    return [
        i / fps
        for i in range(int(dur * fps))
        if ctrl.should_process(cid, i / fps, motion_fn(i / fps))
    ]


def _min_gap(ts):
    return min((b - a for a, b in zip(ts, ts[1:])), default=float("inf"))


class SamplingControllerTests(unittest.TestCase):
    def test_coasts_at_base_fps_when_quiet(self):
        # Capture at 30 fps but, with no motion, process only ~base_fps (4/s) — most frames skipped (cheap).
        c = SamplingController(SensePolicy(base_fps=4, max_fps=15, escalate_threshold=0.15))
        ts = _feed(c, "g", fps=30, dur=3.0, motion_fn=lambda t: 0.0)
        self.assertLessEqual(len(ts), 14)  # ≈ 4/s × 3s, never the full 90 captured frames
        self.assertGreaterEqual(len(ts), 9)

    def test_escalates_on_motion_burst_then_relaxes(self):
        # A motion burst in [1.0, 1.5] should densify processing during the burst + hold; quiet stays sparse.
        c = SamplingController(SensePolicy(base_fps=3, max_fps=15, escalate_threshold=0.15, hold_s=0.5))
        ts = _feed(c, "g", fps=30, dur=3.0, motion_fn=lambda t: 1.0 if 1.0 <= t < 1.5 else 0.0)
        quiet = [x for x in ts if x < 1.0]
        burst = [x for x in ts if 1.0 <= x <= 2.0]  # burst + the 0.5s saccade hold
        self.assertGreater(len(burst), 3 * len(quiet) / 3)  # markedly denser than the quiet ~3/s baseline
        self.assertGreaterEqual(len(burst), 8)  # approaching max_fps during the burst+hold

    def test_never_exceeds_max_fps(self):
        # Motion pinned high (always escalated) -> processing capped at max_fps (the cost ceiling).
        c = SamplingController(SensePolicy(base_fps=4, max_fps=15, escalate_threshold=0.15))
        ts = _feed(c, "g", fps=60, dur=2.0, motion_fn=lambda t: 1.0)
        self.assertGreaterEqual(_min_gap(ts), 1.0 / 15 - 1e-6)
        self.assertLessEqual(len(ts), 15 * 2 + 2)

    def test_per_concern_isolation(self):
        c = SamplingController()
        c.configure("slow", SensePolicy(base_fps=2, max_fps=2))
        c.configure("fast", SensePolicy(base_fps=10, max_fps=10))
        slow = _feed(c, "slow", fps=30, dur=3.0, motion_fn=lambda t: 0.0)
        fast = _feed(c, "fast", fps=30, dur=3.0, motion_fn=lambda t: 0.0)
        self.assertGreater(len(fast), len(slow) * 2)  # independent density per concern

    def test_reset_clears_timing(self):
        c = SamplingController(SensePolicy(base_fps=1, max_fps=1))
        self.assertTrue(c.should_process("g", 0.0, 0.0))
        self.assertFalse(c.should_process("g", 0.1, 0.0))  # within the 1 fps interval
        c.reset("g")
        self.assertTrue(c.should_process("g", 0.2, 0.0))  # post-reset processes immediately

    def test_sense_firewall_keeps_valid_drops_garbage_and_clamps(self):
        ok = coerce_authored_sense({"base_fps": 4, "max_fps": 15, "escalate_threshold": 0.2})
        self.assertEqual(ok["sense"], {"base_fps": 4.0, "max_fps": 15.0, "escalate_threshold": 0.2})
        # inverted envelope is repaired (max >= base), out-of-range threshold dropped, rates clamped to 60.
        fixed = coerce_authored_sense({"base_fps": 30, "max_fps": 2, "escalate_threshold": 5})
        self.assertGreaterEqual(fixed["sense"]["max_fps"], fixed["sense"]["base_fps"])
        self.assertNotIn("escalate_threshold", fixed["sense"])
        self.assertEqual(coerce_authored_sense({"base_fps": 999})["sense"]["base_fps"], 60.0)
        self.assertEqual(coerce_authored_sense("nope"), {})
        self.assertEqual(coerce_authored_sense({"base_fps": -1}), {})


if __name__ == "__main__":
    unittest.main()
