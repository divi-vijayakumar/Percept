"""Phase 3 (L1): the agent ASSEMBLES the PerceptionSpec input — strip from the FrameWindow + ROI.

A recording vision stub captures what ``judge`` actually receives, so we assert the agent built the
temporal strip (earlier frames, oldest→newest, current excluded) and resolved the ROI per the spec
(focus / hint / full), without any model intelligence. Single-frame goals must still call judge with
``frames=None`` (byte-for-byte today).
"""

import asyncio
import unittest

from percept.adapters import Box, Goal, Verdict
from percept.agent import Percept


class RecordingVision:
    """Records every judge() call; returns a fixed low-confidence verdict (no intelligence)."""

    def __init__(self):
        self.calls: list[dict] = []

    async def judge(self, condition, frame, *, roi=None, exemplar=None, corrections=None,
                    frames=None, audio=None):
        self.calls.append({"frame": frame, "roi": roi, "frames": frames, "audio": audio})
        return Verdict(satisfied=False, confidence=0.5, evidence="stub")

    async def detect(self, frame, goals):
        return []

    async def detect_events(self, frames, goals):
        return []


def _agent(goal: Goal):
    rec = RecordingVision()
    a = Percept.create(vision=rec, discover_plugins=False)
    a.add_goal(goal)
    return a, rec


def _watch(perception: dict, **state) -> Goal:
    st = {"type": "watch", "count": 0, **state}
    if perception:
        st["perception"] = perception
    return Goal(id="g", condition="c", say="s", state=st)


class TestStripAssembly(unittest.TestCase):
    def test_strip_pulls_earlier_frames_from_the_window(self):
        a, rec = _agent(_watch({"temporal": "strip", "n_frames": 3}))

        async def go():
            for tok in ("f0", "f1", "f2"):
                await a.perceive_judged(tok, t=0.0)

        asyncio.run(go())
        last = rec.calls[-1]
        self.assertEqual(last["frame"], "f2")  # the current frame is passed as `frame`
        self.assertEqual(last["frames"], ["f0", "f1"])  # earlier context, oldest→newest, current excluded

    def test_single_frame_goal_passes_no_strip(self):
        a, rec = _agent(_watch({}))  # default perception = single

        async def go():
            for tok in ("f0", "f1"):
                await a.perceive_judged(tok, t=0.0)

        asyncio.run(go())
        self.assertTrue(all(c["frames"] is None for c in rec.calls))  # byte-for-byte today


class TestRoiResolution(unittest.TestCase):
    def _first_roi(self, goal: Goal):
        a, rec = _agent(goal)
        asyncio.run(a.perceive_judged("x", t=0.0))
        return rec.calls[0]["roi"]

    def test_roi_hint_revives_region_hint(self):
        g = _watch({"roi": "hint"}, region_hint=[0.1, 0.2, 0.3, 0.4])
        self.assertEqual(self._first_roi(g), Box(0.1, 0.2, 0.3, 0.4))

    def test_roi_full_is_whole_frame(self):
        self.assertIsNone(self._first_roi(_watch({"roi": "full"})))

    def test_roi_hint_without_a_hint_falls_back(self):
        # no region_hint authored + no focus -> falls back to None (never worse than today)
        self.assertIsNone(self._first_roi(_watch({"roi": "hint"})))


if __name__ == "__main__":
    unittest.main()
