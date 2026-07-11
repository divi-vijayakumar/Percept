"""PR 1.5 — a per-goal `exemplar` reaches the judge.

Regression guard: all three real backends implement the deictic `exemplar=` path ("is this THE
mug <reference>?"), but the agent's judge path never passed it, so the individuation capability was
unreachable. This asserts `goal.state['exemplar']` is threaded into `Vision.judge(exemplar=...)`,
and that a goal without one passes `exemplar=None` (byte-for-byte).
"""
from __future__ import annotations

import asyncio
import unittest

import percept
from percept.adapters import Goal, Verdict


class _CapturingVision:
    """Records the exemplar the agent passed to judge()."""

    def __init__(self):
        self.seen_exemplar = "unset"

    async def judge(self, condition, frame, *, roi=None, exemplar=None, corrections=None,
                    frames=None, audio=None, context=None):
        self.seen_exemplar = exemplar
        return Verdict(satisfied=False, confidence=0.9, evidence="n/a")

    async def detect(self, frame, goals):
        return []

    async def detect_events(self, frames, goals):
        return []


def _run(goal_state) -> object:
    vision = _CapturingVision()
    agent = percept.Percept.create(vision=vision, discover_plugins=False)
    agent.add_goal(Goal(id="mug", condition="the mug is here", say="hi", state=goal_state))

    async def go():
        await agent.perceive_judged(("image/jpeg", b"x"), source="camera", t=1.0)

    asyncio.run(go())
    return vision.seen_exemplar


class ExemplarIngressTests(unittest.TestCase):
    def test_exemplar_reaches_judge(self):
        ref = ("image/jpeg", b"REFERENCE_MUG_BYTES")
        self.assertEqual(_run({"exemplar": ref}), ref)

    def test_no_exemplar_passes_none(self):
        self.assertIsNone(_run({}))


if __name__ == "__main__":
    unittest.main()
