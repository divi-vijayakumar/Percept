"""PR 1.4 — the absence-native differentiators fire through the runtime front door.

Regression guard: `check_observability` ("I can't see the baby") and `check_deadlines` ("overdue")
are the SDK's differentiators, but `percept.runtime.run` only drove `check_episode_boundary`, so a
builder using the documented entrypoint got neither. These assert `run()` now emits coverage-loss and
overdue incidents — while a default (mild, no-deadline) goal stays byte-for-byte silent.
"""
from __future__ import annotations

import asyncio
import unittest

from percept import Goal, Percept
from percept.adapters import Verdict
from percept.backends.fake import FakeVision
from percept.clock import FakeClock
from percept.edge.tier0 import gray_frame
from percept.runtime import run

W, H = 8, 8
STILL = gray_frame(W, H, 0)
MOVED = gray_frame(W, H, 255)


def _core(*, state) -> Percept:
    vision = FakeVision(script={"x": Verdict(satisfied=False, confidence=0.9, evidence="not yet")})
    # FakeClock(0): the goal is armed BEFORE run(), so a deadline duration must resolve on the same 0-based
    # timeline the synthetic source stamps its frames on (the worker drives check_deadlines with frame t).
    core = Percept.create(vision=vision, discover_plugins=False, clock=FakeClock(0.0))
    core.add_goal(Goal(id="g", condition="the thing is present", say="fired", state=state))
    return core


async def _collect(core, source, **kw):
    return [inc async for inc in run(core, source=source, **kw)]


class RuntimeAbsenceTests(unittest.TestCase):
    def test_coverage_loss_fires_in_run(self):
        # severe concern: a confident read at t=1 (MOVED escalates) sets coverage; quiet frames then
        # let staleness (2s) elapse -> "I can't see it" fires.
        core = _core(state={"miss_cost": "severe"})
        source = [(0.0, STILL, "x"), (1.0, MOVED, "x")] + [(float(t), STILL, "x") for t in range(2, 7)]
        incidents = asyncio.run(_collect(core, source, coverage_staleness_s=2.0))
        self.assertTrue(any(i.concern_id == "g" for i in incidents), "coverage-loss should fire")

    def test_overdue_deadline_fires_in_run(self):
        core = _core(state={"rearm": "deadline", "deadline_after_s": 3.0})
        source = [(float(t), STILL, "x") for t in range(6)]  # never satisfied; crosses t=3
        incidents = asyncio.run(_collect(core, source))
        self.assertTrue(any(i.concern_id == "g" for i in incidents), "overdue should fire past the deadline")

    def test_default_goal_no_absence_fires(self):
        # mild, no deadline -> neither absence path participates (byte-for-byte the pre-PR path).
        core = _core(state={})
        source = [(float(t), STILL, "x") for t in range(6)]
        incidents = asyncio.run(_collect(core, source, coverage_staleness_s=1.0))
        self.assertEqual(incidents, [])


if __name__ == "__main__":
    unittest.main()
