"""PR 2.2 — 24x7 hardening: the runtime rehydrates on start and saves on exit.

Regression guard: SqlitePersistence/snapshot/rehydrate existed but `run()` never called them, so a
restarted 24/7 watch re-armed from scratch (double-fires, lost goals). These assert run() saves state
on exit and rehydrates armed goals on start — while the default (persistence=None) is byte-for-byte.
"""
from __future__ import annotations

import asyncio
import unittest

from percept import Goal, Percept
from percept.adapters import Verdict
from percept.backends.fake import FakeVision
from percept.edge.tier0 import gray_frame
from percept.persistence import InMemoryPersistence
from percept.runtime import run

STILL = gray_frame(8, 8, 0)
SRC = [(float(i), STILL, "x") for i in range(3)]


def _core():
    core = Percept.create(vision=FakeVision(script={"x": Verdict(satisfied=False, confidence=0.9)}),
                          discover_plugins=False)
    return core


async def _drain(core, **kw):
    return [inc async for inc in run(core, source=SRC, **kw)]


class RuntimePersistenceTests(unittest.TestCase):
    def test_run_saves_state_on_exit(self):
        store = InMemoryPersistence()
        core = _core()
        core.add_goal(Goal(id="zone", condition="a person is here", say="alert"))
        asyncio.run(_drain(core, persistence=store))
        self.assertIsNotNone(store.load("percept"))  # a final save happened

    def test_restart_rehydrates_armed_goals(self):
        store = InMemoryPersistence()
        # session 1: arm a goal, run, save on exit
        c1 = _core()
        c1.add_goal(Goal(id="zone", condition="a person is here", say="alert"))
        asyncio.run(_drain(c1, persistence=store))

        # session 2: a FRESH core with NO goals, restore=True -> the armed goal comes back
        c2 = _core()
        self.assertEqual(c2.goals.active(), [])
        asyncio.run(_drain(c2, persistence=store, restore=True))
        self.assertIn("zone", [g.id for g in c2.goals.active()])

    def test_no_persistence_is_byte_for_byte(self):
        core = _core()
        core.add_goal(Goal(id="zone", condition="x", say="y"))
        # must not raise / must not require a store
        incidents = asyncio.run(_drain(core))
        self.assertEqual(incidents, [])


if __name__ == "__main__":
    unittest.main()
