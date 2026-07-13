"""AGENT-D / D3b — voice-arm id collisions must never silently drop a watch.

The voice-arm slug was ``condition.replace(' ', '-')[:40]`` and ``GoalRegistry.create`` is a bare
dict write (last-writer-wins, SILENT). So two conditions sharing a 40-char prefix collapse to ONE
armed goal — a lost user instruction, with no signal. These tests pin:
  * two long conditions sharing a 40-char prefix arm as TWO distinct goals (hash-suffixed slug);
  * a genuine id overwrite in the registry is REPORTED (loud warning), never silent.
"""
from __future__ import annotations

import asyncio
import unittest

import percept
from percept.adapters import Goal
from percept.goals import GoalRegistry
from percept.runtime.session import RuntimeSession


def _core():
    return percept.Percept.create(vision="fake", discover_plugins=False)


class D3bIdCollisionTests(unittest.TestCase):
    def test_two_long_conditions_sharing_a_40char_prefix_arm_as_two_goals(self):
        # Both conditions are identical for the first ~50 chars, differing only in the final word.
        c1 = "the red car is parked in the driveway near the mailbox on monday"
        c2 = "the red car is parked in the driveway near the mailbox on tuesday"

        async def go():
            sess = RuntimeSession(_core())
            await sess._arm_goal_tool({"condition": c1})  # noqa: SLF001
            await sess._arm_goal_tool({"condition": c2})  # noqa: SLF001
            return (await sess._whats_armed_tool({}))["armed"]  # noqa: SLF001

        armed = asyncio.run(go())
        ids = {a["id"] for a in armed}
        self.assertEqual(len(armed), 2, f"both watches must survive; got {armed}")
        self.assertEqual(len(ids), 2, "the two goals must have DISTINCT ids (no silent overwrite)")

    def test_registry_create_reports_a_genuine_overwrite(self):
        reg = GoalRegistry()
        reg.create(Goal(id="dup", condition="first watch", say="!"))
        with self.assertLogs("percept.goals", level="WARNING") as cm:
            reg.create(Goal(id="dup", condition="second watch", say="!"))
        self.assertTrue(any("dup" in m for m in cm.output),
                        "an id overwrite must be reported loudly, not lost silently")


if __name__ == "__main__":
    unittest.main()
