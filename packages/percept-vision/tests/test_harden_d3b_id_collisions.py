"""AGENT-D / D3b — goal-id collisions must never silently drop a watch.

``GoalRegistry.create`` is a bare dict write (last-writer-wins). Re-using an id silently DROPS the prior
goal — a lost watch instruction, with no signal. This pins the registry-level guard that survived the
v0.6 voice diet: a genuine id overwrite is REPORTED (loud warning), never lost silently.

(The old voice-arm slug path — ``RuntimeSession._arm_slug`` + the thin ``arm_goal`` tool that hash-
suffixed a condition into a collision-resistant id — moved out of the SDK with the voice embodiment. The
one shipped voice app now routes raw language through ``core.arm_from_language``, where the conductor
authors the goal id; so the slug-collision case lives with the template, and this file guards the
registry invariant that backstops every author.)
"""
from __future__ import annotations

import unittest

from percept.adapters import Goal
from percept.goals import GoalRegistry


class D3bIdCollisionTests(unittest.TestCase):
    def test_registry_create_reports_a_genuine_overwrite(self):
        reg = GoalRegistry()
        reg.create(Goal(id="dup", condition="first watch", say="!"))
        with self.assertLogs("percept.goals", level="WARNING") as cm:
            reg.create(Goal(id="dup", condition="second watch", say="!"))
        self.assertTrue(any("dup" in m for m in cm.output),
                        "an id overwrite must be reported loudly, not lost silently")


if __name__ == "__main__":
    unittest.main()
