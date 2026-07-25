"""AGENT-D / D3b — goal-id collisions must never drop a watch.

Goal ids are lifetime-unique inside ``GoalRegistry``. A duplicate is rejected before mutation, including
after complete/cancel, so direct registry callers cannot overwrite a standing watch or inherit its state.

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
    def test_registry_create_rejects_a_genuine_overwrite(self):
        reg = GoalRegistry()
        first = reg.create(Goal(id="dup", condition="first watch", say="!"))
        with self.assertRaisesRegex(ValueError, "lifetime-unique"):
            reg.create(Goal(id="dup", condition="second watch", say="!"))
        self.assertIs(reg.get("dup"), first)


if __name__ == "__main__":
    unittest.main()
