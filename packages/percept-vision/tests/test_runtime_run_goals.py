"""R-0: percept.run(source=..., goals=[...]) convenience — builds a core internally so the one-line
quickstart is real. The canonical run(core, source=...) form is unchanged and takes precedence."""
from __future__ import annotations

import asyncio
import unittest

import percept
from percept.adapters import Goal
from percept.edge.tier0 import gray_frame

STILL = gray_frame(8, 8, 0)


async def _frames(n=3):
    for i in range(n):
        yield (float(i), STILL, "tok")


class RunGoalsTests(unittest.TestCase):
    def test_run_with_goal_strings_builds_a_core(self):
        async def go():
            incs = []
            async for inc in percept.run(
                source=_frames(), goals=["the milk is boiling over"], vision="fake"
            ):
                incs.append(inc)
            return incs
        asyncio.run(go())  # completes without a pre-built core (no TypeError)

    def test_goal_coercion_str_dict_goal(self):
        from percept.runtime.worker import _as_goal
        g0 = _as_goal("a cup is up", 0)
        self.assertEqual((g0.id, g0.condition, g0.say), ("g0", "a cup is up", "a cup is up"))
        g1 = _as_goal({"condition": "smoke"}, 1)
        self.assertEqual((g1.id, g1.condition), ("g1", "smoke"))
        g2 = _as_goal(Goal(id="x", condition="c", say="s"), 2)
        self.assertEqual(g2.id, "x")  # passthrough

    def test_no_core_and_no_goals_raises(self):
        async def go():
            async for _ in percept.run(source=_frames(), vision="fake"):
                pass
        with self.assertRaises(TypeError):
            asyncio.run(go())

    def test_explicit_core_still_works(self):
        async def go():
            core = percept.Percept.create(vision="fake", discover_plugins=False)
            core.add_goal(Goal(id="z", condition="x", say="hi"))
            async for _ in percept.run(core, source=_frames()):
                pass
        asyncio.run(go())  # canonical positional-core form unchanged


if __name__ == "__main__":
    unittest.main()
