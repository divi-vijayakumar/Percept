"""SensorPlan compiler (Build 5) — the deterministic goal->plan compiler.

Proves: (1) every goal type compiles to a typed plan that reflects what it will actually run (incl. the
opt-in Build 2/3 knobs); (2) compiling never changes firing (program_of is read, not rewritten).
"""

from __future__ import annotations

import unittest

from percept import Goal
from percept.behavior import program_of
from percept.sensorplan import (
    PRIMITIVES,
    SensorPlan,
    sensor_plan_for,
)


def goal(condition="x", **state):
    return Goal(id="g", condition=condition, say="!", state=state)


class CompilerTests(unittest.TestCase):
    def test_rhythm_goal_compiles_to_periodic_plan(self):
        plan = sensor_plan_for(goal("a completed squat repetition", type="rhythm"))
        self.assertIsInstance(plan, SensorPlan)
        self.assertEqual(plan.primitives, ("rhythm", "single"))
        self.assertEqual(plan.event_model["kind"], "periodic")
        # opt-in individuation knobs surface at their defaults (off)
        self.assertEqual(plan.event_model["smooth_s"], 0.0)
        self.assertFalse(plan.event_model["canonical_select"])

    def test_rhythm_plan_surfaces_authored_knobs(self):
        g = goal("rep", type="rhythm", rhythm={"channel": "limb-vertical", "smooth_s": 0.15,
                                               "canonical_select": True, "cadence_s": 2.0})
        plan = sensor_plan_for(g)
        self.assertEqual(plan.event_model["smooth_s"], 0.15)
        self.assertTrue(plan.event_model["canonical_select"])
        self.assertEqual(plan.event_model["period_hint_s"], 2.0)
        self.assertEqual(plan.event_model["channel"], "limb-vertical")

    def test_watch_goal_is_edge_judge_plan(self):
        plan = sensor_plan_for(goal("the door is open"))  # default watch -> judge
        self.assertEqual(plan.primitives[0], "judge")
        self.assertEqual(plan.event_model["kind"], "edge")

    def test_count_and_transition_and_absence_plans(self):
        self.assertEqual(sensor_plan_for(goal("a cup", type="count")).event_model["kind"], "distinct")
        tr = sensor_plan_for(goal("x", type="transition", **{"from": "raw", "to": "cooked"}))
        self.assertEqual(tr.event_model["kind"], "transition")
        ab = sensor_plan_for(goal("x", rearm="deadline", deadline_after_s=30.0))
        self.assertEqual(ab.event_model["kind"], "absence")
        self.assertEqual(ab.event_model["deadline_after_s"], 30.0)

    def test_as_dict_carries_the_compiled_shape(self):
        g = goal("rep", type="rhythm")
        plan = sensor_plan_for(g)
        d = plan.as_dict()
        self.assertEqual(d["primitives"], ["rhythm", "single"])
        self.assertEqual(d["event_model"]["kind"], "periodic")
        self.assertIn("cadence", d)
        self.assertIn("perception", d)

    def test_compiling_does_not_change_program(self):
        g = goal("rep", type="rhythm")
        before = program_of(g)
        sensor_plan_for(g)
        after = program_of(g)  # compiling a plan must not alter the compiled behavior program
        self.assertEqual((before.sense, before.perception), (after.sense, after.perception))

    def test_registry_primitives_are_self_describing(self):
        for name, prim in PRIMITIVES.items():
            self.assertEqual(prim.name, name)
            self.assertIn(prim.kind, {"sense", "perception"})
            self.assertTrue(prim.when_to_use)  # the cue sensor_plan_for reads to compose it


if __name__ == "__main__":
    unittest.main()
