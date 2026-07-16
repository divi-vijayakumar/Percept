"""SensorPlan compiler (Build 5) — the deterministic goal->plan compiler and the authored-plan firewall.

Proves: (1) every goal type compiles to a typed plan that reflects what it will actually run (incl. the
opt-in Build 2/3 knobs); (2) an untrusted authored plan is reduced to safe, bounded goal.state keys that
the EXISTING consumers understand; (3) compiling never changes firing (program_of is read, not rewritten).
"""

from __future__ import annotations

import unittest

from percept import Goal
from percept.behavior import program_of
from percept.sensorplan import (
    PRIMITIVES,
    SensorPlan,
    attach_sensor_plan,
    coerce_authored_sensor_plan,
    sensor_plan_for,
)


def goal(condition="x", **state):
    return Goal(id="g", condition=condition, say="!", state=state)


class CompilerTests(unittest.TestCase):
    def test_rhythm_goal_compiles_to_periodic_plan(self):
        plan = sensor_plan_for(goal("a completed squat repetition", type="rhythm"))
        self.assertIsInstance(plan, SensorPlan)
        self.assertEqual(plan.primitives, ("rhythm", "single"))
        self.assertEqual(plan.measurements, ("pose-channels",))
        self.assertEqual(plan.event_model["kind"], "periodic")
        self.assertEqual(plan.confirmation, {"vlm": "sparse", "when": "candidate_boundary"})
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
        self.assertEqual(plan.confirmation["vlm"], "each_escalation")
        self.assertEqual(plan.measurements, ("vlm",))

    def test_count_and_transition_and_absence_plans(self):
        self.assertEqual(sensor_plan_for(goal("a cup", type="count")).event_model["kind"], "distinct")
        tr = sensor_plan_for(goal("x", type="transition", **{"from": "raw", "to": "cooked"}))
        self.assertEqual(tr.event_model["kind"], "transition")
        ab = sensor_plan_for(goal("x", rearm="deadline", deadline_after_s=30.0))
        self.assertEqual(ab.event_model["kind"], "absence")
        self.assertEqual(ab.event_model["deadline_after_s"], 30.0)

    def test_attach_stores_plan_dict_in_state(self):
        g = goal("rep", type="rhythm")
        attach_sensor_plan(g)
        self.assertIn("sensor_plan", g.state)
        self.assertEqual(g.state["sensor_plan"]["measurements"], ["pose-channels"])

    def test_compiling_does_not_change_program(self):
        g = goal("rep", type="rhythm")
        before = program_of(g)
        attach_sensor_plan(g)
        after = program_of(g)  # storing the plan must not alter the compiled behavior program
        self.assertEqual((before.sense, before.perception), (after.sense, after.perception))


class FirewallTests(unittest.TestCase):
    def test_unknown_primitives_dropped_known_kept(self):
        out = coerce_authored_sensor_plan({"primitives": ["rhythm", "telekinesis", "window"]})
        self.assertEqual(out["sensor_plan"]["primitives"], ["rhythm", "window"])

    def test_cadence_delegates_to_sense_firewall_and_clamps_fps(self):
        out = coerce_authored_sensor_plan({"cadence": {"base_fps": 2, "max_fps": 999}})
        self.assertEqual(out["sense"]["max_fps"], 60.0)  # _MAX_SENSE_FPS clamp via coerce_authored_sense
        self.assertEqual(out["sense"]["base_fps"], 2.0)

    def test_event_model_knobs_validated_via_rhythm_firewall(self):
        ok = coerce_authored_sensor_plan(
            {"event_model": {"channel": "hip", "smooth_s": 0.2, "canonical_select": True, "period_hint_s": 2.0}}
        )
        self.assertEqual(ok["rhythm"]["smooth_s"], 0.2)
        self.assertIs(ok["rhythm"]["canonical_select"], True)
        self.assertEqual(ok["rhythm"]["cadence_s"], 2.0)
        # unbounded / wrong-typed knobs are dropped by the delegated firewall
        bad = coerce_authored_sensor_plan({"event_model": {"smooth_s": 5.0, "canonical_select": "yes"}})
        self.assertNotIn("rhythm", bad)

    def test_perception_clamped(self):
        out = coerce_authored_sensor_plan({"perception": {"temporal": "strip", "n_frames": 99, "roi": "bogus"}})
        self.assertEqual(out["perception"]["temporal"], "strip")
        self.assertEqual(out["perception"]["n_frames"], 6)  # _MAX_STRIP
        self.assertNotIn("roi", out["perception"])  # out-of-vocab dropped

    def test_confirmation_closed_vocab(self):
        out = coerce_authored_sensor_plan({"confirmation": {"vlm": "sparse", "when": "haxx"}})
        self.assertEqual(out["sensor_plan"]["confirmation"], {"vlm": "sparse"})

    def test_non_dict_and_empty_return_empty(self):
        self.assertEqual(coerce_authored_sensor_plan("nope"), {})
        self.assertEqual(coerce_authored_sensor_plan({}), {})
        self.assertEqual(coerce_authored_sensor_plan({"primitives": ["nonsense"]}), {})

    def test_no_code_injection_only_typed_scalars_read(self):
        # a string where a number is expected is dropped (only _num-validated scalars are read into state)
        out = coerce_authored_sensor_plan({"cadence": {"max_fps": "__import__('os').system('rm -rf /')"}})
        self.assertEqual(out, {})

    def test_registry_primitives_are_self_describing(self):
        for name, prim in PRIMITIVES.items():
            self.assertEqual(prim.name, name)
            self.assertIn(prim.kind, {"sense", "perception"})
            self.assertTrue(prim.when_to_use)  # the cue the conductor reads to compose it


if __name__ == "__main__":
    unittest.main()
