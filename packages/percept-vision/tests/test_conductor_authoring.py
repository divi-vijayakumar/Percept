"""Conductor authoring (L1) — the LLM can author behavior from NL, but every authored value is GUARDED.

coerce_authored_behavior is the firewall on the untrusted conductor "behavior" block (sibling of the
fire_when AST whitelist): closed vocab, malformed entries dropped to a safe watch, numerics clamped.
_coerce_decision wires the validated block into goal.state. Both are pure (no SDK / no keys).
"""

import unittest

from percept.backends.anthropic_text import _coerce_decision
from percept.behavior import coerce_authored_behavior


class TestCoerceAuthoredBehavior(unittest.TestCase):
    def test_never_and_each_event(self):
        self.assertEqual(coerce_authored_behavior({"rearm": "never"}), {"rearm": "never"})
        self.assertEqual(coerce_authored_behavior({"rearm": "each_event"}), {"rearm": "each_event"})

    def test_interval_valid_and_clamped(self):
        self.assertEqual(
            coerce_authored_behavior({"rearm": "interval", "cadence_s": 1200}),
            {"rearm": "interval", "cadence_s": 1200.0},
        )
        # a busy-loop cadence is clamped to the floor, never trusted
        self.assertEqual(
            coerce_authored_behavior({"rearm": "interval", "cadence_s": 0.01}),
            {"rearm": "interval", "cadence_s": 1.0},
        )

    def test_interval_without_positive_cadence_is_dropped(self):
        self.assertEqual(coerce_authored_behavior({"rearm": "interval"}), {})
        self.assertEqual(coerce_authored_behavior({"rearm": "interval", "cadence_s": 0}), {})

    def test_deadline_valid_and_negative_dropped(self):
        self.assertEqual(
            coerce_authored_behavior({"rearm": "deadline", "deadline_s": 300}),
            {"rearm": "deadline", "deadline_s": 300.0},
        )
        self.assertEqual(coerce_authored_behavior({"rearm": "deadline", "deadline_s": -5}), {})

    def test_transition_needs_from_and_to(self):
        self.assertEqual(
            coerce_authored_behavior({"type": "transition", "from": "asleep", "to": "awake"}),
            {"type": "transition", "from": "asleep", "to": "awake"},
        )
        self.assertEqual(coerce_authored_behavior({"type": "transition", "from": "asleep"}), {})

    def test_consequence_closed_vocab(self):
        self.assertEqual(
            coerce_authored_behavior(
                {"consequence": {"miss_cost": "severe", "false_fire_cost": "irreversible", "protected_floor": True}}
            ),
            {"miss_cost": "severe", "false_fire_cost": "irreversible", "protected_floor": True},
        )
        self.assertEqual(coerce_authored_behavior({"consequence": {"miss_cost": "catastrophic"}}), {})

    def test_fire_when_validated_through_the_ast_firewall(self):
        # a valid nth-distinct predicate is kept; window clamped/typed
        self.assertEqual(
            coerce_authored_behavior({"accumulate": {"fire_when": "n == 3"}}), {"fire_when": "n == 3"}
        )
        self.assertEqual(
            coerce_authored_behavior({"accumulate": {"fire_when": "n >= 2", "window_s": 600}}),
            {"fire_when": "n >= 2", "window_s": 600.0},
        )
        # an UNSAFE expression never reaches goal.state (same AST whitelist as firing time)
        self.assertEqual(coerce_authored_behavior({"accumulate": {"fire_when": "__import__('os')"}}), {})
        self.assertEqual(coerce_authored_behavior({"accumulate": {"fire_when": "open('x')"}}), {})

    def test_unknown_and_malformed_are_dropped(self):
        self.assertEqual(coerce_authored_behavior({"rearm": "explode"}), {})  # out-of-vocab rearm
        self.assertEqual(coerce_authored_behavior({"type": "wormhole"}), {})  # out-of-vocab type
        self.assertEqual(coerce_authored_behavior({"rearm": "deadline", "deadline_s": "__import__"}), {})
        self.assertEqual(coerce_authored_behavior(None), {})
        self.assertEqual(coerce_authored_behavior("interval"), {})
        self.assertEqual(coerce_authored_behavior({"rearm": "interval", "cadence_s": True}), {})  # bool != num


class TestCoerceDecisionWiring(unittest.TestCase):
    def _goal(self, behavior):
        payload = {
            "route": "respond_direct",
            "goal_ops": [{"op": "create", "goal": {"id": "g", "condition": "c", "say": "s", "behavior": behavior}}],
        }
        return _coerce_decision(payload).goal_ops[0].goal

    def test_authored_interval_reaches_goal_state(self):
        g = self._goal({"rearm": "interval", "cadence_s": 1200})
        self.assertEqual(g.state["rearm"], "interval")
        self.assertEqual(g.state["cadence_s"], 1200.0)

    def test_authored_transition_reaches_goal_state(self):
        g = self._goal({"type": "transition", "from": "asleep", "to": "awake"})
        self.assertEqual((g.state["type"], g.state["from"], g.state["to"]), ("transition", "asleep", "awake"))

    def test_no_behavior_leaves_default_state(self):
        g = self._goal(None)
        self.assertEqual(g.state, {"type": "watch", "count": 0})

    def test_malformed_behavior_leaves_default_state(self):
        g = self._goal({"rearm": "explode", "deadline_s": -1})
        self.assertEqual(g.state, {"type": "watch", "count": 0})  # nothing unsafe leaks in


if __name__ == "__main__":
    unittest.main()
