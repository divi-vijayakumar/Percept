"""L1: conductor reply parsing — robust to fenced / prose-wrapped JSON (regression for the
silent respond_with_vision fallback that hid goal authoring). Pure functions, no SDK/model."""

import unittest

from percept.backends.anthropic_text import _coerce_decision, _extract_json

WATCH = (
    '{"route":"respond_direct","look":"none","say":"Watching.",'
    '"goal_ops":[{"op":"create","goal":{"id":"g-coffee",'
    '"condition":"the user is drinking coffee","say":"careful"}}]}'
)


class TestExtractJson(unittest.TestCase):
    def test_bare_json_passes_through(self):
        self.assertEqual(_extract_json(WATCH), WATCH)

    def test_strips_json_code_fence(self):
        fenced = f"```json\n{WATCH}\n```"
        self.assertEqual(_extract_json(fenced), WATCH)

    def test_strips_plain_code_fence(self):
        self.assertEqual(_extract_json(f"```\n{WATCH}\n```"), WATCH)

    def test_extracts_object_from_surrounding_prose(self):
        wrapped = f"Sure, here is the decision:\n{WATCH}\nLet me know if that helps."
        self.assertEqual(_extract_json(wrapped), WATCH)


class TestCoerceDecision(unittest.TestCase):
    def test_watch_command_yields_goal_op(self):
        import json

        decision = _coerce_decision(json.loads(WATCH))
        self.assertEqual(decision.route, "respond_direct")
        self.assertEqual(len(decision.goal_ops), 1)
        op = decision.goal_ops[0]
        self.assertEqual(op.op, "create")
        self.assertEqual(op.goal.condition, "the user is drinking coffee")

    def test_one_shot_has_no_goal_ops(self):
        decision = _coerce_decision(
            {"route": "respond_with_vision", "look": "request_frame", "goal_ops": []}
        )
        self.assertEqual(decision.goal_ops, [])
        self.assertEqual(decision.look, "request_frame")

    def test_missing_fields_default_safely(self):
        decision = _coerce_decision({})
        self.assertEqual(decision.route, "respond_direct")
        self.assertEqual(decision.goal_ops, [])


if __name__ == "__main__":
    unittest.main()
