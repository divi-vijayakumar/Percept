"""L1: the goal registry — apply ops, lifecycle, active filtering."""

import unittest

from percept.adapters import Goal
from percept.goals import GoalOp, GoalRegistry


def g(gid="g1", condition="user drinks coffee", say="careful"):
    return Goal(id=gid, condition=condition, say=say)


class TestGoalRegistry(unittest.TestCase):
    def setUp(self):
        self.reg = GoalRegistry()

    def test_create_via_apply(self):
        self.reg.apply([GoalOp(op="create", goal=g())])
        self.assertEqual(self.reg.status("g1"), "active")
        self.assertEqual(len(self.reg.active()), 1)

    def test_complete_removes_from_active(self):
        self.reg.create(g())
        self.reg.apply([GoalOp(op="complete", goal_id="g1")])
        self.assertEqual(self.reg.status("g1"), "complete")
        self.assertEqual(self.reg.active(), [])

    def test_cancel_removes_from_active(self):
        self.reg.create(g())
        self.reg.apply([GoalOp(op="cancel", goal_id="g1")])
        self.assertEqual(self.reg.status("g1"), "cancelled")

    def test_update_mutates_fields(self):
        self.reg.create(g())
        self.reg.apply([GoalOp(op="update", goal_id="g1", fields={"say": "new line"})])
        self.assertEqual(self.reg.get("g1").say, "new line")

    def test_direct_create_rejects_duplicate_id_even_after_cancel(self):
        self.reg.create(g())
        self.reg.apply([GoalOp(op="cancel", goal_id="g1")])

        with self.assertRaisesRegex(ValueError, "lifetime-unique"):
            self.reg.create(g(condition="replacement"))

        self.assertEqual(self.reg.get("g1").condition, "user drinks coffee")
        self.assertEqual(self.reg.status("g1"), "cancelled")

    def test_update_rejects_executable_fields_and_non_string_say(self):
        self.reg.create(g())
        for fields in (
            {"condition": "changed"},
            {"source": "other"},
            {"state": {}},
            {"id": "other"},
            {"origin": "restored"},
        ):
            with self.subTest(fields=fields):
                with self.assertRaisesRegex(ValueError, "only the non-executable 'say'"):
                    self.reg.apply([GoalOp(op="update", goal_id="g1", fields=fields)])
        with self.assertRaisesRegex(TypeError, "must be a string"):
            self.reg.apply([GoalOp(op="update", goal_id="g1", fields={"say": 3})])

        self.assertEqual(self.reg.get("g1").condition, "user drinks coffee")
        self.assertEqual(self.reg.get("g1").say, "careful")

    def test_create_requires_goal(self):
        with self.assertRaises(ValueError):
            self.reg.apply([GoalOp(op="create")])

    def test_op_on_missing_goal_raises(self):
        with self.assertRaises(KeyError):
            self.reg.apply([GoalOp(op="complete", goal_id="nope")])

    def test_batch_apply(self):
        self.reg.apply(
            [
                GoalOp(op="create", goal=g("g1")),
                GoalOp(op="create", goal=g("g2")),
                GoalOp(op="cancel", goal_id="g1"),
            ]
        )
        self.assertEqual({x.id for x in self.reg.active()}, {"g2"})

    def test_batch_is_fully_prevalidated_before_mutation(self):
        self.reg.create(g())

        with self.assertRaisesRegex(KeyError, "missing"):
            self.reg.apply([
                GoalOp(op="update", goal_id="g1", fields={"say": "mutated"}),
                GoalOp(op="complete", goal_id="missing"),
            ])

        self.assertEqual(self.reg.get("g1").say, "careful")
        self.assertEqual(self.reg.status("g1"), "active")


if __name__ == "__main__":
    unittest.main()
