"""Consequence's three calibration-free jobs and protected_floor (L1).

These are how a coarse consequence earns its keep today while the threshold job stays inert (R3):
attention allocation, hard-quadrant escalation, verification-required — none needs a calibrated
confidence number. protected_floor pins miss_cost at max.
"""

import unittest

from percept.adapters import Consequence, Goal, Verdict
from percept.agent import Percept
from percept.backends.fake import FakeVision
from percept.concern import (
    attention_weight,
    consequence_of,
    is_hard_quadrant,
    requires_verification,
)
from percept.gate import ASK, FIRE

YES = Verdict(True, 0.95, "yes")


def _agent():
    return Percept.create(vision=FakeVision(script={}), discover_plugins=False)


class TestConsequenceOf(unittest.TestCase):
    def test_defaults_mild(self):
        self.assertEqual(consequence_of(Goal(id="g", condition="c", say="s")), Consequence("mild", "mild"))

    def test_protected_floor_pins_miss_at_max(self):
        g = Goal(id="g", condition="c", say="s")
        g.state["protected_floor"] = True
        g.state["miss_cost"] = "trivial"  # refinement tries to relax it...
        c = consequence_of(g)
        self.assertEqual(c.miss_cost, "severe")  # ...the floor pins it at max regardless
        self.assertTrue(c.protected_floor)

    def test_reads_authored_costs(self):
        g = Goal(id="g", condition="c", say="s")
        g.state.update({"miss_cost": "severe", "false_fire_cost": "irreversible"})
        c = consequence_of(g)
        self.assertEqual((c.miss_cost, c.false_fire_cost), ("severe", "irreversible"))


class TestJobs(unittest.TestCase):
    def test_attention_weight_ranks_by_miss_cost(self):
        self.assertGreater(attention_weight(Consequence("severe", "mild")), attention_weight(Consequence("mild", "mild")))
        self.assertGreater(attention_weight(Consequence("mild", "mild")), attention_weight(Consequence("trivial", "mild")))

    def test_hard_quadrant_needs_both_high(self):
        self.assertTrue(is_hard_quadrant(Consequence("severe", "severe")))
        self.assertTrue(is_hard_quadrant(Consequence("severe", "irreversible")))
        self.assertFalse(is_hard_quadrant(Consequence("mild", "mild")))
        self.assertFalse(is_hard_quadrant(Consequence("severe", "mild")))  # only one side high

    def test_requires_verification_on_high_false_fire(self):
        self.assertTrue(requires_verification(Consequence("mild", "severe")))
        self.assertTrue(requires_verification(Consequence("mild", "irreversible")))
        self.assertFalse(requires_verification(Consequence("severe", "mild")))


class TestEscalationWiring(unittest.TestCase):
    def test_hard_quadrant_fire_escalates_to_ask(self):
        a = _agent()
        g = Goal(id="g", condition="the stove is unattended", say="turn off the stove")
        g.state.update({"miss_cost": "severe", "false_fire_cost": "irreversible"})  # hard quadrant
        ev = a._fire_concern(g, YES, ("g", "e"))
        self.assertEqual(ev.action, ASK)  # escalate to the human, do not auto-fire
        self.assertIn("confirm", ev.text)

    def test_default_concern_fires_normally(self):
        a = _agent()
        g = Goal(id="g", condition="c", say="the dog is on the couch")  # mild/mild
        ev = a._fire_concern(g, YES, ("g", "e"))
        self.assertEqual((ev.action, ev.text), (FIRE, "the dog is on the couch"))  # byte-for-byte


class TestAttentionOrder(unittest.TestCase):
    def test_orders_active_concerns_by_stakes(self):
        a = _agent()
        a.add_goal(Goal(id="low", condition="c", say="s"))  # mild
        g_hi = Goal(id="hi", condition="c", say="s")
        g_hi.state["miss_cost"] = "severe"
        a.add_goal(g_hi)
        a.add_goal(Goal(id="mid", condition="c", say="s"))  # mild
        self.assertEqual(a.attention_order()[0].id, "hi")  # severe-miss tracked first


if __name__ == "__main__":
    unittest.main()
