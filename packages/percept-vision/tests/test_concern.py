"""Concern DTOs, Goal compatibility shim, and upstream adapters (L1, no backends or API keys).

Guards the surviving A0 invariants: the Goal->Concern shim reproduces today's defaults byte-for-byte
(consequence mild, tracking watch/on_false) and ``threshold_for`` is pinned to the gate's 0.7 (R15).
The ``derive_belief`` / ``observe`` belief-log spike was removed (never wired into the runtime); see
percept/concern.py header.
"""

import unittest

from percept.adapters import (
    Concern,
    Consequence,
    Goal,
    Priors,
    Subject,
    TrackingBehavior,
)
from percept.concern import threshold_for
from percept.gate import Gate


class TestCompatShim(unittest.TestCase):
    def test_from_goal_reproduces_todays_defaults(self):
        g = Goal(id="g1", condition="the dog is on the couch", say="dog on couch")
        c = Concern.from_goal(g)
        self.assertEqual(c.id, "g1")
        self.assertEqual(c.condition, "the dog is on the couch")
        self.assertEqual(c.say, "dog on couch")
        self.assertEqual(c.source, "camera")
        # default consequence = mild/mild, not protected
        self.assertEqual(c.consequence, Consequence(miss_cost="mild", false_fire_cost="mild"))
        self.assertFalse(c.consequence.protected_floor)
        # default tracking = watch / on_false (today's hardcoded re-arm)
        self.assertEqual(c.tracking, TrackingBehavior(pattern="watch", rearm="on_false"))
        self.assertEqual(c.observations, [])
        self.assertEqual(c.identity, Subject(kind="entity", ref="the dog is on the couch"))
        self.assertEqual(c.expectations, Priors())


class TestThresholdPin(unittest.TestCase):
    def test_default_threshold_equals_gate_threshold(self):
        # R15: an existing goal's threshold must be byte-for-byte the gate default (0.7).
        self.assertEqual(threshold_for(Consequence()), Gate().threshold)
        self.assertEqual(threshold_for(Consequence()), 0.7)

    def test_threshold_is_inert_across_consequences(self):
        # R3: at today's calibration the threshold job is a no-op — every consequence maps to the
        # gate default. (This test PINS that honesty; it will change only when calibration lands.)
        for cons in (
            Consequence(miss_cost="severe", false_fire_cost="trivial"),
            Consequence(miss_cost="trivial", false_fire_cost="irreversible"),
            Consequence(miss_cost="severe", false_fire_cost="severe", protected_floor=True),
        ):
            self.assertEqual(threshold_for(cons), 0.7)


class TestFrozenMembraneUnchanged(unittest.TestCase):
    def test_goal_shape_unchanged(self):
        # The existing Goal DTO is untouched by A0 (added beside, not edited).
        g = Goal(id="g", condition="c", say="s")
        self.assertEqual(g.source, "camera")
        self.assertEqual(g.state, {"type": "watch", "count": 0})
        self.assertEqual(g.expiry, "until_cancelled")


if __name__ == "__main__":
    unittest.main()
