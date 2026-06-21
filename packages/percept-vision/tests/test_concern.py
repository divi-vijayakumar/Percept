"""A0 — Concern DTOs + Goal compat shim + the upstream adapters (L1, no backends, no API keys).

Guards the A0 invariants: the Goal->Concern shim reproduces today's defaults byte-for-byte
(consequence mild, tracking watch/on_false), ``threshold_for`` is pinned to the gate's 0.7 (R15),
the observation log is bounded (R14), and ``derive_belief`` yields observability-from-recency.
"""

import unittest

from percept.adapters import (
    Concern,
    Consequence,
    Evidence,
    Goal,
    Priors,
    Subject,
    TrackingBehavior,
    Verdict,
)
from percept.concern import OBS_CAP, derive_belief, observe, threshold_for
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


class TestBoundedLog(unittest.TestCase):
    def test_observe_is_bounded_and_keeps_most_recent(self):
        c = Concern.from_goal(Goal(id="g", condition="x", say="y"))
        for i in range(OBS_CAP + 100):
            observe(c, Evidence(t=float(i), source_id="cam", confidence=0.9, satisfied=True))
        self.assertEqual(len(c.observations), OBS_CAP)  # R14: never unbounded
        self.assertEqual(c.observations[-1].t, float(OBS_CAP + 99))  # most recent kept
        self.assertEqual(c.observations[0].t, float(100))  # oldest trimmed, order preserved


class TestDeriveBelief(unittest.TestCase):
    def test_no_observations_is_zero_coverage(self):
        c = Concern.from_goal(Goal(id="g", condition="x", say="y"))
        v, freshness = derive_belief(c, now=10.0)
        self.assertFalse(v.satisfied)
        self.assertEqual(v.confidence, 0.0)
        self.assertEqual(freshness, 0.0)  # absence of evidence != evidence of absence

    def test_fresh_observation_full_freshness(self):
        c = Concern.from_goal(Goal(id="g", condition="x", say="y"))
        observe(c, Evidence(t=10.0, source_id="cam", confidence=0.92, satisfied=True))
        v, freshness = derive_belief(c, now=10.0)
        self.assertTrue(v.satisfied)
        self.assertEqual(v.confidence, 0.92)
        self.assertAlmostEqual(freshness, 1.0)

    def test_stale_observation_decays_freshness(self):
        # Observability: a confident belief whose last support is old reads as LOW coverage.
        c = Concern.from_goal(Goal(id="g", condition="x", say="y"))
        observe(c, Evidence(t=0.0, source_id="cam", confidence=0.95, satisfied=True))
        v, freshness = derive_belief(c, now=30.0, half_life_s=30.0)
        self.assertTrue(v.satisfied)
        self.assertEqual(v.confidence, 0.95)
        self.assertAlmostEqual(freshness, 0.5)  # one half-life elapsed

    def test_coverage_scales_freshness(self):
        c = Concern.from_goal(Goal(id="g", condition="x", say="y"))
        observe(c, Evidence(t=5.0, source_id="cam", confidence=0.9, satisfied=True, coverage=0.5))
        _v, freshness = derive_belief(c, now=5.0)
        self.assertAlmostEqual(freshness, 0.5)


class TestFrozenMembraneUnchanged(unittest.TestCase):
    def test_goal_shape_unchanged(self):
        # The existing Goal DTO is untouched by A0 (added beside, not edited).
        g = Goal(id="g", condition="c", say="s")
        self.assertEqual(g.source, "camera")
        self.assertEqual(g.state, {"type": "watch", "count": 0})
        self.assertEqual(g.expiry, "until_cancelled")


if __name__ == "__main__":
    unittest.main()
