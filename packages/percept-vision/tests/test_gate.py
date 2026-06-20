"""L1: the three-state gate matrix — deterministic, no backends."""

import unittest

from percept.adapters import Verdict
from percept.gate import ASK, FIRE, SILENT, Gate


def yes(conf=0.9):
    return Verdict(satisfied=True, confidence=conf)


def no(conf=0.9):
    return Verdict(satisfied=False, confidence=conf)


class TestGate(unittest.TestCase):
    def setUp(self):
        self.gate = Gate(threshold=0.7, fresh_min=0.5)
        self.k = ("g1", "e1")

    def test_fires_on_rising_edge(self):
        self.assertEqual(self.gate.evaluate(self.k, yes()), FIRE)

    def test_silent_on_repeat_while_true(self):
        self.gate.evaluate(self.k, yes())
        self.assertEqual(self.gate.evaluate(self.k, yes()), SILENT)  # already fired, still true

    def test_confident_no_is_silent_and_rearms(self):
        self.assertEqual(self.gate.evaluate(self.k, yes()), FIRE)
        self.assertEqual(self.gate.evaluate(self.k, no()), SILENT)  # re-arms
        self.assertEqual(self.gate.evaluate(self.k, yes()), FIRE)  # fires again after re-arm

    def test_low_confidence_asks_and_never_fires(self):
        self.assertEqual(self.gate.evaluate(self.k, yes(conf=0.4)), ASK)
        # asking did not disarm — a later confident yes still fires
        self.assertEqual(self.gate.evaluate(self.k, yes()), FIRE)

    def test_stale_supporting_fact_asks(self):
        self.assertEqual(self.gate.evaluate(self.k, yes(), freshness=0.0), ASK)

    def test_refractory_single_fire_per_key(self):
        self.assertEqual(self.gate.evaluate(self.k, yes()), FIRE)
        self.assertEqual(self.gate.evaluate(self.k, yes()), SILENT)
        self.assertEqual(self.gate.evaluate(self.k, yes()), SILENT)

    def test_new_entity_key_gets_fresh_edge(self):
        self.gate.evaluate(("g1", "e1"), yes())  # e1 fires
        self.assertEqual(self.gate.evaluate(("g1", "e2"), yes()), FIRE)  # e2 is fresh

    def test_reset_goal_clears_edge(self):
        self.gate.evaluate(self.k, yes())  # fired, disarmed
        self.gate.reset_goal("g1")
        self.assertEqual(self.gate.evaluate(self.k, yes()), FIRE)  # edge reopened

    def test_ask_at_threshold_boundary(self):
        # exactly at threshold is confident (>=), just below asks
        self.assertEqual(self.gate.evaluate(("g", "a"), yes(conf=0.7)), FIRE)
        self.assertEqual(self.gate.evaluate(("g", "b"), yes(conf=0.69)), ASK)


if __name__ == "__main__":
    unittest.main()
