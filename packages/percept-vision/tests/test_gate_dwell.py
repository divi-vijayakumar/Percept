"""R-1: sustained-duration (hold_s) dwell on the three-state gate.

With hold_s set, an armed key fires only after the condition has held confidently-true for hold_s
seconds (measured on the stream time `now`); a confident-false blip in between resets the dwell. With
hold_s=None (the default) the gate is byte-for-byte the rising-edge gate.
"""
from __future__ import annotations

import unittest

from percept.adapters import Goal, Verdict
from percept.behavior import _coerce_hold_s, program_of
from percept.gate import ASK, FIRE, SILENT, Gate

YES = Verdict(satisfied=True, confidence=0.99)
NO = Verdict(satisfied=False, confidence=0.99)
UNSURE = Verdict(satisfied=True, confidence=0.1)  # below threshold -> unknown -> ASK


class GateDwellTests(unittest.TestCase):
    def test_no_hold_s_is_the_rising_edge_gate(self):
        g = Gate()
        self.assertEqual(g.evaluate("k", YES), FIRE)     # fires immediately (byte-for-byte)
        self.assertEqual(g.evaluate("k", YES), SILENT)   # already fired

    def test_dwell_fires_only_after_hold_s_of_sustained_truth(self):
        g = Gate()
        self.assertEqual(g.evaluate("k", YES, hold_s=10, now=0.0), SILENT)   # timer starts
        self.assertEqual(g.evaluate("k", YES, hold_s=10, now=5.0), SILENT)   # 5s < 10s
        self.assertEqual(g.evaluate("k", YES, hold_s=10, now=9.9), SILENT)
        self.assertEqual(g.evaluate("k", YES, hold_s=10, now=10.0), FIRE)    # 10s reached -> fire
        self.assertEqual(g.evaluate("k", YES, hold_s=10, now=11.0), SILENT)  # already fired

    def test_a_transient_blip_resets_the_dwell(self):
        g = Gate()
        self.assertEqual(g.evaluate("k", YES, hold_s=10, now=0.0), SILENT)   # timer starts at 0
        self.assertEqual(g.evaluate("k", YES, hold_s=10, now=8.0), SILENT)   # 8s in
        self.assertEqual(g.evaluate("k", NO, hold_s=10, now=8.5), SILENT)    # blip -> reset + re-arm
        self.assertEqual(g.evaluate("k", YES, hold_s=10, now=9.0), SILENT)   # timer restarts at 9
        self.assertEqual(g.evaluate("k", YES, hold_s=10, now=18.0), SILENT)  # 9s since restart < 10
        self.assertEqual(g.evaluate("k", YES, hold_s=10, now=19.0), FIRE)    # 10s since restart -> fire

    def test_ask_does_not_break_the_dwell(self):
        g = Gate()
        self.assertEqual(g.evaluate("k", YES, hold_s=10, now=0.0), SILENT)   # timer starts
        self.assertEqual(g.evaluate("k", UNSURE, hold_s=10, now=3.0), ASK)   # uncertain -> ASK, timer kept
        self.assertEqual(g.evaluate("k", YES, hold_s=10, now=10.0), FIRE)    # dwell measured from t0=0

    def test_zero_or_negative_hold_s_is_immediate(self):
        g = Gate()
        self.assertEqual(g.evaluate("k", YES, hold_s=0, now=0.0), FIRE)      # 0 -> no dwell


class ProgramHoldSTests(unittest.TestCase):
    def _goal(self, **state):
        g = Goal(id="ppe", condition="the worker has no hard hat", say="put your hard hat on")
        g.state.update(state)
        return g

    def test_program_of_reads_hold_s_from_goal_state(self):
        self.assertEqual(program_of(self._goal(hold_s=10)).hold_s, 10.0)

    def test_absent_hold_s_is_none_byte_for_byte(self):
        self.assertIsNone(program_of(self._goal()).hold_s)

    def test_hold_s_coercion_firewall(self):
        self.assertEqual(_coerce_hold_s(10), 10.0)
        self.assertEqual(_coerce_hold_s(2.5), 2.5)
        for bad in ("10s", 0, -5, True, None, [10]):
            self.assertIsNone(_coerce_hold_s(bad))  # malformed -> no dwell (never suppress firing)


if __name__ == "__main__":
    unittest.main()
