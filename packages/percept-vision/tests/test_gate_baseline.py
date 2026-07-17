"""Gate fix: `require_false_first` — the v0.1 "already present at start -> SILENT" rule.

Validated on a PERSISTENT-state scenario (a condition that is true for a sustained span), which is
Percept's target case. The historical default (fire on first true) is asserted unchanged.
"""
from __future__ import annotations

import unittest

from percept.gate import Gate, FIRE, SILENT
from percept.adapters import Verdict

YES = Verdict(satisfied=True, confidence=0.95, evidence="present")
NO = Verdict(satisfied=False, confidence=0.95, evidence="absent")


class RequireFalseFirstTests(unittest.TestCase):
    def test_already_present_at_start_is_silent(self):
        """A persistent condition ALREADY true on the first frame must not fire (cold-start suppression)."""
        g = Gate(threshold=0.5)
        # first observation is YES -> already present -> SILENT (not a rising edge we watched)
        self.assertEqual(g.evaluate("k", YES, require_false_first=True), SILENT)
        # stays true for a sustained span -> still SILENT (no cold-start fire, no re-fire)
        for _ in range(5):
            self.assertEqual(g.evaluate("k", YES, require_false_first=True), SILENT)

    def test_genuine_false_to_true_fires_exactly_once(self):
        """A real false->true onset fires once; the sustained true span after it does not re-fire."""
        g = Gate(threshold=0.5)
        self.assertEqual(g.evaluate("k", NO, require_false_first=True), SILENT)   # baseline established
        self.assertEqual(g.evaluate("k", YES, require_false_first=True), FIRE)    # rising edge
        for _ in range(5):  # persists true -> exactly one fire
            self.assertEqual(g.evaluate("k", YES, require_false_first=True), SILENT)

    def test_rearm_after_absence_then_second_onset_fires_again(self):
        """After the condition goes away and returns, a second onset is a fresh rising edge -> fires."""
        g = Gate(threshold=0.5)
        g.evaluate("k", YES, require_false_first=True)   # already-present -> SILENT
        g.evaluate("k", NO, require_false_first=True)    # gone -> baseline/armed
        self.assertEqual(g.evaluate("k", YES, require_false_first=True), FIRE)   # 1st real onset
        g.evaluate("k", NO, require_false_first=True)    # gone again -> re-armed
        self.assertEqual(g.evaluate("k", YES, require_false_first=True), FIRE)   # 2nd onset fires again

    def test_default_off_is_byte_for_byte_historical(self):
        """Without the flag, first-true still FIREs (existing rising-edge behavior preserved)."""
        g = Gate(threshold=0.5)
        self.assertEqual(g.evaluate("k", YES), FIRE)          # historical: fire on first true
        self.assertEqual(g.evaluate("k", YES), SILENT)        # already fired


if __name__ == "__main__":
    unittest.main()
