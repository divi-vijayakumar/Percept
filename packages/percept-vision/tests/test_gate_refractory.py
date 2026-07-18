"""L3 (MVP plan E4, precision track): time-refractory on the three-state gate.

Precision forensic finding (eval/viability/JUDGE_LOSS_DEEPDIVE.md, local): ~12% of false fires on the
charades-e2e track were RE-FIRES — the gate re-arming and firing again within one true episode because
the judge briefly flickered confident-false mid-episode. ``refractory_s`` suppresses a fire landing
within N seconds of the same key's last real fire, even across such a blip. Off by default (0/None) is
byte-for-byte the historical gate — asserted here alongside the on-behavior.
"""
from __future__ import annotations

import unittest

from percept.adapters import Verdict
from percept.gate import FIRE, SILENT, Gate

YES = Verdict(satisfied=True, confidence=0.95, evidence="present")
NO = Verdict(satisfied=False, confidence=0.95, evidence="absent")


class RefractorySuppressesReFireTests(unittest.TestCase):
    def test_reflicker_within_refractory_window_is_suppressed(self):
        """The exact re-fire pattern from the forensic: YES fires, a spurious NO blip re-arms the edge,
        then YES again within the refractory window — the second fire must be suppressed."""
        g = Gate(refractory_s=5.0)
        self.assertEqual(g.evaluate("k", YES, now=0.0), FIRE)     # real fire, t=0
        self.assertEqual(g.evaluate("k", NO, now=1.0), SILENT)    # spurious blip -> re-arms the edge
        self.assertEqual(g.evaluate("k", YES, now=2.0), SILENT)   # re-fire at t=2 (< 5s since t=0) -> suppressed

    def test_fire_after_refractory_elapses_is_a_genuine_new_episode(self):
        """Once refractory_s has elapsed since the last real fire, a fresh false->true edge fires again —
        this is a genuine second episode, not a re-fire."""
        g = Gate(refractory_s=5.0)
        self.assertEqual(g.evaluate("k", YES, now=0.0), FIRE)     # real fire, t=0
        self.assertEqual(g.evaluate("k", NO, now=1.0), SILENT)    # goes away
        self.assertEqual(g.evaluate("k", YES, now=6.0), FIRE)     # t=6, 6s since last fire >= 5s -> fires

    def test_refractory_off_by_default_is_byte_for_byte(self):
        """No refractory configured (the default) behaves exactly like the historical gate: every
        false->true edge fires, refractory or no."""
        g = Gate()
        self.assertEqual(g.evaluate("k", YES, now=0.0), FIRE)
        self.assertEqual(g.evaluate("k", NO, now=0.1), SILENT)
        self.assertEqual(g.evaluate("k", YES, now=0.2), FIRE)     # immediate re-fire, no refractory set

    def test_zero_or_negative_refractory_is_off(self):
        g = Gate(refractory_s=0.0)
        self.assertEqual(g.evaluate("k", YES, now=0.0), FIRE)
        self.assertEqual(g.evaluate("k", NO, now=0.1), SILENT)
        self.assertEqual(g.evaluate("k", YES, now=0.2), FIRE)

    def test_per_call_refractory_overrides_gate_default(self):
        """A per-call ``refractory_s`` (e.g. a per-goal WatchProfile override) takes precedence over the
        gate-wide default."""
        g = Gate(refractory_s=100.0)  # gate-wide default would suppress
        self.assertEqual(g.evaluate("k", YES, now=0.0), FIRE)
        self.assertEqual(g.evaluate("k", NO, now=1.0), SILENT)
        # per-call override of 1s -> 5s since last fire clears it, fires despite the 100s gate default
        self.assertEqual(g.evaluate("k", YES, now=6.0, refractory_s=1.0), FIRE)

    def test_refractory_requires_now_to_take_effect(self):
        """Without a stream clock (``now=None``), refractory cannot be measured — falls back to
        byte-for-byte immediate-fire (matches ``hold_s``'s ``now``-required contract)."""
        g = Gate(refractory_s=5.0)
        self.assertEqual(g.evaluate("k", YES), FIRE)       # now=None -> no refractory tracking possible
        self.assertEqual(g.evaluate("k", NO), SILENT)
        self.assertEqual(g.evaluate("k", YES), FIRE)       # still immediate: no `now` supplied anywhere

    def test_suppressed_fire_still_consumes_the_edge(self):
        """A refractory-suppressed 'fire' behaves like a real fire for edge state: the key goes disarmed,
        so it does not fire again on the very next tick even inside the window."""
        g = Gate(refractory_s=5.0)
        self.assertEqual(g.evaluate("k", YES, now=0.0), FIRE)
        self.assertEqual(g.evaluate("k", NO, now=1.0), SILENT)
        self.assertEqual(g.evaluate("k", YES, now=2.0), SILENT)   # suppressed re-fire
        self.assertEqual(g.evaluate("k", YES, now=2.5), SILENT)   # still armed=False -> silent, not a 2nd suppression edge
        self.assertEqual(g.evaluate("k", NO, now=3.0), SILENT)    # re-arms
        self.assertEqual(g.evaluate("k", YES, now=6.5), FIRE)     # >= 5s since the ORIGINAL fire at t=0 -> fires

    def test_hold_s_and_refractory_compose(self):
        """A dwell-gated fire is still subject to refractory once it actually fires."""
        g = Gate(refractory_s=5.0)
        self.assertEqual(g.evaluate("k", YES, hold_s=2, now=0.0), SILENT)  # dwell timer starts
        self.assertEqual(g.evaluate("k", YES, hold_s=2, now=2.0), FIRE)    # dwell satisfied -> fires, t=2
        self.assertEqual(g.evaluate("k", NO, now=2.5), SILENT)             # blip re-arms
        self.assertEqual(g.evaluate("k", YES, hold_s=0, now=3.0), SILENT)  # 1s since fire < 5s -> suppressed

    def test_snapshot_restore_round_trips_refractory_memory(self):
        """A restarted 24/7 watch must not forget an in-progress refractory window (or it would re-fire
        early right after restart). Uses a ``(goal_id, entity)`` tuple key — the snapshot format's
        actual key shape (see :meth:`Gate.snapshot`'s docstring)."""
        key = ("goal-a", "e1")
        g = Gate(refractory_s=5.0)
        self.assertEqual(g.evaluate(key, YES, now=0.0), FIRE)
        self.assertEqual(g.evaluate(key, NO, now=1.0), SILENT)  # re-arm
        snap = g.snapshot()

        g2 = Gate(refractory_s=5.0)
        g2.restore(snap)
        self.assertEqual(g2.evaluate(key, YES, now=2.0), SILENT)  # still within 5s of the restored fire
        self.assertEqual(g2.evaluate(key, NO, now=3.0), SILENT)
        self.assertEqual(g2.evaluate(key, YES, now=6.0), FIRE)    # 6s since original fire -> clears

    def test_rearm_clears_refractory_memory(self):
        """An explicit rearm (e.g. a correction, or a time-bucket rollover) is a fresh episode boundary —
        it must not leave a stale refractory stamp behind."""
        g = Gate(refractory_s=100.0)
        self.assertEqual(g.evaluate("k", YES, now=0.0), FIRE)
        g.rearm("k")
        self.assertEqual(g.evaluate("k", YES, now=1.0), FIRE)  # would be suppressed but rearm cleared it

    def test_reset_goal_clears_refractory_memory(self):
        g = Gate(refractory_s=100.0)
        self.assertEqual(g.evaluate(("goal-a", "e1"), YES, now=0.0), FIRE)
        g.reset_goal("goal-a")
        self.assertEqual(g.evaluate(("goal-a", "e1"), YES, now=1.0), FIRE)  # reset -> fresh, no suppression


if __name__ == "__main__":
    unittest.main()
