"""R-1: sustained-duration (hold_s) dwell on the three-state gate.

With hold_s set, an armed key fires only after the condition has held confidently-true for hold_s
seconds (measured on the stream time `now`); a confident-false blip in between resets the dwell. With
hold_s=None (the default) the gate is byte-for-byte the rising-edge gate.
"""
from __future__ import annotations

import json
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

    def test_ask_breaks_positive_dwell_support(self):
        g = Gate()
        self.assertEqual(g.evaluate("k", YES, hold_s=10, now=0.0), SILENT)   # timer starts
        self.assertEqual(g.evaluate("k", UNSURE, hold_s=10, now=3.0), ASK)   # unknown breaks support
        self.assertEqual(g.evaluate("k", YES, hold_s=10, now=10.0), SILENT)  # timer restarts at 10
        self.assertEqual(g.evaluate("k", YES, hold_s=10, now=20.0), FIRE)

    def test_ask_without_hold_preserves_rising_edge_compatibility(self):
        g = Gate()
        self.assertEqual(g.evaluate("k", UNSURE), ASK)
        self.assertEqual(g.evaluate("k", YES), FIRE)

    def test_zero_or_negative_hold_s_is_immediate(self):
        g = Gate()
        self.assertEqual(g.evaluate("k", YES, hold_s=0, now=0.0), FIRE)      # 0 -> no dwell


class RestartMidDwellTests(unittest.TestCase):
    """Restart while a dwell is in flight — `restore()` drops `_dwell_since` ON PURPOSE.

    `snapshot()` carries the in-progress dwell "for inspection but is NOT restored" (gate.py:723-725);
    `restore()` hard-resets both dwell fields (gate.py:767-768) because the pre-restart start stamp sits
    on a timeline the new process has no anchor to rebase from. The consequence — the one a user
    actually feels — is that a restart mid-dwell costs the accumulated time: the dwell restarts from
    zero, and a full fresh `hold_s` must elapse before anything fires. These tests pin that in both
    directions, including the stale-wall-clock case the failure mode names.
    """

    HOLD_S = 5.0
    PENDING = ("door_open", "front")   # dwell in flight at snapshot time
    FIRED = ("window_open", "rear")    # no dwell; already fired, no_evict, has a last_fire stamp

    def _gate_mid_dwell(self) -> Gate:
        """A gate with `PENDING` two ticks into a 5 s dwell and `FIRED` already fired."""
        g = Gate()
        # A no-dwell key that really fires, so armed/last_fire/no_evict have something to round-trip.
        out = g.decide(self.FIRED, YES, now=0.0, no_evict=True)
        self.assertEqual((out.action, out.reason), (FIRE, "rising_edge"))
        # ...and the dwell key, driven into flight but not yet satisfied.
        for now in (0.0, 1.0):
            out = g.decide(self.PENDING, YES, hold_s=self.HOLD_S, now=now)
            self.assertEqual((out.action, out.reason), (SILENT, "dwell_pending"))
        return g

    def _restored(self) -> tuple[Gate, Gate, dict]:
        """(pre-restart gate, a fresh gate restored from its snapshot through JSON, the snapshot)."""
        before = self._gate_mid_dwell()
        # Through JSON: the snapshot docstring promises it survives the persistence backend, and JSON
        # is what turns the tuple keys into the lists `restore` re-tuples.
        snap = json.loads(json.dumps(before.snapshot()))
        after = Gate()
        after.restore(snap)
        return before, after, snap

    # -- what crosses the restart boundary -------------------------------------------------

    def test_snapshot_carries_the_in_flight_dwell_for_inspection(self):
        snap = self._gate_mid_dwell().snapshot()
        self.assertEqual(snap["dwell_since"], [[list(self.PENDING), 0.0]])  # the START tick, not 1.0
        # The support-continuity map is not serialized at all — there is no key to inspect it under.
        self.assertNotIn("dwell_last_supported", snap)

    def test_restore_drops_both_dwell_maps_on_purpose(self):
        _before, after, snap = self._restored()
        self.assertEqual(snap["dwell_since"], [[list(self.PENDING), 0.0]])  # it WAS in the payload...
        self.assertEqual(after._dwell_since, {})                            # ...and is deliberately dropped
        self.assertEqual(after._dwell_last_supported, {})

    def test_restore_round_trips_armed_touched_no_evict_and_last_fire(self):
        before, after, _snap = self._restored()
        self.assertEqual(after._armed, {self.FIRED: False})
        self.assertEqual(after._armed, before._armed)
        self.assertEqual(after._touched, {self.FIRED: 0.0, self.PENDING: 1.0})
        self.assertEqual(after._touched, before._touched)
        self.assertEqual(after._no_evict, {self.FIRED})
        self.assertEqual(after._no_evict, before._no_evict)
        self.assertEqual(after._last_fire, {self.FIRED: 0.0})
        self.assertEqual(after._last_fire, before._last_fire)

    def test_a_purely_pending_key_has_no_armed_entry_and_restores_armed(self):
        # The dwell branches (gate.py:607-632) never write `armed`, so a key that has only ever been
        # dwell_pending is absent from the armed map — and absent from the snapshot's armed list.
        before, after, snap = self._restored()
        self.assertNotIn(self.PENDING, before._armed)
        self.assertNotIn(list(self.PENDING), [k for k, _v in snap["armed"]])
        self.assertNotIn(self.PENDING, after._armed)
        # ...so it defaults to armed, not to already_fired. Pinned behaviorally: drop `hold_s` and the
        # restored key takes the rising edge rather than reading as this episode's spent fire.
        out = after.decide(self.PENDING, YES, now=2.0)
        self.assertEqual((out.action, out.reason), (FIRE, "rising_edge"))

    # -- the consequence, pinned in both directions ----------------------------------------

    def test_without_a_restart_the_dwell_fires_at_hold_s(self):
        # Control for the two tests below: on this timeline, now=5.0 IS the satisfying tick.
        g = self._gate_mid_dwell()
        out = g.decide(self.PENDING, YES, hold_s=self.HOLD_S, now=5.0)
        self.assertEqual((out.action, out.reason), (FIRE, "dwell_satisfied"))

    def test_after_restore_the_next_true_tick_restarts_the_timer_from_zero(self):
        _before, after, _snap = self._restored()
        out = after.decide(self.PENDING, YES, hold_s=self.HOLD_S, now=2.0)
        # Never an immediate FIRE crediting the 1 s served before the restart.
        self.assertEqual((out.action, out.reason), (SILENT, "dwell_pending"))
        self.assertEqual(after._dwell_since[self.PENDING], 2.0)  # re-anchored to the post-restart tick

    def test_after_restore_a_full_fresh_hold_s_must_elapse(self):
        _before, after, _snap = self._restored()
        after.decide(self.PENDING, YES, hold_s=self.HOLD_S, now=2.0)  # timer restarts here
        # 5.0 satisfied the PRE-restart dwell (see the control above); post-restart it is only 3 s in.
        self.assertEqual(after.evaluate(self.PENDING, YES, hold_s=self.HOLD_S, now=5.0), SILENT)
        self.assertEqual(after.evaluate(self.PENDING, YES, hold_s=self.HOLD_S, now=6.9), SILENT)
        out = after.decide(self.PENDING, YES, hold_s=self.HOLD_S, now=7.0)  # 2.0 + 5.0
        self.assertEqual((out.action, out.reason), (FIRE, "dwell_satisfied"))

    def test_a_stale_wall_clock_after_restore_starts_a_new_dwell_not_a_fire(self):
        # The named failure mode: restored, then handed a `now` far in the future. A resurrected
        # `_dwell_since=0.0` would read as 10_000 s of proven continuity and fire on the first tick.
        _before, after, _snap = self._restored()
        out = after.decide(self.PENDING, YES, hold_s=self.HOLD_S, now=10_000.0)
        self.assertEqual((out.action, out.reason), (SILENT, "dwell_pending"))
        self.assertEqual(after.evaluate(self.PENDING, YES, hold_s=self.HOLD_S, now=10_004.9), SILENT)
        out = after.decide(self.PENDING, YES, hold_s=self.HOLD_S, now=10_005.0)
        self.assertEqual((out.action, out.reason), (FIRE, "dwell_satisfied"))

    def test_a_restored_dwell_never_fires_early_and_never_stops_firing(self):
        # "fires early or never" — the two halves of the failure mode, on one restored gate, for a
        # backwards clock too (a restart onto an earlier timeline is the same broken-anchor case).
        _before, after, _snap = self._restored()
        self.assertEqual(after.evaluate(self.PENDING, YES, hold_s=self.HOLD_S, now=-100.0), SILENT)
        self.assertEqual(after._dwell_since[self.PENDING], -100.0)
        self.assertEqual(after.evaluate(self.PENDING, YES, hold_s=self.HOLD_S, now=-96.0), SILENT)
        self.assertEqual(after.evaluate(self.PENDING, YES, hold_s=self.HOLD_S, now=-95.0), FIRE)


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
