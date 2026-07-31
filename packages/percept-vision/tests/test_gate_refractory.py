"""L3 (MVP plan E4, precision track): time-refractory on the three-state gate.

Precision forensic finding (eval/viability/JUDGE_LOSS_DEEPDIVE.md, local): ~12% of false fires on the
charades-e2e track were RE-FIRES — the gate re-arming and firing again within one true episode because
the judge briefly flickered confident-false mid-episode. ``refractory_s`` suppresses a fire landing
within N seconds of the same key's last real fire, even across such a blip. Off by default (0/None) is
byte-for-byte the historical gate — asserted here alongside the on-behavior.
"""
from __future__ import annotations

import copy
import unittest

from percept.adapters import Verdict
from percept.gate import FIRE, SILENT, Gate

YES = Verdict(satisfied=True, confidence=0.95, evidence="present")
NO = Verdict(satisfied=False, confidence=0.95, evidence="absent")

# The snapshot format's real key shape (``Gate.snapshot`` docstring): ``(goal_id, entity)``.
KEY = ("goal-a", "e1")


def _fired_snapshot() -> dict:
    """A snapshot of a gate that really fired ``KEY`` at t=0 with a 5s refractory window open.

    Every field ``restore`` reads is populated: ``armed`` (disarmed), ``touched``, ``no_evict``
    and ``last_fire``. This is what a 24/7 watch persists mid-episode — in the CURRENT (v2)
    self-describing record shape, plus the v1 mirror ``snapshot()`` still publishes.
    """
    g = Gate(refractory_s=5.0)
    assert g.evaluate(KEY, YES, now=0.0, no_evict=True) == FIRE
    return g.snapshot()


def _legacy_fired_snapshot() -> dict:
    """The SAME gate state in the PRE-v2 wire shape: five independent pair-lists and no ``keys``.

    Hand-built rather than taken from ``snapshot()`` because that is the point — this is a payload
    written by an older build, sitting in a persistence backend across an upgrade. ADR-006 §6 makes
    the serialized gate key tuple a persistence contract, so ``restore`` must still read it.
    """
    return {
        "armed": [[list(KEY), False]],
        "dwell_since": [],
        "touched": [[list(KEY), 0.0]],
        "no_evict": [list(KEY)],
        "last_fire": [[list(KEY), 0.0]],
    }


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


class RefractoryBoundaryTests(unittest.TestCase):
    """§4.1 Refractory / "Exactly at the boundary".

    ``gate.py:597`` is ``(now - last_fire) < effective_refractory`` — a **strict** ``<``. So the
    inclusive side of the window is the FIRING side: at ``delta == refractory_s`` the fire is not
    suppressed. Nothing pinned that before, and an off-by-epsilon (``<=``) flips a real second
    episode landing exactly on the boundary into silence — the "real event delivered as none"
    failure. These four tests are the pin, on both the gate-wide default and the per-call override.
    """

    def test_exactly_at_the_refractory_boundary_fires(self):
        g = Gate(refractory_s=5.0)
        self.assertEqual(g.evaluate("k", YES, now=0.0), FIRE)    # real fire, stamps _last_fire=0.0
        self.assertEqual(g.evaluate("k", NO, now=1.0), SILENT)   # re-arms; without this the edge is spent
        out = g.decide("k", YES, now=5.0)                        # delta == 5.0 == refractory_s
        self.assertEqual(out.action, FIRE)
        self.assertEqual(out.reason, "rising_edge")

    def test_one_millisecond_inside_the_refractory_boundary_is_suppressed(self):
        """The sibling of the boundary test: one tick earlier is the suppressing side."""
        g = Gate(refractory_s=5.0)
        self.assertEqual(g.evaluate("k", YES, now=0.0), FIRE)
        self.assertEqual(g.evaluate("k", NO, now=1.0), SILENT)
        out = g.decide("k", YES, now=4.999)                      # delta 4.999 < 5.0
        self.assertEqual(out.action, SILENT)
        self.assertEqual(out.reason, "refractory")

    def test_per_call_refractory_boundary_is_also_the_firing_side(self):
        """A per-call ``refractory_s=`` takes precedence over the gate-wide default (``:594``), and
        carries the same inclusive-side semantics. The 100s gate default would suppress here."""
        g = Gate(refractory_s=100.0)
        self.assertEqual(g.evaluate("k", YES, now=10.0), FIRE)   # _last_fire=10.0
        self.assertEqual(g.evaluate("k", NO, now=11.0), SILENT)
        out = g.decide("k", YES, now=15.0, refractory_s=5.0)     # delta == 5.0 == the override
        self.assertEqual(out.action, FIRE)
        self.assertEqual(out.reason, "rising_edge")

    def test_per_call_refractory_just_inside_the_boundary_is_suppressed(self):
        g = Gate(refractory_s=100.0)
        self.assertEqual(g.evaluate("k", YES, now=10.0), FIRE)
        self.assertEqual(g.evaluate("k", NO, now=11.0), SILENT)
        out = g.decide("k", YES, now=14.999, refractory_s=5.0)
        self.assertEqual(out.action, SILENT)
        self.assertEqual(out.reason, "refractory")


class RestoreOfAMalformedSnapshotTests(unittest.TestCase):
    """§4.1 Rising-edge latch / "``restore()`` of a malformed/legacy snapshot".

    Pass bar: **no raise, no lost suppression**. ``test_harden_c_bounds.py`` covers only the one
    legacy shape (no ``touched``/``no_evict``); everything below is the rest of the row — each field
    omitted in turn, and the hostile shapes a persistence backend can hand back after a redeploy.

    Both WIRE SHAPES are exercised, because ``restore`` must read both:

    * the **v2 self-describing records** ``snapshot()`` writes today (``_fired_snapshot``), and
    * the **v1 pair-lists** a snapshot taken by an older build carries (``_legacy_fired_snapshot``).

    The field-omission and flattening cases live on the v1 helper on purpose: they are statements
    about the pair-list shape, and v2 is immune to them BY CONSTRUCTION — a record names its own
    fields, so there is no positional slot for a lost nesting level to slide into. That immunity is
    itself asserted below (``test_a_tampered_v1_mirror_cannot_corrupt_a_v2_restore``).
    """

    def test_each_v1_snapshot_field_omitted_in_turn_restores_and_keeps_the_suppression(self):
        """Drop one field at a time from a v1 pair-list snapshot. Nothing may raise, and the key that
        was disarmed pre-restart must still be silent on its next true tick inside the window.

        Which mechanism carries the suppression differs, and that is the point of the reason check:
        with ``armed`` present the latch does it (``already_fired``); with ``armed`` dropped the
        restored refractory stamp does it (``refractory``). The two are independent belts.
        """
        snap = _legacy_fired_snapshot()
        cases = {
            "last_fire": "already_fired",  # legacy pre-refractory snapshot
            "touched": "already_fired",    # legacy pre-eviction snapshot
            "no_evict": "already_fired",   # legacy pre-eviction snapshot
            "armed": "refractory",         # the latch is gone; the refractory stamp still suppresses
        }
        for field, expected_reason in cases.items():
            with self.subTest(omitted=field):
                partial = copy.deepcopy(snap)
                partial.pop(field)
                g = Gate(refractory_s=5.0)
                g.restore(partial)  # must not raise
                out = g.decide(KEY, YES, now=0.5)
                self.assertEqual(out.action, SILENT)
                self.assertEqual(out.reason, expected_reason)

    def test_each_v2_record_field_nulled_in_turn_restores_and_keeps_the_suppression(self):
        """The same matrix against the v2 records. A field a writer could not populate arrives as
        ``null`` on the record rather than as a missing top-level list, and reads the same way."""
        cases = {
            "last_fire": "already_fired",
            "touched": "already_fired",
            "no_evict": "already_fired",
            "armed": "refractory",
        }
        for field, expected_reason in cases.items():
            with self.subTest(nulled=field):
                partial = copy.deepcopy(_fired_snapshot())
                for record in partial["keys"]:
                    record[field] = None
                g = Gate(refractory_s=5.0)
                g.restore(partial)  # must not raise
                out = g.decide(KEY, YES, now=0.5)
                self.assertEqual(out.action, SILENT)
                self.assertEqual(out.reason, expected_reason)

    def test_a_legacy_snapshot_without_last_fire_keeps_the_edge_but_has_no_refractory_memory(self):
        """Documented behavior, pinned: a snapshot written before the refractory knob existed
        restores the armed latch but carries no window to resume. A mid-episode confident-false blip
        therefore re-fires immediately after the restart — unavoidable, because the stamp genuinely
        is not in the payload. Contrast the tampered-stamp case below, where the stamp IS in the
        payload and used to be thrown away."""
        snap = _legacy_fired_snapshot()
        snap.pop("last_fire")
        g = Gate(refractory_s=5.0)
        g.restore(snap)
        self.assertEqual(g.decide(KEY, YES, now=0.5).reason, "already_fired")  # the latch survived
        self.assertEqual(g.decide(KEY, NO, now=1.0).action, SILENT)            # blip re-arms
        out = g.decide(KEY, YES, now=2.0)
        self.assertEqual(out.action, FIRE)          # 2s after the pre-restart fire, but no memory of it
        self.assertEqual(out.reason, "rising_edge")

    def test_a_v1_pair_list_snapshot_still_round_trips_in_full(self):
        """The persistence contract ADR-006 §6 pins: the gate key tuple serializes as a list, and a
        payload in the pre-v2 shape restores every field it carries."""
        g = Gate(refractory_s=5.0)
        g.restore(_legacy_fired_snapshot())
        self.assertEqual(g._armed, {KEY: False})
        self.assertEqual(g._touched, {KEY: 0.0})
        self.assertEqual(g._no_evict, {KEY})
        self.assertEqual(g._last_fire, {KEY: 0.0})

    def test_snapshot_still_publishes_the_v1_mirror_for_an_older_reader(self):
        """v2 writes the pair-lists too, so a snapshot taken by THIS build restores on an older one.
        Dropping them would strand any deployment mid-rollout."""
        snap = _fired_snapshot()
        self.assertEqual(snap["armed"], [[list(KEY), False]])
        self.assertEqual(snap["touched"], [[list(KEY), 0.0]])
        self.assertEqual(snap["no_evict"], [list(KEY)])
        self.assertEqual(snap["last_fire"], [[list(KEY), 0.0]])

    def test_none_and_empty_snapshots_are_total_no_ops_on_a_populated_gate(self):
        """``Percept.restore`` calls ``restore(snap.get("gate"))`` — a state blob written by an older
        build has no ``gate`` key at all, so ``None`` reaches ``restore``. The falsy guard must
        leave a live gate completely alone, not blank it."""
        g = Gate(refractory_s=5.0)
        self.assertEqual(g.evaluate(KEY, YES, now=0.0, no_evict=True), FIRE)
        before = (dict(g._armed), dict(g._last_fire), dict(g._touched), set(g._no_evict))
        g.restore(None)
        g.restore({})
        after = (dict(g._armed), dict(g._last_fire), dict(g._touched), set(g._no_evict))
        self.assertEqual(before, after)
        self.assertEqual(g.decide(KEY, YES, now=0.5).reason, "already_fired")  # suppression intact

    # -- the hostile shapes ---------------------------------------------------------------------
    def test_corrupt_snapshot_shapes_are_handled_without_an_untyped_raise(self):
        """BUG-D-02-02. ``restore`` used to validate nothing, so the shape of the payload leaked out
        as whatever builtin exception the comprehension happened to hit — ``AttributeError`` for a
        non-dict, ``TypeError`` for a null field, ``ValueError`` for a mistyped stamp, none of them
        naming the offending field. A restart path may not be stopped by a garbled snapshot."""
        raised = {}
        for label, snap in (
            ("non-dict", [[KEY[0], KEY[1]]]),
            ("null armed", {"armed": None}),
            ("non-float stamp", {"last_fire": [[list(KEY), "abc"]]}),
            ("string payload", "armed"),
            ("armed is a string", {"armed": "goal-a"}),
            ("records are not objects", {"keys": [["goal-a", "e1"]]}),
        ):
            g = Gate(refractory_s=5.0)
            try:
                g.restore(snap)
            except Exception as exc:  # noqa: BLE001 - the whole point is which type escapes
                raised[label] = type(exc).__name__
        self.assertEqual(raised, {}, "restore() must not raise an untyped error on a corrupt snapshot")

    def test_a_rejected_top_level_payload_leaves_the_live_gate_untouched(self):
        """Declining a whole payload is the fail-safe direction: whatever the gate already knew about
        suppression survives, rather than being replaced by a corrupt half."""
        g = Gate(refractory_s=5.0)
        self.assertEqual(g.evaluate(KEY, YES, now=0.0, no_evict=True), FIRE)
        g.restore([[KEY[0], KEY[1]]])          # a list, not a mapping -> declined
        self.assertEqual(g._armed, {KEY: False})
        self.assertEqual(g._last_fire, {KEY: 0.0})
        self.assertEqual(g.decide(KEY, YES, now=0.5).reason, "already_fired")

    def test_a_corrupt_entry_is_skipped_with_a_log_never_silently(self):
        """A skipped entry is a data loss, so it is traced (ADR-001 §9: never a silent drop)."""
        g = Gate(refractory_s=5.0)
        with self.assertLogs("percept.snapshot", level="INFO") as caught:
            g.restore({"armed": [list(KEY)]})
        self.assertTrue(
            any("snapshot.entry_skipped" in line for line in caught.output), caught.output
        )

    def test_a_failed_restore_does_not_duplicate_the_notification(self):
        """BUG-D-02-01, the user-visible bar for the row ("no lost suppression"): whatever ``restore``
        does with a corrupt payload, the watch must not send a second notification for an episode it
        already notified about 2 seconds and one process restart ago.

        The old restore assigned ``_armed`` before ``_last_fire`` and raised in between, keeping the
        disarmed latch and dropping the refractory window — the two belts are INDEPENDENT, so losing
        exactly one is strictly worse than losing both. The v2 records carry both or neither."""
        snap = _fired_snapshot()                       # real fire at t=0, 5s window
        snap["last_fire"] = [[list(KEY), "abc"]]       # the v1 mirror is unparseable; v2 is authoritative
        g = Gate(refractory_s=5.0)
        g.restore(snap)
        self.assertEqual(g.decide(KEY, NO, now=1.0).action, SILENT)  # the mid-episode flicker re-arms
        out = g.decide(KEY, YES, now=2.0)                            # 2s < 5s since the last real fire
        self.assertEqual(out.action, SILENT)
        self.assertEqual(out.reason, "refractory")

    def test_a_tampered_v1_mirror_cannot_corrupt_a_v2_restore(self):
        """The structural half of the fix: the mirror is an OUTPUT for older readers, never an input
        to this build's restore. Garbling every one of its fields changes nothing."""
        snap = _fired_snapshot()
        snap.update({"armed": None, "touched": "x", "no_evict": 7, "last_fire": [[["z"], "abc"]]})
        g = Gate(refractory_s=5.0)
        g.restore(snap)
        self.assertEqual(g._armed, {KEY: False})
        self.assertEqual(g._last_fire, {KEY: 0.0})
        self.assertEqual(g._no_evict, {KEY})

    def test_a_corrupt_stamp_never_costs_only_one_of_the_two_belts(self):
        """The invariant behind BUG-D-02-01, stated directly. A restore either keeps a key's latch
        AND its refractory memory, or keeps neither — it can never keep the latch alone, which is
        the combination that re-notifies on the next blip."""
        for snap in (
            {"armed": [[list(KEY), False]], "last_fire": [[list(KEY), "abc"]]},
            {"keys": [{"key": list(KEY), "armed": False, "last_fire": "abc"}]},
        ):
            with self.subTest(shape=sorted(snap)):
                g = Gate(refractory_s=5.0)
                g.restore(snap)
                latched = g._armed.get(KEY) is False
                remembered = KEY in g._last_fire
                self.assertFalse(
                    latched and not remembered,
                    "restore kept the disarmed latch but dropped the refractory window",
                )

    def test_a_flattened_armed_entry_does_not_silently_manufacture_a_key(self):
        """BUG-D-02-03. The correct v1 entry is ``[[goal, entity], False]``; one level of nesting lost
        gives ``[goal, entity]``, which unpacks as ``k=goal, v=entity``. ``tuple("goal-a")`` is then a
        six-character tuple and ``bool("e1")`` is True — a key nobody will ever evaluate, armed, while
        the real key is absent and an absent key defaults to ARMED. It must be SKIPPED, never
        reinterpreted."""
        g = Gate(refractory_s=5.0)
        g.restore({"armed": [list(KEY)]})
        self.assertNotIn(tuple("goal-a"), g._armed, "restore() invented a per-character gate key")
        self.assertEqual(g._armed, {})

    def test_the_v2_record_shape_makes_a_lost_nesting_level_unrepresentable(self):
        """Why the record form exists: a v2 record names its own fields, so there is no positional
        slot a flattened key can occupy. The flattened form is not a mis-parse, it is not a record."""
        g = Gate(refractory_s=5.0)
        g.restore({"keys": [list(KEY)]})
        self.assertEqual(g._armed, {})
        self.assertEqual(g._keys, {})


if __name__ == "__main__":
    unittest.main()
