"""— provisional incident + superseding revision (not a wide settle window).

For a correction arriving 21-34s after the original claim, Percept emits the finding PROVISIONALLY on
the rising edge and, when the mutation channel classifies a later turn as replace/retract on the
SAME episode (``EpisodeFoldResult.supersedes``), emits a SUPERSEDING revision that carries the link
back to the prior incident — so the consumer sees BOTH the original and the correction with the
relation between them, instead of a coequal unlinked peer. It does NOT withhold every incident for a
settle window wide enough to catch a 34s retraction (that would destroy latency for every claim).

This module pins:

* the fold's ``supersedes`` link surfaced on the fire (``behavior.Fired``);
* the runtime-local annotation carried across the emission boundary (``incidents.make_incident``);
* the delivery-boundary ``SupersessionSink`` resolving revision -> prior incident id + TRACING it;
* the SLO-bounded ``SettleWindow`` (default provisional-immediate, ceiling at the reflex budget).
"""

import asyncio
import unittest

from percept.adapters import Goal, Verdict
from percept.behavior import BehaviorExecutor, BehaviorProgram, revision_id
from percept.episodes import EpisodeLog, EpisodePolicy
from percept.gate import Gate
from percept.incidents import incident_revision, incident_supersedes, make_incident
from percept.runtime.sinks import (
    SupersessionLedger,
    SupersessionSink,
    superseded_incident_id,
)
from percept.slo import (
    SETTLE_WINDOW_DEFAULT_S,
    SETTLE_WINDOW_MAX_S,
    SettleWindow,
)

_YES = Verdict(satisfied=True, confidence=1.0, evidence="scripted true")


def _goal(**state):
    return Goal(id="g", condition="c", say="fired", state={"type": "watch", **state})


class _CountingMetrics:
    def __init__(self):
        self.seen = []

    def incr(self, key):
        self.seen.append(key)


# --------------------------------------------------------------------------------------------------
# behavior.py — the fold's supersedes link surfaces on the fire.
# --------------------------------------------------------------------------------------------------


class TestFireCarriesSupersession(unittest.TestCase):
    def _drive_update(self, mutation):
        """A rises + fires (provisional), then a same-episode ``mutation`` correction fires a
        superseding revision. Returns (first_fire, correction_fire, episode_id)."""
        gate = Gate()
        metrics = _CountingMetrics()
        ex = BehaviorExecutor(gate, metrics=metrics)
        goal, prog = _goal(), BehaviorProgram(sense="judge")
        log = EpisodeLog("g", policy=EpisodePolicy(continuity_gap_s=1000.0))
        key = (goal.id, "scene")

        d0 = log.observe_decision(0.0, "s", subject_key="s")
        first = ex.fire_judged(goal, _YES, key, prog, t=0.0, fold=ex.fold_episode(d0))
        d1 = log.observe_decision(25.0, "s", subject_key="s")  # a 21-34s-later correction
        correction = ex.fire_judged(
            goal, _YES, key, prog, t=25.0, fold=ex.fold_episode(d1, mutation=mutation)
        )
        return first, correction, d0.episode.id, metrics

    def test_provisional_fire_has_no_supersedes(self):
        first, _correction, episode_id, _m = self._drive_update("updated")
        self.assertIsNotNone(first)
        self.assertEqual(first.action, "fire")
        self.assertIsNone(first.supersedes)                    # original == provisional
        self.assertEqual(first.revision_id, revision_id(episode_id, 1))

    def test_replace_fires_a_linked_superseding_revision(self):
        _first, correction, episode_id, metrics = self._drive_update("updated")
        self.assertIsNotNone(correction)
        self.assertEqual(correction.action, "fire")
        self.assertEqual(correction.supersedes, revision_id(episode_id, 1))   # links to r1
        self.assertEqual(correction.revision_id, revision_id(episode_id, 2))  # is r2
        self.assertIn("gate.superseding_revision", metrics.seen)              # never silent

    def test_retract_also_fires_a_linked_superseding_revision(self):
        _first, correction, episode_id, metrics = self._drive_update("retracted")
        self.assertIsNotNone(correction)
        self.assertEqual(correction.supersedes, revision_id(episode_id, 1))
        self.assertIn("gate.superseding_revision", metrics.seen)

    def test_fold_none_fire_carries_no_link_byte_for_byte(self):
        # The additive guarantee: a fire with no episode fold has the legacy shape (both None).
        gate = Gate()
        ex = BehaviorExecutor(gate)
        goal, prog = _goal(), BehaviorProgram(sense="judge")
        fired = ex.fire_judged(goal, _YES, (goal.id, "e"), prog, t=0.0)
        self.assertIsNotNone(fired)
        self.assertIsNone(fired.supersedes)
        self.assertIsNone(fired.revision_id)

    def test_unchanged_confirmation_is_not_a_superseding_revision(self):
        # A re-statement is idempotently suppressed, so it never becomes a correction.
        gate = Gate()
        metrics = _CountingMetrics()
        ex = BehaviorExecutor(gate, metrics=metrics)
        goal, prog = _goal(), BehaviorProgram(sense="judge")
        log = EpisodeLog("g", policy=EpisodePolicy(continuity_gap_s=1000.0))
        key = (goal.id, "scene")
        for t in (0.0, 25.0):
            d = log.observe_decision(t, "s", subject_key="s")
            ex.fire_judged(goal, _YES, key, prog, t=t, fold=ex.fold_episode(d))
        self.assertNotIn("gate.superseding_revision", metrics.seen)


# --------------------------------------------------------------------------------------------------
# incidents.py — the link crosses the emission boundary as a runtime-local annotation.
# --------------------------------------------------------------------------------------------------


class _FakeFire:
    def __init__(self, *, revision_id=None, supersedes=None):
        self.goal_id = "g"
        self.text = "fired"
        self.verdict = _YES
        self.entity_id = "s"
        self.revision_id = revision_id
        self.supersedes = supersedes


class TestIncidentCarriesAnnotation(unittest.TestCase):
    def test_make_incident_copies_revision_and_supersedes(self):
        fire = _FakeFire(revision_id="g:ep1#r2", supersedes="g:ep1#r1")
        inc = make_incident(fire, ts=25.0)
        self.assertEqual(incident_revision(inc), "g:ep1#r2")
        self.assertEqual(incident_supersedes(inc), "g:ep1#r1")

    def test_annotation_absent_defaults_none(self):
        inc = make_incident(_FakeFire(), ts=1.0)
        self.assertIsNone(incident_revision(inc))
        self.assertIsNone(incident_supersedes(inc))

    def test_annotation_is_not_a_wire_field(self):
        # The link is RUNTIME-LOCAL: it must never leak into the frozen wire shape / a webhook payload.
        import dataclasses

        from percept.contracts import Incident
        from percept.runtime.sinks import incident_payload

        field_names = {f.name for f in dataclasses.fields(Incident)}
        self.assertNotIn("_revision_id", field_names)
        self.assertNotIn("_supersedes", field_names)
        inc = make_incident(_FakeFire(revision_id="g:ep1#r2", supersedes="g:ep1#r1"), ts=1.0)
        payload = incident_payload(inc)
        self.assertNotIn("_revision_id", payload)
        self.assertNotIn("_supersedes", payload)


# --------------------------------------------------------------------------------------------------
# sinks.py — the delivery boundary resolves revision -> prior incident id and TRACES it.
# --------------------------------------------------------------------------------------------------


class TestSupersessionLedger(unittest.TestCase):
    def test_records_and_resolves(self):
        led = SupersessionLedger()
        led.record("g:ep1#r1", "inc-A")
        self.assertEqual(led.resolve("g:ep1#r1"), "inc-A")
        self.assertIsNone(led.resolve("g:ep1#r9"))

    def test_bounded_drops_oldest_with_trace(self):
        evicted = []
        led = SupersessionLedger(max_records=2, on_evict=lambda rev, why: evicted.append((rev, why)))
        led.record("r1", "inc-1")
        led.record("r2", "inc-2")
        led.record("r3", "inc-3")  # evicts r1
        self.assertIsNone(led.resolve("r1"))
        self.assertEqual(evicted, [("r1", "max_records")])
        self.assertEqual(len(led), 2)


class TestSupersessionSink(unittest.TestCase):
    def _incident(self, inc_id, *, revision_id=None, supersedes=None):
        fire = _FakeFire(revision_id=revision_id, supersedes=supersedes)
        return make_incident(fire, ts=1.0, id=inc_id)

    def test_resolves_prior_incident_id_and_traces(self):
        delivered, traced = [], []
        sink = SupersessionSink(delivered.append, on_supersede=traced.append)
        original = self._incident("inc-A", revision_id="g:ep1#r1")
        correction = self._incident("inc-B", revision_id="g:ep1#r2", supersedes="g:ep1#r1")

        sink(original)     # provisional — recorded, nothing to relate
        sink(correction)   # superseding — resolves to inc-A

        self.assertEqual([i.id for i in delivered], ["inc-A", "inc-B"])
        self.assertEqual(superseded_incident_id(correction), "inc-A")
        self.assertEqual(len(traced), 1)
        self.assertEqual(traced[0]["incident_id"], "inc-B")
        self.assertEqual(traced[0]["supersedes_incident_id"], "inc-A")
        self.assertEqual(traced[0]["supersedes_revision"], "g:ep1#r1")

    def test_supersession_is_traced_even_when_prior_aged_out(self):
        # Never silent: an unresolved link (prior revision gone) is STILL traced, with None.
        traced = []
        sink = SupersessionSink(lambda i: None, on_supersede=traced.append)
        correction = self._incident("inc-B", revision_id="g:ep1#r2", supersedes="g:ep1#r1")
        sink(correction)  # r1 was never recorded
        self.assertEqual(len(traced), 1)
        self.assertIsNone(traced[0]["supersedes_incident_id"])
        self.assertIsNone(superseded_incident_id(correction))

    def test_provisional_incident_passes_through_untraced(self):
        delivered, traced = [], []
        sink = SupersessionSink(delivered.append, on_supersede=traced.append)
        sink(self._incident("inc-A", revision_id="g:ep1#r1"))
        self.assertEqual([i.id for i in delivered], ["inc-A"])
        self.assertEqual(traced, [])  # nothing to relate

    def test_plain_incident_no_annotation_is_byte_for_byte(self):
        delivered = []
        sink = SupersessionSink(delivered.append)
        inc = make_incident(_FakeFire(), ts=1.0, id="inc-X")
        sink(inc)
        self.assertEqual([i.id for i in delivered], ["inc-X"])
        self.assertIsNone(superseded_incident_id(inc))

    def test_async_sink_is_delegated(self):
        delivered = []

        async def _asink(i):
            delivered.append(i.id)

        sink = SupersessionSink(_asink)
        correction = self._incident("inc-B", revision_id="g:ep1#r2", supersedes="g:ep1#r1")
        result = sink(correction)
        asyncio.run(result)  # coroutine returned as-is for the caller to await
        self.assertEqual(delivered, ["inc-B"])


# --------------------------------------------------------------------------------------------------
# slo.py — the settle window is concern-specific and SLO-bounded; the DEFAULT is provisional-immediate.
# --------------------------------------------------------------------------------------------------


class TestSettleWindow(unittest.TestCase):
    def test_default_is_provisional_immediate(self):
        w = SettleWindow()
        self.assertEqual(w.seconds, SETTLE_WINDOW_DEFAULT_S)
        self.assertEqual(SETTLE_WINDOW_DEFAULT_S, 0.0)
        self.assertTrue(w.is_immediate)
        self.assertFalse(w.holds(0.0))  # an immediate window never withholds

    def test_declared_small_window_holds_until_it_closes(self):
        w = SettleWindow(seconds=1.5)
        self.assertFalse(w.is_immediate)
        self.assertTrue(w.holds(0.5))   # still settling
        self.assertTrue(w.holds(1.49))
        self.assertFalse(w.holds(1.5))  # window closed -> emit

    def test_ceiling_is_the_reflex_budget(self):
        self.assertEqual(SETTLE_WINDOW_MAX_S, 2.0)
        SettleWindow(seconds=SETTLE_WINDOW_MAX_S)  # exactly at the ceiling is allowed

    def test_wide_window_to_catch_a_late_retraction_is_rejected(self):
        # A 34s late-retraction window is explicitly refused — a loud error, not a silent
        # latency regression on every claim.
        with self.assertRaises(ValueError):
            SettleWindow(seconds=34.0)

    def test_negative_window_is_rejected(self):
        with self.assertRaises(ValueError):
            SettleWindow(seconds=-1.0)

    def test_resolve_normalises_declarations(self):
        self.assertTrue(SettleWindow.resolve(None).is_immediate)         # declared nothing
        self.assertEqual(SettleWindow.resolve(1.0).seconds, 1.0)          # bare float wrapped
        w = SettleWindow(seconds=0.5)
        self.assertIs(SettleWindow.resolve(w), w)                         # passthrough
        with self.assertRaises(ValueError):
            SettleWindow.resolve(99.0)                                    # bound still enforced


if __name__ == "__main__":
    unittest.main()
