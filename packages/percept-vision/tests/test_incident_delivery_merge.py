"""The per-watch delivery merge — one occurrence, one notification.

Scope of the thing under test (``percept.behavior.INCIDENT_MERGE_WINDOW_S``): a SECOND incident for
the SAME gate key landing inside the window is a re-report of the occurrence already delivered, and
is dropped at the delivery seam. It is a notification-volume guard only — the judge sees nothing
different and :class:`~percept.gate.Gate` decides nothing different, so it cannot move precision or
recall on the occurrence the client already heard about.

The two properties that make it safe are pinned here as hard assertions: the merge is PER-WATCH (two
watches firing in the same instant are two keys and both survive), and a stream with no duplicates is
untouched (every gate decision, reason and fire is identical with the merge on and off).

Two further properties are pinned below the window itself: the anchor map stays bounded even when
the caller's clock never moves (``_MERGE_MAX_KEYS``), and the episode/rhythm route
(:meth:`BehaviorExecutor.fire_episode`) bypasses this seam entirely — a fact the window's own
derivation note declares but nothing used to assert.
"""

from __future__ import annotations

import unittest

from percept.adapters import Goal, Verdict
from percept.behavior import (
    INCIDENT_MERGE_WINDOW_S,
    _MERGE_MAX_KEYS,
    BehaviorExecutor,
    BehaviorProgram,
    EpisodeFoldResult,
)
from percept.gate import FIRE, SILENT, Gate
from percept.observability import Metrics

YES = Verdict(satisfied=True, confidence=0.95, evidence="yes")
NO = Verdict(satisfied=False, confidence=0.95, evidence="no")
PROG = BehaviorProgram(sense="judge")


def _goal(goal_id: str = "g") -> Goal:
    return Goal(id=goal_id, condition="c", say="fired", state={"type": "watch"})


def _new_episode(episode_id: str) -> EpisodeFoldResult:
    """The fold the clip/occurrence path builds for every reported event: a fresh episode that
    re-arms its own edge. This is what makes the gate re-fire, and therefore what produces the
    duplicate deliveries this merge exists to collapse."""
    return EpisodeFoldResult(
        episode_id=episode_id,
        match="new",
        mutation="none",
        should_rearm=True,
        should_evaluate_gate=True,
    )


class MergeWindowTests(unittest.TestCase):
    def _fire(self, ex, goal, t, key=("g", "scene"), episode=None):
        fold = _new_episode(episode) if episode is not None else None
        return ex.fire_judged(goal, YES, key, PROG, t=t, fold=fold)

    def test_two_incidents_for_one_watch_inside_the_window_collapse_to_the_first(self):
        ex = BehaviorExecutor(Gate())
        goal = _goal()

        first = self._fire(ex, goal, t=10.0, episode="e1")
        second = self._fire(ex, goal, t=10.4, episode="e2")

        self.assertIsNotNone(first)
        self.assertEqual(first.action, FIRE)
        self.assertIsNone(second)  # ...and it is the FIRST that survives, not the last
        self.assertEqual(first.verdict.evidence, "yes")

    def test_two_watches_firing_inside_the_window_both_survive(self):
        # The non-negotiable one: the merge is keyed exactly like the gate, so two DIFFERENT watches
        # firing 0.1s apart are two different occurrences and must never collapse into one.
        ex = BehaviorExecutor(Gate())
        left, right = _goal("left"), _goal("right")

        a = ex.fire_judged(left, YES, ("left", "scene"), PROG, t=10.0)
        b = ex.fire_judged(right, YES, ("right", "scene"), PROG, t=10.1)

        self.assertIsNotNone(a)
        self.assertIsNotNone(b)
        self.assertEqual([a.goal_id, b.goal_id], ["left", "right"])

    def test_same_watch_but_different_entities_both_survive(self):
        # The gate key is (goal, entity); two entities are two edges, so they are two notifications.
        ex = BehaviorExecutor(Gate())
        goal = _goal()

        a = ex.fire_judged(goal, YES, ("g", "cup-1"), PROG, t=10.0)
        b = ex.fire_judged(goal, YES, ("g", "cup-2"), PROG, t=10.1)

        self.assertIsNotNone(a)
        self.assertIsNotNone(b)

    def test_incidents_outside_the_window_both_survive(self):
        ex = BehaviorExecutor(Gate())
        goal = _goal()

        first = self._fire(ex, goal, t=10.0, episode="e1")
        second = self._fire(ex, goal, t=10.0 + INCIDENT_MERGE_WINDOW_S + 0.05, episode="e2")

        self.assertIsNotNone(first)
        self.assertIsNotNone(second)

    def test_the_window_boundary_is_half_open(self):
        # A fire landing exactly ``window`` later is a separate occurrence, never a merged one.
        # Uses a binary-exact window so the boundary is testable without float drift.
        ex = BehaviorExecutor(Gate(), merge_window_s=0.5)
        goal = _goal()

        self._fire(ex, goal, t=0.0, episode="e1")
        inside = self._fire(ex, goal, t=0.25, episode="e2")
        at_the_boundary = self._fire(ex, goal, t=0.5, episode="e3")

        self.assertIsNone(inside)
        self.assertIsNotNone(at_the_boundary)

    def test_a_real_rep_at_the_measured_floor_is_never_merged(self):
        # The tightest rep-to-rep gap measured across the repcount corpus is 0.79s. On the shipped
        # path those reps arrive as judged occurrences through this seam, so the window has to clear
        # that floor for every one of them.
        ex = BehaviorExecutor(Gate())
        goal = _goal()

        reps = [self._fire(ex, goal, t=round(i * 0.79, 2), episode=f"rep{i}") for i in range(6)]

        self.assertTrue(all(rep is not None for rep in reps))

    def test_a_suppressed_duplicate_does_not_slide_the_window_forward(self):
        # Anchored on the last DELIVERED incident, never on a dropped one — so a dense duplicate
        # stream cannot suppress indefinitely.
        ex = BehaviorExecutor(Gate())
        goal = _goal()
        fires = [self._fire(ex, goal, t=t, episode=f"e{i}")
                 for i, t in enumerate((10.0, 10.6, 11.1))]

        self.assertIsNotNone(fires[0])
        self.assertIsNone(fires[1])       # 0.6s after the delivered one -> merged
        self.assertIsNotNone(fires[2])    # 1.1s after it -> a new notification, not 0.5s after 10.6

    def test_a_confident_false_between_two_fires_is_a_proven_boundary(self):
        # The condition was OBSERVED to stop holding, so the next true is a new occurrence and not a
        # re-report — even when both land on the same timestamp.
        ex = BehaviorExecutor(Gate())
        goal = _goal()

        first = ex.fire_judged(goal, YES, ("g", "scene"), PROG, t=10.0)
        ex.fire_judged(goal, NO, ("g", "scene"), PROG, t=10.0)
        second = ex.fire_judged(goal, YES, ("g", "scene"), PROG, t=10.0)

        self.assertIsNotNone(first)
        self.assertIsNotNone(second)


class NoDuplicatesIsUnchangedTests(unittest.TestCase):
    """A run with no duplicates must be byte-for-byte what it was before the merge existed."""

    #: Rising edges 2.5s apart with a confident-false between them — no two incidents ever land
    #: inside the window, so nothing is eligible to merge.
    _STREAM = [(0.0, YES), (1.0, NO), (2.5, YES), (3.0, NO), (5.0, YES)]

    def _run(self, merge_window_s):
        ex = BehaviorExecutor(Gate(), merge_window_s=merge_window_s)
        goal = _goal()
        trace = []
        for t, verdict in self._STREAM:
            outcomes = []
            fired = ex.fire_judged(
                goal, verdict, ("g", "scene"), PROG, t=t,
                outcome_sink=outcomes.append,
            )
            trace.append((
                t,
                [(o.action, o.reason) for o in outcomes],
                None if fired is None else (fired.action, fired.text),
            ))
        return trace

    def test_merge_on_and_off_produce_an_identical_trace(self):
        self.assertEqual(self._run(INCIDENT_MERGE_WINDOW_S), self._run(0.0))

    def test_every_rising_edge_still_fires(self):
        fires = [row for row in self._run(INCIDENT_MERGE_WINDOW_S) if row[2] is not None]
        self.assertEqual(len(fires), 3)


class WindowConfigurationTests(unittest.TestCase):
    def test_the_default_window_is_the_documented_constant(self):
        self.assertEqual(BehaviorExecutor(Gate()).merge_window_s, INCIDENT_MERGE_WINDOW_S)
        self.assertEqual(INCIDENT_MERGE_WINDOW_S, 0.7)

    def test_the_window_stays_under_the_measured_rep_floor(self):
        # 183 rep-to-rep gaps across 20 repcount cases bottom out at 0.79s, and on the SHIPPED path a
        # repcount goal is a judged watch running through THIS seam. A window at or above that floor
        # deletes real, separately-counted occurrences instead of judgment-cadence artifacts.
        self.assertLess(INCIDENT_MERGE_WINDOW_S, 0.79)
        # ...and it stays wide enough to cover the strides that actually produce the duplicates.
        for stride in (1 / 3, 0.5, 2 / 3):
            self.assertLess(stride, INCIDENT_MERGE_WINDOW_S)

    def test_the_window_is_configurable(self):
        ex = BehaviorExecutor(Gate(), merge_window_s=5.0)
        goal = _goal()

        self.assertEqual(ex.merge_window_s, 5.0)
        self.assertIsNotNone(ex.fire_judged(goal, YES, ("g", "scene"), PROG, t=0.0,
                                            fold=_new_episode("e1")))
        # 2.0s apart would survive the 0.7s default; under a 5.0s window it merges.
        self.assertIsNone(ex.fire_judged(goal, YES, ("g", "scene"), PROG, t=2.0,
                                         fold=_new_episode("e2")))

    def test_a_zero_window_disables_the_merge_entirely(self):
        ex = BehaviorExecutor(Gate(), merge_window_s=0.0)
        goal = _goal()

        self.assertIsNotNone(ex.fire_judged(goal, YES, ("g", "scene"), PROG, t=0.0,
                                            fold=_new_episode("e1")))
        self.assertIsNotNone(ex.fire_judged(goal, YES, ("g", "scene"), PROG, t=0.1,
                                            fold=_new_episode("e2")))

    def test_merged_duplicates_are_metered_never_silent(self):
        class Metrics:
            def __init__(self):
                self.counts = {}

            def incr(self, key):
                self.counts[key] = self.counts.get(key, 0) + 1

        metrics = Metrics()
        ex = BehaviorExecutor(Gate(), metrics=metrics)
        goal = _goal()
        ex.fire_judged(goal, YES, ("g", "scene"), PROG, t=0.0, fold=_new_episode("e1"))
        ex.fire_judged(goal, YES, ("g", "scene"), PROG, t=0.4, fold=_new_episode("e2"))

        self.assertEqual(metrics.counts.get("gate.merged_duplicate"), 1)
        self.assertEqual(metrics.counts.get("gate.fire"), 2)  # the GATE still decided twice


class GateIsUntouchedTests(unittest.TestCase):
    """The merge is a delivery guard. It must not reach into the gate's decision or its state."""

    def test_a_merged_delivery_still_recorded_a_real_gate_fire(self):
        gate = Gate()
        ex = BehaviorExecutor(gate)
        goal = _goal()
        outcomes = []

        ex.fire_judged(goal, YES, ("g", "scene"), PROG, t=0.0, fold=_new_episode("e1"),
                       outcome_sink=outcomes.append)
        merged = ex.fire_judged(goal, YES, ("g", "scene"), PROG, t=0.4, fold=_new_episode("e2"),
                                outcome_sink=outcomes.append)

        self.assertIsNone(merged)
        # The gate DECIDED twice, and both times it decided to fire. Untouched means the decision and
        # the edge state are the gate's own — not that the drop is invisible: the third outcome is the
        # delivery seam reporting, in the gate's typed vocabulary, that IT withheld the second
        # notification (finding reflex-05). Without it a reader of the last outcome is told the gate
        # went silent on a rising edge.
        gate_decisions = [o for o in outcomes if o.reason != "merged_duplicate"]
        self.assertEqual([o.action for o in gate_decisions], [FIRE, FIRE])
        self.assertEqual([o.reason for o in gate_decisions], ["rising_edge", "rising_edge"])
        self.assertEqual(
            [(o.action, o.reason, o.key, o.at) for o in outcomes if o.reason == "merged_duplicate"],
            [(SILENT, "merged_duplicate", ("g", "scene"), 0.4)],
        )
        self.assertFalse(gate._armed[("g", "scene")])  # the edge is spent, exactly as the gate left it

    def test_a_merged_delivery_does_not_advance_an_nth_distinct_count(self):
        # A re-report of an occurrence the client already heard about is not a distinct occurrence.
        ex = BehaviorExecutor(Gate())
        goal = _goal()
        goal.state["fire_when"] = "n == 2"
        from percept.behavior import program_of

        prog = program_of(goal)
        ex.fire_judged(goal, YES, ("g", "scene"), prog, t=0.0, fold=_new_episode("e1"))
        ex.fire_judged(goal, YES, ("g", "scene"), prog, t=0.4, fold=_new_episode("e2"))

        self.assertEqual(ex.ctx("g").count, 1)  # the merged one was never counted


class DegenerateClockBackstopTests(unittest.TestCase):
    """``_MERGE_MAX_KEYS`` — the only thing bounding the anchor map when the clock never moves.

    ``_note_merge_anchor`` retires a stamp with ``t - v < window``, which can only ever fire once
    ``t`` has moved past it. A caller handing the executor a constant ``t`` — a stalled source
    clock, a batch replay that forgot to thread time — therefore prunes NOTHING, and the map would
    otherwise grow one permanent entry per distinct gate key. The cap is the backstop. What follows
    pins that the bound holds, and which direction it errs in when it drops a stamp.
    """

    #: Comfortably past the cap, so the eviction runs many times over.
    _OVERFLOW = _MERGE_MAX_KEYS + 200

    def _flood(self, ex, goal, stamp_of):
        """One distinct occurrence per key. Every key is fresh, so every one is a real gate fire
        and none of them is eligible to merge — the map fills purely from delivered occurrences."""
        return sum(
            ex.fire_judged(goal, YES, ("g", f"cup-{i}"), PROG, t=stamp_of(i)) is not None
            for i in range(self._OVERFLOW)
        )

    def test_a_clock_that_never_advances_cannot_grow_the_anchor_map_without_bound(self):
        ex = BehaviorExecutor(Gate())
        fires = self._flood(ex, _goal(), lambda i: 7.5)

        # Every one of them was a distinct watch-key occurrence, so the merge suppressed nothing.
        self.assertEqual(fires, self._OVERFLOW)
        self.assertLessEqual(len(ex._merge_anchors), _MERGE_MAX_KEYS)
        # ...and the cap is doing all of the work: under a frozen clock ``t - v`` is always 0, so
        # the prune retires nothing and the map would otherwise hold every key ever delivered.
        self.assertEqual(len(ex._merge_anchors), _MERGE_MAX_KEYS)

    def test_the_bound_also_holds_for_a_clock_that_moves_backwards(self):
        # Source time walking backwards. Every stamp already in the map is in the FUTURE of ``t``,
        # so ``t - v`` is negative and the prune deliberately keeps it (a future stamp is another
        # key's clock running ahead, not a stale one) — the cap is again the only bound.
        ex = BehaviorExecutor(Gate())
        fires = self._flood(ex, _goal(), lambda i: 10_000.0 - i)

        self.assertEqual(fires, self._OVERFLOW)
        self.assertLessEqual(len(ex._merge_anchors), _MERGE_MAX_KEYS)

    def test_an_evicted_anchor_fails_open_and_never_suppresses_a_real_occurrence(self):
        # The safety direction. Losing an anchor may cost a notification-volume saving; it must
        # never cost a notification. A dropped anchor has to DELIVER, exactly as a key that never
        # fired before does.
        ex = BehaviorExecutor(Gate())
        goal = _goal()
        self._flood(ex, goal, lambda i: 7.5)

        # Oldest-stamped-first: the FIRST keys flooded in are the ones the cap gave up.
        evicted = ("g", "cup-0")
        retained = ("g", f"cup-{self._OVERFLOW - 1}")
        self.assertNotIn(evicted, ex._merge_anchors)
        self.assertIn(retained, ex._merge_anchors)

        # A fresh episode on each key re-arms the gate, so both reach the delivery seam and the
        # only thing that can differ between them is the anchor.
        after_eviction = ex.fire_judged(goal, YES, evicted, PROG, t=7.5, fold=_new_episode("x"))
        still_anchored = ex.fire_judged(goal, YES, retained, PROG, t=7.5, fold=_new_episode("y"))

        self.assertIsNotNone(after_eviction)  # dropped anchor -> delivered, never silently swallowed
        self.assertIsNone(still_anchored)     # a live anchor still merges — nothing else changed

    def test_the_cap_keeps_the_most_recently_stamped_keys_when_every_timestamp_is_equal(self):
        """BUG-D-06-01, closed. The survivors are the LAST ``_MERGE_MAX_KEYS`` stamped.

        Under a frozen clock every stamp ties, so a sort by timestamp has nothing to order by:
        Python's sort is stable and its reverse sort does not reorder equal elements, so
        ``sorted(..., reverse=True)[:1024]`` kept the 1024 EARLIEST-inserted keys — "newest wins"
        was true only for a clock that moves, i.e. exactly not the case the cap exists for. The
        bound is now insertion-ordered (``BoundedRecencyMap``), which is a total order whatever the
        clock does.
        """
        ex = BehaviorExecutor(Gate())
        for i in range(self._OVERFLOW):
            ex._note_merge_anchor(("g", f"cup-{i}"), 7.5)

        kept = list(ex._merge_anchors)
        self.assertEqual(len(kept), _MERGE_MAX_KEYS)
        self.assertEqual(kept[0], ("g", f"cup-{self._OVERFLOW - _MERGE_MAX_KEYS}"))
        self.assertEqual(kept[-1], ("g", f"cup-{self._OVERFLOW - 1}"))

    def test_the_backstop_keeps_the_stamp_it_just_took(self):
        ex = BehaviorExecutor(Gate())
        for i in range(self._OVERFLOW):
            ex._note_merge_anchor(("g", f"cup-{i}"), 7.5)

        newest = ("g", f"cup-{self._OVERFLOW - 1}")
        self.assertIn(newest, ex._merge_anchors)

    def test_a_key_restamped_under_a_frozen_clock_is_moved_back_out_of_the_eviction_line(self):
        """Recency is refreshed on every stamp, not only on first insert.

        Without the pop-then-reinsert a key that fires on every tick would sit at the front of the
        map forever, holding its ORIGINAL position, and be evicted ahead of a neighbour that has not
        been seen since it was created — under a frozen clock, where the timestamps cannot break the
        tie, that is the whole difference between an LRU and a FIFO.
        """
        ex = BehaviorExecutor(Gate())
        hot = ("g", "cup-0")
        for i in range(_MERGE_MAX_KEYS):
            ex._note_merge_anchor(("g", f"cup-{i}"), 7.5)
        ex._note_merge_anchor(hot, 7.5)          # re-stamped -> moves to the back
        ex._note_merge_anchor(("g", "cup-new"), 7.5)  # overflows by one

        self.assertIn(hot, ex._merge_anchors, "the re-stamped key must not be the one evicted")
        self.assertNotIn(("g", "cup-1"), ex._merge_anchors, "…the next-oldest goes instead")
        self.assertEqual(len(ex._merge_anchors), _MERGE_MAX_KEYS)

    def test_every_capped_drop_is_reported_with_a_reason(self):
        """ADR-001 §9: a drop forced by a bound is never a silent truncation.

        The old three-line truncation dropped up to 200 anchors per call with no log, no metric and
        no reason. Each drop now reaches the executor's metrics AND a structured log line — losing
        an anchor fails open, so it costs a duplicate notification, which is cheap but not free.
        """
        metrics = Metrics()
        ex = BehaviorExecutor(Gate(), metrics)
        with self.assertLogs("percept.behavior", level="INFO") as caught:
            for i in range(_MERGE_MAX_KEYS + 3):
                ex._note_merge_anchor(("g", f"cup-{i}"), 7.5)

        self.assertEqual(metrics.get("merge_anchor.evicted"), 3)
        reasons = [getattr(rec, "fields", {}).get("reason") for rec in caught.records]
        self.assertEqual(reasons, ["max_keys"] * 3)
        self.assertTrue(all("merge_anchor.evicted" in line for line in caught.output))

    def test_the_window_prune_is_not_an_eviction_and_stays_quiet(self):
        """The other half of the same rule, in the other direction. A stamp older than the window
        has no suppressing power left, so retiring it costs nothing and must NOT be reported —
        otherwise every ordinary delivery on a moving clock emits a drop line."""
        metrics = Metrics()
        ex = BehaviorExecutor(Gate(), metrics)
        ex._note_merge_anchor(("g", "a"), 0.0)
        ex._note_merge_anchor(("g", "b"), 100.0)  # 'a' is far outside the 0.7s window

        self.assertEqual(dict(ex._merge_anchors), {("g", "b"): 100.0})
        self.assertEqual(metrics.get("merge_anchor.evicted"), 0)


class _MergeSpy(BehaviorExecutor):
    """Records every consultation of the delivery merge, so "this route never touches it" is an
    assertion about calls made and not merely about an end state that happens to look empty."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.merge_checks: list = []
        self.anchor_stamps: list = []

    def _merges_into_prior_delivery(self, key, t, *args, **kwargs):
        self.merge_checks.append((key, t))
        return super()._merges_into_prior_delivery(key, t, *args, **kwargs)

    def _note_merge_anchor(self, key, t):
        self.anchor_stamps.append((key, t))
        return super()._note_merge_anchor(key, t)


class EpisodeRouteBypassesTheMergeTests(unittest.TestCase):
    """``fire_episode`` — the episode/rhythm route — goes straight to the gate.

    The merge window's derivation note names this route as the one that bypasses the seam, and that
    exemption is load-bearing in both directions: an episode fire must not be swallowed by a judged
    delivery's anchor, and must not leave an anchor that swallows a later judged delivery.
    """

    _EPISODE = BehaviorProgram(sense="episode")

    #: The singleton gate key ``fire_episode`` folds every episode of one goal onto.
    _KEY = ("g", "__episode__")

    def test_two_new_episodes_inside_the_window_both_fire(self):
        ex = BehaviorExecutor(Gate())
        goal = _goal()

        first = ex.fire_episode(goal, self._EPISODE, n=1, is_new=True, t=0.0, episode_id="e1")
        second = ex.fire_episode(goal, self._EPISODE, n=2, is_new=True, t=0.2, episode_id="e2")

        self.assertIsNotNone(first)
        self.assertIsNotNone(second)
        self.assertEqual([first.action, second.action], [FIRE, FIRE])
        self.assertEqual([first.entity_id, second.entity_id], [1, 2])

        # The control that keeps this from passing for the wrong reason: the SAME 0.2s spacing on
        # the judged route, with the same re-arm, is merged away.
        self.assertLess(0.2, INCIDENT_MERGE_WINDOW_S)
        control, judged_goal = BehaviorExecutor(Gate()), _goal("judged")
        self.assertIsNotNone(control.fire_judged(judged_goal, YES, ("judged", "scene"), PROG,
                                                 t=0.0, fold=_new_episode("e1")))
        self.assertIsNone(control.fire_judged(judged_goal, YES, ("judged", "scene"), PROG,
                                              t=0.2, fold=_new_episode("e2")))

    def test_the_episode_route_never_consults_or_stamps_the_merge(self):
        ex = _MergeSpy(Gate())
        goal = _goal()

        # The spy is live — the judged route does both.
        ex.fire_judged(goal, YES, ("g", "scene"), PROG, t=0.0)
        self.assertEqual(ex.merge_checks, [(("g", "scene"), 0.0)])
        self.assertEqual(ex.anchor_stamps, [(("g", "scene"), 0.0)])

        ex.merge_checks.clear()
        ex.anchor_stamps.clear()
        for i, t in enumerate((0.1, 0.2, 0.3), start=1):
            self.assertIsNotNone(
                ex.fire_episode(goal, self._EPISODE, n=i, is_new=True, t=t, episode_id=f"e{i}")
            )

        self.assertEqual(ex.merge_checks, [])
        self.assertEqual(ex.anchor_stamps, [])

    def test_an_episode_fire_neither_suppresses_nor_is_suppressed_by_a_judged_delivery(self):
        ex = BehaviorExecutor(Gate())
        goal = _goal()

        episode = ex.fire_episode(goal, self._EPISODE, n=1, is_new=True, t=0.0, episode_id="e1")
        self.assertIsNotNone(episode)
        self.assertEqual(ex._merge_anchors, {})  # it left nothing behind to suppress with

        # ...so a judged delivery on the very same key 0.1s later is delivered, not merged away.
        judged = ex.fire_judged(goal, YES, self._KEY, PROG, t=0.1, fold=_new_episode("x"))
        self.assertIsNotNone(judged)
        self.assertEqual(ex._merge_anchors, {self._KEY: 0.1})

        # ...and the anchor that judged delivery DID leave cannot reach the episode route: the next
        # episode lands 0.2s later, well inside the window, and still fires.
        again = ex.fire_episode(goal, self._EPISODE, n=2, is_new=True, t=0.3, episode_id="e2")
        self.assertIsNotNone(again)
        self.assertLess(0.3 - 0.1, INCIDENT_MERGE_WINDOW_S)
        self.assertEqual(ex._merge_anchors, {self._KEY: 0.1})  # unmoved by the episode fire


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
