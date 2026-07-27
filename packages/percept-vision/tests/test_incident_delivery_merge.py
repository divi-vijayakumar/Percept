"""The per-watch delivery merge — one occurrence, one notification.

Scope of the thing under test (``percept.behavior.INCIDENT_MERGE_WINDOW_S``): a SECOND incident for
the SAME gate key landing inside the window is a re-report of the occurrence already delivered, and
is dropped at the delivery seam. It is a notification-volume guard only — the judge sees nothing
different and :class:`~percept.gate.Gate` decides nothing different, so it cannot move precision or
recall on the occurrence the client already heard about.

The two properties that make it safe are pinned here as hard assertions: the merge is PER-WATCH (two
watches firing in the same instant are two keys and both survive), and a stream with no duplicates is
untouched (every gate decision, reason and fire is identical with the merge on and off).
"""

from __future__ import annotations

import unittest

from percept.adapters import Goal, Verdict
from percept.behavior import (
    INCIDENT_MERGE_WINDOW_S,
    BehaviorExecutor,
    BehaviorProgram,
    EpisodeFoldResult,
)
from percept.gate import FIRE, Gate

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
        self.assertEqual([o.action for o in outcomes], [FIRE, FIRE])
        self.assertEqual([o.reason for o in outcomes], ["rising_edge", "rising_edge"])
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


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
