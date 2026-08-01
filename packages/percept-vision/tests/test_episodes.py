"""L1: episode counting — new-vs-continuation dedup, window bound, Percept count goal (stub judge).

Also pins two properties of the episode/gate seam that nothing else did: the fold's re-arm must
stand down when the gate is anchoring on a first-observed false, and the episode-owned incident
ledger's growth is deliberately unbounded.
"""

import unittest

from percept.adapters import Goal, Verdict
from percept.agent import Percept
from percept.behavior import BehaviorExecutor, BehaviorProgram, EpisodeIncidentLedger
from percept.episodes import EpisodeLog, EpisodePolicy
from percept.gate import Gate

# Stub judges (NOT intelligence — they exercise the dedup logic deterministically).
ALWAYS_NEW = lambda summaries, exemplar: None  # noqa: E731
ALWAYS_CONTINUE = lambda summaries, exemplar: summaries[-1]["id"] if summaries else None  # noqa: E731


class TestEpisodeLog(unittest.TestCase):
    def test_sips_are_one_episode(self):
        log = EpisodeLog("g1")
        for t in (1.0, 2.0, 3.0):  # sip, sip, sip
            log.observe(t, "cup", ALWAYS_CONTINUE)
        self.assertEqual(log.count(), 1)

    def test_fresh_cup_increments(self):
        log = EpisodeLog("g1")
        log.observe(1.0, "cup", ALWAYS_NEW)
        log.observe(2.0, "cup", ALWAYS_CONTINUE)  # same cup
        ep, is_new = log.observe(3.0, "cup", ALWAYS_NEW)  # fresh cup
        self.assertTrue(is_new)
        self.assertEqual(log.count(), 2)

    def test_continuation_extends_last_seen(self):
        log = EpisodeLog("g1")
        ep, _ = log.observe(1.0, "cup", ALWAYS_NEW)
        same, is_new = log.observe(5.0, "cup", ALWAYS_CONTINUE)
        self.assertFalse(is_new)
        self.assertEqual(same.id, ep.id)
        self.assertEqual(same.last_seen, 5.0)

    def test_window_bound_prunes_old_episodes(self):
        log = EpisodeLog("g1", window_s=10.0)
        log.observe(1.0, "cup", ALWAYS_NEW)
        # a long gap past the window -> the old episode falls out of "today's cups"
        self.assertEqual(log.count(now=100.0), 0)

    def test_max_episodes_bound(self):
        log = EpisodeLog("g1", max_episodes=2)
        for t in range(5):
            log.observe(float(t), f"cup{t}", ALWAYS_NEW)
        self.assertEqual(log.count(), 2)

    def test_max_episodes_drop_is_traced_with_reason(self):
        # ADR-001 §9 / the implementation: the episode buffer's count bound must never drop silently.
        dropped = []
        log = EpisodeLog("g1", max_episodes=2, on_drop=lambda ep, reason: dropped.append((ep.id, reason)))
        for t in range(4):
            log.observe(float(t), f"cup{t}", ALWAYS_NEW)
        self.assertEqual(log.count(), 2)
        self.assertEqual(dropped, [("g1:ep1", "max_episodes"), ("g1:ep2", "max_episodes")])

    def test_window_bound_drop_is_traced_with_reason(self):
        dropped = []
        log = EpisodeLog("g1", window_s=10.0, on_drop=lambda ep, reason: dropped.append((ep.id, reason)))
        log.observe(1.0, "cup", ALWAYS_NEW)
        self.assertEqual(log.count(now=100.0), 0)
        self.assertEqual(dropped, [("g1:ep1", "window_expired")])

    def test_default_drop_is_traced_via_log_not_silent(self):
        # No on_drop supplied: the default path must still observably log the drop.
        log = EpisodeLog("g1", max_episodes=1)
        with self.assertLogs("percept.episodes", level="INFO") as cm:
            log.observe(1.0, "cupA", ALWAYS_NEW)
            log.observe(2.0, "cupB", ALWAYS_NEW)
        self.assertTrue(any("episode.dropped" in line for line in cm.output))


class TestPerceptCountEvent(unittest.TestCase):
    def _agent(self):
        a = Percept.create(discover_plugins=False)
        a.add_goal(
            Goal(
                id="cups",
                condition="the user finished a cup of coffee",
                say="that's another cup",
                state={"type": "count", "count": 0},
            )
        )
        return a

    def test_count_event_updates_goal_state(self):
        a = self._agent()
        _, n1 = a.count_event("cups", "cupA", ALWAYS_NEW, t=1.0)
        a.count_event("cups", "cupA", ALWAYS_CONTINUE, t=2.0)  # same cup, no increment
        _, n3 = a.count_event("cups", "cupB", ALWAYS_NEW, t=3.0)  # new cup
        self.assertTrue(n1 and n3)
        self.assertEqual(a.goals.get("cups").state["count"], 2)
        self.assertEqual(a.metrics.get("episode.new"), 2)
        self.assertEqual(a.metrics.get("episode.continuation"), 1)


class _SpyGate(Gate):
    """Records every forced re-arm. ``gate.py`` stays mechanically unchanged.

    Deliberately copied rather than imported: test modules in this tree never import each other, so
    the same helper is duplicated in ``tests/test_y3_episode_idempotency.py``.
    """

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.rearms: list = []

    def rearm(self, key, **kw):
        self.rearms.append(key)
        super().rearm(key, **kw)


_TRUE = Verdict(satisfied=True, confidence=1.0, evidence="e")
_FALSE = Verdict(satisfied=False, confidence=1.0, evidence="e")
_KEY = ("g", "scene")


class TestNewEpisodeRearmVersusTheStartAnchor(unittest.TestCase):
    """``behavior.py:1373`` — ``fold.should_rearm`` and ``require_false_first`` are the one pair of
    settings that want opposite things from the same gate key, and nothing pinned the combination.

    ``require_false_first`` says a key may fire only after this deployment WATCHED the condition
    become false. It is enforced on a key the gate has never seen — which is exactly the key a first
    episode arrives on. A brand-new episode's fold asks for that key to be armed, because a fresh
    episode's first fire must not be suppressed by a disarm some other subject left behind. If the
    fold won, the start anchor would be deleted and the watch would report a condition that was
    already true when it started: the single thing the flag exists to refuse.
    """

    def _fixture(self, *, require_false_first):
        gate = _SpyGate(require_false_first=require_false_first)
        ex = BehaviorExecutor(gate)
        goal = Goal(id="g", condition="c", say="fired", state={"type": "watch"})
        prog = BehaviorProgram(sense="judge")
        log = EpisodeLog("g", policy=EpisodePolicy(continuity_gap_s=1000.0))
        return gate, ex, goal, prog, log

    @staticmethod
    def _step(ex, goal, prog, log, t, verdict):
        """Fold one observation of the same subject and run it through the fire path."""
        fold = ex.fold_episode(log.observe_decision(t, "s", subject_key="s"))
        reasons: list = []
        fired = ex.fire_judged(
            goal, verdict, _KEY, prog, t=t, fold=fold,
            outcome_sink=lambda outcome: reasons.append(outcome.reason),
        )
        return fold, fired, reasons

    def test_the_fold_really_does_ask_to_rearm_on_a_brand_new_episode(self):
        # The precondition for everything below: without it the guard would be untested by
        # construction rather than honoured.
        _, ex, _, _, log = self._fixture(require_false_first=True)
        fold = ex.fold_episode(log.observe_decision(0.0, "s", subject_key="s"))
        self.assertEqual(fold.match, "new")
        self.assertTrue(fold.should_rearm)
        self.assertTrue(fold.should_evaluate_gate)

    def test_start_anchored_refuses_the_rearm_and_the_anchor_survives(self):
        gate, ex, goal, prog, log = self._fixture(require_false_first=True)

        fold, fired, reasons = self._step(ex, goal, prog, log, 0.0, _TRUE)
        self.assertTrue(fold.should_rearm)          # the fold wants the edge armed...
        self.assertEqual(gate.rearms, [])           # ...and the guard refuses it outright
        self.assertIsNone(fired)                    # so the already-true start stays silent
        self.assertEqual(reasons, ["start_anchored"])
        # The anchor IS the disarmed record. If the re-arm had leaked through, the key would read
        # armed here and the first observation would have fired.
        self.assertIs(gate._armed[_KEY], False)

        # Now prove the anchor is not merely intact but still load-bearing: a confident-false is the
        # observed false->true baseline the flag demands, and the true after it is a real rising edge.
        _, silent, false_reasons = self._step(ex, goal, prog, log, 1.0, _FALSE)
        self.assertIsNone(silent)
        self.assertEqual(false_reasons, ["not_satisfied"])
        self.assertIs(gate._armed[_KEY], True)      # the gate armed it, on evidence

        _, later, true_reasons = self._step(ex, goal, prog, log, 2.0, _TRUE)
        self.assertIsNotNone(later)
        self.assertEqual(later.action, "fire")
        self.assertEqual(true_reasons, ["rising_edge"])
        self.assertEqual(gate.rearms, [])           # never forced, start to finish

    def test_without_the_guard_the_new_episode_rearm_deletes_the_anchor(self):
        # The counter-proof that the guard is load-bearing, mirroring the "without the fold" idiom in
        # tests/test_y3_episode_idempotency.py. Same flag, same fold, same verdict — but the re-arm is
        # applied by hand, as an unguarded ``should_rearm`` would. The condition that was ALREADY true
        # when the watch started is now reported as a rising edge: the exact user-visible defect the
        # flag exists to refuse, and the reason the guard cannot be relaxed to "re-arm anyway".
        gate, ex, goal, prog, log = self._fixture(require_false_first=True)
        fold = ex.fold_episode(log.observe_decision(0.0, "s", subject_key="s"))
        gate.rearm(_KEY)
        reasons: list = []
        fired = ex.fire_judged(
            goal, _TRUE, _KEY, prog, t=0.0, fold=fold,
            outcome_sink=lambda outcome: reasons.append(outcome.reason),
        )
        self.assertIsNotNone(fired)
        self.assertEqual(fired.action, "fire")
        self.assertEqual(reasons, ["rising_edge"])  # not start_anchored — the anchor is gone

    def test_the_identical_observation_rearms_and_fires_with_the_flag_off(self):
        # Same fold, same verdict, same key — only the flag differs. This is what makes the test
        # above a guard and not an accident of the fixture.
        gate, ex, goal, prog, log = self._fixture(require_false_first=False)
        fold, fired, reasons = self._step(ex, goal, prog, log, 0.0, _TRUE)
        self.assertTrue(fold.should_rearm)
        self.assertEqual(gate.rearms, [_KEY])       # exactly once
        self.assertIsNotNone(fired)
        self.assertEqual(fired.action, "fire")
        self.assertEqual(reasons, ["rising_edge"])


class TestIncidentLedgerGrowthIsUnboundedByDecision(unittest.TestCase):
    """``behavior.py:626-652`` — ``EpisodeIncidentLedger._emitted`` is a plain dict with no cap, no
    eviction and no sweep. This test is the explicit recorded decision that it stays that way.

    Growth is linear and unbounded: one small integer entry per DISTINCT episode id, retained for the
    life of the executor. A watch running continuously that opens a million distinct episodes holds a
    million entries. That is accepted rather than fixed, because the ledger is the only thing that
    stops a gate-key TTL sweep from manufacturing a duplicate incident — evicting from it would
    reintroduce exactly the duplicate it exists to prevent, and an eviction would be silent and
    unrecoverable in a way a gate-key eviction is not. Re-confirming an existing episode costs
    nothing, so the bound to watch is distinct episodes, not runtime.
    """

    def test_five_thousand_distinct_episodes_are_all_retained(self):
        ledger = EpisodeIncidentLedger()
        ids = [f"g:ep{n}" for n in range(1, 5001)]
        for episode_id in ids:
            ledger.record(episode_id)
        ledger.record(ids[0])  # a second incident on the first episode — same entry, higher revision
        ledger.record(ids[0])

        self.assertEqual(len(ledger._emitted), 5000)  # nothing evicted, nothing coalesced
        self.assertEqual(ledger.emitted(ids[0]), 3)
        self.assertEqual(ledger.revision(ids[0]), "g:ep1#r3")
        self.assertEqual(ledger.emitted(ids[-1]), 1)
        self.assertEqual(ledger.revision(ids[-1]), "g:ep5000#r1")
        # The oldest entry is as readable as the newest — there is no recency window.
        self.assertEqual(ledger.emitted(ids[1]), 1)

    def test_the_ledger_exposes_no_bound_to_configure_or_observe(self):
        # A drift guard on the decision above. If a cap, a sweep or a drop hook ever appears, the
        # decision has been reversed and this test is where that gets re-argued — including whether
        # the drop is traced, which every other bounded buffer in this module already does.
        ledger = EpisodeIncidentLedger()
        for name in ("max_episodes", "max_entries", "evict", "sweep", "prune", "on_drop", "clear"):
            self.assertFalse(
                hasattr(ledger, name),
                f"the ledger grew {name!r} — growth is no longer unbounded, revisit this test",
            )


if __name__ == "__main__":
    unittest.main()
