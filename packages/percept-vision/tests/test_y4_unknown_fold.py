"""— the OPERATIONAL semantics of an ``'unknown'`` subject inside the episode fold.

The merged subject contract already settled the TYPE and episode-log default: ``_subject_key_of`` returns ``None``
for an ``'unknown'`` ref, so the log MINTS a fresh episode per unknown and two unknowns NEVER collapse
(they are NaN-like). That is re-pinned here, not re-litigated. This suite adds only the two open rules:

  1. The FOLD's gate governance — a visible abstention must never REARM the gate (no wrong suppression)
     and is never granted episode-owned idempotency, EVEN THOUGH the subject resolver minted an episode for it.
  2. The OPT-IN single-focus provisional attach — a narrowly-scoped source MAY continue the sole open
     bound episode with an unknown sighting, RECORDED (never a silent merge).

These are stdlib-only, deterministic, no backend. They assert four explicit design questions:
mint / rearm / continue-a-bound-episode / merge-on-later-bind.
"""

from __future__ import annotations

import unittest

from percept.adapters import Verdict
from percept.behavior import (
    EPISODE_MATCHES,
    BehaviorExecutor,
    episode_fold,
)
from percept.contracts import SubjectRef
from percept.episodes import (
    CONTINUATION_REASONS,
    EpisodeLog,
    EpisodePolicy,
    subject_is_unknown,
)
from percept.gate import Gate


def _bound(key: str) -> SubjectRef:
    return SubjectRef(
        scheme="content", namespace="acme-erp", key=key, resolution="bound",
        scope="durable", strategy="judge",
    )


def _unknown() -> SubjectRef:
    return SubjectRef(
        scheme="content", namespace="acme-erp", key=None, resolution="unknown",
        scope="session", strategy="judge",
    )


class SubjectIsUnknownTests(unittest.TestCase):
    def test_discriminates_the_three_states(self) -> None:
        self.assertTrue(subject_is_unknown(Verdict(True, 0.4, subject=_unknown())))
        self.assertFalse(subject_is_unknown(Verdict(True, 0.9, subject=_bound("INV-1"))))
        # absence of a subject is NOT unknown — it is a different debugging fact
        self.assertFalse(subject_is_unknown(Verdict(True, 0.9)))
        # an exemplar with no subject attribute at all is byte-for-byte unchanged (not unknown)
        self.assertFalse(subject_is_unknown(object()))


class VocabularyTests(unittest.TestCase):
    def test_unknown_is_a_first_class_match(self) -> None:
        self.assertIn("unknown", EPISODE_MATCHES)

    def test_provisional_continuation_reason_is_registered(self) -> None:
        self.assertIn("unknown_provisional_continuation", CONTINUATION_REASONS)


class FoldGateGovernanceTests(unittest.TestCase):
    """Question 2: does an unknown REARM the gate? No; that is this suite's core invariant."""

    def _log(self, gap: float = 30.0) -> EpisodeLog:
        return EpisodeLog("g", policy=EpisodePolicy(continuity_gap_s=gap))

    def test_unknown_that_minted_a_new_episode_does_not_rearm(self) -> None:
        # Y5 mints an episode for an unknown (is_new=True). Without Y4 the fold would read that as
        # ``new`` and REARM. Y4 forces ``unknown`` + no rearm.
        log = self._log()
        d = log.observe_decision(0.0, Verdict(True, 0.4, subject=_unknown()))
        self.assertTrue(d.is_new)  # The subject resolver minted a fresh episode.
        f = episode_fold(d, emitted=0, unknown=True)
        self.assertEqual(f.match, "unknown")
        self.assertEqual(f.mutation, "none")
        self.assertFalse(f.should_rearm)          # a visible abstention never rearms
        self.assertTrue(f.should_evaluate_gate)   # gate-only, like unbound
        self.assertIsNone(f.supersedes)

    def test_unknown_with_no_decision_is_unknown_not_unbound(self) -> None:
        f = episode_fold(None, unknown=True)
        self.assertEqual(f.match, "unknown")      # distinct from "unbound" (no subject at all)
        self.assertIsNone(f.episode_id)
        self.assertFalse(f.should_rearm)
        self.assertTrue(f.should_evaluate_gate)

    def test_bound_new_still_rearms_only_the_unknown_case_changed(self) -> None:
        # Contrast: a BOUND new episode rearms exactly as before; only the unknown path changed.
        log = self._log()
        d = log.observe_decision(0.0, Verdict(True, 0.9, subject=_bound("INV-1")))
        f = episode_fold(d, emitted=0)  # unknown defaults False
        self.assertEqual(f.match, "new")
        self.assertTrue(f.should_rearm)

    def test_unknown_never_granted_episode_idempotency(self) -> None:
        # Even if a prior unknown "emitted", the fold does not treat a later unknown as an idempotent
        # continuation of it — unknowns are NaN-like, never the same episode. It stays gate-only.
        log = self._log()
        d = log.observe_decision(1.0, Verdict(True, 0.4, subject=_unknown()))
        f = episode_fold(d, emitted=5, unknown=True)  # emitted is ignored on the unknown path
        self.assertEqual(f.match, "unknown")
        self.assertTrue(f.should_evaluate_gate)  # NOT suppressed by a bogus idempotency attribution
        self.assertFalse(f.should_rearm)


class MintAndCollapseTests(unittest.TestCase):
    """Question 1: mint or attach to nothing. Re-pins the existing default."""

    def test_two_unknowns_never_collapse_default(self) -> None:
        log = EpisodeLog("g", policy=EpisodePolicy(continuity_gap_s=30.0))
        d0 = log.observe_decision(0.0, Verdict(True, 0.4, subject=_unknown()))
        d1 = log.observe_decision(1.0, Verdict(True, 0.4, subject=_unknown()))
        self.assertEqual(log.count(), 2)                 # Y5: never fabricate shared identity
        self.assertNotEqual(d0.episode.id, d1.episode.id)
        self.assertEqual(d0.reason, "new_subject")
        # and both fold to gate-only, neither rearms
        for d in (d0, d1):
            f = episode_fold(d, unknown=True)
            self.assertEqual(f.match, "unknown")
            self.assertFalse(f.should_rearm)


class LateBindMergeTests(unittest.TestCase):
    """Question 4: unknown then a later BIND — merge, or already minted two? Answer: two, no silent merge."""

    def test_unknown_then_bound_mints_two_no_silent_merge(self) -> None:
        log = EpisodeLog("g", policy=EpisodePolicy(continuity_gap_s=30.0))
        d_unknown = log.observe_decision(0.0, Verdict(True, 0.4, subject=_unknown()))
        d_bound = log.observe_decision(1.0, Verdict(True, 0.9, subject=_bound("INV-1")))
        self.assertTrue(d_bound.is_new)                       # opened its own episode
        self.assertFalse(d_bound.subject_matched)             # did NOT silently adopt the unknown
        self.assertNotEqual(d_unknown.episode.id, d_bound.episode.id)
        self.assertEqual(log.count(), 2)


class ProvisionalSingleFocusAttachTests(unittest.TestCase):
    """Question 3: may an unknown CONTINUE the single active bound episode? Only under the opt-in."""

    def _log(self, *, single_focus: bool, gap: float = 30.0) -> EpisodeLog:
        return EpisodeLog(
            "g",
            policy=EpisodePolicy(
                continuity_gap_s=gap, unknown_attaches_to_active_bound=single_focus
            ),
        )

    def test_default_off_unknown_does_not_attach(self) -> None:
        log = self._log(single_focus=False)
        log.observe_decision(0.0, Verdict(True, 0.9, subject=_bound("INV-1")))
        d = log.observe_decision(5.0, Verdict(True, 0.4, subject=_unknown()))
        self.assertTrue(d.is_new)                 # minted its own, did NOT ride the bound episode
        self.assertEqual(log.count(), 2)

    def test_optin_unknown_continues_sole_open_bound_episode(self) -> None:
        log = self._log(single_focus=True)
        d_bound = log.observe_decision(0.0, Verdict(True, 0.9, subject=_bound("INV-1")))
        d = log.observe_decision(5.0, Verdict(True, 0.4, subject=_unknown()))  # inside the gap
        self.assertFalse(d.is_new)                # rode the open bound episode
        self.assertEqual(d.reason, "unknown_provisional_continuation")
        self.assertTrue(d.subject_matched)
        self.assertEqual(d.episode.id, d_bound.episode.id)
        self.assertEqual(log.count(), 1)          # no duplicate minted

    def test_optin_attach_is_recorded_not_silent(self) -> None:
        # The provisional attach must leave a trace — a recorded revision on the episode.
        log = self._log(single_focus=True)
        d_bound = log.observe_decision(0.0, Verdict(True, 0.9, subject=_bound("INV-1")))
        n_before = len(d_bound.episode.revisions)
        log.observe_decision(5.0, Verdict(True, 0.4, subject=_unknown()))
        # confirmed coalesces the timeline but advances last_confirmed — the attach is observable.
        self.assertEqual(d_bound.episode.observations, 2)
        self.assertEqual(d_bound.episode.last_confirmed, 5.0)
        self.assertGreaterEqual(len(d_bound.episode.revisions), n_before)

    def test_optin_provisional_continuation_folds_as_continuation_not_unknown(self) -> None:
        # A provisional continuation is a REAL continuation — the fold must treat it as one (idempotent
        # once emitted), NOT as the gate-only unknown path.
        log = self._log(single_focus=True)
        log.observe_decision(0.0, Verdict(True, 0.9, subject=_bound("INV-1")))
        d = log.observe_decision(5.0, Verdict(True, 0.4, subject=_unknown()))
        f = episode_fold(d, emitted=1, unknown=True)
        self.assertEqual(f.match, "continuation")
        self.assertFalse(f.should_evaluate_gate)  # already emitted + unchanged -> idempotent
        self.assertFalse(f.should_rearm)

    def test_optin_boundary_gap_exceeded_does_not_attach(self) -> None:
        # No boundary event is required for the attach; a gap past continuity IS a boundary.
        log = self._log(single_focus=True, gap=10.0)
        log.observe_decision(0.0, Verdict(True, 0.9, subject=_bound("INV-1")))
        d = log.observe_decision(45.0, Verdict(True, 0.4, subject=_unknown()))  # past the 10s gap
        self.assertTrue(d.is_new)                 # boundary -> stays unbound, mints new
        self.assertEqual(d.reason, "new_subject")
        self.assertEqual(log.count(), 2)

    def test_optin_two_open_bound_episodes_refuses_to_guess(self) -> None:
        # With two things in focus an unknown could belong to either -> it must NOT attach to one.
        log = self._log(single_focus=True)
        log.observe_decision(0.0, Verdict(True, 0.9, subject=_bound("INV-1")))
        log.observe_decision(1.0, Verdict(True, 0.9, subject=_bound("INV-2")))
        d = log.observe_decision(2.0, Verdict(True, 0.4, subject=_unknown()))
        self.assertTrue(d.is_new)                 # refused to guess -> unbound, minted new
        self.assertEqual(log.count(), 3)

    def test_optin_bound_returns_after_provisional_attach_no_duplicate(self) -> None:
        # The payoff: an unknown rides the bound episode, then the SAME bound subject returns and
        # continues that very episode. One episode throughout -> no duplicate, no merge needed.
        log = self._log(single_focus=True)
        d0 = log.observe_decision(0.0, Verdict(True, 0.9, subject=_bound("INV-1")))
        log.observe_decision(5.0, Verdict(True, 0.4, subject=_unknown()))
        d2 = log.observe_decision(10.0, Verdict(True, 0.9, subject=_bound("INV-1")))
        self.assertFalse(d2.is_new)
        self.assertEqual(d2.episode.id, d0.episode.id)
        self.assertEqual(log.count(), 1)


class FoldGateIntegrationTests(unittest.TestCase):
    """End to end: an unknown reaching the gate via the fold does not manufacture a duplicate fire."""

    def _executor(self):
        gate = Gate(threshold=0.7, fresh_min=0.5)
        return BehaviorExecutor(gate), gate

    def test_unknown_after_a_bound_fire_does_not_refire_shared_key(self) -> None:
        # A bound subject fires and disarms the shared (goal, entity) key. An unknown on the SAME key
        # then folds ``unknown`` -> should_rearm=False, so it consults the disarmed gate and stays
        # SILENT. If the unknown had rearmed (the legacy ``new`` reading) it would refire — the exact
        # wrong suppression the unknown-fold contract forbids.
        from percept.adapters import Goal
        from percept.behavior import BehaviorProgram, program_of

        ex, _gate = self._executor()
        goal = Goal(id="g", condition="c", say="fired", state={"type": "watch"})
        program: BehaviorProgram = program_of(goal)
        key = (goal.id, "scene")

        log = EpisodeLog("g", policy=EpisodePolicy(continuity_gap_s=30.0))
        # bound fire
        v_bound = Verdict(True, 0.95, subject=_bound("INV-1"))
        d_bound = log.observe_decision(0.0, v_bound)
        fold_bound = ex.fold_episode(d_bound)
        fired = ex.fire_judged(goal, v_bound, key, program, t=0.0, fold=fold_bound)
        self.assertIsNotNone(fired)               # the bound subject fired (disarmed the key)

        # unknown on the same key
        v_unknown = Verdict(True, 0.95, subject=_unknown())
        d_unknown = log.observe_decision(5.0, v_unknown)
        fold_unknown = ex.fold_episode(d_unknown, unknown=subject_is_unknown(v_unknown))
        self.assertEqual(fold_unknown.match, "unknown")
        self.assertFalse(fold_unknown.should_rearm)
        again = ex.fire_judged(goal, v_unknown, key, program, t=5.0, fold=fold_unknown)
        self.assertIsNone(again)                  # <- no duplicate fire: the unknown did not rearm


if __name__ == "__main__":
    unittest.main()
