"""— the honest gap left by C2 (``test_c2_profile_wiring.py``): C2 wired a resolved
SourceProfile into ADMISSION (``Settings.edge.*``) end-to-end through the compile path, but the
IDENTITY/EPISODE seams ``percept.identity_policy.select_identity_policy`` and
``SourceProfile.default_episode_policy`` stayed built, tested in isolation, and never actually
threaded into ``percept.agent.Percept`` — a concern on a profiled core that authored no
``identity_policy`` / ``episode`` block got exactly the same profile-blind defaults as a bare,
profile-less core.

This closes it: ``Percept.source_profile`` (populated by ``create()``/``configure()`` and by
``percept.watchplan_apply.apply_watch_plan``) now backs a FALLBACK underneath the three existing
call sites' own defaults — ``count_event_decision``, ``Percept._document_episode_fold``,
``Percept._claim_episode_fold`` (episode policy) and the identity gate the latter two run
(``Percept._resolved_identity_policy``, which powers their ``content_key`` / ``claim_slot`` checks).
The precedence, proven below:

1. explicit ``goal.state`` authoring ALWAYS wins (never touched by this card);
2. absent that, a RESOLVED ``source_profile`` supplies its own default (identity_strategy /
   default_episode_policy) — this is the new behavior;
3. absent BOTH, the pre-C2F byte-for-byte default (the hardcoded ``EPISODE_POLICIES[...]`` per call
   site; ``identity_policy_of`` returning ``None`` and gate-only ``fold=None``).
"""

from __future__ import annotations

import unittest

from percept.adapters import Goal, Verdict
from percept.agent import Percept
from percept.backends.fake import FakeVision
from percept.config import GateSettings, Settings
from percept.contracts import SubjectRef
from percept.episodes import EPISODE_POLICIES, EpisodePolicy
from percept.source_profiles_default import (
    SCREEN_DOCUMENT,
    SCREEN_UNKNOWN_CONTINUITY_GAP_S,
)


def _content_ref(key: "str | None", *, resolution: str = "bound") -> SubjectRef:
    return SubjectRef(
        scheme="content", namespace="workflow", key=key, resolution=resolution,
        scope="durable", strategy="test", confidence=0.9,
    )


def _build(*, source_profile=None, state: "dict | None" = None) -> Percept:
    settings = Settings(gate=GateSettings(threshold=0.7, validity_window_s=1000.0))
    a = Percept.create(
        vision=FakeVision(), settings=settings, discover_plugins=False, source_profile=source_profile,
    )
    a.add_goal(Goal(
        id="g1", condition="a document is on screen", say="doc seen",
        state=state if state is not None else {"type": "watch", "count": 0},
    ))
    return a


# ----------------------------------------------------------------------------------------------
# 1. screen_document's default episode policy (incl. the Y17 unknown-continuity gap) flows into a
#    document-mode goal that declares nothing on identity_policy / episode.
# ----------------------------------------------------------------------------------------------


class TestProfileDefaultFlowsIntoAnUndeclaredGoal(unittest.TestCase):
    def test_undeclared_goal_now_resolves_content_key_from_the_profile(self):
        # Pre-C2F this goal (no identity_policy authored) would gate-only forever, on ANY profile.
        a = _build(source_profile=SCREEN_DOCUMENT)
        goal = a.goals.get("g1")
        self.assertEqual(a._resolved_identity_policy(goal), "content_key")

    def test_bound_subject_now_governs_and_carries_the_profiles_episode_policy(self):
        a = _build(source_profile=SCREEN_DOCUMENT)
        goal = a.goals.get("g1")
        verdict = Verdict(True, 0.9, "an invoice", subject=_content_ref("INV-1"))
        fold = a._document_episode_fold(goal, verdict, 0.0)
        self.assertIsNotNone(fold, "an undeclared goal on a profiled core must now govern")
        log = a.episodes["g1"]
        # NOT the hardcoded document_presence default (continuity_gap_s=120.0) -- the resolved
        # profile's OWN default_episode_policy replaced it entirely.
        self.assertNotEqual(log.policy, EPISODE_POLICIES["document_presence"])
        self.assertEqual(log.policy, SCREEN_DOCUMENT.default_episode_policy)
        self.assertIsNone(log.policy.continuity_gap_s)
        self.assertEqual(log.policy.unknown_continuity_gap_s, SCREEN_UNKNOWN_CONTINUITY_GAP_S)

    def test_unknown_continuity_gap_actually_governs_two_illegible_sightings(self):
        # The behavioral proof, not just a policy-value assertion: two UNKNOWN (unreadable content
        # key) sightings a coast-tick apart fold into the SAME provisional episode instead of
        # minting a rival -- exactly the Y17 capability SCREEN_DOCUMENT declares.
        a = _build(source_profile=SCREEN_DOCUMENT)
        goal = a.goals.get("g1")
        v_unknown = Verdict(True, 0.4, "an illegible document", subject=_content_ref(None, resolution="unknown"))
        first = a._document_episode_fold(goal, v_unknown, 0.0)
        second = a._document_episode_fold(goal, v_unknown, SCREEN_UNKNOWN_CONTINUITY_GAP_S - 1.0)
        self.assertIsNotNone(first)
        self.assertIsNotNone(second)
        log = a.episodes["g1"]
        self.assertEqual(log.count(), 1, "the two illegible sightings must collapse into ONE episode")


# ----------------------------------------------------------------------------------------------
# 2. explicit goal.state authoring still wins over a resolved profile.
# ----------------------------------------------------------------------------------------------


class TestExplicitGoalStateOverridesTheProfile(unittest.TestCase):
    def test_explicit_identity_policy_beats_the_profiles_default(self):
        # SCREEN_DOCUMENT declares identity_strategy="content_key"; this concern explicitly opts
        # into claim_slot instead -- the concern's own choice must win.
        a = _build(
            source_profile=SCREEN_DOCUMENT,
            state={"type": "watch", "count": 0, "identity_policy": "claim_slot"},
        )
        goal = a.goals.get("g1")
        self.assertEqual(a._resolved_identity_policy(goal), "claim_slot")
        verdict = Verdict(True, 0.9, "asserted", subject=_content_ref("slot-1"))
        # content_key-scoped fold must NOT govern this concern despite the profile default.
        self.assertIsNone(a._document_episode_fold(goal, verdict, 0.0))
        # the claim_slot-scoped fold DOES govern it.
        self.assertIsNotNone(a._claim_episode_fold(goal, verdict, 0.0))

    def test_explicit_episode_block_beats_the_profiles_default_episode_policy(self):
        a = _build(
            source_profile=SCREEN_DOCUMENT,
            state={"type": "watch", "count": 0, "episode": {"continuity_gap_s": 42.0}},
        )
        goal = a.goals.get("g1")
        verdict = Verdict(True, 0.9, "an invoice", subject=_content_ref("INV-1"))
        a._document_episode_fold(goal, verdict, 0.0)
        log = a.episodes["g1"]
        self.assertEqual(log.policy.continuity_gap_s, 42.0, "the authored value must win")
        self.assertNotEqual(log.policy.continuity_gap_s, SCREEN_DOCUMENT.default_episode_policy.continuity_gap_s)


# ----------------------------------------------------------------------------------------------
# 3. no resolved profile = today's behavior, byte-for-byte (equivalence assertion).
# ----------------------------------------------------------------------------------------------


class TestNoProfileIsByteForByte(unittest.TestCase):
    def test_percept_source_profile_defaults_to_none(self):
        a = _build(source_profile=None)
        self.assertIsNone(a.source_profile)

    def test_undeclared_goal_still_never_governs_without_a_profile(self):
        a = _build(source_profile=None)
        goal = a.goals.get("g1")
        self.assertIsNone(a._resolved_identity_policy(goal))
        verdict = Verdict(True, 0.9, "an invoice", subject=_content_ref("INV-1"))
        self.assertIsNone(a._document_episode_fold(goal, verdict, 0.0))
        self.assertIsNone(a._claim_episode_fold(goal, verdict, 0.0))
        self.assertNotIn("g1", a.episodes)

    def test_declared_content_key_still_gets_the_hardcoded_document_presence_default(self):
        a = _build(
            source_profile=None,
            state={"type": "watch", "count": 0, "identity_policy": "content_key"},
        )
        goal = a.goals.get("g1")
        verdict = Verdict(True, 0.9, "an invoice", subject=_content_ref("INV-1"))
        a._document_episode_fold(goal, verdict, 0.0)
        log = a.episodes["g1"]
        self.assertEqual(log.policy, EPISODE_POLICIES["document_presence"])

    def test_count_event_decision_default_policy_unchanged_without_a_profile(self):
        a = _build(source_profile=None, state={"type": "count", "count": 0})
        a.count_event_decision("g1", "cup", t=0.0)
        log = a.episodes["g1"]
        self.assertEqual(log.policy, EpisodePolicy())


if __name__ == "__main__":
    unittest.main()
