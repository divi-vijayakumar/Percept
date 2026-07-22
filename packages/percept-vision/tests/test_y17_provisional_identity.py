"""Provisional UNKNOWN identity (``UN_<episode_id>``) + reconciled distinct-subject counting.

Covers the seam that widens :class:`~percept.contracts.SubjectRef` to a THREE-state resolution with a
``'provisional'`` state, the matching episode-log accounting (an unidentified sighting is stamped
``UN_<episode_id>`` instead of vanishing into ``subject_key=None``), the per-``SourceProfile``
unknown-continuity default (``screen_document`` folds a close-in-time run of illegible sightings into
one provisional subject; other shipped profiles do not), and the ``IdentityLedger`` oracle now using
the SAME accounting (bound subjects + unknown episodes) the runtime does — plus its opt-in,
never-silent cap-and-evict, on top of keeping everything by default.

Stdlib only — no backends, no keys. These assert the CONTRACT INVARIANTS and the episode-log/oracle
mechanics, not a model's ability to read a key off a real screen (a benchmark question, not this one).
"""

from __future__ import annotations

import unittest

from percept.contracts import PROVISIONAL_KEY_PREFIX, SubjectRef, provisional_subject
from percept.episodes import EpisodeLog, EpisodePolicy, UNKNOWN_KEY_PREFIX
from percept.eval.identity_ledger import Binding, IdentityLedger, Sighting
from percept.source_profiles_default import CONVERSATION, PHYSICAL_SCENE, SCREEN_DOCUMENT


def _unknown(*, scheme: str = "content", ns: str = "acme-erp") -> SubjectRef:
    return SubjectRef(scheme=scheme, namespace=ns, key=None, resolution="unknown",
                       scope="session", strategy="judge")


class ProvisionalSubjectRefTests(unittest.TestCase):
    def test_prefix_constants_agree(self) -> None:
        # contracts.PROVISIONAL_KEY_PREFIX and episodes.UNKNOWN_KEY_PREFIX must be the same string —
        # episodes.py deliberately duck-types instead of importing contracts, so this pins them equal.
        self.assertEqual(PROVISIONAL_KEY_PREFIX, "UN_")
        self.assertEqual(UNKNOWN_KEY_PREFIX, PROVISIONAL_KEY_PREFIX)

    def test_provisional_subject_builds_a_valid_ref(self) -> None:
        ref = provisional_subject("g1:ep1")
        self.assertEqual(ref.resolution, "provisional")
        self.assertEqual(ref.key, "UN_g1:ep1")
        self.assertEqual(ref.scope, "session")

    def test_provisional_requires_un_prefixed_key(self) -> None:
        with self.assertRaises(ValueError):
            SubjectRef("episode", "", "not-prefixed", "provisional", "session", "episode_scoped")
        with self.assertRaises(ValueError):
            SubjectRef("episode", "", None, "provisional", "session", "episode_scoped")

    def test_provisional_requires_session_scope(self) -> None:
        with self.assertRaises(ValueError):
            SubjectRef("episode", "", "UN_g1:ep1", "provisional", "durable", "episode_scoped")

    def test_provisional_equal_within_same_episode_id(self) -> None:
        a = provisional_subject("g1:ep1")
        b = provisional_subject("g1:ep1")
        self.assertEqual(a, b)
        self.assertEqual(hash(a), hash(b))

    def test_provisional_differs_across_episodes(self) -> None:
        self.assertNotEqual(provisional_subject("g1:ep1"), provisional_subject("g1:ep2"))

    def test_provisional_never_equals_bound_even_sharing_key_string(self) -> None:
        bound = SubjectRef("episode", "acme", "UN_g1:ep1", "bound", "durable", "client")
        prov = provisional_subject("g1:ep1")
        self.assertNotEqual(bound, prov)

    def test_provisional_never_equals_unknown(self) -> None:
        self.assertNotEqual(provisional_subject("g1:ep1"), _unknown())

    def test_unknown_still_nan_like(self) -> None:
        a = _unknown()
        self.assertFalse(a == a)


class EpisodeLogProvisionalIdentityTests(unittest.TestCase):
    def test_unknown_sighting_mints_un_prefixed_subject_key(self) -> None:
        log = EpisodeLog("g1")

        class V:
            subject = _unknown()

        decision = log.observe_decision(0.0, V())
        self.assertTrue(decision.is_new)
        self.assertEqual(decision.episode.subject_key, f"UN_{decision.episode.id}")

    def test_two_unrelated_unknowns_still_never_collapse(self) -> None:
        log = EpisodeLog("g1")

        class V:
            subject = _unknown()

        d1 = log.observe_decision(0.0, V())
        d2 = log.observe_decision(1.0, V())
        self.assertEqual(log.count(), 2)
        self.assertNotEqual(d1.episode.subject_key, d2.episode.subject_key)

    def test_unknown_continuity_gap_folds_close_sightings(self) -> None:
        pol = EpisodePolicy(unknown_continuity_gap_s=5.0)
        log = EpisodeLog("g1", policy=pol)

        class V:
            subject = _unknown()

        d1 = log.observe_decision(0.0, V())
        d2 = log.observe_decision(3.0, V())  # inside the gap -> continues d1's episode
        d3 = log.observe_decision(30.0, V())  # well outside -> a fresh provisional episode
        self.assertFalse(d2.is_new)
        self.assertEqual(d2.reason, "unknown_temporal_continuation")
        self.assertEqual(d1.episode.id, d2.episode.id)
        self.assertTrue(d3.is_new)
        self.assertNotEqual(d3.episode.id, d1.episode.id)
        self.assertEqual(log.count(), 2)

    def test_unknown_continuity_does_not_apply_beyond_the_gap(self) -> None:
        pol = EpisodePolicy(unknown_continuity_gap_s=2.0)
        log = EpisodeLog("g1", policy=pol)

        class V:
            subject = _unknown()

        log.observe_decision(0.0, V())
        d2 = log.observe_decision(10.0, V())
        self.assertTrue(d2.is_new)
        self.assertEqual(log.count(), 2)

    def test_single_focus_bound_attach_unaffected_by_provisional_episodes(self) -> None:
        # A provisional (UN_-prefixed) episode must never be mistaken for "the one bound thing in
        # focus" by the Y4 single-focus attach.
        pol = EpisodePolicy(unknown_attaches_to_active_bound=True)
        log = EpisodeLog("g1", policy=pol)

        class Unk:
            subject = _unknown()

        # Two open provisional episodes (never collapse) -- neither is a bound episode, so a THIRD
        # unknown sighting must still mint its own fresh episode, not attach to either.
        log.observe_decision(0.0, Unk())
        log.observe_decision(100.0, Unk())
        d3 = log.observe_decision(200.0, Unk())
        self.assertTrue(d3.is_new)
        self.assertEqual(d3.reason, "new_subject")
        self.assertEqual(log.count(), 3)

    def test_provisional_subject_key_round_trips_through_subject_match(self) -> None:
        # A caller that hands back the SAME provisional key later continues that episode, same as a
        # bound key would -- Episode.provisional identity behaves like any other subject_key match.
        log = EpisodeLog("g1")

        class Unk:
            subject = _unknown()

        d1 = log.observe_decision(0.0, Unk())
        key = d1.episode.subject_key
        self.assertTrue(key.startswith("UN_"))

        class Prov:
            subject = provisional_subject(d1.episode.id)

        d2 = log.observe_decision(1.0, Prov())
        self.assertFalse(d2.is_new)
        self.assertEqual(d2.episode.id, d1.episode.id)


class SourceProfileUnknownContinuityTests(unittest.TestCase):
    def test_screen_document_defaults_to_temporal_proximity_continuity(self) -> None:
        pol = SCREEN_DOCUMENT.default_episode_policy
        self.assertIsInstance(pol, EpisodePolicy)
        self.assertIsNotNone(pol.unknown_continuity_gap_s)
        self.assertGreater(pol.unknown_continuity_gap_s, 0)

    def test_other_shipped_profiles_default_off(self) -> None:
        # Byte-for-byte the Y5 default for sources where "probably the same unidentified thing" is
        # not a defensible read (a physical scene can hold many distinct fast-moving unknowns).
        self.assertIsNone(PHYSICAL_SCENE.default_episode_policy.unknown_continuity_gap_s)
        self.assertIsNone(CONVERSATION.default_episode_policy.unknown_continuity_gap_s)

    def test_default_episode_policy_is_a_capability_not_a_mandate(self) -> None:
        # A concern is free to ignore the profile default entirely (episode_policy_of with its own
        # default), matching identity_policy's "profile declares, concern selects" shape.
        from percept.episodes import episode_policy_of

        class Goal:
            state: dict = {}

        got = episode_policy_of(Goal(), default=SCREEN_DOCUMENT.default_episode_policy)
        self.assertEqual(got.unknown_continuity_gap_s, SCREEN_DOCUMENT.default_episode_policy.unknown_continuity_gap_s)
        # No default supplied at all -> the ordinary EpisodePolicy() floor, unaffected by the profile.
        self.assertIsNone(episode_policy_of(Goal()).unknown_continuity_gap_s)


class IdentityLedgerReconciledCountTests(unittest.TestCase):
    def test_lone_unknown_sighting_counts_as_one_subject(self) -> None:
        # The exact fixture case this reconciles: subject_identity/unreadable_key_binds_unknown
        # expects distinct_subject_count == 1 for a single illegible sighting.
        ledger = IdentityLedger()
        ledger.observe(Sighting(t=0.0, subject_key=None, binding=Binding.UNKNOWN))
        self.assertEqual(ledger.distinct_subject_count, 1)
        self.assertEqual(ledger.unknown_count, 1)
        self.assertEqual(ledger.unknown_episode_count, 1)

    def test_bound_and_unknown_both_contribute(self) -> None:
        ledger = IdentityLedger()
        ledger.observe(Sighting(t=0.0, subject_key="INV-1"))
        ledger.observe(Sighting(t=1.0, subject_key=None, binding=Binding.UNKNOWN))
        ledger.observe(Sighting(t=2.0, subject_key=None, binding=Binding.UNKNOWN))
        self.assertEqual(ledger.distinct_subject_count, 3)  # 1 bound + 2 unrelated unknowns

    def test_unknown_continuity_gap_folds_a_close_run(self) -> None:
        ledger = IdentityLedger(unknown_continuity_gap_s=5.0)
        ledger.observe(Sighting(t=0.0, subject_key=None, binding=Binding.UNKNOWN))
        ledger.observe(Sighting(t=2.0, subject_key=None, binding=Binding.UNKNOWN))  # folds in
        ledger.observe(Sighting(t=20.0, subject_key=None, binding=Binding.UNKNOWN))  # new episode
        self.assertEqual(ledger.unknown_count, 3)
        self.assertEqual(ledger.unknown_episode_count, 2)
        self.assertEqual(ledger.distinct_subject_count, 2)


class IdentityLedgerEvictionTests(unittest.TestCase):
    def test_keeps_all_by_default(self) -> None:
        ledger = IdentityLedger()
        for i, key in enumerate(["A", "B", "C", "D", "E"]):
            ledger.observe(Sighting(t=float(i), subject_key=key))
        self.assertEqual(ledger.distinct_subject_count, 5)
        self.assertEqual(ledger.eviction_count, 0)
        self.assertEqual(ledger.evictions, [])

    def test_opt_in_cap_evicts_least_recently_seen(self) -> None:
        ledger = IdentityLedger(max_subjects=2)
        ledger.observe(Sighting(t=0.0, subject_key="A"))
        ledger.observe(Sighting(t=1.0, subject_key="B"))
        ledger.observe(Sighting(t=2.0, subject_key="C"))  # evicts A (least recently seen)
        self.assertEqual(ledger.distinct_subject_count, 2)
        self.assertEqual(ledger.subject_keys(), {"B", "C"})

    def test_eviction_is_never_silent(self) -> None:
        seen: list[tuple[str, str]] = []
        ledger = IdentityLedger(max_subjects=1, on_evict=lambda key, reason: seen.append((key, reason)))
        ledger.observe(Sighting(t=0.0, subject_key="A"))
        ledger.observe(Sighting(t=1.0, subject_key="B"))
        self.assertEqual(ledger.eviction_count, 1)
        self.assertEqual(len(ledger.evictions), 1)
        self.assertEqual(seen, ledger.evictions)

    def test_evicted_subject_reopens_as_new_not_reopen(self) -> None:
        from percept.eval.identity_ledger import FireKind

        ledger = IdentityLedger(max_subjects=1)
        ledger.observe(Sighting(t=0.0, subject_key="A"))
        ledger.observe(Sighting(t=1.0, subject_key="B"))  # evicts A
        out = ledger.observe(Sighting(t=2.0, subject_key="A"))  # A was forgotten -> a fresh OPEN
        self.assertTrue(out.fired)
        self.assertEqual(out.fire_kind, FireKind.OPEN)


if __name__ == "__main__":
    unittest.main()
