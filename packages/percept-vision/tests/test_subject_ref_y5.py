"""— SubjectRef: namespace, tri-state resolution, resolver precedence, conflict as a
first-class outcome, and the ``Verdict.subject`` -> ``episodes._subject_match`` wiring.

Stdlib only, no backends, no keys. These assert the documented contract invariants — not a
model's ability to read a key off a real screen (that is a benchmark question, not a capability one).
"""

from __future__ import annotations

import unittest

from percept.contracts import (
    CONTRACT_VERSION,
    SUBJECT_RESOLVER_PRECEDENCE,
    SubjectConflict,
    SubjectRef,
    SubjectResolution,
    Verdict,
    resolve_subject,
)
from percept.episodes import EpisodeLog


def _bound(key: str, *, scheme: str = "content", ns: str = "acme-erp",
           scope: str = "durable", strategy: str = "judge", confidence=None) -> SubjectRef:
    return SubjectRef(
        scheme=scheme, namespace=ns, key=key, resolution="bound",
        scope=scope, strategy=strategy, confidence=confidence,
    )


def _unknown(*, scheme: str = "content", ns: str = "acme-erp",
             scope: str = "session", strategy: str = "judge") -> SubjectRef:
    return SubjectRef(
        scheme=scheme, namespace=ns, key=None, resolution="unknown",
        scope=scope, strategy=strategy,
    )


class ContractVersionTests(unittest.TestCase):
    def test_contract_at_least_1_4_0(self) -> None:
        # SubjectRef bumped the contract to 1.4.0; later additive changes move it forward. That floor
        # is what matters here: SubjectRef + Verdict.subject must be present at or above 1.4.0.
        major, minor, _patch = (int(p) for p in CONTRACT_VERSION.split("."))
        self.assertEqual(major, 1)
        self.assertGreaterEqual(minor, 4)


class SubjectRefInvariantTests(unittest.TestCase):
    def test_bound_requires_namespace(self) -> None:
        with self.assertRaises(ValueError):
            SubjectRef("content", "", "PO-1", "bound", "durable", "judge")

    def test_bound_requires_key(self) -> None:
        with self.assertRaises(ValueError):
            SubjectRef("content", "acme", None, "bound", "durable", "judge")
        with self.assertRaises(ValueError):
            SubjectRef("content", "acme", "", "bound", "durable", "judge")

    def test_unknown_requires_key_is_none(self) -> None:
        # Fixture invariant: unreadable key -> resolution='unknown', key=None, which
        # __post_init__ must REJECT if a key is present.
        with self.assertRaises(ValueError):
            SubjectRef("content", "acme", "PO-1", "unknown", "session", "judge")
        # key=None is the valid unknown
        ref = SubjectRef("content", "acme", None, "unknown", "session", "judge")
        self.assertEqual(ref.resolution, "unknown")
        self.assertIsNone(ref.key)

    def test_resolution_is_a_closed_vocabulary(self) -> None:
        # bound / unknown / provisional only — an arbitrary fourth string is refused.
        with self.assertRaises(ValueError):
            SubjectRef("content", "acme", "PO-1", "resolved", "durable", "judge")  # type: ignore[arg-type]

    def test_confidence_not_defaulted_to_one(self) -> None:
        self.assertIsNone(_bound("PO-1").confidence)
        with self.assertRaises(ValueError):
            _bound("PO-1", confidence=1.5)
        self.assertEqual(_bound("PO-1", confidence=0.4).confidence, 0.4)

    def test_absence_is_none_not_a_ref(self) -> None:
        # The absence of a subject is `SubjectRef | None` == None, never a third binding value.
        self.assertIsNone(Verdict(True, 0.9).subject)


class SubjectRefEqualityTests(unittest.TestCase):
    def test_two_unknown_never_equal(self) -> None:
        a, b = _unknown(), _unknown()
        self.assertNotEqual(a, b)
        # NaN-like: an unknown ref is not even equal to itself by value...
        self.assertFalse(a == a)
        # ...yet it remains findable in a container by identity (hash is identity-based).
        self.assertIn(a, {a})
        self.assertIn(a, [a])

    def test_bound_equal_on_scheme_ns_key(self) -> None:
        a = _bound("PO-1", scope="durable", strategy="client", confidence=0.9)
        b = _bound("PO-1", scope="session", strategy="judge", confidence=0.1)
        # scope/strategy/confidence are provenance, not identity
        self.assertEqual(a, b)
        self.assertEqual(hash(a), hash(b))
        self.assertEqual(len({a, b}), 1)  # dedupes in a set

    def test_bound_differs_on_namespace(self) -> None:
        # 'PO-4471' is NOT globally unique across ERPs — namespace disambiguates.
        self.assertNotEqual(_bound("PO-4471", ns="erp-a"), _bound("PO-4471", ns="erp-b"))

    def test_bound_vs_unknown_never_equal(self) -> None:
        self.assertNotEqual(_bound("PO-1"), _unknown())


class ResolverPrecedenceTests(unittest.TestCase):
    def test_precedence_order_constant(self) -> None:
        self.assertEqual(SUBJECT_RESOLVER_PRECEDENCE, ("client", "judge", "track"))

    def test_client_beats_judge_beats_track(self) -> None:
        c, j, t = _bound("PO-C", strategy="client"), _bound("PO-J", strategy="judge"), _bound("PO-T", strategy="track")
        self.assertEqual(resolve_subject(client=c, judge=j, track=t).subject.strategy, "client")
        self.assertEqual(resolve_subject(judge=j, track=t).subject.strategy, "judge")
        self.assertEqual(resolve_subject(track=t).subject.strategy, "track")

    def test_agreeing_resolvers_are_not_a_conflict(self) -> None:
        c = _bound("PO-1", strategy="client")
        j = _bound("PO-1", strategy="judge")  # same identity, different provenance
        res = resolve_subject(client=c, judge=j)
        self.assertFalse(res.has_conflict)
        self.assertEqual(res.subject.strategy, "client")

    def test_disagreement_is_a_first_class_conflict(self) -> None:
        c = _bound("PO-CLIENT", strategy="client")
        j = _bound("PO-JUDGE", strategy="judge")
        res = resolve_subject(client=c, judge=j)
        self.assertIsInstance(res, SubjectResolution)
        # winner is the higher-precedence ref — deterministic, never a coin flip
        self.assertEqual(res.subject, c)
        self.assertTrue(res.has_conflict)
        self.assertEqual(len(res.conflicts), 1)
        conflict = res.conflicts[0]
        self.assertIsInstance(conflict, SubjectConflict)
        self.assertEqual(conflict.winner, c)
        self.assertEqual(conflict.loser, j)

    def test_no_bound_resolver_falls_through_to_unknown(self) -> None:
        u = _unknown()
        res = resolve_subject(judge=u)
        self.assertEqual(res.subject.resolution, "unknown")
        self.assertIsNone(res.subject.key)
        self.assertFalse(res.has_conflict)

    def test_synthesized_unknown_keeps_provenance(self) -> None:
        # No bound resolver + no explicit unknown fallback -> synthesize an honest unknown that
        # carries the seed's scheme/namespace/scope (never fabricates a key).
        seed = _unknown(scheme="track", ns="cam-3", scope="source")
        res = resolve_subject(track=seed)
        self.assertEqual(res.subject.resolution, "unknown")
        self.assertIsNone(res.subject.key)
        self.assertEqual(res.subject.scheme, "track")
        self.assertEqual(res.subject.namespace, "cam-3")
        self.assertEqual(res.subject.scope, "source")

    def test_explicit_unknown_fallback_must_be_unknown(self) -> None:
        with self.assertRaises(ValueError):
            resolve_subject(unknown=_bound("PO-1"))

    def test_no_resolvers_at_all_raises(self) -> None:
        with self.assertRaises(ValueError):
            resolve_subject()


class ScopeDurableDocTests(unittest.TestCase):
    def test_durable_docstring_disclaims_state_persistence(self) -> None:
        doc = SubjectRef.__doc__ or ""
        self.assertIn("durable", doc)
        # It documents IDENTIFIER stability, NOT that episode state is persisted.
        self.assertIn("SESSION-ONLY", doc)


class SubjectRefSchemaTests(unittest.TestCase):
    def test_subjectref_in_generated_schema(self) -> None:
        from percept.contracts import jsonschema as js

        defs = js.generate_all()["$defs"]
        self.assertIn("SubjectRef", defs)
        s = defs["SubjectRef"]
        # required = fields without a default (confidence is defaulted, so it is optional)
        self.assertEqual(set(s["required"]), {"scheme", "namespace", "key", "resolution", "scope", "strategy"})
        self.assertEqual(s["properties"]["resolution"]["enum"], ["bound", "unknown", "provisional"])
        self.assertEqual(s["properties"]["scope"]["enum"], ["session", "source", "durable"])

    def test_verdict_schema_refs_subjectref(self) -> None:
        from percept.contracts import jsonschema as js

        v = js.generate_all()["$defs"]["Verdict"]
        anyof = v["properties"]["subject"]["anyOf"]
        self.assertIn({"$ref": "#/$defs/SubjectRef"}, anyof)


class EpisodeSubjectWiringTests(unittest.TestCase):
    """``episodes._subject_match`` prefers ``verdict.subject.key`` when bound, keeps the judge match."""

    def test_bound_verdict_subject_drives_continuation(self) -> None:
        log = EpisodeLog("g1")
        v1 = Verdict(True, 0.9, subject=_bound("INV-2291"))
        ep1, new1 = log.observe(0.0, v1)
        self.assertTrue(new1)
        # same bound subject -> continues the SAME episode, no judge needed
        v1b = Verdict(True, 0.9, subject=_bound("INV-2291"))
        ep2, new2 = log.observe(1.0, v1b)
        self.assertFalse(new2)
        self.assertEqual(ep1.id, ep2.id)
        # A different bound subject in the same region creates a distinct episode.
        v2 = Verdict(True, 0.9, subject=_bound("INV-2294"))
        ep3, new3 = log.observe(2.0, v2)
        self.assertTrue(new3)
        self.assertNotEqual(ep3.id, ep1.id)
        self.assertEqual(log.count(), 2)

    def test_unknown_subject_never_collapses_two_sightings(self) -> None:
        log = EpisodeLog("g1")
        # An unknown-subject verdict carries no match key -> without a judge, each opens a new episode
        # rather than fabricating shared identity.
        log.observe(0.0, Verdict(True, 0.4, subject=_unknown()))
        log.observe(1.0, Verdict(True, 0.4, subject=_unknown()))
        self.assertEqual(log.count(), 2)

    def test_explicit_subject_key_still_wins_over_verdict_subject(self) -> None:
        log = EpisodeLog("g1")
        log.observe(0.0, Verdict(True, 0.9, subject=_bound("FROM-VERDICT")))
        # An explicit subject_key= takes precedence over the exemplar's own subject.
        _, new = log.observe(1.0, Verdict(True, 0.9, subject=_bound("FROM-VERDICT")), subject_key="EXPLICIT")
        self.assertTrue(new)  # different key -> new episode
        self.assertEqual(log.count(), 2)

    def test_exemplar_without_subject_is_unchanged(self) -> None:
        # An exemplar that is not subject-bearing (a plain object) behaves byte-for-byte as before:
        # no judge, no subject_key -> every observation opens a new episode.
        log = EpisodeLog("g1")
        log.observe(0.0, object())
        log.observe(1.0, object())
        self.assertEqual(log.count(), 2)


if __name__ == "__main__":
    unittest.main()
