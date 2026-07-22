"""ADR-003 evidence dereference — the handle shape, the resolver, four typed failures, retention.

These tests pin the ADR-003 guarantees, not an implementation: the resolver never returns a silent
empty (it raises one of four typed failures), the opaque handle is excluded from egress by default,
media is opt-in and off the wire, and retention is a caller-parameterised mechanism whose ONLY fixed
policy is the cross-class eviction order (FIRE evicted last).
"""

import unittest

from percept.clock import FakeClock
from percept.episodes import EvidenceHandle
from percept.evidence import (
    EVICTION_ORDER,
    EvidenceAccessDenied,
    EvidenceArtifact,
    EvidenceCorrupt,
    EvidenceExpired,
    EvidenceStore,
    EvidenceUnavailable,
    RetentionPolicy,
    evidence_ref_of,
    resolve_evidence,
)


class TestArtifactShapeAndEgress(unittest.TestCase):
    def test_put_resolve_roundtrip_metadata(self):
        s = EvidenceStore()
        art = s.put(
            b"jpegbytes",
            kind="frame",
            media_type="image/jpeg",
            capture_start_ts=10.0,
            capture_end_ts=10.0,
        )
        got = resolve_evidence(art, store=s)
        self.assertEqual(got, art)
        self.assertEqual(got.kind, "frame")
        self.assertEqual(got.media_type, "image/jpeg")
        self.assertEqual(got.expires_at, None)  # default policy sets no TTL

    def test_capture_end_defaults_to_start_for_instantaneous_frame(self):
        s = EvidenceStore()
        art = s.put(b"x", kind="frame", media_type="image/jpeg", capture_start_ts=5.0)
        self.assertEqual(art.capture_start_ts, art.capture_end_ts)

    def test_ref_is_opaque_minted_not_a_path(self):
        s = EvidenceStore()
        art = s.put(b"x", kind="frame", media_type="image/jpeg", capture_start_ts=1.0)
        # An opaque minted key — unguessable body, no filesystem/URL semantics a caller may rely on.
        self.assertTrue(art.ref.startswith("ev_"))
        self.assertNotIn("/", art.ref)
        self.assertNotIn(".", art.ref)

    def test_to_egress_excludes_the_handle_by_default(self):
        s = EvidenceStore()
        art = s.put(b"x", kind="crop", media_type="image/jpeg", capture_start_ts=1.0)
        head = art.to_egress()
        # The ref is the local-only tail — structurally absent from the egress head.
        self.assertNotIn("ref", head)
        self.assertEqual(
            set(head), {"kind", "media_type", "capture_start_ts", "capture_end_ts", "sha256", "expires_at"}
        )
        # No bytes ever leave via egress either.
        self.assertNotIn("data", head)
        self.assertNotIn("media", head)

    def test_egress_may_opt_in_to_the_bare_handle_only(self):
        s = EvidenceStore()
        art = s.put(b"x", kind="frame", media_type="image/jpeg", capture_start_ts=1.0)
        head = art.to_egress(include_ref=True)
        self.assertEqual(head["ref"], art.ref)
        # Even opted-in, egress carries the handle string, never the bytes.
        self.assertNotIn("data", head)

    def test_egress_ref_is_not_a_dataclass_field(self):
        # EGRESS_FIELDS is a ClassVar; it must not appear in the frozen dataclass fields.
        from dataclasses import fields

        names = {f.name for f in fields(EvidenceArtifact)}
        self.assertNotIn("EGRESS_FIELDS", names)
        self.assertIn("ref", names)


class TestResolveHandleForms(unittest.TestCase):
    def test_resolves_bare_ref_string(self):
        s = EvidenceStore()
        art = s.put(b"x", kind="frame", media_type="image/jpeg", capture_start_ts=1.0)
        self.assertEqual(resolve_evidence(art.ref, store=s).ref, art.ref)

    def test_resolves_reused_y14_evidence_handle(self):
        # ADR-003: reuse Y14's EvidenceHandle shape, do not fork it. resolve_evidence must accept it.
        s = EvidenceStore()
        art = s.put(b"x", kind="frame", media_type="image/jpeg", capture_start_ts=1.0)
        handle = EvidenceHandle(ref=art.ref)
        self.assertEqual(resolve_evidence(handle, store=s).ref, art.ref)

    def test_ref_of_rejects_empty_and_typeless(self):
        with self.assertRaises(ValueError):
            evidence_ref_of("")
        with self.assertRaises(TypeError):
            evidence_ref_of(object())
        with self.assertRaises(ValueError):
            evidence_ref_of(EvidenceHandle(ref=None))  # a handle that never named a stored artifact


class TestTypedFailures(unittest.TestCase):
    def test_unknown_ref_raises_unavailable_not_none(self):
        s = EvidenceStore()
        with self.assertRaises(EvidenceUnavailable):
            resolve_evidence("ev_deadbeef", store=s)

    def test_expired_raises_expired_distinct_from_unavailable(self):
        clock = FakeClock()
        s = EvidenceStore(policy=RetentionPolicy(ttls={"fire": 100.0}), clock=clock)
        art = s.put(b"x", kind="frame", media_type="image/jpeg", capture_start_ts=0.0)
        self.assertEqual(art.expires_at, 100.0)
        clock.advance(101_000)  # ms — past the 100s TTL
        with self.assertRaises(EvidenceExpired):
            resolve_evidence(art, store=s)
        # And it STAYS expired (tombstoned), not degraded to "never seen".
        with self.assertRaises(EvidenceExpired):
            resolve_evidence(art, store=s)

    def test_access_denied_when_authorizer_refuses(self):
        s = EvidenceStore(authorizer=lambda artifact, cls: cls != "fire")
        art = s.put(b"x", kind="frame", media_type="image/jpeg", capture_start_ts=1.0, retention_class="fire")
        with self.assertRaises(EvidenceAccessDenied):
            resolve_evidence(art, store=s)
        # A different class the authorizer permits resolves fine.
        ok = s.put(b"y", kind="frame", media_type="image/jpeg", capture_start_ts=1.0, retention_class="ask")
        self.assertEqual(resolve_evidence(ok, store=s).ref, ok.ref)

    def test_corrupt_bytes_raise_corrupt_on_open(self):
        s = EvidenceStore()
        art = s.put(b"original", kind="frame", media_type="image/jpeg", capture_start_ts=1.0)
        # Tamper with the stored bytes behind the recorded sha256.
        s._entries[art.ref].data = b"tampered!"
        with self.assertRaises(EvidenceCorrupt):
            s.open_bytes(art)
        # resolve() (metadata only) still succeeds — corruption is a bytes-read fact.
        self.assertEqual(resolve_evidence(art, store=s).ref, art.ref)


class TestMediaIsOptInAndLocal(unittest.TestCase):
    def test_open_bytes_is_a_distinct_local_operation(self):
        s = EvidenceStore()
        art = s.put(b"pixels", kind="frame", media_type="image/jpeg", capture_start_ts=1.0)
        # Resolving metadata never yields bytes; you must ask for them explicitly and locally.
        self.assertEqual(s.open_bytes(art), b"pixels")

    def test_redaction_clears_bytes_keeps_head(self):
        s = EvidenceStore()
        art = s.put(b"pixels", kind="frame", media_type="image/jpeg", capture_start_ts=1.0)
        s.redact(art)
        # Metadata head survives (health dashboard); bytes are gone -> Unavailable, not Corrupt.
        self.assertEqual(resolve_evidence(art, store=s).kind, "frame")
        with self.assertRaises(EvidenceUnavailable):
            s.open_bytes(art)

    def test_redaction_can_run_ahead_of_ttl(self):
        clock = FakeClock()
        s = EvidenceStore(policy=RetentionPolicy(ttls={"fire": 1000.0}), clock=clock)
        art = s.put(b"pixels", kind="frame", media_type="image/jpeg", capture_start_ts=0.0)
        s.redact(art)  # distinct trigger, fires well before the 1000s window closes
        with self.assertRaises(EvidenceUnavailable):
            s.open_bytes(art)


class TestBundles(unittest.TestCase):
    def test_bundle_children_share_ref_namespace_and_resolve_ordered(self):
        s = EvidenceStore()
        parent = s.put_bundle(
            [(b"overview", "image/jpeg"), (b"crop", "image/jpeg"), (b"audio", "audio/wav")],
            capture_start_ts=1.0,
            capture_end_ts=2.0,
        )
        self.assertEqual(parent.kind, "bundle")
        children = s.resolve_bundle(parent)
        self.assertEqual(len(children), 3)
        # Children share the parent's ref namespace.
        for i, child in enumerate(children):
            self.assertEqual(child.ref, f"{parent.ref}/{i}")
        # Bytes are reachable per child, locally.
        self.assertEqual(s.open_bytes(children[0]), b"overview")
        self.assertEqual(s.open_bytes(children[2]), b"audio")

    def test_resolve_bundle_rejects_a_non_bundle(self):
        s = EvidenceStore()
        art = s.put(b"x", kind="frame", media_type="image/jpeg", capture_start_ts=1.0)
        with self.assertRaises(ValueError):
            s.resolve_bundle(art)


class TestRetentionMechanism(unittest.TestCase):
    def test_default_policy_never_expires(self):
        clock = FakeClock()
        s = EvidenceStore(clock=clock)  # default policy: no numbers
        art = s.put(b"x", kind="frame", media_type="image/jpeg", capture_start_ts=0.0)
        clock.advance(10_000_000)
        self.assertEqual(resolve_evidence(art, store=s).ref, art.ref)
        self.assertEqual(s.reconcile(), [])

    def test_reconcile_sweeps_expired(self):
        clock = FakeClock()
        s = EvidenceStore(policy=RetentionPolicy(ttls={"fire": 50.0}), clock=clock)
        art = s.put(b"x", kind="frame", media_type="image/jpeg", capture_start_ts=0.0)
        clock.advance(60_000)
        evicted = s.reconcile()
        self.assertEqual(evicted, [art.ref])
        self.assertEqual(len(s), 0)
        with self.assertRaises(EvidenceExpired):
            resolve_evidence(art, store=s)

    def test_explicit_expires_at_overrides_policy(self):
        clock = FakeClock()
        s = EvidenceStore(policy=RetentionPolicy(ttls={"fire": 50.0}), clock=clock)
        art = s.put(
            b"x", kind="frame", media_type="image/jpeg", capture_start_ts=0.0, expires_at=None
        )
        self.assertIsNone(art.expires_at)
        clock.advance(1_000_000)
        self.assertEqual(resolve_evidence(art, store=s).ref, art.ref)

    def test_eviction_order_constant_is_fire_last(self):
        self.assertEqual(EVICTION_ORDER, ("silent_debug", "sampled_debug", "ask", "fire"))

    def test_byte_pressure_evicts_fire_last(self):
        # A global ceiling forces eviction; FIRE must survive while debug/ask are shed first.
        s = EvidenceStore(policy=RetentionPolicy(max_bytes=30))
        blob = b"x" * 20
        fire = s.put(blob, kind="frame", media_type="image/jpeg", capture_start_ts=1.0, retention_class="fire")
        ask = s.put(blob, kind="frame", media_type="image/jpeg", capture_start_ts=1.0, retention_class="ask")
        silent = s.put(
            blob, kind="frame", media_type="image/jpeg", capture_start_ts=1.0, retention_class="silent_debug"
        )
        evicted = set(s.enforce_budget())
        # silent_debug and ask shed first; fire kept until last.
        self.assertIn(silent.ref, evicted)
        self.assertIn(ask.ref, evicted)
        self.assertNotIn(fire.ref, evicted)
        self.assertLessEqual(s.total_bytes(), 30)
        self.assertEqual(resolve_evidence(fire, store=s).ref, fire.ref)

    def test_per_class_subbudget_protects_fire_from_debug_storm(self):
        # A debug-capture storm cannot starve FIRE: its per-class sub-budget caps it.
        s = EvidenceStore(policy=RetentionPolicy(class_max_bytes={"silent_debug": 10}))
        blob = b"x" * 8
        s.put(blob, kind="frame", media_type="image/jpeg", capture_start_ts=1.0, retention_class="silent_debug")
        s.put(blob, kind="frame", media_type="image/jpeg", capture_start_ts=2.0, retention_class="silent_debug")
        fire = s.put(blob, kind="frame", media_type="image/jpeg", capture_start_ts=1.0, retention_class="fire")
        s.enforce_budget()
        # The silent-debug class is trimmed to its cap; FIRE is untouched.
        self.assertLessEqual(s._class_bytes("silent_debug"), 10)
        self.assertEqual(resolve_evidence(fire, store=s).ref, fire.ref)

    def test_no_eviction_when_no_ceiling_configured(self):
        s = EvidenceStore()  # no max_bytes -> the store never evicts under pressure the owner did not price
        for i in range(5):
            s.put(b"x" * 1000, kind="frame", media_type="image/jpeg", capture_start_ts=float(i))
        self.assertEqual(s.enforce_budget(), [])
        self.assertEqual(len(s), 5)


class TestLabelledExtension(unittest.TestCase):
    def test_extend_for_label_is_bounded_by_policy(self):
        clock = FakeClock()
        s = EvidenceStore(
            policy=RetentionPolicy(ttls={"fire": 50.0}, labelled_ceiling_s=200.0), clock=clock
        )
        art = s.put(b"x", kind="frame", media_type="image/jpeg", capture_start_ts=0.0)
        self.assertEqual(art.expires_at, 50.0)
        extended = s.extend_for_label(art)
        self.assertEqual(extended.expires_at, 200.0)  # now(0) + bounded ceiling, not indefinite
        clock.advance(100_000)  # past the original 50s window
        self.assertEqual(resolve_evidence(art, store=s).ref, art.ref)

    def test_extend_refuses_without_a_ceiling(self):
        s = EvidenceStore(policy=RetentionPolicy(ttls={"fire": 50.0}))  # no labelled_ceiling_s
        art = s.put(b"x", kind="frame", media_type="image/jpeg", capture_start_ts=0.0)
        with self.assertRaises(ValueError):
            s.extend_for_label(art)


class TestPersistenceSeam(unittest.TestCase):
    def test_snapshot_carries_index_not_bytes(self):
        s = EvidenceStore(policy=RetentionPolicy(ttls={"fire": 50.0}), clock=FakeClock())
        art = s.put(b"pixels", kind="frame", media_type="image/jpeg", capture_start_ts=0.0)
        snap = s.snapshot()
        entry = snap["entries"][art.ref]
        # The index carries metadata + backing location markers, and explicitly NOT the pixels.
        self.assertEqual(entry["artifact"]["ref"], art.ref)
        self.assertEqual(entry["retention_class"], "fire")
        self.assertTrue(entry["has_bytes"])
        self.assertNotIn("data", entry)


if __name__ == "__main__":
    unittest.main()
