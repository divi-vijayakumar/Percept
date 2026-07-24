"""P06-001 — WatchPlan.judge made operational: ArtifactBinding + ResolvedJudgeSpec + judge provenance.

Pins ONLY the additive contract shapes this incident lands (stdlib only; no resolver, no runtime — the
resolver that consumes ``WatchPlan.judge`` against the deployment catalog is tested elsewhere):

- ``ArtifactBinding`` construction + normalization + validation rejections.
- ``WatchPlan.judge_context`` round-trip WITH and WITHOUT a binding, and the BACK-COMPAT INVARIANT that
  an unset binding leaves ``to_dict()`` byte-for-byte identical to a plan built without the field.
- ``WatchPlan.from_dict`` reconstruction + rejection of a half-specified (ref-without-hash /
  hash-without-ref) binding.
- ``ResolvedJudgeSpec.to_dict`` round-trip.
- ``WatchReceipt`` new-field defaults.
- version constants: ``CONTRACT_VERSION == "1.10.0"`` and ``WATCH_VERSION == "1.1.0"``.

The two new DTOs are imported from ``percept.contracts.watchplan`` (their defining module); the pre-
existing DTOs re-export from ``percept.contracts`` as before.
"""

from __future__ import annotations

import json
import unittest

from percept.contracts import CONTRACT_VERSION
from percept.contracts.watchplan import (
    ArtifactBinding,
    ResolvedJudgeSpec,
    WatchPlan,
    WatchReceipt,
)
from percept.watch import WATCH_VERSION

_DIGEST = "a" * 64  # a valid bare-lowercase 64-char SHA-256 hex digest


class TestArtifactBinding(unittest.TestCase):
    def test_minimal_construction_defaults(self) -> None:
        b = ArtifactBinding(ref="prompt.envelope@v3", sha256=_DIGEST)
        self.assertEqual(b.ref, "prompt.envelope@v3")
        self.assertEqual(b.sha256, _DIGEST)
        self.assertEqual(b.media_type, "text/plain; charset=utf-8")
        self.assertIsNone(b.schema_version)

    def test_is_frozen(self) -> None:
        b = ArtifactBinding(ref="r", sha256=_DIGEST)
        with self.assertRaises(Exception):
            b.ref = "other"  # type: ignore[misc]

    def test_sha256_prefix_stripped_and_lowercased(self) -> None:
        upper = "A" * 64
        b = ArtifactBinding(ref="r", sha256="sha256:" + upper)
        self.assertEqual(b.sha256, "a" * 64)  # prefix stripped, lowercased to bare hex
        # A mixed-case bare digest normalizes identically to its prefixed form.
        self.assertEqual(ArtifactBinding(ref="r", sha256=upper).sha256, "a" * 64)

    def test_sha256_prefix_is_case_insensitive(self) -> None:
        b = ArtifactBinding(ref="r", sha256="SHA256:" + ("B" * 64))
        self.assertEqual(b.sha256, "b" * 64)

    def test_grammar_allows_full_token_alphabet(self) -> None:
        b = ArtifactBinding(ref="A-Za-z0.9_@-", sha256=_DIGEST)
        self.assertEqual(b.ref, "A-Za-z0.9_@-")

    def test_empty_ref_rejected(self) -> None:
        with self.assertRaises(ValueError):
            ArtifactBinding(ref="", sha256=_DIGEST)
        with self.assertRaises(ValueError):
            ArtifactBinding(ref="   ", sha256=_DIGEST)

    def test_ref_with_path_separator_rejected(self) -> None:
        for bad in ("a/b", "a\\b", "../etc", "dir/name"):
            with self.assertRaises(ValueError):
                ArtifactBinding(ref=bad, sha256=_DIGEST)

    def test_ref_with_disallowed_char_rejected(self) -> None:
        for bad in ("has space", "name!", "co:lon", "na#me"):
            with self.assertRaises(ValueError):
                ArtifactBinding(ref=bad, sha256=_DIGEST)

    def test_short_digest_rejected(self) -> None:
        with self.assertRaises(ValueError):
            ArtifactBinding(ref="r", sha256="a" * 63)

    def test_long_digest_rejected(self) -> None:
        with self.assertRaises(ValueError):
            ArtifactBinding(ref="r", sha256="a" * 65)

    def test_non_hex_digest_rejected(self) -> None:
        with self.assertRaises(ValueError):
            ArtifactBinding(ref="r", sha256="g" * 64)

    def test_empty_digest_rejected(self) -> None:
        with self.assertRaises(ValueError):
            ArtifactBinding(ref="r", sha256="")

    def test_to_dict_round_trip(self) -> None:
        b = ArtifactBinding(ref="r", sha256=_DIGEST, media_type="application/json", schema_version="2")
        self.assertEqual(
            b.to_dict(),
            {"ref": "r", "sha256": _DIGEST, "media_type": "application/json", "schema_version": "2"},
        )


class TestWatchPlanJudgeContext(unittest.TestCase):
    def test_default_judge_context_is_none(self) -> None:
        self.assertIsNone(WatchPlan(condition="x").judge_context)

    def test_to_dict_byte_identical_when_unset(self) -> None:
        # BACK-COMPAT INVARIANT: an unset judge_context must not perturb the wire at all — same key
        # SET and same serialized JSON as a plan whose class had never grown the field.
        plan = WatchPlan(
            condition="a person is at the door",
            sense="vision",
            detectors=["motion"],
            explicit={"Settings.edge.salience_stat": "block_max"},
            id="p1",
        )
        d = plan.to_dict()
        legacy_keys = {
            "condition", "sense", "target", "admission", "evidence", "detectors", "extractors",
            "judge", "consequence", "rearm", "uncertainty", "schedule", "explicit", "source",
            "profile", "id",
        }
        self.assertEqual(set(d), legacy_keys)
        # No judge_context_* key leaks onto the wire when the binding is unset.
        self.assertFalse(any(k.startswith("judge_context") for k in d))
        # And the serialized bytes are exactly what from_dict reconstructs back (stable plan_hash input).
        self.assertEqual(WatchPlan.from_dict(json.loads(json.dumps(d))).to_dict(), d)

    def test_to_dict_emits_flat_keys_when_set(self) -> None:
        b = ArtifactBinding(ref="rubric@v1", sha256=_DIGEST, media_type="application/json")
        d = WatchPlan(condition="x", judge_context=b).to_dict()
        self.assertEqual(d["judge_context_ref"], "rubric@v1")
        self.assertEqual(d["judge_context_sha256"], _DIGEST)
        self.assertEqual(d["judge_context_media_type"], "application/json")
        # schema_version key omitted when the binding carries none.
        self.assertNotIn("judge_context_schema_version", d)

    def test_to_dict_emits_schema_version_only_when_set(self) -> None:
        b = ArtifactBinding(ref="r", sha256=_DIGEST, schema_version="7")
        d = WatchPlan(condition="x", judge_context=b).to_dict()
        self.assertEqual(d["judge_context_schema_version"], "7")

    def test_round_trip_with_judge_context(self) -> None:
        b = ArtifactBinding(ref="r", sha256=_DIGEST, media_type="application/json", schema_version="7")
        plan = WatchPlan(condition="x", judge_context=b, id="p9")
        back = WatchPlan.from_dict(json.loads(json.dumps(plan.to_dict())))
        self.assertEqual(back.judge_context, b)
        self.assertEqual(back.to_dict(), plan.to_dict())

    def test_from_dict_without_judge_context_keys(self) -> None:
        back = WatchPlan.from_dict({"condition": "x"})
        self.assertIsNone(back.judge_context)

    def test_from_dict_rejects_ref_without_hash(self) -> None:
        with self.assertRaises(ValueError):
            WatchPlan.from_dict({"condition": "x", "judge_context_ref": "r"})

    def test_from_dict_rejects_hash_without_ref(self) -> None:
        with self.assertRaises(ValueError):
            WatchPlan.from_dict({"condition": "x", "judge_context_sha256": _DIGEST})

    def test_from_dict_normalizes_prefixed_digest(self) -> None:
        back = WatchPlan.from_dict(
            {"condition": "x", "judge_context_ref": "r", "judge_context_sha256": "sha256:" + ("A" * 64)}
        )
        assert back.judge_context is not None
        self.assertEqual(back.judge_context.sha256, "a" * 64)

    def test_judge_context_must_be_binding_or_none(self) -> None:
        with self.assertRaises(ValueError):
            WatchPlan(condition="x", judge_context="not-a-binding")  # type: ignore[arg-type]


class TestResolvedJudgeSpec(unittest.TestCase):
    def _spec(self) -> ResolvedJudgeSpec:
        return ResolvedJudgeSpec(
            requested_ref="fast-judge",
            resolved_ref="fast-judge",
            factory_ref="factory:anthropic_text",
            provider_ref="anthropic",
            model_requested="claude-x",
            model_resolved="claude-x",
            prompt_envelope_sha256=_DIGEST,
            context_ref="rubric@v1",
            context_sha256=_DIGEST,
        )

    def test_to_dict_round_trip(self) -> None:
        spec = self._spec()
        d = spec.to_dict()
        self.assertEqual(
            d,
            {
                "requested_ref": "fast-judge",
                "resolved_ref": "fast-judge",
                "factory_ref": "factory:anthropic_text",
                "provider_ref": "anthropic",
                "model_requested": "claude-x",
                "model_resolved": "claude-x",
                "prompt_envelope_sha256": _DIGEST,
                "context_ref": "rubric@v1",
                "context_sha256": _DIGEST,
            },
        )
        # Re-constructing from the dict yields an equal spec (plain, order-independent round-trip).
        self.assertEqual(ResolvedJudgeSpec(**d), spec)

    def test_optional_fields_accept_none(self) -> None:
        spec = ResolvedJudgeSpec(
            requested_ref=None,
            resolved_ref="default-judge",
            factory_ref="f",
            provider_ref="p",
            model_requested="m",
            model_resolved="m",
            prompt_envelope_sha256=None,
            context_ref=None,
            context_sha256=None,
        )
        self.assertIsNone(spec.requested_ref)
        self.assertIsNone(spec.context_ref)


class TestWatchReceiptJudgeFields(unittest.TestCase):
    def test_new_fields_default(self) -> None:
        r = WatchReceipt(status="applied")
        self.assertIsNone(r.judge_requested)
        self.assertIsNone(r.judge_resolved)
        self.assertIsNone(r.judge_context_ref)
        self.assertIsNone(r.judge_context_sha256)
        self.assertFalse(r.fallback_used)

    def test_new_fields_serialize(self) -> None:
        r = WatchReceipt(
            status="applied_with_warnings",
            judge_requested="fast-judge",
            judge_resolved="default-judge",
            judge_context_ref="rubric@v1",
            judge_context_sha256=_DIGEST,
            fallback_used=True,
        )
        d = r.to_dict()
        self.assertEqual(d["judge_requested"], "fast-judge")
        self.assertEqual(d["judge_resolved"], "default-judge")
        self.assertEqual(d["judge_context_ref"], "rubric@v1")
        self.assertEqual(d["judge_context_sha256"], _DIGEST)
        self.assertTrue(d["fallback_used"])


class TestVersionConstants(unittest.TestCase):
    def test_contract_version(self) -> None:
        self.assertEqual(CONTRACT_VERSION, "1.10.0")

    def test_watch_version(self) -> None:
        self.assertEqual(WATCH_VERSION, "1.1.0")


if __name__ == "__main__":
    unittest.main()
