"""— the portable trace bundle: two explicit modes, media OFF by default, hashes verified.

The bar these tests hold: a bundle exported by someone debugging accounting screens at 2am must not
carry pixels unless a human explicitly said so, must say plainly which mode it is in, and must be
provably intact after transport.
"""

from __future__ import annotations

import json
import unittest
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory

from percept.gate import Gate
from percept.replay import gate_config_of, replay_cognition
from percept.trace_bundle import (
    BUNDLE_SUFFIX,
    BundleError,
    BundlePolicy,
    EvidenceItem,
    MediaReference,
    TraceBundle,
    export_bug_report,
)
from percept.tracing import RunTraceManifest, TracePolicy, TraceRecorder


def _recorder(run_id="run-123"):
    manifest = RunTraceManifest.create(
        run_id=run_id, source="camera", policy=TracePolicy(level="full"),
        gate_config=gate_config_of(Gate()),
    )
    rec = TraceRecorder(manifest)
    rec.record("source.frame", source_ts=1.0)
    rec.record("gate.classified", source_ts=1.0, goal_id="g1", truth="yes", action="fire",
               satisfied=True, confidence=0.9, freshness=1.0, window_title="Ledger Q3.xlsx")
    rec.record("incident.created", source_ts=1.0, goal_id="g1", incident_id="i1")
    return rec


_PIXELS = EvidenceItem(name="frame-001.jpg", data=b"\xff\xd8pretend-jpeg", media_type="image/jpeg")


class TestMediaIsOptIn(unittest.TestCase):
    def test_default_bundle_is_local_and_drops_offered_media(self):
        bundle = TraceBundle.from_recorder(_recorder(), evidence=[_PIXELS])
        self.assertEqual(bundle.mode, "local")
        self.assertEqual(bundle.evidence, ())
        self.assertEqual(bundle.media_omitted, 1)
        self.assertEqual(bundle.media_omitted_reasons, {"policy_opt_out": 1})

    def test_default_policy_has_media_off(self):
        self.assertFalse(BundlePolicy().include_media)
        self.assertFalse(BundlePolicy.local().include_media)
        self.assertTrue(BundlePolicy.portable().include_media)

    def test_local_bundle_writes_no_evidence_bytes_anywhere_on_disk(self):
        bundle = TraceBundle.from_recorder(_recorder(), evidence=[_PIXELS])
        with TemporaryDirectory() as tmp:
            root = bundle.write(Path(tmp) / "run-123")
            blobs = b"".join(p.read_bytes() for p in root.rglob("*") if p.is_file())
            self.assertNotIn(b"pretend-jpeg", blobs)
            self.assertFalse((root / "evidence").exists())
            header = json.loads((root / "manifest.json").read_text())
            self.assertEqual(header["mode"], "local")
            self.assertFalse(header["media_included"])
            self.assertEqual(header["media_omitted"], 1)

    def test_export_bug_report_defaults_to_no_media(self):
        with TemporaryDirectory() as tmp:
            out = export_bug_report(_recorder(), Path(tmp) / "run-123", evidence=[_PIXELS])
            self.assertNotIn(b"pretend-jpeg", out.read_bytes())
            with zipfile.ZipFile(out) as zf:
                self.assertFalse([n for n in zf.namelist() if n.startswith("evidence/")])

    def test_portable_mode_packs_media_only_when_asked(self):
        bundle = TraceBundle.from_recorder(
            _recorder(), policy=BundlePolicy.portable(), evidence=[_PIXELS]
        )
        self.assertEqual(bundle.mode, "portable")
        with TemporaryDirectory() as tmp:
            root = bundle.write(Path(tmp) / "run-123")
            self.assertEqual((root / "evidence" / "frame-001.jpg").read_bytes(), _PIXELS.data)

    def test_media_bounds_drop_loudly_not_silently(self):
        policy = BundlePolicy.portable(max_media_files=1)
        bundle = TraceBundle.from_recorder(
            _recorder(), policy=policy,
            evidence=[_PIXELS, EvidenceItem(name="frame-002.jpg", data=b"x")],
        )
        self.assertEqual(len(bundle.evidence), 1)
        self.assertEqual(bundle.media_omitted_reasons, {"max_media_files": 1})

    def test_media_references_travel_without_the_bytes(self):
        ref = MediaReference(uri="file:///clips/a.mp4", sha256="abc", bytes_len=10)
        bundle = TraceBundle.from_recorder(_recorder(), media_refs=[ref])
        with TemporaryDirectory() as tmp:
            root = bundle.write(Path(tmp) / "run-123")
            back = TraceBundle.read(root)
        self.assertEqual(back.media_refs[0].uri, "file:///clips/a.mp4")
        self.assertEqual(back.mode, "local")

    def test_redaction_strips_named_keys_from_every_event(self):
        bundle = TraceBundle.from_recorder(
            _recorder(), policy=BundlePolicy(redact_keys=("window_title",))
        )
        self.assertTrue(all("window_title" not in r.data for r in bundle.records))


class TestBundleLayoutAndIntegrity(unittest.TestCase):
    def test_layout_is_manifest_events_summary(self):
        bundle = TraceBundle.from_recorder(_recorder(), policy=BundlePolicy.portable(),
                                           evidence=[_PIXELS])
        with TemporaryDirectory() as tmp:
            root = bundle.write(Path(tmp) / "run-123")
            self.assertTrue(root.name.endswith(BUNDLE_SUFFIX))
            names = sorted(str(p.relative_to(root)) for p in root.rglob("*") if p.is_file())
            self.assertEqual(
                names, ["events.jsonl", "evidence/frame-001.jpg", "manifest.json", "summary.json"]
            )
            lines = (root / "events.jsonl").read_text().strip().splitlines()
            self.assertEqual(len(lines), len(bundle.records))
            self.assertEqual(json.loads(lines[0])["stage"], "source.frame")

    def test_round_trip_through_a_single_file(self):
        rec = _recorder()
        bundle = TraceBundle.from_recorder(rec)
        with TemporaryDirectory() as tmp:
            archive = bundle.write_archive(Path(tmp) / "run-123")
            self.assertTrue(archive.name.endswith(".ptrace.zip"))
            back = TraceBundle.read(archive)
        self.assertEqual(back.manifest.run_id, "run-123")
        self.assertEqual(len(back.records), len(bundle.records))
        self.assertEqual(back.summary.incidents, bundle.summary.incidents)
        self.assertEqual(back.summary.to_dict(), bundle.summary.to_dict())

    def test_every_file_is_content_hashed_and_verifies(self):
        bundle = TraceBundle.from_recorder(_recorder(), policy=BundlePolicy.portable(),
                                           evidence=[_PIXELS])
        with TemporaryDirectory() as tmp:
            root = bundle.write(Path(tmp) / "run-123")
            self.assertEqual(TraceBundle.verify_path(root), [])
            archive = bundle.write_archive(Path(tmp) / "run-123")
            self.assertEqual(TraceBundle.verify_path(archive), [])

    def test_tampering_is_caught(self):
        bundle = TraceBundle.from_recorder(_recorder())
        with TemporaryDirectory() as tmp:
            root = bundle.write(Path(tmp) / "run-123")
            (root / "events.jsonl").write_text("{}\n")
            problems = TraceBundle.verify_path(root)
        self.assertTrue(any("events.jsonl" in p for p in problems), problems)

    def test_missing_file_is_caught(self):
        bundle = TraceBundle.from_recorder(_recorder())
        with TemporaryDirectory() as tmp:
            root = bundle.write(Path(tmp) / "run-123")
            (root / "summary.json").unlink()
            self.assertIn("missing file: summary.json", TraceBundle.verify_path(root))

    def test_reading_a_non_bundle_raises_typed(self):
        with TemporaryDirectory() as tmp:
            junk = Path(tmp) / "notes.txt"
            junk.write_text("hi")
            with self.assertRaises(BundleError):
                TraceBundle.read(junk)


class TestRetention(unittest.TestCase):
    def test_expiry_is_stamped_by_default(self):
        bundle = TraceBundle.from_recorder(_recorder(), wall_clock=lambda: 1000.0)
        self.assertEqual(bundle.expires_wall_ts, 1000.0 + 30 * 86400.0)
        self.assertFalse(bundle.expired(now=1000.0))
        self.assertTrue(bundle.expired(now=1000.0 + 31 * 86400.0))

    def test_keeping_forever_must_be_typed_explicitly(self):
        bundle = TraceBundle.from_recorder(
            _recorder(), policy=BundlePolicy(retention_days=None), wall_clock=lambda: 1000.0
        )
        self.assertIsNone(bundle.expires_wall_ts)
        self.assertFalse(bundle.expired(now=1e12))

    def test_verify_reports_an_expired_bundle(self):
        bundle = TraceBundle.from_recorder(
            _recorder(), policy=BundlePolicy(retention_days=1.0), wall_clock=lambda: 0.0
        )
        with TemporaryDirectory() as tmp:
            root = bundle.write(Path(tmp) / "run-123")
            problems = TraceBundle.verify_path(root, now=10 * 86400.0)
        self.assertIn("bundle is past its retention/expiry stamp", problems)


class TestBundleStatesItsReplayContract(unittest.TestCase):
    def test_manifest_never_claims_media_reevaluation(self):
        header = TraceBundle.from_recorder(_recorder()).manifest_dict()
        self.assertTrue(header["replay"]["cognition_replay"])
        self.assertTrue(header["replay"]["cognition_replay_deterministic"])
        self.assertFalse(header["replay"]["media_reevaluation"])
        self.assertIn("COGNITION REPLAY", header["replay"]["statement"])

    def test_a_bundle_round_tripped_through_disk_replays_to_the_same_digest(self):
        rec = _recorder()
        live = replay_cognition(rec.manifest, list(rec.records))
        bundle = TraceBundle.from_recorder(rec)
        with TemporaryDirectory() as tmp:
            archive = bundle.write_archive(Path(tmp) / "run-123")
            back = TraceBundle.read(archive)
        self.assertEqual(back.replay().digest, live.digest)
        self.assertEqual(back.replay().incidents, 1)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
