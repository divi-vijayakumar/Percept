"""Public benchmark import/CLI shim."""

from __future__ import annotations

import unittest
import json
import os
import subprocess
import sys
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory

from percept import bench
from eval.bench import card
from eval.bench.types import TrackResult


ROOT = Path(__file__).resolve().parents[3]


def _write_manifest(path: Path, *, status: str = "draft") -> None:
    path.write_text(
        json.dumps(
            {
                "version": "test",
                "status": status,
                "tracks": [
                    {"id": "relational", "source": "self", "status": status, "metrics": ["f1"]}
                ],
                "splits": {
                    "dev": [],
                    "test": [
                        {
                            "id": "case-1",
                            "source": "self-recorded",
                            "track": "relational",
                        }
                    ],
                },
            }
        )
    )


class TestPublicBenchShim(unittest.TestCase):
    def test_exports_manifest_and_report_helpers(self):
        self.assertTrue(hasattr(bench, "Manifest"))
        self.assertTrue(hasattr(bench, "run_manifest"))
        self.assertTrue(callable(bench.main))
        self.assertTrue(callable(bench.scorecard_main))

    def test_scorecard_subcommand_writes_card(self):
        with TemporaryDirectory() as d:
            out = Path(d) / "scorecard.txt"
            manifest = Path(d) / "MANIFEST.json"
            _write_manifest(manifest)
            with redirect_stdout(StringIO()):
                rc = bench.main(["scorecard", "--offline", "--manifest", str(manifest), "--out", str(out)])
            self.assertEqual(rc, 0)
            text = out.read_text()
            self.assertIn("Percept Benchmark Snapshot", text)
            self.assertIn("sha256:", text)
            self.assertNotIn("escalation recall", text)

    def test_scorecard_require_release_rejects_draft_card(self):
        with TemporaryDirectory() as d:
            out = Path(d) / "scorecard.txt"
            manifest = Path(d) / "MANIFEST.json"
            _write_manifest(manifest)
            buf = StringIO()
            with redirect_stdout(buf):
                rc = bench.main(
                    ["scorecard", "--manifest", str(manifest), "--out", str(out), "--require-release"]
                )
            self.assertEqual(rc, 2)
            self.assertIn("RELEASE GATE: BLOCKED", buf.getvalue())
            self.assertFalse(out.exists())

    def test_release_gate_allows_dense_modeled_cost_reference(self):
        result = TrackResult(
            track="pt-action-e2e",
            cells={
                ("dense-modeled", "dense_frame_calls"): 100,
                ("e2e", "candidate_calls"): 10,
            },
            measured=["e2e:candidate_calls"],
            modeled=["dense-modeled:dense_frame_calls"],
            coverage=[{
                "dataset": "pt",
                "expected": 1,
                "observed": 1,
                "failed": 0,
                "skipped": 0,
                "case_ids": ["case-1"],
            }],
        )
        self.assertEqual(card.stamp_of("full", [result], "manifest sha256:abc"), "RELEASE")

    def test_release_gate_rejects_diagnostics_and_legacy_lanes(self):
        diagnostic = TrackResult(
            track="counting",
            cells={("edge", "obo"): 0.0},
            diagnostic=True,
            excluded=["counting: diagnostic"],
            coverage=[{"dataset": "repcount", "expected": 1, "observed": 1, "failed": 0, "skipped": 0}],
        )
        legacy = TrackResult(
            track="legacy",
            cells={("raw", "precision"): 1.0},
            coverage=[{"dataset": "x", "expected": 1, "observed": 1, "failed": 0, "skipped": 0}],
        )
        self.assertEqual(card.stamp_of("full", [diagnostic], "manifest sha256:abc"), "DRAFT")
        self.assertEqual(card.stamp_of("full", [legacy], "manifest sha256:abc"), "DRAFT")

    def test_scorecard_renders_long_track_names_readably(self):
        result = TrackResult(
            track="proactiveqa-e2e",
            cells={("e2e", "pauc@0.5_proxy"): 33.5},
            measured=["e2e:pauc@0.5_proxy"],
        )
        text = card.render_card([result])
        self.assertIn("proactiveqa-e2e  pauc@0.5_proxy", text)

    def test_audit_subcommand_accepts_draft_artifacts(self):
        with TemporaryDirectory() as d:
            manifest = Path(d) / "MANIFEST.json"
            baselines = Path(d) / "baselines.json"
            manifest.write_text(
                """
                {
                  "version": "golden-v1-draft",
                  "status": "draft",
                  "tracks": [{"id": "relational", "source": "self", "status": "draft", "metrics": ["f1"]}],
                  "splits": {"dev": [], "test": [
                    {"id": "case-1", "source": "self-recorded", "track": "relational"}
                  ]}
                }
                """
            )
            baselines.write_text(
                """
                {
                  "baselines": [{
                    "id": "ref",
                    "system": "Reference",
                    "kind": "reference_bar",
                    "controlled_ab": false,
                    "source": "https://example.com",
                    "metric": "f1",
                    "comparability": "context only"
                  }]
                }
                """
            )
            buf = StringIO()
            with redirect_stdout(buf):
                rc = bench.main(["audit", "--manifest", str(manifest), "--baselines", str(baselines)])
            self.assertEqual(rc, 0)
            self.assertIn("ARTIFACT-AUDIT PASS", buf.getvalue())
            self.assertIn("Cloud cost gate : NOT-RUN", buf.getvalue())

    def test_audit_subcommand_rejects_empty_draft_corpus(self):
        with TemporaryDirectory() as d:
            manifest = Path(d) / "MANIFEST.json"
            baselines = Path(d) / "baselines.json"
            manifest.write_text(
                '{"version":"draft","status":"draft","tracks":'
                '[{"id":"x","source":"self","status":"draft","metrics":["f1"]}],'
                '"splits":{"dev":[],"test":[]}}'
            )
            baselines.write_text('{"baselines":[]}')
            with redirect_stdout(StringIO()) as out:
                rc = bench.main(
                    ["audit", "--manifest", str(manifest), "--baselines", str(baselines)]
                )
            self.assertEqual(rc, 1)
            self.assertIn("FAILED", out.getvalue())

    def test_module_invocation_executes_and_missing_artifact_fails(self):
        env = {**os.environ, "PYTHONPATH": str(ROOT / "packages/percept-vision")}
        ok = subprocess.run(
            [
                sys.executable,
                "-m",
                "percept.bench",
                "audit",
                "--manifest",
                str(ROOT / "eval/golden-v1/MANIFEST.json"),
                "--baselines",
                str(ROOT / "packages/percept-vision/eval/baselines.json"),
            ],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
        )
        self.assertEqual(ok.returncode, 0, ok.stdout + ok.stderr)
        self.assertIn("PERCEPT BENCHMARK ARTIFACT AUDIT", ok.stdout)
        missing = subprocess.run(
            [sys.executable, "-m", "percept.bench", "audit", "--manifest", "does-not-exist.json"],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(missing.returncode, 0)
        self.assertIn("FAILED", missing.stdout)

    def test_audit_release_mode_rejects_draft_manifest(self):
        buf = StringIO()
        with redirect_stdout(buf):
            rc = bench.main(
                [
                    "audit",
                    "--release",
                    "--manifest",
                    str(ROOT / "eval/golden-v1/MANIFEST.json"),
                    "--baselines",
                    str(ROOT / "packages/percept-vision/eval/baselines.json"),
                ]
            )
        self.assertEqual(rc, 1)
        self.assertIn("AUDIT FAILED", buf.getvalue())

    def test_release_gate_rejects_empty_incomplete_and_non_vacuous_results(self):
        self.assertEqual(card.stamp_of("full", [], "manifest sha256:abc"), "DRAFT")
        incomplete = TrackResult(
            track="watch-e2e",
            cells={("e2e", "notification_recall"): None},
            measured=["e2e:notification_recall"],
            coverage=[{"dataset": "x", "expected": 1, "observed": 1}],
        )
        no_fire = TrackResult(
            track="watch-e2e",
            cells={("e2e", "notification_recall"): 0.0},
            measured=["e2e:notification_recall"],
            coverage=[{"dataset": "x", "expected": 1, "observed": 1}],
            qualifying_events=0,
        )
        modeled = TrackResult(
            track="watch-e2e",
            cells={("e2e", "notification_recall"): 1.0},
            modeled=["e2e:notification_recall"],
            coverage=[{"dataset": "x", "expected": 1, "observed": 1}],
        )
        for result in (incomplete, no_fire, modeled):
            self.assertEqual(card.stamp_of("full", [result], "manifest sha256:abc"), "DRAFT")

    def test_scorecard_execution_rejects_failed_or_all_empty_results(self):
        self.assertTrue(card.execution_blockers([]))
        self.assertTrue(
            card.execution_blockers(
                [TrackResult(track="watch", cells={("e2e", "recall"): None})]
            )
        )
        self.assertTrue(
            card.execution_blockers(
                [
                    TrackResult(
                        track="watch",
                        cells={("e2e", "recall"): 1.0},
                        status="failed",
                    )
                ]
            )
        )


if __name__ == "__main__":
    unittest.main()
