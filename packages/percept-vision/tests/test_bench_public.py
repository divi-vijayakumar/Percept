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
from eval.bench import card as snapshot
from eval.bench.types import (
    TRACK_RESULT_SCHEMA_V2,
    Measurement,
    TrackResult,
)


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
            schema_version=TRACK_RESULT_SCHEMA_V2,
            contract={
                "runtime_path": (
                    "WatchPlan -> ConfigResolver -> ResolvedWatch -> "
                    "RuntimeCompiler -> worker -> JudgePool -> Incident"
                ),
                "qualification_scope": "owned",
                "plan_hashes": ["a" * 64],
                "component_snapshot_hashes": ["b" * 64],
                "evidence": "native clip",
                "decision_output": "Incident",
                "dense_baseline": {
                    "unit": "watch opportunities",
                    "formula": "duration * fps * active Watches",
                    "capture_fps": 2.0,
                    "active_watches": 1,
                },
            },
            measurements={
                "candidate_calls": Measurement(
                    value=10,
                    unit="logical_calls",
                    conditioning="one pinned case",
                    denominator=1,
                ),
            },
            reconciliation={
                "checks": {"balanced": True},
                "source_dropped_frames": 0,
                "forbidden_evidence_fallbacks": 0,
                "superseded_before_launch": 0,
                "revision_dropped_before_launch": 0,
                "deadline_dropped_before_launch": 0,
                "deadline_expired_after_launch": 0,
                "final_provider_inflight": 0,
                "final_pool_inflight": 0,
            },
            quality_gates=[{"id": "fixture", "status": "pass"}],
        )
        self.assertEqual(snapshot.stamp_of("full", [result], "manifest sha256:abc"), "RELEASE")

    def test_release_rejects_v1_even_for_non_pt_tracks(self):
        result = TrackResult(
            track="custom-runtime",
            cells={("e2e", "notification_recall"): 1.0},
            measured=["e2e:notification_recall"],
            coverage=[{
                "dataset": "owned",
                "expected": 1,
                "observed": 1,
                "case_ids": ["case-1"],
            }],
            qualifying_events=1,
        )

        self.assertEqual(
            snapshot.stamp_of(
                "full",
                [result],
                "manifest sha256:abc",
            ),
            "DRAFT",
        )
        self.assertTrue(
            any(
                "release requires percept.benchmark.track_result/2"
                in reason
                for reason in snapshot.release_blockers([result])
            )
        )

    def test_pt_release_requires_versioned_qualification_envelope(self):
        legacy = TrackResult(
            track="pt-action-e2e",
            cells={("e2e", "notification_recall"): 1.0},
            measured=["e2e:notification_recall"],
            coverage=[{
                "dataset": "pt",
                "expected": 1,
                "observed": 1,
                "case_ids": ["case-1"],
            }],
            qualifying_events=1,
        )
        self.assertEqual(
            snapshot.stamp_of("full", [legacy], "manifest sha256:abc"),
            "DRAFT",
        )
        self.assertTrue(
            any(
                "track_result/2" in reason
                for reason in snapshot.release_blockers([legacy])
            )
        )

    def test_v2_measurement_denominator_and_gate_fail_closed(self):
        result = TrackResult(
            track="watch-e2e",
            cells={("e2e", "notification_recall"): 1.0},
            measured=["e2e:notification_recall"],
            coverage=[{
                "dataset": "owned",
                "expected": 1,
                "observed": 1,
                "case_ids": ["case-1"],
            }],
            schema_version=TRACK_RESULT_SCHEMA_V2,
            contract={"path": "compiled-runtime"},
            measurements={
                "notification_recall": {
                    "value": 1.0,
                    "unit": "ratio",
                    "conditioning": "all positive occurrences",
                    "numerator": 1,
                    "denominator": 0,
                },
            },
            reconciliation={"final_inflight": 0},
            quality_gates=[{"id": "boundedness", "status": "blocked"}],
        )

        blockers = snapshot.release_blockers([result])

        self.assertTrue(any("denominator must be positive" in item for item in blockers))
        self.assertTrue(any("boundedness=blocked" in item for item in blockers))

    def test_coverage_preserves_event_recall_denominator(self):
        result = TrackResult(
            track="pt-action-e2e",
            coverage=[{
                "dataset": "perception-test/action",
                "expected": 1,
                "observed": 1,
                "positive_events": 2,
            }],
        )

        coverage = result.normalized_coverage()

        self.assertEqual(coverage[0].positive_events, 2)

    def test_result_json_round_trip_preserves_resolved_configuration(self):
        from tools.percept_benchmark import _json_safe_result, _track_result_from_json

        result = TrackResult(
            track="pt-action-e2e",
            cells={("e2e", "notification_recall"): 1.0},
            configuration={
                "provider": "openrouter",
                "watch": {
                    "rearm": "each_event",
                    "perception": {
                        "temporal": "strip",
                        "n_frames": 5,
                        "sampling": "uniform",
                        "window_s": 2.0,
                    },
                },
            },
        )

        restored = _track_result_from_json(
            json.loads(json.dumps(_json_safe_result(result)))
        )

        self.assertEqual(restored.configuration, result.configuration)
        self.assertEqual(restored.cells, result.cells)

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
        self.assertEqual(snapshot.stamp_of("full", [diagnostic], "manifest sha256:abc"), "DRAFT")
        self.assertEqual(snapshot.stamp_of("full", [legacy], "manifest sha256:abc"), "DRAFT")

    def test_scorecard_renders_long_track_names_readably(self):
        result = TrackResult(
            track="proactiveqa-e2e",
            cells={("e2e", "pauc@0.5_proxy"): 33.5},
            measured=["e2e:pauc@0.5_proxy"],
        )
        text = snapshot.render_card([result])
        self.assertIn("proactiveqa-e2e  pauc@0.5_proxy", text)

    def test_scorecard_renders_v2_measurement_semantics_and_gates(self):
        result = TrackResult(
            track="watch-e2e",
            cells={("e2e", "notification_recall"): 1.0},
            measured=["e2e:notification_recall"],
            schema_version=TRACK_RESULT_SCHEMA_V2,
            contract={"path": "compiled-runtime"},
            measurements={
                "occurrence_recall": Measurement(
                    value=1.0,
                    unit="ratio",
                    conditioning="all positive occurrences",
                    numerator=2,
                    denominator=2,
                )
            },
            reconciliation={"fires": 2, "incidents": 2},
            quality_gates=[{"id": "reconciliation", "status": "pass"}],
        )

        text = snapshot.render_card(
            [result],
            sdk_artifact=(
                "percept-vision 1.0.0 wheel "
                f"sha256:{'a' * 64} build:{'b' * 40} contract:1.10.0"
            ),
        )

        self.assertIn(
            "watch-e2e:percept.benchmark.track_result/2",
            text,
        )
        self.assertIn(
            "watch-e2e:occurrence_recall "
            "[ratio; all positive occurrences; denominator=2]",
            text,
        )
        self.assertIn("watch-e2e:reconciliation=pass", text)
        self.assertIn("sdk-artifact: percept-vision 1.0.0 wheel", text)
        self.assertIn(
            "video / PT action          : candidate",
            text,
        )
        self.assertIn(
            "audio Watch API           : preview; bounded runtime-conformance only",
            text,
        )
        self.assertIn(
            "execution status is not quality status",
            text,
        )

    def test_release_matrix_does_not_overclaim_free_pack_accuracy(self):
        matrix = "\n".join(snapshot._oss_qualification_matrix("RELEASE"))
        self.assertIn("video / PT action          : provider-qualified", matrix)
        self.assertIn("Physical Scene starter    : available; mechanics-conformance only", matrix)
        self.assertIn("Ego Motion starter        : available; mechanics-conformance only", matrix)
        self.assertIn("no pack-specific quality/latency/cost claim", matrix)

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
        self.assertEqual(snapshot.stamp_of("full", [], "manifest sha256:abc"), "DRAFT")
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
            self.assertEqual(snapshot.stamp_of("full", [result], "manifest sha256:abc"), "DRAFT")

    def test_scorecard_execution_rejects_failed_or_all_empty_results(self):
        self.assertTrue(snapshot.execution_blockers([]))
        self.assertTrue(
            snapshot.execution_blockers(
                [TrackResult(track="watch", cells={("e2e", "recall"): None})]
            )
        )
        self.assertTrue(
            snapshot.execution_blockers(
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
