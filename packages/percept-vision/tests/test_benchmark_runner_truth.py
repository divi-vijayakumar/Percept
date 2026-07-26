"""Collection acceptance is derived from pinned manifest tracks and case IDs."""

from __future__ import annotations

import json
import os
import runpy
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


ROOT = Path(__file__).resolve().parents[3]
TOOL = ROOT / "tools/percept_benchmark.py"


def _manifest(
    path: Path,
    runner_specs: dict[str, tuple[str, str, int]],
) -> None:
    tracks = []
    cases = []
    for runner, (logical_track, result_track, count) in runner_specs.items():
        tracks.append(
            {
                "id": logical_track,
                "source": "self",
                "status": "draft",
                "metrics": ["f1"],
                "runners": [{"id": runner, "result_track": result_track}],
            }
        )
        cases.extend(
            {
                "id": f"{logical_track}-case-{index}",
                "source": "self-recorded",
                "track": logical_track,
            }
            for index in range(count)
        )
    path.write_text(
        json.dumps(
            {
                "version": "test-draft",
                "status": "draft",
                "tracks": tracks,
                "splits": {"dev": [], "test": cases},
            }
        )
    )


def _result(path: Path, track: str, case_ids: list[str]) -> None:
    expected = len(case_ids)
    path.write_text(
        json.dumps(
            {
                "track": track,
                "cells": {"core-replay:f1": 1.0},
                "fmts": {},
                "headline": {"f1": "1.0"},
                "measured": ["core-replay:f1"],
                "modeled": [],
                "excluded": [],
                "reference": {},
                "calls": 0,
                "notes": "deterministic fixture",
                "coverage": [
                    {
                        "dataset": "fixture",
                        "expected": expected,
                        "observed": expected,
                        "failed": 0,
                        "skipped": 0,
                        "case_ids": case_ids,
                    }
                ],
                "diagnostic": False,
                "status": "pass",
                "qualifying_events": None,
            }
        )
    )


class BenchmarkRunnerTruthTests(unittest.TestCase):
    def test_all_alias_resolves_every_registered_track(self):
        namespace = runpy.run_path(str(TOOL), run_name="benchmark_tool_test")
        self.assertEqual(
            namespace["_resolve_tracks"]("all"),
            list(namespace["TRACK_MODULES"]),
        )

    def test_collect_passes_for_metadata_derived_arbitrary_coverage(self):
        with TemporaryDirectory() as d:
            root = Path(d)
            run_dir = root / "run"
            results = run_dir / "results"
            results.mkdir(parents=True)
            (run_dir / "run.json").write_text(
                json.dumps({"tracks": ["relational"], "live": False, "commit": "test"})
            )
            case_ids = [f"relational-track-case-{index}" for index in range(37)]
            _result(results / "relational.json", "relational", case_ids)
            manifest = root / "MANIFEST.json"
            _manifest(
                manifest,
                {"relational": ("relational-track", "relational", len(case_ids))},
            )
            out = root / "benchmark_card.txt"
            run = subprocess.run(
                [
                    sys.executable,
                    str(TOOL),
                    "collect",
                    "--run-dir",
                    str(run_dir),
                    "--manifest",
                    str(manifest),
                    "--out",
                    str(out),
                ],
                cwd=ROOT,
                env={**os.environ, "PYTHONPATH": str(ROOT / "packages/percept-vision")},
                capture_output=True,
                text=True,
            )
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
            self.assertGreater(out.stat().st_size, 0)
            self.assertIn("COLLECTION STATUS: PASS", run.stdout)
            self.assertIn("expected=37 observed=37", run.stdout)

    def test_sampled_collect_uses_deterministic_manifest_prefix(self):
        with TemporaryDirectory() as d:
            root = Path(d)
            run_dir = root / "run"
            results = run_dir / "results"
            results.mkdir(parents=True)
            (run_dir / "run.json").write_text(
                json.dumps(
                    {
                        "tracks": ["pt-action-e2e"],
                        "live": True,
                        "commit": "test",
                        "n": 1,
                    }
                )
            )
            _result(
                results / "pt-action-e2e.json",
                "pt-action-e2e",
                ["perception-test-action-case-0"],
            )
            manifest = root / "MANIFEST.json"
            _manifest(
                manifest,
                {
                    "pt-action-e2e": (
                        "perception-test-action",
                        "pt-action-e2e",
                        3,
                    )
                },
            )
            run = subprocess.run(
                [
                    sys.executable,
                    str(TOOL),
                    "collect",
                    "--run-dir",
                    str(run_dir),
                    "--manifest",
                    str(manifest),
                    "--mode",
                    "sampled",
                ],
                cwd=ROOT,
                env={**os.environ, "PYTHONPATH": str(ROOT / "packages/percept-vision")},
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
            self.assertIn("expected=1 observed=1", run.stdout)

    def test_collect_rejects_missing_selected_track_artifact(self):
        with TemporaryDirectory() as d:
            root = Path(d)
            run_dir = root / "run"
            results = run_dir / "results"
            results.mkdir(parents=True)
            (run_dir / "run.json").write_text(
                json.dumps(
                    {
                        "tracks": ["relational", "counting"],
                        "live": False,
                        "commit": "test",
                    }
                )
            )
            _result(
                results / "relational.json",
                "relational",
                ["relational-track-case-0"],
            )
            manifest = root / "MANIFEST.json"
            _manifest(
                manifest,
                {
                    "relational": ("relational-track", "relational", 1),
                    "counting": ("counting-track", "counting", 1),
                },
            )
            run = subprocess.run(
                [
                    sys.executable,
                    str(TOOL),
                    "collect",
                    "--run-dir",
                    str(run_dir),
                    "--manifest",
                    str(manifest),
                ],
                cwd=ROOT,
                env={**os.environ, "PYTHONPATH": str(ROOT / "packages/percept-vision")},
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(run.returncode, 0)
            self.assertIn("benchmark collection incomplete", run.stderr)
            self.assertIn("missing=counting", run.stderr)

    def test_live_runner_blocks_before_workers_without_key_or_budget(self):
        with TemporaryDirectory() as d:
            root = Path(d)
            run = subprocess.run(
                [
                    sys.executable,
                    str(TOOL),
                    "run",
                    "--live",
                    "--tracks",
                    "edge",
                    "--manifest",
                    str(ROOT / "eval/golden-v1/MANIFEST.json"),
                ],
                cwd=ROOT,
                env={
                    **os.environ,
                    "PYTHONPATH": str(ROOT / "packages/percept-vision"),
                    "PERCEPT_BENCH_RUNS_DIR": str(root / "runs"),
                    "PERCEPT_VISION_BACKEND": "gemini",
                    "GEMINI_API_KEY": "",
                    "GOOGLE_API_KEY": "",
                    "GOOGLE_AISTUDIO_API_KEY": "",
                    "PERCEPT_BENCH_BUDGET_PLAN": "",
                },
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(run.returncode, 0)
            self.assertIn("BENCHMARK LIVE GATE: BLOCKED", run.stderr)
            self.assertFalse((root / "runs").exists())

    def test_collect_validates_runner_and_distinct_result_identity(self):
        with TemporaryDirectory() as d:
            root = Path(d)
            run_dir = root / "run"
            results = run_dir / "results"
            results.mkdir(parents=True)
            (run_dir / "run.json").write_text(
                json.dumps({"tracks": ["edge"], "live": False, "commit": "test"})
            )
            _result(results / "edge.json", "edge", ["action-track-case-0"])
            manifest = root / "MANIFEST.json"
            _manifest(
                manifest,
                {"edge": ("action-track", "edge-recall", 1)},
            )
            run = subprocess.run(
                [
                    sys.executable,
                    str(TOOL),
                    "collect",
                    "--run-dir",
                    str(run_dir),
                    "--manifest",
                    str(manifest),
                ],
                cwd=ROOT,
                env={**os.environ, "PYTHONPATH": str(ROOT / "packages/percept-vision")},
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(run.returncode, 0)
            self.assertIn("TrackResult identity 'edge'", run.stderr)
            self.assertIn("result_track 'edge-recall'", run.stderr)

    def test_collect_accepts_explicit_edge_to_edge_recall_mapping(self):
        with TemporaryDirectory() as d:
            root = Path(d)
            run_dir = root / "run"
            results = run_dir / "results"
            results.mkdir(parents=True)
            (run_dir / "run.json").write_text(
                json.dumps({"tracks": ["edge"], "live": False, "commit": "test"})
            )
            _result(
                results / "edge.json",
                "edge-recall",
                ["action-track-case-0"],
            )
            manifest = root / "MANIFEST.json"
            _manifest(
                manifest,
                {"edge": ("action-track", "edge-recall", 1)},
            )
            run = subprocess.run(
                [
                    sys.executable,
                    str(TOOL),
                    "collect",
                    "--run-dir",
                    str(run_dir),
                    "--manifest",
                    str(manifest),
                ],
                cwd=ROOT,
                env={**os.environ, "PYTHONPATH": str(ROOT / "packages/percept-vision")},
                capture_output=True,
                text=True,
            )
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
            self.assertIn("COLLECTION STATUS: PASS", run.stdout)

    def test_collect_rejects_undeclared_runner(self):
        with TemporaryDirectory() as d:
            root = Path(d)
            run_dir = root / "run"
            results = run_dir / "results"
            results.mkdir(parents=True)
            (run_dir / "run.json").write_text(
                json.dumps({"tracks": ["edge"], "live": False, "commit": "test"})
            )
            _result(results / "edge.json", "edge-recall", ["action-track-case-0"])
            manifest = root / "MANIFEST.json"
            _manifest(
                manifest,
                {"different-runner": ("action-track", "edge-recall", 1)},
            )
            run = subprocess.run(
                [
                    sys.executable,
                    str(TOOL),
                    "collect",
                    "--run-dir",
                    str(run_dir),
                    "--manifest",
                    str(manifest),
                ],
                cwd=ROOT,
                env={**os.environ, "PYTHONPATH": str(ROOT / "packages/percept-vision")},
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(run.returncode, 0)
            self.assertIn("selected runner(s) not declared", run.stderr)

    def test_collect_blocks_manifest_without_explicit_runner_mapping(self):
        with TemporaryDirectory() as d:
            root = Path(d)
            run_dir = root / "run"
            results = run_dir / "results"
            results.mkdir(parents=True)
            (run_dir / "run.json").write_text(
                json.dumps({"tracks": ["edge"], "live": False, "commit": "test"})
            )
            _result(results / "edge.json", "edge-recall", ["action-case-0"])
            manifest = root / "MANIFEST.json"
            manifest.write_text(
                json.dumps(
                    {
                        "version": "draft",
                        "status": "draft",
                        "tracks": [{
                            "id": "action-track",
                            "source": "self",
                            "status": "draft",
                            "metrics": ["recall"],
                        }],
                        "splits": {
                            "dev": [],
                            "test": [{
                                "id": "action-case-0",
                                "source": "self",
                                "track": "action-track",
                            }],
                        },
                    }
                )
            )
            run = subprocess.run(
                [
                    sys.executable,
                    str(TOOL),
                    "collect",
                    "--run-dir",
                    str(run_dir),
                    "--manifest",
                    str(manifest),
                ],
                cwd=ROOT,
                env={**os.environ, "PYTHONPATH": str(ROOT / "packages/percept-vision")},
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(run.returncode, 0)
            self.assertIn("no explicit runners mapping", run.stderr)

    def test_collect_rejects_coverage_not_pinned_by_manifest(self):
        with TemporaryDirectory() as d:
            root = Path(d)
            run_dir = root / "run"
            results = run_dir / "results"
            results.mkdir(parents=True)
            (run_dir / "run.json").write_text(
                json.dumps({"tracks": ["relational"], "live": False, "commit": "test"})
            )
            _result(
                results / "relational.json",
                "relational",
                ["relational-track-case-0", "invented-case"],
            )
            manifest = root / "MANIFEST.json"
            _manifest(
                manifest,
                {"relational": ("relational-track", "relational", 1)},
            )
            run = subprocess.run(
                [
                    sys.executable,
                    str(TOOL),
                    "collect",
                    "--run-dir",
                    str(run_dir),
                    "--manifest",
                    str(manifest),
                ],
                cwd=ROOT,
                env={**os.environ, "PYTHONPATH": str(ROOT / "packages/percept-vision")},
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(run.returncode, 0)
            self.assertIn("does not match pinned manifest", run.stderr)
            self.assertIn("undeclared case IDs", run.stderr)

    def test_collect_blocks_selected_runner_with_no_pinned_cases(self):
        with TemporaryDirectory() as d:
            root = Path(d)
            run_dir = root / "run"
            results = run_dir / "results"
            results.mkdir(parents=True)
            (run_dir / "run.json").write_text(
                json.dumps({"tracks": ["edge"], "live": False, "commit": "test"})
            )
            _result(results / "edge.json", "edge-recall", [])
            manifest = root / "MANIFEST.json"
            _manifest(
                manifest,
                {
                    "edge": ("action-track", "edge-recall", 0),
                    "relational": ("relational-track", "relational", 1),
                },
            )
            run = subprocess.run(
                [
                    sys.executable,
                    str(TOOL),
                    "collect",
                    "--run-dir",
                    str(run_dir),
                    "--manifest",
                    str(manifest),
                ],
                cwd=ROOT,
                env={**os.environ, "PYTHONPATH": str(ROOT / "packages/percept-vision")},
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(run.returncode, 0)
            self.assertIn("selected runner has no pinned test cases", run.stderr)

    def test_collect_rejects_omitted_required_logical_track(self):
        with TemporaryDirectory() as d:
            root = Path(d)
            run_dir = root / "run"
            results = run_dir / "results"
            results.mkdir(parents=True)
            (run_dir / "run.json").write_text(
                json.dumps({"tracks": ["relational"], "live": False, "commit": "test"})
            )
            _result(
                results / "relational.json",
                "relational",
                ["relational-track-case-0"],
            )
            manifest = root / "MANIFEST.json"
            _manifest(
                manifest,
                {
                    "relational": ("relational-track", "relational", 1),
                    "counting": ("counting-track", "counting", 1),
                },
            )
            data = json.loads(manifest.read_text())
            next(row for row in data["tracks"] if row["id"] == "counting-track")[
                "required"
            ] = True
            manifest.write_text(json.dumps(data))
            run = subprocess.run(
                [
                    sys.executable,
                    str(TOOL),
                    "collect",
                    "--run-dir",
                    str(run_dir),
                    "--manifest",
                    str(manifest),
                ],
                cwd=ROOT,
                env={**os.environ, "PYTHONPATH": str(ROOT / "packages/percept-vision")},
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(run.returncode, 0)
            self.assertIn("omits required pinned-manifest track", run.stderr)


if __name__ == "__main__":
    unittest.main()
