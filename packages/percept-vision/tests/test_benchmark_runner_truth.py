"""Collection acceptance is derived from pinned manifest tracks and case IDs."""

from __future__ import annotations

import hashlib
import json
import os
import runpy
import subprocess
import sys
import unittest
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace as NS
from unittest.mock import patch


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
    def test_launch_defaults_select_only_the_runtime_v2_track(self):
        namespace = runpy.run_path(
            str(TOOL),
            run_name="benchmark_launch_defaults_test",
        )

        args = namespace["build_parser"]().parse_args(["run"])

        self.assertEqual(namespace["SAFE_LIVE_TRACKS"], ("pt-action-e2e",))
        self.assertEqual(args.tracks, "pt-action-e2e")
        self.assertEqual(args.n, 1)
        self.assertEqual(
            namespace["_resolve_tracks"]("stable"),
            ["pt-action-e2e"],
        )

    def test_live_runner_requires_an_exact_installed_wheel(self):
        namespace = runpy.run_path(
            str(TOOL),
            run_name="benchmark_wheel_required_test",
        )
        args = namespace["build_parser"]().parse_args(["run", "--live"])
        self.assertIsNone(args.sdk_wheel)
        with self.assertRaisesRegex(
            SystemExit,
            "--sdk-wheel is required",
        ):
            namespace["_installed_sdk_artifact"](
                args.sdk_wheel,
                required=True,
            )

    def test_wheel_identity_reads_canonical_metadata_and_build_stamp(self):
        namespace = runpy.run_path(
            str(TOOL),
            run_name="benchmark_wheel_metadata_test",
        )
        with TemporaryDirectory() as d:
            wheel = Path(d) / "percept_vision-1.0.0-py3-none-any.whl"
            build_info = "\n".join(
                [
                    "PACKAGE_VERSION = '1.0.0'",
                    f"GIT_COMMIT = '{'a' * 40}'",
                    "CONTRACT_VERSION = '1.10.0'",
                    "BUILD_TIMESTAMP = '2026-07-26T00:00:00+00:00'",
                ]
            )
            with zipfile.ZipFile(wheel, "w") as archive:
                archive.writestr("percept/_build_info.py", build_info)
                archive.writestr(
                    "percept_vision-1.0.0.dist-info/METADATA",
                    "Metadata-Version: 2.4\nName: percept-vision\nVersion: 1.0.0\n",
                )
                archive.writestr(
                    "percept_vision-1.0.0.dist-info/RECORD",
                    "percept_vision-1.0.0.dist-info/RECORD,,\n",
                )

            values, record_name, rows = namespace["_wheel_build_info"](wheel)

        self.assertEqual(values["GIT_COMMIT"], "a" * 40)
        self.assertEqual(values["PACKAGE_VERSION"], "1.0.0")
        self.assertEqual(
            record_name,
            "percept_vision-1.0.0.dist-info/RECORD",
        )
        self.assertEqual(len(rows), 1)

    def test_wheel_identity_rejects_metadata_build_version_mismatch(self):
        namespace = runpy.run_path(
            str(TOOL),
            run_name="benchmark_wheel_metadata_mismatch_test",
        )
        with TemporaryDirectory() as d:
            wheel = Path(d) / "percept_vision-1.0.0-py3-none-any.whl"
            build_info = "\n".join(
                [
                    "PACKAGE_VERSION = '1.0.0'",
                    f"GIT_COMMIT = '{'b' * 40}'",
                    "CONTRACT_VERSION = '1.10.0'",
                    "BUILD_TIMESTAMP = '2026-07-26T00:00:00+00:00'",
                ]
            )
            with zipfile.ZipFile(wheel, "w") as archive:
                archive.writestr("percept/_build_info.py", build_info)
                archive.writestr(
                    "percept_vision-1.0.0.dist-info/METADATA",
                    "Metadata-Version: 2.4\nName: percept-vision\nVersion: 9.9.9\n",
                )
                archive.writestr(
                    "percept_vision-1.0.0.dist-info/RECORD",
                    "percept_vision-1.0.0.dist-info/RECORD,,\n",
                )

            with self.assertRaisesRegex(
                SystemExit,
                "METADATA identity",
            ):
                namespace["_wheel_build_info"](wheel)

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

    def test_collect_uses_recorded_live_provenance_not_ambient_environment(self):
        with TemporaryDirectory() as d:
            root = Path(d)
            run_dir = root / "run"
            results = run_dir / "results"
            results.mkdir(parents=True)
            model = "qwen/qwen3-vl-32b-instruct"
            (run_dir / "run.json").write_text(
                json.dumps(
                    {
                        "tracks": ["pt-action-e2e"],
                        "live": True,
                        "commit": "test",
                        "n": 1,
                        "vision": {
                            "provider": "openrouter",
                            "model": model,
                        },
                        "budget_plan": {
                            "source": "qwen-n1-budget.json",
                            "canonical_sha256": "a" * 64,
                            "identity": {
                                "version": 1,
                                "provider": "openrouter",
                                "model": model,
                                "region": "us",
                                "approval_scope": "N=1",
                            },
                        },
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
                        1,
                    )
                },
            )
            out = root / "scorecard.txt"
            env = {
                **os.environ,
                "PYTHONPATH": str(ROOT / "packages/percept-vision"),
                "PERCEPT_VISION_BACKEND": "gemini",
                "PERCEPT_VISION_MODEL": "ambient-model-must-not-appear",
            }
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
                    "--out",
                    str(out),
                ],
                cwd=ROOT,
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
            rendered = out.read_text()
            self.assertIn(f"backbone: openrouter / {model}", rendered)
            self.assertIn(
                f"budget-plan: qwen-n1-budget.json canonical-sha256:{'a' * 64}",
                rendered,
            )
            self.assertNotIn("ambient-model-must-not-appear", rendered)

    def test_collect_labels_legacy_live_provenance_unknown_instead_of_guessing(self):
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
                        1,
                    )
                },
            )
            out = root / "scorecard.txt"
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
                    "--out",
                    str(out),
                ],
                cwd=ROOT,
                env={
                    **os.environ,
                    "PYTHONPATH": str(ROOT / "packages/percept-vision"),
                    "PERCEPT_VISION_MODEL": "ambient-model-must-not-appear",
                },
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
            rendered = out.read_text()
            self.assertIn(
                "backbone: unknown (legacy live run metadata did not record this value)",
                rendered,
            )
            self.assertNotIn("ambient-model-must-not-appear", rendered)

    def test_v2_result_reuse_requires_matching_identity_and_checksum(self):
        namespace = runpy.run_path(
            str(TOOL),
            run_name="benchmark_result_reuse_test",
        )
        existing_result_ok = namespace["_existing_result_ok"]
        checksum_path = namespace["_result_checksum_path"]
        with TemporaryDirectory() as d:
            result_path = Path(d) / "edge.json"
            _result(result_path, "edge-recall", ["case-1"])
            payload = json.loads(result_path.read_text())
            payload["configuration"] = {
                "benchmark_artifact": {
                    "schema_version": "percept.benchmark.track_artifact/2",
                    "run_identity_sha256": "a" * 64,
                    "runner": "edge",
                    "live": False,
                    "n": 1,
                    "sdk_artifact": {
                        "kind": "wheel",
                        "sha256": "c" * 64,
                    },
                }
            }
            result_path.write_text(json.dumps(payload))
            digest = hashlib.sha256(result_path.read_bytes()).hexdigest()
            checksum_path(result_path).write_text(
                f"{digest}  {result_path.name}\n"
            )

            self.assertTrue(
                existing_result_ok(
                    result_path,
                    run_identity="a" * 64,
                    runner="edge",
                    sdk_artifact={
                        "kind": "wheel",
                        "sha256": "c" * 64,
                    },
                )
            )
            self.assertFalse(
                existing_result_ok(
                    result_path,
                    run_identity="b" * 64,
                    runner="edge",
                    sdk_artifact={
                        "kind": "wheel",
                        "sha256": "c" * 64,
                    },
                )
            )
            self.assertFalse(
                existing_result_ok(
                    result_path,
                    run_identity="a" * 64,
                    runner="edge",
                    sdk_artifact={
                        "kind": "wheel",
                        "sha256": "d" * 64,
                    },
                )
            )
            result_path.write_text(result_path.read_text() + "\n")
            self.assertFalse(
                existing_result_ok(
                    result_path,
                    run_identity="a" * 64,
                    runner="edge",
                    sdk_artifact={
                        "kind": "wheel",
                        "sha256": "c" * 64,
                    },
                )
            )

    def test_track_worker_rejects_wheel_changed_after_parent_preflight(self):
        namespace = runpy.run_path(
            str(TOOL),
            run_name="benchmark_worker_wheel_change_test",
        )
        with TemporaryDirectory() as d:
            root = Path(d)
            result_path = root / "result.json"
            manifest = root / "manifest.json"
            manifest.write_text("{}")
            args = NS(
                live=False,
                track="edge",
                manifest=str(manifest),
                n=1,
                out=str(result_path),
                run_identity="a" * 64,
                sdk_wheel=str(root / "candidate.whl"),
                sdk_artifact_identity="b" * 64,
            )
            observed = {
                "kind": "wheel",
                "sha256": "c" * 64,
            }
            with patch.dict(
                namespace["cmd_run_track"].__globals__,
                {"_installed_sdk_artifact": lambda *_args, **_kwargs: observed},
            ):
                with self.assertRaisesRegex(
                    SystemExit,
                    "changed after run preflight",
                ):
                    namespace["cmd_run_track"](args)
            self.assertFalse(result_path.exists())

    def test_background_command_forwards_exact_wheel_path(self):
        namespace = runpy.run_path(
            str(TOOL),
            run_name="benchmark_background_wheel_test",
        )
        wheel = "/private/tmp/percept/exact.whl"
        args = namespace["build_parser"]().parse_args(
            ["run", "--background", "--sdk-wheel", wheel]
        )
        command = namespace["_background_cmd"](args, "run-1")
        index = command.index("--sdk-wheel")
        self.assertEqual(command[index + 1], wheel)

    def test_make_launch_command_requires_explicit_wheel_and_interpreter(self):
        makefile = (ROOT / "Makefile").read_text()
        self.assertIn("BENCH_PY ?= $(PY)", makefile)
        self.assertIn("SDK_WHEEL ?=", makefile)
        self.assertIn("$(BENCH_PY) tools/percept_benchmark.py run", makefile)
        self.assertIn("$(SDK_WHEEL_FLAG)", makefile)

    def test_run_track_returns_nonzero_for_blocked_result_and_writes_checksum(self):
        from eval.bench.types import TrackResult

        namespace = runpy.run_path(
            str(TOOL),
            run_name="benchmark_terminal_status_test",
        )
        with TemporaryDirectory() as d:
            root = Path(d)
            result_path = root / "blocked.json"
            manifest = root / "manifest.json"
            manifest.write_text("{}")
            module = NS(
                score=lambda **_kwargs: TrackResult(
                    track="edge-recall",
                    cells={("edge", "candidate_recall"): None},
                    status="blocked",
                )
            )
            args = NS(
                live=False,
                track="edge",
                manifest=str(manifest),
                n=1,
                out=str(result_path),
                run_identity="c" * 64,
            )
            with patch.object(
                namespace["importlib"],
                "import_module",
                return_value=module,
            ):
                rc = namespace["cmd_run_track"](args)

            self.assertEqual(rc, 2)
            self.assertTrue(result_path.is_file())
            self.assertTrue(namespace["_valid_result_checksum"](result_path))
            payload = json.loads(result_path.read_text())
            self.assertEqual(payload["status"], "blocked")
            self.assertEqual(
                payload["configuration"]["benchmark_artifact"][
                    "run_identity_sha256"
                ],
                "c" * 64,
            )

    def test_resume_identity_rejects_changed_manifest_or_code(self):
        namespace = runpy.run_path(
            str(TOOL),
            run_name="benchmark_resume_identity_test",
        )
        identity = {
            "schema_version": "percept.benchmark.run/2",
            "source": {"commit": "abc", "dirty": False},
            "manifest": {"path": "/tmp/manifest", "sha256": "a" * 64},
            "live": False,
            "n": 1,
            "tracks": ["edge"],
            "mode": "sampled",
            "fps": "1",
            "device": "local",
            "seeds": 1,
            "vision": {"provider": "fake", "model": "scripted"},
            "budget_plan": None,
        }
        existing = {
            "schema_version": "percept.benchmark.run/2",
            "qualification_identity": identity,
            "qualification_identity_sha256": namespace[
                "_canonical_sha256"
            ](identity),
        }
        namespace["_assert_resume_identity"](existing, dict(identity))
        changed = json.loads(json.dumps(identity))
        changed["source"]["commit"] = "def"
        with self.assertRaisesRegex(
            SystemExit,
            "changed qualification identity: source",
        ):
            namespace["_assert_resume_identity"](existing, changed)

    def test_release_identity_requires_and_reconciles_exact_sdk_artifact(self):
        namespace = runpy.run_path(
            str(TOOL),
            run_name="benchmark_release_wheel_identity_test",
        )
        with TemporaryDirectory() as d:
            manifest = Path(d) / "MANIFEST.json"
            manifest.write_text("{}")
            sdk_artifact = {
                "kind": "wheel",
                "sha256": "d" * 64,
                "git_commit": "a" * 40,
            }
            identity = {
                "schema_version": "percept.benchmark.run/2",
                "source": {"commit": "a" * 40, "dirty": False},
                "manifest": {
                    "path": str(manifest.resolve()),
                    "sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
                },
                "live": True,
                "n": 1,
                "tracks": ["pt-action-e2e"],
                "mode": "sampled",
                "fps": "1",
                "device": "local",
                "seeds": 1,
                "vision": {
                    "provider": "openrouter",
                    "model": "qwen/test",
                },
                "budget_plan": {"identity": "test"},
                "sdk_artifact": sdk_artifact,
            }
            meta = {
                "schema_version": "percept.benchmark.run/2",
                "commit": "a" * 40,
                "live": True,
                "n": 1,
                "tracks": ["pt-action-e2e"],
                "mode": "sampled",
                "fps": "1",
                "device": "local",
                "seeds": 1,
                "vision": identity["vision"],
                "budget_plan": identity["budget_plan"],
                "sdk_artifact": sdk_artifact,
                "qualification_identity": identity,
                "qualification_identity_sha256": namespace[
                    "_canonical_sha256"
                ](identity),
            }

            self.assertEqual(
                namespace["_validated_run_identity"](
                    meta,
                    manifest=manifest,
                    require_release=True,
                ),
                meta["qualification_identity_sha256"],
            )
            changed = dict(meta)
            changed["sdk_artifact"] = {
                **sdk_artifact,
                "sha256": "e" * 64,
            }
            with self.assertRaisesRegex(
                SystemExit,
                "sdk_artifact",
            ):
                namespace["_validated_run_identity"](
                    changed,
                    manifest=manifest,
                    require_release=True,
                )

    def test_preregistered_acceptance_rules_use_typed_measurements(self):
        from eval.bench.types import Measurement, TrackResult

        namespace = runpy.run_path(
            str(TOOL),
            run_name="benchmark_acceptance_gate_test",
        )
        result = TrackResult(
            track="pt-action-e2e",
            measurements={
                "occurrence_recall": Measurement(
                    value=0.6,
                    unit="ratio",
                    conditioning="all positive occurrences",
                    numerator=6,
                    denominator=10,
                ),
                "max_action_onset_to_incident": Measurement(
                    value=2.2,
                    unit="seconds",
                    conditioning="matched positive occurrences",
                    denominator=6,
                ),
            },
        )
        namespace["_apply_acceptance_gates"](
            result,
            {
                "occurrence_recall": {
                    "operator": ">=",
                    "threshold": 0.59,
                    "basis": "reviewed ladder",
                },
                "max_action_onset_to_incident": {
                    "operator": "<=",
                    "threshold": 2.0,
                    "basis": "public reflex SLO",
                },
                "missing_measurement": {
                    "operator": ">=",
                    "threshold": 0.1,
                    "basis": "reviewed ladder",
                },
            },
        )

        gates = {gate["id"]: gate for gate in result.quality_gates}
        self.assertEqual(
            gates["acceptance:occurrence_recall"]["status"],
            "pass",
        )
        self.assertEqual(
            gates["acceptance:max_action_onset_to_incident"]["status"],
            "fail",
        )
        self.assertEqual(
            gates["acceptance:missing_measurement"]["status"],
            "fail",
        )

    def test_collect_rejects_live_provider_budget_identity_mismatch(self):
        with TemporaryDirectory() as d:
            root = Path(d)
            run_dir = root / "run"
            results = run_dir / "results"
            results.mkdir(parents=True)
            model = "qwen/qwen3-vl-32b-instruct"
            (run_dir / "run.json").write_text(
                json.dumps(
                    {
                        "tracks": ["pt-action-e2e"],
                        "live": True,
                        "commit": "test",
                        "n": 1,
                        "vision": {
                            "provider": "openrouter",
                            "model": model,
                        },
                        "budget_plan": {
                            "source": "budget.json",
                            "canonical_sha256": "b" * 64,
                            "identity": {
                                "provider": "gemini",
                                "model": model,
                            },
                        },
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
                        1,
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
            self.assertNotEqual(run.returncode, 0)
            self.assertIn(
                "provider/model does not match budget_plan identity",
                run.stderr,
            )

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
