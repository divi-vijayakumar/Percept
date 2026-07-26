"""Subprocess truthfulness checks for the OSS and live benchmark launch gates."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace as NS
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[3]
SDK = ROOT / "packages/percept-vision"


def _env(**updates: str) -> dict[str, str]:
    env = {**os.environ, "PYTHONPATH": str(SDK)}
    env.update(updates)
    return env


def _budget_plan_json(
    *,
    provider: str = "gemini",
    model: str = "gemini-test",
    media_minutes: float = 5,
) -> str:
    return json.dumps(
        {
            "version": 1,
            "provider": provider,
            "model": model,
            "region": "us",
            "price_snapshot": {
                "as_of": "2026-07-25",
                "currency": "USD",
                "source": "provider price sheet snapshot",
                "input_usd_per_million_tokens": 1.0,
                "output_usd_per_million_tokens": 2.0,
                "media_usd_per_minute": 0.1,
            },
            "caps": {
                "usd": 1.0,
                "requests": 10,
                "tokens": 10000,
                "media_minutes": media_minutes,
                "wall_clock_seconds": 300,
                "concurrency": 1,
                "request_timeout_seconds": 30,
            },
            "design": {
                "arms": ["gated", "matched-ungated"],
                "seeds": [1],
                "retries": 0,
                "cache": {"enabled": False, "policy": "disabled"},
            },
            "stages": [{"name": "smoke", "caps": {"requests": 2, "usd": 0.2}}],
            "early_stop": {
                "stop_on_cap": True,
                "max_error_rate": 0.1,
                "min_successful_cases": 1,
            },
            "approval": {
                "approved": True,
                "approved_by": "benchmark-owner",
                "approved_at": "2026-07-25T00:00:00Z",
                "scope": "test-only plan; no provider construction",
            },
            "enforcement": {
                "mode": "reserve_per_attempt",
                "stage": "smoke",
                "tokens_per_attempt": 1000,
                "usd_per_attempt": 0.1,
            },
        }
    )


class BenchmarkLaunchGateTests(unittest.TestCase):
    def test_openrouter_budget_plan_activates_hard_enforcement(self):
        from eval.bench.budget import live_budget_blockers, validate_budget_plan

        plan = validate_budget_plan(
            _budget_plan_json(
                provider="openrouter",
                model="qwen/qwen3-vl-32b-instruct",
            ),
            provider="openrouter",
            model="qwen/qwen3-vl-32b-instruct",
        )
        self.assertEqual(plan["caps"]["requests"], 10)
        blockers = live_budget_blockers(
            provider="openrouter",
            model="qwen/qwen3-vl-32b-instruct",
            raw=_budget_plan_json(
                provider="openrouter",
                model="qwen/qwen3-vl-32b-instruct",
            ),
        )
        self.assertEqual(blockers, [])

    def test_budget_plan_rejects_non_positive_caps(self):
        from eval.bench.budget import BudgetPlanError, validate_budget_plan

        raw = json.loads(_budget_plan_json())
        raw["caps"]["usd"] = 0
        with self.assertRaises(BudgetPlanError):
            validate_budget_plan(
                json.dumps(raw),
                provider="gemini",
                model="gemini-test",
            )

    def test_budget_plan_can_be_loaded_from_file(self):
        from eval.bench.budget import validate_budget_plan

        with TemporaryDirectory() as d:
            path = Path(d) / "budget-plan.json"
            path.write_text(_budget_plan_json())
            plan = validate_budget_plan(
                str(path),
                provider="gemini",
                model="gemini-test",
            )
            self.assertEqual(plan["approval"]["approved"], True)

    def test_offline_gating_benchmark_executes_and_measures_reduction(self):
        run = subprocess.run(
            [sys.executable, "-m", "percept.harness.bench_gating"],
            cwd=ROOT,
            env=_env(PERCEPT_MODE="live"),
            capture_output=True,
            text=True,
        )
        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
        self.assertIn("CALL-REDUCTION DIAGNOSTIC: PASS", run.stdout)
        self.assertIn("not a matched-quality/same-archetype comparison", run.stdout)
        self.assertIn("vision calls (gated)", run.stdout)
        self.assertIn("vision calls (ungated)", run.stdout)

    def test_scorecard_and_reproduce_write_nonempty_artifacts(self):
        with TemporaryDirectory() as d:
            scorecard = Path(d) / "scorecard.txt"
            args = [
                "--manifest",
                str(ROOT / "eval/golden-v1/MANIFEST.json"),
                "--out",
                str(scorecard),
            ]
            run = subprocess.run(
                [sys.executable, "-m", "percept.bench", "scorecard", "--offline", *args],
                cwd=ROOT,
                env=_env(PERCEPT_MODE="offline"),
                capture_output=True,
                text=True,
            )
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
            self.assertGreater(scorecard.stat().st_size, 0)
            self.assertIn("Cloud provider cost gate: BLOCKED", run.stdout)

            reproduced = Path(d) / "reproduced.txt"
            run = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "percept.bench",
                    "reproduce",
                    "--manifest",
                    str(ROOT / "eval/golden-v1/MANIFEST.json"),
                    "--baselines",
                    str(ROOT / "packages/percept-vision/eval/baselines.json"),
                    "--out",
                    str(reproduced),
                ],
                cwd=ROOT,
                env=_env(PERCEPT_MODE="offline"),
                capture_output=True,
                text=True,
            )
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
            self.assertGreater(reproduced.stat().st_size, 0)
            self.assertIn("REPRODUCTION: PASS", run.stdout)
            self.assertIn("OSS LAUNCH QUALIFICATION: BLOCKED", run.stdout)

    def test_live_scorecard_blocks_before_provider_call_without_key_or_budget(self):
        env = _env(
            PERCEPT_VISION_BACKEND="gemini",
            GEMINI_API_KEY="",
            GOOGLE_API_KEY="",
            GOOGLE_AISTUDIO_API_KEY="",
            PERCEPT_BENCH_BUDGET_PLAN="",
        )
        run = subprocess.run(
            [
                sys.executable,
                "-m",
                "percept.bench",
                "scorecard",
                "--live",
                "--manifest",
                str(ROOT / "eval/golden-v1/MANIFEST.json"),
            ],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
        )
        self.assertEqual(run.returncode, 2, run.stdout + run.stderr)
        self.assertIn("BENCHMARK LIVE GATE: BLOCKED", run.stdout)
        self.assertIn("credential missing", run.stdout)
        self.assertIn("budget plan invalid", run.stdout)

    def test_release_scorecard_blocks_on_absent_media_before_provider_call(self):
        with TemporaryDirectory() as d:
            manifest = Path(d) / "MANIFEST.json"
            manifest.write_text(
                json.dumps(
                    {
                        "version": "release-test",
                        "status": "release",
                        "tracks": [
                            {
                                "id": "x",
                                "source": "self",
                                "status": "ready",
                                "metrics": ["recall"],
                                "runners": [{"id": "watch", "result_track": "watch-e2e"}],
                            }
                        ],
                        "splits": {
                            "dev": [],
                            "test": [
                                {
                                    "id": "case",
                                    "source": "self-recorded",
                                    "license": "owned",
                                    "track": "x",
                                    "media_path": str(Path(d) / "missing.mp4"),
                                    "media_sha256": "a" * 64,
                                }
                            ],
                        },
                    }
                )
            )
            run = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "percept.bench",
                    "scorecard",
                    "--live",
                    "--require-release",
                    "--mode",
                    "full",
                    "--manifest",
                    str(manifest),
                ],
                cwd=ROOT,
                env=_env(
                    PERCEPT_VISION_BACKEND="gemini",
                    GEMINI_API_KEY="test-only-not-used",
                    PERCEPT_VISION_MODEL="gemini-test",
                    PERCEPT_BENCH_BUDGET_PLAN=_budget_plan_json(),
                ),
                capture_output=True,
                text=True,
            )
            self.assertEqual(run.returncode, 2, run.stdout + run.stderr)
            self.assertIn("BENCHMARK LIVE GATE: BLOCKED", run.stdout)
            self.assertIn("release media missing/empty", run.stdout)
            self.assertIn("hard spend enforcement is unavailable", run.stdout)

    def test_qwen_n20_preflight_reports_pinned_capacity_without_provider_call(self):
        model = "qwen/qwen3-vl-32b-instruct"
        run = subprocess.run(
            [
                sys.executable,
                "tools/percept_benchmark.py",
                "run",
                "--live",
                "--tracks",
                "pt-action-e2e",
                "--n",
                "20",
                "--workers",
                "1",
                "--attempts",
                "1",
                "--manifest",
                str(ROOT / "eval/golden-v1/MANIFEST.json"),
                "--no-prefetch-pose",
            ],
            cwd=ROOT,
            env=_env(
                PERCEPT_VISION_BACKEND="openrouter",
                PERCEPT_VISION_MODEL=model,
                OPENROUTER_API_KEY="offline-presence-check-only",
                OPENROUTER_KEY="",
                PERCEPT_BENCH_DATA_DIR=str(ROOT / ".missing-benchmark-test-cache"),
                PERCEPT_BENCH_BUDGET_PLAN=_budget_plan_json(
                    provider="openrouter",
                    model=model,
                ),
            ),
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertNotEqual(run.returncode, 0, run.stdout + run.stderr)
        output = run.stdout + run.stderr
        self.assertIn(
            "requested N=20, but only 0 of 20 licensed pinned cases "
            "have verified local annotations/media",
            output,
        )
        self.assertNotIn("provider response", output)

    def test_oversized_selected_media_blocks_before_provider_construction(self):
        from eval.bench.run import _live_prerequisite_blockers
        from percept import registry

        model = "qwen/qwen3-vl-32b-instruct"
        with (
            patch.dict(
                os.environ,
                {
                    "PERCEPT_VISION_BACKEND": "openrouter",
                    "PERCEPT_VISION_MODEL": model,
                    "OPENROUTER_API_KEY": "offline-presence-check-only",
                    "PERCEPT_BENCH_BUDGET_PLAN": _budget_plan_json(
                        provider="openrouter",
                        model=model,
                        media_minutes=1,
                    ),
                },
                clear=False,
            ),
            patch(
                "eval.bench.perception_test.golden_action_capacity",
                return_value={"licensed": 2, "verified_local": 2},
            ),
            patch(
                "eval.bench.perception_test.select_golden_action_events",
                return_value=[{"path": Path("/offline/a.mp4")}],
            ),
            patch(
                "eval.bench.perception_test.action_media_minutes",
                return_value=1.25,
            ),
            patch.object(registry, "make_vision") as provider_factory,
        ):
            blockers = _live_prerequisite_blockers(
                manifest="offline-manifest.json",
                release=False,
                n=1,
                tracks=["pt-action-e2e"],
            )
        self.assertIn(
            "selected pinned media is 1.250000 minutes, "
            "exceeding the effective media cap of 1.000000 minutes",
            blockers,
        )
        provider_factory.assert_not_called()

    def test_non_finite_media_metadata_blocks_before_provider_construction(self):
        from eval.bench import perception_test
        from eval.bench.run import _live_prerequisite_blockers
        from percept import registry

        class Capture:
            def isOpened(self):
                return True

            def get(self, field):
                if field == 1:
                    return float("nan")
                return 30.0

            def release(self):
                pass

        fake_cv2 = NS(
            CAP_PROP_FPS=1,
            CAP_PROP_FRAME_COUNT=2,
            VideoCapture=lambda _path: Capture(),
        )
        model = "qwen/qwen3-vl-32b-instruct"
        perception_test.media_duration_s.cache_clear()
        try:
            with (
                patch.dict(
                    os.environ,
                    {
                        "PERCEPT_VISION_BACKEND": "openrouter",
                        "PERCEPT_VISION_MODEL": model,
                        "OPENROUTER_API_KEY": "offline-presence-check-only",
                        "PERCEPT_BENCH_BUDGET_PLAN": _budget_plan_json(
                            provider="openrouter",
                            model=model,
                        ),
                    },
                    clear=False,
                ),
                patch.dict(sys.modules, {"cv2": fake_cv2}),
                patch(
                    "eval.bench.perception_test.golden_action_capacity",
                    return_value={"licensed": 1, "verified_local": 1},
                ),
                patch(
                    "eval.bench.perception_test.select_golden_action_events",
                    return_value=[{"path": Path("/offline/nan.mp4")}],
                ),
                patch.object(registry, "make_vision") as provider_factory,
            ):
                blockers = _live_prerequisite_blockers(
                    manifest="offline-manifest.json",
                    release=False,
                    n=1,
                    tracks=["pt-action-e2e"],
                )
        finally:
            perception_test.media_duration_s.cache_clear()
        self.assertTrue(
            any("invalid duration metadata" in blocker for blocker in blockers),
            blockers,
        )
        provider_factory.assert_not_called()

    def test_eval_live_writes_blocked_status_without_prerequisites(self):
        with TemporaryDirectory() as d:
            artifact = Path(d) / "results.json"
            run = subprocess.run(
                [sys.executable, "-m", "eval.run"],
                cwd=ROOT,
                env=_env(
                    PERCEPT_LIVE="1",
                    PERCEPT_VISION_BACKEND="gemini",
                    GEMINI_API_KEY="",
                    GOOGLE_API_KEY="",
                    GOOGLE_AISTUDIO_API_KEY="",
                    PERCEPT_BENCH_BUDGET_PLAN="",
                    PERCEPT_EVAL_OUT=str(artifact),
                ),
                capture_output=True,
                text=True,
            )
            self.assertEqual(run.returncode, 2, run.stdout + run.stderr)
            self.assertEqual(json.loads(artifact.read_text())["status"], "BLOCKED")
            self.assertIn("L2 STATUS: BLOCKED", run.stderr)


if __name__ == "__main__":
    unittest.main()
