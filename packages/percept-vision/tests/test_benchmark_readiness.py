"""Offline qualification for paid benchmark admission, case identity, and latency truth."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import unittest
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace as NS
from unittest.mock import patch

from eval.bench import perception_test as pt
from eval.bench import track_perception_test_e2e as pt_track
from eval.bench.budget import (
    BudgetExceeded,
    BudgetPlanError,
    build_budget_enforcer,
    validate_budget_plan,
)
from eval.bench.track_perception_test_e2e import (
    ACTION_BRIEF_WINDOW_S,
    ACTION_TEMPORAL_FRAMES,
    ACTION_TEMPORAL_SAMPLING,
    ACTION_TEMPORAL_WINDOW_S,
    EventRun,
    _action_goal_state,
    _call_reductions,
    _decision_observation,
    _event_precision,
    _fire_records,
    _median,
    _precision,
    admission_cause,
)
from percept.adapters import Verdict
from percept.backends import openrouter
from percept.backends.openrouter import OpenRouterVision
from percept.contracts import JudgmentTiming


def _plan(
    *,
    requests: int = 4,
    tokens: int = 4000,
    usd: float = 0.4,
    tokens_per_attempt: int = 1000,
    usd_per_attempt: float = 0.1,
    retries: int = 1,
) -> str:
    return json.dumps(
        {
            "version": 1,
            "provider": "openrouter",
            "model": "qwen/qwen3-vl-32b-instruct",
            "region": "us",
            "price_snapshot": {
                "as_of": "2026-07-25",
                "currency": "USD",
                "source": "approved snapshot",
                "input_usd_per_million_tokens": 1.0,
                "output_usd_per_million_tokens": 2.0,
                "media_usd_per_minute": 0.0,
            },
            "caps": {
                "usd": usd,
                "requests": requests,
                "tokens": tokens,
                "media_minutes": 1,
                "wall_clock_seconds": 60,
                "concurrency": 1,
                "request_timeout_seconds": 5,
            },
            "design": {
                "arms": ["gated", "matched"],
                "seeds": [1],
                "retries": retries,
                "cache": {"enabled": False, "policy": "disabled"},
            },
            "stages": [
                {
                    "name": "canary",
                    "caps": {"requests": requests, "tokens": tokens, "usd": usd},
                }
            ],
            "early_stop": {
                "stop_on_cap": True,
                "max_error_rate": 1.0,
                "min_successful_cases": 1,
            },
            "approval": {
                "approved": True,
                "approved_by": "test",
                "approved_at": "2026-07-25T00:00:00Z",
                "scope": "offline synthetic test",
            },
            "enforcement": {
                "mode": "reserve_per_attempt",
                "stage": "canary",
                "tokens_per_attempt": tokens_per_attempt,
                "usd_per_attempt": usd_per_attempt,
            },
        }
    )


def _enforcer(**kwargs):
    return build_budget_enforcer(
        _plan(**kwargs),
        provider="openrouter",
        model="qwen/qwen3-vl-32b-instruct",
    )


class ActionBenchmarkConfigurationTests(unittest.TestCase):
    def test_action_watch_is_temporal_occurrence_aware_and_annotation_independent(self):
        # The helper deliberately accepts no event/annotation argument. Every selected case gets
        # exactly this preregistered product configuration; ground-truth interval duration is scoring
        # data only and cannot leak into admission or temporal-window width.
        state = _action_goal_state()
        self.assertEqual(state["rearm"], "each_event")
        self.assertEqual(state["sense"]["brief_window_s"], ACTION_BRIEF_WINDOW_S)
        self.assertEqual(
            state["perception"],
            {
                "temporal": "strip",
                "n_frames": ACTION_TEMPORAL_FRAMES,
                "sampling": ACTION_TEMPORAL_SAMPLING,
                "window_s": ACTION_TEMPORAL_WINDOW_S,
            },
        )
        self.assertGreaterEqual(ACTION_TEMPORAL_FRAMES, 3)


class BudgetAdmissionTests(unittest.TestCase):
    def test_non_finite_plan_numbers_are_rejected(self):
        for value in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(value=value):
                plan = json.loads(_plan())
                plan["caps"]["usd"] = value
                with self.assertRaisesRegex(BudgetPlanError, "caps.usd must be positive"):
                    validate_budget_plan(
                        json.dumps(plan),
                        provider="openrouter",
                        model="qwen/qwen3-vl-32b-instruct",
                    )

    def test_request_cap_stops_before_a_second_transport(self):
        budget = _enforcer(requests=1)
        first = budget.begin_attempt()
        budget.fail_attempt(first)
        with self.assertRaisesRegex(BudgetExceeded, "request budget exhausted"):
            budget.begin_attempt()
        self.assertEqual(budget.snapshot()["physical_attempts"], 1)

    def test_failed_attempt_retains_token_reservation(self):
        budget = _enforcer(
            requests=2,
            tokens=1000,
            usd=1.0,
            tokens_per_attempt=1000,
            usd_per_attempt=0.1,
        )
        first = budget.begin_attempt()
        budget.fail_attempt(first)
        with self.assertRaisesRegex(BudgetExceeded, "token budget exhausted"):
            budget.begin_attempt()

    def test_failed_attempt_retains_usd_reservation(self):
        budget = _enforcer(
            requests=2,
            tokens=10000,
            usd=0.1,
            tokens_per_attempt=1000,
            usd_per_attempt=0.1,
        )
        first = budget.begin_attempt()
        budget.fail_attempt(first)
        with self.assertRaisesRegex(BudgetExceeded, "USD budget exhausted"):
            budget.begin_attempt()

    def test_oversized_metered_attempt_is_recorded_then_blocks(self):
        budget = _enforcer(
            requests=2,
            tokens=2000,
            usd=0.2,
            tokens_per_attempt=1000,
            usd_per_attempt=0.1,
        )
        reservation = budget.begin_attempt()
        with self.assertRaisesRegex(BudgetExceeded, "exceeded the approved"):
            budget.complete_attempt(
                reservation,
                {"input_tokens": 900, "output_tokens": 200, "cost_usd": 0.11},
            )
        snapshot = budget.snapshot()
        self.assertEqual(snapshot["committed_tokens"], 1100)
        self.assertAlmostEqual(snapshot["billed_usd"], 0.11)
        with self.assertRaisesRegex(BudgetExceeded, "token budget exhausted"):
            budget.begin_attempt()

    def test_non_finite_billed_cost_is_rejected_and_reservation_retained(self):
        for value in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(value=value):
                budget = _enforcer()
                reservation = budget.begin_attempt()
                with self.assertRaisesRegex(BudgetExceeded, "invalid token or billed-cost"):
                    budget.complete_attempt(
                        reservation,
                        {
                            "input_tokens": 20,
                            "output_tokens": 10,
                            "cost_usd": value,
                        },
                    )
                snapshot = budget.snapshot()
                self.assertEqual(snapshot["failed_attempts"], 1)
                self.assertEqual(snapshot["metered_attempts"], 0)
                self.assertEqual(snapshot["committed_tokens"], 1000)
                self.assertEqual(snapshot["committed_usd"], 0.1)
                self.assertEqual(snapshot["billed_usd"], 0.0)

    def test_late_response_fails_reconciliation_and_retains_reservation(self):
        now = 0.0

        def clock():
            return now

        budget = build_budget_enforcer(
            _plan(),
            provider="openrouter",
            model="qwen/qwen3-vl-32b-instruct",
            clock=clock,
        )
        reservation = budget.begin_attempt()
        now = 61.0
        with self.assertRaisesRegex(BudgetExceeded, "wall-clock budget exhausted"):
            budget.complete_attempt(
                reservation,
                {"input_tokens": 20, "output_tokens": 10, "cost_usd": 0.01},
            )
        snapshot = budget.snapshot()
        self.assertEqual(snapshot["failed_attempts"], 1)
        self.assertEqual(snapshot["metered_attempts"], 0)
        self.assertEqual(snapshot["committed_tokens"], 1000)
        self.assertEqual(snapshot["committed_usd"], 0.1)
        self.assertEqual(snapshot["billed_usd"], 0.0)

    def test_openrouter_counts_each_retry_as_a_physical_attempt(self):
        class FlakyCompletions:
            def __init__(self) -> None:
                self.calls = 0

            async def create(self, **_kwargs):
                self.calls += 1
                if self.calls == 1:
                    raise RuntimeError("synthetic retry")
                return NS(
                    choices=[
                        NS(
                            message=NS(
                                content='{"satisfied": true, "confidence": 0.9, "evidence": "seen"}'
                            )
                        )
                    ],
                    usage=NS(prompt_tokens=20, completion_tokens=10, cost=0.01),
                )

        completions = FlakyCompletions()
        client = NS(chat=NS(completions=completions))
        budget = _enforcer(retries=1)
        vision = OpenRouterVision(
            NS(models=NS(vision_model="qwen/qwen3-vl-32b-instruct")),
            client=client,
            request_budget=budget,
        )
        verdict = asyncio.run(vision.judge("action?", b"frame"))
        self.assertTrue(verdict.satisfied)
        self.assertEqual(completions.calls, 2)
        snapshot = budget.snapshot()
        self.assertEqual(snapshot["physical_attempts"], 2)
        self.assertEqual(snapshot["failed_attempts"], 1)
        self.assertEqual(snapshot["metered_attempts"], 1)
        self.assertEqual(snapshot["input_tokens"], 20)
        self.assertEqual(snapshot["output_tokens"], 10)
        self.assertAlmostEqual(snapshot["billed_usd"], 0.01)

    def test_benchmark_adapter_disables_sdk_retry_layer(self):
        captured: dict = {}

        def client_factory(**kwargs):
            captured.update(kwargs)
            return NS()

        with (
            patch.object(openrouter, "_require_openai", return_value=client_factory),
            patch.dict(
                os.environ,
                {"OPENROUTER_API_KEY": "offline-construction-check"},
                clear=False,
            ),
        ):
            vision = OpenRouterVision(
                NS(models=NS(vision_model="qwen/qwen3-vl-32b-instruct")),
                request_budget=_enforcer(),
            )
            vision._ensure_client()
        self.assertEqual(captured["max_retries"], 0)

    def test_generic_adapter_preserves_sdk_default_retry_behavior(self):
        captured: dict = {}

        def client_factory(**kwargs):
            captured.update(kwargs)
            return NS()

        with (
            patch.object(openrouter, "_require_openai", return_value=client_factory),
            patch.dict(
                os.environ,
                {"OPENROUTER_API_KEY": "offline-construction-check"},
                clear=False,
            ),
        ):
            vision = OpenRouterVision(
                NS(models=NS(vision_model="qwen/qwen3-vl-32b-instruct")),
            )
            vision._ensure_client()
        self.assertNotIn("max_retries", captured)

    def test_unmetered_response_is_not_accepted(self):
        response = NS(
            choices=[
                NS(message=NS(content='{"satisfied": true, "confidence": 0.9, "evidence": "seen"}'))
            ]
        )
        calls = 0

        async def create(**_kwargs):
            nonlocal calls
            calls += 1
            return response

        completions = NS(create=create)
        client = NS(chat=NS(completions=completions))
        budget = _enforcer(requests=2, retries=1)
        vision = OpenRouterVision(
            NS(models=NS(vision_model="qwen/qwen3-vl-32b-instruct")),
            client=client,
            request_budget=budget,
        )
        with self.assertRaisesRegex(BudgetExceeded, "omitted enforceable"):
            asyncio.run(vision.judge("action?", b"frame"))
        self.assertEqual(calls, 1)
        self.assertEqual(budget.snapshot()["physical_attempts"], 1)

    def test_pre_attempt_cap_exhaustion_never_retries_transport(self):
        calls = 0

        async def create(**_kwargs):
            nonlocal calls
            calls += 1
            raise RuntimeError("synthetic transport failure")

        budget = _enforcer(requests=1, retries=2)
        vision = OpenRouterVision(
            NS(models=NS(vision_model="qwen/qwen3-vl-32b-instruct")),
            client=NS(chat=NS(completions=NS(create=create))),
            request_budget=budget,
        )
        with self.assertRaisesRegex(BudgetExceeded, "request budget exhausted"):
            asyncio.run(vision.judge("action?", b"frame"))
        self.assertEqual(calls, 1)
        self.assertEqual(budget.snapshot()["physical_attempts"], 1)


class TerminalBenchmarkFailureTests(unittest.TestCase):
    class _AlwaysGate:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def evaluate(self, _frame, *, ts):
            return NS(
                escalate=True,
                signal=NS(motion=1.0, structural=0.0, semantic=0.0),
                threshold=0.1,
                ts=ts,
            )

    @staticmethod
    def _events() -> list[dict]:
        return [
            {
                "row_id": f"case-{index}",
                "video_id": f"video-{index}",
                "label": "Turning something on",
                "onset_s": 0.0,
                "end_s": 1.0,
                "positive_windows": [[0.0, 1.0]],
                "path": Path(f"/offline/video-{index}.mp4"),
            }
            for index in (1, 2)
        ]

    def _run_terminal_failure(
        self,
        *,
        requests: int,
        retries: int,
        unmetered_response: bool = False,
    ) -> tuple[object, int, int, int]:
        transport_calls = 0
        source_starts = 0
        decoded_frames = 0

        async def create(**_kwargs):
            nonlocal transport_calls
            transport_calls += 1
            if unmetered_response:
                return NS(
                    choices=[
                        NS(
                            message=NS(
                                content='{"satisfied": true, "confidence": 0.9, "evidence": "seen"}'
                            )
                        )
                    ],
                )
            raise RuntimeError("synthetic exhausted transport")

        client = NS(chat=NS(completions=NS(create=create)))

        def make_vision(settings, *, request_budget):
            return OpenRouterVision(
                settings,
                client=client,
                request_budget=request_budget,
            )

        def source(*_args, **_kwargs):
            nonlocal source_starts
            source_starts += 1

            async def frames():
                nonlocal decoded_frames
                for timestamp in (0.0, 0.5):
                    decoded_frames += 1
                    yield timestamp, object(), b"frame"

            return frames()

        env = {
            "PERCEPT_VISION_BACKEND": "openrouter",
            "PERCEPT_VISION_MODEL": "qwen/qwen3-vl-32b-instruct",
            "PERCEPT_BENCH_BUDGET_PLAN": _plan(
                requests=requests,
                retries=retries,
            ),
        }
        with (
            patch.dict(os.environ, env, clear=False),
            patch.object(
                pt_track.pt,
                "select_golden_action_events",
                return_value=self._events(),
            ),
            patch.object(
                pt_track.pt,
                "action_media_minutes",
                return_value=0.25,
            ),
            patch.object(pt_track, "bench_vision", side_effect=make_vision),
            patch.object(pt_track, "open_source", side_effect=source),
            patch.object(pt_track, "MotionGate", self._AlwaysGate),
            patch.object(pt_track, "compile_watchspec", return_value=NS()),
        ):
            result = pt_track.score(live=True, n=2)
        return result, transport_calls, source_starts, decoded_frames

    def _assert_terminal_result(self, result, *, physical_attempts: int) -> None:
        self.assertEqual(result.status, "failed")
        self.assertEqual(result.qualifying_events, 0)
        self.assertIsNone(result.cells[("e2e", "notification_recall")])
        self.assertIsNone(result.cells[("e2e", "notification_precision")])
        self.assertIsNone(result.cells[("e2e", "call_reduction")])
        self.assertEqual(
            result.cells[("e2e", "physical_provider_attempts")],
            physical_attempts,
        )
        self.assertEqual(result.cells[("e2e", "selected_media_minutes")], 0.25)
        coverage = result.coverage[0]
        self.assertEqual(coverage["expected"], 2)
        self.assertEqual(coverage["observed"], 0)
        self.assertEqual(coverage["failed"], 1)
        self.assertEqual(coverage["skipped"], 1)
        self.assertTrue(result.invalid_reasons())

    def test_swallowed_budget_exhaustion_stops_frames_cases_and_metrics(self):
        result, transport_calls, source_starts, decoded_frames = self._run_terminal_failure(
            requests=1, retries=1
        )
        self.assertEqual(transport_calls, 1)
        self.assertEqual(source_starts, 1)
        self.assertEqual(decoded_frames, 1)
        self.assertIn("BudgetExceeded", result.excluded[0])
        self._assert_terminal_result(result, physical_attempts=1)

    def test_exhausted_retry_stops_frames_cases_and_metrics(self):
        result, transport_calls, source_starts, decoded_frames = self._run_terminal_failure(
            requests=4, retries=1
        )
        self.assertEqual(transport_calls, 2)
        self.assertEqual(source_starts, 1)
        self.assertEqual(decoded_frames, 1)
        self.assertIn("RuntimeError", result.excluded[0])
        self._assert_terminal_result(result, physical_attempts=2)

    def test_swallowed_usage_reconciliation_failure_stops_the_run(self):
        result, transport_calls, source_starts, decoded_frames = self._run_terminal_failure(
            requests=4,
            retries=1,
            unmetered_response=True,
        )
        self.assertEqual(transport_calls, 1)
        self.assertEqual(source_starts, 1)
        self.assertEqual(decoded_frames, 1)
        self.assertIn("omitted enforceable", result.excluded[0])
        self._assert_terminal_result(result, physical_attempts=1)


class FixtureReplayCoverageTests(unittest.TestCase):
    @staticmethod
    def _event() -> dict:
        return {
            "row_id": "case-1",
            "video_id": "video-1",
            "label": "Turning something on",
            "onset_s": 0.0,
            "end_s": 1.0,
            "positive_windows": [[0.0, 1.0]],
            "path": Path("/offline/video-1.mp4"),
        }

    def test_pt_fixture_uses_bounded_lossless_replay(self):
        captured: dict = {}

        class NoCallVision:
            async def judge(self, *_args, **_kwargs):
                raise AssertionError("empty fixture replay must not call a provider")

        def source(*_args, **kwargs):
            captured.update(kwargs)

            async def frames():
                if False:
                    yield None

            return frames()

        with (
            patch.object(pt_track, "bench_vision", return_value=NoCallVision()),
            patch.object(pt_track, "open_source", side_effect=source),
        ):
            run = asyncio.run(pt_track._score_one(self._event(), _enforcer()))

        self.assertIsNone(run.error)
        self.assertEqual(run.dropped_frames, 0)
        self.assertEqual(captured["maxsize"], pt_track._FIXTURE_QUEUE_SIZE)
        self.assertGreater(captured["maxsize"], 0)
        self.assertEqual(captured["queue_policy"], "block")
        self.assertEqual(captured["dropped"], [0])

    def test_any_fixture_drop_fails_closed_and_invalidates_coverage(self):
        class NoCallVision:
            async def judge(self, *_args, **_kwargs):
                raise AssertionError("empty fixture replay must not call a provider")

        def source(*_args, **kwargs):
            async def frames():
                kwargs["dropped"][0] += 3
                if False:
                    yield None

            return frames()

        event = self._event()
        env = {
            "PERCEPT_VISION_BACKEND": "openrouter",
            "PERCEPT_VISION_MODEL": "qwen/qwen3-vl-32b-instruct",
            "PERCEPT_BENCH_BUDGET_PLAN": _plan(),
        }
        with (
            patch.dict(os.environ, env, clear=False),
            patch.object(
                pt_track.pt,
                "select_golden_action_events",
                return_value=[event],
            ),
            patch.object(
                pt_track.pt,
                "action_media_minutes",
                return_value=0.25,
            ),
            patch.object(pt_track, "bench_vision", return_value=NoCallVision()),
            patch.object(pt_track, "open_source", side_effect=source),
        ):
            result = pt_track.score(live=True, n=1)

        self.assertEqual(result.status, "failed")
        self.assertEqual(result.cells[("e2e", "source_dropped_frames")], 3)
        self.assertIsNone(result.cells[("e2e", "notification_recall")])
        self.assertEqual(result.coverage[0]["expected"], 1)
        self.assertEqual(result.coverage[0]["observed"], 0)
        self.assertEqual(result.coverage[0]["failed"], 1)
        self.assertIn("fixture replay dropped 3 decoded frame", result.excluded[0])

    def test_ambient_language_seams_cannot_hijack_code_armed_track(self):
        class NoCallVision:
            async def judge(self, *_args, **_kwargs):
                raise AssertionError("empty fixture replay must not call a provider")

        async def source(*_args, **_kwargs):
            if False:
                yield None

        hostile = {
            "PERCEPT_MODE": "live",
            "PERCEPT_CONDUCTOR_BACKEND": "hostile-unregistered-conductor",
            "PERCEPT_SPECIALIST_BACKEND": "hostile-unregistered-specialist",
        }
        with (
            patch.dict(os.environ, hostile, clear=False),
            patch.object(pt_track, "bench_vision", return_value=NoCallVision()),
            patch.object(pt_track, "open_source", side_effect=source),
        ):
            run = asyncio.run(pt_track._score_one(self._event(), _enforcer()))

        self.assertIsNone(run.error)
        self.assertEqual(run.frames, 0)
        self.assertEqual(run.judge_calls, 0)

    def test_trace_distinguishes_gate_cadence_and_union_admission(self):
        self.assertEqual(admission_cause(gate=True, cadence=False), "gate")
        self.assertEqual(admission_cause(gate=False, cadence=True), "cadence")
        self.assertEqual(admission_cause(gate=True, cadence=True), "gate+cadence")
        self.assertEqual(admission_cause(gate=False, cadence=False), "none")

        class FalseVision:
            async def judge(self, *_args, **_kwargs):
                return Verdict(
                    satisfied=False,
                    confidence=0.9,
                    evidence="not yet",
                )

        class SuppressedGate:
            def __init__(self, *_args, **_kwargs):
                pass

            def evaluate(self, _frame, *, ts):
                return NS(
                    escalate=False,
                    signal=NS(
                        motion=0.0,
                        structural=0.0,
                        semantic=0.0,
                    ),
                    threshold=0.1,
                    ts=ts,
                )

        async def source(*_args, **_kwargs):
            yield 0.0, object(), b"frame"

        with (
            patch.object(pt_track, "bench_vision", return_value=FalseVision()),
            patch.object(pt_track, "open_source", side_effect=source),
            patch.object(pt_track, "MotionGate", SuppressedGate),
        ):
            run = asyncio.run(pt_track._score_one(self._event(), _enforcer()))

        self.assertIsNone(run.error)
        self.assertEqual(len(run.trace), 1)
        self.assertEqual(run.trace[0]["edge"]["decision"], "suppress")
        self.assertEqual(run.trace[0]["admission_cause"], "cadence")


class PinnedCaseTests(unittest.TestCase):
    @staticmethod
    def _write_fixture(root: Path) -> Path:
        annotation = {
            "video_b": {
                "metadata": {"video_id": "video_b", "frame_rate": 30},
                "action_localisation": [
                    {"label": "Action:b", "timestamps": [2_000_000, 3_000_000]},
                    {"label": "Action:other", "timestamps": [4_000_000, 5_000_000]},
                    {"label": "Action:b", "timestamps": [7_000_000, 8_000_000]},
                ],
            },
            "video_a": {
                "metadata": {"video_id": "video_a", "frame_rate": 30},
                "action_localisation": [
                    {"label": "Action:a", "timestamps": [1_000_000, 2_000_000]}
                ],
            },
        }
        annotation_path = root / "sample.json"
        annotation_path.write_text(json.dumps(annotation))
        annotation_hash = hashlib.sha256(annotation_path.read_bytes()).hexdigest()
        rows = []
        for case_id, video_id in (("case-b", "video_b"), ("case-a", "video_a")):
            media_path = root / f"{video_id}.mp4"
            media_path.write_bytes(video_id.encode())
            rows.append(
                {
                    "id": case_id,
                    "source": "owned",
                    "license": "owned",
                    "track": "perception-test-action",
                    "video_id": video_id,
                    "media_path": str(media_path),
                    "media_sha256": hashlib.sha256(media_path.read_bytes()).hexdigest(),
                    "annotation_path": str(annotation_path),
                    "annotation_sha256": annotation_hash,
                    "action_segments": 1,
                }
            )
        manifest = root / "MANIFEST.json"
        manifest.write_text(
            json.dumps(
                {
                    "version": "test",
                    "status": "draft",
                    "tracks": [
                        {
                            "id": "perception-test-action",
                            "source": "owned",
                            "status": "ready",
                            "metrics": ["recall"],
                        }
                    ],
                    "splits": {"dev": [], "test": rows},
                }
            )
        )
        return manifest

    def test_selection_uses_manifest_order_and_stable_case_ids(self):
        with TemporaryDirectory() as raw:
            manifest = self._write_fixture(Path(raw))
            events = pt.select_golden_action_events(2, manifest=manifest)
            self.assertEqual([event["row_id"] for event in events], ["case-b", "case-a"])
            self.assertEqual([event["video_id"] for event in events], ["video_b", "video_a"])
            self.assertEqual(events[0]["positive_windows"], [[2.0, 3.0], [7.0, 8.0]])

    def test_zipped_valid_annotations_join_by_video_id(self):
        with TemporaryDirectory() as raw:
            archive = Path(raw) / "annotations.zip"
            payload = {"video-a": {"action_localisation": [{"label": "wave"}]}}
            with zipfile.ZipFile(archive, "w") as bundle:
                bundle.writestr("valid.json", json.dumps(payload))
            self.assertEqual(pt.load_sample_annotations(archive), payload)

    def test_catch_all_action_label_cannot_define_a_watch_case(self):
        with TemporaryDirectory() as raw:
            root = Path(raw)
            annotation_path = root / "annotations.json"
            annotation_path.write_text(
                json.dumps(
                    {
                        "video-a": {
                            "metadata": {"video_id": "video-a", "frame_rate": 30},
                            "action_localisation": [
                                {"label": "Other", "timestamps": [1_000_000, 2_000_000]}
                            ],
                        }
                    }
                )
            )
            row = {
                "id": "case-a",
                "video_id": "video-a",
                "annotation_path": str(annotation_path),
                "media_path": str(root / "video-a.mp4"),
            }
            self.assertIsNone(pt._event_from_row(row, "action_localisation"))

    def test_first_grounded_label_binds_all_of_its_occurrences(self):
        with TemporaryDirectory() as raw:
            root = Path(raw)
            annotation_path = root / "annotations.json"
            annotation_path.write_text(
                json.dumps(
                    {
                        "video-a": {
                            "metadata": {"video_id": "video-a", "frame_rate": 30},
                            "action_localisation": [
                                {"label": "Other", "timestamps": [1_000_000, 2_000_000]},
                                {
                                    "label": "Turning something on",
                                    "timestamps": [3_000_000, 4_000_000],
                                },
                                {
                                    "label": "Turning something on",
                                    "timestamps": [6_000_000, 7_000_000],
                                },
                            ],
                        }
                    }
                )
            )
            event = pt._event_from_row(
                {
                    "id": "case-a",
                    "video_id": "video-a",
                    "annotation_path": str(annotation_path),
                    "media_path": str(root / "video-a.mp4"),
                },
                "action_localisation",
            )
            self.assertIsNotNone(event)
            assert event is not None
            self.assertEqual(event["label"], "Turning something on")
            self.assertEqual(event["positive_windows"], [[3.0, 4.0], [6.0, 7.0]])

    def test_public_manifest_declares_exact_disjoint_n20_capacity(self):
        capacity = pt.golden_action_capacity(manifest=pt.GOLDEN_MANIFEST)
        self.assertEqual(capacity["licensed"], 20)
        rows = pt.golden_action_rows(manifest=pt.GOLDEN_MANIFEST)
        self.assertEqual(len({row["id"] for row in rows}), 20)
        self.assertEqual(len({row["video_id"] for row in rows}), 20)
        with self.assertRaisesRegex(
            pt.GoldenSetCapacityError,
            r"requested N=21.*only 20 licensed action cases",
        ):
            pt.select_golden_action_events(
                21,
                manifest=pt.GOLDEN_MANIFEST,
                require_media=False,
            )

    def test_precision_never_matches_another_cases_window(self):
        case_a = EventRun(
            case_id="case-a",
            video_id="video-a",
            label="a",
            goal="a",
            onset_s=0.0,
            end_s=1.0,
        )
        case_b = EventRun(
            case_id="case-b",
            video_id="video-b",
            label="b",
            goal="b",
            onset_s=5.0,
            end_s=7.0,
        )
        self.assertEqual(_precision([("case-a", 6.0)], [case_a, case_b]), 0.0)
        self.assertEqual(_precision([("case-b", 6.0)], [case_a, case_b]), 1.0)

    def test_repeated_label_windows_are_scored_as_distinct_positive_events(self):
        run = EventRun(
            case_id="case-a",
            video_id="video-a",
            label="wave",
            goal="wave",
            onset_s=1.0,
            end_s=2.0,
            positive_windows=[(1.0, 2.0), (5.0, 6.0)],
            escalations=[1.5, 5.5],
        )
        run.fires.extend(
            [
                NS(source_media_ts=1.6),
                NS(source_media_ts=5.6),
            ]
        )
        self.assertEqual(run.edge_hits, 2)
        self.assertEqual(run.fire_hits, 2)
        self.assertEqual(_precision([("case-a", 5.5)], [run]), 1.0)

    def test_duplicate_fire_in_one_window_is_a_precision_false_positive(self):
        run = EventRun(
            case_id="case-a",
            video_id="video-a",
            label="wave",
            goal="wave",
            onset_s=1.0,
            end_s=2.0,
        )
        run.fires.extend(
            [
                NS(source_media_ts=1.5),
                NS(source_media_ts=1.6),
            ]
        )
        self.assertEqual(run.fire_hits, 1)
        self.assertEqual(len(run.matched_fire_events), 1)
        self.assertEqual(_event_precision([run]), 0.5)

    def test_cost_reduction_uses_physical_attempts_including_retry(self):
        physical, logical = _call_reductions(
            dense_calls=10,
            logical_calls=2,
            physical_attempts=3,
        )
        self.assertAlmostEqual(physical, 0.7)
        self.assertAlmostEqual(logical, 0.8)


class LatencySemanticsTests(unittest.TestCase):
    def test_even_latency_sample_uses_conventional_median(self):
        self.assertEqual(_median([1.0, 3.0]), 2.0)

    def test_action_to_fire_adds_source_delay_but_judge_latency_stays_separate(self):
        run = EventRun(
            case_id="case-a",
            video_id="video-a",
            label="a",
            goal="a",
            onset_s=10.0,
            end_s=12.0,
        )
        timing = JudgmentTiming(
            ingest_observed_ts=100.0,
            admitted_ts=100.05,
            judge_started_ts=100.1,
            judge_completed_ts=100.35,
            decision_emitted_ts=100.4,
            source_media_ts=10.5,
        )
        observation = _decision_observation(
            NS(judgment=NS(timing=timing)),
            run,
        )
        self.assertIsNotNone(observation)
        assert observation is not None
        self.assertAlmostEqual(observation.observed_to_fire_s, 0.4)
        self.assertAlmostEqual(observation.judge_latency_s, 0.25)
        self.assertAlmostEqual(observation.action_onset_to_fire_s, 0.9)
        self.assertNotEqual(
            observation.action_onset_to_fire_s,
            observation.judge_latency_s,
        )

    def test_later_window_action_latency_uses_later_window_onset(self):
        run = EventRun(
            case_id="case-a",
            video_id="video-a",
            label="a",
            goal="a",
            onset_s=1.0,
            end_s=2.0,
            positive_windows=[(1.0, 2.0), (10.0, 12.0)],
        )
        timing = JudgmentTiming(
            ingest_observed_ts=100.0,
            admitted_ts=100.05,
            judge_started_ts=100.1,
            judge_completed_ts=100.35,
            decision_emitted_ts=100.4,
            source_media_ts=10.5,
        )
        observation = _decision_observation(
            NS(judgment=NS(timing=timing)),
            run,
        )
        self.assertIsNotNone(observation)
        assert observation is not None
        self.assertAlmostEqual(observation.action_onset_to_fire_s, 0.9)

    def test_overlapping_window_details_use_aggregate_one_to_one_assignment(self):
        run = EventRun(
            case_id="case-a",
            video_id="video-a",
            label="a",
            goal="a",
            onset_s=0.0,
            end_s=5.0,
            positive_windows=[(0.0, 5.0), (4.0, 6.0)],
        )

        def observation(source_media_ts: float, observed_s: float):
            timing = JudgmentTiming(
                ingest_observed_ts=100.0,
                admitted_ts=100.01,
                judge_started_ts=100.02,
                judge_completed_ts=100.05,
                decision_emitted_ts=100.0 + observed_s,
                source_media_ts=source_media_ts,
            )
            value = _decision_observation(
                NS(judgment=NS(timing=timing)),
                run,
            )
            assert value is not None
            return value

        run.fires.extend(
            [
                observation(4.2, 0.1),
                observation(4.3, 0.2),
            ]
        )
        run.reconcile_fire_windows()
        records = _fire_records(run)
        self.assertEqual(run.fire_hits, 2)
        self.assertAlmostEqual(records[0]["source_detection_latency_s"], 4.2)
        self.assertAlmostEqual(records[0]["action_onset_to_fire_s"], 4.3)
        self.assertAlmostEqual(records[1]["source_detection_latency_s"], 0.3)
        self.assertAlmostEqual(records[1]["action_onset_to_fire_s"], 0.5)


if __name__ == "__main__":
    unittest.main()
