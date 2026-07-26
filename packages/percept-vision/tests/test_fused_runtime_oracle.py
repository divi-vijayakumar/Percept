"""Public-product adversarial tests for fused Watches.

The implementation DTOs are used only by ``fused_bundle`` to inject deterministic accepted media at
the private provider bridge.  Every behavioral assertion is made at the public Watch, FireEvent,
Incident, callback, provider-call, or trace surface.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import struct
import types
import unittest
from dataclasses import FrozenInstanceError, dataclass, is_dataclass
from unittest.mock import patch

from percept import Percept
from percept.adapters import Verdict
from percept.backends.fake import FakeVision
from percept.backends.gemini import GeminiVision
from percept.contracts import AudioFrame, WatchPlan
from percept.direct_judge import DirectJudge, DirectResult, DirectRouting
from percept.events import AudioUnit
from percept.goals import GoalOp
from percept.harness.tier0 import gray_frame
from percept.incidents import make_incident
from percept.runtime.envelope import SourceEnvelope, SourceFormat
from percept.runtime.worker import RuntimeStats, run
from percept.watchplan_apply import apply_watch_plan


W, H = 48, 36
STILL = gray_frame(W, H, 0)
CHANGED = gray_frame(W, H, 255)
JPEG = b"\xff\xd8fused-runtime-oracle\xff\xd9"
VIDEO_PAYLOAD = ("image/jpeg", JPEG)


class CountingVision:
    SUPPORTS_FUSED_AUDIO_INPUT = True

    def __init__(self, verdicts=None):
        self.verdicts = dict(verdicts or {})
        self.detect_calls = []
        self.judge_calls = []

    async def detect(self, frame, goals):
        self.detect_calls.append((frame, tuple(goal.id for goal in goals)))
        return []

    async def judge(self, condition, frame, **kwargs):
        self.judge_calls.append(
            {
                "condition": condition,
                "frame": frame,
                "audio": kwargs.get("audio"),
                "kwargs": dict(kwargs),
            }
        )
        answer = self.verdicts.get(condition, Verdict(False, 0.99, "not present"))
        if isinstance(answer, BaseException):
            raise answer
        return answer


class ProviderResponse:
    usage_metadata = None

    def __init__(self, verdict: Verdict):
        self.text = json.dumps(
            {
                "satisfied": verdict.satisfied,
                "confidence": verdict.confidence,
                "evidence": verdict.evidence,
            }
        )


class ProviderModels:
    def __init__(self, answers=None, blocked=()):
        self.answers = dict(answers or {})
        self.blocked = set(blocked)
        self.calls = []
        self.slow_started = asyncio.Event()
        self.all_started = asyncio.Event()
        self.slow_release = asyncio.Event()
        self.started_conditions = set()

    async def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        rendered = str(kwargs)
        condition = next(
            (candidate for candidate in self.answers if candidate in rendered),
            "",
        )
        if condition in self.blocked:
            self.started_conditions.add(condition)
            self.slow_started.set()
            if self.started_conditions == self.blocked:
                self.all_started.set()
            await self.slow_release.wait()
        answer = self.answers.get(
            condition,
            Verdict(False, 0.99, "not present"),
        )
        if isinstance(answer, BaseException):
            raise answer
        return ProviderResponse(answer)


class ProviderClient:
    def __init__(self, answers=None, blocked=()):
        self.models = ProviderModels(answers, blocked)
        self.aio = types.SimpleNamespace(models=self.models)


def trusted_core(*, answers=None, blocked=(), **kwargs):
    client = ProviderClient(answers, blocked)
    vision = GeminiVision(client=client)

    async def no_detection(_frame, _goals):
        return []

    # Detection is not the fused provider-judgment seam under test. Keep the exact trusted Gemini
    # judge/binding while replacing only this unrelated object-detection call with an offline no-op.
    vision.detect = no_detection
    core = Percept.create(
        vision=vision,
        discover_plugins=False,
        **kwargs,
    )
    return core, client.models


class ExplodingIterable:
    def __init__(self):
        self.iterated = False

    def __iter__(self):
        self.iterated = True
        raise AssertionError("source was iterated before fused preflight rejected the topology")


async def collect(core, *, source, **kwargs):
    return [item async for item in run(core, source=source, **kwargs)]


async def bounded(coro, timeout: float = 2.0):
    return await asyncio.wait_for(coro, timeout=timeout)


def apply(core, goal_id: str, condition: str, sense: str) -> None:
    receipt = apply_watch_plan(
        core,
        WatchPlan(id=goal_id, condition=condition, sense=sense),
    )
    if receipt.status != "applied":
        raise AssertionError(receipt)


def application_for(core, goal_id: str) -> str:
    return core._watch_runtime_keys[goal_id]


def audio_unit(seq: int, start: float, samples: tuple[int, ...], energy: float):
    pcm = struct.pack("<" + "h" * len(samples), *samples)
    return (
        start + len(samples) / 4,
        AudioFrame(
            pcm=pcm,
            sample_rate=4,
            channels=1,
            sample_format="s16le",
            t_start=start,
            t_end=start + len(samples) / 4,
        ),
        energy,
    )


def video_envelope(seq: int, at: float, *, epoch: int = 0, changed: bool = False):
    return SourceEnvelope(
        source_id="fixture-container",
        modality="video",
        epoch=epoch,
        sequence=seq,
        observed_time=100.0 + at,
        source_time=at,
        format=SourceFormat(media_type="image/jpeg", width=W, height=H),
        payload=VIDEO_PAYLOAD,
        signal=CHANGED if changed else STILL,
    )


def audio_envelope(seq: int, start: float, end: float, *, epoch: int = 0):
    sample_rate = 4
    sample_count = max(1, round((end - start) * sample_rate))
    end = start + sample_count / sample_rate
    frame = AudioFrame(
        pcm=struct.pack("<" + "h" * sample_count, *([1000] * sample_count)),
        sample_rate=sample_rate,
        channels=1,
        sample_format="s16le",
        t_start=start,
        t_end=end,
    )
    return SourceEnvelope(
        source_id="fixture-container",
        modality="audio",
        epoch=epoch,
        sequence=seq,
        observed_time=100.0 + end,
        source_time=end,
        format=SourceFormat(
            media_type="audio/pcm",
            sample_rate=sample_rate,
            channels=1,
            sample_format="s16le",
        ),
        payload=frame,
        signal=0.5,
        audio_direction="input",
    )


def fused_bundle(
    *,
    watch_revision: int,
    application: str = "runtime-oracle",
    include_audio: bool = True,
    video_payload=VIDEO_PAYLOAD,
    audio_payload: AudioFrame | None = None,
    policy_window_s: float = 2.0,
):
    """The one private-constructor adapter in this public-runtime test module."""
    from percept.fused_sync import (
        _AcceptedMediaUnit,
        _FusedSyncPolicy,
        _FusedSynchronizer,
    )

    policy = _FusedSyncPolicy(
        window_s=policy_window_s,
        allowed_lateness_s=0.25,
        max_skew_s=0.5,
        min_clock_confidence=0.9,
        max_units_per_modality=16,
        max_bytes_per_modality=1024 * 1024,
        version=f"oracle-{policy_window_s}",
    )
    sync = _FusedSynchronizer(
        policy,
        watch_application=application,
        watch_revision=watch_revision,
    )
    video = _AcceptedMediaUnit(
        modality="video",
        payload=video_payload,
        source_id="fixture-video",
        source_epoch=0,
        source_sequence=0,
        generation=0,
        watch_application=application,
        watch_revision=watch_revision,
        event_start=1.0,
        event_end=1.0,
        observed_start=101.0,
        observed_end=101.0,
        clock_domain="source",
        clock_authority="fixture-container",
        clock_confidence=1.0,
        provenance={"carrier": "video", "seq": 0},
        source_watermark=2.0,
        byte_size=(
            len(video_payload[1])
            if isinstance(video_payload, tuple)
            else len(video_payload)
        ),
    )
    sync.accept(video)
    if include_audio:
        audio_payload = audio_payload or AudioFrame(
            pcm=struct.pack("<hhhh", 1, 2, 3, 4),
            sample_rate=4,
            channels=1,
            sample_format="s16le",
            t_start=0.0,
            t_end=1.0,
        )
        canonical_audio = AudioUnit(
            seq=0,
            t_start=0.0,
            t_end=1.0,
            event_t=1.0,
            chunk=audio_payload,
            sample_rate=audio_payload.sample_rate,
            channels=audio_payload.channels,
            sample_format=audio_payload.sample_format,
            direction="input",
            source_id="fixture-audio",
            source_epoch=0,
            source_sequence=0,
            observed_time=101.0,
            source_time=1.0,
        )
        sync.accept(
            _AcceptedMediaUnit(
                modality="audio",
                payload=canonical_audio,
                source_id="fixture-audio",
                source_epoch=0,
                source_sequence=0,
                generation=0,
                watch_application=application,
                watch_revision=watch_revision,
                event_start=0.0,
                event_end=1.0,
                observed_start=100.0,
                observed_end=101.0,
                clock_domain="source",
                clock_authority="fixture-container",
                clock_confidence=1.0,
                provenance={"carrier": "audio", "seq": 0},
                source_watermark=2.0,
                byte_size=len(audio_payload.pcm or audio_payload.encoded or b""),
            )
        )
    return sync.bundle_for(video)


def trace_rows_with_identity(rows, evidence_seq, judgment_event_id):
    """Centralized private trace lookup while fused carrier/stage names remain under review."""
    matches = []
    for row in rows:
        data = getattr(row, "data", {}) or {}
        if (
            data.get("evidence_seq") == evidence_seq
            or data.get("dispatch_seq") == evidence_seq
            or data.get("judgment_event_id") == judgment_event_id
        ):
            matches.append(row)
    return matches


def provider_bytes(value, *, _seen=None):
    """Collect exact byte leaves from provider request parts without depending on google-genai reprs."""
    seen = set() if _seen is None else _seen
    if id(value) in seen:
        return []
    seen.add(id(value))
    if isinstance(value, bytes):
        return [value]
    if isinstance(value, dict):
        return [
            blob
            for key, item in value.items()
            for blob in provider_bytes((key, item), _seen=seen)
        ]
    if isinstance(value, (tuple, list)):
        return [
            blob
            for item in value
            for blob in provider_bytes(item, _seen=seen)
        ]
    blobs = []
    for name in ("data", "inline_data", "parts", "contents"):
        if hasattr(value, name):
            blobs.extend(provider_bytes(getattr(value, name), _seen=seen))
    return blobs


class FusedPublicPreflightAndNoninterferenceTests(unittest.TestCase):
    def test_unsupported_clock_rejects_before_either_source_is_iterated(self):
        core = Percept.create(vision=FakeVision(), discover_plugins=False)
        apply(core, "fused", "both channels matter", "fused")
        video = ExplodingIterable()
        audio = ExplodingIterable()

        with self.assertRaisesRegex(ValueError, "timebase"):
            asyncio.run(
                collect(
                    core,
                    source=video,
                    audio_source=audio,
                    timebase="uncalibrated-cross-device",
                )
            )

        self.assertFalse(video.iterated)
        self.assertFalse(audio.iterated)
        self.assertEqual(core.metrics.get("judge.latency.count"), 0)

    def test_untrusted_provider_rejects_at_apply_before_calls_or_runtime_state(self):
        vision = CountingVision()
        core = Percept.create(vision=vision, discover_plugins=False)
        receipt = apply_watch_plan(
            core,
            WatchPlan(id="fused", condition="both channels matter", sense="fused"),
        )
        self.assertEqual(receipt.status, "rejected")
        self.assertEqual(receipt.dropped_fields[0].code, "audio_input_backend_unsupported")
        self.assertEqual(core.goals.all(), [])
        self.assertEqual(vision.detect_calls, [])
        self.assertEqual(vision.judge_calls, [])

    def test_audio_only_rejects_fused_watch_before_source_iteration_or_provider_spend(self):
        vision = FakeVision()
        core = Percept.create(vision=vision, discover_plugins=False)
        apply(core, "fused", "both channels matter", "fused")
        source = ExplodingIterable()
        judge_calls_before = core.metrics.get("judge.latency.count")

        with self.assertRaisesRegex(ValueError, "audio-only.*audio"):
            asyncio.run(
                collect(
                    core,
                    source=None,
                    audio_source=source,
                    judge_dispatch="inline",
                )
        )

        self.assertFalse(source.iterated)
        self.assertEqual(core.metrics.get("judge.latency.count"), judge_calls_before)

    def test_video_only_run_passes_no_fused_kwarg_and_spawns_no_runtime_task(self):
        vision = CountingVision()
        core = Percept.create(vision=vision, discover_plugins=False)
        apply(core, "video", "the scene changed", "video")
        calls = []
        original = core.perceive_judged

        async def observed(*args, **kwargs):
            calls.append(dict(kwargs))
            return await original(*args, **kwargs)

        core.perceive_judged = observed
        original_create_task = asyncio.create_task
        spawned = []

        def recording_create_task(coro, *args, **kwargs):
            spawned.append(coro)
            return original_create_task(coro, *args, **kwargs)

        with patch("percept.runtime.worker.asyncio.create_task", recording_create_task):
            asyncio.run(
                collect(
                    core,
                    source=[(0.0, STILL, "still"), (1.0, CHANGED, "changed")],
                    judge_dispatch="inline",
                )
            )

        self.assertTrue(calls)
        self.assertTrue(
            all("_fused_bundles_by_goal" not in kwargs for kwargs in calls)
        )
        self.assertEqual(spawned, [])
        self.assertEqual(len(vision.judge_calls), 1)

    def test_audio_only_run_stays_on_existing_audio_bridge_with_no_fused_kwarg(self):
        vision = FakeVision()
        core = Percept.create(vision=vision, discover_plugins=False)
        apply(core, "audio", "the sound matters", "audio")
        audio_calls = []
        judged_calls = []
        original_audio = core.perceive_audio_judged
        original_judged = core.perceive_judged

        async def observed_audio(*args, **kwargs):
            audio_calls.append(dict(kwargs))
            return await original_audio(*args, **kwargs)

        async def observed_judged(*args, **kwargs):
            judged_calls.append(dict(kwargs))
            return await original_judged(*args, **kwargs)

        core.perceive_audio_judged = observed_audio
        core.perceive_judged = observed_judged
        stats = RuntimeStats()
        spawned = []
        original_create_task = asyncio.create_task

        def recording_create_task(coro, *args, **kwargs):
            spawned.append(coro)
            return original_create_task(coro, *args, **kwargs)

        with patch("percept.runtime.worker.asyncio.create_task", recording_create_task):
            asyncio.run(
                collect(
                    core,
                    source=None,
                    audio_source=[
                        audio_unit(0, 0.0, (0, 0), 0.0),
                        audio_unit(1, 0.5, (32767, 32767), 1.0),
                    ],
                    audio_wake={"high": 0.1, "low": 0.05},
                    audio_settle_s=0.0,
                    judge_dispatch="inline",
                    stats=stats,
                )
            )

        self.assertTrue(audio_calls)
        self.assertTrue(judged_calls)
        self.assertTrue(
            all("_fused_bundles_by_goal" not in kwargs for kwargs in audio_calls)
        )
        self.assertTrue(
            all("_fused_bundles_by_goal" not in kwargs for kwargs in judged_calls)
        )
        self.assertEqual(stats.judge_calls, core.metrics.get("judge.latency.count"))
        self.assertEqual(spawned, [])


class FusedCompilerPolicyTests(unittest.TestCase):
    def test_compiler_freezes_one_distinct_policy_per_fused_watch(self):
        vision = FakeVision()
        core = Percept.create(vision=vision, discover_plugins=False)
        apply(core, "fused-a", "first fused condition", "fused")
        apply(core, "fused-b", "second fused condition", "fused")
        apply(core, "video", "video condition", "video")

        first = core._watch_runtime_bindings["fused-a"][0].fused_sync_policy
        second = core._watch_runtime_bindings["fused-b"][0].fused_sync_policy
        video = core._watch_runtime_bindings["video"][0].fused_sync_policy
        self.assertIsNotNone(first)
        self.assertIsNotNone(second)
        self.assertIsNot(first, second)
        self.assertEqual(first, second)
        self.assertTrue(is_dataclass(first))
        with self.assertRaises(FrozenInstanceError):
            first.window_s = 999.0
        self.assertIsNone(video)

    def test_fusion_adds_no_public_run_tunable_profile_or_registry(self):
        import percept.fused_sync as fused_sync

        public_run_parameters = set(inspect.signature(run).parameters)
        forbidden = {
            name
            for name in public_run_parameters
            if any(token in name for token in ("fused", "sync", "skew", "lateness"))
        }
        self.assertEqual(forbidden, set())
        self.assertFalse(hasattr(fused_sync, "FusedSyncProfile"))
        self.assertFalse(hasattr(fused_sync, "FusedSyncRegistry"))


@dataclass
class AlwaysConfidentRule:
    applies_to: DirectRouting = DirectRouting()

    def evaluate(self, condition, frame, roi, facts):
        return DirectResult(
            outcome="confident",
            verdict=Verdict(True, 0.999, "direct shortcut"),
        )


class FusedPublicIsolationAndFailureTests(unittest.TestCase):
    def test_direct_judge_cannot_turn_missing_audio_into_a_confident_fire(self):
        direct = DirectJudge([AlwaysConfidentRule()], verify_cadence={"medium": 0})
        core, models = trusted_core(
            answers={"both channels matter": Verdict(True, 0.99, "provider yes")},
            direct_judge=direct,
        )
        apply(core, "fused", "both channels matter", "fused")
        bundle = fused_bundle(
            watch_revision=core.goals.revision,
            application=application_for(core, "fused"),
            include_audio=False,
        )

        async def drive():
            completion = await core.perceive_fused_judged(
                VIDEO_PAYLOAD,
                t=1.0,
                goal_ids=("fused",),
                _fused_bundles_by_goal={"fused": bundle},
            )
            return await core.reduce_fused_completion(completion)

        events = asyncio.run(drive())

        self.assertFalse(any(event.action == "fire" for event in events))
        self.assertTrue(any(event.action == "ask" for event in events))
        self.assertEqual(models.calls, [])

    def test_visual_watch_cannot_wake_or_spend_for_incomplete_fused_watch(self):
        core, models = trusted_core(
            answers={
                "visual": Verdict(True, 0.99, "visual yes"),
                "fused": Verdict(True, 0.99, "fused yes"),
            }
        )
        apply(core, "visual", "visual", "video")
        apply(core, "fused", "fused", "fused")
        incomplete = fused_bundle(
            watch_revision=core.goals.revision,
            application=application_for(core, "fused"),
            include_audio=False,
        )

        async def drive():
            visual = await core.perceive_judged(
                VIDEO_PAYLOAD,
                t=1.0,
                goal_ids=("visual",),
            )
            completion = await core.perceive_fused_judged(
                VIDEO_PAYLOAD,
                t=1.0,
                goal_ids=("fused",),
                _fused_bundles_by_goal={"fused": incomplete},
            )
            fused = await core.reduce_fused_completion(completion)
            return visual + fused

        events = asyncio.run(drive())

        self.assertEqual(
            len(models.calls),
            1,
        )
        self.assertEqual(
            [event.goal_id for event in events if event.action == "fire"],
            ["visual"],
        )
        self.assertTrue(
            any(event.goal_id == "fused" and event.action == "ask" for event in events)
        )

    def test_provider_failure_for_complete_fusion_asks_and_never_fires(self):
        core, models = trusted_core(
            answers={"fused": RuntimeError("provider unavailable")}
        )
        apply(core, "fused", "fused", "fused")
        bundle = fused_bundle(
            watch_revision=core.goals.revision,
            application=application_for(core, "fused"),
        )

        async def drive():
            completion = await core.perceive_fused_judged(
                VIDEO_PAYLOAD,
                t=1.0,
                goal_ids=("fused",),
                _fused_bundles_by_goal={"fused": bundle},
            )
            return await core.reduce_fused_completion(completion)

        events = asyncio.run(drive())

        self.assertFalse(any(event.action == "fire" for event in events))
        self.assertTrue(any(event.action == "ask" for event in events))
        self.assertGreaterEqual(len(models.calls), 1)

    def test_hung_deliberate_watch_does_not_delay_reflex_watch(self):
        async def drive():
            core, models = trusted_core(
                answers={
                    "deliberate": Verdict(True, 0.99, "slow answer"),
                    "reflex": Verdict(True, 0.99, "fast answer"),
                },
                blocked={"deliberate"},
            )
            apply(core, "deliberate", "deliberate", "fused")
            apply(core, "reflex", "reflex", "fused")
            revision = core.goals.revision
            slow_bundle = fused_bundle(
                watch_revision=revision,
                application=application_for(core, "deliberate"),
                policy_window_s=30.0,
            )
            fast_bundle = fused_bundle(
                watch_revision=revision,
                application=application_for(core, "reflex"),
                policy_window_s=0.5,
            )
            slow = asyncio.create_task(
                core.perceive_fused_judged(
                    VIDEO_PAYLOAD,
                    t=1.0,
                    goal_ids=("deliberate",),
                    _fused_bundles_by_goal={"deliberate": slow_bundle},
                )
            )
            await asyncio.wait_for(models.slow_started.wait(), timeout=0.25)
            fast_completion = await asyncio.wait_for(
                core.perceive_fused_judged(
                    VIDEO_PAYLOAD,
                    t=1.0,
                    goal_ids=("reflex",),
                    _fused_bundles_by_goal={"reflex": fast_bundle},
                ),
                timeout=0.25,
            )
            fast_events = await core.reduce_fused_completion(fast_completion)
            self.assertFalse(slow.done())
            models.slow_release.set()
            slow_completion = await slow
            await core.reduce_fused_completion(slow_completion)
            return fast_events

        events = asyncio.run(bounded(drive()))
        self.assertTrue(any(event.goal_id == "reflex" for event in events))

    def test_overlapping_provider_waits_commit_same_time_watches_in_isolated_state(self):
        async def drive():
            core, models = trusted_core(
                answers={
                    "watch-a": Verdict(True, 0.99, "a"),
                    "watch-b": Verdict(True, 0.99, "b"),
                },
                blocked={"watch-a", "watch-b"},
            )
            apply(core, "watch-a", "watch-a", "fused")
            apply(core, "watch-b", "watch-b", "fused")
            revision = core.goals.revision
            dispatch_order = ("watch-a", "watch-b")
            reduction_order = ("watch-b", "watch-a")
            bundles = {
                goal_id: fused_bundle(
                    watch_revision=revision,
                    application=application_for(core, goal_id),
                )
                for goal_id in dispatch_order
            }
            tasks = [
                asyncio.create_task(
                    core.perceive_fused_judged(
                        VIDEO_PAYLOAD,
                        t=7.0,
                        goal_ids=(goal_id,),
                        _fused_bundles_by_goal={goal_id: bundles[goal_id]},
                    )
                )
                for goal_id in dispatch_order
            ]
            await asyncio.wait_for(models.all_started.wait(), timeout=0.25)
            self.assertTrue(all(not task.done() for task in tasks))
            models.slow_release.set()
            completions = await asyncio.gather(*tasks)
            completions_by_goal = {
                completion.goal_id: completion for completion in completions
            }
            events = []
            for goal_id in reduction_order:
                events.extend(
                    await core.reduce_fused_completion(completions_by_goal[goal_id])
                )
            return core, events, reduction_order

        core, events, reduction_order = asyncio.run(bounded(drive()))
        fires = [event for event in events if event.action == "fire"]
        self.assertEqual(tuple(event.goal_id for event in fires), reduction_order)
        self.assertEqual(
            {event.goal_id: event.judgment.evidence_seq for event in fires},
            {"watch-a": 1, "watch-b": 1},
        )
        self.assertNotEqual(fires[0].judgment.key, fires[1].judgment.key)
        self.assertEqual(core.metrics.get("judge.latency.count"), 2)

    def test_provider_completion_after_watch_revision_cannot_emit_or_mutate(self):
        async def drive():
            core, models = trusted_core(
                answers={"stale": Verdict(True, 0.99, "obsolete answer")},
                blocked={"stale"},
            )
            apply(core, "stale", "stale", "fused")
            revision = core.goals.revision
            bundle = fused_bundle(
                watch_revision=revision,
                application=application_for(core, "stale"),
            )
            task = asyncio.create_task(
                core.perceive_fused_judged(
                    VIDEO_PAYLOAD,
                    t=1.0,
                    expected_goals_revision=revision,
                    goal_ids=("stale",),
                    _fused_bundles_by_goal={"stale": bundle},
                )
            )
            await asyncio.wait_for(models.slow_started.wait(), timeout=0.25)
            models.slow_release.set()
            completion = await task
            core.apply_goal_ops(
                [GoalOp(op="update", goal_id="stale", fields={"say": "new text"})]
            )
            state_before_reduce = core.gate.snapshot()
            log_before_reduce = tuple(core.action_log.entries())
            events = await core.reduce_fused_completion(completion)
            return (
                core,
                events,
                state_before_reduce,
                core.gate.snapshot(),
                log_before_reduce,
                tuple(core.action_log.entries()),
            )

        core, events, state_before, state_after, log_before, log_after = asyncio.run(
            bounded(drive())
        )
        self.assertEqual(events, [])
        self.assertEqual(state_after, state_before)
        self.assertEqual(log_after, log_before)
        self.assertEqual(len(core.action_log), 0)

    def test_provider_completion_after_source_generation_cannot_emit_or_mutate(self):
        async def drive():
            core, models = trusted_core(
                answers={"stale": Verdict(True, 0.99, "obsolete answer")},
                blocked={"stale"},
            )
            apply(core, "stale", "stale", "fused")
            revision = core.goals.revision
            generation = core.source_continuity_generation("camera")
            bundle = fused_bundle(
                watch_revision=revision,
                application=application_for(core, "stale"),
            )
            task = asyncio.create_task(
                core.perceive_fused_judged(
                    VIDEO_PAYLOAD,
                    t=1.0,
                    expected_source_generation=generation,
                    goal_ids=("stale",),
                    _fused_bundles_by_goal={"stale": bundle},
                )
            )
            await asyncio.wait_for(models.slow_started.wait(), timeout=0.25)
            models.slow_release.set()
            completion = await task
            core.advance_source_continuity("camera")
            state_before_reduce = core.gate.snapshot()
            log_before_reduce = tuple(core.action_log.entries())
            events = await core.reduce_fused_completion(completion)
            return (
                core,
                events,
                state_before_reduce,
                core.gate.snapshot(),
                log_before_reduce,
                tuple(core.action_log.entries()),
            )

        core, events, state_before, state_after, log_before, log_after = asyncio.run(
            bounded(drive())
        )
        self.assertEqual(events, [])
        self.assertEqual(state_after, state_before)
        self.assertEqual(log_after, log_before)
        self.assertEqual(len(core.action_log), 0)


class FusedRealWorkerContinuityTests(unittest.TestCase):
    def test_video_gap_invalidates_old_audio_and_new_generation_recovers(self):
        token = f"{VIDEO_PAYLOAD}|audio"
        core = Percept.create(
            vision=FakeVision(script={token: Verdict(True, 0.99, "joined")}),
            discover_plugins=False,
        )
        apply(core, "fused", "joined", "fused")

        async def drive():
            first_audio_committed = asyncio.Event()
            gap_processed = asyncio.Event()
            new_audio_committed = asyncio.Event()
            original_push_audio = core.push_audio
            pushes = 0

            def observed_push(*args, **kwargs):
                nonlocal pushes
                result = original_push_audio(*args, **kwargs)
                pushes += 1
                (first_audio_committed if pushes == 1 else new_audio_committed).set()
                return result

            core.push_audio = observed_push

            async def audio_source():
                yield audio_envelope(0, 0.0, 1.0)
                await gap_processed.wait()
                yield audio_envelope(1, 1.0, 2.0)

            async def video_source():
                await first_audio_committed.wait()
                # First visible sequence 1 is a real source gap and generation boundary.
                yield video_envelope(1, 1.0)
                gap_processed.set()
                await new_audio_committed.wait()
                yield video_envelope(2, 2.0, changed=True)

            asks = []
            incidents = await collect(
                core,
                source=video_source(),
                audio_source=audio_source(),
                audio_gate=None,
                timebase="media",
                judge_dispatch="inline",
                on_ask=asks.append,
            )
            return incidents, asks

        incidents, _asks = asyncio.run(bounded(drive()))
        self.assertEqual(len(incidents), 1)
        self.assertEqual(incidents[0].concern_id, "fused")
        self.assertEqual(core.metrics.get("judge.latency.count"), 1)

    def test_audio_gap_purges_pre_gap_video_and_recovers_with_post_gap_pair(self):
        token = f"{VIDEO_PAYLOAD}|audio"
        core = Percept.create(
            vision=FakeVision(script={token: Verdict(True, 0.99, "joined")}),
            discover_plugins=False,
        )
        apply(core, "fused", "joined", "fused")

        async def drive():
            first_audio_committed = asyncio.Event()
            first_video_processed = asyncio.Event()
            gapped_audio_committed = asyncio.Event()
            original_push_audio = core.push_audio
            pushes = 0

            def observed_push(*args, **kwargs):
                nonlocal pushes
                result = original_push_audio(*args, **kwargs)
                pushes += 1
                (
                    first_audio_committed
                    if pushes == 1
                    else gapped_audio_committed
                ).set()
                return result

            core.push_audio = observed_push

            async def audio_source():
                yield audio_envelope(0, 0.0, 1.0)
                await first_video_processed.wait()
                # Missing sequence 1 is a real audio discontinuity.
                yield audio_envelope(2, 1.0, 2.0)

            async def video_source():
                await first_audio_committed.wait()
                yield video_envelope(0, 1.0)
                first_video_processed.set()
                await gapped_audio_committed.wait()
                yield video_envelope(1, 2.0, changed=True)

            return await collect(
                core,
                source=video_source(),
                audio_source=audio_source(),
                audio_gate=None,
                timebase="media",
                judge_dispatch="inline",
            )

        incidents = asyncio.run(bounded(drive()))
        self.assertEqual(len(incidents), 1)
        self.assertEqual(core.metrics.get("judge.latency.count"), 1)

    def test_eos_with_nonempty_tail_truncated_audio_never_confidently_fires(self):
        token = f"{VIDEO_PAYLOAD}|audio"
        core = Percept.create(
            vision=FakeVision(script={token: Verdict(True, 0.99, "joined")}),
            discover_plugins=False,
        )
        apply(core, "fused", "joined", "fused")

        async def drive():
            audio_committed = asyncio.Event()
            original_push_audio = core.push_audio

            def observed_push(*args, **kwargs):
                result = original_push_audio(*args, **kwargs)
                audio_committed.set()
                return result

            core.push_audio = observed_push

            async def audio_source():
                yield audio_envelope(0, 0.0, 1.25)

            async def video_source():
                await audio_committed.wait()
                yield video_envelope(0, 1.0)
                yield video_envelope(1, 2.0, changed=True)

            asks = []
            incidents = await collect(
                core,
                source=video_source(),
                audio_source=audio_source(),
                audio_gate=None,
                timebase="media",
                judge_dispatch="inline",
                on_ask=asks.append,
            )
            return incidents, asks

        incidents, asks = asyncio.run(bounded(drive()))
        self.assertEqual(incidents, [])
        self.assertTrue(asks)
        self.assertEqual(core.metrics.get("judge.latency.count"), 0)


class FusedPublicEvidenceIdentityTests(unittest.TestCase):
    def test_exact_provider_payload_event_incident_and_trace_share_identity(self):
        video_payload = ("image/jpeg", b"\xff\xd8exact-video\xff\xd9")
        audio_payload = AudioFrame(
            pcm=struct.pack("<hhhh", 7, 8, 9, 10),
            sample_rate=4,
            channels=1,
            sample_format="s16le",
            t_start=0.0,
            t_end=1.0,
        )
        traces = []
        core, models = trusted_core(
            answers={"fused": Verdict(True, 0.99, "both present")},
            on_trace=traces.append,
        )
        apply(core, "fused", "fused", "fused")
        bundle = fused_bundle(
            watch_revision=core.goals.revision,
            application=application_for(core, "fused"),
            video_payload=video_payload,
            audio_payload=audio_payload,
        )

        async def drive():
            completion = await core.perceive_fused_judged(
                video_payload,
                t=1.0,
                goal_ids=("fused",),
                _fused_bundles_by_goal={"fused": bundle},
            )
            return await core.reduce_fused_completion(completion)

        events = asyncio.run(drive())
        fire = next(event for event in events if event.action == "fire")
        incident = make_incident(fire, ts=1.0)
        attached = fire._fused_evidence_bundle
        incident_attached = incident._fused_evidence_bundle
        provider_blobs = provider_bytes(models.calls[0])
        self.assertIn(video_payload[1], provider_blobs)
        self.assertTrue(any(audio_payload.pcm in blob for blob in provider_blobs))
        self.assertIsNotNone(fire.judgment)
        self.assertIs(attached, incident_attached)
        self.assertEqual(attached.evidence_seq, fire.judgment.evidence_seq)
        self.assertEqual(
            incident.evidence_seq,
            fire.judgment.evidence_seq,
        )
        self.assertEqual(
            incident.judgment_event_id,
            next(
                row.data["judgment_event_id"]
                for row in traces
                if row.data.get("judgment_event_id") is not None
            ),
        )
        self.assertTrue(
            trace_rows_with_identity(
                traces,
                fire.judgment.evidence_seq,
                incident.judgment_event_id,
            )
        )
        self.assertTrue(any(row.stage == "source.fused" for row in traces))

    def test_ask_silent_and_degraded_paths_keep_bundle_identity_in_trace(self):
        scenarios = (
            ("ask", None, False),
            ("silent", Verdict(False, 0.99, "not present"), True),
            ("degraded", RuntimeError("provider failed"), True),
        )
        for name, answer, include_audio in scenarios:
            with self.subTest(name=name):
                verdicts = {} if answer is None else {"fused": answer}
                traces = []
                core, _models = trusted_core(
                    answers=verdicts,
                    on_trace=traces.append,
                )
                apply(core, "fused", "fused", "fused")
                bundle = fused_bundle(
                    watch_revision=core.goals.revision,
                    application=application_for(core, "fused"),
                    include_audio=include_audio,
                )
                async def drive():
                    completion = await core.perceive_fused_judged(
                        VIDEO_PAYLOAD,
                        t=1.0,
                        goal_ids=("fused",),
                        _fused_bundles_by_goal={"fused": bundle},
                    )
                    return await core.reduce_fused_completion(completion)

                events = asyncio.run(drive())
                self.assertFalse(any(event.action == "fire" for event in events))
                if name != "silent":
                    self.assertTrue(any(event.action == "ask" for event in events))
                    ask = next(event for event in events if event.action == "ask")
                    self.assertIsNotNone(ask._fused_evidence_bundle)
                identity_rows = [
                    row
                    for row in traces
                    if getattr(row, "data", {}).get("evidence_seq") is not None
                    or getattr(row, "data", {}).get("dispatch_seq") is not None
                    or getattr(row, "data", {}).get("judgment_event_id") is not None
                ]
                self.assertTrue(identity_rows)


if __name__ == "__main__":
    unittest.main()
