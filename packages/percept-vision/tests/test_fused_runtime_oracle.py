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
import sys
import types
import unittest
from copy import deepcopy
from dataclasses import FrozenInstanceError, dataclass, is_dataclass, replace
from unittest.mock import patch

from percept import Percept
from percept.agent import _SEQ_MAX_KEYS
from percept.adapters import Verdict
from percept.backends.fake import FakeVision
from percept.backends.gemini import GeminiVision
from percept.contracts import AudioFrame, WatchPlan
from percept.direct_judge import DirectJudge, DirectResult, DirectRouting
from percept.events import AudioUnit
from percept.goals import GoalOp
from percept.harness.tier0 import CivilTime, gray_frame
from percept.incidents import make_incident
from percept.runtime.envelope import SourceEnvelope, SourceFormat
from percept.runtime.worker import RuntimeStats, _JudgeScheduler, run
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
        self.release_by_condition = {}
        self.cancel_resistant = set()
        self.cancelled_conditions = set()
        self.cancellation_started = asyncio.Event()
        self.cancellation_release = asyncio.Event()
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
            release = self.release_by_condition.get(
                condition,
                self.slow_release,
            )
            try:
                await release.wait()
            except asyncio.CancelledError:
                if condition not in self.cancel_resistant:
                    raise
                self.cancelled_conditions.add(condition)
                self.cancellation_started.set()
                await self.cancellation_release.wait()
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


def provider_wire_modules() -> dict[str, types.ModuleType]:
    """Minimal google-genai wire types for the already-injected fake provider."""
    google = types.ModuleType("google")
    genai = types.ModuleType("google.genai")
    google_types = types.ModuleType("google.genai.types")

    class Part:
        @staticmethod
        def from_bytes(**kwargs):
            return types.SimpleNamespace(**kwargs)

    class GenerateContentConfig:
        def __init__(self, **kwargs):
            vars(self).update(kwargs)

    google.genai = genai
    genai.types = google_types
    google_types.Part = Part
    google_types.GenerateContentConfig = GenerateContentConfig
    return {
        "google": google,
        "google.genai": genai,
        "google.genai.types": google_types,
    }


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


def prepare_fused_job(
    core,
    bundle,
    *,
    goal_id: str,
    source: str = "camera",
    source_t: float = 1.0,
    goals_revision: int | None = None,
    source_generation: int | None = None,
    fused_generation: int | None = None,
):
    goals_revision = (
        core.goals.revision if goals_revision is None else goals_revision
    )
    source_generation = (
        core.source_continuity_generation(source)
        if source_generation is None
        else source_generation
    )
    fused_generation = (
        bundle.video_units[-1].generation
        if fused_generation is None
        else fused_generation
    )
    job = core.prepare_fused_judgment(
        bundle,
        goal_id=goal_id,
        source=source,
        source_t=source_t,
        goals_revision=goals_revision,
        source_generation=source_generation,
        fused_generation=fused_generation,
    )
    if job is None:
        raise AssertionError("valid fused fixture was not admitted")
    return job


async def flush_fused_traces(core) -> None:
    for trace_id, trace_t, stage, data in core.drain_fused_trace_records():
        await core._emit_trace(trace_id, trace_t, stage, **data)


async def dispatch_and_reduce_fused(
    core,
    bundle,
    *,
    goal_id: str,
    source: str = "camera",
    source_t: float = 1.0,
):
    job = prepare_fused_job(
        core,
        bundle,
        goal_id=goal_id,
        source=source,
        source_t=source_t,
    )
    completion = await core.dispatch_fused_judgment(job)
    events = core.reduce_fused_completion(
        completion,
        live_fused_generation=job.token.fused_generation,
    )
    await flush_fused_traces(core)
    return events


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


def audio_envelope(
    seq: int,
    start: float,
    end: float,
    *,
    epoch: int = 0,
    energy: float = 0.5,
):
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
        signal=energy,
        audio_direction="input",
    )


def fused_bundle(
    *,
    watch_revision: int,
    application: str = "runtime-oracle",
    include_audio: bool = True,
    video_payload=VIDEO_PAYLOAD,
    audio_payload: AudioFrame | None = None,
    policy_window_s: float = 4.0,
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
        if audio_payload is None:
            # The public fused Watch currently compiles a four-second leading window.  Supply its
            # entire configured interval so this helper means "complete" even when the private
            # synchronizer policy under test asks for a shorter or longer interval.
            audio_start = 1.0 - max(4.0, policy_window_s)
            sample_count = round((1.0 - audio_start) * 4)
            audio_payload = AudioFrame(
                pcm=struct.pack("<" + "h" * sample_count, *range(sample_count)),
                sample_rate=4,
                channels=1,
                sample_format="s16le",
                t_start=audio_start,
                t_end=1.0,
            )
        audio_start = audio_payload.t_start
        audio_end = audio_payload.t_end
        assert audio_start is not None
        assert audio_end is not None
        canonical_audio = AudioUnit(
            seq=0,
            t_start=audio_start,
            t_end=audio_end,
            event_t=audio_end,
            chunk=audio_payload,
            sample_rate=audio_payload.sample_rate,
            channels=audio_payload.channels,
            sample_format=audio_payload.sample_format,
            direction="input",
            source_id="fixture-audio",
            source_epoch=0,
            source_sequence=0,
            observed_time=100.0 + audio_end,
            source_time=audio_end,
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
                event_start=audio_start,
                event_end=audio_end,
                observed_start=100.0 + audio_start,
                observed_end=100.0 + audio_end,
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
            for item in value.values()
            for blob in provider_bytes(item, _seen=seen)
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

        events = asyncio.run(
            dispatch_and_reduce_fused(core, bundle, goal_id="fused")
        )

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
            fused = await dispatch_and_reduce_fused(
                core,
                incomplete,
                goal_id="fused",
            )
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
        traces = []
        core, models = trusted_core(
            answers={"fused": RuntimeError("provider unavailable")},
            on_trace=traces.append,
        )
        apply(core, "fused", "fused", "fused")
        # Seed the in-progress dwell directly: the compiled fused route does not itself author a
        # duration Watch, but the failure reducer must still invoke the shared pending-dwell clear.
        core.goals.get("fused").state["hold_s"] = 5.0
        key = ("fused", None)
        core.gate._dwell_since[key] = 0.25
        core.gate._dwell_last_supported[key] = 0.5
        durable_gate_before = {
            name: deepcopy(getattr(core.gate, name))
            for name in ("_armed", "_last_fire", "_touched", "_no_evict")
        }
        bundle = fused_bundle(
            watch_revision=core.goals.revision,
            application=application_for(core, "fused"),
        )

        events = asyncio.run(
            dispatch_and_reduce_fused(core, bundle, goal_id="fused")
        )

        self.assertEqual(events, [])
        self.assertGreaterEqual(len(models.calls), 1)
        self.assertNotIn(key, core.gate._dwell_since)
        self.assertNotIn(key, core.gate._dwell_last_supported)
        self.assertEqual(
            {
                name: getattr(core.gate, name)
                for name in durable_gate_before
            },
            durable_gate_before,
        )
        degraded = [row for row in traces if row.stage == "judge.degraded"]
        classified = [row for row in traces if row.stage == "gate.classified"]
        self.assertEqual(len(degraded), 1)
        self.assertEqual(len(classified), 1)
        self.assertEqual(classified[0].data["action"], "ask")
        self.assertTrue(classified[0].data["degraded"])
        self.assertEqual(
            [
                (kind, health_key)
                for kind, health_key, _cause in core._judge_health_log
            ],
            [("judge_degraded", key)],
        )

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
            slow_job = prepare_fused_job(
                core,
                slow_bundle,
                goal_id="deliberate",
            )
            fast_job = prepare_fused_job(
                core,
                fast_bundle,
                goal_id="reflex",
            )
            slow = asyncio.create_task(
                core.dispatch_fused_judgment(slow_job)
            )
            await asyncio.wait_for(models.slow_started.wait(), timeout=0.25)
            fast_completion = await asyncio.wait_for(
                core.dispatch_fused_judgment(fast_job),
                timeout=0.25,
            )
            fast_events = core.reduce_fused_completion(
                fast_completion,
                live_fused_generation=fast_job.token.fused_generation,
            )
            self.assertFalse(slow.done())
            models.slow_release.set()
            slow_completion = await slow
            core.reduce_fused_completion(
                slow_completion,
                live_fused_generation=slow_job.token.fused_generation,
            )
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
            jobs = {
                goal_id: prepare_fused_job(
                    core,
                    bundles[goal_id],
                    goal_id=goal_id,
                    source_t=7.0,
                )
                for goal_id in dispatch_order
            }
            tasks = [
                asyncio.create_task(
                    core.dispatch_fused_judgment(jobs[goal_id])
                )
                for goal_id in dispatch_order
            ]
            await asyncio.wait_for(models.all_started.wait(), timeout=0.25)
            self.assertTrue(all(not task.done() for task in tasks))
            models.slow_release.set()
            completions = await asyncio.gather(*tasks)
            completions_by_goal = {
                completion.job.token.goal_id: completion
                for completion in completions
            }
            events = []
            for goal_id in reduction_order:
                events.extend(
                    core.reduce_fused_completion(
                        completions_by_goal[goal_id],
                        live_fused_generation=jobs[goal_id].token.fused_generation,
                    )
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
            job = prepare_fused_job(
                core,
                bundle,
                goal_id="stale",
                goals_revision=revision,
            )
            task = asyncio.create_task(
                core.dispatch_fused_judgment(job)
            )
            await asyncio.wait_for(models.slow_started.wait(), timeout=0.25)
            models.slow_release.set()
            completion = await task
            core.apply_goal_ops(
                [GoalOp(op="update", goal_id="stale", fields={"say": "new text"})]
            )
            state_before_reduce = core.gate.snapshot()
            log_before_reduce = tuple(core.action_log.entries())
            events = core.reduce_fused_completion(
                completion,
                live_fused_generation=job.token.fused_generation,
            )
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
            job = prepare_fused_job(
                core,
                bundle,
                goal_id="stale",
                source_generation=generation,
            )
            task = asyncio.create_task(
                core.dispatch_fused_judgment(job)
            )
            await asyncio.wait_for(models.slow_started.wait(), timeout=0.25)
            models.slow_release.set()
            completion = await task
            core.advance_source_continuity("camera")
            state_before_reduce = core.gate.snapshot()
            log_before_reduce = tuple(core.action_log.entries())
            events = core.reduce_fused_completion(
                completion,
                live_fused_generation=job.token.fused_generation,
            )
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


class FusedPromotionBlockerTests(unittest.TestCase):
    def test_completed_result_invalidated_before_drain_has_admission_drop_receipt(self):
        async def drive():
            core, _models = trusted_core(
                answers={"completed": Verdict(True, 0.99, "completed")}
            )
            apply(core, "completed", "completed", "fused")
            bundle = fused_bundle(
                watch_revision=core.goals.revision,
                application=application_for(core, "completed"),
            )
            job = prepare_fused_job(
                core,
                bundle,
                goal_id="completed",
            )

            async def dispatch(prepared, _source=None, **_ignored):
                return await core.dispatch_fused_judgment(prepared)

            scheduler = _JudgeScheduler(
                core,
                "camera",
                RuntimeStats(),
                perceive=dispatch,
            )
            scheduler.submit(1.0, job)
            await scheduler.flush()
            core.advance_source_continuity("camera")
            scheduler.invalidate_source_continuity(("source_gap",))
            drained = scheduler.drain()
            self.assertEqual(len(drained), 1)
            events = core.reduce_fused_completion(
                drained[0][1],
                live_fused_generation=job.token.fused_generation,
            )
            return job, events, core.drain_fused_trace_records()

        job, events, drop_receipts = asyncio.run(bounded(drive()))
        self.assertEqual(events, [])
        drops = [
            data
            for _trace_id, _at, stage, data in drop_receipts
            if stage == "judgment.dropped"
        ]
        self.assertEqual(len(drops), 1)
        data = drops[0]
        self.assertEqual(data["admission_id"], job.token.admission_id)
        self.assertEqual(data["reason"], "source_generation")
        self.assertEqual(data["slot"], "completed")

    def test_duplicate_completion_replay_stays_inert_past_reducer_key_bound(self):
        core, _models = trusted_core()
        goal_ids = tuple(f"replay-{index}" for index in range(_SEQ_MAX_KEYS + 2))
        receipt = apply_watch_plan(
            core,
            [
                WatchPlan(id=goal_id, condition=goal_id, sense="fused")
                for goal_id in goal_ids
            ],
        )
        self.assertEqual(receipt.status, "applied")
        revision = core.goals.revision

        def reduce_once(goal_id):
            bundle = fused_bundle(
                watch_revision=revision,
                application=application_for(core, goal_id),
                include_audio=False,
            )
            job = prepare_fused_job(core, bundle, goal_id=goal_id)
            completion = core.complete_fused_locally(job)
            events = core.reduce_fused_completion(
                completion,
                live_fused_generation=job.token.fused_generation,
            )
            return job, completion, events

        original_job, original_completion, original_events = reduce_once(goal_ids[0])
        self.assertEqual([event.action for event in original_events], ["ask"])

        state_before_duplicate = core.gate.snapshot()
        log_before_duplicate = tuple(core.action_log.entries())
        self.assertEqual(
            core.reduce_fused_completion(
                original_completion,
                live_fused_generation=original_job.token.fused_generation,
            ),
            [],
        )
        self.assertEqual(core.gate.snapshot(), state_before_duplicate)
        self.assertEqual(tuple(core.action_log.entries()), log_before_duplicate)
        core.drain_fused_trace_records()

        for goal_id in goal_ids[1:]:
            _job, _completion, events = reduce_once(goal_id)
            self.assertEqual([event.action for event in events], ["ask"])

        state_before_replay = core.gate.snapshot()
        log_before_replay = tuple(core.action_log.entries())
        self.assertEqual(
            core.reduce_fused_completion(
                original_completion,
                live_fused_generation=original_job.token.fused_generation,
            ),
            [],
        )
        self.assertEqual(core.gate.snapshot(), state_before_replay)
        self.assertEqual(tuple(core.action_log.entries()), log_before_replay)
        drops = [
            data
            for _trace_id, _at, stage, data in core.drain_fused_trace_records()
            if stage == "judgment.dropped"
        ]
        self.assertEqual(len(drops), 1)
        self.assertEqual(drops[0]["admission_id"], original_job.token.admission_id)

    def test_audio_acceptance_across_fused_watches_is_transactional(self):
        async def drive():
            core, models = trusted_core(
                answers={
                    "good fused": Verdict(True, 0.99, "good"),
                    "rejecting fused": Verdict(True, 0.99, "bad"),
                }
            )
            apply(core, "good", "good fused", "fused")
            apply(core, "rejecting", "rejecting fused", "fused")
            rejecting_application = application_for(core, "rejecting")
            audio_processed = asyncio.Event()

            async def audio_source():
                yield audio_envelope(0, 0.0, 1.0)
                audio_processed.set()

            async def video_source():
                await audio_processed.wait()
                yield video_envelope(0, 1.0)
                yield video_envelope(1, 2.0, changed=True)

            from percept.fused_sync import _FusedSynchronizer

            original_accept = _FusedSynchronizer.accept

            def reject_one(synchronizer, accepted):
                if (
                    accepted.modality == "audio"
                    and synchronizer.watch_application == rejecting_application
                ):
                    raise RuntimeError("adversarial per-Watch audio rejection")
                return original_accept(synchronizer, accepted)

            asks = []
            with patch.object(_FusedSynchronizer, "accept", reject_one):
                incidents = await collect(
                    core,
                    source=video_source(),
                    audio_source=audio_source(),
                    audio_gate=None,
                    timebase="media",
                    judge_dispatch="inline",
                    on_ask=asks.append,
                )
            return core, models, incidents, asks

        core, models, incidents, asks = asyncio.run(bounded(drive()))
        self.assertEqual(incidents, [])
        self.assertEqual(models.calls, [])
        self.assertTrue(asks)
        self.assertEqual(len(core.action_log.entries("fire")), 0)

    def test_mixed_lanes_reduce_reverse_completion_without_cross_goal_mutation(self):
        async def drive():
            traces = []
            core, models = trusted_core(
                answers={
                    "ordinary lane": Verdict(True, 0.99, "ordinary"),
                    "fused lane": Verdict(True, 0.99, "fused"),
                },
                blocked={"ordinary lane", "fused lane"},
                on_trace=traces.append,
            )
            apply(core, "ordinary", "ordinary lane", "video")
            apply(core, "fused", "fused lane", "fused")
            models.release_by_condition = {
                condition: asyncio.Event()
                for condition in ("ordinary lane", "fused lane")
            }
            bundle = fused_bundle(
                watch_revision=core.goals.revision,
                application=application_for(core, "fused"),
            )
            fused_job = prepare_fused_job(
                core,
                bundle,
                goal_id="fused",
                source_t=7.0,
            )

            async def dispatch(prepared, _source=None, **_ignored):
                return await core.dispatch_fused_judgment(prepared)

            ordinary_scheduler = _JudgeScheduler(
                core,
                "camera",
                RuntimeStats(),
            )
            fused_scheduler = _JudgeScheduler(
                core,
                "camera",
                RuntimeStats(),
                perceive=dispatch,
            )
            ordinary_scheduler.submit(
                7.0,
                VIDEO_PAYLOAD,
                goal_ids=("ordinary",),
            )
            fused_scheduler.submit(7.0, fused_job)
            try:
                await asyncio.wait_for(models.all_started.wait(), timeout=0.25)
            except TimeoutError as exc:
                raise AssertionError(
                    ("provider lanes did not both start", models.started_conditions, len(models.calls))
                ) from exc

            models.release_by_condition["fused lane"].set()
            await asyncio.wait_for(fused_scheduler.flush(), timeout=0.25)
            self.assertEqual(ordinary_scheduler.active, 1)
            fused_results = fused_scheduler.drain()
            self.assertEqual(len(fused_results), 1)
            fused_events = core.reduce_fused_completion(
                fused_results[0][1],
                live_fused_generation=fused_job.token.fused_generation,
            )
            await flush_fused_traces(core)

            models.release_by_condition["ordinary lane"].set()
            await asyncio.wait_for(ordinary_scheduler.flush(), timeout=0.25)
            ordinary_results = ordinary_scheduler.drain()
            self.assertEqual(len(ordinary_results), 1)
            ordinary_events = ordinary_results[0][1]
            return core, traces, fused_events + ordinary_events

        core, traces, events = asyncio.run(bounded(drive()))
        fires = [event for event in events if event.action == "fire"]
        self.assertEqual(
            tuple(event.goal_id for event in fires),
            ("fused", "ordinary"),
        )
        self.assertEqual(
            {event.goal_id for event in fires},
            {"ordinary", "fused"},
        )
        self.assertEqual(
            [record.goal_id for record in core.action_log.entries("fire")],
            ["fused", "ordinary"],
        )
        self.assertEqual(
            {event.judgment.key for event in fires},
            {("ordinary", None), ("fused", None)},
        )
        traced_goals = {
            row.goal_id or row.data.get("goal_id")
            for row in traces
            if row.goal_id is not None or row.data.get("goal_id") is not None
        }
        self.assertEqual(traced_goals, {"ordinary", "fused"})

    def test_mixed_inflight_revision_and_source_invalidation_is_cognition_inert(self):
        async def drive():
            traces = []
            core, models = trusted_core(
                answers={
                    "ordinary stale": Verdict(True, 0.99, "ordinary"),
                    "fused stale": Verdict(True, 0.99, "fused"),
                },
                blocked={"ordinary stale", "fused stale"},
                on_trace=traces.append,
            )
            apply(core, "ordinary", "ordinary stale", "video")
            apply(core, "fused", "fused stale", "fused")
            models.release_by_condition = {
                condition: asyncio.Event()
                for condition in ("ordinary stale", "fused stale")
            }
            revision = core.goals.revision
            source_generation = core.source_continuity_generation("camera")
            bundle = fused_bundle(
                watch_revision=revision,
                application=application_for(core, "fused"),
            )
            fused_job = prepare_fused_job(
                core,
                bundle,
                goal_id="fused",
                goals_revision=revision,
                source_generation=source_generation,
            )
            ordinary_task = asyncio.create_task(
                core.perceive_judged(
                    VIDEO_PAYLOAD,
                    "camera",
                    t=9.0,
                    expected_goals_revision=revision,
                    expected_source_generation=source_generation,
                    goal_ids=("ordinary",),
                )
            )
            fused_task = asyncio.create_task(
                core.dispatch_fused_judgment(fused_job)
            )
            await asyncio.wait_for(models.all_started.wait(), timeout=0.25)

            core.apply_goal_ops(
                [
                    GoalOp(
                        op="update",
                        goal_id="fused",
                        fields={"say": "new fused text"},
                    )
                ]
            )
            core.advance_source_continuity("camera")
            gate_before = core.gate.snapshot()
            log_before = tuple(core.action_log.entries())
            trace_count_before = len(traces)

            models.release_by_condition["fused stale"].set()
            models.release_by_condition["ordinary stale"].set()
            ordinary_events, fused_completion = await asyncio.gather(
                ordinary_task,
                fused_task,
            )
            fused_events = core.reduce_fused_completion(
                fused_completion,
                live_fused_generation=fused_job.token.fused_generation,
            )
            await flush_fused_traces(core)
            return (
                core,
                traces[trace_count_before:],
                ordinary_events,
                fused_events,
                gate_before,
                log_before,
            )

        core, new_traces, ordinary, fused, gate_before, log_before = asyncio.run(
            bounded(drive())
        )
        self.assertEqual(ordinary, [])
        self.assertEqual(fused, [])
        self.assertEqual(core.gate.snapshot(), gate_before)
        self.assertEqual(tuple(core.action_log.entries()), log_before)
        self.assertTrue(new_traces)
        self.assertTrue(
            all(row.stage == "judgment.dropped" for row in new_traces)
        )
        self.assertEqual(
            {
                row.goal_id or row.data.get("goal_id")
                for row in new_traces
            },
            {"fused"},
        )

    def test_mixed_eos_hung_lane_is_bounded_and_reflex_result_survives(self):
        async def drive():
            current = asyncio.current_task()
            baseline_tasks = {
                task
                for task in asyncio.all_tasks()
                if task is not current
            }
            core, models = trusted_core(
                answers={
                    "hung deliberate": Verdict(True, 0.99, "deliberate"),
                    "completed reflex": Verdict(True, 0.99, "reflex"),
                },
                blocked={"hung deliberate"},
            )
            models.cancel_resistant.add("hung deliberate")
            apply(core, "deliberate", "hung deliberate", "fused")
            apply(core, "reflex", "completed reflex", "fused")
            audio_processed = asyncio.Event()

            async def audio_source():
                yield audio_envelope(0, 0.0, 1.0)
                yield audio_envelope(1, 1.0, 2.0)
                audio_processed.set()

            async def video_source():
                await audio_processed.wait()
                yield video_envelope(0, 1.0)
                yield video_envelope(1, 2.0, changed=True)

            health = []
            incidents = await collect(
                core,
                source=video_source(),
                audio_source=audio_source(),
                audio_gate=None,
                timebase="media",
                judge_dispatch="async",
                judge_stall_after_s=0.05,
                on_health=health.append,
            )
            effects_before_release = (
                len(models.calls),
                tuple(core.action_log.entries()),
            )
            models.cancellation_release.set()
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            effects_after_release = (
                len(models.calls),
                tuple(core.action_log.entries()),
            )
            lingering = [
                task
                for task in asyncio.all_tasks()
                if (
                    task is not asyncio.current_task()
                    and task not in baseline_tasks
                    and not task.done()
                )
            ]
            return (
                core,
                models,
                incidents,
                health,
                effects_before_release,
                effects_after_release,
                lingering,
            )

        # Keep this a cold-start test: the injected provider needs only the
        # stable wire constructors, not a cache-priming import of the entire
        # optional google-genai SDK before the hard outer deadline begins.
        with patch.dict(sys.modules, provider_wire_modules()):
            (
                core,
                models,
                incidents,
                health,
                effects_before_release,
                effects_after_release,
                lingering,
            ) = asyncio.run(bounded(drive(), timeout=1.0))
        self.assertEqual(
            [incident.concern_id for incident in incidents],
            ["reflex"],
            (
                models.started_conditions,
                len(models.calls),
                [(event.kind, event.data) for event in health],
            ),
        )
        self.assertIn("hung deliberate", models.cancelled_conditions)
        self.assertEqual(lingering, [])
        self.assertEqual(effects_after_release, effects_before_release)
        self.assertEqual(
            [record.goal_id for record in core.action_log.entries("fire")],
            ["reflex"],
        )
        reported = {
            event.kind
            for event in health
            if event.kind
            in {
                "judge_scheduler_stalled",
                "judge_source_discontinuity_dropped",
                "judgment_dropped",
                "judge_eos_timeout",
            }
        }
        self.assertTrue(reported)

    def test_provider_owned_nested_verdict_is_frozen_before_completion(self):
        async def drive():
            core, _models = trusted_core()
            apply(core, "nested", "nested", "fused")
            bundle = fused_bundle(
                watch_revision=core.goals.revision,
                application=application_for(core, "nested"),
            )
            original_job = prepare_fused_job(
                core,
                bundle,
                goal_id="nested",
            )
            provider_owned = Verdict(
                True,
                0.99,
                "nested",
                usage={
                    "input_tokens": 3,
                    "provider": {"ids": ["original"]},
                },
                value={"amounts": [1, {"currency": "USD"}]},
                normalized_value={"amounts": [1]},
            )

            async def provider(*_args, **_kwargs):
                return provider_owned

            job = replace(original_job, dispatch=provider)
            completion = await core.dispatch_fused_judgment(job)
            provider_owned.satisfied = False
            provider_owned.confidence = 0.0
            provider_owned.evidence = "mutated"
            provider_owned.usage["provider"]["ids"].append("mutated")
            provider_owned.value["amounts"][1]["currency"] = "EUR"
            provider_owned.normalized_value["amounts"].append(2)
            frozen = completion.verdict.to_verdict()
            events = core.reduce_fused_completion(
                completion,
                live_fused_generation=job.token.fused_generation,
            )
            return frozen, events

        frozen, events = asyncio.run(bounded(drive()))
        self.assertTrue(frozen.satisfied)
        self.assertEqual(frozen.confidence, 0.99)
        self.assertEqual(frozen.evidence, "nested")
        self.assertEqual(
            frozen.usage["provider"],
            {"ids": ["original"]},
        )
        self.assertEqual(
            frozen.value,
            {"amounts": [1, {"currency": "USD"}]},
        )
        self.assertEqual(frozen.normalized_value, {"amounts": [1]})
        self.assertEqual(
            [event.goal_id for event in events if event.action == "fire"],
            ["nested"],
        )

    def test_provider_failure_is_pure_for_all_cognition_state_owners(self):
        class TrapCache:
            def __init__(self):
                self.layers = []
                self.marker = {"nested": ["unchanged"]}

            def resolve(self, _key):
                raise AssertionError("fused failure touched verdict cache")

            def store(self, *_args, **_kwargs):
                raise AssertionError("fused failure touched verdict cache")

        async def drive():
            core, _models = trusted_core(
                answers={"failure": RuntimeError("provider unavailable")}
            )
            apply(core, "failure", "failure", "fused")
            core.cache = TrapCache()
            bundle = fused_bundle(
                watch_revision=core.goals.revision,
                application=application_for(core, "failure"),
            )
            job = prepare_fused_job(core, bundle, goal_id="failure")
            state_before = {
                "gate": deepcopy(core.gate.snapshot()),
                "documents": deepcopy(core._doc_episode_hwm),
                "claims": deepcopy(core._claim_states),
                "claim_hwm": deepcopy(core._claim_hwm),
                "claim_episode": deepcopy(core._claim_episode_id),
                "episodes": deepcopy(core.episodes),
                "cache": deepcopy(core.cache.marker),
                "actions": tuple(core.action_log.entries()),
                "supported": deepcopy(core.last_supported),
                "satisfied": deepcopy(core.last_satisfied),
            }
            completion = await core.dispatch_fused_judgment(job)
            events = core.reduce_fused_completion(
                completion,
                live_fused_generation=job.token.fused_generation,
            )
            state_after = {
                "gate": deepcopy(core.gate.snapshot()),
                "documents": deepcopy(core._doc_episode_hwm),
                "claims": deepcopy(core._claim_states),
                "claim_hwm": deepcopy(core._claim_hwm),
                "claim_episode": deepcopy(core._claim_episode_id),
                "episodes": deepcopy(core.episodes),
                "cache": deepcopy(core.cache.marker),
                "actions": tuple(core.action_log.entries()),
                "supported": deepcopy(core.last_supported),
                "satisfied": deepcopy(core.last_satisfied),
            }
            return completion, events, state_before, state_after

        completion, events, state_before, state_after = asyncio.run(
            bounded(drive())
        )
        self.assertIsNotNone(completion.failure)
        self.assertEqual(events, [])
        self.assertEqual(state_after, state_before)

    def test_exact_bundle_identity_mismatch_is_typed_and_validation_is_pure(self):
        core, models = trusted_core()
        apply(core, "identity", "identity", "fused")
        bundle = fused_bundle(
            watch_revision=core.goals.revision,
            application=application_for(core, "identity"),
            include_audio=False,
        )
        job = prepare_fused_job(core, bundle, goal_id="identity")
        completion = core.complete_fused_locally(job)
        forged_bundle = replace(
            job.bundle,
            evidence_seq=job.token.evidence_seq + 1,
        )
        forged = replace(
            completion,
            job=replace(job, bundle=forged_bundle),
        )
        gate_before = deepcopy(core.gate.snapshot())
        actions_before = tuple(core.action_log.entries())
        episodes_before = deepcopy(core.episodes)

        events = core.reduce_fused_completion(
            forged,
            live_fused_generation=job.token.fused_generation,
        )

        self.assertEqual(events, [])
        self.assertEqual(models.calls, [])
        self.assertEqual(core.gate.snapshot(), gate_before)
        self.assertEqual(tuple(core.action_log.entries()), actions_before)
        self.assertEqual(core.episodes, episodes_before)
        drops = [
            data
            for _trace_id, _at, stage, data in core.drain_fused_trace_records()
            if stage == "judgment.dropped"
        ]
        self.assertEqual(
            drops,
            [
                {
                    "goal_id": "identity",
                    "admission_id": job.token.admission_id,
                    "evidence_seq": job.token.evidence_seq,
                    "reason": "evidence_identity",
                    "slot": "completed",
                }
            ],
        )

    def test_scheduler_submit_after_close_or_fatal_publishes_nothing(self):
        async def drive():
            core, _models = trusted_core()

            async def succeed(*_args, **_kwargs):
                return []

            closed_stats = RuntimeStats()
            closed = _JudgeScheduler(
                core,
                "camera",
                closed_stats,
                perceive=succeed,
            )
            await closed.close(drain=True)
            closed_before = (
                closed_stats.judge_calls,
                closed_stats.perceive_dispatches,
                closed_stats.judge_superseded,
            )
            closed.submit(1.0, object())

            async def explode(*_args, **_kwargs):
                raise RuntimeError("fatal judge")

            fatal_stats = RuntimeStats()
            fatal = _JudgeScheduler(
                core,
                "camera",
                fatal_stats,
                perceive=explode,
            )
            fatal.submit(1.0, object())
            await fatal.flush()
            with self.assertRaisesRegex(RuntimeError, "fatal judge"):
                fatal.drain()
            fatal_before = (
                fatal_stats.judge_calls,
                fatal_stats.perceive_dispatches,
                fatal_stats.judge_superseded,
            )
            fatal.submit(2.0, object())
            return closed, closed_stats, closed_before, fatal, fatal_stats, fatal_before

        closed, closed_stats, closed_before, fatal, fatal_stats, fatal_before = (
            asyncio.run(bounded(drive()))
        )
        self.assertEqual((closed.active, closed.pending), (0, 0))
        self.assertEqual(len(closed._results), 0)
        self.assertEqual(closed.drain_trace(), [])
        self.assertEqual(
            (
                closed_stats.judge_calls,
                closed_stats.perceive_dispatches,
                closed_stats.judge_superseded,
            ),
            closed_before,
        )
        self.assertEqual((fatal.active, fatal.pending), (0, 0))
        self.assertEqual(len(fatal._results), 0)
        self.assertEqual(fatal.drain_trace(), [])
        self.assertEqual(
            (
                fatal_stats.judge_calls,
                fatal_stats.perceive_dispatches,
                fatal_stats.judge_superseded,
            ),
            fatal_before,
        )

    def test_projection_failure_publishes_no_ghost_admission(self):
        core, _models = trusted_core()
        apply(core, "projection", "projection", "fused")
        bundle = fused_bundle(
            watch_revision=core.goals.revision,
            application=application_for(core, "projection"),
        )
        sequencer_before = core._sequencer
        latest_before = dict(getattr(core, "_fused_latest_admission", {}))
        counter_before = getattr(core, "_fused_admission_counter", 0)

        with patch(
            "copy.deepcopy",
            side_effect=RuntimeError("adversarial projection failure"),
        ):
            with self.assertRaisesRegex(RuntimeError, "projection failure"):
                core.prepare_fused_judgment(
                    bundle,
                    goal_id="projection",
                    source="camera",
                    source_t=1.0,
                    goals_revision=core.goals.revision,
                    source_generation=core.source_continuity_generation("camera"),
                    fused_generation=bundle.video_units[-1].generation,
                )

        self.assertIs(core._sequencer, sequencer_before)
        self.assertEqual(
            dict(getattr(core, "_fused_latest_admission", {})),
            latest_before,
        )
        self.assertEqual(
            getattr(core, "_fused_admission_counter", 0),
            counter_before,
        )


class ExternalAcousticAdmissionOracleTests(unittest.TestCase):
    @staticmethod
    def _condition_calls(models, condition: str) -> int:
        return sum(condition in str(call) for call in models.calls)

    @staticmethod
    def _audio_plan(
        goal_id: str,
        condition: str,
        *,
        schedule=None,
        cadence_s: float | None = None,
    ) -> WatchPlan:
        explicit = (
            {"Goal.state.cadence.min_judge_interval": cadence_s}
            if cadence_s is not None
            else {}
        )
        return WatchPlan(
            id=goal_id,
            condition=condition,
            sense="audio",
            schedule=schedule,
            explicit=explicit,
        )

    def test_visual_motion_never_admits_or_spends_external_audio_watch(self):
        async def drive():
            core, models = trusted_core(
                answers={
                    "visual motion": Verdict(True, 0.99, "visual"),
                    "audio only": Verdict(True, 0.99, "audio"),
                }
            )
            apply(core, "visual", "visual motion", "video")
            receipt = apply_watch_plan(
                core,
                self._audio_plan("audio", "audio only"),
            )
            self.assertEqual(receipt.status, "applied")
            audio_processed = asyncio.Event()
            quiet = (
                audio_envelope(0, 0.0, 1.0, energy=0.0),
                audio_envelope(1, 1.0, 2.0, energy=0.0),
            )

            async def audio_source():
                for envelope in quiet:
                    yield envelope
                audio_processed.set()

            async def video_source():
                await audio_processed.wait()
                yield video_envelope(0, 1.0)
                yield video_envelope(1, 2.0, changed=True)

            stats = RuntimeStats()
            incidents = await collect(
                core,
                source=video_source(),
                audio_source=audio_source(),
                audio_gate="acoustic-onset",
                audio_settle_s=0.0,
                audio_wake={"high": 0.1, "low": 0.05},
                timebase="media",
                judge_dispatch="inline",
                stats=stats,
            )
            return models, incidents, stats

        models, incidents, stats = asyncio.run(bounded(drive()))
        self.assertEqual(
            [incident.concern_id for incident in incidents],
            ["visual"],
        )
        self.assertEqual(self._condition_calls(models, "visual motion"), 1)
        self.assertEqual(self._condition_calls(models, "audio only"), 0)
        self.assertEqual(stats.admission_by_goal.get("visual"), 1)
        self.assertIsNone(stats.admission_by_goal.get("audio"))
        self.assertEqual((stats.judge_calls, stats.perceive_dispatches), (1, 1))

    def test_acoustic_onset_admits_exact_audio_without_synthetic_video(self):
        async def drive():
            core, models = trusted_core(
                answers={
                    "visual only": Verdict(True, 0.99, "visual"),
                    "heard onset": Verdict(True, 0.99, "audio"),
                }
            )
            apply(core, "visual", "visual only", "video")
            receipt = apply_watch_plan(
                core,
                self._audio_plan("audio", "heard onset"),
            )
            self.assertEqual(receipt.status, "applied")
            quiet = audio_envelope(0, 0.0, 1.0, energy=0.0)
            onset = audio_envelope(1, 1.0, 2.0, energy=1.0)
            audio_processed = asyncio.Event()

            async def audio_source():
                yield quiet
                yield onset
                audio_processed.set()

            async def video_source():
                await audio_processed.wait()
                yield video_envelope(0, 1.0)
                yield video_envelope(1, 2.0)

            stats = RuntimeStats()
            incidents = await collect(
                core,
                source=video_source(),
                audio_source=audio_source(),
                audio_gate="acoustic-onset",
                audio_settle_s=0.0,
                audio_wake={"high": 0.1, "low": 0.05},
                timebase="media",
                judge_dispatch="inline",
                stats=stats,
            )
            return models, incidents, stats, quiet, onset

        models, incidents, stats, quiet, onset = asyncio.run(bounded(drive()))
        self.assertEqual([incident.concern_id for incident in incidents], ["audio"])
        self.assertEqual(self._condition_calls(models, "heard onset"), 1)
        self.assertEqual(self._condition_calls(models, "visual only"), 0)
        self.assertEqual((stats.judge_calls, stats.perceive_dispatches), (1, 1))
        provider_blobs = provider_bytes(models.calls[0])
        expected_pcm = quiet.payload.pcm + onset.payload.pcm
        self.assertTrue(any(expected_pcm in blob for blob in provider_blobs))
        self.assertTrue(
            all(VIDEO_PAYLOAD[1] not in blob for blob in provider_blobs)
        )
        snippet = incidents[0].snippet
        self.assertEqual(snippet.kind, "audio")
        self.assertEqual(
            [frame.source_sequence for frame in snippet.frames],
            [0, 1],
        )

    def test_external_audio_schedule_and_cadence_are_per_watch(self):
        async def drive():
            conditions = ("fast audio", "slow audio", "closed audio")
            core, models = trusted_core(
                answers={
                    condition: Verdict(False, 0.99, condition)
                    for condition in conditions
                }
            )
            receipt = apply_watch_plan(
                core,
                [
                    self._audio_plan("fast", "fast audio"),
                    self._audio_plan("slow", "slow audio", cadence_s=10.0),
                    self._audio_plan(
                        "closed",
                        "closed audio",
                        schedule={"days": [2]},
                    ),
                ],
            )
            self.assertEqual(receipt.status, "applied")
            first_pair = asyncio.Event()
            first_video = asyncio.Event()
            second_pair = asyncio.Event()

            async def audio_source():
                yield audio_envelope(0, 0.0, 1.0, energy=0.0)
                yield audio_envelope(1, 1.0, 2.0, energy=1.0)
                first_pair.set()
                await first_video.wait()
                yield audio_envelope(2, 2.0, 3.0, energy=0.0)
                yield audio_envelope(3, 3.0, 4.0, energy=1.0)
                second_pair.set()

            async def video_source():
                await first_pair.wait()
                yield video_envelope(0, 2.0)
                first_video.set()
                await second_pair.wait()
                yield video_envelope(1, 4.0)

            stats = RuntimeStats()
            incidents = await collect(
                core,
                source=video_source(),
                audio_source=audio_source(),
                audio_gate="acoustic-onset",
                audio_settle_s=0.0,
                audio_wake={
                    "high": 0.1,
                    "low": 0.05,
                    "refractory_s": 0.0,
                },
                timebase="media",
                clock=lambda: CivilTime(weekday=1, minute_of_day=600),
                judge_dispatch="inline",
                stats=stats,
            )
            return models, incidents, stats

        models, incidents, stats = asyncio.run(bounded(drive()))
        self.assertEqual(incidents, [])
        self.assertEqual(self._condition_calls(models, "fast audio"), 2)
        self.assertEqual(self._condition_calls(models, "slow audio"), 1)
        self.assertEqual(self._condition_calls(models, "closed audio"), 0)
        self.assertEqual(stats.admission_by_goal.get("fast"), 2)
        self.assertEqual(stats.admission_by_goal.get("slow"), 1)
        self.assertIsNone(stats.admission_by_goal.get("closed"))
        self.assertEqual((stats.judge_calls, stats.perceive_dispatches), (2, 2))

    def test_fused_motion_and_audio_wake_share_cadence_without_double_spend(self):
        async def drive():
            core, models = trusted_core(
                answers={"both lanes": Verdict(True, 0.99, "joined")}
            )
            receipt = apply_watch_plan(
                core,
                WatchPlan(
                    id="fused",
                    condition="both lanes",
                    sense="fused",
                    explicit={
                        "Goal.state.cadence.min_judge_interval": 10.0,
                    },
                ),
            )
            self.assertEqual(receipt.status, "applied")
            audio_processed = asyncio.Event()

            async def audio_source():
                yield audio_envelope(0, 0.0, 1.0, energy=0.0)
                yield audio_envelope(1, 1.0, 2.0, energy=1.0)
                audio_processed.set()

            async def video_source():
                await audio_processed.wait()
                yield video_envelope(0, 1.0)
                yield video_envelope(1, 2.0, changed=True)

            stats = RuntimeStats()
            health = []
            incidents = await collect(
                core,
                source=video_source(),
                audio_source=audio_source(),
                audio_gate="acoustic-onset",
                audio_settle_s=0.0,
                audio_wake={"high": 0.1, "low": 0.05},
                timebase="media",
                judge_dispatch="inline",
                stats=stats,
                on_health=health.append,
            )
            return models, incidents, stats, health

        models, incidents, stats, health = asyncio.run(bounded(drive()))
        self.assertEqual([incident.concern_id for incident in incidents], ["fused"])
        self.assertEqual(self._condition_calls(models, "both lanes"), 1)
        self.assertEqual(stats.admission_by_goal, {"fused": 1})
        self.assertEqual((stats.judge_calls, stats.perceive_dispatches), (1, 1))
        escalations = [event for event in health if event.kind == "escalation"]
        self.assertEqual(len(escalations), 1)
        self.assertEqual(escalations[0].data["cause"], "both")

    def test_dynamic_regate_preserves_survivor_and_removes_external_ids(self):
        async def drive():
            conditions = ("survivor audio", "removed audio", "new audio")
            core, models = trusted_core(
                answers={
                    condition: Verdict(False, 0.99, condition)
                    for condition in conditions
                }
            )
            receipt = apply_watch_plan(
                core,
                [
                    self._audio_plan(
                        "survivor",
                        "survivor audio",
                        cadence_s=10.0,
                    ),
                    self._audio_plan("removed", "removed audio"),
                ],
            )
            self.assertEqual(receipt.status, "applied")
            first_pair = asyncio.Event()
            first_video = asyncio.Event()
            second_pair = asyncio.Event()

            async def audio_source():
                yield audio_envelope(0, 0.0, 1.0, energy=0.0)
                yield audio_envelope(1, 1.0, 2.0, energy=1.0)
                first_pair.set()
                await first_video.wait()
                yield audio_envelope(2, 2.0, 3.0, energy=0.0)
                yield audio_envelope(3, 3.0, 4.0, energy=1.0)
                second_pair.set()

            async def video_source():
                await first_pair.wait()
                yield video_envelope(0, 2.0)
                core.apply_goal_ops(
                    [GoalOp(op="complete", goal_id="removed")]
                )
                added = apply_watch_plan(
                    core,
                    self._audio_plan("new", "new audio"),
                )
                self.assertEqual(added.status, "applied")
                first_video.set()
                await second_pair.wait()
                yield video_envelope(1, 4.0)

            stats = RuntimeStats()
            incidents = await collect(
                core,
                source=video_source(),
                audio_source=audio_source(),
                audio_gate="acoustic-onset",
                audio_settle_s=0.0,
                audio_wake={
                    "high": 0.1,
                    "low": 0.05,
                    "refractory_s": 0.0,
                },
                timebase="media",
                judge_dispatch="inline",
                stats=stats,
            )
            return models, incidents, stats

        models, incidents, stats = asyncio.run(bounded(drive()))
        self.assertEqual(incidents, [])
        self.assertEqual(self._condition_calls(models, "survivor audio"), 1)
        self.assertEqual(self._condition_calls(models, "removed audio"), 1)
        self.assertEqual(self._condition_calls(models, "new audio"), 1)
        self.assertEqual(stats.admission_by_goal.get("survivor"), 1)
        self.assertEqual(stats.admission_by_goal.get("removed"), 1)
        self.assertEqual(stats.admission_by_goal.get("new"), 1)


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
        # Cross-lane pacing deadlock regression: the audio lane waits until the pre-gap video was
        # consumed, while the video lane then waits until the accepted gap carrier is committed.
        # ``bounded`` below makes any cyclic wait a deterministic failure, not a hung test process.
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
    def test_provider_byte_walker_reaches_actual_trusted_request_parts(self):
        video_payload = ("image/jpeg", b"\xff\xd8byte-walker-video\xff\xd9")
        audio_payload = AudioFrame(
            pcm=struct.pack("<hhhh", 11, 12, 13, 14),
            sample_rate=4,
            channels=1,
            sample_format="s16le",
            t_start=0.0,
            t_end=1.0,
        )
        core, models = trusted_core(
            answers={"byte walker": Verdict(True, 0.99, "both present")}
        )

        verdict = asyncio.run(
            core.vision.judge(
                "byte walker",
                video_payload,
                frames=[video_payload],
                audio=audio_payload,
            )
        )

        self.assertTrue(verdict.satisfied)
        blobs = provider_bytes(models.calls[0])
        self.assertIn(video_payload[1], blobs)
        self.assertTrue(any(audio_payload.pcm in blob for blob in blobs))

    def test_exact_provider_payload_event_incident_and_trace_share_identity(self):
        video_payload = ("image/jpeg", b"\xff\xd8exact-video\xff\xd9")
        audio_payload = AudioFrame(
            pcm=struct.pack("<" + "h" * 16, *range(7, 23)),
            sample_rate=4,
            channels=1,
            sample_format="s16le",
            t_start=-3.0,
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

        events = asyncio.run(
            dispatch_and_reduce_fused(core, bundle, goal_id="fused")
        )
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

                events = asyncio.run(
                    dispatch_and_reduce_fused(core, bundle, goal_id="fused")
                )
                self.assertFalse(any(event.action == "fire" for event in events))
                if name == "ask":
                    self.assertTrue(any(event.action == "ask" for event in events))
                    ask = next(event for event in events if event.action == "ask")
                    self.assertIsNotNone(ask._fused_evidence_bundle)
                elif name == "degraded":
                    self.assertEqual(events, [])
                    degraded = [
                        row for row in traces if row.stage == "judge.degraded"
                    ]
                    classified = [
                        row for row in traces if row.stage == "gate.classified"
                    ]
                    self.assertEqual(len(degraded), 1)
                    self.assertEqual(len(classified), 1)
                    self.assertEqual(classified[0].data["action"], "ask")
                    self.assertTrue(classified[0].data["degraded"])
                    self.assertEqual(
                        [
                            (kind, key)
                            for kind, key, _cause in core._judge_health_log
                        ],
                        [("judge_degraded", ("fused", None))],
                    )
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
