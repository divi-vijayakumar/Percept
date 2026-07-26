"""Adversarial oracle for the unified cognition-owner lifecycle.

This module deliberately tests the private split-phase boundary.  Public mixed-runtime tests can
only report that a topology was rejected; these tests identify *why* it is not yet promotable:
provider work mutated cognition, a lifecycle token was not validated, scheduler teardown advanced a
shared generation, or the runtime still lacks the single ordered owner queue.

Run this file against the moving implementation, never this oracle worktree's package::

    PYTHONPATH=/private/tmp/percept-launch/fused-sync-v2/packages/percept-vision \
      python -m pytest --import-mode=importlib -q \
      packages/percept-vision/tests/test_shared_owner_lifecycle_oracle.py
"""

from __future__ import annotations

import asyncio
import inspect
import struct
from copy import deepcopy
from dataclasses import fields, is_dataclass, replace
from types import MappingProxyType

import pytest

from percept import Goal, Percept
from percept.adapters import Verdict
from percept.contracts import AudioFrame, WatchPlan
from percept.events import AudioUnit
from percept.watchplan_apply import apply_watch_plan


class _BarrierVision:
    """A provider double whose completions can be released in an adversarial order."""

    def __init__(self, goal_ids: tuple[str, ...]) -> None:
        self.started = {goal_id: asyncio.Event() for goal_id in goal_ids}
        self.release = {goal_id: asyncio.Event() for goal_id in goal_ids}

    async def detect(self, _frame, _goals):
        return []

    async def judge(self, condition, _frame, **_kwargs):
        self.started[condition].set()
        await self.release[condition].wait()
        return Verdict(True, 0.99, f"{condition} completed")


def _watch_core(
    goal_ids: tuple[str, ...] = ("ordinary",),
    *,
    vision=None,
    goal_state: dict | None = None,
    **create_kwargs,
) -> Percept:
    if vision is None:
        vision = _BarrierVision(goal_ids)
    core = Percept.create(
        vision=vision,
        discover_plugins=False,
        **create_kwargs,
    )
    for goal_id in goal_ids:
        core.add_goal(
            Goal(
                id=goal_id,
                condition=goal_id,
                say=f"{goal_id} fired",
                source="camera",
                state=deepcopy(goal_state or {"type": "watch"}),
            )
        )
    return core


async def _prepare_ordinary_jobs(
    core: Percept,
    goal_ids: tuple[str, ...],
    *,
    frame="frame",
    t: float = 1.0,
):
    """Exercise the owner-only observation and judgment admission seams."""
    revision = core.goals.revision
    generation = core.source_continuity_generation("camera")
    candidate = core.prepare_observation_candidate(
        frame,
        "camera",
        t=t,
        expected_goals_revision=revision,
        expected_source_generation=generation,
        goal_ids=goal_ids,
    )
    assert candidate is not None, "ordinary observation candidate was not admitted"
    assert core.launch_observation_candidate(candidate)
    observation = await core.dispatch_observation_candidate(candidate)
    reduction = core.reduce_observation_completion(observation)
    assert reduction.stale_reason is None
    jobs = tuple(
        core.prepare_frame_judgment(reduction, goal_id=goal_id)
        for goal_id in goal_ids
    )
    assert all(job is not None for job in jobs), (
        "ordinary FRAME_WATCH jobs must use the split-phase preparer"
    )
    return jobs


def _semantic_fingerprint(core: Percept) -> dict:
    """State that a provider task and stale completion are forbidden to mutate."""
    executor_contexts = getattr(core.executor, "_ctx", None)
    if executor_contexts is None:
        executor_contexts = getattr(core.executor, "_contexts", {})
    graph_state = {
        "entities": deepcopy(getattr(core.graph, "_entities", {})),
        "counter": getattr(core.graph, "_counter", None),
        "last_frame_ids": tuple(getattr(core.graph, "_last_frame_ids", ())),
    }
    return {
        "gate": deepcopy(core.gate.snapshot()),
        "executor": deepcopy(executor_contexts),
        "graph": graph_state,
        "window": deepcopy(core._window("camera").frames()),
        "actions": tuple(deepcopy(core.action_log.entries())),
        "episodes": deepcopy(core.episodes),
        "documents": deepcopy(getattr(core, "_doc_episode_hwm", {})),
        "claims": deepcopy(getattr(core, "_claim_states", {})),
        "claim_hwm": deepcopy(getattr(core, "_claim_hwm", {})),
        "supported": deepcopy(core.last_supported),
        "satisfied": deepcopy(core.last_satisfied),
    }


def _worker_source() -> str:
    import percept.runtime.worker as worker

    return inspect.getsource(worker)


def _fused_bundle(*, watch_revision: int, application: str):
    """Build one exact, complete private bundle without involving transport timing."""
    from percept.fused_sync import (
        _AcceptedMediaUnit,
        _FusedSyncPolicy,
        _FusedSynchronizer,
    )

    policy = _FusedSyncPolicy(
        window_s=4.0,
        allowed_lateness_s=0.25,
        max_skew_s=0.5,
        min_clock_confidence=0.9,
        max_units_per_modality=16,
        max_bytes_per_modality=1024 * 1024,
        version="shared-owner-oracle",
    )
    sync = _FusedSynchronizer(
        policy,
        watch_application=application,
        watch_revision=watch_revision,
    )
    video_payload = ("image/jpeg", b"\xff\xd8shared-owner-oracle\xff\xd9")
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
        clock_authority="fixture",
        clock_confidence=1.0,
        provenance=MappingProxyType({"carrier": "video"}),
        source_watermark=2.0,
        byte_size=len(video_payload[1]),
    )
    assert sync.accept(video)
    samples = tuple(range(16))
    audio_frame = AudioFrame(
        pcm=struct.pack("<" + "h" * len(samples), *samples),
        sample_rate=4,
        channels=1,
        sample_format="s16le",
        t_start=-3.0,
        t_end=1.0,
    )
    audio_unit = AudioUnit(
        seq=0,
        t_start=-3.0,
        t_end=1.0,
        event_t=1.0,
        chunk=audio_frame,
        sample_rate=4,
        channels=1,
        sample_format="s16le",
        direction="input",
        source_id="fixture-audio",
        source_epoch=0,
        source_sequence=0,
        observed_time=101.0,
        source_time=1.0,
    )
    assert sync.accept(
        _AcceptedMediaUnit(
            modality="audio",
            payload=audio_unit,
            source_id="fixture-audio",
            source_epoch=0,
            source_sequence=0,
            generation=0,
            watch_application=application,
            watch_revision=watch_revision,
            event_start=-3.0,
            event_end=1.0,
            observed_start=97.0,
            observed_end=101.0,
            clock_domain="source",
            clock_authority="fixture",
            clock_confidence=1.0,
            provenance=MappingProxyType({"carrier": "audio"}),
            source_watermark=2.0,
            byte_size=len(audio_frame.pcm),
        )
    )
    return sync.bundle_for(video)


def test_blocked_provider_completion_cannot_mutate_semantics_before_owner_reduction():
    async def drive():
        vision = _BarrierVision(("ordinary",))
        core = _watch_core(vision=vision)
        job = (await _prepare_ordinary_jobs(core, ("ordinary",)))[0]
        before_dispatch = _semantic_fingerprint(core)

        task = asyncio.create_task(core.dispatch_judgment(job))
        await asyncio.wait_for(vision.started["ordinary"].wait(), timeout=0.25)
        assert _semantic_fingerprint(core) == before_dispatch

        vision.release["ordinary"].set()
        completion = await asyncio.wait_for(task, timeout=0.25)
        # Merely returning from the provider lane is not owner reduction.
        assert _semantic_fingerprint(core) == before_dispatch

        events = core.reduce_judgment_completion(completion)
        assert [event.action for event in events] == ["fire"]
        assert _semantic_fingerprint(core) != before_dispatch

    asyncio.run(drive())


def test_reverse_provider_completion_is_inert_until_explicit_owner_reduction():
    async def drive():
        goal_ids = ("ordinary-a", "ordinary-b")
        vision = _BarrierVision(goal_ids)
        core = _watch_core(goal_ids, vision=vision)
        jobs = await _prepare_ordinary_jobs(core, goal_ids)
        baseline = _semantic_fingerprint(core)
        tasks = {
            job.token.goal_id: asyncio.create_task(core.dispatch_judgment(job))
            for job in jobs
        }
        await asyncio.wait_for(
            asyncio.gather(
                *(vision.started[goal_id].wait() for goal_id in goal_ids)
            ),
            timeout=0.25,
        )

        vision.release["ordinary-b"].set()
        completion_b = await asyncio.wait_for(tasks["ordinary-b"], timeout=0.25)
        assert _semantic_fingerprint(core) == baseline
        vision.release["ordinary-a"].set()
        completion_a = await asyncio.wait_for(tasks["ordinary-a"], timeout=0.25)
        assert _semantic_fingerprint(core) == baseline

        events = []
        events.extend(core.reduce_judgment_completion(completion_b))
        events.extend(core.reduce_judgment_completion(completion_a))
        assert [event.goal_id for event in events] == ["ordinary-b", "ordinary-a"]
        assert [row.goal_id for row in core.action_log.entries("fire")] == [
            "ordinary-b",
            "ordinary-a",
        ]

    asyncio.run(drive())


@pytest.mark.parametrize(
    ("field_name", "replacement"),
    (
        ("goal_object_token", object()),
        ("source_generation", 999),
        ("judge_binding_token", object()),
        ("watch_runtime_token", object()),
    ),
)
def test_stale_ordinary_tokens_are_semantically_inert(field_name, replacement):
    async def drive():
        vision = _BarrierVision(("ordinary",))
        core = _watch_core(vision=vision)
        job = (await _prepare_ordinary_jobs(core, ("ordinary",)))[0]
        task = asyncio.create_task(core.dispatch_judgment(job))
        await vision.started["ordinary"].wait()
        vision.release["ordinary"].set()
        completion = await task
        forged_token = replace(job.token, **{field_name: replacement})
        forged = replace(completion, job=replace(job, token=forged_token))
        before = _semantic_fingerprint(core)
        events = core.reduce_judgment_completion(forged)
        assert events == []
        assert _semantic_fingerprint(core) == before

    asyncio.run(drive())


def test_stale_goal_revision_is_semantically_inert():
    async def drive():
        from percept.goals import GoalOp

        vision = _BarrierVision(("ordinary",))
        core = _watch_core(vision=vision)
        job = (await _prepare_ordinary_jobs(core, ("ordinary",)))[0]
        task = asyncio.create_task(core.dispatch_judgment(job))
        await vision.started["ordinary"].wait()
        vision.release["ordinary"].set()
        completion = await task
        core.apply_goal_ops(
            [
                GoalOp(
                    op="update",
                    goal_id="ordinary",
                    fields={"say": "new presentation"},
                )
            ]
        )
        before = _semantic_fingerprint(core)
        assert core.reduce_judgment_completion(completion) == []
        assert _semantic_fingerprint(core) == before

    asyncio.run(drive())


def test_stale_fused_generation_is_semantically_inert():
    from percept.backends.fake import FakeVision

    core = Percept.create(
        vision=FakeVision(script={"fused": Verdict(True, 0.99, "complete")}),
        discover_plugins=False,
    )
    receipt = apply_watch_plan(
        core,
        WatchPlan(id="fused", condition="fused", sense="fused"),
    )
    assert receipt.status == "applied", receipt
    bundle = _fused_bundle(
        watch_revision=core.goals.revision,
        application=core._watch_runtime_keys["fused"],
    )
    job = core.prepare_fused_judgment(
        bundle,
        goal_id="fused",
        source="camera",
        source_t=1.0,
        goals_revision=core.goals.revision,
        source_generation=core.source_continuity_generation("camera"),
        fused_generation=0,
    )
    assert job is not None
    completion = asyncio.run(core.dispatch_fused_judgment(job))
    before = _semantic_fingerprint(core)
    assert (
        core.reduce_fused_completion(
            completion,
            live_fused_generation=job.token.fused_generation + 1,
        )
        == []
    )
    assert _semantic_fingerprint(core) == before


def test_generic_validator_reports_each_lifecycle_coordinate_without_mutation():
    """A silent ``[]`` is not a stale receipt; the owner needs one typed reason."""
    validator = getattr(Percept, "validate_judgment_token", None)
    assert callable(validator), (
        "missing seam: Percept.validate_judgment_token must be a read-only generic "
        "validator for goal/source/runtime/judge/fused coordinates"
    )
    signature = inspect.signature(validator)
    assert tuple(signature.parameters)[:2] == ("self", "token")
    source = inspect.getsource(validator)
    for coordinate in (
        "goal_object_token",
        "goals_registry_revision",
        "watch_application",
        "watch_runtime_token",
        "judge_binding_token",
        "source_generation",
        "fused_generation",
        "evidence_seq",
    ):
        assert coordinate in source, f"validator omits stale coordinate {coordinate!r}"


def test_one_completion_carrier_and_one_global_ready_ordinal_cover_every_route():
    import percept.agent as agent
    import percept.runtime.worker as worker

    completion_type = getattr(agent, "_JudgmentCompletion", None)
    assert completion_type is not None and is_dataclass(completion_type)
    assert "completion_ordinal" in {field.name for field in fields(completion_type)}

    # One generic preparer is load-bearing: route-specific full-core coroutines would restore a
    # second cognition owner for transition/process/audio.
    preparer = getattr(Percept, "prepare_judgment", None)
    assert callable(preparer), (
        "missing seam: one Percept.prepare_judgment must admit ordinary, transition, "
        "process, audio, fused, and local completions"
    )

    scheduler = getattr(worker, "_ProviderScheduler", None)
    assert scheduler is not None, (
        "missing seam: transport-only _ProviderScheduler with a shared owner-ready callback"
    )
    scheduler_source = inspect.getsource(scheduler)
    assert "completion_ordinal" in scheduler_source
    assert "ready" in scheduler_source or "callback" in scheduler_source
    assert "perceive_judged" not in scheduler_source
    assert "advance_source_continuity" not in scheduler_source


@pytest.mark.parametrize(
    "local_route",
    ("direct", "extractor", "process_qa", "transition", "process"),
)
def test_local_and_multi_call_routes_enter_the_same_ordered_completion_protocol(
    local_route,
):
    """These routes must not fall back to the old stateful ``perceive_judged`` task."""
    import percept.agent as agent

    completion_type = getattr(agent, "_JudgmentCompletion", None)
    assert completion_type is not None
    protocol = getattr(Percept, "prepare_judgment", None)
    assert callable(protocol), (
        f"missing seam for {local_route}: generic local/multi-call judgment preparation"
    )
    local_completion = getattr(Percept, "complete_judgment_locally", None)
    assert callable(local_completion), (
        f"missing seam for {local_route}: local results must enter the generic completion "
        "queue without provider spend"
    )


def test_audio_pump_is_command_only_and_never_mutates_owner_state():
    source = _worker_source()
    start = source.find("async def _pump_audio")
    assert start >= 0
    end = source.find("\n        async def ", start + 1)
    pump = source[start:] if end < 0 else source[start:end]
    forbidden = {
        "core.push_audio(": "AudioWindow",
        "synchronizer.accept(": "fused synchronizer",
        "_advance_fused_generation(": "fused generation",
        "advance_source_continuity(": "source generation",
        "_queue_fused_bundle(": "judgment admission",
        "prepare_fused_judgment(": "judgment admission",
    }
    hits = {
        mutation: owner
        for mutation, owner in forbidden.items()
        if mutation in pump
    }
    assert not hits, (
        "audio pump is a second cognition owner; it must enqueue canonical media/boundary "
        f"commands only, but directly mutates {hits}"
    )
    assert "put" in pump or "enqueue" in pump, (
        "missing seam: audio pump has no bounded owner-command publication"
    )


def test_transport_lane_close_never_advances_shared_generation_per_lane():
    from percept.runtime.worker import RuntimeStats, _JudgeScheduler

    async def drive():
        core = _watch_core()
        before = core.source_continuity_generation("camera")
        lanes = [
            _JudgeScheduler(core, "camera", RuntimeStats())
            for _ in range(4)
        ]
        for lane in lanes:
            await lane.close(drain=False)
        return before, core.source_continuity_generation("camera")

    before, after = asyncio.run(drive())
    assert after == before, (
        "provider scheduler teardown advanced shared source continuity once per lane; "
        "only the runtime owner may advance a real boundary exactly once"
    )


def test_owner_command_queue_is_bounded_and_has_one_monotonic_ordinal():
    import percept.runtime.worker as worker

    queue_type = next(
        (
            getattr(worker, name)
            for name in (
                "_OwnerCommandQueue",
                "_RuntimeOwnerQueue",
                "_OwnerReadyQueue",
            )
            if hasattr(worker, name)
        ),
        None,
    )
    assert queue_type is not None, (
        "missing seam: bounded owner-command queue shared by audio, observation, "
        "ordinary, transition, process, audio judgment, and fused judgment"
    )
    source = inspect.getsource(queue_type)
    assert "max" in source.lower(), "owner command queue has no explicit hard bound"
    assert "ordinal" in source, "owner command queue does not author a global ordinal"
    assert "overflow" in source.lower(), (
        "owner command overflow must become a typed continuity boundary"
    )


def test_incidents_actions_and_traces_materialize_before_any_async_callback():
    import percept.runtime.worker as worker

    materialize = getattr(worker, "_materialize_reduction", None)
    deliver = getattr(worker, "_deliver_materialized", None)
    assert callable(materialize) and not inspect.iscoroutinefunction(materialize), (
        "missing seam: synchronous _materialize_reduction must freeze every Ask/Incident, "
        "action, and trace description immediately after owner reduction"
    )
    assert callable(deliver) and inspect.iscoroutinefunction(deliver), (
        "missing seam: async callbacks must consume already-materialized outputs"
    )
    materialize_source = inspect.getsource(materialize)
    assert "await " not in materialize_source
    assert "make_incident" in materialize_source
    assert "judgment_trace_id" in materialize_source


def test_eos_joins_all_lanes_concurrently_and_never_uses_unbounded_ordinary_flush():
    source = _worker_source()
    assert "await _judge_sched.flush()" not in source, (
        "ordinary EOS still has an unbounded sequential flush outside the shared lane deadline"
    )
    assert "_ProviderScheduler" in source, (
        "missing seam: EOS cannot join every route until all routes use transport-only lanes"
    )
    assert "asyncio.as_completed" in source or "asyncio.gather" in source
    assert "judge_eos_timeout" in source


def test_mixed_topology_is_supported_only_through_the_shared_owner():
    source = _worker_source()
    assert "_UnsafeMixedTopologyError" not in source, (
        "missing shared-owner promotion seam: runtime still rejects mixed ordinary/audio/fused "
        "topology instead of routing it through one owner"
    )
    assert "perceive=core.perceive_judged" not in source
    assert "core.perceive_audio_judged(" not in source
    assert "core.perceive_judged(" not in source
