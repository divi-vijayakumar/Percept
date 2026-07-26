"""Independent oracle for the final global-runtime-owner lifecycle packet.

Run this file against an implementation worktree, never this oracle worktree's
package.  The queue behavior test deliberately skips when the required owner
queue does not exist; the adjacent architecture test remains a hard failure.
That distinction keeps "promotion seam absent" separate from a broken test
fixture or an incorrect implementation of an existing seam.
"""

from __future__ import annotations

import asyncio
import inspect
import time
from dataclasses import replace

import pytest

from percept import Goal, Percept
from percept.adapters import Verdict
from percept.backends.fake import FakeVision
from percept.edge.tier0 import gray_frame


_OWNER_QUEUE_NAMES = (
    "_OwnerCommandQueue",
    "_RuntimeOwnerQueue",
    "_OwnerReadyQueue",
)
_OWNER_QUEUE_CAPACITY = 4


def _ordinary_core(*, usage: bool = False) -> Percept:
    verdict = Verdict(
        False,
        0.99,
        "not satisfied",
        usage={"input_tokens": 1, "cost_usd": 0.001} if usage else None,
    )
    core = Percept.create(
        vision=FakeVision(script={"receipt": verdict}),
        discover_plugins=False,
    )
    core.add_goal(
        Goal(
            id="ordinary",
            condition="ordinary",
            say="ordinary fired",
            source="camera",
            state={"type": "watch"},
        )
    )
    return core


async def _ordinary_completion(core: Percept):
    revision = core.goals.revision
    generation = core.source_continuity_generation("camera")
    candidate = core.prepare_observation_candidate(
        "receipt",
        "camera",
        t=1.0,
        expected_goals_revision=revision,
        expected_source_generation=generation,
        goal_ids=("ordinary",),
    )
    assert candidate is not None
    assert core.launch_observation_candidate(candidate)
    observation = await core.dispatch_observation_candidate(candidate)
    reduction = core.reduce_observation_completion(observation)
    assert reduction.stale_reason is None
    job = core.prepare_frame_judgment(reduction, goal_id="ordinary")
    assert job is not None
    return await core.dispatch_judgment(job)


def _forged_completion(base, ordinal: int):
    token = replace(
        base.job.token,
        admission_id=f"global-owner-oracle:{ordinal}",
        evidence_seq=ordinal,
    )
    return replace(
        base,
        job=replace(base.job, token=token),
        completion_ordinal=ordinal,
    )


def _owner_queue_type(worker):
    found = [
        getattr(worker, name)
        for name in _OWNER_QUEUE_NAMES
        if hasattr(worker, name)
    ]
    return found[0] if len(found) == 1 else None


def _construct_owner_queue(queue_type):
    signature = inspect.signature(queue_type)
    for parameter in ("maxsize", "capacity", "limit"):
        if parameter in signature.parameters:
            return queue_type(**{parameter: _OWNER_QUEUE_CAPACITY})
    pytest.fail(
        "owner queue exists but has no explicit maxsize/capacity/limit constructor "
        "contract"
    )


def _publish_owner_item(queue, *, kind: str, payload: dict):
    publish = getattr(queue, "publish", None)
    assert callable(publish) and not inspect.iscoroutinefunction(publish), (
        "owner queue must synchronously assign an ordinal and publish one "
        "command/completion"
    )
    try:
        return publish(kind=kind, payload=payload)
    except TypeError:
        return publish(kind, payload)


def _drain_owner_items(queue):
    drain = getattr(queue, "drain", None)
    assert callable(drain) and not inspect.iscoroutinefunction(drain), (
        "the runtime owner needs a synchronous ordered drain"
    )
    rows = drain()
    return tuple(rows or ())


def _row_ordinal(row):
    if isinstance(row, dict):
        return row.get("ordinal")
    return getattr(row, "ordinal", None)


def _row_payload(row):
    if isinstance(row, dict):
        return row.get("payload")
    return getattr(row, "payload", None)


def test_provider_accounting_uses_completion_ordinal_not_bounded_identity_cache():
    """The behavioral test below proves the high-water and explicit-gap semantics."""
    source = inspect.getsource(Percept._record_completed_provider_calls)
    assert "completion_ordinal" in source, (
        "provider accounting ignores the global completion ordinal"
    )
    assert "_FUSED_ADMISSION_LEDGER_MAX" not in source, (
        "provider accounting still relies on the 1,024-entry admission-id cache"
    )


def test_provider_accounting_survives_1024_churn_gap_and_duplicate_replay():
    """Count each real receipt once even after cache horizon and out-of-order gap fill."""
    core = _ordinary_core(usage=True)
    base = asyncio.run(_ordinary_completion(core))

    # Publish 1..1024, then leap over 1025, advance the high-water, and finally
    # fill 1025.  A high-water+gap receipt ledger counts every ordinal once.
    completion_order = (
        tuple(range(1, 1025))
        + tuple(range(1026, 1030))
        + (1025,)
    )
    completions = {
        ordinal: _forged_completion(base, ordinal)
        for ordinal in completion_order
    }
    for ordinal in completion_order:
        assert core.reduce_judgment_completion(completions[ordinal]) == []

    assert core.metrics.get("funnel.judged") == 1029
    assert core.metrics.get("judge.tokens.input") == 1029
    assert core.metrics.get("judge.cost.usd") == pytest.approx(1.029)

    # Both a very old ordinal and the ordinal that originally filled the gap
    # are duplicate replays.  Neither may publish factual spend again.
    assert core.reduce_judgment_completion(completions[1]) == []
    assert core.reduce_judgment_completion(completions[1025]) == []
    assert core.metrics.get("funnel.judged") == 1029
    assert core.metrics.get("judge.tokens.input") == 1029
    assert core.metrics.get("judge.cost.usd") == pytest.approx(1.029)


def test_runtime_exposes_one_bounded_cross_route_owner_queue():
    import percept.runtime.worker as worker

    queue_types = [
        (name, getattr(worker, name))
        for name in _OWNER_QUEUE_NAMES
        if hasattr(worker, name)
    ]
    assert len(queue_types) == 1, (
        "missing architecture: exactly one bounded owner command/completion "
        f"queue is required; found {[name for name, _type in queue_types]}"
    )
    _name, queue_type = queue_types[0]
    source = inspect.getsource(queue_type)
    for word in ("max", "ordinal", "overflow"):
        assert word in source.lower(), (
            f"owner queue omits required {word!r} semantics"
        )

    scheduler = getattr(worker, "_ProviderScheduler", None)
    assert scheduler is not None, (
        "missing architecture: provider lanes still use the stateful "
        "_JudgeScheduler"
    )
    scheduler_source = inspect.getsource(scheduler)
    assert "ready" in scheduler_source or "callback" in scheduler_source
    assert "completion_ordinal" in scheduler_source
    for forbidden in (
        "perceive_judged",
        "advance_source_continuity",
        "reduce_judgment",
        "reduce_fused",
    ):
        assert forbidden not in scheduler_source

    worker_source = inspect.getsource(worker)
    for route in ("ordinary", "transition", "process", "audio", "fused"):
        assert route in worker_source, f"worker owner protocol omits {route!r}"


def test_reverse_blocked_routes_publish_one_global_ready_order():
    """A blocked ordinary lane cannot head-of-line block later-ready routes."""
    import percept.runtime.worker as worker

    queue_type = _owner_queue_type(worker)
    if queue_type is None:
        pytest.skip(
            "architecture-required: global owner queue is absent; the hard "
            "architecture test reports the promotion failure"
        )
    queue = _construct_owner_queue(queue_type)

    # Admission order is ordinary, transition, process, audio, fused. Ordinary
    # remains blocked while the other lanes complete in exact reverse order.
    ready = (
        ("fused", 4),
        ("audio", 3),
        ("process", 2),
        ("transition", 1),
    )
    for route, admission_rank in ready:
        _publish_owner_item(
            queue,
            kind="judgment_ready",
            payload={"route": route, "admission_rank": admission_rank},
        )
    first = _drain_owner_items(queue)
    assert len(first) == len(ready)
    first_ordinals = tuple(_row_ordinal(row) for row in first)
    assert all(type(ordinal) is int for ordinal in first_ordinals)
    assert first_ordinals == tuple(
        range(first_ordinals[0], first_ordinals[0] + len(first))
    )
    assert tuple(_row_payload(row)["route"] for row in first) == tuple(
        route for route, _rank in ready
    )

    _publish_owner_item(
        queue,
        kind="judgment_ready",
        payload={"route": "ordinary", "admission_rank": 0},
    )
    second = _drain_owner_items(queue)
    assert len(second) == 1
    assert _row_payload(second[0])["route"] == "ordinary"
    assert _row_ordinal(second[0]) == first_ordinals[-1] + 1


def test_audio_pump_publishes_commands_and_never_mutates_owner_state():
    import percept.runtime.worker as worker

    source = inspect.getsource(worker)
    start = source.index("async def _pump_audio")
    end = source.index("audio_task =", start)
    pump = source[start:end]
    forbidden = (
        "core.push_audio(",
        "synchronizer.accept(",
        "_advance_fused_generation(",
        "advance_source_continuity(",
        "_queue_fused_bundle(",
        "prepare_fused_judgment(",
    )
    hits = tuple(item for item in forbidden if item in pump)
    assert hits == (), (
        "audio pump remains a cognition owner through direct mutations: "
        f"{hits}"
    )
    for command in ("AudioAccepted", "AudioBoundary", "AudioEOS"):
        assert command in pump, (
            f"audio pump does not publish the canonical {command} owner command"
        )
    assert ".put" in pump or "publish(" in pump or "enqueue(" in pump


def test_provider_lane_close_does_not_advance_shared_source_generation():
    import percept.runtime.worker as worker

    scheduler_type = getattr(
        worker,
        "_ProviderScheduler",
        getattr(worker, "_JudgeScheduler", None),
    )
    assert scheduler_type is not None
    close_source = inspect.getsource(scheduler_type.close)
    assert "advance_source_continuity" not in close_source
    assert "advance_generation" not in close_source

    # Exercise the legacy constructor while it remains the live lane. The
    # transport-only replacement may intentionally have a different,
    # core-free constructor; its close source above is the relevant contract.
    if scheduler_type.__name__ != "_JudgeScheduler":
        return

    async def drive():
        core = _ordinary_core()
        before = core.source_continuity_generation("camera")
        lanes = [
            scheduler_type(core, "camera", worker.RuntimeStats())
            for _ in range(8)
        ]
        for lane in lanes:
            await lane.close(drain=False)
        return before, core.source_continuity_generation("camera")

    before, after = asyncio.run(drive())
    assert after == before, (
        "each provider lane independently advanced the shared source boundary; "
        "the runtime owner must advance it exactly once"
    )


def test_all_outputs_materialize_before_the_first_async_callback(monkeypatch):
    """Observe real worker ordering without depending on a proposed helper signature."""
    import percept.runtime.worker as worker

    still = gray_frame(8, 8, 0)
    moved = gray_frame(8, 8, 255)
    core = Percept.create(
        vision=FakeVision(
            script={
                "clear": Verdict(False, 0.99, "clear"),
                "hit": Verdict(True, 0.99, "hit"),
            }
        ),
        discover_plugins=False,
    )
    for goal_id in ("first", "second"):
        core.add_goal(
            Goal(
                id=goal_id,
                condition=goal_id,
                say=f"{goal_id} fired",
                source="camera",
            )
        )

    created: list[str] = []
    callback_views: list[tuple[str, ...]] = []
    real_make_incident = worker.make_incident

    def recording_make_incident(fire, *args, **kwargs):
        created.append(fire.goal_id)
        return real_make_incident(fire, *args, **kwargs)

    async def on_incident(_incident):
        callback_views.append(tuple(created))
        await asyncio.sleep(0)

    monkeypatch.setattr(worker, "make_incident", recording_make_incident)

    async def drive():
        source = [
            (0.0, still, "clear"),
            (1.0, moved, "hit"),
        ]
        return [
            incident
            async for incident in worker.run(
                core,
                source=source,
                on_incident=on_incident,
                judge_dispatch="inline",
            )
        ]

    incidents = asyncio.run(drive())
    assert len(incidents) == 2
    assert len(created) == 2
    assert len(callback_views) == 2
    assert len(callback_views[0]) == 2, (
        "the first callback ran before the second output was materialized"
    )


def test_eos_joins_every_provider_lane_under_one_concurrent_deadline():
    import percept.runtime.worker as worker

    source = inspect.getsource(worker)
    assert "_ProviderScheduler" in source
    assert (
        "asyncio.as_completed" in source
        or "asyncio.gather" in source
        or "TaskGroup" in source
    )
    assert "judge_eos_timeout" in source
    assert "await _judge_sched.flush()" not in source, (
        "ordinary-only EOS still bypasses the bounded concurrent lane join"
    )


def test_ordinary_only_eos_obeys_the_same_hard_deadline():
    import percept.runtime.worker as worker

    class SlowVision:
        def __init__(self):
            self.started = False

        async def detect(self, _frame, _goals):
            return []

        async def judge(self, _condition, _frame, **_kwargs):
            self.started = True
            await asyncio.sleep(0.30)
            return Verdict(False, 0.99, "late")

    vision = SlowVision()
    core = Percept.create(vision=vision, discover_plugins=False)
    core.add_goal(
        Goal(
            id="slow",
            condition="slow",
            say="slow",
            source="camera",
        )
    )
    health = []
    still = gray_frame(8, 8, 0)
    moved = gray_frame(8, 8, 255)

    async def drive():
        return [
            incident
            async for incident in worker.run(
                core,
                source=[
                    (0.0, still, "clear"),
                    (1.0, moved, "slow"),
                ],
                on_health=health.append,
                judge_dispatch="async",
                judge_stall_after_s=0.02,
            )
        ]

    started = time.perf_counter()
    incidents = asyncio.run(drive())
    elapsed = time.perf_counter() - started
    assert vision.started
    assert incidents == []
    assert elapsed < 0.15, (
        f"ordinary-only EOS waited {elapsed:.3f}s for provider completion "
        "instead of enforcing the shared hard deadline"
    )
    assert any(event.kind == "judge_eos_timeout" for event in health)
