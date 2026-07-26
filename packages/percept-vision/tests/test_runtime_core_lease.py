"""Fail-closed lifecycle probes for the public same-core runtime lease."""

from __future__ import annotations

import asyncio

import pytest

from percept import ConcurrentCoreRunError, Goal, Percept
from percept.adapters import Verdict
from percept.edge.tier0 import gray_frame
from percept.runtime import InterestSubscription, run
from percept.runtime import worker as worker_module


STILL = gray_frame(8, 8, 0)
MOVED = gray_frame(8, 8, 255)


class RecordingVision:
    def __init__(self, *, satisfied: bool = False) -> None:
        self.calls: list[str] = []
        self.satisfied = satisfied

    async def detect(self, _frame, _goals):
        return []

    async def judge(self, _condition, frame, **_kwargs):
        self.calls.append(str(frame))
        return Verdict(
            self.satisfied,
            0.99,
            f"{frame} verdict",
            usage={"input_tokens": 1, "cost_usd": 0.001},
        )


class BlockingVision(RecordingVision):
    def __init__(self) -> None:
        super().__init__()
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def judge(self, condition, frame, **kwargs):
        self.calls.append(str(frame))
        self.started.set()
        await self.release.wait()
        return Verdict(
            False,
            0.99,
            f"{frame} verdict",
            usage={"input_tokens": 1, "cost_usd": 0.001},
        )


class FailOnceVision(RecordingVision):
    async def judge(self, condition, frame, **kwargs):
        self.calls.append(str(frame))
        if len(self.calls) == 1:
            raise RuntimeError("synthetic provider failure")
        return Verdict(
            False,
            0.99,
            f"{frame} verdict",
            usage={"input_tokens": 1, "cost_usd": 0.001},
        )


def _core(vision) -> Percept:
    core = Percept.create(vision=vision, discover_plugins=False)
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


def _source(token: str):
    return [
        (0.0, STILL, f"{token}:prime"),
        (1.0, MOVED, f"{token}:receipt"),
    ]


async def _drain(core, source, **kwargs):
    return [
        incident
        async for incident in run(
            core,
            source=source,
            judge_dispatch="async",
            **kwargs,
        )
    ]


@pytest.mark.asyncio
async def test_same_core_rejects_before_source_subscription_or_provider_work_across_threads():
    vision = RecordingVision()
    core = _core(vision)
    source_blocked = asyncio.Event()
    source_release = asyncio.Event()
    rejected_source_pulls = 0

    async def held_source():
        yield 0.0, STILL, "held:prime"
        source_blocked.set()
        await source_release.wait()

    class PoisonSource:
        def __aiter__(self):
            nonlocal rejected_source_pulls
            rejected_source_pulls += 1
            return self

        async def __anext__(self):
            nonlocal rejected_source_pulls
            rejected_source_pulls += 1
            raise AssertionError("rejected source must not be pulled")

    active = asyncio.create_task(_drain(core, held_source()))
    await asyncio.wait_for(source_blocked.wait(), timeout=1.0)
    calls_before = list(vision.calls)
    rejected_subscription = InterestSubscription()

    def overlapping_thread():
        return asyncio.run(
            _drain(
                core,
                PoisonSource(),
                interest_subscription=rejected_subscription,
            )
        )

    with pytest.raises(ConcurrentCoreRunError, match="already has an active runtime"):
        await asyncio.to_thread(overlapping_thread)

    assert rejected_source_pulls == 0
    assert vision.calls == calls_before
    assert rejected_subscription.closed is False

    active.cancel()
    with pytest.raises(asyncio.CancelledError):
        await active
    assert id(core) not in worker_module._ACTIVE_CORE_RUNS

    # Cancellation released the lease, so normal sequential reuse remains supported.
    assert await _drain(core, _source("after-cancel")) == []
    assert vision.calls == ["after-cancel:receipt"]


@pytest.mark.asyncio
async def test_different_cores_run_concurrently_without_serialization():
    left_vision = BlockingVision()
    right_vision = BlockingVision()
    left = asyncio.create_task(_drain(_core(left_vision), _source("left")))
    right = asyncio.create_task(_drain(_core(right_vision), _source("right")))

    await asyncio.wait_for(
        asyncio.gather(left_vision.started.wait(), right_vision.started.wait()),
        timeout=1.0,
    )
    assert left_vision.calls == ["left:receipt"]
    assert right_vision.calls == ["right:receipt"]
    left_vision.release.set()
    right_vision.release.set()
    assert await asyncio.gather(left, right) == [[], []]


@pytest.mark.asyncio
async def test_preflight_source_and_callback_failures_release_the_lease():
    vision = RecordingVision()
    core = _core(vision)

    with pytest.raises(TypeError, match="exact InterestSubscription"):
        await _drain(core, _source("preflight"), interest_subscription=object())

    async def failing_source():
        raise RuntimeError("source failed before first unit")
        yield  # pragma: no cover

    with pytest.raises(RuntimeError, match="source failed before first unit"):
        await _drain(core, failing_source())

    def failing_frame(_t, _frame):
        raise RuntimeError("frame callback failed")

    with pytest.raises(RuntimeError, match="frame callback failed"):
        await _drain(core, _source("callback"), on_frame=failing_frame)

    assert id(core) not in worker_module._ACTIVE_CORE_RUNS
    assert await _drain(core, _source("after-errors")) == []
    assert vision.calls == ["after-errors:receipt"]


@pytest.mark.asyncio
async def test_provider_failure_releases_lease_for_the_next_run():
    vision = FailOnceVision()
    core = _core(vision)

    # Provider failure is normalized into a failed completion by the cognition
    # owner, but the finite runtime still must relinquish ownership at EOS.
    assert await _drain(core, _source("provider-failure")) == []
    assert id(core) not in worker_module._ACTIVE_CORE_RUNS
    assert await _drain(core, _source("after-provider-failure")) == []
    assert vision.calls == [
        "provider-failure:receipt",
        "after-provider-failure:receipt",
    ]


@pytest.mark.asyncio
async def test_adversarial_trace_teardown_restores_trace_and_releases_lease():
    vision = RecordingVision()
    core = _core(vision)

    async def original_trace(*_args, **_kwargs):
        return None

    async def run_trace(*_args, **_kwargs):
        return None

    async def fail_runtime_end(event):
        if event.kind == "runtime_ended":
            raise RuntimeError("terminal health sink failed")

    core.on_trace = original_trace
    with pytest.raises(RuntimeError, match="terminal health sink failed"):
        await _drain(
            core,
            _source("teardown"),
            on_trace=run_trace,
            on_health=fail_runtime_end,
        )

    assert core.on_trace is original_trace
    assert id(core) not in worker_module._ACTIVE_CORE_RUNS
    assert await _drain(core, _source("after-teardown")) == []


@pytest.mark.asyncio
async def test_generator_exit_releases_lease_for_sequential_reuse():
    vision = RecordingVision(satisfied=True)
    core = _core(vision)
    stream = run(
        core,
        source=_source("generator-exit"),
        judge_dispatch="inline",
    )
    incident = await anext(stream)
    assert incident is not None
    await stream.aclose()

    assert id(core) not in worker_module._ACTIVE_CORE_RUNS
    calls_before = len(vision.calls)
    await _drain(core, _source("after-generator-exit"))
    assert len(vision.calls) == calls_before + 1


def test_sequential_reuse_preserves_provider_receipt_high_water():
    vision = RecordingVision()
    core = _core(vision)

    async def exercise():
        assert await _drain(core, _source("first")) == []
        first_high_water = core._provider_completion_high_water
        assert await _drain(core, _source("second")) == []
        return first_high_water, core._provider_completion_high_water

    first, second = asyncio.run(exercise())
    assert (first, second) == (1, 2)
    assert core.metrics.get("funnel.judged") == 2
    assert core.metrics.get("judge.tokens.input") == 2
    assert core.metrics.get("judge.cost.usd") == 0.002
    assert worker_module._ACTIVE_CORE_RUNS == {}
