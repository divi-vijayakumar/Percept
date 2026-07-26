"""Public conformance tests for the core runtime lease."""

from __future__ import annotations

import asyncio
import inspect

import pytest

import percept
import percept.runtime
from percept import Goal, Percept
from percept.adapters import Verdict
from percept.edge.tier0 import gray_frame
from percept.interest import InterestSubscription
from percept.runtime import run
from percept.runtime import worker as worker_module


STILL = gray_frame(8, 8, 0)
MOVED = gray_frame(8, 8, 255)


class _Vision:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def detect(self, _frame, _goals):
        return []

    async def judge(self, _condition, frame, **_kwargs):
        self.calls.append(str(frame))
        return Verdict(
            False,
            0.99,
            f"{frame} false",
            usage={"input_tokens": 1, "cost_usd": 0.001},
        )


def _core(vision=None) -> Percept:
    core = Percept.create(vision=vision or _Vision(), discover_plugins=False)
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


async def _drain(core: Percept, source, **kwargs) -> list:
    judge_dispatch = kwargs.pop("judge_dispatch", "async")
    return [
        incident
        async for incident in run(
            core,
            source=source,
            judge_dispatch=judge_dispatch,
            **kwargs,
        )
    ]


@pytest.mark.asyncio
async def test_overlap_rejection_precedes_every_rejected_invocation_side_effect():
    core = _core()
    active_source_started = asyncio.Event()
    rejected_source_pulls = 0
    rejected_health_calls = 0

    async def original_trace(*_args, **_kwargs):
        return None

    async def active_trace(*_args, **_kwargs):
        return None

    async def rejected_trace(*_args, **_kwargs):
        raise AssertionError("rejected trace must not be installed or called")

    async def held_source():
        active_source_started.set()
        await asyncio.Event().wait()
        yield  # pragma: no cover

    class PoisonSource:
        def __aiter__(self):
            nonlocal rejected_source_pulls
            rejected_source_pulls += 1
            raise AssertionError("rejected source must not be iterated")

    async def rejected_health(_event):
        nonlocal rejected_health_calls
        rejected_health_calls += 1
        raise AssertionError("rejected health callback must not run")

    core.on_trace = original_trace
    active = asyncio.create_task(
        _drain(core, held_source(), on_trace=active_trace)
    )
    await asyncio.wait_for(active_source_started.wait(), timeout=1.0)
    assert core.on_trace is active_trace

    subscription = InterestSubscription()
    rejected = run(
        core,
        source=PoisonSource(),
        on_trace=rejected_trace,
        on_health=rejected_health,
        interest_subscription=subscription,
    )
    with pytest.raises(
        worker_module.ConcurrentCoreRunError,
        match="already has an active runtime",
    ):
        await anext(rejected)

    assert rejected_source_pulls == 0
    assert rejected_health_calls == 0
    assert subscription._run_id is None
    assert subscription.closed is False
    assert core.on_trace is active_trace

    active.cancel()
    with pytest.raises(asyncio.CancelledError):
        await active
    assert core.on_trace is original_trace
    assert id(core) not in worker_module._ACTIVE_CORE_RUNS


def test_public_error_and_decorated_generator_preserve_runtime_surface():
    error = worker_module.ConcurrentCoreRunError
    assert percept.ConcurrentCoreRunError is error
    assert percept.runtime.ConcurrentCoreRunError is error
    assert inspect.isasyncgenfunction(run)
    signature = inspect.signature(run)
    assert tuple(signature.parameters)[:2] == ("core", "source")
    assert signature.parameters["source"].kind is inspect.Parameter.KEYWORD_ONLY


@pytest.mark.asyncio
async def test_bind_preflight_startup_callback_and_source_failures_release_lease():
    core = _core()

    bad_session = InterestSubscription()
    with pytest.raises(ValueError, match="interest_session_id"):
        await _drain(
            core,
            [],
            interest_subscription=bad_session,
            interest_session_id="",
        )
    assert bad_session.closed is True
    assert id(core) not in worker_module._ACTIVE_CORE_RUNS

    preflight_subscription = InterestSubscription()
    with pytest.raises(ValueError, match="judge_dispatch"):
        await _drain(
            core,
            [],
            judge_dispatch="invalid",
            interest_subscription=preflight_subscription,
        )
    assert preflight_subscription.closed is True
    assert id(core) not in worker_module._ACTIVE_CORE_RUNS

    async def fail_runtime_started(event):
        if event.kind == "runtime_started":
            raise RuntimeError("startup callback failed")

    with pytest.raises(RuntimeError, match="startup callback failed"):
        await _drain(core, [], on_health=fail_runtime_started)
    assert id(core) not in worker_module._ACTIVE_CORE_RUNS

    async def failing_source():
        raise RuntimeError("source failed")
        yield  # pragma: no cover

    with pytest.raises(RuntimeError, match="source failed"):
        await _drain(core, failing_source())
    assert id(core) not in worker_module._ACTIVE_CORE_RUNS

    assert await _drain(core, _source("after-failures")) == []
    assert id(core) not in worker_module._ACTIVE_CORE_RUNS


@pytest.mark.asyncio
async def test_fatal_split_dispatch_failure_releases_lease_and_preserves_reuse():
    vision = _Vision()
    core = _core(vision)
    original_dispatch = core.dispatch_judgment
    fatal_calls = 0

    async def fatal_dispatch(_job):
        nonlocal fatal_calls
        fatal_calls += 1
        raise RuntimeError("fatal dispatch failure")

    core.dispatch_judgment = fatal_dispatch
    with pytest.raises(RuntimeError, match="fatal dispatch failure"):
        await _drain(core, _source("fatal"))
    assert fatal_calls == 1
    assert id(core) not in worker_module._ACTIVE_CORE_RUNS

    core.dispatch_judgment = original_dispatch
    assert await _drain(core, _source("after-fatal")) == []
    assert vision.calls == ["after-fatal:receipt"]
    assert core._provider_completion_high_water == 1
    assert core.metrics.get("funnel.judged") == 1
    assert id(core) not in worker_module._ACTIVE_CORE_RUNS
