"""Core-run lease finalization across normal, failed-close, and cancelled exits."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

import pytest

import percept.runtime.worker as worker
from percept.interest import InterestRunError, InterestSubscription


class _Core:
    def __init__(self) -> None:
        self.advanced: list[str] = []

    def advance_source_continuity(self, source: str) -> None:
        self.advanced.append(source)


class _ForgedCancelled(asyncio.CancelledError):
    pass


class _Stream:
    def __init__(
        self,
        items: tuple[Any, ...] = (),
        *,
        close_error: BaseException | None = None,
        block: bool = False,
    ) -> None:
        self._items = list(items)
        self._close_error = close_error
        self._block = block
        self.started = asyncio.Event()
        self.close_calls = 0

    def __aiter__(self) -> "_Stream":
        return self

    async def __anext__(self) -> Any:
        self.started.set()
        if self._items:
            return self._items.pop(0)
        if self._block:
            await asyncio.Event().wait()
        raise StopAsyncIteration

    async def aclose(self) -> None:
        self.close_calls += 1
        if self._close_error is not None:
            raise self._close_error


@worker._exclusive_core_runtime
def _runtime(
    core: Any,
    *,
    stream: _Stream,
    gate_source: str = "camera",
) -> AsyncIterator[Any]:
    del core, gate_source
    return stream


@worker._exclusive_core_runtime
def _construction_failure(
    core: Any,
    *,
    error: BaseException,
) -> AsyncIterator[Any]:
    del core
    raise error


@worker._exclusive_core_runtime
def _iteration_failure(
    core: Any,
    *,
    error: BaseException,
) -> AsyncIterator[Any]:
    del core

    async def fail() -> AsyncIterator[Any]:
        raise error
        yield  # pragma: no cover - makes this an async generator

    return fail()


async def _collect(stream: AsyncIterator[Any]) -> list[Any]:
    return [item async for item in stream]


def test_close_error_is_truthful_and_never_strands_the_core_lease(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    core = _Core()
    close_error = RuntimeError("inner close failed")
    broken = _Stream(close_error=close_error)
    releases: list[Any] = []
    original_release = worker._release_core_run

    def observed_release(lease: Any) -> None:
        releases.append(lease)
        original_release(lease)

    monkeypatch.setattr(worker, "_release_core_run", observed_release)

    async def drive() -> None:
        with pytest.raises(RuntimeError, match="inner close failed") as caught:
            await _collect(_runtime(core, stream=broken))
        assert caught.value is close_error
        assert broken.close_calls == 1
        assert len(releases) == 1

        healthy = _Stream(("next-run",))
        assert await _collect(_runtime(core, stream=healthy)) == ["next-run"]
        assert healthy.close_calls == 1
        assert len(releases) == 2

    asyncio.run(drive())


def test_synchronous_stream_construction_failure_releases_exactly_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    core = _Core()
    construction_error = RuntimeError("stream construction failed")
    releases: list[Any] = []
    original_release = worker._release_core_run

    def observed_release(lease: Any) -> None:
        releases.append(lease)
        original_release(lease)

    monkeypatch.setattr(worker, "_release_core_run", observed_release)

    async def drive() -> None:
        with pytest.raises(RuntimeError, match="stream construction failed") as caught:
            await _collect(_construction_failure(core, error=construction_error))
        assert caught.value is construction_error
        assert len(releases) == 1

        healthy = _Stream(("next-run",))
        assert await _collect(_runtime(core, stream=healthy)) == ["next-run"]
        assert len(releases) == 2

    asyncio.run(drive())


def test_stale_release_cannot_delete_a_newer_same_core_lease() -> None:
    core = _Core()
    first = worker._acquire_core_run(core)
    worker._release_core_run(first)
    second = worker._acquire_core_run(core)
    try:
        worker._release_core_run(first)
        with pytest.raises(worker.ConcurrentCoreRunError):
            worker._acquire_core_run(core)
    finally:
        worker._release_core_run(second)

    third = worker._acquire_core_run(core)
    worker._release_core_run(third)


def test_generator_exit_hostile_teardown_holds_core_lease_until_quiescent() -> None:
    core = _Core()

    async def drive() -> None:
        release = asyncio.Event()
        generator_exit_seen = asyncio.Event()
        exits: list[type[BaseException]] = []

        async def hostile_provider() -> None:
            while not release.is_set():
                try:
                    await release.wait()
                except BaseException as exc:  # noqa: BLE001 - adversarial provider
                    exits.append(type(exc))
                    if isinstance(exc, GeneratorExit):
                        generator_exit_seen.set()

        provider_task = asyncio.create_task(hostile_provider())

        async def own_until_quiescent() -> None:
            lease = worker._acquire_core_run(core)
            try:
                await worker._cancel_and_join_task(provider_task)
            finally:
                worker._release_core_run(lease)

        owner_task = asyncio.create_task(own_until_quiescent())
        await asyncio.wait_for(generator_exit_seen.wait(), timeout=0.5)
        assert not owner_task.done()
        with pytest.raises(worker.ConcurrentCoreRunError):
            worker._acquire_core_run(core)

        release.set()
        await asyncio.wait_for(owner_task, timeout=0.5)
        assert provider_task.done()
        assert provider_task not in worker._FORCE_CLOSED_PROVIDER_TASKS
        assert asyncio.CancelledError in exits
        assert GeneratorExit in exits

    asyncio.run(drive())

    reusable = worker._acquire_core_run(core)
    worker._release_core_run(reusable)


def test_normal_finalization_does_not_advance_source_continuity() -> None:
    core = _Core()
    stream = _Stream(("done",))

    assert asyncio.run(_collect(_runtime(core, stream=stream))) == ["done"]
    assert stream.close_calls == 1
    assert core.advanced == []


def test_cancelled_finalization_advances_once_and_releases_for_reuse() -> None:
    core = _Core()

    async def drive() -> None:
        blocked = _Stream(block=True)
        task = asyncio.create_task(
            _collect(_runtime(core, stream=blocked, gate_source="microphone"))
        )
        await blocked.started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        assert blocked.close_calls == 1
        assert core.advanced == ["microphone"]

        healthy = _Stream(("next-run",))
        assert await _collect(_runtime(core, stream=healthy)) == ["next-run"]
        assert healthy.close_calls == 1
        assert core.advanced == ["microphone"]

    asyncio.run(drive())


def test_asyncio_run_shutdown_finalizer_releases_without_context_leak() -> None:
    core = _Core()
    abandoned = _Stream(("started",), block=True)

    async def consume_one_then_abandon() -> None:
        stream = _runtime(core, stream=abandoned)
        assert await anext(stream) == "started"

    asyncio.run(consume_one_then_abandon())

    assert abandoned.close_calls == 1
    assert core.advanced == ["camera"]
    healthy = _Stream(("next-run",))
    assert asyncio.run(_collect(_runtime(core, stream=healthy))) == [
        "next-run"
    ]


def test_production_interest_run_owns_one_cancellation_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    core = _Core()

    async def drive() -> None:
        async def completed_worker(**_kwargs: Any) -> AsyncIterator[Any]:
            if False:
                yield None

        monkeypatch.setattr(worker, "_run_worker", completed_worker)
        assert await _collect(
            worker.run(
                core,
                source=(),
                interest_subscription=InterestSubscription(),
                gate_source="production-source",
            )
        ) == []
        assert core.advanced == []

        started = asyncio.Event()

        async def blocked_worker(**_kwargs: Any) -> AsyncIterator[Any]:
            started.set()
            await asyncio.Event().wait()
            if False:
                yield None

        monkeypatch.setattr(worker, "_run_worker", blocked_worker)
        task = asyncio.create_task(
            _collect(
                worker.run(
                    core,
                    source=(),
                    interest_subscription=InterestSubscription(),
                    gate_source="production-source",
                )
            )
        )
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert core.advanced == ["production-source"]

    asyncio.run(drive())


def test_production_interest_cancel_with_delivery_failure_advances_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    core = _Core()

    async def drive() -> None:
        started = asyncio.Event()

        async def blocked_worker(**_kwargs: Any) -> AsyncIterator[Any]:
            started.set()
            await asyncio.Event().wait()
            if False:
                yield None

        delivery_error = RuntimeError("interest close failed")

        def fail_finish(
            _subscription: InterestSubscription,
            _events: Any,
        ) -> None:
            raise delivery_error

        monkeypatch.setattr(worker, "_run_worker", blocked_worker)
        monkeypatch.setattr(InterestSubscription, "_finish", fail_finish)
        task = asyncio.create_task(
            _collect(
                worker.run(
                    core,
                    source=(),
                    interest_subscription=InterestSubscription(),
                    gate_source="nested-cancel",
                )
            )
        )
        await started.wait()
        task.cancel()
        with pytest.raises(InterestRunError) as caught:
            await task
        assert isinstance(caught.value.worker_error, asyncio.CancelledError)
        assert caught.value.delivery_error is delivery_error
        assert core.advanced == ["nested-cancel"]

    asyncio.run(drive())


def test_public_interest_errors_cannot_forge_cancellation_provenance() -> None:
    core = _Core()
    forged = InterestRunError(
        asyncio.CancelledError(),
        RuntimeError("forged delivery failure"),
    )
    cyclic = InterestRunError(
        RuntimeError("cyclic worker failure"),
        RuntimeError("cyclic delivery failure"),
    )
    cyclic.worker_error = cyclic
    malformed = InterestRunError(
        RuntimeError("malformed worker failure"),
        RuntimeError("malformed delivery failure"),
    )
    malformed.worker_error = object()  # type: ignore[assignment]

    async def drive() -> None:
        for error in (forged, cyclic, malformed):
            with pytest.raises(InterestRunError) as caught:
                await _collect(_iteration_failure(core, error=error))
            assert caught.value is error
            assert core.advanced == []

    asyncio.run(drive())


def test_direct_forged_cancelled_subclass_has_zero_authority() -> None:
    core = _Core()
    forged = _ForgedCancelled("publicly raised, not Task.cancel")

    async def drive() -> None:
        with pytest.raises(_ForgedCancelled) as caught:
            await _collect(_iteration_failure(core, error=forged))
        assert caught.value is forged

    asyncio.run(drive())
    assert core.advanced == []


def test_production_forged_cancelled_subclass_has_zero_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    core = _Core()
    forged = _ForgedCancelled("worker raised this directly")

    async def forged_worker(**_kwargs: Any) -> AsyncIterator[Any]:
        raise forged
        yield  # pragma: no cover

    monkeypatch.setattr(worker, "_run_worker", forged_worker)

    async def drive() -> None:
        with pytest.raises(_ForgedCancelled) as caught:
            await _collect(
                worker.run(
                    core,
                    source=(),
                    interest_subscription=InterestSubscription(),
                    gate_source="forged-production",
                )
            )
        assert caught.value is forged

    asyncio.run(drive())
    assert core.advanced == []


def test_explicit_production_run_close_is_one_cancelled_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    core = _Core()
    terminal_reasons: list[str] = []
    original_close_all = worker.LiveInterestTracker.close_all

    def observed_close_all(
        tracker: Any,
        reason: str,
    ) -> Any:
        terminal_reasons.append(reason)
        return original_close_all(tracker, reason)

    async def yielded_worker(**_kwargs: Any) -> AsyncIterator[Any]:
        yield "incident"
        await asyncio.Event().wait()

    monkeypatch.setattr(worker, "_run_worker", yielded_worker)
    monkeypatch.setattr(
        worker.LiveInterestTracker,
        "close_all",
        observed_close_all,
    )

    async def drive() -> None:
        run_stream = worker.run(
            core,
            source=(),
            interest_subscription=InterestSubscription(),
            gate_source="consumer-close",
        )
        assert await anext(run_stream) == "incident"
        await run_stream.aclose()

    asyncio.run(drive())
    assert terminal_reasons == ["cancelled"]
    assert core.advanced == ["consumer-close"]
