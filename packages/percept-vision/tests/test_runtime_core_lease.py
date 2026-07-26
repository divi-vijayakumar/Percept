"""Core-run lease finalization across normal, failed-close, and cancelled exits."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

import pytest

import percept.runtime.worker as worker
from percept.interest import InterestSubscription


class _Core:
    def __init__(self) -> None:
        self.advanced: list[str] = []

    def advance_source_continuity(self, source: str) -> None:
        self.advanced.append(source)


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


async def _collect(stream: AsyncIterator[Any]) -> list[Any]:
    return [item async for item in stream]


def test_close_error_is_truthful_and_never_strands_the_core_lease(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    core = _Core()
    close_error = RuntimeError("inner close failed")
    broken = _Stream(close_error=close_error)
    releases: list[tuple[int, Any] | None] = []
    original_release = worker._release_core_run

    def observed_release(lease: tuple[int, Any] | None) -> None:
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
    releases: list[tuple[int, Any] | None] = []
    original_release = worker._release_core_run

    def observed_release(lease: tuple[int, Any] | None) -> None:
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
