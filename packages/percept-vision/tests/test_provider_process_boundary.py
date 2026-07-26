"""External provider lifetime: no Python scheduling escape can outlive return."""
from __future__ import annotations

import asyncio
import contextvars
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from percept import Goal, Percept
from percept.adapters import Verdict
from percept.harness.tier0 import Frame, gray_frame
from percept.runtime import provider_process
from percept.runtime.provider_process import (
    ProviderWatchdogExpired,
    run_provider_process,
)
from percept.runtime.worker import _ACTIVE_CORE_RUNS
from percept.runtime.worker import run as worker_run


def _production_boundary_core(vision: Any) -> Percept:
    """Build a core without the repository test harness's shared-memory grant."""
    core = Percept.create(vision=vision, discover_plugins=False)
    core.__dict__.pop(
        "_reviewed_test_in_process_provider_capability",
        None,
    )
    return core


def _late_write(path: str, label: str, delay_s: float = 0.2) -> None:
    time.sleep(delay_s)
    with open(path, "a", encoding="utf-8") as stream:
        stream.write(label + "\n")


async def _late_async_write(path: str, label: str) -> None:
    await asyncio.sleep(0.2)
    with open(path, "a", encoding="utf-8") as stream:
        stream.write(label + "\n")


class _EscapeProvider:
    def __init__(self, effects: str) -> None:
        self.effects = effects

    async def __call__(self) -> tuple[str, bool]:
        loop = asyncio.get_running_loop()

        # A provider-created process inherits the containment process group.
        try:
            descendant = os.fork()
        except OSError:
            # The production sandbox denies process descendants outright.
            descendant = -1
        fork_denied = descendant == -1
        if descendant == 0:  # pragma: no cover - only on a weaker platform
            _late_write(self.effects, "forked-descendant")
            os._exit(0)

        # Bypass create_task and ambient Context directly.
        asyncio.Task(
            _late_async_write(self.effects, "direct-task"),
            loop=loop,
        )
        empty = contextvars.Context()
        if sys.version_info >= (3, 11):
            loop.create_task(
                _late_async_write(self.effects, "empty-context-task"),
                context=empty,
            )
        else:
            empty.run(
                loop.create_task,
                _late_async_write(self.effects, "empty-context-task"),
            )

        # Replace the loop factory and create another descendant through it.
        def replacement_factory(
            factory_loop: asyncio.AbstractEventLoop,
            coroutine: Any,
            **_kwargs: Any,
        ) -> asyncio.Task[Any]:
            return asyncio.Task(coroutine, loop=factory_loop)

        loop.set_task_factory(replacement_factory)
        loop.create_task(
            _late_async_write(self.effects, "replacement-factory")
        )

        # A scheduled callback may itself create non-async work.
        loop.call_soon(
            lambda: threading.Thread(
                target=_late_write,
                args=(self.effects, "call-soon-thread"),
                daemon=False,
            ).start()
        )
        loop.call_soon(
            lambda: threading.Thread(
                target=_late_write,
                args=(self.effects, "empty-context-call-soon"),
                daemon=False,
            ).start(),
            context=empty,
        )
        loop.run_in_executor(
            None,
            _late_write,
            self.effects,
            "executor-thread",
        )

        try:
            await asyncio.sleep(0)
            return ("frozen-result", fork_denied)
        finally:
            # Teardown remains contained even when the root creates new work
            # while leaving its own finally block.
            asyncio.Task(
                _late_async_write(self.effects, "finally-direct-task"),
                loop=loop,
            )
            empty.run(
                loop.create_task,
                _late_async_write(self.effects, "finally-empty-context"),
            )


class _ResistantProvider:
    def __init__(self, effects: str) -> None:
        self.effects = effects

    async def __call__(self) -> None:
        threading.Thread(
            target=_late_write,
            args=(self.effects, "resistant-thread"),
            daemon=False,
        ).start()
        while True:
            try:
                await asyncio.Event().wait()
            except BaseException:  # noqa: BLE001 - adversarial provider
                continue


class _SelfCancellingProvider:
    async def __call__(self) -> str:
        current = asyncio.current_task()
        assert current is not None
        current.cancel()
        try:
            await asyncio.sleep(0)
        except asyncio.CancelledError:
            pass
        return "still-owned-by-parent"


def test_terminal_result_kills_every_async_thread_and_process_escape(
    tmp_path: Path,
) -> None:
    effects = str(tmp_path / "late-effects.txt")

    async def drive() -> None:
        assert await run_provider_process(
            _EscapeProvider(effects),
            (),
            {},
            watchdog_s=1.0,
        ) == ("frozen-result", True)
        assert provider_process._ACTIVE_PROVIDER_PROCESSES == set()

    asyncio.run(drive())
    time.sleep(0.35)
    assert not Path(effects).exists()


def test_watchdog_kills_and_reaps_indefinitely_resistant_provider(
    tmp_path: Path,
) -> None:
    effects = str(tmp_path / "watchdog-late-effects.txt")

    async def drive() -> None:
        started = time.monotonic()
        with pytest.raises(ProviderWatchdogExpired):
            await run_provider_process(
                _ResistantProvider(effects),
                (),
                {},
                watchdog_s=0.05,
            )
        assert time.monotonic() - started < 0.75
        assert provider_process._ACTIVE_PROVIDER_PROCESSES == set()

    asyncio.run(drive())
    time.sleep(0.35)
    assert not Path(effects).exists()


def test_parent_cancellation_kills_and_reaps_before_propagation(
    tmp_path: Path,
) -> None:
    effects = str(tmp_path / "cancel-late-effects.txt")

    async def drive() -> None:
        invocation = asyncio.create_task(
            run_provider_process(
                _ResistantProvider(effects),
                (),
                {},
                watchdog_s=5.0,
            )
        )
        await asyncio.sleep(0.05)
        invocation.cancel()
        with pytest.raises(asyncio.CancelledError):
            await invocation
        assert provider_process._ACTIVE_PROVIDER_PROCESSES == set()

    asyncio.run(drive())
    time.sleep(0.35)
    assert not Path(effects).exists()


def test_cancellation_during_subprocess_launch_still_obtains_and_reaps_handle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_spawn = asyncio.create_subprocess_exec

    async def drive() -> None:
        entered = asyncio.Event()
        release = asyncio.Event()

        async def delayed_spawn(*args: Any, **kwargs: Any) -> Any:
            entered.set()
            await release.wait()
            return await original_spawn(*args, **kwargs)

        monkeypatch.setattr(
            asyncio,
            "create_subprocess_exec",
            delayed_spawn,
        )
        invocation = asyncio.create_task(
            run_provider_process(
                _SelfCancellingProvider(),
                (),
                {},
                watchdog_s=1.0,
            )
        )
        await entered.wait()
        invocation.cancel()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await invocation
        assert provider_process._ACTIVE_PROVIDER_PROCESSES == set()

    asyncio.run(drive())


def test_child_cannot_cancel_the_live_parent_owner_task() -> None:
    async def drive() -> None:
        owner = asyncio.current_task()
        assert owner is not None

        assert await run_provider_process(
            _SelfCancellingProvider(),
            (),
            {},
            watchdog_s=1.0,
        ) == "still-owned-by-parent"
        await asyncio.sleep(0)
        assert not owner.cancelled()

    asyncio.run(drive())


class _UnmarkedCustomVision:
    def __init__(self, effects: str) -> None:
        self.effects = effects
        self.parent_visible_calls = 0

    async def detect(self, _frame: Any, _goals: Any) -> list[Any]:
        return []

    async def judge(
        self,
        _condition: str,
        _frame: Any,
        **_kwargs: Any,
    ) -> Verdict:
        self.parent_visible_calls += 1
        threading.Thread(
            target=_late_write,
            args=(self.effects, "adapter-thread"),
            daemon=False,
        ).start()
        return Verdict(
            satisfied=True,
            confidence=0.99,
            evidence="isolated",
        )


def test_worker_releases_core_only_after_provider_process_is_dead(
    tmp_path: Path,
) -> None:
    effects = str(tmp_path / "worker-late-effects.txt")
    vision = _UnmarkedCustomVision(effects)
    assert not hasattr(vision, "RUNTIME_EXECUTION_BOUNDARY")
    core = _production_boundary_core(vision)
    core.add_goal(Goal(id="g", condition="motion", say="seen"))
    still = gray_frame(8, 8, 0)
    moving_pixels = [0] * 64
    for index in range(40):
        moving_pixels[index] = 255
    moving = Frame(w=8, h=8, pixels=tuple(moving_pixels))

    async def source():
        yield (0.0, still, "f0")
        await asyncio.sleep(0)
        yield (1.0, moving, "f1")
        await asyncio.sleep(0)

    async def drive() -> list[Any]:
        return [
            incident
            async for incident in worker_run(
                core,
                source=source(),
                judge_dispatch="async",
            )
        ]

    incidents = asyncio.run(drive())
    assert len(incidents) == 1
    assert incidents[0].verdict.evidence == "isolated"
    assert vision.parent_visible_calls == 0
    assert provider_process._ACTIVE_PROVIDER_PROCESSES == set()
    assert id(core) not in _ACTIVE_CORE_RUNS

    time.sleep(0.35)
    assert not Path(effects).exists()


class _BoundarySpoofingVision(_UnmarkedCustomVision):
    RUNTIME_EXECUTION_BOUNDARY = "in_process"


def test_custom_adapter_cannot_self_authorize_in_process_execution(
    tmp_path: Path,
) -> None:
    vision = _BoundarySpoofingVision(str(tmp_path / "must-not-run.txt"))
    core = _production_boundary_core(vision)

    async def drive() -> None:
        with pytest.raises(
            ValueError,
            match="reserved for authenticated SDK/test fixtures",
        ):
            await core._invoke_provider_call(
                vision.judge,
                ("condition", "frame"),
                {},
            )
        assert vision.parent_visible_calls == 0

    asyncio.run(drive())
