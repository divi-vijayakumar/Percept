from __future__ import annotations

import asyncio
import time

import pytest

from percept import Verdict
from percept.backends.judge_pool import (
    JudgeCapability,
    JudgePool,
    JudgeRoute,
)
from percept.harness.tier0 import Frame, gray_frame
from percept.runtime.worker import RuntimeStats
from percept.runtime.worker import run as worker_run


class _Judge:
    def __init__(
        self,
        verdict: Verdict,
        *,
        delay_s: float = 0.0,
        failure: Exception | None = None,
    ) -> None:
        self.verdict = verdict
        self.delay_s = delay_s
        self.failure = failure
        self.calls = 0
        self.contexts: list[dict | None] = []

    async def judge(self, _condition, _frame, **kwargs):
        self.calls += 1
        self.contexts.append(kwargs.get("context"))
        await asyncio.sleep(self.delay_s)
        if self.failure is not None:
            raise self.failure
        return self.verdict


def _cap(
    *shapes: str,
    latency_slo_s: float = 0.05,
    max_inflight: int = 1,
):
    return JudgeCapability(
        evidence_shapes=frozenset(shapes),
        latency_slo_s=latency_slo_s,
        modes=frozenset({"reflex", "deliberate"}),
        max_inflight=max_inflight,
    )


@pytest.mark.asyncio
async def test_reflex_pool_hedges_and_first_confident_judge_wins():
    slow = _Judge(
        Verdict(False, 0.95, "slow no"),
        delay_s=0.2,
    )
    fast = _Judge(
        Verdict(True, 0.91, "fast yes"),
        delay_s=0.01,
    )
    pool = JudgePool(
        [
            JudgeRoute("slow", slow, _cap("single_image"), priority=0),
            JudgeRoute("fast", fast, _cap("single_image"), priority=1),
        ],
        deadline_s=0.1,
        hedge_after_s=0.005,
    )

    started = time.monotonic()
    verdict = await pool.judge("condition", b"frame")
    elapsed = time.monotonic() - started

    assert verdict.satisfied is True
    assert elapsed < 0.1
    assert fast.calls == 1
    assert any(
        receipt.route == "fast" and receipt.outcome == "decisive"
        for receipt in pool.receipts()
    )


@pytest.mark.asyncio
async def test_deadline_is_an_outcome_bound_not_a_health_only_threshold():
    slow = _Judge(
        Verdict(True, 0.99, "too late"),
        delay_s=0.2,
    )
    pool = JudgePool(
        [
            JudgeRoute(
                "slow",
                slow,
                _cap("single_image", latency_slo_s=0.01),
            )
        ],
        deadline_s=0.03,
    )

    started = time.monotonic()
    verdict = await pool.judge("condition", b"frame")
    elapsed = time.monotonic() - started

    assert verdict.satisfied is False
    assert verdict.confidence == 0.0
    assert "deadline expired" in verdict.evidence
    assert elapsed < 0.1
    assert [receipt.outcome for receipt in pool.receipts()][-2:] == [
        "deadline",
        "cancelled",
    ]


@pytest.mark.asyncio
async def test_parent_cancellation_cancels_friendly_route_and_releases_capacity():
    entered = asyncio.Event()

    class Friendly:
        async def judge(self, *_args, **_kwargs):
            entered.set()
            await asyncio.Event().wait()

    pool = JudgePool(
        [JudgeRoute("friendly", Friendly(), _cap("single_image"))],
        deadline_s=1.0,
    )
    task = asyncio.create_task(pool.judge("condition", b"frame"))
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await asyncio.sleep(0)

    assert pool.snapshot()["routes"]["friendly"]["inflight"] == 0
    assert pool.receipts()[-1].outcome == "cancelled"


@pytest.mark.asyncio
async def test_parent_cancellation_keeps_hostile_route_physically_occupied():
    entered = asyncio.Event()
    release = asyncio.Event()

    class Hostile:
        async def judge(self, *_args, **_kwargs):
            entered.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                await release.wait()
            return Verdict(False, 0.9, "released")

    pool = JudgePool(
        [JudgeRoute("hostile", Hostile(), _cap("single_image"))],
        deadline_s=1.0,
    )
    task = asyncio.create_task(pool.judge("condition", b"frame"))
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await asyncio.sleep(0)

    assert pool.snapshot()["routes"]["hostile"]["inflight"] == 1
    unavailable = await pool.judge("condition", b"next")
    assert unavailable.confidence == 0.0
    assert pool.receipts()[-1].outcome == "capacity"

    release.set()
    for _ in range(10):
        await asyncio.sleep(0)
        if pool.snapshot()["routes"]["hostile"]["inflight"] == 0:
            break
    assert pool.snapshot()["routes"]["hostile"]["inflight"] == 0


@pytest.mark.asyncio
async def test_simultaneous_hedges_record_every_physical_completion():
    entered = 0
    both_entered = asyncio.Event()
    release = asyncio.Event()

    class BarrierJudge:
        def __init__(self, evidence):
            self.evidence = evidence

        async def judge(self, *_args, **_kwargs):
            nonlocal entered
            entered += 1
            if entered == 2:
                both_entered.set()
            await release.wait()
            return Verdict(True, 0.99, self.evidence)

    pool = JudgePool(
        [
            JudgeRoute("first", BarrierJudge("first"), _cap("single_image")),
            JudgeRoute("second", BarrierJudge("second"), _cap("single_image")),
        ],
        deadline_s=0.2,
        max_parallel=2,
        hedge_after_s=0.0,
    )
    task = asyncio.create_task(pool.judge("condition", b"frame"))
    await both_entered.wait()
    release.set()
    verdict = await task

    assert verdict.evidence == "first"
    assert [
        (receipt.route, receipt.outcome)
        for receipt in pool.receipts()
    ] == [("first", "decisive"), ("second", "decisive")]


@pytest.mark.asyncio
async def test_simultaneous_terminal_failure_wins_over_decisive_hedge():
    entered = 0
    both_entered = asyncio.Event()
    release = asyncio.Event()

    class BarrierJudge:
        def __init__(self, *, fail=False):
            self.fail = fail

        async def judge(self, *_args, **_kwargs):
            nonlocal entered
            entered += 1
            if entered == 2:
                both_entered.set()
            await release.wait()
            if self.fail:
                raise RuntimeError("quota")
            return Verdict(True, 0.99, "grounded")

    pool = JudgePool(
        [
            JudgeRoute("winner", BarrierJudge(), _cap("single_image")),
            JudgeRoute(
                "terminal",
                BarrierJudge(fail=True),
                _cap("single_image"),
            ),
        ],
        deadline_s=0.2,
        max_parallel=2,
        hedge_after_s=0.0,
        terminal_outcomes={"provider_error"},
    )
    task = asyncio.create_task(pool.judge("condition", b"frame"))
    await both_entered.wait()
    release.set()
    with pytest.raises(RuntimeError, match="quota"):
        await task

    assert {
        (receipt.route, receipt.outcome)
        for receipt in pool.receipts()
    } == {
        ("winner", "decisive"),
        ("terminal", "provider_error"),
    }
    terminal = pool.take_terminal_error()
    assert isinstance(terminal, RuntimeError)
    assert str(terminal) == "quota"


@pytest.mark.asyncio
async def test_reflex_rejects_route_whose_declared_slo_exceeds_deadline():
    judge = _Judge(Verdict(True, 0.99, "would be yes"))
    pool = JudgePool(
        [
            JudgeRoute(
                "deliberate-only-speed",
                judge,
                _cap("single_image", latency_slo_s=4.0),
            )
        ],
        mode="reflex",
        deadline_s=1.8,
    )

    verdict = await pool.judge("condition", b"frame")

    assert verdict.satisfied is False
    assert judge.calls == 0
    assert pool.receipts()[-1].outcome == "no_route"


@pytest.mark.asyncio
async def test_native_clip_request_never_silently_flattens_to_ordered_images():
    strip = _Judge(Verdict(True, 0.99, "strip"))
    clip = _Judge(Verdict(True, 0.95, "native clip"))
    pool = JudgePool(
        [
            JudgeRoute("strip", strip, _cap("ordered_images")),
            JudgeRoute("clip", clip, _cap("video_clip")),
        ],
        deadline_s=0.1,
    )

    verdict = await pool.judge(
        "condition",
        b"clip-payload",
        context={"_percept_evidence_shape": "video_clip"},
    )

    assert verdict.evidence == "native clip"
    assert strip.calls == 0
    assert clip.calls == 1


@pytest.mark.asyncio
async def test_context_stream_request_does_not_fall_back_to_stateless_strip():
    strip = _Judge(Verdict(True, 0.99, "stateless"))
    observer = _Judge(Verdict(True, 0.91, "context held"))
    pool = JudgePool(
        [
            JudgeRoute("strip", strip, _cap("ordered_images")),
            JudgeRoute("observer", observer, _cap("context_stream")),
        ],
        deadline_s=0.1,
    )

    verdict = await pool.judge(
        "condition",
        b"latest",
        frames=[b"earlier"],
        context={"_percept_evidence_shape": "context_stream"},
    )

    assert verdict.evidence == "context held"
    assert observer.calls == 1
    assert strip.calls == 0


@pytest.mark.asyncio
async def test_uncertain_primary_does_not_preempt_confident_hedge():
    uncertain = _Judge(
        Verdict(False, 0.2, "unknown"),
        delay_s=0.001,
    )
    decisive = _Judge(
        Verdict(True, 0.9, "grounded"),
        delay_s=0.01,
    )
    pool = JudgePool(
        [
            JudgeRoute("uncertain", uncertain, _cap("single_image")),
            JudgeRoute("decisive", decisive, _cap("single_image")),
        ],
        deadline_s=0.1,
        hedge_after_s=0.0,
    )

    verdict = await pool.judge("condition", b"frame")

    assert verdict.satisfied is True
    assert verdict.evidence == "grounded"


@pytest.mark.asyncio
async def test_route_capacity_is_bounded_while_a_prior_call_is_running():
    entered = asyncio.Event()

    class HeldJudge:
        def __init__(self) -> None:
            self.calls = 0

        async def judge(self, *_args, **_kwargs):
            self.calls += 1
            entered.set()
            await asyncio.sleep(0.05)
            return Verdict(True, 0.99, "late")

    held = HeldJudge()
    pool = JudgePool(
        [
            JudgeRoute(
                "held",
                held,
                _cap(
                    "single_image",
                    latency_slo_s=0.1,
                    max_inflight=1,
                ),
            )
        ],
        deadline_s=0.2,
    )

    first_task = asyncio.create_task(
        pool.judge("condition", b"frame-1")
    )
    await entered.wait()
    second = await pool.judge("condition", b"frame-2")
    first = await first_task

    assert first.satisfied is True
    assert second.confidence == 0.0
    assert held.calls == 1
    assert any(
        receipt.route == "held" and receipt.outcome == "capacity"
        for receipt in pool.receipts()
    )
    assert pool.snapshot()["routes"]["held"]["inflight"] == 0


@pytest.mark.asyncio
async def test_occupied_primary_falls_through_to_lower_priority_route():
    entered = asyncio.Event()
    release = asyncio.Event()

    class HeldPrimary:
        async def judge(self, *_args, **_kwargs):
            entered.set()
            await release.wait()
            return Verdict(True, 0.99, "primary")

    fallback = _Judge(Verdict(True, 0.95, "fallback"))
    pool = JudgePool(
        [
            JudgeRoute(
                "primary",
                HeldPrimary(),
                _cap("single_image", max_inflight=1),
                priority=0,
            ),
            JudgeRoute(
                "fallback",
                fallback,
                _cap("single_image", max_inflight=1),
                priority=1,
            ),
        ],
        deadline_s=0.2,
        hedge_after_s=0.0,
        max_parallel=1,
    )

    first = asyncio.create_task(pool.judge("condition", b"first"))
    await entered.wait()
    second = await pool.judge("condition", b"second")

    assert second.evidence == "fallback"
    assert fallback.calls == 1
    assert any(
        receipt.route == "primary" and receipt.outcome == "capacity"
        for receipt in pool.receipts()
    )

    release.set()
    assert (await first).evidence == "primary"


def test_capability_and_pool_configuration_fail_loudly():
    judge = _Judge(Verdict(False, 0.0, "unknown"))
    with pytest.raises(ValueError):
        JudgeCapability(frozenset(), latency_slo_s=1.0)
    with pytest.raises(ValueError):
        JudgeCapability(
            frozenset({"sprite"}),  # type: ignore[arg-type]
            latency_slo_s=1.0,
        )
    with pytest.raises(ValueError):
        JudgePool([])
    with pytest.raises(ValueError):
        JudgePool(
            [
                JudgeRoute("same", judge, _cap("single_image")),
                JudgeRoute("same", judge, _cap("single_image")),
            ]
        )


@pytest.mark.asyncio
async def test_pool_is_on_the_real_async_runtime_watch_path():
    width = height = 8
    still = gray_frame(width, height, 0)
    pixels = [0] * (width * height)
    for index in range(40):
        pixels[index] = 255
    motion = Frame(width, height, tuple(pixels))

    slow = _Judge(Verdict(False, 0.95, "slow"), delay_s=0.2)
    fast = _Judge(Verdict(True, 0.95, "fast"), delay_s=0.005)
    pool = JudgePool(
        [
            JudgeRoute("slow", slow, _cap("single_image")),
            JudgeRoute("fast", fast, _cap("single_image")),
        ],
        deadline_s=0.05,
        hedge_after_s=0.001,
    )
    from percept import Goal, Percept

    core = Percept.create(vision=pool, discover_plugins=False)
    core.add_goal(
        Goal(
            id="arrival",
            condition="someone arrived",
            say="arrival",
        )
    )
    stats = RuntimeStats()

    async def source():
        yield 0.0, still, b"still"
        await asyncio.sleep(0)
        yield 1.0, motion, b"motion"
        await asyncio.sleep(0)

    incidents = [
        incident
        async for incident in worker_run(
            core,
            source=source(),
            stats=stats,
            judge_dispatch="async",
        )
    ]

    assert len(incidents) == 1
    assert incidents[0].concern_id == "arrival"
    assert stats.judge_calls == 1
    assert any(
        receipt.route == "fast" and receipt.outcome == "decisive"
        for receipt in pool.receipts()
    )
