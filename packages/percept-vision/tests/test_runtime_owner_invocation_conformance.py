"""Independent test-only conformance probes for runtime-owner invocation boundaries."""

from __future__ import annotations

import asyncio

from percept import Goal, Percept
from percept.adapters import Verdict
from percept.edge.tier0 import gray_frame
from percept.runtime import run


STILL = gray_frame(8, 8, 0)
MOVED = gray_frame(8, 8, 255)


def _source(token: str):
    return [
        (0.0, STILL, f"{token}:prime"),
        (1.0, MOVED, f"{token}:receipt"),
    ]


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


async def _drain(core: Percept, token: str, *, on_trace=None) -> list:
    return [
        incident
        async for incident in run(
            core,
            source=_source(token),
            judge_dispatch="async",
            on_trace=on_trace,
        )
    ]


class _RecordingVision:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def detect(self, _frame, _goals):
        return []

    async def judge(self, condition, frame, **kwargs):
        self.calls.append(
            {
                "condition": condition,
                "frame": frame,
                "frames": tuple(kwargs.get("frames") or ()),
            }
        )
        return Verdict(
            False,
            0.99,
            f"{frame} not satisfied",
            usage={"input_tokens": 1, "cost_usd": 0.001},
        )


def test_sequential_public_runs_start_at_a_fresh_source_lifecycle_boundary():
    """A second finite source must not inherit the first run's temporal pixels."""

    vision = _RecordingVision()
    core = _core(vision)

    async def exercise():
        assert await _drain(core, "first") == []
        assert await _drain(core, "second") == []

    asyncio.run(exercise())

    assert [call["frame"] for call in vision.calls] == [
        "first:receipt",
        "second:receipt",
    ]
    assert "first:receipt" not in repr(vision.calls[1]["frames"]), (
        "run() lifecycle boundary leaked the prior finite source's frame "
        "window into the next provider request"
    )
    assert core.metrics.get("funnel.judged") == 2
    assert core.metrics.get("judge.tokens.input") == 2
    assert core.metrics.get("judge.cost.usd") == 0.002


class _ConcurrentVision:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.all_started = asyncio.Event()
        self.release = {
            "left:receipt": asyncio.Event(),
            "right:receipt": asyncio.Event(),
        }

    async def detect(self, _frame, _goals):
        return []

    async def judge(self, _condition, frame, **_kwargs):
        self.calls.append(frame)
        if len(self.calls) == 2:
            self.all_started.set()
        await self.release[frame].wait()
        return Verdict(
            False,
            0.99,
            f"{frame} not satisfied",
            usage={"input_tokens": 1, "cost_usd": 0.001},
        )


def test_concurrent_public_runs_share_one_owner_or_reject_before_provider_work():
    """One core cannot have two independent owner ordinals or callback lifecycles."""

    async def exercise():
        vision = _ConcurrentVision()
        core = _core(vision)
        completion_ordinals = []
        original_reduce = core.reduce_judgment_completion

        def record_reduction(completion):
            completion_ordinals.append(completion.completion_ordinal)
            return original_reduce(completion)

        core.reduce_judgment_completion = record_reduction

        async def original_trace(*_args, **_kwargs):
            return None

        async def left_trace(*_args, **_kwargs):
            return None

        async def right_trace(*_args, **_kwargs):
            return None

        core.on_trace = original_trace
        tasks = [
            asyncio.create_task(_drain(core, "left", on_trace=left_trace)),
            asyncio.create_task(_drain(core, "right", on_trace=right_trace)),
        ]
        try:
            await asyncio.wait_for(vision.all_started.wait(), timeout=1.0)
        except TimeoutError:
            # A conforming implementation may serialize or reject the second
            # run before it can spend provider work.
            for release in vision.release.values():
                release.set()
        else:
            # Finish the invocation that installed its callback wrapper first.
            # Its teardown must not clobber or orphan the still-live owner.
            vision.release["left:receipt"].set()
            await tasks[0]
            vision.release["right:receipt"].set()
        outcomes = await asyncio.gather(*tasks, return_exceptions=True)
        return (
            core,
            vision,
            original_trace,
            completion_ordinals,
            outcomes,
        )

    (
        core,
        vision,
        original_trace,
        completion_ordinals,
        outcomes,
    ) = asyncio.run(exercise())
    failures = [outcome for outcome in outcomes if isinstance(outcome, BaseException)]
    violations = []

    if failures:
        if len(vision.calls) > 1:
            violations.append(
                "a rejected concurrent run started more than one provider call"
            )
    else:
        if completion_ordinals != [1, 2]:
            violations.append(
                "concurrent runs minted duplicate per-run completion ordinals "
                f"{completion_ordinals!r}"
            )
        if core.metrics.get("funnel.judged") != 2:
            violations.append(
                "two provider receipts were not both accounted: "
                f"funnel.judged={core.metrics.get('funnel.judged')!r}"
            )
        if core.metrics.get("judge.tokens.input") != 2:
            violations.append(
                "two provider receipts lost token accounting: "
                f"judge.tokens.input={core.metrics.get('judge.tokens.input')!r}"
            )
        if core.metrics.get("judge.cost.usd") != 0.002:
            violations.append(
                "two provider receipts lost cost accounting: "
                f"judge.cost.usd={core.metrics.get('judge.cost.usd')!r}"
            )

    if core.on_trace is not original_trace:
        violations.append(
            "overlapping run() teardown left another invocation's trace "
            "wrapper installed on the shared core"
        )
    assert not violations, "; ".join(violations)
