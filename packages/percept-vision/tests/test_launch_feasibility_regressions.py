"""Deterministic launch-feasibility regressions for transition latency and source gaps.

These tests spend no provider budget.  They pin two architectural facts exposed by the
Traffic_Data control run:

* one transition judgment is one scheduler job but two sequential provider calls, so the
  historical 1.9 second outer budget cannot contain two otherwise-healthy 1.35 second calls;
* a sequence gap in an otherwise unchanged source epoch resets continuity and cancels an active
  async judgment exactly like a real reconnect, even though only the gap accounts missing frames.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from percept import Goal, Percept
from percept.adapters import Verdict
from percept.harness.tier0 import gray_frame
from percept.runtime import RuntimeStats, SourceEnvelope, SourceFormat, run
from percept.runtime.worker import _JudgeScheduler


SOURCE_FORMAT = SourceFormat(media_type="application/x-percept-opaque")
STILL = gray_frame(8, 8, 0)
MOVED = gray_frame(8, 8, 255)


class FixedLatencyVision:
    """A cancellation-sensitive provider double with a fixed latency per judge call."""

    def __init__(self, delay_s: float = 1.35) -> None:
        self.delay_s = delay_s
        self.calls: list[str] = []
        self.cancelled = 0

    async def detect(self, frame, goals):
        return []

    async def judge(self, condition, frame, **kwargs):
        self.calls.append(condition)
        try:
            await asyncio.sleep(self.delay_s)
        except asyncio.CancelledError:
            self.cancelled += 1
            raise
        return Verdict(satisfied=False, confidence=0.99, evidence="deterministic")


def _core(vision, goal: Goal) -> Percept:
    core = Percept.create(vision=vision, discover_plugins=False)
    core.add_goal(goal)
    return core


async def _prepared_job(core: Percept, goal_id: str):
    candidate = core.prepare_observation_candidate(
        "frame",
        "camera",
        t=1.0,
        expected_goals_revision=core.goals.revision,
        expected_source_generation=core.source_continuity_generation("camera"),
        goal_ids=(goal_id,),
    )
    assert candidate is not None
    assert core.launch_observation_candidate(candidate)
    reduction = core.reduce_observation_completion(
        await core.dispatch_observation_candidate(candidate)
    )
    job = core.prepare_judgment(reduction, goal_id=goal_id)
    assert job is not None
    return job


async def _dispatch_one_job(core: Percept, job, *, deadline_s: float):
    stats = RuntimeStats()
    scheduler = _JudgeScheduler(
        core,
        "camera",
        stats,
        decision_deadline_s=deadline_s,
    )
    started = time.monotonic()
    assert scheduler.submit(1.0, (job,))
    await scheduler.flush()
    elapsed = time.monotonic() - started
    return stats, scheduler.drain(), elapsed


def test_transition_pair_exceeds_old_outer_deadline_while_one_call_fits():
    """The old 1.9s budget measures a transition's job, not its two serial calls.

    Each fake provider call takes 1.35s: a normal target-state judgment finishes with ample margin,
    while the transition starts its second healthy call and is then cancelled by the outer job
    deadline.  The broad elapsed-time bounds diagnose gross scheduler drift without relying on a
    millisecond-precise CI clock.
    """

    async def drive():
        target_vision = FixedLatencyVision()
        target = _core(
            target_vision,
            Goal(id="target", condition="the target state is true", say="target"),
        )
        target_job = await _prepared_job(target, "target")
        target_budget = _JudgeScheduler(
            target, "camera", None, decision_deadline_s=1.9
        )._batch_deadline_s((target_job,))
        target_stats, target_rows, target_elapsed = await _dispatch_one_job(
            target, target_job, deadline_s=1.9
        )

        transition_vision = FixedLatencyVision()
        transition = _core(
            transition_vision,
            Goal(
                id="transition",
                condition="the user woke up",
                say="awake",
                state={
                    "type": "transition",
                    "from": "the user is asleep",
                    "to": "the user is awake",
                    "count": 0,
                },
            ),
        )
        transition_job = await _prepared_job(transition, "transition")
        transition_budget = _JudgeScheduler(
            transition, "camera", None, decision_deadline_s=1.9
        )._batch_deadline_s((transition_job,))
        transition_stats, transition_rows, transition_elapsed = (
            await _dispatch_one_job(transition, transition_job, deadline_s=1.9)
        )

        return {
            "target": (
                target_vision,
                target_stats,
                target_rows,
                target_elapsed,
                target_budget,
            ),
            "transition": (
                transition_vision,
                transition_stats,
                transition_rows,
                transition_elapsed,
                transition_budget,
            ),
        }

    result = asyncio.run(drive())
    target_vision, target_stats, target_rows, target_elapsed, target_budget = (
        result["target"]
    )
    (
        transition_vision,
        transition_stats,
        transition_rows,
        transition_elapsed,
        transition_budget,
    ) = result["transition"]

    assert target_budget == pytest.approx(1.9)
    assert len(target_rows) == 1 and len(target_rows[0][1]) == 1
    assert target_stats.judge_deadline_expired == 0
    assert target_vision.calls == ["the target state is true"]
    assert target_vision.cancelled == 0
    assert 1.2 <= target_elapsed < 1.9

    # The scheduler sees one transition JOB and therefore assigns one 1.9s budget, even though
    # dispatch_judgment starts two provider CALLS in series.
    assert transition_budget == pytest.approx(1.9)
    assert transition_rows == [(1.0, ())]
    assert transition_stats.judge_deadline_expired == 1
    assert transition_vision.calls == ["the user is asleep", "the user is awake"]
    assert transition_vision.cancelled == 1
    assert 1.75 <= transition_elapsed < 2.4


class BlockingFirstJudgmentVision:
    """Blocks only the first call so a stream boundary can cancel visible active work."""

    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.calls = 0
        self.cancelled = 0
        self.completed = 0

    async def detect(self, frame, goals):
        return []

    async def judge(self, condition, frame, **kwargs):
        self.calls += 1
        if self.calls == 1:
            self.started.set()
            try:
                await self.release.wait()
            except asyncio.CancelledError:
                self.cancelled += 1
                raise
        self.completed += 1
        return Verdict(satisfied=False, confidence=0.99, evidence="clear")


def _envelope(
    sequence: int,
    *,
    epoch: int = 0,
    observed_time: float,
    source_time: float,
    signal,
) -> SourceEnvelope:
    return SourceEnvelope(
        source_id="camera",
        modality="video",
        epoch=epoch,
        sequence=sequence,
        observed_time=observed_time,
        source_time=source_time,
        format=SOURCE_FORMAT,
        payload=f"epoch-{epoch}-sequence-{sequence}",
        signal=signal,
    )


@pytest.mark.parametrize(
    ("boundary", "expected_event", "expected_missing"),
    [
        (
            _envelope(
                3,
                observed_time=3.0,
                source_time=3.0,
                signal=MOVED,
            ),
            "source_gap",
            1,
        ),
        (
            _envelope(
                0,
                epoch=1,
                observed_time=3.0,
                source_time=0.0,
                signal=MOVED,
            ),
            "source_reconnect",
            0,
        ),
    ],
    ids=("same-epoch-queue-gap", "new-epoch-reconnect"),
)
def test_gap_and_reconnect_both_cancel_active_async_judgment(
    boundary: SourceEnvelope,
    expected_event: str,
    expected_missing: int,
):
    """A same-epoch queue gap currently has reconnect-strength cancellation semantics."""

    async def drive():
        vision = BlockingFirstJudgmentVision()
        core = _core(
            vision,
            Goal(id="watch", condition="something changed", say="changed"),
        )
        stats = RuntimeStats()
        health = []

        async def source():
            yield _envelope(
                0,
                observed_time=0.0,
                source_time=0.0,
                signal=STILL,
            )
            yield _envelope(
                1,
                observed_time=1.0,
                source_time=1.0,
                signal=MOVED,
            )
            await asyncio.wait_for(vision.started.wait(), timeout=1.0)
            yield boundary
            # Let the scheduler's cancellation callback settle before making a failed test
            # releasable; on the expected path release is unnecessary because cancellation wins.
            for _ in range(4):
                await asyncio.sleep(0)
            vision.release.set()

        incidents = [
            incident
            async for incident in run(
                core,
                source=source(),
                judge_dispatch="async",
                timebase="media",
                stats=stats,
                on_health=health.append,
            )
        ]
        return vision, stats, health, incidents

    vision, stats, health, incidents = asyncio.run(drive())
    kinds = [event.kind for event in health]
    boundary_event = next(event for event in health if event.kind == expected_event)

    assert incidents == []
    assert vision.calls == 1
    assert vision.cancelled == 1
    assert vision.completed == 0
    assert stats.source_discontinuities == 1
    assert stats.source_missing_sequences == expected_missing
    assert boundary_event.data["missing_sequences"] == expected_missing
    assert stats.judge_source_discontinuity_active_dropped == 1
    assert stats.judge_source_discontinuity_pending_dropped == 0
    assert "judge_source_discontinuity_dropped" in kinds


@pytest.mark.xfail(
    strict=True,
    reason=(
        "proposal: a same-epoch sampling/queue gap should reset temporal evidence without "
        "cancelling an immutable pre-gap judgment"
    ),
)
def test_same_epoch_queue_gap_should_preserve_active_pre_gap_judgment():
    """Executable policy proposal separating lossy sampling from a true source boundary.

    The gap remains visible in coverage and must still prevent a future temporal window from
    spanning missing frames.  The desired difference is narrower: provider work already launched
    from an immutable pre-gap evidence snapshot remains valid for its original timestamp and is not
    discarded merely because the capture queue chose freshness over backlog.
    """

    async def drive():
        vision = BlockingFirstJudgmentVision()
        core = _core(
            vision,
            Goal(id="watch", condition="something changed", say="changed"),
        )
        stats = RuntimeStats()

        async def source():
            yield _envelope(
                0,
                observed_time=0.0,
                source_time=0.0,
                signal=STILL,
            )
            yield _envelope(
                1,
                observed_time=1.0,
                source_time=1.0,
                signal=MOVED,
            )
            await asyncio.wait_for(vision.started.wait(), timeout=1.0)
            yield _envelope(
                3,
                observed_time=3.0,
                source_time=3.0,
                signal=MOVED,
            )
            vision.release.set()

        incidents = [
            incident
            async for incident in run(
                core,
                source=source(),
                judge_dispatch="async",
                timebase="media",
                stats=stats,
            )
        ]
        return vision, stats, incidents

    vision, stats, incidents = asyncio.run(drive())
    assert incidents == []
    assert stats.source_missing_sequences == 1
    assert stats.source_discontinuities == 1
    assert stats.judge_source_discontinuity_active_dropped == 0
    assert vision.cancelled == 0
    assert vision.completed == 1
