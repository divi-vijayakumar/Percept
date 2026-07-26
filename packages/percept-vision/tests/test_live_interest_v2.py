"""Public live VID admission lifecycle contract for ``percept.run``."""

from __future__ import annotations

import asyncio
import threading

import pytest

from percept import Goal, InterestIntervalConfig, Tier0MotionInterest
from percept.adapters import Verdict
from percept.backends.fake import FakeVision
from percept.edge.tier0 import CivilTime, GatedWatchLoop, gray_frame
from percept.goals import GoalOp
from percept.harness import tier0 as tier0_module
from percept.runtime import (
    InterestBackpressureError,
    InterestCallbackUnsupportedError,
    InterestSubscription,
    SourceEnvelope,
    SourceFormat,
    run,
)


OPAQUE = SourceFormat(media_type="application/x-percept-opaque")
BLACK = gray_frame(2, 2, 0)
WHITE = gray_frame(2, 2, 255)
GRAY = gray_frame(2, 2, 128)


def _envelope(
    sequence: int,
    signal,
    *,
    epoch: int = 0,
    source_id: str = "video:test",
    source_time: float | None = None,
    observed_time: float | None = None,
    source_format: SourceFormat = OPAQUE,
    discontinuity: str | None = None,
) -> SourceEnvelope:
    return SourceEnvelope(
        source_id=source_id,
        modality="video",
        epoch=epoch,
        sequence=sequence,
        source_time=10.0 + sequence if source_time is None else source_time,
        observed_time=100.0 + sequence if observed_time is None else observed_time,
        format=source_format,
        payload="clear",
        signal=signal,
        discontinuity=discontinuity,
    )


def _core(*watch_ids: str):
    core = FakeVision(
        script={
            "clear": Verdict(
                satisfied=False,
                confidence=0.99,
                evidence="condition is not satisfied",
            )
        }
    )
    from percept import Percept

    agent = Percept.create(vision=core, discover_plugins=False)
    for watch_id in watch_ids:
        agent.add_goal(
            Goal(id=watch_id, condition=f"{watch_id} condition", say=f"{watch_id} fired")
        )
    return agent


@pytest.mark.asyncio
async def test_run_emits_exact_per_watch_lifecycle_with_source_and_observer_clocks() -> None:
    subscription = InterestSubscription()
    events = []

    async def consume() -> None:
        async for event in subscription:
            events.append(event)

    consumer = asyncio.create_task(consume())
    async for _ in run(
        _core("watch-b", "watch-a"),
        source=[_envelope(0, BLACK), _envelope(1, WHITE), _envelope(2, BLACK)],
        interest_subscription=subscription,
        interest_session_id="test-session",
        judge_dispatch="inline",
    ):
        pass
    await consumer

    assert [(event.watch_id, event.kind) for event in events] == [
        ("watch-a", "interest.entered"),
        ("watch-b", "interest.entered"),
        ("watch-a", "interest.updated"),
        ("watch-b", "interest.updated"),
        ("watch-a", "interest.exited"),
        ("watch-b", "interest.exited"),
    ]
    entered, updated, exited = events[0], events[2], events[-1]
    assert entered.source_start == 11.0
    assert entered.observed_start == 101.0
    assert updated.source_end == 12.0
    assert updated.observed_end == 102.0
    assert exited.reason == "end_of_stream"
    assert exited.source_end == 12.0
    assert exited.observed_end == 102.0
    assert subscription.closed is True


async def _collect(source, *watch_ids: str, session_id: str = "test-session"):
    subscription = InterestSubscription()
    events = []

    async def consume() -> None:
        async for event in subscription:
            events.append(event)

    consumer = asyncio.create_task(consume())
    async for _ in run(
        _core(*(watch_ids or ("watch",))),
        source=source,
        interest_subscription=subscription,
        interest_session_id=session_id,
        judge_dispatch="inline",
    ):
        pass
    await consumer
    return events


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("tail", "reason", "missing"),
    [
        (_envelope(3, BLACK), "source_gap", 1),
        (
            _envelope(0, BLACK, epoch=1, observed_time=200.0),
            "source_reconnect",
            0,
        ),
        (
            _envelope(
                0,
                BLACK,
                epoch=1,
                observed_time=200.0,
                discontinuity="capture_overflow",
            ),
            "source_capture_overflow",
            0,
        ),
    ],
)
async def test_discontinuity_closes_at_last_admitted_unit(
    tail: SourceEnvelope, reason: str, missing: int
) -> None:
    events = await _collect([_envelope(0, BLACK), _envelope(1, WHITE), tail])
    assert [event.kind for event in events] == [
        "interest.entered",
        "interest.exited",
    ]
    exited = events[-1]
    assert exited.reason == reason
    assert exited.sequence_end == 1
    assert exited.event_sequence == tail.sequence
    assert exited.event_source_epoch == tail.epoch
    assert exited.missing_sequences == missing
    assert exited.coverage_kind == "observed_after_discontinuity"
    assert reason in exited.source_events


@pytest.mark.asyncio
async def test_format_change_closes_without_cross_format_diff() -> None:
    changed = SourceFormat(
        media_type="application/x-percept-opaque",
        width=2,
        height=2,
    )
    events = await _collect(
        [_envelope(0, BLACK), _envelope(1, WHITE), _envelope(2, BLACK, source_format=changed)]
    )
    assert [event.kind for event in events] == [
        "interest.entered",
        "interest.exited",
    ]
    assert events[-1].reason == "source_format_change"
    assert events[-1].sequence_end == 1


@pytest.mark.asyncio
async def test_duplicate_and_reordered_envelopes_create_no_lifecycle_units() -> None:
    events = await _collect(
        [
            _envelope(0, BLACK),
            _envelope(1, WHITE),
            _envelope(1, WHITE, observed_time=102.0),
            _envelope(0, BLACK, observed_time=103.0),
            _envelope(2, BLACK, observed_time=104.0),
        ]
    )
    assert [event.kind for event in events] == [
        "interest.entered",
        "interest.updated",
        "interest.exited",
    ]
    assert events[1].event_sequence == 2


@pytest.mark.asyncio
async def test_signal_loss_closes_at_prior_admitted_unit() -> None:
    events = await _collect(
        [_envelope(0, BLACK), _envelope(1, WHITE), _envelope(2, WHITE)]
    )
    assert [event.kind for event in events] == [
        "interest.entered",
        "interest.exited",
    ]
    assert events[-1].reason == "signal_lost"
    assert events[-1].sequence_end == 1
    assert events[-1].event_sequence == 2


@pytest.mark.asyncio
async def test_fixed_cadence_wait_has_typed_cadence_reason() -> None:
    core = _core("fixed")
    goal = core.goals.get("fixed")
    assert goal is not None
    goal.state["cadence"] = {
        "min_judge_interval": 10.0,
        "gate_on": "fixed",
    }
    subscription = InterestSubscription()
    consumer = asyncio.create_task(_consume_all(subscription))
    async for _ in run(
        core,
        source=[_envelope(0, BLACK), _envelope(1, BLACK)],
        interest_subscription=subscription,
        judge_dispatch="inline",
    ):
        pass
    events = await consumer
    assert [event.kind for event in events] == [
        "interest.entered",
        "interest.exited",
    ]
    assert events[0].reason == "fixed_cadence"
    assert events[0].admission_cause == "fixed"
    assert events[-1].reason == "cadence_limited"
    assert events[-1].sequence_end == 0
    assert events[-1].event_sequence == 1


@pytest.mark.asyncio
async def test_reconnect_reentry_gets_a_distinct_epoch_interval() -> None:
    events = await _collect(
        [
            _envelope(0, BLACK),
            _envelope(1, WHITE),
            _envelope(0, BLACK, epoch=1, observed_time=200.0),
            _envelope(1, WHITE, epoch=1, observed_time=201.0),
        ]
    )
    assert [event.kind for event in events] == [
        "interest.entered",
        "interest.exited",
        "interest.entered",
        "interest.exited",
    ]
    assert events[1].reason == "source_reconnect"
    assert events[0].interval_id != events[2].interval_id
    assert events[0].source_epoch == 0
    assert events[2].source_epoch == 1


@pytest.mark.asyncio
async def test_replay_and_live_share_the_motion_lifecycle_vocabulary() -> None:
    replay_interest = Tier0MotionInterest(
        source="camera",
        config=InterestIntervalConfig(enter_threshold=0.7, exit_threshold=0.2),
    )
    replay_core = _core("watch")
    replay = await GatedWatchLoop(
        replay_core,
        list(replay_core.goals.active()),
        threshold=0.7,
        source="camera",
        interest=replay_interest,
    ).run(
        [(10.0, BLACK), (11.0, WHITE), (12.0, BLACK)],
        interest_stream_epoch="video:test:0",
        interest_observed_times=(100.0, 101.0, 102.0),
    )
    live = await _collect(
        [
            _envelope(0, BLACK),
            _envelope(1, WHITE),
            _envelope(2, BLACK),
        ]
    )
    assert [event.kind for event in live] == [
        event.kind for event in replay.interest_events
    ]
    assert [event.reason for event in live] == [
        event.reason for event in replay.interest_events
    ]
    assert live[0].admission_cause == "motion"


@pytest.mark.asyncio
async def test_ids_bind_session_media_and_watch_without_source_secret() -> None:
    secret = "rtsp://alice:password@example.invalid/private"
    source = [
        _envelope(0, BLACK, source_id=secret),
        _envelope(1, WHITE, source_id=secret),
    ]
    same = await _collect(source, "a", "b", session_id="same")
    replay = await _collect(source, "a", "b", session_id="same")
    other_run = await _collect(source, "a", "b", session_id="other")
    other_media = await _collect(
        [
            _envelope(0, BLACK, source_id=secret),
            _envelope(1, GRAY, source_id=secret),
        ],
        "a",
        "b",
        session_id="same",
    )
    assert [event.interval_id for event in same] == [
        event.interval_id for event in replay
    ]
    assert [event.evidence_id for event in same] == [
        event.evidence_id for event in replay
    ]
    assert same[0].interval_id != same[1].interval_id
    assert same[0].interval_id != other_run[0].interval_id
    assert same[0].interval_id != other_media[0].interval_id
    assert same[0].source_id.startswith("source_")
    assert "alice" not in repr(same[0])
    assert "password" not in repr(same[0])


@pytest.mark.asyncio
async def test_gate_policy_supplies_cadence_and_schedule_exit_reasons() -> None:
    core = _core("scheduled", "cadenced")
    scheduled = core.goals.get("scheduled")
    cadenced = core.goals.get("cadenced")
    assert scheduled is not None and cadenced is not None
    scheduled.state["schedule"] = {"windows": [[0, 100]], "days": [1]}
    cadenced.state["cadence"] = {
        "min_judge_interval": 10.0,
        "gate_on": "motion",
    }
    civil = iter(
        [
            CivilTime(weekday=1, minute_of_day=50),
            CivilTime(weekday=1, minute_of_day=50),
            CivilTime(weekday=1, minute_of_day=200),
        ]
    )
    subscription = InterestSubscription()
    events = []

    async def consume() -> None:
        async for event in subscription:
            events.append(event)

    consumer = asyncio.create_task(consume())
    async for _ in run(
        core,
        source=[_envelope(0, BLACK), _envelope(1, WHITE), _envelope(2, BLACK)],
        clock=lambda: next(civil),
        interest_subscription=subscription,
        judge_dispatch="inline",
    ):
        pass
    await consumer
    exits = {event.watch_id: event for event in events if event.kind == "interest.exited"}
    assert exits["scheduled"].reason == "schedule_inactive"
    assert exits["cadenced"].reason == "cadence_limited"


@pytest.mark.asyncio
async def test_pause_closes_and_resume_cold_starts_gate() -> None:
    states = iter((False, False, True, False, False))
    subscription = InterestSubscription()
    events = []

    async def consume() -> None:
        async for event in subscription:
            events.append(event)

    consumer = asyncio.create_task(consume())
    async for _ in run(
        _core("watch"),
        source=[
            _envelope(0, BLACK),
            _envelope(1, WHITE),
            _envelope(2, BLACK),
            _envelope(3, WHITE),
            _envelope(4, BLACK),
        ],
        paused=lambda: next(states),
        interest_subscription=subscription,
        judge_dispatch="inline",
    ):
        pass
    await consumer
    assert [event.kind for event in events] == [
        "interest.entered",
        "interest.exited",
        "interest.entered",
        "interest.exited",
    ]
    assert events[1].reason == "paused"
    assert events[1].sequence_end == 1
    assert events[1].event_sequence == 2
    assert events[2].sequence_start == 4


@pytest.mark.asyncio
async def test_watch_removal_closes_only_removed_watch() -> None:
    core = _core("remove", "keep")
    subscription = InterestSubscription()
    events = []

    def remove_on_frame(t, _frame) -> None:
        if t == 11.0:
            core.goals.apply([GoalOp(op="cancel", goal_id="remove")])

    async def consume() -> None:
        async for event in subscription:
            events.append(event)

    consumer = asyncio.create_task(consume())
    async for _ in run(
        core,
        source=[_envelope(0, BLACK), _envelope(1, WHITE), _envelope(2, BLACK)],
        on_frame=remove_on_frame,
        interest_subscription=subscription,
        judge_dispatch="inline",
    ):
        pass
    await consumer
    removed = [event for event in events if event.watch_id == "remove"]
    kept = [event for event in events if event.watch_id == "keep"]
    assert [event.kind for event in removed] == [
        "interest.entered",
        "interest.exited",
    ]
    assert removed[-1].reason == "watch_removed"
    assert [event.kind for event in kept] == [
        "interest.entered",
        "interest.updated",
        "interest.exited",
    ]


@pytest.mark.asyncio
async def test_cancellation_and_worker_failure_close_open_intervals() -> None:
    async def blocked_source():
        yield _envelope(0, BLACK)
        yield _envelope(1, WHITE)
        await asyncio.Event().wait()

    subscription = InterestSubscription()
    events = []
    entered = asyncio.Event()

    async def consume() -> None:
        async for event in subscription:
            events.append(event)
            if event.kind == "interest.entered":
                entered.set()

    consumer = asyncio.create_task(consume())
    task = asyncio.create_task(
        _drain_run(
            _core("watch"),
            blocked_source(),
            subscription,
        )
    )
    await asyncio.wait_for(entered.wait(), timeout=1.0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await consumer
    assert events[-1].reason == "cancelled"

    async def failing_source():
        yield _envelope(0, BLACK)
        yield _envelope(1, WHITE)
        raise RuntimeError("source failed")

    failed_subscription = InterestSubscription()
    failed_events = []

    async def consume_failed() -> None:
        async for event in failed_subscription:
            failed_events.append(event)

    failed_consumer = asyncio.create_task(consume_failed())
    with pytest.raises(RuntimeError, match="source failed"):
        await _drain_run(_core("watch"), failing_source(), failed_subscription)
    await failed_consumer
    assert failed_events[-1].reason == "worker_error"


async def _drain_run(core, source, subscription) -> None:
    async for _ in run(
        core,
        source=source,
        interest_subscription=subscription,
        judge_dispatch="inline",
    ):
        pass


def _assert_balanced_lifecycle(events) -> None:
    active = set()
    for event in events:
        if event.kind == "interest.entered":
            assert event.interval_id not in active
            active.add(event.interval_id)
        elif event.kind == "interest.updated":
            assert event.interval_id in active
        else:
            assert event.kind == "interest.exited"
            assert event.interval_id in active
            active.remove(event.interval_id)
    assert active == set()


@pytest.mark.asyncio
async def test_terminal_coordinate_uses_last_admitted_unit_after_hook_failure() -> None:
    subscription = InterestSubscription()
    events = []

    def fail_on_frame(t, _frame) -> None:
        if t == 12.0:
            raise RuntimeError("hook failed")

    async def consume() -> None:
        async for event in subscription:
            events.append(event)

    consumer = asyncio.create_task(consume())
    with pytest.raises(RuntimeError, match="hook failed"):
        async for _ in run(
            _core("watch"),
            source=[_envelope(0, BLACK), _envelope(1, WHITE), _envelope(2, BLACK)],
            on_frame=fail_on_frame,
            interest_subscription=subscription,
            judge_dispatch="inline",
        ):
            pass
    await consumer
    assert events[-1].reason == "worker_error"
    assert events[-1].sequence_end == 1
    assert events[-1].event_sequence == 2
    assert events[-1].event_source_time == 12.0


@pytest.mark.asyncio
async def test_bounded_channel_fails_loud_and_reserves_terminal_exit() -> None:
    subscription = InterestSubscription(max_batches=1)
    with pytest.raises(InterestBackpressureError):
        await _drain_run(
            _core("watch"),
            [
                _envelope(0, BLACK),
                _envelope(1, WHITE),
                _envelope(2, BLACK),
            ],
            subscription,
        )
    events = [event async for event in subscription]
    assert subscription.closed is True
    assert [event.kind for event in events] == [
        "interest.entered",
        "interest.updated",
        "interest.exited",
    ]
    assert events[-1].reason == "worker_error"
    assert events[-1].sequence_end == 2
    _assert_balanced_lifecycle(events)


@pytest.mark.asyncio
async def test_overflow_preserves_multi_watch_update_and_new_interval_before_closure() -> None:
    core = _core("a")
    added = False

    def add_watch_on_frame(t, _frame) -> None:
        nonlocal added
        if t == 11.0 and not added:
            added = True
            core.add_goal(Goal(id="b", condition="b condition", say="b fired"))

    subscription = InterestSubscription(max_batches=1)
    with pytest.raises(InterestBackpressureError):
        async for _ in run(
            core,
            source=[_envelope(0, BLACK), _envelope(1, WHITE), _envelope(2, BLACK)],
            on_frame=add_watch_on_frame,
            interest_subscription=subscription,
            judge_dispatch="inline",
        ):
            pass
    events = [event async for event in subscription]
    assert [(event.watch_id, event.kind) for event in events] == [
        ("a", "interest.entered"),
        ("a", "interest.updated"),
        ("b", "interest.entered"),
        ("a", "interest.exited"),
        ("b", "interest.exited"),
    ]
    assert [event.reason for event in events[-2:]] == ["worker_error", "worker_error"]
    _assert_balanced_lifecycle(events)


@pytest.mark.asyncio
async def test_overflow_preserves_rejected_signal_loss_exit() -> None:
    subscription = InterestSubscription(max_batches=1)
    with pytest.raises(InterestBackpressureError):
        await _drain_run(
            _core("a"),
            [_envelope(0, BLACK), _envelope(1, WHITE), _envelope(2, WHITE)],
            subscription,
        )
    events = [event async for event in subscription]
    assert [event.kind for event in events] == [
        "interest.entered",
        "interest.exited",
    ]
    assert events[-1].reason == "signal_lost"
    _assert_balanced_lifecycle(events)


@pytest.mark.asyncio
async def test_overflow_preserves_admission_change_before_new_terminal_exit() -> None:
    core = _core("a")
    changed = False

    def change_admission_on_frame(t, _frame) -> None:
        nonlocal changed
        if t == 11.0 and not changed:
            changed = True
            goal = core.goals.get("a")
            assert goal is not None
            goal.state["cadence"] = {
                "min_judge_interval": 1.0,
                "gate_on": "fixed",
            }
            core.goals.apply(
                [GoalOp(op="update", goal_id="a", fields={"say": "a still fired"})]
            )

    subscription = InterestSubscription(max_batches=1)
    with pytest.raises(InterestBackpressureError):
        async for _ in run(
            core,
            source=[_envelope(0, BLACK), _envelope(1, WHITE), _envelope(2, BLACK)],
            on_frame=change_admission_on_frame,
            interest_subscription=subscription,
            judge_dispatch="inline",
        ):
            pass
    events = [event async for event in subscription]
    assert [event.kind for event in events] == [
        "interest.entered",
        "interest.exited",
        "interest.entered",
        "interest.exited",
    ]
    assert [event.reason for event in events] == [
        "tier0_motion",
        "admission_changed",
        "fixed_cadence",
        "worker_error",
    ]
    assert [event.admission_cause for event in events] == [
        "motion",
        "motion",
        "fixed",
        "fixed",
    ]
    assert events[0].interval_id == events[1].interval_id
    assert events[2].interval_id == events[3].interval_id
    assert events[0].interval_id != events[2].interval_id
    _assert_balanced_lifecycle(events)


@pytest.mark.asyncio
async def test_disabled_path_hashes_nothing_and_enabled_path_adds_no_judgments(
    monkeypatch,
) -> None:
    def unexpected_hash(*_args, **_kwargs):
        raise AssertionError("disabled lifecycle must not hash causal media")

    def unexpected_outcome(*_args, **_kwargs):
        raise AssertionError("disabled lifecycle must not allocate outcome DTOs")

    monkeypatch.setattr(tier0_module, "_motion_evidence_id", unexpected_hash)
    monkeypatch.setattr(tier0_module, "WatchAdmissionOutcome", unexpected_outcome)
    disabled = _core("watch")
    async for _ in run(
        disabled,
        source=[_envelope(0, BLACK), _envelope(1, WHITE), _envelope(2, BLACK)],
        judge_dispatch="inline",
    ):
        pass
    monkeypatch.undo()

    disabled = _core("watch")
    enabled = _core("watch")
    from percept.runtime import RuntimeStats

    disabled_stats = RuntimeStats()
    enabled_stats = RuntimeStats()
    source = [_envelope(0, BLACK), _envelope(1, WHITE), _envelope(2, BLACK)]
    async for _ in run(
        disabled,
        source=source,
        stats=disabled_stats,
        judge_dispatch="inline",
    ):
        pass
    subscription = InterestSubscription()
    consumer = asyncio.create_task(_consume_all(subscription))
    async for _ in run(
        enabled,
        source=source,
        stats=enabled_stats,
        interest_subscription=subscription,
        judge_dispatch="inline",
    ):
        pass
    await consumer
    assert enabled_stats.judge_calls == disabled_stats.judge_calls
    assert enabled_stats.perceive_dispatches == disabled_stats.perceive_dispatches


async def _consume_all(subscription) -> list:
    return [event async for event in subscription]


@pytest.mark.asyncio
async def test_callback_and_impostor_rejected_before_source_read() -> None:
    reads = 0
    callback_called = False
    threads_before = {thread.ident for thread in threading.enumerate()}

    async def source():
        nonlocal reads
        reads += 1
        yield _envelope(0, BLACK)

    def callback(_event) -> None:
        nonlocal callback_called
        callback_called = True

    with pytest.raises(InterestCallbackUnsupportedError, match="InterestSubscription"):
        async for _ in run(_core("watch"), source=source(), on_interest=callback):
            pass
    with pytest.raises(TypeError, match="exact InterestSubscription"):
        async for _ in run(
            _core("watch"),
            source=source(),
            interest_subscription=object(),  # type: ignore[arg-type]
        ):
            pass
    assert reads == 0
    assert callback_called is False
    assert {thread.ident for thread in threading.enumerate()} == threads_before

    with pytest.raises(TypeError, match="cannot be subclassed"):
        class InvalidSubscription(InterestSubscription):
            pass


@pytest.mark.asyncio
async def test_subscription_binds_before_preflight_and_always_closes() -> None:
    subscription = InterestSubscription()
    consumer = asyncio.create_task(_consume_all(subscription))
    with pytest.raises(ValueError, match="judge_dispatch"):
        async for _ in run(
            _core("watch"),
            source=[_envelope(0, BLACK)],
            judge_dispatch="invalid",
            interest_subscription=subscription,
        ):
            pass
    assert await consumer == []
    assert subscription.closed is True

    callback_subscription = InterestSubscription()
    callback_consumer = asyncio.create_task(_consume_all(callback_subscription))
    called = False

    def callback(_event) -> None:
        nonlocal called
        called = True

    with pytest.raises(InterestCallbackUnsupportedError):
        async for _ in run(
            _core("watch"),
            source=[_envelope(0, BLACK)],
            on_interest=callback,
            interest_subscription=callback_subscription,
        ):
            pass
    assert await callback_consumer == []
    assert callback_subscription.closed is True
    assert called is False

    subclass_code_ran = False

    class SessionIdImpostor(str):
        def strip(self, *_args, **_kwargs):
            nonlocal subclass_code_ran
            subclass_code_ran = True
            raise AssertionError("session id subclass executed caller code")

    impostor_subscription = InterestSubscription()
    impostor_consumer = asyncio.create_task(_consume_all(impostor_subscription))
    with pytest.raises(ValueError, match="interest_session_id"):
        async for _ in run(
            _core("watch"),
            source=[_envelope(0, BLACK)],
            interest_session_id=SessionIdImpostor("impostor"),
            interest_subscription=impostor_subscription,
        ):
            pass
    assert await impostor_consumer == []
    assert impostor_subscription.closed is True
    assert subclass_code_ran is False


@pytest.mark.asyncio
async def test_subscription_instance_cannot_replace_sdk_producer_methods() -> None:
    subscription = InterestSubscription()

    def caller_code(*_args, **_kwargs):
        raise AssertionError("producer executed caller-controlled method")

    for method_name in ("_bind", "_publish", "_finish"):
        with pytest.raises(AttributeError):
            setattr(subscription, method_name, caller_code)

    events_task = asyncio.create_task(_consume_all(subscription))
    await _drain_run(
        _core("watch"),
        [_envelope(0, BLACK), _envelope(1, WHITE)],
        subscription,
    )
    events = await events_task
    assert [event.kind for event in events] == [
        "interest.entered",
        "interest.exited",
    ]

    from percept.runtime import worker

    source = __import__("inspect").getsource(worker)
    assert "InterestSubscription._bind(" in source
    assert "InterestSubscription._publish(" in source
    assert "InterestSubscription._finish(" in source


@pytest.mark.asyncio
async def test_cross_source_change_is_rejected_before_gate_diff() -> None:
    with pytest.raises(ValueError, match="one video source_id"):
        async for _ in run(
            _core("watch"),
            source=[
                _envelope(0, BLACK, source_id="video:first"),
                _envelope(0, WHITE, source_id="video:second"),
            ],
            judge_dispatch="inline",
        ):
            pass
