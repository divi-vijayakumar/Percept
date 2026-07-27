from __future__ import annotations

import asyncio

import pytest

from percept import Goal, Percept, Verdict
from percept.backends.anthropic import _parse_verdict
from percept.backends.judge_pool import JudgeCapability, JudgePool, JudgeRoute
from percept.contracts import ClipEvent, judgment_event_id
from percept.harness.tier0 import Frame, gray_frame
from percept.runtime import NativeVideoVision
from percept.runtime.worker import run as worker_run


def _motion(index: int) -> Frame:
    pixels = [0] * 64
    start = 0 if index % 2 else 24
    for offset in range(40):
        pixels[(start + offset) % 64] = 255
    return Frame(8, 8, tuple(pixels))


def test_parser_accepts_ordered_bounded_clip_events():
    verdict = _parse_verdict(
        '{"satisfied": true, "confidence": 0.96, "evidence": "three", '
        '"phase": "post", "action_present": false, "events": ['
        '{"t_start": 0.1, "t_end": 0.2, "label": "first"},'
        '{"t_start": 0.4, "t_end": 0.5, "label": "second"},'
        '{"t_start": 0.7, "t_end": 0.8, "label": "third"}]}'
    )

    assert [(event.t_start, event.t_end) for event in verdict.events] == [
        (0.1, 0.2),
        (0.4, 0.5),
        (0.7, 0.8),
    ]


def test_parser_fails_closed_on_unordered_clip_events():
    verdict = _parse_verdict(
        '{"satisfied": true, "confidence": 0.96, "evidence": "bad", '
        '"phase": "post", "action_present": false, "events": ['
        '{"t_start": 0.7, "t_end": 0.8},'
        '{"t_start": 0.4, "t_end": 0.5}]}'
    )

    assert verdict.confidence == 0.0
    assert verdict.events == []


def test_occurrence_evidence_link_discriminates_one_dispatch_without_renumbering():
    ordinary = judgment_event_id("goal", None, 7)
    first = judgment_event_id("goal", None, 7, "episode#1")
    second = judgment_event_id("goal", None, 7, "episode#2")

    assert ordinary == "goal##7"
    assert first == "goal##7#episode%231"
    assert len({ordinary, first, second}) == 3


@pytest.mark.asyncio
async def test_async_worker_emits_every_event_reported_in_one_window():
    class ScriptedNativeJudge:
        SUPPORTS_NATIVE_VIDEO_TRANSPORT = True

        def __init__(self) -> None:
            self.calls = 0
            self.frames = []

        async def judge(self, _condition, frame, **_kwargs):
            self.calls += 1
            self.frames.append(frame)
            events = (
                [
                    ClipEvent(0.1, 0.2, "first"),
                    ClipEvent(0.4, 0.5, "second"),
                    ClipEvent(0.7, 0.8, "third"),
                ]
                if self.calls == 1
                else [
                    ClipEvent(0.1, 0.2, "first"),
                    ClipEvent(0.4, 0.5, "second"),
                    ClipEvent(0.7, 0.8, "third"),
                    ClipEvent(1.0, 1.1, "fourth"),
                ]
            )
            return Verdict(
                satisfied=True,
                confidence=0.96,
                evidence="bounded temporal occurrences",
                phase="post",
                action_present=False,
                events=events,
            )

    provider = ScriptedNativeJudge()
    native = NativeVideoVision(
        provider,
        encoder=lambda frames, **_kwargs: (
            b"mp4:" + bytes(str(len(frames)), "ascii")
        ),
        require_native_transport=True,
    )
    pool = JudgePool(
        [
            JudgeRoute(
                "native",
                native,
                JudgeCapability(
                    evidence_shapes=frozenset({"video_clip"}),
                    latency_slo_s=0.2,
                    modes=frozenset({"reflex"}),
                    max_inflight=1,
                ),
            )
        ],
        deadline_s=0.3,
        max_parallel=1,
    )
    core = Percept.create(vision=pool, discover_plugins=False)
    core.add_goal(
        Goal(
            id="action",
            condition="someone performs the action",
            say="action",
            state={
                "type": "watch",
                "rearm": "each_event",
                "salience_stat": "block_max",
                "perception": {
                    "temporal": "strip",
                    "n_frames": 5,
                    "sampling": "recent",
                    "window_s": 2.0,
                },
            },
        )
    )

    async def source():
        for index, source_t in enumerate((0.0, 1.0, 2.0)):
            yield source_t, (
                _motion(1) if index == 1 else gray_frame(8, 8, 0)
            ), f"frame-{index}".encode()
            if index == 1:
                for _ in range(100):
                    if provider.calls >= 1:
                        break
                    await asyncio.sleep(0.001)
                await asyncio.sleep(0.02)

    traces = []
    incidents = [
        incident
        async for incident in worker_run(
            core,
            source=source(),
            judge_dispatch="async",
            judge_decision_deadline_s=0.8,
            on_trace=traces.append,
        )
    ]

    assert provider.calls == 2
    # Every event the FIRST window reported is emitted: one look enumerating three separate
    # occurrences 0.3s apart is the provider's explicit testimony that they ARE separate, and the
    # delivery merge never applies within a single enumeration (behavior.INCIDENT_MERGE_WINDOW_S).
    # The SECOND window's "fourth" is a CROSS-judgment delivery landing 0.3s after the third
    # incident for the same watch — inside the merge window, so it is dropped as a re-report.
    assert len(incidents) == 3
    assert len({incident.revision_id for incident in incidents}) == 3
    assert [incident.ts for incident in incidents] == pytest.approx(
        [0.2, 0.5, 0.8]
    )
    incident_links = [incident.judgment_event_id for incident in incidents]
    assert len(set(incident_links)) == 3
    judgment_links = [
        trace.data["judgment_event_id"]
        for trace in traces
        if trace.stage == "judgment.event"
    ]
    assert judgment_links == incident_links
    assert all(frame[0] == "video/mp4" for frame in provider.frames)


@pytest.mark.asyncio
async def test_public_manual_path_deduplicates_overlapping_windows():
    class OverlapJudge:
        SUPPORTS_NATIVE_VIDEO_TRANSPORT = True

        def __init__(self) -> None:
            self.calls = 0

        async def judge(self, _condition, _frame, **_kwargs):
            self.calls += 1
            events = (
                [
                    ClipEvent(0.1, 0.2, "first"),
                    ClipEvent(0.4, 0.5, "second"),
                    ClipEvent(0.7, 0.8, "third"),
                ]
                if self.calls == 1
                else [
                    ClipEvent(0.1, 0.2, "first"),
                    ClipEvent(0.4, 0.5, "second"),
                    ClipEvent(0.7, 0.8, "third"),
                    ClipEvent(1.1, 1.2, "fourth"),
                ]
            )
            return Verdict(
                True,
                0.96,
                "overlapping temporal window",
                phase="post",
                action_present=False,
                events=events,
            )

    provider = OverlapJudge()
    pool = JudgePool(
        [
            JudgeRoute(
                "native",
                NativeVideoVision(
                    provider,
                    encoder=lambda frames, **_kwargs: (
                        b"mp4:" + bytes(str(len(frames)), "ascii")
                    ),
                    require_native_transport=True,
                ),
                JudgeCapability(
                    evidence_shapes=frozenset({"video_clip"}),
                    latency_slo_s=0.2,
                    modes=frozenset({"reflex"}),
                ),
            )
        ],
        deadline_s=0.3,
        max_parallel=1,
    )
    core = Percept.create(vision=pool, discover_plugins=False)
    core.add_goal(
        Goal(
            id="action",
            condition="someone performs the action",
            say="action",
            state={
                "type": "watch",
                "rearm": "each_event",
                "perception": {
                    "temporal": "strip",
                    "n_frames": 5,
                    "sampling": "recent",
                    "window_s": 2.0,
                },
            },
        )
    )

    initial = await core.perceive_judged(b"f0", t=0.0)
    assert [event for event in initial if event.action == "fire"] == []
    first = await core.perceive_judged(b"f1", t=1.0)
    second = await core.perceive_judged(b"f2", t=2.0)

    assert provider.calls == 2
    assert len(first) == 3  # one look, three enumerated occurrences -> never merged against itself
    # The second window's three restatements are overlap-suppressed as before; its genuinely new
    # "fourth" lands 0.4s after the third delivery for the same watch, so the delivery merge
    # (behavior.INCIDENT_MERGE_WINDOW_S) drops it as a re-report rather than a second notification.
    assert len(second) == 0
    all_fires = [*first, *second]
    assert len({fire.revision_id for fire in all_fires}) == 3
    assert [
        fire.judgment.timing.source_media_ts for fire in all_fires
    ] == pytest.approx([0.2, 0.5, 0.8])
