from __future__ import annotations

import asyncio

import pytest

from percept import Goal, Percept, Verdict
from percept.backends.judge_pool import JudgeCapability, JudgePool, JudgeRoute
from percept.harness.tier0 import Frame, gray_frame
from percept.runtime import NativeVideoVision
from percept.runtime.worker import run as worker_run


def _motion() -> Frame:
    return Frame(8, 8, tuple([255] * 40 + [0] * 24))


@pytest.mark.asyncio
async def test_vid_admission_retains_nonadmitted_preroll_for_native_clip():
    class RecordingJudge:
        SUPPORTS_NATIVE_VIDEO_TRANSPORT = True

        def __init__(self) -> None:
            self.frames = []

        async def judge(self, _condition, frame, **_kwargs):
            self.frames.append(frame)
            return Verdict(False, 0.95, "not yet")

    provider = RecordingJudge()
    pool = JudgePool(
        [
            JudgeRoute(
                "native",
                NativeVideoVision(
                    provider,
                    encoder=lambda frames, **_kwargs: (
                        b"mp4:" + str(len(frames)).encode("ascii")
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
        yield 0.0, gray_frame(8, 8, 0), b"frame-0"
        yield 1.0, _motion(), b"frame-1"
        for _ in range(100):
            if provider.frames:
                break
            await asyncio.sleep(0.001)

    traces = []
    assert [
        incident
        async for incident in worker_run(
            core,
            source=source(),
            judge_dispatch="async",
            judge_decision_deadline_s=0.8,
            on_trace=traces.append,
        )
    ] == []
    assert provider.frames == [("video/mp4", b"mp4:2")], [
        (trace.stage, trace.data) for trace in traces
    ]
