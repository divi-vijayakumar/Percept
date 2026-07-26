from __future__ import annotations

import asyncio
import threading
import time

import pytest

from percept import Verdict
from percept.backends.judge_pool import JudgeCapability, JudgePool, JudgeRoute
from percept.runtime.video_clip import NativeVideoVision, encode_mp4_clip


class _RecordingVision:
    def __init__(self) -> None:
        self.calls = []

    async def judge(self, condition, frame, **kwargs):
        self.calls.append((condition, frame, kwargs))
        return Verdict(
            satisfied=True,
            confidence=0.96,
            evidence="native temporal evidence",
        )


@pytest.mark.asyncio
async def test_event_chain_assembles_native_clip_inside_bounded_route():
    recording = _RecordingVision()
    encoder_calls = []

    def encoder(frames, **kwargs):
        encoder_calls.append((frames, kwargs))
        return b"encoded-mp4"

    native = NativeVideoVision(
        recording,
        fps=5.0,
        max_frames=8,
        encoder=encoder,
    )
    pool = JudgePool(
        [
            JudgeRoute(
                "native-video",
                native,
                JudgeCapability(
                    evidence_shapes=frozenset({"video_clip"}),
                    latency_slo_s=0.05,
                    modes=frozenset({"reflex"}),
                ),
            )
        ],
        deadline_s=0.1,
    )

    verdict = await pool.judge(
        "one discrete action",
        b"current-jpeg",
        frames=[b"before-jpeg"],
        context={"event_chain": True},
    )

    assert verdict.evidence == "native temporal evidence"
    assert encoder_calls[0][0] == [b"before-jpeg", b"current-jpeg"]
    _, frame, kwargs = recording.calls[0]
    assert frame == ("video/mp4", b"encoded-mp4")
    assert kwargs["frames"] is None
    assert kwargs["context"]["_percept_evidence_shape"] == "video_clip"
    assert kwargs["context"]["_percept_clip_frame_count"] == 2
    assert pool.receipts()[-1].evidence_shape == "video_clip"


@pytest.mark.asyncio
async def test_event_chain_never_falls_back_to_ordered_images():
    strip = _RecordingVision()
    pool = JudgePool(
        [
            JudgeRoute(
                "strip",
                strip,
                JudgeCapability(
                    evidence_shapes=frozenset({"ordered_images"}),
                    latency_slo_s=0.05,
                    modes=frozenset({"reflex"}),
                ),
            )
        ],
        deadline_s=0.1,
    )

    verdict = await pool.judge(
        "one discrete action",
        b"current-jpeg",
        frames=[b"before-jpeg"],
        context={"event_chain": True},
    )

    assert verdict.confidence == 0.0
    assert strip.calls == []
    assert pool.receipts()[-1].outcome == "no_route"


@pytest.mark.asyncio
async def test_first_temporal_admission_abstains_without_spending_provider_call():
    recording = _RecordingVision()
    native = NativeVideoVision(
        recording,
        encoder=lambda *_args, **_kwargs: pytest.fail(
            "single-frame evidence must not invoke the encoder"
        ),
    )
    pool = JudgePool(
        [
            JudgeRoute(
                "native-video",
                native,
                JudgeCapability(
                    evidence_shapes=frozenset({"video_clip"}),
                    latency_slo_s=0.05,
                    modes=frozenset({"reflex"}),
                ),
            )
        ],
        deadline_s=0.1,
    )

    verdict = await pool.judge(
        "one discrete action",
        b"first-jpeg",
        context={"event_chain": True},
    )

    assert verdict.confidence == 0.0
    assert "at least two retained frames" in verdict.evidence
    assert recording.calls == []
    assert [receipt.outcome for receipt in pool.receipts()] == ["uncertain"]


@pytest.mark.asyncio
async def test_native_clip_preserves_actual_retained_source_span():
    recording = _RecordingVision()
    encoder_calls = []

    def encoder(frames, **kwargs):
        encoder_calls.append((frames, kwargs))
        return b"encoded-mp4"

    native = NativeVideoVision(
        recording,
        fps=6.0,
        max_frames=5,
        encoder=encoder,
    )
    await native.judge(
        "one discrete action",
        b"f4",
        frames=[b"f0", b"f1", b"f2", b"f3"],
        context={
            "event_chain": True,
            "_percept_temporal_span_s": 2.0,
        },
    )

    assert encoder_calls[0][1]["fps"] == 2.0
    assert (
        recording.calls[0][2]["context"]["_percept_clip_fps"]
        == 2.0
    )


def test_mp4_encoder_produces_decodable_bounded_native_video():
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    encoded_frames = []
    for index in range(4):
        image = np.full((24, 32, 3), index * 60, dtype=np.uint8)
        ok, payload = cv2.imencode(".jpg", image)
        assert ok
        encoded_frames.append(("image/jpeg", payload.tobytes()))

    clip = encode_mp4_clip(
        encoded_frames,
        fps=4.0,
        max_frames=4,
        max_dimension=64,
    )
    assert clip

    # OpenCV VideoCapture does not portably accept bytes, so validate the
    # container mechanics through a temporary file without retaining it.
    import tempfile
    from pathlib import Path

    handle = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    path = Path(handle.name)
    try:
        handle.write(clip)
        handle.close()
        capture = cv2.VideoCapture(str(path))
        count = 0
        while True:
            ok, _ = capture.read()
            if not ok:
                break
            count += 1
        capture.release()
    finally:
        path.unlink(missing_ok=True)

    assert count == 4


@pytest.mark.asyncio
async def test_stalled_encoder_does_not_block_deadline_or_queue_more_encodes():
    started = threading.Event()
    release = threading.Event()
    encode_calls = 0

    def stalled_encoder(_frames, **_kwargs):
        nonlocal encode_calls
        encode_calls += 1
        started.set()
        release.wait(timeout=2.0)
        return b"encoded-after-release"

    native = NativeVideoVision(
        _RecordingVision(),
        encoder=stalled_encoder,
    )
    pool = JudgePool(
        [
            JudgeRoute(
                "native-video",
                native,
                JudgeCapability(
                    evidence_shapes=frozenset({"video_clip"}),
                    latency_slo_s=0.02,
                    modes=frozenset({"reflex"}),
                ),
            )
        ],
        deadline_s=0.03,
    )
    heartbeat = 0
    stop = asyncio.Event()

    async def beat():
        nonlocal heartbeat
        while not stop.is_set():
            heartbeat += 1
            await asyncio.sleep(0.002)

    beater = asyncio.create_task(beat())
    started_at = time.monotonic()
    verdict = await pool.judge(
        "one discrete action",
        b"current",
        frames=[b"before"],
        context={"event_chain": True},
    )
    elapsed = time.monotonic() - started_at

    assert started.wait(timeout=0.1)
    assert verdict.confidence == 0.0
    assert elapsed < 0.15
    assert heartbeat > 2

    second = await pool.judge(
        "one discrete action",
        b"current-2",
        frames=[b"before-2"],
        context={"event_chain": True},
    )
    assert second.confidence == 0.0
    assert encode_calls == 1
    assert any(
        receipt.outcome == "provider_error"
        for receipt in pool.receipts()
    )

    release.set()
    stop.set()
    await beater
    await asyncio.sleep(0.02)
