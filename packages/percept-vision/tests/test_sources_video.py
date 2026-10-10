"""File sources through OpenCV, on a video generated at test time. Skipped without the extra."""

import asyncio

import numpy as np
import pytest

from percept.sources import open_source

cv2 = pytest.importorskip("cv2")

W, H, FPS, N = 1280, 720, 10.0, 20


@pytest.fixture(scope="module")
def video(tmp_path_factory):
    path = tmp_path_factory.mktemp("video") / "synthetic.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), FPS, (W, H))
    assert writer.isOpened()
    for i in range(N):
        image = np.full((H, W, 3), (i * 10) % 255, np.uint8)
        cv2.putText(image, str(i), (100, 300), cv2.FONT_HERSHEY_SIMPLEX, 5, (255, 255, 255), 8)
        writer.write(image)
    writer.release()
    return str(path)


async def collect(source, until=None):
    out = []
    async for frame in source.frames():
        out.append(frame)
        if until and until(out):
            break
    await source.aclose()
    return out


def test_file_source_reads_every_frame_downscaled(video):
    src = open_source(video, stream_id="clip")
    frames = asyncio.run(collect(src))
    assert len(frames) == N
    assert [f.seq for f in frames] == list(range(N))
    assert {f.stream_epoch for f in frames} == {0} and {f.stream_id for f in frames} == {"clip"}
    assert all(f.image.shape == (360, 640, 3) for f in frames)
    assert np.all(np.diff([f.t for f in frames]) > 0)
    assert np.allclose(np.diff([f.t_wall for f in frames]), 1 / FPS)
    assert src.link == "ended"


def test_file_source_fps_cap(video):
    frames = asyncio.run(collect(open_source(video, fps=5)))
    assert len(frames) == N // 2
    assert np.allclose(np.diff([f.t_wall for f in frames]), 0.2)


def test_looping_file_bumps_epoch_and_resets_seq(video):
    src = open_source(video, loop=True, max_side=320)
    frames = asyncio.run(collect(src, until=lambda out: out[-1].stream_epoch == 2))
    assert [f.seq for f in frames[:N + 1]] == list(range(N)) + [0]
    assert frames[N].stream_epoch == 1
    assert all(max(f.image.shape[:2]) <= 320 for f in frames)
    assert np.all(np.diff([f.t for f in frames]) >= 0)
