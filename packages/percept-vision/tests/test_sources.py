"""Sources without cameras or network: fake readers, PushSource, ffmpeg plumbing."""

import asyncio
import io
import shutil
import subprocess
import sys
import threading
import time

import numpy as np
import pytest

import percept
from percept.clips import PreRoll
from percept.contracts import Source
from percept.sources import (
    CaptureSource,
    FfmpegReader,
    PushSource,
    Reader,
    downscale,
    ffmpeg_command,
    open_source,
    read_ppm,
)

IMG = np.zeros((720, 1280, 3), np.uint8)


class ScriptedReader(Reader):
    """Delivers ``n`` frames (media time None: live), then fails the read."""

    def __init__(self, n: int, media_fps: float | None = None, image=IMG):
        self.n, self.i, self.media_fps, self.image = n, 0, media_fps, image

    def read(self):
        if self.i >= self.n:
            if self.media_fps is not None:
                return None  # clean EOF
            raise ConnectionError("read failed")
        self.i += 1
        media = None if self.media_fps is None else (self.i - 1) / self.media_fps
        return self.image, media


def scripted(script, source_box):
    """An opener walking ``script``: "fail" raises, an int delivers that many frames."""
    steps = iter(script)

    def opener():
        step = next(steps, None)
        if step is None:
            source_box[0]._stop()
            raise ConnectionError("script done")
        if step == "fail":
            raise ConnectionError("camera unreachable")
        return ScriptedReader(step)

    return opener


async def collect(source, limit=None):
    out = []
    async for frame in source.frames():
        out.append(frame)
        if limit and len(out) >= limit:
            break
    await source.aclose()
    return out


def test_reconnect_backoff_epochs_and_health():
    box, health, captured, delays = [None], [], [], []
    src = CaptureSource("cam", scripted(["fail", "fail", 3, "fail", "fail", "fail", "fail", "fail", 2], box),
                        live=True, jitter=0.0, on_health=health.append, on_frame=captured.append,
                        sleep=delays.append)
    box[0] = src
    assert isinstance(src, Source)
    asyncio.run(collect(src))  # frames() ends without raising
    assert delays == [2, 4, 2, 4, 8, 16, 30, 30, 2]  # the last connection drops too
    assert [(f.stream_epoch, f.seq) for f in captured] == [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1)]
    ts = [f.t for f in captured]
    assert ts == sorted(ts)
    assert [h.kind for h in health] == ["link_down", "link_up", "link_down", "link_up", "link_down"]
    assert "unreachable" in health[0].detail and health[1].detail == "epoch 0"
    assert src.link == "ended" and src.epoch == 1 and src.reconnects == 1
    assert all(f.image.shape == (360, 640, 3) for f in captured)


def test_backoff_jitter_stays_in_band():
    box, delays = [None], []
    src = CaptureSource("cam", scripted(["fail"] * 8, box), live=True, jitter=0.1, sleep=delays.append)
    box[0] = src
    asyncio.run(collect(src))
    nominal = [2, 4, 8, 16, 30, 30, 30, 30]
    assert all(0.9 * n <= d <= 1.1 * n for d, n in zip(delays, nominal))


def test_starting_epoch_is_configurable():
    box, captured = [None], []
    src = CaptureSource("cam", scripted([1, 1], box), live=True, epoch=7, on_frame=captured.append,
                        sleep=lambda s: None)
    box[0] = src
    asyncio.run(collect(src))
    assert [f.stream_epoch for f in captured] == [7, 8]


def test_slow_consumer_gets_latest_and_capture_is_not_blocked():
    done = threading.Event()

    class Fast(Reader):
        i = 0

        def read(self):
            if self.i == 200:
                done.set()
                time.sleep(0.01)
                return None
            self.i += 1
            return IMG[:8, :8], None

    src = CaptureSource("cam", Fast, live=True, sleep=lambda s: None)

    async def run():
        seen = []
        async for frame in src.frames():
            seen.append(frame.seq)
            assert len(src._handoff.items) <= 1
            await asyncio.sleep(0.02)
            if done.is_set() and len(seen) > 1:
                break
        await src.aclose()
        return seen

    t0 = time.monotonic()
    seen = asyncio.run(run())
    assert done.is_set() and time.monotonic() - t0 < 5
    assert len(seen) < 100 and src.dropped > 100
    assert seen == sorted(seen)


def test_file_like_source_is_lossless_and_fps_capped():
    src = CaptureSource("file", lambda: ScriptedReader(50, media_fps=25.0), live=False, fps=5.0,
                        wall=lambda: 1_000.0)
    frames = asyncio.run(collect(src))
    assert len(frames) == 10
    assert [f.seq for f in frames] == list(range(10))
    assert np.allclose(np.diff([f.t_wall for f in frames]), 0.2)
    assert frames[0].t_wall == 1_000.0 and src.link == "ended"


def test_file_that_fails_to_open_ends_with_health():
    health = []

    def opener():
        raise OSError("corrupt")

    src = CaptureSource("file", opener, live=False, on_health=health.append)
    assert asyncio.run(collect(src)) == []
    assert [h.kind for h in health] == ["link_down"]


def test_aclose_unblocks_a_blocked_read_promptly():
    released = threading.Event()

    class Blocking(Reader):
        def read(self):
            released.wait(10)
            return None

        def abort(self):
            released.set()

    src = CaptureSource("cam", Blocking, live=True)

    async def run():
        it = src.frames().__aiter__()
        task = asyncio.ensure_future(it.__anext__())
        await asyncio.sleep(0.05)
        t0 = time.monotonic()
        await src.aclose()
        elapsed = time.monotonic() - t0
        with pytest.raises(StopAsyncIteration):
            await task
        return elapsed

    assert asyncio.run(run()) < 2
    assert not src._thread.is_alive()


def test_preroll_fed_at_capture_rate_and_clip_is_threadsafe():
    ring = PreRoll(seconds=8, fps=2)
    src = CaptureSource("file", lambda: ScriptedReader(100, media_fps=10.0), live=False, preroll=ring,
                        wall=lambda: 5_000.0)
    asyncio.run(collect(src))
    assert len(ring) == 17  # 8 s at 2 fps, both ends
    clip = src.clip((5_005.0, 5_009.9), n=4)
    assert [f.t_wall for f in clip] == [5_005.0, 5_006.5, 5_008.0, 5_009.5]
    assert src.keyframe((5_005.0, 5_009.9)).t_wall == 5_009.5


# --------------------------------------------------------------------------- PushSource


def test_push_source_bounded_drop_oldest_monotonic():
    src = PushSource("app", maxlen=8)
    for i in range(20):
        src.push(np.full((100, 2000), i, np.uint8), t_wall=100.0 + i)
    src.close()
    frames = asyncio.run(collect(src))
    assert [f.seq for f in frames] == list(range(12, 20))
    assert src.dropped == 12
    assert all(f.image.ndim == 2 and max(f.image.shape) <= 640 for f in frames)
    assert [f.t for f in frames] == sorted(f.t for f in frames)
    assert frames[0].t_wall == 112.0
    with pytest.raises(RuntimeError):
        src.push(IMG)


def test_push_source_from_another_thread():
    src = PushSource()

    def producer():
        for _ in range(30):
            src.push(IMG[:10, :10])
            time.sleep(0.001)
        src.close()

    async def run():
        threading.Thread(target=producer).start()
        return await collect(src)

    frames = asyncio.run(run())
    assert frames and [f.seq for f in frames] == sorted({f.seq for f in frames})


# --------------------------------------------------------------------------- downscale and imports


def test_downscale_without_opencv(monkeypatch):
    monkeypatch.setitem(sys.modules, "cv2", None)
    image = np.arange(1080 * 1920, dtype=np.uint32).reshape(1080, 1920).astype(np.uint8)
    small = downscale(image, 640)
    assert max(small.shape) <= 640 and small.dtype == np.uint8
    assert downscale(IMG[:100, :100], 640) is not None


def test_import_percept_does_not_import_cv2():
    code = "import sys, percept; assert 'cv2' not in sys.modules; assert percept.open_source"
    subprocess.run([sys.executable, "-c", code], check=True)
    assert "open_source" in percept.__all__ and "PushSource" in percept.__all__


def test_missing_opencv_is_a_clear_error(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "cv2", None)
    path = tmp_path / "v.mp4"
    path.write_bytes(b"")
    with pytest.raises(ImportError, match=r"percept-vision\[video\]"):
        open_source(str(path))
    with pytest.raises(FileNotFoundError):
        open_source(str(tmp_path / "missing.mp4"))


def test_missing_ffmpeg_is_a_clear_error():
    with pytest.raises(RuntimeError, match="ffmpeg"):
        open_source("https://example.com/live/index.m3u8", ffmpeg="no-such-ffmpeg-binary")


# --------------------------------------------------------------------------- ffmpeg


def test_ffmpeg_command_for_hls_and_files():
    hls = ffmpeg_command("https://cdn.example.com/live/index.m3u8?token=x", fps=2)
    assert hls[hls.index("-live_start_index") + 1] == "-1"
    assert "-reconnect" in hls and "-rw_timeout" in hls
    assert hls.index("-live_start_index") < hls.index("-i")
    assert hls[hls.index("-vf") + 1].startswith("fps=2,scale=")
    assert hls[-4:] == ["image2pipe", "-vcodec", "ppm", "-"]
    dash = ffmpeg_command("https://cdn.example.com/stream.mpd")
    assert "-live_start_index" not in dash and "-reconnect" in dash
    local = ffmpeg_command("/videos/clip.mp4", max_side=320)
    assert "-reconnect" not in local and "min(320,iw)" in local[local.index("-vf") + 1]


def ppm(image_rgb: np.ndarray) -> bytes:
    h, w = image_rgb.shape[:2]
    return f"P6\n{w} {h}\n255\n".encode() + image_rgb.tobytes()


def test_read_ppm_parses_frames_as_bgr():
    rgb = np.zeros((2, 3, 3), np.uint8)
    rgb[..., 0] = 200  # red
    stream = io.BytesIO(ppm(rgb) + ppm(rgb[:1]) + ppm(rgb)[:-4])
    first = read_ppm(stream)
    assert first.shape == (2, 3, 3) and (first[..., 2] == 200).all() and (first[..., 0] == 0).all()
    assert read_ppm(stream).shape == (1, 3, 3)
    assert read_ppm(stream) is None  # torn frame at the end
    assert read_ppm(io.BytesIO(b"")) is None
    with pytest.raises(ValueError):
        read_ppm(io.BytesIO(b"P5\n1 1\n255\n\x00"))


def test_ffmpeg_reader_subprocess_is_killed_on_aclose():
    frame = ppm(np.zeros((4, 4, 3), np.uint8))
    script = (f"import sys, time; sys.stdout.buffer.write({frame!r} * 3); sys.stdout.flush(); time.sleep(60)")
    readers = []

    def opener():
        readers.append(FfmpegReader([sys.executable, "-c", script]))
        return readers[-1]

    src = CaptureSource("hls", opener, live=True)

    async def run():
        n = 0
        async for _ in src.frames():
            n += 1
            if n == 1:
                break
        t0 = time.monotonic()
        await src.aclose()
        return time.monotonic() - t0

    assert asyncio.run(run()) < 3
    assert readers[0].proc.poll() is not None


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is not installed")
def test_real_ffmpeg_reads_a_generated_file(tmp_path):
    path = tmp_path / "test.mp4"
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
                    "testsrc=size=1280x720:rate=10", "-t", "2", "-pix_fmt", "yuv420p", str(path)], check=True)
    src = open_source(str(path), backend="ffmpeg", fps=5)
    frames = asyncio.run(collect(src))
    assert 8 <= len(frames) <= 11
    assert all(f.image.shape == (360, 640, 3) for f in frames)
    assert [f.seq for f in frames] == list(range(len(frames)))
    assert np.allclose(np.diff([f.t_wall for f in frames]), 0.2)  # media time, not decode speed
    assert src.link == "ended"
    ring = PreRoll(seconds=8, fps=2)
    frames = asyncio.run(collect(open_source("file://" + str(path), backend="ffmpeg", preroll=ring)))
    assert len(frames) == 20 and np.allclose(np.diff([f.t_wall for f in frames]), 0.1)  # ffprobe rate
    assert 4 <= len(ring) <= 5  # 2 s of video in half-second slots


# --------------------------------------------------------------------------- review regressions

FAKE_FFMPEG = '''#!{exe}
import os, sys, time
url = sys.argv[sys.argv.index("-i") + 1]
frame = b"P6\\n4 4\\n255\\n" + bytes(48)
sys.stdout.buffer.write(frame * int(os.environ.get("FAKE_FRAMES", "0")))
sys.stdout.flush()
if os.environ.get("FAKE_STREAM"):  # a healthy 20 fps stream, honouring an -vf fps= cap like ffmpeg
    vf = sys.argv[sys.argv.index("-vf") + 1]
    rate = float(vf.split(",")[0][4:]) if vf.startswith("fps=") else 20.0
    while True:
        sys.stdout.buffer.write(frame)
        sys.stdout.flush()
        time.sleep(1 / rate)
sys.stderr.write("Connection refused while opening " + url + "\\n")
sys.stderr.flush()
time.sleep(float(os.environ.get("FAKE_SLEEP", "0")))
sys.exit(int(os.environ.get("FAKE_CODE", "0")))
'''


@pytest.fixture
def fake_ffmpeg(tmp_path):
    path = tmp_path / "ffmpeg"
    path.write_text(FAKE_FFMPEG.format(exe=sys.executable))
    path.chmod(0o755)
    return str(path)


async def until_health(src, predicate, limit_s=8.0):
    """Drive frames() until ``predicate(frames, health)`` holds, then close."""
    frames, health = [], []
    src.on_health = health.append

    async def consume():
        async for frame in src.frames():
            frames.append(frame)

    task = asyncio.ensure_future(consume())
    t0 = time.monotonic()
    while not predicate(frames, health) and time.monotonic() - t0 < limit_s and not task.done():
        await asyncio.sleep(0.02)
    await src.aclose()
    await asyncio.wait_for(task, 5)
    return frames, health


@pytest.mark.parametrize("uri, live", [
    ("http://cam.local/video.mjpg", True),
    ("https://cam.local/live/stream", True),
    ("rtsp://cam.local/stream", True),
    ("http://media.local/clip.mp4", False),
    ("https://media.local/a/b.MKV?sig=1", False),
])
def test_url_liveness_classification(uri, live, fake_ffmpeg):
    assert open_source(uri, backend="ffmpeg", ffmpeg=fake_ffmpeg).live is live


def test_live_override(fake_ffmpeg):
    assert open_source("http://media.local/clip.mp4", live=True, backend="ffmpeg", ffmpeg=fake_ffmpeg).live
    assert not open_source("http://cam.local/video.mjpg", live=False, backend="ffmpeg", ffmpeg=fake_ffmpeg).live


def test_http_camera_outage_reconnects_instead_of_ending(fake_ffmpeg, monkeypatch):
    monkeypatch.setenv("FAKE_FRAMES", "2")
    src = open_source("http://cam.local/video.mjpg", backend="ffmpeg", ffmpeg=fake_ffmpeg, backoff=(0.05, 0.05))
    frames, health = asyncio.run(until_health(src, lambda f, h: any(x.stream_epoch >= 1 for x in f)))
    assert any(f.stream_epoch >= 1 for f in frames)
    assert "link_down" in [h.kind for h in health]


def test_stalled_live_stream_is_torn_down_and_retried(fake_ffmpeg, monkeypatch):
    monkeypatch.setenv("FAKE_SLEEP", "60")  # alive, answering nothing: a playlist that never grows
    src = open_source("https://cdn.local/live/index.m3u8", ffmpeg=fake_ffmpeg, stall_timeout=0.3,
                      backoff=(0.05, 0.05))
    t0 = time.monotonic()
    _, health = asyncio.run(until_health(src, lambda f, h: len(src.delays) >= 2))
    assert time.monotonic() - t0 < 6
    assert health and health[0].kind == "link_down" and "stalled" in health[0].detail
    assert len(src.delays) >= 2


def test_stall_watchdog_aborts_a_blocked_reader():
    aborted = []

    class Hung(Reader):
        def __init__(self):
            self.gate = threading.Event()

        def read(self):
            self.gate.wait(30)
            return None

        def abort(self):
            aborted.append(1)
            self.gate.set()

    src = CaptureSource("cam", Hung, live=True, stall_timeout=0.2, backoff=(0.05, 0.05))
    _, health = asyncio.run(until_health(src, lambda f, h: bool(h)))
    assert aborted and "stalled" in health[0].detail


def test_credentials_never_leave_the_uri(fake_ffmpeg, monkeypatch):
    monkeypatch.setenv("FAKE_FRAMES", "1")
    monkeypatch.setenv("FAKE_CODE", "1")
    uri = "rtsp://admin:s3cret@127.0.0.1:1/stream?token=t0k"
    src = open_source(uri, backend="ffmpeg", ffmpeg=fake_ffmpeg, backoff=(0.05, 0.05))
    assert src.stream_id == "rtsp://127.0.0.1:1/stream"
    names = []
    frames, health = asyncio.run(until_health(
        src, lambda f, h: names.extend(t.name for t in threading.enumerate()) or len(h) >= 1))
    assert frames and health and "Connection refused" in health[0].detail
    seen = [src.stream_id, repr(src), str(src), *names, *(repr(f) for f in frames), *(repr(h) for h in health)]
    assert not [s for s in seen if "s3cret" in s or "t0k" in s]
    with pytest.raises(RuntimeError) as err:
        open_source(uri, ffmpeg="no-such-ffmpeg-binary", backend="ffmpeg")
    assert "s3cret" not in str(err.value)


def test_backoff_and_jitter_are_validated():
    for kwargs in (dict(jitter=1.5), dict(jitter=-0.1), dict(backoff=(-1.0, 2.0)), dict(backoff=(3.0, 2.0)),
                   dict(backoff=(float("nan"), 2.0)), dict(stall_timeout=0)):
        with pytest.raises(ValueError):
            CaptureSource("cam", Reader, live=True, **kwargs)


def test_loop_carries_t_wall_forward_and_keeps_the_ring():
    ring = PreRoll(seconds=8, fps=2)
    src = CaptureSource("file", lambda: ScriptedReader(20, media_fps=10.0), live=False, loop=True,
                        preroll=ring, wall=lambda: 1_000.0)

    async def run():
        out = []
        async for frame in src.frames():
            out.append(frame)
            if frame.stream_epoch == 2:
                break
        await src.aclose()
        return out

    frames = asyncio.run(run())
    walls = [f.t_wall for f in frames]
    assert np.all(np.diff(walls) > 0)
    assert np.allclose(np.diff(walls), 0.1)
    assert len(ring) > 8


def test_epoch_is_none_before_the_first_frame():
    assert CaptureSource("cam", Reader, live=True).epoch is None


def test_read_ppm_rejects_oversized_frames():
    with pytest.raises(ValueError):
        read_ppm(io.BytesIO(b"P6\n4000 4000\n255\n"), max_pixels=640 * 640)
    with pytest.raises(ValueError):
        read_ppm(io.BytesIO(b"P6\n0 4\n255\n"))


def test_low_fps_live_ffmpeg_is_not_mistaken_for_a_stall(fake_ffmpeg, monkeypatch):
    monkeypatch.setenv("FAKE_STREAM", "1")
    src = open_source("https://cdn.local/live/index.m3u8", ffmpeg=fake_ffmpeg, fps=0.5, stall_timeout=0.3,
                      backoff=(0.05, 0.05))
    t0 = time.monotonic()
    frames, health = asyncio.run(until_health(src, lambda f, h: time.monotonic() - t0 > 2.5))
    assert health == [] and frames and {f.stream_epoch for f in frames} == {0}


def test_live_ffmpeg_command_caps_fps_reader_side(fake_ffmpeg):
    live = open_source("https://cdn.local/live/index.m3u8", ffmpeg=fake_ffmpeg, fps=2)
    assert live.fps == 2
    reader = live.opener()
    try:
        assert not reader.proc.args[reader.proc.args.index("-vf") + 1].startswith("fps=")
    finally:
        reader.close()


def test_stall_limit_is_at_least_two_frame_periods():
    assert CaptureSource("cam", Reader, live=True, fps=1.0, stall_timeout=0.3).stall_limit == 2.0
    assert CaptureSource("cam", Reader, live=True, fps=10.0, stall_timeout=30).stall_limit == 30
    assert CaptureSource("cam", Reader, live=True, stall_timeout=None).stall_limit is None

    class Slow(Reader):  # a healthy 1 fps camera
        def read(self):
            time.sleep(0.5)
            return IMG[:8, :8], None

    src = CaptureSource("cam", Slow, live=True, fps=1.0, stall_timeout=0.3)
    t0 = time.monotonic()
    frames, health = asyncio.run(until_health(src, lambda f, h: time.monotonic() - t0 > 1.6))
    assert health == [] and len(frames) >= 1


def test_stall_check_ignores_a_reader_that_is_no_longer_current():
    aborted = []

    class R(Reader):
        def abort(self):
            aborted.append(self)

    src = CaptureSource("cam", Reader, live=True, stall_timeout=1.0, clock=lambda: 100.0)
    old, new = R(), R()
    src._reader, src._progress = new, 0.0  # a fresh connection replaced `old` after it was sampled
    assert not src._check_stall(old)
    assert not src._stalled and aborted == []
    assert src._check_stall(new) and src._stalled and aborted == [new]
