"""Sources: where frames come from.

:func:`open_source` opens a camera, file or stream and returns a
:class:`~percept.contracts.Source`. :class:`PushSource` is for apps that already own capture.

Capture never waits for the consumer. A blocking reader runs on a background thread and hands
frames to the event loop through a one-slot "latest frame": when the consumer is slow the
older frame is replaced, never queued. Every frame is offered to the optional ``on_frame``
hook and :class:`~percept.clips.PreRoll` on the capture thread first, so the pre-roll ring
sees the stream at capture rate even when the consumer does not. Files are the exception: a
file has nothing to lose by waiting, so a file source hands over every sampled frame.

Live sources (RTSP, webcams, HLS/DASH) reconnect forever with jittered exponential backoff
(2 s doubling to 30 s by default). ``stream_epoch`` increases by one each time a connection
delivers its first frame, and ``seq`` restarts at 0 in every epoch. ``t`` is the process's
monotonic clock and ``t_wall`` Unix time, both read when the frame arrives; for a file both
advance with the video's own timeline instead, so a file reads the same at any decode speed.

Link changes are reported, never raised: ``frames()`` does not raise for a camera that is
down. They arrive as :class:`~percept.contracts.Health` (``link_down`` once per outage,
``link_up`` when frames flow again) through ``on_health``, called on the event loop thread,
and the current state is always readable as ``source.link`` (``"connecting"``, ``"up"``,
``"down"`` or ``"ended"``).

OpenCV is the ``video`` extra (``pip install "percept-vision[video]"``) and is imported only
when a source needs it. HLS and DASH are read by an ``ffmpeg`` subprocess, which must be on
``PATH``.
"""

from __future__ import annotations

import asyncio
import os
import random
import shutil
import subprocess
import threading
import time
from collections import deque
from typing import Any, AsyncIterator, Callable

import numpy as np

from percept.clips import Interval, PreRoll
from percept.contracts import Frame, Health

MANIFEST_SUFFIXES = (".m3u8", ".m3u", ".mpd")
_FFMPEG_ENV = "OPENCV_FFMPEG_CAPTURE_OPTIONS"
_ENV_LOCK = threading.Lock()


def _cv2() -> Any:
    try:
        import cv2
    except ImportError as exc:
        raise ImportError('this source needs OpenCV: pip install "percept-vision[video]"') from exc
    return cv2


def downscale(image: np.ndarray, max_side: int) -> np.ndarray:
    """Shrink ``image`` so its longest side is at most ``max_side``, by area averaging."""
    h, w = image.shape[:2]
    if max(h, w) <= max_side:
        return image
    scale = max_side / max(h, w)
    size = (max(1, int(w * scale)), max(1, int(h * scale)))
    try:
        import cv2
    except ImportError:  # integer block mean: the same area average, without OpenCV
        k = -(-max(h, w) // max_side)
        hh, ww = h // k * k, w // k * k
        blocks = image[:hh, :ww].reshape(hh // k, k, ww // k, k, *image.shape[2:])
        return blocks.mean(axis=(1, 3)).round().astype(np.uint8)
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


class _Handoff:
    """Thread-to-loop frame handoff: a bounded deque, drop-oldest or (``lossless``) blocking."""

    def __init__(self, capacity: int, lossless: bool = False) -> None:
        self.items: deque[Frame] = deque()
        self.capacity, self.lossless = capacity, lossless
        self.cond = threading.Condition()
        self.closed = False
        self.dropped = 0
        self.loop: asyncio.AbstractEventLoop | None = None
        self.ready: asyncio.Event | None = None

    def put(self, frame: Frame, stop: threading.Event | None = None) -> None:
        with self.cond:
            if self.lossless:
                while len(self.items) >= self.capacity and not self.closed and not (stop and stop.is_set()):
                    self.cond.wait(0.1)
            while len(self.items) >= self.capacity:
                self.items.popleft()
                self.dropped += 1
            self.items.append(frame)
        self.wake()

    def close(self) -> None:
        with self.cond:
            self.closed = True
            self.cond.notify_all()
        self.wake()

    def wake(self) -> None:
        if self.loop is not None and self.ready is not None:
            try:
                self.loop.call_soon_threadsafe(self.ready.set)
            except RuntimeError:  # the loop is closed: nobody is listening
                pass

    def bind(self) -> None:
        self.loop, self.ready = asyncio.get_running_loop(), asyncio.Event()

    async def drain(self) -> AsyncIterator[Frame]:
        assert self.ready is not None, "bind() first"
        while True:
            self.ready.clear()
            with self.cond:
                frame = self.items.popleft() if self.items else None
                done = self.closed and frame is None
                self.cond.notify_all()
            if frame is not None:
                yield frame
            elif done:
                return
            else:
                await self.ready.wait()


class _Base:
    """What every source shares: downscaling, the capture hooks, the handoff and closing."""

    def __init__(self, stream_id: str, *, max_side: int, on_frame: Callable[[Frame], None] | None,
                 preroll: PreRoll | None, capacity: int = 1, lossless: bool = False) -> None:
        if not stream_id:
            raise ValueError("stream_id must not be empty")
        self.stream_id = stream_id
        self.max_side = max_side
        self.on_frame = on_frame
        self.preroll = preroll
        self.hook_errors = 0
        self._preroll_lock = threading.Lock()
        self._handoff = _Handoff(capacity, lossless)
        self._started = False

    @property
    def dropped(self) -> int:
        """Frames replaced in the handoff before the consumer took them."""
        return self._handoff.dropped

    def _emit(self, frame: Frame, stop: threading.Event | None = None) -> None:
        if self.preroll is not None:
            with self._preroll_lock:
                self.preroll.add(frame)
        if self.on_frame is not None:
            try:
                self.on_frame(frame)
            except Exception:  # noqa: BLE001 - a broken hook must not stop capture
                self.hook_errors += 1
        self._handoff.put(frame, stop)

    def clip(self, interval: Interval, n: int = 4) -> tuple[Frame, ...]:
        """Thread-safe :meth:`PreRoll.select` on this source's ring (empty without one)."""
        if self.preroll is None:
            return ()
        with self._preroll_lock:
            return self.preroll.select(interval, n)

    def keyframe(self, interval: Interval, at: float | None = None) -> Frame | None:
        """Thread-safe :meth:`PreRoll.keyframe` on this source's ring."""
        if self.preroll is None:
            return None
        with self._preroll_lock:
            return self.preroll.keyframe(interval, at)

    def _start(self) -> None:
        pass

    async def frames(self) -> AsyncIterator[Frame]:
        """Yield frames until the source ends or is closed. Iterate it once."""
        if self._started:
            raise RuntimeError("frames() can be iterated once per source")
        self._started = True
        self._handoff.bind()
        drain = self._handoff.drain()
        self._start()
        try:
            async for frame in drain:
                yield frame
        finally:
            self._stop()
            await drain.aclose()

    def _stop(self) -> None:
        self._handoff.close()

    async def aclose(self) -> None:
        self._stop()

    async def __aenter__(self) -> _Base:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()


class PushSource(_Base):
    """A source your app feeds: ``push(image)`` from any thread, consume ``frames()``.

    Holds at most ``maxlen`` frames; when full, the oldest is dropped. ``t`` is read from the
    monotonic clock at push time and ``t_wall`` defaults to ``time.time()``.
    """

    def __init__(self, stream_id: str = "push", *, max_side: int = 640, maxlen: int = 8, epoch: int = 0,
                 on_frame: Callable[[Frame], None] | None = None, preroll: PreRoll | None = None) -> None:
        super().__init__(stream_id, max_side=max_side, on_frame=on_frame, preroll=preroll, capacity=maxlen)
        self.epoch = epoch
        self._seq = 0
        self._last_t = -float("inf")
        self._lock = threading.Lock()

    def push(self, image: np.ndarray, t_wall: float | None = None) -> Frame:
        if self._handoff.closed:
            raise RuntimeError("push() on a closed PushSource")
        with self._lock:
            t = max(time.monotonic(), self._last_t)
            frame = Frame(self.stream_id, self.epoch, self._seq, t, time.time() if t_wall is None else t_wall,
                          downscale(np.ascontiguousarray(image), self.max_side))
            self._seq += 1
            self._last_t = t
            self._emit(frame)
        return frame

    def close(self) -> None:
        """End the stream: ``frames()`` finishes once the frames already pushed are taken."""
        self._stop()


class Reader:
    """A blocking frame reader. ``read`` returns ``(image, media_s)`` or None at end."""

    def read(self) -> tuple[np.ndarray, float | None] | None:
        raise NotImplementedError

    def abort(self) -> None:
        """Called from another thread to unblock a pending ``read``, if possible."""

    def close(self) -> None:
        pass


class CaptureSource(_Base):
    """A source read by a blocking :class:`Reader` on a background thread.

    ``opener`` makes a connected reader and raises when it can't. ``live=True`` reconnects
    forever; ``live=False`` (a file) ends at EOF, or restarts it in a new epoch if ``loop``.
    ``sleep`` (default: an interruptible wait), ``clock``, ``wall`` and ``rng`` are injectable.
    """

    def __init__(self, stream_id: str, opener: Callable[[], Reader], *, live: bool, loop: bool = False,
                 fps: float | None = None, max_side: int = 640, epoch: int = 0,
                 backoff: tuple[float, float] = (2.0, 30.0), jitter: float = 0.1,
                 on_health: Callable[[Health], None] | None = None,
                 on_frame: Callable[[Frame], None] | None = None, preroll: PreRoll | None = None,
                 sleep: Callable[[float], Any] | None = None, clock: Callable[[], float] = time.monotonic,
                 wall: Callable[[], float] = time.time, rng: Callable[[], float] = random.random) -> None:
        super().__init__(stream_id, max_side=max_side, on_frame=on_frame, preroll=preroll, lossless=not live)
        if fps is not None and not fps > 0:
            raise ValueError("fps must be positive")
        if not 0 < backoff[0] <= backoff[1]:
            raise ValueError("backoff is (first, max) with 0 < first <= max")
        self.opener, self.live, self.loop, self.fps = opener, live, loop, fps
        self.backoff, self.jitter = backoff, jitter
        self.on_health = on_health
        self.link = "connecting"
        self.epoch = epoch - 1          # the first connection's first frame makes it `epoch`
        self.reconnects = 0             # connections after the first that delivered frames
        self.delays: deque[float] = deque(maxlen=64)  # recent backoff delays, for inspection
        self._first_epoch = epoch
        self._stop_event = threading.Event()
        self._sleep = sleep or self._stop_event.wait
        self._clock, self._wall, self._rng = clock, wall, rng
        self._reader: Reader | None = None
        self._thread: threading.Thread | None = None

    def _start(self) -> None:
        self._thread = threading.Thread(target=self._run, name=f"percept-capture-{self.stream_id}", daemon=True)
        self._thread.start()

    def _health(self, kind: str, detail: str) -> None:
        self.link = "up" if kind == "link_up" else "down"
        if self.on_health is None:
            return
        event = Health(kind, self.stream_id, self._wall(), detail)  # type: ignore[arg-type]
        loop = self._handoff.loop
        try:
            if loop is not None:
                loop.call_soon_threadsafe(self.on_health, event)
            else:
                self.on_health(event)
        except RuntimeError:
            pass

    def _run(self) -> None:
        failures = 0
        last_t = -float("inf")
        try:
            while not self._stop_event.is_set():
                got, error, last_t = self._connection(last_t)
                if self._stop_event.is_set():
                    break
                if not self.live:  # a file: EOF ends it, or restarts it in a new epoch if looping
                    if error is not None or not got:
                        self._health("link_down", error or "the file has no frames")
                    if self.loop and got and error is None:
                        continue
                    break
                if got:
                    failures = 0
                if self.link != "down":
                    self._health("link_down", error or "stream ended")
                delay = min(self.backoff[1], self.backoff[0] * 2 ** min(failures, 30))
                delay *= 1 + self.jitter * (2 * self._rng() - 1)
                failures += 1
                self.delays.append(delay)
                self._sleep(delay)
        finally:
            self.link = "ended"
            self._handoff.close()

    def _connection(self, last_t: float) -> tuple[bool, str | None, float]:
        """Read one connection to its end. Returns (delivered frames, error, last t)."""
        try:
            reader = self.opener()
        except Exception as exc:  # noqa: BLE001 - every open failure is retried the same way
            return False, f"{type(exc).__name__}: {exc}", last_t
        self._reader = reader
        got, error, seq, due = False, None, 0, None
        t0, w0 = max(self._clock(), last_t), self._wall()
        period = 1.0 / self.fps if self.fps else 0.0
        try:
            while not self._stop_event.is_set():
                item = reader.read()
                if item is None:
                    break
                image, media_s = item
                if media_s is None:
                    t, t_wall = max(self._clock(), last_t), self._wall()
                else:
                    t, t_wall = max(t0 + media_s, last_t), w0 + media_s
                if period:
                    if due is not None and t < due - 0.1 * period:
                        continue
                    due = (t if due is None or t - due > period else due) + period
                if not got:
                    got = True
                    self.epoch += 1
                    self.reconnects = self.epoch - self._first_epoch
                    if self.link == "down":
                        self._health("link_up", f"epoch {self.epoch}")
                    self.link = "up"
                frame = Frame(self.stream_id, self.epoch, seq, t, t_wall,
                              downscale(np.ascontiguousarray(image), self.max_side))
                seq, last_t = seq + 1, t
                self._emit(frame, self._stop_event)
        except Exception as exc:  # noqa: BLE001 - a read failure is a link failure
            error = f"{type(exc).__name__}: {exc}"
        finally:
            self._reader = None
            try:
                reader.close()
            except Exception:  # noqa: BLE001
                pass
        return got, error, last_t

    def _stop(self) -> None:
        self._stop_event.set()
        reader = self._reader
        if reader is not None:
            reader.abort()
        super()._stop()

    async def aclose(self, timeout: float = 5.0) -> None:
        """Stop capture and wait (up to ``timeout``) for the thread and any subprocess to end."""
        self._stop()
        if self._thread is not None:
            await asyncio.to_thread(self._thread.join, timeout)


# --------------------------------------------------------------------------- readers


class Cv2Reader(Reader):
    """OpenCV capture. Files report media time from the frame index."""

    def __init__(self, target: str | int, *, file: bool) -> None:
        cv2 = _cv2()
        if isinstance(target, str) and target.lower().startswith(("rtsp://", "rtsps://")):
            params = [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 10_000, cv2.CAP_PROP_READ_TIMEOUT_MSEC, 10_000]
            with _ENV_LOCK:  # TCP transport; a value the caller exported wins
                preset = os.environ.get(_FFMPEG_ENV)
                os.environ.setdefault(_FFMPEG_ENV, "rtsp_transport;tcp")
                try:
                    self.cap = cv2.VideoCapture(target, cv2.CAP_FFMPEG, params)
                finally:
                    if preset is None:
                        os.environ.pop(_FFMPEG_ENV, None)
        else:
            self.cap = cv2.VideoCapture(target)
        if not self.cap.isOpened():
            self.cap.release()
            raise ConnectionError(f"could not open {target!r}")
        self.src_fps = (self.cap.get(cv2.CAP_PROP_FPS) or 30.0) if file else None
        self.index = 0

    def read(self) -> tuple[np.ndarray, float | None] | None:
        ok, image = self.cap.read()
        if not ok or image is None:
            return None
        media_s = None if self.src_fps is None else self.index / self.src_fps
        self.index += 1
        return image, media_s

    def close(self) -> None:
        self.cap.release()


def ffmpeg_command(uri: str, *, fps: float | None = None, max_side: int = 640, ffmpeg: str = "ffmpeg") -> list[str]:
    """The ffmpeg command that decodes ``uri`` to a stream of PPM (raw RGB) frames on stdout."""
    cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin"]
    low = uri.lower().split("?", 1)[0]
    if low.startswith(("http://", "https://")):
        # Reconnect inside ffmpeg on brief HTTP hiccups; rw_timeout turns a dead socket into EOF.
        cmd += ["-reconnect", "1", "-reconnect_streamed", "1", "-reconnect_delay_max", "5",
                "-rw_timeout", "15000000"]
    if low.endswith((".m3u8", ".m3u")):
        cmd += ["-live_start_index", "-1"]  # open at the live edge, not the head of the window
    filters = ([f"fps={fps:g}"] if fps else []) + [
        f"scale=w='min({max_side},iw)':h='min({max_side},ih)':force_original_aspect_ratio=decrease"]
    return cmd + ["-i", uri, "-an", "-sn", "-vf", ",".join(filters), "-f", "image2pipe", "-vcodec", "ppm", "-"]


def _read_exact(stream: Any, n: int) -> bytes | None:
    chunks, left = [], n
    while left:
        chunk = stream.read(left)
        if not chunk:
            return None
        chunks.append(chunk)
        left -= len(chunk)
    return b"".join(chunks)


def read_ppm(stream: Any) -> np.ndarray | None:
    """Read one binary PPM (P6) frame from ``stream`` as BGR; None at a clean or torn end."""
    fields: list[bytes] = []
    token = b""
    while len(fields) < 4:
        byte = stream.read(1)
        if not byte:
            return None
        if byte.isspace():
            if token:
                fields.append(token)
                token = b""
        else:
            token += byte
    if fields[0] != b"P6" or fields[3] != b"255":
        raise ValueError(f"not an 8-bit PPM frame: {b' '.join(fields)!r}")
    w, h = int(fields[1]), int(fields[2])
    data = _read_exact(stream, w * h * 3)
    if data is None:
        return None
    return np.frombuffer(data, np.uint8).reshape(h, w, 3)[..., ::-1].copy()


class FfmpegReader(Reader):
    """Frames decoded by an ffmpeg subprocess. ``abort`` kills it, so no process outlives us."""

    def __init__(self, command: list[str]) -> None:
        self.proc = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                     stderr=subprocess.DEVNULL)

    def read(self) -> tuple[np.ndarray, float | None] | None:
        image = read_ppm(self.proc.stdout)
        return None if image is None else (image, None)

    def abort(self) -> None:
        if self.proc.poll() is None:
            self.proc.kill()

    def close(self) -> None:
        self.abort()
        try:
            self.proc.wait(timeout=5)
        finally:
            if self.proc.stdout is not None:
                self.proc.stdout.close()


# --------------------------------------------------------------------------- open_source


def open_source(uri: str, *, stream_id: str | None = None, fps: float | None = None, max_side: int = 640,
                loop: bool = False, backend: str = "auto", epoch: int = 0,
                backoff: tuple[float, float] = (2.0, 30.0),
                on_health: Callable[[Health], None] | None = None,
                on_frame: Callable[[Frame], None] | None = None,
                preroll: PreRoll | None = None, ffmpeg: str = "ffmpeg") -> CaptureSource:
    """Open a camera, file or stream as a :class:`~percept.contracts.Source`.

    ``uri`` is ``rtsp://``/``rtsps://`` (OpenCV), ``webcam:N`` (OpenCV device N), an HLS or
    DASH manifest URL ending in ``.m3u8``/``.mpd`` (ffmpeg), or a file path or other URL
    (OpenCV). ``backend="ffmpeg"`` forces the ffmpeg reader for any input. ``fps`` caps the
    sampling rate; frames are shrunk so their longest side is at most ``max_side``. Nothing is
    read until ``frames()`` is iterated. Missing OpenCV or ffmpeg raises here, not later.
    """
    if backend not in ("auto", "cv2", "ffmpeg"):
        raise ValueError("backend is 'auto', 'cv2' or 'ffmpeg'")
    low = uri.lower()
    path = uri[7:] if low.startswith("file://") else uri
    is_url = "://" in low
    manifest = low.startswith(("http://", "https://")) and low.split("?", 1)[0].endswith(MANIFEST_SUFFIXES)
    webcam = low.startswith("webcam:")
    live = webcam or manifest or low.startswith(("rtsp://", "rtsps://", "rtmp://", "srt://", "udp://"))
    if not is_url and not webcam and not os.path.exists(path):
        raise FileNotFoundError(f"no such video file: {path!r}")
    common: dict[str, Any] = dict(live=live, loop=loop, max_side=max_side, epoch=epoch, backoff=backoff,
                                  on_health=on_health, on_frame=on_frame, preroll=preroll)
    name = stream_id or uri
    if backend == "ffmpeg" or (backend == "auto" and manifest):
        if shutil.which(ffmpeg) is None:
            raise RuntimeError(f"{uri!r} is read with ffmpeg, and no {ffmpeg!r} binary is on PATH")
        command = ffmpeg_command(path, fps=fps, max_side=max_side, ffmpeg=ffmpeg)
        return CaptureSource(name, lambda: FfmpegReader(command), **common)
    _cv2()
    target: str | int = int(low.split(":", 1)[1].lstrip("/") or 0) if webcam else path
    return CaptureSource(name, lambda: Cv2Reader(target, file=not live and not is_url), fps=fps, **common)
