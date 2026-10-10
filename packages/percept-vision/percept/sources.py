"""Sources: where frames come from.

:func:`open_source` opens a camera, file or stream and returns a
:class:`~percept.contracts.Source`. :class:`PushSource` is for apps that already own capture.

Capture never waits for the consumer. A blocking reader runs on a background thread and hands
frames to the event loop through a one-slot "latest frame": when the consumer is slow the
older frame is replaced, never queued. Every frame is offered to the optional ``on_frame``
hook and :class:`~percept.clips.PreRoll` on the capture thread first, so the pre-roll ring
sees the stream at capture rate even when the consumer does not. Files are the exception: a
file has nothing to lose by waiting, so a file source hands over every sampled frame.

Live sources (every ``scheme://`` URL except one naming a container file such as ``.mp4``,
and webcams) reconnect forever with jittered exponential backoff (2 s doubling to 30 s by
default), and a live connection that delivers nothing for ``stall_timeout`` seconds is torn
down and retried. ``stream_epoch`` increases by one each time a connection delivers its first
frame, and ``seq`` restarts at 0 in every epoch.

Clocks: ``Frame.t`` is the stream clock, monotonic within an epoch and never lower than the
epoch before. For a live source it is ``time.monotonic()`` at arrival and ``t_wall`` is Unix
time at arrival. For a file both follow the video's own timeline, anchored at open (and
carried forward across ``loop`` restarts), so a file reads the same at any decode speed and
``t`` may run ahead of ``time.monotonic()``. Compare ``frame.t`` only with another
``frame.t``. ``Health.t_wall`` is always real wall time.

Credentials never leave the URI through Percept: the default ``stream_id`` drops
``user:password@``, the query string and the fragment, and health details are scrubbed of
credentials. Two places outside Percept's control still see the raw URI: OpenCV's own logger
may print it to the process's stderr (set ``OPENCV_LOG_LEVEL=ERROR`` or ``OFF`` to silence
it), and the ffmpeg/ffprobe command line carries it, so it is visible in local process
listings.

Link changes are reported, never raised: ``frames()`` does not raise for a camera that is
down. They arrive as :class:`~percept.contracts.Health` (``link_down`` once per outage,
``link_up`` when frames flow again) through ``on_health`` and every listener added with
``add_health_listener`` (a ``Monitor`` adds one when it watches the source), called on the event
loop thread, and the current state is always readable as ``source.link`` (``"connecting"``,
``"up"``, ``"down"`` or ``"ended"``). A :class:`PushSource` has no connection of its own: the app
reports its outages with ``set_link``.

OpenCV is the ``video`` extra (``pip install "percept-vision[video]"``) and is imported only
when a source needs it. HLS and DASH are read by an ``ffmpeg`` subprocess, which must be on
``PATH``.
"""

from __future__ import annotations

import asyncio
import math
import os
import random
import re
import shutil
import subprocess
import threading
import time
from collections import deque
from typing import Any, AsyncIterator, Callable
from urllib.parse import urlsplit, urlunsplit

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
    """What every source shares: downscaling, the capture hooks, health listeners, the handoff
    and closing."""

    def __init__(self, stream_id: str, *, max_side: int, on_frame: Callable[[Frame], None] | None,
                 preroll: PreRoll | None, capacity: int = 1, lossless: bool = False,
                 on_health: Callable[[Health], None] | None = None,
                 wall: Callable[[], float] = time.time) -> None:
        if not stream_id:
            raise ValueError("stream_id must not be empty")
        self.stream_id = stream_id
        self.max_side = max_side
        self.on_frame = on_frame
        self.on_health = on_health
        self.preroll = preroll
        self.hook_errors = 0
        self._listeners: list[Callable[[Health], None]] = []
        self._wall = wall
        self._preroll_lock = threading.Lock()
        self._handoff = _Handoff(capacity, lossless)
        self._started = False

    @property
    def dropped(self) -> int:
        """Frames replaced in the handoff before the consumer took them."""
        return self._handoff.dropped

    def add_health_listener(self, listener: Callable[[Health], None]) -> None:
        """Also send this source's link health (``link_down`` / ``link_up``) to ``listener``,
        on the event loop thread, after ``on_health``. Adding a listener twice has no effect."""
        if not callable(listener):
            raise TypeError("a health listener is callable")
        if listener not in self._listeners:
            self._listeners.append(listener)

    def remove_health_listener(self, listener: Callable[[Health], None]) -> None:
        """Stop sending health to ``listener`` (no error if it was not added)."""
        if listener in self._listeners:
            self._listeners.remove(listener)

    def _health(self, kind: str, detail: str) -> None:
        """Report a link change to ``on_health`` and every listener. Once ``frames()`` runs they
        are called on the event loop thread, always through its queue, so reports arrive in the
        order they were made; before that, directly on the caller's thread."""
        if self.on_health is None and not self._listeners:
            return
        event = Health(kind, self.stream_id, self._wall(), scrub(detail))  # type: ignore[arg-type]
        loop = self._handoff.loop
        try:
            if loop is not None:
                loop.call_soon_threadsafe(self._publish, event)
            else:
                self._publish(event)
        except RuntimeError:  # the loop is closed: nobody is listening
            pass

    def _publish(self, event: Health) -> None:
        for listener in [self.on_health, *self._listeners]:
            if listener is None:
                continue
            try:
                listener(event)
            except Exception:  # noqa: BLE001 - one broken listener must not starve the others
                self.hook_errors += 1

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

    The app owns the connection, so it reports outages: ``set_link(False, detail)`` when its
    feed is gone (health ``link_down``, and a ``link_down`` blind interval in a monitor watching
    this source) and ``set_link(True)`` when it is back (``link_up``); a frame pushed while the
    link is down also marks it up. Coming back up starts a new ``epoch`` with ``seq`` at 0, as a
    reconnect does. ``Monitor.set_link`` does the same for a stream the monitor created itself
    from ``Monitor.push``, whose ``PushSource`` the app never holds.
    """

    def __init__(self, stream_id: str = "push", *, max_side: int = 640, maxlen: int = 8, epoch: int = 0,
                 on_frame: Callable[[Frame], None] | None = None, preroll: PreRoll | None = None,
                 on_health: Callable[[Health], None] | None = None) -> None:
        super().__init__(stream_id, max_side=max_side, on_frame=on_frame, preroll=preroll, capacity=maxlen,
                         on_health=on_health)
        self.epoch = epoch
        self.link = "up"
        self._seq = 0
        self._last_t = -float("inf")
        self._lock = threading.RLock()   # reentrant: a listener called while it is held may push

    def set_link(self, up: bool, detail: str = "") -> None:
        """Report the app's feed as up or down, from any thread. Repeating the current state
        reports nothing.

        The change and its report happen under one lock, so concurrent ``set_link`` and
        ``push`` calls report strictly alternating ``link_down`` / ``link_up``. Coming back up
        starts a new epoch: an answer to a request sent before the outage is then stale and
        changes nothing, and held states are rechecked on the new epoch's frames.
        """
        with self._lock:
            if self._handoff.closed or (self.link == "up") == bool(up):
                return
            self.link = "up" if up else "down"
            if up:
                self.epoch += 1
                self._seq = 0
            self._health("link_up" if up else "link_down",
                         detail or (f"epoch {self.epoch}" if up else "reported down by the application"))

    def push(self, image: np.ndarray, t_wall: float | None = None) -> Frame:
        if self._handoff.closed:
            raise RuntimeError("push() on a closed PushSource")
        with self._lock:
            if self.link == "down":
                self.set_link(True, "frames pushed again")
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
    """A blocking frame reader. ``read`` returns ``(image, media_s)`` or None at end.

    ``media_s`` is the frame's position on the video's own timeline, or None for a live
    reader whose frames are stamped with the clock when they arrive.
    """

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

    Clocks. ``Frame.t`` is the stream clock: monotonic within an epoch and never lower than
    the epoch before it. For a live reader it is ``time.monotonic()`` at arrival. For a file it
    is the media timeline anchored at open, so it can run ahead of ``time.monotonic()`` while
    a file decodes faster than real time; compare ``frame.t`` only with another ``frame.t``.
    ``Frame.t_wall`` follows the same rule (Unix time at arrival, or the media timeline
    anchored at the wall clock when the file opened, carried forward across loop restarts).
    ``Health.t_wall`` is always real wall time.

    A live connection that delivers no frame for ``stall_timeout`` seconds is aborted and treated
    as a failure (``link_down``, then backoff and reconnect). The limit also bounds the time to
    the first frame of a connection (for ffmpeg, which opens without blocking), and with an
    ``fps`` cap it is never shorter than two frame periods. Aborting kills an ffmpeg reader
    at once; an OpenCV read can't be interrupted, so it ends at its own read timeout, and
    ``aclose`` may return while the thread finishes that bounded read.

    ``epoch`` is None until the first frame. ``sleep`` (default: an interruptible wait),
    ``clock``, ``wall`` and ``rng`` are injectable for tests.
    """

    def __init__(self, stream_id: str, opener: Callable[[], Reader], *, live: bool, loop: bool = False,
                 fps: float | None = None, max_side: int = 640, epoch: int = 0,
                 backoff: tuple[float, float] = (2.0, 30.0), jitter: float = 0.1,
                 stall_timeout: float | None = 30.0,
                 on_health: Callable[[Health], None] | None = None,
                 on_frame: Callable[[Frame], None] | None = None, preroll: PreRoll | None = None,
                 sleep: Callable[[float], Any] | None = None, clock: Callable[[], float] = time.monotonic,
                 wall: Callable[[], float] = time.time, rng: Callable[[], float] = random.random) -> None:
        super().__init__(stream_id, max_side=max_side, on_frame=on_frame, preroll=preroll, lossless=not live,
                         on_health=on_health, wall=wall)
        if fps is not None and not (math.isfinite(fps) and fps > 0):
            raise ValueError("fps must be positive")
        if not (len(backoff) == 2 and all(math.isfinite(b) for b in backoff) and 0 < backoff[0] <= backoff[1]):
            raise ValueError("backoff is (first, max) seconds with 0 < first <= max")
        if not (math.isfinite(jitter) and 0 <= jitter < 1):
            raise ValueError("jitter is a fraction in [0, 1)")
        if stall_timeout is not None and not (math.isfinite(stall_timeout) and stall_timeout > 0):
            raise ValueError("stall_timeout must be positive or None")
        self.opener, self.live, self.loop, self.fps = opener, live, loop, fps
        self.backoff, self.jitter, self.stall_timeout = backoff, jitter, stall_timeout
        self.link = "connecting"
        self.epoch: int | None = None   # the first connection's first frame makes it the start epoch
        self.reconnects = 0             # connections after the first that delivered frames
        self.delays: deque[float] = deque(maxlen=64)  # recent backoff delays, for inspection
        self._next_epoch = epoch
        self._stop_event = threading.Event()
        self._sleep = sleep or self._stop_event.wait
        self._clock, self._rng = clock, rng
        self._reader: Reader | None = None
        self._progress = 0.0            # clock() at the last sign of life of the current connection
        self._stalled = False
        self._conn_lock = threading.Lock()  # guards _reader/_progress/_stalled between threads
        self._last = (-math.inf, -math.inf, 0.0)  # (t, t_wall, media step) of the last frame
        self._thread: threading.Thread | None = None

    def __repr__(self) -> str:
        return f"CaptureSource({self.stream_id!r}, link={self.link!r}, epoch={self.epoch})"

    def _start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="percept-capture", daemon=True)
        self._thread.start()
        if self.live and self.stall_timeout is not None:  # a file's handoff may wait on the consumer
            threading.Thread(target=self._watchdog, name="percept-capture-watchdog", daemon=True).start()

    @property
    def stall_limit(self) -> float | None:
        """The effective stall timeout: ``stall_timeout``, but at least two ``fps`` periods."""
        if self.stall_timeout is None:
            return None
        return max(self.stall_timeout, 2.0 / self.fps) if self.fps else self.stall_timeout

    def _watchdog(self) -> None:
        limit = self.stall_limit
        assert limit is not None
        while not self._stop_event.wait(min(1.0, limit / 4)):
            self._check_stall(self._reader)

    def _check_stall(self, reader: Reader | None) -> bool:
        """Abort ``reader`` if it is still the current connection and has gone quiet too long."""
        limit = self.stall_limit
        with self._conn_lock:
            if (reader is None or reader is not self._reader or self._stalled or limit is None
                    or self._clock() - self._progress <= limit):
                return False
            self._stalled = True
        reader.abort()
        return True

    def _health(self, kind: str, detail: str) -> None:
        self.link = "up" if kind == "link_up" else "down"
        super()._health(kind, detail)

    def _run(self) -> None:
        failures = 0
        try:
            while not self._stop_event.is_set():
                got, error = self._connection()
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

    def _connection(self) -> tuple[bool, str | None]:
        """Read one connection to its end. Returns (delivered frames, error)."""
        try:
            reader = self.opener()
        except Exception as exc:  # noqa: BLE001 - every open failure is retried the same way
            return False, f"{type(exc).__name__}: {exc}"
        with self._conn_lock:
            self._progress, self._stalled, self._reader = self._clock(), False, reader
        got, error, seq, due, prev_media = False, None, 0, None, None
        last_t, last_wall, step = self._last
        t0, w0 = max(self._clock(), last_t + step), max(self._wall(), last_wall + step)
        period = 1.0 / self.fps if self.fps else 0.0
        try:
            while not self._stop_event.is_set():
                item = reader.read()
                if item is None:
                    break
                self._progress = self._clock()
                image, media_s = item
                if media_s is None:
                    t, t_wall = max(self._clock(), last_t), max(self._wall(), last_wall)
                else:
                    t, t_wall = max(t0 + media_s, last_t), max(w0 + media_s, last_wall)
                    step = media_s - prev_media if prev_media is not None else step
                    prev_media = media_s
                if period:
                    if due is not None and t < due - 0.1 * period:
                        continue
                    due = (t if due is None or t - due > period else due) + period
                if not got:
                    got = True
                    self.reconnects += self.epoch is not None
                    self.epoch = self._next_epoch
                    self._next_epoch += 1
                    if self.link == "down":
                        self._health("link_up", f"epoch {self.epoch}")
                    self.link = "up"
                frame = Frame(self.stream_id, self.epoch, seq, t, t_wall,
                              downscale(np.ascontiguousarray(image), self.max_side))
                seq, last_t, last_wall = seq + 1, t, t_wall
                self._last = (t, t_wall, step)
                self._emit(frame, self._stop_event)
        except Exception as exc:  # noqa: BLE001 - a read failure is a link failure
            error = f"{type(exc).__name__}: {exc}"
        finally:
            with self._conn_lock:
                self._reader = None
            try:
                reader.close()
            except Exception:  # noqa: BLE001
                pass
        if self._stalled and not self._stop_event.is_set():
            error = f"stalled: no frame for {self.stall_limit:g} s"
        return got, error

    def _stop(self) -> None:
        self._stop_event.set()
        reader = self._reader
        if reader is not None:
            reader.abort()
        super()._stop()

    async def aclose(self, timeout: float = 5.0) -> None:
        """Stop capture and wait (up to ``timeout``) for the thread and any subprocess to end.

        An OpenCV read in progress can't be interrupted; the thread then ends after that read
        returns (bounded by the read timeout) and this may return first.
        """
        self._stop()
        if self._thread is not None:
            await asyncio.to_thread(self._thread.join, timeout)


# --------------------------------------------------------------------------- readers

_URL = re.compile(r"(?i)\b[a-z][a-z0-9+.-]*://[^\s'\"<>]+")


def redact(uri: str) -> str:
    """A URI safe to show: no credentials, query string or fragment."""
    parts = urlsplit(uri)
    if not parts.scheme or not parts.netloc:
        return uri
    return urlunsplit((parts.scheme, parts.netloc.rsplit("@", 1)[-1], parts.path, "", ""))


def scrub(text: str) -> str:
    """Redact every URL in ``text`` (credentials, query string and fragment removed)."""
    return _URL.sub(lambda m: redact(m.group(0)), text)


class Cv2Reader(Reader):
    """OpenCV capture. Files report media time from the frame index."""

    TIMEOUT_MS = 10_000

    def __init__(self, target: str | int, *, file: bool, label: str = "") -> None:
        cv2 = _cv2()
        if isinstance(target, str) and "://" in target:
            params = [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, self.TIMEOUT_MS, cv2.CAP_PROP_READ_TIMEOUT_MSEC, self.TIMEOUT_MS]
            rtsp = target.lower().startswith(("rtsp://", "rtsps://"))
            with _ENV_LOCK:  # TCP transport for RTSP; a value the caller exported wins
                preset = os.environ.get(_FFMPEG_ENV)
                if rtsp:
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
            raise ConnectionError(f"could not open {label or 'the source'}")
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


def probe_rate(path: str, ffprobe: str = "ffprobe") -> float | None:
    """The video's frame rate from ``ffprobe``, or None when it can't be read."""
    try:
        out = subprocess.run([ffprobe, "-v", "error", "-select_streams", "v:0", "-show_entries",
                              "stream=r_frame_rate", "-of", "csv=p=0", path],
                             capture_output=True, text=True, timeout=15, stdin=subprocess.DEVNULL).stdout
        num, _, den = out.strip().splitlines()[0].partition("/")
        rate = float(num) / float(den or 1)
    except (OSError, subprocess.SubprocessError, ValueError, IndexError, ZeroDivisionError):
        return None
    return rate if math.isfinite(rate) and rate > 0 else None


def _read_exact(stream: Any, n: int) -> bytes | None:
    chunks, left = [], n
    while left:
        chunk = stream.read(left)
        if not chunk:
            return None
        chunks.append(chunk)
        left -= len(chunk)
    return b"".join(chunks)


def read_ppm(stream: Any, max_pixels: int | None = None) -> np.ndarray | None:
    """Read one binary PPM (P6) frame from ``stream`` as BGR; None at a clean or torn end.

    ``max_pixels`` rejects a header announcing a larger frame (``ValueError``).
    """
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
        elif len(token) > 16:
            raise ValueError("malformed PPM header")
        else:
            token += byte
    if fields[0] != b"P6" or fields[3] != b"255" or not (fields[1].isdigit() and fields[2].isdigit()):
        raise ValueError(f"not an 8-bit PPM frame: {b' '.join(fields)[:40]!r}")
    w, h = int(fields[1]), int(fields[2])
    if w < 1 or h < 1 or (max_pixels is not None and w * h > max_pixels):
        raise ValueError(f"PPM frame size {w}x{h} is out of bounds")
    data = _read_exact(stream, w * h * 3)
    if data is None:
        return None
    return np.frombuffer(data, np.uint8).reshape(h, w, 3)[..., ::-1].copy()


class FfmpegReader(Reader):
    """Frames decoded by an ffmpeg subprocess. ``abort`` kills it, so no process outlives us.

    With ``rate`` (frames per second of the output), frames carry media time from their index,
    so a file reads the same at any decode speed. The last line ffmpeg wrote to stderr, with
    credentials removed, is kept as ``last_error`` and reported when the process fails.
    """

    def __init__(self, command: list[str], *, rate: float | None = None, max_pixels: int | None = None) -> None:
        self.rate, self.max_pixels, self.index = rate, max_pixels, 0
        self.last_error = ""
        self.proc = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE)
        self._stderr = threading.Thread(target=self._drain, name="percept-ffmpeg-stderr", daemon=True)
        self._stderr.start()

    def _drain(self) -> None:
        assert self.proc.stderr is not None
        for raw in self.proc.stderr:
            line = raw.decode("utf-8", "replace").strip()
            if line:
                self.last_error = scrub(line)[:300]

    def read(self) -> tuple[np.ndarray, float | None] | None:
        image = read_ppm(self.proc.stdout, self.max_pixels)
        if image is None:
            try:
                code = self.proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                return None
            self._stderr.join(1)
            if code != 0:
                raise ConnectionError(f"ffmpeg exited with {code}: {self.last_error or 'no message'}")
            return None
        media_s = None if self.rate is None else self.index / self.rate
        self.index += 1
        return image, media_s

    def abort(self) -> None:
        if self.proc.poll() is None:
            self.proc.kill()

    def close(self) -> None:
        self.abort()
        try:
            self.proc.wait(timeout=5)
        finally:
            for pipe in (self.proc.stdout, self.proc.stderr):
                if pipe is not None:
                    pipe.close()


# --------------------------------------------------------------------------- open_source

CONTAINER_SUFFIXES = (".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v")


def open_source(uri: str, *, stream_id: str | None = None, fps: float | None = None, max_side: int = 640,
                loop: bool = False, live: bool | None = None, backend: str = "auto", epoch: int = 0,
                backoff: tuple[float, float] = (2.0, 30.0), stall_timeout: float | None = 30.0,
                on_health: Callable[[Health], None] | None = None,
                on_frame: Callable[[Frame], None] | None = None,
                preroll: PreRoll | None = None, ffmpeg: str = "ffmpeg") -> CaptureSource:
    """Open a camera, file or stream as a :class:`~percept.contracts.Source`.

    ``uri`` is ``rtsp://``/``rtsps://`` (OpenCV), ``webcam:N`` (OpenCV device N), an HLS or
    DASH manifest URL ending in ``.m3u8``/``.mpd`` (ffmpeg), or a file path, ``file://`` URL
    or other URL (OpenCV). ``backend="ffmpeg"`` forces the ffmpeg reader for any input.

    Every ``scheme://`` URL is live (reconnects forever) unless its path ends in a container
    suffix such as ``.mp4``; local files are not. ``live`` overrides the guess. ``fps`` caps
    the sampling rate; frames are shrunk so their longest side is at most ``max_side``.

    ``stream_id`` defaults to the URI without credentials, query string or fragment; the raw
    URI never appears in frames, health events or errors. Nothing is read until ``frames()``
    is iterated. Missing OpenCV or ffmpeg raises here, not later.
    """
    if backend not in ("auto", "cv2", "ffmpeg"):
        raise ValueError("backend is 'auto', 'cv2' or 'ffmpeg'")
    low = uri.lower()
    local = low.startswith("file://") or "://" not in low
    path = uri[7:] if low.startswith("file://") else uri
    webcam = low.startswith("webcam:")
    url_path = urlsplit(low).path
    manifest = not local and url_path.endswith(MANIFEST_SUFFIXES)
    if live is None:
        live = webcam or (not local and not url_path.endswith(CONTAINER_SUFFIXES))
    if local and not webcam and not os.path.exists(path):
        raise FileNotFoundError(f"no such video file: {path!r}")
    name = stream_id or ("webcam:" + low.split(":", 1)[1].lstrip("/") if webcam else redact(uri))
    common: dict[str, Any] = dict(live=live, loop=loop, max_side=max_side, epoch=epoch, backoff=backoff,
                                  stall_timeout=stall_timeout, on_health=on_health, on_frame=on_frame,
                                  preroll=preroll)
    if backend == "ffmpeg" or (backend == "auto" and manifest):
        binary = shutil.which(ffmpeg)
        if binary is None:
            raise RuntimeError(f"{name} is read with ffmpeg, and no {ffmpeg!r} binary is on PATH")
        # Live: cap fps on our side, so the pipe (and the stall watchdog) sees every decoded
        # frame. Files: cap inside ffmpeg, whose output rate then gives each frame's media time.
        command = ffmpeg_command(path, fps=None if live else fps, max_side=max_side, ffmpeg=ffmpeg)
        ffprobe = os.path.join(os.path.dirname(binary), "ffprobe")
        rates: list[float | None] = []

        def ffmpeg_opener() -> Reader:
            if not live and not rates:  # media time for files: the output rate, probed once
                rates.append(fps or probe_rate(path, ffprobe if os.path.exists(ffprobe) else "ffprobe"))
            return FfmpegReader(command, rate=None if live else rates[0], max_pixels=max_side * max_side)

        return CaptureSource(name, ffmpeg_opener, fps=fps if live else None, **common)
    _cv2()
    target: str | int = int(low.split(":", 1)[1].lstrip("/") or 0) if webcam else path
    return CaptureSource(name, lambda: Cv2Reader(target, file=not live, label=name), fps=fps, **common)
