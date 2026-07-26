"""Capture sources for the runtime worker.

``open_source(uri)`` opens a cv2-readable URI (file / RTSP / ``webcam:0`` / ``webcam://0``) on a
**background thread** and yields versioned source envelopes for programmable Watches:
  * ``gate_frame`` — a small gray :class:`~percept.edge.tier0.Frame` for the motion gate (still-vs-moved
    doesn't need full resolution),
  * ``judge_frame`` — a ``("image/jpeg", bytes)`` pair the VLM judges.

cv2 (``opencv-python-headless``) is the ``[runtime]``/``[rtsp]`` extra — imported lazily, never in the core.
The hardened transport (rtsp-tcp, reconnect/backoff, avfoundation ``-video_size``, ONVIF) is the ``[rtsp]``
follow-up; v1 reuses the deterministic decode pattern from ``eval/count_video.py``.
"""

from __future__ import annotations

import asyncio
import math
import queue
import threading
import time
from dataclasses import dataclass
from typing import AsyncIterator, Callable, Literal

from percept.runtime.envelope import SourceEnvelope, SourceFormat

GATE_W, GATE_H = 48, 36  # tier-0 downsample (still vs moved; full res not needed for the gate)

# A-4 delivered-fps misreport heuristic. Measure the RAW read rate over a rolling wall-clock window
# and compare it to the source's REPORTED CAP_PROP_FPS. A camera that reports 30 but delivers 15 keeps
# every Nth frame at half the intended rate — the gate/fovea-coast Nyquist guarantee silently halves.
# Only evaluated once ``>= _FPS_MIN_EVAL_S`` of wall time has actually elapsed, so an offline/instant
# fixture (span ~0) can never false-fire.
_FPS_WINDOW_S = 2.0  # rolling window the delivered rate is measured over
_FPS_MIN_EVAL_S = 1.0  # minimum elapsed wall time before we trust a measurement
_FPS_TOL = 0.25  # relative gap (|measured-reported|/reported) that counts as a misreport
_POLL = object()


@dataclass(frozen=True)
class SourceConfig:
    """Runtime capture tuning for webcam/video/RTSP sources."""

    fps: float = 4.0
    queue_size: int = 64
    reconnects: int = 0
    reconnect_backoff_s: float = 0.5
    rtsp_tcp: bool = True
    # Clock selected by the transitional tuple view.  Every envelope carries observed monotonic time
    # independently and carries decoder source time only when the decoder actually provides it.
    timebase: Literal["media", "wall"] = "media"
    # Live streams prefer fresh evidence over stale backlog. Deterministic file replays may instead
    # apply bounded producer backpressure so an evaluator observes every selected decoded frame.
    queue_policy: Literal["latest", "block"] = "latest"


def _normalize_uri(uri: str):
    if uri == "webcam":
        return 0
    if uri.startswith("webcam://"):
        return int(uri.removeprefix("webcam://") or 0)
    if uri.startswith("webcam:"):
        return int(uri.split(":", 1)[1] or 0)
    if uri.startswith("screen://"):
        raise NotImplementedError(
            "screen:// sources need a screen grabber (mss) — see docs/scenarios/screenshare.md"
        )
    return uri


def _put_latest(q: "queue.Queue", item, dropped: "list[int] | None" = None) -> None:
    """Put without letting a slow consumer make capture block forever.

    On a full queue the oldest frame is discarded to make room for the latest; ``dropped`` (a 1-elem
    shared counter) is bumped so the worker can surface a ``frames_dropped`` telemetry count.
    """
    try:
        q.put(item, timeout=0.1)
    except queue.Full:
        try:
            q.get_nowait()
            if dropped is not None:
                dropped[0] += 1
        except queue.Empty:
            pass
        q.put(item, timeout=0.1)


def _put_blocking(q: "queue.Queue", item, stop: threading.Event) -> bool:
    """Put into a bounded queue without dropping evidence.

    This policy is for finite, deterministic fixture replay. It deliberately backpressures the
    decoder, but still polls ``stop`` so cancelling a consumer cannot strand the capture thread.
    """
    while not stop.is_set():
        try:
            q.put(item, timeout=0.1)
            return True
        except queue.Full:
            continue
    return False


def _put_frame(
    q: "queue.Queue",
    item,
    config: SourceConfig,
    stop: threading.Event,
    dropped: "list[int] | None" = None,
) -> bool:
    if config.queue_policy == "latest":
        _put_latest(q, item, dropped)
        return True
    if config.queue_policy == "block":
        return _put_blocking(q, item, stop)
    raise ValueError(f"unknown source queue policy: {config.queue_policy!r}")


def _put_terminal(q: "queue.Queue", item, stop: threading.Event) -> None:
    """Publish an error/end marker after queued evidence instead of evicting that evidence."""
    while not stop.is_set():
        try:
            q.put(item, timeout=0.1)
            return
        except queue.Full:
            continue


def _get_polling(q: "queue.Queue", timeout: float = 0.5):
    """Bound one executor-backed queue wait so cancellation cannot strand a worker thread."""
    try:
        return q.get(timeout=timeout)
    except queue.Empty:
        return _POLL


def _open_capture(cv2, uri: str, config: SourceConfig):
    source = _normalize_uri(uri)
    if isinstance(source, str) and source.startswith("rtsp://") and config.rtsp_tcp:
        return cv2.VideoCapture(source, cv2.CAP_FFMPEG)
    return cv2.VideoCapture(source)


def _source_time(
    cap, cv2, uri: str, *, previous: float | None = None, after_first: bool = False
) -> float | None:
    """Read a decoder-provided media timestamp, or return ``None``.

    Device/network sources do not have a trustworthy media clock.  A reported FPS plus a local frame
    counter is a cadence estimate, not a source timestamp, so this adapter deliberately does not turn
    ``i / fps`` into provenance.
    """
    source = _normalize_uri(uri)
    if not isinstance(source, str):
        return None
    if source.lower().startswith(("rtsp://", "rtsps://", "rtp://", "udp://", "tcp://")):
        return None
    prop = getattr(cv2, "CAP_PROP_POS_MSEC", None)
    if prop is None:
        return None
    try:
        value = float(cap.get(prop)) / 1000.0
    except Exception:  # noqa: BLE001 - optional backend property
        return None
    if not math.isfinite(value) or value < 0.0:
        return None
    # OpenCV backends commonly report a permanent 0.0 when POS_MSEC is unsupported.  Zero is valid
    # for the first decoded frame, but a non-advancing decoder clock is not timestamp provenance.
    if after_first and previous is not None and value <= previous:
        return None
    return value


def _video_format(bgr) -> SourceFormat:
    shape = getattr(bgr, "shape", ())
    height = int(shape[0]) if len(shape) >= 2 and int(shape[0]) > 0 else None
    width = int(shape[1]) if len(shape) >= 2 and int(shape[1]) > 0 else None
    return SourceFormat(media_type="image/jpeg", width=width, height=height)


def _capture(
    uri: str,
    config: SourceConfig,
    q: "queue.Queue",
    stop: threading.Event,
    dropped: "list[int] | None" = None,
    *,
    now: "Callable[[], float] | None" = None,
    health: "list | None" = None,
) -> None:
    """Blocking capture loop, run on a daemon thread; pushes frames into ``q`` until stopped/EOF.

    ``now`` is the observation clock stamped on every envelope and used by the delivered-fps measure
    (injectable so tests need no real sleeping; defaults to ``time.monotonic``). ``health`` mirrors the
    ``dropped`` shared-list telemetry pattern: the capture thread appends ``(kind, data)`` events (today
    only ``("fps_misreported", {"reported", "measured"})``) for the worker to surface via ``on_health``."""
    import cv2  # lazy: the [runtime]/[rtsp] extra

    from percept.edge.tier0 import Frame

    if now is None:
        now = time.monotonic
    attempts = 0
    epoch = 0
    try:
        while not stop.is_set():
            cap = _open_capture(cv2, uri, config)
            if not cap.isOpened():
                cap.release()
                raise RuntimeError(f"could not open runtime source: {uri!r}")
            src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            step = max(1, round(src_fps / config.fps))
            i = 0
            sequence = 0
            previous_source_time: float | None = None
            read_ts: "list[float]" = []  # wall times of recent raw reads (A-4 delivered-fps window)
            fps_reported = False  # emit the misreport at most once per open
            try:
                while not stop.is_set():
                    ok, bgr = cap.read()
                    if not ok:
                        break
                    # A-4: measure the RAW read rate over a rolling wall-clock window and flag a source
                    # whose delivered fps diverges from its reported CAP_PROP_FPS. Emit-only (does NOT
                    # touch decimation or either envelope clock). Guarded by `health`.
                    if health is not None and not fps_reported:
                        tw = now()
                        read_ts.append(tw)
                        cutoff = tw - _FPS_WINDOW_S
                        while read_ts and read_ts[0] < cutoff:
                            read_ts.pop(0)
                        span = read_ts[-1] - read_ts[0]
                        if span >= _FPS_MIN_EVAL_S and len(read_ts) >= 2:
                            measured = (len(read_ts) - 1) / span
                            if abs(measured - src_fps) > _FPS_TOL * src_fps:
                                health.append(
                                    ("fps_misreported", {"reported": src_fps, "measured": measured})
                                )
                                fps_reported = True
                    if i % step == 0:
                        observed_time = now()
                        source_time = _source_time(
                            cap,
                            cv2,
                            uri,
                            previous=previous_source_time,
                            after_first=sequence > 0,
                        )
                        if source_time is not None:
                            previous_source_time = source_time
                        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
                        g = cv2.resize(gray, (GATE_W, GATE_H))
                        gate_frame = Frame(
                            w=GATE_W, h=GATE_H, pixels=tuple(int(p) for p in g.flatten())
                        )
                        encoded, buf = cv2.imencode(".jpg", bgr)
                        jpeg = buf.tobytes() if encoded and buf is not None else b""
                        published = _put_frame(
                            q,
                            SourceEnvelope(
                                source_id=f"video:{uri}",
                                modality="video",
                                epoch=epoch,
                                sequence=sequence,
                                observed_time=observed_time,
                                source_time=source_time,
                                format=_video_format(bgr),
                                payload=("image/jpeg", jpeg),
                                signal=gate_frame,
                                legacy_timebase="observed"
                                if config.timebase == "wall"
                                else "source",
                            ),
                            config,
                            stop,
                            dropped,
                        )
                        if not published:
                            break
                        sequence += 1
                    i += 1
            finally:
                cap.release()
            if attempts >= config.reconnects:
                break
            attempts += 1
            epoch += 1
            time.sleep(config.reconnect_backoff_s)
    except Exception as exc:
        _put_terminal(q, exc, stop)
    finally:
        _put_terminal(q, None, stop)  # end sentinel never evicts a captured envelope


async def open_source(
    uri: str,
    *,
    fps: float = 4.0,
    maxsize: int = 64,
    reconnects: int = 0,
    reconnect_backoff_s: float = 0.5,
    rtsp_tcp: bool = True,
    queue_policy: "Literal['latest', 'block']" = "latest",
    timebase: "Literal['media', 'wall']" = "media",
    dropped: "list[int] | None" = None,
    health: "list | None" = None,
) -> AsyncIterator[SourceEnvelope]:
    """Threaded capture: read frames off a daemon thread (so capture never blocks the consumer/event
    loop), yield :class:`~percept.runtime.envelope.SourceEnvelope` items asynchronously. Stops the
    thread on consumer exit.

    The envelope always preserves observation time and source time independently.  ``timebase`` only
    selects the transitional tuple view; source time remains ``None`` when the decoder supplied none.
    ``queue_policy="latest"`` keeps live capture fresh by evicting stale queued frames (the default).
    ``queue_policy="block"`` is a bounded, lossless mode for deterministic finite-file evaluation.
    ``health`` is an optional shared list the capture thread appends A-4 ``fps_misreported`` events to."""
    if queue_policy not in ("latest", "block"):
        raise ValueError(f"unknown source queue policy: {queue_policy!r}")
    if queue_policy == "block" and maxsize <= 0:
        raise ValueError("queue_policy='block' requires a positive bounded maxsize")
    config = SourceConfig(
        fps=fps,
        queue_size=maxsize,
        reconnects=reconnects,
        reconnect_backoff_s=reconnect_backoff_s,
        rtsp_tcp=rtsp_tcp,
        queue_policy=queue_policy,
        timebase=timebase,
    )
    q: "queue.Queue" = queue.Queue(maxsize=config.queue_size)
    stop = threading.Event()
    thread = threading.Thread(
        target=_capture,
        args=(uri, config, q, stop, dropped),
        kwargs={"health": health},
        daemon=True,
    )
    thread.start()
    try:
        while True:
            item = await asyncio.to_thread(_get_polling, q)
            if item is _POLL:
                continue
            if item is None:
                break
            if isinstance(item, Exception):
                raise item
            yield item
    finally:
        stop.set()
