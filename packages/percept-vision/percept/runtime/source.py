"""Capture sources for the runtime worker.

``open_source(uri)`` opens a cv2-readable URI (file / RTSP / ``webcam:0`` / ``webcam://0``) on a **background thread** and
yields ``(t, gate_frame, judge_frame)``:
  * ``gate_frame`` — a small gray :class:`~percept.edge.tier0.Frame` for the motion gate (still-vs-moved
    doesn't need full resolution),
  * ``judge_frame`` — a ``("image/jpeg", bytes)`` pair the VLM judges.

cv2 (``opencv-python-headless``) is the ``[runtime]``/``[rtsp]`` extra — imported lazily, never in the core.
The hardened transport (rtsp-tcp, reconnect/backoff, avfoundation ``-video_size``, ONVIF) is the ``[rtsp]``
follow-up; v1 reuses the deterministic decode pattern from ``eval/count_video.py``.
"""
from __future__ import annotations

import asyncio
import queue
import threading
import time
from dataclasses import dataclass
from typing import AsyncIterator, Callable, Literal

GATE_W, GATE_H = 48, 36  # tier-0 downsample (still vs moved; full res not needed for the gate)

# A-4 delivered-fps misreport heuristic. Measure the RAW read rate over a rolling wall-clock window
# and compare it to the source's REPORTED CAP_PROP_FPS. A camera that reports 30 but delivers 15 keeps
# every Nth frame at half the intended rate — the gate/fovea-coast Nyquist guarantee silently halves.
# Only evaluated once ``>= _FPS_MIN_EVAL_S`` of wall time has actually elapsed, so an offline/instant
# fixture (span ~0) can never false-fire.
_FPS_WINDOW_S = 2.0      # rolling window the delivered rate is measured over
_FPS_MIN_EVAL_S = 1.0    # minimum elapsed wall time before we trust a measurement
_FPS_TOL = 0.25          # relative gap (|measured-reported|/reported) that counts as a misreport


@dataclass(frozen=True)
class SourceConfig:
    """Runtime capture tuning for webcam/video/RTSP sources."""

    fps: float = 4.0
    queue_size: int = 64
    reconnects: int = 0
    reconnect_backoff_s: float = 0.5
    rtsp_tcp: bool = True
    # Timestamp origin for the yielded ``t`` (A-clock switch, introduced NO-OP). ``"media"`` = today's
    # arithmetic byte-for-byte (``t = i / src_fps``, unclamped reported fps). ``"wall"`` = ``time.monotonic()``
    # stamped per KEPT frame. Default stays ``"media"`` — a later PR flips it.
    timebase: Literal["media", "wall"] = "media"


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


def _open_capture(cv2, uri: str, config: SourceConfig):
    source = _normalize_uri(uri)
    if isinstance(source, str) and source.startswith("rtsp://") and config.rtsp_tcp:
        return cv2.VideoCapture(source, cv2.CAP_FFMPEG)
    return cv2.VideoCapture(source)


def _capture(uri: str, config: SourceConfig, q: "queue.Queue", stop: threading.Event,
             dropped: "list[int] | None" = None, *, now: "Callable[[], float] | None" = None,
             health: "list | None" = None) -> None:
    """Blocking capture loop, run on a daemon thread; pushes frames into ``q`` until stopped/EOF.

    ``now`` is the wall clock used for ``timebase="wall"`` stamps AND for the A-4 delivered-fps measure
    (injectable so tests need no real sleeping; defaults to ``time.monotonic``). ``health`` mirrors the
    ``dropped`` shared-list telemetry pattern: the capture thread appends ``(kind, data)`` events (today
    only ``("fps_misreported", {"reported", "measured"})``) for the worker to surface via ``on_health``."""
    import cv2  # lazy: the [runtime]/[rtsp] extra

    from percept.edge.tier0 import Frame

    if now is None:
        now = time.monotonic
    attempts = 0
    try:
        while not stop.is_set():
            cap = _open_capture(cv2, uri, config)
            if not cap.isOpened():
                cap.release()
                raise RuntimeError(f"could not open runtime source: {uri!r}")
            src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            step = max(1, round(src_fps / config.fps))
            i = 0
            read_ts: "list[float]" = []   # wall times of recent raw reads (A-4 delivered-fps window)
            fps_reported = False           # emit the misreport at most once per open
            try:
                while not stop.is_set():
                    ok, bgr = cap.read()
                    if not ok:
                        break
                    # A-4: measure the RAW read rate over a rolling wall-clock window and flag a source
                    # whose delivered fps diverges from its reported CAP_PROP_FPS. Emit-only (does NOT
                    # touch `step`/`t`), so the media lane stays byte-for-byte. Guarded by `health`.
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
                                health.append(("fps_misreported",
                                               {"reported": src_fps, "measured": measured}))
                                fps_reported = True
                    if i % step == 0:
                        t = now() if config.timebase == "wall" else i / src_fps
                        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
                        g = cv2.resize(gray, (GATE_W, GATE_H))
                        gate_frame = Frame(w=GATE_W, h=GATE_H, pixels=tuple(int(p) for p in g.flatten()))
                        _, buf = cv2.imencode(".jpg", bgr)
                        _put_latest(q, (t, gate_frame, ("image/jpeg", buf.tobytes())), dropped)
                    i += 1
            finally:
                cap.release()
            if attempts >= config.reconnects:
                break
            attempts += 1
            time.sleep(config.reconnect_backoff_s)
    except Exception as exc:
        _put_latest(q, exc)
    finally:
        _put_latest(q, None)  # end sentinel


async def open_source(
    uri: str,
    *,
    fps: float = 4.0,
    maxsize: int = 64,
    reconnects: int = 0,
    reconnect_backoff_s: float = 0.5,
    rtsp_tcp: bool = True,
    timebase: "Literal['media', 'wall']" = "media",
    dropped: "list[int] | None" = None,
    health: "list | None" = None,
) -> AsyncIterator[tuple]:
    """Threaded capture: read frames off a daemon thread (so capture never blocks the consumer/event
    loop), yield ``(t, gate_frame, judge_frame)`` asynchronously. Stops the thread on consumer exit.

    ``timebase`` selects the ``t`` origin (``"media"`` = today's ``i/src_fps``; ``"wall"`` = monotonic).
    ``health`` is an optional shared list the capture thread appends A-4 ``fps_misreported`` events to."""
    config = SourceConfig(
        fps=fps,
        queue_size=maxsize,
        reconnects=reconnects,
        reconnect_backoff_s=reconnect_backoff_s,
        rtsp_tcp=rtsp_tcp,
        timebase=timebase,
    )
    q: "queue.Queue" = queue.Queue(maxsize=config.queue_size)
    stop = threading.Event()
    thread = threading.Thread(
        target=_capture, args=(uri, config, q, stop, dropped), kwargs={"health": health}, daemon=True
    )
    thread.start()
    try:
        while True:
            item = await asyncio.to_thread(q.get)
            if item is None:
                break
            if isinstance(item, Exception):
                raise item
            yield item
    finally:
        stop.set()
