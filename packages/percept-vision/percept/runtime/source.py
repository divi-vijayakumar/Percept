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
from typing import AsyncIterator

GATE_W, GATE_H = 48, 36  # tier-0 downsample (still vs moved; full res not needed for the gate)


@dataclass(frozen=True)
class SourceConfig:
    """Runtime capture tuning for webcam/video/RTSP sources."""

    fps: float = 4.0
    queue_size: int = 64
    reconnects: int = 0
    reconnect_backoff_s: float = 0.5
    rtsp_tcp: bool = True


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


def _put_latest(q: "queue.Queue", item) -> None:
    """Put without letting a slow consumer make capture block forever."""
    try:
        q.put(item, timeout=0.1)
    except queue.Full:
        try:
            q.get_nowait()
        except queue.Empty:
            pass
        q.put(item, timeout=0.1)


def _open_capture(cv2, uri: str, config: SourceConfig):
    source = _normalize_uri(uri)
    if isinstance(source, str) and source.startswith("rtsp://") and config.rtsp_tcp:
        return cv2.VideoCapture(source, cv2.CAP_FFMPEG)
    return cv2.VideoCapture(source)


def _capture(uri: str, config: SourceConfig, q: "queue.Queue", stop: threading.Event) -> None:
    """Blocking capture loop, run on a daemon thread; pushes frames into ``q`` until stopped/EOF."""
    import cv2  # lazy: the [runtime]/[rtsp] extra

    from percept.edge.tier0 import Frame

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
            try:
                while not stop.is_set():
                    ok, bgr = cap.read()
                    if not ok:
                        break
                    if i % step == 0:
                        t = i / src_fps
                        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
                        g = cv2.resize(gray, (GATE_W, GATE_H))
                        gate_frame = Frame(w=GATE_W, h=GATE_H, pixels=tuple(int(p) for p in g.flatten()))
                        _, buf = cv2.imencode(".jpg", bgr)
                        _put_latest(q, (t, gate_frame, ("image/jpeg", buf.tobytes())))
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
) -> AsyncIterator[tuple]:
    """Threaded capture: read frames off a daemon thread (so capture never blocks the consumer/event
    loop), yield ``(t, gate_frame, judge_frame)`` asynchronously. Stops the thread on consumer exit."""
    config = SourceConfig(
        fps=fps,
        queue_size=maxsize,
        reconnects=reconnects,
        reconnect_backoff_s=reconnect_backoff_s,
        rtsp_tcp=rtsp_tcp,
    )
    q: "queue.Queue" = queue.Queue(maxsize=config.queue_size)
    stop = threading.Event()
    thread = threading.Thread(target=_capture, args=(uri, config, q, stop), daemon=True)
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
