"""Native short-video evidence for motion-sensitive judges.

Ordered images and sprites are useful, lower-fidelity evidence shapes. They
are not substitutes for motion. ``NativeVideoVision`` converts a bounded
retained frame window into one MP4 before invoking a video-capable judge, so
the provider receives a real temporal clip and the surrounding ``JudgePool``
deadline includes clip assembly.

OpenCV and NumPy are imported lazily. This module therefore keeps the SDK core
importable without extras while making the feature available through
``percept-vision[runtime]``.
"""

from __future__ import annotations

import asyncio
import base64
import math
import tempfile
import threading
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from ..adapters import Box, Verdict


def _encoded_image_bytes(frame: Any) -> bytes:
    if isinstance(frame, tuple) and len(frame) == 2:
        media_type, payload = frame
        if not str(media_type).lower().startswith("image/"):
            raise TypeError(
                "native video assembly accepts encoded image frames only"
            )
        if isinstance(payload, (bytes, bytearray)):
            return bytes(payload)
    if isinstance(frame, dict) and frame.get("type") == "image":
        source = frame.get("source", {})
        if not str(source.get("media_type", "image/jpeg")).lower().startswith(
            "image/"
        ):
            raise TypeError(
                "native video assembly accepts encoded image frames only"
            )
        data = source.get("data")
        if isinstance(data, str):
            return base64.b64decode(data, validate=True)
    if isinstance(frame, (bytes, bytearray)):
        return bytes(frame)
    raise TypeError(
        "native video assembly needs JPEG/PNG bytes, an image media tuple, "
        f"or an image block; got {type(frame).__name__}"
    )


def encode_mp4_clip(
    frames: Sequence[Any],
    *,
    fps: float = 4.0,
    max_frames: int = 16,
    max_dimension: int = 768,
    max_bytes: int = 12 * 1024 * 1024,
    max_frame_bytes: int = 8 * 1024 * 1024,
    max_input_bytes: int = 32 * 1024 * 1024,
    max_pixels: int = 16_000_000,
) -> bytes:
    """Encode two or more image frames as one bounded MP4 byte string."""
    if (
        isinstance(fps, bool)
        or not isinstance(fps, (int, float))
        or not math.isfinite(float(fps))
        or float(fps) <= 0
    ):
        raise ValueError("fps must be a positive finite number")
    for name, value, minimum in (
        ("max_frames", max_frames, 2),
        ("max_dimension", max_dimension, 16),
        ("max_bytes", max_bytes, 1),
        ("max_frame_bytes", max_frame_bytes, 1),
        ("max_input_bytes", max_input_bytes, 1),
        ("max_pixels", max_pixels, 1),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < minimum
        ):
            raise ValueError(f"{name} must be an integer >= {minimum}")

    total_frames = len(frames)
    selected = [
        frames[index]
        for index in range(max(0, total_frames - max_frames), total_frames)
    ]
    if len(selected) < 2:
        raise ValueError("native video evidence requires at least two frames")

    try:
        import cv2  # noqa: PLC0415 - [runtime] extra, intentionally lazy
        import numpy as np  # noqa: PLC0415 - [runtime] extra, intentionally lazy
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "native video evidence needs percept-vision[runtime]"
        ) from exc

    encoded = []
    input_bytes = 0
    for item in selected:
        raw = _encoded_image_bytes(item)
        if len(raw) > max_frame_bytes:
            raise ValueError(
                "one retained frame exceeds "
                f"max_frame_bytes={max_frame_bytes}"
            )
        input_bytes += len(raw)
        if input_bytes > max_input_bytes:
            raise ValueError(
                "retained frames exceed "
                f"max_input_bytes={max_input_bytes}"
            )
        encoded.append(raw)

    decoded = []
    for raw in encoded:
        image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("one retained frame is not a decodable image")
        height, width = image.shape[:2]
        if height * width > max_pixels:
            raise ValueError(
                f"one retained frame exceeds max_pixels={max_pixels}"
            )
        decoded.append(image)

    height, width = decoded[0].shape[:2]
    scale = min(1.0, max_dimension / max(height, width))
    output_size = (
        max(2, int(round(width * scale)) // 2 * 2),
        max(2, int(round(height * scale)) // 2 * 2),
    )

    handle = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    output_path = Path(handle.name)
    handle.close()
    writer = None
    try:
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            float(fps),
            output_size,
        )
        if not writer.isOpened():
            raise RuntimeError("OpenCV could not initialize an MP4 encoder")
        for image in decoded:
            if (image.shape[1], image.shape[0]) != output_size:
                image = cv2.resize(image, output_size)
            writer.write(image)
        writer.release()
        writer = None
        payload = output_path.read_bytes()
    finally:
        if writer is not None:
            writer.release()
        output_path.unlink(missing_ok=True)

    if not payload:
        raise RuntimeError("MP4 encoder produced no evidence")
    if len(payload) > max_bytes:
        raise ValueError(
            f"native video evidence exceeds max_bytes={max_bytes}"
        )
    return payload


class NativeVideoVision:
    """Vision adapter that delivers retained images as one native video clip."""

    def __init__(
        self,
        adapter: Any,
        *,
        fps: float = 4.0,
        max_frames: int = 16,
        encoder: Callable[..., bytes] = encode_mp4_clip,
        require_native_transport: bool = False,
    ) -> None:
        if not callable(getattr(adapter, "judge", None)):
            raise ValueError("NativeVideoVision adapter needs judge()")
        if require_native_transport and not bool(
            getattr(adapter, "SUPPORTS_NATIVE_VIDEO_TRANSPORT", False)
        ):
            raise ValueError(
                "NativeVideoVision requires an adapter that declares "
                "SUPPORTS_NATIVE_VIDEO_TRANSPORT"
            )
        self._adapter = adapter
        self._fps = fps
        self._max_frames = max_frames
        self._encoder = encoder
        # Encoding runs outside the event loop. A non-blocking semaphore keeps
        # cancellation-hostile encoder work from becoming an unbounded thread
        # queue after a reflex deadline has already expired.
        self._encode_slot = threading.BoundedSemaphore(1)

    def _encode(self, frames: list[Any], fps: float) -> bytes:
        if not self._encode_slot.acquire(blocking=False):
            raise RuntimeError("native video encoder capacity exhausted")
        try:
            return self._encoder(
                frames,
                fps=fps,
                max_frames=self._max_frames,
            )
        finally:
            self._encode_slot.release()

    async def judge(
        self,
        condition: str,
        frame: Any,
        *,
        roi: Box | None = None,
        exemplar: Any = None,
        corrections: dict | None = None,
        frames: list | None = None,
        audio: Any = None,
        context: dict | None = None,
    ):
        if audio is not None:
            raise ValueError(
                "NativeVideoVision is a video_clip route, not fused evidence"
            )
        history = [item for item in (frames or []) if item is not frame]
        clip_frames = [*history, frame]
        if len(clip_frames) < 2:
            return Verdict(
                satisfied=False,
                confidence=0.0,
                evidence=(
                    "native video evidence unavailable: at least two "
                    "retained frames are required"
                ),
            )
        clip_fps = self._fps
        temporal_span_s = (context or {}).get("_percept_temporal_span_s")
        if (
            isinstance(temporal_span_s, (int, float))
            and not isinstance(temporal_span_s, bool)
            and math.isfinite(float(temporal_span_s))
            and float(temporal_span_s) > 0
        ):
            # Preserve the actual first-to-last source span. For N frames,
            # (N-1)/fps is the interval between the first and last samples.
            clip_fps = max(
                0.25,
                min(
                    30.0,
                    (len(clip_frames) - 1) / float(temporal_span_s),
                ),
            )
        payload = await asyncio.to_thread(
            self._encode,
            clip_frames,
            clip_fps,
        )
        clip_context = {
            **(context or {}),
            "_percept_evidence_shape": "video_clip",
            "_percept_clip_frame_count": min(
                len(clip_frames),
                self._max_frames,
            ),
            "_percept_clip_fps": clip_fps,
        }
        return await self._adapter.judge(
            condition,
            ("video/mp4", payload),
            roi=roi,
            exemplar=exemplar,
            corrections=corrections,
            frames=None,
            audio=None,
            context=clip_context,
        )

    async def detect(self, frame: Any, goals: list[Any]) -> list[Any]:
        method = getattr(self._adapter, "detect", None)
        if not callable(method):
            return []
        return await method(frame, goals)

    async def detect_events(
        self,
        frames: list,
        goals: list[Any],
    ) -> list[Any]:
        method = getattr(self._adapter, "detect_events", None)
        if not callable(method):
            return []
        return await method(frames, goals)

    def take_terminal_error(self) -> BaseException | None:
        method = getattr(self._adapter, "take_terminal_error", None)
        return method() if callable(method) else None


__all__ = ["NativeVideoVision", "encode_mp4_clip"]
