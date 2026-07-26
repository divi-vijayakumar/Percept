"""C1 event capsule — fold the recent frame window into ONE sprite montage used AS the judge frame.

Why: the SDK's single-flight judge scheduler (active + one latest-pending) judges the *freshest*
admitted frame and supersedes older ones. A brief event (a sip, a hand-raise, a sugar pour ~1–2 s)
is shorter than one request/response judge (~3–5 s), so the salient frame is dropped before it is
ever judged, so the watch never fires.

Fix (coverage, not latency): make every judge frame a montage of the recent window — {first,
peak-salience, last}. Whatever frame the scheduler happens to judge now *carries the event inside
it*, so the pour can't be superseded away. One image's token cost via ``percept.sprite.make_sprite``.
Latency stays at the request/response floor (~3–5 s); that is the accepted trade.

Pure numpy + PIL + the SDK's ``make_sprite`` — deliberately NO pipecat/cv2 import, so both the live
``media.VideoTee`` and the headless (cv2-based) video test can use it without the libav clash.
"""
from __future__ import annotations

import io
from collections import deque
from typing import Any

import numpy as np
from PIL import Image

from percept.sprite import make_sprite


class CapsuleBuilder:
    """Rolling event-capsule builder. Push each captured RGB frame; read the current montage as the
    judge frame. Bounded memory (tiles are downsized and windowed)."""

    def __init__(self, *, window: int = 6, tiles: int = 3, tile_px: int = 512,
                 jpeg_quality: int = 70) -> None:
        self._tiles = max(1, tiles)
        self._tile_px = tile_px
        self._quality = jpeg_quality
        self._buf: deque = deque(maxlen=window)   # (tile_rgb_uint8, salience)
        self._prev_gray: Any = None

    def push(self, rgb: Any) -> None:
        """Add one RGB ndarray (HxWx3, uint8-ish). Computes cheap frame-diff salience for peak pick."""
        pil = Image.fromarray(np.asarray(rgb).astype(np.uint8), "RGB")
        gray = np.asarray(pil.convert("L").resize((96, 54)), dtype=np.int16)
        sal = 0.0 if self._prev_gray is None else float(np.abs(gray - self._prev_gray).mean())
        self._prev_gray = gray
        h, w = np.asarray(rgb).shape[:2]
        sc = self._tile_px / max(h, w)
        tile = pil.resize((max(1, int(w * sc)), max(1, int(h * sc)))) if sc < 1 else pil
        self._buf.append((np.asarray(tile).astype(np.uint8), sal))

    def ready(self) -> bool:
        return len(self._buf) > 0

    def montage_jpeg(self) -> tuple[str, bytes]:
        """Current capsule as a ('image/jpeg', bytes) pair: {first, peak-salience, last} tiles → montage."""
        items = list(self._buf)
        first = items[0][0]
        last = items[-1][0]
        peak = max(items, key=lambda it: it[1])[0]
        # de-dup by identity so a tiny window doesn't tile the same frame three times
        picks: list = []
        for tile in (first, peak, last):
            if not any(tile is p for p in picks):
                picks.append(tile)
            if len(picks) >= self._tiles:
                break
        sprite = make_sprite(picks, max_cols=self._tiles)
        arr = sprite if sprite is not None else np.concatenate(picks, axis=1)
        bio = io.BytesIO()
        Image.fromarray(np.asarray(arr).astype(np.uint8), "RGB").save(
            bio, format="JPEG", quality=self._quality)
        return ("image/jpeg", bio.getvalue())


__all__ = ["CapsuleBuilder"]
