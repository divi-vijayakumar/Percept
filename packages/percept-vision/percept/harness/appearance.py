"""Appearance signatures for re-identification — the embedder the ``EntityGraph`` / ``AppearanceIdentity``
seam expects (``embedder(frame, box) -> Signature``).

A cheap, dependency-light colour descriptor: the HSV (hue×saturation) histogram of a detection's box crop,
L1-normalised and flattened to a plain tuple of floats — the opaque ``Signature`` the core compares by
cosine. This is the measures-layer encoder that turns a person crop into an appearance embedding for re-id;
it is the validated probe mechanism (hold the subject by appearance through tracking gaps), now reusable.

It lives here (not in the pure-stdlib core) because it needs numpy/opencv; the core only ever sees the
resulting float tuple, so injecting it keeps the cognition layer dependency-free.
"""
from __future__ import annotations

from typing import Any


def appearance_signature(frame: Any, box: Any, *, bins: int = 16) -> tuple[float, ...] | None:
    """HSV hue×sat histogram of ``box``'s region of ``frame``, L1-normalised → a tuple of floats.

    ``box`` is normalised (``x, y, w, h`` in 0..1); ``frame`` is an ``HxWx3`` BGR array. Returns ``None`` for
    a degenerate/empty crop or a non-image frame (so the graph degrades to pure continuity, never crashes).
    """
    try:
        import cv2
        import numpy as np
    except Exception:
        return None
    arr = np.asarray(frame)
    if arr.ndim != 3 or arr.shape[2] < 3:
        return None
    h, w = arr.shape[:2]
    x0, y0 = max(0, int(box.x * w)), max(0, int(box.y * h))
    x1, y1 = min(w, int((box.x + box.w) * w)), min(h, int((box.y + box.h) * h))
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None
    hsv = cv2.cvtColor(arr[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [bins, bins], [0, 180, 0, 256])
    total = float(hist.sum())
    if total <= 0.0:
        return None
    return tuple(float(v) for v in (hist / total).flatten())


__all__ = ["appearance_signature"]
