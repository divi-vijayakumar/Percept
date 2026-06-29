"""Sprite — Build 6: fold N frames into ONE montage image, temporal context at ~single-frame token cost.

A video-LLM charges per image. A temporal *strip* (N frames) gives the eye motion/ordering context but
costs ~N images of tokens; a *sprite* lays those frames out as one grid image, so the VLM sees the
sequence at the price of (about) one image. The investigation measured the trade: for "what happened /
what are the steps" synthesis, a sprite cut observation redundancy 3.0->1.5 at ~single-frame token cost,
while a strip's per-frame fidelity (1.0) cost ~3.6x the tokens. So:

  * Use a sprite for **what / steps** synthesis (token-budgeted temporal context).
  * Use a strip when **ordering, small motion, or fine detail / exact boundary timing** matters — a grid
    trades temporal resolution for context.
  * **VLM-only.** Edge detectors read the original measurement stream, never a montage (see the brief).

Pure numpy (imported lazily so the stdlib core stays import-clean — numpy is already an extra for the
vision/bench paths that produce real frames). Returns ``None`` for non-image input so the caller can fall
back to a plain strip.
"""

from __future__ import annotations

import math


def make_sprite(frames, max_cols: int = 3, pad: int = 2, bg: int = 0):
    """Lay out image ``frames`` (HxW or HxWxC uint8-ish arrays) into a single grid montage.

    Tiles are placed left→right, top→bottom (reading order = time order) in a grid at most ``max_cols``
    wide, separated by ``pad`` px of ``bg``. Mismatched sizes are nearest-neighbour resized to the largest
    frame's H×W so the grid is regular. Returns one ``uint8`` ndarray, or ``None`` if no frame is a usable
    image (so the caller degrades to a strip)."""
    try:
        import numpy as np
    except ImportError:  # numpy is an extra; without it there is no montage, fall back to a strip
        return None

    arrs = []
    for f in frames or []:
        a = np.asarray(f)
        if a.ndim >= 2 and a.shape[0] > 0 and a.shape[1] > 0:
            arrs.append(a)
    if not arrs:
        return None

    th = max(a.shape[0] for a in arrs)
    tw = max(a.shape[1] for a in arrs)
    ch = 3 if any(a.ndim == 3 and a.shape[2] >= 3 for a in arrs) else 1

    def to_tile(a):
        if a.ndim == 2:
            a = a[:, :, None]
        a = a[:, :, :ch] if a.shape[2] >= ch else np.repeat(a[:, :, :1], ch, axis=2)
        if a.shape[0] != th or a.shape[1] != tw:  # nearest-neighbour resize (no cv2 dependency)
            ys = (np.arange(th) * a.shape[0] // th).clip(0, a.shape[0] - 1)
            xs = (np.arange(tw) * a.shape[1] // tw).clip(0, a.shape[1] - 1)
            a = a[ys][:, xs]
        return a.astype(np.uint8)

    tiles = [to_tile(a) for a in arrs]
    n = len(tiles)
    cols = min(max_cols, n)
    rows = math.ceil(n / cols)
    height = rows * th + pad * (rows + 1)
    width = cols * tw + pad * (cols + 1)
    canvas = np.full((height, width, ch), bg, dtype=np.uint8)
    for i, tile in enumerate(tiles):
        r, c = divmod(i, cols)
        y = pad + r * (th + pad)
        x = pad + c * (tw + pad)
        canvas[y:y + th, x:x + tw, :] = tile
    return canvas


__all__ = ["make_sprite"]
