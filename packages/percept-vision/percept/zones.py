"""Zones: where in the frame a condition looks.

A zone is a polygon in normalised image coordinates (``x`` right, ``y`` down, both in
``[0, 1]``). A pixel belongs to a polygon when its centre does (even-odd rule), so the same
polygon gives a consistent mask at any resolution.

Composition: no include zones means the whole frame; exclude zones are subtracted from it.

Pure: numpy only, no I/O.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from percept.contracts import Polygon

__all__ = ["crop_box", "polygon_mask", "zone_mask"]


def polygon_mask(polygon: Polygon, width: int, height: int) -> np.ndarray:
    """Boolean ``height x width`` mask of the pixels whose centres lie inside ``polygon``."""
    if width <= 0 or height <= 0:
        raise ValueError(f"mask size must be positive, got {width}x{height}")
    px = (np.arange(width, dtype=np.float64) + 0.5) / width
    py = (np.arange(height, dtype=np.float64) + 0.5) / height
    x, y = np.meshgrid(px, py)
    inside = np.zeros((height, width), dtype=bool)
    n = len(polygon)
    for i in range(n):
        xi, yi = float(polygon[i][0]), float(polygon[i][1])
        xj, yj = float(polygon[i - 1][0]), float(polygon[i - 1][1])
        if yi == yj:
            continue  # a horizontal edge never crosses a horizontal ray
        crosses = (yi > y) != (yj > y)
        x_cross = (xj - xi) * (y - yi) / (yj - yi) + xi
        inside ^= crosses & (x < x_cross)
    return inside


@lru_cache(maxsize=256)
def _zone_mask(zones: tuple[Polygon, ...], exclude: tuple[Polygon, ...], width: int, height: int) -> np.ndarray:
    if zones:
        mask = np.zeros((height, width), dtype=bool)
        for polygon in zones:
            mask |= polygon_mask(polygon, width, height)
            # A zone too small or thin to cover a pixel centre still watches the pixel at its
            # vertex centroid, rather than going silently blind.
            cx = sum(p[0] for p in polygon) / len(polygon)
            cy = sum(p[1] for p in polygon) / len(polygon)
            mask[min(int(cy * height), height - 1), min(int(cx * width), width - 1)] = True
    else:
        mask = np.ones((height, width), dtype=bool)
    for polygon in exclude:
        mask &= ~polygon_mask(polygon, width, height)
    mask.flags.writeable = False
    return mask


def zone_mask(
    zones: tuple[Polygon, ...] = (),
    exclude_zones: tuple[Polygon, ...] = (),
    width: int = 48,
    height: int = 36,
) -> np.ndarray:
    """The pixels a condition watches: the union of ``zones`` (or everything) minus ``exclude_zones``.

    Every include polygon covers at least the pixel holding its vertex centroid; exclude zones
    may still remove everything.

    Returns a read-only boolean ``height x width`` array. Results are memoised, since a
    condition's zones are fixed and the same grid is asked for on every frame.
    """
    norm = lambda polys: tuple(tuple((float(x), float(y)) for x, y in p) for p in polys)  # noqa: E731
    return _zone_mask(norm(zones), norm(exclude_zones), int(width), int(height))


def crop_box(
    zones: tuple[Polygon, ...], width: int, height: int, margin: float = 0.05
) -> tuple[int, int, int, int]:
    """Pixel box ``(x0, y0, x1, y1)``, half-open, around the include ``zones`` for cropping.

    The box is the bounding box of every include polygon, grown by ``margin`` (a fraction of
    the frame's width and height) on each side so the judge sees a little context, then
    clamped to the frame. No zones means the whole frame. Exclude zones are not cut out of a
    crop: a rectangle can't express them, and the judge's prompt carries the condition text.
    """
    if width <= 0 or height <= 0:
        raise ValueError(f"frame size must be positive, got {width}x{height}")
    if not zones:
        return 0, 0, width, height
    xs = [float(p[0]) for poly in zones for p in poly]
    ys = [float(p[1]) for poly in zones for p in poly]
    x0 = min(width - 1, max(0, int(np.floor((min(xs) - margin) * width))))
    y0 = min(height - 1, max(0, int(np.floor((min(ys) - margin) * height))))
    x1 = min(width, int(np.ceil((max(xs) + margin) * width)))
    y1 = min(height, int(np.ceil((max(ys) + margin) * height)))
    return x0, y0, max(x1, x0 + 1), max(y1, y0 + 1)
