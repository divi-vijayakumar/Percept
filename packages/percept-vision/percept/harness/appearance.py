"""Appearance signatures for re-identification — the embedder the ``EntityGraph`` / ``AppearanceIdentity``
seam expects (``embedder(frame, box) -> Signature``).

A cheap, dependency-light colour descriptor: the joint HSV (hue×saturation×value) histogram of a
detection's box crop, L1-normalised and flattened to a plain tuple of floats — the opaque ``Signature``
the core compares by cosine. This is the measures-layer encoder that turns a person crop into an
appearance embedding for re-id; it is the validated probe mechanism (hold the subject by appearance
through tracking gaps), now reusable.

The value axis is part of the *joint* distribution rather than a separate marginal appended to a
hue×saturation histogram. Marginals cannot do this job: every achromatic pixel (black, grey, white)
shares one hue×saturation cell, so with concatenated marginals a black subject and a white subject
still overlap on the whole chroma block, and no fixed weighting of the two blocks separates
black-from-white and red-from-blue at the same time — pushing the weight onto value to split the
greys collapses the hues, and vice versa. The joint histogram puts them in different cells outright.
The cost is dimensionality: ``bins × bins × vbins`` (2048 at the defaults, versus 256 for the
hue×saturation descriptor this replaces), which is why the value axis is quantised coarsely — value
is the axis illumination moves most, so it is the least identity-bearing one and deserves the
fewest bins.

It lives here (not in the pure-stdlib core) because it needs numpy/opencv; the core only ever sees the
resulting float tuple, so injecting it keeps the cognition layer dependency-free.

Every ``None`` return is a distinct cause, and each one is named on this module's logger at DEBUG
(``logging.getLogger("percept.harness.appearance")``). The return value stays a bare ``None`` because
the caller contract is "no appearance evidence, degrade to pure continuity"; the log is how an
operator tells "this deployment has no opencv" apart from "this one crop was unusable".
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

#: Above this share of the mass in a single bin the crop is treated as carrying no identity
#: information: a one-cell descriptor scores cosine 1.0 against any other one-cell descriptor that
#: quantises to the same cell, so it asserts a maximally confident match on a crop with no structure
#: (a blank wall, a blown-out region, a subject in shadow). Better to hand the graph nothing.
DEGENERATE_BIN_MASS = 0.98

#: Upper bound on either histogram axis. Not a tuning knob — a guard so a mistyped ``bins`` cannot
#: turn into a multi-gigabyte allocation inside ``calcHist``.
MAX_BINS = 64


def _bad_axis(n: Any) -> bool:
    """True when ``n`` is not a usable histogram axis length."""
    return not isinstance(n, int) or isinstance(n, bool) or n < 1 or n > MAX_BINS


def appearance_signature(
    frame: Any, box: Any, *, bins: int = 16, vbins: int = 8
) -> tuple[float, ...] | None:
    """Joint HSV histogram of ``box``'s region of ``frame``, L1-normalised → a tuple of floats.

    ``box`` is normalised (``x, y, w, h`` in 0..1); ``frame`` is an ``HxWx3`` BGR array of any dtype.
    The signature has ``bins * bins * vbins`` dimensions (hue × saturation × value).

    Returns ``None`` — never raises, never a partial descriptor — when there is no appearance evidence
    to give, so the graph degrades to pure continuity. The causes, each logged at DEBUG:

    * numpy/opencv are not installed in this deployment;
    * ``bins``/``vbins`` are not integers in ``1..MAX_BINS``;
    * ``frame`` is not an ``HxWx3`` array (a 4-channel BGRA frame is refused, not silently stripped:
      its alpha cannot be honoured without compositing, and histogramming un-composited pixels would
      count fully transparent ones as black);
    * the crop is empty, off-frame, or smaller than 2×2;
    * the crop is (near-)uniform — more than :data:`DEGENERATE_BIN_MASS` of the mass in one bin.

    Frames that are not ``uint8`` are normalised to ``uint8`` BGR first. This matters because
    ``cv2.cvtColor(..., COLOR_BGR2HSV)`` emits hue in 0..360 for float input but 0..180 for ``uint8``:
    normalising the input keeps one histogram range instead of a dtype-conditional one that has to
    stay in step with cvtColor's conventions. Float frames carry a real 0..1 vs 0..255 ambiguity; it is
    resolved from the **whole frame's** maximum (not the crop's), so every box on a frame is scaled
    identically — a frame peaking at or below 1.0 is read as 0..1 and scaled by 255, otherwise it is
    read as already being 0..255. (A 0..255 float frame that really does peak below 1.0 is black, and
    is refused as degenerate either way.) See :func:`_as_uint8_bgr` for the wider-depth cases.
    """
    try:
        import cv2
        import numpy as np
    except Exception:
        logger.debug("appearance_signature -> None: numpy/opencv unavailable in this deployment")
        return None
    if _bad_axis(bins) or _bad_axis(vbins):
        logger.debug(
            "appearance_signature -> None: bins=%r/vbins=%r are not integers in 1..%d", bins, vbins, MAX_BINS
        )
        return None
    arr = np.asarray(frame)
    if arr.ndim != 3 or arr.shape[2] != 3:
        logger.debug(
            "appearance_signature -> None: frame is not an HxWx3 BGR array (shape=%r)",
            getattr(arr, "shape", None),
        )
        return None
    h, w = arr.shape[:2]
    x0, y0 = max(0, int(box.x * w)), max(0, int(box.y * h))
    x1, y1 = min(w, int((box.x + box.w) * w)), min(h, int((box.y + box.h) * h))
    if x1 - x0 < 2 or y1 - y0 < 2:
        logger.debug("appearance_signature -> None: crop is empty/off-frame/smaller than 2x2")
        return None
    try:
        crop = _as_uint8_bgr(np, arr, arr[y0:y1, x0:x1])
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1, 2], None, [bins, bins, vbins], [0, 180, 0, 256, 0, 256])
    except Exception as exc:  # an object/complex/otherwise unconvertible buffer that got this far
        logger.debug("appearance_signature -> None: %s buffer could not be converted (%s)", arr.dtype, exc)
        return None
    total = float(hist.sum())
    if total <= 0.0:
        logger.debug("appearance_signature -> None: histogram is empty (no pixel fell in range)")
        return None
    norm = hist / total
    dominant = float(norm.max())
    if dominant > DEGENERATE_BIN_MASS:
        logger.debug(
            "appearance_signature -> None: crop is near-uniform (%.4f of the mass in one bin) and carries "
            "no identity information", dominant
        )
        return None
    # ``tolist`` converts in C and already yields plain Python floats — a per-element Python loop over
    # the (now 8x larger) descriptor would be the most expensive line in the embedder.
    return tuple(norm.ravel().tolist())


def _as_uint8_bgr(np: Any, frame: Any, crop: Any) -> Any:
    """``crop`` as ``uint8`` 0..255, with the scale decided once from the whole ``frame``.

    Deciding per-frame rather than per-crop keeps every box on one frame on the same scale, so two
    detections in one image stay comparable. One rule covers every dtype, keyed on the frame's peak:
    at most 1.0 is the 0..1 convention; at most 255 is already 8-bit-ranged whatever the container
    dtype (an ``int32`` buffer holding 0..255 is not a 31-bit image); above that it is a wider-depth
    image, scaled from its dtype's full range (or, for floats, from its own peak — the only scale
    there is).
    """
    if crop.dtype == np.uint8:
        return crop
    if np.issubdtype(frame.dtype, np.floating):
        finite = frame[np.isfinite(frame)]
        peak = float(finite.max()) if finite.size else 0.0
    else:
        peak = float(frame.max()) if frame.size else 0.0
    if peak <= 1.0:
        scale = 255.0
    elif peak <= 255.0:
        scale = 1.0
    elif np.issubdtype(crop.dtype, np.integer):
        scale = 255.0 / float(np.iinfo(crop.dtype).max)
    else:
        scale = 255.0 / peak
    out = np.nan_to_num(crop.astype(np.float64), nan=0.0, posinf=255.0, neginf=0.0) * scale
    return np.clip(np.rint(out), 0.0, 255.0).astype(np.uint8)


__all__ = ["appearance_signature", "DEGENERATE_BIN_MASS", "MAX_BINS"]
