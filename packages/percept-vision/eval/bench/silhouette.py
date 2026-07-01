"""Silhouette-PCA rep counter — the OFFLINE, REGIME-SPECIFIC fallback for occluded whole-body motion.

Lifted from the validated POC (``scratch/silhouette_pca.py`` + the motion helpers in
``scratch/generic_motion_hero_tracker.py``) into tracked eval code. It exists for ONE case the pose path
can't reach: an action whose counting joints are occluded (pommel-horse legs ≈ 0 keypoint confidence) but
whose whole-body silhouette is strongly periodic. The rhythm hero routes to it ONLY when the measured
prior-channel keypoint coverage is low — never by action name.

What it is (and what it is NOT):
  * **Offline / batch** — it builds a median background and runs SVD over the whole-clip mask matrix. This
    is an analysis-mode counter, NOT part of the streaming core. A real-time form would need an incremental
    PCA / windowed reformulation; explicitly out of scope here.
  * **Static-camera** — background subtraction assumes the camera doesn't move. False for body-cam / PTZ.
  * **Principled, not tuned** — the ROI is inferred from motion heat, the principal component is chosen by
    its OWN periodicity (autocorr), and the refractory gap is derived from that component's period. One knob
    set for every clip (no hardcoded PC index, no per-clip ROI) — that is what keeps it from overfitting.

It also exports :func:`dominant_period`, the autocorr periodicity confidence the A/B runner uses on the POSE
signal too, so "is this actually periodic?" is measured in the SAME units for both phase sources.
"""

from __future__ import annotations

GRID = 32           # silhouette masks are resized to GRID×GRID before stacking
TOPK = 6            # consider the top-K principal components; pick by periodicity (NOT a fixed index)
GAP_FRAC = 0.6      # refractory = GAP_FRAC × the component's OWN period (derived, not hardcoded seconds)
PROM_FRAC = 0.08    # prominence floor as a fraction of signal range (same for every clip)


def dominant_period(sig, fps):
    """Autocorrelation period of a 1-D signal -> ``(period_s | None, confidence)``. The confidence is the
    autocorr peak height in (0, 1] — a clean periodic signal peaks near 1, noise stays low. Used for BOTH
    the silhouette PC and the pose channel so the periodicity gate is comparable across phase sources."""
    import numpy as np

    y = np.asarray(sig, float)
    if y.std() < 1e-9:
        return None, 0.0
    y = (y - y.mean()) / (y.std() + 1e-9)
    ac = np.correlate(y, y, "full")[len(y) - 1:]
    ac = ac / ac[0]
    lo, hi = max(2, int(round(0.35 * fps))), min(len(ac) - 1, int(round(4.5 * fps)))
    peaks = [(float(ac[k]), k) for k in range(lo, hi) if ac[k] > ac[k - 1] and ac[k] >= ac[k + 1]]
    if not peaks:
        return None, 0.0
    conf, lag = max(peaks)
    return lag / fps, conf


def _moving_average(v, window):
    import numpy as np

    v = np.asarray(v, float)
    if window <= 1:
        return v
    pad = window // 2
    return np.convolve(np.pad(v, (pad, pad), "edge"), np.ones(window) / window, "valid")[:len(v)]


def _count_extrema(sig, fps, period):
    """Count prominent extrema with a refractory gap derived from the signal's own period."""
    import numpy as np

    y = _moving_average(sig, 5)
    amp = float(np.percentile(y, 90) - np.percentile(y, 10))
    if amp <= 1e-9:
        return 0
    mg = max(1, int(round(GAP_FRAC * period * fps)))
    prom = PROM_FRAC * amp
    med = float(np.median(y))
    ev = []
    for i in range(1, len(y) - 1):
        peak = y[i] >= y[i - 1] and y[i] > y[i + 1]
        trough = y[i] <= y[i - 1] and y[i] < y[i + 1]
        if not (peak or trough):
            continue
        loc = y[max(0, i - mg):min(len(y), i + mg + 1)]
        if max(abs(y[i] - loc.min()), abs(y[i] - loc.max())) < prom:
            continue
        if ev and i - ev[-1] < mg:
            if abs(y[i] - med) > abs(y[ev[-1]] - med):
                ev[-1] = i
            continue
        ev.append(i)
    return len(ev)


# --- motion-foreground helpers (static-camera background subtraction) ------------------------------------

def _build_background(video, roi, samples=80):
    import cv2
    import numpy as np

    cap = cv2.VideoCapture(str(video))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    x, y, w, h = roi
    frames = []
    for fid in np.linspace(0, max(0, n - 1), min(samples, max(1, n)), dtype=int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fid))
        ok, fr = cap.read()
        if ok:
            frames.append(fr[y:y + h, x:x + w])
    cap.release()
    if not frames:
        raise ValueError("no frames")
    return np.median(np.stack(frames).astype(np.float32), axis=0).astype(np.uint8)


def _motion_mask(roi_frame, bg):
    import cv2
    import numpy as np

    diff = cv2.absdiff(roi_frame, bg)
    gray = cv2.GaussianBlur(cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    floor = (gray > 24).astype(np.uint8) * 255
    mask = cv2.bitwise_and(mask, floor)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((13, 13), np.uint8))


def infer_motion_roi(video):
    """The dominant-motion region of a static-camera clip — the routine ROI, inferred (not hand-cropped)."""
    import cv2
    import numpy as np

    cap = cv2.VideoCapture(str(video))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    ok, first = cap.read()
    h, w = first.shape[:2]
    cap.release()
    full = (0, 0, w, h)
    bg = _build_background(video, full, 60)
    heat = np.zeros((h, w), np.float32)
    cap = cv2.VideoCapture(str(video))
    for fid in np.linspace(0, max(0, n - 1), min(120, max(1, n)), dtype=int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fid))
        ok, fr = cap.read()
        if not ok:
            continue
        heat += (_motion_mask(fr, bg) > 0).astype(np.float32)
    cap.release()
    heat = cv2.GaussianBlur(heat, (0, 0), 9)
    thr = max(2.0, float(np.percentile(heat, 88)))
    hot = (heat >= thr).astype(np.uint8) * 255
    hot = cv2.morphologyEx(hot, cv2.MORPH_CLOSE, np.ones((31, 31), np.uint8))
    num, _, stats, _ = cv2.connectedComponentsWithStats(hot)
    best = None
    for label in range(1, num):
        x, y, bw, bh, area = (int(v) for v in stats[label])
        if area < 300:
            continue
        border = x <= 2 or y <= 2 or x + bw >= w - 2 or y + bh >= h - 2
        aspect = bw / max(1, bh)
        compact = area / max(1, bw * bh)
        score = area * compact
        if border:
            score *= 0.35
        if aspect > 6 or aspect < 0.15:
            score *= 0.5
        if best is None or score > best[0]:
            best = (score, (x, y, bw, bh))
    if best is None:
        return full
    x, y, bw, bh = best[1]
    px, py = int(0.18 * bw) + 20, int(0.25 * bh) + 20
    return (max(0, x - px), max(0, y - py),
            min(w, x + bw + px) - max(0, x - px), min(h, y + bh + py) - max(0, y - py))


def _silhouette_matrix(video, roi):
    import cv2
    import numpy as np

    bg = _build_background(video, roi)
    x0, y0, rw, rh = roi
    cap = cv2.VideoCapture(str(video))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    rows = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        mask = _motion_mask(frame[y0:y0 + rh, x0:x0 + rw], bg)
        rows.append(cv2.resize(mask, (GRID, GRID)).astype(np.float32).flatten())
    cap.release()
    return fps, np.stack(rows)


def silhouette_count(video, roi=None) -> dict:
    """Count whole-body reps via the most-periodic silhouette principal component.

    Returns ``{count, period_confidence, period_s, pc, roi, dur}``. ``count`` is 0 / ``period_confidence``
    0.0 when no component is periodic (so the caller can refuse a non-periodic fallback). ``roi`` is
    inferred from motion heat unless supplied."""
    import numpy as np

    roi = roi if roi is not None else infer_motion_roi(video)
    fps, X = _silhouette_matrix(video, roi)
    dur = len(X) / fps
    X = X - X.mean(axis=0)
    U, S, _ = np.linalg.svd(X, full_matrices=False)
    scores = U[:, :TOPK] * S[:TOPK]
    ranked = []
    for j in range(min(TOPK, scores.shape[1])):
        period, conf = dominant_period(scores[:, j], fps)
        if period is not None:
            ranked.append((conf, j, period))
    if not ranked:
        return {"count": 0, "period_confidence": 0.0, "period_s": None, "pc": None, "roi": roi, "dur": dur}
    conf, pc, period = max(ranked)  # the MOST-periodic component, not a fixed index
    return {
        "count": _count_extrema(scores[:, pc], fps, period),
        "period_confidence": float(conf),
        "period_s": float(period),
        "pc": int(pc),
        "roi": roi,
        "dur": dur,
    }


__all__ = ["silhouette_count", "dominant_period", "infer_motion_roi", "GRID", "TOPK"]
