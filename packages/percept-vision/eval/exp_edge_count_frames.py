"""Edge-count bench on RAW FRAMES — the edge path's real claim: a cheap MOTION signal counts reps where
the VLM phase signal can't (RepCount-Δ0).

For each rep clip it decodes frames → a downscaled grayscale motion signal (frame-diff, the tier-0 edge
computation, NO model) → counts reps two ways and compares to the ground-truth count:
  * peaks   — motion-periodicity (count motion oscillation peaks)
  * period  — autocorrelation dominant period (duration / period) — robust to the "2 peaks per rep"
              symmetry of a down-up rep; a candidate better-registered detector if peaks over-count
No API, no VLM — this is the *cheap edge signal*. Head-to-head vs the VLM is the cached exp_edge_count.

CLIPS: any ``eval/perception_probe/<name>/clip.json`` that has a ``count`` field (ground-truth reps) and
``frames``. Record with: ``make_probe_clip <video> --name bench --count 8 --condition bench-press`` (rep mode).

Run: PYTHONPATH=packages/percept-vision:packages/percept-harness python3.12 -m eval.exp_edge_count_frames
"""

from __future__ import annotations

import io
import json
from pathlib import Path

PROBE_DIR = Path(__file__).resolve().parent / "perception_probe"
GW, GH = 48, 36  # downscaled luma grid for the motion signal (tier-0 diffs a small plane)


def _to_frame(jpeg):
    from PIL import Image

    from percept_harness.tier0 import Frame

    _, raw = jpeg
    img = Image.open(io.BytesIO(raw)).convert("L").resize((GW, GH))
    return Frame(w=GW, h=GH, pixels=tuple(int(p) for p in img.getdata()))


def _motion_series(frames):
    """frames: [(t, (media,bytes))] -> [(t, motion 0..1)] (frame-diff vs the previous frame)."""
    from percept_harness.tier0 import motion_score

    out, prev = [], None
    for t, jpeg in frames:
        f = _to_frame(jpeg)
        out.append((t, motion_score(prev, f)))
        prev = f
    return out


def _count_peaks(series) -> int:
    from percept_harness.detectors import MotionPeriodicity

    det = MotionPeriodicity(min_height=0.01, rel_drop=0.5)
    return sum(len(det.push(t, v)) for t, v in series)


def _count_period(series) -> float:
    """Autocorrelation dominant period -> reps = duration / period. Robust to 2-peaks-per-rep."""
    import numpy as np

    if len(series) < 6:
        return 0.0
    ts = [t for t, _ in series]
    v = np.array([x for _, x in series], dtype=float)
    v = v - v.mean()
    ac = np.correlate(v, v, mode="full")[len(v) - 1:]
    if ac[0] <= 0:
        return 0.0
    ac = ac / ac[0]
    # first local maximum after the zero-lag peak (skip the initial descent)
    lag = None
    for i in range(2, len(ac) - 1):
        if ac[i] > ac[i - 1] and ac[i] >= ac[i + 1] and ac[i] > 0.2:
            lag = i
            break
    if not lag:
        return 0.0
    dt = (ts[-1] - ts[0]) / (len(ts) - 1)
    period_s = lag * dt
    return (ts[-1] - ts[0]) / period_s if period_s > 0 else 0.0


def main() -> int:
    clips = [p for p in sorted(PROBE_DIR.glob("*/clip.json"))
             if "count" in json.loads(p.read_text())]
    if not clips:
        print(f"\n  No rep clips found under {PROBE_DIR}/<name>/clip.json with a `count` field.")
        print("  Record 2-3 short rep videos (bench/squat/jumpjacks, ~10-20s) and run, e.g.:")
        print("    make_probe_clip bench.mov --name bench --count 8 --condition bench-press")
        print("  Then re-run this bench.")
        return 0

    rows = []
    for cp in clips:
        spec = json.loads(cp.read_text())
        frames = [(float(t), (_media(b), _b64(b))) for t, b in spec["frames"]]
        series = _motion_series(frames)
        peaks = _count_peaks(series)
        period = _count_period(series)
        rows.append((cp.parent.name, spec.get("condition", "?"), spec["count"], peaks, round(period, 1)))

    L = ["", "=" * 78, "  EDGE-COUNT ON RAW FRAMES — cheap motion signal (no VLM)", "=" * 78,
         f"  {'clip':<16}{'action':<16}{'gt':>4}{'peaks':>8}{'period':>9}"]
    for name, act, gt, pk, pr in rows:
        L.append(f"  {name:<16}{act[:15]:<16}{gt:>4}{pk:>8}{pr:>9}")
    pk_obo = 100.0 * sum(abs(pk - gt) <= 1 for _, _, gt, pk, _ in rows) / len(rows)
    pr_obo = 100.0 * sum(abs(pr - gt) <= 1 for _, _, gt, _, pr in rows) / len(rows)
    L += ["", f"  peaks  OBO={pk_obo:.0f}   |   period OBO={pr_obo:.0f}   (vs cached VLM signal: 15.0)",
          "  If motion (peaks/period) beats the VLM's 15.0 OBO, the edge path is the RepCount fix.",
          "=" * 78, ""]
    print("\n".join(L))
    return 0


def _media(b64: str) -> str:
    import base64
    raw = base64.b64decode(b64)
    return "image/jpeg" if raw[:3] == b"\xff\xd8\xff" else "image/png"


def _b64(b64: str) -> bytes:
    import base64
    return base64.b64decode(b64)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
