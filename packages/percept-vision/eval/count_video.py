"""Edge-count a real video — decode (cv2, no ffmpeg) -> cheap motion signal -> count distinct events.

Tests the edge path's claim on a real clip: a frame-diff MOTION signal (the tier-0 edge computation, no
model) counts discrete movements/gestures/reps. Reports two methods:
  * peaks  — motion-periodicity (one boundary per motion bump) — right for DISCRETE gestures/reps
  * period — autocorrelation dominant period (duration/period) — right for strictly PERIODIC reps
Optionally saves thumbnails for inspection, and scores against a known --count.

Run: PYTHONPATH=packages/percept-vision:packages/percept-harness python3.12 -m eval.count_video <video> \\
        [--fps 5] [--count N] [--refractory 0.3] [--thumbs DIR]
"""

from __future__ import annotations

import argparse
from pathlib import Path

GW, GH = 48, 36


def decode(video: str, target_fps: float):
    """-> (list[(t, small_gray_Frame)], list[(t, bgr_full)]) sampled at ~target_fps."""
    import cv2

    from percept.harness.tier0 import Frame

    cap = cv2.VideoCapture(video)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(src_fps / target_fps)))
    small, full, i = [], [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % step == 0:
            t = i / src_fps
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            g = cv2.resize(gray, (GW, GH))
            small.append((t, Frame(w=GW, h=GH, pixels=tuple(int(p) for p in g.flatten()))))
            full.append((t, frame))
        i += 1
    cap.release()
    return small, full


def motion_series(small):
    from percept.harness.tier0 import motion_score

    out, prev = [], None
    for t, f in small:
        out.append((t, motion_score(prev, f)))
        prev = f
    return out


def count_peaks(series, *, min_height, rel_drop, refractory):
    from percept.harness.detectors import MotionPeriodicity

    det = MotionPeriodicity(min_height=min_height, rel_drop=rel_drop, refractory_s=refractory)
    bs = []
    for t, v in series:
        bs.extend(det.push(t, v))
    return bs


def count_period(series):
    import numpy as np

    if len(series) < 6:
        return 0.0
    ts = [t for t, _ in series]
    v = np.array([x for _, x in series], float)
    v = v - v.mean()
    ac = np.correlate(v, v, "full")[len(v) - 1:]
    if ac[0] <= 0:
        return 0.0
    ac /= ac[0]
    lag = next((i for i in range(2, len(ac) - 1) if ac[i] > ac[i - 1] and ac[i] >= ac[i + 1] and ac[i] > 0.2), None)
    if not lag:
        return 0.0
    dt = (ts[-1] - ts[0]) / (len(ts) - 1)
    return (ts[-1] - ts[0]) / (lag * dt)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--fps", type=float, default=5.0)
    ap.add_argument("--count", type=int, default=None)
    ap.add_argument("--refractory", type=float, default=0.3)
    ap.add_argument("--thumbs", default=None, help="dir to save evenly-spaced thumbnails for inspection")
    ap.add_argument("--n-thumbs", type=int, default=10)
    ap.add_argument("--vlm", action="store_true", help="also run the VLM-confirm path (Gemini per-frame + count rising edges)")
    ap.add_argument("--condition", default="a person is holding up a wooden block that shows a letter of the alphabet (not a geometric shape, not empty)")
    ap.add_argument("--vlm-fps", type=float, default=1.5, help="VLM sampling rate (cost control)")
    args = ap.parse_args()

    small, full = decode(args.video, args.fps)
    series = motion_series(small)
    dur = series[-1][0] if series else 0.0
    print(f"\n  {Path(args.video).name}: {len(small)} frames @ ~{args.fps}fps, {dur:.1f}s")
    mvals = [v for _, v in series]
    print(f"  motion signal: min={min(mvals):.3f} max={max(mvals):.3f} mean={sum(mvals)/len(mvals):.3f}")

    if args.thumbs:
        import cv2
        d = Path(args.thumbs)
        d.mkdir(parents=True, exist_ok=True)
        idx = [round(i * (len(full) - 1) / (args.n_thumbs - 1)) for i in range(args.n_thumbs)]
        for k, j in enumerate(idx):
            t, bgr = full[j]
            cv2.imwrite(str(d / f"t{k:02d}_{t:05.1f}s.jpg"), bgr)
        print(f"  saved {len(idx)} thumbnails -> {d}")

    print("\n  EDGE (motion only — counts movements, blind to meaning):")
    for mh in (0.005, 0.01, 0.02, 0.04):
        bs = count_peaks(series, min_height=mh, rel_drop=0.5, refractory=args.refractory)
        print(f"    min_height={mh:<6} -> {len(bs)} motion peaks  @ t={[round(b.t,1) for b in bs]}")
    edge_peaks = len(count_peaks(series, min_height=0.01, rel_drop=0.5, refractory=args.refractory))
    print(f"  autocorr period count: {count_period(series):.1f}")

    if args.vlm:
        import asyncio

        import cv2

        from eval._dotenv import load_dotenv
        load_dotenv()
        from percept import registry
        from percept.config import load_settings
        from percept.gate import classify

        settings = load_settings()
        vision = registry.make_vision("gemini", settings)
        thr, fmin = settings.gate.threshold, 0.5  # fresh_min: the scorecard's classify default
        stride = max(1, round(args.fps / args.vlm_fps))
        sampled = full[::stride]

        def _jpg(bgr):
            _, buf = cv2.imencode(".jpg", bgr)
            return ("image/jpeg", buf.tobytes())

        async def _go():
            sem = asyncio.Semaphore(6)

            async def one(t, bgr):
                async with sem:
                    v = await vision.judge(args.condition, _jpg(bgr))
                    return t, classify(v, 1.0, thr, fmin)

            return await asyncio.gather(*(one(t, b) for t, b in sampled))

        sig = asyncio.run(_go())
        rising, prev, tl = 0, False, []
        for _, truth in sig:
            yes = truth == "yes"
            rising += yes and not prev
            tl.append("Y" if yes else ("." if truth == "no" else "?"))
            prev = yes
        print(f"\n  VLM-CONFIRM (semantics: \"{args.condition[:48]}...\"):")
        print(f"    per-frame @ {args.vlm_fps}fps (Y=yes · .=no · ?=unsure): {''.join(tl)}")
        print(f"    distinct confirmed episodes (rising edges of confident-yes) = {rising}")
        print(f"\n  >>> edge proposes ~{edge_peaks} motion events · VLM-confirm counts {rising} (per this condition)")

    if args.count is not None:
        print(f"  ground truth: {args.count}")
    print()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
