"""Pose-based rep count (MediaPipe BlazePose) — the structural edge skill that fixes fast-rep aliasing.

Runs pose per frame -> an 'openness' waveform = ankle-spread/shoulder-width + wrist-elevation -> counts
oscillation peaks = reps. Where frame-diff motion conflates the open+close of each rep (over-counts) and
the VLM aliases on fast jumps (under-counts), the pose openness signal is ONE clean peak per rep, across
variations (legs-only contributes via ankle-spread; full jacks add wrist-elevation).

This is the first MODEL-backed edge skill (vs pure DSP). BlazePose is edge-class (mobile, real-time CPU,
~3MB lite model); lazy-imported so the L1 core never touches it.

Run: PYTHONPATH=packages/percept-vision:packages/percept-harness python3.12 -m eval.pose_count <video> [--count N]
"""

from __future__ import annotations

import argparse

from eval.count_video import decode  # reuse the cv2 decoder


def openness_series(full):
    """full: [(t, bgr)] -> [(t, openness)]; holds the last value through frames with no detected pose.

    BlazePose (Tasks API) -> 33 normalized landmarks; openness = ankle-spread/shoulder-width + wrist-elevation."""
    from pathlib import Path

    import cv2
    import mediapipe as mp
    import numpy as np
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    model = str(Path(__file__).resolve().parent / "pose_landmarker_lite.task")
    opts = vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=model),
        running_mode=vision.RunningMode.IMAGE,
    )
    out, last = [], 0.0
    with vision.PoseLandmarker.create_from_options(opts) as lmk:
        for t, bgr in full:
            rgb = np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            res = lmk.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
            if res.pose_landmarks:
                lm = res.pose_landmarks[0]                       # first person, 33 landmarks
                spread = abs(lm[27].x - lm[28].x)               # ankle horizontal distance
                sw = abs(lm[11].x - lm[12].x) or 0.1            # shoulder width (scale-normalize)
                wrist_up = max(0.0, lm[11].y - lm[15].y) + max(0.0, lm[12].y - lm[16].y)  # wrists above shoulders
                last = spread / sw + wrist_up
            out.append((t, last))
    return out


def count_reps(series, *, min_height, rel_drop, refractory):
    from percept.harness.detectors import MotionPeriodicity

    det = MotionPeriodicity(min_height=min_height, rel_drop=rel_drop, refractory_s=refractory)
    return [b for t, v in series for b in det.push(t, v)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--fps", type=float, default=6.0)
    ap.add_argument("--count", type=int, default=None)
    ap.add_argument("--refractory", type=float, default=0.4)
    args = ap.parse_args()

    _, full = decode(args.video, args.fps)
    series = openness_series(full)
    vals = [v for _, v in series]
    print(f"\n  POSE openness (ankle-spread/shoulder + wrist-elevation): {len(series)} frames @ ~{args.fps}fps")
    print(f"  signal: min={min(vals):.2f} max={max(vals):.2f} mean={sum(vals)/len(vals):.2f}")
    print("\n  rep count (peaks of the openness waveform):")
    for mh, rd in [(0.5, 0.35), (0.6, 0.4), (0.8, 0.4), (1.0, 0.5)]:
        bs = count_reps(series, min_height=mh, rel_drop=rd, refractory=args.refractory)
        print(f"    min_height={mh} rel_drop={rd} -> {len(bs)} reps @ t={[round(b.t, 1) for b in bs]}")
    if args.count is not None:
        print(f"\n  ground truth: {args.count}   (edge motion-diff: 15 · VLM-loose: 5)")
    print()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
