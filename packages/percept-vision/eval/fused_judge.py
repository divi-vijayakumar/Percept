"""Fused (vision+audio) judge on a real video — the ACOUSTIC test.

Some states are told mostly by SOUND: "is the hairdryer working" is the motor whir, not a still frame.
This decodes frames (cv2) + audio (PyAV) -> a per-window energy signal -> acoustic-onset boundaries, and
judges a representative frame two ways: vision-only vs fused (frame + the audio clip). If fused flips a
vision-ambiguous verdict the right way, audio earned its place (the Phase-5 lever, finally on real data).

Run: PERCEPT_LIVE=1 PYTHONPATH=packages/percept-vision:packages/percept-harness python3.12 -m eval.fused_judge \\
        <video> --condition "the hairdryer is working" [--t 2.0] [--thumbs DIR] [--vlm]
"""

from __future__ import annotations

import argparse
import io
import wave
from pathlib import Path

from eval.count_video import decode


def extract_audio(path):
    """-> (mono float32 samples in [-1,1], sample_rate) via PyAV (bundles ffmpeg; no system ffmpeg)."""
    import av
    import numpy as np

    c = av.open(str(path))
    sr, chunks = None, []
    for fr in c.decode(audio=0):
        sr = fr.sample_rate
        a = fr.to_ndarray().astype("float32")
        if a.ndim == 2:
            a = a.mean(axis=0)  # downmix to mono
        chunks.append(a.reshape(-1))
    c.close()
    if not chunks:
        return None, None
    aud = np.concatenate(chunks)
    if np.abs(aud).max() > 2:  # int PCM -> normalize
        aud = aud / 32768.0
    return aud, sr


def energy_series(aud, sr, win_s=0.2):
    import numpy as np

    w = max(1, int(win_s * sr))
    return [(i / sr, float(np.sqrt(np.mean(aud[i:i + w] ** 2)))) for i in range(0, max(1, len(aud) - 1), w)]


def to_wav(aud, sr, t0, t1):
    import numpy as np

    seg = aud[int(t0 * sr):int(t1 * sr)]
    pcm = (np.clip(seg, -1, 1) * 32767).astype("<i2").tobytes()
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm)
    return ("audio/wav", buf.getvalue())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--condition", required=True)
    ap.add_argument("--fps", type=float, default=4.0)
    ap.add_argument("--t", type=float, default=None, help="timestamp to judge (default: loudest window)")
    ap.add_argument("--audio-s", type=float, default=2.0)
    ap.add_argument("--thumbs", default=None)
    ap.add_argument("--n-thumbs", type=int, default=8)
    ap.add_argument("--vlm", action="store_true")
    args = ap.parse_args()

    _, full = decode(args.video, args.fps)
    aud, sr = extract_audio(args.video)
    dur = full[-1][0]
    print(f"\n  {Path(args.video).name}: {len(full)} frames @ ~{args.fps}fps, {dur:.1f}s, "
          f"audio={'%d Hz' % sr if sr else 'NONE'}")

    if aud is not None:
        es = energy_series(aud, sr)
        emax = max(e for _, e in es) or 1e-9
        # compact energy sparkline + acoustic-onset boundaries
        bars = " .:-=+*#@"
        spark = "".join(bars[min(len(bars) - 1, int(e / emax * (len(bars) - 1)))] for _, e in es)
        print(f"  audio energy (max={emax:.3f}):  {spark}")
        from percept_harness.detectors import AcousticOnset
        det = AcousticOnset(high=0.3 * emax, low=0.15 * emax)
        onsets = [b.t for t, e in es for b in det.push(t, e)]
        print(f"  acoustic-onset boundaries @ t={[round(o, 1) for o in onsets]}")
        t_loud = max(es, key=lambda te: te[1])[0]
    else:
        t_loud = dur / 2

    if args.thumbs:
        import cv2
        d = Path(args.thumbs)
        d.mkdir(parents=True, exist_ok=True)
        idx = [round(i * (len(full) - 1) / (args.n_thumbs - 1)) for i in range(args.n_thumbs)]
        for k, j in enumerate(idx):
            cv2.imwrite(str(d / f"t{k:02d}_{full[j][0]:05.1f}s.jpg"), full[j][1])
        print(f"  saved {len(idx)} thumbnails -> {d}")

    if args.vlm:
        import asyncio

        import cv2

        from eval._dotenv import load_dotenv
        load_dotenv()
        from percept import registry
        from percept.config import load_settings

        t = args.t if args.t is not None else t_loud
        bgr = min(full, key=lambda tb: abs(tb[0] - t))[1]
        _, buf = cv2.imencode(".jpg", bgr)
        frame = ("image/jpeg", buf.tobytes())
        clip = to_wav(aud, sr, max(0, t - args.audio_s), t) if aud is not None else None
        vision = registry.make_vision("gemini", load_settings())

        async def go():
            v_only = await vision.judge(args.condition, frame)
            v_fused = await vision.judge(args.condition, frame, audio=clip) if clip else None
            return v_only, v_fused

        v_only, v_fused = asyncio.run(go())
        print(f"\n  judging \"{args.condition}\" at t={t:.1f}s:")
        print(f"    vision-only : satisfied={v_only.satisfied}  conf={v_only.confidence:.2f}  ({v_only.evidence[:60]})")
        if v_fused:
            print(f"    +audio fused: satisfied={v_fused.satisfied}  conf={v_fused.confidence:.2f}  ({v_fused.evidence[:60]})")
    print()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
