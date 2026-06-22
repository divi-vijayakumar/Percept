"""Turn a phone video into a Phase-0 probe clip — frames + audio, base64, the format the probe reads.

You record ~3 short clips on your phone, then run this once per clip; it uses ffmpeg to sample frames
and extract the audio and writes ``eval/perception_probe/<name>/clip.json``. Then run the probe.

Example:
    PYTHONPATH=packages/percept-vision python3.12 -m eval.make_probe_clip milk.mov \\
        --name milk-boil --shape continuous --expected true --ambiguous-t 6.5 \\
        --condition "the milk is about to boil over"

    # discrete (a few reps; the ambiguous moment is mid-rep):
    ... bench.mov --name bench --shape discrete --expected true --ambiguous-t 3.0 \\
        --condition "the person is at the bottom of a bench-press rep"

    # control (single-state; an escalation here SHOULD stay flat):
    ... stove.mov --name stove --shape control --expected true --ambiguous-t 2.0 --no-audio \\
        --condition "the stove burner is on"

Then: PERCEPT_LIVE=1 PYTHONPATH=packages/percept-vision python3.12 -m eval.exp_perception_spec_probe
"""

from __future__ import annotations

import argparse
import base64
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

OUT_ROOT = Path(__file__).resolve().parent / "perception_probe"


def _require_ffmpeg() -> str:
    ff = shutil.which("ffmpeg")
    if not ff:
        raise SystemExit(
            "ffmpeg not found. Install it (macOS: `brew install ffmpeg`) — it's the only dependency "
            "for turning a video into probe frames + audio."
        )
    return ff


def _extract_frames(ff: str, video: Path, tmp: Path, fps: float) -> list[tuple[float, bytes]]:
    pat = str(tmp / "f_%04d.jpg")
    subprocess.run(
        [ff, "-y", "-i", str(video), "-vf", f"fps={fps}", "-q:v", "3", pat],
        check=True, capture_output=True,
    )
    frames: list[tuple[float, bytes]] = []
    for i, fp in enumerate(sorted(tmp.glob("f_*.jpg"))):
        frames.append((i / fps, fp.read_bytes()))  # frame i ≈ i/fps seconds
    if not frames:
        raise SystemExit("ffmpeg produced no frames — check the video path/format.")
    return frames


def _extract_audio(ff: str, video: Path, tmp: Path) -> bytes | None:
    out = tmp / "audio.wav"
    proc = subprocess.run(
        [ff, "-y", "-i", str(video), "-ac", "1", "-ar", "16000", "-f", "wav", str(out)],
        capture_output=True,
    )
    if proc.returncode != 0 or not out.exists():
        print("  (no audio track found — writing a vision-only clip)")
        return None
    return out.read_bytes()


def main() -> int:
    ap = argparse.ArgumentParser(description="Build a Phase-0 probe clip from a video.")
    ap.add_argument("video", type=Path)
    ap.add_argument("--name", required=True, help="clip dir name under eval/perception_probe/")
    ap.add_argument("--condition", required=True, help="condition to judge (perception) or action label (rep mode)")
    ap.add_argument("--count", type=int, default=None, help="REP MODE: ground-truth rep count (for the edge-count bench)")
    ap.add_argument("--shape", choices=["continuous", "discrete", "control"], help="perception mode")
    ap.add_argument("--expected", choices=["true", "false"], help="perception mode: ground truth at the ambiguous moment")
    ap.add_argument("--ambiguous-t", type=float, help="perception mode: seconds; the frame a single look gets wrong")
    ap.add_argument("--fps", type=float, default=3.0, help="frame sampling rate (default 3; use 5-6 for fast reps)")
    ap.add_argument("--no-audio", action="store_true", help="skip audio extraction")
    args = ap.parse_args()

    if not args.video.exists():
        raise SystemExit(f"video not found: {args.video}")
    if args.count is None and not (args.shape and args.expected and args.ambiguous_t is not None):
        raise SystemExit("provide --count (rep mode) OR --shape + --expected + --ambiguous-t (perception mode)")
    ff = _require_ffmpeg()
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        frames = _extract_frames(ff, args.video, tmp, args.fps)
        audio = None if args.no_audio else _extract_audio(ff, args.video, tmp)

    clip = {
        "condition": args.condition,
        "frames": [[round(t, 3), base64.b64encode(raw).decode("ascii")] for t, raw in frames],
    }
    if args.count is not None:  # rep mode (edge-count bench)
        clip["count"] = args.count
    else:  # perception mode (strip/roi probe)
        clip.update(shape=args.shape, expected=args.expected == "true", ambiguous_t=args.ambiguous_t)
    if audio is not None:
        clip["audio"] = {"mime": "audio/wav", "data": base64.b64encode(audio).decode("ascii")}

    out_dir = OUT_ROOT / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "clip.json").write_text(json.dumps(clip))
    print(f"  wrote {out_dir / 'clip.json'}  ({len(frames)} frames @ {args.fps}fps, "
          f"audio={'yes' if audio else 'no'}, shape={args.shape})")
    return 0


if __name__ == "__main__":  # pragma: no cover — local tooling
    raise SystemExit(main())
