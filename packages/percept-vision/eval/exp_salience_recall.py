"""Recall ablation for the tier-0 salience MODES (A block_max/p95, B ego_compensate, C audio fusion).

Edge-only, no VLM, no API keys. Reuses the Perception Test annotations (offline, from the cached
annotation zips) against the videos on disk, plus ProactiveVideoQA EGO for the egocentric case, and
the same "did the gate escalate inside the annotated event window?" scoring the escalation track uses.

Shapes (pick with --shape):

  static  — Mode A. Localized events on a STATIC camera (PT action). Sweeps `mean` vs `block_max`/`p95`
            across thresholds; this is the calibration behind the per-stat defaults in EdgeSettings.
            The whole-frame mean is near-blind (recall ~0.06 @ 0.10); block_max@0.20 recovers ~0.85.
  ego     — Mode B. A WEARABLE camera (ProactiveVideoQA EGO cooking). `mean` vs `ego_compensate`:
            ego cuts calls/min (fewer head-motion false wakes) at some recall cost.
  sound   — Mode C. Audio fusion on PT SOUND events whose source is NOT on screen (is_visible=0), so
            only audio can wake the gate. `motion-only` vs a `wake_policy` fusing `audio_event`:
            motion-only recall ~0.28; +audio ~0.70. Needs ffmpeg + soundfile for energy extraction.

Run:
    PYTHONPATH=packages/percept-vision python3 -m eval.exp_salience_recall --shape static --n 80
    PYTHONPATH=packages/percept-vision python3 -m eval.exp_salience_recall --shape ego   --n 6
    PYTHONPATH=packages/percept-vision python3 -m eval.exp_salience_recall --shape sound --n 40
"""

from __future__ import annotations

import argparse
import io
import json
import subprocess

from percept.adapters import WatchCondition, WatchSpec
from percept.config import load_settings
from percept.harness.detectors import WakePolicy
from percept.harness.tier0 import Frame, MotionGate

from eval.bench.perception_test import REPO_ROOT, load_annotations, TASK_KEY

GW, GH = 48, 36  # the gate frame (divisible by the 8x6 block grid)
PT_VIDEOS = REPO_ROOT / "eval/bench-staging/perception_test/raw/videos"
EGO_RAW = REPO_ROOT / "eval/bench-staging/proactiveqa/raw"


def _local_pt_ids() -> set[str]:
    return {p.stem for p in PT_VIDEOS.glob("*.mp4")}


def _pt_events(task: str, n: int, *, is_visible: int | None = None) -> list[dict]:
    """One event per LOCAL PT video for `task` (offline annotations x on-disk videos)."""
    data = load_annotations(task)
    key = TASK_KEY[task]
    local = _local_pt_ids()
    out: list[dict] = []
    for vid in sorted(data):
        v = data[vid]
        vi = v["metadata"]["video_id"]
        if vi not in local:
            continue
        seg = next((s for s in v.get(key, [])
                    if is_visible is None or s.get("is_visible") == is_visible), None)
        if not seg:
            continue
        ts = seg["timestamps"]
        out.append({"path": PT_VIDEOS / f"{vi}.mp4", "lo": ts[0] / 1e6, "hi": ts[1] / 1e6})
        if n and len(out) >= n:
            break
    return out


def _ego_events(n: int, max_s: float = 150.0) -> list[dict]:
    subset = json.loads((EGO_RAW / "EGO_subset.json").read_text())
    vids = EGO_RAW / "EGO_videos"
    out: list[dict] = []
    for item in subset:
        path = vids / str(item.get("video", ""))
        span = next((a["related_timespan"] for a in item.get("answer", [])
                     if a.get("related_timespan") and a["related_timespan"][0] < max_s), None)
        if not path.exists() or not span:
            continue
        out.append({"path": path, "lo": float(span[0]), "hi": float(span[1]), "max_s": max_s})
        if n and len(out) >= n:
            break
    return out


def _frames(path, fps: float = 4.0, max_s: float | None = None):
    import cv2

    cap = cv2.VideoCapture(str(path))
    src = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(src / fps)))
    out, i = [], 0
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        t = i / src
        if max_s is not None and t > max_s:
            break
        if i % step == 0:
            g = cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), (GW, GH))
            out.append((t, Frame(w=GW, h=GH, pixels=tuple(int(p) for p in g.flatten()))))
        i += 1
    cap.release()
    return out


def _energy_fn(path):
    """Per-clip-normalized (0..1) audio RMS as a function of time, via ffmpeg -> soundfile."""
    import numpy as np
    import soundfile as sf

    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-ac", "1", "-ar", "16000", "-f", "wav", "pipe:1"],
        capture_output=True,
    ).stdout
    wav, sr = sf.read(io.BytesIO(raw))
    win = int(0.25 * sr)
    e = np.array([np.sqrt(np.mean(wav[i:i + win] ** 2)) for i in range(0, max(1, len(wav) - win), win)])
    peak = e.max() if len(e) and e.max() > 0 else 1.0
    return lambda t: float(e[min(int(t / 0.25), len(e) - 1)] / peak) if len(e) else 0.0


def _gate(*, thr, salience_stat="mean", ego=False, wake_policy=None):
    spec = WatchSpec(conditions=[WatchCondition(
        goal_id="c", anchor_text="x", threshold=thr, gate_on="motion",
        salience_stat=salience_stat, ego_compensate=ego,
    )])
    return MotionGate(spec, wake_policy=wake_policy)


def _recall(clips, gate_factory, energy=False):
    hits = calls = frames = dur = 0
    for fs, efn, lo, hi in clips:
        g = gate_factory()
        esc = [t for t, f in fs
               if g.evaluate(f, ts=t, **({"energy": efn(t)} if energy else {})).escalate]
        calls += len(esc)
        frames += len(fs)
        dur += fs[-1][0] if fs else 0
        if any(lo <= t <= hi + 0.5 for t in esc):
            hits += 1
    cpm = calls / (dur / 60) if dur else 0.0
    return (hits / len(clips) if clips else 0.0), cpm, calls


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", choices=["static", "ego", "sound"], default="static")
    ap.add_argument("--n", type=int, default=40)
    args = ap.parse_args()
    load_settings()  # exercise config load; the sweeps below set thresholds explicitly

    if args.shape == "static":
        events = _pt_events("action", args.n)
        clips = [(_frames(e["path"]), None, e["lo"], min(e["lo"] + 1.5, e["hi"] + 0.5)) for e in events]
        clips = [c for c in clips if c[0]]
        print(f"STATIC / localized — PT action, N={len(clips)} (per-stat threshold sweep = calibration)")
        print(f"  {'stat':10s}{'thr':>6s}{'recall':>8s}{'calls/min':>11s}")
        for stat in ("mean", "block_max", "p95"):
            for th in (0.05, 0.10, 0.15, 0.20, 0.25, 0.30):
                r, cpm, _ = _recall(clips, lambda st=stat, t=th: _gate(thr=t, salience_stat=st))
                print(f"  {stat:10s}{th:>6.2f}{r:>8.2f}{cpm:>11.1f}")
            print()

    elif args.shape == "ego":
        events = _ego_events(args.n or 6)
        clips = [(_frames(e["path"], fps=2.0, max_s=e["max_s"]), None, e["lo"], e["hi"]) for e in events]
        clips = [c for c in clips if c[0]]
        print(f"EGO / wearable — ProactiveVideoQA EGO, N={len(clips)}  (mean vs ego_compensate)")
        print(f"  {'variant':16s}{'recall':>8s}{'calls/min':>11s}")
        for name, ego in (("mean", False), ("ego k=4", True)):
            r, cpm, _ = _recall(clips, lambda e=ego: _gate(thr=0.10, ego=e))
            print(f"  {name:16s}{r:>8.2f}{cpm:>11.1f}")

    else:  # sound — Mode C audio fusion
        events = _pt_events("sound", args.n, is_visible=0)
        clips = [(_frames(e["path"]), _energy_fn(e["path"]), e["lo"], e["hi"]) for e in events]
        clips = [c for c in clips if c[0]]
        wp = WakePolicy(weights={"change": 1.0, "audio_event": 1.0})
        print(f"SOUND / invisible source (is_visible=0) — PT sound, N={len(clips)}  (motion vs +audio)")
        print(f"  {'config':22s}{'recall':>8s}{'calls':>8s}")
        r, _, c = _recall(clips, lambda: _gate(thr=0.10))
        print(f"  {'motion-only thr0.10':22s}{r:>8.2f}{c:>8d}")
        for th in (0.3, 0.4, 0.5):
            r, _, c = _recall(clips, lambda t=th: _gate(thr=t, wake_policy=wp), energy=True)
            print(f"  {'motion+audio thr' + format(th, '.1f'):22s}{r:>8.2f}{c:>8d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
