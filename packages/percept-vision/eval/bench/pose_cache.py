"""RTMO detection cache — run the pose provider ONCE per clip, store all-persons detections to JSON, so
the hero A/B (``eval.bench.hero_ab``) can replay the association + count layers cheaply. RTMO-L on CPU is
the slow part; caching it decouples every association/count experiment from full-frame pose inference.

This is EVAL TOOLING, not a product feature. The cache lives under ``bench-staging`` (gitignored data) —
only this code is tracked. Keyed by ``(video_id, fps, provider)`` so changing the rate re-caches cleanly.

Run (one-time warm + cache of the fixed 8-clip manifest):
    PYTHONPATH=packages/percept-vision python -m eval.bench.pose_cache
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from eval.bench.repcount import _STAGING  # the raw-clip dir; cache is a sibling

CACHE_DIR = _STAGING.parent / "pose_cache"

# The handoff's fixed N=8 (explicit list, NOT sample_repcount — that excludes pommel + "others").
MANIFEST = ["stu10_5", "train143", "stu8_23", "test538", "stu8_34", "train1665", "stu4_28", "stu13_33"]


def cache_path(video_id: str, fps: int, provider: str) -> Path:
    return CACHE_DIR / f"{video_id}.fps{fps}.{provider}.json"


def cache_detections(video, fps: int = 6, provider: str | None = None, *, force: bool = False) -> Path:
    """Decode ``video`` at ``fps``, run the pose provider on every frame, write all-persons detections.

    Each frame stores ``{t, dets:[{bbox, conf, person_id, kps:[[x,y,conf]*17]}]}``. Idempotent: returns the
    existing cache unless ``force``. RTMO bbox is ``(x1,y1,x2,y2)`` normalized; BlazePose has none (None)."""
    import cv2

    from eval.count_video import decode
    from percept_harness.pose import make_pose_provider

    video = Path(video)
    vid = video.stem
    prov = make_pose_provider(provider)
    pname = getattr(prov, "name", provider or "rtmo")
    out_path = cache_path(vid, fps, pname)
    if out_path.exists() and not force:
        return out_path

    prov.warm()
    _small, full = decode(str(video), fps)
    frames = []
    for t, bgr in full:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)  # the provider expects RGB (appearance later uses BGR)
        dets = prov.detect(rgb)
        frames.append({
            "t": float(t),
            "dets": [
                {
                    "bbox": [float(v) for v in d.bbox] if d.bbox else None,
                    "conf": float(d.confidence),
                    "person_id": d.person_id,
                    "kps": [[float(k.x), float(k.y), float(k.confidence)] for k in d.keypoints],
                }
                for d in dets
            ],
        })

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"video_id": vid, "fps": fps, "provider": pname, "frames": frames}))
    return out_path


def load_detections(video_id: str, fps: int = 6, provider: str = "rtmo") -> list[dict] | None:
    """The cached per-frame detection list (``[{t, dets}]``), or ``None`` if not cached yet."""
    p = cache_path(video_id, fps, provider)
    if not p.exists():
        return None
    return json.loads(p.read_text())["frames"]


def reconstruct(frame_dets: list[dict]):
    """One cached frame's ``dets`` -> ``list[PoseDetection]`` (so the eye/measures layer sees real poses)."""
    from percept_harness.pose import PoseDetection, PoseKeypoint

    out = []
    for d in frame_dets:
        kps = tuple(PoseKeypoint(x, y, c) for x, y, c in d["kps"])
        bbox = tuple(d["bbox"]) if d.get("bbox") else None
        out.append(PoseDetection(
            keypoints=kps, confidence=float(d.get("conf", 1.0)),
            person_id=d.get("person_id"), provider="cache", bbox=bbox,
        ))
    return out


def main() -> None:
    fps = int(os.environ.get("PERCEPT_HERO_FPS", "6"))
    force = os.environ.get("PERCEPT_HERO_RECACHE") == "1"
    for vid in MANIFEST:
        video = _STAGING / f"{vid}.mp4"
        if not video.exists():
            print(f"  MISSING {video}")
            continue
        path = cache_detections(video, fps, force=force)
        frames = json.loads(path.read_text())["frames"]
        max_people = max((len(f["dets"]) for f in frames), default=0)
        print(f"  cached {vid:10s} {len(frames):4d} frames @ {fps}fps  max_people={max_people}  -> {path.name}")


if __name__ == "__main__":  # pragma: no cover
    main()
