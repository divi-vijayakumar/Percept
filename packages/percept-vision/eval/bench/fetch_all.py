"""Fetch + PERSIST all benchmark data into eval/bench-staging/ (gitignored, survives across sessions)
so the full live run never has to re-fetch. Idempotent: the fetchers cache to disk, so re-runs skip
what's already there. Writes a MANIFEST.json describing what's cached.

Run:
    PYTHONPATH=packages/percept-vision python3.12 -m eval.bench.fetch_all [--videos N] [--repcount N]

Sources:
  * Perception Test (CC-BY-4.0)  — sound + action annotations (~550KB) + N range-fetched ~2MB videos/task
  * RepCount-A (lmms-lab lance)  — meta + N materialized video blobs
  * golden (relational)          — already local (fixtures-golden); just verified + recorded
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
_DEFAULT_STAGE_ROOT = Path(__file__).resolve().parents[1] / "bench-staging"
_STAGE_ROOT = Path(os.environ.get("PERCEPT_BENCH_DATA_DIR", _DEFAULT_STAGE_ROOT)).expanduser()
STAGING = _STAGE_ROOT if _STAGE_ROOT.is_absolute() else REPO_ROOT / _STAGE_ROOT


def _pt(manifest: dict, n_videos: int) -> None:
    from eval.bench import perception_test as pt

    for task in ("sound", "action"):
        ann = pt.load_annotations(task)
        manifest["sources"][f"perception_test/{task}/annotations"] = {"ok": True, "rows": len(ann)}
        print(f"PT {task}: annotations cached ({len(ann)} rows)")
        events = pt.sample_events(task, n_videos, is_visible=0) if task == "sound" else pt.sample_events(task, n_videos)
        got = 0
        for e in events:
            try:
                pt.fetch_video(e["video_id"])
                got += 1
            except Exception as ex:  # one bad video shouldn't sink the run
                print(f"  PT {task} video {e.get('video_id')}: {type(ex).__name__}")
        manifest["sources"][f"perception_test/{task}/videos"] = {"ok": True, "cached": got, "asked": len(events)}
        print(f"PT {task}: {got}/{len(events)} videos cached")


def _repcount(manifest: dict, n: int) -> None:
    from eval.bench import repcount as rc

    meta = rc.load_meta()
    manifest["sources"]["repcount/meta"] = {"ok": True, "rows": len(meta)}
    print(f"RepCount: meta cached ({len(meta)} rows)")
    # sample_repcount carries the lance row `index` take_blobs needs + the `video_id`
    sample = rc.sample_repcount(n)
    got = 0
    for row in sample:
        try:
            rc.fetch_video(row["index"], row["video_id"])
            got += 1
        except Exception as ex:
            print(f"  repcount video {row.get('video_id')}: {type(ex).__name__}: {str(ex)[:60]}")
    manifest["sources"]["repcount/videos"] = {"ok": True, "cached": got, "asked": len(sample)}
    print(f"RepCount: {got}/{len(sample)} videos materialized")


def _golden(manifest: dict) -> None:
    from percept.eval import find_golden_root

    r = find_golden_root()
    ok = bool(r and r.exists())
    manifest["sources"]["golden"] = {"ok": ok, "path": str(r)}
    print(f"golden (relational): {'present' if ok else 'MISSING'} at {r}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", type=int, default=12, help="PT videos to range-fetch per task")
    ap.add_argument("--repcount", type=int, default=12, help="RepCount videos to materialize")
    args = ap.parse_args()

    manifest: dict = {"fetched_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "sources": {}}
    for name, fn in (
        ("perception_test", lambda: _pt(manifest, args.videos)),
        ("repcount", lambda: _repcount(manifest, args.repcount)),
        ("golden", lambda: _golden(manifest)),
    ):
        try:
            fn()
        except Exception as ex:
            print(f"{name} FAILED: {type(ex).__name__}: {str(ex)[:140]}")
            manifest["sources"][name] = {"ok": False, "error": f"{type(ex).__name__}: {str(ex)[:140]}"}

    STAGING.mkdir(parents=True, exist_ok=True)
    (STAGING / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nmanifest -> {STAGING / 'MANIFEST.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
