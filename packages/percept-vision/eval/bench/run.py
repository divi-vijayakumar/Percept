"""Phase-A v0.5 runner — score the tracks and emit the DRAFT Snapshot Card.

Offline smoke (no API, renders the card with stub cells):
    PYTHONPATH=packages/percept-vision:packages/percept-harness python3.12 -m eval.bench.run
Live sampled:
    PERCEPT_LIVE=1 PYTHONPATH=... python3.12 -m eval.bench.run --live --n 20

Live mode intentionally skips legacy dense/raw media-bearing tracks. Publishable benchmark runs should
use edge/E2E candidate calls only; dense call counts are modeled from the sampling schedule.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
from pathlib import Path

from . import card
from . import (
    track_acoustic_pt,
    track_counting,
    track_cuespeak,
    track_edge_recall,
    track_escalation_policy,
    track_perception_test_e2e,
    track_relational,
    track_timeliness,
    track_timespeak,
)


def _commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:  # pragma: no cover
        return "?"


def _manifest_ref(path: str | None) -> str:
    if not path:
        return ""
    p = Path(path)
    digest = hashlib.sha256(p.read_bytes()).hexdigest()[:12]
    return f"{p.as_posix()} sha256:{digest}"


def _backend_label(live: bool) -> str:
    if not live:
        return "fake (offline stub)"
    return os.environ.get("PERCEPT_VISION_MODEL") or os.environ.get("PERCEPT_GEMINI_MODEL") or "gemini"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true", help="run real Gemini (else offline stub cells)")
    ap.add_argument("--n", type=int, default=20, help="relational sample size")
    ap.add_argument("--out", default=None, help="also write the card to this path")
    ap.add_argument("--manifest", default=None, help="path to the pinned data manifest to stamp on the card")
    ap.add_argument("--mode", default="sampled", choices=["sampled", "full"], help="scorecard mode")
    ap.add_argument("--device", default="?", help="device/runtime descriptor for the card header")
    ap.add_argument("--fps", default="1", help="sampling/capture fps for the card header")
    ap.add_argument("--seeds", type=int, default=1, help="number of seeds/runs represented")
    ap.add_argument(
        "--require-release",
        action="store_true",
        help="exit non-zero unless the rendered card is stamped RELEASE",
    )
    args = ap.parse_args(argv)

    if args.live:
        from eval._dotenv import load_dotenv
        load_dotenv()

    results = []
    if args.live:
        tracks = (
            track_perception_test_e2e,
            track_edge_recall,
            track_escalation_policy,
            track_counting,
            track_cuespeak,
            track_timespeak,
        )
    else:
        tracks = (track_relational, track_counting, track_acoustic_pt, track_timeliness, track_edge_recall,
                  track_cuespeak, track_timespeak)
    for mod in tracks:
        try:
            results.append(mod.score(live=args.live, n=args.n))
        except Exception as exc:  # a track failure must not sink the whole card
            from .types import TrackResult
            results.append(TrackResult(track=getattr(mod, "TRACK", mod.__name__),
                                       notes=f"FAILED: {type(exc).__name__}: {exc}",
                                       excluded=[f"{getattr(mod, 'TRACK', '?')}: {exc}"]))

    backend = _backend_label(args.live)
    auroc = next((r.cells.get(("core-replay", "auroc")) for r in results if r.track == "relational"), None)
    manifest_ref = _manifest_ref(args.manifest)
    stamp = card.stamp_of(args.mode, results, manifest_ref)
    out = card.render_card(
        results,
        mode=args.mode,
        backend=backend,
        commit=_commit(),
        manifest=manifest_ref,
        fps=args.fps,
        device=args.device,
        seeds=args.seeds,
        auroc=auroc,
    )
    print(out)
    if args.out:
        from pathlib import Path
        Path(args.out).write_text(out)
        print(f"  card written -> {args.out}")
    if args.require_release and stamp != "RELEASE":
        print("SCORECARD RELEASE GATE FAILED: rendered card is DRAFT")
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
