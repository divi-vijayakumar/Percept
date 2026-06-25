"""Phase-A v0.5 runner — score the tracks and emit the DRAFT Snapshot Card.

Offline smoke (no API, renders the card with stub cells):
    PYTHONPATH=packages/percept-vision:packages/percept-harness python3.12 -m eval.bench.run
Live sampled (real Gemini):
    PERCEPT_LIVE=1 PYTHONPATH=... python3.12 -m eval.bench.run --live --n 20
"""

from __future__ import annotations

import argparse
import subprocess

from . import card
from . import (
    track_acoustic_pt,
    track_counting,
    track_cuespeak,
    track_edge_recall,
    track_relational,
    track_timeliness,
    track_timespeak,
)


def _commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:  # pragma: no cover
        return "?"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true", help="run real Gemini (else offline stub cells)")
    ap.add_argument("--n", type=int, default=20, help="relational sample size")
    ap.add_argument("--out", default=None, help="also write the card to this path")
    args = ap.parse_args(argv)

    if args.live:
        from eval._dotenv import load_dotenv
        load_dotenv()

    results = []
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

    backend = "gemini-2.5-flash" if args.live else "fake (offline stub)"
    auroc = next((r.cells.get(("core", "auroc")) for r in results if r.track == "relational"), None)
    out = card.render_card(results, mode="sampled", backend=backend, commit=_commit(), fps="1", auroc=auroc)
    print(out)
    if args.out:
        from pathlib import Path
        Path(args.out).write_text(out)
        print(f"  card written -> {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
