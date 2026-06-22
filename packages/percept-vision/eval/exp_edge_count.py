"""Edge-count bench — is the bottleneck the COUNTER or the SIGNAL? (offline, cached RepCount signals)

The cached ``repcount_signals.json`` holds the Gemini PHASE verdicts (the RepCount-Δ0 source — often
stuck/flickering). Feeding them to the NEW edge detector (``motion-periodicity``) vs the existing
event-track counter (false→true edges) on the SAME signals isolates the question: is a better *counting
algorithm* the lever, or is the *signal* the problem?

If both score ~the same (~15% OBO), the counter isn't the lever — the SIGNAL is, and the edge path's real
claim (a cheap MOTION signal oscillates cleanly where the VLM phase doesn't) can only be tested on raw
frames (record a few rep clips → exp_edge_count_frames). This run needs no API and no frames.

Run: PYTHONPATH=packages/percept-vision:packages/percept-harness python3.12 -m eval.exp_edge_count
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]  # repo root (caches live here)


def count_edges(seq) -> int:
    """The existing percept event-track count: false→true rising edges of the phase signal."""
    n, prev = 0, False
    for _, b in seq:
        if b and not prev:
            n += 1
        prev = bool(b)
    return n


def count_motion_periodicity(seq) -> int:
    """The NEW edge detector on the same signal (bool→{0,1}); counts confirmed oscillation peaks."""
    from percept_harness.detectors import MotionPeriodicity

    det = MotionPeriodicity(min_height=0.5, rel_drop=0.5)
    n = 0
    for t, b in seq:
        n += len(det.push(float(t), 1.0 if b else 0.0))
    return n


def _obo(rows) -> float:
    return 100.0 * sum(abs(p - g) <= 1 for g, p in rows) / len(rows) if rows else 0.0


def _mae(rows) -> float:
    return sum(abs(p - g) for g, p in rows) / len(rows) if rows else 0.0


def main() -> int:
    sig_path = ROOT / "repcount_signals.json"
    if not sig_path.exists():
        raise SystemExit(f"no cached signals at {sig_path}")
    signals = json.loads(sig_path.read_text())

    ce_rows, mp_rows, detail = [], [], []
    for vid, d in signals.items():
        seq, gt = d["seq"], d["gt"]
        ce, mp = count_edges(seq), count_motion_periodicity(seq)
        ce_rows.append((gt, ce))
        mp_rows.append((gt, mp))
        detail.append((vid, d.get("action", "?"), gt, ce, mp))

    L = ["", "=" * 76, "  EDGE-COUNT BENCH (offline) — counter vs counter on the cached VLM phase signal",
         "=" * 76, f"  {'video':<14}{'action':<14}{'gt':>4}{'count_edges':>13}{'motion-periodicity':>20}"]
    for vid, act, gt, ce, mp in detail:
        L.append(f"  {vid:<14}{act:<14}{gt:>4}{ce:>13}{mp:>20}")
    L += ["", f"  count_edges (existing) : OBO={_obo(ce_rows):.1f}  MAE={_mae(ce_rows):.2f}",
          f"  motion-periodicity (new): OBO={_obo(mp_rows):.1f}  MAE={_mae(mp_rows):.2f}",
          "",
          "  Read: if these are ~equal, the COUNTER is not the lever — the SIGNAL is. The edge path's real",
          "  claim (cheap MOTION oscillates where the VLM phase is stuck) needs raw frames, not this cache.",
          "=" * 76, ""]
    print("\n".join(L))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
