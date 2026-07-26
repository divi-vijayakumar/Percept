"""Reliability / ECE over the golden set's existing labels.

Turns the "1/405 frames < 0.70" anecdote into a single reproducible scalar answering the only question
that matters for consequence-as-threshold: **does the model's confidence DISCRIMINATE right from
wrong?** If it doesn't, no threshold on that axis can separate a true fire from a false one, and the
spec's "fire the child at a lower threshold" lever is dead on arrival (R3) — the fix has to come from a
different signal (categorical/selective ASK, or a higher-capacity judge), not from tuning the threshold.

Offline: reads ``calibration_capture.json`` (the per-frame ``(satisfied, confidence)`` capture
``calibrate.py`` already writes) — no ``PERCEPT_LIVE``, no model calls. Per-frame ground truth is crisp
only on **silent** scenes (every frame should be *unsatisfied*), so the per-frame reliability set is
those frames — a clean mix of correct rejections and confident false-fires, which is exactly the trust
question. ``fire``/``ask`` scenes lack a per-frame rising edge in the capture, so they're summarized at
the scene level as context under the honest-reduction rule, not over-claimed as frame labels.

The metric math here (``ece``/``auroc``/``discrimination``/``reliability_bins``) is pure and
L1-testable on synthetic pairs — no golden set required. ``main()`` does the IO.
"""

from __future__ import annotations

# --- pure metric core (no IO; unit-tested on synthetic pairs) ------------------------------------
# A "pair" is (confidence: float in [0,1], was_correct: bool).


def reliability_bins(pairs: list[tuple[float, bool]], n_bins: int = 10) -> list[dict]:
    """Bin pairs by confidence; per bin report n, mean confidence, observed accuracy.

    A calibrated model sits on the diagonal (acc ≈ conf in every bin). The last bin is closed on the
    right so confidence == 1.0 lands somewhere.
    """
    bins = []
    for i in range(n_bins):
        lo, hi = i / n_bins, (i + 1) / n_bins
        last = i == n_bins - 1
        sel = [(c, ok) for c, ok in pairs if (lo <= c < hi) or (last and c == hi)]
        if not sel:
            bins.append({"lo": lo, "hi": hi, "n": 0, "conf": 0.0, "acc": 0.0})
            continue
        n = len(sel)
        bins.append(
            {
                "lo": lo,
                "hi": hi,
                "n": n,
                "conf": sum(c for c, _ in sel) / n,
                "acc": sum(1.0 for _, ok in sel if ok) / n,
            }
        )
    return bins


def ece(pairs: list[tuple[float, bool]], n_bins: int = 10) -> float:
    """Expected Calibration Error: Σ_b (n_b/N)·|acc(b) − conf(b)|. 0 == perfectly calibrated."""
    if not pairs:
        return 0.0
    n = len(pairs)
    return sum((b["n"] / n) * abs(b["acc"] - b["conf"]) for b in reliability_bins(pairs, n_bins) if b["n"])


def auroc(pairs: list[tuple[float, bool]]) -> float | None:
    """AUROC of confidence vs correctness (Mann–Whitney): P(conf_right > conf_wrong), ties at 0.5.

    0.5 == confidence is no better than chance at telling right from wrong (the lever is dead). Returns
    None when one class is absent (undefined). O(n²) — fine for a few hundred frames.
    """
    pos = [c for c, ok in pairs if ok]
    neg = [c for c, ok in pairs if not ok]
    if not pos or not neg:
        return None
    wins = ties = 0
    for p in pos:
        for q in neg:
            if p > q:
                wins += 1
            elif p == q:
                ties += 1
    return (wins + 0.5 * ties) / (len(pos) * len(neg))


def discrimination(pairs: list[tuple[float, bool]]) -> dict:
    """Does confidence separate right from wrong? mean-conf-right vs mean-conf-wrong + AUROC."""
    right = [c for c, ok in pairs if ok]
    wrong = [c for c, ok in pairs if not ok]
    mcr = sum(right) / len(right) if right else None
    mcw = sum(wrong) / len(wrong) if wrong else None
    return {
        "n_right": len(right),
        "n_wrong": len(wrong),
        "mean_conf_right": mcr,
        "mean_conf_wrong": mcw,
        "gap": (mcr - mcw) if (mcr is not None and mcw is not None) else None,
        "auroc": auroc(pairs),
    }


def below_threshold(cases: list[dict], threshold: float = 0.7) -> tuple[int, int]:
    """The "1/405 < 0.70" headline: how many of all captured confidences fall below the gate default."""
    confs = [conf for c in cases for _, conf in c.get("verdicts", [])]
    return sum(1 for x in confs if x < threshold), len(confs)


def silent_frame_pairs(cases: list[dict]) -> list[tuple[float, bool]]:
    """Per-frame (confidence, was_correct) over SILENT scenes — the crisp-negative reliability set.

    On a silent scene the correct verdict is ``satisfied == False`` for every frame, so
    ``was_correct = not model_satisfied``. This set contains both correct rejections (the right answer)
    and confident false-fires (the wrong one) — exactly what discrimination needs.
    """
    pairs: list[tuple[float, bool]] = []
    for c in cases:
        if c.get("expected") != "silent":
            continue
        for satisfied, conf in c["verdicts"]:
            pairs.append((float(conf), not bool(satisfied)))
    return pairs


def reliability_report(cases: list[dict], *, n_bins: int = 10, threshold: float = 0.7) -> dict:
    """Full reliability scalar set from captured cases. ``discriminative`` is the go/no-go for any
    threshold/consequence-calibration work (R3)."""
    pairs = silent_frame_pairs(cases)
    disc = discrimination(pairs)
    n_below, n_total = below_threshold(cases, threshold)
    auc = disc["auroc"]
    discriminative = auc is not None and abs(auc - 0.5) >= 0.10  # honest heuristic; tracked over time
    return {
        "n_frames_total": n_total,
        "n_reliability_frames": len(pairs),  # crisp silent-scene frames
        "below_threshold": {"threshold": threshold, "n_below": n_below, "n": n_total},
        "ece": ece(pairs, n_bins),
        "bins": reliability_bins(pairs, n_bins),
        "discrimination": disc,
        "discriminative": discriminative,
        "verdict": (
            "confidence discriminates — a threshold lever is viable"
            if discriminative
            else "confidence does NOT discriminate (AUROC≈0.5); the threshold is the wrong lever — "
            "use categorical/selective ASK or a higher-capacity judge"
        ),
    }


# --- CLI (IO; mirrors calibrate.py house style) --------------------------------------------------
def _format(report: dict) -> str:
    d = report["discrimination"]
    bt = report["below_threshold"]
    lines = ["", "=" * 70, "  RELIABILITY / ECE — does confidence discriminate?", "=" * 70, ""]
    lines.append(f"  frames captured: {report['n_frames_total']}   reliability set (silent): {report['n_reliability_frames']}")
    lines.append(f"  below threshold {bt['threshold']}: {bt['n_below']}/{bt['n']}   (the '1/405' headline)")
    lines += ["", "  RELIABILITY DIAGRAM (confidence bin -> n / mean-conf / accuracy):"]
    for b in report["bins"]:
        if not b["n"]:
            continue
        lines.append(f"    [{b['lo']:.1f},{b['hi']:.1f})  n={b['n']:>4}  conf={b['conf']:.3f}  acc={b['acc']:.3f}")
    mcr = "n/a" if d["mean_conf_right"] is None else f"{d['mean_conf_right']:.3f}"
    mcw = "n/a" if d["mean_conf_wrong"] is None else f"{d['mean_conf_wrong']:.3f}"
    auc = "n/a" if d["auroc"] is None else f"{d['auroc']:.3f}"
    lines += [
        "",
        f"  ECE: {report['ece']:.4f}    (0 == perfectly calibrated)",
        f"  mean-conf-right={mcr}  mean-conf-wrong={mcw}  (right={d['n_right']}, wrong={d['n_wrong']})",
        f"  AUROC(conf vs correct): {auc}    discriminative: {report['discriminative']}",
        "",
        f"  VERDICT: {report['verdict']}",
        "",
        "=" * 70,
        "",
    ]
    return "\n".join(lines)


def main() -> int:  # pragma: no cover — CLI entry / IO
    import json
    import os
    from pathlib import Path

    from percept.eval import find_golden_root

    from ._dotenv import load_dotenv

    load_dotenv()
    root = find_golden_root()
    cap_path = Path(os.environ.get("PERCEPT_CALIB_CAPTURE", root / "calibration_capture.json"))
    if not cap_path.exists():
        print(f"reliability: no capture at {cap_path} — run `python -m eval.calibrate` (PERCEPT_LIVE=1) first.")
        return 1
    cases = json.loads(cap_path.read_text())
    print(f"loaded {len(cases)} cases from {cap_path} (offline)")
    report = reliability_report(cases)
    print(_format(report))
    out = Path(os.environ.get("PERCEPT_RELIABILITY_OUT", "reliability.json"))
    out.write_text(json.dumps(report, indent=2, default=str))
    print(f"  wrote {out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
