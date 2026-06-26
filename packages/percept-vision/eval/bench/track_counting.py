"""COUNTING track — RepCount-A E2E counting.

RepCount-A is a repetition-counting task, not a frame-level VLM task. This track does not call a raw
VLM baseline. It samples across all valid action types, runs the current local Percept counting path,
and reports unresolved/fallback pressure explicitly. Detector-specific or pose-friendly subsets belong
in diagnostic cards, not the publishable RepCount-A row.

This exercises the existing Percept architecture: the core/brain owns routine start/stop semantics,
the action-specific unit to count, and the fallback policy. The edge attempts the cheap measurement. If
a hard action such as pommelhorse defeats the current detector, the row stays in the sample and is
reported as a failure or unresolved case.

Metrics (averaged over videos): obo (off-by-one accuracy, |pred-gt|<=1) and mae (|pred-gt|).
Reference = TransRAC (CVPR'22) on RepCount-A, shown only as an external bar.
"""

from __future__ import annotations

import os
from collections import Counter

from . import repcount
from .types import TrackResult

TRACK = "counting"

# TransRAC (CVPR'22) RepCount-A test: OBO 0.29, MAE 0.44 — external reference bar.
_REFERENCE = {
    "obo": {"value": 0.29, "by": "TransRAC CVPR'22"},
    "mae": {"value": 0.44, "by": "TransRAC"},
}
_FMTS = {"obo": "{:.2f}", "mae": "{:.2f}", "unresolved_rate": "{:.2f}"}
_MEASURED = ["e2e obo", "e2e mae", "e2e unresolved_rate"]


def _short_error(exc: Exception) -> str:
    msg = str(exc).replace("\n", " ").strip()
    if "missing Python module" in msg:
        return msg.split(". ", 1)[0]
    if "No module named" in msg:
        return msg.split(". ", 1)[0]
    return msg[:120]


def _result(cells, headline, notes, *, calls, measured, modeled, excluded=None) -> TrackResult:
    return TrackResult(
        track=TRACK,
        cells=cells,
        fmts=_FMTS,
        headline=headline,
        measured=measured,
        modeled=modeled,
        reference=_REFERENCE,
        calls=calls,
        notes=notes,
        excluded=list(excluded or []),
    )


def score(*, live: bool, n: int) -> TrackResult:
    if not live:
        # OFFLINE stub — no dataset, no API. Modeled until the full action-set runner executes.
        cells = {
            ("raw", "obo"): None,
            ("raw", "mae"): None,
            ("raw", "unresolved_rate"): None,
            ("core", "obo"): None,
            ("core", "mae"): None,
            ("core", "unresolved_rate"): None,
            ("e2e", "obo"): 0.45,
            ("e2e", "mae"): 1.8,
            ("e2e", "unresolved_rate"): 0.20,
        }
        headline = {
            "counting RepCount-A full-E2E OBO (N)": "0.45 (N=stub) [SOTA TransRAC 0.29]",
            "counting full-E2E MAE": "1.80",
            "counting unresolved rate": "0.20",
        }
        return _result(
            cells,
            headline,
            "offline stub",
            calls=0,
            measured=[],
            modeled=_MEASURED,
        )

    if os.environ.get("PERCEPT_ENABLE_POSE_COUNT") != "1":
        cells = {
            ("raw", "obo"): None,
            ("raw", "mae"): None,
            ("raw", "unresolved_rate"): None,
            ("core", "obo"): None,
            ("core", "mae"): None,
            ("core", "unresolved_rate"): None,
            ("e2e", "obo"): None,
            ("e2e", "mae"): None,
            ("e2e", "unresolved_rate"): None,
        }
        return _result(
            cells,
            {
                "counting RepCount-A full-E2E OBO (N)": "n/a (pose runner disabled)",
                "counting full-E2E MAE": "n/a",
                "counting unresolved rate": "n/a",
            },
            "pose counting skipped locally; set PERCEPT_ENABLE_POSE_COUNT=1 to run the configured pose provider",
            calls=0,
            measured=[],
            modeled=[],
            excluded=["counting: pose runner disabled (PERCEPT_ENABLE_POSE_COUNT!=1)"],
        )

    ev = repcount.sample_repcount_unbiased(n)

    e2e_obos, e2e_maes = [], []
    per_video = []
    unresolved = []
    for e in ev:
        path = repcount.fetch_video(e["index"], e["video_id"])
        gt = e["count"]

        try:
            e2e_pred = repcount.pose_rep_count(path)
        except Exception as exc:
            unresolved.append((e["video_id"], e["action"], type(exc).__name__, _short_error(exc)))
            e2e_obos.append(0.0)
            e2e_maes.append(1.0)
            per_video.append((e["video_id"], e["action"], gt, None))
            continue

        if e2e_pred <= 0:
            unresolved.append((e["video_id"], e["action"], "no_count"))

        denom = max(gt, 1)  # normalized MAE matches RepCount-A reference convention.
        e2e_obos.append(1.0 if abs(e2e_pred - gt) <= 1 else 0.0)
        e2e_maes.append(abs(e2e_pred - gt) / denom)
        per_video.append((e["video_id"], e["action"], gt, e2e_pred))

    def _mean(xs):
        return sum(xs) / len(xs) if xs else 0.0

    e2e_obo, e2e_mae = _mean(e2e_obos), _mean(e2e_maes)
    unresolved_rate = len(unresolved) / len(ev) if ev else 0.0

    cells = {
        ("raw", "obo"): None,
        ("raw", "mae"): None,
        ("raw", "unresolved_rate"): None,
        ("core", "obo"): None,
        ("core", "mae"): None,
        ("core", "unresolved_rate"): None,
        ("e2e", "obo"): e2e_obo,
        ("e2e", "mae"): e2e_mae,
        ("e2e", "unresolved_rate"): unresolved_rate,
    }
    headline = {
        "counting RepCount-A full-E2E OBO (N)":
            f"{e2e_obo:.2f} (N={len(ev)}) [SOTA TransRAC 0.29]",
        "counting full-E2E MAE": f"{e2e_mae:.2f}",
        "counting unresolved rate": f"{unresolved_rate:.2f}",
    }
    mix = Counter(e["action"] for e in ev)
    notes = (
        f"unbiased action sample; no raw VLM baseline. action mix: {dict(mix)}; "
        f"unresolved: {unresolved}; per-video (vid,action,gt,e2e): {per_video}"
    )
    return _result(
        cells,
        headline,
        notes,
        calls=0,
        measured=_MEASURED,
        modeled=[],
    )
