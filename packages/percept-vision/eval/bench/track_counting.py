"""COUNTING track — RepCount-A Percept E2E counting.

RepCount-A is a repetition-counting task, not a frame-level VLM task. This track does not call a raw
VLM baseline. It samples across all valid action types, creates a Percept count goal for each routine,
lets registered edge detectors propose repetition boundaries, and lets the core own the count via
``perceive_boundary``. Provider/detector failures are explicit failed coverage, never cached as success.

Metrics (averaged over videos): obo (off-by-one accuracy, |pred-gt|<=1) and mae (|pred-gt|).
Reference = TransRAC (CVPR'22) on RepCount-A, shown only as an external bar.
"""

from __future__ import annotations

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
_MEASURED = ["e2e:obo", "e2e:mae", "e2e:unresolved_rate"]


def _short_error(exc: Exception) -> str:
    msg = str(exc).replace("\n", " ").strip()
    if "missing Python module" in msg:
        return msg.split(". ", 1)[0]
    if "No module named" in msg:
        return msg.split(". ", 1)[0]
    return msg[:120]


def _result(
    cells,
    headline,
    notes,
    *,
    calls,
    measured,
    modeled,
    excluded=None,
    coverage=None,
    diagnostic=False,
) -> TrackResult:
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
        coverage=list(coverage or []),
        diagnostic=diagnostic,
    )


def score(*, live: bool, n: int) -> TrackResult:
    if not live:
        # OFFLINE stub — no dataset, no pose runtime. Modeled diagnostic until live runs the detector path.
        cells = {
            ("e2e", "obo"): 0.45,
            ("e2e", "mae"): 1.8,
            ("e2e", "unresolved_rate"): 0.20,
        }
        headline = {
            "counting RepCount-A E2E OBO (N)": "0.45 (N=stub) [SOTA TransRAC 0.29]",
            "counting E2E MAE": "1.80",
            "counting unresolved rate": "0.20",
        }
        return _result(
            cells,
            headline,
            "offline modeled stub; live mode runs edge-boundaries -> core count",
            calls=0,
            measured=[],
            modeled=_MEASURED,
            diagnostic=True,
        )

    ev = repcount.sample_repcount_unbiased(n)

    e2e_obos, e2e_maes = [], []
    per_video = []
    unresolved = []
    blind = []  # hero counts flagged escalate-worthy (low coverage/conf) — honesty signal, not a failure
    for e in ev:
        path = repcount.fetch_video(e["index"], e["video_id"])
        gt = e["count"]

        try:
            # The counting track runs the PER-PERFORMER hero path: the eye measures every person, the
            # EntityGraph assigns stable ids, and the brain selects the performer (the multi-person fix for
            # pommel horse / spotters). Channel prior comes from the manifest, never per-action ifs.
            counted = repcount.percept_e2e_rep_count(path, action=e["action"], hero=True)
            e2e_pred = counted["count"]
        except Exception as exc:
            unresolved.append((e["video_id"], e["action"], type(exc).__name__, _short_error(exc)))
            e2e_obos.append(0.0)
            e2e_maes.append(1.0)
            per_video.append((e["video_id"], e["action"], gt, None))
            continue

        if e2e_pred <= 0:
            unresolved.append((e["video_id"], e["action"], "no_count"))

        # OBSERVABILITY: the per-performer hero marks a low-coverage / low-confidence count as BLIND
        # (escalate-worthy — arm C). Surface it so the track never reports a blind count as a clean one.
        # It is an honesty signal, NOT a coverage failure, so it stays out of unresolved/failed.
        hero = counted.get("hero")
        if hero and hero.get("escalate"):
            blind.append((e["video_id"], e["action"], hero.get("blind_reason")))

        denom = max(gt, 1)  # normalized MAE matches RepCount-A reference convention.
        e2e_obos.append(1.0 if abs(e2e_pred - gt) <= 1 else 0.0)
        e2e_maes.append(abs(e2e_pred - gt) / denom)
        per_video.append(
            (
                e["video_id"],
                e["action"],
                gt,
                e2e_pred,
                counted["detectors"],
                counted["frames"],
                len(counted["boundaries"]),
                [round(b["t"], 2) for b in counted["boundaries"][:20]],
                # hero observability: who the brain counted + the phase source + blind flag
                {"subject": hero.get("subject"), "phase": hero.get("phase_source"),
                 "cov": round(hero.get("channel_coverage", 0.0), 3),
                 "htc": round(hero.get("hero_track_coverage", 0.0), 3),
                 "escalate": hero.get("escalate")} if hero else None,
            )
        )

    def _mean(xs):
        return sum(xs) / len(xs) if xs else 0.0

    e2e_obo, e2e_mae = _mean(e2e_obos), _mean(e2e_maes)
    unresolved_rate = len(unresolved) / len(ev) if ev else 0.0

    cells = {
        ("e2e", "obo"): e2e_obo,
        ("e2e", "mae"): e2e_mae,
        ("e2e", "unresolved_rate"): unresolved_rate,
    }
    headline = {
        "counting RepCount-A E2E OBO (N)":
            f"{e2e_obo:.2f} (N={len(ev)}) [SOTA TransRAC 0.29]",
        "counting E2E MAE": f"{e2e_mae:.2f}",
        "counting unresolved rate": f"{unresolved_rate:.2f}",
    }
    mix = Counter(e["action"] for e in ev)
    notes = (
        f"unbiased action sample; per-performer hero -> core count; no raw VLM baseline. "
        f"action mix: {dict(mix)}; unresolved: {unresolved}; "
        f"blind/escalate-worthy (hero observability gate, arm-C candidates): {blind}; "
        f"per-video (vid,action,gt,e2e,detectors,frames,boundaries,boundary_ts,hero): {per_video}"
    )
    failed = len([u for u in unresolved if len(u) >= 4])
    return _result(
        cells,
        headline,
        notes,
        calls=0,
        measured=_MEASURED,
        modeled=[],
        coverage=[{
            "dataset": "repcount-a",
            "expected": len(ev),
            "observed": len(ev) - failed,
            "failed": failed,
            "skipped": 0,
            "notes": "edge-boundary detector(s) -> Percept core count; no VLM calls",
        }],
    )
