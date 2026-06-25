"""Phase B-4: edge event-recall — does the motion-gate escalate on PT event onsets? (validates amendment #3).

The card models e2e-relational accuracy as `core × edge-call-reduction`, ASSUMING the motion-gate escalates
on every true event. This measures that assumption directly: over PT action onsets, the fraction where motion
clears the gate threshold somewhere in the event window. If it's < ~1.0, the edge SKIPS low-motion events and
the honest modeling is `e2e accuracy = core × edge_event_recall`, not ≈ core. Pure motion — no VLM calls.
"""

from __future__ import annotations

from . import perception_test as pt
from .types import TrackResult

TRACK = "edge-recall"
GW, GH = 48, 36


def _motion_series(path, fps: float = 4.0):
    import cv2

    from percept_harness.tier0 import Frame, motion_score

    from eval.count_video import decode

    _, full = decode(str(path), fps)
    out, prev = [], None
    for t, bgr in full:
        g = cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), (GW, GH))
        f = Frame(w=GW, h=GH, pixels=tuple(int(p) for p in g.flatten()))
        out.append((t, motion_score(prev, f)))
        prev = f
    return out


def _score_events(events: list[dict], *, thr: float) -> tuple[float | None, float]:
    escalates, max_motions = [], []
    for e in events:
        series = _motion_series(e["path"])
        lo, hi = e["onset_s"], min(e["onset_s"] + 1.5, e["end_s"] + 0.5)
        win = [m for t, m in series if lo <= t <= hi]
        mm = max(win) if win else 0.0
        max_motions.append(mm)
        escalates.append(mm >= thr)
    recall = (sum(escalates) / len(escalates)) if escalates else None
    avg_m = (sum(max_motions) / len(max_motions)) if max_motions else 0.0
    return recall, avg_m


def _result(recall, *, thr: float, n_events: int, avg_m: float, notes: str) -> TrackResult:
    return TrackResult(
        track=TRACK,
        cells={("e2e", "event_recall"): recall},
        fmts={"event_recall": "{:.2f}"},
        headline={
            f"edge event-recall @thr={thr:.2f} (motion clears gate on onset)":
                f"{recall:.2f} (N={n_events}) -> e2e_acc = core x {recall:.2f}" if recall is not None else "n/a",
        },
        measured=["e2e:event_recall"],
        notes=f"{notes}; thr={thr}; avg max-motion in window={avg_m:.3f}",
    )


def score(*, live: bool, n: int) -> TrackResult:
    from percept.config import load_settings

    thr = load_settings().edge.motion_gate
    if not live:
        try:
            events = pt.golden_action_events(require_media=True)
            if events:
                recall, avg_m = _score_events(events[:n], thr=thr)
                return _result(recall, thr=thr, n_events=len(events[:n]), avg_m=avg_m,
                               notes="golden-v1 Perception Test sample")
        except Exception as exc:
            return TrackResult(
                track=TRACK,
                cells={("e2e", "event_recall"): None},
                fmts={"event_recall": "{:.2f}"},
                headline={"edge event-recall (motion-gate escalates on onset)": "n/a (local golden media unavailable)"},
                measured=[],
                modeled=[],
                excluded=["edge-recall: local golden media unavailable"],
                notes=f"offline stub; golden local measurement unavailable: {type(exc).__name__}: {exc}",
            )
        return TrackResult(
            track=TRACK,
            cells={("e2e", "event_recall"): None},
            fmts={"event_recall": "{:.2f}"},
            headline={"edge event-recall (motion-gate escalates on onset)": "n/a (local golden media unavailable)"},
            measured=[],
            modeled=[],
            excluded=["edge-recall: local golden media unavailable"],
            notes="offline stub; golden local media unavailable",
        )

    events = pt.sample_events("action", n)
    for e in events:
        e["path"] = pt.fetch_video(e["video_id"])
    recall, avg_m = _score_events(events, thr=thr)
    return _result(recall, thr=thr, n_events=len(events), avg_m=avg_m, notes=f"manifest={pt.manifest_hash(events)}")
