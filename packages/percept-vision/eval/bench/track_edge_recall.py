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


def score(*, live: bool, n: int) -> TrackResult:
    if not live:
        return TrackResult(
            track=TRACK, cells={("e2e", "event_recall"): 0.90}, fmts={"event_recall": "{:.2f}"},
            headline={"edge event-recall (motion-gate escalates on onset)": "0.90 (offline stub)"},
            notes="offline stub",
        )

    from percept.config import load_settings

    thr = load_settings().edge.motion_gate
    events = pt.sample_events("action", n)
    escalates, max_motions = [], []
    for e in events:
        series = _motion_series(pt.fetch_video(e["video_id"]))
        lo, hi = e["onset_s"], min(e["onset_s"] + 1.5, e["end_s"] + 0.5)
        win = [m for t, m in series if lo <= t <= hi]
        mm = max(win) if win else 0.0
        max_motions.append(mm)
        escalates.append(mm >= thr)
    recall = (sum(escalates) / len(escalates)) if escalates else None
    avg_m = (sum(max_motions) / len(max_motions)) if max_motions else 0.0

    return TrackResult(
        track=TRACK,
        cells={("e2e", "event_recall"): recall},
        fmts={"event_recall": "{:.2f}"},
        headline={
            f"edge event-recall @thr={thr:.2f} (motion clears gate on onset)":
                f"{recall:.2f} (N={len(events)}) -> e2e_acc = core x {recall:.2f}" if recall is not None else "n/a",
        },
        measured=["e2e:event_recall"],
        notes=f"manifest={pt.manifest_hash(events)}; thr={thr}; avg max-motion in window={avg_m:.3f}",
    )
