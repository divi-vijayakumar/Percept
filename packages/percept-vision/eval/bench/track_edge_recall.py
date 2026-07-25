"""Edge event-recall — does the conductor wake on PT event onsets?

This is an edge-only measurement. It does not call the VLM and it does not claim final alert quality.
It asks whether the cheap conductor would wake near annotated Perception Test action onsets, and how
many off-window candidate calls it would create.

The legacy metric stays motion-only for continuity. The policy metric runs the actual server-side
``MotionGate`` with ``WakePolicy`` so benchmark changes track the edge architecture, not a stale shortcut.
Pure edge signals only — no VLM calls.
"""

from __future__ import annotations

from . import perception_test as pt
from .types import TrackResult

TRACK = "edge-recall"
GW, GH = 48, 36
POLICY_CHANGE_WEIGHT = 2.0


def _frame_series(path, fps: float = 4.0):
    import cv2

    from percept.harness.tier0 import Frame

    cap = cv2.VideoCapture(str(path))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(src_fps / fps)))
    out = []
    i = 0
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        if i % step == 0:
            t = i / src_fps
            g = cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), (GW, GH))
            out.append((t, Frame(w=GW, h=GH, pixels=tuple(int(p) for p in g.flatten()))))
        i += 1
    cap.release()
    return out


def _motion_series(path, fps: float = 4.0):
    from percept.harness.tier0 import motion_score

    out, prev = [], None
    for t, f in _frame_series(path, fps=fps):
        out.append((t, motion_score(prev, f)))
        prev = f
    return out


def _event_window(e: dict) -> tuple[float, float]:
    return e["onset_s"], min(e["onset_s"] + 1.5, e["end_s"] + 0.5)


def _score_motion_events(events: list[dict], *, thr: float) -> tuple[float | None, float]:
    escalates, max_motions = [], []
    for e in events:
        series = _motion_series(e["path"])
        lo, hi = _event_window(e)
        win = [m for t, m in series if lo <= t <= hi]
        mm = max(win) if win else 0.0
        max_motions.append(mm)
        escalates.append(mm >= thr)
    recall = (sum(escalates) / len(escalates)) if escalates else None
    avg_m = (sum(max_motions) / len(max_motions)) if max_motions else 0.0
    return recall, avg_m


def _score_policy_events(
    events: list[dict], *, thr: float, change_weight: float = POLICY_CHANGE_WEIGHT
) -> tuple[float | None, float | None, int, int]:
    from percept.adapters import WatchCondition, WatchSpec
    from percept.harness.detectors import WakePolicy
    from percept.harness.tier0 import MotionGate

    hit_events = 0
    tp_frames = 0
    fp_frames = 0
    for e in events:
        lo, hi = _event_window(e)
        gate = MotionGate(
            WatchSpec(
                conditions=[
                    WatchCondition(
                        goal_id="pt-action",
                        anchor_text="the action event is starting",
                        threshold=thr,
                    )
                ]
            ),
            wake_policy=WakePolicy(weights={"change": change_weight}),
        )
        event_hit = False
        for t, frame in _frame_series(e["path"]):
            woke = gate.evaluate(frame, ts=t).escalate
            if not woke:
                continue
            if lo <= t <= hi:
                tp_frames += 1
                event_hit = True
            else:
                fp_frames += 1
        if event_hit:
            hit_events += 1
    recall = (hit_events / len(events)) if events else None
    precision = (tp_frames / (tp_frames + fp_frames)) if (tp_frames + fp_frames) else None
    return recall, precision, tp_frames, fp_frames


def _result(
    motion_recall,
    policy_recall,
    policy_precision,
    *,
    thr: float,
    n_events: int,
    avg_m: float,
    policy_tp: int,
    policy_fp: int,
    notes: str,
) -> TrackResult:
    cells = {
        ("edge", "motion_recall"): motion_recall,
        ("edge", "policy_recall"): policy_recall,
        ("edge", "policy_precision"): policy_precision,
    }
    fmts = {
        "motion_recall": "{:.2f}",
        "policy_recall": "{:.2f}",
        "policy_precision": "{:.2f}",
    }
    return TrackResult(
        track=TRACK,
        cells=cells,
        fmts=fmts,
        headline={
            f"edge event-recall @thr={thr:.2f} (motion clears gate on onset)":
                f"{motion_recall:.2f} (N={n_events})"
                if motion_recall is not None else "n/a",
            f"edge policy-recall @thr={thr:.2f} (WakePolicy change×{POLICY_CHANGE_WEIGHT:g})":
                f"{policy_recall:.2f} (precision proxy {policy_precision:.2f}, TP/FP frames {policy_tp}/{policy_fp})"
                if policy_recall is not None and policy_precision is not None else "n/a",
        },
        measured=["edge:motion_recall", "edge:policy_recall", "edge:policy_precision"],
        coverage=[{
            "dataset": "perception-test/action",
            "expected": n_events,
            "observed": n_events,
            "failed": 0,
            "skipped": 0,
            "notes": "edge-only wake windows; no VLM calls",
        }],
        notes=f"{notes}; thr={thr}; avg max-motion in window={avg_m:.3f}; "
        f"policy change_weight={POLICY_CHANGE_WEIGHT}",
    )


def _unavailable(exc: Exception | None = None) -> TrackResult:
    suffix = f": {type(exc).__name__}: {exc}" if exc is not None else ""
    return TrackResult(
        track=TRACK,
        cells={
            ("edge", "motion_recall"): None,
            ("edge", "policy_recall"): None,
            ("edge", "policy_precision"): None,
        },
        fmts={
            "motion_recall": "{:.2f}",
            "policy_recall": "{:.2f}",
            "policy_precision": "{:.2f}",
        },
        headline={"edge event-recall / policy-recall": "n/a (local golden media unavailable)"},
        measured=[],
        modeled=[],
        excluded=["edge-recall: local golden media unavailable"],
        notes=f"offline stub; golden local measurement unavailable{suffix}",
        status="skipped",
    )


def score(*, live: bool, n: int) -> TrackResult:
    from percept.config import load_settings

    thr = load_settings().edge.motion_gate
    if not live:
        try:
            events = pt.golden_action_events(require_media=True)
            if events:
                sample = events[:n]
                motion_recall, avg_m = _score_motion_events(sample, thr=thr)
                policy_recall, policy_precision, policy_tp, policy_fp = _score_policy_events(sample, thr=thr)
                return _result(
                    motion_recall,
                    policy_recall,
                    policy_precision,
                    thr=thr,
                    n_events=len(sample),
                    avg_m=avg_m,
                    policy_tp=policy_tp,
                    policy_fp=policy_fp,
                    notes="golden-v1 Perception Test sample",
                )
        except Exception as exc:
            return _unavailable(exc)
        return _unavailable()

    events = pt.sample_events("action", n)
    for e in events:
        e["path"] = pt.fetch_video(e["video_id"])
    motion_recall, avg_m = _score_motion_events(events, thr=thr)
    policy_recall, policy_precision, policy_tp, policy_fp = _score_policy_events(events, thr=thr)
    return _result(
        motion_recall,
        policy_recall,
        policy_precision,
        thr=thr,
        n_events=len(events),
        avg_m=avg_m,
        policy_tp=policy_tp,
        policy_fp=policy_fp,
        notes=f"manifest={pt.manifest_hash(events)}",
    )
