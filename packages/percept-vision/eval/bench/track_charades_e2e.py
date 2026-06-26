"""Charades notification E2E benchmark.

Spec shape:

1. Convert every Charades action interval on a sampled video into a Percept watch goal.
2. Run the product path over the video: edge gate -> candidate Gemini judgments -> core fires.
3. Score notification hits inside annotated intervals and call reduction against a modeled dense loop.

This track deliberately does not claim dense temporal-localization parity with TML. Charades has
start/end intervals, while Percept's current public watch contract emits rising-edge notifications.
We therefore report interval notification metrics as the publishable E2E cells and keep sparse span
mIoU as a diagnostic cell derived only from candidate judgments.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from percept import Goal, Percept
from percept.adapters import Goal as ContractGoal
from percept.config import load_settings
from percept.edge.tier0 import MotionGate, compile_watchspec
from percept.gate import FIRE
from percept.runtime.source import open_source

from .types import TrackResult

TRACK = "charades-e2e"
ROOT = Path(__file__).resolve().parents[4]
RAW = ROOT / "eval" / "bench-staging" / "charades" / "raw"
VID_DIR = RAW / "Charades_v1_480"
SEL = RAW / "selection.json"
FPS = 1.0
WINDOW_PAD_S = 1.0


@dataclass
class ActionGoal:
    goal_id: str
    condition: str
    desc: str
    segments: list[tuple[float, float]]


@dataclass
class VideoRun:
    video_id: str
    path: Path
    length_s: float
    goals: list[ActionGoal]
    frames: int = 0
    escalations: list[float] = field(default_factory=list)
    fires: list[tuple[str, float, float]] = field(default_factory=list)
    asks: list[tuple[str, float]] = field(default_factory=list)
    verdicts: dict[str, list[tuple[float, bool, float]]] = field(default_factory=dict)
    judge_calls: int = 0
    error: str | None = None


class RecordingVision:
    """A Gemini judge wrapper for E2E benchmarks.

    The generic entity detector is intentionally a no-op here: Charades scoring is goal judgment,
    not object graph quality. This keeps call accounting tied to candidate condition judgments.
    """

    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self.current_t: float | None = None
        self.calls = 0
        self.by_condition: dict[str, list[tuple[float, bool, float]]] = {}

    async def detect(self, frame: Any, goals: list[ContractGoal]) -> list:
        return []

    async def detect_events(self, frames: list, goals: list[ContractGoal]) -> list:
        return []

    async def judge(self, condition: str, frame: Any, **kwargs: Any):
        self.calls += 1
        verdict = await self.inner.judge(condition, frame, **kwargs)
        t = 0.0 if self.current_t is None else float(self.current_t)
        self.by_condition.setdefault(condition, []).append(
            (t, bool(verdict.satisfied), float(verdict.confidence))
        )
        return verdict


def _condition(desc: str) -> str:
    text = desc.strip()
    if not text:
        text = "the annotated action"
    return "someone is " + text[0].lower() + text[1:]


def _load_rows(n: int) -> list[dict[str, Any]]:
    if not SEL.exists():
        return []
    rows = json.loads(SEL.read_text())
    out = []
    for row in rows[:n]:
        path = VID_DIR / f"{row['id']}.mp4"
        if path.exists():
            out.append(row)
    return out


def _goals_for(row: dict[str, Any]) -> list[ActionGoal]:
    by_desc: dict[str, list[tuple[float, float]]] = {}
    for desc, start, end in row.get("actions", []):
        by_desc.setdefault(str(desc), []).append((float(start), float(end)))
    return [
        ActionGoal(
            goal_id=f"charades-{row['id']}-{idx}",
            condition=_condition(desc),
            desc=desc,
            segments=segments,
        )
        for idx, (desc, segments) in enumerate(sorted(by_desc.items()))
    ]


def _inside_any(t: float, goals: list[ActionGoal], goal_id: str | None = None) -> bool:
    for goal in goals:
        if goal_id is not None and goal.goal_id != goal_id:
            continue
        for start, end in goal.segments:
            if start <= t <= end + WINDOW_PAD_S:
                return True
    return False


def _interval_metrics(runs: list[VideoRun]) -> dict[str, float | int | None]:
    tp = fp = fn = 0
    latencies: list[float] = []
    for run in runs:
        fires_by_goal: dict[str, list[float]] = {}
        for goal_id, t, _conf in run.fires:
            fires_by_goal.setdefault(goal_id, []).append(t)
        for goal in run.goals:
            used: set[int] = set()
            gfires = sorted(fires_by_goal.get(goal.goal_id, []))
            for start, end in goal.segments:
                hit = None
                for idx, t in enumerate(gfires):
                    if idx not in used and start <= t <= end + WINDOW_PAD_S:
                        hit = idx
                        break
                if hit is None:
                    fn += 1
                else:
                    used.add(hit)
                    tp += 1
                    latencies.append(gfires[hit] - start)
            fp += len(gfires) - len(used)
    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    median_latency = sorted(latencies)[len(latencies) // 2] if latencies else None
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "median_latency_s": median_latency}


def _candidate_metrics(runs: list[VideoRun]) -> dict[str, float | int | None]:
    segments = 0
    edge_hits = 0
    candidates = 0
    candidate_hits = 0
    for run in runs:
        for t in run.escalations:
            candidates += 1
            if _inside_any(t, run.goals):
                candidate_hits += 1
        for goal in run.goals:
            for start, end in goal.segments:
                segments += 1
                if any(start <= t <= end + WINDOW_PAD_S for t in run.escalations):
                    edge_hits += 1
    return {
        "candidate_recall": edge_hits / segments if segments else None,
        "candidate_precision_proxy": candidate_hits / candidates if candidates else None,
        "candidate_frames": candidates,
    }


def _iou(a: tuple[float, float], b: tuple[float, float]) -> float:
    lo = max(a[0], b[0])
    hi = min(a[1], b[1])
    inter = max(0.0, hi - lo)
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0.0


def _sparse_spans(points: list[tuple[float, bool, float]]) -> list[tuple[float, float]]:
    spans: list[tuple[float, float]] = []
    start: float | None = None
    last_t: float | None = None
    for t, sat, conf in sorted(points):
        last_t = t
        yes = sat and conf >= 0.5
        if yes and start is None:
            start = t
        elif not yes and start is not None:
            if t > start:
                spans.append((start, t))
            start = None
    if start is not None and last_t is not None:
        spans.append((start, max(start + (1.0 / FPS), last_t + (1.0 / FPS))))
    return spans


def _sparse_miou(runs: list[VideoRun]) -> float | None:
    scores: list[float] = []
    for run in runs:
        condition_to_goal = {g.condition: g for g in run.goals}
        for condition, points in run.verdicts.items():
            goal = condition_to_goal.get(condition)
            if goal is None:
                continue
            pred = _sparse_spans(points)
            for gt in goal.segments:
                scores.append(max((_iou(gt, p) for p in pred), default=0.0))
    return sum(scores) / len(scores) if scores else None


async def _score_one(row: dict[str, Any]) -> VideoRun:
    from percept.backends.gemini import GeminiVision

    goals = _goals_for(row)
    run = VideoRun(
        video_id=str(row["id"]),
        path=VID_DIR / f"{row['id']}.mp4",
        length_s=float(row.get("length") or 0.0),
        goals=goals,
    )
    settings = load_settings()
    recorder = RecordingVision(GeminiVision(settings))
    core = Percept.create(vision=recorder, settings=settings, discover_plugins=False)
    for goal in goals:
        core.add_goal(Goal(id=goal.goal_id, condition=goal.condition, say=f"Charades action detected: {goal.desc}"))
    spec = compile_watchspec(list(core.goals.active()), threshold=settings.edge.motion_gate)
    gate = MotionGate(spec, settings=settings)

    try:
        async for t, gate_frame, judge_frame in open_source(str(run.path), fps=FPS):
            run.frames += 1
            decision = gate.evaluate(gate_frame, ts=float(t))
            if not decision.escalate:
                continue
            run.escalations.append(float(t))
            recorder.current_t = float(t)
            events = await core.perceive_judged(judge_frame, "charades", t=float(t))
            for event in events:
                if event.action == FIRE:
                    conf = float(event.verdict.confidence) if event.verdict else 0.0
                    run.fires.append((event.goal_id, float(t), conf))
                else:
                    run.asks.append((event.goal_id, float(t)))
    except Exception as exc:
        run.error = f"{type(exc).__name__}: {exc}"
    finally:
        recorder.current_t = None
        run.judge_calls = recorder.calls
        run.verdicts = recorder.by_condition
    return run


async def _score_all(rows: list[dict[str, Any]]) -> list[VideoRun]:
    runs: list[VideoRun] = []
    for row in rows:
        runs.append(await _score_one(row))
    return runs


def _unavailable(reason: str) -> TrackResult:
    return TrackResult(
        track=TRACK,
        cells={
            ("e2e", "notification_recall"): None,
            ("e2e", "notification_precision"): None,
            ("edge", "candidate_recall"): None,
            ("edge", "candidate_precision_proxy"): None,
            ("e2e", "call_reduction"): None,
            ("dense-modeled", "dense_goal_frame_calls"): None,
        },
        headline={"Charades notification E2E": "n/a"},
        excluded=[f"{TRACK}: {reason}"],
        notes=reason,
    )


def score(*, live: bool, n: int) -> TrackResult:
    if not live:
        return _unavailable("live backend required; offline fake cannot assert Charades visual acuity")

    rows = _load_rows(n)
    if not rows:
        return _unavailable(f"no Charades selection found at {SEL}")

    runs = asyncio.run(_score_all(rows))
    failed = [r for r in runs if r.error]
    observed = len(runs) - len(failed)

    interval = _interval_metrics(runs)
    candidates = _candidate_metrics(runs)
    dense_calls = sum(r.frames * len(r.goals) for r in runs)
    e2e_calls = sum(r.judge_calls for r in runs)
    call_reduction = 1.0 - (e2e_calls / dense_calls) if dense_calls else None
    sparse_miou = _sparse_miou(runs)

    cells = {
        ("e2e", "notification_recall"): interval["recall"],
        ("e2e", "notification_precision"): interval["precision"],
        ("edge", "candidate_recall"): candidates["candidate_recall"],
        ("edge", "candidate_precision_proxy"): candidates["candidate_precision_proxy"],
        ("e2e", "median_latency_s"): interval["median_latency_s"],
        ("e2e", "call_reduction"): call_reduction,
        ("e2e", "candidate_calls"): e2e_calls,
        ("dense-modeled", "dense_goal_frame_calls"): dense_calls,
        ("e2e", "sparse_span_miou_diagnostic"): sparse_miou,
    }
    per_video = [
        {
            "video_id": r.video_id,
            "frames": r.frames,
            "goals": len(r.goals),
            "segments": sum(len(g.segments) for g in r.goals),
            "escalations": len(r.escalations),
            "judge_calls": r.judge_calls,
            "fires": [{"goal_id": gid, "t": round(t, 3), "confidence": round(conf, 3)} for gid, t, conf in r.fires],
            "error": r.error,
        }
        for r in runs
    ]
    return TrackResult(
        track=TRACK,
        cells=cells,
        fmts={
            "notification_recall": "{:.2f}",
            "notification_precision": "{:.2f}",
            "candidate_recall": "{:.2f}",
            "candidate_precision_proxy": "{:.2f}",
            "median_latency_s": "{:.2f}",
            "call_reduction": "{:.2f}",
            "candidate_calls": "{:.0f}",
            "dense_goal_frame_calls": "{:.0f}",
            "sparse_span_miou_diagnostic": "{:.2f}",
        },
        headline={
            "Charades notif recall / precision": (
                f"{interval['recall']:.2f} / {interval['precision']:.2f} (N={len(runs)})"
                if interval["recall"] is not None and interval["precision"] is not None
                else "n/a"
            ),
            "Charades calls dense-modeled -> E2E": f"{dense_calls} -> {e2e_calls}",
        },
        measured=[
            "e2e:notification_recall",
            "e2e:notification_precision",
            "edge:candidate_recall",
            "edge:candidate_precision_proxy",
            "e2e:median_latency_s",
            "e2e:call_reduction",
            "e2e:candidate_calls",
            "e2e:sparse_span_miou_diagnostic",
        ],
        modeled=["dense-modeled:dense_goal_frame_calls"],
        calls=e2e_calls,
        coverage=[{
            "dataset": "charades/test-selection",
            "expected": len(rows),
            "observed": observed,
            "failed": len(failed),
            "skipped": 0,
            "notes": "goal-labeled Charades videos through edge-gated Percept watch path",
        }],
        notes=f"selection={SEL}; fps={FPS}; metric=notification interval hit, sparse mIoU diagnostic only; per_video={per_video}",
    )
