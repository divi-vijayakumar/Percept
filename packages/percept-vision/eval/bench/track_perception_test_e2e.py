"""Perception Test action E2E notification benchmark.

Spec shape:

1. Convert the Perception Test action label into a Percept watch goal.
2. Run the product path over the video: runtime edge gate -> candidate Gemini calls -> core fire/ask.
3. Score whether an alert landed inside the annotated action window, plus candidate recall/precision,
   latency, and call reduction against a modeled dense-frame loop.

This is intentionally narrow: action localization only. Sound and richer multimodal tracks should add
parallel E2E adapters rather than reusing the legacy dense VLM loops.
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field

from percept import Goal, Percept
from percept.runtime import RuntimeHealthEvent, RuntimeStats, run

from . import perception_test as pt
from .types import TrackResult

TRACK = "pt-action-e2e"
FPS = 2.0
WINDOW_PAD_S = 0.5


@dataclass
class EventRun:
    video_id: str
    label: str
    goal: str
    onset_s: float
    end_s: float
    frames: int = 0
    judge_calls: int = 0
    fires: list[float] = field(default_factory=list)
    asks: list[float] = field(default_factory=list)
    escalations: list[float] = field(default_factory=list)
    error: str | None = None

    @property
    def lo(self) -> float:
        return self.onset_s

    @property
    def hi(self) -> float:
        return self.end_s + WINDOW_PAD_S

    @property
    def edge_hit(self) -> bool:
        return any(self.lo <= t <= self.hi for t in self.escalations)

    @property
    def fire_hit(self) -> bool:
        return any(self.lo <= t <= self.hi for t in self.fires)

    @property
    def first_latency_s(self) -> float | None:
        hits = [t - self.lo for t in self.fires if self.lo <= t <= self.hi]
        return min(hits) if hits else None


def _action_goal(label: str) -> str:
    raw = label.split(":", 1)[-1]
    words = re.sub(r"[_-]+", " ", raw).strip().lower()
    if not words:
        words = "the annotated action"
    return f"someone is {words}"


async def _score_one(event: dict) -> EventRun:
    goal_text = _action_goal(event["label"])
    result = EventRun(
        video_id=event["video_id"],
        label=event["label"],
        goal=goal_text,
        onset_s=float(event["onset_s"]),
        end_s=float(event["end_s"]),
    )
    stats = RuntimeStats()
    escalations: list[float] = []
    asks: list[float] = []

    core = Percept.create(vision="gemini", discover_plugins=False)
    core.add_goal(
        Goal(
            id=f"pt-{event['video_id']}",
            condition=goal_text,
            say=f"Perception Test action detected: {goal_text}",
        )
    )

    async def on_health(ev: RuntimeHealthEvent) -> None:
        if ev.kind == "escalation" and ev.ts is not None:
            escalations.append(float(ev.ts))

    async def on_ask(_ev) -> None:
        asks.append(-1.0)

    try:
        async for incident in run(
            core,
            source=str(event["path"]),
            fps=FPS,
            stats=stats,
            on_health=on_health,
            on_ask=on_ask,
            snippet_kind="frame",
        ):
            result.fires.append(float(incident.ts))
    except Exception as exc:
        result.error = f"{type(exc).__name__}: {exc}"

    result.frames = stats.frames
    result.judge_calls = stats.judge_calls
    result.escalations = escalations
    result.asks = asks
    return result


def _mean(xs: list[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def _precision(times: list[float], runs: list[EventRun]) -> float | None:
    if not times:
        return None
    hits = 0
    for t in times:
        if any(r.lo <= t <= r.hi for r in runs):
            hits += 1
    return hits / len(times)


def _unavailable(reason: str) -> TrackResult:
    return TrackResult(
        track=TRACK,
        cells={
            ("e2e", "notification_recall"): None,
            ("edge", "candidate_recall"): None,
            ("edge", "candidate_precision_proxy"): None,
            ("e2e", "notification_precision"): None,
            ("e2e", "median_latency_s"): None,
            ("e2e", "call_reduction"): None,
        },
        headline={"Perception Test action E2E": "n/a"},
        excluded=[f"{TRACK}: {reason}"],
        notes=reason,
    )


async def _score_all(events: list[dict]) -> list[EventRun]:
    runs: list[EventRun] = []
    for event in events:
        runs.append(await _score_one(event))
    return runs


def score(*, live: bool, n: int) -> TrackResult:
    if not live:
        return _unavailable("live backend required; offline fake cannot assert visual acuity")

    events = pt.sample_events("action", n)
    for event in events:
        event["path"] = pt.fetch_video(event["video_id"])

    runs = asyncio.run(_score_all(events)) if events else []
    failed = [r for r in runs if r.error]
    observed = len(runs) - len(failed)

    edge_recall = sum(1 for r in runs if r.edge_hit) / len(runs) if runs else None
    notification_recall = sum(1 for r in runs if r.fire_hit) / len(runs) if runs else None
    all_candidates = [t for r in runs for t in r.escalations]
    all_fires = [t for r in runs for t in r.fires]
    candidate_precision = _precision(all_candidates, runs)
    notification_precision = _precision(all_fires, runs)
    latencies = [x for r in runs if (x := r.first_latency_s) is not None]
    latency = sorted(latencies)[len(latencies) // 2] if latencies else None
    dense_calls = sum(r.frames for r in runs)
    e2e_calls = sum(r.judge_calls for r in runs)
    call_reduction = 1.0 - (e2e_calls / dense_calls) if dense_calls else None

    cells = {
        ("e2e", "notification_recall"): notification_recall,
        ("edge", "candidate_recall"): edge_recall,
        ("edge", "candidate_precision_proxy"): candidate_precision,
        ("e2e", "notification_precision"): notification_precision,
        ("e2e", "median_latency_s"): latency,
        ("e2e", "call_reduction"): call_reduction,
        ("dense-modeled", "dense_frame_calls"): dense_calls,
        ("e2e", "candidate_calls"): e2e_calls,
    }
    per_event = [
        {
            "video_id": r.video_id,
            "label": r.label,
            "goal": r.goal,
            "window": [round(r.lo, 3), round(r.hi, 3)],
            "frames": r.frames,
            "judge_calls": r.judge_calls,
            "edge_hit": r.edge_hit,
            "fire_hit": r.fire_hit,
            "fires": [round(t, 3) for t in r.fires],
            "escalations": [round(t, 3) for t in r.escalations[:20]],
            "error": r.error,
        }
        for r in runs
    ]
    return TrackResult(
        track=TRACK,
        cells=cells,
        fmts={
            "notification_recall": "{:.2f}",
            "candidate_recall": "{:.2f}",
            "candidate_precision_proxy": "{:.2f}",
            "notification_precision": "{:.2f}",
            "median_latency_s": "{:.2f}",
            "call_reduction": "{:.2f}",
            "dense_frame_calls": "{:.0f}",
            "candidate_calls": "{:.0f}",
        },
        headline={
            "PT action E2E recall / edge recall": (
                f"{notification_recall:.2f} / {edge_recall:.2f} (N={len(runs)})"
                if notification_recall is not None and edge_recall is not None
                else "n/a"
            ),
            "PT action calls dense-modeled -> E2E": f"{dense_calls} -> {e2e_calls}",
        },
        measured=[
            "e2e:notification_recall",
            "edge:candidate_recall",
            "edge:candidate_precision_proxy",
            "e2e:notification_precision",
            "e2e:median_latency_s",
            "e2e:call_reduction",
            "e2e:candidate_calls",
        ],
        modeled=["dense-modeled:dense_frame_calls"],
        calls=e2e_calls,
        coverage=[{
            "dataset": "perception-test/action",
            "expected": len(events),
            "observed": observed,
            "failed": len(failed),
            "skipped": 0,
            "notes": "runtime E2E path over sampled PT action rows",
        }],
        notes=f"manifest={pt.manifest_hash(events)}; fps={FPS}; per_event={per_event}",
    )
