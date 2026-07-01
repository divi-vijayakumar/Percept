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

from typing import Any

from percept import Goal, Percept
from percept.adapters import Goal as ContractGoal
from percept.config import load_settings
from percept.edge.tier0 import MotionGate, compile_watchspec
from percept.gate import FIRE
from percept.runtime.source import open_source
from percept.sampling import SamplingController, SensePolicy
from percept.sensorplan import sensor_plan_for

from . import perception_test as pt
from .types import TrackResult, bench_vision

TRACK = "pt-action-e2e"
FPS = 2.0  # legacy coarse capture floor — the SensorPlan's sensitive cadence raises the EFFECTIVE rate
WINDOW_PAD_S = 0.5
# Headroom so the capture stream itself isn't the bottleneck: decode at least this dense when the goal's
# SensorPlan asks for a sensitive above-Nyquist cadence (the controller still subsamples to the plan's fps).
_CAPTURE_HEADROOM_FPS = 8.0


def cadence_policy(cadence: dict) -> SensePolicy:
    """Compile a SensorPlan ``cadence`` dict into the fovea's :class:`SensePolicy` (the processing-density
    envelope). Absent knobs fall back to the SensePolicy defaults — so a plain (non-brief) judge goal is
    byte-for-byte the controller default. This is the seam that lets the plan's SENSITIVE cadence drive
    sampling density without any per-activity constant in the adapter."""
    d = SensePolicy()
    return SensePolicy(
        base_fps=cadence.get("base_fps") or d.base_fps,
        max_fps=cadence.get("max_fps") or d.max_fps,
        escalate_threshold=(
            cadence["escalate_threshold"] if cadence.get("escalate_threshold") is not None else d.escalate_threshold
        ),
        hold_s=d.hold_s,
    )


def capture_fps(cadence: dict) -> float:
    """The rate to DECODE the stream at: dense enough that the controller's sensitive cadence isn't starved
    by a coarse capture floor, but never below the legacy ``FPS`` (so non-brief goals decode as before)."""
    want = cadence.get("max_fps") or cadence.get("base_fps") or 0.0
    return max(FPS, min(_CAPTURE_HEADROOM_FPS, float(want)))


def judged_timestamps(
    samples: list[tuple[float, float]], cadence: dict, *, gate_escalations: set[float] | None = None
) -> list[float]:
    """PURE coverage helper (no video, no model) — given a captured ``(t, motion)`` stream and a SensorPlan
    ``cadence``, return the timestamps that would be JUDGED.

    A frame is judged when EITHER the motion gate escalated it (``gate_escalations``, the legacy path —
    unchanged firing logic) OR the fovea's :class:`SamplingController` admits it at the plan's cadence (the
    coverage floor: coast at ``base_fps``, escalate to ``max_fps`` on motion). The union is the fix: a
    quiet-but-in-window frame the motion gate skipped is still judged because the sensitive base_fps coasts
    over it. With the default (non-brief) cadence the controller coasts at the SensePolicy default and the
    union is dominated by the gate exactly as before."""
    ctrl = SamplingController(cadence_policy(cadence))
    esc = gate_escalations or set()
    out: list[float] = []
    for t, motion in samples:
        if ctrl.should_process("judge", t, motion) or t in esc:
            out.append(t)
    return out


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
    trace: list[dict[str, Any]] = field(default_factory=list)
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


class RecordingVision:
    """Gemini judge wrapper that records the latest verdict for benchmark tracing."""

    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self.calls = 0
        self.last_verdict: dict[str, Any] | None = None

    async def detect(self, frame: Any, goals: list[ContractGoal]) -> list:
        return []

    async def detect_events(self, frames: list, goals: list[ContractGoal]) -> list:
        return []

    async def judge(self, condition: str, frame: Any, **kwargs: Any):
        self.calls += 1
        verdict = await self.inner.judge(condition, frame, **kwargs)
        self.last_verdict = {
            "condition": condition,
            "satisfied": bool(verdict.satisfied),
            "confidence": float(verdict.confidence),
            "evidence": str(getattr(verdict, "evidence", "") or "")[:240],
        }
        return verdict


def _silent_reason(verdict: dict[str, Any] | None, threshold: float) -> str:
    if verdict is None:
        return "silent: no_verdict_recorded"
    if not verdict.get("satisfied"):
        return "silent: verdict_not_satisfied"
    if float(verdict.get("confidence") or 0.0) < threshold:
        return "silent: low_confidence"
    return "silent: suppressed_or_refractory"


async def _score_one(event: dict) -> EventRun:
    goal_text = _action_goal(event["label"])
    result = EventRun(
        video_id=event["video_id"],
        label=event["label"],
        goal=goal_text,
        onset_s=float(event["onset_s"]),
        end_s=float(event["end_s"]),
    )
    settings = load_settings()
    vision = RecordingVision(bench_vision(settings))  # PERCEPT_VISION_BACKEND (gemini | gemini-live | ...)
    core = Percept.create(vision=vision, settings=settings, discover_plugins=False)
    goal_id = f"pt-{event['video_id']}"
    # The pt-action notification goal IS a "tell me when X briefly happens" goal: the GT window's duration
    # is the goal's brief-window PRIOR (data the goal-setter has — not a per-activity constant). It compiles
    # the judge sense to a SENSITIVE above-Nyquist cadence so a sub-second action is covered. A non-brief
    # goal carries no such prior and is byte-for-byte the legacy cadence.
    brief_window_s = max(0.0, float(event["end_s"]) - float(event["onset_s"]))
    goal = Goal(
        id=goal_id,
        condition=goal_text,
        say=f"Perception Test action detected: {goal_text}",
        state={"type": "watch", "count": 0, "sense": {"brief_window_s": brief_window_s}},
    )
    core.add_goal(goal)
    plan = sensor_plan_for(goal)
    cadence = plan.cadence
    policy = cadence_policy(cadence)
    ctrl = SamplingController(policy)
    spec = compile_watchspec(list(core.goals.active()), threshold=settings.edge.motion_gate)
    gate = MotionGate(spec, settings=settings)

    try:
        async for t, gate_frame, judge_frame in open_source(str(event["path"]), fps=capture_fps(cadence)):
            result.frames += 1
            decision = gate.evaluate(gate_frame, ts=float(t))
            # JUDGE on EITHER the motion gate's escalation (legacy firing logic — UNCHANGED) OR the fovea's
            # sensitive cadence coasting over a quiet-but-in-window frame (the coverage fix). The motion
            # signal drives the controller's saccade, so a real motion burst still densifies to max_fps.
            coast = ctrl.should_process("judge", float(t), float(decision.signal.motion))
            if not (decision.escalate or coast):
                continue
            result.escalations.append(float(t))
            vision.last_verdict = None
            events = await core.perceive_judged(judge_frame, "pt-action", t=float(t))
            action = "silent"
            for fire in events or []:
                action = getattr(fire, "action", "silent")
                if action == FIRE:
                    result.fires.append(float(t))
                else:
                    result.asks.append(float(t))
            verdict = vision.last_verdict
            result.trace.append({
                "ts": round(float(t), 3),
                "edge": {
                    "decision": "escalate",
                    "motion": round(float(decision.signal.motion), 4),
                    "structural": round(float(decision.signal.structural), 4),
                    "semantic": round(float(decision.signal.semantic), 4),
                    "threshold": round(float(decision.threshold), 4),
                },
                "goal_id": goal_id,
                "goal": goal_text,
                "gemini_verdict": {
                    "satisfied": bool(verdict.get("satisfied")) if verdict else None,
                    "confidence": round(float(verdict.get("confidence") or 0.0), 3) if verdict else None,
                    "evidence": str(verdict.get("evidence", "") if verdict else "")[:240],
                },
                "core_gate_action": action,
                "reason": f"{action}: emitted" if action != "silent" else _silent_reason(verdict, settings.gate.threshold),
            })
    except Exception as exc:
        result.error = f"{type(exc).__name__}: {exc}"

    result.judge_calls = vision.calls
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
            "trace": r.trace,
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
        notes=(
            f"manifest={pt.manifest_hash(events)}; capture_floor_fps={FPS}; "
            f"cadence=sensorplan-sensitive(brief_window); per_event={per_event}"
        ),
    )
