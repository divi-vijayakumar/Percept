"""Perception Test action E2E notification benchmark.

Spec shape:

1. Convert the Perception Test action label into a Percept watch goal.
2. Run the product path over the video: runtime edge gate -> budgeted candidate judge calls -> core fire/ask.
3. Score whether an alert landed inside the annotated action window, plus candidate recall/precision,
   latency, and call reduction against a modeled dense-frame loop.

This is intentionally narrow: action localization only. Sound and richer multimodal tracks should add
parallel E2E adapters rather than reusing the legacy dense VLM loops.
"""

from __future__ import annotations

import asyncio
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median
from typing import Any

from percept import Goal, Percept
from percept.adapters import Goal as ContractGoal
from percept.behavior import program_of
from percept.config import load_settings
from percept.contracts import JudgmentTiming
from percept.edge.tier0 import MotionGate, compile_watchspec
from percept.gate import FIRE, Gate
from percept.runtime.source import open_source
from percept.sampling import SamplingController, SensePolicy
from percept.sensorplan import sensor_plan_for

from . import perception_test as pt
from .budget import BudgetEnforcer, build_budget_enforcer
from .types import TrackResult, bench_vision

TRACK = "pt-action-e2e"
FPS = 2.0  # legacy coarse capture floor — the SensorPlan's sensitive cadence raises the EFFECTIVE rate
WINDOW_PAD_S = 0.5
# Preregistered PRODUCT configuration, not a value read from the benchmark annotation.  The former
# adapter copied each ground-truth interval's duration into the Watch's sensing prior, which leaks
# scoring data into runtime admission.  A client configuring a general action Watch does not know the
# future occurrence duration, so every PT case now gets this same conservative one-second prior.
ACTION_BRIEF_WINDOW_S = 1.0
ACTION_TEMPORAL_WINDOW_S = 2.0
ACTION_TEMPORAL_FRAMES = 5
ACTION_TEMPORAL_SAMPLING = "uniform"
# Headroom so the capture stream itself isn't the bottleneck: decode at least this dense when the goal's
# SensorPlan asks for a sensitive above-Nyquist cadence (the controller still subsamples to the plan's fps).
_CAPTURE_HEADROOM_FPS = 8.0
# A finite fixture must be lossless, but an unbounded decoded-frame backlog would turn provider
# latency into an uncontrolled memory spike. A small blocking queue bounds memory and backpressures
# only the benchmark decoder; ``open_source`` retains latest-frame semantics everywhere else.
_FIXTURE_QUEUE_SIZE = 8


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
            cadence["escalate_threshold"]
            if cadence.get("escalate_threshold") is not None
            else d.escalate_threshold
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


def admission_cause(*, gate: bool, cadence: bool) -> str:
    """Name why a frame reached the judge without conflating the two admission paths."""
    if gate and cadence:
        return "gate+cadence"
    if gate:
        return "gate"
    if cadence:
        return "cadence"
    return "none"


@dataclass
class DecisionObservation:
    """A case-bound decision with source alignment and monotonic latency kept separate."""

    case_id: str
    video_id: str
    source_media_ts: float
    observed_to_fire_s: float
    judge_latency_s: float
    action_onset_to_fire_s: float | None
    timing: JudgmentTiming
    revision_id: str | None = None


def _matched_event_pairs(
    timestamps: list[float],
    windows: list[tuple[float, float]],
) -> list[tuple[int, int]]:
    """Greedily one-to-one match timestamps to interval events by earliest end."""
    unmatched = list(range(len(windows)))
    pairs: list[tuple[int, int]] = []
    for timestamp_index, timestamp in sorted(
        enumerate(timestamps),
        key=lambda item: (item[1], item[0]),
    ):
        candidates = [
            index
            for index in unmatched
            if windows[index][0] <= timestamp <= windows[index][1] + WINDOW_PAD_S
        ]
        if not candidates:
            continue
        chosen = min(candidates, key=lambda index: windows[index][1])
        unmatched.remove(chosen)
        pairs.append((timestamp_index, chosen))
    return pairs


@dataclass
class EventRun:
    case_id: str
    video_id: str
    label: str
    goal: str
    onset_s: float
    end_s: float
    positive_windows: list[tuple[float, float]] = field(default_factory=list)
    frames: int = 0
    dropped_frames: int = 0
    judge_calls: int = 0
    fires: list[DecisionObservation] = field(default_factory=list)
    asks: list[float] = field(default_factory=list)
    escalations: list[float] = field(default_factory=list)
    trace: list[dict[str, Any]] = field(default_factory=list)
    configuration: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    @property
    def lo(self) -> float:
        return self.onset_s

    @property
    def hi(self) -> float:
        return self.end_s + WINDOW_PAD_S

    @property
    def windows(self) -> list[tuple[float, float]]:
        return self.positive_windows or [(self.onset_s, self.end_s)]

    def contains(self, timestamp: float) -> bool:
        return any(onset <= timestamp <= end + WINDOW_PAD_S for onset, end in self.windows)

    @property
    def edge_hit(self) -> bool:
        return self.edge_hits > 0

    @property
    def edge_hits(self) -> int:
        return len(_matched_event_pairs(self.escalations, self.windows))

    @property
    def fire_hit(self) -> bool:
        return self.fire_hits > 0

    @property
    def fire_hits(self) -> int:
        return len(
            _matched_event_pairs(
                [fire.source_media_ts for fire in self.fires],
                self.windows,
            )
        )

    @property
    def matched_fire_events(
        self,
    ) -> list[tuple[DecisionObservation, tuple[float, float]]]:
        return [
            (self.fires[fire_index], self.windows[window_index])
            for fire_index, window_index in _matched_event_pairs(
                [fire.source_media_ts for fire in self.fires],
                self.windows,
            )
        ]

    def reconcile_fire_windows(self) -> None:
        """Bind each FIRE to the same one-to-one window used by aggregate metrics."""
        for fire in self.fires:
            fire.action_onset_to_fire_s = None
        for fire, window in self.matched_fire_events:
            fire.action_onset_to_fire_s = (
                fire.source_media_ts - window[0]
            ) + fire.observed_to_fire_s


def _action_goal(label: str) -> str:
    raw = label.split(":", 1)[-1]
    words = re.sub(r"[_-]+", " ", raw).strip().lower()
    if not words:
        words = "the annotated action"
    return f"someone performs one discrete occurrence of {words}"


def _action_goal_state() -> dict[str, Any]:
    """The fixed, annotation-independent action Watch used by every benchmark case."""
    return {
        "type": "watch",
        "count": 0,
        "rearm": "each_event",
        "sense": {"brief_window_s": ACTION_BRIEF_WINDOW_S},
        "perception": {
            "temporal": "strip",
            "n_frames": ACTION_TEMPORAL_FRAMES,
            "sampling": ACTION_TEMPORAL_SAMPLING,
            "window_s": ACTION_TEMPORAL_WINDOW_S,
        },
    }


class RecordingVision:
    """Judge wrapper that records calls, verdicts, and swallowed backend failures."""

    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self.calls = 0
        self.last_verdict: dict[str, Any] | None = None
        self.last_error: Exception | None = None

    async def detect(self, frame: Any, goals: list[ContractGoal]) -> list:
        return []

    async def detect_events(self, frames: list, goals: list[ContractGoal]) -> list:
        return []

    async def judge(self, condition: str, frame: Any, **kwargs: Any):
        self.calls += 1
        self.last_verdict = None
        self.last_error = None
        try:
            verdict = await self.inner.judge(condition, frame, **kwargs)
        except Exception as exc:
            # Percept deliberately degrades backend exceptions to ASK/SILENT. Preserve the
            # original exception so a paid benchmark can terminate instead of scoring a
            # partial run as if degradation were a model judgment.
            self.last_error = exc
            raise
        self.last_verdict = {
            "condition": condition,
            "satisfied": bool(verdict.satisfied),
            "confidence": float(verdict.confidence),
            "evidence": str(getattr(verdict, "evidence", "") or "")[:240],
            "phase": str(getattr(verdict, "phase", "") or "") or None,
            "action_present": getattr(verdict, "action_present", None),
            "events": list(getattr(verdict, "events", []) or []),
            "usage": dict(getattr(verdict, "usage", {}) or {}),
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


def _decision_observation(
    fire: Any,
    run: EventRun,
) -> DecisionObservation | None:
    """Materialize one timing-bearing decision without mixing clock domains."""
    judgment = getattr(fire, "judgment", None)
    timing = getattr(judgment, "timing", None)
    if timing is None or timing.source_media_ts is None:
        return None
    source_media_ts = float(timing.source_media_ts)
    observed_to_fire_s = timing.ingest_to_decision_s()
    matched = _matched_event_pairs([source_media_ts], run.windows)
    action_onset = run.windows[matched[0][1]][0] if matched else None
    return DecisionObservation(
        case_id=run.case_id,
        video_id=run.video_id,
        source_media_ts=source_media_ts,
        observed_to_fire_s=observed_to_fire_s,
        judge_latency_s=timing.judge_duration_s(),
        action_onset_to_fire_s=(
            (source_media_ts - action_onset) + observed_to_fire_s
            if action_onset is not None
            else None
        ),
        timing=timing,
        revision_id=getattr(fire, "revision_id", None),
    )


async def _score_one(event: dict, request_budget: BudgetEnforcer) -> EventRun:
    goal_text = _action_goal(event["label"])
    result = EventRun(
        case_id=event["row_id"],
        video_id=event["video_id"],
        label=event["label"],
        goal=goal_text,
        onset_s=float(event["onset_s"]),
        end_s=float(event["end_s"]),
        positive_windows=[
            (float(onset), float(end))
            for onset, end in event.get(
                "positive_windows",
                [[event["onset_s"], event["end_s"]]],
            )
        ],
    )
    settings = load_settings()
    vision = RecordingVision(bench_vision(settings, request_budget=request_budget))
    # This benchmark arms its goal in code and only exercises the supplied live vision adapter.
    # Explicit scripted values isolate unused language/specialist seams from ambient deployment env.
    core = Percept.create(
        vision=vision,
        specialist="scripted",
        conductor="scripted-conductor",
        stt="fake",
        tts="fake",
        settings=settings,
        discover_plugins=False,
    )
    goal_id = f"pt-{event['video_id']}"
    goal = Goal(
        id=goal_id,
        condition=goal_text,
        say=f"Perception Test action detected: {goal_text}",
        state=_action_goal_state(),
    )
    core.add_goal(goal)
    plan = sensor_plan_for(goal)
    cadence = plan.cadence
    policy = cadence_policy(cadence)
    ctrl = SamplingController(policy)
    spec = compile_watchspec(list(core.goals.active()), threshold=settings.edge.motion_gate)
    gate = MotionGate(spec, settings=settings)
    program = program_of(goal)
    result.configuration = {
        "goal_state": dict(goal.state),
        "resolved_perception": {
            "modality": program.perception.modality,
            "temporal": program.perception.temporal,
            "n_frames": program.perception.n_frames,
            "sampling": program.perception.sampling,
            "window_s": program.perception.window_s,
        },
        "resolved_cadence": dict(cadence),
        "capture_fps": capture_fps(cadence),
    }
    dropped_frames = [0]

    try:
        async for t, gate_frame, judge_frame in open_source(
            str(event["path"]),
            fps=capture_fps(cadence),
            maxsize=_FIXTURE_QUEUE_SIZE,
            queue_policy="block",
            dropped=dropped_frames,
        ):
            result.frames += 1
            decision = gate.evaluate(gate_frame, ts=float(t))
            # JUDGE on EITHER the motion gate's escalation (legacy firing logic — UNCHANGED) OR the fovea's
            # sensitive cadence coasting over a quiet-but-in-window frame (the coverage fix). The motion
            # signal drives the controller's saccade, so a real motion burst still densifies to max_fps.
            coast = ctrl.should_process("judge", float(t), float(decision.signal.motion))
            if not (decision.escalate or coast):
                continue
            cause = admission_cause(gate=bool(decision.escalate), cadence=coast)
            edge_trace = {
                "decision": "escalate" if decision.escalate else "suppress",
                "motion": round(float(decision.signal.motion), 4),
                "structural": round(float(decision.signal.structural), 4),
                "semantic": round(float(decision.signal.semantic), 4),
                "threshold": round(float(decision.threshold), 4),
            }
            result.escalations.append(float(t))
            vision.last_verdict = None
            events = await core.perceive_judged(judge_frame, "pt-action", t=float(t))
            if vision.last_error is not None:
                result.error = f"{type(vision.last_error).__name__}: {vision.last_error}"
                result.trace.append(
                    {
                        "ts": round(float(t), 3),
                        "case_id": result.case_id,
                        "goal_id": goal_id,
                        "edge": edge_trace,
                        "admission_cause": cause,
                        "core_gate_action": "invalid",
                        "reason": f"backend failure: {result.error}",
                    }
                )
                break
            action = "silent"
            for fire in events or []:
                action = getattr(fire, "action", "silent")
                if action == FIRE:
                    observation = _decision_observation(fire, result)
                    if observation is None:
                        result.error = (
                            "FIRE omitted JudgmentTiming/source_media_ts; "
                            "action-to-fire latency cannot be qualified"
                        )
                    else:
                        result.fires.append(observation)
                else:
                    result.asks.append(float(t))
            verdict = vision.last_verdict
            result.trace.append(
                {
                    "ts": round(float(t), 3),
                    "edge": edge_trace,
                    "admission_cause": cause,
                    "goal_id": goal_id,
                    "case_id": result.case_id,
                    "goal": goal_text,
                    "judge_verdict": {
                        "satisfied": bool(verdict.get("satisfied")) if verdict else None,
                        "confidence": round(float(verdict.get("confidence") or 0.0), 3)
                        if verdict
                        else None,
                        "evidence": str(verdict.get("evidence", "") if verdict else "")[:240],
                        "phase": verdict.get("phase") if verdict else None,
                        "action_present": verdict.get("action_present") if verdict else None,
                    },
                    "core_gate_action": action,
                    "reason": f"{action}: emitted"
                    if action != "silent"
                    else _silent_reason(verdict, settings.gate.threshold),
                }
            )
            if result.error:
                break
    except Exception as exc:
        result.error = f"{type(exc).__name__}: {exc}"

    result.dropped_frames = dropped_frames[0]
    if result.dropped_frames and result.error is None:
        result.error = (
            f"fixture replay dropped {result.dropped_frames} decoded frame(s); "
            "benchmark coverage is invalid"
        )
    result.reconcile_fire_windows()
    result.judge_calls = vision.calls
    return result


def _mean(xs: list[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def _median(values: list[float]) -> float | None:
    return float(median(values)) if values else None


def _precision(samples: list[tuple[str, float]], runs: list[EventRun]) -> float | None:
    """Score timestamps against the union of positive windows for the same case."""
    if not samples:
        return None
    by_case = {run.case_id: run for run in runs}
    hits = 0
    for case_id, timestamp in samples:
        run = by_case.get(case_id)
        if run is not None and run.contains(timestamp):
            hits += 1
    return hits / len(samples)


def _event_precision(runs: list[EventRun]) -> float | None:
    """One-to-one FIRE precision; duplicate fires in one true window are false positives."""
    predictions = sum(len(run.fires) for run in runs)
    if not predictions:
        return None
    return sum(run.fire_hits for run in runs) / predictions


def _fire_records(run: EventRun) -> list[dict[str, Any]]:
    """Serialize FIRE details from the reconciled one-to-one event assignment."""
    records: list[dict[str, Any]] = []
    for fire in run.fires:
        source_latency = (
            fire.action_onset_to_fire_s - fire.observed_to_fire_s
            if fire.action_onset_to_fire_s is not None
            else None
        )
        records.append(
            {
                "source_media_ts": round(fire.source_media_ts, 3),
                "source_detection_latency_s": (
                    round(source_latency, 6) if source_latency is not None else None
                ),
                "observed_to_fire_s": round(fire.observed_to_fire_s, 6),
                "action_onset_to_fire_s": (
                    round(fire.action_onset_to_fire_s, 6)
                    if fire.action_onset_to_fire_s is not None
                    else None
                ),
                "judge_latency_s": round(fire.judge_latency_s, 6),
                "decision_emitted_ts": round(
                    fire.timing.decision_emitted_ts,
                    6,
                ),
                "revision_id": fire.revision_id,
            }
        )
    return records


def _call_reductions(
    *,
    dense_calls: int,
    logical_calls: int,
    physical_attempts: int,
) -> tuple[float | None, float | None]:
    """Return cost-facing physical and diagnostic logical reductions."""
    if not dense_calls:
        return None, None
    return (
        1.0 - (physical_attempts / dense_calls),
        1.0 - (logical_calls / dense_calls),
    )


def _unavailable(reason: str) -> TrackResult:
    return TrackResult(
        track=TRACK,
        cells={
            ("e2e", "notification_recall"): None,
            ("edge", "candidate_recall"): None,
            ("edge", "candidate_precision_proxy"): None,
            ("e2e", "notification_precision"): None,
            ("e2e", "median_source_detection_latency_s"): None,
            ("e2e", "median_observed_to_fire_s"): None,
            ("e2e", "median_action_onset_to_fire_s"): None,
            ("e2e", "median_judge_latency_s"): None,
            ("e2e", "call_reduction"): None,
            ("e2e", "positive_events"): None,
            ("e2e", "matched_fire_events"): None,
            ("e2e", "total_fire_predictions"): None,
        },
        headline={"Perception Test action E2E": "n/a"},
        excluded=[f"{TRACK}: {reason}"],
        notes=reason,
        status="skipped",
    )


def _failed_run(
    *,
    events: list[dict],
    runs: list[EventRun],
    request_budget: BudgetEnforcer,
    selected_media_minutes: float,
) -> TrackResult:
    """Return ledger-only evidence for a terminal benchmark failure; no partial score."""
    budget = request_budget.snapshot()
    physical_attempts = int(budget["physical_attempts"])
    failures = [run for run in runs if run.error]
    reason = failures[0].error if failures else "unknown terminal benchmark failure"
    cells = {
        ("e2e", "notification_recall"): None,
        ("edge", "candidate_recall"): None,
        ("edge", "candidate_precision_proxy"): None,
        ("e2e", "notification_precision"): None,
        ("e2e", "median_source_detection_latency_s"): None,
        ("e2e", "median_observed_to_fire_s"): None,
        ("e2e", "median_action_onset_to_fire_s"): None,
        ("e2e", "median_judge_latency_s"): None,
        ("e2e", "call_reduction"): None,
        ("e2e", "logical_admission_reduction"): None,
        ("e2e", "positive_events"): None,
        ("e2e", "matched_fire_events"): None,
        ("e2e", "total_fire_predictions"): None,
        ("dense-modeled", "dense_frame_calls"): None,
        ("e2e", "logical_judge_calls"): sum(run.judge_calls for run in runs),
        ("e2e", "physical_provider_attempts"): physical_attempts,
        ("e2e", "input_tokens"): budget["input_tokens"],
        ("e2e", "output_tokens"): budget["output_tokens"],
        ("e2e", "billed_usd"): budget["billed_usd"],
        ("e2e", "committed_budget_usd"): budget["committed_usd"],
        ("e2e", "selected_media_minutes"): selected_media_minutes,
        ("e2e", "source_dropped_frames"): sum(run.dropped_frames for run in runs),
    }
    measured = [f"{lane}:{metric}" for lane, metric in cells if cells[(lane, metric)] is not None]
    return TrackResult(
        track=TRACK,
        cells=cells,
        fmts={
            "logical_judge_calls": "{:.0f}",
            "physical_provider_attempts": "{:.0f}",
            "input_tokens": "{:.0f}",
            "output_tokens": "{:.0f}",
            "billed_usd": "${:.6f}",
            "committed_budget_usd": "${:.6f}",
            "selected_media_minutes": "{:.6f}",
            "source_dropped_frames": "{:.0f}",
        },
        headline={"Perception Test action E2E": "INVALID — terminal benchmark failure"},
        measured=measured,
        excluded=[f"{TRACK}: terminal benchmark failure: {reason}"],
        calls=physical_attempts,
        qualifying_events=0,
        coverage=[
            {
                "dataset": "perception-test/action",
                "expected": len(events),
                "observed": sum(1 for run in runs if not run.error),
                "failed": len(failures),
                "skipped": max(0, len(events) - len(runs)),
                "case_ids": [event["row_id"] for event in events],
                "notes": "terminal benchmark failure stopped the current case and all later cases",
            }
        ],
        notes=f"budget={budget}; attempted_cases={[run.case_id for run in runs]}",
        status="failed",
    )


async def _score_all(
    events: list[dict],
    request_budget: BudgetEnforcer,
) -> list[EventRun]:
    runs: list[EventRun] = []
    for event in events:
        run = await _score_one(event, request_budget)
        runs.append(run)
        if run.error:
            break
    return runs


def score(*, live: bool, n: int) -> TrackResult:
    if not live:
        return _unavailable("live backend required; offline fake cannot assert visual acuity")

    manifest = Path(os.environ.get("PERCEPT_BENCH_MANIFEST", str(pt.GOLDEN_MANIFEST)))
    try:
        events = pt.select_golden_action_events(
            n,
            manifest=manifest,
            require_media=True,
        )
    except pt.GoldenSetCapacityError as exc:
        result = _unavailable(str(exc))
        result.status = "blocked"
        return result

    backend = os.environ.get("PERCEPT_VISION_BACKEND", "gemini")
    model = os.environ.get("PERCEPT_VISION_MODEL")
    settings = load_settings()
    request_budget = build_budget_enforcer(
        os.environ.get("PERCEPT_BENCH_BUDGET_PLAN", ""),
        provider=backend,
        model=model,
    )
    try:
        selected_media_minutes = pt.action_media_minutes(events)
    except (ImportError, OSError, ValueError) as exc:
        result = _unavailable(f"selected media duration could not be verified: {exc}")
        result.status = "blocked"
        return result
    if selected_media_minutes > request_budget.media_minutes_cap + 1e-12:
        result = _unavailable(
            f"selected pinned media is {selected_media_minutes:.6f} minutes, "
            f"exceeding the effective media cap of "
            f"{request_budget.media_minutes_cap:.6f} minutes"
        )
        result.status = "blocked"
        return result

    runs = asyncio.run(_score_all(events, request_budget)) if events else []
    failed = [r for r in runs if r.error]
    if failed:
        return _failed_run(
            events=events,
            runs=runs,
            request_budget=request_budget,
            selected_media_minutes=selected_media_minutes,
        )
    observed = len(runs) - len(failed)

    positive_events = sum(len(run.windows) for run in runs)
    matched_fire_events = sum(run.fire_hits for run in runs)
    total_fire_predictions = sum(len(run.fires) for run in runs)
    edge_recall = sum(run.edge_hits for run in runs) / positive_events if positive_events else None
    notification_recall = matched_fire_events / positive_events if positive_events else None
    all_candidates = [(r.case_id, t) for r in runs for t in r.escalations]
    candidate_precision = _precision(all_candidates, runs)
    notification_precision = _event_precision(runs)
    source_latencies = [
        fire.source_media_ts - window[0] for run in runs for fire, window in run.matched_fire_events
    ]
    observed_latencies = [
        fire.observed_to_fire_s for run in runs for fire, _window in run.matched_fire_events
    ]
    action_latencies = [
        (fire.source_media_ts - window[0]) + fire.observed_to_fire_s
        for run in runs
        for fire, window in run.matched_fire_events
    ]
    judge_latencies = [
        fire.judge_latency_s for run in runs for fire, _window in run.matched_fire_events
    ]

    dense_calls = sum(r.frames for r in runs)
    logical_calls = sum(r.judge_calls for r in runs)
    budget = request_budget.snapshot()
    physical_attempts = int(budget["physical_attempts"])
    call_reduction, logical_admission_reduction = _call_reductions(
        dense_calls=dense_calls,
        logical_calls=logical_calls,
        physical_attempts=physical_attempts,
    )

    cells = {
        ("e2e", "notification_recall"): notification_recall,
        ("edge", "candidate_recall"): edge_recall,
        ("edge", "candidate_precision_proxy"): candidate_precision,
        ("e2e", "notification_precision"): notification_precision,
        ("e2e", "median_source_detection_latency_s"): _median(source_latencies),
        ("e2e", "median_observed_to_fire_s"): _median(observed_latencies),
        ("e2e", "median_action_onset_to_fire_s"): _median(action_latencies),
        ("e2e", "median_judge_latency_s"): _median(judge_latencies),
        ("e2e", "call_reduction"): call_reduction,
        ("e2e", "logical_admission_reduction"): logical_admission_reduction,
        ("e2e", "positive_events"): positive_events,
        ("e2e", "matched_fire_events"): matched_fire_events,
        ("e2e", "total_fire_predictions"): total_fire_predictions,
        ("dense-modeled", "dense_frame_calls"): dense_calls,
        ("e2e", "logical_judge_calls"): logical_calls,
        ("e2e", "physical_provider_attempts"): physical_attempts,
        ("e2e", "input_tokens"): budget["input_tokens"],
        ("e2e", "output_tokens"): budget["output_tokens"],
        ("e2e", "billed_usd"): budget["billed_usd"],
        ("e2e", "committed_budget_usd"): budget["committed_usd"],
        ("e2e", "selected_media_minutes"): selected_media_minutes,
        ("e2e", "source_dropped_frames"): sum(run.dropped_frames for run in runs),
    }
    per_event = [
        {
            "case_id": r.case_id,
            "video_id": r.video_id,
            "label": r.label,
            "goal": r.goal,
            "positive_windows": [
                [round(onset, 3), round(end + WINDOW_PAD_S, 3)] for onset, end in r.windows
            ],
            "positive_event_count": len(r.windows),
            "edge_hits": r.edge_hits,
            "fire_hits": r.fire_hits,
            "frames": r.frames,
            "source_dropped_frames": r.dropped_frames,
            "judge_calls": r.judge_calls,
            "edge_hit": r.edge_hit,
            "fire_hit": r.fire_hit,
            "fires": _fire_records(r),
            "escalations": [round(t, 3) for t in r.escalations[:20]],
            "trace": r.trace,
            "configuration": r.configuration,
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
            "median_source_detection_latency_s": "{:.3f}",
            "median_observed_to_fire_s": "{:.3f}",
            "median_action_onset_to_fire_s": "{:.3f}",
            "median_judge_latency_s": "{:.3f}",
            "call_reduction": "{:.2f}",
            "logical_admission_reduction": "{:.2f}",
            "positive_events": "{:.0f}",
            "matched_fire_events": "{:.0f}",
            "total_fire_predictions": "{:.0f}",
            "dense_frame_calls": "{:.0f}",
            "logical_judge_calls": "{:.0f}",
            "physical_provider_attempts": "{:.0f}",
            "input_tokens": "{:.0f}",
            "output_tokens": "{:.0f}",
            "billed_usd": "${:.6f}",
            "committed_budget_usd": "${:.6f}",
            "selected_media_minutes": "{:.6f}",
            "source_dropped_frames": "{:.0f}",
        },
        headline={
            "PT action E2E recall / edge recall": (
                f"{notification_recall:.2f} / {edge_recall:.2f} (N={len(runs)})"
                if notification_recall is not None and edge_recall is not None
                else "n/a"
            ),
            "PT action calls dense-modeled -> logical -> physical": (
                f"{dense_calls} -> {logical_calls} -> {physical_attempts}"
            ),
        },
        measured=[
            "e2e:notification_recall",
            "edge:candidate_recall",
            "edge:candidate_precision_proxy",
            "e2e:notification_precision",
            "e2e:median_source_detection_latency_s",
            "e2e:median_observed_to_fire_s",
            "e2e:median_action_onset_to_fire_s",
            "e2e:median_judge_latency_s",
            "e2e:call_reduction",
            "e2e:logical_admission_reduction",
            "e2e:positive_events",
            "e2e:matched_fire_events",
            "e2e:total_fire_predictions",
            "e2e:logical_judge_calls",
            "e2e:physical_provider_attempts",
            "e2e:input_tokens",
            "e2e:output_tokens",
            "e2e:billed_usd",
            "e2e:committed_budget_usd",
            "e2e:selected_media_minutes",
            "e2e:source_dropped_frames",
        ],
        modeled=["dense-modeled:dense_frame_calls"],
        calls=physical_attempts,
        qualifying_events=matched_fire_events,
        coverage=[
            {
                "dataset": "perception-test/action",
                "expected": len(events),
                "observed": observed,
                "failed": len(failed),
                "skipped": 0,
                "case_ids": [event["row_id"] for event in events],
                "positive_events": positive_events,
                "notes": (
                    "runtime E2E path over exact pinned PT action manifest rows; "
                    "recall denominator is every same-label positive window"
                ),
            }
        ],
        notes=(
            f"manifest={manifest}; pinned_cases={[event['row_id'] for event in events]}; "
            f"capture_floor_fps={FPS}; cadence=sensorplan-sensitive(fixed_product_prior); "
            f"budget={budget}; per_event={per_event}"
        ),
        configuration={
            "provider": backend,
            "model": model,
            "watch": _action_goal_state(),
            "evidence_contract": {
                "ordered_frames": True,
                "event_chain": True,
                "required_phase_vocabulary": ["pre", "during", "post"],
                "required_action_present": True,
                "occurrence_rearm": "each_event",
            },
            "admission": {
                "rule": "motion_gate_or_sensitive_cadence",
                "capture_floor_fps": FPS,
                "capture_headroom_fps": _CAPTURE_HEADROOM_FPS,
                "fixture_queue_size": _FIXTURE_QUEUE_SIZE,
                "fixture_queue_policy": "block",
                "motion_gate": settings.edge.motion_gate,
                "salience_stat": settings.edge.salience_stat,
                "cadence_formula": {
                    "base_fps": f"2/{ACTION_BRIEF_WINDOW_S}",
                    "max_fps": "3*base_fps",
                    "escalate_threshold": 0.05,
                },
            },
            "decision": {
                "confidence_threshold": settings.gate.threshold,
                "freshness_min": Gate().fresh_min,
                "require_false_first": settings.gate.require_false_first,
                "refractory_s": settings.gate.refractory_s,
            },
            "cache": {
                "enabled": settings.cache.enabled,
                "threshold": settings.cache.threshold,
                "ttl_s": settings.cache.ttl_s,
            },
            "scoring": {
                "positive_unit": "same-label occurrence window",
                "matching": "greedy_one_to_one_earliest_end",
                "late_fire_pad_s": WINDOW_PAD_S,
            },
            "budget_plan": request_budget.plan,
        },
    )
