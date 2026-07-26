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
import math
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median
from typing import Any

from percept import InterestSubscription, Percept
from percept.adapters import Goal as ContractGoal
from percept.backends.judge_pool import (
    JudgeCapability,
    JudgePool,
    JudgeRoute,
)
from percept.behavior import program_of
from percept.config import load_settings
from percept.contracts import JudgmentTiming
from percept.contracts.watchplan import ResolvedWatch, WatchPlan
from percept.gate import Gate
from percept.incidents import incident_judgment
from percept.resolver import ConfigResolver
from percept.runtime import NativeVideoVision, RuntimeStats
from percept.runtime.source import open_source
from percept.runtime.worker import run as worker_run
from percept.sampling import SamplingController, SensePolicy
from percept.sensorplan import sensor_plan_for
from percept.source_profiles_default import PHYSICAL_SCENE
from percept.watchplan_apply import apply_watch_plan

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
# One hard action budget, split so provider/evidence work cannot consume the
# full user-visible reflex envelope. The measured launch gate remains strict
# action-onset -> Incident under 2 seconds.
ACTION_PROVIDER_DEADLINE_S = 1.5
ACTION_DECISION_DEADLINE_S = 1.9


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
    incident_observed_ts: float
    ingest_to_decision_s: float
    judge_latency_s: float
    admitted_to_incident_s: float
    decision_to_incident_s: float
    timing: JudgmentTiming
    source_detection_latency_s: float | None = None
    action_onset_to_incident_s: float | None = None
    evidence_available_to_incident_s: float | None = None
    revision_id: str | None = None


@dataclass(frozen=True)
class ReplayClockAnchor:
    """One explicit mapping between fixture media time and replay monotonic time."""

    media_ts: float
    monotonic_ts: float

    def monotonic_for(self, media_ts: float) -> float:
        return self.monotonic_ts + (float(media_ts) - self.media_ts)


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
    health: list[dict[str, Any]] = field(default_factory=list)
    interest_events: list[dict[str, Any]] = field(default_factory=list)
    judge_receipts: list[dict[str, Any]] = field(default_factory=list)
    configuration: dict[str, Any] = field(default_factory=dict)
    replay_anchor: ReplayClockAnchor | None = None
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
            fire.source_detection_latency_s = None
            fire.action_onset_to_incident_s = None
            fire.evidence_available_to_incident_s = None
        for fire, window in self.matched_fire_events:
            fire.source_detection_latency_s = fire.source_media_ts - window[0]
            if self.replay_anchor is not None:
                fire.action_onset_to_incident_s = (
                    fire.incident_observed_ts
                    - self.replay_anchor.monotonic_for(window[0])
                )
                fire.evidence_available_to_incident_s = (
                    fire.incident_observed_ts
                    - self.replay_anchor.monotonic_for(window[1])
                )


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
        # PT actions are localized hand/object motion. Pin the measured Tier-0
        # operating point in the benchmark itself; never inherit an ambient
        # deployment-wide mean-stat default.
        "salience_stat": "block_max",
        "sense": {"brief_window_s": ACTION_BRIEF_WINDOW_S},
        "perception": {
            "temporal": "strip",
            "n_frames": ACTION_TEMPORAL_FRAMES,
            "sampling": ACTION_TEMPORAL_SAMPLING,
            "window_s": ACTION_TEMPORAL_WINDOW_S,
        },
    }


def _action_watch_plan(*, goal_id: str, condition: str) -> WatchPlan:
    """The public, annotation-independent WatchPlan executed for every PT case."""
    state = _action_goal_state()
    perception = state["perception"]
    return WatchPlan(
        id=goal_id,
        condition=condition,
        sense="video",
        rearm=state["rearm"],
        explicit={
            "Goal.state.sense.brief_window_s": state["sense"]["brief_window_s"],
            "Goal.state.perception.temporal": perception["temporal"],
            "Goal.state.perception.n_frames": perception["n_frames"],
            "Goal.state.perception.sampling": perception["sampling"],
            "Goal.state.perception.window_s": perception["window_s"],
            "WatchProfile.salience_stat": state["salience_stat"],
        },
    )


class _BenchmarkResolver(ConfigResolver):
    """Capture the exact ResolvedWatch that ``apply_watch_plan`` compiled and armed."""

    def __init__(self) -> None:
        super().__init__()
        self.applied: ResolvedWatch | None = None

    def resolve(self, *args: Any, **kwargs: Any) -> ResolvedWatch:
        resolved = super().resolve(*args, **kwargs)
        self.applied = resolved
        return resolved


_BENCHMARK_RESOLVED_PATHS = (
    "Goal.state.rearm",
    "Goal.state.sense.brief_window_s",
    "Goal.state.perception.temporal",
    "Goal.state.perception.n_frames",
    "Goal.state.perception.sampling",
    "Goal.state.perception.window_s",
    "Settings.edge.salience_stat",
    "Settings.edge.motion_gate_block_max",
    "Settings.gate.threshold",
    "Settings.gate.require_false_first",
    "Settings.gate.refractory_s",
    "SourceProfile.default_timebase",
)


def _json_value(value: Any) -> Any:
    """Normalize immutable configuration snapshots for the JSON benchmark artifact."""
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted(_json_value(item) for item in value)
    return value


def _resolved_watch_record(resolved: ResolvedWatch) -> dict[str, Any]:
    paths = [
        path for path in _BENCHMARK_RESOLVED_PATHS if path in resolved.effective
    ]
    return {
        "plan_hash": resolved.plan_hash,
        "component_snapshot_hash": resolved.component_snapshot_hash,
        "optimization_lock_hash": resolved.optimization_lock_hash,
        "source_profile": (
            _json_value(dict(resolved.source_profile))
            if resolved.source_profile is not None
            else None
        ),
        "domain_pack": (
            _json_value(dict(resolved.domain_pack))
            if resolved.domain_pack is not None
            else None
        ),
        "execution_policy": (
            _json_value(dict(resolved.execution_policy))
            if resolved.execution_policy is not None
            else None
        ),
        "components": _json_value(dict(resolved.components)),
        "effective": {
            path: _json_value(resolved.effective[path]) for path in paths
        },
        "provenance": {
            path: _json_value(resolved.provenance[path].to_dict())
            for path in paths
        },
    }


class RecordingVision:
    """Judge wrapper that records calls, verdicts, and swallowed backend failures."""

    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self.SUPPORTS_NATIVE_VIDEO_TRANSPORT = bool(
            getattr(inner, "SUPPORTS_NATIVE_VIDEO_TRANSPORT", False)
        )
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
    *,
    incident_observed_ts: float | None = None,
) -> DecisionObservation | None:
    """Materialize one timing-bearing decision without mixing clock domains."""
    judgment = getattr(fire, "judgment", None) or incident_judgment(fire)
    timing = getattr(judgment, "timing", None)
    if timing is None or timing.source_media_ts is None:
        return None
    source_media_ts = float(timing.source_media_ts)
    incident_ts = (
        time.monotonic()
        if incident_observed_ts is None
        else float(incident_observed_ts)
    )
    if incident_ts < timing.decision_emitted_ts:
        raise ValueError(
            "Incident observation precedes decision emission on the monotonic clock"
        )
    return DecisionObservation(
        case_id=run.case_id,
        video_id=run.video_id,
        source_media_ts=source_media_ts,
        incident_observed_ts=incident_ts,
        ingest_to_decision_s=timing.ingest_to_decision_s(),
        judge_latency_s=timing.judge_duration_s(),
        admitted_to_incident_s=incident_ts - timing.admitted_ts,
        decision_to_incident_s=incident_ts - timing.decision_emitted_ts,
        timing=timing,
        revision_id=getattr(fire, "revision_id", None),
    )


async def _paced_fixture_source(
    path: str,
    *,
    fps: float,
    result: EventRun,
    dropped: list[int],
):
    """Replay a finite file at its media cadence while preserving envelopes."""
    started_wall: float | None = None
    first_media: float | None = None
    async for envelope in open_source(
        path,
        fps=fps,
        maxsize=_FIXTURE_QUEUE_SIZE,
        queue_policy="block",
        timebase="media",
        dropped=dropped,
    ):
        if envelope.source_time is None:
            raise ValueError(
                "paced PT qualification requires decoder source timestamps"
            )
        media_t = float(envelope.source_time)
        if first_media is None:
            first_media = media_t
            started_wall = time.monotonic()
            result.replay_anchor = ReplayClockAnchor(
                media_ts=media_t,
                monotonic_ts=started_wall,
            )
        assert started_wall is not None
        target = started_wall + (media_t - first_media)
        remaining = target - time.monotonic()
        if remaining > 0:
            await asyncio.sleep(remaining)
        result.frames += 1
        yield envelope


def _receipt_record(receipt: Any) -> dict[str, Any]:
    return {
        "route": receipt.route,
        "evidence_shape": receipt.evidence_shape,
        "mode": receipt.mode,
        "outcome": receipt.outcome,
        "latency_s": round(float(receipt.latency_s), 6),
        "deadline_s": float(receipt.deadline_s),
        "confidence": receipt.confidence,
        "exception_type": receipt.exception_type,
    }


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
    goal_id = f"pt-{event['video_id']}"
    provider = RecordingVision(
        bench_vision(settings, request_budget=request_budget)
    )

    def on_receipt(receipt: Any) -> None:
        record = _receipt_record(receipt)
        result.judge_receipts.append(record)
        if receipt.outcome in {
            "provider_error",
            "malformed",
            "deadline",
            "capacity",
            "no_route",
        }:
            result.error = (
                f"reflex judge route failed closed: outcome={receipt.outcome}; "
                f"route={receipt.route}; exception={receipt.exception_type}"
            )

    vision = JudgePool(
        [
            JudgeRoute(
                "native-video-primary",
                NativeVideoVision(
                    provider,
                    # The adapter recomputes clip fps from the retained source span.
                    # This bounded capture ceiling is only the no-span fallback.
                    fps=_CAPTURE_HEADROOM_FPS,
                    max_frames=ACTION_TEMPORAL_FRAMES,
                    require_native_transport=True,
                ),
                JudgeCapability(
                    evidence_shapes=frozenset({"video_clip"}),
                    latency_slo_s=ACTION_PROVIDER_DEADLINE_S,
                    modes=frozenset({"reflex"}),
                    max_inflight=1,
                ),
            )
        ],
        mode="reflex",
        deadline_s=ACTION_PROVIDER_DEADLINE_S,
        max_parallel=1,
        on_receipt=on_receipt,
        terminal_outcomes={
            "provider_error",
            "malformed",
            "deadline",
            "capacity",
            "no_route",
        },
    )
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
    plan = _action_watch_plan(goal_id=goal_id, condition=goal_text)
    resolver = _BenchmarkResolver()
    receipt = apply_watch_plan(
        core,
        plan,
        resolver=resolver,
        source_profile=PHYSICAL_SCENE,
    )
    if receipt.status != "applied" or receipt.watch_ids != (goal_id,):
        result.error = (
            "WatchPlan failed to arm exactly through the compiled product path: "
            f"status={receipt.status}; watch_ids={receipt.watch_ids!r}; "
            f"dropped={[item.code for item in receipt.dropped_fields]!r}; "
            f"warnings={[item.code for item in receipt.warnings]!r}"
        )
        result.configuration = {
            "watch_plan": plan.to_dict(),
            "watch_receipt": receipt.to_dict(),
        }
        return result
    if resolver.applied is None:
        result.error = "WatchPlan applied without an inspectable ResolvedWatch"
        return result
    goal = core.goals.get(goal_id)
    runtime_binding = core._watch_runtime_for(goal_id)
    if goal is None or runtime_binding is None:
        result.error = "WatchPlan receipt claimed applied but compiled runtime binding is absent"
        return result
    compiled = runtime_binding[0]
    cadence = sensor_plan_for(goal).cadence
    program = program_of(goal)
    result.configuration = {
        "architecture": (
            "WatchPlan -> ConfigResolver -> ResolvedWatch -> "
            "RuntimeCompiler -> async worker"
        ),
        "watch_plan": plan.to_dict(),
        "watch_receipt": receipt.to_dict(),
        "resolved_watch": _resolved_watch_record(resolver.applied),
        "runtime_application_key": core._watch_runtime_keys.get(goal_id),
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
        "tier0": {
            "salience_stat": compiled.settings.edge.salience_stat,
            "threshold": compiled.settings.edge.motion_gate_block_max,
            "threshold_setting": "Settings.edge.motion_gate_block_max",
        },
        "evidence_shape": "video_clip",
        "provider_deadline_s": ACTION_PROVIDER_DEADLINE_S,
        "decision_deadline_s": ACTION_DECISION_DEADLINE_S,
    }
    dropped_frames = [0]
    stats = RuntimeStats()
    interest_subscription = InterestSubscription(max_batches=64)

    def on_trace(trace_event: Any) -> None:
        row = {
            "trace_id": trace_event.trace_id,
            "stage": trace_event.stage,
            "ts": trace_event.t,
            "goal_id": trace_event.goal_id,
            "data": dict(trace_event.data),
        }
        result.trace.append(row)
        if (
            trace_event.stage == "edge.decision"
            and trace_event.data.get("goal_ids")
            and trace_event.t is not None
        ):
            result.escalations.append(float(trace_event.t))

    def on_health(event: Any) -> None:
        result.health.append(
            {
                "kind": event.kind,
                "ts": event.ts,
                "data": dict(event.data or {}),
            }
        )

    try:
        paced_source = _paced_fixture_source(
            str(event["path"]),
            fps=capture_fps(cadence),
            result=result,
            dropped=dropped_frames,
        )
        runtime = worker_run(
            core,
            source=paced_source,
            fps=capture_fps(cadence),
            stats=stats,
            gate_source="camera",
            judge_dispatch="async",
            judge_stall_after_s=ACTION_DECISION_DEADLINE_S,
            judge_decision_deadline_s=ACTION_DECISION_DEADLINE_S,
            timebase="media",
            on_trace=on_trace,
            on_health=on_health,
            interest_subscription=interest_subscription,
            interest_session_id=f"pt:{result.case_id}",
        )

        async def collect_interest() -> None:
            async for event in interest_subscription:
                result.interest_events.append(event.as_dict())

        interest_task = asyncio.create_task(collect_interest())
        try:
            async for incident in runtime:
                observation = _decision_observation(
                    incident,
                    result,
                    incident_observed_ts=time.monotonic(),
                )
                if observation is None:
                    result.error = (
                        "Incident omitted runtime-local JudgmentTiming/source_media_ts; "
                        "action-to-fire latency cannot be qualified"
                    )
                    break
                result.fires.append(observation)
        finally:
            await runtime.aclose()
            await interest_task
    except Exception as exc:
        result.error = f"{type(exc).__name__}: {exc}"

    result.dropped_frames = dropped_frames[0]
    if result.dropped_frames and result.error is None:
        result.error = (
            f"fixture replay dropped {result.dropped_frames} decoded frame(s); "
            "benchmark coverage is invalid"
        )
    result.reconcile_fire_windows()
    if result.fires and result.replay_anchor is None and result.error is None:
        result.error = (
            "fixture replay omitted its media-to-monotonic clock anchor; "
            "onset-to-Incident latency cannot be qualified"
        )
    result.judge_calls = provider.calls
    result.configuration["runtime_stats"] = {
        "frames": stats.frames,
        "escalations": stats.escalations,
        "judge_calls": stats.judge_calls,
        "judge_superseded": stats.judge_superseded,
        "judge_revision_dropped": stats.judge_revision_dropped,
        "judge_deadline_dropped": stats.judge_deadline_dropped,
        "judge_deadline_expired": stats.judge_deadline_expired,
        "goal_admissions": stats.goal_admissions,
        "admission_by_goal": dict(stats.admission_by_goal),
        "admission_by_cause": dict(stats.admission_by_cause),
        "incidents": stats.incidents,
    }
    return result


def _mean(xs: list[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def _median(values: list[float]) -> float | None:
    return float(median(values)) if values else None


def _percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    index = max(
        0,
        min(len(ordered) - 1, math.ceil(len(ordered) * quantile) - 1),
    )
    return ordered[index]


def _timely_occurrence_recall(
    latencies: list[float],
    *,
    positive_events: int,
    threshold_s: float = 2.0,
) -> float | None:
    """Timely true positives divided by every positive occurrence, including misses."""
    if positive_events <= 0:
        return None
    return (
        sum(float(latency) <= threshold_s for latency in latencies)
        / positive_events
    )


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
        records.append(
            {
                "source_media_ts": round(fire.source_media_ts, 3),
                "source_detection_latency_s": (
                    round(fire.source_detection_latency_s, 6)
                    if fire.source_detection_latency_s is not None
                    else None
                ),
                "ingest_to_decision_s": round(fire.ingest_to_decision_s, 6),
                "admitted_to_incident_s": round(
                    fire.admitted_to_incident_s,
                    6,
                ),
                "decision_to_incident_s": round(
                    fire.decision_to_incident_s,
                    6,
                ),
                "action_onset_to_incident_s": (
                    round(fire.action_onset_to_incident_s, 6)
                    if fire.action_onset_to_incident_s is not None
                    else None
                ),
                "evidence_available_to_incident_s": (
                    round(fire.evidence_available_to_incident_s, 6)
                    if fire.evidence_available_to_incident_s is not None
                    else None
                ),
                "judge_latency_s": round(fire.judge_latency_s, 6),
                "incident_observed_ts": round(
                    fire.incident_observed_ts,
                    6,
                ),
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
            ("e2e", "median_ingest_to_decision_s"): None,
            ("e2e", "median_action_onset_to_incident_s"): None,
            ("e2e", "p95_action_onset_to_incident_s"): None,
            ("e2e", "max_action_onset_to_incident_s"): None,
            ("e2e", "median_evidence_available_to_incident_s"): None,
            ("e2e", "timely_occurrence_recall_at_2s"): None,
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
        ("e2e", "median_ingest_to_decision_s"): None,
        ("e2e", "median_action_onset_to_incident_s"): None,
        ("e2e", "p95_action_onset_to_incident_s"): None,
        ("e2e", "max_action_onset_to_incident_s"): None,
        ("e2e", "median_evidence_available_to_incident_s"): None,
        ("e2e", "timely_occurrence_recall_at_2s"): None,
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
    ingest_to_decision_latencies = [
        fire.ingest_to_decision_s
        for run in runs
        for fire, _window in run.matched_fire_events
    ]
    action_to_incident_latencies = [
        fire.action_onset_to_incident_s
        for run in runs
        for fire, _window in run.matched_fire_events
        if fire.action_onset_to_incident_s is not None
    ]
    evidence_to_incident_latencies = [
        fire.evidence_available_to_incident_s
        for run in runs
        for fire, _window in run.matched_fire_events
        if fire.evidence_available_to_incident_s is not None
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
        ("e2e", "median_ingest_to_decision_s"): _median(
            ingest_to_decision_latencies
        ),
        ("e2e", "median_action_onset_to_incident_s"): _median(
            action_to_incident_latencies
        ),
        ("e2e", "p95_action_onset_to_incident_s"): _percentile(
            action_to_incident_latencies,
            0.95,
        ),
        ("e2e", "max_action_onset_to_incident_s"): (
            max(action_to_incident_latencies)
            if action_to_incident_latencies
            else None
        ),
        ("e2e", "median_evidence_available_to_incident_s"): _median(
            evidence_to_incident_latencies
        ),
        ("e2e", "timely_occurrence_recall_at_2s"): _timely_occurrence_recall(
            action_to_incident_latencies,
            positive_events=positive_events,
        ),
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
            "health": r.health,
            "interest_events": r.interest_events,
            "judge_receipts": r.judge_receipts,
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
            "median_ingest_to_decision_s": "{:.3f}",
            "median_action_onset_to_incident_s": "{:.3f}",
            "p95_action_onset_to_incident_s": "{:.3f}",
            "max_action_onset_to_incident_s": "{:.3f}",
            "median_evidence_available_to_incident_s": "{:.3f}",
            "timely_occurrence_recall_at_2s": "{:.2f}",
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
            "e2e:median_ingest_to_decision_s",
            "e2e:median_action_onset_to_incident_s",
            "e2e:p95_action_onset_to_incident_s",
            "e2e:max_action_onset_to_incident_s",
            "e2e:median_evidence_available_to_incident_s",
            "e2e:timely_occurrence_recall_at_2s",
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
                "native_video_clip": True,
                "ordered_images_are_not_equivalent": True,
                "ordered_image_fallback": False,
                "event_chain": True,
                "required_phase_vocabulary": ["pre", "during", "post"],
                "required_action_present": True,
                "occurrence_rearm": "each_event",
                "occurrence_close_quiet_s": ACTION_TEMPORAL_WINDOW_S,
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
                "provider_deadline_s": ACTION_PROVIDER_DEADLINE_S,
                "absolute_decision_deadline_s": ACTION_DECISION_DEADLINE_S,
                "reflex_action_to_fire_s": 2.0,
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
                "latency_clock": (
                    "annotation media time anchored to monotonic replay start; "
                    "measured through local Incident observation"
                ),
                "timely_recall_threshold_s": 2.0,
                "timely_recall_denominator": "all positive occurrences",
                "evidence_available_time": "annotation occurrence end",
            },
            "budget_plan": request_budget.plan,
        },
    )
