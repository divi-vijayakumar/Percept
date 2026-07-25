"""Preregistered L1 matched-Watch execution receipts.

This module is deliberately narrow.  It executes the shipped
``facility_observation@1/package_appeared`` transition Watch through the real resolver/compiler and
runtime worker for an exhaustive sampled baseline and one adaptive operating point.  The bundled
judge is scripted and proves contract mechanics only; no receipt produced here is provider-measured
or launch-qualifying.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from percept import Percept
from percept._safe_json import SafeInputReceipt, SafeJSONInputError, load_json_file
from percept.adapters import Verdict
from percept.config import Settings
from percept.domain_pack import watch_template
from percept.harness.tier0 import gray_frame
from percept.packs_default import FACILITY_OBSERVATION_V1
from percept.resolver import ConfigResolver
from percept.runtime.worker import RuntimeHealthEvent, RuntimeStats, run
from percept.sourceprofile import SourceProfile
from percept.tracing import content_hash
from percept.watchplan_apply import apply_watch_plan

__all__ = [
    "MatchedWatchInputError",
    "ProviderCallReceipt",
    "CandidateReceipt",
    "ArmReceipt",
    "MatchedWatchReceipt",
    "load_matched_watch_manifest",
    "run_matched_watch",
    "run_matched_watch_async",
    "validate_matched_arms",
    "derive_viability_rows",
    "main",
]

_SCHEMA = "percept.eval.matched_watch/v1"
_MEDIA_SCHEMA = "percept.eval.matched_watch.synthetic_video/v1"
_RECEIPT_SCHEMA = "percept.eval.matched_watch.receipt/v1"
_ARCHETYPE = "physical_scene/package_appeared"
_TEMPLATE = "facility_observation@1/package_appeared"
_PACK = "facility_observation@1"
_EVIDENCE_CLASSES = ("scripted_contract", "admission_only")
_SHA_PATHS = {
    "Settings.edge.motion_gate",
    "Settings.edge.motion_gate_block_max",
    "Settings.edge.motion_gate_p95",
    "SourceProfile.parameters.salience_threshold",
}

_MANIFEST_KEYS = {
    "schema",
    "case_id",
    "archetype",
    "split",
    "media",
    "replay",
    "watch",
    "judge",
    "adaptive",
    "events",
    "evidence_classification",
}
_MEDIA_REF_KEYS = {"path", "sha256"}
_REPLAY_KEYS = {"fps", "timebase", "source_id"}
_WATCH_KEYS = {"template_ref", "parameters", "domain_pack"}
_JUDGE_KEYS = {"adapter", "model", "prompt_sha256"}
_ADAPTIVE_KEYS = {"operating_point", "salience_stat", "threshold"}
_EVENT_KEYS = {
    "id",
    "eligible_at_s",
    "window_start_s",
    "window_end_s",
    "review_authority",
    "label_digest",
}
_MEDIA_KEYS = {"schema", "width", "height", "frames"}
_FRAME_KEYS = {"t", "fill", "token", "from", "to"}
_VERDICT_KEYS = {"satisfied", "confidence", "evidence", "error"}


class MatchedWatchInputError(ValueError):
    """A matched replay input or execution receipt is unsafe, incomplete, or incomparable."""


def _fail(message: str) -> None:
    raise MatchedWatchInputError(message)


def _require(condition: bool, message: str) -> None:
    if not condition:
        _fail(message)


def _object(value: Any, context: str) -> dict[str, Any]:
    _require(isinstance(value, Mapping), f"{context} must be an object")
    return dict(value)


def _exact(value: Mapping[str, Any], keys: set[str], context: str) -> None:
    missing = sorted(keys - set(value))
    extra = sorted(set(value) - keys)
    _require(not missing, f"{context} is missing required fields {missing!r}")
    _require(not extra, f"{context} has unsupported fields {extra!r}")


def _text(value: Any, context: str) -> str:
    _require(isinstance(value, str) and bool(value.strip()), f"{context} must be non-empty text")
    return value.strip()


def _number(value: Any, context: str, *, minimum: float | None = None) -> float:
    _require(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value)),
        f"{context} must be a finite number",
    )
    result = float(value)
    if minimum is not None:
        _require(result >= minimum, f"{context} must be >= {minimum}")
    return result


def _sha(value: Any, context: str) -> str:
    result = _text(value, context)
    _require(
        result.startswith("sha256:")
        and len(result) == 71
        and all(c in "0123456789abcdef" for c in result[7:]),
        f"{context} must be sha256:<64 lowercase hex>",
    )
    return result


def _canonical_sha(value: Any) -> str:
    return f"sha256:{content_hash(value)}"


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, set):
        return tuple(sorted(_freeze(item) for item in value))
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


def _safe_receipt_dict(receipt: SafeInputReceipt) -> dict[str, Any]:
    return {
        "path": receipt.path,
        "canonical_path": receipt.canonical_path,
        "byte_count": receipt.byte_count,
        "sha256": receipt.sha256,
        "device": receipt.device,
        "inode": receipt.inode,
        "modified_time_ns": receipt.modified_time_ns,
    }


@dataclass(frozen=True)
class _VerdictSpec:
    satisfied: bool
    confidence: float
    evidence: str
    error: str | None


@dataclass(frozen=True)
class _FrameSpec:
    index: int
    t: float
    fill: int
    token: str
    from_verdict: _VerdictSpec
    to_verdict: _VerdictSpec


@dataclass(frozen=True)
class _EventSpec:
    id: str
    eligible_at_s: float
    start: float
    end: float
    review_authority: str
    label_digest: str


@dataclass(frozen=True)
class _Manifest:
    document: Mapping[str, Any]
    input_receipt: SafeInputReceipt
    media_receipt: SafeInputReceipt
    case_id: str
    split: str
    fps: float
    source_id: str
    watch_parameters: Mapping[str, Any]
    judge_identity: Mapping[str, str]
    operating_point: str
    salience_stat: str
    threshold: float
    events: tuple[_EventSpec, ...]
    frames: tuple[_FrameSpec, ...]
    width: int
    height: int
    evidence_classification: str


@dataclass(frozen=True)
class _JudgePayload:
    index: int
    token: str
    t: float

    def __str__(self) -> str:
        return self.token


@dataclass(frozen=True)
class ProviderCallReceipt:
    sequence: int
    candidate_index: int
    candidate_t: float
    predicate: str
    condition_sha256: str
    status: str
    satisfied: bool | None
    confidence: float | None
    evidence: str | None
    error: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CandidateReceipt:
    index: int
    t: float
    edge_decision: Mapping[str, Any]
    admitted: bool
    runtime_dispatched: bool
    provider_calls: tuple[ProviderCallReceipt, ...]
    decision: str | None
    transition_outcome: str | None
    incident_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "t": self.t,
            "edge_decision": _thaw(self.edge_decision),
            "admitted": self.admitted,
            "runtime_dispatched": self.runtime_dispatched,
            "provider_calls": [item.to_dict() for item in self.provider_calls],
            "decision": self.decision,
            "transition_outcome": self.transition_outcome,
            "incident_ids": list(self.incident_ids),
        }


@dataclass(frozen=True)
class ArmReceipt:
    arm: str
    source_profile: Mapping[str, Any]
    resolved_watch: Mapping[str, Any]
    apply_receipt: Mapping[str, Any]
    compiled_binding: Mapping[str, Any]
    judge_identity: Mapping[str, str]
    configuration_sha256: str
    candidate_timestamps: tuple[float, ...]
    candidates: tuple[CandidateReceipt, ...]
    incidents: tuple[Mapping[str, Any], ...]
    asks: tuple[Mapping[str, Any], ...]
    traces: tuple[Mapping[str, Any], ...]
    health: tuple[Mapping[str, Any], ...]
    runtime_stats: Mapping[str, Any]
    runtime_dispatches: int
    provider_call_attempts: int
    provider_call_successes: int
    provider_call_failures: int
    valid: bool
    blockers: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "arm": self.arm,
            "source_profile": _thaw(self.source_profile),
            "resolved_watch": _thaw(self.resolved_watch),
            "apply_receipt": _thaw(self.apply_receipt),
            "compiled_binding": _thaw(self.compiled_binding),
            "judge_identity": _thaw(self.judge_identity),
            "configuration_sha256": self.configuration_sha256,
            "candidate_timestamps": list(self.candidate_timestamps),
            "candidates": [item.to_dict() for item in self.candidates],
            "incidents": [_thaw(item) for item in self.incidents],
            "asks": [_thaw(item) for item in self.asks],
            "traces": [_thaw(item) for item in self.traces],
            "health": [_thaw(item) for item in self.health],
            "runtime_stats": _thaw(self.runtime_stats),
            "runtime_dispatches": self.runtime_dispatches,
            "provider_call_attempts": self.provider_call_attempts,
            "provider_call_successes": self.provider_call_successes,
            "provider_call_failures": self.provider_call_failures,
            "valid": self.valid,
            "blockers": list(self.blockers),
        }


@dataclass(frozen=True)
class MatchedWatchReceipt:
    manifest: Mapping[str, Any]
    manifest_input_receipt: Mapping[str, Any]
    media_input_receipt: Mapping[str, Any]
    watch_plan: Mapping[str, Any]
    watch_plan_sha256: str
    judge_identity: Mapping[str, str]
    event_windows: tuple[Mapping[str, Any], ...]
    evidence_classification: str
    package_version: str
    baseline: ArmReceipt
    adaptive: ArmReceipt
    comparable: bool
    comparability_blockers: tuple[str, ...]
    run_timestamp: str

    @property
    def launch_qualified(self) -> bool:
        return False

    def content_dict(self) -> dict[str, Any]:
        return {
            "schema": _RECEIPT_SCHEMA,
            "manifest": _thaw(self.manifest),
            "manifest_input_receipt": _thaw(self.manifest_input_receipt),
            "media_input_receipt": _thaw(self.media_input_receipt),
            "watch_plan": _thaw(self.watch_plan),
            "watch_plan_sha256": self.watch_plan_sha256,
            "judge_identity": _thaw(self.judge_identity),
            "event_windows": [_thaw(item) for item in self.event_windows],
            "evidence_classification": self.evidence_classification,
            "package_version": self.package_version,
            "baseline": self.baseline.to_dict(),
            "adaptive": self.adaptive.to_dict(),
            "comparable": self.comparable,
            "comparability_blockers": list(self.comparability_blockers),
            "launch_qualified": False,
            "qualification_reason": (
                "L1 scripted/admission evidence proves execution mechanics only; "
                "provider-measured release qualification is unavailable"
            ),
        }

    def hash_content_dict(self) -> dict[str, Any]:
        result = self.content_dict()
        for key in ("manifest_input_receipt", "media_input_receipt"):
            full = result[key]
            result[key] = {"sha256": full["sha256"], "byte_count": full["byte_count"]}
        for arm_name in ("baseline", "adaptive"):
            arm = result[arm_name]
            arm["incidents"] = [
                {
                    key: value
                    for key, value in incident.items()
                    if key not in {"id", "judgment_event_id"}
                }
                for incident in arm["incidents"]
            ]
            for candidate in arm["candidates"]:
                candidate["incident_count"] = len(candidate.pop("incident_ids"))
            # Raw traces contain real local timing and generated correlation ids. Their deterministic
            # semantic projection is already bound above as candidates/provider calls/incidents/stats.
            arm.pop("traces")
            arm.pop("health")
            arm["runtime_stats"].pop("last_wall_t", None)
        return result

    @property
    def content_sha256(self) -> str:
        return _canonical_sha(self.hash_content_dict())

    def to_dict(self) -> dict[str, Any]:
        result = self.content_dict()
        result["content_sha256"] = self.content_sha256
        result["run_timestamp"] = self.run_timestamp
        return result

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)


def _parse_verdict(value: Any, context: str) -> _VerdictSpec:
    doc = _object(value, context)
    _exact(doc, _VERDICT_KEYS, context)
    _require(isinstance(doc["satisfied"], bool), f"{context}.satisfied must be boolean")
    confidence = _number(doc["confidence"], f"{context}.confidence", minimum=0.0)
    _require(confidence <= 1.0, f"{context}.confidence must be <= 1")
    evidence = doc["evidence"]
    _require(isinstance(evidence, str), f"{context}.evidence must be text")
    error = doc["error"]
    _require(error is None or isinstance(error, str), f"{context}.error must be text or null")
    return _VerdictSpec(doc["satisfied"], confidence, evidence, error)


def load_matched_watch_manifest(path: str | Path) -> _Manifest:
    """Load and freeze a strict manifest plus its safely snapshotted synthetic media."""

    try:
        loaded = load_json_file(path)
    except SafeJSONInputError as exc:
        raise MatchedWatchInputError(f"unsafe matched-Watch manifest: {exc}") from exc
    doc = _object(loaded.value, "manifest")
    _exact(doc, _MANIFEST_KEYS, "manifest")
    _require(doc["schema"] == _SCHEMA, f"manifest.schema must be {_SCHEMA!r}")
    case_id = _text(doc["case_id"], "manifest.case_id")
    _require(doc["archetype"] == _ARCHETYPE, f"manifest.archetype must be {_ARCHETYPE!r}")
    split = _text(doc["split"], "manifest.split")

    media_ref = _object(doc["media"], "manifest.media")
    _exact(media_ref, _MEDIA_REF_KEYS, "manifest.media")
    expected_media_sha = _sha(media_ref["sha256"], "manifest.media.sha256")
    media_path = Path(_text(media_ref["path"], "manifest.media.path"))
    if not media_path.is_absolute():
        media_path = Path(loaded.receipt.canonical_path).parent / media_path
    try:
        media_loaded = load_json_file(media_path)
    except SafeJSONInputError as exc:
        raise MatchedWatchInputError(f"unsafe matched-Watch media: {exc}") from exc
    _require(
        media_loaded.receipt.sha256 == expected_media_sha,
        "manifest media digest does not match the descriptor-backed media snapshot",
    )
    media = _object(media_loaded.value, "media")
    _exact(media, _MEDIA_KEYS, "media")
    _require(media["schema"] == _MEDIA_SCHEMA, f"media.schema must be {_MEDIA_SCHEMA!r}")
    width = int(_number(media["width"], "media.width", minimum=1))
    height = int(_number(media["height"], "media.height", minimum=1))

    raw_frames = media["frames"]
    _require(isinstance(raw_frames, list) and raw_frames, "media.frames must be non-empty")
    frames: list[_FrameSpec] = []
    prior_t: float | None = None
    for index, raw in enumerate(raw_frames):
        context = f"media.frames[{index}]"
        frame = _object(raw, context)
        _exact(frame, _FRAME_KEYS, context)
        t = _number(frame["t"], f"{context}.t", minimum=0.0)
        _require(prior_t is None or t > prior_t, "media frame timestamps must be strictly increasing")
        prior_t = t
        fill = frame["fill"]
        _require(
            isinstance(fill, int) and not isinstance(fill, bool) and 0 <= fill <= 255,
            f"{context}.fill must be an integer in [0, 255]",
        )
        frames.append(
            _FrameSpec(
                index,
                t,
                fill,
                _text(frame["token"], f"{context}.token"),
                _parse_verdict(frame["from"], f"{context}.from"),
                _parse_verdict(frame["to"], f"{context}.to"),
            )
        )

    replay = _object(doc["replay"], "manifest.replay")
    _exact(replay, _REPLAY_KEYS, "manifest.replay")
    fps = _number(replay["fps"], "manifest.replay.fps", minimum=0.000001)
    _require(replay["timebase"] == "media", "manifest.replay.timebase must be 'media'")
    source_id = _text(replay["source_id"], "manifest.replay.source_id")

    watch = _object(doc["watch"], "manifest.watch")
    _exact(watch, _WATCH_KEYS, "manifest.watch")
    _require(watch["template_ref"] == _TEMPLATE, f"manifest.watch.template_ref must be {_TEMPLATE!r}")
    _require(watch["domain_pack"] == _PACK, f"manifest.watch.domain_pack must be {_PACK!r}")
    parameters = _object(watch["parameters"], "manifest.watch.parameters")

    judge = _object(doc["judge"], "manifest.judge")
    _exact(judge, _JUDGE_KEYS, "manifest.judge")
    judge_identity = {
        "adapter": _text(judge["adapter"], "manifest.judge.adapter"),
        "model": _text(judge["model"], "manifest.judge.model"),
        "prompt_sha256": _sha(judge["prompt_sha256"], "manifest.judge.prompt_sha256"),
    }
    _require(
        judge_identity["adapter"] == "scripted_transition/v1",
        "L1 matched-Watch manifests require adapter 'scripted_transition/v1'",
    )

    adaptive = _object(doc["adaptive"], "manifest.adaptive")
    _exact(adaptive, _ADAPTIVE_KEYS, "manifest.adaptive")
    stat = _text(adaptive["salience_stat"], "manifest.adaptive.salience_stat")
    _require(stat in {"mean", "block_max", "p95"}, "unsupported adaptive salience_stat")
    threshold = _number(adaptive["threshold"], "manifest.adaptive.threshold", minimum=0.0)
    _require(threshold <= 1.0, "manifest.adaptive.threshold must be <= 1")

    raw_events = doc["events"]
    _require(isinstance(raw_events, list) and raw_events, "manifest.events must be non-empty")
    events: list[_EventSpec] = []
    ids: set[str] = set()
    for index, raw in enumerate(raw_events):
        context = f"manifest.events[{index}]"
        event = _object(raw, context)
        _exact(event, _EVENT_KEYS, context)
        event_id = _text(event["id"], f"{context}.id")
        _require(event_id not in ids, f"duplicate event id {event_id!r}")
        ids.add(event_id)
        start = _number(event["window_start_s"], f"{context}.window_start_s", minimum=0.0)
        end = _number(event["window_end_s"], f"{context}.window_end_s", minimum=0.0)
        eligible = _number(event["eligible_at_s"], f"{context}.eligible_at_s", minimum=0.0)
        _require(start <= eligible <= end, f"{context}.eligible_at_s must be inside its window")
        events.append(
            _EventSpec(
                event_id,
                eligible,
                start,
                end,
                _text(event["review_authority"], f"{context}.review_authority"),
                _sha(event["label_digest"], f"{context}.label_digest"),
            )
        )
    for frame in frames:
        owners = [event.id for event in events if event.start <= frame.t <= event.end]
        _require(len(owners) <= 1, f"candidate t={frame.t} belongs to multiple event windows {owners!r}")
    for index, left in enumerate(events):
        for right in events[index + 1 :]:
            _require(
                left.end < right.start or right.end < left.start,
                f"event windows {left.id!r} and {right.id!r} overlap",
            )

    evidence_classification = _text(
        doc["evidence_classification"], "manifest.evidence_classification"
    )
    _require(
        evidence_classification in _EVIDENCE_CLASSES,
        f"evidence classification must be one of {_EVIDENCE_CLASSES!r}",
    )
    return _Manifest(
        MappingProxyType(doc),
        loaded.receipt,
        media_loaded.receipt,
        case_id,
        split,
        fps,
        source_id,
        MappingProxyType(parameters),
        MappingProxyType(judge_identity),
        _text(adaptive["operating_point"], "manifest.adaptive.operating_point"),
        stat,
        threshold,
        tuple(events),
        tuple(frames),
        width,
        height,
        evidence_classification,
    )


class _RecordingTransitionJudge:
    SUPPORTS_FUSED_AUDIO_INPUT = True

    def __init__(self, frames: Sequence[_FrameSpec], from_condition: str, to_condition: str) -> None:
        self._frames = tuple(frames)
        self._from = from_condition
        self._to = to_condition
        self.calls: list[ProviderCallReceipt] = []

    async def detect(self, frame: Any, goals: Any) -> list[Any]:
        return []

    async def judge(self, condition: str, frame: Any, **kwargs: Any) -> Verdict:
        if not isinstance(frame, _JudgePayload):
            raise TypeError("scripted transition judge requires the bound candidate payload")
        spec = self._frames[frame.index]
        if condition == self._from:
            predicate, answer = "from", spec.from_verdict
        elif condition == self._to:
            predicate, answer = "to", spec.to_verdict
        else:
            predicate, answer = "unknown", _VerdictSpec(False, 0.0, "", "unexpected condition")
        sequence = len(self.calls) + 1
        if answer.error is not None:
            self.calls.append(
                ProviderCallReceipt(
                    sequence,
                    frame.index,
                    frame.t,
                    predicate,
                    _canonical_sha(condition),
                    "error",
                    None,
                    None,
                    None,
                    answer.error,
                )
            )
            raise RuntimeError(answer.error)
        self.calls.append(
            ProviderCallReceipt(
                sequence,
                frame.index,
                frame.t,
                predicate,
                _canonical_sha(condition),
                "success",
                answer.satisfied,
                answer.confidence,
                answer.evidence,
                None,
            )
        )
        return Verdict(answer.satisfied, answer.confidence, answer.evidence)


def _profile(arm: str, stat: str, threshold: float) -> SourceProfile:
    return SourceProfile(
        name=f"matched-watch-{stat}",
        modalities=frozenset({"video"}),
        temporal_unit="frame",
        admission_strategy="pixel-change",
        evidence_strategy="frame",
        identity_strategy="scene",
        default_timebase="media",
        parameters={"salience_stat": stat, "salience_threshold": threshold},
        notes="preregistered matched-Watch L1 execution profile",
    )


def _trace_dict(event: Any) -> dict[str, Any]:
    return {
        "trace_id": event.trace_id,
        "stage": event.stage,
        "t": event.t,
        "goal_id": event.goal_id,
        "data": dict(event.data),
    }


def _health_dict(event: RuntimeHealthEvent) -> dict[str, Any]:
    return {"kind": event.kind, "ts": event.ts, "data": dict(event.data or {})}


def _incident_dict(incident: Any) -> dict[str, Any]:
    return {
        "id": incident.id,
        "concern_id": incident.concern_id,
        "ts": float(incident.ts),
        "message": incident.message,
        "entity_id": incident.entity_id,
        "reason": incident.reason,
        "watchplan_id": incident.watchplan_id,
        "judgment_event_id": getattr(incident, "judgment_event_id", None),
        "evidence_origin": getattr(incident, "evidence_origin", None),
    }


def _profile_dict(profile: SourceProfile) -> dict[str, Any]:
    return {
        "name": profile.name,
        "modalities": sorted(profile.modalities),
        "temporal_unit": profile.temporal_unit,
        "admission_strategy": profile.admission_strategy,
        "evidence_strategy": profile.evidence_strategy,
        "identity_strategy": profile.identity_strategy,
        "default_timebase": profile.default_timebase,
        "parameters": dict(profile.parameters),
        "stability": profile.stability,
        "notes": profile.notes,
    }


async def _run_arm(manifest: _Manifest, plan: Any, arm: str) -> ArmReceipt:
    threshold = 0.0 if arm == "baseline" else manifest.threshold
    profile = _profile(arm, manifest.salience_stat, threshold)
    from_condition = str(plan.explicit["Goal.state.from"])
    to_condition = str(plan.explicit["Goal.state.to"])
    adapter = _RecordingTransitionJudge(manifest.frames, from_condition, to_condition)
    core = Percept.create(
        vision=adapter,
        settings=Settings(),
        discover_plugins=False,
        mode="offline",
    )
    resolver = ConfigResolver()
    resolved = resolver.resolve(
        plan,
        source_profile=profile,
        domain_pack=FACILITY_OBSERVATION_V1,
        settings=core.settings,
    )
    apply_receipt = apply_watch_plan(
        core,
        plan,
        resolver=resolver,
        source_profile=profile,
        domain_pack=FACILITY_OBSERVATION_V1,
        settings=core.settings,
    )
    _require(
        apply_receipt.status in {"applied", "applied_with_warnings"},
        f"{arm} WatchPlan apply failed: {apply_receipt.to_dict()!r}",
    )
    _require(len(apply_receipt.watch_ids) == 1, f"{arm} armed an unexpected Watch set")
    goal_id = apply_receipt.watch_ids[0]
    bound = core._watch_runtime_bindings.get(goal_id)
    _require(bound is not None, f"{arm} compiled runtime binding is missing")
    compiled = bound[0]

    stats = RuntimeStats()
    traces: list[Any] = []
    health: list[RuntimeHealthEvent] = []
    asks: list[Any] = []
    source = [
        (
            frame.t,
            gray_frame(manifest.width, manifest.height, frame.fill),
            _JudgePayload(frame.index, frame.token, frame.t),
        )
        for frame in manifest.frames
    ]
    incidents = [
        incident
        async for incident in run(
            core,
            source=source,
            fps=manifest.fps,
            gate_source=manifest.source_id,
            timebase="media",
            judge_dispatch="inline",
            stats=stats,
            on_trace=traces.append,
            on_health=health.append,
            on_ask=asks.append,
            regate_on_goal_change=False,
            coverage_floor=compiled.coverage_floor,
        )
    ]

    edge = [item for item in traces if item.stage == "edge.decision" and item.goal_id is None]
    transitions = [
        item for item in traces if item.stage == "transition.decision" and item.goal_id == goal_id
    ]
    candidate_ts = tuple(frame.t for frame in manifest.frames)
    blockers: list[str] = []
    if len(edge) != len(manifest.frames):
        blockers.append("edge trace count does not equal candidate count")
    edge_by_t = {float(item.t): item for item in edge if item.t is not None}
    if tuple(sorted(edge_by_t)) != candidate_ts:
        blockers.append("edge trace timestamps do not equal enumerated candidate timestamps")
    transition_by_t = {float(item.t): item for item in transitions if item.t is not None}
    incident_by_t: dict[float, list[Any]] = {}
    for incident in incidents:
        incident_by_t.setdefault(float(incident.ts), []).append(incident)
    ask_by_t: dict[float, list[Any]] = {}
    for ask in asks:
        ask_t = getattr(ask, "ts", None)
        if isinstance(ask_t, (int, float)):
            ask_by_t.setdefault(float(ask_t), []).append(ask)
    calls_by_index: dict[int, list[ProviderCallReceipt]] = {}
    for call in adapter.calls:
        calls_by_index.setdefault(call.candidate_index, []).append(call)

    candidates: list[CandidateReceipt] = []
    for frame in manifest.frames:
        edge_event = edge_by_t.get(frame.t)
        edge_data = dict(edge_event.data) if edge_event is not None else {}
        admitted = goal_id in tuple(edge_data.get("goal_ids", ()))
        transition = transition_by_t.get(frame.t)
        dispatched = transition is not None
        frame_calls = tuple(calls_by_index.get(frame.index, ()))
        frame_incidents = tuple(item.id for item in incident_by_t.get(frame.t, ()))
        if dispatched:
            if frame_incidents:
                decision = "fire"
            elif ask_by_t.get(frame.t):
                decision = "ask"
            elif transition.data.get("outcome") == "judge_degraded" or any(
                call.status != "success" for call in frame_calls
            ):
                decision = "missing"
            else:
                decision = "silent"
        else:
            decision = None
        candidates.append(
            CandidateReceipt(
                frame.index,
                frame.t,
                _freeze(edge_data),
                admitted,
                dispatched,
                frame_calls,
                decision,
                (str(transition.data.get("outcome")) if transition is not None else None),
                frame_incidents,
            )
        )

    admitted_count = sum(item.admitted for item in candidates)
    dispatch_count = sum(item.runtime_dispatched for item in candidates)
    failures = sum(item.status != "success" for item in adapter.calls)
    successes = len(adapter.calls) - failures
    if stats.frames != len(candidates):
        blockers.append("RuntimeStats.frames does not equal candidate count")
    if stats.goal_admissions != admitted_count or stats.escalations != admitted_count:
        blockers.append("RuntimeStats admission counts do not equal edge traces")
    if stats.perceive_dispatches != dispatch_count or stats.judge_calls != dispatch_count:
        blockers.append("RuntimeStats dispatch counts do not equal transition traces")
    if dispatch_count != admitted_count:
        blockers.append("admitted candidates and inline runtime dispatches differ")
    if len(adapter.calls) != 2 * dispatch_count:
        blockers.append("transition provider subcall count is incomplete")
    if failures:
        blockers.append("provider subcall failure")
    if stats.frames_dropped or stats.sink_errors:
        blockers.append("runtime dropped frames or reported sink errors")
    if (
        stats.judge_superseded
        or stats.judge_revision_dropped
        or stats.judge_source_discontinuity_dropped
    ):
        blockers.append("async/revision work loss is forbidden")
    if stats.source_excluded or stats.source_discontinuities or stats.source_missing_sequences:
        blockers.append("source exclusion, gap, or discontinuity is forbidden")
    if any(item.kind == "runtime_error" for item in health):
        blockers.append("runtime_error health event")
    if arm == "baseline":
        if threshold != 0.0:
            blockers.append("baseline threshold is not exactly 0.0")
        if admitted_count != len(candidates) or dispatch_count != len(candidates):
            blockers.append("baseline is not exhaustive over sampled candidates")
        if stats.coast_admits or stats.fixed_admits or stats.audio_wakes:
            blockers.append("baseline used a forbidden non-signal admission path")

    resolved_dict = resolved.to_dict()
    compiled_dict = {
        "goal_id": compiled.goal.id,
        "timebase": compiled.timebase,
        "coverage_floor": compiled.coverage_floor,
        "watchspec_kwargs": dict(compiled.watchspec_kwargs),
        "components": dict(compiled.components),
    }
    config_sha = _canonical_sha(
        {
            "effective": resolved_dict["effective"],
            "source_profile": _profile_dict(profile),
            "compiled_binding": compiled_dict,
        }
    )
    return ArmReceipt(
        arm,
        _freeze(_profile_dict(profile)),
        _freeze(resolved_dict),
        _freeze(apply_receipt.to_dict()),
        _freeze(compiled_dict),
        _freeze(manifest.judge_identity),
        config_sha,
        candidate_ts,
        tuple(candidates),
        tuple(_freeze(_incident_dict(item)) for item in incidents),
        tuple(
            _freeze(
                {
                    "goal_id": getattr(item, "goal_id", None),
                    "action": getattr(item, "action", None),
                    "text": getattr(item, "text", None),
                }
            )
            for item in asks
        ),
        tuple(_freeze(_trace_dict(item)) for item in traces),
        tuple(_freeze(_health_dict(item)) for item in health),
        _freeze(asdict(stats)),
        dispatch_count,
        len(adapter.calls),
        successes,
        failures,
        not blockers,
        tuple(blockers),
    )


def _sanitized_effective(arm: ArmReceipt) -> dict[str, Any]:
    effective = dict(arm.resolved_watch["effective"])
    return {key: value for key, value in effective.items() if key not in _SHA_PATHS}


def _sanitized_profile(arm: ArmReceipt) -> dict[str, Any]:
    profile = _thaw(arm.source_profile)
    profile["parameters"].pop("salience_threshold", None)
    return profile


def _sanitized_binding(arm: ArmReceipt) -> dict[str, Any]:
    binding = _thaw(arm.compiled_binding)
    kwargs = binding["watchspec_kwargs"]
    kwargs.pop("threshold", None)
    kwargs.pop("stat_thresholds", None)
    return binding


def _binding_consistent(arm: ArmReceipt) -> bool:
    profile = arm.source_profile
    stat = profile["parameters"]["salience_stat"]
    threshold = float(profile["parameters"]["salience_threshold"])
    kwargs = arm.compiled_binding["watchspec_kwargs"]
    effective = arm.resolved_watch["effective"]
    threshold_path = {
        "mean": "Settings.edge.motion_gate",
        "block_max": "Settings.edge.motion_gate_block_max",
        "p95": "Settings.edge.motion_gate_p95",
    }[stat]
    return (
        kwargs["salience_stat"] == stat
        and float(kwargs["stat_thresholds"][stat]) == threshold
        and effective["Settings.edge.salience_stat"] == stat
        and float(effective[threshold_path]) == threshold
    )


def validate_matched_arms(
    baseline: ArmReceipt,
    adaptive: ArmReceipt,
    *,
    watch_plan_sha256: str | None = None,
    expected_media_sha256: str | None = None,
) -> tuple[str, ...]:
    """Return every comparability blocker; an empty tuple means the arms are matched."""

    blockers: list[str] = []
    if baseline.arm != "baseline" or adaptive.arm != "adaptive":
        blockers.append("receipt arms must be baseline then adaptive")
    if baseline.candidate_timestamps != adaptive.candidate_timestamps:
        blockers.append("candidate timestamp sequences differ")
    baseline_plan = baseline.resolved_watch.get("plan")
    adaptive_plan = adaptive.resolved_watch.get("plan")
    if baseline_plan != adaptive_plan:
        blockers.append("rendered WatchPlan differs")
    if baseline.resolved_watch.get("plan_hash") != adaptive.resolved_watch.get("plan_hash"):
        blockers.append("resolved plan hashes differ")
    if baseline.resolved_watch.get("component_snapshot_hash") != adaptive.resolved_watch.get(
        "component_snapshot_hash"
    ):
        blockers.append("component snapshots differ")
    if _sanitized_effective(baseline) != _sanitized_effective(adaptive):
        blockers.append("arms differ outside approved admission configuration")
    if _sanitized_profile(baseline) != _sanitized_profile(adaptive):
        blockers.append("SourceProfiles differ outside approved admission configuration")
    if _sanitized_binding(baseline) != _sanitized_binding(adaptive):
        blockers.append("compiled bindings differ outside approved admission configuration")
    if baseline.judge_identity != adaptive.judge_identity:
        blockers.append("judge adapter/model/prompt identities differ")
    if not _binding_consistent(baseline) or not _binding_consistent(adaptive):
        blockers.append("SourceProfile/resolved/compiled admission bindings are inconsistent")
    if float(baseline.source_profile["parameters"]["salience_threshold"]) != 0.0:
        blockers.append("baseline compiled admission threshold is not exactly 0.0")
    if (
        watch_plan_sha256 is not None
        and f"sha256:{baseline.resolved_watch.get('plan_hash')}" != watch_plan_sha256
    ):
        blockers.append("resolved WatchPlan does not match the receipt WatchPlan identity")
    if expected_media_sha256 is not None and not expected_media_sha256.startswith("sha256:"):
        blockers.append("media identity is malformed")
    blockers.extend(f"baseline: {item}" for item in baseline.blockers)
    blockers.extend(f"adaptive: {item}" for item in adaptive.blockers)
    return tuple(blockers)


async def run_matched_watch_async(path: str | Path) -> MatchedWatchReceipt:
    manifest = load_matched_watch_manifest(path)
    canonical_plan = watch_template(_TEMPLATE, manifest.watch_parameters)
    # Fresh render + core per arm. Sequential replay keeps append order deterministic and prevents
    # any shallow nested state in a future WatchPlan field from becoming a cross-arm channel.
    baseline = await _run_arm(
        manifest,
        watch_template(_TEMPLATE, manifest.watch_parameters),
        "baseline",
    )
    adaptive = await _run_arm(
        manifest,
        watch_template(_TEMPLATE, manifest.watch_parameters),
        "adaptive",
    )
    plan_dict = canonical_plan.to_dict()
    plan_sha = _canonical_sha(plan_dict)
    blockers = validate_matched_arms(
        baseline,
        adaptive,
        watch_plan_sha256=plan_sha,
        expected_media_sha256=manifest.media_receipt.sha256,
    )
    try:
        package_version = version("percept-vision")
    except PackageNotFoundError:
        package_version = "source-tree"
    events = tuple(
        MappingProxyType(
            {
                "id": event.id,
                "eligible_at_s": event.eligible_at_s,
                "window_start_s": event.start,
                "window_end_s": event.end,
                "review_authority": event.review_authority,
                "label_digest": event.label_digest,
            }
        )
        for event in manifest.events
    )
    return MatchedWatchReceipt(
        _freeze(dict(manifest.document)),
        _freeze(_safe_receipt_dict(manifest.input_receipt)),
        _freeze(_safe_receipt_dict(manifest.media_receipt)),
        _freeze(plan_dict),
        plan_sha,
        _freeze(manifest.judge_identity),
        events,
        manifest.evidence_classification,
        package_version,
        baseline,
        adaptive,
        not blockers,
        blockers,
        datetime.now(timezone.utc).isoformat(),
    )


def run_matched_watch(path: str | Path) -> MatchedWatchReceipt:
    """Synchronous entry point for the finite L1 replay."""

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(run_matched_watch_async(path))
    raise RuntimeError("run_matched_watch() cannot be called from an active event loop; await the async API")


def _event_rows(receipt: MatchedWatchReceipt, arm: ArmReceipt) -> tuple[list[dict[str, Any]], list[str]]:
    events: list[dict[str, Any]] = []
    fire_ids: list[str] = []
    known_incidents = {item["id"]: item for item in arm.incidents}
    aligned_incidents: set[str] = set()
    for event in receipt.event_windows:
        candidates = [
            item
            for item in arm.candidates
            if event["window_start_s"] <= item.t <= event["window_end_s"]
        ]
        fires = [
            incident
            for incident in arm.incidents
            if event["window_start_s"] <= float(incident["ts"]) <= event["window_end_s"]
        ]
        _require(len(fires) <= 1, f"event {event['id']!r} has multiple aligned incidents")
        _require(
            not any(str(item["id"]) in aligned_incidents for item in fires),
            "one incident aligned to multiple event windows",
        )
        asks = [item for item in candidates if item.decision == "ask"]
        missing = [item for item in candidates if item.decision == "missing"]
        if fires:
            outcome = "fire"
            fire_ids.append(str(event["id"]))
            aligned_incidents.update(str(item["id"]) for item in fires)
        elif asks:
            outcome = "ask"
        elif missing:
            outcome = "missing"
        else:
            outcome = "silent"
        events.append(
            {
                "id": event["id"],
                "candidate_windows": len(candidates),
                "admitted_candidates": sum(item.admitted for item in candidates),
                "outcome": outcome,
            }
        )
    for incident_id in known_incidents:
        if incident_id not in aligned_incidents:
            fire_ids.append(incident_id)
    return events, fire_ids


def derive_viability_rows(receipt: MatchedWatchReceipt) -> tuple[dict[str, Any], dict[str, Any]]:
    """Derive the v3 viability rows exclusively from an immutable execution receipt."""

    _require(receipt.evidence_classification in _EVIDENCE_CLASSES, "unsupported evidence classification")
    _require(receipt.comparable, f"matched arms are not comparable: {receipt.comparability_blockers!r}")
    _require(receipt.baseline.valid and receipt.adaptive.valid, "one or more arm receipts are invalid")
    rows: list[dict[str, Any]] = []
    for arm in (receipt.baseline, receipt.adaptive):
        observed_calls = [
            call
            for candidate in arm.candidates
            for call in candidate.provider_calls
        ]
        _require(
            [call.sequence for call in observed_calls] == list(range(1, len(observed_calls) + 1)),
            "provider call sequence is incomplete or reordered",
        )
        for candidate in arm.candidates:
            if candidate.runtime_dispatched:
                _require(candidate.admitted, "runtime dispatched a non-admitted candidate")
                _require(
                    [call.predicate for call in candidate.provider_calls] == ["from", "to"],
                    "transition dispatch does not carry exact from/to provider subcalls",
                )
                _require(
                    all(
                        call.candidate_index == candidate.index
                        and call.candidate_t == candidate.t
                        and call.status == "success"
                        for call in candidate.provider_calls
                    ),
                    "provider subcall provenance/status mismatch",
                )
            else:
                _require(
                    not candidate.provider_calls,
                    "non-dispatched candidate carries provider subcalls",
                )
        _require(
            len(observed_calls) == arm.provider_call_attempts
            and sum(call.status == "success" for call in observed_calls)
            == arm.provider_call_successes
            and sum(call.status != "success" for call in observed_calls)
            == arm.provider_call_failures,
            "provider subcall aggregates do not match append-only records",
        )
        events, fire_ids = _event_rows(receipt, arm)
        decisions = {key: 0 for key in ("fire", "ask", "silent", "missing")}
        for candidate in arm.candidates:
            if candidate.runtime_dispatched:
                _require(candidate.decision in decisions, "runtime dispatch is missing a decision")
                decisions[str(candidate.decision)] += 1
        _require(
            sum(decisions.values()) == arm.runtime_dispatches,
            "decision counts do not sum to runtime dispatches",
        )
        rows.append(
            {
                "case_id": receipt.manifest["case_id"],
                "archetype": receipt.manifest["archetype"],
                "operating_point": receipt.manifest["adaptive"]["operating_point"],
                # The viability manifest fingerprints the adaptive operating point; the baseline's
                # own executable configuration remains independently bound in the arm receipt.
                "configuration_sha256": receipt.adaptive.configuration_sha256,
                "arm": arm.arm,
                "evidence_kind": "synthetic",
                "candidate_windows_total": len(arm.candidates),
                "candidate_windows_admitted": sum(item.admitted for item in arm.candidates),
                "runtime_dispatches": arm.runtime_dispatches,
                "judge_calls": arm.provider_call_attempts,
                "decisions": decisions,
                "event_windows": events,
                "fire_event_ids": fire_ids,
            }
        )
    baseline, adaptive = rows
    _require(
        baseline["candidate_windows_admitted"] == baseline["candidate_windows_total"]
        and baseline["runtime_dispatches"] == baseline["candidate_windows_total"],
        "derived baseline is not exhaustive",
    )
    return baseline, adaptive


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out")
    args = parser.parse_args(argv)
    try:
        receipt = run_matched_watch(args.manifest)
        payload = receipt.to_json()
    except MatchedWatchInputError as exc:
        parser.error(str(exc))
    if args.out:
        try:
            with Path(args.out).open("x", encoding="utf-8") as stream:
                stream.write(payload + "\n")
        except FileExistsError:
            parser.error("--out is append-only and already exists; choose a new receipt path")
    else:
        print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
