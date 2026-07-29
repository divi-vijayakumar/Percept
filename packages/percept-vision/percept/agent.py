"""The Percept orchestrator — the reactive watch loop (Phase 1).

One firing path: for each active goal, judge the plain-language condition over the in-focus
ROI (with any user-asserted corrections), run the verdict through the three-state gate, and emit
fires/asks on the rising edge. Later phases add *beside* this — the spoken turn (Phase 3),
episodes (Phase 4), events (Phase 5) — without editing the gate/graph contracts.
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import math
import time
import warnings
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Callable, Hashable

from . import registry
from .adapters import (
    AudioFrame,
    Box,
    Decision,
    Goal,
    GoalOrigin,
    Observation,
    TurnResult,
    Verdict,
)
from .behavior import (
    BehaviorExecutor,
    EpisodeFoldResult,
    ShadowObservation,
    normalize_event_chain_verdict,
    program_of,
)
from .behavior_routes import (
    BehaviorRoute,
    UndrivenBehaviorRouteWarning,
    UnsupportedBehaviorRouteError,
    classify_armed_routes,
    goal_behavior_route,
    require_worker_routes,
    undriven_route_message,
)
from .duration_routes import duration_policy, require_duration_routes
from .cache import CacheCascade, CacheKey, FrameHashLayer
from .concern import attention_weight, consequence_of, requires_verification, resolution_ready
from .config import ModeError, Settings, load_settings, missing_credential, resolve_mode
from .contracts import JudgmentEvent, JudgmentTimingBuilder, SubjectRef, judgment_event_id
from .escalation import MAX_PERCEPTION_LEVEL, escalate_perception, should_escalate
from .consent import ActionLog, ActionRecord, ConsentDenied, ConsentProfile
from .entity import EntityGraph
from .extraction import CONTEXT_KEY as EXTRACTOR_CONTEXT_KEY
from .claims import ClaimState, concern_authority_of
from .episodes import (
    EPISODE_POLICIES,
    ContinuationDecision,
    Episode,
    EpisodeJudge,
    EpisodeLog,
    EpisodePolicy,
    MutationCheck,
    episode_policy_of,
    subject_is_unknown,
)
from .identity_policy import (
    PER_ENTITY_IDENTITY_POLICIES,
    SCENE_IDENTITY,
    identity_policy_of,
    select_identity_policy,
    transition_latch_key,
    watch_reducer_entity_id,
)
from .judge_context import JUDGE_STATIC_CONTEXT_KEY, ReservedContextKeyError
from .events import AudioUnit, AudioWindow, EventLog, FrameWindow
from .gate import FIRE, Gate, GateAction, GateOutcome, classify
from .goals import GoalOp, GoalRegistry
from .observability import Funnel, Metrics, OTelExporter, TraceEvent, get_logger, health, judge_usage
from .rhythm import RhythmTracker
from .routing import RouteEngine
from .scheduler import Scheduler
from .tracing import STAGE_JUDGMENT_EVENT, STAGE_SHADOW_FOLD, STAGE_TRANSITION_DECISION


#: LRU ceiling on the judgment evidence-sequencer's per-key counters. Matches the gate's own
#: default ``max_keys`` so the observability map can never outgrow the gate state it mirrors.
_SEQ_MAX_KEYS = 10_000
_PROVIDER_ACCOUNTING_FALLBACK_MAX = 1_024


@dataclass
class FireEvent:
    """What the watch loop emits when the gate acts."""

    goal_id: str
    action: GateAction  # "fire" | "ask"
    text: str
    entity_id: str | None = None
    verdict: Verdict | None = None
    # The frozen :class:`percept.contracts.JudgmentEvent` for this decision (ADR-001 §1): the
    # gate key, the evidence sequence, the action, and the five-stamp two-domain timing the funnel
    # actually took. Populated on the judged watch path (``perceive_judged`` → ``_fire_concern``)
    # and on a fired live transition (``_perceive_transition``), whose event is marked
    # ``evidence_origin="transition"``. ``None`` on episode/rhythm/boundary/absence/process-QA paths
    # and on legacy direct transition-executor calls that supply no timing — an absent event, never
    # an invented one. Appended LAST with a ``None`` default: existing positional constructions and
    # consumers are unchanged.
    judgment: JudgmentEvent | None = None
    # The enriched judgment envelope carried from the executor's ``Fired`` so it survives to the
    # delivered Incident (``make_incident`` reads these off the fire). All OPTIONAL/defaulted: an
    # episode/rhythm/boundary/absence fire that sets none of them is byte-for-byte unchanged.
    reason: str | None = None
    modality: str | None = None
    roi: object | None = None
    provenance: str | None = None
    # The episode-owned provisional/superseding link carried from the executor's ``Fired``
    # (``behavior.Fired.revision_id``/``.supersedes``) so a live delivery — ``incidents.make_incident``
    # in the ``worker.py`` path — can put the correction relation on the wire instead of reading None on
    # every fire. ``revision_id`` is THIS fire's stable episode-revision handle
    # (``<episode_id>#r<n>``); ``supersedes`` names the prior revision this one corrects, and is
    # ``None`` for an original/provisional fire. Both OPTIONAL/defaulted and appended LAST: every
    # existing positional ``FireEvent(...)`` construction and every consumer that ignores them sees
    # byte-for-byte today's shape.
    revision_id: str | None = None
    supersedes: str | None = None
    # Exact immutable audio evidence submitted to the provider for this decision. Populated only by
    # the compiled audio-only path; the worker builds the delivered snippet from this same tuple.
    audio_units: tuple[AudioUnit, ...] | None = None


@dataclass(frozen=True)
class _JudgmentAdmissionToken:
    admission_id: str
    goal_id: str
    modality: str
    route: str
    goal_object_token: object
    watch_application: str | None
    watch_revision: int
    goals_registry_revision: int
    watch_runtime_token: object | None
    judge_binding_token: object
    source_id: str
    source_generation: int
    fused_generation: int | None
    evidence_seq: int
    parent_source_trace_id: str
    judgment_trace_id: str


@dataclass(frozen=True)
class _ProviderRequest:
    dispatch: Any
    condition: str
    frame: Any
    frames: tuple[Any, ...]
    audio: Any
    roi: Any
    exemplar: Any
    corrections: tuple[tuple[str, str], ...]
    context: Any
    include_frames: bool = False
    include_audio: bool = False
    include_context: bool = False
    include_exemplar: bool = True


class _TemporalStrip(list):
    """Selected temporal frames with exact source-coordinate provenance."""

    def __init__(
        self,
        frames: list[Any],
        *,
        source_start_t: float,
        source_end_t: float,
        source_offsets_s: tuple[float, ...],
    ) -> None:
        super().__init__(frames)
        self.source_start_t = source_start_t
        self.source_end_t = source_end_t
        self.source_offsets_s = source_offsets_s


@dataclass(frozen=True)
class _TransitionProviderRequest:
    """Two immutable provider calls that form one transition judgment."""

    before: _ProviderRequest
    after: _ProviderRequest


@dataclass(frozen=True)
class _ReducerInputs:
    source_t: float
    ingest_observed_ts: float
    admitted_ts: float
    reducer_entity_id: str | None
    freshness: float
    focus_token: Any = None
    cache_key: Any = None
    cache_store: bool = False
    route_inputs: Any = None


@dataclass(frozen=True)
class _WatchReducerInputs:
    """Frozen reducer-only state for ordinary and local watch routes."""

    process_qa: bool = False
    cached_verdict: _VerdictSnapshot | None = None
    cache_audit: bool = False
    extractor_facts: _FrozenMapping | None = None
    direct_hit: bool = False


@dataclass(frozen=True)
class _TransitionReducerInputs:
    identity_policy: str | None
    latch_key: Hashable
    latch_key_source: str
    focus_entity_id: str | None
    strip_len: int
    enriched: bool
    extractor_facts: _FrozenMapping | None = None


@dataclass(frozen=True)
class _ProcessReducerInputs:
    process_key: tuple[str]


@dataclass(frozen=True)
class _JudgmentJob:
    token: _JudgmentAdmissionToken
    modality: str
    evidence: Any
    provider: _ProviderRequest | _TransitionProviderRequest
    reducer: _ReducerInputs
    adapter: Any
    local_verdict: _VerdictSnapshot | None = None
    trace_temporal: str | None = None
    trace_roi: bool = False
    trace_provenance: tuple[tuple[str, Any], ...] = ()

    @property
    def bundle(self) -> Any:
        return self.evidence

    @property
    def dispatch(self) -> Any:
        return self.provider.dispatch

    @property
    def condition(self) -> str:
        return self.provider.condition

    @property
    def frame(self) -> Any:
        return self.provider.frame

    @property
    def frames(self) -> tuple[Any, ...]:
        return self.provider.frames

    @property
    def audio(self) -> Any:
        return self.provider.audio

    @property
    def roi(self) -> Any:
        return self.provider.roi

    @property
    def exemplar(self) -> Any:
        return self.provider.exemplar

    @property
    def corrections(self) -> tuple[tuple[str, str], ...]:
        return self.provider.corrections

    @property
    def context(self) -> Any:
        return self.provider.context

    @property
    def source_t(self) -> float:
        return self.reducer.source_t

    @property
    def ingest_observed_ts(self) -> float:
        return self.reducer.ingest_observed_ts

    @property
    def admitted_ts(self) -> float:
        return self.reducer.admitted_ts


@dataclass(frozen=True)
class _ProviderFailure:
    kind: str
    exception_type: str
    safe_message: str


@dataclass(frozen=True)
class _FrozenMapping:
    items: tuple[tuple[Any, Any], ...]


def _freeze_provider_value(value: Any) -> Any:
    """Detach and recursively freeze provider-owned structured values."""
    from .contracts import ClipEvent

    if value is None or type(value) in (bool, int, float, str, bytes):
        return value
    if type(value) is ClipEvent:
        return ClipEvent(value.t_start, value.t_end, value.label)
    if type(value) is SubjectRef:
        return value
    if isinstance(value, dict):
        return _FrozenMapping(
            tuple(
                (_freeze_provider_value(key), _freeze_provider_value(item))
                for key, item in value.items()
            )
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_provider_value(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_freeze_provider_value(item) for item in value)
    raise ValueError(
        f"provider Verdict contains unsupported mutable value {type(value).__name__}"
    )


def _thaw_provider_value(value: Any) -> Any:
    if type(value) is _FrozenMapping:
        return {
            _thaw_provider_value(key): _thaw_provider_value(item)
            for key, item in value.items
        }
    if isinstance(value, tuple):
        return [_thaw_provider_value(item) for item in value]
    if isinstance(value, frozenset):
        return {_thaw_provider_value(item) for item in value}
    return value


@dataclass(frozen=True)
class _VerdictSnapshot:
    satisfied: bool
    confidence: float
    evidence: str
    phase: str
    action_present: bool | None
    running_answer: str
    usage: _FrozenMapping
    events: tuple[Any, ...]
    subject: SubjectRef | None
    operation: str
    value: Any
    normalized_value: Any
    target_revision_id: str | None
    asserted_by: str | None

    @classmethod
    def capture(cls, verdict: Verdict) -> "_VerdictSnapshot":
        if not _is_valid_verdict(verdict):
            raise ValueError("provider returned a malformed Verdict")
        usage = _freeze_provider_value(verdict.usage)
        if type(usage) is not _FrozenMapping:
            raise ValueError("provider Verdict usage must be a mapping")
        return cls(
            satisfied=verdict.satisfied,
            confidence=float(verdict.confidence),
            evidence=str(verdict.evidence),
            phase=str(verdict.phase),
            action_present=verdict.action_present,
            running_answer=str(verdict.running_answer),
            usage=usage,
            events=tuple(_freeze_provider_value(event) for event in verdict.events),
            subject=_freeze_provider_value(verdict.subject),
            operation=str(verdict.operation),
            value=_freeze_provider_value(verdict.value),
            normalized_value=_freeze_provider_value(verdict.normalized_value),
            target_revision_id=verdict.target_revision_id,
            asserted_by=verdict.asserted_by,
        )

    def to_verdict(self) -> Verdict:
        return Verdict(
            satisfied=self.satisfied,
            confidence=self.confidence,
            evidence=self.evidence,
            phase=self.phase,
            action_present=self.action_present,
            running_answer=self.running_answer,
            usage=_thaw_provider_value(self.usage),
            events=[_thaw_provider_value(event) for event in self.events],
            subject=self.subject,
            operation=self.operation,
            value=_thaw_provider_value(self.value),
            normalized_value=_thaw_provider_value(self.normalized_value),
            target_revision_id=self.target_revision_id,
            asserted_by=self.asserted_by,
        )


def _readonly_provider_value(value: Any) -> Any:
    """Recursively detach input state while preserving read-only mapping access."""
    import copy
    from types import MappingProxyType

    if value is None or type(value) in (bool, int, float, str, bytes):
        return value
    if isinstance(value, dict):
        return MappingProxyType(
            {
                copy.deepcopy(key): _readonly_provider_value(item)
                for key, item in value.items()
            }
        )
    if isinstance(value, (list, tuple)):
        return tuple(_readonly_provider_value(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_readonly_provider_value(item) for item in value)
    return copy.deepcopy(value)


def _provider_input_copy(value: Any) -> Any:
    """Give one provider call a detached mutable view of frozen request data."""
    import copy

    if value is None or type(value) in (bool, int, float, str, bytes):
        return value
    if isinstance(value, Mapping):
        return {
            copy.deepcopy(key): _provider_input_copy(item)
            for key, item in value.items()
        }
    if isinstance(value, tuple):
        return tuple(_provider_input_copy(item) for item in value)
    if isinstance(value, frozenset):
        return {_provider_input_copy(item) for item in value}
    return copy.deepcopy(value)


def _goal_provider_projection(goal: Goal) -> Goal:
    """Exact Goal-shaped, deeply read-only projection for detector providers."""
    return Goal(
        id=goal.id,
        condition=goal.condition,
        say=goal.say,
        source=goal.source,
        modality=goal.modality,
        state=_readonly_provider_value(goal.state),
        expiry=goal.expiry,
        origin=goal.origin,
    )


@dataclass(frozen=True)
class _ObservationAdmissionToken:
    admission_id: str
    source_id: str
    source_generation: int
    goals_registry_revision: int
    goal_object_tokens: tuple[tuple[str, object], ...]
    adapter_token: object
    parent_source_trace_id: str
    observation_trace_id: str


@dataclass(frozen=True)
class _ObservationCandidate:
    token: _ObservationAdmissionToken
    dispatch: Any
    frame: Any
    goals: tuple[Goal, ...]
    source_t: float
    ingest_observed_ts: float


@dataclass(frozen=True)
class _ObservationSnapshot:
    label: str
    box: Box
    confidence: float
    attrs: _FrozenMapping

    @classmethod
    def capture(cls, observation: Observation) -> "_ObservationSnapshot":
        if type(observation) is not Observation or type(observation.box) is not Box:
            raise ValueError("detector returned a malformed Observation")
        attrs = _freeze_provider_value(observation.attrs)
        if type(attrs) is not _FrozenMapping:
            raise ValueError("Observation attrs must be a mapping")
        return cls(
            label=str(observation.label),
            box=Box(
                float(observation.box.x),
                float(observation.box.y),
                float(observation.box.w),
                float(observation.box.h),
            ),
            confidence=float(observation.confidence),
            attrs=attrs,
        )

    def to_observation(self) -> Observation:
        return Observation(
            label=self.label,
            box=self.box,
            confidence=self.confidence,
            attrs=_thaw_provider_value(self.attrs),
        )


@dataclass(frozen=True)
class _ObservationCompletion:
    candidate: _ObservationCandidate
    observations: tuple[_ObservationSnapshot, ...]
    failure: _ProviderFailure | None
    started_ts: float
    completed_ts: float


@dataclass(frozen=True)
class _ObservationReduction:
    admission_id: str
    stale_reason: str | None
    failure: _ProviderFailure | None
    focus: Any = None
    candidate: _ObservationCandidate | None = None


@dataclass(frozen=True)
class _ProviderCallCompletion:
    """One frozen provider result with no live-core references."""

    verdict: _VerdictSnapshot
    failure: _ProviderFailure | None
    usage: tuple[tuple[str, Any], ...]
    judge_latency_s: float
    judge_started_ts: float
    judge_completed_ts: float
    provider_spent: bool = True


@dataclass(frozen=True)
class _WatchJudgmentResult:
    call: _ProviderCallCompletion


@dataclass(frozen=True)
class _ProcessJudgmentResult:
    """A provider verdict whose only semantic output is a process-log delta."""

    call: _ProviderCallCompletion


@dataclass(frozen=True)
class _TransitionJudgmentResult:
    """The ordered before/after reads for one atomic transition tick."""

    before: _ProviderCallCompletion
    after: _ProviderCallCompletion | None


@dataclass(frozen=True)
class _JudgmentCompletion:
    """Provider-only result; cognition changes only during owner reduction.

    A transition's two provider reads share this one completion object, and
    therefore receive one owner-ready completion ordinal when the scheduler
    publishes it.  The scheduler assigns ``completion_ordinal``; direct/manual
    dispatch leaves it ``None``.
    """

    job: _JudgmentJob
    result: _WatchJudgmentResult | _ProcessJudgmentResult | _TransitionJudgmentResult
    completion_ordinal: int | None = None

    @property
    def calls(self) -> tuple[_ProviderCallCompletion, ...]:
        if type(self.result) is _TransitionJudgmentResult:
            return (
                (self.result.before,)
                if self.result.after is None
                else (self.result.before, self.result.after)
            )
        return (self.result.call,)

    @property
    def verdict(self) -> _VerdictSnapshot:
        if type(self.result) is _TransitionJudgmentResult:
            raise AttributeError("paired transition completion has no single verdict")
        return self.result.call.verdict

    @property
    def failure(self) -> _ProviderFailure | None:
        return next((call.failure for call in self.calls if call.failure is not None), None)

    @property
    def usage(self) -> tuple[tuple[str, Any], ...]:
        if len(self.calls) == 1:
            return self.calls[0].usage
        combined: dict[str, Any] = {}
        for call in self.calls:
            for key, value in call.usage:
                prior = combined.get(key)
                combined[key] = (
                    prior + value
                    if isinstance(prior, (int, float))
                    and isinstance(value, (int, float))
                    else value
                )
        return tuple(combined.items())

    @property
    def judge_latency_s(self) -> float:
        return sum(call.judge_latency_s for call in self.calls)

    @property
    def judge_started_ts(self) -> float:
        return self.calls[0].judge_started_ts

    @property
    def judge_completed_ts(self) -> float:
        return self.calls[-1].judge_completed_ts


_FUSED_ADMISSION_LEDGER_MAX = 1024


# -- The resolved WatchProfile.plan_mode -> the concern's PerceptionSpec -----------------------
# ``plan_mode`` was carried onto the core before anything read it. This module makes it LIVE: a resolved profile whose
# plan_mode is "window" makes the judge see an ORDERED FRAME WINDOW (``judge(frames=[...])``, oldest→
# newest, the anchor frame passed separately) instead of one still frame. That is the validated win
# (dynamic-goal recall rose five points, and to ~nine-in-ten with gate-relax; see
# percept/profiles_default.py::H0_PROVENANCE for the measured numbers): the eye hands the model TIME,
# not a montage. No number lives here -- this is a plan SHAPE, not a threshold.

#: Plan modes ranked by how much time the judge is given. The profile's plan_mode is a FLOOR, never a
#: downgrade — a concern that AUTHORED a richer spec keeps it (the brain's explicit HOW outranks a
#: context default), and a profile can only raise a cheaper spec up to its own rung.
#:
#: ``event_capsule`` sits ABOVE ``window`` (rank 3): it is a composite superset shape (pre-roll +
#: episode boundary + early snapshot, see :mod:`percept.capsule`), never a subset of what a plain
#: window gives the judge. Giving it the top rank means the existing floor comparison already does
#: the right thing for it with no extra branching — an authored ``event_capsule`` spec's rank always
#: beats any profile's rank, so it is never floored/downgraded by an active plan_mode. The reverse
#: direction (a profile raising a cheaper spec UP TO capsule) never happens: ``event_capsule`` is
#: deliberately absent from :data:`percept.watchprofile.PLAN_MODES`, the vocabulary a resolved
#: ``WatchProfile.plan_mode`` is validated against, so ``mode`` below can never legitimately be
#: ``"event_capsule"`` — no profile may default/upgrade a concern TO it yet. ``_planned_perception``
#: also refuses it explicitly as a target, so this holds even for a caller that bypasses profile
#: validation and passes the raw string directly.
_PLAN_RANK: dict[str, int] = {"single": 0, "strip": 1, "window": 2, "event_capsule": 3}

#: Deprecated plan modes -> their replacement. ``sprite`` folded the context frames into
#: ONE montage image to save VLM tokens; an ordered window beats it unconditionally (see
#: ``percept/profile_resolution.py``: "sprite is strictly worse and must never be selected"), so the plan
#: path no longer builds a montage. ``percept.sprite.make_sprite`` itself is untouched and still importable
#: — only the PLAN path is deprecated, and it degrades to the strictly better mode rather than failing.
_DEPRECATED_PLAN_MODES: dict[str, str] = {"sprite": "window"}

#: Frames a profile-raised strip/window asks for when the authored spec never sized one (same floor the
#: escalation ladder uses, so "the profile widened the plan" and "the ladder widened the plan" agree).
_PLAN_MIN_FRAMES = 3


def _normalize_plan_mode(mode: str) -> str:
    """Map a DEPRECATED plan mode onto its replacement, warning once per call site. A live
    mode is returned unchanged — this is a no-op on every supported plan."""
    replacement = _DEPRECATED_PLAN_MODES.get(mode)
    if replacement is None:
        return mode
    warnings.warn(
        f"plan_mode/temporal {mode!r} is deprecated and now behaves as {replacement!r}: an ordered "
        "frame window beats a sprite montage unconditionally on the dynamic-goal eval set, so the plan "
        "path no longer builds a montage. Set the plan mode to 'window' explicitly.",
        DeprecationWarning,
        stacklevel=3,
    )
    return replacement


def _planned_perception(base, plan_mode: str | None):
    """Overlay the RESOLVED ``WatchProfile.plan_mode`` onto a concern's base ``PerceptionSpec``.

    Two things happen here, in one place:

    1. the deprecation shim — a ``sprite`` plan (authored on the spec OR carried on the profile) is
       normalized to ``window``;
    2. the profile FLOOR — ``plan_mode`` raises a cheaper spec to its rung and never lowers a richer
       authored one (``_PLAN_RANK``). An authored ``event_capsule`` spec ranks above every profile
       plan_mode, so this floor can never silently downgrade it back to a plain strip/window — it
       always survives, whichever of the four profile plan_modes is active.

    ``plan_mode=None`` (no profile, or a profile-less construction) returns the base spec unchanged, so
    the legacy path stays byte-for-byte. An unrecognized mode is ignored rather than trusted — the
    vocabulary lives in :data:`percept.behavior._VALID_TEMPORAL`, not in a string a caller invented.
    ``event_capsule`` itself is refused as an escalation TARGET even though it has a rank: no profile
    may raise a cheaper spec up to it yet (it is deliberately absent from
    :data:`percept.watchprofile.PLAN_MODES`), only an authored spec may start there."""
    spec = base
    temporal = _normalize_plan_mode(spec.temporal)
    if temporal != spec.temporal:  # authored sprite -> window (shim), before any ranking
        spec = replace(spec, temporal=temporal)
    if not plan_mode:
        return spec
    mode = _normalize_plan_mode(str(plan_mode))
    if mode not in _PLAN_RANK or mode == "event_capsule":
        return spec  # unknown/unsupported plan mode, or a profile trying to select capsule: refused
    if _PLAN_RANK[mode] <= _PLAN_RANK.get(spec.temporal, 0):
        return spec  # the authored spec already sees at least this much time — never downgrade
    n_frames = spec.n_frames if spec.n_frames >= _PLAN_MIN_FRAMES else _PLAN_MIN_FRAMES
    return replace(spec, temporal=mode, n_frames=n_frames)


def _escalated_perception(base, level: int):
    """Apply the escalation ladder to a base ``PerceptionSpec``: rung 1 adds a temporal strip, rung 2
    adds fused audio. Never downgrades an already-richer authored spec (only escalates the cheap parts)."""
    if level <= 0:
        return base
    spec = base
    if spec.temporal == "single":
        spec = replace(spec, temporal="strip", n_frames=max(3, spec.n_frames))
    if level >= 2 and spec.modality == "vision":
        spec = replace(spec, modality="fused", audio_s=spec.audio_s or 2.0)
    return spec


def _trace_reason(truth: str, action: str, gate_reason: str | None = None) -> str:
    """Human-readable WHY for ``gate.classified``.

    The typed ``gate_reason`` comes from the actual :class:`GateOutcome`; truth/action provide the
    compatibility fallback for paths that did not evaluate the gate. This is observational only.
    """
    if action == FIRE:
        return f"fire: {gate_reason or 'rising_edge'}"
    if action == "ask":
        return f"ask: {gate_reason or 'low_confidence_or_stale'}"
    # silent
    if gate_reason is not None:
        label = "verdict_not_satisfied" if gate_reason == "not_satisfied" else gate_reason
        return f"silent: {label}"
    if truth == "no":
        return "silent: verdict_not_satisfied"
    if truth == "yes":
        return "silent: already_fired_or_not_rising"
    return "silent: suppressed"


def _concat_audio(chunks: list) -> Any:
    """Concatenate audio chunks into one clip for ``judge(audio=...)``.

    The AudioWindow the runtime pump fills (``runtime/audio.py``'s ``open_audio_source`` — mic,
    wav, AND video-container/ffmpeg demux) holds TYPED :class:`~percept.contracts.AudioFrame` chunks,
    not raw bytes. This function used to only recognize ``(mime, bytes)`` pairs and raw ``bytes`` —
    an ``AudioFrame`` chunk matched NEITHER case and fell through to the ``chunks[-1]`` opaque-descriptor
    branch, silently handing the judge just the LAST chunk (often a short partial final read) instead of
    the concatenated window. Handled here as its own case: concatenate every frame's raw ``.pcm`` bytes
    into one buffer and return ONE typed ``AudioFrame`` carrying the TRUE ``sample_rate``/``channels``/
    ``sample_format`` the frames themselves were captured with (read off the frames, never hardcoded —
    they already carry the real demux/mic rate from ``runtime/audio.py``'s capture config).

    An already-ENCODED frame (``.encoded`` set, ``.pcm`` empty — a pre-recorded clip carried as a
    WAV/Opus blob rather than raw PCM) has nothing to concatenate; it is passed straight through
    unmodified, preserving prior behaviour for that shape.

    Non-``AudioFrame`` chunks keep the original two cases byte-for-byte: ``(mime, bytes)`` pairs or raw
    ``bytes``/``bytearray`` concatenate; any other opaque descriptor passes the latest chunk through.
    """
    if not chunks:
        return None
    first = chunks[0]
    if isinstance(first, AudioFrame):
        if not all(isinstance(c, AudioFrame) for c in chunks):
            return None
        pcm_frames = [c for c in chunks if c.pcm]
        if not pcm_frames:
            return chunks[-1]  # encoded-only (or empty) frames — nothing to concat, pass the latest through
        if len(pcm_frames) != len(chunks):
            return None  # never flatten a mixed encoded/raw sequence
        formats = {
            (c.sample_rate, c.channels, c.sample_format)
            for c in pcm_frames
        }
        if len(formats) != 1:
            return None  # split/resample belongs upstream; never mislabel mixed PCM as the first format
        bytes_per_sample = {"s16le": 2, "f32le": 4, "u8": 1}
        if any(
            len(c.pcm) % (bytes_per_sample[c.sample_format] * c.channels)
            for c in pcm_frames
        ):
            return None
        head, tail = pcm_frames[0], pcm_frames[-1]
        return AudioFrame(
            pcm=b"".join(bytes(c.pcm) for c in pcm_frames),
            sample_rate=head.sample_rate,
            channels=head.channels,
            sample_format=head.sample_format,
            t_start=head.t_start,
            t_end=tail.t_end,
        )
    if isinstance(first, tuple) and len(first) == 2 and isinstance(first[1], (bytes, bytearray)):
        if not all(
            isinstance(c, tuple)
            and len(c) == 2
            and isinstance(c[1], (bytes, bytearray))
            and str(c[0]) == str(first[0])
            for c in chunks
        ):
            return None
        return (str(first[0]), b"".join(bytes(c[1]) for c in chunks))
    if isinstance(first, (bytes, bytearray)):
        if not all(isinstance(c, (bytes, bytearray)) for c in chunks):
            return None
        return b"".join(bytes(c) for c in chunks)
    return chunks[-1]  # opaque descriptor — pass the latest through


def _is_valid_verdict(obj: Any) -> bool:
    """Shape-validate a judge response BEFORE any caller dereferences ``.confidence``/``.satisfied``
    (ADR-001 §10 invariant 4). The backend contract is :class:`Verdict`; anything else — a
    plain/garbled object (malformed), or a ``Verdict``-shaped-but-incomplete response with the wrong
    field types (partial/truncated JSON) — must be rejected here, never dereferenced blindly."""
    return (
        isinstance(obj, Verdict)
        and isinstance(obj.satisfied, bool)
        and isinstance(obj.confidence, (int, float))
    )


_GATE_HEALTH_LOG_MAX = 256
"""Growth bound: the most queued gate-suppression entries :attr:`Percept._gate_health_log`
holds between drains. The runtime loop drains it EVERY tick, so this is never reached in a real
watch; it exists so a core driven WITHOUT ``percept.runtime.worker.run`` (a bare ``perceive_judged``
harness, which never drains) cannot grow the queue without bound over a 24/7 run. Oldest entries are
shed first, and every suppression is already emitted as its own structured ``gate.suppressed`` log
line by :meth:`percept.gate.Gate._note_suppressed`, so shedding never loses the fact itself."""


CONSECUTIVE_DEGRADE_ESCALATE = 5
"""Tunable policy knob: the number of CONSECUTIVE degraded judge ticks (raised /
malformed / partial -- anything :class:`_JudgeDegraded`) for the SAME watch key before the
fail-closed boundary escalates from a per-tick ``"judge_degraded"`` health event to a louder,
ONE-TIME ``"judge_persistent_failure"`` event. Escalating never halts the run (degrade+continue is
the chosen fail-closed policy) -- it only makes an otherwise-silent, persistently-failing watch
LOUD. The streak for a key resets to zero the instant that SAME key produces any well-formed
verdict again (see :meth:`Percept._note_judge_ok`)."""


class _JudgeDegraded(Exception):
    """Internal-only fail-closed signal: the judge call raised (timeout / provider error)
    or returned a malformed/partial response. Raised exclusively by
    :meth:`Percept._safe_timed_judge` and caught by every caller of :meth:`Percept._judge`
    (``perceive_judged``, ``_observe_and_log``) — it must never escape :class:`Percept` itself.
    Carries the ORIGINAL failure (the raised exception, or the bad object returned) for logging only;
    it never reaches ``percept.gate.classify`` or the gate."""

    def __init__(self, cause: Any) -> None:
        self.cause = cause
        super().__init__(f"judge call failed closed: {cause!r}")


class JudgeBindingInvariantError(RuntimeError):
    """Fail-closed dispatch invariant (ADR-009): a goal was MARKED judge-requesting (a
    per-goal judge binding was expected) but no binding is present at dispatch time.

    Rather than silently fall back to the deployment-default judge — which would run a goal that
    explicitly requested a specialist judge against the wrong evaluator — the dispatch path raises
    this and spends ZERO judge calls. Defined LOCALLY here (a ``RuntimeError`` subclass) so this
    module stays self-contained: the parallel ``percept.judge_registry`` build declares the same
    name for the merged wave, and the merge gate must reconcile the two to one home.
    """


class CompiledWatchSnapshotUnsupportedError(RuntimeError):
    """A compiled Watch carries session-local executable state that OSS snapshots cannot restore."""


def _judge_binding_trace(binding: Any) -> dict:
    """The additive judge-provenance keys a trace event carries when (and ONLY when) a per-goal judge
    binding is in force: ``judge_resolved`` (the resolved judge ref), ``judge_context_sha256``
    (the bound static-context digest, or ``None``), and ``fallback_used`` (always ``False`` — this path
    is reached only when the requested judge bound exactly).

    Stamped from the BINDING, never from the adapter's returned Verdict, so a judge can never set or
    override its own provenance. Returns ``{}`` when ``binding`` is ``None`` — every trace event emitted
    without a binding keeps its exact prior shape.
    """
    if binding is None:
        return {}
    return {
        "judge_resolved": binding.spec.resolved_ref,
        "judge_context_sha256": binding.spec.context_sha256,
        "fallback_used": False,
    }


class DeterministicAuditor:
    """Decides which cache HITs to re-perceive (false-serve audit) — **no RNG** (rule 2).

    False-serve is the cache's analog of false-fire: re-perceive an
    ``audit_rate`` fraction of HITs through real vision and compare; a divergence is a
    **false-serve** and the fresh read wins. The decision is a deterministic *counter* so the L1
    lane is reproducible: with ``audit_rate = r`` we audit whenever the running HIT count crosses
    the next ``1/r`` boundary (a stratified 1-in-N, not a coin flip). ``r <= 0`` never audits;
    ``r >= 1`` audits every HIT. Inject a custom predicate for other policies.
    """

    def __init__(self, audit_rate: float) -> None:
        self._rate = audit_rate
        self._hits = 0
        self._audited = 0

    def __call__(self, key: object) -> bool:
        if self._rate <= 0.0:
            self._hits += 1
            return False
        if self._rate >= 1.0:
            self._hits += 1
            self._audited += 1
            return True
        self._hits += 1
        # audit when the ideal audited-count (hits*rate) has advanced past what we've done.
        if self._hits * self._rate >= self._audited + 1:
            self._audited += 1
            return True
        return False


def _resolve(seam: str, value: Any, cfg: Settings):
    """Accept either a backend *name* (resolved via the registry) or an adapter instance."""
    if isinstance(value, str):
        return registry.make(seam, value, cfg)
    return value  # already an adapter (handy for tests / custom wiring)


def _is_scripted_backend(obj: Any) -> bool:
    """True iff ``obj`` is one of Percept's in-core scripted doubles (``percept.backends.fake``).

    A real vendor adapter (``percept.backends.anthropic`` …) OR any custom/test double counts as
    NOT scripted. The point is to catch a REAL pipeline silently paired with a scripted double — not
    to classify third-party wiring — so identity is by the module (the class is the FACT), never a
    heuristic on the requested name (a registry miss / plugin / stray ``PERCEPT_*_BACKEND`` can land
    you on a scripted double after you asked for real; the string is the request, the class the fact).
    """
    return type(obj).__module__ == "percept.backends.fake"


# One announced vocabulary: "offline" (the mode) selects "scripted" (the backends).
_SCRIPTED_SEAMS = ("vision", "specialist", "conductor", "stt", "tts")
_BANNER_EMITTED = False  # once-per-process guard for the loud stderr banner

#: How many of ONE frame's admitted watch goals may hold a judge call at the same time when no
#: adapter declares a capacity of its own (see ``Percept._judge_batch_slots``).
#:
#: WHY A BOUND AT ALL. A frame admits N goals and each spends its own judge call. Awaiting them one
#: after another makes the frame's makespan N x the per-call latency, which IS the batch period and
#: therefore IS the onset->incident latency (the runtime's judge scheduler holds a single in-flight
#: slot, so nothing newer is admitted until the whole batch resolves). Unbounded fan-out is the
#: opposite failure: N simultaneous provider requests off one key.
#:
#: WHY 8. Measured single-call service time on the shipped deliberate route is p50 1.653s / p95
#: 2.283s (lognormal mean ~1.69s). At 8 slots a frame sustains 8 / 1.69 ~ 4.7 judge calls/s (~285
#: requests per minute against one credential) — inside what a single un-negotiated provider key is
#: normally granted — while a 73-goal frame falls from 73 serial calls to ceil(73/8) = 10 waves.
#: Measured offline against that distribution (tests/test_judge_batch_concurrency.py --measure):
#: N=73 costs 126.8s at 1 slot, 16.6s at 8, 7.4s at 20, 2.5s at 73.
#:
#: WHAT IT IS NOT. It is not a target: pooling turns a SUM into a MAX, and the MAX of N draws from
#: the latency distribution sits near its N/(N+1) quantile, so a large frame is floored by the
#: distribution's TAIL no matter how many slots it gets. Raising this number buys waves, never the
#: floor. A deployment that has negotiated real provider concurrency declares it instead — either by
#: passing ``judge_concurrency=`` or by declaring ``JudgeCapability.max_inflight`` on its routes —
#: and the smaller of the two always wins.
_DEFAULT_JUDGE_CONCURRENCY = 8

#: Returned by :meth:`Percept._judge_slot` for a goal that reached the front of the concurrency
#: queue only AFTER the frame went stale. It is the batched analogue of the sequential loop's
#: ``if _stale(): break`` — the call is simply never spent, so a goals-revision change still cannot
#: be charged for work nobody will commit. The apply pass breaks on its own ``_stale()`` check
#: before it can observe this value; the identity check there is a belt, not the mechanism.
_BATCH_STALE = object()

#: "no gate key has governed this scene occurrence yet" — distinct from ``None``, which is itself a
#: real gate key (the non-entity scope a concern gets when nothing is in focus).
_NO_SCENE_KEY = object()


def _scripted_seams(core: Any) -> list[str]:
    """The seams whose CONSTRUCTED backend is a scripted double, in a stable order."""
    return [s for s in _SCRIPTED_SEAMS if _is_scripted_backend(getattr(core, s, None))]


def _emit_banner(mode: str, scripted: list[str]) -> None:
    """The loud stderr banner — once per process, suppressed under pytest (895 banners self-defeat)."""
    global _BANNER_EMITTED
    import os
    import sys

    if _BANNER_EMITTED or "pytest" in sys.modules or os.environ.get("PERCEPT_NO_BANNER"):
        return
    _BANNER_EMITTED = True
    header = "OFFLINE" if mode == "offline" else "SCRIPTED BACKENDS"
    sys.stderr.write(
        f"[percept] ⚠ {header} — scripted backends: {', '.join(scripted)}.\n"
        f"          This proves WIRING, not intelligence. A scripted conductor authors NO goals.\n"
    )


def guard_live_language_path(core: Any, *, allow_scripted_brain: bool = False) -> None:
    """FATAL seam: refuse to wire a LIVE voice/language path behind a SCRIPTED brain.

    The conductor is the language-authoring seam — it turns an utterance into ``goal_ops``. A scripted
    conductor's default ``Decision(route="ignore")`` silently discards every spoken "watch/tell me
    when…", so the user wired a voice that can't author. That is the demo bug; refuse it at the
    language↔conductor crossing (``on_speech_endpoint`` / ``arm_from_language``) and at the live-voice
    wiring points ``run(on_utterance=…)`` / ``session.start(on_utterance=…)`` — NOT at ``create()``, which
    cannot yet know a language path is coming. (An app that wires a voice ENGINE — see
    ``templates/realtime_voice_watch_agent`` — calls this itself at ``PerceptBridge.attach``.)

    Skipped when the whole run is declared scripted (``mode="offline"`` — everything is announced
    scripted, including the voice) or when the ugly-on-purpose ``allow_scripted_brain`` hatch is set.
    """
    if allow_scripted_brain or getattr(core, "_allow_scripted_brain", False):
        return
    if getattr(core, "_mode", "live") == "offline":
        return
    conductor = getattr(core, "conductor", None)
    if conductor is not None and _is_scripted_backend(conductor):
        raise ModeError(
            "scripted brain behind a live voice: a language path (on_utterance / an attached "
            "transport) is wired, but the conductor is SCRIPTED — it authors NO goals, so every "
            "spoken 'watch X' / 'tell me when…' is silently dropped. Fix: select a real conductor "
            "(conductor='claude'/'gemini' or PERCEPT_CONDUCTOR_BACKEND=…) with its API key; or, to "
            "test the voice plumbing only, run mode='offline' or pass allow_scripted_brain=True."
        )


# Backwards-compatible alias: this seam was named ``_guard_live_language_path`` before it was made a
# public template import. The underscore name is kept so existing internal callers and
# ``from percept.agent import _guard_live_language_path`` sites keep working; new code imports the
# public ``guard_live_language_path``.
_guard_live_language_path = guard_live_language_path


@dataclass
class Percept:
    """The cognition core: backends behind the seams + the memory structures + the gate."""

    vision: Any
    specialist: Any
    conductor: Any
    stt: Any
    tts: Any
    settings: Settings
    metrics: Metrics
    graph: EntityGraph
    gate: Gate
    goals: GoalRegistry
    cache: CacheCascade
    clock: Callable[[], float] = field(default=time.monotonic)
    # Which cache HITs to re-perceive for false-serve auditing. Deterministic by default
    # (no RNG on the L1 path); injectable for custom policies. ``None`` -> never audit.
    audit: Callable[[object], bool] | None = None
    router: RouteEngine | None = None  # local fast-path; None -> always defer (built in create)
    on_trace: "Callable[[Any], None] | None" = None  # opt-in per-frame decision tracer; None -> no-op
    episodes: dict = field(default_factory=dict)  # goal_id -> EpisodeLog (counting goals; also
    # reused by _document_episode_fold for a content-key watch goal, and for the scene-scoped
    # occurrence of a per-entity-identity (camera) watch goal — a goal is never more than one of
    # these types, so the map is shared safely)
    # The per-goal high-water mark of the last ``t`` GOVERNED into a content-key concern's episode
    # log (see ``_document_episode_fold``). Session-only, deliberately absent from
    # snapshot()/restore() like ``episodes`` itself.
    _doc_episode_hwm: dict = field(default_factory=dict)  # goal_id -> highest t governed so far
    # The OPEN scene occurrence of a per-entity-identity (camera) watch goal, so a confident-false —
    # the same proven occurrence boundary that re-arms the gate — can close it and let the next
    # occurrence open a fresh episode. Absent from the map = no occurrence is currently open.
    _scene_episode_id: dict = field(default_factory=dict)  # goal_id -> open episode id
    # The gate key that scene occurrence is currently being GOVERNED under. One key per goal, never a
    # per-entity map: it exists so the fold can tell "the same key re-entering" (which the episode
    # owns) from "a different entity's key" (which the gate's per-entity edge owns, unchanged).
    _scene_episode_key: dict = field(default_factory=dict)  # goal_id -> gate key
    # CLAIM MODE: the per-goal claim-slot twins of the document-mode fields above.
    # ``_claim_states`` holds the authoritative :class:`~percept.claims.ClaimState` (value + its own
    # revision ledger) per claim-slot goal; ``_claim_hwm`` is the same out-of-order high-water mark as
    # ``_doc_episode_hwm``; ``_claim_episode_id`` remembers the ONE persistent episode a claim-slot
    # concern keeps in the shared ``episodes`` map (a fixed subject_key == goal.id, unlike a
    # content-key concern's many per-document episodes), so :meth:`episode_current` can look it up
    # without reaching into ``EpisodeLog`` internals.
    _claim_states: dict = field(default_factory=dict)  # goal_id -> ClaimState
    _claim_hwm: dict = field(default_factory=dict)  # goal_id -> highest t governed so far
    _claim_episode_id: dict = field(default_factory=dict)  # goal_id -> episode id
    edge_counts: dict = field(default_factory=dict)  # goal_id -> running tally of edge detector boundaries
    rhythm: dict = field(default_factory=dict)  # goal_id -> RhythmTracker (the SELECTED subject's; brain count)
    _rhythm_multi: dict = field(default_factory=dict)  # (goal_id, entity_id) -> RhythmTracker (per-person streams)
    windows: dict = field(default_factory=dict)  # source -> FrameWindow (temporal perception)
    audio_windows: dict = field(default_factory=dict)  # source -> AudioWindow (fused/acoustic perception)
    # Source-continuity generation per runtime source. Async judgments capture this value at launch
    # and re-check it before committing cache/gate/executor state. A capture gap advances the
    # generation so cancellation-resistant stale work cannot spend the next rising edge.
    _source_continuity_generations: dict = field(default_factory=dict)
    perception_cooldown_s: float = 5.0  # the perception-rung refractory (shorter than the LLM's 30s)
    event_log: Any = None  # EventLog — temporal dedup of detected events (built in create)
    scheduler: Any = None  # Scheduler — time-triggered proactive fires (built in create)
    last_supported: dict = field(default_factory=dict)  # goal_id -> t of last CONFIDENT observation
    last_satisfied: dict = field(default_factory=dict)  # goal_id -> t of last confident-YES (deadline)
    consent: ConsentProfile = field(default_factory=ConsentProfile)  # the act-modality gate
    action_log: ActionLog = field(default_factory=ActionLog)  # audit: every fire + act
    _executor: Any = None  # BehaviorExecutor — the unified firing path (lazily bound to self.gate)
    # The per-key monotonic evidence counter (percept.sequencing.EvidenceSequencer), stamped at
    # judge-DISPATCH time so every JudgmentEvent carries the same ``evidence_seq`` a GateDecision
    # would. Lazily built and LRU-bounded (``_SEQ_MAX_KEYS``) exactly like the gate's own key maps, so
    # a 24/7 multi-entity watch cannot leak through it. Observability only — nothing branches on it.
    _sequencer: Any = None
    otel: OTelExporter | None = None  # optional OTel mirror for the funnel; None -> in-process only
    _funnel: Any = None  # Funnel — the standard telemetry funnel (lazily bound to self.metrics)
    # The announced-scripted state. Set by create(): the resolved mode ("offline"/"live"),
    # the seams whose CONSTRUCTED backend is a scripted double, and the ugly escape hatch. Read by
    # health() (the single scripted-state payload) and by the language-path FATAL seam. NOT construction inputs the
    # caller passes as dataclass fields — create() fills them.
    _mode: str = "live"
    _scripted: tuple = ()
    _allow_scripted_brain: bool = False
    # -- fail-closed boundary: consecutive-degradation breaker + the health-event queue
    # worker.py's run() loop drains into its on_health(RuntimeHealthEvent) channel every tick (the
    # SAME pattern as _JudgeScheduler.drain_trace() -- Percept queues, the runtime drains + emits;
    # Percept itself never calls on_health directly, that channel belongs to worker.py). Keyed by
    # the SAME ``(goal_id, entity_id)`` key the gate/executor already use ("the reducer key").
    _judge_degrade_streak: dict = field(default_factory=dict)  # key -> consecutive degraded-tick count
    _judge_health_log: list = field(default_factory=list)  # [(kind, key, cause), ...] queued for drain
    # -- per-goal judge binding (fail-closed dispatch). ``_judge_bindings`` maps a goal/concern
    # id -> a BoundJudge-shaped object (``.spec`` ResolvedJudgeSpec, ``.adapter`` the judge to call,
    # ``.static_context`` a ResolvedJudgeContext | None) registered by watchplan_apply's COMMIT phase and
    # released on goal removal/re-arm. ``_judge_requesting_goal_ids`` is the set of goals that MUST carry a
    # binding: a goal in this set with NO table entry is an invariant violation and the dispatch path
    # raises ``JudgeBindingInvariantError`` rather than falling back to ``self.vision``. Both empty (the
    # default) => every judge call routes to ``self.vision``, byte-for-byte the behavior before per-goal
    # bindings existed.
    _judge_bindings: dict = field(default_factory=dict)  # goal_id -> BoundJudge-shaped binding
    _judge_requesting_goal_ids: set = field(default_factory=set)  # goal ids that MUST have a binding
    _judge_binding_tokens: dict = field(default_factory=dict)  # goal_id -> identity of applied binding
    # Exact compiled audio/fused dispatch. Unlike the general judge registry, these bindings retain
    # the SDK-owned class function captured at Watch preparation and never dynamically re-read
    # ``self.vision`` / ``adapter.judge`` after arming.
    _audio_judge_bindings: dict = field(default_factory=dict)
    _audio_judge_binding_tokens: dict = field(default_factory=dict)
    # RuntimeCompiler output bound per programmable Watch.  Values are ``(CompiledWatch,
    # SourceProfile | None)`` tuples registered atomically with the goal by ``apply_watch_plan``.
    # Keeping the already-compiled object keyed by goal lets a shared core execute different
    # admission, temporal, episode, and source-profile choices without last-arm-wins mutation.
    _watch_runtime_bindings: dict = field(default_factory=dict)
    # Executable identity paired with ``_watch_runtime_bindings``. The apply ledger consults this
    # before treating an old receipt as an idempotent hit, so replacing goal ``g`` with a different
    # compiled Watch does not make a later re-apply of the first plan a false no-op.
    _watch_runtime_keys: dict = field(default_factory=dict)
    # -- consequence-gate GA: the SAME queue-here / drain-in-worker pattern for GATE SUPPRESSION.
    # ``Gate`` calls ``_note_gate_suppressed`` for every decision whose reason is in
    # ``percept.gate.SUPPRESSION_REASONS`` (start_anchored / refractory) -- i.e. a fire the baseline
    # rising-edge gate WOULD have emitted, withheld by an opt-in precision guard. Both guards are off
    # by default, so a default-configured core never appends here at all. worker.py's run() loop drains
    # this into ``on_health(RuntimeHealthEvent("gate_suppressed"))`` every tick; Percept never calls
    # on_health itself. Bounded (see _GATE_HEALTH_LOG_MAX) so a core driven WITHOUT the runtime loop
    # (a bare perceive_judged harness, which never drains) cannot leak on a 24/7 watch.
    _gate_health_log: list = field(default_factory=list)  # [(key, reason, at), ...] queued for drain
    # -- episode-memory reset: the SAME queue-here / drain-in-worker pattern. snapshot() does
    # NOT carry ``episodes`` (the per-goal EpisodeLog map — session-only for v1.4, ADR-006),
    # so ``restore()`` cannot rehydrate grouping/counting memory: after a redeploy every counting goal
    # restarts at zero episodes. That is a documented decision, but a SILENT one — the known failure
    # mode is an operator under-counting after a restart with nothing telling them why. restore() queues
    # ONE ``episode_memory_reset`` entry here when it rehydrates active counting goals with an empty
    # episode map; worker.py's run() drains it into ``on_health(RuntimeHealthEvent("episode_memory_reset"))``
    # right after load_state, so the amnesia is SEEN. Percept never calls on_health itself.
    _restore_health_log: list = field(default_factory=list)  # [(kind, data), ...] queued for drain
    # -- WatchProfile knobs applied at CORE CREATION time (never a mid-run mutation) ----------------
    # The resolved fovea cadence a WatchProfile's ``cadence`` carries (see percept.profile_wiring).
    # ``None`` (the default -- no profile, or a profile-less ``settings=`` construction) means no
    # override: byte-for-byte the behavior before profile resolution existed. Set once, in create(); nothing in this class mutates it
    # afterward -- a live cadence change is a NEW core, not a field write on this one.
    sense_policy: Any = None  # SensePolicy | None -- see percept.sampling.SensePolicy
    # The resolved WatchProfile.plan_mode ("single"/"strip"/"window"; "sprite" is deprecated and shimmed
    # to "window"). LIVE: _planned_perception() floors every concern's PerceptionSpec with it on
    # the judging path, so plan_mode="window" makes the judge see an ordered frame window
    # (``judge(frames=[...])``). ``None`` -- no profile override, byte-for-byte the behavior before profile resolution existed.
    plan_mode: str | None = None
    # The optional EvidenceExtractor seam (percept.extraction.EvidenceExtractor). ``None`` (the
    # default) -- perceive_judged calls it for nothing and every judge call is byte-for-byte
    # unchanged. When set, perceive_judged calls .extract(condition, frame, roi) for each active
    # watch goal right before judging, merges the resulting Enrichment.facts into
    # context["extractor"] (a reserved sub-dict -- see percept.extraction.CONTEXT_KEY), and emits
    # an "extractor.enriched" trace stage. GATE-IRRELEVANT: the gate still reads only the judge's
    # own satisfied/confidence.
    extractor: Any = None
    # The optional DirectJudge routing layer (percept.direct_judge.DirectJudge). ``None`` (the
    # default) -- perceive_judged routes nothing and every judge call is byte-for-byte unchanged.
    # When set, it is consulted right before the judge call: CONFIDENT skips the judge entirely
    # (zero vision calls -- its normalized Verdict is used as-is, and the UNCHANGED Gate still
    # classifies it); UNCERTAIN merges its facts into context["extractor"] and the judge is still
    # called; DEGRADED (or no rule matched) falls through to today's judge call, untouched.
    direct_judge: Any = None
    # The RESOLVED SourceProfile object (percept.sourceprofile.SourceProfile), when construction went
    # through the ADR-008 configuration boundary and a profile was actually named — ``None`` (the
    # default; a bare create() / a profile-less configure()) is byte-for-byte the behavior before this
    # field existed: nothing reads it. Populated by :meth:`configure` (the ``source_profile=`` kwarg it
    # resolves) and by :func:`percept.watchplan_apply.apply_watch_plan` (the same kwarg, applied to an
    # EXISTING core). Never mutated elsewhere. Threaded into ``count_event_decision`` /
    # ``_document_episode_fold`` / ``_claim_episode_fold`` as the FALLBACK underneath the existing
    # ``EPISODE_POLICIES[...]`` defaults — an authored ``goal.state`` (explicit episode policy /
    # ``identity_policy``) always wins; see ``percept.identity_policy.select_identity_policy`` and
    # ``percept.sourceprofile.SourceProfile.default_episode_policy``.
    source_profile: Any = None
    # How many of ONE frame's admitted watch goals may hold a judge call at the same time.
    # ``None`` (the default) means :data:`_DEFAULT_JUDGE_CONCURRENCY`. Whatever this resolves to, it
    # is then floored by every capacity the batch's adapters DECLARE (``JudgePool.max_inflight`` /
    # any adapter exposing ``max_inflight``) — see :meth:`_judge_batch_slots`. A declared capacity of
    # 1 therefore keeps the historical strictly-sequential batch, which is the correct reading of
    # that declaration: the pool has no queue, so exceeding it does not slow calls down, it fails
    # them closed on the ``"capacity"`` outcome.
    judge_concurrency: int | None = None

    # -- construction --------------------------------------------------------
    @classmethod
    def create(
        cls,
        *,
        vision: Any = None,
        specialist: Any = None,
        conductor: Any = None,
        stt: Any = None,
        tts: Any = None,
        extractor: Any = None,
        direct_judge: Any = None,
        settings: Settings | None = None,
        clock: Callable[[], float] | None = None,
        discover_plugins: bool = True,
        audit: Callable[[object], bool] | None = None,
        identity: Any = None,
        embedder: Callable[[Any, Box], Any] | None = None,
        on_trace: Callable[[Any], None] | None = None,
        on_health: Callable[[str], None] | None = None,
        mode: str | None = None,
        allow_scripted_brain: bool = False,
        sense_policy: Any = None,
        plan_mode: str | None = None,
        source_profile: Any = None,
        judge_concurrency: int | None = None,
    ) -> "Percept":
        cfg = settings or load_settings()
        # The mode is the switch: explicit kwarg > PERCEPT_MODE env > live. It is never "offline" by
        # NOT typing something — you fall onto scripted backends only by naming the mode or selecting a
        # scripted seam, which is the whole point (faking must be a choice you make and see).
        resolved_mode = resolve_mode(mode)
        offline = resolved_mode == "offline"
        # Per-seam resolution: explicit kwarg > settings.models.* (env/ModelSettings) > mode default.
        # vision/conductor/specialist now default None (config.py); stt/tts still default "fake". An
        # unset seam becomes "scripted" offline. In LIVE mode an unset VISION is the bare-create() trap
        # (no real perception, no key) -> ModeError; conductor/specialist fall back to scripted and are
        # announced LOUDLY below (the code-armed watch is a legitimate pattern, not a crash).
        def _pick(name: str, kwarg: Any, cfg_value: Any, *, required: bool,
                  scripted_default: str = "scripted") -> Any:
            sel = kwarg if kwarg is not None else cfg_value
            if sel is not None:
                return sel
            if offline:
                return scripted_default
            if required:
                raise ModeError(
                    f"no {name} backend selected and this is a LIVE run — Percept will not silently "
                    f"fall back to a scripted double. Fix: pass mode='offline' for keyless scripted "
                    f"backends (proves WIRING, not intelligence), or select a real backend "
                    f"({name}='claude'/'gemini') and set its API key."
                )
            return scripted_default

        vision = _pick("vision", vision, cfg.models.vision, required=True)
        specialist = _pick("specialist", specialist, cfg.models.specialist, required=False)
        # The conductor is the language-authoring seam: an UNSET one defaults to the HONEST scripted double
        # (ScriptedConductor — no-match -> route="respond_direct"), NEVER FakeConductor whose default
        # route="ignore" is the one value indistinguishable from a real conductor declining. So
        # `mode="offline"` — the door people explore through — no longer hands back a brain that discards
        # every utterance and looks like it worked.
        conductor = _pick("conductor", conductor, cfg.models.conductor, required=False,
                          scripted_default="scripted-conductor")
        stt = cfg.models.stt if stt is None else stt
        tts = cfg.models.tts if tts is None else tts
        # LIVE credential check: a SELECTED real backend name whose key is absent raises ModeError
        # naming the exact env var(s) — never construct-then-fail-silently at first call. Per-backend
        # (not hardcoded to one vendor). Skipped offline (a declared real seam offline needs no key —
        # it is not going to make a live call). Adapter instances the caller wired directly are skipped.
        if not offline:
            for seam_name, sel in (("vision", vision), ("specialist", specialist),
                                   ("conductor", conductor), ("stt", stt), ("tts", tts)):
                keys = missing_credential(sel)
                if keys:
                    raise ModeError(
                        f"{seam_name} backend {sel!r} needs a credential and none is set. Fix: export "
                        f"one of {list(keys)}, or run mode='offline' (keyless scripted backends)."
                    )
        if discover_plugins:
            registry.discover_entrypoints()
        clk = clock or time.monotonic
        # Cache ships a contract, not a policy (invariant 6): the cascade is EMPTY (always DEFERs)
        # unless the FrameHashLayer is explicitly enabled via settings — then it sits in front of
        # the VLM. The false-serve auditor defaults to deterministic (no RNG on the L1 path).
        layers = (
            [FrameHashLayer(cfg.cache, clock=clk)] if cfg.cache.enabled else []
        )
        auditor = audit if audit is not None else DeterministicAuditor(cfg.cache.audit_rate)
        obj = cls(
            vision=_resolve("vision", vision, cfg),
            specialist=_resolve("specialist", specialist, cfg),
            conductor=_resolve("conductor", conductor, cfg),
            stt=_resolve("stt", stt, cfg),
            tts=_resolve("tts", tts, cfg),
            extractor=extractor,  # optional; None -> perceive_judged calls nothing, byte-for-byte
            direct_judge=direct_judge,  # optional; None -> every judge call is byte-for-byte
            settings=cfg,
            metrics=Metrics(),
            graph=EntityGraph(
                cfg.entity.iou_match,
                cfg.entity.max_entities,
                clock=clk,
                identity=identity,
                embedder=embedder,
            ),
            gate=Gate(
                threshold=cfg.gate.threshold, clock=clk,
                require_false_first=cfg.gate.require_false_first,
                refractory_s=cfg.gate.refractory_s or None,
            ),
            goals=GoalRegistry(),
            cache=CacheCascade(layers=layers, threshold=cfg.cache.threshold),
            clock=clk,
            audit=auditor,
            on_trace=on_trace,
            router=RouteEngine.default(threshold=cfg.routing.confidence_threshold),
            event_log=EventLog(cfg.events.merge_gap_s),
            scheduler=Scheduler(clock=clk),
            consent=ConsentProfile(
                allow_computer_use=cfg.consent.allow_computer_use,
                allowed_actions=cfg.consent.allowed_actions,
                actuation=getattr(cfg.consent, "actuation", "none"),  # typed envelope declaration
            ),
            action_log=ActionLog(),
            _mode=resolved_mode,
            _allow_scripted_brain=allow_scripted_brain,
            sense_policy=sense_policy,
            plan_mode=plan_mode,
            source_profile=source_profile,
            judge_concurrency=judge_concurrency,
        )
        # GoalRegistry is a public mutation surface. Bind runtime cleanup to its authoritative
        # operation hook so direct ``core.goals.create/apply`` calls obey the same executable-state
        # rules as ``Percept.add_goal/apply_goal_ops``.
        obj.goals._bind_owner_mutation_hook(obj._on_goal_mutations)
        # When the entity graph evicts a track (it caps at max_entities), sweep that entity's derived
        # state — its ``(goal_id, entity_id)`` gate keys AND any cached verdicts resting on it — so a 24/7
        # churn of entities can't leave orphan edges accumulating in the gate. Wired here (not at graph
        # construction) because the gate is built alongside the graph in the same call.
        obj.graph.on_evict = obj._on_entity_evicted
        # Gate SUPPRESSION (a fire withheld by require_false_first / refractory_s) must be visible,
        # not a silent non-event. Wired here for the same reason as on_evict above — the gate is built
        # in this same call, and the hook needs the finished ``obj`` to queue onto. Queue only; the
        # runtime loop drains it into on_health (this class never calls on_health itself).
        obj.gate.on_suppress = obj._note_gate_suppressed
        # Announce scripted backends LOUDLY at construction, in the one consistent vocabulary.
        # This decides on the CONSTRUCTED CLASS (the fact), not the requested name — a registry miss /
        # plugin / stray PERCEPT_*_BACKEND can land a scripted double after you asked for real.
        scripted = _scripted_seams(obj)
        obj._scripted = tuple(scripted)
        if scripted:
            log = get_logger("percept.agent")
            conductor_scripted = _is_scripted_backend(obj.conductor)
            # create() can only LOG — it CANNOT know whether a language path (on_utterance, or an
            # app-side voice engine) will be wired later. So the FATAL "scripted brain behind a live
            # voice" check lives at the language↔conductor crossing (+ the live-voice wiring points);
            # here we only warn.
            if conductor_scripted:
                # The common, LEGITIMATE code-armed pattern: goals via add_goal(), no voice. Non-fatal.
                log.warning(
                    "scripted_conductor: the conductor is SCRIPTED — language goal-authoring is OFF; "
                    "goals must be code-armed via add_goal(). A spoken 'tell me when…' will be ignored. "
                    "(scripted seams: %s; mode=%s)", ", ".join(scripted), resolved_mode,
                )
                if on_health is not None:
                    on_health("scripted_conductor")
            else:
                log.warning(
                    "scripted_backends: %s are SCRIPTED — this proves WIRING, not intelligence. "
                    "(mode=%s)", ", ".join(scripted), resolved_mode,
                )
            _emit_banner(resolved_mode, scripted)
        return obj

    # -- ADR-008 configuration boundary: thin entry points onto create() ----
    @classmethod
    def configure(
        cls,
        *,
        source_profile: Any = None,
        source_profile_explicit: bool = True,
        domain_pack: Any = None,
        execution_policy: Any = None,
        optimization_lock: Any = None,
        situation: "tuple[Any, Any, Any] | None" = None,
        registry: Any = None,
        settings: Settings | None = None,
        resolver: Any = None,
        **create_kwargs: Any,
    ) -> "Percept":
        """Build a core through the ADR-008 configuration boundary — resolve the six-object
        precedence chain (:class:`percept.resolver.ConfigResolver`), compile it to real runtime
        primitives (:func:`percept.runtime_compiler.runtime_compile`), and hand the result to
        :meth:`create` (unchanged). No goal is armed here — arm one afterward via
        :meth:`add_goal` or :func:`percept.watchplan_apply.apply_watch_plan`.

        This is an ADDITIVE door onto :meth:`create`, not a replacement: a bare ``Percept.create()``
        remains byte-for-byte the pre-ADR-008 path. Every keyword this method does not itself
        consume (``vision=``, ``mode=``, ...) passes straight through to :meth:`create` in
        ``create_kwargs``; an explicit ``create_kwargs['settings']``/``['sense_policy']``/
        ``['plan_mode']`` wins over what resolution produced (the caller building the core is always
        the most specific author).
        """
        from .contracts.watchplan import WatchPlan
        from .resolver import ConfigResolver
        from .runtime_compiler import runtime_compile

        # ConfigResolver.resolve requires a WatchPlan (its condition is never armed — configure()
        # resolves DEPLOYMENT/profile knobs only, no goal). The condition text is discarded; this
        # placeholder is not read by anything downstream of runtime_compile's Settings/sense_policy/
        # plan_mode fields.
        placeholder = WatchPlan(condition="percept.configure — deployment/profile resolution only")
        active_resolver = resolver if resolver is not None else ConfigResolver()
        resolved = active_resolver.resolve(
            placeholder,
            source_profile=source_profile,
            source_profile_explicit=source_profile_explicit,
            domain_pack=domain_pack,
            execution_policy=execution_policy,
            optimization_lock=optimization_lock,
            situation=situation,
            registry=registry,
            settings=settings,
        )
        compiled = runtime_compile(resolved)
        create_kwargs.setdefault("settings", compiled.settings)
        create_kwargs.setdefault("sense_policy", compiled.sense_policy)
        create_kwargs.setdefault("plan_mode", compiled.plan_mode)
        # The RESOLVED SourceProfile OBJECT (not ``resolved.source_profile``, which is a plain JSON-
        # shaped snapshot dict per ``ResolvedWatch`` — see its docstring) — the actual object this
        # method was called with, so ``Percept.source_profile`` carries something with
        # ``.default_identity_policy`` / ``.default_episode_policy`` on it, not a dict. ``None`` when
        # this ``configure()`` call named no profile, byte-for-byte no field at all.
        create_kwargs.setdefault("source_profile", source_profile)
        return cls.create(**create_kwargs)

    @classmethod
    def from_bundle(cls, bundle: "dict[str, Any]") -> "Percept":
        """Build a core from a plain mapping of the SAME keywords :meth:`configure` accepts — e.g. a
        deployment bundle loaded from JSON/YAML. Thin: ``Percept.from_bundle(b)`` is exactly
        ``Percept.configure(**b)``, nothing more."""
        return cls.configure(**dict(bundle))

    # -- goal authoring ------------------------------------------------------
    def add_goal(self, goal: Goal, *, _cache_preinvalidated: bool = False) -> Goal:
        if self.goals.get(goal.id) is not None:
            raise ValueError(
                f"goal id {goal.id!r} already exists; same-id replacement is unsupported because "
                "gate, executor, episode, detector, and tier-0 state cannot yet be reset as one unit. "
                "Use a new goal id."
            )
        declared_identity = identity_policy_of(goal)
        if (
            goal.state.get("rearm") == "each_event"
            and declared_identity in {"content_key", "claim_slot"}
        ):
            raise ValueError(
                "rearm='each_event' owns occurrence episode identity and cannot be combined "
                f"with explicit identity_policy={declared_identity!r}"
            )
        # PROVENANCE: the STRUCTURED entry point stamps STRUCTURED_TRUSTED, OVERWRITING
        # whatever the caller supplied — a caller cannot claim interrupt-trust by pre-filling ``origin``.
        # ``replace`` (not in-place set) so this is correct even if Goal is later frozen.
        goal = replace(goal, origin=GoalOrigin.STRUCTURED_TRUSTED)
        self._arm_deadline(goal)  # resolve an authored duration into an absolute stamp at ARM time
        # Fail before mutating the registry or bindings. ``apply_watch_plan`` preflights a whole batch
        # and sets the private flag so a cache failure cannot leave half the batch armed.
        if not _cache_preinvalidated:
            self.invalidate_cache(goal.id)
        # Defensive cleanup for a fresh id; GoalRegistry itself enforces lifetime-unique ids, so this
        # never acts as a same-id replacement path.
        self._release_judge_binding(goal.id)
        self._release_audio_judge_binding(goal.id)
        self._release_watch_runtime(goal.id)
        result = self.goals.create(goal)
        return result

    # -- per-goal judge binding: register/release/select ---------------------------------------------
    def _register_judge_binding(self, goal_id: str, bound: Any) -> None:
        """Bind ``goal_id`` to a BoundJudge-shaped judge (``.spec``/``.adapter``/``.static_context``).

        Called by :func:`percept.watchplan_apply.apply_watch_plan`'s COMMIT phase, AFTER every goal in a
        batch prepared cleanly. Marks the goal judge-requesting so a later dispatch that finds the binding
        gone fails closed (:class:`JudgeBindingInvariantError`) instead of silently using ``self.vision``.
        """
        self._judge_requesting_goal_ids.add(goal_id)
        self._judge_bindings[goal_id] = bound
        self._judge_binding_tokens[goal_id] = id(bound)

    def _release_judge_binding(self, goal_id: str) -> None:
        """Drop ``goal_id``'s judge binding and its judge-requesting mark (goal removal / re-arm).

        Idempotent — releasing an unbound goal is a no-op, so a plain (judge-less) goal's lifecycle is
        byte-for-byte unchanged.
        """
        self._judge_bindings.pop(goal_id, None)
        self._judge_binding_tokens.pop(goal_id, None)
        self._judge_requesting_goal_ids.discard(goal_id)

    def _register_audio_judge_binding(self, goal_id: str, bound: Any) -> None:
        """Commit one immutable compiled audio/fused dispatch binding."""
        self._audio_judge_bindings[goal_id] = bound
        self._audio_judge_binding_tokens[goal_id] = bound.binding_token

    def _release_audio_judge_binding(self, goal_id: str) -> None:
        """Release a compiled audio dispatch token with its Watch lifecycle."""
        self._audio_judge_bindings.pop(goal_id, None)
        self._audio_judge_binding_tokens.pop(goal_id, None)

    def _register_watch_runtime(
        self,
        goal_id: str,
        compiled: Any,
        source_profile: Any = None,
        application_key: str | None = None,
    ) -> None:
        """Bind one compiled Watch to its armed goal without mutating core-wide defaults."""
        self._watch_runtime_bindings[goal_id] = (compiled, source_profile)
        if application_key is not None:
            self._watch_runtime_keys[goal_id] = application_key

    def _release_watch_runtime(self, goal_id: str) -> None:
        """Release an applied Watch's compiled binding and invalidate its apply-ledger hits."""
        self._watch_runtime_bindings.pop(goal_id, None)
        self._invalidate_watch_application(goal_id)

    def _invalidate_watch_application(self, goal_id: str) -> None:
        """Invalidate apply idempotency while preserving a live compiled runtime binding."""
        self._watch_runtime_keys.pop(goal_id, None)
        ledger = getattr(self, "_watchplan_applied_hashes", None)
        if isinstance(ledger, dict):
            for key, value in list(ledger.items()):
                if value and value[0] == goal_id:
                    ledger.pop(key, None)

    def _check_armed_routes(self, ops: "tuple[GoalOp, ...]") -> None:
        """THE arm-time behavior-route gate — one check on every path that arms a Goal.

        ``percept`` declares eight behavior routes; the frame worker drives four. The other four used
        to arm in total silence: the goal went ``active``, no frame ever selected it, no incident ever
        landed, and nothing said why. That is the worst failure an SDK can have, and it became
        reachable in practice once :mod:`percept.conductor` started authoring real goals to
        ``type="count"``.

        This runs from :meth:`_on_goal_mutations`, the ``GoalRegistry`` owner hook, which is invoked
        by ``GoalRegistry.apply`` for the whole batch BEFORE any mutation. That is deliberately the
        one seam rather than three: :meth:`add_goal` (structured), :meth:`apply_goal_ops` (the
        conductor's language path) and a direct ``core.goals.create``/``apply`` all funnel through it,
        so no arm can slip past, and a raise here aborts the batch atomically with nothing armed.

        Raise vs. warn is decided by :func:`~percept.behavior_routes.classify_armed_routes`, not here:

        * ``UNSUPPORTED`` — nothing implements it and no host API can feed it, so the goal is dead on
          arrival. **Raise.**
        * a host-driven external route (``count``/``event``/``remind``/``scheduler``) — the driver
          exists and ships (:meth:`perceive_boundary`, :meth:`push_frame` + :meth:`perceive_events`,
          :meth:`schedule_every`/:meth:`schedule_at` + :meth:`tick`); what is missing is the host
          WIRING. Every one of those APIs resolves the goal BY ID, so refusing the arm would make them
          unreachable — it would delete three shipped capabilities to fix a message. **Warn loudly**,
          name the exact call, and let a deployment that wants it fatal filter
          :class:`~percept.behavior_routes.UndrivenBehaviorRouteWarning` to ``"error"``.

        Note ``percept.watchplan_apply.apply_watch_plan`` still REJECTS these routes one layer up: its
        contract is "arm a Watch that ``percept.run`` drives", and it fails in PREPARE, before this
        hook ever sees the goal. That is unchanged — only its message now names the same remedy.
        """
        armed = [op.goal for op in ops if op.op == "create" and op.goal is not None]
        if not armed:
            return
        undrivable, host_driven = classify_armed_routes(armed)
        if undrivable:
            raise UnsupportedBehaviorRouteError(
                undrivable, driver="Percept.add_goal/apply_goal_ops", verb="arm"
            )
        for item in host_driven:
            message = undriven_route_message(item)
            # Both channels on purpose: warnings.warn reaches an interactive/test caller (and is what
            # a strict deployment escalates to an error); the logger reaches a 24/7 deployment where
            # nobody is watching stderr, which is exactly where a silent no-op used to live.
            warnings.warn(message, UndrivenBehaviorRouteWarning, stacklevel=2)
            get_logger("percept.agent").warning("%s", message)
            self.metrics.incr(f"goal.armed.undriven_route.{item.route.value}")

    def _on_goal_mutations(self, ops: tuple[GoalOp, ...]) -> None:
        """Keep executable state coherent for every GoalRegistry mutation path."""
        # Fail/flag the batch's ARM routes before any per-op state work below mutates anything.
        self._check_armed_routes(ops)
        for op in ops:
            goal_id = op.goal.id if op.op == "create" and op.goal is not None else op.goal_id
            if goal_id is None:
                continue
            if op.op == "update":
                # Registry validation proves this is a presentation-only ``say`` update. Keep the
                # compiled runtime and judge binding, but invalidate the old plan's exact receipt.
                self._invalidate_watch_application(goal_id)
            elif op.op in ("complete", "cancel"):
                self._release_judge_binding(goal_id)
                self._release_audio_judge_binding(goal_id)
                self._release_watch_runtime(goal_id)

    def _watch_runtime_for(self, goal: "Goal | str") -> Any:
        goal_id = goal if isinstance(goal, str) else goal.id
        return self._watch_runtime_bindings.get(goal_id)

    def _settings_for_goal(self, goal: Goal) -> Any:
        bound = self._watch_runtime_for(goal)
        return bound[0].settings if bound is not None else self.settings

    def _source_profile_for_goal(self, goal: Goal) -> Any:
        bound = self._watch_runtime_for(goal)
        if bound is not None and bound[1] is not None:
            return bound[1]
        return self.source_profile

    def _plan_mode_for_goal(self, goal: Goal) -> str | None:
        bound = self._watch_runtime_for(goal)
        if bound is not None:
            return bound[0].plan_mode
        return self.plan_mode

    def _select_judge_adapter(self, goal: Goal) -> tuple[Any, Any]:
        """Return ``(adapter, bound_or_None)`` for ``goal``'s judge call — the EXACT fail-closed rule:

        * a binding exists -> ``(bound.adapter, bound)`` (route to the bound judge);
        * no binding but the goal is MARKED judge-requesting -> raise :class:`JudgeBindingInvariantError`
          (fail closed; NO adapter is called, so zero judge calls are spent);
        * neither -> ``(self.vision, None)`` (the deployment default, byte-for-byte unchanged).

        Raised OUTSIDE the ``_safe_timed_judge`` boundary (see :meth:`_judge`) so the invariant error
        propagates rather than degrading to ASK/SILENT like a live judge fault.
        """
        goal_id = getattr(goal, "id", None)
        bound = self._judge_bindings.get(goal_id)
        if bound is not None:
            return bound.adapter, bound
        if goal_id in self._judge_requesting_goal_ids:
            raise JudgeBindingInvariantError(
                f"goal {goal_id!r} is marked judge-requesting but has no bound judge; refusing to fall "
                f"back to the deployment-default judge (fail-closed dispatch, ADR-009)"
            )
        return self.vision, None

    def _audio_judge_binding_for(self, goal: Goal) -> Any:
        """Return the intact immutable binding required by a compiled audio/fused Watch."""
        binding = self._audio_judge_bindings.get(goal.id)
        runtime = self._watch_runtime_for(goal)
        modality = (
            program_of(goal).perception.modality if runtime is not None else "vision"
        )
        if modality not in ("audio", "fused"):
            return None
        if (
            binding is None
            or self._audio_judge_binding_tokens.get(goal.id) is not binding.binding_token
        ):
            raise JudgeBindingInvariantError(
                f"compiled {modality} Watch {goal.id!r} has no intact immutable audio judge "
                "binding; refusing dynamic adapter fallback"
            )
        return binding

    def _arm_deadline(self, goal: Goal) -> None:
        """Resolve a ``deadline`` concern's authored DURATION into an absolute ``deadline_at`` on THIS
        agent's clock, at the moment the goal is armed (one field, one meaning, one clock).

        ``deadline_after_s`` (a duration) -> ``deadline_at = clock() + deadline_after_s``. The executor
        only ever compares against ``deadline_at``, on the same timeline the worker stamps frames on
        (wall for live capture). A legacy absolute ``deadline_s`` in ``goal.state`` (an old author or a
        v1 snapshot) has NO recoverable duration base, so it is refused LOUDLY — never silently reused
        under a new meaning that would fire on the wrong frame or never fire at all."""
        st = goal.state
        if st.get("rearm") != "deadline":
            return
        if st.get("deadline_at") is not None:
            return  # already absolute (e.g. a verification concern that stamped it directly)
        if "deadline_after_s" in st:
            st["deadline_at"] = self.clock() + float(st["deadline_after_s"])
            return
        if "deadline_s" in st:
            raise ValueError(
                f"goal {goal.id!r} carries a legacy 'deadline_s' (ambiguous absolute-vs-duration, the "
                "A-2 clock bug). Re-author as 'deadline_after_s' (a duration from arm). Refusing to arm "
                "a deadline concern whose stamp would fire on the wrong timeline or never fire."
            )

    def _episode_default(
        self, goal: Goal, fallback: "EpisodePolicy | None"
    ) -> "EpisodePolicy | None":
        """The :class:`~percept.episodes.EpisodePolicy` an episode-owning call site should pass as
        ``episode_policy_of(goal, default=...)`` when the goal itself authors nothing.

        The source profile resolved for this Watch (falling back to the core's
        ``self.source_profile``) wins over
        the call site's own hardcoded ``fallback`` (``EPISODE_POLICIES['document_presence']`` /
        ``['conversational_claim']`` / ``None``) — a source class's declared continuation semantics
        (:attr:`~percept.sourceprofile.SourceProfile.default_episode_policy`) are more specific than
        the generic per-mode default. ``self.source_profile is None`` (a bare ``create()``, or a
        profile-less ``configure()``) returns ``fallback`` unchanged — byte-for-byte today's behavior.
        Explicit ``goal.state['episode']`` authoring always wins regardless (handled inside
        :func:`~percept.episodes.episode_policy_of` itself, upstream of this default).
        """
        source_profile = self._source_profile_for_goal(goal)
        if source_profile is not None:
            return source_profile.default_episode_policy
        return fallback

    def _resolved_identity_policy(self, goal: Goal) -> str | None:
        """The identity policy THIS concern is running under, for the document/claim episode-fold
        gates below. Concern-declared (``goal.state['identity_policy']``) always wins; with no
        resolved source profile this is exactly :func:`~percept.identity_policy.identity_policy_of`
        — ``None`` when the goal declared nothing, byte-for-byte today's behavior (an undeclared goal
        never enters document/claim episode-fold mode on a profile-less core). When a profile WAS
        resolved, an undeclared goal instead inherits the profile's capability via
        :func:`~percept.identity_policy.select_identity_policy` (``source_profile.identity_strategy``)
        — the ADR-005 "profile declares, concern selects" shape this call site never actually threaded
        before this wiring.
        """
        source_profile = self._source_profile_for_goal(goal)
        if source_profile is not None:
            return select_identity_policy(goal, source_profile)
        return identity_policy_of(goal)

    # -- episode counting (the cup problem) ---------------------------------
    def count_event(
        self, goal_id: str, exemplar: Any, judge: "EpisodeJudge | None" = None, *,
        t: float | None = None,
        subject_key: str | None = None,
        mutation: "MutationCheck | None" = None,
    ) -> tuple[Any, bool]:
        """Fold a candidate counting-event into the goal's episode log; sync ``goal.state['count']``.

        ``judge`` answers the SUBJECT question only — which existing episode is this observation
        about (an LLM in real use; a stub in L1 — never assert backend intelligence with a fake).
        Whether that subject's episode is CONTINUED is a separate decision, evaluated against the
        concern's :class:`~percept.episodes.EpisodePolicy`. Returns ``(episode, is_new)``;
        :meth:`count_event_decision` returns the typed reason as well.
        For goals whose ``state['type'] == 'count'`` (e.g. "how many cups today").
        """
        return self.count_event_decision(
            goal_id, exemplar, judge, t=t, subject_key=subject_key, mutation=mutation
        )[:2]

    def count_event_decision(
        self, goal_id: str, exemplar: Any, judge: "EpisodeJudge | None" = None, *,
        t: float | None = None,
        subject_key: str | None = None,
        mutation: "MutationCheck | None" = None,
    ) -> tuple[Any, bool, ContinuationDecision]:
        """:meth:`count_event` plus the typed :class:`~percept.episodes.ContinuationDecision`.

        The decision separates the three collapsed questions: ``subject_matched`` (was
        this the same subject at all), ``is_new`` + ``reason`` (did it continue, and if not WHY),
        and the mutation seam. ``subject_matched and is_new`` — a same-subject observation that
        opened a SECOND episode — is counted as ``episode.denied.<reason>`` so a client can see the
        split in metrics, not only in logs.
        """
        now = self.clock() if t is None else t
        log = self.episodes.get(goal_id)
        if log is None:
            goal = self.goals.get(goal_id)
            goal_settings = self._settings_for_goal(goal) if goal is not None else self.settings
            log = EpisodeLog(
                goal_id,
                window_s=goal_settings.episodes.window_s,
                max_episodes=goal_settings.episodes.max_episodes,
                policy=(
                    episode_policy_of(goal, default=self._episode_default(goal, None))
                    if goal is not None else None
                ),
            )
            self.episodes[goal_id] = log
        decision = log.observe_decision(
            now, exemplar, judge, subject_key=subject_key, mutation=mutation
        )
        goal = self.goals.get(goal_id)
        if goal is not None:
            goal.state["count"] = log.count(now)
        self.metrics.incr("episode.new" if decision.is_new else "episode.continuation")
        if decision.denied:
            self.metrics.incr(f"episode.denied.{decision.reason}")
        return decision.episode, decision.is_new, decision

    # -- the event track (temporal perception) ------------------------------
    def _window(self, source: str) -> FrameWindow:
        window = self.windows.get(source)
        if window is None:
            window = FrameWindow(self.settings.events.window)
            self.windows[source] = window
        return window

    def push_frame(self, frame: Any, source: str = "camera", *, t: float | None = None) -> None:
        """Add a frame to the source's rolling window (for event-only flows; perceive_judged also feeds it)."""
        self._window(source).push(self.clock() if t is None else t, frame)

    def _audio_window(self, source: str) -> AudioWindow:
        win = self.audio_windows.get(source)
        if win is None:
            win = AudioWindow(getattr(self.settings.events, "audio_window_s", 4.0))
            self.audio_windows[source] = win
        return win

    def push_audio(self, chunk: Any, source: str = "camera", *, t: float | None = None):
        """Feed captured, replayed, or application-pushed audio into one ordered source timeline.

        An :class:`~percept.events.AudioUnit` retains sequence, interval, format, participant/direction,
        and cheap-boundary metadata. Legacy opaque chunks remain supported with ``t=`` and receive a
        local sequence number. The returned ``AudioPushResult`` makes discontinuities observable.
        """
        window = self._audio_window(source)
        if isinstance(chunk, AudioUnit):
            return window.push_unit(chunk)
        from percept.runtime.audio import normalize_audio_unit

        at = self.clock() if t is None else t
        unit = normalize_audio_unit((at, chunk, None), seq=window.next_seq)
        return window.push_unit(unit)

    def _audio_clip(self, source: str, spec) -> Any:
        """Assemble one audio clip for a fused/audio ``PerceptionSpec`` from the AudioWindow, else None."""
        if spec.modality not in ("audio", "fused") or not spec.audio_s:
            return None
        return _concat_audio(self._audio_window(source).clip(spec.audio_s))

    async def perceive_events(self, source: str = "camera") -> list[FireEvent]:
        """Detect events over the rolling window, dedup them, and run event goals through the gate.

        A *new* deduped event is a rising edge -> FIRE (once, via the gate's refractory on the
        event id); a continuation is SILENT; a low-confidence event is ASK. Counting falls out as
        ``goal.state['count'] = distinct events of the type``.
        """
        window = self.windows.get(source)
        if window is None or len(window) == 0:
            return []
        detected = await self.vision.detect_events(window.frames(), self.goals.active())
        out: list[FireEvent] = []
        for ev in detected:
            distinct, _is_new = self.event_log.observe(
                ev.type, ev.t, participants=ev.participants, confidence=ev.confidence
            )
            for goal in self.goals.active():
                if goal.state.get("type") != "event" or goal.state.get("event_type") != ev.type:
                    continue
                verdict = Verdict(satisfied=True, confidence=distinct.confidence)
                ev_fire = self._fire_concern(
                    goal, verdict, (goal.id, distinct.id), metric="event.gate"
                )
                goal.state["count"] = self.event_log.count(ev.type)
                if ev_fire is not None:
                    out.append(ev_fire)
        latest = window.latest()
        self._log_fires(out, latest[0] if latest else 0.0)
        return out

    # -- the scheduler (time-triggered proactive fires) ---------------------
    def schedule_every(self, interval_s: float, goal_id: str, *, now: float | None = None):
        return self.scheduler.every(interval_s, goal_id, now=now)

    def schedule_at(self, when_s: float, goal_id: str):
        return self.scheduler.at(when_s, goal_id)

    async def tick(self, *, t: float | None = None) -> list[FireEvent]:
        """Fire due timers via the conductor (timing in the clock; content judged by the model)."""
        now = self.clock() if t is None else t
        out: list[FireEvent] = []
        for timer in self.scheduler.due(now):
            goal = self.goals.get(timer.goal_id)
            say = goal.say if goal is not None else ""
            decision = await self.conductor.run("scheduler_tick", say, {"goal_id": timer.goal_id})
            if decision.route == "proactive_speak":
                out.append(FireEvent(timer.goal_id, FIRE, decision.say or say, None, None))
        self._log_fires(out, now)
        return out

    def apply_goal_ops(self, ops: list[GoalOp]) -> None:
        # Conductor-authored creates are untrusted. Copy instead of mutating the caller's GoalOp, and
        # resolve deadline durations before the authoritative registry is allowed to commit anything.
        prepared: list[GoalOp] = []
        for op in ops:
            if op.op == "create" and op.goal is not None:
                goal = replace(
                    op.goal,
                    origin=GoalOrigin.LANGUAGE_AUTHORED,
                    state=dict(op.goal.state),
                )
                self._arm_deadline(goal)
                prepared.append(GoalOp(op="create", goal=goal))
            else:
                prepared.append(
                    GoalOp(
                        op=op.op,
                        goal=op.goal,
                        goal_id=op.goal_id,
                        fields=dict(op.fields),
                    )
                )

        # GoalRegistry is the authoritative semantic firewall and validates the WHOLE batch. Run it
        # before fallible cache work, then let its mutation hook keep compiled runtime state coherent.
        self.goals.validate(prepared)
        for op in prepared:
            if op.op in ("update", "complete", "cancel") and op.goal_id is not None:
                self.invalidate_cache(op.goal_id)
        self.goals.apply(prepared)

    # -- unified firing via the general executor ----------------------------
    @property
    def executor(self) -> BehaviorExecutor:
        """The single firing path, lazily bound to THIS agent's gate + metrics (shared state)."""
        if self._executor is None:
            self._executor = BehaviorExecutor(self.gate, self.metrics)
        return self._executor

    def _admit_evidence(self, key: tuple) -> int:
        """Reserve this judgment's ``evidence_seq`` at judge-DISPATCH time (ADR-001 §8).

        The live watch path is strictly single-flight, so nothing depends on the sequence today — it
        exists so a :class:`~percept.contracts.JudgmentEvent` emitted here correlates with the
        ``GateDecision`` a future worker-pool reducer would produce for the same key, instead of
        carrying a placeholder. LRU-bounded like the gate's own key maps.
        """
        if self._sequencer is None:
            from .sequencing import EvidenceSequencer  # noqa: PLC0415 — avoid an import cycle at module load
            self._sequencer = EvidenceSequencer()
        seq = self._sequencer.admit(key)
        tracked = self._sequencer.tracked_keys()  # first-admission order
        if len(tracked) > _SEQ_MAX_KEYS:
            for stale in tracked[: len(tracked) - _SEQ_MAX_KEYS]:
                self._sequencer.forget(stale)
        return seq

    @property
    def funnel(self) -> Funnel:
        """The standard telemetry funnel (seen→gated→judged→refused→fired), bound to THIS agent's
        metrics. The harness loop (percept.harness.tier0) AND the production runtime loop
        (percept.runtime.worker.run) both emit seen/gated via ``core.funnel``; the watch path
        here (perceive_judged) emits judged/refused/fired and the judge latency + token/cost, for
        either driver. Lazily built; OTel mirror attached when ``self.otel`` is set."""
        if self._funnel is None:
            self._funnel = Funnel(self.metrics, otel=self.otel)
        return self._funnel

    def _fire_concern(
        self,
        goal: Goal,
        verdict: Verdict,
        base_key: tuple,
        *,
        freshness: float = 1.0,
        t: float = 0.0,
        metric: str = "gate",
        timing: JudgmentTimingBuilder | None = None,
        evidence_seq: int = 0,
        fold: "EpisodeFoldResult | None" = None,
        outcome_sink: "Callable[[GateOutcome], None] | None" = None,
    ) -> FireEvent | None:
        """One firing path — delegates to the :class:`~percept.behavior.BehaviorExecutor` (the general
        ``sense·key·accumulate`` executor that now owns the per-concern re-arm key-synthesis, the
        never-contract, the hard-quadrant escalation, and the confusion substrate). This method is a
        thin boundary that adapts the executor's ``Fired`` back to the legacy ``FireEvent`` shape; the
        watch + event paths route through here, so both now run on the unified executor (gate unchanged).

        ``timing``/``evidence_seq`` are the OPTIONAL judgment-funnel stamps the caller collected
        for this unit of evidence; when present the executor freezes them into a
        :class:`~percept.contracts.JudgmentEvent` that rides out on ``FireEvent.judgment``. Callers
        that take no stamps (the event path) pass nothing and get ``judgment=None`` — additive only.

        ``fold`` (OPTIONAL) is the episode-owned idempotency decision to GOVERN this fire with (see
        ``_document_episode_fold``); ``None`` (every caller that doesn't build one) is byte-for-byte
        the gate-only path this method has always run.
        """
        fired = self.executor.fire_judged(
            goal, verdict, base_key, program_of(goal), freshness=freshness, t=t, metric=metric,
            timing=timing, evidence_seq=evidence_seq, fold=fold, outcome_sink=outcome_sink,
        )
        if fired is None:
            return None
        return self._fire_event_from_executor(fired)

    @staticmethod
    def _fire_event_from_executor(fired: Any) -> FireEvent:
        return FireEvent(fired.goal_id, fired.action, fired.text, fired.entity_id, fired.verdict,
                         fired.judgment, reason=fired.reason, modality=fired.modality,
                         roi=fired.roi, provenance=fired.provenance,
                         revision_id=fired.revision_id, supersedes=fired.supersedes)

    def _fire_clip_occurrences(
        self,
        goal: Goal,
        verdict: Verdict,
        base_key: tuple,
        *,
        source_t: float,
        context: Any,
        freshness: float,
        timing: JudgmentTimingBuilder | None,
        evidence_seq: int,
        outcome_sink: "Callable[[GateOutcome], None] | None" = None,
    ) -> list[FireEvent]:
        """Govern a provider's bounded clip-relative occurrence list."""
        if not verdict.events:
            return []
        if not isinstance(context, Mapping):
            return []
        span_s = context.get("_percept_temporal_span_s")
        if (
            isinstance(span_s, bool)
            or not isinstance(span_s, (int, float))
            or not math.isfinite(float(span_s))
            or float(span_s) <= 0.0
        ):
            return []
        clip_end_t = float(source_t)
        clip_start_t = clip_end_t - float(span_s)

        def timing_for_source(event_source_t: float):
            if timing is None:
                return None
            return JudgmentTimingBuilder(
                clock=timing.clock,
                ingest_observed_ts=timing.ingest_observed_ts,
                admitted_ts=timing.admitted_ts,
                judge_started_ts=timing.judge_started_ts,
                judge_completed_ts=timing.judge_completed_ts,
                source_media_ts=event_source_t,
            )

        return [
            self._fire_event_from_executor(fired)
            for fired in self.executor.fire_clip_events(
                goal,
                verdict,
                base_key,
                program_of(goal),
                clip_start_t=clip_start_t,
                clip_end_t=clip_end_t,
                freshness=freshness,
                timing_for_source=timing_for_source,
                evidence_seq=evidence_seq,
                outcome_sink=outcome_sink,
            )
        ]

    def perceive_boundary(self, goal_id: str, *, t: float = 0.0) -> FireEvent | None:
        """Consume one EDGE boundary (a detector's timed event — a rep peak, an audio onset) as a
        distinct occurrence and route it through the executor's nth-distinct accumulator (reusing
        :meth:`BehaviorExecutor.fire_episode`). This is the **brain-counts** half of
        edge-proposes·brain-counts·VLM-confirms: the count comes from the cheap edge signal, NOT a
        per-frame VLM call. Fires per the goal's ``accumulate.fire_when`` (e.g. "n == 10"), or on every
        boundary when none is authored — counting reps without spending a single vision token."""
        goal = self.goals.get(goal_id)
        if goal is None:
            return None
        n = self.edge_counts.get(goal_id, 0) + 1
        self.edge_counts[goal_id] = n
        fired = self.executor.fire_episode(goal, program_of(goal), n=n, is_new=True, t=t)
        if fired is None:
            return None
        self.metrics.incr("edge.boundary.fire")
        return FireEvent(fired.goal_id, fired.action, fired.text, fired.entity_id, fired.verdict)

    def _rhythm_tracker(self, goal) -> RhythmTracker:
        prior = goal.state.get("rhythm") if isinstance(goal.state.get("rhythm"), dict) else {}
        kw = {"amp_gate_frac": prior["amp_gate_frac"]} if "amp_gate_frac" in prior else {}
        if "smooth_s" in prior:  # opt-in rate-invariance smoothing (Build 2); default off keeps legacy counts
            kw["smooth_s"] = prior["smooth_s"]
        if prior.get("canonical_select"):  # opt-in rate-invariant channel selection (Build 3); default off
            kw["canonical_select"] = True
        return RhythmTracker(prior_channel=prior.get("channel"), expected_rom=prior.get("expected_rom"), **kw)

    def _fire_rep(self, goal, n: int, stop_t: float) -> FireEvent | None:
        fired = self.executor.fire_episode(goal, program_of(goal), n=n, is_new=True, t=stop_t)
        if fired is None:
            return None
        self.metrics.incr("rhythm.rep.fire")
        return FireEvent(fired.goal_id, fired.action, fired.text, fired.entity_id, fired.verdict)

    def perceive_rhythm(
        self, goal_id: str, *, t: float, channels: dict[str, float], entity_id: str | None = None
    ) -> FireEvent | None:
        """The **rhythm sense** (the brain-counts half done right): fold one frame's measured pose/motion
        CHANNELS into the goal's :class:`~percept.rhythm.RhythmTracker` and fire only when a *complete* rep
        is causally confirmed (rest→peak→return-to-rest). The count is the tracker's confirmed-rep tally —
        the eye measures, the brain decides which excursions are reps. No VLM call, no per-action tuning.

        ``channels`` is the eye's decomposed measurement vector for this frame (e.g. the ``pose-channels``
        detector's basis). A confirmed rep IS a distinct episode, so the firing reuses
        :meth:`BehaviorExecutor.fire_episode` (gate + ``fire_when``) — same path, new sense.

        ``entity_id`` makes counting PER-PERSON: each subject gets its own tracker and the brain SELECTS the
        performer (the entity with the most reps on the prior channel — a still bystander reps ~0), so the
        goal's reported count is the performer's. ``None`` (the default) is the legacy single-subject stream
        the benchmark/runtime read directly; behaviour there is unchanged."""
        goal = self.goals.get(goal_id)
        if goal is None or not channels:
            return None
        if entity_id is None:  # legacy single-subject stream — unchanged
            tracker = self.rhythm.get(goal_id)
            if tracker is None:
                tracker = self.rhythm[goal_id] = self._rhythm_tracker(goal)
            rep = tracker.observe(t, channels)
            if rep is None:
                return None
            n = tracker.count
            self.edge_counts[goal_id] = n  # the brain-owned count, read uniformly by the bench/runtime
            goal.state["count"] = n
            return self._fire_rep(goal, n, rep.stop_t)
        # per-entity: the brain selects the performer (most reps for this goal) and reports their count
        key = (goal_id, entity_id)
        tracker = self._rhythm_multi.get(key)
        if tracker is None:
            tracker = self._rhythm_multi[key] = self._rhythm_tracker(goal)
        rep = tracker.observe(t, channels)
        perf_key, perf = max(((k, tr) for k, tr in self._rhythm_multi.items() if k[0] == goal_id),
                             key=lambda kt: kt[1].count)
        self.rhythm[goal_id] = perf  # so core.rhythm[goal_id] / .count read the SELECTED subject uniformly
        n = perf.count
        self.edge_counts[goal_id] = n
        goal.state["count"] = n
        goal.state["subject"] = perf_key[1]  # who the brain is counting (the selected performer entity)
        if rep is None or key != perf_key:  # fire only when THE PERFORMER confirms a rep this frame
            return None
        return self._fire_rep(goal, n, rep.stop_t)

    # -- observability: coverage-loss fires with the concern's consequence ----
    def check_observability(
        self,
        *,
        t: float | None = None,
        staleness_s: float | None = None,
        goal_ids: "tuple[str, ...] | None" = None,
    ) -> list[FireEvent]:
        """Fire when a high-consequence concern LOSES perceptual coverage.

        Observability falls out of belief-from-observations: a concern whose last *confident*
        observation is older than ``staleness_s`` has gone blind. Losing coverage of a high-miss-cost
        concern INHERITS its consequence — "I can't see the baby" fires with the baby's urgency. This
        is perception-ROBUST: it fires on the ABSENCE of evidence, never by trusting a verdict.

        De-duped via the UNCHANGED gate: coverage-loss is a ``(C, R)`` fire (C = lost; R = on_false)
        on a dedicated ``__coverage__`` key, so it fires ONCE per loss episode and re-arms when
        coverage returns. Only ``miss_cost == 'severe'`` concerns inherit it (default mild -> silent,
        byte-for-byte). ``last is None`` (never yet observed) is not a loss — acquisition, not blindness.
        """
        now = self.clock() if t is None else t
        bound = staleness_s if staleness_s is not None else self.settings.gate.validity_window_s
        out: list[FireEvent] = []
        selected = None if goal_ids is None else frozenset(goal_ids)
        for goal in self.goals.active():
            if selected is not None and goal.id not in selected:
                continue
            if goal.state.get("miss_cost", "mild") != "severe":
                continue
            last = self.last_supported.get(goal.id)
            lost = last is not None and (now - last) > bound
            # observability is a derived FACT (we KNOW we haven't seen it), not a perceptual guess
            verdict = Verdict(satisfied=lost, confidence=1.0, evidence="coverage")
            # per-goal singleton -> exempt from the R-1 TTL sweep (a dropped edge would re-fire coverage-loss)
            action = self.gate.evaluate((goal.id, "__coverage__"), verdict, no_evict=True)
            self.metrics.incr(f"coverage.gate.{action}")
            if action == FIRE:
                out.append(FireEvent(goal.id, FIRE, self._coverage_text(goal), None, verdict))
        self._log_fires(out, now)
        return out

    def _coverage_text(self, goal: Goal) -> str:
        return f"I've lost sight of what I'm watching: {goal.condition}"

    def _escalate_text(self, goal: Goal) -> str:
        return f"High stakes either way — confirm before I act: {goal.condition}?"

    def attention_order(self) -> list[Goal]:
        """Attention allocation: order active concerns by consequence stakes so a
        budget-limited perception loop spends on the highest-consequence concerns first. Pure stakes,
        no confidence — a calibration-free job."""
        return sorted(
            self.goals.active(),
            key=lambda g: attention_weight(consequence_of(g)),
            reverse=True,
        )

    # -- absence-native firing: an expected concern overdue -----------------
    def check_deadlines(
        self, *, t: float | None = None, goal_ids: "tuple[str, ...] | None" = None
    ) -> list[FireEvent]:
        """An EXPECTED concern not satisfied by its deadline fires "overdue".

        Absence IS the firing — there is no positive event to perceive, so this is perception-robust
        by construction: it fires on the LACK of a confident-yes, never on a guess. Rides the existing
        tick/scheduler loop and feeds a synthetic verdict to the UNCHANGED gate on a dedicated
        ``__overdue__`` key -> fires ONCE per overdue episode, re-arms if the expected event arrives.
        Only ``rearm == 'deadline'`` concerns (with a resolved ``deadline_at``) participate -> byte-for-byte.
        """
        now = self.clock() if t is None else t
        out: list[FireEvent] = []
        selected = None if goal_ids is None else frozenset(goal_ids)
        for goal in self.goals.active():
            if selected is not None and goal.id not in selected:
                continue
            if goal.state.get("rearm") != "deadline":
                continue
            # absence firing (synthetic overdue verdict on the __overdue__ key) lives in the executor;
            # satisfaction is read from the agent's last_satisfied stamp.
            fired = self.executor.fire_absence(
                goal, program_of(goal), now=now, satisfied=goal.id in self.last_satisfied
            )
            if fired is not None:
                out.append(
                    FireEvent(fired.goal_id, fired.action, fired.text, fired.entity_id, fired.verdict)
                )
        self._log_fires(out, now)
        return out

    # -- Item 4: the episode boundary fires the consolidated process answer -------------------
    async def check_episode_boundary(
        self,
        *,
        t: float | None = None,
        inactivity_s: float | None = None,
        goal_ids: "tuple[str, ...] | None" = None,
    ) -> list[FireEvent]:
        """A process activity that has gone quiet (no NEW step for ``inactivity_s``) ENDED — summarize the
        logged deltas, emit the answer, and wipe. Cognition-owned, perception-robust (fires on the
        ABSENCE of new steps, never a guess). Mirrors :meth:`check_observability`: a synthetic verdict
        through the UNCHANGED gate on a dedicated ``__episode__`` key -> fires ONCE per episode and re-arms
        when the next step is logged. Only ``type=='process'`` goals participate -> byte-for-byte.
        Run on the worker's cadence (it must see QUIET frames — the boundary IS an absence of activity)."""
        now = self.clock() if t is None else t
        bound = inactivity_s if inactivity_s is not None else self.settings.gate.validity_window_s
        out: list[FireEvent] = []
        selected = None if goal_ids is None else frozenset(goal_ids)
        for goal in self.goals.active():
            if selected is not None and goal.id not in selected:
                continue
            gtype = goal.state.get("type")
            ctx = self.executor.ctx(goal.id)
            ended = (
                bool(ctx.process_log)
                and ctx.last_step_at != float("-inf")
                and (now - ctx.last_step_at) > bound
            )
            if gtype == "process":
                verdict = Verdict(satisfied=ended, confidence=1.0, evidence="episode")
                action = self.gate.evaluate((goal.id, "__episode__"), verdict, no_evict=True)  # singleton
                self.metrics.incr(f"episode.gate.{action}")
                if action == FIRE:
                    answer = await self.summarize_process(goal.id, wipe=True)  # consolidate + reset
                    v = Verdict(satisfied=True, confidence=1.0, evidence=answer, running_answer=answer)
                    out.append(FireEvent(goal.id, FIRE, answer or self._episode_text(goal), None, v))
                continue
            # process-QA (mvp2) watch goal: a quiet episode end RESOLVES whatever was confidently
            # accumulated, with the coverage floor relaxed by one (``at_boundary``) — but still never
            # resolving an empty/uncertain answer. Same answer-path resolver as the in-tick path.
            if goal.state.get("process_qa") and ended:
                resolved = self._resolve_process_qa(goal, ctx, 1.0, at_boundary=True)
                if resolved is not None:
                    ctx.process_log = []  # end the episode → wipe accumulated coverage, re-arm
                    out.append(resolved)
        self._log_fires(out, now)
        return out

    def _episode_text(self, goal: Goal) -> str:
        return f"Here's what I saw for: {goal.condition}"

    def _overdue_text(self, goal: Goal) -> str:
        return f"Still waiting on: {goal.condition}"

    # -- the watch path ------------------------------------------------------
    async def _emit_trace(self, trace_id: str, t: float, stage: str, *, goal_id: str | None = None,
                          **data: Any) -> None:
        """Invoke ``on_trace`` safely: await an async callback, and never let a bad tracer break the
        loop (observability must not crash perception). Caller guards on ``self.on_trace is not None``
        first so no trace kwargs are built on the default (untraced) hot path."""
        cb = self.on_trace
        if cb is None:
            return
        try:
            result = cb(TraceEvent(trace_id=trace_id, stage=stage, t=t, goal_id=goal_id, data=data))
            if inspect.isawaitable(result):
                await result
        except Exception:  # noqa: BLE001 — a tracer is observability; swallow its failures
            pass

    def _shadow_observe(self, verdict: Any) -> "ShadowObservation":
        """SHADOW: the resolved subject + the proposed :class:`ShadowObservation` for a
        watch-path verdict. ADVISORY ONLY — it OBSERVES what the fold would propose and governs
        nothing (no gate call, no re-arm, no suppression). The watch path binds no continuation
        decision, so ``decision=None`` (an ``unbound`` fold, or ``unknown`` when the judge saw a
        subject it could not identify); the subject is the :class:`~percept.contracts.SubjectRef`
        the judge emitted, if any. Never raises — a bad verdict yields an unbound shadow."""
        subject = getattr(verdict, "subject", None)
        if not isinstance(subject, SubjectRef):
            subject = None
        unknown = subject is not None and subject.resolution == "unknown"
        operation = getattr(verdict, "operation", None)
        # An unknown subject carries no key: it is a visible abstention, never a bound identity that
        # could drive a fold, so drop it to None on the ref carrier and mark the fold unknown.
        subject_ref = subject if (subject is not None and subject.resolution == "bound") else None
        return self.executor.shadow_observe(
            None, subject=subject_ref, operation=operation, unknown=unknown
        )

    def _event_chain_advisory(
        self,
        goal: Goal,
        entity_id: Hashable,
        verdict: Verdict,
    ) -> dict[str, Any]:
        """Inspectable occurrence state for an explicit ``each_event`` Watch."""
        if goal.state.get("rearm") != "each_event":
            return {}
        state = self.executor.ctx(goal.id).occurrences.get(entity_id)
        return {
            "event_phase": getattr(verdict, "phase", "") or None,
            "event_action_present": getattr(verdict, "action_present", None),
            "event_episode_id": state.episode_id if state is not None else None,
            "event_occurrence_sequence": state.sequence if state is not None else 0,
            "event_episode_emitted": state.emitted if state is not None else False,
            "event_pre_since": state.pre_since if state is not None else None,
            "event_close_quiet_s": float(
                program_of(goal).perception.window_s or 0.0
            ),
        }

    @staticmethod
    def _event_chain_provider_advisory(
        goal: Goal,
        verdict: Verdict,
    ) -> dict[str, Any]:
        """Raw structured provider tuple, present only for explicit event Watches.

        Ordinary Watch traces remain byte-for-byte unchanged; the matched-watch reconstruction
        contract intentionally treats extra judge fields as a different receipt.
        """
        if goal.state.get("rearm") != "each_event":
            return {}
        return {
            "phase": getattr(verdict, "phase", "") or None,
            "action_present": getattr(verdict, "action_present", None),
        }

    def _document_episode_fold(
        self,
        goal: Goal,
        verdict: Verdict,
        t: float,
        *,
        truth: "str | None" = None,
        key: "tuple | None" = None,
    ) -> "EpisodeFoldResult | None":
        """Episode grouping + :class:`~percept.behavior.EpisodeFoldResult`-owned idempotency, GOVERNING,
        scoped to content-key concerns and to the SCENE occurrence of a per-entity-identity concern.

        The shadow-mode advisory (``_shadow_observe``) always folds ``decision=None`` — an ``unbound``
        shadow that changes nothing, for every concern. This method is the narrow exception: when a
        concern SELECTS the content-key resolver (``goal.state['identity_policy'] == 'content_key'``,
        the per-concern selector in ``identity_policy.py`` — screen/document scope, never inferred from
        the source), it maintains a REAL per-goal :class:`~percept.episodes.EpisodeLog` and folds the
        resulting :class:`~percept.episodes.ContinuationDecision` through the SAME pure
        :meth:`~percept.behavior.BehaviorExecutor.fold_episode` the shadow path already exercises — and
        the caller (:meth:`perceive_judged`) now THREADS the answer into ``_fire_concern(fold=...)`` so
        it actually governs the gate interaction (idempotent suppression + re-arm; see
        :func:`percept.behavior.episode_fold`).

        A concern that resolves to a PER-ENTITY (scene-object track) resolver instead — the shipped
        default of the physical-scene profile, i.e. every ordinary camera watch — has no content key to
        group by, so it took the ``None`` branch and got no fold at all: alone among the source classes
        it had no TTL-invariant idempotency, and a gate-key eviction mid-occurrence could re-report a
        condition that never stopped holding. It now folds on a SCENE-scoped subject
        (:meth:`_scene_episode_fold`), NOT a per-entity one — see that method for the forfeit that
        buys and the guards that keep it honest.

        Every other concern — including a claim-slot / conversational (mutation) goal — returns
        ``None`` here, so ``_fire_concern`` runs its byte-for-byte gate-only path and stays shadow-only
        via the UNTOUCHED ``_shadow_observe``/``STAGE_SHADOW_FOLD`` trace. This method never changes
        gate mechanics itself — it only decides whether a fold is built at all.

        ``truth`` (the caller's already-computed :func:`~percept.gate.classify` result) and ``key`` (the
        gate key this observation will be evaluated under) are read ONLY by the scene branch — the
        content-key path below is byte-for-byte unchanged. It needs ``truth`` to tell an occurrence that
        ENDED from one that is merely unproven, and ``key`` to leave the gate's per-entity edge alone.
        ``truth=None`` (a direct caller that took no freshness reading) is re-derived as-if fresh.

        Two things this method refuses to do:

        * **fabricate a content key** — a verdict carrying no subject at all is not grouped (returns
          ``None``, the same "nothing to fold" answer an undeclared concern gets); an ``unknown``
          subject folds through :func:`percept.behavior.episode_fold`'s own visible-abstention rule
          (never rearms, never a manufactured key).
        * **let an out-of-order observation corrupt the live log** — this per-goal log only ever
          advances in time (a high-water mark on ``t``). Async/off-loop judge dispatch can resolve
          completions out of chronological order; feeding a stale ``t`` into the log would rewind an
          episode's ``last_seen`` and could fabricate a spurious continuation/new-episode divergence on
          a LATER, in-order observation. A late arrival instead degrades this one tick to the gate-only
          path (``None``) — the same safe answer an undeclared concern always gets — and is counted
          (``episode.document.out_of_order_skipped``), never silently swallowed.
        """
        # An event-chain Watch owns occurrence identity. A source-profile default such as
        # screen_document/content_key must not silently add a second, incompatible episode fold.
        # Explicit content_key + each_event is rejected at arm time in add_goal().
        if goal.state.get("rearm") == "each_event":
            return None
        resolved_policy = self._resolved_identity_policy(goal)
        if resolved_policy in PER_ENTITY_IDENTITY_POLICIES:
            return self._scene_episode_fold(goal, verdict, t, truth=truth, key=key)
        if resolved_policy != "content_key":
            return None
        subject = getattr(verdict, "subject", None)
        if subject is None:
            return None  # nothing to group — gate-only, exactly like an undeclared concern
        hwm = self._doc_episode_hwm.get(goal.id)
        if hwm is not None and t < hwm:
            self.metrics.incr("episode.document.out_of_order_skipped")
            return None
        self._doc_episode_hwm[goal.id] = t
        log = self.episodes.get(goal.id)
        if log is None:
            goal_settings = self._settings_for_goal(goal)
            log = EpisodeLog(
                goal.id,
                window_s=goal_settings.episodes.window_s,
                max_episodes=goal_settings.episodes.max_episodes,
                policy=episode_policy_of(
                    goal,
                    default=self._episode_default(
                        goal, EPISODE_POLICIES["document_presence"]
                    ),
                ),
            )
            self.episodes[goal.id] = log
        decision = log.observe_decision(t, verdict)
        unknown = subject_is_unknown(verdict)
        return self.executor.fold_episode(
            decision, operation=getattr(verdict, "operation", None), unknown=unknown
        )

    def _scene_episode_fold(
        self,
        goal: Goal,
        verdict: Verdict,
        t: float,
        *,
        truth: "str | None" = None,
        key: "tuple | None" = None,
    ) -> "EpisodeFoldResult | None":
        """SCENE MODE — the camera twin of :meth:`_document_episode_fold`'s content-key path.

        A concern whose resolved identity policy is a per-entity, scene-object track resolver
        (:data:`~percept.identity_policy.PER_ENTITY_IDENTITY_POLICIES` — what the physical-scene profile
        declares, so what an ordinary camera watch inherits) supplies neither a content key nor a claim
        slot, so it built no fold at all and was the one source class with NO episode-owned idempotency:
        while a condition kept holding, a TTL/LRU eviction of its gate key re-created that key ARMED and
        re-reported an occurrence that never stopped. This method closes that hole by folding on a
        SCENE-scoped subject — one open occurrence per CONCERN (the same scope
        :attr:`Percept.episodes` is keyed by), named by the recognized
        :data:`~percept.identity_policy.SCENE_IDENTITY` scope — through the SAME
        :class:`~percept.episodes.EpisodeLog` and the SAME pure
        :meth:`~percept.behavior.BehaviorExecutor.fold_episode` the document and claim paths use. The
        continuation policy comes from the resolved source profile
        (:attr:`~percept.sourceprofile.SourceProfile.default_episode_policy`, via
        :meth:`_episode_default`), never from a constant invented here.

        **What scene keying forfeits, deliberately.** A scene singleton CANNOT COUNT SUBJECTS. Within one
        gate key, an occurrence that has already been reported is reported once: two people arriving
        inside the same unbroken occurrence are one incident, not two. Counting subjects is what the
        gate's PER-ENTITY key exists for, and that key is left entirely alone here — hence the
        ``key`` guard below. Per-entity EPISODES are refused on measured grounds: scene entity ids churn
        (58,101 ids in 82 minutes, p50 track lifetime 0.0s), so a per-entity episode key would mint a
        brand-new never-emitted episode almost every tick — which is a brand-new gate key defaulting to
        ARMED — and would turn the never-silent drop/deny/evict tracing into a log-line-per-tick
        firehose.

        The five guards that keep this honest:

        * **only the default rising-edge re-arm.** A scene occurrence is the mirror image of the gate's
          own on_false edge, so it governs only concerns that run that edge. ``never`` (fire-once-ever),
          ``interval`` (a cadence owns the re-arm), ``deadline`` (absence) and ``each_event`` (the
          occurrence chain already owns identity) each carry their own re-arm rule; a fold suppressing
          the gate ahead of them would silently delete their firing, so they keep the gate-only path.
        * **only a judged frame-watch route.** Counting/rhythm/event goals own
          :attr:`Percept.episodes` for their own subjects; a scene subject must never be written into
          those logs.
        * **it governs ONE gate key at a time — the gate's per-entity edge is untouched.** The scene
          episode grants idempotency for re-entries of the key it is being governed under; when the
          observation arrives on a DIFFERENT key (focus turned over to another entity) the fold declines
          to govern that tick and hands the decision back to the gate, byte-for-byte the behavior from
          before this method existed. That is what keeps "a second person is a second key, and a second
          key fires" true, while still making a TTL/LRU eviction of a STABLE key a non-event — the
          episode remembers the emission that the evicted key forgot. One key is remembered per concern,
          never a per-entity map, so the refused entity firehose stays refused.
        * **an occurrence ENDS on the same proven boundary the gate re-arms on.** A confident-false
          closes the open episode (counted, ``episode.scene.closed``) and degrades this tick to
          gate-only, so the NEXT time the condition holds it is a genuinely new episode that re-arms and
          fires. Without that boundary a scene singleton — whose subject key never changes — would fold
          as one endless continuation and the watch would fire exactly once, ever.
        * **an abstention changes nothing.** An ``unknown`` (low-confidence or stale) read neither opens,
          continues, nor closes the occurrence, and returns ``None`` — byte-for-byte the ASK path.

        The out-of-order high-water mark is the same one the content-key path keeps (a goal is only ever
        one of the two), counted separately as ``episode.scene.out_of_order_skipped``.
        """
        if goal.state.get("rearm", "on_false") != "on_false":
            return None
        if goal_behavior_route(goal) is not BehaviorRoute.FRAME_WATCH:
            return None
        if truth is None:
            # A direct caller that took no freshness reading: read the verdict as-if fresh, exactly
            # what the fused-evidence call site passes (freshness=1.0). Never a second gate call —
            # ``classify`` is the pure pre-edge mapping, not gate state.
            truth = classify(verdict, 1.0, self.gate.threshold, self.gate.fresh_min)
        if truth == "unknown":
            return None  # a refusal to guess is not an occurrence boundary and not an observation
        hwm = self._doc_episode_hwm.get(goal.id)
        if hwm is not None and t < hwm:
            self.metrics.incr("episode.scene.out_of_order_skipped")
            return None
        self._doc_episode_hwm[goal.id] = t
        log = self.episodes.get(goal.id)
        if log is None:
            goal_settings = self._settings_for_goal(goal)
            log = EpisodeLog(
                goal.id,
                window_s=goal_settings.episodes.window_s,
                max_episodes=goal_settings.episodes.max_episodes,
                # No hardcoded fallback: with no resolved profile this is the documented default
                # EpisodePolicy(), and authored ``goal.state['episode']`` still wins over both.
                policy=episode_policy_of(goal, default=self._episode_default(goal, None)),
            )
            self.episodes[goal.id] = log
        if truth == "no":
            episode_id = self._scene_episode_id.pop(goal.id, None)
            self._scene_episode_key.pop(goal.id, None)
            if episode_id is not None:
                log.close(episode_id, t=t)
                self.metrics.incr("episode.scene.closed")
            return None
        # A fixed subject_key (the scene scope): like a claim slot, this is ONE persistent thing rather
        # than many distinct subjects, so there is no identity to resolve and the "unknown subject"
        # branch is unreachable — an explicit subject_key always wins in observe_decision. Any
        # SubjectRef the judge happened to emit is deliberately not consulted: keying on it is the
        # per-entity episode this method refuses to build.
        decision = log.observe_decision(t, verdict, subject_key=SCENE_IDENTITY)
        self._scene_episode_id[goal.id] = decision.episode.id
        governed_key = self._scene_episode_key.get(goal.id, _NO_SCENE_KEY)
        self._scene_episode_key[goal.id] = key
        if governed_key is not _NO_SCENE_KEY and governed_key != key:
            # Focus turned over: this observation belongs to a gate key the scene occurrence has not
            # been speaking for. Whether a different entity's edge fires is the GATE's question and the
            # answer must not change, so hand it back untouched. The occurrence itself was still
            # advanced above — declining to govern a tick is not pretending it was unobserved.
            self.metrics.incr("episode.scene.entity_turnover_skipped")
            return None
        return self.executor.fold_episode(decision)

    def _claim_episode_fold(
        self, goal: Goal, verdict: Verdict, t: float
    ) -> "EpisodeFoldResult | None":
        """CLAIM MODE — the claim-slot twin of :meth:`_document_episode_fold`.

        Document mode flipped episode grouping + idempotency to GOVERNING for concerns that SELECT the
        content-key resolver; this method flips the SAME seam for concerns that SELECT the claim-slot
        one (``goal.state['identity_policy'] == 'claim_slot'``) — a goal is never both, so the two
        methods are mutually exclusive by construction and either may be called unconditionally by the
        caller. Every other concern (including an undeclared one) still gets ``fold=None`` here, exactly
        as it always has.

        The three pieces this wires together for claim-slot concerns:

        1. **The mutation channel is authority-gated FIRST** (``percept.claims``): the verdict's speech
           act only reaches the episode fold as its EFFECTIVE operation —
           :meth:`~percept.claims.ClaimState.apply` downgrades a ``replace``/``retract``/``close`` from
           an asserter this concern's :class:`~percept.contracts.ConcernAuthority`
           (:func:`~percept.claims.concern_authority_of`, ``goal.state['authority']``) does not permit
           to a non-mutating ``mention`` — TRACED (``claim.authority_downgrade``), never silent — so a
           seller's textbook replace can never move a buyer-authoritative value. A concern that
           authored no authority gets the unrestricted default, byte-for-byte.
        2. **The SAME episode fold document mode governs with**: the effective operation feeds
           :meth:`~percept.behavior.BehaviorExecutor.fold_episode` exactly like a content-key concern's
           verdict does, via the SAME ``episode_mutation_of`` map — so an applied ``replace``/``retract``
           re-arms and fires a superseding revision (``fold.supersedes``), and the gate/incident-delivery
           machinery (``_fire_concern(fold=...)`` → ``BehaviorExecutor.fire_judged`` →
           ``SupersessionSink``) is UNCHANGED. A downgraded (unauthorized) mutation folds as a plain
           ``confirmed`` re-statement — it never re-arms, never supersedes, never moves the value.
        3. **The corrected value + its own revision chain, on the episode** — unlike a content-key
           concern (which never carries a "value"), a claim slot's whole point is a corrected VALUE.
           Whenever the claim state machine actually APPLIES a write (``outcome.applied``), the episode's
           ``value`` is updated (:meth:`~percept.episodes.EpisodeLog.annotate`) and, for a genuine state
           change (``updated``/``retracted``/``closed`` — never the initial ``opened`` row, already
           recorded by ``observe_decision``), a :class:`~percept.episodes.Revision` is appended
           (:meth:`~percept.episodes.EpisodeLog.record_revision`) naming the value the slot became. A
           downgraded or non-mutating act writes nothing — the value and the chain stay exactly as they
           were. :meth:`episode_current` reads both back off the episode.

        Like :meth:`_document_episode_fold`, this method refuses to fabricate: a stale/out-of-order ``t``
        degrades the tick to gate-only (counted, ``episode.claim.out_of_order_skipped``, never silently
        swallowed) rather than risk rewinding the log's high-water mark.
        """
        if goal.state.get("rearm") == "each_event":
            return None
        if self._resolved_identity_policy(goal) != "claim_slot":
            return None
        hwm = self._claim_hwm.get(goal.id)
        if hwm is not None and t < hwm:
            self.metrics.incr("episode.claim.out_of_order_skipped")
            return None
        self._claim_hwm[goal.id] = t
        log = self.episodes.get(goal.id)
        if log is None:
            goal_settings = self._settings_for_goal(goal)
            log = EpisodeLog(
                goal.id,
                window_s=goal_settings.episodes.window_s,
                max_episodes=goal_settings.episodes.max_episodes,
                policy=episode_policy_of(
                    goal,
                    default=self._episode_default(
                        goal, EPISODE_POLICIES["conversational_claim"]
                    ),
                ),
            )
            self.episodes[goal.id] = log
        state = self._claim_states.get(goal.id)
        if state is None:
            state = ClaimState(claim_id=goal.id)
            self._claim_states[goal.id] = state
        authority = concern_authority_of(goal)
        outcome = state.apply(
            getattr(verdict, "operation", "assert"),
            value=getattr(verdict, "value", None),
            normalized_value=getattr(verdict, "normalized_value", None),
            asserted_by=getattr(verdict, "asserted_by", None),
            authority=authority,
        )
        # A fixed subject_key (== goal.id): a claim slot is ONE persistent thing, not many distinct
        # documents, so there is no identity to resolve here — unlike content-key grouping, this never
        # takes the "unknown subject" branch (an explicit subject_key always wins in observe_decision).
        decision = log.observe_decision(t, verdict, subject_key=goal.id)
        self._claim_episode_id[goal.id] = decision.episode.id
        fold = self.executor.fold_episode(decision, operation=outcome.effective)
        if outcome.applied:
            log.annotate(decision.episode, value=outcome.value_after)
            if not decision.is_new and fold.mutation in ("updated", "retracted", "closed"):
                # The initial write's "opened" row was already recorded by observe_decision; only a
                # genuine SUBSEQUENT state change (an accepted replace/retract/close) appends a new one.
                log.record_revision(
                    decision.episode, fold.mutation, t=t, value=outcome.value_after
                )
        return fold

    def episode_current(self, goal_id: str) -> "Episode | None":
        """The current authoritative episode for a claim-slot concern.

        ``.value`` is the corrected value (the last APPLIED assert/replace/retract, never a downgraded
        or merely-mentioned one) and ``.revisions`` is the bounded state-change chain, so a correction
        is never a silent overwrite: the revision it superseded stays on the timeline and the new row
        names what the slot became. Returns ``None`` when this goal has no claim-slot episode yet —
        nothing observed, or the goal never selected ``identity_policy='claim_slot'``.
        """
        log = self.episodes.get(goal_id)
        episode_id = self._claim_episode_id.get(goal_id)
        if log is None or episode_id is None:
            return None
        return log.get(episode_id)

    def prepare_observation_candidate(
        self,
        frame: Any,
        source: str,
        *,
        t: float,
        expected_goals_revision: int,
        expected_source_generation: int,
        goal_ids: "tuple[str, ...] | None" = None,
    ) -> "_ObservationCandidate | None":
        """Owner-only admission for one state-free entity-detection request."""
        import copy

        if (
            type(source) is not str
            or not source
            or not isinstance(t, (int, float))
            or isinstance(t, bool)
            or not math.isfinite(float(t))
        ):
            raise ValueError("observation admission requires a source and finite event time")
        if (
            self.goals.revision != expected_goals_revision
            or self.source_continuity_generation(source)
            != expected_source_generation
        ):
            return None
        selected = frozenset(goal_ids) if goal_ids is not None else None
        goals = tuple(
            goal
            for goal in self.goals.active()
            if goal.source == source
            and (selected is None or goal.id in selected)
            and (
                program_of(goal).perception.modality == "vision"
                or (
                    program_of(goal).perception.modality == "fused"
                    and self._watch_runtime_for(goal) is None
                )
            )
            and goal_behavior_route(goal)
            in (
                BehaviorRoute.FRAME_WATCH,
                BehaviorRoute.FRAME_TRANSITION,
                BehaviorRoute.FRAME_PROCESS,
            )
        )
        if not goals:
            return None
        # Finish every fallible copy/projection before publishing an admission
        # counter or token.
        frozen_frame = copy.deepcopy(frame)
        projected_goals = tuple(_goal_provider_projection(goal) for goal in goals)
        observed_at = self.clock()
        counter = getattr(self, "_observation_admission_counter", 0) + 1
        self._observation_admission_counter = counter
        parent_trace_id = f"{source}@{float(t)}"
        admission_id = (
            f"observation:{source}:{expected_goals_revision}:"
            f"{expected_source_generation}:{counter}"
        )
        token = _ObservationAdmissionToken(
            admission_id=admission_id,
            source_id=source,
            source_generation=expected_source_generation,
            goals_registry_revision=expected_goals_revision,
            goal_object_tokens=tuple((goal.id, goal) for goal in goals),
            adapter_token=self.vision,
            parent_source_trace_id=parent_trace_id,
            observation_trace_id=f"{parent_trace_id}:observation:{counter}",
        )
        return _ObservationCandidate(
            token=token,
            dispatch=self.vision.detect,
            frame=frozen_frame,
            goals=projected_goals,
            source_t=float(t),
            ingest_observed_ts=observed_at,
        )

    async def dispatch_observation_candidate(
        self,
        candidate: "_ObservationCandidate",
    ) -> "_ObservationCompletion":
        """Provider-only detection; no graph/window/cognition state is read or mutated."""
        import copy

        if type(candidate) is not _ObservationCandidate:
            raise ValueError("observation dispatch requires an exact admitted candidate")
        started = self.clock()
        failure = None
        snapshots = ()
        try:
            observations = await candidate.dispatch(
                # The launched candidate is owner evidence (Window now, graph on
                # reduction).  A detector receives a detached copy so retained
                # provider aliases can never rewrite either owner structure.
                copy.deepcopy(candidate.frame),
                list(candidate.goals),
            )
            if not isinstance(observations, (list, tuple)):
                raise ValueError("detector returned a malformed observation collection")
            snapshots = tuple(
                _ObservationSnapshot.capture(observation)
                for observation in observations
            )
        except Exception as exc:  # noqa: BLE001 - provider failure becomes typed owner input
            failure = _ProviderFailure(
                kind="observation_provider_failure",
                exception_type=type(exc).__name__,
                safe_message="observation provider unavailable",
            )
        return _ObservationCompletion(
            candidate=candidate,
            observations=snapshots,
            failure=failure,
            started_ts=started,
            completed_ts=self.clock(),
        )

    def launch_observation_candidate(
        self,
        candidate: "_ObservationCandidate",
        *,
        runtime_trace: bool = False,
    ) -> bool:
        """Owner-only lane-accept commit: push one launched frame exactly once."""
        if type(candidate) is not _ObservationCandidate:
            raise ValueError("observation launch requires an exact admitted candidate")
        token = candidate.token
        if (
            self.goals.revision != token.goals_registry_revision
            or self.source_continuity_generation(token.source_id)
            != token.source_generation
            or self.vision is not token.adapter_token
        ):
            return False
        for goal_id, goal_token in token.goal_object_tokens:
            if self.goals.get(goal_id) is not goal_token:
                return False
        launched = getattr(self, "_observation_launched", None)
        if launched is None:
            launched = {}
            self._observation_launched = launched
        if token.source_id in launched:
            return False
        self._window(token.source_id).push(candidate.source_t, candidate.frame)
        launched[token.source_id] = token.admission_id
        if runtime_trace and self.on_trace is not None:
            queue = getattr(self, "_judgment_trace_queue", None)
            if queue is None:
                queue = []
                self._judgment_trace_queue = queue
            queue.append(
                (
                    token.parent_source_trace_id,
                    candidate.source_t,
                    "source.frame",
                    {},
                )
            )
        return True

    def reduce_observation_completion(
        self,
        completion: "_ObservationCompletion",
    ) -> "_ObservationReduction":
        """Owner-only synchronous validation and graph/window commit."""
        if type(completion) is not _ObservationCompletion:
            raise ValueError("observation reducer requires an exact completion")
        candidate = completion.candidate
        token = candidate.token
        launched = getattr(self, "_observation_launched", {})
        launched_here = launched.get(token.source_id) == token.admission_id
        stale_reason = None
        if not launched_here:
            stale_reason = "not_launched"
        elif self.goals.revision != token.goals_registry_revision:
            stale_reason = "goal_revision"
        elif self.source_continuity_generation(token.source_id) != token.source_generation:
            stale_reason = "source_generation"
        elif self.vision is not token.adapter_token:
            stale_reason = "adapter_binding"
        else:
            for goal_id, goal_token in token.goal_object_tokens:
                if self.goals.get(goal_id) is not goal_token:
                    stale_reason = "goal_object"
                    break
        if launched_here:
            launched.pop(token.source_id, None)
        if stale_reason is not None or completion.failure is not None:
            return _ObservationReduction(
                admission_id=token.admission_id,
                stale_reason=stale_reason,
                failure=completion.failure,
                candidate=candidate,
            )
        observations = [
            snapshot.to_observation() for snapshot in completion.observations
        ]
        self.graph.ingest(
            observations,
            candidate.frame,
            token.source_id,
            t=candidate.source_t,
        )
        return _ObservationReduction(
            admission_id=token.admission_id,
            stale_reason=None,
            failure=None,
            focus=self.graph.in_focus(),
            candidate=candidate,
        )

    def prepare_frame_judgment(
        self,
        reduction: "_ObservationReduction",
        *,
        goal_id: str,
        temporal_window: "tuple[tuple[float, Any], ...] | None" = None,
    ) -> "_JudgmentJob | None":
        """Owner-only admission for one ordinary FRAME_WATCH provider call.

        This is the split-phase equivalent of the simple watch branch in
        :meth:`perceive_judged`. Transition/process and local cache/direct
        completions join this same carrier in the following slices.
        """
        import copy
        from types import MappingProxyType

        if type(reduction) is not _ObservationReduction:
            raise ValueError("frame judgment requires an exact observation reduction")
        candidate = reduction.candidate
        if (
            reduction.stale_reason is not None
            or reduction.failure is not None
            or type(candidate) is not _ObservationCandidate
        ):
            return None
        token_by_id = dict(candidate.token.goal_object_tokens)
        goal = self.goals.get(goal_id)
        modality = (
            program_of(goal).perception.modality
            if goal is not None
            else None
        )
        legacy_fused = (
            goal is not None
            and modality == "fused"
            and self._watch_runtime_for(goal) is None
        )
        if (
            goal is None
            or token_by_id.get(goal_id) is not goal
            or self.goals.revision
            != candidate.token.goals_registry_revision
            or self.source_continuity_generation(candidate.token.source_id)
            != candidate.token.source_generation
            or (modality != "vision" and not legacy_fused)
            or goal_behavior_route(goal) is not BehaviorRoute.FRAME_WATCH
        ):
            return None
        ctx = self.executor.ctx(goal.id)
        spec, strip = self._resolve_strip(
            goal,
            candidate.token.source_id,
            perception_level=ctx.perception_level,
        )
        if temporal_window is not None:
            if (
                not temporal_window
                or float(temporal_window[-1][0]) != candidate.source_t
            ):
                raise ValueError(
                    "temporal evidence window must end at the admitted source frame"
                )
            strip = self._strip_frames_from_entries(
                list(temporal_window),
                spec,
            )
        focus = reduction.focus
        roi = self._resolve_roi(goal, focus, spec)
        corrections = focus.asserted() if focus else {}
        adapter, bound = self._select_judge_adapter(goal)
        audio = self._audio_clip(candidate.token.source_id, spec)
        context: dict[str, Any] = {}
        process_qa = bool(goal.state.get("process_qa"))
        if process_qa:
            context.update(
                {
                    "process_qa": True,
                    "running_answer": ctx.running_answer,
                }
            )
        extractor_facts = None
        extractor_ctx: dict[str, Any] | None = None
        if self.extractor is not None:
            enrichment = self.extractor.extract(
                goal.condition, candidate.frame, roi
            )
            extractor_facts = _freeze_provider_value(dict(enrichment.facts))
            if type(extractor_facts) is not _FrozenMapping:
                raise ValueError("extractor facts must be a mapping")
            if not enrichment.is_empty:
                extractor_ctx = {
                    EXTRACTOR_CONTEXT_KEY: enrichment.to_dict()
                }
                context.update(extractor_ctx)
        direct_result = None
        if self.direct_judge is not None:
            direct_result = self.direct_judge.route(
                goal.condition,
                candidate.frame,
                roi,
                facts=(extractor_ctx or {})
                .get(EXTRACTOR_CONTEXT_KEY, {})
                .get("facts", {}),
                stakes=goal.state.get("stakes", "medium"),
            )
            if (
                direct_result.outcome == "uncertain"
                and direct_result.facts
            ):
                prior = dict(context.get(EXTRACTOR_CONTEXT_KEY) or {})
                prior["facts"] = {
                    **prior.get("facts", {}),
                    **direct_result.facts,
                }
                context[EXTRACTOR_CONTEXT_KEY] = prior
        if bound is not None and bound.static_context is not None:
            context[JUDGE_STATIC_CONTEXT_KEY] = bound.static_context.decoded_text
        if self._resolved_identity_policy(goal) == "claim_slot":
            context["claim_mode"] = True
        if goal.state.get("rearm") == "each_event":
            context["event_chain"] = True
            context.update(
                self._temporal_evidence_context(strip)
            )
        base_key = (
            goal.id,
            watch_reducer_entity_id(goal, focus.id if focus else None),
        )
        cache_key = None
        cache_store = False
        cached_verdict = None
        cache_audit = False
        local_verdict = None
        direct_hit = bool(
            direct_result is not None
            and direct_result.outcome == "confident"
        )
        if direct_hit:
            local_verdict = _VerdictSnapshot.capture(direct_result.verdict)
        elif not strip and not context and audio is None:
            cache_key = CacheKey(
                goal_id=goal.id,
                frame=candidate.frame,
                roi=roi,
                entity_id=base_key[1],
                condition=goal.condition,
            )
            hit = self.cache.resolve(cache_key)
            if hit.status == "HIT" and hit.verdict is not None:
                cached_verdict = _VerdictSnapshot.capture(hit.verdict)
                cache_audit = bool(
                    self.audit is not None and self.audit(cache_key)
                )
                if not cache_audit:
                    local_verdict = cached_verdict
            else:
                cache_store = True
        parent_trace = candidate.token.parent_source_trace_id
        projection_counter = getattr(
            self, "_judgment_projection_counter", 0
        ) + 1
        self._judgment_projection_counter = projection_counter
        admission_id = (
            f"judgment:{goal.id}:"
            f"{candidate.token.goals_registry_revision}:p{projection_counter}"
        )
        runtime = self._watch_runtime_for(goal)
        token = _JudgmentAdmissionToken(
            admission_id=admission_id,
            goal_id=goal.id,
            modality=str(modality),
            route=BehaviorRoute.FRAME_WATCH.value,
            goal_object_token=goal,
            watch_application=self._watch_runtime_keys.get(goal.id),
            watch_revision=candidate.token.goals_registry_revision,
            goals_registry_revision=candidate.token.goals_registry_revision,
            watch_runtime_token=runtime[0] if runtime is not None else None,
            judge_binding_token=bound if bound is not None else adapter,
            source_id=candidate.token.source_id,
            source_generation=candidate.token.source_generation,
            fused_generation=None,
            evidence_seq=0,
            parent_source_trace_id=parent_trace,
            judgment_trace_id=parent_trace,
        )
        return _JudgmentJob(
            token=token,
            modality=str(modality),
            evidence=candidate,
            provider=_ProviderRequest(
                dispatch=adapter.judge,
                condition=str(goal.condition),
                frame=copy.deepcopy(candidate.frame),
                frames=tuple(copy.deepcopy(strip or ())),
                audio=copy.deepcopy(audio),
                roi=copy.deepcopy(roi),
                exemplar=copy.deepcopy(goal.state.get("exemplar")),
                corrections=tuple(
                    (str(key), str(value)) for key, value in corrections.items()
                ),
                context=MappingProxyType(
                    {key: _readonly_provider_value(value) for key, value in context.items()}
                ),
                include_frames=bool(strip),
                include_audio=audio is not None,
                include_context=bool(context),
            ),
            reducer=_ReducerInputs(
                source_t=candidate.source_t,
                ingest_observed_ts=candidate.ingest_observed_ts,
                admitted_ts=0.0,
                reducer_entity_id=base_key[1],
                freshness=self._freshness(focus, candidate.source_t, goal.id),
                focus_token=focus,
                cache_key=cache_key,
                cache_store=cache_store,
                route_inputs=_WatchReducerInputs(
                    process_qa=process_qa,
                    cached_verdict=cached_verdict,
                    cache_audit=cache_audit,
                    extractor_facts=extractor_facts,
                    direct_hit=direct_hit,
                ),
            ),
            adapter=adapter,
            local_verdict=local_verdict,
            trace_temporal=spec.temporal,
            trace_roi=roi is not None,
            trace_provenance=tuple(_judge_binding_trace(bound).items()),
        )

    def _prepare_routed_frame_goal(
        self,
        reduction: "_ObservationReduction",
        *,
        goal_id: str,
        route: BehaviorRoute,
    ) -> tuple[_ObservationCandidate, Goal] | None:
        """Validate an observation reduction before route-specific admission."""
        if type(reduction) is not _ObservationReduction:
            raise ValueError("frame judgment requires an exact observation reduction")
        candidate = reduction.candidate
        if (
            reduction.stale_reason is not None
            or reduction.failure is not None
            or type(candidate) is not _ObservationCandidate
        ):
            return None
        token_by_id = dict(candidate.token.goal_object_tokens)
        goal = self.goals.get(goal_id)
        if (
            goal is None
            or token_by_id.get(goal_id) is not goal
            or self.goals.revision != candidate.token.goals_registry_revision
            or self.source_continuity_generation(candidate.token.source_id)
            != candidate.token.source_generation
            or program_of(goal).perception.modality != "vision"
            or goal_behavior_route(goal) is not route
        ):
            return None
        return candidate, goal

    def _publish_routed_judgment(
        self,
        *,
        goal: Goal,
        candidate: "_ObservationCandidate",
        route: BehaviorRoute,
        adapter: Any,
        bound: Any,
        reducer_entity_id: str | None,
        provider: "_ProviderRequest | _TransitionProviderRequest",
        route_inputs: Any,
        freshness: float,
        focus_token: Any = None,
    ) -> "_JudgmentJob":
        """Publish one fully projected owner admission after all fallible work."""
        runtime = self._watch_runtime_for(goal)
        parent_trace = candidate.token.parent_source_trace_id
        projection_counter = getattr(
            self, "_judgment_projection_counter", 0
        ) + 1
        self._judgment_projection_counter = projection_counter
        admission_id = (
            f"judgment:{goal.id}:"
            f"{candidate.token.goals_registry_revision}:p{projection_counter}"
        )
        token = _JudgmentAdmissionToken(
            admission_id=admission_id,
            goal_id=goal.id,
            modality="vision",
            route=route.value,
            goal_object_token=goal,
            watch_application=self._watch_runtime_keys.get(goal.id),
            watch_revision=candidate.token.goals_registry_revision,
            goals_registry_revision=candidate.token.goals_registry_revision,
            watch_runtime_token=runtime[0] if runtime is not None else None,
            judge_binding_token=bound if bound is not None else adapter,
            source_id=candidate.token.source_id,
            source_generation=candidate.token.source_generation,
            fused_generation=None,
            evidence_seq=0,
            parent_source_trace_id=parent_trace,
            judgment_trace_id=parent_trace,
        )
        return _JudgmentJob(
            token=token,
            modality="vision",
            evidence=candidate,
            provider=provider,
            reducer=_ReducerInputs(
                source_t=candidate.source_t,
                ingest_observed_ts=candidate.ingest_observed_ts,
                admitted_ts=0.0,
                reducer_entity_id=reducer_entity_id,
                freshness=freshness,
                focus_token=focus_token,
                route_inputs=route_inputs,
            ),
            adapter=adapter,
            trace_provenance=tuple(_judge_binding_trace(bound).items()),
        )

    def prepare_transition_judgment(
        self,
        reduction: "_ObservationReduction",
        *,
        goal_id: str,
    ) -> "_JudgmentJob | None":
        """Owner-only admission for one atomic before/after transition tick."""
        import copy
        from types import MappingProxyType

        prepared = self._prepare_routed_frame_goal(
            reduction,
            goal_id=goal_id,
            route=BehaviorRoute.FRAME_TRANSITION,
        )
        if prepared is None:
            return None
        candidate, goal = prepared
        focus = reduction.focus
        declared = identity_policy_of(goal)
        latch_key, latch_key_source = transition_latch_key(
            declared, focus.id if focus else None
        )
        # Exact legacy behavior: a per-entity transition with no subject does
        # not admit provider work or touch the latch.
        if declared in PER_ENTITY_IDENTITY_POLICIES and focus is None:
            return None
        from_cond = str(goal.state["from"])
        to_cond = str(goal.state["to"])
        adapter, bound = self._select_judge_adapter(goal)
        roi: Box | None = focus.box if focus else None
        corrections = focus.asserted() if focus else {}
        _, strip = self._resolve_strip(goal, candidate.token.source_id)
        judge_context: dict[str, Any] | None = None
        enriched = False
        extractor_facts = None
        if self.extractor is not None:
            enrichment = self.extractor.extract(goal.condition, candidate.frame, roi)
            if not enrichment.is_empty:
                judge_context = {EXTRACTOR_CONTEXT_KEY: enrichment.to_dict()}
                enriched = True
                extractor_facts = _freeze_provider_value(
                    dict(enrichment.facts)
                )
                if type(extractor_facts) is not _FrozenMapping:
                    raise ValueError("extractor facts must be a mapping")
        if bound is not None and bound.static_context is not None:
            if judge_context is not None and JUDGE_STATIC_CONTEXT_KEY in judge_context:
                raise ReservedContextKeyError(
                    f"goal {goal.id!r} judge context already carries the reserved key "
                    f"{JUDGE_STATIC_CONTEXT_KEY!r}; a bound static context cannot be injected over it"
                )
            judge_context = dict(judge_context) if judge_context else {}
            judge_context[JUDGE_STATIC_CONTEXT_KEY] = bound.static_context.decoded_text
        frozen_frame = copy.deepcopy(candidate.frame)
        frozen_strip = tuple(copy.deepcopy(strip or ()))
        frozen_roi = copy.deepcopy(roi)
        frozen_exemplar = copy.deepcopy(goal.state.get("exemplar"))
        frozen_corrections = tuple(
            (str(key), str(value)) for key, value in corrections.items()
        )
        frozen_context = MappingProxyType(
            {
                key: _readonly_provider_value(value)
                for key, value in (judge_context or {}).items()
            }
        )
        request_common = dict(
            dispatch=adapter.judge,
            frame=frozen_frame,
            frames=frozen_strip,
            audio=None,
            roi=frozen_roi,
            exemplar=frozen_exemplar,
            corrections=frozen_corrections,
            context=frozen_context,
            include_frames=bool(strip),
            include_context=judge_context is not None,
            include_exemplar=False,
        )
        route_inputs = _TransitionReducerInputs(
            identity_policy=declared,
            latch_key=latch_key,
            latch_key_source=latch_key_source,
            focus_entity_id=focus.id if focus else None,
            strip_len=len(strip) if strip else 0,
            enriched=enriched,
            extractor_facts=extractor_facts,
        )
        return self._publish_routed_judgment(
            goal=goal,
            candidate=candidate,
            route=BehaviorRoute.FRAME_TRANSITION,
            adapter=adapter,
            bound=bound,
            reducer_entity_id=latch_key,
            provider=_TransitionProviderRequest(
                before=_ProviderRequest(condition=from_cond, **request_common),
                after=_ProviderRequest(condition=to_cond, **request_common),
            ),
            route_inputs=route_inputs,
            freshness=1.0,
            focus_token=focus,
        )

    def prepare_process_judgment(
        self,
        reduction: "_ObservationReduction",
        *,
        goal_id: str,
    ) -> "_JudgmentJob | None":
        """Owner-only admission for one typed process-log observation."""
        import copy
        from types import MappingProxyType

        prepared = self._prepare_routed_frame_goal(
            reduction,
            goal_id=goal_id,
            route=BehaviorRoute.FRAME_PROCESS,
        )
        if prepared is None:
            return None
        candidate, goal = prepared
        adapter, bound = self._select_judge_adapter(goal)
        ctx = self.executor.ctx(goal.id)
        _, strip = self._resolve_strip(
            goal,
            candidate.token.source_id,
            perception_level=ctx.perception_level,
        )
        context: dict[str, Any] = {"observe_step": True}
        if bound is not None and bound.static_context is not None:
            context[JUDGE_STATIC_CONTEXT_KEY] = bound.static_context.decoded_text
        process_key = (goal.id,)
        return self._publish_routed_judgment(
            goal=goal,
            candidate=candidate,
            route=BehaviorRoute.FRAME_PROCESS,
            adapter=adapter,
            bound=bound,
            reducer_entity_id=None,
            provider=_ProviderRequest(
                dispatch=adapter.judge,
                condition=str(goal.condition),
                frame=copy.deepcopy(candidate.frame),
                frames=tuple(copy.deepcopy(strip or ())),
                audio=None,
                roi=None,
                exemplar=copy.deepcopy(goal.state.get("exemplar")),
                corrections=(),
                context=MappingProxyType(
                    {
                        key: _readonly_provider_value(value)
                        for key, value in context.items()
                    }
                ),
                # _judge's process mode historically forwards these keys even
                # when the strip/audio values are empty.
                include_frames=True,
                include_audio=True,
                include_context=True,
            ),
            route_inputs=_ProcessReducerInputs(process_key=process_key),
            freshness=1.0,
        )

    def prepare_judgment(
        self,
        reduction: "_ObservationReduction",
        *,
        goal_id: str,
        temporal_window: "tuple[tuple[float, Any], ...] | None" = None,
    ) -> "_JudgmentJob | None":
        """One owner preparation facade for every frame-governed Watch route."""
        goal = self.goals.get(goal_id)
        if goal is None:
            return None
        route = goal_behavior_route(goal)
        if route is BehaviorRoute.FRAME_TRANSITION:
            return self.prepare_transition_judgment(reduction, goal_id=goal_id)
        if route is BehaviorRoute.FRAME_PROCESS:
            return self.prepare_process_judgment(reduction, goal_id=goal_id)
        if route is BehaviorRoute.FRAME_WATCH:
            return self.prepare_frame_judgment(
                reduction,
                goal_id=goal_id,
                temporal_window=temporal_window,
            )
        return None

    @staticmethod
    def _judgment_base_key(job: "_JudgmentJob") -> Hashable:
        route_inputs = job.reducer.route_inputs
        if type(route_inputs) is _ProcessReducerInputs:
            return route_inputs.process_key
        return (job.token.goal_id, job.reducer.reducer_entity_id)

    def _prepared_judgment_reason(self, job: "_JudgmentJob") -> str | None:
        """Read-only lifecycle validation before a bounded lane accepts a job."""
        if type(job) is not _JudgmentJob:
            return "job_type"
        token = job.token
        if (
            token.evidence_seq != 0
            or not token.admission_id
            or not token.judgment_trace_id
        ):
            return "already_launched"
        goal = self.goals.get(token.goal_id)
        runtime = self._watch_runtime_for(token.goal_id)
        bound = self._judge_bindings.get(token.goal_id)
        if (
            goal is None
            or goal is not token.goal_object_token
            or self.goals.revision != token.goals_registry_revision
            or program_of(goal).perception.modality != token.modality
            or goal_behavior_route(goal).value != token.route
        ):
            return "goal_revision"
        if (runtime[0] if runtime is not None else None) is not token.watch_runtime_token:
            return "runtime_binding"
        if self._watch_runtime_keys.get(token.goal_id) != token.watch_application:
            return "watch_application"
        if token.watch_revision != token.goals_registry_revision:
            return "watch_revision"
        if (bound if bound is not None else self.vision) is not token.judge_binding_token:
            return "judge_binding"
        if (
            self.source_continuity_generation(token.source_id)
            != token.source_generation
        ):
            return "source_generation"
        return None

    def launch_judgments(
        self,
        jobs: "tuple[_JudgmentJob, ...]",
        *,
        runtime_trace: bool = False,
    ) -> "tuple[_JudgmentJob, ...]":
        """Atomically publish a prepared batch after a bounded lane accepts it.

        Preparation is projection only.  Evidence sequences and latest-admitted
        coordinates become visible together in this synchronous owner segment;
        a rejected scheduler offer therefore cannot stale accepted work.
        """
        prepared = tuple(jobs)
        if not prepared:
            return ()
        reasons = tuple(self._prepared_judgment_reason(job) for job in prepared)
        if any(reason is not None for reason in reasons):
            raise ValueError(
                "judgment batch is not launchable: "
                + ",".join(reason or "ok" for reason in reasons)
            )
        # Validate the whole batch before the first sequence/latest mutation.
        launched: list[_JudgmentJob] = []
        for job in prepared:
            base_key = self._judgment_base_key(job)
            evidence_seq = self._admit_evidence(base_key)
            token = replace(
                job.token,
                evidence_seq=evidence_seq,
            )
            launched_job = replace(
                job,
                token=token,
                reducer=replace(job.reducer, admitted_ts=self.clock()),
            )
            latest = getattr(self, "_judgment_latest_admission", None)
            if latest is None:
                latest = {}
                self._judgment_latest_admission = latest
            latest[base_key] = token.admission_id
            admission_keys = getattr(self, "_judgment_admission_keys", None)
            if admission_keys is None:
                admission_keys = {}
                self._judgment_admission_keys = admission_keys
            admission_keys[token.admission_id] = base_key
            while len(admission_keys) > _FUSED_ADMISSION_LEDGER_MAX:
                admission_keys.pop(next(iter(admission_keys)), None)
            if (
                runtime_trace
                and token.route == BehaviorRoute.FRAME_WATCH.value
            ):
                self._queue_judgment_trace(
                    launched_job,
                    "judge.started",
                    goal_id=token.goal_id,
                    temporal=job.trace_temporal,
                    roi=job.trace_roi,
                    **dict(job.trace_provenance),
                )
            launched.append(launched_job)
        return tuple(launched)

    def launch_judgment(self, job: "_JudgmentJob") -> "_JudgmentJob":
        """Publish exactly one prepared judgment after lane acceptance."""
        if type(job) is not _JudgmentJob:
            raise ValueError("judgment launch requires an exact prepared job")
        if job.token.evidence_seq > 0 and job.token.admission_id:
            return job
        return self.launch_judgments((job,))[0]

    def validate_judgment_token(
        self,
        token: "_JudgmentAdmissionToken",
        *,
        live_fused_generation: int | None = None,
    ) -> str | None:
        """Read-only typed lifecycle validation for every judgment carrier."""
        if type(token) is not _JudgmentAdmissionToken:
            return "token_type"
        goal = self.goals.get(token.goal_id)
        runtime = self._watch_runtime_for(token.goal_id)
        bound = self._judge_bindings.get(token.goal_id)
        if goal is None or goal is not token.goal_object_token:
            return "goal_object_token"
        if self.goals.revision != token.goals_registry_revision:
            return "goals_registry_revision"
        if program_of(goal).perception.modality != token.modality:
            return "modality"
        if goal_behavior_route(goal).value != token.route:
            return "route"
        if self._watch_runtime_keys.get(token.goal_id) != token.watch_application:
            return "watch_application"
        if token.watch_revision != token.goals_registry_revision:
            return "watch_revision"
        if (runtime[0] if runtime is not None else None) is not token.watch_runtime_token:
            return "watch_runtime_token"
        if (bound if bound is not None else self.vision) is not token.judge_binding_token:
            return "judge_binding_token"
        if (
            self.source_continuity_generation(token.source_id)
            != token.source_generation
        ):
            return "source_generation"
        if (
            token.fused_generation is not None
            and live_fused_generation is not None
            and token.fused_generation != live_fused_generation
        ):
            return "fused_generation"
        admission_keys = getattr(self, "_judgment_admission_keys", {})
        base_key = admission_keys.get(token.admission_id)
        latest = getattr(self, "_judgment_latest_admission", {})
        if (
            base_key is None
            or latest.get(base_key) != token.admission_id
            or self._sequencer is None
            or self._sequencer.latest_admitted(base_key) != token.evidence_seq
        ):
            return "evidence_seq"
        reduced = getattr(self, "_judgment_reduced_admissions", {})
        if reduced.get(base_key) == token.admission_id:
            return "duplicate_completion"
        return None

    def _judgment_stale_reason(self, job: "_JudgmentJob") -> str | None:
        """Compatibility-named route check; includes exact watch_application identity."""
        token = job.token
        reason = self.validate_judgment_token(token)
        return "superseded_admission" if reason == "evidence_seq" else reason

    def _mark_judgment_reduced(self, job: "_JudgmentJob") -> None:
        reduced = getattr(self, "_judgment_reduced_admissions", None)
        if reduced is None:
            reduced = {}
            self._judgment_reduced_admissions = reduced
        reduced[self._judgment_base_key(job)] = job.token.admission_id

    def _queue_judgment_trace(
        self,
        job: "_JudgmentJob",
        stage: str,
        **data: Any,
    ) -> None:
        """Materialize a trace row without invoking an async/user callback."""
        if self.on_trace is None:
            return
        queue = getattr(self, "_judgment_trace_queue", None)
        if queue is None:
            queue = []
            self._judgment_trace_queue = queue
        queue.append(
            (
                job.token.judgment_trace_id,
                job.source_t,
                stage,
                data,
            )
        )
        if len(queue) > 1024:
            del queue[: len(queue) - 1024]

    def drain_judgment_trace_rows(
        self,
    ) -> tuple[tuple[str, float, str, dict[str, Any]], ...]:
        """Owner-only transfer of already-materialized shared-pipeline traces."""
        queue = getattr(self, "_judgment_trace_queue", None)
        if not queue:
            return ()
        rows = tuple(queue)
        queue.clear()
        return rows

    def _record_completed_provider_calls(
        self,
        completion: "_JudgmentCompletion",
    ) -> bool:
        """Apply provider accounting once, including for stale semantic results.

        A completion proves the provider call was launched and spent. Semantic
        token invalidation may make its verdict inert, but cannot erase that
        cost. Runtime-owned completions use a monotonic completion ordinal with
        an exact high-water/gap ledger. Direct/manual completions without that
        runtime coordinate retain a bounded admission-id compatibility ledger.
        """
        completion_ordinal = completion.completion_ordinal
        if completion_ordinal is not None:
            if not self._accept_provider_completion_ordinal(completion_ordinal):
                return False
        else:
            accounted = getattr(
                self, "_judgment_accounted_admissions", None
            )
            if accounted is None:
                accounted = {}
                self._judgment_accounted_admissions = accounted
            admission_id = completion.job.token.admission_id
            if admission_id in accounted:
                return False
            accounted[admission_id] = True
            while len(accounted) > _PROVIDER_ACCOUNTING_FALLBACK_MAX:
                accounted.pop(next(iter(accounted)), None)
        token = completion.job.token
        for call in completion.calls:
            self.funnel.judged(
                concern=token.goal_id,
                camera=token.source_id,
            )
            if not call.provider_spent:
                continue
            # Legacy _timed_judge records latency/usage only when the adapter
            # returned; a raised/malformed call has no successful receipt.
            if (
                call.failure is None
                or call.failure.kind == "malformed_response"
            ):
                self.funnel.record_judge(
                    latency_s=call.judge_latency_s,
                    **dict(call.usage),
                )
                if call.usage:
                    self._queue_judgment_trace(
                        completion.job,
                        "judge.usage",
                        model=self._judge_model(completion.job.adapter),
                        **dict(call.usage),
                        **dict(completion.job.trace_provenance),
                    )
        return True

    def _accept_provider_completion_ordinal(self, ordinal: int) -> bool:
        """Accept one global provider receipt coordinate exactly once.

        ``high_water`` records the largest observed ordinal. Missing lower
        ordinals are retained as compact disjoint intervals, so reverse
        completion and a late gap fill remain valid while replay of either an
        old receipt or the filled gap remains inert forever. Interval storage
        avoids allocating ``O(ordinal)`` memory for a large coordinate jump.
        """
        if type(ordinal) is not int or ordinal < 1:
            return False
        high_water = int(
            getattr(self, "_provider_completion_high_water", 0)
        )
        gaps = list(
            getattr(self, "_provider_completion_gap_ranges", ())
        )
        if ordinal > high_water:
            if ordinal > high_water + 1:
                gaps.append((high_water + 1, ordinal - 1))
            self._provider_completion_high_water = ordinal
            self._provider_completion_gap_ranges = tuple(gaps)
            return True
        for index, (start, end) in enumerate(gaps):
            if ordinal < start:
                return False
            if ordinal > end:
                continue
            replacement = []
            if start < ordinal:
                replacement.append((start, ordinal - 1))
            if ordinal < end:
                replacement.append((ordinal + 1, end))
            gaps[index : index + 1] = replacement
            self._provider_completion_gap_ranges = tuple(gaps)
            return True
        return False

    def _reduce_transition_completion(
        self,
        completion: "_JudgmentCompletion",
    ) -> list[FireEvent]:
        job = completion.job
        token = job.token
        result = completion.result
        route_inputs = job.reducer.route_inputs
        if (
            type(result) is not _TransitionJudgmentResult
            or type(route_inputs) is not _TransitionReducerInputs
            or type(job.provider) is not _TransitionProviderRequest
        ):
            raise ValueError("transition reducer requires an exact paired result")
        if self._judgment_stale_reason(job) is not None:
            self._record_completed_provider_calls(completion)
            return []
        goal = self.goals.get(token.goal_id)
        assert goal is not None  # exact token validation above
        self._mark_judgment_reduced(job)
        if route_inputs.extractor_facts is not None:
            self._queue_judgment_trace(
                job,
                "extractor.enriched",
                goal_id=goal.id,
                facts=_thaw_provider_value(route_inputs.extractor_facts),
            )
        self._record_completed_provider_calls(completion)
        base_key = self._judgment_base_key(job)
        if completion.failure is not None or result.after is None:
            failure = completion.failure
            assert failure is not None
            self._degrade_tick(
                base_key,
                failure.safe_message,
                goal_id=goal.id,
                label=" (transition)",
                action="applying nothing this tick",
            )
            self._queue_judgment_trace(
                job,
                STAGE_TRANSITION_DECISION,
                goal_id=goal.id,
                entity_id=route_inputs.latch_key,
                from_truth=None,
                to_truth=None,
                armed=None,
                fired_latch=None,
                outcome="judge_degraded",
                strip_len=route_inputs.strip_len,
                enriched=route_inputs.enriched,
                identity_policy=route_inputs.identity_policy,
                latch_key=route_inputs.latch_key,
                latch_key_source=route_inputs.latch_key_source,
                focus_entity_id=route_inputs.focus_entity_id,
                from_confidence=None,
                to_confidence=None,
                from_evidence=None,
                to_evidence=None,
            )
            return []
        before = result.before.verdict.to_verdict()
        after = result.after.verdict.to_verdict()
        self._note_judge_ok(base_key)
        timing = JudgmentTimingBuilder(clock=self.clock)
        timing.mark_ingest_observed(at=job.ingest_observed_ts)
        timing.mark_admitted(at=job.admitted_ts)
        timing.mark_judge_started(at=result.before.judge_started_ts)
        timing.mark_judge_completed(at=result.after.judge_completed_ts)
        timing.set_source_media_ts(job.source_t)
        tick = self.executor.transition_decide(
            goal,
            before,
            after,
            focus_id=route_inputs.latch_key,
            timing=timing,
            evidence_seq=token.evidence_seq,
            # The SOURCE timeline the judged path already gates on (``_fire_concern(t=job.source_t)``).
            # It is what makes the gate's refractory window and the delivery merge measure real elapsed
            # stream time on this path instead of nothing at all.
            now=job.source_t,
        )
        self._queue_judgment_trace(
            job,
            STAGE_TRANSITION_DECISION,
            goal_id=goal.id,
            entity_id=route_inputs.latch_key,
            from_truth=tick.from_truth,
            to_truth=tick.to_truth,
            armed=tick.armed,
            fired_latch=tick.fired_latch,
            outcome=tick.outcome,
            strip_len=route_inputs.strip_len,
            enriched=route_inputs.enriched,
            identity_policy=route_inputs.identity_policy,
            latch_key=route_inputs.latch_key,
            latch_key_source=route_inputs.latch_key_source,
            focus_entity_id=route_inputs.focus_entity_id,
            from_confidence=before.confidence,
            to_confidence=after.confidence,
            from_evidence=before.evidence,
            to_evidence=after.evidence,
        )
        fired = tick.fired
        if fired is None:
            return []
        goal.state["count"] = goal.state.get("count", 0) + 1
        self.metrics.incr("transition.fire")
        if fired.action == FIRE:
            self.funnel.fired(concern=goal.id, camera=token.source_id)
        event = FireEvent(
            fired.goal_id,
            fired.action,
            fired.text,
            fired.entity_id,
            fired.verdict,
            judgment=fired.judgment,
            provenance=fired.provenance,
            modality=fired.modality,
            # The gate's own typed reason, now that this path HAS a gate decision to report. It was
            # dropped here (and hard-coded None upstream), so every transition Incident reached the
            # wire with an empty ``reason`` while judged Incidents carried one.
            reason=fired.reason,
        )
        if event.judgment is not None:
            self._queue_judgment_trace(
                job,
                STAGE_JUDGMENT_EVENT,
                goal_id=goal.id,
                action=event.judgment.action,
                dispatch_seq=event.judgment.evidence_seq,
                entity_id=route_inputs.latch_key,
                judgment_event_id=judgment_event_id(
                    goal.id,
                    route_inputs.latch_key,
                    event.judgment.evidence_seq,
                ),
                evidence_origin=event.judgment.evidence_origin or "transition",
            )
        # Freeze the one audit record in the same synchronous owner segment.
        # Trace rows are data only here; no async/user callback has run.
        self._log_fires([event], job.source_t)
        return [event]

    def _reduce_process_completion(
        self,
        completion: "_JudgmentCompletion",
    ) -> list[FireEvent]:
        job = completion.job
        result = completion.result
        route_inputs = job.reducer.route_inputs
        if (
            type(result) is not _ProcessJudgmentResult
            or type(route_inputs) is not _ProcessReducerInputs
            or type(job.provider) is not _ProviderRequest
        ):
            raise ValueError("process reducer requires an exact log-only result")
        if self._judgment_stale_reason(job) is not None:
            self._record_completed_provider_calls(completion)
            return []
        goal = self.goals.get(job.token.goal_id)
        assert goal is not None
        self._mark_judgment_reduced(job)
        self._record_completed_provider_calls(completion)
        if result.call.failure is not None:
            self._degrade_tick(
                route_inputs.process_key,
                result.call.failure.safe_message,
                goal_id=goal.id,
                label=" (observe_step)",
                action="logging nothing this tick",
            )
            return []
        self._note_judge_ok(route_inputs.process_key)
        verdict = result.call.verdict.to_verdict()
        step = (getattr(verdict, "evidence", "") or "").strip()
        if step and step != "not enough evidence":
            ctx = self.executor.ctx(goal.id)
            ctx.process_log.append(step)
            ctx.last_step_at = job.source_t
        return []

    def reduce_judgment_completion(
        self,
        completion: "_JudgmentCompletion",
    ) -> list[FireEvent]:
        """Owner-only validation and cognition commit for every non-fused frame route."""
        if type(completion) is not _JudgmentCompletion:
            raise ValueError("judgment reducer requires an exact provider completion")
        job = completion.job
        token = job.token
        from .fused_sync import _FusedEvidenceBundle

        if type(job.bundle) is _FusedEvidenceBundle:
            raise ValueError("fused completion requires its live generation")
        if type(completion.result) is _TransitionJudgmentResult:
            return self._reduce_transition_completion(completion)
        if type(completion.result) is _ProcessJudgmentResult:
            return self._reduce_process_completion(completion)
        if type(completion.result) is not _WatchJudgmentResult:
            raise ValueError("judgment reducer requires a typed result variant")
        goal = self.goals.get(token.goal_id)
        if self._judgment_stale_reason(job) is not None:
            self._record_completed_provider_calls(completion)
            return []
        assert goal is not None
        base_key = self._judgment_base_key(job)
        self._mark_judgment_reduced(job)
        self._record_completed_provider_calls(completion)
        route_inputs = job.reducer.route_inputs
        if type(route_inputs) is not _WatchReducerInputs:
            raise ValueError("watch reducer requires exact route inputs")
        if route_inputs.extractor_facts is not None:
            self._queue_judgment_trace(
                job,
                "extractor.enriched",
                goal_id=goal.id,
                facts=_thaw_provider_value(route_inputs.extractor_facts),
            )
        if route_inputs.direct_hit:
            self.metrics.incr("direct_judge.hit")
        verdict = completion.verdict.to_verdict()
        if route_inputs.cached_verdict is not None:
            cached = route_inputs.cached_verdict.to_verdict()
            if route_inputs.cache_audit:
                self.metrics.incr("cache.audit")
                if completion.failure is not None:
                    self._degrade_tick(
                        base_key,
                        completion.failure.safe_message,
                        goal_id=goal.id,
                        label=" (cache audit)",
                        action=(
                            "discarding prior dwell support and serving "
                            "the cached read"
                        ),
                    )
                    verdict = cached
                elif verdict.satisfied != cached.satisfied:
                    self.metrics.incr("cache.false_serve")
                    if job.reducer.cache_key is not None:
                        self.cache.store(
                            job.reducer.cache_key,
                            verdict,
                            verdict.confidence,
                        )
                else:
                    verdict = cached
            else:
                self.metrics.incr("cache.hit")
                verdict = cached
        elif completion.failure is not None:
            self.funnel.refused(concern=goal.id, camera=token.source_id)
            self._degrade_tick(
                base_key,
                completion.failure.safe_message,
                goal_id=goal.id,
                label="",
                action="degrading this tick to ASK/SILENT",
            )
            self._queue_judgment_trace(
                job,
                "judge.degraded",
                goal_id=goal.id,
                cause=repr(RuntimeError(completion.failure.safe_message)),
            )
            self._queue_judgment_trace(
                job,
                "gate.classified",
                goal_id=goal.id,
                truth="unknown",
                action="ask",
                entity_id=job.reducer.reducer_entity_id,
                degraded=True,
            )
            return []
        elif job.reducer.cache_store and job.reducer.cache_key is not None:
            self.cache.store(
                job.reducer.cache_key,
                verdict,
                verdict.confidence,
            )
        provider_verdict = verdict
        clip_events_present = bool(
            goal.state.get("rearm") == "each_event"
            and verdict.events
        )
        if (
            goal.state.get("rearm") == "each_event"
            and not clip_events_present
        ):
            verdict = normalize_event_chain_verdict(verdict)
        self._note_judge_ok(base_key)
        timing = JudgmentTimingBuilder(clock=self.clock)
        timing.mark_ingest_observed(at=job.ingest_observed_ts)
        timing.mark_admitted(at=job.admitted_ts)
        timing.mark_judge_started(at=completion.judge_started_ts)
        timing.mark_judge_completed(at=completion.judge_completed_ts)
        timing.set_source_media_ts(job.source_t)
        freshness = job.reducer.freshness
        truth = classify(
            verdict,
            freshness,
            self.gate.threshold,
            self.gate.fresh_min,
        )
        ctx = self.executor.ctx(goal.id)
        if truth == "unknown":
            # THE refusal counter for this tick — the one and only site on this path (the degraded
            # branch above owns the ticks that never reached a verdict). Nothing downstream may
            # count the same abstention again; see the emission loop below.
            self.funnel.refused(concern=goal.id, camera=token.source_id)
        else:
            self.last_supported[goal.id] = job.source_t
            ctx.perception_level = 0
        if truth == "yes" or clip_events_present:
            self.last_satisfied[goal.id] = job.source_t
        resolved: FireEvent | None = None
        if route_inputs.process_qa and verdict.running_answer:
            ctx.running_answer = verdict.running_answer
            ctx.process_log.append(verdict.running_answer)
            if len(ctx.process_log) > 256:
                del ctx.process_log[: len(ctx.process_log) - 256]
            ctx.last_step_at = job.source_t
            resolved = self._resolve_process_qa(
                goal,
                ctx,
                verdict.confidence,
            )
        doc_fold = self._document_episode_fold(
            goal, verdict, job.source_t, truth=truth, key=base_key
        )
        claim_fold = self._claim_episode_fold(goal, verdict, job.source_t)
        gate_outcomes: list[GateOutcome] = []
        if clip_events_present:
            clip_fires = self._fire_clip_occurrences(
                goal,
                verdict,
                base_key,
                source_t=job.source_t,
                context=job.context,
                freshness=freshness,
                timing=timing,
                evidence_seq=token.evidence_seq,
                outcome_sink=gate_outcomes.append,
            )
            fire = clip_fires[-1] if clip_fires else None
            events = list(clip_fires)
        else:
            fire = self._fire_concern(
                goal,
                verdict,
                base_key,
                freshness=freshness,
                t=job.source_t,
                timing=timing,
                evidence_seq=token.evidence_seq,
                fold=doc_fold if doc_fold is not None else claim_fold,
                outcome_sink=gate_outcomes.append,
            )
            events = [] if fire is None else [fire]
        # ONE refusal per judged tick, counted at CLASSIFICATION (the `truth == "unknown"` branch
        # above) — never again here at EMISSION. A gate ASK is exactly an unknown truth, so bumping
        # both points counted every abstention twice and made ``funnel.refused`` exceed
        # ``funnel.judged``, which no one-per-tick counter can honestly do. This loop therefore
        # counts fires only, byte-for-byte the same accounting the inline twin (``perceive_judged``)
        # has always used: the async dispatch mode reports the SAME funnel as the inline one.
        for emitted in events:
            if emitted.action == FIRE:
                self.funnel.fired(concern=goal.id, camera=token.source_id)
        if resolved is not None:
            self.funnel.fired(concern=goal.id, camera=token.source_id)
            events.append(resolved)
        action = (
            "fire"
            if resolved is not None
            or (fire is not None and fire.action == FIRE)
            else fire.action
            if fire is not None
            else "silent"
        )
        gate_reason = gate_outcomes[-1].reason if gate_outcomes else None
        self._queue_judgment_trace(
            job,
            "judge.finished",
            goal_id=goal.id,
            satisfied=provider_verdict.satisfied,
            confidence=provider_verdict.confidence,
            **self._event_chain_provider_advisory(goal, provider_verdict),
            **dict(job.trace_provenance),
        )
        self._queue_judgment_trace(
            job,
            "gate.classified",
            goal_id=goal.id,
            truth=truth,
            action=action,
            reason=_trace_reason(truth, action, gate_reason),
            gate_reason=gate_reason,
            entity_id=job.reducer.reducer_entity_id,
            satisfied=verdict.satisfied,
            confidence=verdict.confidence,
            freshness=freshness,
            hold_s=program_of(goal).hold_s,
            **self._event_chain_advisory(
                goal, job.reducer.reducer_entity_id, verdict
            ),
        )
        for emitted in events:
            if emitted.judgment is None:
                continue
            self._queue_judgment_trace(
                job,
                STAGE_JUDGMENT_EVENT,
                goal_id=goal.id,
                action=emitted.judgment.action,
                dispatch_seq=emitted.judgment.evidence_seq,
                entity_id=job.reducer.reducer_entity_id,
                judgment_event_id=judgment_event_id(
                    goal.id,
                    job.reducer.reducer_entity_id,
                    emitted.judgment.evidence_seq,
                    emitted.judgment.occurrence_id,
                ),
                evidence_origin=(
                    emitted.judgment.evidence_origin or "judged"
                ),
                **(
                    {
                        "occurrence_revision_id": emitted.revision_id,
                        "occurrence_source_t": (
                            emitted.judgment.timing.source_media_ts
                        ),
                    }
                    if clip_events_present
                    else {}
                ),
            )
        self._queue_judgment_trace(
            job,
            STAGE_SHADOW_FOLD,
            goal_id=goal.id,
            **self._shadow_observe(verdict).to_advisory(),
        )
        self._log_fires(events, job.source_t)
        return events

    async def perceive_judged(
        self, frame: Any, source: str = "camera", *, t: float | None = None,
        expected_goals_revision: int | None = None,
        expected_source_generation: int | None = None,
        goal_ids: "tuple[str, ...] | None" = None,
        _audio_units_by_goal: "dict[str, tuple[AudioUnit, ...]] | None" = None,
        _decision_terminal: Any = None,
    ) -> list[FireEvent]:
        """Judge selected active *watch* goals against ``frame`` and emit rising-edge actions.

        Event / scheduler goals run through their own paths (:meth:`perceive_events`, :meth:`tick`)
        but feed the **same three-state gate** — one gate, two perception tracks.

        ``goal_ids`` is the admission result from the runtime's per-Watch tier-0 gate. ``None``
        preserves the manual API's historical "all active goals" behavior; a tuple restricts both
        detector context and judgment to those Watches, so one Watch's permissive admission settings
        cannot spend calls or mutate gate state for another.

        ``expected_goals_revision`` (the non-blocking-dispatch seam): the goal-registry revision this
        frame was ADMITTED under. When set, the call aborts — returning without applying anything —
        the moment ``self.goals.revision`` no longer matches, checked **after every network await and
        before the mutation it feeds**. This is what makes an OFF-LOOP judgment safe: while the vision
        call is in flight the conductor may disarm a goal or safely update its announcement, and a stale
        verdict must not flip a rising-edge latch, write a disposition/action record, stamp
        coverage/deadline satisfaction, or judge old evidence as-if under the newly authored plan.
        Filtering the *returned* fires would be too late — the gate mutates inside this call
        (:meth:`_fire_concern` / the executor), so the abort lives here, before the gate.

        ``expected_source_generation`` is the matching transport-continuity guard. The runtime
        captures it when async work launches and advances it at a sequence gap/reconnect/format break,
        before yielding control. A stale result may still finish if its judge suppresses task
        cancellation, but it cannot populate the verdict cache or mutate cognition state.

        ``_decision_terminal`` is the equivalent scheduler-owned latch for an absolute decision
        deadline. It is private runtime plumbing: once terminal, a cancellation-resistant audio
        provider may physically return later, but its result cannot reach any cognition mutation.

        Abort semantics (what is protected vs what may proceed):

        - **Aborted on staleness** — every goal-conditioned application: gate/latch evaluation,
          executor context (running answer / process log / transition state), ``last_supported`` /
          ``last_satisfied`` stamps, action-log records, and the confusion auto-refinement pass.
        - **Allowed to proceed** — frame-level, goal-independent perception: the rolling FrameWindow
          push and the entity-graph ingest of ``vision.detect`` observations. The graph records what
          was IN the frame regardless of which goals asked; that world-state stays valid across a
          goal-registry change. Executable same-id replacement is rejected, while the sole safe update
          (``say``) still advances the revision so an in-flight result cannot emit stale presentation.
        - **Mid-loop staleness** (multi-goal): goals already gated under the matching revision keep
          their (already-applied) fires — swallowing them would strand a flipped latch with no emitted
          incident — and all *subsequent* judgment/mutation is aborted. With a single active goal this
          reduces to the plain contract: abort ⇒ ``[]`` and untouched state.

        ``None`` (the default) never checks — byte-for-byte the inline path.
        """
        now = self.clock() if t is None else t
        audio_only = _audio_units_by_goal is not None
        active_goals = list(self.goals.active())
        if goal_ids is not None:
            selected = frozenset(goal_ids)
            active_goals = [goal for goal in active_goals if goal.id in selected]
        if audio_only:
            active_goals = [
                goal
                for goal in active_goals
                if program_of(goal).perception.modality == "audio"
            ]
        require_duration_routes(
            active_goals,
            driver="Percept.perceive_judged",
            require_worker_cadence=False,
        )
        unsupported_goals = [
            goal
            for goal in active_goals
            if goal_behavior_route(goal) is BehaviorRoute.UNSUPPORTED
        ]
        if unsupported_goals:
            # External event/boundary/rhythm/scheduler Goals are intentionally skipped below and
            # remain driven by their dedicated APIs. A malformed/non-canonical armed Goal has no
            # legitimate driver, so refuse before frame buffering, detection, or provider calls.
            require_worker_routes(unsupported_goals, driver="Percept.perceive_judged")

        def _stale() -> bool:
            return (
                bool(
                    _decision_terminal is not None
                    and getattr(_decision_terminal, "terminal", False)
                )
                or (
                    expected_goals_revision is not None
                    and self.goals.revision != expected_goals_revision
                )
                or (
                    expected_source_generation is not None
                    and self.source_continuity_generation(source) != expected_source_generation
                )
            )
        # Per-frame decision trace (opt-in via on_trace; no-op otherwise). trace_id correlates this
        # frame's stages; derived from source+timestamp — deterministic WHEN t is supplied (as in
        # tests). With t=None it uses the wall clock, like any production run.
        trace_id = f"{source}@{now}"
        tracing = self.on_trace is not None  # guard so the default (untraced) path builds no trace kwargs
        # The judge-usage record emitted deep inside _timed_judge belongs to THIS frame's trace, so
        # park the id where that wrapper can find it (it takes only the judge's own args). One frame
        # is in judgment at a time per core in both dispatch modes, so this is the current frame's id.
        self._usage_trace_id = trace_id
        self._usage_source_ts = now
        # The judgment clock (ADR-001 §1/§5). Two domains, never mixed: every stamp below is read
        # from THIS core's injected ``clock`` (monotonic in production, FakeClock in tests), while the
        # caller's stream timeline ``t`` — when one was supplied — rides along ONLY as the untrusted
        # ``source_media_ts``. ``ingest_observed_ts`` is stamped here, at local receipt of the frame:
        # the honest thing this method knows. Capture-side stamps upstream of the core (a camera's own
        # capture time, the runtime worker's tier-0 admission) are NOT threaded through this seam, so
        # they are deliberately left out rather than approximated from a later read.
        observed_stamps = (
            [
                unit.observed_time
                for units in (_audio_units_by_goal or {}).values()
                for unit in units
                if unit.observed_time is not None
            ]
            if audio_only else []
        )
        ingest_observed_at = max(observed_stamps) if observed_stamps else self.clock()

        if audio_only:
            if tracing:
                from .audio_input import audio_frame_from_units

                audio_payloads = {}
                for goal in active_goals:
                    units = tuple((_audio_units_by_goal or {}).get(goal.id, ()))
                    if not units:
                        continue
                    binding = self._audio_judge_binding_for(goal)
                    provider_audio = audio_frame_from_units(
                        units, capability=binding.capability
                    )
                    mime, payload = provider_audio.as_pair()
                    audio_payloads[goal.id] = {
                        "sha256": hashlib.sha256(
                            mime.lower().encode("ascii") + b"\0" + payload
                        ).hexdigest(),
                        "mime": mime,
                        "bytes": len(payload),
                        "t_start": provider_audio.t_start,
                        "t_end": provider_audio.t_end,
                    }
                await self._emit_trace(
                    trace_id,
                    now,
                    "source.audio",
                    units={
                        goal_id: tuple(
                            (
                                hashlib.sha256(unit.source_id.encode("utf-8")).hexdigest()[:16],
                                unit.source_epoch,
                                unit.source_sequence,
                                unit.seq,
                                unit.t_start,
                                unit.t_end,
                                unit.observed_time,
                                unit.source_time,
                            )
                            for unit in units
                        )
                        for goal_id, units in (_audio_units_by_goal or {}).items()
                    },
                    provider_payloads=audio_payloads,
                )
            focus = None
        else:
            if tracing:
                await self._emit_trace(trace_id, now, "source.frame")
            self._window(source).push(now, frame)
            if _stale():
                return []
            observations = await self.vision.detect(frame, active_goals)
            self.graph.ingest(observations, frame, source, t=now)
            focus = self.graph.in_focus()
            if _stale():
                return []

        events: list[FireEvent] = []
        # ==========================================================================================
        # ONE FRAME, N GOALS: dispatch pass, then apply pass.
        #
        # WHY TWO PASSES. Every admitted goal spends its own judge call, and a call is ~1.7s of
        # PROVIDER wall-clock in which this coroutine has nothing to do. Awaiting them one after
        # another made a frame's makespan the SUM of N calls; because the runtime's judge scheduler
        # holds a single in-flight slot (ADR-001 §9) and admits nothing new until this coroutine
        # returns, that sum WAS the batch period and therefore WAS the onset->incident latency. A
        # 73-goal frame measured 81.6s per batch against a 2s product gate, with 99.6% of admissions
        # superseded — not because any goal was starved (every goal was judged, Gini 0.0) but
        # because they all queued behind each other.
        #
        # Pass 1 walks ``active_goals`` IN ORDER and, for each watch goal, does the (cheap, local)
        # preparation and launches its judge call against a bounded slot pool. Pass 2 walks the SAME
        # list IN THE SAME ORDER and applies each result. So the SUM becomes a MAX while everything
        # order-dependent is untouched: evidence-sequence reservation, funnel accounting, gate and
        # executor mutation, and the order of the returned ``events`` all still follow goal order.
        #
        # FAIRNESS IS UNCHANGED, AND IS THE POINT. Pass 1 launches exactly one call per admitted
        # goal; ``asyncio.Semaphore`` hands out slots first-come-first-served, so a goal late in the
        # list waits for a slot but is never skipped, deprioritized, or dropped. Concurrency here is
        # a queue, never a lottery.
        #
        # WHAT PASS 1 IS NOT ALLOWED TO DO: judge. It reserves sequence numbers and spends funnel
        # counters exactly where the sequential loop did, but every gate/latch/executor mutation
        # stays in pass 2, behind the same ``_stale()`` checks, in the same order.
        # ==========================================================================================
        watch_goals = [
            goal for goal in active_goals
            if goal_behavior_route(goal) is BehaviorRoute.FRAME_WATCH
        ]
        judge_slots = asyncio.Semaphore(self.effective_judge_concurrency(watch_goals))
        dispatched: dict[str, dict] = {}
        try:
            for goal in active_goals:
                if _stale():
                    break  # revision moved during dispatch — launch nothing further under it
                route = goal_behavior_route(goal)
                if audio_only and route is not BehaviorRoute.FRAME_WATCH:
                    raise RuntimeError(
                        f"audio-only execution supports watch goals; {goal.id!r} has route {route.value!r}"
                    )
                if route is not BehaviorRoute.FRAME_WATCH:
                    continue  # transition/process/external routes are driven by the apply pass
                ctx = self.executor.ctx(goal.id)
                # HOW to look = the authored/preset spec, floored by the resolved profile's plan_mode —
                # "window" hands the judge an ordered clip, never a montage — then escalated by this
                # concern's current ladder rung (single→strip→+audio) when it's been recurrently ambiguous
                # (set in _deliberate_confused). Same resolution the transition path now shares via
                # _resolve_strip; strip is the temporal strip per the spec (None = single).
                spec, strip = self._resolve_strip(goal, source, perception_level=ctx.perception_level)
                if audio_only:
                    strip = None
                roi = self._resolve_roi(goal, focus, spec)
                corrections = focus.asserted() if focus else {}
                evidence_units = tuple((_audio_units_by_goal or {}).get(goal.id, ()))
                if audio_only:
                    from .audio_input import audio_frame_from_units

                    binding = self._audio_judge_binding_for(goal)
                    audio = audio_frame_from_units(
                        evidence_units, capability=binding.capability
                    )
                elif (
                    self._watch_runtime_for(goal) is not None
                    and spec.modality == "fused"
                ):
                    from .audio_input import audio_frame_from_units

                    binding = self._audio_judge_binding_for(goal)
                    evidence_units = self._audio_window(source).clip_units(
                        spec.audio_s or self.settings.events.audio_window_s,
                        now=now,
                    )
                    audio = (
                        audio_frame_from_units(
                            evidence_units, capability=binding.capability
                        )
                        if evidence_units else None
                    )
                else:
                    audio = self._audio_clip(source, spec)
                if (
                    self._watch_runtime_for(goal) is not None
                    and spec.modality in ("audio", "fused")
                    and audio is None
                ):
                    raise RuntimeError(
                        f"compiled {spec.modality} Watch {goal.id!r} has no audio evidence in source "
                        f"{source!r}; refusing to judge it as vision-only."
                    )
                self.funnel.judged(concern=goal.id, camera=source)  # a judge call is being spent
                # Item 2 process-QA: opt-in goals carry a per-concern running answer in/out of the judge.
                # GATE-IRRELEVANT — the verdict's satisfied/confidence still drive the (unchanged) gate; the
                # running answer is just threaded through ConcernContext. Default goals pass context=None and
                # are byte-for-byte unchanged.
                process_ctx = (
                    {"process_qa": True, "running_answer": ctx.running_answer}
                    if goal.state.get("process_qa") else None
                )
                # The EvidenceExtractor seam (percept.extraction): ``None`` by default -- every call
                # below is skipped and every judge call stays byte-for-byte. When set, it runs at the
                # SAME point the judge is about to be called (frame + roi already resolved), and its
                # facts are merged into context under the RESERVED "extractor" sub-dict — visible to
                # the judge and, independently, to the trace (so "did the facts reach the judge" is
                # answerable from the .ptrace alone, not just from reading this call site).
                extractor_ctx = None
                if self.extractor is not None and not audio_only:
                    enrichment = self.extractor.extract(goal.condition, frame, roi)
                    if tracing:
                        await self._emit_trace(trace_id, now, "extractor.enriched", goal_id=goal.id,
                                               facts=dict(enrichment.facts))
                    if not enrichment.is_empty:
                        extractor_ctx = {EXTRACTOR_CONTEXT_KEY: enrichment.to_dict()}
                judge_context = (
                    process_ctx if extractor_ctx is None else
                    {**(process_ctx or {}), **extractor_ctx}
                )
                # DirectJudge routing (percept.direct_judge): ``None`` by default -- direct_result stays
                # None, the block below is skipped, and every judge call is byte-for-byte unchanged.
                # CONFIDENT is handled inside the try/except below (it substitutes the judge call
                # itself, spending zero vision calls); UNCERTAIN's facts are merged into the SAME
                # reserved "extractor" context key the EvidenceExtractor seam uses, here, BEFORE the
                # judge call, so a direct rule's partial signal reaches the judge exactly like an
                # extractor's would.
                direct_result = None
                if self.direct_judge is not None and not audio_only:
                    direct_result = self.direct_judge.route(
                        goal.condition, frame, roi,
                        facts=(extractor_ctx or {}).get(EXTRACTOR_CONTEXT_KEY, {}).get("facts", {}),
                        stakes=goal.state.get("stakes", "medium"),
                    )
                    if direct_result.outcome == "uncertain" and direct_result.facts:
                        prior = dict((judge_context or {}).get(EXTRACTOR_CONTEXT_KEY) or {})
                        prior["facts"] = {**prior.get("facts", {}), **direct_result.facts}
                        judge_context = {**(judge_context or {}), EXTRACTOR_CONTEXT_KEY: prior}
                # CLAIM-MODE seam (1.5): mark the judge context when THIS concern resolves to the
                # claim-slot identity policy, so a parsing backend (claude/gemini/openrouter) advertises
                # the correction contract and folds ``operation``/``value`` off the reply. A reserved
                # control key read by _judge (context plumbing) and the backends; every non-claim concern
                # leaves judge_context untouched, byte-for-byte. The fold itself stays authority-gated in
                # _claim_episode_fold — this only lets a LIVE reply reach it.
                if self._resolved_identity_policy(goal) == "claim_slot":
                    judge_context = {**(judge_context or {}), "claim_mode": True}
                # ``rearm=each_event`` is a temporal event-chain contract, not a level poll. The
                # backend must return structured pre/during/post metadata; BehaviorExecutor consumes
                # that metadata to own one session-local episode per distinct occurrence.
                if goal.state.get("rearm") == "each_event":
                    judge_context = {**(judge_context or {}), "event_chain": True}
                    judge_context.update(
                        self._temporal_evidence_context(strip)
                    )
                # Computed BEFORE the judge call (not just on success) so cache, degradation, sequencing,
                # gate, evidence and trace all use ONE effective reducer entity. Only a concern-declared
                # identity_policy="scene" opts out of volatile focus identity; undeclared/profile-only
                # and every other policy preserve the historical focus-scoped reducer.
                reducer_entity_id = watch_reducer_entity_id(
                    goal, focus.id if focus else None
                )
                base_key = (goal.id, reducer_entity_id)
                # Open this judgment's timing record. ``admitted_ts`` is stamped HERE: tier-0
                # detection and focus resolution are done and this concern is about to spend a judge
                # call, which is exactly "admitted to the judge funnel" (ADR-001 §5). ``source_media_ts``
                # is the caller's stream timeline when one was passed and stays None otherwise — it is
                # never back-filled from the monotonic clock, and never enters latency math.
                judgment_timing = JudgmentTimingBuilder(clock=self.clock)
                judgment_timing.mark_ingest_observed(at=ingest_observed_at)
                judgment_timing.mark_admitted()
                source_stamps = [
                    unit.source_time for unit in evidence_units
                    if unit.source_time is not None
                ]
                judgment_timing.set_source_media_ts(
                    max(source_stamps) if audio_only and source_stamps else t
                )
                evidence_seq = self._admit_evidence(base_key)  # reserved at DISPATCH, per ADR-001 §8
                # Runtime-stamped judge provenance for this goal's trace events — additive keys
                # present ONLY when a per-goal binding exists (empty dict otherwise, so an unbound goal's
                # judge.started/finished are byte-for-byte unchanged).
                judge_prov = _judge_binding_trace(self._judge_bindings.get(goal.id))
                if tracing:
                    await self._emit_trace(trace_id, now, "judge.started", goal_id=goal.id,
                                           temporal=getattr(spec, "temporal", None), roi=roi is not None,
                                           **judge_prov)
                if direct_result is not None and direct_result.outcome == "confident":
                    # The confident DirectJudge path: use its normalized Verdict AS the judge's
                    # verdict — self._judge / self.vision are never called (zero vision calls).
                    # The Gate stays sole authority: `verdict` flows into the exact same
                    # freshness/classify/gate path below as a real judge call would. It occupies no
                    # concurrency slot: there is no provider call to bound.
                    judgment_timing.mark_judge_started()
                    self.metrics.incr("direct_judge.hit")
                    direct_verdict = direct_result.verdict
                    judgment_timing.mark_judge_completed()
                    judge_task = None
                else:
                    direct_verdict = None
                    judge_task = asyncio.create_task(
                        self._judge_slot(
                            judge_slots, goal, frame, roi, corrections,
                            entity_id=reducer_entity_id, frames=strip, audio=audio,
                            context=judge_context, stale=_stale, timing=judgment_timing,
                        )
                    )
                    judge_task.add_done_callback(self._consume_batch_task)
                dispatched[goal.id] = {
                    "ctx": ctx,
                    "evidence_units": evidence_units,
                    "process_ctx": process_ctx,
                    "judge_context": judge_context,
                    "reducer_entity_id": reducer_entity_id,
                    "base_key": base_key,
                    "judgment_timing": judgment_timing,
                    "evidence_seq": evidence_seq,
                    "judge_prov": judge_prov,
                    "task": judge_task,
                    "direct_verdict": direct_verdict,
                }

            for goal in active_goals:
                if _stale():
                    break  # revision moved during a prior goal's judge — stop before judging under it
                route = goal_behavior_route(goal)
                if audio_only and route is not BehaviorRoute.FRAME_WATCH:
                    raise RuntimeError(
                        f"audio-only execution supports watch goals; {goal.id!r} has route {route.value!r}"
                    )
                if route is BehaviorRoute.FRAME_TRANSITION:
                    ev = await self._perceive_transition(
                        goal, frame, focus, now=now, t=t, ingest_observed_at=ingest_observed_at,
                        trace_id=trace_id, tracing=tracing, source=source, stale=_stale,
                    )
                    if ev is not None:
                        events.append(ev)
                    continue
                if route is BehaviorRoute.FRAME_PROCESS:
                    # Item 4: log a delta, never fire here
                    await self._observe_and_log(goal, frame, source, now, stale=_stale)
                    continue
                if route is not BehaviorRoute.FRAME_WATCH:
                    continue  # external primitives are driven by their dedicated public APIs
                prepared = dispatched.get(goal.id)
                if prepared is None:
                    break  # the dispatch pass stopped here (staleness); nothing was launched beyond it
                ctx = prepared["ctx"]
                evidence_units = prepared["evidence_units"]
                process_ctx = prepared["process_ctx"]
                judge_context = prepared["judge_context"]
                reducer_entity_id = prepared["reducer_entity_id"]
                base_key = prepared["base_key"]
                judgment_timing = prepared["judgment_timing"]
                evidence_seq = prepared["evidence_seq"]
                judge_prov = prepared["judge_prov"]
                judge_task = prepared["task"]
                try:
                    verdict = (
                        prepared["direct_verdict"] if judge_task is None
                        else await judge_task
                    )
                    if verdict is _BATCH_STALE:
                        break  # queued behind the bound when the revision moved: never dispatched
                except _JudgeDegraded as degraded:
                    if _stale():
                        break  # stale transport/goal work cannot alter the degradation breaker either
                    # Fail-closed boundary (ADR-001 §10 invariant 4): the judge call raised (timeout /
                    # provider error) or returned a malformed/partial response. NEVER read .confidence/
                    # .satisfied off that — degrade THIS goal's tick to an explicit abstention (ASK-class:
                    # same telemetry a real low-confidence read gets), never a FIRE, and never let the gate
                    # see it at all. Fail-closed bookkeeping preserves the armed/disarmed edge and
                    # last-fire/refractory memory, but clears pending positive-dwell support: a failed read
                    # cannot prove continuity. No-hold state is unchanged. The exception is swallowed HERE
                    # — it must never escape to worker.py's run() loop and end the watch.
                    self.funnel.refused(concern=goal.id, camera=source)
                    self._degrade_tick(base_key, degraded.cause, goal_id=goal.id, label="",
                                       action="degrading this tick to ASK/SILENT")
                    if tracing:
                        await self._emit_trace(
                            trace_id, now, "judge.degraded", goal_id=goal.id, cause=repr(degraded.cause),
                        )
                        await self._emit_trace(trace_id, now, "gate.classified", goal_id=goal.id,
                                               truth="unknown", action="ask",
                                               entity_id=reducer_entity_id, degraded=True)
                    continue  # next goal this SAME frame; the watch loop (and run) continues either way
                if _stale():
                    break  # source/goal generation moved while the judge was in flight: commit nothing
                provider_verdict = verdict
                clip_events_present = bool(
                    goal.state.get("rearm") == "each_event"
                    and verdict.events
                )
                if (
                    goal.state.get("rearm") == "each_event"
                    and not clip_events_present
                ):
                    verdict = normalize_event_chain_verdict(verdict)
                self._note_judge_ok(base_key)  # a well-formed verdict resets this key's degrade streak
                resolved: FireEvent | None = None
                if process_ctx is not None and getattr(verdict, "running_answer", ""):
                    ctx.running_answer = verdict.running_answer  # accumulate the model's rewritten summary
                    # Accumulated-coverage stamp (answer-path only): each evidence-bearing observation that
                    # folded into the running answer is a unit of coverage. Reuse the per-concern process_log
                    # the boundary already owns, so len(process_log) IS the evidence count — no gate state.
                    # Bounded so a 24/7 process_qa concern that never hits a boundary can't leak memory; the
                    # resolver only reads it as a >= floor count, so a cap never weakens resolution.
                    ctx.process_log.append(verdict.running_answer)
                    if len(ctx.process_log) > 256:
                        del ctx.process_log[: len(ctx.process_log) - 256]
                    ctx.last_step_at = now
                    # RESOLVE the running answer once it's complete/confident enough (general — coverage +
                    # confidence, never the question text). De-duped through the UNCHANGED gate on a
                    # dedicated __resolved__ key, exactly like the coverage/episode/overdue absence fires.
                    resolved = self._resolve_process_qa(goal, ctx, verdict.confidence)
                freshness = self._freshness(focus, now, goal.id)
                truth = classify(verdict, freshness, self.gate.threshold, self.gate.fresh_min)
                if truth == "unknown":
                    self.funnel.refused(concern=goal.id, camera=source)  # abstained — refused to guess (ASK)
                else:
                    self.last_supported[goal.id] = now  # a confident read = perceptual coverage
                    ctx.perception_level = 0  # decay: the (escalated) look resolved → back to the cheap default
                if truth == "yes" or clip_events_present:
                    self.last_satisfied[goal.id] = now  # deadline: the expected event happened
                # base_key computed earlier (before the judge call) — reused here unchanged.
                # GOVERNING for content-key, per-entity-identity (its SCENE occurrence) and claim-slot
                # concerns; every other concern gets fold=None (byte-for-byte, unchanged). A goal is
                # never more than one of those types (identity_policy selects exactly one), so at most
                # one of these returns non-None.
                doc_fold = self._document_episode_fold(
                    goal, verdict, now, truth=truth, key=base_key
                )
                claim_fold = self._claim_episode_fold(goal, verdict, now)
                gate_outcomes: list[GateOutcome] = []
                if clip_events_present:
                    clip_fires = self._fire_clip_occurrences(
                        goal,
                        verdict,
                        base_key,
                        source_t=now,
                        context=judge_context,
                        freshness=freshness,
                        timing=judgment_timing,
                        evidence_seq=evidence_seq,
                        outcome_sink=gate_outcomes.append,
                    )
                    ev_fire = clip_fires[-1] if clip_fires else None
                else:
                    ev_fire = self._fire_concern(
                        goal, verdict, base_key, freshness=freshness, t=now,
                        timing=judgment_timing, evidence_seq=evidence_seq,
                        fold=doc_fold if doc_fold is not None else claim_fold,
                        outcome_sink=gate_outcomes.append,
                    )
                    clip_fires = [] if ev_fire is None else [ev_fire]
                for emitted in clip_fires:
                    if audio_only:
                        emitted.audio_units = evidence_units
                    if emitted.action == FIRE:
                        self.funnel.fired(concern=goal.id, camera=source)  # a rising-edge fire emitted
                    events.append(emitted)
                if resolved is not None:  # the process-QA running answer crossed the bar this tick
                    self.funnel.fired(concern=goal.id, camera=source)
                    events.append(resolved)
                if tracing:
                    # Provider completion through cognition/action-log commit above is one
                    # synchronous owner segment.  Trace callbacks run only after that commit,
                    # so a tracer cannot change the Watch revision between stale validation
                    # and gate/executor mutation.
                    await self._emit_trace(
                        trace_id,
                        now,
                        "judge.finished",
                        goal_id=goal.id,
                        satisfied=getattr(provider_verdict, "satisfied", None),
                        confidence=getattr(provider_verdict, "confidence", None),
                        **self._event_chain_provider_advisory(
                            goal, provider_verdict
                        ),
                        **judge_prov,
                    )
                    # AFTER both fire paths: label a FIRE whenever an incident was actually emitted this tick
                    # — either the gate fired OR a resolved process-QA answer crossed the bar (which fires even
                    # when the gate abstained/ASK'd on this same tick). Only ask/silent when nothing fired.
                    if resolved is not None or (ev_fire is not None and ev_fire.action == FIRE):
                        action = "fire"
                    elif ev_fire is not None:
                        action = ev_fire.action
                    else:
                        action = "silent"
                    # The gate's own INPUTS ride along with its output, so a recorded trace is
                    # replayable through the (deterministic) gate without the original frame or any
                    # model call — see percept.replay. Purely additive trace data; nothing branches on it.
                    gate_reason = (
                        gate_outcomes[-1].reason
                        if gate_outcomes and resolved is None
                        else None
                    )
                    await self._emit_trace(trace_id, now, "gate.classified", goal_id=goal.id,
                                           truth=truth, action=action,
                                           reason=_trace_reason(truth, action, gate_reason),
                                           gate_reason=gate_reason,
                                           entity_id=reducer_entity_id,
                                           satisfied=getattr(verdict, "satisfied", None),
                                           confidence=getattr(verdict, "confidence", None),
                                           freshness=freshness,
                                           hold_s=program_of(goal).hold_s,
                                           **self._event_chain_advisory(
                                               goal, reducer_entity_id, verdict
                                           ))
                    # Project the frozen JudgmentEvent (the one production construction site,
                    # behavior.py:_judgment) into the trace stream so the SAME evidence this decision
                    # judged is resolvable from the .ptrace. Only when THIS tick's fire actually carried
                    # one (the judged watch path only — a degraded tick never reaches here, and
                    # episode/rhythm/boundary/absence fires take no judge-funnel stamps and build no
                    # JudgmentEvent at all; live transition fires emit their row in
                    # ``_perceive_transition``). The RECORDER (not this caller) stamps
                    # ``data["evidence_seq"]`` as the seq of this SAME trace_id's ``source.frame``
                    # record; ``dispatch_seq`` carries the JudgmentEvent's OWN ADR-001 §8 per-key dispatch
                    # sequence, a distinct number, so neither meaning is lost (see
                    # tracing.STAGE_JUDGMENT_EVENT). Purely additive observability — governs nothing.
                    for emitted in clip_fires:
                        if emitted.judgment is None:
                            continue
                        await self._emit_trace(
                            trace_id, now, STAGE_JUDGMENT_EVENT, goal_id=goal.id,
                            action=emitted.judgment.action,
                            dispatch_seq=emitted.judgment.evidence_seq,
                            entity_id=reducer_entity_id,
                            # The OPAQUE, INJECTIVE evidence link a client audit resolves: the SAME handle the
                            # delivered Incident carries (make_incident derives it from the same goal/entity/
                            # seq), so exactly one judgment.event row answers each incident.created. A None
                            # evidence_origin on the judged path reads as "judged" — stamp it explicitly here.
                            judgment_event_id=judgment_event_id(
                                goal.id,
                                reducer_entity_id,
                                emitted.judgment.evidence_seq,
                                emitted.judgment.occurrence_id,
                            ),
                            evidence_origin=(
                                emitted.judgment.evidence_origin or "judged"
                            ),
                            **(
                                {
                                    "occurrence_revision_id": emitted.revision_id,
                                    "occurrence_source_t": (
                                        emitted.judgment.timing.source_media_ts
                                    ),
                                }
                                if clip_events_present
                                else {}
                            ),
                        )
                    # SHADOW — emit the subject + the fold the model WOULD have proposed, as ADVISORY
                    # trace data. It changes NOTHING (computed after the gate already classified, governs
                    # no firing) and lands in the ``shadow.fold`` stage so a client's media-OFF .ptrace
                    # carries something gradeable. Only built when a tracer is attached (default path does
                    # no shadow work at all).
                    await self._emit_trace(trace_id, now, STAGE_SHADOW_FOLD, goal_id=goal.id,
                                           **self._shadow_observe(verdict).to_advisory())
        finally:
            # Whatever the apply pass did not consume — a mid-loop ``break`` on staleness, a raised
            # binding invariant, a cancelled frame — must not outlive this call holding a provider
            # connection. Cancelling here is what keeps "the batch is bounded" true in the failure
            # paths too; a cancellation-hostile adapter still shows up as route occupancy in the
            # pool's own snapshot rather than as a leaked task.
            for prepared in dispatched.values():
                task = prepared["task"]
                if task is not None and not task.done():
                    task.cancel()
        self._log_fires(events, now)
        if _stale():
            return events  # aborted mid-loop: skip auto-refinement (it authors goal changes)
        if not audio_only:
            await self._deliberate_confused(now)
        return events

    async def perceive_audio_judged(
        self,
        audio_units_by_goal: "dict[str, tuple[AudioUnit, ...]]",
        source: str = "camera",
        *,
        t: float | None = None,
        expected_goals_revision: int | None = None,
        expected_source_generation: int | None = None,
        goal_ids: "tuple[str, ...] | None" = None,
        _decision_terminal: Any = None,
    ) -> list[FireEvent]:
        """Judge exact frozen audio evidence without synthesizing a video frame."""
        from .audio_input import canonicalize_audio_unit

        frozen = {}
        for goal_id, units in audio_units_by_goal.items():
            goal = self.goals.get(goal_id)
            if goal is None:
                raise ValueError(f"audio evidence names unknown goal {goal_id!r}")
            binding = self._audio_judge_binding_for(goal)
            checked = []
            for unit in tuple(units):
                if (
                    type(unit) is not AudioUnit
                    or type(unit.source_id) is not str
                    or not unit.source_id
                    or unit.source_epoch is None
                    or unit.source_sequence is None
                    or unit.observed_time is None
                ):
                    raise ValueError("audio judgment requires provenance-complete exact AudioUnits")
                checked.append(
                    canonicalize_audio_unit(
                        unit,
                        capability=binding.capability,
                        source_id=unit.source_id,
                        source_epoch=unit.source_epoch,
                        source_sequence=unit.source_sequence,
                        observed_time=unit.observed_time,
                        source_time=unit.source_time,
                    )
                )
            if not checked:
                raise ValueError(f"audio evidence for goal {goal_id!r} is empty")
            frozen[goal_id] = tuple(checked)
        return await self.perceive_judged(
            None,
            source,
            t=t,
            expected_goals_revision=expected_goals_revision,
            expected_source_generation=expected_source_generation,
            goal_ids=goal_ids,
            _audio_units_by_goal=frozen,
            _decision_terminal=_decision_terminal,
        )

    def prepare_fused_judgment(
        self,
        bundle: Any,
        *,
        goal_id: str,
        source: str,
        source_t: float,
        goals_revision: int,
        source_generation: int,
        fused_generation: int,
    ) -> "_JudgmentJob | None":
        """Owner-only synchronous admission: freeze every provider input and token."""
        import copy
        from types import MappingProxyType

        from .audio_input import audio_frame_from_units
        from .fused_sync import _FusedEvidenceBundle

        if type(bundle) is not _FusedEvidenceBundle:
            raise ValueError("fused admission requires an exact frozen bundle")
        goal = self.goals.get(goal_id)
        if goal is None or program_of(goal).perception.modality != "fused":
            return None
        if self.goals.revision != goals_revision:
            return None
        binding = self._audio_judge_binding_for(goal)
        runtime = self._watch_runtime_for(goal)
        if (
            runtime is None
            or bundle.watch_application != binding.application_key
            or bundle.watch_revision != goals_revision
            or bundle.video_units[-1].generation != fused_generation
        ):
            raise ValueError("fused bundle application does not match its immutable judge binding")
        # Complete every fallible projection before reserving sequence/latest
        # admission state.  Admission publication below is one synchronous,
        # non-raising owner segment.
        anchor = bundle.video_units[-1]
        observed = [
            unit.observed_end for unit in (*bundle.video_units, *bundle.audio_units)
        ]
        audio = (
            audio_frame_from_units(
                tuple(unit.payload for unit in bundle.audio_units),
                capability=binding.capability,
            )
            if bundle.complete
            else None
        )
        exemplar = copy.deepcopy(goal.state.get("exemplar"))
        provider_context: dict[str, Any] = {
            "_percept_frames_are_earlier": True
        }
        if self._resolved_identity_policy(goal) == "claim_slot":
            provider_context["claim_mode"] = True
        if goal.state.get("rearm") == "each_event":
            provider_context["event_chain"] = True
        frozen_source_t = float(source_t)
        ingest_observed_ts = max(observed) if observed else self.clock()
        admitted_ts = self.clock()
        base_key = (goal.id, None)
        evidence_seq = self._admit_evidence(base_key)
        bundle = bundle.with_judgment_identity(
            evidence_seq=evidence_seq,
            judgment_event_id=judgment_event_id(goal.id, None, evidence_seq),
        )
        counter = getattr(self, "_fused_admission_counter", 0) + 1
        self._fused_admission_counter = counter
        parent_trace_id = (
            f"{anchor.source_id}:{anchor.source_epoch}:{anchor.source_sequence}:"
            f"{anchor.generation}"
        )
        admission_id = f"{goal.id}:{goals_revision}:{counter}"
        token = _JudgmentAdmissionToken(
            admission_id=admission_id,
            goal_id=goal.id,
            modality="fused",
            route=goal_behavior_route(goal).value,
            goal_object_token=goal,
            watch_application=binding.application_key,
            watch_revision=bundle.watch_revision,
            goals_registry_revision=goals_revision,
            watch_runtime_token=runtime[0],
            judge_binding_token=binding.binding_token,
            source_id=source,
            source_generation=source_generation,
            fused_generation=fused_generation,
            evidence_seq=evidence_seq,
            parent_source_trace_id=parent_trace_id,
            judgment_trace_id=f"{parent_trace_id}:{goal.id}:{evidence_seq}",
        )
        latest = getattr(self, "_fused_latest_admission", None)
        if latest is None:
            latest = {}
            self._fused_latest_admission = latest
        latest[base_key] = admission_id
        reduced = getattr(self, "_fused_reduced_admissions", None)
        if reduced is None:
            reduced = {}
            self._fused_reduced_admissions = reduced
        ledger_max = getattr(
            self, "_fused_admission_ledger_max", _FUSED_ADMISSION_LEDGER_MAX
        )
        while len(latest) > ledger_max:
            evicted_key = next(iter(latest))
            latest.pop(evicted_key, None)
            # Admission and reduction identity are one replay firewall.
            # Evicting both means an old completion fails latest-admission
            # validation; retaining one side independently is forbidden.
            reduced.pop(evicted_key, None)
        return _JudgmentJob(
            token=token,
            modality="fused",
            evidence=bundle,
            provider=_ProviderRequest(
                dispatch=binding.dispatch,
                condition=str(goal.condition),
                frame=anchor.payload,
                frames=tuple(unit.payload for unit in bundle.video_units[:-1]),
                audio=audio,
                roi=None,
                exemplar=exemplar,
                corrections=(),
                context=MappingProxyType(provider_context),
                include_frames=True,
                include_audio=True,
                include_context=True,
            ),
            reducer=_ReducerInputs(
                source_t=frozen_source_t,
                ingest_observed_ts=ingest_observed_ts,
                admitted_ts=admitted_ts,
                reducer_entity_id=None,
                freshness=1.0,
            ),
            adapter=binding.adapter,
            trace_temporal=program_of(goal).perception.temporal,
            trace_roi=False,
        )

    async def _dispatch_provider_request(
        self,
        request: "_ProviderRequest",
    ) -> "_ProviderCallCompletion":
        """Execute one frozen request without reading or mutating cognition."""
        if type(request) is not _ProviderRequest:
            raise ValueError("provider dispatch requires an exact frozen request")
        failure = None
        usage: dict[str, Any] = {}
        judge_latency_s = 0.0
        judge_started_ts = self.clock()
        provider_started = self.clock()
        try:
            kwargs: dict[str, Any] = {
                "roi": _provider_input_copy(request.roi),
                "corrections": dict(request.corrections) or None,
            }
            if request.include_exemplar:
                kwargs["exemplar"] = _provider_input_copy(request.exemplar)
            if request.include_frames:
                kwargs["frames"] = (
                    list(_provider_input_copy(request.frames)) or None
                )
            if request.include_audio:
                kwargs["audio"] = _provider_input_copy(request.audio)
            if request.include_context:
                kwargs["context"] = _provider_input_copy(request.context)
            provider_verdict = await request.dispatch(
                request.condition,
                _provider_input_copy(request.frame),
                **kwargs,
            )
        except Exception as exc:  # noqa: BLE001 - one provider completion fails closed
            failure = _ProviderFailure(
                kind="provider_failure",
                exception_type=type(exc).__name__,
                safe_message="judgment provider unavailable",
            )
            verdict = _VerdictSnapshot.capture(
                Verdict(
                    satisfied=False,
                    confidence=0.0,
                    evidence="judgment provider unavailable",
                )
            )
        else:
            # _timed_judge historically records latency/usage after any
            # provider RETURN and only then does _safe_timed_judge reject a
            # malformed result. Preserve that factual receipt while keeping
            # the invalid object out of cognition.
            usage = judge_usage(provider_verdict)
            if not _is_valid_verdict(provider_verdict):
                failure = _ProviderFailure(
                    kind="malformed_response",
                    exception_type=type(provider_verdict).__name__,
                    safe_message="judgment provider returned malformed data",
                )
                verdict = _VerdictSnapshot.capture(
                    Verdict(
                        satisfied=False,
                        confidence=0.0,
                        evidence="judgment provider returned malformed data",
                    )
                )
            else:
                verdict = _VerdictSnapshot.capture(provider_verdict)
        judge_latency_s = self.clock() - provider_started
        return _ProviderCallCompletion(
            verdict=verdict,
            failure=failure,
            usage=tuple(usage.items()),
            judge_latency_s=judge_latency_s,
            judge_started_ts=judge_started_ts,
            judge_completed_ts=self.clock(),
        )

    async def dispatch_judgment(
        self,
        job: "_JudgmentJob",
    ) -> "_JudgmentCompletion":
        """Provider-only phase: no live goal, binding, gate, trace, or metrics reads."""
        if type(job) is not _JudgmentJob:
            raise ValueError("provider dispatch requires an exact admitted job")
        job = self.launch_judgment(job)
        if job.local_verdict is not None:
            return self.complete_judgment_locally(job)
        if type(job.provider) is _TransitionProviderRequest:
            before = await self._dispatch_provider_request(job.provider.before)
            after = (
                None
                if before.failure is not None
                else await self._dispatch_provider_request(job.provider.after)
            )
            result: (
                _WatchJudgmentResult
                | _ProcessJudgmentResult
                | _TransitionJudgmentResult
            ) = _TransitionJudgmentResult(before=before, after=after)
        elif type(job.provider) is _ProviderRequest:
            from .fused_sync import _FusedEvidenceBundle

            if (
                type(job.bundle) is _FusedEvidenceBundle
                and not job.bundle.complete
            ):
                now = self.clock()
                call = _ProviderCallCompletion(
                    verdict=_VerdictSnapshot.capture(
                        Verdict(
                            satisfied=False,
                            confidence=0.0,
                            evidence="fused coverage uncertainty: "
                            + ",".join(
                                reason.value for reason in job.bundle.coverage_reasons
                            ),
                        )
                    ),
                    failure=None,
                    usage=(),
                    judge_latency_s=0.0,
                    judge_started_ts=now,
                    judge_completed_ts=now,
                )
            else:
                call = await self._dispatch_provider_request(job.provider)
            result = (
                _ProcessJudgmentResult(call)
                if job.token.route == BehaviorRoute.FRAME_PROCESS.value
                else _WatchJudgmentResult(call)
            )
        else:
            raise ValueError("provider dispatch requires a typed provider request")
        return _JudgmentCompletion(
            job=job,
            result=result,
        )

    def complete_judgment_locally(
        self,
        job: "_JudgmentJob",
        verdict: "Verdict | None" = None,
    ) -> "_JudgmentCompletion":
        """Enter cache/direct local results into the generic completion protocol."""
        if type(job) is not _JudgmentJob:
            raise ValueError("local completion requires an exact judgment job")
        job = self.launch_judgment(job)
        snapshot = (
            _VerdictSnapshot.capture(verdict)
            if verdict is not None
            else job.local_verdict
        )
        if snapshot is None:
            raise ValueError("local completion requires a frozen local verdict")
        now = self.clock()
        return _JudgmentCompletion(
            job=job,
            result=_WatchJudgmentResult(
                _ProviderCallCompletion(
                    verdict=snapshot,
                    failure=None,
                    usage=(),
                    judge_latency_s=0.0,
                    judge_started_ts=now,
                    judge_completed_ts=now,
                    provider_spent=False,
                )
            ),
        )

    async def dispatch_fused_judgment(
        self,
        job: "_JudgmentJob",
    ) -> "_JudgmentCompletion":
        """Compatibility-named fused entrypoint over the generic dispatcher."""
        if type(job) is not _JudgmentJob or job.modality != "fused":
            raise ValueError("fused provider dispatch requires an exact admitted job")
        return await self.dispatch_judgment(job)

    def complete_fused_locally(
        self,
        job: "_JudgmentJob",
    ) -> "_JudgmentCompletion":
        """Build an uncertain no-provider completion without occupying a lane."""
        if type(job) is not _JudgmentJob or job.modality != "fused" or job.bundle.complete:
            raise ValueError("local fused completion requires one uncertain admitted job")
        now = self.clock()
        verdict = _VerdictSnapshot.capture(
            Verdict(
                satisfied=False,
                confidence=0.0,
                evidence="fused coverage uncertainty: "
                + ",".join(reason.value for reason in job.bundle.coverage_reasons),
            )
        )
        return _JudgmentCompletion(
            job=job,
            result=_WatchJudgmentResult(
                _ProviderCallCompletion(
                    verdict=verdict,
                    failure=None,
                    usage=(),
                    judge_latency_s=0.0,
                    judge_started_ts=now,
                    judge_completed_ts=now,
                )
            ),
        )

    def reduce_fused_completion(
        self,
        completion: "_JudgmentCompletion | None",
        *,
        live_fused_generation: int,
    ) -> list[FireEvent]:
        """Owner-only token validation and synchronous cognition commit."""
        from .fused_sync import _FusedEvidenceBundle

        if completion is None:
            return []
        if type(completion) is not _JudgmentCompletion:
            raise ValueError("fused reducer requires an exact provider completion")
        job = completion.job
        token = job.token
        goal = self.goals.get(token.goal_id)
        runtime = self._watch_runtime_for(token.goal_id)
        binding = self._audio_judge_bindings.get(token.goal_id)
        base_key = (token.goal_id, None)
        latest = getattr(self, "_fused_latest_admission", {})
        reduced = getattr(self, "_fused_reduced_admissions", None)
        if reduced is None:
            reduced = {}
            self._fused_reduced_admissions = reduced
        stale_reason = None
        if (
            goal is None
            or goal is not token.goal_object_token
            or token.modality != "fused"
            or program_of(goal).perception.modality != token.modality
            or goal_behavior_route(goal).value != token.route
        ):
            stale_reason = "goal_revision"
        elif self.goals.revision != token.goals_registry_revision:
            stale_reason = "goal_revision"
        elif runtime is None or runtime[0] is not token.watch_runtime_token:
            stale_reason = "runtime_binding"
        elif (
            binding is None
            or binding.binding_token is not token.judge_binding_token
            or self._audio_judge_binding_tokens.get(token.goal_id)
            is not token.judge_binding_token
        ):
            stale_reason = "judge_binding"
        elif (
            self.source_continuity_generation(token.source_id)
            != token.source_generation
        ):
            stale_reason = "source_generation"
        elif live_fused_generation != token.fused_generation:
            stale_reason = "fused_generation"
        elif (
            latest.get(base_key) != token.admission_id
            or self._sequencer.latest_admitted(base_key) != token.evidence_seq
        ):
            stale_reason = "superseded_admission"
        elif reduced.get(base_key) == token.admission_id:
            stale_reason = "duplicate_completion"
        if stale_reason is not None:
            trace_queue = getattr(self, "_fused_trace_queue", None)
            if trace_queue is None:
                trace_queue = []
                self._fused_trace_queue = trace_queue
            trace_queue.append(
                (
                    token.judgment_trace_id,
                    job.source_t,
                    "judgment.dropped",
                    {
                        "goal_id": token.goal_id,
                        "admission_id": token.admission_id,
                        "evidence_seq": token.evidence_seq,
                        "reason": stale_reason,
                        "slot": "completed",
                    },
                )
            )
            return []
        bundle = job.bundle
        if type(bundle) is not _FusedEvidenceBundle:
            raise ValueError("fused completion lost its exact evidence bundle")
        if (
            bundle.watch_application != token.watch_application
            or bundle.watch_revision != token.watch_revision
            or bundle.evidence_seq != token.evidence_seq
            or bundle.judgment_event_id
            != judgment_event_id(goal.id, None, token.evidence_seq)
        ):
            trace_queue = getattr(self, "_fused_trace_queue", None)
            if trace_queue is None:
                trace_queue = []
                self._fused_trace_queue = trace_queue
            trace_queue.append(
                (
                    token.judgment_trace_id,
                    job.source_t,
                    "judgment.dropped",
                    {
                        "goal_id": token.goal_id,
                        "admission_id": token.admission_id,
                        "evidence_seq": token.evidence_seq,
                        "reason": "evidence_identity",
                        "slot": "completed",
                    },
                )
            )
            return []
        reduced[base_key] = token.admission_id
        verdict = completion.verdict.to_verdict()
        timing = JudgmentTimingBuilder(clock=self.clock)
        timing.mark_ingest_observed(at=job.ingest_observed_ts)
        timing.mark_admitted(at=job.admitted_ts)
        timing.mark_judge_started(at=completion.judge_started_ts)
        timing.mark_judge_completed(at=completion.judge_completed_ts)
        timing.set_source_media_ts(job.source_t)
        usage = dict(completion.usage)
        if bundle.complete:
            self.funnel.judged(concern=goal.id, camera=token.source_id)
            self.funnel.record_judge(
                latency_s=completion.judge_latency_s,
                **usage,
            )
        if completion.failure is not None:
            self.funnel.refused(concern=goal.id, camera=token.source_id)
            self._degrade_tick(
                base_key,
                completion.failure.safe_message,
                goal_id=goal.id,
                label="",
                action="degrading this fused tick to ASK",
            )
            trace_queue = getattr(self, "_fused_trace_queue", None)
            if trace_queue is None:
                trace_queue = []
                self._fused_trace_queue = trace_queue
            if self.on_trace is not None:
                trace_queue.extend(
                    [
                        (
                            token.judgment_trace_id,
                            job.source_t,
                            "source.fused",
                            {
                                "goal_id": goal.id,
                                "judgment_event_id": bundle.judgment_event_id,
                                "evidence_seq": bundle.evidence_seq,
                                "policy_version": bundle.policy_version,
                                "coverage_reasons": tuple(
                                    reason.value
                                    for reason in bundle.coverage_reasons
                                ),
                            },
                        ),
                        (
                            token.judgment_trace_id,
                            job.source_t,
                            "judge.degraded",
                            {
                                "goal_id": goal.id,
                                "cause": completion.failure.safe_message,
                                "failure_code": completion.failure.kind,
                                "exception_type": completion.failure.exception_type,
                            },
                        ),
                        (
                            token.judgment_trace_id,
                            job.source_t,
                            "gate.classified",
                            {
                                "goal_id": goal.id,
                                "truth": "unknown",
                                "action": "ask",
                                "entity_id": None,
                                "degraded": True,
                            },
                        ),
                    ]
                )
            return []
        provider_verdict = verdict
        if goal.state.get("rearm") == "each_event":
            verdict = normalize_event_chain_verdict(verdict)
        if bundle.complete:
            self._note_judge_ok(base_key)
        truth = classify(
            verdict,
            1.0,
            self.gate.threshold,
            self.gate.fresh_min,
        )
        if truth != "unknown":
            self.last_supported[goal.id] = job.source_t
        if truth == "yes":
            self.last_satisfied[goal.id] = job.source_t
        doc_fold = self._document_episode_fold(
            goal, verdict, job.source_t, truth=truth, key=base_key
        )
        claim_fold = self._claim_episode_fold(
            goal, verdict, job.source_t
        )
        gate_outcomes: list[GateOutcome] = []
        fire = self._fire_concern(
            goal,
            verdict,
            base_key,
            freshness=1.0,
            t=job.source_t,
            timing=timing,
            evidence_seq=token.evidence_seq,
            fold=doc_fold if doc_fold is not None else claim_fold,
            outcome_sink=gate_outcomes.append,
        )
        events: list[FireEvent] = []
        if fire is not None:
            fire._fused_evidence_bundle = bundle
            if fire.action == FIRE:
                self.funnel.fired(concern=goal.id, camera=token.source_id)
            else:
                self.funnel.refused(concern=goal.id, camera=token.source_id)
            events.append(fire)
        trace_queue = getattr(self, "_fused_trace_queue", None)
        if trace_queue is None:
            trace_queue = []
            self._fused_trace_queue = trace_queue
        if self.on_trace is not None:
            action = fire.action if fire is not None else "silent"
            gate_reason = gate_outcomes[-1].reason if gate_outcomes else None
            trace_queue.extend(
                [
                    (
                        token.judgment_trace_id,
                        job.source_t,
                        "source.fused",
                        {
                            "goal_id": goal.id,
                            "judgment_event_id": bundle.judgment_event_id,
                            "evidence_seq": bundle.evidence_seq,
                            "policy_version": bundle.policy_version,
                            "actual_skew_s": bundle.actual_skew_s,
                            "video_watermark": bundle.video_watermark,
                            "audio_watermark": bundle.audio_watermark,
                            "coverage_reasons": tuple(
                                reason.value for reason in bundle.coverage_reasons
                            ),
                        },
                    ),
                    (
                        token.judgment_trace_id,
                        job.source_t,
                        "judge.finished",
                        {
                            "goal_id": goal.id,
                            "satisfied": provider_verdict.satisfied,
                            "confidence": provider_verdict.confidence,
                            "fused": True,
                            **self._event_chain_provider_advisory(
                                goal, provider_verdict
                            ),
                        },
                    ),
                    (
                        token.judgment_trace_id,
                        job.source_t,
                        "gate.classified",
                        {
                            "goal_id": goal.id,
                            "truth": truth,
                            "action": action,
                            "reason": _trace_reason(truth, action, gate_reason),
                            "gate_reason": gate_reason,
                            "entity_id": None,
                            "satisfied": verdict.satisfied,
                            "confidence": verdict.confidence,
                            "freshness": 1.0,
                            "hold_s": program_of(goal).hold_s,
                            **self._event_chain_advisory(
                                goal, None, verdict
                            ),
                        },
                    ),
                ]
            )
            if usage:
                trace_queue.append(
                    (
                        token.judgment_trace_id,
                        job.source_t,
                        "judge.usage",
                        {
                            "model": self._judge_model(job.adapter),
                            **usage,
                        },
                    )
                )
            if fire is not None and fire.judgment is not None:
                trace_queue.append(
                    (
                        token.judgment_trace_id,
                        job.source_t,
                        STAGE_JUDGMENT_EVENT,
                        {
                            "goal_id": goal.id,
                            "action": fire.judgment.action,
                            "dispatch_seq": fire.judgment.evidence_seq,
                            "entity_id": None,
                            "judgment_event_id": bundle.judgment_event_id,
                            "evidence_origin": fire.judgment.evidence_origin or "judged",
                            "fused_policy_version": bundle.policy_version,
                            "fused_actual_skew_s": bundle.actual_skew_s,
                            "fused_video_watermark": bundle.video_watermark,
                            "fused_audio_watermark": bundle.audio_watermark,
                            "fused_coverage_reasons": tuple(
                                reason.value for reason in bundle.coverage_reasons
                            ),
                            "fused_judgment_event_id": bundle.judgment_event_id,
                            "fused_evidence_seq": bundle.evidence_seq,
                        },
                    )
                )
            trace_queue.append(
                (
                    token.judgment_trace_id,
                    job.source_t,
                    STAGE_SHADOW_FOLD,
                    {
                        "goal_id": goal.id,
                        **self._shadow_observe(verdict).to_advisory(),
                    },
                )
            )
            if len(trace_queue) > 1024:
                del trace_queue[: len(trace_queue) - 1024]
        self._log_fires(events, job.source_t)
        return events

    def drain_fused_trace_records(self) -> list[tuple[str, float, str, dict]]:
        records = getattr(self, "_fused_trace_queue", [])
        self._fused_trace_queue = []
        return records

    def record_fused_admission_drop(
        self,
        job: "_JudgmentJob",
        *,
        reason: str,
        slot: str = "admitted",
    ) -> None:
        if (
            type(job) is not _JudgmentJob
            or not reason
            or slot not in ("admitted", "pending", "completed")
        ):
            raise ValueError("fused admission drop requires an exact job and reason")
        queue = getattr(self, "_fused_trace_queue", None)
        if queue is None:
            queue = []
            self._fused_trace_queue = queue
        queue.append(
            (
                job.token.judgment_trace_id,
                job.source_t,
                "judgment.dropped",
                {
                    "goal_id": job.token.goal_id,
                    "admission_id": job.token.admission_id,
                    "evidence_seq": job.token.evidence_seq,
                    "reason": reason,
                    "slot": slot,
                },
            )
        )

    async def _perceive_transition(
        self, goal, frame, focus, *, now: float, t: float | None, ingest_observed_at: float,
        trace_id: str, tracing: bool, source: str, stale: Callable[[], bool] | None = None,
    ) -> FireEvent | None:
        """Fire on a state change A->B, but ONLY after the 'from' state A was witnessed.

        A transition goal carries ``state={"type":"transition","from":<cond A>,"to":<cond B>}``.
        Per frame we judge BOTH states; observing a confident A arms (and re-arms) the goal, and a
        confident B *after* A has been seen fires once. This is what a single-frame judge can't do
        ("waking up" looks like "awake"): you must have seen the *before* state. Joining mid-stream
        already in B (never saw A) correctly never fires.

        The transition path spends TWO real judge calls (``from`` and ``to``) but produces ONE frozen
        :class:`~percept.contracts.JudgmentEvent` for the fire — built from REAL two-domain stamps that
        bracket both judge calls, and marked ``evidence_origin="transition"``. That is what lets a
        transition fire correlate to its delivered Incident (via :func:`judgment_event_id`) exactly like
        the judged watch path, and it emits ``transition.decision`` EVERY tick (fire or not) so the
        state machine is observable from the ``.ptrace`` alone.
        """
        from_cond, to_cond = goal.state.get("from"), goal.state.get("to")
        if not from_cond or not to_cond:
            return None
        # Per-goal judge routing (fail-closed) — selected BEFORE any judge call AND before any side
        # effect (evidence admission, funnel.judged), exactly like _judge routes the watch path. A goal
        # MARKED judge-requesting but with no binding raises JudgeBindingInvariantError here (propagates;
        # NO adapter called, zero judge calls, zero evidence admitted) rather than silently falling back
        # to the deployment-default judge — the failure class this incident forbids on EVERY dispatch
        # path. No binding => (self.vision, None): every line below is byte-for-byte the unbound path.
        adapter, bound = self._select_judge_adapter(goal)
        roi: Box | None = focus.box if focus else None
        corrections = focus.asserted() if focus else {}
        # The rising-edge latch is scoped by the concern's DECLARED identity policy, defaulting to the
        # stable "scene" — NOT the volatile focus.id, which would split the latch (miss the edge) the
        # moment the in-focus entity changes between the arming (A) and completing (B) observations.
        # Use the CONCERN-declared policy (identity_policy_of), NOT _resolved_identity_policy /
        # select_identity_policy: a profile default (e.g. screen_document's content_key) must not
        # re-scope this latch. transition_latch_key fails toward the stable "scene" for every policy
        # except a per-entity track/appearance resolver with an entity actually in focus.
        declared = identity_policy_of(goal)
        latch_key, latch_key_source = transition_latch_key(declared, focus.id if focus else None)
        # A per-entity transition with NO subject in focus has nothing to attribute an A→B edge to:
        # spend no judges and touch no latch. Latching the shared "scene" key here would let a focus
        # dropout ARM and FIRE across the very entities this policy isolates (a from=yes then a to=yes,
        # both under no-focus, would fire under entity_id="scene"). Skipping instead DEFERS the edge to
        # when the subject is back in focus — the arm on its real key is untouched. Emitted as its own
        # transition.decision row (from/to None ⇒ no judges ran) so the skip is auditable.
        if declared in PER_ENTITY_IDENTITY_POLICIES and focus is None:
            if tracing:
                await self._emit_trace(
                    trace_id, now, STAGE_TRANSITION_DECISION, goal_id=goal.id, entity_id=latch_key,
                    from_truth=None, to_truth=None, armed=None, fired_latch=None, outcome="idle",
                    strip_len=0, enriched=False,
                    identity_policy=declared, latch_key=latch_key, latch_key_source=latch_key_source,
                    focus_entity_id=None,
                    from_confidence=None, to_confidence=None, from_evidence=None, to_evidence=None,
                )
            return None
        # Judge from/to over the SAME resolved temporal strip the watch path uses — not bare single
        # frames. _resolve_strip floors the authored spec by the compiled plan_mode; ``strip`` is None
        # (the degenerate single-frame case) when the plan is 'single' or the window holds no earlier
        # frames, keeping every single-frame transition goal byte-for-byte (no frames= passed). This is
        # what lets a state change caught only across frames (a status flip that lands between samples /
        # renders a frame late) fire — a bare single-frame judge of the anchor alone misses it.
        _, strip = self._resolve_strip(goal, source)
        # Extractor enrichment threaded exactly like perceive_judged — extract ONCE per tick off
        # goal.condition + this frame/roi, emit the extractor.enriched trace stage, and merge the facts
        # into context= for BOTH the from and the to judge call (the reserved EXTRACTOR_CONTEXT_KEY
        # sub-dict). None by default (no extractor) -> no context= kwarg, byte-for-byte.
        judge_context = None
        enriched = False
        if self.extractor is not None:
            enrichment = self.extractor.extract(goal.condition, frame, roi)
            if tracing:
                await self._emit_trace(trace_id, now, "extractor.enriched", goal_id=goal.id,
                                       facts=dict(enrichment.facts))
            if not enrichment.is_empty:
                judge_context = {EXTRACTOR_CONTEXT_KEY: enrichment.to_dict()}
                enriched = True
        # Static-context injection at the transition dispatch site — the SAME rule _judge applies: only
        # when the binding carries a verified static context; a caller/extractor context already holding
        # the reserved key is a bug (ReservedContextKeyError, raised BEFORE any judge call); otherwise
        # COPY (never mutate) the extractor context and inject the decoded text under the reserved key so
        # it reaches BOTH the 'from' and 'to' judge call. No binding / no static context => untouched.
        if bound is not None and bound.static_context is not None:
            if judge_context is not None and JUDGE_STATIC_CONTEXT_KEY in judge_context:
                raise ReservedContextKeyError(
                    f"goal {goal.id!r} judge context already carries the reserved key "
                    f"{JUDGE_STATIC_CONTEXT_KEY!r}; a bound static context cannot be injected over it"
                )
            judge_context = dict(judge_context) if judge_context else {}
            judge_context[JUDGE_STATIC_CONTEXT_KEY] = bound.static_context.decoded_text
        # Pass frames=/context= ONLY when they exist, so a single-frame goal with no extractor calls
        # ``vision.judge`` with the exact single-frame signature (some Vision doubles accept neither kwarg).
        judge_kwargs: dict[str, Any] = {}
        if strip is not None:
            judge_kwargs["frames"] = strip
        if judge_context is not None:
            judge_kwargs["context"] = judge_context
        # Reserve this judgment's evidence_seq at judge-DISPATCH (ADR-001 §8), keyed by the SAME stable
        # (goal, latch_key) the executor, degrade-streak and judgment_event_id use — so evidence-seq,
        # degrade streak and judgment key are ONE key, BEFORE the first judge call.
        evidence_seq = self._admit_evidence((goal.id, latch_key))
        # REAL stamps only — no fabrications. ingest/admit are the frame's honest local times; the
        # source timeline ``t`` (when supplied) rides as the untrusted source_media_ts; the judge window
        # brackets BOTH judge calls (the two the transition sense actually spends).
        timing = JudgmentTimingBuilder(clock=self.clock)
        timing.mark_ingest_observed(at=ingest_observed_at)
        timing.mark_admitted()
        timing.set_source_media_ts(t)
        timing.mark_judge_started()
        # Both judge calls now route through the metered, fail-closed boundary (never adapter.judge
        # directly), so the transition path records the same judge.latency/usage telemetry as the
        # watch path AND degrades — never crashes, never phantom-fires — on a timeout/malformed verdict.
        try:
            self.funnel.judged(concern=goal.id, camera=source)  # the 'from' judge call is being spent
            from_v = await self._safe_timed_judge(
                from_cond, frame, roi=roi, corrections=corrections or None,
                _adapter=adapter, _binding=bound, **judge_kwargs,
            )
            self.funnel.judged(concern=goal.id, camera=source)  # the 'to' judge call is being spent
            to_v = await self._safe_timed_judge(
                to_cond, frame, roi=roi, corrections=corrections or None,
                _adapter=adapter, _binding=bound, **judge_kwargs,
            )
        except _JudgeDegraded as degraded:
            if stale is not None and stale():
                return None
            # A failed/malformed from|to judge fails CLOSED exactly like _observe_and_log: apply nothing
            # this tick — do NOT call the executor, do NOT mutate the latch (a phantom arm/fire off a
            # garbled response is the false-fire this path must never produce) — and never crash the
            # watch. The degrade streak is keyed by the SAME (goal, latch_key) the fire path uses.
            self._degrade_tick((goal.id, latch_key), degraded.cause, goal_id=goal.id,
                               label=" (transition)", action="applying nothing this tick")
            if tracing:
                await self._emit_trace(
                    trace_id, now, STAGE_TRANSITION_DECISION, goal_id=goal.id, entity_id=latch_key,
                    from_truth=None, to_truth=None, armed=None, fired_latch=None,
                    outcome="judge_degraded",
                    strip_len=(len(strip) if strip else 0), enriched=enriched,
                    identity_policy=declared, latch_key=latch_key, latch_key_source=latch_key_source,
                    focus_entity_id=focus.id if focus else None,
                    from_confidence=None, to_confidence=None, from_evidence=None, to_evidence=None,
                )
            return None
        timing.mark_judge_completed()
        if stale is not None and stale():
            return None  # goals changed while the from/to judges were in flight — apply nothing
        self._note_judge_ok((goal.id, latch_key))  # a well-formed from+to resets this key's degrade streak
        # Perception (judging from/to) stays here; the ORDERING WITNESS (seen_from) lives in the
        # executor's isolated per-concern context, and the FIRE itself is the gate's — the executor
        # routes this sense's rising and falling edges through ``Gate.decide`` on the SAME
        # ``(goal, latch_key)`` key used for evidence-seq admission and the degrade streak above. The
        # executor hands back the tick facts so this method never reaches into that ctx.
        tick = self.executor.transition_decide(
            goal, from_v, to_v, focus_id=latch_key, timing=timing, evidence_seq=evidence_seq,
            # The gate's clock for this tick: the SOURCE timeline when the caller supplied one (the
            # same timeline ``_fire_concern`` gates the judged path on, and the one the observed
            # duplicate cadence was measured against), else this frame's local observation time. A
            # bring-your-own-capture caller that threads no ``t`` still gets a monotonic clock rather
            # than a clockless gate.
            now=t if t is not None else now,
        )
        transition_trace = (
            {
                "goal_id": goal.id,
                "entity_id": latch_key,
                "from_truth": tick.from_truth,
                "to_truth": tick.to_truth,
                "armed": tick.armed,
                "fired_latch": tick.fired_latch,
                "outcome": tick.outcome,
                "strip_len": len(strip) if strip else 0,
                "enriched": enriched,
                "identity_policy": declared,
                "latch_key": latch_key,
                "latch_key_source": latch_key_source,
                "focus_entity_id": focus.id if focus else None,
                "from_confidence": from_v.confidence,
                "to_confidence": to_v.confidence,
                "from_evidence": from_v.evidence,
                "to_evidence": to_v.evidence,
            }
            if tracing
            else None
        )
        fired = tick.fired
        if fired is None:
            if transition_trace is not None:
                await self._emit_trace(
                    trace_id,
                    now,
                    STAGE_TRANSITION_DECISION,
                    **transition_trace,
                )
            return None
        goal.state["count"] = goal.state.get("count", 0) + 1
        self.metrics.incr("transition.fire")
        if fired.action == FIRE:
            self.funnel.fired(concern=goal.id, camera=source)  # a transition rising-edge fire emitted
        ev = FireEvent(
            fired.goal_id, fired.action, fired.text, fired.entity_id, fired.verdict,
            judgment=fired.judgment, provenance=fired.provenance, modality=fired.modality,
            # The gate's own typed reason — see the twin construction in
            # ``_reduce_transition_completion``. Dropping it here is what left every transition
            # Incident on the wire with an empty ``reason`` beside a populated judged one.
            reason=fired.reason,
        )
        # Materialize the audit action before any decision/event callback.
        # The frame-level aggregate calls _log_fires again, but that seam is
        # idempotent on this exact event object.
        self._log_fires([ev], now)
        if transition_trace is not None:
            await self._emit_trace(
                trace_id,
                now,
                STAGE_TRANSITION_DECISION,
                **transition_trace,
            )
        # The SAME STAGE_JUDGMENT_EVENT row shape the watch path emits — so the recorder's frame-seq
        # stamping (data["evidence_seq"] = this trace_id's source.frame seq) works unchanged — but with
        # evidence_origin="transition" and the judgment_event_id the delivered Incident will also carry.
        if tracing and ev.judgment is not None:
            await self._emit_trace(
                trace_id, now, STAGE_JUDGMENT_EVENT, goal_id=goal.id,
                action=ev.judgment.action, dispatch_seq=ev.judgment.evidence_seq, entity_id=latch_key,
                judgment_event_id=judgment_event_id(goal.id, latch_key, ev.judgment.evidence_seq),
                evidence_origin=ev.judgment.evidence_origin or "transition",
            )
        return ev

    async def _observe_and_log(
        self, goal: Goal, frame: Any, source: str, now: float,
        *, stale: Callable[[], bool] | None = None,
    ) -> None:
        """Item 4 (log→summarize→wipe): observe THIS clip's step delta and append it to the per-concern
        log. NEVER fires here — a process goal's answer is produced by :meth:`summarize_process` at the
        episode boundary / on demand. The gate is untouched (no verdict routed)."""
        ctx = self.executor.ctx(goal.id)
        _, strip = self._resolve_strip(goal, source, perception_level=ctx.perception_level)
        self.funnel.judged(concern=goal.id, camera=source)  # an observe call is being spent
        # Process goals have no entity/focus concept here; a 1-tuple key never collides with a watch
        # goal's (goal_id, entity_id) 2-tuple key (see perceive_judged's base_key).
        process_key = (goal.id,)
        try:
            verdict = await self._judge(
                goal, frame, None, {}, frames=strip, context={"observe_step": True}, stale=stale
            )
        except _JudgeDegraded as degraded:
            if stale is not None and stale():
                return
            # A failed/malformed observe-step judge logs NOTHING this tick (never a phantom step
            # off a garbled/timed-out response) and never crashes the run — this goal never touches
            # the gate anyway (see the docstring above), so there is nothing further to degrade.
            self._degrade_tick(process_key, degraded.cause, goal_id=goal.id,
                               label=" (observe_step)", action="logging nothing this tick")
            return
        if stale is not None and stale():
            return  # goals changed while the observe call was in flight — don't log a step under it
        self._note_judge_ok(process_key)  # a well-formed verdict resets this key's degrade streak
        step = (getattr(verdict, "evidence", "") or "").strip()
        if step and step != "not enough evidence":
            ctx.process_log.append(step)  # the VLM never sees this back — cheap append only
            ctx.last_step_at = now

    async def summarize_process(self, goal_id: str, *, wipe: bool = False) -> str:
        """Consolidate a process goal's logged step-deltas into the ordered distinct steps (Item 4).

        On-demand ("what have you seen?") with ``wipe=False``; the episode boundary calls it with
        ``wipe=True`` (end the episode → emit → reset the log). Returns "" if nothing has been logged.
        The summarize is gate-irrelevant — it never routes a verdict; only the boundary does (and through
        the unchanged gate). Backends that don't implement ``summarize_process`` yield "" gracefully.
        """
        ctx = self.executor.ctx(goal_id)
        log = list(ctx.process_log)
        summarizer = getattr(self.vision, "summarize_process", None)
        if not log or summarizer is None:
            return ctx.running_answer
        goal = self.goals.get(goal_id)
        condition = goal.condition if goal is not None else ""
        answer = await summarizer(condition, log)
        ctx.running_answer = answer or ctx.running_answer
        if wipe:
            ctx.process_log = []
        return ctx.running_answer

    def _resolve_process_qa(self, goal: Goal, ctx, confidence: float, *, at_boundary: bool = False
                            ) -> FireEvent | None:
        """Emit a RESOLVED process-QA answer once the accumulated running answer is complete/confident
        enough (Item 2 mvp2). This closes the gap behind ``unresolved_rate == 1.0``: the running answer
        accumulates per escalation but the gate only fires on a confident-YES verdict, so an
        evidence-gathering judge never RESOLVES — the answer the metric can score is never produced.

        GENERAL, not per-question: :func:`~percept.concern.resolution_ready` reads only accumulated
        COVERAGE (``len(ctx.process_log)`` — the evidence-bearing observations folded in), SUBSTANCE, and
        CONFIDENCE. An unseen process question resolves with zero new branches. The gate stays UNCHANGED:
        de-dup rides a synthetic verdict on a dedicated ``(goal, __resolved__)`` key — the same mechanism
        :meth:`check_observability` / :meth:`check_episode_boundary` / :meth:`check_deadlines` already
        use — so it resolves ONCE per accumulation and re-arms only when fresh evidence pushes a new,
        different answer. Precision-guarded: a thin/empty/uncertain running answer never resolves."""
        answer = (ctx.running_answer or "").strip()
        ready = resolution_ready(
            answer, len(ctx.process_log), confidence, at_boundary=at_boundary
        )
        # The answer TEXT used to be folded into the gate key (``(goal, __resolved__, answer)``) — a
        # permanent key per distinct answer, unbounded over a long process. The discriminator now lives in
        # ctx: a *changed* answer (more evidence rewrote it) rolls it over and re-arms the ONE singleton
        # key, so a new answer resolves again while a repeated identical answer fires only once. Marked
        # no_evict (a per-goal singleton, exempt from the TTL sweep).
        key = (goal.id, "__resolved__")
        if ctx.last_resolved_answer is not None and answer != ctx.last_resolved_answer:
            self.gate.rearm(key)
        ctx.last_resolved_answer = answer
        verdict = Verdict(satisfied=ready, confidence=confidence, evidence=answer, running_answer=answer)
        action = self.gate.evaluate(key, verdict, no_evict=True)
        self.metrics.incr(f"process_qa.resolve.{action}")
        if action != FIRE:
            return None
        return FireEvent(goal.id, FIRE, goal.say or answer, None, verdict)

    def _resolve_roi(self, goal: Goal, focus, spec) -> Box | None:
        """ROI per the ``PerceptionSpec``: ``focus`` = the in-focus entity box (today); ``hint`` = the
        authored ``region_hint`` (reviving the previously-dead field); ``full`` = the whole frame."""
        if spec.roi == "full":
            return None
        if spec.roi == "hint":
            hint = goal.state.get("region_hint")
            if isinstance(hint, Box):
                return hint
            if isinstance(hint, (list, tuple)) and len(hint) == 4:
                return Box(*hint)
            # no usable hint authored -> fall back to the focus box (never worse than today)
        return focus.box if focus else None

    def _resolve_strip(self, goal, source: str, *, perception_level: int = 0):
        """Resolve the perception spec — the authored/preset ``PerceptionSpec`` FLOORED by the resolved
        ``plan_mode`` and then ESCALATED by this concern's ladder rung — and assemble the temporal strip
        off the rolling window. Returns ``(spec, strip)``; ``strip`` is ``None`` for single-frame
        perception (``plan_mode`` 'single', an authored single spec, or no earlier frames in the window).

        Shared by the watch path (:meth:`perceive_judged` / :meth:`_observe_and_log`) and the A→B
        transition path (:meth:`_perceive_transition`) so BOTH floor to the SAME strip the compiled plan
        mode asks for, rather than the transition path judging bare single frames. The transition path
        passes no ``perception_level`` (it is not on the confusion-escalation ladder), so it gets the
        plan-mode floor only — still single when the plan is 'single', a strip when the plan widened one."""
        spec = _escalated_perception(
            _planned_perception(
                program_of(goal).perception, self._plan_mode_for_goal(goal)
            ),
            perception_level,
        )
        return spec, self._strip_frames(source, spec)

    def _strip_frames(self, source: str, spec) -> list | None:
        """Assemble the temporal strip/window per the ``PerceptionSpec`` from the rolling FrameWindow:
        the earlier-context frames (oldest→newest, EXCLUDING the current frame, which the backend appends
        last). Returns None for single-frame perception — byte-for-byte today.

        LATEST-WINS: the window is assembled HERE, inside the judging tick, against a FrameWindow
        whose last entry is the frame being judged (``perceive_judged`` pushes before it judges). The
        async scheduler (``percept/runtime/worker.py::_JudgeScheduler``) is single-flight and drops a
        superseded candidate BEFORE it ever reaches this method, so a frame that lost the latest-wins
        race contributes nothing to any window, and the window a launched judgment sees is always
        anchored on its own admitted frame — never on a newer one that arrived mid-flight."""
        return self._strip_frames_from_entries(
            self._window(source).frames(),
            spec,
        )

    @staticmethod
    def _strip_frames_from_entries(
        window: list[tuple[float, Any]],
        spec: Any,
    ) -> list | None:
        """Select an exact temporal strip from a retained source window.

        The manual SDK path supplies the core's cognition window. The runtime
        supplies its independently bounded capture ring so VID admission does
        not erase pre-action context merely because earlier frames were not
        themselves judged.
        """
        spec = replace(spec, temporal=_normalize_plan_mode(spec.temporal))  # shim: sprite -> window
        if spec.temporal == "single":
            return None
        earlier = window[:-1]  # drop the current frame (passed as `frame`)
        if not earlier:
            return None
        # A clip window (window_s) bounds the strip by TIME — keep only the earlier frames within
        # window_s seconds of the current frame, so "a 4s clip" means the last 4s, not a frame count.
        if getattr(spec, "window_s", None):
            now_t = window[-1][0]  # current frame's timestamp
            cutoff = now_t - spec.window_s
            earlier = [(t, f) for t, f in earlier if t >= cutoff]
            if not earlier:
                return None
        want = (spec.n_frames - 1) if spec.temporal == "strip" else len(earlier)
        if want <= 0:
            return None
        if spec.sampling == "uniform" and len(earlier) > want:
            if want == 1:
                picked = [earlier[-1]]
            else:
                step = (len(earlier) - 1) / (want - 1)
                picked = [earlier[round(i * step)] for i in range(want)]
        else:  # "recent": the last `want` earlier frames
            picked = earlier[-want:]
        # Ordered, oldest→newest. The sprite montage that used to collapse these into ONE image is gone
        # from the plan path: the judge sees the real ordered frames.
        strip = [f for _, f in picked]
        # Keep source coordinates attached to this exact selection. Recovering
        # them later by frame identity is incorrect when capture reuses a
        # bytes/object instance at multiple timestamps.
        return _TemporalStrip(
            strip,
            source_start_t=float(picked[0][0]),
            source_end_t=float(window[-1][0]),
            source_offsets_s=tuple(
                float(t) - float(picked[0][0])
                for t, _ in [*picked, window[-1]]
            ),
        )

    def _temporal_evidence_context(
        self,
        strip: list | None,
    ) -> dict[str, Any]:
        """Describe the actual retained temporal span without exposing timestamps."""
        if not strip:
            return {}
        start_t = getattr(strip, "source_start_t", None)
        end_t = getattr(strip, "source_end_t", None)
        if not isinstance(start_t, (int, float)) or not isinstance(
            end_t,
            (int, float),
        ):
            return {}
        span_s = max(0.0, float(end_t) - float(start_t))
        result = {
            "_percept_temporal_frame_count": len(strip) + 1,
            "_percept_temporal_span_s": span_s,
        }
        offsets = getattr(strip, "source_offsets_s", None)
        if (
            isinstance(offsets, tuple)
            and len(offsets) == len(strip) + 1
            and all(isinstance(value, (int, float)) for value in offsets)
        ):
            result["_percept_temporal_offsets_s"] = offsets
        return result

    async def _timed_judge(self, *args: Any, **kwargs: Any) -> Verdict:
        """Call the vision judge, timing wall-clock latency and capturing token/cost off the verdict.

        ONE place wraps the real cloud call so every judge path (strip/audio, cache-miss, audit)
        records the same telemetry: ``judge.latency`` (always — we time it here) plus optional
        ``judge.tokens.*`` / ``judge.cost.usd`` when the backend annotated its verdict
        (:func:`~percept.observability.judge_usage`). Backends that don't report usage leave those
        hooks unused; the timing is unconditional.

        THE COST WIRE. The same usage dict is also emitted as a ``judge.usage`` TRACE
        event, which is what carries token/cost out to ``TraceSummary`` (the funnel's counters never
        reached the trace surface, so every traced run reported ``usd: 0.0`` no matter what it spent).
        It is emitted HERE, not at the runtime's judge-result drain, because this wrapper is the only
        site EVERY judge call passes through — inline and async dispatch, cache-miss, strip/audio and
        clip audit alike. A drained runtime result carries FIRES only; metering there would omit
        every judgment that didn't fire and report a confident undercount.
        """
        # Per-goal routing: ``_adapter``/``_binding`` are threaded in by :meth:`_judge` (the only
        # goal-aware caller); popped here so they never reach ``adapter.judge``. When absent (e.g.
        # ``judge_clip``, or any goal with no binding) ``adapter`` is ``self.vision`` and ``binding`` is
        # None -> the call and its trace are byte-for-byte the unbound path.
        adapter = kwargs.pop("_adapter", None) or self.vision
        binding = kwargs.pop("_binding", None)
        audio_dispatch = kwargs.pop("_audio_dispatch", None)
        t0 = self.clock()
        if audio_dispatch is not None:
            verdict = await audio_dispatch(*args, **kwargs)
        else:
            verdict = await adapter.judge(*args, **kwargs)
        usage = judge_usage(verdict)
        self.funnel.record_judge(latency_s=self.clock() - t0, **usage)
        if usage and self.on_trace is not None:
            await self._emit_trace(
                getattr(self, "_usage_trace_id", None) or f"judge@{t0}",
                (
                    self._usage_source_ts
                    if getattr(self, "_usage_source_ts", None) is not None
                    else t0
                ),
                "judge.usage",  # percept.tracing.STAGE_JUDGE_USAGE (string here to keep agent import-light)
                model=self._judge_model(adapter),
                **usage,
                **_judge_binding_trace(binding),  # additive provenance, ONLY when a binding exists
            )
        return verdict

    def _judge_model(self, adapter: Any = None) -> str | None:
        """The model id the vision judge bills against, when the adapter exposes one.

        Read off the adapter (public ``model`` first, then the backends' ``_model``); ``None`` for a
        fake/duck-typed judge that names no model. ``None`` means the call is NOT priced — see
        :func:`percept.slo.price_usage_or_none`. We never substitute a default price here: a guessed
        dollar figure printed as a run's cost is worse than no figure at all.
        """
        v = self.vision if adapter is None else adapter
        for attr in ("model", "_model"):
            m = getattr(v, attr, None)
            if isinstance(m, str) and m:
                return m
        return None

    async def _safe_timed_judge(self, *args: Any, **kwargs: Any) -> Verdict:
        """Fail-closed boundary around a live judge call (ADR-001 §10 invariant 4).

        A judge call fails one of two ways: it RAISES (timeout / provider error — the designed,
        typed failure signal a real backend's retry-then-raise contract already produces, see
        ``percept.backends._resilience.with_retry``), or it RETURNS something that is not a
        well-formed :class:`Verdict` (malformed / partial — e.g. a plain dict, or a dict-shaped
        truncated-JSON payload missing required fields). Neither shape may reach
        ``percept.gate.classify`` — which dereferences ``.confidence``/``.satisfied`` with no guard
        of its own — un-validated. Both failure shapes are normalized to :class:`_JudgeDegraded` here
        and caught by every caller of :meth:`_judge` (``perceive_judged``, ``_observe_and_log``),
        which degrade THAT TICK to ASK/SILENT (never FIRE, never a crash) and let the watch continue
        to the next frame. A well-formed ``Verdict`` is returned completely unchanged — this wrapper
        is a byte-for-byte no-op on the happy path.
        """
        try:
            verdict = await self._timed_judge(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001 - ANY judge-call failure fails closed, not just the
            # two typed modes (timeout/provider_error) fake_judge injects — a real backend can raise
            # its own transport exception too, and this boundary must catch all of them alike.
            raise _JudgeDegraded(exc) from exc
        if not _is_valid_verdict(verdict):
            raise _JudgeDegraded(verdict)
        return verdict

    def _degrade_tick(self, key: Hashable, cause: Any, *, goal_id: str, label: str, action: str) -> None:
        """Shared fail-closed bookkeeping for a degraded judge tick: count it, queue the health
        escalation for ``key``, and warn. Each caller keeps its OWN path-specific side effects
        (``funnel.refused``, the trace emit, the return value) — this is only the common core the
        watch / observe-step / transition degrade handlers all repeat verbatim."""
        goal = self.goals.get(goal_id)
        if goal is not None:
            hold_s, _ = duration_policy(goal.state)
            if hold_s is not None:
                # A failed/malformed provider response supplies no continuity evidence. Remove only
                # pending dwell; an already-fired edge stays disarmed until a supported false read.
                self.gate.clear_pending_dwell_goal(goal_id)
        self.metrics.incr("judge.degraded")
        self._note_judge_degraded(key, cause)  # queues on_health for worker.py
        get_logger("percept.agent").warning(
            "judge.degraded%s: goal=%r cause=%r — %s (fail-closed)", label, goal_id, cause, action,
        )

    def _note_judge_degraded(self, key: Hashable, cause: Any) -> None:
        """Record a degraded judge tick for ``key`` (the SAME ``(goal_id, entity_id)`` key
        the gate/executor already use) and escalate ONCE if its consecutive-degrade streak crosses
        :data:`CONSECUTIVE_DEGRADE_ESCALATE`. Only QUEUES the health-log entries --
        ``percept/runtime/worker.py``'s ``run()`` loop drains :attr:`_judge_health_log` every tick
        into its own ``on_health(RuntimeHealthEvent)`` channel (see that module); ``Percept`` never
        calls ``on_health`` itself."""
        streak = self._judge_degrade_streak.get(key, 0) + 1
        self._judge_degrade_streak[key] = streak
        self._judge_health_log.append(("judge_degraded", key, cause))
        if streak == CONSECUTIVE_DEGRADE_ESCALATE:  # exactly-once at the crossing, never re-fired
            self._judge_health_log.append(("judge_persistent_failure", key, cause))

    def _note_gate_suppressed(self, decision: Any) -> None:
        """Record one GATE SUPPRESSION (a :class:`~percept.gate.GateOutcome` whose reason is
        in :data:`~percept.gate.SUPPRESSION_REASONS`) for the runtime to surface.

        Only QUEUES -- ``percept/runtime/worker.py``'s ``run()`` loop drains :attr:`_gate_health_log`
        every tick into its ``on_health(RuntimeHealthEvent("gate_suppressed"))`` channel, the same
        contract :meth:`_note_judge_degraded` uses. Bounded: a core driven without that loop never
        drains, so the oldest entries are shed rather than accumulating forever on a 24/7 watch (the
        gate already logged each one structurally, so shedding loses no ground truth)."""
        self._gate_health_log.append(
            (decision.key, decision.reason, decision.at)
        )
        if len(self._gate_health_log) > _GATE_HEALTH_LOG_MAX:
            del self._gate_health_log[: len(self._gate_health_log) - _GATE_HEALTH_LOG_MAX]

    def _note_judge_ok(self, key: Hashable) -> None:
        """A well-formed verdict for ``key`` resets its consecutive-degrade streak -- ANY
        successful judge call (regardless of the verdict's confidence/truth) proves the judge is
        reachable again."""
        self._judge_degrade_streak.pop(key, None)

    async def judge_clip(
        self,
        condition: str,
        frames: list,
        *,
        roi: Box | None = None,
        corrections: dict | None = None,
        audio: Any = None,
    ) -> Verdict:
        """Judge a plain-language ``condition`` over a CLIP (multiple frames), returning a ``Verdict``.

        The public temporal-judging seam: a thin, ergonomic wrapper over the existing
        ``Vision.judge(..., frames=...)`` path for callers who already hold a clip (e.g. a concern that
        declared ``window_s``). ``frames`` is the clip oldest→newest; the LAST frame is the anchor passed
        as ``frame`` and the earlier ones are the temporal strip (matching the agent's per-tick
        assembly). A single-element clip degrades to a normal single-frame judge. Timing/usage telemetry
        flows through the same ``_timed_judge`` wrapper as the watch path, so a clip judge is observable
        identically. Raises ``ValueError`` on an empty clip — there is nothing to judge."""
        if not frames:
            raise ValueError("judge_clip requires a non-empty clip (oldest→newest frames)")
        anchor = frames[-1]
        strip = list(frames[:-1]) or None
        return await self._timed_judge(
            condition, anchor, roi=roi, corrections=corrections or None,
            frames=strip, audio=audio,
        )

    @staticmethod
    def _declared_judge_capacity(adapter: Any) -> int | None:
        """The concurrent-decision capacity an adapter DECLARES, or ``None`` if it declares none.

        :class:`~percept.backends.judge_pool.JudgePool` publishes this as ``max_inflight`` (the
        minimum occupancy its routes grant). Any other adapter that wants to be batched under a
        bound may expose the same attribute. A duck-typed/fake judge declares nothing and is
        governed purely by :data:`_DEFAULT_JUDGE_CONCURRENCY` / ``judge_concurrency``.
        """
        value = getattr(adapter, "max_inflight", None)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            return None
        return value

    def effective_judge_concurrency(self, goals: "Iterable[Any]" = ()) -> int:
        """How many judge calls ONE frame's batch may hold at once — the resolved, floored bound.

        ``judge_concurrency`` (or :data:`_DEFAULT_JUDGE_CONCURRENCY`) is the caller's ceiling; every
        DECLARED adapter capacity in this batch then floors it. The floor is not advisory: the pool
        has no queue, so a caller that fans out past a route's ``max_inflight`` does not wait, it
        collects ``"capacity"``-shaped UNKNOWN verdicts (and, in a deployment that lists ``capacity``
        in ``terminal_outcomes``, kills the run). Bounding the fan-out HERE converts that
        fail-closed overflow into ordinary queueing, which is what the caller wanted in the first
        place.

        Reads ``self.vision`` plus the per-goal bound adapters actually present in ``goals`` — a
        batch routed to a strictly-bounded judge is not sped up by another goal's roomier one.
        """
        ceiling = (
            _DEFAULT_JUDGE_CONCURRENCY
            if self.judge_concurrency is None
            else int(self.judge_concurrency)
        )
        declared = [self._declared_judge_capacity(self.vision)]
        for goal in goals:
            binding = self._judge_bindings.get(getattr(goal, "id", goal))
            adapter = getattr(binding, "adapter", None)
            if adapter is not None:
                declared.append(self._declared_judge_capacity(adapter))
        floors = [value for value in declared if value is not None]
        if floors:
            ceiling = min(ceiling, min(floors))
        return max(1, ceiling)

    async def _judge_slot(
        self,
        slots: "asyncio.Semaphore",
        goal: Goal,
        frame: Any,
        roi: Box | None,
        corrections: dict[str, str],
        *,
        entity_id: str | None,
        frames: list | None,
        audio: Any,
        context: dict | None,
        stale: "Callable[[], bool] | None",
        timing: Any,
    ) -> "Verdict | object":
        """One goal's judge call, holding one of the batch's bounded concurrency slots.

        Everything that makes a judgment a JUDGMENT (routing, cache, fail-closed normalization to
        :class:`_JudgeDegraded`) stays in :meth:`_judge`; this only owns the slot and the two timing
        marks. ``mark_judge_started`` fires AFTER the slot is acquired, so time spent queued behind
        the bound is attributed to admitted->started (where it belongs) instead of inflating the
        provider's measured latency.
        """
        async with slots:
            if stale is not None and stale():
                # The batched form of the sequential loop's ``if _stale(): break``. This goal was
                # launched, then waited its turn, and by the time a slot opened the goals revision /
                # source generation had moved. Judging now would spend a call whose verdict the apply
                # pass is already forbidden to commit, so spend nothing and say so.
                return _BATCH_STALE
            timing.mark_judge_started()
            verdict = await self._judge(
                goal, frame, roi, corrections, entity_id=entity_id,
                frames=frames, audio=audio, context=context, stale=stale,
            )
            timing.mark_judge_completed()
            return verdict

    @staticmethod
    def _consume_batch_task(task: "asyncio.Task") -> None:
        """Retrieve a batch task's exception so an un-awaited one never warns at GC."""
        if task.cancelled():
            return
        try:
            task.exception()
        except (asyncio.CancelledError, Exception):
            pass

    async def _judge(
        self,
        goal: Goal,
        frame: Any,
        roi: Box | None,
        corrections: dict[str, str],
        *,
        entity_id: str | None = None,
        frames: list | None = None,
        audio: Any = None,
        context: dict | None = None,
        stale: "Callable[[], bool] | None" = None,
    ) -> Verdict:
        # Per-goal judge routing (fail-closed) — selected BEFORE any judge call and OUTSIDE the
        # _safe_timed_judge boundary, so a marked-but-unbound goal raises JudgeBindingInvariantError here
        # (propagates; never degraded to ASK) and NO adapter is called. No binding => (self.vision, None):
        # the whole block below is byte-for-byte the unbound path.
        audio_binding = self._audio_judge_binding_for(goal)
        audio_dispatch = None
        if audio_binding is not None:
            bound = self._judge_bindings.get(goal.id)
            if bound is not None and bound.adapter is not audio_binding.adapter:
                raise JudgeBindingInvariantError(
                    f"goal {goal.id!r} audio dispatch no longer matches its selected adapter"
                )
            if bound is None and goal.id in self._judge_requesting_goal_ids:
                raise JudgeBindingInvariantError(
                    f"goal {goal.id!r} requested a per-goal judge but its binding is missing"
                )
            adapter = audio_binding.adapter
            audio_dispatch = audio_binding.dispatch
        else:
            adapter, bound = self._select_judge_adapter(goal)
        # Static-context injection at the dispatch site: only when the binding carries a verified static
        # context. A caller context already holding the reserved key is a bug (ReservedContextKeyError);
        # otherwise COPY the caller's dict (never mutate it) and inject the decoded text under the reserved
        # key. ``force_context`` then guarantees the injected context reaches the adapter even on the
        # single-frame path (which normally forwards no context).
        force_context = bound is not None and bound.static_context is not None
        if force_context:
            if context is not None and JUDGE_STATIC_CONTEXT_KEY in context:
                raise ReservedContextKeyError(
                    f"goal {goal.id!r} judge context already carries the reserved key "
                    f"{JUDGE_STATIC_CONTEXT_KEY!r}; a bound static context cannot be injected over it"
                )
            context = dict(context) if context else {}
            context[JUDGE_STATIC_CONTEXT_KEY] = bound.static_context.decoded_text
        process_mode = bool(
            context
            and (context.get("process_qa") or context.get("observe_step") or context.get(EXTRACTOR_CONTEXT_KEY))
        )
        # CLAIM-MODE (1.5): a claim-slot concern's reply carries a correction channel (operation/value)
        # the backend must be TOLD to parse. Route it like process_mode — bypass the single-frame cache
        # (a "replace"/"retract" verdict must never be re-served from a frame-hashed slot) and forward
        # ``context`` so ``context["claim_mode"]`` reaches the backend. A non-claim goal has no such key,
        # so claim_mode stays False and the path below is byte-for-byte the pre-1.5 cache path.
        claim_mode = bool(context and context.get("claim_mode"))
        event_mode = bool(context and context.get("event_chain"))
        # PR 1.5: a per-goal reference image ("is this THE mug <exemplar>?") — deictic/individuation
        # grounding the backends already implement but the watch path never supplied. Stable per goal
        # (set in goal.state), so it rides the goal's cache scope; None (the default) is byte-for-byte.
        exemplar = goal.state.get("exemplar")
        if frames or audio or process_mode or claim_mode or event_mode or force_context:
            # The strip/audio (escalation) path is rare-by-design and multi-frame; bypass the
            # single-frame cache (its key dHashes ONE frame) so a strip verdict never pollutes — nor is
            # served from — a single-frame slot. The cache stays single-frame-pure. A process-QA judge
            # ALSO bypasses (its running answer/step accumulates per-call), and so does an
            # extractor-enriched judge (context["extractor"] is per-call, computed off THIS frame —
            # must never be cached/served under a different frame's slot). A CLAIM-MODE judge bypasses
            # too: a correction (replace/retract) must never be cached and re-served under a frame hash.
            # ``context`` is passed in process_mode OR claim_mode so a plain strip judge with neither is
            # byte-for-byte.
            # ``force_context`` also forwards ``context`` here so an injected static context reaches the
            # adapter on a plain single-frame binding (no process/claim/strip). With no binding this is
            # exactly the unbound rule (context only on process/claim).
            extra = {
                "context": context
            } if (process_mode or claim_mode or event_mode or force_context) else {}
            # The strip/audio/process-QA judge is on the live watch path (perceive_judged /
            # _observe_and_log) — a malformed/timeout/partial response here must degrade the tick,
            # never crash it, so this goes through the fail-closed boundary (may raise
            # _JudgeDegraded, caught by every caller of _judge).
            return await self._safe_timed_judge(
                goal.condition, frame, roi=roi, exemplar=exemplar, corrections=corrections or None,
                frames=frames, audio=audio, _adapter=adapter, _binding=bound,
                _audio_dispatch=audio_dispatch, **extra,
            )
        # A structured key so the FrameHashLayer can ROI-crop + dHash the frame and scope its
        # entry to (goal, entity); the legacy stub cascade ignores the extra fields harmlessly.
        cache_key = CacheKey(
            goal_id=goal.id, frame=frame, roi=roi, entity_id=entity_id, condition=goal.condition
        )
        hit = self.cache.resolve(cache_key)
        if hit.status == "HIT" and hit.verdict is not None:
            self.metrics.incr("cache.hit")
            return await self._audit_or_use(
                goal,
                frame,
                roi,
                corrections,
                cache_key,
                hit.verdict,
                adapter=adapter,
                binding=bound,
                stale=stale,
            )
        # Same fail-closed boundary for the cache-miss (single-frame) judge — the other half of
        # the live watch path. A well-formed Verdict is cached exactly as before; a degraded response
        # raises _JudgeDegraded BEFORE `cache.store` is ever reached, so a failure is never cached.
        verdict = await self._safe_timed_judge(
            goal.condition, frame, roi=roi, exemplar=exemplar, corrections=corrections or None,
            _adapter=adapter, _binding=bound, _audio_dispatch=audio_dispatch,
        )
        if stale is None or not stale():
            self.cache.store(cache_key, verdict, verdict.confidence)
        return verdict

    async def _audit_or_use(
        self,
        goal: Goal,
        frame: Any,
        roi: Box | None,
        corrections: dict[str, str],
        cache_key: Any,
        cached: Verdict,
        *,
        adapter: Any = None,
        binding: Any = None,
        stale: "Callable[[], bool] | None" = None,
    ) -> Verdict:
        """Trust-but-verify a cache HIT.

        On the audited fraction (decided by ``self.audit`` — deterministic, no RNG on the L1
        path), re-perceive through real vision and compare to the cached verdict. A divergence in
        the ``satisfied`` bit is a **false-serve**: the fresh read wins and replaces the stale
        cache entry. Un-audited HITs are served as-is. ``audit is None`` -> never audit.
        """
        if self.audit is None or not self.audit(cache_key):
            return cached
        self.metrics.incr("cache.audit")
        # The audit re-perceive is ALSO a live judge call on the watch path — guard it the same
        # way. A degraded audit response must not crash the tick (which already has a perfectly good
        # cached verdict in hand); fall back to serving `cached` unchanged, exactly as an un-audited
        # HIT would. The false-serve upgrade only ever happens off a WELL-FORMED fresh verdict.
        try:
            fresh = await self._safe_timed_judge(
                goal.condition, frame, roi=roi, exemplar=goal.state.get("exemplar"),
                corrections=corrections or None, _adapter=adapter, _binding=binding,
            )
        except _JudgeDegraded as degraded:
            self._degrade_tick(
                (goal.id, getattr(cache_key, "entity_id", None)),
                degraded.cause,
                goal_id=goal.id,
                label=" (cache audit)",
                action="discarding prior dwell support and serving the cached read",
            )
            return cached
        if fresh.satisfied != cached.satisfied:
            self.metrics.incr("cache.false_serve")  # the cache's analog of false-fire
            if stale is None or not stale():
                self.cache.store(cache_key, fresh, fresh.confidence)  # fresh read wins
            return fresh
        return cached

    def _freshness(self, focus, now: float, goal_id: str | None = None) -> float:
        """Freshness in [0,1]: 1.0 = supported/current, 0.0 = stale (-> ASK, invariant 2).

        Two independent staleness signals, combined as the STRICTER (min) — either going to 0.0 drives
        freshness below ``fresh_min`` -> ASK, matching invariant 2's OR ("low confidence OR stale -> ASK"):

        * ASSERTED-fact age (always on): a user-asserted attribute older than ``validity_window_s`` is stale.
        * PERCEPTION recency (A-3b, opt-in via ``gate.stale_ask``): a goal not confidently read within
          ``validity_window_s`` is stale. ``last_supported[goal_id]`` is stamped on every confident read;
          because ``classify`` runs BEFORE that stamp, a goal with NO prior read (missing entry) is treated
          as fresh — the current judgment IS the fresh perception, so a first-ever read must not ASK.

        With ``stale_ask`` OFF (default) the perception term is never consulted and this is byte-for-byte
        the asserted-only path.
        """
        asserted = 1.0
        if focus is not None:
            confirmed = [a.last_confirmed for a in focus.attributes.values() if a.source == "asserted"]
            if confirmed:
                age = now - max(confirmed)
                asserted = 1.0 if age < self.settings.gate.validity_window_s else 0.0
        if not self.settings.gate.stale_ask or goal_id is None:
            return asserted  # OFF: byte-for-byte the asserted-fact freshness
        perception = 1.0
        last = self.last_supported.get(goal_id)
        if last is not None:
            age = now - last
            perception = 1.0 if age < self.settings.gate.validity_window_s else 0.0
        return min(asserted, perception)

    def _ask_text(self, goal: Goal) -> str:
        return f'About "{goal.condition}" — can you confirm?'

    # -- corrections (voice -> memory) --------------------------------------
    def assert_attribute(
        self, key: str, value: str, *, entity_id: str | None = None, t: float | None = None
    ) -> str | None:
        """Bind a user correction to an entity (defaults to in-focus), then re-open judgment.

        Asserted beats observed; the goal's edge state is cleared so the corrected fact is
        re-judged rather than suppressed by a stale rising edge, and the cache is invalidated.
        """
        now = self.clock() if t is None else t
        target = entity_id
        if target is None:
            focus = self.graph.in_focus()
            target = focus.id if focus else None
        if target is None:
            return None
        self.graph.assert_attribute(target, key, value, t=now)
        for goal in self.goals.active():
            self.gate.reset_goal(goal.id)  # a correction re-opens the edge
        # cache invalidation — assertion-override (2nd in the order goal-change -> assertion ->
        # TTL -> L0): drop every cached verdict resting on the corrected entity so the asserted
        # fact is re-judged, never served stale. Empty/stub cascades are a harmless no-op.
        self.invalidate_cache_entity(target)
        return target

    def _on_entity_evicted(self, entity_id: str) -> None:
        """Eviction hook: an entity died (graph cap) -> drop its gate keys and its cached verdicts, so no
        orphan state survives it. Ties ``FrameHashLayer.invalidate_entity`` to entity DEATH (it previously
        fired only on assertion-override), closing the adjacent-audit gap."""
        self.gate.reset_entity(entity_id)
        self.invalidate_cache_entity(entity_id)

    def invalidate_cache(self, goal_id: str | None = None) -> None:
        """Goal-change invalidation: drop a goal's cached verdicts (``None`` clears all)."""
        for layer in self.cache.layers:
            inv = getattr(layer, "invalidate", None)
            if callable(inv):
                inv(goal_id)

    def invalidate_cache_entity(self, entity_id: str | None) -> None:
        """Assertion-override invalidation: drop cached verdicts resting on an entity."""
        for layer in self.cache.layers:
            inv = getattr(layer, "invalidate_entity", None)
            if callable(inv):
                inv(entity_id)

    def note_mention(self, text: str, *, t: float | None = None) -> str | None:
        """Fuse a spoken mention onto the in-focus entity (voice -> visible-entity)."""
        now = self.clock() if t is None else t
        return self.graph.note_mention(text, t=now)

    # -- the spoken turn (Phase 3) ------------------------------------------
    # Added *beside* perceive_judged, never inside it: the reactive watch loop is untouched.
    # The turn is router -> conductor -> specialist -> TTS, with the same "when in doubt,
    # defer" trust posture as the gate: the local router only fast-paths confident,
    # non-authoring intents; everything else falls to the conductor, the one tier that writes
    # standing state (authors goals).

    async def on_speech_endpoint(
        self, utterance: str, *, frame: Any = None, t: float | None = None
    ) -> TurnResult:
        """Handle one finalized spoken turn end-to-end.

        Pipeline: fuse any spoken mention onto the in-focus entity (voice -> visible-entity)
        -> local route -> if confident, execute locally (stream a reply via the specialist)
        -> else **defer to the conductor**, apply its goal ops, and act on its route /
        proactive-speak. Returns a :class:`~percept.adapters.TurnResult` for the caller/log.
        """
        now = self.clock() if t is None else t
        self.note_mention(utterance, t=now)

        local = self.router.route(utterance) if self.router is not None else None
        if local is not None:
            self.metrics.incr(f"route.local.{local.route}")
            reply = await self._execute(local, utterance, frame=frame)
            return TurnResult(route=local.route, source="router", reply=reply)

        # DEFER: the conductor routes + authors goals (the only tier that writes state).
        # FATAL seam: guard HERE, at the language↔conductor crossing, not at the wiring points.
        # A spoken turn that reaches a SCRIPTED brain is silently discarded (FakeConductor -> route="ignore").
        # session.ask() -> on_speech_endpoint bypasses run()/start(), so guarding only the wiring points
        # left a door open; guarding the crossing covers every door with one check. (Router-
        # handled turns above never reach here, so a turn the conductor was never needed for isn't refused.)
        _guard_live_language_path(self)
        self.metrics.incr("route.defer")
        context = self._turn_context()
        decision = await self.conductor.run("speech", utterance, context)
        reply = await self._apply_decision(decision, utterance, frame=frame)
        return TurnResult(route=decision.route, source="conductor", reply=reply)

    async def arm_from_language(self, command: str, *, t: float | None = None) -> Decision:
        """Author + arm a standing watch from RAW LANGUAGE, WITHOUT executing a spoken reply.

        The explicit ``watch``-verb path, distinct from :meth:`on_speech_endpoint` (which also *answers*
        the turn by streaming the specialist). A realtime wire has already said "on it", so the conductor's
        only job here is to author the watchspec; streaming the specialist's frame answer back would poison
        the held voice session and add seconds of latency. Returns the conductor's :class:`Decision` — its
        ``say`` is the confirmation, its ``goal_ops`` are what got armed.
        """
        now = self.clock() if t is None else t
        self.note_mention(command, t=now)
        _guard_live_language_path(self)  # FATAL seam: raw language -> a scripted brain authors nothing
        decision = await self.conductor.run("speech", command, self._turn_context())
        if decision.goal_ops:
            self.apply_goal_ops(decision.goal_ops)
            self.metrics.incr("goals.authored")
        return decision

    async def on_scene_delta(
        self, event: str, context: dict | None = None, *, frame: Any = None
    ) -> TurnResult:
        """A proactive turn: a scene delta (person entered / something appeared) reaches the
        conductor so it can speak *unprompted* — not only fire pre-armed goals.

        ``event`` is a short delta label (e.g. ``"person_entered"``). There is no utterance,
        so the router is bypassed entirely; the conductor decides whether to proactively speak
        and/or author a goal. Apply path is shared with the spoken turn.
        """
        ctx = {**self._turn_context(), **(context or {})}
        decision = await self.conductor.run(event, "", ctx)
        reply = await self._apply_decision(decision, "", frame=frame)
        return TurnResult(route=decision.route, source="conductor", reply=reply)

    async def _apply_decision(
        self, decision: Decision, utterance: str, *, frame: Any = None
    ) -> str:
        """Fold a conductor :class:`Decision` into state + speech.

        Order matters: author goals *first* (standing-state write), then act on the route.
        """
        if decision.goal_ops:
            self.apply_goal_ops(decision.goal_ops)
            self.metrics.incr("goals.authored")
        return await self._execute(decision, utterance, frame=frame)

    async def deliberate(
        self, goal: Goal, *, t: float | None = None, cooldown_s: float = 30.0
    ) -> Decision | None:
        """Posture-B escalation (the consequence×novelty dial): when a concern is recurrently AMBIGUOUS
        and its stakes warrant it, re-invoke the conductor to REFINE the program (or decide to ask the
        human) instead of re-asking every tick. Consequence-gated + rate-limited via the per-concern
        ConfusionSignal + cooldown (the deliberative analog of the gate's refractory). Returns the
        conductor's Decision, or ``None`` when escalation isn't warranted.
        """
        now = self.clock() if t is None else t
        ctx = self.executor.ctx(goal.id)
        if not should_escalate(ctx, consequence_of(goal), now, cooldown_s=cooldown_s):
            return None
        ctx.last_escalated_at = now  # stamp the deliberative refractory (prevents LLM thrash)
        self.metrics.incr("deliberate.escalate")
        context = {
            **self._turn_context(),
            "goal": {"id": goal.id, "condition": goal.condition, "state": dict(goal.state)},
            "reason": "recurrent_ambiguity",
            "recent_confidence": list(ctx.confusion.recent_conf),
        }
        decision = await self.conductor.run("refine_program", goal.condition, context)
        if decision is not None and decision.goal_ops:
            self.apply_goal_ops(decision.goal_ops)  # the LLM refined the program (the in-loop path)
            self.metrics.incr("deliberate.refined")
        return decision

    async def _deliberate_confused(self, now: float) -> None:
        """Posture B in the perceive loop, as a LADDER: a recurrently-ambiguous concern first escalates
        its PERCEPTION (single→strip→+audio — cheap, better-looking) and only once that's maxed and it is
        STILL ambiguous does it re-invoke the LLM (expensive, more-thinking). Both rungs ride the same
        consequence-set ambiguity threshold + a refractory; perception uses a shorter cooldown so it
        always climbs first. A routine/calm concern costs nothing — neither rung fires.
        """
        for goal in list(self.goals.active()):
            ctx = self.executor.ctx(goal.id)
            if ctx.perception_level < MAX_PERCEPTION_LEVEL and escalate_perception(
                ctx, consequence_of(goal), now, cooldown_s=self.perception_cooldown_s
            ):
                self.metrics.incr("perception.escalate")
                continue  # took a perception rung this tick — give the better look a chance before the LLM
            await self.deliberate(goal, t=now)  # perception maxed (or not warranted) → consider the LLM rung

    async def _execute(self, decision: Decision, utterance: str, *, frame: Any = None) -> str:
        """Carry out a routed/conducted :class:`Decision`: speak (stream) or stay silent.

        ``respond_with_vision`` streams grounded on a frame; ``respond_direct`` /
        ``task_with_tools`` stream frame-free; ``proactive_speak`` speaks the conductor's
        ``say`` (or streams if absent); ``ignore`` / ``act`` emit no speech here.
        """
        route = decision.route
        if route == "ignore" or route == "act":
            return ""
        if route == "proactive_speak" and decision.say:
            return self._speak(decision.say)
        grounded = frame if decision.look == "request_frame" else None
        prompt = decision.say or utterance
        return await self._stream_reply(prompt, frame=grounded)

    async def _stream_reply(self, prompt: str, *, frame: Any = None) -> str:
        """Stream specialist tokens to the TTS sink, returning the assembled reply."""
        chunks: list[str] = []
        async for token in self.specialist.stream(prompt, frame=frame):
            self.tts.push(token)
            chunks.append(token)
        self.tts.flush()
        self.metrics.incr("specialist.reply")
        return "".join(chunks).strip()

    def _speak(self, text: str) -> str:
        """Push a fixed line straight to TTS (no model turn)."""
        self.tts.push(text)
        self.tts.flush()
        self.metrics.incr("tts.speak")
        return text

    def _turn_context(self) -> dict:
        """The conductor's read-only view of standing state for a turn / scene delta."""
        focus = self.graph.in_focus()
        return {
            "active_goals": [
                {"id": g.id, "condition": g.condition} for g in self.goals.active()
            ],
            "in_focus": focus.id if focus else None,
        }

    # -- consent + audit (SDK primitives; the strip UI is project 2) --------
    def _log_fires(self, events: list[FireEvent], t: float) -> None:
        """Record each materialized event once in the action audit log."""
        for ev in events:
            if getattr(ev, "_action_materialized", False):
                continue
            event_t = t
            if (
                ev.judgment is not None
                and ev.judgment.timing.source_media_ts is not None
            ):
                event_t = ev.judgment.timing.source_media_ts
            self.action_log.record(
                ActionRecord(
                    kind=ev.action,
                    goal_id=ev.goal_id,
                    detail={"text": ev.text, "entity_id": ev.entity_id},
                    t=event_t,
                )
            )
            ev._action_materialized = True

    def act(
        self, goal_id: str, action: str, detail: dict | None = None, *, t: float | None = None
    ) -> ActionRecord:
        """Perform a computer-use action — **flag-gated, scoped, logged**.

        Allowed only when the goal's ``modality == "act"`` AND the consent profile permits the
        action. Every attempt is logged (allowed or blocked — nothing acts silently); a blocked
        attempt raises :class:`ConsentDenied`. The returned record is the event the assistant's
        consent strip renders.
        """
        now = self.clock() if t is None else t
        goal = self.goals.get(goal_id)
        modality_ok = goal is not None and goal.modality == "act"
        source_id = goal.source if goal is not None else None
        allowed = bool(modality_ok and self.consent.permits(action, source_id))
        record = self.action_log.record(
            ActionRecord(
                kind="act" if allowed else "act_blocked",
                goal_id=goal_id,
                detail={"action": action, **(detail or {})},
                t=now,
                allowed=allowed,
            )
        )
        self.metrics.incr("act.allowed" if allowed else "act.blocked")
        if not allowed:
            if not modality_ok:
                why = "goal modality is not 'act'"
            elif getattr(self.consent, "actuation", "none") == "physical":
                # ADR-007: an OUT-of-envelope declaration, not a permission miss. Say so,
                # so a client debugging at 2am reads the refusal instead of hunting an allowlist.
                from .envelope import OUT_OF_ENVELOPE  # noqa: PLC0415 — local, avoids import cycle

                why = (
                    "ConsentProfile.actuation='physical' — "
                    + OUT_OF_ENVELOPE["robotics_control_loop"].sentence()
                )
            else:
                why = (
                    f"action {action!r} not permitted "
                    f"(allow_computer_use={self.consent.allow_computer_use})"
                )
            raise ConsentDenied(f"act blocked for goal {goal_id!r}: {why}")
        return record

    def actuate(
        self,
        goal_id: str,
        action: str,
        *,
        expect: str | None = None,
        verify_within: float = 30.0,
        detail: dict | None = None,
        t: float | None = None,
    ) -> ActionRecord:
        """A world-changing action under the full safety contract.

        Goes through :meth:`act` (consent + per-source + modality enforced; a blocked attempt still
        raises :class:`ConsentDenied`), and — for a concern whose false-fire cost is high enough to
        REQUIRE VERIFICATION — instantiates a **verification concern** that confirms the action's
        effect actually landed. "Every world-changing action instantiates a verification concern"
        is enforced HERE, not left to the caller. OSS stays notify-first: actuate performs
        no side effect beyond the audited ``act()`` and arming the verification concern.
        """
        now = self.clock() if t is None else t
        record = self.act(goal_id, action, detail, t=now)  # consent gate; raises if blocked
        goal = self.goals.get(goal_id)
        if goal is not None and requires_verification(consequence_of(goal)):
            v = self._instantiate_verification(goal, action, expect, now, verify_within)
            record.detail["verification_goal_id"] = v.id
        return record

    def _instantiate_verification(
        self, goal: Goal, action: str, expect: str | None, t: float, verify_within: float
    ) -> Goal:
        """Acting creates a concern that checks the effect LANDED — reusing the deadline/absence-firing
        machinery: a ``deadline`` concern that fires "couldn't confirm" if the expected post-condition
        isn't observed within ``verify_within``. High miss-cost: an unverified world-change is exactly
        what must not be silently dropped. Confirmation (a confident-yes on the effect) cancels it."""
        vid = f"{goal.id}::verify::{action}"
        v = Goal(
            id=vid,
            condition=expect or f"the effect of {action!r} is confirmed",
            say=f"couldn't confirm the effect of {action!r}",
            source=goal.source,
        )
        v.state["rearm"] = "deadline"
        # ``t`` is this agent's clock reading (actuate's ``now``); the worker stamps frames on that SAME
        # clock (wall for live capture), so this absolute deadline is comparable against ``check_deadlines``
        # -> the verification concern can actually fire (A-1: previously a monotonic stamp vs a stream t).
        v.state["deadline_at"] = t + verify_within
        v.state["miss_cost"] = "severe"
        v.state["verifies"] = goal.id
        self.add_goal(v)
        self.action_log.record(
            ActionRecord(
                kind="verify_created",
                goal_id=goal.id,
                detail={"verification_goal_id": vid, "action": action},
                t=t,
            )
        )
        return v

    # -- persistence (endurance: survive a disconnect) ----------------------
    def snapshot(self) -> dict:
        """A JSON-able snapshot of durable state: goals, entities, the action log, and (v2) the gate's
        armed edge state + deadlines rebased into EPOCH time.

        Absolute deadlines live on the monotonic clock, which resets on restart — a raw monotonic value
        is meaningless after a reboot. Each ``deadline_at`` is converted to a wall-clock (``time.time``)
        instant here and rebased back into the fresh monotonic frame in :meth:`restore`. The
        ``clock_anchor`` records the ``(epoch, monotonic)`` pair the conversion was taken against.

        DELIBERATELY ABSENT: ``episodes`` (the per-goal :class:`~percept.episodes.EpisodeLog` map). Its
        omission is the ratified v1.4 policy (SESSION-ONLY, ADR-006), not an oversight — an
        :class:`EpisodeLog` is process-lifetime state with no rebasable duration base (see
        ``percept.episodes``). ``test_z2_episode_persistence`` asserts the key stays out of this dict, so
        a future accidental add TRIPS a test and forces the policy to be revisited deliberately rather
        than flipped by accident. The forward seam that will one day carry it is
        :meth:`Episode.to_dict` / :meth:`Episode.from_dict` — present but intentionally NOT wired here."""
        if (
            self._watch_runtime_bindings
            or self._watch_runtime_keys
            or getattr(self, "_watchplan_applied_hashes", None)
        ):
            raise CompiledWatchSnapshotUnsupportedError(
                "OSS snapshots do not support compiled Watches: their runtime, source-profile, "
                "judge, and episode state is session-only. Refusing to produce a partial recovery "
                "artifact. Snapshot a plain-goal core, or use a durable hosted runtime."
            )
        now_mono = self.clock()
        now_epoch = time.time()
        goals = self.goals.snapshot()
        for item in goals:
            st = item.get("goal", {}).get("state", {})
            da = st.get("deadline_at")
            if da is not None:
                st["deadline_at_epoch"] = now_epoch + (da - now_mono)  # monotonic -> epoch (rebasable)
                del st["deadline_at"]
        return {
            "version": 2,
            "clock_anchor": [now_epoch, now_mono],
            "goals": goals,
            "gate": self.gate.snapshot(),
            "entities": self.graph.snapshot(),
            "action_log": [asdict(r) for r in self.action_log.entries()],
        }

    def restore(self, snap: dict) -> "Percept":
        """Rehydrate goals / entities / action-log / gate from a :meth:`snapshot`, rebasing epoch
        deadlines into THIS process's monotonic frame.

        v1 back-compat is a TRAP: a v1 goal carries an old absolute ``deadline_s`` with no recoverable
        duration base, so it cannot be rebased. Accepting it silently would leave the concern with no
        ``deadline_at`` and it would never fire again — the exact silent-inertness this pass exists to
        kill. So a legacy ``deadline_s`` is handled LOUDLY: a warning + metric, then re-resolved as a
        duration-from-restore-now so the concern still fires."""
        # v1 vs v2 is detected structurally (a v2 goal carries deadline_at_epoch; a v1 goal a legacy
        # deadline_s), so a missing/renumbered version field can never silently mis-route a deadline.
        # Compiled Watch snapshots are refused by snapshot(); restore therefore only accepts the
        # plain-goal durability contract and starts with no session-local runtime/judge bindings.
        self._watch_runtime_bindings.clear()
        self._watch_runtime_keys.clear()
        self._judge_bindings.clear()
        self._judge_binding_tokens.clear()
        self._judge_requesting_goal_ids.clear()
        self._audio_judge_bindings.clear()
        self._audio_judge_binding_tokens.clear()
        ledger = getattr(self, "_watchplan_applied_hashes", None)
        if isinstance(ledger, dict):
            ledger.clear()
        self.goals.restore(snap.get("goals", []))
        self.graph.restore(snap.get("entities", []))
        self.action_log = ActionLog()
        for rd in snap.get("action_log", []):
            self.action_log.record(ActionRecord(**rd))

        now_mono = self.clock()
        now_epoch = time.time()
        for goal in self.goals.all():
            st = goal.state
            if st.get("rearm") != "deadline":
                continue
            if "deadline_at_epoch" in st:  # v2: rebase epoch -> the fresh monotonic frame
                remaining = st.pop("deadline_at_epoch") - now_epoch
                st["deadline_at"] = now_mono + remaining
            elif "deadline_at" not in st and "deadline_after_s" not in st and "deadline_s" in st:
                legacy = st.pop("deadline_s")  # v1 legacy absolute — LOUD, never silent
                try:
                    st["deadline_at"] = now_mono + float(legacy)
                except (TypeError, ValueError):
                    st["deadline_at"] = now_mono
                self.metrics.incr("snapshot.deadline_unmigratable")
                warnings.warn(
                    f"restore(): goal {goal.id!r} carried a legacy absolute 'deadline_s' from a v1 "
                    f"snapshot that cannot be rebased (no duration base). Re-resolved as a duration from "
                    f"restore-time so it still fires — verify the new deadline. Re-author with "
                    f"'deadline_after_s' to remove this warning.",
                    stacklevel=2,
                )

        # Episode memory is SESSION-ONLY (ADR-006) — snapshot() never carried the
        # ``episodes`` map, so every counting goal we just rehydrated will restart at zero episodes and a
        # subject last seen before the restart will open a NEW episode afterwards. That is a decision, not
        # a bug, but it is a failure mode when it is SILENT: an operator watching a counting goal
        # under-counts after a redeploy with nothing saying why. Queue a single loud ``episode_memory_reset``
        # health event whenever we restored active counting goals into an empty episode map, so the amnesia
        # is observable. Queue-only — worker.py drains it into on_health (Percept never calls on_health).
        counting = [g for g in self.goals.all() if g.state.get("type") == "count"]
        occurrence_watches = [
            g for g in self.goals.all() if g.state.get("rearm") == "each_event"
        ]
        if (counting or occurrence_watches) and not self.episodes:
            self._restore_health_log.append(
                (
                    "episode_memory_reset",
                    {
                        "reason": "session_only_v1.4",
                        "counting_goals": len(counting),
                        "occurrence_goals": len(occurrence_watches),
                        "goal_ids": [
                            g.id for g in (*counting, *occurrence_watches)
                        ],
                        "detail": (
                            "episodes are session-only (ADR-006): counting/occurrence memory was "
                            "NOT restored; these goals restart their episode lifecycle and may "
                            "under-count or repeat an occurrence after recovery."
                        ),
                    },
                )
            )

        self.gate.restore(snap.get("gate"))
        return self

    def save_state(self, persistence: Any, key: str = "percept") -> None:
        """Persist the snapshot through a ``Persistence`` backend (in-memory / sqlite)."""
        persistence.save(key, self.snapshot())

    def load_state(self, persistence: Any, key: str = "percept") -> bool:
        """Rehydrate from a ``Persistence`` backend. Returns False if nothing was stored."""
        snap = persistence.load(key)
        if snap is None:
            return False
        self.restore(snap)
        return True

    # -- introspection -------------------------------------------------------
    def health(self) -> dict:
        backends = {
            "vision": type(self.vision).__name__,
            "specialist": type(self.specialist).__name__,
            "conductor": type(self.conductor).__name__,
            "stt": type(self.stt).__name__,
            "tts": type(self.tts).__name__,
        }
        from . import __version__

        payload = health(__version__, backends, self.metrics)
        # The single scripted-state payload — one vocabulary, asked not guessed. ``mode`` is the
        # resolved switch ("offline"/"live"); ``scripted`` lists the seams whose CONSTRUCTED backend is a
        # scripted double (computed at create() from the class, the fact — not the requested name).
        payload["mode"] = self._mode
        payload["scripted"] = list(self._scripted)
        return payload

    def source_continuity_generation(self, source: str = "camera") -> int:
        """Return the current generation of temporally continuous evidence for ``source``."""
        return int(self._source_continuity_generations.get(source, 0))

    def clear_pending_dwell(
        self,
        source: str = "camera",
        *,
        goal_ids: "tuple[str, ...] | None" = None,
    ) -> None:
        """Invalidate unproven duration support for selected Watches on one source."""

        selected = None if goal_ids is None else frozenset(goal_ids)
        for goal in self.goals.all():
            if (
                getattr(goal, "source", "camera") == source
                and (selected is None or goal.id in selected)
            ):
                hold_s, _ = duration_policy(goal.state)
                if hold_s is not None:
                    self.gate.clear_pending_dwell_goal(goal.id)

    def advance_source_continuity(self, source: str = "camera") -> int:
        """Invalidate in-flight work admitted before a source discontinuity.

        This does not re-arm the consequence gate: a transport gap is not evidence that a condition
        became false. It only advances the commit generation and clears the temporal frame window so
        later judgments cannot combine frames from opposite sides of the gap.
        """
        self.clear_pending_dwell(source)
        generation = self.source_continuity_generation(source) + 1
        self._source_continuity_generations[source] = generation
        window = self.windows.get(source)
        if window is not None and hasattr(window, "clear"):
            window.clear()
        return generation


__all__ = ["Percept", "FireEvent", "guard_live_language_path"]
