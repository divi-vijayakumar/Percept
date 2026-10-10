"""The types every part of Percept agrees on.

Values are frozen dataclasses. Behaviour is limited to validation and pure conversion
(``Decision.to_dict`` / ``from_dict``). Extension points are ``Protocol`` classes: a
``Source`` brings frames, a ``Trigger`` decides when to look, a ``Judge`` decides what is
true, a ``Sink`` receives decisions and a ``Store`` persists state.

This module imports nothing else from ``percept`` and does no I/O.
"""

from __future__ import annotations

import base64
import math
from dataclasses import dataclass, field, fields
from typing import Any, AsyncIterator, Literal, Mapping, Protocol, runtime_checkable

import numpy as np

CONTRACT_VERSION = "1.0"

Kind = Literal["state", "event"]
Answer = Literal["yes", "no", "unknown"]
VerifyMode = Literal["auto", "original", "fresh", "off"]
DecisionType = Literal["met", "cleared", "uncertain", "expired"]
Failure = Literal["timeout", "unavailable", "malformed", "missing", "refused", "budget"]
HealthKind = Literal[
    "judge_unavailable",
    "degraded",
    "link_down",
    "link_up",
    "budget_exhausted",
    "verify_deferred",
    "scene_reset",
    "capacity",
    "store_lag",
]

# A zone is a polygon in normalised image coordinates: (x, y) with 0 <= x, y <= 1.
Point = tuple[float, float]
Polygon = tuple[Point, ...]

DECISION_TYPES: tuple[str, ...] = ("met", "cleared", "uncertain", "expired")
ANSWERS: tuple[str, ...] = ("yes", "no", "unknown")
FAILURES: tuple[str, ...] = ("timeout", "unavailable", "malformed", "missing", "refused", "budget")
HEALTH_KINDS: tuple[str, ...] = (
    "judge_unavailable", "degraded", "link_down", "link_up", "budget_exhausted",
    "verify_deferred", "scene_reset", "capacity", "store_lag",
)


def _require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def _finite(value: float) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value)


def _check_polygon(polygon: Polygon, name: str) -> None:
    _require(len(polygon) >= 3, f"{name}: a polygon needs at least 3 points")
    for point in polygon:
        _require(len(point) == 2, f"{name}: points are (x, y) pairs")
        x, y = point
        _require(_finite(x) and _finite(y) and 0 <= x <= 1 and 0 <= y <= 1,
                 f"{name}: coordinates are normalised to [0, 1], got {point}")


# --------------------------------------------------------------------------- frames


@dataclass(frozen=True)
class Frame:
    """One decoded frame from a stream.

    ``seq`` and ``t`` are monotonic within one ``stream_epoch``; the epoch increases on every
    reconnect and restart, so values from different epochs are never compared. ``t`` is
    seconds on the process's monotonic clock; ``t_wall`` is Unix time for anything that is
    persisted or shown to a person.
    """

    stream_id: str
    stream_epoch: int
    seq: int
    t: float
    t_wall: float
    image: np.ndarray = field(compare=False, repr=False)

    def __post_init__(self) -> None:
        _require(bool(self.stream_id), "Frame.stream_id must not be empty")
        _require(self.stream_epoch >= 0 and self.seq >= 0, "Frame epoch and seq are non-negative")
        _require(_finite(self.t) and _finite(self.t_wall), "Frame times must be finite")
        _require(isinstance(self.image, np.ndarray) and self.image.ndim in (2, 3),
                 "Frame.image is an HxW or HxWxC numpy array")


@dataclass(frozen=True)
class EvidenceRef:
    """A pointer to a frame. Ephemeral references may not survive a restart."""

    stream_id: str
    stream_epoch: int
    seq: int
    t_wall: float
    ephemeral: bool = True


# --------------------------------------------------------------------------- conditions


@dataclass(frozen=True)
class Condition:
    """A plain-language condition to watch for, and how to decide about it.

    ``kind="state"`` is something that stays true for a while (a door is open) and is
    held, rechecked and cleared on evidence. ``kind="event"`` happens at a moment (a person
    falls); it is verified on the moment's own evidence, reported once, then rearms.
    """

    id: str
    text: str
    kind: Kind = "state"
    trigger: Any = None                      # a Trigger; None means the default for the kind
    judge: str | None = None                 # route name inside a cascade judge
    zones: tuple[Polygon, ...] = ()
    exclude_zones: tuple[Polygon, ...] = ()
    not_when: tuple[str, ...] = ()
    min_confidence: float = 0.6              # applies to yes and no alike
    confirm: int = 1                         # confident yes answers needed; the first one counts
    confirm_window_s: float = 20.0
    verify: VerifyMode = "auto"              # auto: events verify the original interval, states don't
    fast_path_confidence: float | None = None
    candidate_deadline_s: float = 30.0
    on_unverified: Literal["uncertain", "health"] = "uncertain"
    rearm_s: float = 120.0                   # events
    recheck_every_s: float = 60.0            # states
    clear_after_no: int = 3                  # states
    max_open_s: float | None = None          # states: emits `expired`, stays held
    cooldown_s: float = 0.0
    emit_cleared: bool = False
    version: int = 1

    def __post_init__(self) -> None:
        _require(bool(self.id) and self.id.strip() == self.id, "Condition.id must be a non-empty token")
        _require("/" not in self.id, "Condition.id may not contain '/'")
        _require(bool(self.text.strip()), "Condition.text must not be empty")
        _require(self.kind in ("state", "event"), f"Condition.kind must be 'state' or 'event', got {self.kind!r}")
        _require(self.verify in ("auto", "original", "fresh", "off"), f"unknown verify mode {self.verify!r}")
        _require(not (self.kind == "event" and self.verify == "fresh"),
                 "events can't verify on a fresh frame: a later frame doesn't show the event itself")
        _require(not (self.kind == "state" and self.verify == "original"),
                 "states verify on a fresh frame or not at all")
        _require(0.0 <= self.min_confidence <= 1.0, "min_confidence is in [0, 1]")
        _require(self.fast_path_confidence is None or self.min_confidence <= self.fast_path_confidence <= 1.0,
                 "fast_path_confidence is in [min_confidence, 1]")
        _require(isinstance(self.confirm, int) and self.confirm >= 1, "confirm is an integer >= 1")
        _require(isinstance(self.clear_after_no, int) and self.clear_after_no >= 1, "clear_after_no is >= 1")
        _require(self.on_unverified in ("uncertain", "health"), "on_unverified is 'uncertain' or 'health'")
        _require(isinstance(self.version, int) and self.version >= 1, "version is an integer >= 1")
        for name in ("confirm_window_s", "candidate_deadline_s", "rearm_s", "recheck_every_s"):
            value = getattr(self, name)
            _require(_finite(value) and value > 0, f"{name} must be a positive number")
        _require(_finite(self.cooldown_s) and self.cooldown_s >= 0, "cooldown_s must be >= 0")
        _require(self.max_open_s is None or (_finite(self.max_open_s) and self.max_open_s > 0),
                 "max_open_s must be positive or None")
        for polygon in self.zones:
            _check_polygon(polygon, "zones")
        for polygon in self.exclude_zones:
            _check_polygon(polygon, "exclude_zones")
        _require(all(isinstance(s, str) and s.strip() for s in self.not_when),
                 "not_when entries are non-empty strings")

    @property
    def verify_mode(self) -> Literal["original", "fresh", "off"]:
        """The verification actually used, with ``auto`` resolved by kind."""
        if self.verify == "auto":
            return "original" if self.kind == "event" else "off"
        return self.verify


@dataclass(frozen=True)
class CounterExample:
    """A false alarm a person labelled, sent to the judge as a 'this is not it' example."""

    note: str
    thumbnail_jpeg: bytes | None = None


# --------------------------------------------------------------------------- judging


@dataclass(frozen=True)
class Usage:
    """What inference cost. Attempts count every outbound request, retries included."""

    attempts: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    latency_s: float = 0.0

    def __post_init__(self) -> None:
        _require(self.attempts >= 0 and self.tokens_in >= 0 and self.tokens_out >= 0,
                 "Usage counts are non-negative")
        _require(_finite(self.cost_usd) and self.cost_usd >= 0, "Usage.cost_usd is >= 0")
        _require(_finite(self.latency_s) and self.latency_s >= 0, "Usage.latency_s is >= 0")

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            attempts=self.attempts + other.attempts,
            tokens_in=self.tokens_in + other.tokens_in,
            tokens_out=self.tokens_out + other.tokens_out,
            cost_usd=self.cost_usd + other.cost_usd,
            latency_s=self.latency_s + other.latency_s,
        )


@dataclass(frozen=True)
class PriceModel:
    """Provider pricing, used to reserve a worst-case cost before a request is sent."""

    usd_per_mtok_in: float
    usd_per_mtok_out: float
    tokens_per_image: int

    def __post_init__(self) -> None:
        _require(self.usd_per_mtok_in >= 0 and self.usd_per_mtok_out >= 0 and self.tokens_per_image >= 0,
                 "PriceModel values are non-negative")


@dataclass(frozen=True)
class JudgeItem:
    """One condition to judge within a request, with the state it is judged for."""

    condition: Condition
    subject: str = "scene"
    occurrence_gen: int = 0
    counter_examples: tuple[CounterExample, ...] = ()


@dataclass(frozen=True)
class JudgeRequest:
    """Everything a judge needs for one call. All items share the same evidence."""

    request_id: str
    stream_id: str
    stream_epoch: int
    items: tuple[JudgeItem, ...]
    images: tuple[np.ndarray, ...] = field(compare=False, repr=False)
    evidence_t: tuple[float, float]          # wall-clock interval the images cover
    purpose: Literal["detect", "verify"] = "detect"
    deadline_s: float = 8.0
    max_output_tokens: int = 400

    def __post_init__(self) -> None:
        _require(bool(self.request_id), "JudgeRequest.request_id must not be empty")
        _require(len(self.items) >= 1, "JudgeRequest needs at least one item")
        ids = [i.condition.id for i in self.items]
        _require(len(ids) == len(set(ids)), "a request judges each condition at most once")
        _require(len(self.images) >= 1, "JudgeRequest needs at least one image")
        _require(self.evidence_t[0] <= self.evidence_t[1], "evidence_t is (start, end)")
        _require(self.deadline_s > 0 and self.max_output_tokens > 0, "deadline and token limit are positive")


@dataclass(frozen=True)
class Observation:
    """A judge's answer about one condition, tagged with what it was judged for.

    ``failure`` is set when no real answer was obtained (timeout, outage, malformed reply,
    refusal, missing id, budget). A failed observation never changes an occurrence.
    """

    condition_id: str
    subject: str
    answer: Answer
    confidence: float
    reason: str
    request_id: str
    stream_epoch: int
    condition_version: int
    occurrence_gen: int
    evidence_t: tuple[float, float]
    judge: str
    failure: Failure | None = None

    def __post_init__(self) -> None:
        _require(self.answer in ANSWERS, f"Observation.answer must be yes/no/unknown, got {self.answer!r}")
        _require(_finite(self.confidence) and 0.0 <= self.confidence <= 1.0, "confidence is in [0, 1]")
        _require(self.failure is None or self.failure in FAILURES, f"unknown failure {self.failure!r}")
        _require(self.failure is None or self.answer == "unknown", "a failed observation answers 'unknown'")

    @property
    def failed(self) -> bool:
        return self.failure is not None


@dataclass(frozen=True)
class RequestUsage:
    """Usage for one provider request, shared by every observation it produced."""

    request_id: str
    judge: str
    usage: Usage
    failure: Failure | None = None


@dataclass(frozen=True)
class JudgeResult:
    """A judge's reply: exactly one observation per requested condition, plus usage."""

    observations: tuple[Observation, ...]
    usage: tuple[RequestUsage, ...]


# --------------------------------------------------------------------------- outputs


@dataclass(frozen=True)
class Decision:
    """What the agent hears. A pure value and the versioned wire contract.

    ``decision_id`` is stable across redelivery; ``occurrence_id`` groups every decision
    about one real-world occurrence.
    """

    decision_id: str
    occurrence_id: str
    type: DecisionType
    condition_id: str
    stream_id: str
    subject: str
    t_observed: float
    t_decided: float
    confidence: float
    reason: str
    cost: Usage
    trace_id: str
    request_ids: tuple[str, ...] = ()
    evidence: tuple[EvidenceRef, ...] = ()
    keyframe: bytes | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        _require(bool(self.decision_id) and bool(self.occurrence_id), "decision and occurrence ids are required")
        _require(self.type in DECISION_TYPES, f"unknown decision type {self.type!r}")
        _require(_finite(self.t_observed) and _finite(self.t_decided), "decision times are finite")
        _require(0.0 <= self.confidence <= 1.0, "confidence is in [0, 1]")
        _require(self.keyframe is None or len(self.keyframe) <= 40_000, "keyframe is at most 40 KB")

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready form, matching ``decision.schema.json``."""
        return {
            "contract_version": CONTRACT_VERSION,
            "decision_id": self.decision_id,
            "occurrence_id": self.occurrence_id,
            "type": self.type,
            "condition_id": self.condition_id,
            "stream_id": self.stream_id,
            "subject": self.subject,
            "t_observed": self.t_observed,
            "t_decided": self.t_decided,
            "confidence": self.confidence,
            "reason": self.reason,
            "cost": {f.name: getattr(self.cost, f.name) for f in fields(Usage)},
            "trace_id": self.trace_id,
            "request_ids": list(self.request_ids),
            "evidence": [
                {f.name: getattr(e, f.name) for f in fields(EvidenceRef)} for e in self.evidence
            ],
            "keyframe_jpeg_b64": None if self.keyframe is None else base64.b64encode(self.keyframe).decode("ascii"),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Decision:
        version = data.get("contract_version")
        _require(isinstance(version, str) and version.split(".")[0] == CONTRACT_VERSION.split(".")[0],
                 f"unsupported decision contract version {version!r}")
        keyframe = data.get("keyframe_jpeg_b64")
        return cls(
            decision_id=data["decision_id"],
            occurrence_id=data["occurrence_id"],
            type=data["type"],
            condition_id=data["condition_id"],
            stream_id=data["stream_id"],
            subject=data["subject"],
            t_observed=float(data["t_observed"]),
            t_decided=float(data["t_decided"]),
            confidence=float(data["confidence"]),
            reason=data["reason"],
            cost=Usage(**data["cost"]),
            trace_id=data["trace_id"],
            request_ids=tuple(data.get("request_ids", ())),
            evidence=tuple(EvidenceRef(**e) for e in data.get("evidence", ())),
            keyframe=None if keyframe is None else base64.b64decode(keyframe),
        )


@dataclass(frozen=True)
class Health:
    """An operational event, kept separate from decisions. ``count`` aggregates repeats."""

    kind: HealthKind
    stream_id: str
    t_wall: float
    detail: str = ""
    count: int = 1

    def __post_init__(self) -> None:
        _require(self.kind in HEALTH_KINDS, f"unknown health kind {self.kind!r}")
        _require(self.count >= 1, "Health.count is >= 1")


# --------------------------------------------------------------------------- protocols


Due = Mapping[str, str]  # condition_id -> reason the trigger fired


@runtime_checkable
class Source(Protocol):
    """Brings frames. Must keep ``seq``/``t`` monotonic and bump ``stream_epoch`` on reconnect."""

    stream_id: str

    def frames(self) -> AsyncIterator[Frame]: ...


@runtime_checkable
class Trigger(Protocol):
    """Decides when a condition is worth judging. Pure: no I/O, time passed in."""

    def step(self, frame: Frame, state: Any, now: float) -> tuple[Due, Any]: ...


@runtime_checkable
class Judge(Protocol):
    """Decides what is true. Never raises: failures come back as failed observations."""

    name: str
    can_batch: bool
    price: PriceModel | None

    async def judge(self, request: JudgeRequest) -> JudgeResult: ...


@runtime_checkable
class Delivery(Protocol):
    """The envelope a decision is delivered in. Acknowledging removes it from the outbox."""

    decision: Decision
    attempt: int

    async def ack(self) -> None: ...


@runtime_checkable
class Sink(Protocol):
    """Receives decisions, for example to forward them to a webhook."""

    async def emit(self, decision: Decision) -> None: ...


@runtime_checkable
class Store(Protocol):
    """Durable key-value state. ``commit`` is atomic; a ``None`` value deletes the key."""

    def load(self, key: str) -> bytes | None: ...

    def commit(self, writes: Mapping[str, bytes | None]) -> None: ...
