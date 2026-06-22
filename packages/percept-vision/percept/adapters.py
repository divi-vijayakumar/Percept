"""The FROZEN membrane — vendor-neutral DTOs + Protocols.

This module is the contract between the cognition core and everything vendor-specific.
Per the implementation plan (§0, Phase 0) it is **immutable within a phase**: later phases
implement *behind* these interfaces and add *beside* them. If a phase needs to change a
frozen shape here, stop and flag it — do not edit silently.

Cardinal rule (the firing contract): the Vision seam judges a goal's plain-language
*condition* and returns ``Verdict{satisfied, confidence, evidence}``. It never compiles the
goal to a rigid ``attr_key=value`` and exact-matches. One firing path, one gate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

# ---------------------------------------------------------------------------
# DTOs (vendor-neutral nouns)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Box:
    """Axis-aligned box in normalized 0..1 image coordinates."""

    x: float
    y: float
    w: float
    h: float

    def iou(self, o: "Box") -> float:
        """Intersection-over-union with another box. 0.0 when disjoint or degenerate."""
        ax2, ay2 = self.x + self.w, self.y + self.h
        bx2, by2 = o.x + o.w, o.y + o.h
        ix1, iy1 = max(self.x, o.x), max(self.y, o.y)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        inter = iw * ih
        if inter <= 0.0:
            return 0.0
        union = self.w * self.h + o.w * o.h - inter
        return inter / union if union > 0.0 else 0.0


@dataclass
class Observation:
    """A single detected thing in one frame (the entity-attribute substrate)."""

    label: str
    box: Box
    confidence: float
    attrs: dict[str, str] = field(default_factory=dict)


@dataclass
class Verdict:
    """The firing-contract return: is the plain-language condition true right now?"""

    satisfied: bool
    confidence: float
    evidence: str = ""


# who said a fact, used for provenance + freshness reasoning in the core
Provenance = Literal["observed", "asserted", "cached", "llm"]


@dataclass
class Attribute:
    """One fact bound to an entity, carrying provenance and freshness.

    Invariants enforced by the core (not here): assertion beats observation; asserted
    facts go stale and are re-confirmed after a validity window.
    """

    value: str
    source: Provenance
    confidence: float
    last_confirmed: float  # monotonic seconds


@dataclass
class Goal:
    """A standing watch/count/remind goal, stated as a plain-language condition."""

    id: str
    condition: str
    say: str
    source: str = "camera"
    modality: Literal["speak", "act"] = "speak"
    state: dict = field(default_factory=lambda: {"type": "watch", "count": 0})
    expiry: str = "until_cancelled"


# ---------------------------------------------------------------------------
# Concern model (spec §1,§3,§5) — added BESIDE Goal. A0 introduces the DTOs + the
# Goal->Concern compat shim ONLY; the agent/gate are untouched (byte-for-byte). Belief is
# DERIVED from observations (see percept.concern), never stored. Logic lives in concern.py.
# ---------------------------------------------------------------------------

Cost = Literal["trivial", "mild", "severe"]
FalseFireCost = Literal["trivial", "mild", "severe", "irreversible"]


@dataclass(frozen=True)
class Consequence:
    """The decision weight (spec §5): a coarse (miss, false-fire) cost pair.

    NOTE (R3): at today's non-discriminative confidence the *threshold* job is inert, so v1
    consequence earns its keep via attention / escalation / verification (A3), not the threshold.
    ``protected_floor`` pins ``miss_cost`` at maximum — refinement may never relax it.
    """

    miss_cost: Cost = "mild"
    false_fire_cost: FalseFireCost = "mild"
    protected_floor: bool = False


@dataclass(frozen=True)
class Evidence:
    """One timestamped, sourced observation feeding a Concern's log (spec §3).

    Named ``Evidence`` (not ``Observation``) to avoid colliding with the entity-substrate
    ``Observation`` above. Belief is derived from these against expectations; never stored.
    """

    t: float
    source_id: str
    confidence: float
    satisfied: bool
    coverage: float = 1.0


@dataclass(frozen=True)
class TrackingBehavior:
    """How a concern's state is maintained (spec §1,§4).

    ``rearm`` is the per-concern ``R``. S2 proved each value is expressible WITHOUT changing the
    gate: ``on_false``/``each_event``/``interval`` via key-synthesis; ``never`` with the adapter
    contract (do not push a confident-false into the gate); ``deadline`` via the scheduler + an
    absence adapter. The A1 firing collapse consumes this; A0 only carries it.
    """

    pattern: Literal["watch", "transition", "event", "count"] = "watch"
    rearm: Literal["on_false", "never", "each_event", "interval", "deadline"] = "on_false"
    cadence_s: float | None = None
    deadline_s: float | None = None
    exhaustive: bool = False


@dataclass(frozen=True)
class Subject:
    """What a concern is about (spec §1): an entity ref | relation | expectation | event."""

    kind: Literal["entity", "relation", "expectation", "event"] = "entity"
    ref: str = ""


@dataclass(frozen=True)
class Priors:
    """What to expect, when, and what 'satisfied' looks like (spec §1). Minimal in A0."""

    expected_state: str | None = None
    deadline_s: float | None = None


@dataclass
class Concern:
    """The unit of consequence-weighted attention (spec §1) — sits above the entity.

    Carries ``condition``/``say`` so today's Vision seam + fire path work unchanged through the
    shim. Belief is DERIVED from ``observations`` against ``expectations`` (percept.concern), never
    stored. The default produced by :meth:`from_goal` reproduces today's behavior byte-for-byte.
    """

    id: str
    identity: Subject
    condition: str
    say: str
    consequence: Consequence = field(default_factory=Consequence)
    tracking: TrackingBehavior = field(default_factory=TrackingBehavior)
    observations: list[Evidence] = field(default_factory=list)
    expectations: Priors = field(default_factory=Priors)
    source: str = "camera"

    @classmethod
    def from_goal(cls, goal: "Goal") -> "Concern":
        """Compat shim: a ``Goal`` becomes a default ``Concern`` (watch / on_false / mild).

        R15: the defaults are pinned to the measured current gate numbers (threshold 0.7 via
        ``concern.threshold_for``, re-arm ``on_false``) so an existing goal's firing is unchanged.
        """
        return cls(
            id=goal.id,
            identity=Subject(kind="entity", ref=goal.condition),
            condition=goal.condition,
            say=goal.say,
            source=goal.source,
            consequence=Consequence(),
            tracking=TrackingBehavior(),
        )


@dataclass
class Decision:
    """Conductor / router output — what to do with a turn or scene delta."""

    route: Literal[
        "respond_direct",
        "respond_with_vision",
        "task_with_tools",
        "act",
        "proactive_speak",
        "ignore",
    ]
    look: Literal["none", "request_frame"] = "none"
    filler: str | None = None
    say: str | None = None
    goal_ops: list = field(default_factory=list)
    endpoint: Literal["complete", "incomplete", "barge_in", "n/a"] = "n/a"


@dataclass
class STTEvent:
    text: str
    is_final: bool
    ts: float


@dataclass
class TurnResult:
    route: str
    source: str
    reply: str


# ---- edge wire-contract (down: WatchSpec · up: Tier0Signal) ----


@dataclass
class WatchCondition:
    goal_id: str
    anchor_text: str
    threshold: float
    region_hint: Box | None = None
    detectors: list[str] = field(default_factory=list)  # cheap edge signal detectors to run for this goal


@dataclass
class WatchSpec:
    conditions: list[WatchCondition]
    version: int = 1


@dataclass
class Tier0Signal:
    motion: float
    structural: float
    semantic: float
    ts: float


# ---------------------------------------------------------------------------
# Protocols (implement to swap a backend; resolved by name via the registry)
# ---------------------------------------------------------------------------


@runtime_checkable
class Vision(Protocol):
    async def judge(
        self,
        condition: str,
        frame: Any,
        *,
        roi: Box | None = None,
        exemplar: Any = None,
        corrections: dict | None = None,
        frames: list | None = None,
        audio: Any = None,
    ) -> Verdict:
        """Judge whether ``condition`` is true of ``frame`` (optionally cropped to ``roi``).

        ``frames`` (a temporal strip — earlier context frames in time order, oldest→newest, excluding
        ``frame``) and ``audio`` (an opaque clip, bytes or a ``(mime, bytes)`` pair) are OPTIONAL
        escalations the brain assembles per the concern's ``PerceptionSpec``. Both default ``None`` so a
        single-frame call is byte-for-byte today. A backend lacking a modality (e.g. Claude + audio)
        degrades gracefully to what it can read; the returned ``Verdict`` shape is unchanged either way
        (the gate downstream is identical)."""
        ...

    async def detect(self, frame: Any, goals: list[Goal]) -> list[Observation]:
        """Detect goal-relevant things in a frame (the entity-attribute substrate)."""
        ...

    async def detect_events(self, frames: list, goals: list[Goal]) -> list:
        """Detect temporal events over a frame window. Phase 5; defaults to []."""
        ...


@runtime_checkable
class Specialist(Protocol):
    def stream(self, prompt: str, *, frame: Any = None, fast: bool = False):
        """Async-iterate reply tokens, optionally grounded on a frame."""
        ...


@runtime_checkable
class Conductor(Protocol):
    async def run(self, event: str, utterance: str, context: dict) -> Decision:
        """Route a turn / scene delta and author goal ops. The only tier that writes state."""
        ...


@runtime_checkable
class STT(Protocol):
    def stream(self, audio):
        """Async-iterate ``STTEvent``s from an audio stream."""
        ...


@runtime_checkable
class TTS(Protocol):
    def push(self, text: str) -> None: ...
    def flush(self) -> None: ...


@runtime_checkable
class Identity(Protocol):  # Phase 4
    def match(self, obs: Observation, known: list) -> str | None:
        """Return the id of the matching known entity, or None for a new one."""
        ...


@runtime_checkable
class Persistence(Protocol):  # Phase 7
    def save(self, key: str, value: Any) -> None: ...
    def load(self, key: str) -> Any: ...


__all__ = [
    "Box",
    "Observation",
    "Verdict",
    "Provenance",
    "Attribute",
    "Goal",
    "Cost",
    "FalseFireCost",
    "Consequence",
    "Evidence",
    "TrackingBehavior",
    "Subject",
    "Priors",
    "Concern",
    "Decision",
    "STTEvent",
    "TurnResult",
    "WatchCondition",
    "WatchSpec",
    "Tier0Signal",
    "Vision",
    "Specialist",
    "Conductor",
    "STT",
    "TTS",
    "Identity",
    "Persistence",
]
