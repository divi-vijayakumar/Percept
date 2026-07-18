"""percept-contracts — the FROZEN membrane, as a standalone zero-dependency package.

This package is the contract between the cognition core and everything vendor-specific.
Per the implementation plan (§0, Phase 0) it is **immutable within a phase**: later phases
implement *behind* these interfaces and add *beside* them. If a phase needs to change a
frozen shape here, stop and flag it — do not edit silently.

Cardinal rule (the firing contract): the Vision seam judges a goal's plain-language
*condition* and returns ``Verdict{satisfied, confidence, evidence}``. It never compiles the
goal to a rigid ``attr_key=value`` and exact-matches. One firing path, one gate.

This package was extracted out of ``percept.adapters`` so that consumers (clients, the IRA,
the edge twin, dashboards) can import the contract directly instead of re-declaring it. It is
**stdlib only** — no third-party runtime dependencies, by design. ``percept.adapters`` remains
a thin re-export shim so every existing ``from percept.adapters import X`` keeps working.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

# ---------------------------------------------------------------------------
# Global contract version + negotiation
# ---------------------------------------------------------------------------

#: Semantic version of the whole frozen contract surface (DTOs + Protocols + wire shapes).
#: Bump the MAJOR on any breaking change to a frozen shape; MINOR for additive (new optional
#: field / new DTO); PATCH for docs/non-wire tweaks. This is distinct from ``WatchSpec.version``
#: (a wire integer kept for byte-for-byte payload compat).
CONTRACT_VERSION = "1.3.0"  # 1.3: + WatchAction/WatchEvent (M1 — the public watch() stream event)
# 1.2: + Verdict.running_answer (process-QA, optional, gate-irrelevant, back-compat)
# 1.1: + WatchCondition.cadence_s/gate_on + Incident.message (optional, back-compat)


class ContractVersionMismatch(Exception):
    """Raised when two endpoints negotiate incompatible contract MAJOR versions."""


def _major(v: str) -> int:
    return int(v.split(".", 1)[0])


def negotiate(local: str, remote: str) -> bool:
    """Check that ``local`` and ``remote`` contract versions can interoperate.

    Returns ``True`` when the MAJOR versions match (semver: same major == wire-compatible,
    additive minor/patch differences are tolerated). Raises :class:`ContractVersionMismatch`
    when the MAJOR versions differ, because a major bump means a frozen shape changed.
    """
    if _major(local) != _major(remote):
        raise ContractVersionMismatch(
            f"incompatible contract versions: local={local!r} remote={remote!r} "
            f"(major {_major(local)} != {_major(remote)})"
        )
    return True


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
    # --- optional structured fields for temporal/clip judging (Slice 2) ---------------------------------
    # Additive and defaulted: a single-frame judge that sets none of these reproduces today's behaviour
    # byte-for-byte (the gate reads only ``satisfied``/``confidence``). A clip judge MAY fill them:
    phase: str = ""  # "pre" | "during" | "post" | "" — where in the action the latest frame sits
    action_present: bool | None = None  # explicit presence signal for action/counting goals (None = not assessed)
    # Process-QA (Item 2): the model's incremental running answer for a process/QA goal ("what are the
    # steps to X"). GATE-IRRELEVANT — the gate reads only satisfied/confidence; a judge that doesn't do
    # process-QA leaves this "" and reproduces today's firing byte-for-byte. NOT a string-join: it is the
    # model's rewritten summary, threaded per-concern via ConcernContext.running_answer.
    running_answer: str = ""


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
    # B1 — temporal/clip declaration: a concern can say "judge me on a CLIP, not one frame".
    # ``clip`` requests multi-frame perception; ``window_s`` (>0 implies ``clip``) declares the clip's
    # length in seconds, so the brain assembles a time-bounded temporal strip (oldest→newest) and feeds
    # it to ``Vision.judge(..., frames=...)``. Both default off, so a single-frame concern is unchanged.
    clip: bool = False
    window_s: float | None = None

    @property
    def wants_clip(self) -> bool:
        """True when this concern declared it needs a clip (``clip=True`` or a positive ``window_s``)."""
        return self.clip or (self.window_s is not None and self.window_s > 0)

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


@dataclass(frozen=True)
class Schedule:
    """When a watch is *armed* — a coarse, vendor-neutral, timezone-naive time gate (C1).

    The default constructed shape means **always armed** (empty ``windows`` + all seven days),
    so a condition with no schedule behaves byte-for-byte as before. The gate consults
    :meth:`is_active` with an *injected* civil time (no wall-clock read inside pure logic), and
    skips the detector / escalation whenever the current moment falls outside the schedule.

    - ``windows`` — inclusive-start, exclusive-end minute-of-day spans (``0..1440``). Empty means
      "any time of day". A window may wrap past midnight (``start > end``, e.g. 22:00→06:00).
    - ``days`` — ISO weekday numbers that are armed: Monday=1 … Sunday=7. Default = all seven.

    Both halves AND together: armed iff the weekday is in ``days`` AND (no windows, or the
    minute-of-day lands in some window). Minutes (not ``HH:MM`` strings) keep the wire trivially
    typed and language-neutral; :func:`minute_of_day` is the convenience constructor helper.
    """

    windows: tuple[tuple[int, int], ...] = ()
    days: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7)

    def is_active(self, *, weekday: int, minute_of_day: int) -> bool:
        """Is the watch armed at this civil moment? ``weekday`` is ISO (Mon=1…Sun=7).

        ``weekday`` / ``minute_of_day`` are passed in (an injected clock), never read here, so the
        decision is pure and testable. The default ("always") schedule returns ``True`` for every
        moment. A window with ``start > end`` wraps over midnight.
        """
        if weekday not in self.days:
            return False
        if not self.windows:
            return True
        m = minute_of_day
        for start, end in self.windows:
            if start <= end:
                if start <= m < end:
                    return True
            else:  # wraps past midnight (e.g. 22:00 -> 06:00)
                if m >= start or m < end:
                    return True
        return False


def minute_of_day(hour: int, minute: int = 0) -> int:
    """``hour``/``minute`` (24h civil time) to minute-of-day ``0..1440`` — a :class:`Schedule` helper."""
    return hour * 60 + minute


GateOn = Literal["motion", "structural", "semantic", "fixed"]
"""Which signal arms a goal's escalation. ``motion`` (default) = pixel-motion gate — right for physical
events (a person, a package). ``structural``/``semantic`` = the richer Tier0 channels (CLIP/pose), for
content/relational events. ``fixed`` = escalate on a fixed cadence regardless of motion — for low-pixel-
motion content (a screenshare, a dashboard) the motion gate would silently under-sample."""


@dataclass
class WatchCondition:
    goal_id: str
    anchor_text: str
    threshold: float
    region_hint: Box | None = None
    detectors: list[str] = field(default_factory=list)  # cheap edge signal detectors to run for this goal
    schedule: Schedule | None = None  # when armed (None -> always); the gate skips when out of schedule
    cadence_s: float | None = None  # min seconds between escalations for this goal (None -> no cadence cap)
    gate_on: GateOn = "motion"  # which signal arms this goal; "fixed" escalates every cadence_s regardless of motion


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
# Incident (A1) — a fired-event record carrying a captured relevant snippet.
# ---------------------------------------------------------------------------


@dataclass
class IncidentSnippet:
    """The captured *relevant snippet* of media attached to a fired :class:`Incident`.

    A small, transport-neutral container for the evidence the brain chose to capture at the
    moment a concern fired. Exactly one ``kind`` describes what the snippet is:

    - ``"frame"`` — a single still (``media``/``mime``, or a one-element ``frames`` list).
    - ``"clip"``  — a short video window (``frames`` = ordered frames, oldest->newest, and/or an
      encoded blob in ``media``/``mime``, bounded by ``t_start``/``t_end`` in seconds).
    - ``"audio"`` — an opaque audio clip (``media``/``mime``, bounded by ``t_start``/``t_end``).

    All payload fields are optional so the same shape serves a redacted record (media stripped,
    timestamps kept) as serves a full capture. ``media`` is raw ``bytes`` (e.g. PNG/MP4/WAV) with
    its ``mime`` type; ``frames`` is an opaque list of frame objects (backend-defined) for callers
    that prefer decoded frames over an encoded blob. Privacy redaction nulls ``media``/``frames``
    while keeping ``kind``/``mime``/timestamps for the health dashboard.
    """

    kind: Literal["frame", "clip", "audio"]
    media: bytes | None = None
    mime: str | None = None
    frames: list | None = None
    t_start: float | None = None
    t_end: float | None = None


@dataclass(frozen=True)
class AudioFrame:
    """A typed audio signal — the first-class AUDIO modality the eye can read (B2).

    The vendor-neutral, stdlib-only container the brain assembles for an audio/fused
    :class:`Vision.judge` escalation (the acoustic-onset wake path: an onset boundary on the mic
    energy stream → escalate → ``judge(audio=AudioFrame(...))``). It carries a raw PCM buffer and
    its rate/layout, plus optional encoded bytes (e.g. a WAV/Opus blob) so a backend can take
    whichever form it prefers:

    - ``pcm`` — raw little-endian PCM samples (signed 16-bit by default; see ``sample_format``).
    - ``sample_rate`` — samples per second per channel (e.g. ``16000`` for the STT mic stream).
    - ``channels`` — interleaved channel count (1 = mono, the default for a single mic).
    - ``encoded``/``mime`` — an OPTIONAL pre-encoded blob (e.g. ``audio/wav``) for backends that
      want a container rather than raw PCM; ``None`` when only PCM is carried.
    - ``t_start``/``t_end`` — OPTIONAL clip bounds in seconds, for provenance/freshness.

    This is the *typed* form of the previously-opaque ``judge(audio=...)`` payload. The param stays
    typed-but-flexible — a backend still accepts raw ``bytes`` or a ``(mime, bytes)`` pair — but a
    caller that wants a self-describing, schema-validated signal passes an :class:`AudioFrame`. The
    :meth:`as_pair` helper renders it to the legacy ``(mime, bytes)`` shape the Gemini backend's
    ``_audio_part`` already reads, so the typed object flows through the existing path unchanged.
    """

    pcm: bytes = b""
    sample_rate: int = 16000
    channels: int = 1
    sample_format: Literal["s16le", "f32le", "u8"] = "s16le"
    encoded: bytes | None = None
    mime: str | None = None
    t_start: float | None = None
    t_end: float | None = None

    @property
    def duration_s(self) -> float | None:
        """Best-effort clip duration in seconds from ``t_start``/``t_end`` (``None`` if unbounded)."""
        if self.t_start is None or self.t_end is None:
            return None
        return self.t_end - self.t_start

    def as_pair(self) -> tuple[str, bytes]:
        """Render to the legacy ``(mime, bytes)`` audio shape the backends already accept.

        Prefers the pre-``encoded`` blob (with its ``mime``) when present; otherwise returns the raw
        ``pcm`` tagged as ``audio/L16`` (signed 16-bit) / ``audio/pcm`` so the opaque backend path
        (``_audio_part``) consumes the typed object byte-for-byte as it did the opaque one."""
        if self.encoded is not None:
            return (self.mime or "audio/wav", self.encoded)
        mime = "audio/L16" if self.sample_format == "s16le" else "audio/pcm"
        return (f"{mime};rate={self.sample_rate};channels={self.channels}", self.pcm)


@dataclass
class Incident:
    """A fired-event record — the durable trace of one concern firing, with evidence.

    Emitted when a concern fires (e.g. a baby-monitor concern ``"baby awake"`` fires an
    ``Incident`` whose ``snippet`` is a short clip of the wake-up). It carries the firing
    :class:`Verdict` and an optional captured :class:`IncidentSnippet`. The ``id`` is the stable
    handle the rest of the system references later:

    - ``feedback(incident_id, label)`` writes back ``label`` ("real" / "false_alarm").
    - health dashboards list incidents by ``concern_id`` / ``ts``.
    - privacy redaction clears the snippet's media in place, keeping the record.

    ``label`` is ``None`` until a human (or downstream auto-labeller) classifies the fire; the
    suggested vocabulary is ``"real"`` / ``"false_alarm"`` but the field is a free ``str`` so
    callers can use their own scheme without a contract bump.
    """

    id: str
    concern_id: str
    verdict: Verdict
    ts: float
    label: str | None = None  # e.g. "real" / "false_alarm" / None (unlabelled)
    snippet: IncidentSnippet | None = None
    message: str = ""  # the human-facing alert line this fire emitted (the goal's `say`)


# ---------------------------------------------------------------------------
# WatchEvent (M1) — the event the public ``watch()`` entrypoint streams.
# ---------------------------------------------------------------------------

WatchAction = Literal["fire", "ask", "silent"]
"""The three-state gate's outward action. Mirrors ``percept.gate.GateAction`` value-for-value; kept as
an independent ``Literal`` here (not imported) so the contracts package stays dependency-free of the
cognition core, per this module's zero-dependency design."""


@dataclass
class WatchEvent:
    """The stable, versioned event ``percept.watch()`` streams — the M1 frozen public contract.

    One event per gate action taken on a goal this tick: ``"fire"`` (a rising-edge, user-facing
    alert), ``"ask"`` (low-confidence or stale evidence — the gate refuses to guess), or
    ``"silent"`` (no user-facing change). ``"silent"`` is part of the type for forward/observability
    completeness; today's ``watch()`` wraps the existing internal gate wiring byte-for-byte, and that
    wiring only ever surfaces FIRE/ASK to callers (a silent tick simply emits nothing) — see
    ``percept.watch`` for the emission contract.

    Deliberately named ``WatchEvent``, not ``Event``: ``percept.events.Event`` already names the
    *temporal-perception* event (a pass, a rep, a discrete action over a frame window) — a distinct,
    older concept this DTO does not replace or overlap with.
    """

    goal_id: str
    action: WatchAction
    t: float
    text: str = ""
    entity_id: str | None = None
    verdict: Verdict | None = None


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
        audio: "AudioFrame | bytes | tuple[str, bytes] | None" = None,
        context: dict | None = None,
    ) -> Verdict:
        """Judge whether ``condition`` is true of ``frame`` (optionally cropped to ``roi``).

        ``frames`` (a temporal strip — earlier context frames in time order, oldest→newest, excluding
        ``frame``) and ``audio`` (the first-class AUDIO modality — a typed :class:`AudioFrame`, or the
        legacy-opaque ``bytes`` / ``(mime, bytes)`` pair the param still accepts) are OPTIONAL
        escalations the brain assembles per the concern's ``PerceptionSpec``. Both default ``None`` so a
        single-frame call is byte-for-byte today. A backend lacking a modality (e.g. Claude + audio)
        degrades gracefully to what it can read; the returned ``Verdict`` shape is unchanged either way
        (the gate downstream is identical).

        ``context`` is OPTIONAL structured facts from a specialist :class:`Enricher` (e.g. an LPR model's
        ``{"plate": "7XYZ123"}`` for a ``plate-presence`` cue) — read as authoritative, distinct from the
        user-asserted ``corrections``. It defaults ``None`` so an un-enriched call is byte-for-byte today;
        a backend that ignores it degrades gracefully (the seam is wired even when the model is not)."""
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
class Enricher(Protocol):
    """A specialist MODEL that turns a cheap Tier-0 signal into STRUCTURED context for the judge.

    The complement of :class:`Specialist` (an orchestration/Q&A token-stream): an ``Enricher`` is a
    narrow, signal-triggered model whose output is *facts*, not prose. A Tier-0 signal or detector
    declares it needs a specialist (e.g. a ``plate-presence`` cue wants an LPR model); the registry
    resolves the signal name to the ``Enricher``, runs it, and injects its structured output into the
    judge's context — assertion-style facts the VLM trusts over its own read (e.g.
    ``{"plate": "7XYZ123", "region": "US-CA"}``).

    The handoff is one method so a NEW specialist (LPR, fire/smoke classifier, face-id, …) is added by
    REGISTERING an enricher — never by editing Percept core. The structured ``dict`` it returns is the
    contract: it reaches the judge as ``context=`` (read alongside, not instead of, the frame). An
    enricher MAY short-circuit the judge by returning a ``{"verdict": Verdict(...)}`` key — the brain
    uses that verdict directly and skips the VLM call. All keys are optional; an empty dict is a no-op
    enrichment (the judge proceeds exactly as it would untriggered)."""

    def enrich(self, signal: Any, frame: Any = None) -> dict:
        """Run the specialist on a triggering ``signal`` (+ optional ``frame``); return structured facts.

        ``signal`` is the Tier-0 trigger that selected this enricher (a :class:`Tier0Signal`, a detector
        :class:`Boundary`-like object, or a plain name/payload — enricher-defined). ``frame`` is the
        current frame when one is available. Returns a JSON-ish ``dict`` of facts to inject into the judge
        context; ``{"verdict": Verdict}`` to short-circuit; ``{}`` for no-op. MUST NOT raise on a normal
        miss — return ``{}`` instead (the registry treats a raising enricher as a no-op)."""
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
    # version + negotiation
    "CONTRACT_VERSION",
    "ContractVersionMismatch",
    "negotiate",
    # DTOs
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
    "Schedule",
    "minute_of_day",
    "GateOn",
    "WatchCondition",
    "WatchSpec",
    "Tier0Signal",
    "AudioFrame",
    "IncidentSnippet",
    "Incident",
    "WatchAction",
    "WatchEvent",
    # Protocols
    "Vision",
    "Specialist",
    "Enricher",
    "Conductor",
    "STT",
    "TTS",
    "Identity",
    "Persistence",
]
