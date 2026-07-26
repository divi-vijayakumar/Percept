"""percept-contracts — the FROZEN membrane, as a standalone zero-dependency package.

This package is the contract between the cognition core and everything vendor-specific.
It is **immutable within a phase**: later phases
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

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable, ClassVar, Hashable, Literal, Protocol, runtime_checkable

# ---------------------------------------------------------------------------
# Global contract version + negotiation
# ---------------------------------------------------------------------------

#: Semantic version of the whole frozen contract surface (DTOs + Protocols + wire shapes).
#: Bump the MAJOR on any breaking change to a frozen shape; MINOR for additive (new optional
#: field / new DTO); PATCH for docs/non-wire tweaks. This is distinct from ``WatchSpec.version``
#: (a wire integer kept for byte-for-byte payload compat).
CONTRACT_VERSION = "1.10.0"  # 1.10: + judge made operational — WatchPlan.judge_context + judge provenance
# 1.10: WatchPlan.judge becomes OPERATIONAL. Until now WatchPlan.judge was accepted +
# serialized but never consumed; it is now a REQUEST validated at resolution against the deployment-owned
# judge catalog, failing closed (pre-arm error) when the requested judge is unavailable — a request, not
# an override. Additive DTOs + fields, all defaulted so a plan/receipt/ResolvedWatch that names no judge
# is byte-for-byte the pre-1.10 shape:
#   * :class:`ArtifactBinding` — an immutable ref + SHA-256 content-address for a judge-context artifact
#     (prompt envelope / rubric / policy doc); ref grammar ``[A-Za-z0-9._@-]+`` (no path separators),
#     digest normalized to bare-lowercase hex (accepts an optional ``sha256:`` prefix on input).
#   * :class:`ResolvedJudgeSpec` — the resolver's judge OUTPUT (requested/resolved ref, factory,
#     provider, model requested vs resolved, prompt-envelope + context digests).
#   * :attr:`WatchPlan.judge_context` — an OPTIONAL ``ArtifactBinding`` binding the exact judge-context
#     bytes a plan was authored against; projected onto the wire as flat ``judge_context_ref/_sha256/
#     _media_type`` (+ ``_schema_version`` when set) ONLY when bound, so plan_hash for existing plans is
#     stable; ``WatchPlan.from_dict`` reconstructs it and rejects a half-specified (ref-without-hash /
#     hash-without-ref) binding.
#   * :attr:`ResolvedWatch.judge` (+ additive ``components["judge"]`` projection) and
#     :attr:`WatchReceipt.judge_requested`/``judge_resolved``/``judge_context_ref``/``judge_context_sha256``/
#     ``fallback_used`` — judge provenance carried through resolution and onto the applied receipt.
# GATE-IRRELEVANT (the gate reads only satisfied/confidence); Verdict/Incident/JudgmentEvent + the
# judgment/evidence shapes are UNCHANGED; watch.py + WATCH_VERSION move independently (see watch.py).
# 1.9: + evidence-link correlation — judgment_event_id() + EvidenceOrigin
# 1.9: the durable evidence LINK between a judgment and the Incident it produced. Three additive,
# defaulted, GATE-IRRELEVANT fields + one pure helper, so a client audit can prove every delivered
# Incident resolves to EXACTLY ONE judgment record (and vice versa), including the transition path that
# until now emitted no judgment record at all:
#   * :func:`judgment_event_id` — a pure, OPAQUE, INJECTIVE handle ``"{goal}#{entity}#{seq}"`` (each
#     component percent-escaped) joining a ``judgment.event`` trace row to its Incident. Derivable, so it
#     is NOT stored on :class:`JudgmentEvent` (zero information); it IS stored on :class:`Incident`.
#   * :data:`EvidenceOrigin` — a CLOSED ``judged``/``transition`` vocabulary naming which firing path
#     produced the evidence. ``None`` == legacy, read as ``"judged"``.
#   * :attr:`JudgmentEvent.evidence_origin` + :attr:`Incident.judgment_event_id` /
#     :attr:`Incident.evidence_origin` — all appended LAST, defaulted None; a producer/consumer that
#     names none of them sees byte-for-byte the pre-1.9 envelope. watch.py + WATCH_VERSION FROZEN.
# 1.8: + WatchPlan/SettingOrigin/ResolvedWatch/WatchReceipt/FieldDiagnostic
# DTOs (percept.contracts.watchplan, ADR-008 — configuration ownership/resolution/provenance): portable
# typed intent + per-field resolution provenance. A NEW additive module, re-exported from this package's
# top level and from percept.adapters; no existing DTO shape changes. GATE-IRRELEVANT (the gate reads
# only satisfied/confidence; nothing here touches Verdict/Gate) — a consumer that never constructs one of
# these five new types sees byte-for-byte today's contract.
# 1.8 also: + Incident.revision_id/Incident.supersedes — the episode-owned provisional/superseding
# link put ON THE WIRE (it previously existed only as a runtime-local annotation the live delivery path
# never carried through to a webhook consumer). Additive, defaulted None, GATE-IRRELEVANT; watch.py +
# WATCH_VERSION FROZEN. Same batch as the WatchPlan DTOs above — CONTRACT_VERSION does not move again.
# 1.7: + Incident.evidence_seq (additive, defaulted None) — the
# JudgmentEvent.evidence_seq (the per-key ADR-001 §8 dispatch sequence) a fire's judgment already
# carried now rides through to the delivered Incident, so a client can correlate a fire back to the
# judgment record that produced it. GATE-IRRELEVANT; watch.py + WATCH_VERSION FROZEN.
# 1.7 also: + SubjectRef 'provisional' resolution state (a THIRD state beside bound/unknown — a
# session-local UN_<episode_id> handle for a sighting an episode log tracked but never identified) +
# provisional_subject()/PROVISIONAL_KEY_PREFIX (additive, GATE-IRRELEVANT — the gate reads only
# satisfied/confidence; a producer that never emits 'provisional' is byte-for-byte unchanged, and
# 'bound'/'unknown' construction and equality are untouched). Both landed in the same 1.7.0 batch.
# 1.6: + judgment-envelope enrichment: JudgmentEvent + Incident
# gain reason/watchplan_id/roi/modality/provenance (all additive, defaulted, GATE-IRRELEVANT — the gate
# reads only satisfied/confidence, and a consumer that ignores them sees byte-for-byte today's Incident;
# watch.py + WATCH_VERSION FROZEN).
# 1.5: + Verdict claim-mode channel (operation/value/normalized_value/
# target_revision_id/asserted_by) + ConcernAuthority DTO + Concern.authority (all additive, defaulted,
# GATE-IRRELEVANT — watch.py + WATCH_VERSION FROZEN).
# 1.4: + SubjectRef DTO (namespaced, tri-state subject identity) +
# SubjectConflict/SubjectResolution + resolve_subject() + Verdict.subject (all additive, defaulted,
# GATE-IRRELEVANT — watch.py + WATCH_VERSION FROZEN).
# 1.3: + Incident.entity_id (correlation identity survives the emission
# boundary; optional, defaults None, additive).
# 1.2: + WatchAction/WatchEvent (the public watch() stream event; additive).
# 1.1: + AttentionIntent DTO + GoalOrigin enum + Goal.origin (all additive, defaulted, back-compat —
# the effective-attention membrane + durable goal provenance).
# 1.0.0: initial public wire contract (research-preview launch). Consolidates the pre-release surface
# that iterated internally through 1.1–1.3 but never shipped; nothing was published under those, so we
# launch clean at 1.0.0.


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


@dataclass(frozen=True)
class ClipEvent:
    """One timestamped event a clip judge saw inside the judged window (1.3).

    Times are seconds RELATIVE to the start of the judged clip (not wall-clock), because that is
    what a video-native model returns. ``t_end`` is ``None`` for an instantaneous event; ``label``
    is the model's short description (e.g. ``"item concealed in bag"``). These are the incident
    evidence entries for a clip verdict — carried on :attr:`Verdict.events`.
    """

    t_start: float
    t_end: float | None = None
    label: str = ""


# ---------------------------------------------------------------------------
# SubjectRef — the namespaced, tri-state subject identity.
# ---------------------------------------------------------------------------


#: The prefix a `'provisional'` :class:`SubjectRef` key always carries — ``UN_<episode_id>``. Named
#: so nothing constructs the shape by hand: the key is always "this run's episode id, prefixed", never
#: a value that could be mistaken for a durable/biometric identifier.
PROVISIONAL_KEY_PREFIX = "UN_"


@dataclass(frozen=True, eq=False)
class SubjectRef:
    """A namespaced reference to WHAT a judgment is about — the subject identity, resolved or not.

    This lands the identity contract that an earlier revision left a seam for. The failure it fixes:
    a document/entity could be perceived correctly while the SDK could not *say which one* — a ``PO-4471``
    on one ERP is not the same subject as ``PO-4471`` on another, and an unreadable key is neither a
    fabricated identity nor a missing subject. This DTO makes all three expressible, distinctly.

    Fields:

    - ``scheme`` — an OPEN vocabulary naming HOW the subject is identified: ``'track'`` (a live
      tracker id), ``'content'`` (a content-derived key, e.g. an invoice number), ``'slot'`` (a
      positional/UI slot), or any client-defined string. Deliberately not an enum: a client's domain
      scheme must not require a contract bump.
    - ``namespace`` — the tenant / source / system / doc-type the ``key`` is meaningful WITHIN.
      ``'PO-4471'`` is NOT globally unique across ERPs; the namespace is what disambiguates it, so a
      ``bound`` ref REQUIRES a non-empty namespace.
    - ``key`` — the stable identifier within the namespace, or ``None`` when unresolved. A ``bound``
      or ``provisional`` ref REQUIRES a non-empty key; an ``unknown`` ref REQUIRES ``key is None``.
    - ``resolution`` — THREE states: ``'bound'`` (we know which subject this is), ``'unknown'`` (we
      saw a subject but could not identify it and have nothing to key it by), or ``'provisional'`` (we
      saw a subject we could not identify, but it stayed in view long enough to earn a SESSION-LOCAL
      handle — the key is always ``UN_<episode_id>`` (:data:`PROVISIONAL_KEY_PREFIX`), minted by the
      episode log itself from the episode that observed it, never by a judge and never from pixels.
      A ``provisional`` ref is NOT an identity claim: it dies with the run (``scope`` is always
      ``'session'``, enforced below) and it names a SIGHTING, not a re-identified person or object —
      the same discipline :mod:`percept.envelope` enforces for every subject resolver percept ships.
      The ABSENCE of a subject is NOT a fourth resolution value — it is ``SubjectRef | None`` ==
      ``None`` on the carrier (e.g. :attr:`Verdict.subject`). Do not encode "no subject" as an
      ``unknown``/``provisional`` ref with no key; encode it as ``None``.
    - ``scope`` — how far the IDENTIFIER is stable: ``'session'`` (only within this run), ``'source'``
      (stable for this camera/source/document feed), or ``'durable'`` (a content-derived identifier
      stable across runs/restarts). ``scope='durable'`` documents IDENTIFIER stability ONLY. It is
      **NOT** a claim that episode STATE is persisted — percept's episode/continuity state is
      SESSION-ONLY and does not survive a restart. A durable key means only that the same subject
      would RESOLVE to the same key again, never that anything about it was saved. A ``provisional``
      ref may only ever be ``scope='session'`` — a session-local handle claiming durability would be
      exactly the re-identification record percept refuses to produce.
    - ``strategy`` — free-form provenance of HOW this ref was resolved (e.g. ``'client'``, ``'judge'``,
      ``'track-iou'``), for debugging a resolution at 2am.
    - ``confidence`` — OPTIONAL resolver confidence in ``[0, 1]``. Deliberately NOT defaulted to
      ``1.0``: a judge-emitted key is not certain by construction, and a 1.0 default would launder an
      uncertain binding into a certain-looking one. ``None`` means "unstated".

    Equality: two ``'unknown'`` refs NEVER compare equal (an unidentified subject is not "the same"
    as another unidentified subject — nor, like a NaN, as itself). Two ``'bound'`` refs are equal iff
    their ``(scheme, namespace, key)`` triple matches; likewise two ``'provisional'`` refs are equal
    iff that triple matches — which happens exactly when they name the SAME episode, so a provisional
    ref IS stable within the episode that minted it and never collides with a different episode's. A
    ``'bound'`` ref and a ``'provisional'`` ref are never equal to each other even sharing a key
    string, because a provisional handle is never promoted into a real identity by comparison alone.
    ``scope``/``strategy``/``confidence`` are provenance, not identity, and never affect equality.
    """

    scheme: str
    namespace: str
    key: str | None
    resolution: Literal["bound", "unknown", "provisional"]
    scope: Literal["session", "source", "durable"]
    strategy: str
    confidence: float | None = None

    def __post_init__(self) -> None:
        if self.resolution == "bound":
            if not self.namespace:
                raise ValueError(
                    "SubjectRef 'bound' requires a non-empty namespace ('PO-4471' is not globally "
                    "unique across systems — the namespace is what disambiguates the key)."
                )
            if not self.key:
                raise ValueError(
                    "SubjectRef 'bound' requires a non-empty key (a bound ref names WHICH subject)."
                )
        elif self.resolution == "unknown":
            if self.key is not None:
                raise ValueError(
                    "SubjectRef 'unknown' requires key is None — an unresolved subject must not carry "
                    f"a key (got {self.key!r}). The absence of a subject is `SubjectRef | None` == "
                    "None, not an 'unknown' ref bearing a key."
                )
        elif self.resolution == "provisional":
            if not self.key or not self.key.startswith(PROVISIONAL_KEY_PREFIX):
                raise ValueError(
                    "SubjectRef 'provisional' requires a key of the form "
                    f"{PROVISIONAL_KEY_PREFIX!r} + episode id (got {self.key!r}). A provisional "
                    "handle is minted by the episode that observed the sighting, never invented ad "
                    "hoc — use percept.contracts.provisional_subject(episode_id)."
                )
            if self.scope != "session":
                raise ValueError(
                    f"SubjectRef 'provisional' requires scope='session' (got {self.scope!r}). A "
                    "session-local handle claiming source/durable stability would be exactly the "
                    "re-identification record percept refuses to produce — see percept.envelope."
                )
        else:
            raise ValueError(
                "SubjectRef.resolution must be 'bound', 'unknown' or 'provisional', got "
                f"{self.resolution!r}"
            )
        if self.confidence is not None and not (0.0 <= self.confidence <= 1.0):
            raise ValueError(
                f"SubjectRef.confidence must be in [0, 1] or None, got {self.confidence!r}"
            )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SubjectRef):
            return NotImplemented
        # 'unknown' never compares equal to anything, itself included (NaN-like) — an unidentified
        # subject with no key has no basis for sameness. 'bound' and 'provisional' each compare on
        # the identity triple, but only WITHIN their own resolution: a provisional handle is never
        # laundered into a bound identity by an equality check, even if a key string collided.
        if self.resolution == "unknown" or other.resolution == "unknown":
            return False
        if self.resolution != other.resolution:
            return False
        return (self.scheme, self.namespace, self.key) == (
            other.scheme,
            other.namespace,
            other.key,
        )

    def __hash__(self) -> int:
        # 'bound'/'provisional' refs hash on (resolution, scheme, namespace, key), so equal refs
        # share a bucket and dedupe in a set/dict, and a bound/provisional pair sharing a key string
        # still hash differently (resolution is part of the tuple). 'unknown' refs hash on OBJECT
        # IDENTITY: no two unknown refs are equal, so any hash is consistent, and identity hashing
        # keeps the same object findable in a container even though it never compares equal to
        # itself by value.
        if self.resolution in ("bound", "provisional"):
            return hash((self.resolution, self.scheme, self.namespace, self.key))
        return object.__hash__(self)


def provisional_subject(
    episode_id: str, *, scheme: str = "episode", namespace: str = "", strategy: str = "episode_scoped"
) -> SubjectRef:
    """Build the ``'provisional'`` :class:`SubjectRef` for a session-local, unidentified sighting.

    The ONLY door a provisional identity comes through: an episode log observed a subject it could
    not identify and gave it a handle keyed to the episode itself (``UN_<episode_id>``), never to
    pixels or any cross-session signal. Two sightings only ever share this ref when they are the SAME
    episode — a different episode id is a different (unequal) ref, so a provisional handle can never
    accidentally collapse two distinct unidentified subjects, and it can never outlive the run.
    """
    if not isinstance(episode_id, str) or not episode_id.strip():
        raise ValueError("provisional_subject(episode_id=...) must be a non-empty episode id")
    return SubjectRef(
        scheme=scheme,
        namespace=namespace,
        key=f"{PROVISIONAL_KEY_PREFIX}{episode_id}",
        resolution="provisional",
        scope="session",
        strategy=strategy,
    )


@dataclass(frozen=True)
class SubjectConflict:
    """A resolution CONFLICT — two BOUND resolvers disagreed on the subject key.

    A first-class OUTCOME, not a silently-dropped tie. Precedence still deterministically picks the
    ``winner`` (never a coin-flip), but the ``loser`` — the disagreeing lower-precedence bound ref —
    is surfaced so a client debugging identity at 2am can see that two resolvers fought and which
    lost, instead of one binding vanishing without a trace.
    """

    winner: SubjectRef
    loser: SubjectRef


@dataclass(frozen=True)
class SubjectResolution:
    """Typed result of resolving a subject across precedence-ordered resolvers.

    ``subject`` is the winning ref — the highest-precedence BOUND ref when any resolver bound one,
    else the ``unknown`` fallback. ``conflicts`` lists every lower-precedence bound ref that
    disagreed with the winner (empty on the common no-conflict path). A conflict never changes the
    winner — precedence is deterministic — it only makes the disagreement observable.
    """

    subject: SubjectRef
    conflicts: tuple[SubjectConflict, ...] = ()

    @property
    def has_conflict(self) -> bool:
        """True iff at least one lower-precedence bound resolver disagreed with the winner."""
        return bool(self.conflicts)


#: Resolver precedence, highest first: a client-supplied binding beats a judge-emitted one,
#: which beats track identity. ``unknown`` is the floor when nothing bound.
SUBJECT_RESOLVER_PRECEDENCE: tuple[str, ...] = ("client", "judge", "track")


def resolve_subject(
    *,
    client: "SubjectRef | None" = None,
    judge: "SubjectRef | None" = None,
    track: "SubjectRef | None" = None,
    unknown: "SubjectRef | None" = None,
) -> SubjectResolution:
    """Resolve one subject across precedence-ordered resolvers — CONFLICT as a first-class result.

    Precedence (highest first, see :data:`SUBJECT_RESOLVER_PRECEDENCE`): ``client`` (a client-supplied
    bound ref) > ``judge`` (a judge-emitted bound ref) > ``track`` (track identity). The winner is the
    highest-precedence ref whose ``resolution`` is ``'bound'``. When NO resolver bound a subject the
    result is the ``unknown`` fallback — or, if none was supplied, an ``unknown`` ref synthesized from
    the first available candidate's scheme/namespace/scope so provenance is not lost.

    If two BOUND resolvers disagree on identity, this NEVER silently picks one and hides the other: it
    returns the higher-precedence winner AND records each disagreeing lower-precedence ref as a
    :class:`SubjectConflict` in ``conflicts`` (surfaced via :attr:`SubjectResolution.has_conflict`).
    """
    ordered = [ref for ref in (client, judge, track) if ref is not None]
    bound = [ref for ref in ordered if ref.resolution == "bound"]
    if bound:
        winner = bound[0]
        conflicts = tuple(
            SubjectConflict(winner=winner, loser=other) for other in bound[1:] if other != winner
        )
        return SubjectResolution(subject=winner, conflicts=conflicts)
    if unknown is not None:
        if unknown.resolution != "unknown":
            raise ValueError(
                "resolve_subject: the `unknown` fallback must have resolution='unknown', got "
                f"{unknown.resolution!r}"
            )
        return SubjectResolution(subject=unknown, conflicts=())
    if ordered:
        seed = ordered[0]  # carry scheme/namespace/scope provenance into the honest unknown
        return SubjectResolution(
            subject=SubjectRef(
                scheme=seed.scheme,
                namespace=seed.namespace,
                key=None,
                resolution="unknown",
                scope=seed.scope,
                strategy=seed.strategy,
            ),
            conflicts=(),
        )
    raise ValueError("resolve_subject: no resolvers supplied and no `unknown` fallback")


# ---------------------------------------------------------------------------
# Claim-mode / speech-act channel — a value DIFFERENCE is not a REPLACEMENT.
# ---------------------------------------------------------------------------

Operation = Literal["assert", "confirm", "replace", "retract", "close", "mention"]
"""The LOCAL speech act a judge classifies for a value-bearing utterance/finding — classifiable from
the CURRENT turn alone (no prior state), which is what preserves the anchoring mitigation: the prior
value never enters the satisfaction prompt. The six acts, and what each does to a slot's value:

- ``assert``  — states the value as fact. ESTABLISHES the slot when it is empty; does NOT overwrite an
  established value (that requires ``replace``). "The budget is $40K".
- ``confirm``  — re-states the SAME value without change. Value UNCHANGED.
- ``mention``  — the value appears but is not asserted: a question ("Could we do $25K?"), an
  attribution ("Another customer paid $25K") or a negation ("The budget is not $25K"). UNCHANGED.
- ``replace``  — explicitly supersedes the established value. "Actually, the approved budget is $25K".
- ``retract``  — withdraws the value (VALUELESS — carries no replacement). "Scratch what I said." → None.
- ``close``    — terminates the claim/episode. Value withdrawn.

The mutating acts are ``replace``/``retract``/``close`` (they act on an EXISTING value and are the ones
an authority policy gates); ``assert``/``confirm``/``mention`` are non-mutating. Note: containing a
value (four of the five decisive utterances contain ``$25K``) is NOT asserting it — exactly one act
replaces it. The default ``assert`` reproduces today's behaviour byte-for-byte (a plain positive
verdict with no value asserts the condition; with ``value=None`` it establishes nothing)."""

OPERATIONS: tuple[Operation, ...] = ("assert", "confirm", "replace", "retract", "close", "mention")
"""The closed :data:`Operation` vocabulary, ordered. A judge that emits nothing outside this set."""

MUTATING_OPERATIONS: frozenset[str] = frozenset({"replace", "retract", "close"})
"""The acts that MUTATE an existing claim — the ones a :class:`ConcernAuthority` gates. ``assert`` is
absent deliberately: it can only ESTABLISH an empty slot, never overwrite, so it is safe unauthorized;
overwriting an established value is exactly what ``replace`` is, and ``replace`` is gated here."""


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
    # Token/cost accounting for THIS judge call (1.3). GATE-IRRELEVANT and optional: a backend that
    # reports nothing leaves ``{}`` and reproduces today's behaviour byte-for-byte. Suggested keys
    # (read by ``observability.judge_usage``): ``input_tokens`` / ``output_tokens`` / ``cost_usd``.
    usage: dict[str, Any] = field(default_factory=dict)
    # Clip judging (1.3): the timestamped events the judge saw inside the judged window, clip-relative
    # seconds (see :class:`ClipEvent`). Empty for single-frame/strip judges — additive and defaulted.
    events: list[ClipEvent] = field(default_factory=list)
    # Subject identity (1.4): WHICH subject this verdict is about — a namespaced, tri-state
    # :class:`SubjectRef`, or ``None`` when the judge emitted no subject. GATE-IRRELEVANT and appended
    # LAST: the gate reads only ``satisfied``/``confidence``, so a judge that sets nothing reproduces
    # today's firing byte-for-byte, and every existing positional ``Verdict(...)`` keeps its meaning.
    # ``None`` is the ABSENCE of a subject — never confuse it with a ``resolution='unknown'`` ref,
    # which is a subject that was SEEN but not identified. ``episodes._subject_match`` prefers
    # ``verdict.subject.key`` (when bound) over the judge-match fallback.
    subject: "SubjectRef | None" = None
    # Structured value + mutation channel (1.5) — the CLAIM-MODE contract, so the executor
    # never has to parse values out of free-text ``evidence`` (unstable, vendor-dependent) and so a
    # value DIFFERENCE is never mistaken for a REPLACEMENT. All additive, defaulted, GATE-IRRELEVANT
    # (the gate reads only satisfied/confidence), appended LAST so every positional ``Verdict(...)``
    # keeps its meaning and a verdict that sets none of these reproduces today's behaviour:
    #
    # - ``operation`` — the LOCAL speech act (see :data:`Operation`). Default ``assert``: a plain
    #   positive verdict asserts its condition, and ``value=None`` makes that a no-op on any slot.
    # - ``value`` — the slot value this finding carries, as the judge emitted it (e.g. ``"$40K"``).
    #   ``None`` == the finding carries no value (a bare ``retract``, or a non-value condition).
    # - ``normalized_value`` — an OPTIONAL canonical form of ``value`` (e.g. ``40000``) for comparison
    #   across surface spellings; folded in parallel with ``value`` when the judge supplies it.
    # - ``target_revision_id`` — for a ``replace``/``retract``, the revision the judge BELIEVES it acts
    #   on (a hint; the claim ledger computes the authoritative ``supersedes`` itself).
    # - ``asserted_by`` — WHO asserted this (a role/identity, e.g. ``"buyer"``). A :class:`ConcernAuthority`
    #   reads it to decide whether a mutating operation is applied or downgraded. ``None`` == unstated.
    operation: "Operation" = "assert"
    value: Any = None
    normalized_value: Any = None
    target_revision_id: str | None = None
    asserted_by: str | None = None


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


class GoalOrigin(Enum):
    """Durable PROVENANCE of a goal — *who authored it*, which decides its attention TRUST.

    Interrupt authority (may a fire cut into a live turn?) requires durable provenance, not inference
    from which function happens to hold the goal. This enum is **stamped by the SDK entry point** that
    admitted the goal, OVERWRITING whatever a caller supplied — a caller cannot claim trust by filling
    the field. It is persisted with the goal so trust survives ``snapshot``/``restore``:

    - ``STRUCTURED_TRUSTED`` — armed through the structured API (:meth:`Percept.add_goal` /
      ``RuntimeSession.arm``). The developer wrote it in code; it MAY request ``preempt="interrupt"``.
    - ``LANGUAGE_AUTHORED`` — authored by the conductor from raw language (``arm_from_language`` /
      ``apply_goal_ops``). The conductor is an UNTRUSTED author; its interrupt requests are clamped.
    - ``RESTORED`` — rehydrated from a snapshot that predates provenance (no persisted origin). Neutral.
    - ``SYSTEM_TEMPLATE`` — authored by a trusted first-party template/policy; MAY interrupt.

    Default on a freshly-constructed :class:`Goal` is the LEAST-privileged ``LANGUAGE_AUTHORED`` — trust
    is granted only by the API path that stamps it, never by construction."""

    STRUCTURED_TRUSTED = "structured_trusted"
    LANGUAGE_AUTHORED = "language_authored"
    RESTORED = "restored"
    SYSTEM_TEMPLATE = "system_template"


@dataclass(frozen=True)
class Goal:
    """A standing watch/count/remind goal, stated as a plain-language condition."""

    id: str
    condition: str
    say: str
    source: str = "camera"
    modality: Literal["speak", "act"] = "speak"
    state: dict = field(default_factory=lambda: {"type": "watch", "count": 0})
    expiry: str = "until_cancelled"
    # Durable provenance (1.1) — additive + defaulted. STAMPED by the SDK entry point that admits the
    # goal (structured arm -> STRUCTURED_TRUSTED; conductor GoalOps -> LANGUAGE_AUTHORED), overwriting
    # whatever the caller supplied. Persisted with the goal so interrupt-trust survives restore. Default
    # is the LEAST-privileged origin so an un-stamped/directly-constructed goal never carries trust.
    origin: "GoalOrigin" = GoalOrigin.LANGUAGE_AUTHORED


# ---------------------------------------------------------------------------
# Concern model — added BESIDE Goal. This introduces the DTOs + the
# Goal->Concern compat shim ONLY; the agent/gate are untouched (byte-for-byte). Belief is
# DERIVED from observations (see percept.concern), never stored. Logic lives in concern.py.
# ---------------------------------------------------------------------------

Cost = Literal["trivial", "mild", "severe"]
FalseFireCost = Literal["trivial", "mild", "severe", "irreversible"]


@dataclass(frozen=True)
class Consequence:
    """The decision weight: a coarse (miss, false-fire) cost pair.

    NOTE: at today's non-discriminative confidence the *threshold* job is inert, so v1
    consequence earns its keep via attention / escalation / verification, not the threshold.
    ``protected_floor`` pins ``miss_cost`` at maximum — refinement may never relax it.
    """

    miss_cost: Cost = "mild"
    false_fire_cost: FalseFireCost = "mild"
    protected_floor: bool = False


@dataclass(frozen=True)
class ConcernAuthority:
    """CONCERN-LEVEL authority: WHO may issue a MUTATING claim operation against this concern's value.

    The policy the mutation channel needs so that "a seller mentioning $25K must NOT replace
    a BUYER-side stated budget". Authority is evaluated SEPARATELY from the speech act: a well-formed
    ``replace`` from an unauthorized asserter is still a ``replace`` — it is DOWNGRADED (not applied),
    and the downgrade is TRACED (``percept.claims`` logs ``claim.authority_downgrade``), never silent.

    - ``asserters`` — the identities/roles permitted to MUTATE (``replace``/``retract``/``close``) the
      value. Matched against :attr:`Verdict.asserted_by`. An EMPTY set means "no restriction" — the
      idiomatic way to say that is ``Concern.authority = None`` (unrestricted, byte-for-byte today);
      an empty ``ConcernAuthority`` is treated the same rather than locking everyone out by surprise.

    Non-mutating acts (``assert``/``confirm``/``mention``) are ALWAYS permitted: they cannot overwrite
    an established value (``assert`` only establishes an empty slot), so they need no authority. Only
    the acts in :data:`MUTATING_OPERATIONS` are gated.
    """

    asserters: frozenset[str] = frozenset()

    def permits(self, asserted_by: "str | None", operation: str) -> bool:
        """May ``asserted_by`` apply ``operation`` under this policy?

        A non-mutating operation is always permitted. A mutating one requires ``asserted_by`` to be a
        member of :attr:`asserters` — unless ``asserters`` is empty, which is the unrestricted policy.
        """
        if operation not in MUTATING_OPERATIONS:
            return True
        if not self.asserters:
            return True
        return asserted_by is not None and asserted_by in self.asserters


@dataclass(frozen=True)
class Evidence:
    """One timestamped, sourced observation feeding a Concern's log.

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
    """How a concern's state is maintained.

    ``rearm`` is the per-concern ``R``. Each value is expressible WITHOUT changing the
    gate: ``on_false``/``each_event``/``interval`` via key-synthesis; ``never`` with the adapter
    contract (do not push a confident-false into the gate); ``deadline`` via the scheduler + an
    absence adapter. The firing-collapse logic downstream consumes this; this dataclass only
    carries it.
    """

    pattern: Literal["watch", "transition", "event", "count"] = "watch"
    rearm: Literal["on_false", "never", "each_event", "interval", "deadline"] = "on_false"
    cadence_s: float | None = None
    deadline_s: float | None = None
    exhaustive: bool = False


@dataclass(frozen=True)
class Subject:
    """What a concern is about: an entity ref | relation | expectation | event."""

    kind: Literal["entity", "relation", "expectation", "event"] = "entity"
    ref: str = ""


@dataclass(frozen=True)
class Priors:
    """What to expect, when, and what 'satisfied' looks like. Minimal today."""

    expected_state: str | None = None
    deadline_s: float | None = None


@dataclass
class Concern:
    """The unit of consequence-weighted attention — sits above the entity.

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
    # Temporal/clip declaration: a concern can say "judge me on a CLIP, not one frame".
    # ``clip`` requests multi-frame perception; ``window_s`` (>0 implies ``clip``) declares the clip's
    # length in seconds, so the brain assembles a time-bounded temporal strip (oldest→newest) and feeds
    # it to ``Vision.judge(..., frames=...)``. Both default off, so a single-frame concern is unchanged.
    clip: bool = False
    window_s: float | None = None
    # 1.3 — native-clip judging knobs, both inert unless ``wants_clip``: ``fps`` is the sampling rate a
    # video-native judge should read the clip at (None -> backend default; 1fps suits behaviour concerns,
    # sub-second events need more). ``verify_only`` bounds cost: the clip runs ONLY to confirm a candidate
    # FIRE/ASK from the cheaper rungs (single/strip), never as the first look.
    fps: float | None = None
    verify_only: bool = False
    # 1.3 — the version of THIS authored concern (the "versioned detection asset"), so a condition
    # library can roll a concern forward/back and slice metrics per version. Not the contract semver
    # (``CONTRACT_VERSION``) and not the wire integer (``WatchSpec.version``).
    version: int = 1
    # 1.5 — WHO may MUTATE this concern's claimed value. ``None`` (default) == no restriction,
    # byte-for-byte today; a :class:`ConcernAuthority` naming authorized asserters downgrades a mutating
    # operation from anyone else (traced, never silent). Authority is a property of the concern, not the
    # utterance: a seller's textbook ``replace`` against a buyer-authoritative budget is refused here.
    authority: "ConcernAuthority | None" = None

    @property
    def wants_clip(self) -> bool:
        """True when this concern declared it needs a clip (``clip=True`` or a positive ``window_s``)."""
        return self.clip or (self.window_s is not None and self.window_s > 0)

    @classmethod
    def from_goal(cls, goal: "Goal") -> "Concern":
        """Compat shim: a ``Goal`` becomes a default ``Concern`` (watch / on_false / mild).

        The defaults are pinned to the measured current gate numbers (threshold 0.7 via
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


# ---- Production timing contract — two time domains (ADR-001 §5) ----

# Mirrors the *values* of ``percept.gate.GateAction`` ("fire"/"ask"/"silent") without importing
# from ``percept.gate`` — ``percept.contracts`` is the frozen membrane at the bottom of the layering
# (stdlib-only, imported BY ``percept.gate``/``percept.sequencing``, never the other way), so the
# literal is duplicated here rather than creating a reverse/circular import.
JudgmentAction = Literal["fire", "ask", "silent"]

# --- Judgment-envelope enrichment vocabularies -----------------------------------------
# HOW the judgment was perceived + WHERE it came from — the typed, closed vocabularies that ride on
# both the frozen :class:`JudgmentEvent` and the delivered :class:`Incident`. Both are additive and
# always OPTIONAL (``... | None``): a producer that names neither reproduces today's envelope exactly.

#: The perception modality a judgment was made under — mirrors ``percept.behavior.Modality`` (vision /
#: audio / fused) WITHOUT importing it (contracts is the bottom membrane). ``None`` == unstated.
EvidenceModality = Literal["vision", "audio", "fused"]

#: WHERE a judgment came from: ``observed`` — a sensor/VLM read the evidence directly (the judged watch
#: path); ``inferred`` — the finding was derived without a fresh direct observation (e.g. a deadline/
#: absence conclusion: the thing did NOT happen in time). ``None`` == unstated.
JudgmentProvenance = Literal["observed", "inferred"]

#: WHICH firing path produced a judgment's evidence — a CLOSED vocabulary (1.9). ``judged`` — the
#: rising-edge judged watch path (one gate call over one verdict, the common case). ``transition`` — the
#: A→B transition path, which spends TWO real judge calls (the ``from`` and ``to`` states) and fires only
#: after the ``from`` state was witnessed. ``None`` == legacy: a reader treats an absent origin as
#: ``"judged"`` (the only path that emitted a judgment record before 1.9).
EvidenceOrigin = Literal["judged", "transition"]


def judgment_event_id(
    goal_id: str,
    entity_id: "str | None",
    dispatch_seq: int,
    occurrence_id: "str | None" = None,
) -> str:
    """The OPAQUE, INJECTIVE handle joining a ``judgment.event`` trace row to the :class:`Incident`
    it produced: ``"{goal}#{entity}#{seq}"`` for an ordinary decision and
    ``"{goal}#{entity}#{seq}#{occurrence}"`` when one provider dispatch reports multiple temporal
    occurrences. Each component is percent-escaped before joining.

    OPAQUE to consumers: **equality-compare it, never parse it.** The escaping — ``%`` → ``%25``
    then ``#`` → ``%23``, applied IN EACH COMPONENT before the ``#`` separators are inserted —
    guarantees injectivity, even when a raw component contains ``#`` or ``%``. ``occurrence_id=None``
    preserves the original three-component identifier byte-for-byte. A non-``None`` occurrence id is
    an honest discriminator for several decisions derived from one dispatch; it does not fabricate
    additional dispatch sequence numbers. An ``entity_id`` of ``None`` is treated as the empty string
    (a scene-scoped / non-entity fire), so a ``None`` entity and an ``""`` entity share one handle by
    design.
    """
    def _esc(part: str) -> str:
        # ``%`` FIRST so the ``%23`` this step's ``#`` rule emits is never itself re-escaped.
        return part.replace("%", "%25").replace("#", "%23")

    goal = _esc(str(goal_id))
    entity = _esc("" if entity_id is None else str(entity_id))
    seq = _esc(str(dispatch_seq))
    base = f"{goal}#{entity}#{seq}"
    if occurrence_id is None:
        return base
    return f"{base}#{_esc(str(occurrence_id))}"


@dataclass(frozen=True)
class JudgmentTiming:
    """The two-time-domain production timestamp contract for one judgment (ADR-001 §5).

    Two domains, never conflated:

    * **MONOTONIC (production, local clock)** — ``ingest_observed_ts``, ``admitted_ts``,
      ``judge_started_ts``, ``judge_completed_ts``, ``decision_emitted_ts``. ALL latency math
      (dwell/refractory windows, funnel latency, p50/p95/p99) reads ONLY these five stamps, and
      they are sourced from the injectable monotonic clock seam (``percept/clock.py`` —
      ``WallClock``/``FakeClock``), never a wall-clock/NTP-adjustable
      source.
    * **SOURCE (optional, untrusted)** — ``source_media_ts``: the source/fixture/RTSP-PTS
      timeline, when the source exposes one. It may be absent (``None``) or arbitrarily skewed
      from the monotonic domain; it exists for replay/ground-truth ALIGNMENT ONLY (see the
      separate evaluation ground-truth record) and must never be substituted into a
      latency calculation.

    ``ingest_observed_ts`` is when the runtime locally RECEIVED the evidence — NOT camera capture
    time (that would require trusting the untrusted source domain).

    ``decision_emitted_ts`` is the FREEZE point: the :class:`JudgmentEvent` this timing is
    attached to is immutable from that instant on. Delivery/sink timestamps are deliberately NOT
    part of this struct — they live in the ``DeliveryAttempt`` ledger, because a
    frozen JudgmentEvent must never be mutated to append a later delivery stamp.
    """

    ingest_observed_ts: float
    admitted_ts: float
    judge_started_ts: float
    judge_completed_ts: float
    decision_emitted_ts: float
    source_media_ts: float | None = None  # OPTIONAL, untrusted — alignment only, never latency math

    def __post_init__(self) -> None:
        """Construction invariant: the monotonic-domain stamps are non-decreasing in funnel order.

        ``source_media_ts`` is deliberately excluded — it is a separate, possibly unrelated
        timeline and is never compared against the monotonic domain.
        """
        ordered = (
            self.ingest_observed_ts,
            self.admitted_ts,
            self.judge_started_ts,
            self.judge_completed_ts,
            self.decision_emitted_ts,
        )
        for earlier, later in zip(ordered, ordered[1:]):
            if later < earlier:
                raise ValueError(
                    "JudgmentTiming monotonic stamps must be non-decreasing "
                    f"(ingest_observed_ts, admitted_ts, judge_started_ts, judge_completed_ts, "
                    f"decision_emitted_ts): got {ordered!r}"
                )

    # ---- latency helpers — ALL read the monotonic domain only; source_media_ts never enters ----

    def ingest_to_decision_s(self) -> float:
        """Total funnel latency: local receipt of evidence -> frozen decision."""
        return self.decision_emitted_ts - self.ingest_observed_ts

    def admitted_to_decision_s(self) -> float:
        """Judge-funnel latency: tier-0 admission -> frozen decision (ADR-001 §5)."""
        return self.decision_emitted_ts - self.admitted_ts

    def judge_wait_s(self) -> float:
        """Queueing delay between admission and the judge call actually starting."""
        return self.judge_started_ts - self.admitted_ts

    def judge_duration_s(self) -> float:
        """Wall time the judge call itself took."""
        return self.judge_completed_ts - self.judge_started_ts

    def decision_overhead_s(self) -> float:
        """Time from judge completion to the frozen decision (gate/reducer overhead)."""
        return self.decision_emitted_ts - self.judge_completed_ts


@dataclass
class JudgmentTimingBuilder:
    """Accumulates one judgment's monotonic stamps as it moves through the funnel, then freezes.

    Wired through the same injectable clock seam as the rest of run-time cognition (``Clock =
    Callable[[], float]`` in ``percept/clock.py``; ``WallClock`` in production, ``FakeClock`` in
    tests/replay) — nothing here reads a wall clock directly. Each ``mark_*`` records "now" (or an
    explicit ``at=`` override, e.g. for replaying a stream's own admission times) at the moment the
    funnel stage happens; :meth:`freeze` stamps ``decision_emitted_ts`` and produces the immutable
    :class:`JudgmentTiming`.
    """

    clock: Callable[[], float]
    ingest_observed_ts: float | None = None
    admitted_ts: float | None = None
    judge_started_ts: float | None = None
    judge_completed_ts: float | None = None
    source_media_ts: float | None = None

    def mark_ingest_observed(self, *, at: float | None = None) -> float:
        """Stamp local receipt of the evidence — NOT camera capture time."""
        self.ingest_observed_ts = self.clock() if at is None else at
        return self.ingest_observed_ts

    def mark_admitted(self, *, at: float | None = None) -> float:
        """Stamp tier-0 admission to the judge funnel."""
        self.admitted_ts = self.clock() if at is None else at
        return self.admitted_ts

    def mark_judge_started(self, *, at: float | None = None) -> float:
        """Stamp the judge call actually starting."""
        self.judge_started_ts = self.clock() if at is None else at
        return self.judge_started_ts

    def mark_judge_completed(self, *, at: float | None = None) -> float:
        """Stamp the judge call returning."""
        self.judge_completed_ts = self.clock() if at is None else at
        return self.judge_completed_ts

    def set_source_media_ts(self, value: float | None) -> None:
        """Record the OPTIONAL source/fixture/RTSP-PTS timeline value, if the source exposes one."""
        self.source_media_ts = value

    def freeze(self, *, at: float | None = None) -> JudgmentTiming:
        """Stamp ``decision_emitted_ts`` and produce the frozen :class:`JudgmentTiming`.

        Raises ``ValueError`` if an earlier funnel stage was never marked — a judgment cannot
        freeze with gaps in its own monotonic history.
        """
        missing = [
            name
            for name, value in (
                ("ingest_observed_ts", self.ingest_observed_ts),
                ("admitted_ts", self.admitted_ts),
                ("judge_started_ts", self.judge_started_ts),
                ("judge_completed_ts", self.judge_completed_ts),
            )
            if value is None
        ]
        if missing:
            raise ValueError(f"cannot freeze JudgmentTiming, missing stamp(s): {missing!r}")
        return JudgmentTiming(
            ingest_observed_ts=self.ingest_observed_ts,  # type: ignore[arg-type]
            admitted_ts=self.admitted_ts,  # type: ignore[arg-type]
            judge_started_ts=self.judge_started_ts,  # type: ignore[arg-type]
            judge_completed_ts=self.judge_completed_ts,  # type: ignore[arg-type]
            decision_emitted_ts=self.clock() if at is None else at,
            source_media_ts=self.source_media_ts,
        )


@dataclass(frozen=True)
class JudgmentEvent:
    """The frozen output of one pass through the judge funnel (``Verdict -> GateDecision ->
    JudgmentEvent``, ADR-001 §1-5).

    Immutable (frozen dataclass) and frozen in time at ``timing.decision_emitted_ts`` — nothing on
    this event may be mutated after construction, and no delivery/sink timestamp is ever added
    here; those live in the ``DeliveryAttempt`` ledger precisely because that
    ledger records attempts made *against* an already-frozen event.

    ``key``/``evidence_seq`` mirror :class:`percept.sequencing.GateDecision` so a
    JudgmentEvent can be correlated back to the serialized-reducer decision that produced it —
    added here additively; the earlier GateDecision fields are untouched.
    """

    key: Hashable
    evidence_seq: int
    action: JudgmentAction
    timing: JudgmentTiming
    # --- Judgment-envelope enrichment — additive, defaulted, non-latency observability ----
    # The typed WHY/WHERE/HOW of this decision, so the frozen event a client reads at 2am is not silent
    # about the things it already knew. All OPTIONAL: an event built with none of them is byte-for-byte
    # the earlier envelope (and every existing keyword construction keeps its meaning — these append LAST).
    #
    # - ``reason``       — the closed ``percept.gate.GateReason`` code the gate decided by (``rising_edge``/
    #   ``dwell_satisfied``/``low_confidence``/``stale``/...). It already EXISTED on ``GateOutcome`` but only
    #   ever reached the health/trace channel; here it rides on the judgment itself.
    # - ``watchplan_id`` — a stable id of the PLAN this judgment ran under (a ``PlanIdentity.plan_hash``).
    #   The executor does not know the whole plan, so it leaves this ``None``; the runtime that
    #   owns the goal set stamps it on the delivered :class:`Incident`.
    # - ``roi``          — the region the judgment actually fired on, WHEN the judge/gate knows it. Left
    #   ``None`` honestly rather than echoing a ``WatchCondition.region_hint`` INPUT as if it were an output.
    # - ``modality``     — the :data:`EvidenceModality` the judgment was perceived under (vision/audio/fused).
    # - ``provenance``   — :data:`JudgmentProvenance`: ``observed`` (a direct read) vs ``inferred``.
    reason: str | None = None
    watchplan_id: str | None = None
    roi: "Box | None" = None
    modality: "EvidenceModality | None" = None
    provenance: "JudgmentProvenance | None" = None
    # 1.9 — WHICH firing path produced this judgment's evidence (:data:`EvidenceOrigin`): ``judged``
    # (the rising-edge watch path) vs ``transition`` (the A→B path that spends two judge calls). Appended
    # LAST, defaulted None; ``None`` == legacy and a reader treats it as ``"judged"``. A stable
    # ``judgment_event_id`` is deliberately NOT a field here — it is fully derivable from
    # (key's goal/entity, ``evidence_seq``) via :func:`judgment_event_id`, so storing it would carry zero
    # information; it lives on the delivered :class:`Incident`, the record a client actually correlates.
    evidence_origin: "EvidenceOrigin | None" = None
    # One provider dispatch may report several bounded temporal occurrences. Those are distinct
    # decisions over shared evidence, not pretend provider dispatches, so they retain one
    # ``evidence_seq`` and carry this optional discriminator to keep their evidence links injective.
    # ``None`` preserves the ordinary one-dispatch/one-decision contract byte-for-byte.
    occurrence_id: str | None = None


# ---------------------------------------------------------------------------
# GroundTruth — EVALUATION-ONLY ground truth, DISTINCT from the live JudgmentEvent.
# ---------------------------------------------------------------------------

#: Version of the eligibility-annotation RUBRIC below (bump on any wording/criteria change so an
#: existing :class:`GroundTruth` record self-documents which rubric revision produced it). Distinct
#: from ``CONTRACT_VERSION`` (the wire contract) and ``Concern.version`` (a versioned detection asset).
ANNOTATION_RUBRIC_VERSION = "eligibility-rubric-v1"

ANNOTATION_RUBRIC = """\
Eligibility annotation rubric ({version}).

decision_eligible_ts = the timestamp of the FIRST FRAME at which a competent human annotator,
given only the evidence up to and including that frame (no future frames, no foreknowledge of the
fixture's intended answer), could confidently judge whether the watched condition is satisfied.

It is NOT the frame where the underlying physical event starts (``event_start_ts``) — those
coincide only for an instantaneous/unambiguous event. For anything with onset/rise time (a door
swinging open, a person sitting down) eligibility comes strictly AFTER the event start, once
enough of the action has resolved to remove reasonable doubt.

Two substrates produce this timestamp:

1. BY CONSTRUCTION (primary, core gating archetypes) — a synthetic fixture author places the event
   at a known ``event_start_ts`` and DECLARES ``decision_eligible_ts`` deterministically (a fixed,
   documented offset for the archetype). No human judgment is exercised at record-construction
   time; the fixture generator IS the ground truth. Use :meth:`GroundTruth.by_construction`.
2. ANNOTATED REAL CLIPS (secondary, two-annotator) — a real/captured clip has no generator to ask,
   so a human annotator marks the frame. Two independent annotators reduce single-rater bias; a
   disagreement beyond tolerance is adjudicated (a third annotator, or the clip is discarded)
   rather than averaged away blindly. :meth:`GroundTruth.from_two_annotator` is the documented
   INTAKE HOOK for this path — not a full annotation tool/UI, which is out of PRE-MVP VIABILITY
   scope.

``annotation_version`` on every record MUST equal (or explicitly reference) the rubric version that
produced it, so a later rubric revision does not silently corrupt historical golden fixtures.
""".format(version=ANNOTATION_RUBRIC_VERSION)


@dataclass(frozen=True)
class GroundTruth:
    """EVALUATION-ONLY ground truth for one watched event — NEVER a field on a live JudgmentEvent.

    Per ADR-001 §5: the production runtime cannot (and must not pretend
    to) know when an event "really" started or when a human could have confidently judged it —
    those are labels a human annotator or a synthetic fixture generator supplies AFTER THE FACT,
    for evaluation only (recall/precision scoring, the ``golden_eval`` timeline, the
    actionable-latency metric). Keeping this a wholly separate, distinct type — never
    merged into :class:`JudgmentEvent`/:class:`JudgmentTiming` — is a hard invariant of the frozen
    contract: a live event built from real traffic never carries an ``event_start_ts``/
    ``decision_eligible_ts`` field, STRUCTURALLY, not just by convention. It is nullable/absent on
    the live path simply because a live pipeline never constructs one of these.

    See :data:`ANNOTATION_RUBRIC` for the eligibility rubric that MUST have produced
    ``decision_eligible_ts``.

    Fields:

    - ``event_start_ts`` — when the underlying physical/scripted event began, on the fixture's or
      annotator's own timeline (evaluation-side; NOT the runtime's monotonic clock — analogous in
      spirit to the untrusted ``source_media_ts`` on :class:`JudgmentTiming`, but a wholly separate
      timeline belonging to a wholly separate, eval-only type).
    - ``decision_eligible_ts`` — the rubric-defined first-eligible-judgment instant (see
      :data:`ANNOTATION_RUBRIC`). Always ``>= event_start_ts``.
    - ``annotation_version`` — REQUIRED (non-empty); the :data:`ANNOTATION_RUBRIC_VERSION` (or a
      documented successor) that produced this record. A ``golden_eval`` run that silently mixes
      rubric versions is a scoring bug this field exists to catch.
    - ``annotator_or_fixture_id`` — REQUIRED (non-empty) provenance: which fixture generator
      (by-construction path) or which human annotator/annotator-pair (real-clip path) produced
      this record.
    """

    event_start_ts: float
    decision_eligible_ts: float
    annotation_version: str
    annotator_or_fixture_id: str

    def __post_init__(self) -> None:
        if self.decision_eligible_ts < self.event_start_ts:
            raise ValueError(
                "GroundTruth.decision_eligible_ts must be >= event_start_ts (eligibility cannot "
                f"precede the event it is judging): decision_eligible_ts={self.decision_eligible_ts!r} "
                f"< event_start_ts={self.event_start_ts!r}"
            )
        if not self.annotation_version:
            raise ValueError(
                "GroundTruth.annotation_version is required (non-empty) — see ANNOTATION_RUBRIC "
                "and ANNOTATION_RUBRIC_VERSION."
            )
        if not self.annotator_or_fixture_id:
            raise ValueError("GroundTruth.annotator_or_fixture_id is required (non-empty).")

    @classmethod
    def by_construction(
        cls,
        *,
        event_start_ts: float,
        eligible_after_s: float,
        fixture_id: str,
        annotation_version: str = ANNOTATION_RUBRIC_VERSION,
    ) -> "GroundTruth":
        """PRIMARY path for the core gating archetypes: a synthetic fixture
        DECLARES eligibility deterministically as a fixed offset from its own known event start —
        no human judgment, no randomness, so two calls with identical inputs always produce a
        byte-for-byte identical record (reproducible golden fixtures).

        ``eligible_after_s`` is the fixture author's documented reaction-time/resolution offset for
        THIS archetype (see ``eval/viability/ground_truth.py`` for the six core-archetype offsets
        the neutral fixtures use) — MUST be ``>= 0``.
        """
        if eligible_after_s < 0:
            raise ValueError(f"eligible_after_s must be >= 0, got {eligible_after_s!r}")
        return cls(
            event_start_ts=event_start_ts,
            decision_eligible_ts=event_start_ts + eligible_after_s,
            annotation_version=annotation_version,
            annotator_or_fixture_id=fixture_id,
        )

    @classmethod
    def from_two_annotator(
        cls,
        *,
        event_start_ts: float,
        annotator_a_id: str,
        annotator_a_eligible_ts: float,
        annotator_b_id: str,
        annotator_b_eligible_ts: float,
        annotation_version: str = ANNOTATION_RUBRIC_VERSION,
        max_disagreement_s: float = 1.0,
    ) -> "GroundTruth":
        """SECONDARY path for annotated REAL clips — a documented INTAKE HOOK, not a
        full annotation tool/UI. Two independent human annotators each mark their own eligibility
        instant; this reconciles them into ONE record.

        Raises :class:`ValueError` if the two annotators disagree by more than
        ``max_disagreement_s`` — adjudicate with a third annotator, or discard the clip; this hook
        deliberately will not average away a real disagreement. Within tolerance, records the mean
        of the two timestamps and a combined ``annotator_or_fixture_id`` (both ids, so provenance is
        never lost).
        """
        disagreement = abs(annotator_a_eligible_ts - annotator_b_eligible_ts)
        if disagreement > max_disagreement_s:
            raise ValueError(
                f"annotators disagree by {disagreement!r}s (> max_disagreement_s={max_disagreement_s!r}s): "
                f"{annotator_a_id}={annotator_a_eligible_ts!r} vs {annotator_b_id}={annotator_b_eligible_ts!r} "
                "— adjudicate with a third annotator or discard this clip; this hook will not "
                "silently average away a real disagreement."
            )
        reconciled = (annotator_a_eligible_ts + annotator_b_eligible_ts) / 2.0
        return cls(
            event_start_ts=event_start_ts,
            decision_eligible_ts=reconciled,
            annotation_version=annotation_version,
            annotator_or_fixture_id=f"{annotator_a_id}+{annotator_b_id}",
        )


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
    """When a watch is *armed* — a coarse, vendor-neutral, timezone-naive time gate.

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


SalienceStat = Literal["mean", "block_max", "p95"]
"""How the motion channel summarizes a frame's abs-diff. ``mean`` (default) = whole-area mean — right
for scene-filling events. ``block_max``/``p95`` = a coarse block grid's top / 95th-percentile block, so
a *localized* event (a hand, a cup) on a static camera clears threshold even when its whole-frame mean
is ~0. Tracks the camera MOUNT, not the goal: a static desk/security cam picks ``block_max``."""


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
    salience_stat: SalienceStat = "mean"  # localizing statistic for the motion channel (mean -> legacy path)
    ego_compensate: bool = False  # remove whole-frame translation (camera pan / head turn) before diffing


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
    # 1.3 — the audio-energy channel value (mean-of-squares of float PCM in [-1,1], so 0..1) when this
    # signal came from an audio/frame-free ingest path (edge ``processAudio``). ``None`` means "not an
    # audio-carrying signal" — distinct from 0.0 (measured silence) — so the frame path is unchanged.
    energy: float | None = None


# ---------------------------------------------------------------------------
# Incident — a fired-event record carrying a captured relevant snippet.
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
    """A typed audio signal — the first-class AUDIO modality the eye can read.

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

    ``entity_id`` (1.3) is the GENERIC correlation handle of the subject this fire was gated on
    — the same ``entity_id`` that already keys the rising-edge gate (``(goal_id, entity_id)``) and
    rides on ``FireEvent``/``WatchEvent``. Without it two independent subjects tripping the same
    concern are indistinguishable downstream, which reads as spurious out-of-order/duplicate
    divergence. It is a correlation key ONLY: it carries no claim about *who* or *which document*
    the subject is (richer subject/incident relations are ADR-006). ``None`` when the fire was not
    entity-scoped (single-subject streams, boundary fires) — which is the back-compatible default.
    """

    id: str
    concern_id: str
    verdict: Verdict
    ts: float
    label: str | None = None  # e.g. "real" / "false_alarm" / None (unlabelled)
    snippet: IncidentSnippet | None = None
    message: str = ""  # the human-facing alert line this fire emitted (the goal's `say`)
    # A reserved link a webhook consumer follows to label this fire. Populated only by a deployment
    # that serves its own label endpoint; ``None`` on the keyless path. Reserved so the wire shape is
    # stable for consumers that do wire one up.
    label_url: str | None = None
    # 1.3: the correlation handle of the gated subject. Appended LAST so every existing
    # positional construction of Incident(...) keeps its meaning.
    entity_id: str | None = None
    # 1.6 — the enriched judgment envelope, carried THROUGH the emission boundary onto the
    # durable record a client actually receives. Domain probes found percept perceived correctly
    # but was UNOBSERVABLE: the reason a fire was withheld, the plan it ran under, the region/
    # modality/provenance of the evidence — all existed internally yet none reached a delivered Incident.
    # These do. All additive, defaulted, appended LAST (positional constructions unchanged), and a
    # consumer that ignores them sees byte-for-byte today's Incident. See :class:`JudgmentEvent` for the
    # per-field meaning; ``make_incident`` copies them off the fire's ``JudgmentEvent``/``FireEvent``.
    reason: str | None = None
    watchplan_id: str | None = None
    roi: "Box | None" = None
    modality: "EvidenceModality | None" = None
    provenance: "JudgmentProvenance | None" = None
    # 1.7 — the per-key ADR-001 §8 dispatch sequence this fire's :class:`JudgmentEvent` carried
    # (``JudgmentEvent.evidence_seq``), copied off the fire so a delivered Incident carries the SAME
    # correlation handle a ``.ptrace`` judgment record does. Live transition fires carry this
    # sequence too (with ``evidence_origin="transition"``); ``None`` when the fire carried no
    # JudgmentEvent (episode/rhythm/boundary/absence, or a legacy direct transition-executor call
    # without timing) or for a pre-1.7 Incident — an absent link, never a fabricated one. Appended
    # LAST: every existing positional ``Incident(...)`` construction keeps its meaning.
    evidence_seq: int | None = None
    # 1.8 — the episode-owned provisional/superseding link, ON THE WIRE: a claim-mode correction
    # (a later turn the mutation channel classifies replace/retract on the same episode) fires a
    # SUPERSEDING revision, and until now that relation existed only as a runtime-local annotation
    # (``percept.incidents.incident_revision``/``incident_supersedes``) invisible to an out-of-process
    # webhook consumer. ``revision_id`` is THIS incident's stable episode-revision handle
    # (``<episode_id>#r<n>``, :func:`percept.behavior.revision_id`); ``supersedes`` names the PRIOR
    # revision this one corrects, and is ``None`` for an original/provisional incident (the common
    # case). ``make_incident`` copies both straight off the fire's ``FireEvent``/``Fired``. ``None``
    # when the fire carried no episode fold (episode/rhythm/boundary/transition/absence fires, or a
    # pre-1.8 Incident) — an absent link, never a fabricated one. Additive, defaulted, appended LAST:
    # every existing positional ``Incident(...)`` construction keeps its meaning, and a consumer that
    # ignores these two fields sees byte-for-byte the pre-1.8 Incident.
    revision_id: str | None = None
    supersedes: str | None = None
    # 1.9 — the durable EVIDENCE LINK a client audit resolves against. ``judgment_event_id`` is the
    # OPAQUE, INJECTIVE handle (:func:`judgment_event_id`) that a ``judgment.event`` trace row ALSO
    # carries, so a delivered Incident joins back to EXACTLY ONE judgment record (equality-compare only;
    # never parse it). ``evidence_origin`` (:data:`EvidenceOrigin`) names WHICH firing path produced the
    # evidence — ``judged`` vs ``transition`` — so the audit can tell a rising-edge fire from the A→B
    # transition path that, from 1.9 on, also emits a judgment record. Both ``None`` when the fire carried
    # no JudgmentEvent (episode/rhythm/boundary/absence fires take no judge-funnel stamps) or for a
    # pre-1.9 Incident — an absent link, never a fabricated one; a ``None`` origin is read as ``"judged"``.
    # ``make_incident`` derives ``judgment_event_id`` from the fire's (goal_id, entity_id, evidence_seq)
    # and copies ``evidence_origin`` off the fire's ``JudgmentEvent``. Additive, defaulted, appended LAST:
    # every existing positional ``Incident(...)`` keeps its meaning, and a consumer that ignores both sees
    # byte-for-byte the pre-1.9 Incident.
    judgment_event_id: str | None = None
    evidence_origin: "EvidenceOrigin | None" = None

    def bind_labeler(
        self,
        sink: "Callable[[LabelRecord], Any]",
        *,
        verdict_ref: str = "",
        goal_id: str | None = None,
        calibration_version: str | None = None,
        frame_ref: str | None = None,
    ) -> "Incident":
        """Attach the LOCAL label sink + join context so :meth:`record_label` can write a
        :class:`LabelRecord`. The producer (``percept.run(label_store=...)`` /
        :class:`RuntimeSession`) calls this on each emitted incident. The bound state lives in a
        private instance attribute — NOT a dataclass field — so it never serializes, egresses, or
        appears in the JSON schema. Returns ``self`` for chaining."""
        self._label_sink = sink  # type: ignore[attr-defined]
        self._label_ctx = {  # type: ignore[attr-defined]
            "verdict_ref": verdict_ref,
            "goal_id": goal_id if goal_id is not None else self.concern_id,
            "calibration_version": calibration_version,
            "frame_ref": frame_ref,
        }
        return self

    def record_label(
        self,
        good: bool,
        note: str | None = None,
        tags: "tuple[str, ...] | list[str]" = (),
        *,
        source: Literal["api", "voice", "ask_answer", "watch_calibration"] = "api",
    ) -> "LabelRecord":
        """Record a ground-truth label on THIS fire — the feedback intake.

        Builds a :class:`LabelRecord` (``source="api"`` by default — a human via the SDK) and appends it
        to the local label store bound by the producer (``percept.run(label_store=...)``). Named
        ``record_label`` — NOT ``label`` — because :attr:`Incident.label` is already the ``str | None``
        display field; a dataclass field and a method cannot share a name, and renaming the field would
        break every existing reader (``feedback(incident_id, label)``, dashboards). Raises
        :class:`RuntimeError` if no store is bound (there is nothing local to write to)."""
        sink = getattr(self, "_label_sink", None)
        if sink is None:
            raise RuntimeError(
                "no label store bound to this incident — pass label_store= to percept.run()/start() "
                "(or append to a LabelStore directly). incident.record_label() is a LOCAL write."
            )
        ctx = getattr(self, "_label_ctx", {})
        rec = LabelRecord(
            incident_id=self.id,
            verdict_ref=ctx.get("verdict_ref", ""),
            goal_id=ctx.get("goal_id") or self.concern_id,
            ts=time.time(),
            good=bool(good),
            note=note,
            tags=list(tags),
            frame_ref=ctx.get("frame_ref"),
            calibration_version=ctx.get("calibration_version"),
            source=source,
        )
        sink(rec)
        return rec


# ---------------------------------------------------------------------------
# AttentionIntent — the channel-neutral EFFECTIVE-attention membrane.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AttentionIntent:
    """What a fired :class:`Incident` asks of the human's attention — the EFFECTIVE, not authored, request.

    The application embodiment plane (floor, speech, delivery) consumes this instead of reaching into the
    goal's raw ``delivery`` dict, so two hard invariants live in one place (see ``AttentionPolicy``):

    1. **Priority is effective, never authored.** ``preempt`` here is the value AFTER deterministic
       clamping — a conductor (an untrusted language author) cannot grant itself ``"interrupt"`` merely by
       emitting a delivery dict; only a trusted :class:`GoalOrigin` (or an explicit policy allowlist) keeps
       it. The gap between requested and effective is the spec's two-value pattern.
    2. **No channel prescription.** ``preferred_channel`` is a free-form HINT, not a modality enum. The SDK
       must not prescribe application channels: a robot ACT is not just another notification modality (it
       may need separate authorization), so the app — not this contract — decides speech/UI/webhook/action.

    Fields:

    - ``message`` — the human-facing line to deliver (the goal's ``say`` / the incident message).
    - ``goal_id`` / ``incident_id`` — provenance handles back to the firing goal + durable incident record.
    - ``ts`` — the incident timestamp (source clock), or ``None`` when unstamped.
    - ``preempt`` — the EFFECTIVE floor disposition: ``"interrupt"`` (cut into a live turn), ``"queue"``
      (hold behind the current turn), or ``"drop"`` (discard if it cannot be delivered now).
    - ``ttl_s`` — how many seconds a queued intent stays fresh; past it a late nudge is worse than none
      and the floor drops it. ``None`` = no expiry.
    - ``verbatim`` — deliver ``message`` exactly (no paraphrase) — for scripted/compliance alerts.
    - ``preferred_channel`` — an OPTIONAL app-defined hint (e.g. ``"speech"`` / ``"webhook"``); ``None``
      lets the app choose. NEVER an authorization: the app validates capability + permission per channel.
    """

    message: str
    goal_id: str
    incident_id: str
    ts: float | None
    preempt: Literal["interrupt", "queue", "drop"]
    ttl_s: float | None
    verbatim: bool
    preferred_channel: str | None = None


# ---------------------------------------------------------------------------
# LabelRecord — one ground-truth label on a fire.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LabelRecord:
    """A single ground-truth label on a fired verdict.

    **One schema, three producers** (do not fork it): manual :meth:`Incident.record_label`
    (``source="api"``), ASK answers (``source="ask_answer"``), and watch-calibration reviews
    (``source="watch_calibration"``).

    **The egress invariant (media-free head / local-only tail).** The record splits in two: every field
    ABOVE ``frame_ref`` is the *metadata head* and MAY leave the machine; ``frame_ref`` is a LOCAL
    filesystem path (the pixels' whereabouts) that must NEVER egress. :meth:`to_egress` is built from an
    explicit :attr:`EGRESS_FIELDS` whitelist that structurally excludes ``frame_ref`` — so the no-pixels
    rule is a *type-level fact*, not a runtime filter you can forget to apply. The full record (incl.
    ``frame_ref``) is written LOCALLY by the :class:`~percept.labelstore.LabelStore` via
    :meth:`to_local_dict`; only :meth:`to_egress` ever feeds a network path.
    """

    incident_id: str
    verdict_ref: str  # hash/id of the Verdict that fired — the join key to usage (see labelstore.verdict_ref)
    goal_id: str
    ts: float
    good: bool  # was this fire correct? (False = a false alarm the user had to dismiss)
    note: str | None = None
    tags: list[str] = field(default_factory=list)
    frame_ref: str | None = None  # LOCAL filesystem path ONLY; the local-only tail — NEVER egressed
    calibration_version: str | None = None  # which prior policy produced the fire
    source: Literal["api", "voice", "ask_answer", "watch_calibration"] = "api"

    #: The egress whitelist — the metadata head, in declaration order, EXCLUDING ``frame_ref`` (the
    #: local tail). :meth:`to_egress` emits exactly these names, so ``frame_ref`` is structurally
    #: un-emittable. A ``ClassVar`` (not a dataclass field): it never serializes and never reaches the
    #: JSON schema.
    EGRESS_FIELDS: ClassVar[tuple[str, ...]] = (
        "incident_id",
        "verdict_ref",
        "goal_id",
        "ts",
        "good",
        "note",
        "tags",
        "calibration_version",
        "source",
    )

    def to_egress(self) -> dict[str, Any]:
        """The metadata-head dict that MAY leave the machine — structurally cannot contain ``frame_ref``.

        Built from the :attr:`EGRESS_FIELDS` whitelist (which omits ``frame_ref``), so there is no code
        path — no ``**asdict``, no field iteration — by which the local media pointer leaks to egress."""
        return {name: getattr(self, name) for name in self.EGRESS_FIELDS}

    def to_local_dict(self) -> dict[str, Any]:
        """The FULL record — including the local-only ``frame_ref`` — for the on-box
        :class:`~percept.labelstore.LabelStore`. NEVER hand this to a network sink; use
        :meth:`to_egress` for anything that leaves the machine."""
        return {
            "incident_id": self.incident_id,
            "verdict_ref": self.verdict_ref,
            "goal_id": self.goal_id,
            "ts": self.ts,
            "good": self.good,
            "note": self.note,
            "tags": list(self.tags),
            "frame_ref": self.frame_ref,
            "calibration_version": self.calibration_version,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "LabelRecord":
        """Reconstruct a record from a JSONL row (the inverse of :meth:`to_local_dict`). Unknown keys
        are ignored; missing optionals fall back to their defaults, so an older row still loads."""
        return cls(
            incident_id=d["incident_id"],
            verdict_ref=d.get("verdict_ref", ""),
            goal_id=d.get("goal_id", ""),
            ts=d["ts"],
            good=bool(d["good"]),
            note=d.get("note"),
            tags=list(d.get("tags") or []),
            frame_ref=d.get("frame_ref"),
            calibration_version=d.get("calibration_version"),
            source=d.get("source", "api"),
        )


# ---------------------------------------------------------------------------
# WatchEvent — the event the public ``watch()`` entrypoint streams.
# ---------------------------------------------------------------------------

WatchAction = Literal["fire", "ask", "silent"]
"""The three-state gate's outward action. Mirrors ``percept.gate.GateAction`` value-for-value; kept as
an independent ``Literal`` here (not imported) so the contracts package stays dependency-free of the
cognition core, per this module's zero-dependency design."""


@dataclass
class WatchEvent:
    """The stable, versioned event ``percept.watch()`` streams — the frozen public contract.

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
# Configuration ownership / resolution / provenance DTOs (ADR-008). Defined in the sibling
# .watchplan module (kept separate so its own docstring can stay focused) and re-exported here so
# ``from percept.contracts import WatchPlan`` keeps working the same way every other DTO does.
# ---------------------------------------------------------------------------

from .watchplan import (  # noqa: E402 — re-export after the DTOs those bump comments describe
    RECEIPT_STATUSES,
    ArtifactBinding,
    FieldDiagnostic,
    ReceiptStatus,
    ResolvedJudgeSpec,
    ResolvedWatch,
    SettingOrigin,
    WatchPlan,
    WatchReceipt,
)

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

        ``context`` is OPTIONAL structured facts the CALLER supplies (e.g. an LPR model's
        ``{"plate": "7XYZ123"}``, an OCR read's ``{"invoice_number": "INV-2291"}``) — read as
        authoritative, distinct from the user-asserted ``corrections``. There is no registry that fills
        it for you (the dead enricher registry was removed): a caller that wants specialist facts in
        the judge passes them here. It defaults ``None`` so an un-enriched call is byte-for-byte today;
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
    """DEPRECATED — the shape of a function that produces structured judge ``context``.

    **The enricher REGISTRY is gone.** ``percept.harness.enrichers`` (``register_enricher`` /
    ``enrich_signal`` / ``enricher_for``) shipped as "the way to add an OCR/LPR specialist without
    editing core" and then acquired ZERO callers anywhere in the SDK — nothing in the runtime ever ran a
    registered enricher, so registering one had no effect on any watch. It was removed rather than left
    standing as an architecture-shaped seam a client could find, wire up, and get silence from. This
    protocol is retained ONLY so the name does not disappear from a published contract MAJOR
    (``CONTRACT_VERSION`` 1.x); it is scheduled for removal in the next MAJOR and nothing consumes it.

    What replaced it, precisely:

    - to decide **when to look closely** (admission) — register a detector
      (:func:`percept.edge.detectors.register_detector`) or pass a tier-0 adapter
      (``percept.run(..., semantic_adapter=...)``). Both are OCR-capable and both are *admission*.
    - to get **facts into the judge** — call the judge yourself with ``context={...}``:
      :meth:`Vision.judge` accepts it directly. That parameter is live and wired; it simply never had a
      registry behind it.

    See ``docs/reference/registration.md`` ("Admission is not extraction")."""

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


@dataclass(frozen=True)
class UtteranceEvent:
    """A complete spoken user turn, endpointed by a :class:`VoiceTransport` (with AEC applied).

    .. deprecated:: 0.6
        Part of the :class:`VoiceTransport` seam, which moved app-side in the SDK voice diet — no SDK-core
        behavior consumes it after the diet. It stays FUNCTIONAL this release as a migration aid; the
        template owns its own ``SpeechRequest``/``VoiceEngine`` types (``engine_adapter.py``). Slated for
        removal at v0.7/v1.0 unless an SDK-level consumer appears. See ``templates/realtime_voice_watch_agent``.

    ``text`` is the transcript when the transport did its own STT (Pipecat/LiveKit, or a speech-to-speech
    model); ``audio`` carries raw PCM when the transport ships audio for the brain to transcribe. At least
    one is set. ``t`` is the transport clock at the endpoint."""

    text: str | None = None
    audio: "AudioFrame | None" = None
    t: float | None = None


@dataclass(frozen=True)
class Reply:
    """The brain's answer handed back to a :class:`VoiceTransport`. ``text`` for a component transport to
    synthesize (TTS); ``audio`` when the brain already holds speech (e.g. from a live model). Both may be
    ``None`` for a speech-to-speech transport that already spoke on its own.

    .. deprecated:: 0.6
        Part of the :class:`VoiceTransport` seam, which moved app-side in the SDK voice diet. FUNCTIONAL
        this release as a migration aid; the template uses its own ``SpeechRequest`` instead. Slated for
        removal at v0.7/v1.0 unless an SDK-level consumer appears. See ``templates/realtime_voice_watch_agent``."""

    text: str | None = None
    audio: "AudioFrame | None" = None


@runtime_checkable
class VoiceTransport(Protocol):
    """The bidirectional VOICE seam: whoever owns mic + speaker, acoustic echo cancellation (AEC),
    voice-activity endpointing, and barge-in (Pipecat, LiveKit, browser WebRTC, a speech-to-speech
    model). Distinct from the chunk-level :class:`STT`/:class:`TTS` *component* seams — a transport owns
    the whole audio *pipe*.

    .. deprecated:: 0.6
        The voice embodiment moved app-side in the SDK voice diet: ``RuntimeSession.attach_transport`` is
        now a raising stub, and no SDK-core behavior consumes this Protocol. It stays DEFINED (functional)
        this release as a migration seam. The one shipped voice application,
        ``templates/realtime_voice_watch_agent``, owns its own ``VoiceEngine(Protocol)``/``SpeechRequest``
        (``engine_adapter.py``) — the component that actually needs engine portability. Slated for removal
        from core contracts at v0.7/v1.0 unless an SDK-level consumer appears.

    Percept never hand-rolls this (no AEC in Python). Historically a client implemented this Protocol and
    handed it to ``RuntimeSession.attach_transport``, which wired it to cognition (endpointed utterance ->
    frame-grounded turn -> streamed reply; barge-in -> interrupt). That wiring now lives in the template's
    ``PerceptBridge`` against its own ``VoiceEngine`` seam, not here.

    Two archetypes, one interface:
      * component (Pipecat/LiveKit + STT/TTS): delivers ``text``; :meth:`say` synthesizes the reply.
      * speech-to-speech (Gemini Live, OpenAI Realtime): its own model hears + speaks; the brain's role is
        to expose command tools (:meth:`register_tool`), keep the wire GROUNDED (:meth:`sync_state`), and
        speak proactive fires (:meth:`say`). The wire is not an independent oracle — it answers reactive
        questions from what the brain has told it, not from raw frames alone.

    **Resume is transparent.** A real transport's underlying session may end + reconnect (a realtime model
    caps duration; context compacts). Across an internal resume the transport MUST silently re-establish
    registered tools, the frame provider, and the last :meth:`sync_state` — the caller sees ONE continuous
    session and never re-registers anything. All registration methods are therefore idempotent.
    """

    async def start(self) -> None:
        """Open the transport (room / PeerConnection / mic). Idempotent."""
        ...

    async def stop(self) -> None:
        """Tear the transport down. MUST NOT raise (teardown path)."""
        ...

    def on_utterance(self, cb: "Callable[[UtteranceEvent], Awaitable[None] | None]") -> None:
        """Register the up-channel sink for a complete (AEC'd, endpointed) user turn. Sync or async cb."""
        ...

    def on_barge_in(self, cb: "Callable[[], Awaitable[None] | None]") -> None:
        """Register a sink fired when the user starts speaking during playout (-> ``interrupt``)."""
        ...

    def on_event(self, cb: "Callable[[dict], Awaitable[None] | None]") -> None:
        """Optional telemetry sink: ``{"kind": "agent_reply"|"partial"|"error"|..., ...}``. A
        speech-to-speech transport surfaces what it SAID here so the brain can log + arm goals."""
        ...

    def on_user_speaking(self, cb: "Callable[[bool], Awaitable[None] | None]") -> None:
        """Register an up-channel signal that the user's turn started (``True``) / ended (``False``). The
        brain uses it to time proactive fires (interrupt vs. queue vs. drop) and to price ``SILENT``
        correctly during a live conversation — a fire mid-sentence, or stale, is worse than silence."""
        ...

    def on_bot_speaking(self, cb: "Callable[[bool], Awaitable[None] | None]") -> None:
        """Register an up-channel signal that the AGENT's voice output started (``True``) / stopped
        (``False``) — the moment audio actually begins playing, NOT when a reply was enqueued. This is the
        honest end of the audio-to-voice (A2V) latency window: measuring to ``say()``/enqueue understates
        true latency by the model's TTS first-token time. **Optional/additive** — a transport that can't
        observe its own playout simply never calls it (the brain then can't price A2V for that wire, but
        everything else works). Consumers wire it defensively (``hasattr``) so older transports still bind."""
        ...

    def on_transcript(self, cb: "Callable[[str], Awaitable[None] | None]") -> None:
        """Register an up-channel sink for the user's turn TRANSCRIPT (a realtime model emits these
        natively — this reads a field you already pay for, it is NOT adding STT). LOW priority: the brain
        mines it for evidence ("that's my daughter Maya") + latent intent; nobody is waiting on it."""
        ...

    async def say(self, reply: "Reply", *, verbatim: bool = False) -> None:
        """Deliver the brain's line (a reply, or a proactive fire). ``verbatim=False`` (default) = CONVEY:
        the transport may re-render the intent in its own voice (all a speech-to-speech wire can reliably
        do). ``verbatim=True`` requests the EXACT wording — a **capability a transport may not honor**
        (an S2S model best-efforts it; a TTS transport guarantees it). Policy-exact wording that MUST be
        verbatim should also ride the act channel (``on_incident`` records the message verbatim)."""
        ...

    async def sync_state(self, summary: str) -> None:
        """Make a compact BRAIN-STATE summary available to the wire WITHOUT triggering a response
        (armed goals, recent evidence labels, last fires — goal ids, never delivery scripts). Keeps a
        speech-to-speech wire's reactive answers coherent with the brain — so "do you see my coffee?"
        agrees with a caffeine fire 90s ago — and makes any drift in the wire's own memory harmless.
        Idempotent; re-synced transparently on resume.

        "Without triggering a response" is a HARD requirement, not advisory: only :meth:`say` may cause
        unprompted audio. A backend that cannot append context silently (e.g. Gemini Live answers every
        appended turn out loud) must NOT push — stash the summary and expose it for PULL instead (a
        state tool the model calls), which is how the template's ``PipecatVoiceEngine`` implements it."""
        ...

    async def interrupt(self) -> None:
        """Stop the current playout (the downstream half of barge-in). Idempotent."""
        ...

    def set_frame_provider(self, provider: "Callable[[], Any]") -> None:
        """Give the transport a pull for the current visual frame (``('image/jpeg', bytes)`` | None) so a
        speech-to-speech model can attach it per turn (frame-grounded 'what do you see?')."""
        ...

    def register_tool(self, name: str, schema: dict,
                      handler: "Callable[[dict], Awaitable[dict] | dict]") -> None:
        """Expose a tool (e.g. ``arm_goal``) to a transport whose model can call tools. Best-effort: a
        transport that can't host tools drops it and the brain routes tool intent via the conductor."""
        ...


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
    "ClipEvent",
    "SubjectRef",
    "PROVISIONAL_KEY_PREFIX",
    "provisional_subject",
    "SubjectConflict",
    "SubjectResolution",
    "SUBJECT_RESOLVER_PRECEDENCE",
    "resolve_subject",
    "Verdict",
    "Operation",
    "OPERATIONS",
    "MUTATING_OPERATIONS",
    "Provenance",
    "Attribute",
    "Goal",
    "GoalOrigin",
    "Cost",
    "FalseFireCost",
    "Consequence",
    "ConcernAuthority",
    "Evidence",
    "TrackingBehavior",
    "Subject",
    "Priors",
    "Concern",
    "Decision",
    "JudgmentAction",
    "EvidenceOrigin",
    "judgment_event_id",
    "JudgmentTiming",
    "JudgmentTimingBuilder",
    "JudgmentEvent",
    "ANNOTATION_RUBRIC_VERSION",
    "ANNOTATION_RUBRIC",
    "GroundTruth",
    "STTEvent",
    "TurnResult",
    "Schedule",
    "minute_of_day",
    "GateOn",
    "SalienceStat",
    "WatchCondition",
    "WatchSpec",
    "Tier0Signal",
    "AudioFrame",
    "IncidentSnippet",
    "Incident",
    "AttentionIntent",
    "LabelRecord",
    "WatchAction",
    "WatchEvent",
    # Configuration ownership / resolution / provenance DTOs (ADR-008)
    "ReceiptStatus",
    "RECEIPT_STATUSES",
    "FieldDiagnostic",
    "SettingOrigin",
    "WatchPlan",
    "ResolvedWatch",
    "WatchReceipt",
    "ArtifactBinding",
    "ResolvedJudgeSpec",
    # Protocols
    "Vision",
    "Specialist",
    "Enricher",
    "Conductor",
    "STT",
    "TTS",
    "UtteranceEvent",
    "Reply",
    "VoiceTransport",
    "Identity",
    "Persistence",
]
