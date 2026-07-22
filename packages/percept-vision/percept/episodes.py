"""Episode counting — the cup problem (spec §5.5) and the THREE SEPARATED DECISIONS (card Y1).

Counting "cups of coffee" is **episode counting**: not sip-counting (over-counts) and not
IoU-track-counting (mis-counts across gaps). Per counting goal we keep a bounded **episode log**
(the counting window — "today's cups", a handful); the count is the number of distinct episodes.

Card Y1 — subject / continuation / mutation are THREE decisions, not one
------------------------------------------------------------------------
The original design collapsed them into one judge call ("which episode does this continue?"). That
is wrong, and wrong in the expensive direction: it fixes duplicate events by MANUFACTURING FALSE
CONTINUITY. PO-4471 is the same **subject** every time it appears; it is not necessarily the same
**episode**. A PO opened, closed and reopened. A person who leaves and re-enters. A door that opens
twice. The same claim slot established in two separate calls. The same invoice returning after the
continuity window as a new workflow occurrence. All of those are one subject and two episodes.

So the three decisions are named and evaluated separately here:

1. **SUBJECT** — *what is this observation about?* Exact subject equality (a judge-matched episode
   id today; a judge-EMITTED ``subject_key`` once cards Y5/Z3 land — pass ``subject_key=`` and no
   judge call is made). Subject equality is **NECESSARY EVIDENCE** for continuation, never its
   definition.
2. **CONTINUATION** — *does this observation extend the existing episode?* The conjunction below,
   evaluated in :meth:`EpisodeLog.observe_decision`::

       continues = (same_subject
                    and episode.status == "open"          # or reopen is allowed
                    and observation.ts - episode.last_seen < continuity_gap_s
                    and mutation_is_compatible)

3. **MUTATION** — *how does this finding modify the episode?* Card Y2 owns the mutation CHANNEL;
   Y1 lands only its seam: the :data:`MutationCheck` predicate (``(episode, exemplar) -> bool``),
   defaulting to "compatible", which participates in the conjunction as its last conjunct. Y2
   replaces the boolean with a richer mutation without touching the other two decisions.

Every denial is TYPED (:data:`CONTINUATION_REASONS`) and OBSERVABLE (a structured log line plus the
optional ``on_continuation`` hook) — a same-subject observation that opened a NEW episode is exactly
the thing a client debugging percept at 2am needs to see, and it must never be silent.

Continuation semantics are **per concern**, never one global constant: a document-presence episode
(the document is there until it is gone — minutes), a door-open interval (seconds) and a
conversational claim (a whole call) do not agree on any of ``continuity_gap_s``, ``allow_reopen`` or
``terminal_statuses``. That is :class:`EpisodePolicy`, resolved per goal by
:func:`episode_policy_of`.

The judge is injected (an LLM in real use; a deterministic stub in L1) so the dedup *logic* is
testable without a model — and so we never assert backend intelligence with a fake.

PERSISTENCE IS SESSION-ONLY — episodes do NOT survive a restart (ADR-006, card Y9)
---------------------------------------------------------------------------------

An :class:`EpisodeLog` is **process-lifetime state**. ``Percept.snapshot()`` carries ``goals``,
``gate``, ``entities`` and ``action_log``; it does **not** carry ``Percept.episodes`` (the per-goal
log map built in ``Percept.count_event``), so ``Percept.restore()`` cannot rehydrate it. After a
restart every counting goal starts again at zero episodes, and a subject last seen before the restart
is a NEW episode afterwards.

This **blocks any 24/7 viability claim** until a later phase gives episodes a persisted
representation. The bounds below (``window_s`` / ``max_episodes``) are a MEMORY story — they keep a
long-running process from growing without limit. They are not a DURABILITY story and must not be
cited as one.

Do not read a durable IDENTIFIER as durable STATE. A subject reference marked ``scope='durable'``
says the identifier is stable — ``PO-4471`` means the same thing next week — and says nothing at all
about restart-safety of the episode state keyed by it. The failure mode of conflating them is silent
under-counting after a redeploy, which nothing in this module can detect for you.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal

from .events import frame_nbytes
from .observability import get_logger, log

# judge(summaries, exemplar) -> the id of the episode this observation shares a SUBJECT with, or
# None for a new subject. NOTE (Y1): the judge answers decision (1) ONLY. Its answer is evidence for
# continuation, not continuation itself — the returned episode still has to pass the continuation
# conjunction. The name is historical; ``subject_key=`` is the forward-looking seam (Y5/Z3).
EpisodeJudge = Callable[[list[dict], Any], "str | None"]

# The Y2 seam: does this finding MODIFY the episode in a way compatible with continuing it? A
# retraction, a contradicting value or a status flip is incompatible -> a new episode, not a
# silently mutated old one. Default (no check supplied) is "compatible", which is byte-for-byte the
# pre-Y1 behavior.
MutationCheck = Callable[["Episode", Any], bool]

# ================================================================================================
# EVIDENCE HANDLES (card Y14) — an episode retains a HANDLE, never the decoded frame.
# ================================================================================================
#
# The defect: ``Episode.exemplar`` / ``Episode.representative`` are opaque ``Any`` and the counting
# path folded DECODED FRAMES into them. Z1 already proved retention is a fixed plateau of
# ~50 x frame_bytes for the FrameWindow (46 MB at 640x480, ~300 MB at 1080p); an episode log that
# ALSO pins a frame per episode multiplies that, and ``max_episodes`` bounds the COUNT of episodes,
# never their BYTES — so a per-episode frame makes the log's memory scale with source resolution,
# exactly the class of leak Z1's byte budget closed for the window.
#
# The fix is the ADR-003 opaque-handle shape (docs/adr/ADR-003): store a small descriptor — an
# opaque ``ref`` and a cheap ``fingerprint`` — NEVER the pixels. A handle carries no ``nbytes`` /
# ``w`` / ``h`` / ``size`` attribute, so the SAME measuring function Z1 shipped
# (:func:`percept.events.frame_nbytes`) reports it as 0 (unmeasurably small) — retained evidence
# bytes then stop scaling with resolution and with N observations. ``source_nbytes`` records how big
# the frame this handle STANDS IN FOR was, for observability (how much was NOT retained); it is a
# deliberately-named plain int, not ``nbytes``, precisely so ``frame_nbytes`` never mistakes the tiny
# handle for the big frame it replaced.


@dataclass(frozen=True)
class EvidenceHandle:
    """An opaque HANDLE to evidence — a small descriptor, NEVER a decoded frame (card Y14, ADR-003).

    ``ref`` is an opaque store key in the ADR-003 sense: a caller MUST NOT parse it, construct one, or
    assume it is a filesystem path. ``fingerprint`` is a cheap dHash-style signature (see
    :func:`percept.cache.framehash.dhash`) the :class:`EpisodeJudge` can compare on WITHOUT the
    pixels. ``source_nbytes`` is the measured size of the frame this handle replaced (0 = unknown) —
    observability only, and deliberately NOT called ``nbytes`` so :func:`percept.events.frame_nbytes`
    reads the handle as ~0 bytes retained, the property the whole card exists to guarantee."""

    ref: str | None = None
    fingerprint: int | None = None
    source_nbytes: int = 0


#: An exemplar measured (via :func:`percept.events.frame_nbytes`) at or below this many bytes is a
#: CHEAP descriptor — a subject token, a small dict, a pixel-free :class:`~percept.contracts.Verdict`
#: — and is retained AS-IS (byte-for-byte unchanged, so subject/mutation reads on it keep working).
#: Anything larger is a decoded frame and is replaced by an :class:`EvidenceHandle` before storage.
#: This is a memory backstop, not a tunable of cognition; it is intentionally small.
_MAX_INLINE_EVIDENCE_BYTES = 512


def _cheap_fingerprint(frame: Any) -> "int | None":
    """A cheap, pixel-free signature of a frame (best-effort dHash), or ``None`` if not derivable.

    Reuses the perception cache's stdlib dHash so no new hashing path is introduced. Any failure —
    an un-griddable frame, an import problem — yields ``None``: a fingerprint is a bonus for judge
    comparison, never load-bearing for the byte bound, which holds whether or not one is computed."""
    try:
        from .cache.framehash import dhash
    except Exception:  # noqa: BLE001 — a missing cache never blocks handle creation
        return None
    try:
        return dhash(frame)
    except Exception:  # noqa: BLE001 — un-hashable frame: still get a byte-bounded handle
        return None


def evidence_handle(exemplar: Any, *, max_inline_bytes: int = _MAX_INLINE_EVIDENCE_BYTES) -> Any:
    """Return what an episode should RETAIN for ``exemplar`` — a HANDLE if it is a decoded frame.

    An :class:`EvidenceHandle` passes through unchanged (idempotent). A cheap descriptor (measured at
    or below ``max_inline_bytes`` by :func:`percept.events.frame_nbytes`) is returned AS-IS, so string
    tokens, dicts and pixel-free verdicts — everything the counting path used before this card — are
    byte-for-byte unchanged. Anything larger is a decoded frame: it is replaced by a small handle
    (fingerprint + measured ``source_nbytes``) and the pixels are NOT retained. This is the single
    choke point; :class:`EpisodeLog` routes every stored exemplar/representative through it so the
    'store decoded frames' defect cannot reappear by a caller forgetting to convert."""
    if isinstance(exemplar, EvidenceHandle):
        return exemplar
    n = frame_nbytes(exemplar)
    if n <= max_inline_bytes:
        return exemplar
    return EvidenceHandle(fingerprint=_cheap_fingerprint(exemplar), source_nbytes=n)


#: Default status an episode is opened in. Anything in a policy's ``terminal_statuses`` is CLOSED.
OPEN = "open"
CLOSED = "closed"

#: The prefix this log stamps on ``Episode.subject_key`` when it mints a PROVISIONAL identity for a
#: sighting seen but never identified — always ``UN_<episode_id>``, so it is unmistakable in a dump
#: and never collides with a real bound content key (see :func:`percept.contracts.provisional_subject`,
#: which builds the matching ``SubjectRef`` for anything outward-facing that wants the typed form).
#: Duplicated as a plain string constant here — this module deliberately does not import
#: ``percept.contracts`` (it stays duck-typed over ``exemplar.subject``); the two are pinned equal by
#: ``tests/test_y17_provisional_identity.py``.
UNKNOWN_KEY_PREFIX = "UN_"

#: "argument not supplied" sentinel for :meth:`EpisodeLog.annotate` — distinct from ``None``, which is
#: itself a meaningful value there (None = clear/absent). Not part of the public surface.
_UNSET: Any = object()


def _subject_key_of(exemplar: Any) -> "str | None":
    """The judge-emitted (or episode-minted) subject key an exemplar carries, or ``None`` (card Y5).

    Reads a ``SubjectRef`` off ``exemplar.subject`` (e.g. a :class:`~percept.contracts.Verdict`) and
    returns its ``key`` when the ref is ``resolution='bound'`` OR ``resolution='provisional'`` (the
    session-local ``UN_<episode_id>`` handle this log mints for a tracked-but-unidentified sighting —
    see :func:`percept.contracts.provisional_subject`) with a non-empty key. An ``'unknown'`` ref — a
    subject that was seen but not identified AND never earned a provisional handle — is deliberately
    NOT a match key: two genuinely unidentified subjects must never collapse into one episode. Absence
    of a subject (``exemplar.subject is None``, or no ``subject`` attribute at all) also yields
    ``None``, so an exemplar that predates subject identity is byte-for-byte unchanged."""
    subject = getattr(exemplar, "subject", None)
    if subject is None:
        return None
    if getattr(subject, "resolution", None) not in ("bound", "provisional"):
        return None
    key = getattr(subject, "key", None)
    return key if key else None


def subject_is_unknown(exemplar: Any) -> bool:
    """True iff ``exemplar`` carries a subject that was SEEN BUT NOT IDENTIFIED (card Y4/Y5).

    Reads a :class:`~percept.contracts.SubjectRef` off ``exemplar.subject`` and returns ``True`` only
    when its ``resolution`` is ``'unknown'`` — a *visible abstention*, distinct from BOTH a resolved
    ``'bound'`` subject AND the ABSENCE of a subject (``exemplar.subject is None``, which is ``False``
    here). This is the discriminator the FOLD keys on so a visible abstention never REARMS the gate and
    is never granted episode-owned idempotency (two unknowns are NaN-like — never the same episode).
    An exemplar with no ``subject`` attribute at all is byte-for-byte unchanged: it is not unknown."""
    subject = getattr(exemplar, "subject", None)
    if subject is None:
        return False
    return getattr(subject, "resolution", None) == "unknown"

ContinuationReason = Literal[
    # -- continued (is_new=False)
    "continued",              # same subject, open, inside the gap, mutation compatible
    "reopened",               # same subject, TERMINAL but the policy allows reopen -> same episode
    "unknown_provisional_continuation",  # Y4: an UNKNOWN sighting rode the sole open bound episode
    "unknown_temporal_continuation",  # an UNKNOWN sighting rode the most recent open PROVISIONAL
    # episode because it arrived within the source's unknown-continuity gap (no bound episode to
    # attach to at all — this is unknown-to-unknown continuity, distinct from the bound-attach above)
    # -- a NEW episode was opened (is_new=True)
    "new_subject",            # no existing episode shares this subject — nothing to continue
    "gap_exceeded",           # same subject, but idle longer than the concern's continuity_gap_s
    "episode_terminal",       # same subject, but the episode is finished and reopen is not allowed
    "mutation_incompatible",  # same subject and live, but the finding contradicts the episode (Y2)
]

#: The closed vocabulary of WHY an observation did or did not continue an episode. Stable surface:
#: a code is never renamed or repurposed; a new path adds a NEW code.
CONTINUATION_REASONS: tuple[ContinuationReason, ...] = (
    "continued",
    "reopened",
    "unknown_provisional_continuation",
    "unknown_temporal_continuation",
    "new_subject",
    "gap_exceeded",
    "episode_terminal",
    "mutation_incompatible",
)

#: The reasons that mean "a same-subject episode existed and we refused to continue it". These are
#: the interesting ones: each is a place the collapsed design would have manufactured false
#: continuity. They are logged individually so a client can see WHICH rule split the episode.
DENIAL_REASONS: tuple[ContinuationReason, ...] = (
    "gap_exceeded",
    "episode_terminal",
    "mutation_incompatible",
)


@dataclass(frozen=True)
class EpisodePolicy:
    """CONCERN-LEVEL continuation semantics. There is no correct global constant for these.

    * ``continuity_gap_s`` — the maximum idle time (observation ts minus ``episode.last_seen``) that
      still counts as the SAME episode. ``None`` = unbounded (the pre-Y1 behavior; kept as the
      default so an un-authored concern is byte-for-byte unchanged).
    * ``allow_reopen`` — may a TERMINAL episode be reopened by a later same-subject observation
      (a PO reopened) instead of starting a fresh occurrence? Default no.
    * ``terminal_statuses`` — the statuses that mean "this episode is finished". A same-subject
      observation against a terminal episode opens a new one (or reopens, per ``allow_reopen``).
    * ``unknown_attaches_to_active_bound`` — the card-Y4 OPT-IN for a NARROWLY single-focus source.
      Default ``False`` (a visible abstention stays unbound and mints no counted episode competitor —
      the Y5 default). When ``True`` AND the source has exactly ONE open bound episode AND no boundary
      event intervened, an ``'unknown'`` sighting provisionally CONTINUES that sole episode instead of
      minting a rival. Only defensible when the source truly holds one thing in focus: with two open
      bound episodes the sighting could belong to either, so it stays unbound. The attach is RECORDED
      (its own :data:`ContinuationReason` + a revision), never a silent merge.
    * ``unknown_continuity_gap_s`` — UNKNOWN-TO-UNKNOWN temporal-proximity continuity, independent of
      ``unknown_attaches_to_active_bound`` above (that one attaches to a BOUND episode; this one
      continues a PROVISIONAL one). Default ``None`` (off — byte-for-byte the Y5 default: every
      unidentified sighting mints its own fresh episode, keyed ``UN_<episode_id>``, and two unknowns
      never collapse). When set, an unidentified sighting that arrives within this many seconds of the
      most recently seen OPEN provisional episode CONTINUES it (same ``UN_`` key) instead of minting a
      new one — the honest middle ground for a source where "probably the same unidentified thing,
      seen again a moment later" is a defensible read but a full identity claim is not. Declared per
      :class:`~percept.sourceprofile.SourceProfile` (a capability, not a global), never hardcoded.
    """

    continuity_gap_s: float | None = None
    allow_reopen: bool = False
    terminal_statuses: frozenset[str] = frozenset({CLOSED})
    unknown_attaches_to_active_bound: bool = False
    unknown_continuity_gap_s: float | None = None

    def is_terminal(self, status: str) -> bool:
        return status in self.terminal_statuses


#: Shipped starting points, NOT law — three domains whose continuation semantics genuinely differ.
#: Named so a concern can author ``goal.state["episode"] = {"policy": "door_interval"}`` and get a
#: defensible default instead of the unbounded one. Any field may still be overridden alongside it.
EPISODE_POLICIES: dict[str, EpisodePolicy] = {
    # A document is present until it is gone; a short occlusion is the SAME presence episode, and a
    # return after minutes is a new workflow occurrence.
    "document_presence": EpisodePolicy(continuity_gap_s=120.0, allow_reopen=False),
    # A door open/close interval: seconds. A second opening is emphatically a second episode.
    "door_interval": EpisodePolicy(continuity_gap_s=5.0, allow_reopen=False),
    # A conversational claim slot lives as long as the call and may be revisited within it.
    "conversational_claim": EpisodePolicy(continuity_gap_s=1800.0, allow_reopen=True),
    # Explicit "never split on time" — the pre-Y1 behavior, chosen rather than inherited.
    "unbounded": EpisodePolicy(),
}

#: Only these keys are honoured from an authored ``goal.state["episode"]`` block.
_POLICY_KEYS = ("continuity_gap_s", "allow_reopen", "terminal_statuses")
_MAX_TERMINAL_STATUSES = 16
_MAX_STATUS_LEN = 40


def episode_policy_of(goal: Any, default: "EpisodePolicy | None" = None) -> EpisodePolicy:
    """Resolve the CONCERN-LEVEL :class:`EpisodePolicy` for a goal from ``goal.state['episode']``.

    Reads the same validated ``goal.state`` channel every other authored knob uses
    (``percept.behavior.coerce_authored_episode`` is the writer). A goal that authored nothing gets
    ``default`` (or the unbounded legacy policy), so existing concerns are byte-for-byte unchanged.
    """
    base = default if default is not None else EpisodePolicy()
    state = getattr(goal, "state", None)
    spec = state.get("episode") if isinstance(state, dict) else None
    if not isinstance(spec, dict):
        return base
    named = spec.get("policy")
    if isinstance(named, str) and named in EPISODE_POLICIES:
        base = EPISODE_POLICIES[named]
    fields: dict = {}
    gap = spec.get("continuity_gap_s")
    if isinstance(gap, (int, float)) and not isinstance(gap, bool) and gap > 0:
        fields["continuity_gap_s"] = float(gap)
    if isinstance(spec.get("allow_reopen"), bool):
        fields["allow_reopen"] = spec["allow_reopen"]
    terminal = spec.get("terminal_statuses")
    if isinstance(terminal, (list, tuple, set, frozenset)):
        statuses = {s for s in terminal if isinstance(s, str) and 0 < len(s) <= _MAX_STATUS_LEN}
        if statuses:
            fields["terminal_statuses"] = frozenset(sorted(statuses)[:_MAX_TERMINAL_STATUSES])
    if not fields:
        return base
    return EpisodePolicy(**{k: fields.get(k, getattr(base, k)) for k in _POLICY_KEYS})


# ================================================================================================
# REVISION TIMELINE (card Y15) — a 24/7 confirm heartbeat must not be an unbounded log.
# ================================================================================================
#
# An episode carries a small, BOUNDED timeline of the STATE CHANGES it went through — it opened, an
# update was accepted, it was retracted, it closed, it expired out of the window, a stretch went
# unobserved. That timeline is what a client debugging percept at 2am reads to answer "what happened
# to THIS episode, and when?" without re-deriving it from a firehose of ticks.
#
# The stress-test defect (found by the seven-domain eval, card Y15): a long-running watch confirms the
# SAME finding on every confident tick — a heartbeat — and the naive timeline appended a ``confirmed``
# row per tick. Over a 24/7 watch that is an UNBOUNDED log: N identical confirmations become N rows,
# forever. A re-statement of an unchanged fact is not a state change and must not grow the timeline.
#
# So ``confirmed`` COALESCES IN PLACE: it advances ``last_confirmed`` (and ``last_seen``) on the
# existing episode and appends NOTHING. A run of N unchanged confirmations produces exactly ONE row —
# the ``opened`` that began the episode. Only STATE-CHANGING kinds append
# (:data:`STATE_CHANGING_REVISION_KINDS`). The timeline is additionally CAPPED (``max_revisions``)
# with a TRACED drop on overflow, matching the EpisodeLog drop discipline (ADR-001 §9): a
# bound-driven drop is never silent.

RevisionKind = Literal[
    "opened",          # the episode began (or reopened) — the one row a pure heartbeat ever produces
    "confirmed",       # an unchanged re-statement — COALESCES in place, never appends (the Y15 fix)
    "updated",         # an accepted semantic change to the episode's value → a new revision
    "retracted",       # the finding was withdrawn
    "closed",          # the episode reached a terminal status
    "expired",         # the episode fell out of the counting window
    "unobserved_gap",  # a stretch nobody observed was recorded against the episode
]

#: The closed vocabulary of episode state changes. Stable: a code is never renamed or repurposed.
REVISION_KINDS: tuple[RevisionKind, ...] = (
    "opened", "confirmed", "updated", "retracted", "closed", "expired", "unobserved_gap",
)

#: The kinds that APPEND a timeline row. ``confirmed`` is deliberately absent: an unchanged
#: re-confirmation is a heartbeat, not a state change, and coalesces in place (card Y15). A run of N
#: confirmations therefore leaves the timeline at the single ``opened`` row.
STATE_CHANGING_REVISION_KINDS: tuple[RevisionKind, ...] = (
    "opened", "updated", "retracted", "closed", "expired", "unobserved_gap",
)


@dataclass(frozen=True)
class Revision:
    """One STATE CHANGE on an episode's timeline (card Y15).

    A ``confirmed`` heartbeat never becomes a ``Revision`` — it coalesces into the episode's
    ``last_confirmed`` instead — so an appended row always marks a genuine change, never a
    re-statement. ``value``/``incident_id`` carry what changed, when relevant (``None`` = absent)."""

    kind: RevisionKind
    t: float
    value: Any = None
    incident_id: str | None = None


@dataclass
class Episode:
    id: str
    goal_id: str
    started_at: float
    last_seen: float
    exemplar: Any = None
    # Y1: ``exemplar`` is the FIRST observation that opened the episode; ``representative`` is the
    # MOST RECENT one folded into it. Before Y1 the two were set once and never diverged, which made
    # ``representative`` a dead field. It is now updated on every continuation, so a consumer asking
    # "what does this episode look like NOW" gets the current evidence, not the opening frame.
    representative: Any = None
    # Y1 decision (2): an episode has a LIFECYCLE. Only an ``open`` episode can be continued (unless
    # the concern's policy allows reopening a terminal one).
    status: str = OPEN
    # Y1 decision (1): the subject this episode is about. ``None`` today for judge-matched episodes;
    # populated when the caller passes a ``subject_key`` (the Y5/Z3 judge-emitted key seam).
    subject_key: str | None = None
    # How many observations folded into this episode (1 = just the opening one). Pure observability.
    observations: int = 1
    # Y7 decision (1): STATE CONSISTENCY. status/t_end are kept consistent BY CONSTRUCTION through
    # ``EpisodeLog.transition`` — the single authoritative writer. No other code path may assign
    # them; a test asserts it. ``value``/``incident_id`` use None as the one and only ABSENCE
    # sentinel: '' is never an absence marker here, so "no value yet" and "the empty string" can
    # never be silently confused.
    #
    # ``t_end`` is the TERMINAL close time — distinct from ``last_seen`` (the last observation). It is
    # ``None`` while the episode is live, and is DERIVABLE: ``t_end is not None`` iff ``status`` is
    # terminal. ``last_seen`` says "we last SAW this at t"; ``t_end`` says "this episode ENDED at t".
    # A duration is NEVER ``last_seen - started_at`` (that forward-fills the unobserved tail — see
    # :func:`interval_coverage`); an ended episode's extent is ``[started_at, t_end]``, and its truly
    # observed coverage comes from spans, not from the endpoints.
    t_end: float | None = None
    # The current value/finding this episode carries. None == genuinely absent (NEVER '').
    value: Any = None
    # The incident this episode raised, if any. None == no incident (NEVER '').
    incident_id: str | None = None
    # Y15: the most recent CONFIRMED tick (a heartbeat). An unchanged re-confirmation advances this
    # in place WITHOUT appending a revision row, so a 24/7 confirm stream never grows the timeline.
    # ``None`` until the first confirmation; distinct from ``last_seen`` only in intent (a confirm is
    # a POSITIVE re-statement, where ``last_seen`` also moves on a same-subject continuation).
    last_confirmed: float | None = None
    # Y15: the BOUNDED timeline of this episode's STATE CHANGES (opened / updated / retracted /
    # closed / expired / unobserved_gap). A pure heartbeat produces exactly one row — the ``opened``.
    # Written only through :meth:`EpisodeLog.record_revision`, the single coalescing choke point.
    revisions: list["Revision"] = field(default_factory=list)

    # -- Z2 forward seam: a persisted representation, present but NOT wired for v1.4 -------------------
    # Persistence is SESSION-ONLY today (ADR-006 / card Y9): ``Percept.snapshot()`` does not carry the
    # episode map and ``Percept.restore()`` emits an ``episode_memory_reset`` health event instead of
    # rehydrating one. These two methods exist so a LATER phase can turn persistence ON without reshaping
    # the type. The design point that makes that safe: every field here is either an identifier or a
    # STREAM-TIME instant (``started_at`` / ``last_seen`` are frame timestamps on the same clock a
    # snapshot's deadlines are), so they rebase cleanly the way ``deadline_at`` already does — UNLIKE a
    # wall-vs-monotonic ``_dwell_since``-style base, which has no recoverable duration after a reboot and
    # is exactly why the naive snapshot was refused. ``exemplar`` / ``representative`` are opaque payloads
    # the caller owns; a persistence phase must supply their (de)serialization, so they are carried as-is
    # here and a non-JSON-able payload is that phase's problem to solve, not silently dropped.
    def to_dict(self) -> dict:
        """JSON-able view of the episode (Z2 forward seam — see the class note). Not called in v1.4."""
        return {
            "id": self.id,
            "goal_id": self.goal_id,
            "started_at": self.started_at,
            "last_seen": self.last_seen,
            "exemplar": self.exemplar,
            "representative": self.representative,
            "status": self.status,
            "subject_key": self.subject_key,
            "observations": self.observations,
            "last_confirmed": self.last_confirmed,
            "revisions": list(self.revisions),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Episode":
        """Rebuild an :class:`Episode` from :meth:`to_dict` (Z2 forward seam). Not called in v1.4.

        Stream-time fields (``started_at`` / ``last_seen``) are stored as-taken; a persistence phase
        rebases them into the fresh monotonic frame at restore, the same way deadlines are rebased."""
        return cls(
            id=d["id"],
            goal_id=d["goal_id"],
            started_at=d["started_at"],
            last_seen=d["last_seen"],
            exemplar=d.get("exemplar"),
            representative=d.get("representative"),
            status=d.get("status", OPEN),
            subject_key=d.get("subject_key"),
            observations=d.get("observations", 1),
            last_confirmed=d.get("last_confirmed"),
            revisions=list(d.get("revisions", [])),
        )


@dataclass(frozen=True)
class ContinuationDecision:
    """One typed episode decision: the episode, whether it is NEW, and WHY (card Y1).

    ``subject_matched`` answers decision (1) on its own — it is true whenever an existing episode
    shared this observation's subject, INCLUDING the cases where we then refused to continue it.
    That split is the whole point of the card: ``subject_matched and is_new`` is precisely the
    situation the collapsed design got wrong.
    """

    episode: Episode
    is_new: bool
    reason: ContinuationReason
    subject_matched: bool = False
    prior_episode_id: str | None = None  # the same-subject episode that was NOT continued
    gap_s: float | None = None           # observed idle gap against that episode

    @property
    def denied(self) -> bool:
        """True iff a same-subject episode existed and a continuation was refused."""
        return self.reason in DENIAL_REASONS


class EpisodeLog:
    """Bounded per-goal episode log. ``observe`` folds a candidate event into new/continuation.

    Boundedness (ADR-001 §9, card A9 subset): ``window_s`` bounds the log in TIME (episodes idle
    longer than the counting window fall out — "today's cups", not "all-time cups") and
    ``max_episodes`` bounds it in COUNT (a hard ceiling, drop-oldest, the backstop when no window is
    set or the window alone isn't enough). Every bound-driven drop is TRACED — never silent —
    through ``on_drop(episode, reason)`` (``reason`` is ``"window_expired"`` or ``"max_episodes"``);
    the default logs a structured line so a drop is observable even when the caller wires nothing.

    Card Y15: each episode also owns a BOUNDED revision timeline. ``record_revision`` folds a
    ``confirmed`` heartbeat in place (no row) so a 24/7 confirm stream never grows the log; only
    state-changing kinds append, and the per-episode timeline is capped at ``max_revisions`` with a
    TRACED drop (``on_revision_drop`` / ``episode.revision_dropped``) on overflow.

    Card Y1: ``policy`` carries the CONCERN-LEVEL continuation semantics (see :class:`EpisodePolicy`)
    and ``on_continuation(decision)`` observes every typed :class:`ContinuationDecision`. A denied
    continuation is additionally logged, so "why did this subject open a second episode?" is
    answerable from a log file alone.
    """

    def __init__(
        self, goal_id: str, *, window_s: float | None = None, max_episodes: int = 64,
        on_drop: "Callable[[Episode, str], None] | None" = None,
        policy: "EpisodePolicy | None" = None,
        on_continuation: "Callable[[ContinuationDecision], None] | None" = None,
        retain_inline_max_bytes: int = _MAX_INLINE_EVIDENCE_BYTES,
        max_revisions: int = 256,
        on_revision_drop: "Callable[[Episode, Revision], None] | None" = None,
    ) -> None:
        self.goal_id = goal_id
        self.window_s = window_s
        self.max_episodes = max_episodes
        self.on_drop = on_drop
        # Card Y15: a per-episode timeline is bounded in COUNT (max_revisions) with a TRACED
        # drop-oldest on overflow, the same ADR-001 §9 discipline max_episodes uses. `confirmed`
        # heartbeats coalesce and never reach this cap; only state changes count against it.
        self.max_revisions = max_revisions
        self.on_revision_drop = on_revision_drop
        self.policy = policy if policy is not None else EpisodePolicy()
        self.on_continuation = on_continuation
        # Card Y14: an exemplar larger than this (measured by frame_nbytes) is a decoded frame and is
        # replaced by an EvidenceHandle before storage, so retained bytes stay bounded by episode
        # COUNT, never by frame resolution. Cheap descriptors (tokens/dicts/verdicts) pass through.
        self.retain_inline_max_bytes = retain_inline_max_bytes
        self._episodes: list[Episode] = []
        self._counter = 0

    def _retain(self, exemplar: Any) -> Any:
        """What to STORE for ``exemplar`` — a handle if it is a decoded frame (card Y14)."""
        return evidence_handle(exemplar, max_inline_bytes=self.retain_inline_max_bytes)

    def retained_evidence_nbytes(self) -> int:
        """Measured bytes of decoded evidence currently pinned by this log (card Y14 soak metric).

        Sums :func:`percept.events.frame_nbytes` over every stored ``exemplar`` and ``representative``
        — the SAME function Z1's FrameWindow byte budget uses. With the Y14 handle wiring this stays
        ~0 no matter how many observations flow through (handles carry no pixels), which is exactly
        the property the soak test asserts: retained bytes are bounded by episode COUNT, not by N."""
        total = 0
        for e in self._episodes:
            total += frame_nbytes(e.exemplar) + frame_nbytes(e.representative)
        return total

    # -- decision (1): STATE CONSISTENCY (card Y7) -----------------------------------------------
    @staticmethod
    def _absent_if_empty(v: Any) -> Any:
        """Normalize the absence sentinel: ``''`` collapses to ``None`` so absence has ONE spelling.

        The bug this closes: two code paths, one writing ``''`` and one writing ``None`` for "no
        value / no incident", then a third comparing them and silently disagreeing. Here there is
        exactly one absent value — ``None`` — and an empty string can never masquerade as presence.
        """
        return None if isinstance(v, str) and v == "" else v

    def transition(self, episode: "Episode", status: str, *, t: float) -> "Episode":
        """The SINGLE authoritative writer of ``status`` and ``t_end`` (card Y7 §1).

        No other code path may assign ``episode.status`` or ``episode.t_end`` — a test asserts it.
        Routing every lifecycle change through here keeps the pair consistent BY CONSTRUCTION:

        * entering a terminal status stamps ``t_end = t`` (the close time);
        * leaving a terminal status (a reopen) clears ``t_end = None``.

        So ``t_end`` is never stored independently of ``status`` and the two cannot drift: ``t_end is
        not None`` iff ``status`` is terminal under this log's policy.
        """
        episode.status = status
        episode.t_end = t if self.policy.is_terminal(status) else None
        return episode

    def annotate(
        self, episode: "Episode", *, value: Any = _UNSET, incident_id: Any = _UNSET
    ) -> "Episode":
        """Attach the current ``value`` and/or ``incident_id`` to an episode.

        Enforces the absence invariant: ``''`` is normalized to ``None`` on the way in, so an empty
        string can never be mistaken for a present value or a real incident. Does NOT touch
        ``status``/``t_end`` — those change only through :meth:`transition`.
        """
        if value is not _UNSET:
            episode.value = self._absent_if_empty(value)
        if incident_id is not _UNSET:
            episode.incident_id = self._absent_if_empty(incident_id)
        return episode

    def record_revision(
        self, episode: "Episode", kind: "RevisionKind", *, t: float,
        value: Any = None, incident_id: "str | None" = None,
    ) -> "Revision | None":
        """Fold a state change into an episode's BOUNDED timeline — the single coalescing choke point
        (card Y15). Returns the appended :class:`Revision`, or ``None`` when a confirmation coalesced.

        A ``confirmed`` heartbeat COALESCES IN PLACE: it advances ``last_confirmed`` (and
        ``last_seen``, never backwards) and appends NOTHING. A run of N unchanged confirmations
        therefore produces exactly ONE row — the ``opened`` that began the episode — not N. This is
        the whole card: a 24/7 confirm stream must not turn the timeline into an unbounded log.

        Every OTHER (state-changing) kind appends a row. ``kind`` must be in :data:`REVISION_KINDS`;
        an unknown kind is a programming error, raised loudly rather than silently mis-recorded.
        """
        if kind not in REVISION_KINDS:
            raise ValueError(f"unknown revision kind {kind!r}; expected one of {REVISION_KINDS}")
        if kind == "confirmed":
            # Coalesce in place — a heartbeat is not a new row. Advance last_seen/last_confirmed
            # (never backwards), the ONLY state a repeated confirmation is allowed to move.
            if t > episode.last_seen:
                episode.last_seen = t
            episode.last_confirmed = t
            return None
        revision = Revision(kind=kind, t=t, value=value, incident_id=incident_id)
        episode.revisions.append(revision)
        if len(episode.revisions) > self.max_revisions:
            dropped = episode.revisions.pop(0)  # drop-oldest, the same discipline as max_episodes
            self._revision_dropped(episode, dropped)
        return revision

    def _revision_dropped(self, episode: "Episode", revision: "Revision") -> None:
        """ADR-001 §9: a bound-driven revision drop must never be silent.

        Mirrors :meth:`_dropped` for episodes — an ``on_revision_drop`` hook when wired, else a
        structured log line, so a client sees the timeline was capped even with nothing wired."""
        if self.on_revision_drop is not None:
            self.on_revision_drop(episode, revision)
        else:
            log(
                get_logger("percept.episodes"), "episode.revision_dropped",
                goal_id=self.goal_id, episode_id=episode.id,
                kind=revision.kind, t=revision.t, reason="max_revisions",
            )

    def _dropped(self, episode: "Episode", reason: str) -> None:
        """ADR-001 §9: a bound-driven drop must never be silent."""
        if self.on_drop is not None:
            self.on_drop(episode, reason)
        else:
            log(
                get_logger("percept.episodes"), "episode.dropped",
                goal_id=self.goal_id, episode_id=episode.id, reason=reason,
            )

    def _prune(self, now: float) -> None:
        """Drop episodes outside the counting window (keeps the set bounded — the endurance story)."""
        if self.window_s is None:
            return
        keep: list[Episode] = []
        for e in self._episodes:
            if now - e.last_seen <= self.window_s:
                keep.append(e)
            else:
                self._dropped(e, "window_expired")
        self._episodes = keep

    def summaries(self) -> list[dict]:
        """Compact episode descriptors handed to the judge (no pixels on the hot path)."""
        return [
            {
                "id": e.id, "started_at": e.started_at, "last_seen": e.last_seen,
                "status": e.status, "subject_key": e.subject_key,
            }
            for e in self._episodes
        ]

    # -- decision (1): SUBJECT -------------------------------------------------------------------
    def _subject_match(
        self, exemplar: Any, judge: "EpisodeJudge | None", subject_key: "str | None"
    ) -> "Episode | None":
        """Which existing episode shares this observation's SUBJECT? Evidence only — the caller
        still has to run the continuation conjunction on whatever comes back.

        Precedence (card Y5): an explicit ``subject_key`` wins; else a BOUND ``verdict.subject.key``
        carried by the exemplar wins (the judge-emitted identity seam Y1 left); else the injected
        judge answers, as it always did. A key match is exact-equality, most recent first, and makes
        NO judge call.
        """
        if not self._episodes:
            return None
        if subject_key is None:
            subject_key = _subject_key_of(exemplar)
        if subject_key is not None:
            for e in reversed(self._episodes):
                if e.subject_key == subject_key:
                    return e
            return None
        if judge is None:
            return None
        matched_id = judge(self.summaries(), exemplar)
        if matched_id is None:
            return None
        for e in self._episodes:
            if e.id == matched_id:
                return e
        return None

    # -- decision (2): CONTINUATION (with decision (3) as its last conjunct) ----------------------
    @staticmethod
    def continuation_reason(
        episode: Episode, t: float, policy: EpisodePolicy, *, mutation_compatible: bool = True
    ) -> ContinuationReason:
        """Evaluate the continuation conjunction against a SAME-SUBJECT episode.

        This is the card-Y1 shape, in order::

            continues = (same_subject                                   # the caller established this
                         and episode.status == "open"                   # or allow_reopen
                         and observation.ts - episode.last_seen < continuity_gap_s
                         and mutation_is_compatible)

        Returns ``"continued"``/``"reopened"`` when it holds, else the FIRST conjunct that failed.
        The gap is checked before the mutation so the cheap, always-available evidence decides first
        and the (future, Y2) mutation channel is only consulted about a still-live episode. The gap
        comparison is the INCLUSIVE boundary rule: a separation EXACTLY equal to ``continuity_gap_s``
        does NOT continue the episode — it opens a new one (``t - episode.last_seen >= continuity_gap_s``
        counts as exceeded, not just strictly greater).
        """
        if policy.is_terminal(episode.status) and not policy.allow_reopen:
            return "episode_terminal"
        if policy.continuity_gap_s is not None and (t - episode.last_seen) >= policy.continuity_gap_s:
            return "gap_exceeded"
        if not mutation_compatible:
            return "mutation_incompatible"
        return "reopened" if policy.is_terminal(episode.status) else "continued"

    def _open(
        self, t: float, exemplar: Any, subject_key: "str | None",
        reason: ContinuationReason, prior: "Episode | None", gap_s: "float | None",
    ) -> ContinuationDecision:
        self._counter += 1
        retained = self._retain(exemplar)  # Y14: store a HANDLE, never the decoded frame
        episode = Episode(
            id=f"{self.goal_id}:ep{self._counter}",
            goal_id=self.goal_id,
            started_at=t,
            last_seen=t,
            exemplar=retained,
            representative=retained,
            status=OPEN,
            subject_key=subject_key,
        )
        self.record_revision(episode, "opened", t=t)  # Y15: the one row a pure heartbeat produces
        self._episodes.append(episode)
        if len(self._episodes) > self.max_episodes:
            oldest = self._episodes.pop(0)  # oldest falls out of the bounded log
            self._dropped(oldest, "max_episodes")
        return ContinuationDecision(
            episode=episode, is_new=True, reason=reason,
            subject_matched=prior is not None,
            prior_episode_id=prior.id if prior is not None else None,
            gap_s=gap_s,
        )

    def _note(self, decision: ContinuationDecision) -> None:
        """Y1: a refused continuation must never be silent — it is the one event that distinguishes
        "two real occurrences" from "the SDK lost the thread". Logged unconditionally (so a bare
        ``EpisodeLog`` is still debuggable) and handed to ``on_continuation`` when wired."""
        if decision.denied:
            log(
                get_logger("percept.episodes"), "episode.continuation_denied",
                goal_id=self.goal_id, reason=decision.reason,
                prior_episode_id=decision.prior_episode_id,
                episode_id=decision.episode.id, gap_s=decision.gap_s,
            )
        if self.on_continuation is not None:
            self.on_continuation(decision)

    # -- decision (1): the UNKNOWN subject — a visible abstention (card Y4) ----------------------
    def _single_open_bound(self, pol: EpisodePolicy) -> "Episode | None":
        """The SOLE open, subject-BOUND episode, or ``None`` if there are zero or MORE than one.

        The gate for the provisional single-focus attach (card Y4): attaching an unidentified
        sighting is only defensible when the source holds exactly one thing in focus. With two open
        bound episodes an ``'unknown'`` could belong to either, so a guess is refused and the sighting
        stays unbound. A PROVISIONAL episode (``subject_key`` prefixed :data:`UNKNOWN_KEY_PREFIX`) is
        NOT a bound identity and is excluded here — it is the OTHER continuity mechanism's territory
        (:meth:`_open_unknown_episode`), never a stand-in for "the one thing in focus"."""
        found: "Episode | None" = None
        for e in self._episodes:
            if (
                e.subject_key is not None
                and not e.subject_key.startswith(UNKNOWN_KEY_PREFIX)
                and not pol.is_terminal(e.status)
            ):
                if found is not None:
                    return None  # more than one thing in focus -> not a single-focus source
                found = e
        return found

    def _open_unknown_episode(self, pol: EpisodePolicy) -> "Episode | None":
        """The most recently seen OPEN provisional episode (``subject_key`` prefixed
        :data:`UNKNOWN_KEY_PREFIX``), or ``None``. The candidate for unknown-to-unknown temporal
        continuity — unlike :meth:`_single_open_bound`, more than one open provisional episode does
        not refuse the attach: the MOST RECENT one is the honest "probably the same thing, seen a
        moment later" read; an older one falling out of ``unknown_continuity_gap_s`` is exactly what
        the gap check (in the caller) is for."""
        for e in reversed(self._episodes):
            if (
                e.subject_key is not None
                and e.subject_key.startswith(UNKNOWN_KEY_PREFIX)
                and not pol.is_terminal(e.status)
            ):
                return e
        return None

    def _observe_unknown(
        self, t: float, exemplar: Any, pol: EpisodePolicy
    ) -> ContinuationDecision:
        """Fold an UNKNOWN (seen-but-unidentified) observation — the card-Y4 operational semantics,
        widened with a session-local PROVISIONAL identity for whatever episode results.

        Two independent, opt-in continuity mechanisms are tried in order, each OFF by default (the Y5
        behaviour: mint a fresh episode, never collapse):

        1. ``unknown_attaches_to_active_bound`` — single-focus attach to the sole open BOUND episode
           (unchanged from card Y4).
        2. ``unknown_continuity_gap_s`` — unknown-to-unknown temporal-proximity continuity: if the
           sighting arrives within the gap of the most recently seen open PROVISIONAL episode, it
           continues that episode (same ``UN_`` key) instead of minting a rival.

        Either attach is RECORDED — its own :data:`ContinuationReason` and a ``confirmed`` revision —
        so it is observable and never a silent merge. When NEITHER applies, a fresh episode opens and
        is immediately stamped with a PROVISIONAL identity: ``subject_key = "UN_" + episode.id``
        (:data:`UNKNOWN_KEY_PREFIX`; the matching typed ref is
        :func:`percept.contracts.provisional_subject`). That key can never collide with another
        episode's (ids are unique) and is never fabricated ahead of the episode existing, so two
        UNRELATED unknowns still never collapse — the Y5 guarantee holds; only a sighting close enough
        in time to a STILL-OPEN provisional episode ever shares its key."""
        if pol.unknown_attaches_to_active_bound:
            prior = self._single_open_bound(pol)
            if prior is not None and self.continuation_reason(
                prior, t, pol, mutation_compatible=True
            ) == "continued":
                gap_s = t - prior.last_seen
                prior.last_seen = t
                prior.representative = self._retain(exemplar)  # Y14: a HANDLE, never pixels
                prior.observations += 1
                self.record_revision(prior, "confirmed", t=t)  # provisional attach, recorded
                decision = ContinuationDecision(
                    episode=prior, is_new=False, reason="unknown_provisional_continuation",
                    subject_matched=True, prior_episode_id=prior.id, gap_s=gap_s,
                )
                self._note(decision)
                return decision
        if pol.unknown_continuity_gap_s is not None:
            prior = self._open_unknown_episode(pol)
            if prior is not None and (t - prior.last_seen) < pol.unknown_continuity_gap_s:
                gap_s = t - prior.last_seen
                prior.last_seen = t
                prior.representative = self._retain(exemplar)  # Y14: a HANDLE, never pixels
                prior.observations += 1
                self.record_revision(prior, "confirmed", t=t)  # temporal attach, recorded
                decision = ContinuationDecision(
                    episode=prior, is_new=False, reason="unknown_temporal_continuation",
                    subject_matched=True, prior_episode_id=prior.id, gap_s=gap_s,
                )
                self._note(decision)
                return decision
        # Neither continuity mechanism applied: mint a NEW episode and give it a PROVISIONAL identity
        # keyed to itself, so it can never match anything opened before it (no silent merge; card Y4
        # question 4) yet is still a real, countable subject rather than a vanished sighting.
        decision = self._open(t, exemplar, None, "new_subject", None, None)
        decision.episode.subject_key = f"{UNKNOWN_KEY_PREFIX}{decision.episode.id}"
        self._note(decision)
        return decision

    def observe_decision(
        self,
        t: float,
        exemplar: Any,
        judge: "EpisodeJudge | None" = None,
        *,
        subject_key: "str | None" = None,
        mutation: "MutationCheck | None" = None,
        policy: "EpisodePolicy | None" = None,
    ) -> ContinuationDecision:
        """Fold a candidate event at ``t`` into the log and return the TYPED decision (card Y1).

        The three decisions run in order and are individually visible on the result: subject
        (:attr:`ContinuationDecision.subject_matched`), continuation
        (:attr:`ContinuationDecision.is_new` + :attr:`~ContinuationDecision.reason`), and mutation
        (the ``mutation`` predicate, whose refusal surfaces as ``"mutation_incompatible"``).
        """
        pol = policy if policy is not None else self.policy
        self._prune(t)
        # Card Y4: a subject SEEN BUT NOT IDENTIFIED ('unknown'). An explicit ``subject_key`` still
        # wins (the resolver precedence) — only a genuinely unresolved sighting takes the unknown
        # path. Y5 already fixed the TYPE and the default fold (mint a fresh episode; two unknowns
        # never collapse); Y4 adds only the opt-in provisional single-focus attach here. The gate
        # governance (a visible abstention must never REARM) lives in the FOLD, not in this log.
        if subject_key is None and subject_is_unknown(exemplar):
            return self._observe_unknown(t, exemplar, pol)
        # Card Y5: an explicit subject_key wins; else a bound verdict.subject.key carried by the
        # exemplar. Resolve it ONCE here so the matched/opened episode STORES the same key the next
        # sighting will match against (a locally-derived-but-unstored key would never continue).
        if subject_key is None:
            subject_key = _subject_key_of(exemplar)
        prior = self._subject_match(exemplar, judge, subject_key)
        if prior is None:
            decision = self._open(t, exemplar, subject_key, "new_subject", None, None)
            self._note(decision)
            return decision
        gap_s = t - prior.last_seen
        compatible = True if mutation is None else bool(mutation(prior, exemplar))
        reason = self.continuation_reason(prior, t, pol, mutation_compatible=compatible)
        if reason in ("continued", "reopened"):
            prior.last_seen = t              # continuation — extend, don't count again
            prior.representative = self._retain(exemplar)  # LIVE (was dead) — a HANDLE (Y14), not pixels
            prior.observations += 1
            if reason == "reopened":
                self.transition(prior, OPEN, t=t)  # single writer: clears t_end too
                self.record_revision(prior, "opened", t=t)  # Y15: a reopen IS a state change
            else:
                # Y15: an unchanged continuation is a heartbeat — it COALESCES in place (advances
                # last_confirmed) and appends NO revision row, so a 24/7 confirm stream is bounded.
                self.record_revision(prior, "confirmed", t=t)
            decision = ContinuationDecision(
                episode=prior, is_new=False, reason=reason, subject_matched=True,
                prior_episode_id=prior.id, gap_s=gap_s,
            )
            self._note(decision)
            return decision
        # Same subject, refused continuation -> a genuinely NEW occurrence of the SAME subject.
        decision = self._open(t, exemplar, subject_key or prior.subject_key, reason, prior, gap_s)
        self._note(decision)
        return decision

    def observe(
        self,
        t: float,
        exemplar: Any,
        judge: "EpisodeJudge | None" = None,
        *,
        subject_key: "str | None" = None,
        mutation: "MutationCheck | None" = None,
        policy: "EpisodePolicy | None" = None,
    ) -> tuple[Episode, bool]:
        """Fold a candidate event at time ``t`` into the log. Returns ``(episode, is_new)``.

        The historical entry point, unchanged in shape: callers wanting the typed WHY (and the
        subject/continuation split) call :meth:`observe_decision` instead.
        """
        decision = self.observe_decision(
            t, exemplar, judge, subject_key=subject_key, mutation=mutation, policy=policy
        )
        return decision.episode, decision.is_new

    def close(
        self, episode_id: str, status: str = CLOSED, *, t: "float | None" = None
    ) -> "Episode | None":
        """Mark an episode TERMINAL (decision (2)'s lifecycle input). A later same-subject
        observation then opens a new occurrence — or reopens this one, if the concern's policy says
        a reopen is the same episode. Returns the episode, or ``None`` if it is not in the log.

        Goes through :meth:`transition` (the single status/t_end writer), which stamps ``t_end`` with
        the close time — ``t`` when given, else the episode's ``last_seen`` (the last moment it was
        actually observed; the end is never invented later than the last observation)."""
        for e in self._episodes:
            if e.id == episode_id:
                close_t = e.last_seen if t is None else t
                self.transition(e, status, t=close_t)
                if self.policy.is_terminal(status):  # Y15: reaching a terminal status is a state change
                    self.record_revision(e, "closed", t=close_t, incident_id=e.incident_id)
                return e
        return None

    def get(self, episode_id: str) -> "Episode | None":
        for e in self._episodes:
            if e.id == episode_id:
                return e
        return None

    def count(self, now: float | None = None) -> int:
        """Distinct episodes (optionally pruning to the window at ``now`` first)."""
        if now is not None:
            self._prune(now)
        return len(self._episodes)


# ================================================================================================
# INTERVAL COVERAGE (card Y7 §2) — "time accounted for" comes from OBSERVATION, never interpolation.
# ================================================================================================
#
# The decisive review catch: ``started_at -> last_seen`` does NOT prove continuous coverage. An
# observation at t=1 and another at t=20 with NOTHING between them is TWO point observations and one
# UNOBSERVED gap — it is emphatically NOT a 19-second interval. That 19s answer is manufactured by
# forward-filling a gap nobody looked at, which this design rejects outright.
#
# Coverage therefore comes from OBSERVED SPANS or a VALIDITY HORIZON, never from joining the dots:
# an observation at ``t`` is valid through ``min(next_observation, t + observation_valid_for_s)``,
# where ``observation_valid_for_s`` is a CONCERN-LEVEL knob (a door interval, a document presence and
# a conversational claim do not agree on it). Two positive observations merge into one continuous
# span only when the later one falls inside the earlier one's validity horizon; otherwise the stretch
# between them is a GAP, reported WITH ITS REASON. A judge-degraded tick (backpressure, a failed
# call) observed NOTHING — it marks a ``judge_degraded`` gap and is never carried forward as continued
# truth and never recorded as a confident negative. Silence is not evidence.
#
# The default horizon is ``None`` — no forward extension at all, so an un-configured concern gets the
# conservative answer (point observations, everything between them a gap) rather than a fabricated
# duration. The trailing observation is NEVER extended past itself even when a horizon is set: with
# no successor to bound it, extending into the open future would itself be forward-filling.

CoverageKind = Literal["observed", "validity_horizon"]
GapReason = Literal["unobserved", "judge_degraded"]


@dataclass(frozen=True)
class Observation:
    """One tick's worth of interval evidence.

    ``observed`` is False when the judge produced NO verdict this tick (degraded/backpressure) —
    silence, which is never confirmation and never a negative. ``satisfied`` is only meaningful when
    ``observed`` is True; a positive is ``observed and satisfied``. ``degraded`` distinguishes "the
    judge failed here" (a ``judge_degraded`` gap) from "nothing was sampled here" (an ``unobserved``
    gap) — the two are reported differently because they mean different things to someone debugging.
    """

    t: float
    observed: bool
    satisfied: bool = False
    degraded: bool = False

    @property
    def positive(self) -> bool:
        return self.observed and self.satisfied


@dataclass(frozen=True)
class CoverageSpan:
    """A stretch of time actually accounted for by observation."""

    start: float
    end: float
    kind: CoverageKind = "observed"

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass(frozen=True)
class CoverageGap:
    """A stretch of time NOT accounted for, with the reason it is a gap and not truth."""

    start: float
    end: float
    reason: GapReason

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass(frozen=True)
class Coverage:
    """The answer to "how long, and how much of it did we actually observe?".

    ``observed_total_s`` is the sum of span durations ONLY — a gap contributes zero, because an
    unobserved second is not a second of coverage. This is the number a client should quote for
    "time accounted for"; the gaps are what they must NOT quote as either presence or absence.
    """

    spans: tuple[CoverageSpan, ...]
    gaps: tuple[CoverageGap, ...]
    observed_total_s: float


def _gap_reason(observations: "list[Observation]", start: float, end: float) -> GapReason:
    """A gap is ``judge_degraded`` if the judge tried and failed anywhere inside it; else nobody
    looked and it is ``unobserved``. A degraded tick can never upgrade a gap into truth."""
    for o in observations:
        if start < o.t < end and o.degraded:
            return "judge_degraded"
    return "unobserved"


def interval_coverage(
    observations: "list[Observation]",
    *,
    observation_valid_for_s: "float | None" = None,
) -> Coverage:
    """Build observed spans + declared gaps from ticks — WITHOUT forward-filling (card Y7 §2).

    Two consecutive positive observations merge into one span only when the gap between them is
    within ``observation_valid_for_s`` (``None`` = never merge; each positive stands alone). Any
    stretch that is not merged is a GAP carrying its reason (``judge_degraded`` if a degraded tick
    falls inside it, else ``unobserved``). Coverage is defined over ``[first positive, last
    positive]`` only — nothing is claimed before the first or after the last thing observed.

    Returns a :class:`Coverage`; ``observed_total_s`` counts span time only.
    """
    ordered = sorted(observations, key=lambda o: o.t)
    positives = [o for o in ordered if o.positive]
    if not positives:
        return Coverage(spans=(), gaps=(), observed_total_s=0.0)

    spans: list[CoverageSpan] = []
    gaps: list[CoverageGap] = []
    span_start = positives[0].t
    prev_t = positives[0].t
    for cur in positives[1:]:
        dt = cur.t - prev_t
        mergeable = observation_valid_for_s is not None and dt <= observation_valid_for_s
        if mergeable:
            prev_t = cur.t
            continue
        # The current span ends at the last merged positive; the stretch to ``cur`` is a gap.
        spans.append(CoverageSpan(start=span_start, end=prev_t, kind="observed"))
        gaps.append(CoverageGap(start=prev_t, end=cur.t, reason=_gap_reason(ordered, prev_t, cur.t)))
        span_start = cur.t
        prev_t = cur.t
    spans.append(CoverageSpan(start=span_start, end=prev_t, kind="observed"))

    total = sum(s.duration for s in spans)
    return Coverage(spans=tuple(spans), gaps=tuple(gaps), observed_total_s=total)


__all__ = [
    "Episode",
    "Observation",
    "CoverageSpan",
    "CoverageGap",
    "Coverage",
    "CoverageKind",
    "GapReason",
    "interval_coverage",
    "EpisodeJudge",
    "EpisodeLog",
    "EpisodePolicy",
    "EPISODE_POLICIES",
    "episode_policy_of",
    "MutationCheck",
    "EvidenceHandle",
    "evidence_handle",
    "Revision",
    "RevisionKind",
    "REVISION_KINDS",
    "STATE_CHANGING_REVISION_KINDS",
    "ContinuationDecision",
    "ContinuationReason",
    "CONTINUATION_REASONS",
    "DENIAL_REASONS",
    "subject_is_unknown",
    "OPEN",
    "CLOSED",
]
