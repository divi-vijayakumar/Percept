"""The decision reducer: whether the agent hears about an occurrence.

One pure function per stream. Given the stream's state, one input and the current wall-clock
time, :func:`step` returns the next state, the decisions to commit to the outbox, and the
effects the runtime must carry out (timers, verification and recheck requests, health,
counter-examples, traces). It owns the two lifecycles (``state`` and ``event`` conditions),
confirmation, verification bookkeeping, holding, clearing, expiry, rearm, cooldown, snooze,
staleness, identities, capacity and the coverage fate of every observation. Budget, dispatch,
retries, judging, delivery and storage belong to the shell.

Names in brackets, such as [Time], are sections of ``docs/decision-semantics.md``, the public
statement of these rules. The rules in brief:

* Every stored time is wall clock [Time]. Deadlines are data and timers are hints: each input
  that touches a key first applies every deadline that has passed, so a late or lost timer
  never changes an outcome.
* Failed, weak (below ``min_confidence``) and ``unknown`` answers carry no information and
  never change state [Answers]; only a verifier's ``unknown`` resolves a candidate as unverified.
* Answers are checked for staleness by epoch, condition version, occurrence generation,
  purpose and the evidence floor [Stale results].
* The confirm window is measured on evidence time [Confirm window]; verification, rearm,
  cooldown, snooze and recheck on arrival time.

Purity: no I/O, no clock, no randomness. The state passed in is never mutated; keys are frozen
values shared between the old and the new state.

Cost: one step is O(K) in the number of keys on the stream (K <= ``max_keys``), because it
builds a shallow working copy of the key map; everything else a step does touches only the keys
it names, except an epoch start, which visits every key once (O(K log K) for the sorted order).
At the default ``max_keys`` this is negligible.

:func:`step` dispatches on the exact input type: the input classes below are accepted, and
subclasses of them are not.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Callable, Literal, Mapping, Optional, Union

from percept.contracts import Condition, Decision, Fate, Health, Observation, Usage

SCENE = "scene"
DEFAULT_MAX_KEYS = 64
MAX_REQUEST_IDS = 16            # request ids kept for one pending decision: the most recent ones

TimerKind = Literal["window", "verify", "recheck", "max_open", "rearm"]
TIMER_KINDS: tuple[str, ...] = ("window", "verify", "recheck", "max_open", "rearm")

# Lifecycle phases [States] [Events]. A state condition is idle, confirming, verifying or holding; an
# event condition is idle, candidate, verifying or reported. Any phase but idle is "open".
Phase = Literal["idle", "confirming", "candidate", "verifying", "holding", "reported"]
IDLE: Phase = "idle"
CONFIRMING: Phase = "confirming"
CANDIDATE: Phase = "candidate"
VERIFYING: Phase = "verifying"
HOLDING: Phase = "holding"
REPORTED: Phase = "reported"
STATE_PHASES: tuple[str, ...] = (IDLE, CONFIRMING, VERIFYING, HOLDING)
EVENT_PHASES: tuple[str, ...] = (IDLE, CANDIDATE, VERIFYING, REPORTED)

UnverifiedCause = Literal["timeout", "unverified", "restart"]

# Fixed human sentences for decisions the judge did not word [Decision fields]. Agents branch on `cause`.
_UNVERIFIED_REASONS: Mapping[str, str] = MappingProxyType({
    "timeout": "a possible occurrence could not be verified before its deadline",
    "unverified": "the verifier could not tell from the original evidence",
    "restart": "verification was interrupted by a restart",
})
_EXPIRED_REASON = "still true after max_open_s"
_RESOLVED_REASON = "resolved by the application"
_SCENE_RETURN_REASON = "the scene returned to its baseline"
_CONDITION_CHANGED_REASON = "the condition was changed"


# --------------------------------------------------------------------------- inputs


@dataclass(frozen=True)
class Observed:
    """A judge answer routed to this stream, with its deterministic share of the request's usage."""

    observation: Observation
    usage: Usage = Usage()


@dataclass(frozen=True)
class TimerFired:
    """A timer this reducer scheduled has fired. It may be early, late, duplicated or stale."""

    condition_id: str
    subject: str
    kind: TimerKind


@dataclass(frozen=True)
class Resolve:
    """The application says this occurrence is over."""

    occurrence_id: str


@dataclass(frozen=True)
class FalseAlarm:
    """A person labelled this occurrence a false alarm."""

    occurrence_id: str
    note: str = ""


@dataclass(frozen=True)
class Snooze:
    """Suppress met and uncertain until wall time ``until``.

    Replaces any snooze for the same ``(condition_id, subject)``; ``until <= now`` cancels it.
    ``subject=None`` covers every subject of the condition.
    """

    condition_id: str
    until: float
    subject: Optional[str] = None


@dataclass(frozen=True)
class SceneReturn:
    """The trigger saw the scene return to its baseline. ``subject=None``: every subject."""

    condition_id: str
    subject: Optional[str] = None


@dataclass(frozen=True)
class ConditionChanged:
    """A condition was added, updated (version bump) or removed in ``conditions``."""

    condition_id: str


@dataclass(frozen=True)
class EpochStarted:
    """The stream's epoch increased: a reconnect (``restart=False``) or a restore (``True``)."""

    stream_epoch: int
    restart: bool


Input = Union[Observed, TimerFired, Resolve, FalseAlarm, Snooze, SceneReturn, ConditionChanged, EpochStarted]


# --------------------------------------------------------------------------- effects


@dataclass(frozen=True)
class ScheduleTimer:
    """Fire ``TimerFired(condition_id, subject, kind)`` at wall time ``at``.

    Replaces any pending timer with the same ``(condition_id, subject, kind)``.
    """

    condition_id: str
    subject: str
    kind: TimerKind
    at: float


@dataclass(frozen=True)
class CancelTimer:
    """Drop the pending timer with this ``(condition_id, subject, kind)``, if any."""

    condition_id: str
    subject: str
    kind: TimerKind


@dataclass(frozen=True)
class RequestVerify:
    """Judge this condition again with ``purpose="verify"``.

    ``mode="original"`` re-judges exactly ``interval`` from ``stream_epoch``'s pre-roll;
    ``mode="fresh"`` judges a new frame (``interval`` is None). The answering observation
    carries this request's ``stream_epoch``, ``occurrence_gen`` and ``condition_version``.
    The runtime may retry until ``deadline``; the reducer resolves the candidate itself when
    the deadline passes.
    """

    condition_id: str
    subject: str
    occurrence_id: str
    occurrence_gen: int
    condition_version: int
    mode: Literal["original", "fresh"]
    interval: Optional[tuple[float, float]]
    stream_epoch: int
    deadline: float
    absorbed: bool = False      # verifies a successor promoted from an absorbed detection [Absorbed entries]


@dataclass(frozen=True)
class CancelVerify:
    """Drop a queued or in-flight verification.

    Emitted when an occurrence stops verifying for any reason other than a verify answer
    (resolve, false alarm, scene return, condition change, deadline, restart).
    """

    condition_id: str
    subject: str
    occurrence_id: str
    occurrence_gen: int


@dataclass(frozen=True)
class RequestRecheck:
    """Judge a held state again on a fresh frame (``purpose="detect"``)."""

    condition_id: str
    subject: str
    occurrence_id: str
    occurrence_gen: int
    condition_version: int


@dataclass(frozen=True)
class EvidenceGap:
    """An absorbed detection could not be promoted to a successor candidate: its evidence crosses a stream
    epoch (a reconnect happened while the first candidate verified). The runtime reports ``interval`` as an
    ``evidence_unavailable`` gap; nothing is judged and no decision is emitted [Absorbed entries]."""

    condition_id: str
    subject: str
    interval: tuple[float, float]               # the absorbed detection's evidence interval
    cause: Literal["epoch"]


@dataclass(frozen=True)
class EmitHealth:
    """Put ``health`` on the health channel. The reducer uses kinds ``unverified`` and ``capacity``."""

    health: Health


@dataclass(frozen=True)
class StoreCounterExample:
    """Keep a 'this is not it' example for the condition; the runtime attaches the thumbnail."""

    condition_id: str
    occurrence_id: str
    note: str


@dataclass(frozen=True)
class Trace:
    """A trace record. Every ``Observed`` input yields exactly one Trace with a ``fate`` [Coverage fates]."""

    t: float
    event: str
    condition_id: str
    subject: Optional[str] = None
    fate: Optional[Fate] = None
    occurrence_id: Optional[str] = None
    detail: str = ""


Effect = Union[ScheduleTimer, CancelTimer, RequestVerify, CancelVerify, RequestRecheck, EvidenceGap,
               EmitHealth, StoreCounterExample, Trace]


# --------------------------------------------------------------------------- state


KeyId = tuple[str, str]                      # (condition_id, subject)
SnoozeId = tuple[str, Optional[str]]         # (condition_id, subject or None for every subject)


@dataclass(frozen=True)
class Key:
    """Decide state for one ``(condition_id, subject)`` on one stream. Every time is wall clock.

    The fields from ``occurrence_id`` to ``met_confidence`` describe the open occurrence and are
    reset when the key returns to idle; ``gen``, ``last_met``, ``last_uncertain`` and
    ``closed_evidence_end`` survive idle. ``cost``, ``request_ids`` and ``last_touched`` are
    bookkeeping: no lifecycle transition reads them (``last_touched`` only picks an eviction victim).
    """

    condition_id: str
    subject: str
    phase: Phase = IDLE
    gen: int = 0
    occurrence_id: Optional[str] = None
    decisions: int = 0                          # decisions emitted for this occurrence
    yes_count: int = 0                          # confident yeses counted in the window
    window_start: Optional[float] = None        # evidence end of the earliest counted yes
    window_last: Optional[float] = None         # evidence end of the latest counted yes
    window_close_at: Optional[float] = None     # arrival time an unfinished window gives up
    interval: Optional[tuple[float, float]] = None   # hull of the counted yeses' evidence
    interval_epoch: Optional[int] = None        # epoch of the window's first yes
    yes_confidence: float = 0.0                 # confidence of the latest counted yes
    verify_deadline: Optional[float] = None
    absorbed: Optional[Observation] = None      # verifying, event: the first detect yes about newer evidence,
    absorbed_usage: Usage = Usage()             # kept whole (request id, interval, epoch) with its usage
    no_count: int = 0                           # holding: confident no's since the last yes
    held_since: Optional[float] = None
    expires_at: Optional[float] = None          # holding with max_open_s, until it passes
    expired: bool = False
    next_recheck: Optional[float] = None
    rearm_until: Optional[float] = None
    met_emitted: bool = False
    met_pending: bool = False                   # a state's met was suppressed: deliver it later
    met_confidence: float = 0.0
    last_met: Optional[float] = None
    last_uncertain: Optional[float] = None
    closed_evidence_end: Optional[float] = None  # the evidence floor for detect answers [Evidence floor]
    cost: Usage = Usage()
    request_ids: tuple[str, ...] = ()
    last_touched: float = 0.0


_NO_ABSORBED = dict(absorbed=None, absorbed_usage=Usage())
BOOKKEEPING_FIELDS: tuple[str, ...] = ("cost", "request_ids", "last_touched")


def _frozen(mapping: Optional[dict] = None) -> Mapping:
    return MappingProxyType({} if mapping is None else mapping)


@dataclass(frozen=True)
class StreamDecideState:
    """Everything the reducer remembers about one stream. Immutable; see :func:`to_dict`."""

    stream_id: str
    stream_epoch: int
    incarnation: str                            # minted once, at creation; part of every id
    max_keys: int
    next_occurrence: int = 1
    next_gen: int = 1
    versions: Mapping[str, int] = field(default_factory=_frozen)
    keys: Mapping[KeyId, Key] = field(default_factory=_frozen)
    snoozes: Mapping[SnoozeId, float] = field(default_factory=_frozen)


@dataclass(frozen=True)
class StepResult:
    state: StreamDecideState
    decisions: tuple[Decision, ...]
    effects: tuple[Effect, ...]


def new_state(stream_id: str, *, created_at: float, stream_epoch: int = 0,
              max_keys: int = DEFAULT_MAX_KEYS) -> StreamDecideState:
    """A fresh stream state. ``created_at`` (wall clock) mints the incarnation [Identity], so ids
    stay unique even when a snapshot is lost and the stream starts over."""
    if not stream_id:
        raise ValueError("stream_id must not be empty")
    if max_keys < 1:
        raise ValueError("max_keys is >= 1")
    return StreamDecideState(stream_id, stream_epoch, format(int(created_at * 1000), "x"), max_keys)


def gen_for(state: StreamDecideState, condition_id: str, subject: str = SCENE) -> int:
    """The occurrence generation to tag a new judge request with [Identity]. 0 for an absent key."""
    key = state.keys.get((condition_id, subject))
    return 0 if key is None else key.gen


def phase_of(state: StreamDecideState, condition_id: str, subject: str = SCENE) -> str:
    """The lifecycle phase of a key; ``"idle"`` for an absent one."""
    key = state.keys.get((condition_id, subject))
    return IDLE if key is None else key.phase


def decision_view(state: StreamDecideState) -> dict:
    """The snapshot without bookkeeping fields: what "unchanged" means: bookkeeping never changes an outcome."""
    data = to_dict(state)
    for key in data["keys"]:
        for name in BOOKKEEPING_FIELDS:
            del key[name]
    return data


# --------------------------------------------------------------------------- snapshots


SNAPSHOT_FORMAT = 1
_ABSORBED_FIELDS = ("absorbed", "absorbed_usage")
_KEY_FIELDS: tuple[str, ...] = tuple(Key.__dataclass_fields__)
_PHASES: frozenset[str] = frozenset(STATE_PHASES + EVENT_PHASES)


def to_dict(state: StreamDecideState) -> dict:
    """A plain-JSON snapshot, tagged with ``SNAPSHOT_FORMAT``. Keys and snoozes are sorted, so
    equal states give equal dicts."""

    def key_dict(key: Key) -> dict:
        data = {name: getattr(key, name) for name in _KEY_FIELDS}
        data["interval"] = None if key.interval is None else list(key.interval)
        data["request_ids"] = list(key.request_ids)
        data["cost"] = {name: getattr(key.cost, name) for name in Usage.__dataclass_fields__}
        if key.absorbed is None:            # written only while it matters, so most snapshots stay readable by 1.0.0
            for name in _ABSORBED_FIELDS:
                del data[name]
        else:
            data["absorbed"] = {name: getattr(key.absorbed, name) for name in Observation.__dataclass_fields__}
            data["absorbed"]["evidence_t"] = list(key.absorbed.evidence_t)
            data["absorbed_usage"] = {name: getattr(key.absorbed_usage, name) for name in Usage.__dataclass_fields__}
        return data

    return {
        "format": SNAPSHOT_FORMAT,
        "stream_id": state.stream_id,
        "stream_epoch": state.stream_epoch,
        "incarnation": state.incarnation,
        "max_keys": state.max_keys,
        "next_occurrence": state.next_occurrence,
        "next_gen": state.next_gen,
        "versions": dict(sorted(state.versions.items())),
        "keys": [key_dict(state.keys[k]) for k in sorted(state.keys)],
        "snoozes": [[c, s, u] for (c, s), u in sorted(state.snoozes.items(),
                                                       key=lambda item: (item[0][0], item[0][1] or ""))],
    }


def from_dict(data: Mapping) -> StreamDecideState:
    """The inverse of :func:`to_dict`.

    Raises ``ValueError`` for anything that is not a snapshot this version wrote: a missing or
    unknown ``format``, an unknown phase, a missing or unknown field, or a wrong shape.
    """
    try:
        return _from_dict(data)
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        raise ValueError(f"malformed decide snapshot: {exc!r}") from exc


def _from_dict(data: Mapping) -> StreamDecideState:
    if data.get("format") != SNAPSHOT_FORMAT:
        raise ValueError(f"unsupported snapshot format {data.get('format')!r}")
    keys: dict[KeyId, Key] = {}
    for raw in data["keys"]:
        raw = dict(raw)
        if raw.get("phase") not in _PHASES:
            raise ValueError(f"unknown phase {raw.get('phase')!r}")
        raw["interval"] = None if raw["interval"] is None else tuple(raw["interval"])
        if raw.get("absorbed") is not None:
            raw["absorbed"] = Observation(**{**raw["absorbed"], "evidence_t": tuple(raw["absorbed"]["evidence_t"])})
            raw["absorbed_usage"] = Usage(**raw["absorbed_usage"])
        raw["request_ids"] = tuple(raw["request_ids"])
        raw["cost"] = Usage(**raw["cost"])
        key = Key(**raw)
        keys[(key.condition_id, key.subject)] = key
    return StreamDecideState(
        stream_id=data["stream_id"],
        stream_epoch=data["stream_epoch"],
        incarnation=data["incarnation"],
        max_keys=data["max_keys"],
        next_occurrence=data["next_occurrence"],
        next_gen=data["next_gen"],
        versions=_frozen(dict(data["versions"])),
        keys=_frozen(keys),
        snoozes=_frozen({(c, s): u for c, s, u in data["snoozes"]}),
    )


# --------------------------------------------------------------------------- one step


class _Step:
    """The working copy for one :func:`step`: new dicts over the old frozen keys, the outputs,
    and the inputs every rule needs (``now``, ``conditions``). Discarded after the step."""

    def __init__(self, state: StreamDecideState, now: float, conditions: Mapping[str, Condition]) -> None:
        self.now = now
        self.conditions = conditions
        self.stream_id = state.stream_id
        self.incarnation = state.incarnation
        self.max_keys = state.max_keys
        self.epoch = state.stream_epoch
        self.next_occurrence = state.next_occurrence
        self.next_gen = state.next_gen
        self.versions: dict[str, int] = dict(state.versions)
        self.keys: dict[KeyId, Key] = dict(state.keys)
        self.snoozes: dict[SnoozeId, float] = dict(state.snoozes)
        self.decisions: list[Decision] = []
        self.effects: list[Effect] = []

    def emit(self, effect: Effect) -> None:
        self.effects.append(effect)

    def trace(self, event: str, condition_id: str, subject: Optional[str] = None, *,
              fate: Optional[Fate] = None, occurrence_id: Optional[str] = None, detail: str = "") -> None:
        self.effects.append(Trace(self.now, event, condition_id, subject, fate, occurrence_id, detail))

    def put(self, key: Key) -> Key:
        self.keys[(key.condition_id, key.subject)] = key
        return key

    def mint_gen(self) -> int:
        """A new occurrence generation from the per-stream counter [Identity]."""
        gen = self.next_gen
        self.next_gen += 1
        return gen

    def mint_occurrence_id(self, condition_id: str) -> str:
        """``{stream}/{condition}/{incarnation}-{n}`` [Identity]."""
        occurrence_id = f"{self.stream_id}/{condition_id}/{self.incarnation}-{self.next_occurrence}"
        self.next_occurrence += 1
        return occurrence_id

    def result(self) -> StepResult:
        state = StreamDecideState(
            self.stream_id, self.epoch, self.incarnation, self.max_keys, self.next_occurrence,
            self.next_gen, _frozen(self.versions), _frozen(self.keys), _frozen(self.snoozes))
        return StepResult(state, tuple(self.decisions), tuple(self.effects))


def step(state: StreamDecideState, inp: Input, now: float, conditions: Mapping[str, Condition]) -> StepResult:
    """Apply one input at wall time ``now`` under the stream's current ``conditions``.

    Pure and deterministic; ``state`` is never mutated. Raises only for a programming error:
    an unknown input type, or an observation addressed to another stream.
    """
    handler = _HANDLERS.get(type(inp))
    if handler is None:
        raise TypeError(f"unknown input {inp!r}")
    st = _Step(state, now, conditions)
    handler(st, inp)
    return st.result()


# --------------------------------------------------------------------------- key primitives


def _timers(key: Key) -> dict[str, float]:
    """The key's live deadlines by timer kind, in ``TIMER_KINDS`` order. One timer per deadline."""
    deadlines = (("window", key.window_close_at), ("verify", key.verify_deadline),
                 ("recheck", key.next_recheck), ("max_open", key.expires_at), ("rearm", key.rearm_until))
    return {kind: at for kind, at in deadlines if at is not None}


def _raise_floor(key: Key, evidence_end: float) -> Key:
    """Raise (never lower) the evidence floor [Evidence floor]."""
    if key.closed_evidence_end is not None and evidence_end <= key.closed_evidence_end:
        return key
    return replace(key, closed_evidence_end=evidence_end)


def _charge(key: Key, inp: Observed) -> Key:
    """Add an observation's usage to the occurrence's pending cost [Decision fields]."""
    return _charge_usage(key, inp.observation.request_id, inp.usage)


def _charge_usage(key: Key, request_id: str, usage: Usage) -> Key:
    ids = key.request_ids
    if request_id not in ids:
        ids = (ids + (request_id,))[-MAX_REQUEST_IDS:]
    return replace(key, cost=key.cost + usage, request_ids=ids)


def _absorbable(key: Key, cond: Condition, o: Observation, answer: str) -> bool:
    """A confident detect yes about evidence newer than a verifying event candidate's own, with none kept yet
    [Absorbed entries]. Its usage is not charged to the candidate: it goes with the entry it found."""
    return (key.phase == VERIFYING and cond.kind == "event" and o.purpose == "detect" and answer == "yes"
            and key.absorbed is None and key.interval is not None and o.evidence_t[1] > key.interval[1])


def _settled(key: Key) -> Key:
    """Charge a kept absorbed detection to the key after all (it is not promoted) and forget it."""
    if key.absorbed is None:
        return key
    return replace(_charge_usage(key, key.absorbed.request_id, key.absorbed_usage), **_NO_ABSORBED)


def _close(st: _Step, key: Key, *, bump: bool = False, verify_answered: bool = False) -> Key:
    """Return the key to idle: cancel its verification (unless a verify answer closed it) and
    its timers, and reset everything that does not survive idle. ``bump`` mints a new gen."""
    if key.phase == VERIFYING and not verify_answered:
        st.emit(CancelVerify(key.condition_id, key.subject, key.occurrence_id, key.gen))
    for kind in _timers(key):
        st.emit(CancelTimer(key.condition_id, key.subject, kind))
    return Key(key.condition_id, key.subject, gen=st.mint_gen() if bump else key.gen,
               last_met=key.last_met, last_uncertain=key.last_uncertain,
               closed_evidence_end=key.closed_evidence_end, last_touched=st.now)


def _advance(st: _Step, key: Key, cond: Condition) -> Key:
    """Apply every stored deadline that is due [Time]: window give-up, verify deadline, rearm
    end, expiry. The recheck is an action, not a deadline, and is not applied here."""
    now = st.now
    if key.phase in (CONFIRMING, CANDIDATE) and now >= key.window_close_at:
        st.trace("flicker", cond.id, key.subject, occurrence_id=key.occurrence_id, detail="confirm window ended")
        key = _close(st, key)
    if key.phase == VERIFYING and now >= key.verify_deadline:
        key = _unverified(st, key, cond, "timeout", verify_answered=False)
    if key.phase == REPORTED and now >= key.rearm_until:
        st.trace("rearmed", cond.id, key.subject, occurrence_id=key.occurrence_id)
        key = _close(st, _raise_floor(key, key.rearm_until))
    if key.phase == HOLDING and key.expires_at is not None and now >= key.expires_at:
        # Expiry happens once; a met still waiting for delivery is dropped with it [Emission rules].
        key = replace(key, expires_at=None, expired=True, met_pending=False)
        key, _ = _emit(st, key, cond, "expired", "max_open", _EXPIRED_REASON, key.met_confidence, None)
    return key


# --------------------------------------------------------------------------- emission [Emission rules]


def _snoozed(st: _Step, condition_id: str, subject: str) -> bool:
    return any(st.now < st.snoozes.get((condition_id, who), float("-inf")) for who in (subject, None))


def _in_cooldown(cond: Condition, since: Optional[float], now: float) -> bool:
    """Clamped [Time]: a backwards clock step counts as no time elapsed; zero never suppresses."""
    return cond.cooldown_s > 0 and since is not None and max(0.0, now - since) < cond.cooldown_s


def _last_alert(key: Key) -> Optional[float]:
    times = [t for t in (key.last_met, key.last_uncertain) if t is not None]
    return max(times) if times else None


def _suppression(st: _Step, key: Key, cond: Condition, type_: str) -> Optional[str]:
    """Why a decision of ``type_`` must not be emitted now, or None [Emission rules]."""
    if type_ in ("met", "uncertain") and _snoozed(st, cond.id, key.subject):
        return "snoozed"
    if type_ == "met" and _in_cooldown(cond, key.last_met, st.now):
        return "cooldown"
    if type_ == "uncertain" and _in_cooldown(cond, _last_alert(key), st.now):
        return "cooldown"
    if type_ in ("cleared", "expired") and not key.met_emitted:
        return "met_not_emitted"
    if type_ == "cleared" and (cond.kind == "event" or not cond.emit_cleared):
        return "emit_cleared_off"
    return None


def _emit(st: _Step, key: Key, cond: Condition, type_: str, cause: Optional[str], reason: str,
          confidence: float, t_observed: Optional[float]) -> tuple[Key, bool]:
    """Emit a decision about the key's occurrence unless a rule suppresses it (traced).

    The decision carries everything charged since the previous one [Decision fields], and takes the next
    ``decision_id`` of the occurrence. ``t_observed=None`` means ``now``.
    """
    why_not = _suppression(st, key, cond, type_)
    if why_not is not None:
        st.trace("suppressed", cond.id, key.subject, occurrence_id=key.occurrence_id,
                 detail=f"{type_}: {why_not}")
        return key, False
    n = key.decisions + 1
    st.decisions.append(Decision(
        decision_id=f"{key.occurrence_id}/{n}", occurrence_id=key.occurrence_id, type=type_,
        condition_id=key.condition_id, stream_id=st.stream_id, subject=key.subject,
        t_observed=st.now if t_observed is None else t_observed, t_decided=st.now,
        confidence=confidence, reason=reason, cost=key.cost, trace_id=key.occurrence_id,
        request_ids=key.request_ids, cause=cause))
    key = replace(key, decisions=n, cost=Usage(), request_ids=())
    if type_ == "met":
        # `expired` later carries the confidence of the met the agent actually received.
        key = replace(key, met_emitted=True, last_met=st.now, met_confidence=confidence)
    elif type_ == "uncertain":
        key = replace(key, last_uncertain=st.now)
    return key, True


# --------------------------------------------------------------------------- confirmation and acceptance


def _start_window(key: Key, o: Observation) -> Key:
    e = o.evidence_t[1]
    return replace(key, yes_count=1, window_start=e, window_last=e, interval=tuple(o.evidence_t),
                   interval_epoch=o.stream_epoch)


def _count_yes(st: _Step, key: Key, cond: Condition, o: Observation, *, fate: Optional[Fate] = "judged",
               absorbed: bool = False) -> Key:
    """Count a confident yes into the confirm window, on evidence time [Confirm window]. Traces the
    observation's fate, and confirms the occurrence when the count reaches ``confirm``."""
    e = o.evidence_t[1]
    if key.yes_count == 0:
        key, event = _start_window(key, o), "counted"
    elif max(key.window_last, e) - min(key.window_start, e) <= cond.confirm_window_s:
        key = replace(key, yes_count=key.yes_count + 1,
                      window_start=min(key.window_start, e), window_last=max(key.window_last, e),
                      interval=(min(key.interval[0], o.evidence_t[0]), max(key.interval[1], e)))
        event = "counted"
    elif e > key.window_last:
        # The earlier yeses were a flicker; this one starts the window again, same occurrence.
        key, event = _start_window(key, o), "window_restarted"
    else:
        st.trace("older_than_window", cond.id, o.subject, fate="judged", occurrence_id=key.occurrence_id)
        return key
    key = replace(key, yes_confidence=o.confidence)
    st.trace(event, cond.id, o.subject, fate=fate, occurrence_id=key.occurrence_id,
             detail=f"{key.yes_count}/{cond.confirm}")
    if key.yes_count < cond.confirm:
        # An unfinished window gives up once a yes inside it can no longer arrive (arrival time).
        close_at = key.window_start + cond.confirm_window_s + cond.candidate_deadline_s
        st.emit(ScheduleTimer(cond.id, o.subject, "window", close_at))
        return replace(key, window_close_at=close_at)
    return _confirm(st, key, cond, o, absorbed=absorbed)


def _confirm(st: _Step, key: Key, cond: Condition, o: Observation, *, absorbed: bool = False) -> Key:
    """The window is complete: accept now (verification off, or the fast path), or verify [States] [Events]."""
    if key.window_close_at is not None:
        key = replace(key, window_close_at=None)
        st.emit(CancelTimer(cond.id, o.subject, "window"))
    mode = cond.verify_mode
    fast = cond.fast_path_confidence is not None and o.confidence >= cond.fast_path_confidence
    if mode == "off" or fast:
        return _accept(st, key, cond, o.confidence, o.reason)
    deadline = st.now + cond.candidate_deadline_s
    key = replace(key, phase=VERIFYING, verify_deadline=deadline)
    original = mode == "original"
    st.emit(RequestVerify(
        condition_id=cond.id, subject=o.subject, occurrence_id=key.occurrence_id, occurrence_gen=key.gen,
        condition_version=cond.version, mode=mode, interval=key.interval if original else None,
        stream_epoch=key.interval_epoch if original else st.epoch, deadline=deadline, absorbed=absorbed))
    st.emit(ScheduleTimer(cond.id, o.subject, "verify", deadline))
    return key


def _accept(st: _Step, key: Key, cond: Condition, confidence: float, reason: str) -> Key:
    """The occurrence is real: hold the state or report the event, and emit ``met`` [States] [Events].

    A suppressed state ``met`` is remembered and delivered later (``met_pending``, [Emission rules]); a
    suppressed event ``met`` stays silent.
    """
    now = st.now
    if key.verify_deadline is not None:
        key = replace(_settled(key), verify_deadline=None)
        st.emit(CancelTimer(key.condition_id, key.subject, "verify"))
    if cond.kind == "state":
        key = replace(key, phase=HOLDING, held_since=now, no_count=0, next_recheck=now + cond.recheck_every_s)
        st.emit(ScheduleTimer(cond.id, key.subject, "recheck", key.next_recheck))
        if cond.max_open_s is not None:
            key = replace(key, expires_at=now + cond.max_open_s)
            st.emit(ScheduleTimer(cond.id, key.subject, "max_open", key.expires_at))
    else:
        key = replace(key, phase=REPORTED, rearm_until=now + cond.rearm_s)
        st.emit(ScheduleTimer(cond.id, key.subject, "rearm", key.rearm_until))
    key, emitted = _emit(st, key, cond, "met", None, reason, confidence, key.window_start)
    if not emitted and cond.kind == "state":
        key = replace(key, met_pending=True)
    return key


def _unverified(st: _Step, key: Key, cond: Condition, cause: UnverifiedCause, *, verify_answered: bool,
                promote: bool = True) -> Key:
    """on_unverified [Unverified candidates]: a candidate closes without a verified answer.

    Only a candidate supported by a confident yes is reported, under its own occurrence id: as
    ``uncertain`` or as ``unverified`` health. The key returns to idle with no gen bump and no floor, and
    an absorbed detection is promoted to a successor candidate [Absorbed entries] (not on a restart).
    """
    absorbed, usage = key.absorbed, key.absorbed_usage
    key = replace(key, **_NO_ABSORBED)
    if absorbed is not None and not promote:
        key = _charge_usage(key, absorbed.request_id, usage)
    if key.yes_count >= 1:
        reason = _UNVERIFIED_REASONS[cause]
        if cond.on_unverified == "uncertain":
            key, _ = _emit(st, key, cond, "uncertain", cause, reason, key.yes_confidence, key.window_start)
        else:
            st.emit(EmitHealth(Health(kind="unverified", stream_id=st.stream_id, t_wall=st.now, detail=reason,
                                      condition_id=cond.id, subject=key.subject)))
    st.trace("unverified", cond.id, key.subject, occurrence_id=key.occurrence_id, detail=cause)
    key = _close(st, key, verify_answered=verify_answered)
    return key if absorbed is None or not promote else _promote(st, key, cond, absorbed, usage)


def _promote(st: _Step, key: Key, cond: Condition, o: Observation, usage: Usage) -> Key:
    """A verifying candidate closed without ``met`` while holding an absorbed detect yes [Absorbed entries]:
    that yes opens at most one successor occurrence, which is verified on its own evidence (the absorbed
    interval) like any candidate. The key is idle here. The absorbed detect is charged to the successor,
    once; it was never charged to the closed candidate, and its fate was traced when it was absorbed.
    Evidence from another epoch cannot be verified: an ``EvidenceGap`` is reported instead."""
    if o.stream_epoch != st.epoch:
        st.emit(EvidenceGap(cond.id, key.subject, tuple(o.evidence_t), "epoch"))
        st.trace("promotion_refused", cond.id, key.subject, occurrence_id=key.occurrence_id, detail="epoch")
        return key
    key = replace(key, last_touched=st.now, occurrence_id=st.mint_occurrence_id(cond.id), gen=st.mint_gen(),
                  decisions=0, yes_count=0)
    key = _charge_usage(key, o.request_id, usage)
    st.trace("promoted", cond.id, key.subject, occurrence_id=key.occurrence_id, detail=o.request_id)
    if cond.confirm > 1:
        key = replace(key, phase=CANDIDATE)
    return _count_yes(st, key, cond, o, fate=None, absorbed=True)


# --------------------------------------------------------------------------- capacity


def _pinned(key: Key, cond: Optional[Condition], now: float) -> bool:
    """Open keys, and idle keys still inside their cooldown, are never evicted."""
    if key.phase != IDLE:
        return True
    return cond is not None and _in_cooldown(cond, _last_alert(key), now)


def _admit_key(st: _Step, condition_id: str, subject: str) -> Optional[Key]:
    """A new key, evicting the least-recently-touched unpinned key at capacity. None (with a
    ``capacity`` health effect) when every key is pinned. The caller stores the key."""
    if len(st.keys) >= st.max_keys:
        evictable = [k for k in st.keys.values() if not _pinned(k, st.conditions.get(k.condition_id), st.now)]
        if not evictable:
            st.emit(EmitHealth(Health(kind="capacity", stream_id=st.stream_id, t_wall=st.now,
                                      detail=f"{len(st.keys)} keys, none evictable",
                                      condition_id=condition_id, subject=subject)))
            return None
        victim = min(evictable, key=lambda k: (k.last_touched, k.condition_id, k.subject))
        del st.keys[(victim.condition_id, victim.subject)]
        st.trace("evicted", victim.condition_id, victim.subject)
    return Key(condition_id, subject, last_touched=st.now)


# --------------------------------------------------------------------------- observations


AnswerClass = Literal["failed", "unknown", "weak", "yes", "no"]


def _classify(o: Observation, cond: Condition) -> AnswerClass:
    """[Answers] Confidence is symmetric; a model ``unknown`` is unknown at any confidence."""
    if o.failed:
        return "failed"
    if o.answer == "unknown":
        return "unknown"
    if o.confidence < cond.min_confidence:
        return "weak"
    return o.answer


def _stale_reason(st: _Step, key: Optional[Key], cond: Condition, o: Observation) -> Optional[str]:
    """[Stale results], in order: epoch, version, generation, purpose, evidence floor."""
    if o.purpose == "verify" and key is not None and key.phase == VERIFYING and cond.verify_mode == "original":
        expected_epoch = key.interval_epoch    # about the candidate's own evidence
    else:
        expected_epoch = st.epoch
    if o.stream_epoch != expected_epoch:
        return "epoch"
    if o.condition_version != cond.version:
        return "version"
    if o.occurrence_gen != (0 if key is None else key.gen):
        return "generation"
    if o.purpose == "verify" and (key is None or key.phase != VERIFYING):
        return "purpose"
    if o.purpose == "detect" and key is not None and key.closed_evidence_end is not None \
            and o.evidence_t[1] <= key.closed_evidence_end:
        return "evidence"
    return None


def _on_observed(st: _Step, inp: Observed) -> None:
    """[Order of processing]: stream check, version sync, advance, stale check, charge, failed,
    weak, phase table."""
    o = inp.observation
    if o.stream_id != st.stream_id:
        raise ValueError(f"observation for stream {o.stream_id!r} sent to {st.stream_id!r}")
    cid, subject = o.condition_id, o.subject
    _sync_condition(st, cid)
    cond = st.conditions.get(cid)
    if cond is None:
        st.trace("stale_result", cid, subject, fate="stale", detail="unknown_condition")
        return
    key = st.keys.get((cid, subject))
    if key is not None:
        key = st.put(_advance(st, key, cond))

    reason = _stale_reason(st, key, cond, o)
    if reason is not None:
        st.trace("stale_result", cid, subject, fate="stale", detail=reason)
        return
    answer = _classify(o, cond)
    if key is not None and key.phase != IDLE and not _absorbable(key, cond, o, answer):
        key = st.put(_charge(key, inp))
    occurrence_id = None if key is None else key.occurrence_id
    if answer == "failed":
        st.trace("failure", cid, subject, fate="failed", occurrence_id=occurrence_id, detail=o.failure or "")
        return
    if answer == "weak":
        st.trace("weak_answer", cid, subject, fate="judged_weak", occurrence_id=occurrence_id)
        return
    if key is not None and answer != "unknown":
        key = st.put(replace(key, last_touched=st.now))

    if o.purpose == "verify":
        _on_verify_answer(st, key, cond, o, answer)
    elif answer == "unknown":
        # The model can't tell: no information, in every phase [Answers].
        phase = IDLE if key is None else key.phase
        st.trace(f"unknown_while_{phase}", cid, subject, fate="judged_weak", occurrence_id=occurrence_id)
    else:
        table = _STATE_DETECT if cond.kind == "state" else _EVENT_DETECT
        table[IDLE if key is None else key.phase](st, key, cond, inp, answer)


def _on_verify_answer(st: _Step, key: Key, cond: Condition, o: Observation, answer: AnswerClass) -> None:
    """A verify answer for a key in ``verifying`` (the stale check guarantees the phase)."""
    occurrence_id = key.occurrence_id
    if answer == "yes":
        st.trace("verified", cond.id, o.subject, fate="judged", occurrence_id=occurrence_id)
        key = _accept(st, key, cond, o.confidence, o.reason)
    elif answer == "no":
        st.trace("verify_rejected", cond.id, o.subject, fate="judged", occurrence_id=occurrence_id)
        floor = key.interval[1] if cond.verify_mode == "original" else o.evidence_t[1]
        absorbed, usage = key.absorbed, key.absorbed_usage
        key = _close(st, _raise_floor(replace(key, **_NO_ABSORBED), floor), verify_answered=True)
        if absorbed is not None:
            key = _promote(st, key, cond, absorbed, usage)
    else:  # looking at the evidence again, the verifier can't tell
        st.trace("verify_unknown", cond.id, o.subject, fate="judged_weak", occurrence_id=occurrence_id)
        key = _unverified(st, key, cond, "unverified", verify_answered=True)
    st.put(key)


DetectHandler = Callable[[_Step, Optional[Key], Condition, Observed, AnswerClass], None]


def _idle_detect(st: _Step, key: Optional[Key], cond: Condition, inp: Observed, answer: AnswerClass) -> None:
    """Idle, both kinds: a confident yes opens an occurrence; a no is nothing."""
    o = inp.observation
    if answer == "no":
        st.trace("no_while_idle", cond.id, o.subject, fate="judged")
        return
    if key is None:
        key = _admit_key(st, cond.id, o.subject)
        if key is None:
            st.trace("capacity_refused", cond.id, o.subject, fate="dropped_capacity")
            return
    key = replace(key, last_touched=st.now, occurrence_id=st.mint_occurrence_id(cond.id),
                  gen=st.mint_gen(), decisions=0, yes_count=0)
    key = _charge(key, inp)
    st.trace("opened", cond.id, o.subject, occurrence_id=key.occurrence_id)
    if cond.confirm > 1:
        key = replace(key, phase=CONFIRMING if cond.kind == "state" else CANDIDATE)
    st.put(_count_yes(st, key, cond, o))


def _confirming_detect(st: _Step, key: Key, cond: Condition, inp: Observed, answer: AnswerClass) -> None:
    """State, confirming: a yes counts; a no from inside the window is a flicker [States]."""
    o = inp.observation
    if answer == "yes":
        st.put(_count_yes(st, key, cond, o))
    elif o.evidence_t[1] >= key.window_start:
        st.trace("flicker", cond.id, o.subject, fate="judged", occurrence_id=key.occurrence_id,
                 detail="confident no while confirming")
        st.put(_close(st, key))
    else:
        st.trace("older_than_window", cond.id, o.subject, fate="judged", occurrence_id=key.occurrence_id)


def _candidate_detect(st: _Step, key: Key, cond: Condition, inp: Observed, answer: AnswerClass) -> None:
    """Event, candidate: a yes counts; a no is not evidence against an event [Events]."""
    o = inp.observation
    if answer == "yes":
        st.put(_count_yes(st, key, cond, o))
    else:
        st.trace("ignored_while_candidate", cond.id, o.subject, fate="judged", occurrence_id=key.occurrence_id)


def _verifying_detect(st: _Step, key: Key, cond: Condition, inp: Observed, answer: AnswerClass) -> None:
    """Verifying, both kinds: only the verification decides; detections are absorbed. An event's first
    confident yes about evidence newer than the candidate's is kept whole, with its usage [Absorbed entries]."""
    o = inp.observation
    if _absorbable(key, cond, o, answer):
        st.put(replace(key, absorbed=o, absorbed_usage=inp.usage))
    st.trace("awaiting_verification", cond.id, o.subject, fate="judged", occurrence_id=key.occurrence_id)


def _holding_detect(st: _Step, key: Key, cond: Condition, inp: Observed, answer: AnswerClass) -> None:
    """State, holding [States]: a yes resets the no-count and may deliver a deferred met; enough
    confident no's clear the state and set the evidence floor."""
    o = inp.observation
    occurrence_id = key.occurrence_id
    if answer == "yes":
        key = replace(key, no_count=0)
        st.trace("held", cond.id, o.subject, fate="judged", occurrence_id=occurrence_id)
        if key.met_pending and not key.expired:
            key, emitted = _emit(st, key, cond, "met", "deferred", o.reason, o.confidence, key.window_start)
            if emitted:
                key = replace(key, met_pending=False)
    else:
        key = replace(key, no_count=key.no_count + 1)
        st.trace("no_while_holding", cond.id, o.subject, fate="judged", occurrence_id=occurrence_id,
                 detail=f"{key.no_count}/{cond.clear_after_no}")
        if key.no_count >= cond.clear_after_no:
            end = o.evidence_t[1]
            key, _ = _emit(st, key, cond, "cleared", "no", o.reason, o.confidence, end)
            key = _close(st, _raise_floor(key, end))
    st.put(key)


def _reported_detect(st: _Step, key: Key, cond: Condition, inp: Observed, answer: AnswerClass) -> None:
    """Event, reported [Events]: a repeat is absorbed without extending rearm; events never clear."""
    event = "repeat_absorbed" if answer == "yes" else "ignored_while_reported"
    st.trace(event, cond.id, inp.observation.subject, fate="judged", occurrence_id=key.occurrence_id)


_STATE_DETECT: Mapping[str, DetectHandler] = MappingProxyType({
    IDLE: _idle_detect, CONFIRMING: _confirming_detect, VERIFYING: _verifying_detect, HOLDING: _holding_detect,
})
_EVENT_DETECT: Mapping[str, DetectHandler] = MappingProxyType({
    IDLE: _idle_detect, CANDIDATE: _candidate_detect, VERIFYING: _verifying_detect, REPORTED: _reported_detect,
})


# --------------------------------------------------------------------------- timers


def _on_timer(st: _Step, inp: TimerFired) -> None:
    """Sync, remember the stored deadline for this kind, advance; then reschedule an early
    timer, ignore one with no deadline, or act on a due recheck."""
    _sync_condition(st, inp.condition_id)
    cond = st.conditions.get(inp.condition_id)
    key = st.keys.get((inp.condition_id, inp.subject))
    if cond is None or key is None:
        st.trace("timer_ignored", inp.condition_id, inp.subject, detail=inp.kind)
        return
    due_at = _timers(key).get(inp.kind)
    key = _advance(st, key, cond)
    if due_at is None:
        st.trace("timer_ignored", cond.id, key.subject, detail=inp.kind)
    elif st.now < due_at:
        st.emit(ScheduleTimer(cond.id, key.subject, inp.kind, due_at))   # early: ask again
    elif inp.kind == "recheck" and key.phase == HOLDING:
        st.emit(RequestRecheck(cond.id, key.subject, key.occurrence_id, key.gen, cond.version))
        key = replace(key, next_recheck=st.now + cond.recheck_every_s)
        st.emit(ScheduleTimer(cond.id, key.subject, "recheck", key.next_recheck))
    # any other due deadline was applied by _advance
    st.put(key)


# --------------------------------------------------------------------------- controls


def _find_open(st: _Step, occurrence_id: str) -> Optional[Key]:
    for key in st.keys.values():
        if key.phase != IDLE and key.occurrence_id == occurrence_id:
            return key
    return None


def _condition_of(stream_id: str, occurrence_id: str) -> Optional[str]:
    """The condition id inside an occurrence id of this stream (ids parse right to left [Identity])."""
    prefix = stream_id + "/"
    if not occurrence_id.startswith(prefix):
        return None
    rest = occurrence_id[len(prefix):]
    if "/" not in rest:
        return None
    return rest.rsplit("/", 1)[0]


def _close_externally(st: _Step, key: Key, cond: Condition, cause: str, reason: str) -> Key:
    """Resolve, false alarm, scene return: the world was declared changed. A held state emits
    ``cleared``; the key goes idle with a gen bump [Identity]."""
    if cond.kind == "state" and key.phase == HOLDING:
        key, _ = _emit(st, key, cond, "cleared", cause, reason, 1.0, None)
    st.trace("closed", cond.id, key.subject, occurrence_id=key.occurrence_id, detail=cause)
    return _close(st, key, bump=True)


def _close_open_occurrence(st: _Step, occurrence_id: str, cause: str, reason: str) -> None:
    """Find the open occurrence, sync its condition, advance it, and close it if still open."""
    key = _find_open(st, occurrence_id)
    if key is not None:
        _sync_condition(st, key.condition_id)      # may close or remove it
        key = _find_open(st, occurrence_id)
    if key is not None:
        key = st.put(_advance(st, key, st.conditions[key.condition_id]))   # may close it
        if key.phase == IDLE:
            key = None
    if key is None:
        st.trace("control_ignored", _condition_of(st.stream_id, occurrence_id) or "?",
                 occurrence_id=occurrence_id, detail=f"{cause}: not open")
        return
    st.put(_close_externally(st, key, st.conditions[key.condition_id], cause, reason))


def _on_resolve(st: _Step, inp: Resolve) -> None:
    _close_open_occurrence(st, inp.occurrence_id, "resolve", _RESOLVED_REASON)


def _on_false_alarm(st: _Step, inp: FalseAlarm) -> None:
    """Store the counter-example (even for a closed occurrence), then close it if open."""
    cid = _condition_of(st.stream_id, inp.occurrence_id)
    if cid is None or cid not in st.conditions:
        st.trace("control_ignored", cid or "?", occurrence_id=inp.occurrence_id, detail="false_alarm: unknown")
        return
    st.emit(StoreCounterExample(cid, inp.occurrence_id, inp.note))
    reason = "marked a false alarm" + (f": {inp.note}" if inp.note else "")
    _close_open_occurrence(st, inp.occurrence_id, "false_alarm", reason)


def _on_snooze(st: _Step, inp: Snooze) -> None:
    """Replace semantics; ``until <= now`` cancels. Snoozes that have ended are dropped too.
    A snooze for a condition not in ``conditions`` is ignored, so none can leak."""
    _sync_condition(st, inp.condition_id)
    if inp.condition_id not in st.conditions:
        st.trace("control_ignored", inp.condition_id, inp.subject, detail="snooze: unknown")
        return
    target = (inp.condition_id, inp.subject)
    st.snoozes = {k: until for k, until in st.snoozes.items() if until > st.now and k != target}
    if inp.until > st.now:
        st.snoozes[target] = inp.until
        st.trace("snoozed", inp.condition_id, inp.subject, detail=f"until {inp.until}")
    else:
        st.trace("snooze_cancelled", inp.condition_id, inp.subject)


def _on_scene_return(st: _Step, inp: SceneReturn) -> None:
    """Closes open state occurrences of the condition (one subject, or all); events ignore it."""
    _sync_condition(st, inp.condition_id)
    cond = st.conditions.get(inp.condition_id)
    if cond is None:
        st.trace("control_ignored", inp.condition_id, inp.subject, detail="scene_return: unknown")
        return
    for ident in sorted(i for i in st.keys if i[0] == cond.id and inp.subject in (None, i[1])):
        key = st.put(_advance(st, st.keys[ident], cond))
        if key.phase == IDLE:
            continue
        if cond.kind == "event":
            st.trace("scene_return_ignored", cond.id, key.subject, occurrence_id=key.occurrence_id)
            continue
        st.put(_close_externally(st, key, cond, "scene_return", _SCENE_RETURN_REASON))


# --------------------------------------------------------------------------- conditions


def _sync_condition(st: _Step, condition_id: str, idents: Optional[list[KeyId]] = None) -> bool:
    """Bring the recorded version of one condition in line with ``conditions``.

    Updated: every key of the condition closes with a gen bump (a held state emits ``cleared``,
    cause ``condition_changed``). Removed: its keys, recorded version and snoozes are deleted,
    with no decision. Returns True when anything changed. O(1) when nothing did; otherwise
    O(K), unless the caller passes the condition's key ids (``idents``, sorted).
    """
    cond = st.conditions.get(condition_id)
    recorded = st.versions.get(condition_id)
    if (cond is None and recorded is None) or (cond is not None and recorded == cond.version):
        return False
    if idents is None:
        idents = sorted(i for i in st.keys if i[0] == condition_id)
    for ident in idents:
        key = st.keys[ident]
        if cond is None:
            st.trace("condition_removed", condition_id, key.subject, occurrence_id=key.occurrence_id)
            _close(st, key)
            del st.keys[ident]
            continue
        if key.phase == HOLDING:
            key, _ = _emit(st, key, cond, "cleared", "condition_changed", _CONDITION_CHANGED_REASON, 1.0, None)
        st.trace("condition_changed", condition_id, key.subject, occurrence_id=key.occurrence_id,
                 detail=f"version {recorded} -> {cond.version}")
        st.put(_close(st, key, bump=True))
    if cond is None:
        st.versions.pop(condition_id, None)
        if any(k[0] == condition_id for k in st.snoozes):
            st.snoozes = {k: until for k, until in st.snoozes.items() if k[0] != condition_id}
    else:
        st.versions[condition_id] = cond.version
    return True


def _on_condition_changed(st: _Step, inp: ConditionChanged) -> None:
    if not _sync_condition(st, inp.condition_id):
        st.trace("condition_unchanged", inp.condition_id)


# --------------------------------------------------------------------------- epochs


def _on_epoch(st: _Step, inp: EpochStarted) -> None:
    """Set the epoch, sync every condition that has a recorded version or a key (so removed
    conditions are purged), then per key in sorted order: advance, apply restart or reconnect
    handling, and re-emit every live deadline as a timer. O(K log K + C log C): keys are grouped
    by condition once, not scanned once per changed condition."""
    if inp.stream_epoch <= st.epoch:
        st.trace("epoch_ignored", "*", detail=f"{inp.stream_epoch} <= {st.epoch}")
        return
    st.epoch = inp.stream_epoch
    st.trace("epoch_started", "*", detail=f"{inp.stream_epoch} restart={inp.restart}")
    by_condition: dict[str, list[KeyId]] = {}
    for ident in sorted(st.keys):
        by_condition.setdefault(ident[0], []).append(ident)
    for cid in sorted(set(st.versions) | set(by_condition)):
        _sync_condition(st, cid, by_condition.get(cid, []))
    for ident in sorted(st.keys):
        cond = st.conditions[ident[0]]
        key = _advance(st, st.keys[ident], cond)
        st.put(_resume(st, key, cond, restart=inp.restart))


def _resume(st: _Step, key: Key, cond: Condition, *, restart: bool) -> Key:
    """One key at the start of a new epoch [Restart and reconnect].

    A fresh verification is requested again for the new epoch (same deadline). On a restart,
    an event candidate or original-interval verification resolves via on_unverified with cause
    ``restart``; on a reconnect it stays. A held state stays held, emits no met and is
    rechecked now.
    """
    if key.phase == VERIFYING and cond.verify_mode == "fresh":
        st.emit(RequestVerify(
            condition_id=cond.id, subject=key.subject, occurrence_id=key.occurrence_id, occurrence_gen=key.gen,
            condition_version=cond.version, mode="fresh", interval=None, stream_epoch=st.epoch,
            deadline=key.verify_deadline))
    elif restart and key.phase in (CANDIDATE, VERIFYING):
        # The absorbed detection's frames died with the process: no successor [Absorbed entries].
        key = _unverified(st, key, cond, "restart", verify_answered=False, promote=False)
    if key.phase == HOLDING:
        st.emit(RequestRecheck(cond.id, key.subject, key.occurrence_id, key.gen, cond.version))
        key = replace(key, next_recheck=st.now + cond.recheck_every_s)
    for kind, at in sorted(_timers(key).items()):
        st.emit(ScheduleTimer(cond.id, key.subject, kind, at))
    return key


_HANDLERS: Mapping[type, Callable[[_Step, Input], None]] = MappingProxyType({
    Observed: _on_observed,
    TimerFired: _on_timer,
    Resolve: _on_resolve,
    FalseAlarm: _on_false_alarm,
    Snooze: _on_snooze,
    SceneReturn: _on_scene_return,
    ConditionChanged: _on_condition_changed,
    EpochStarted: _on_epoch,
})
