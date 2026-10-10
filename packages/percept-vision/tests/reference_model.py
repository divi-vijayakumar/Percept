"""Reference model of the decision reducer: a test oracle, not shipped code.

This is the simplest implementation of the transition specification that is obviously
correct when read top to bottom. Speed doesn't matter here; readability does. The real
reducer (``percept.decide``) must produce exactly the same outputs for the same inputs, so
the scenario and property tests are written against the API below and run against both.

The API
-------
``new_state(stream_id, *, created_at, stream_epoch=0, max_keys=DEFAULT_MAX_KEYS)``
    A fresh per-stream state.
``step(state, inp, now, conditions) -> StepResult(state, decisions, effects)``
    Pure: the argument ``state`` is never mutated; ``now`` is wall-clock seconds.
``to_dict(state)`` / ``from_dict(data)``
    A JSON-ready snapshot and its inverse.
``gen_for(state, condition_id, subject)``
    The occurrence generation the runtime tags a new judge request with.
``phase_of(state, condition_id, subject)`` / ``decision_view(state)``
    Inspection for tests: the lifecycle phase of a key, and the state without bookkeeping.

Everything else in the shell (budget, dispatch, judge calls, delivery) is out of scope.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
from typing import Literal, Mapping, Optional, Union

from percept.contracts import Condition, Decision, Fate, Health, Observation, Usage

SCENE = "scene"
DEFAULT_MAX_KEYS = 64
MAX_REQUEST_IDS = 16          # request ids kept per pending decision (the most recent ones)

TimerKind = Literal["window", "verify", "recheck", "max_open", "rearm"]
TIMER_KINDS: tuple[str, ...] = ("window", "verify", "recheck", "max_open", "rearm")

# Phases. A state condition is idle, confirming, verifying or holding; an event condition is
# idle, candidate, verifying or reported. "Open" means any phase other than idle.
STATE_PHASES = ("idle", "confirming", "verifying", "holding")
EVENT_PHASES = ("idle", "candidate", "verifying", "reported")


# --------------------------------------------------------------------------- inputs


@dataclass(frozen=True)
class Observed:
    """A judge observation, plus its attributed share of the request's usage."""

    observation: Observation
    usage: Usage = Usage()


@dataclass(frozen=True)
class TimerFired:
    """A timer the reducer scheduled has fired (the runtime may fire early or late)."""

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
    """Suppress met and uncertain until ``until`` (wall clock); replaces any earlier snooze
    for the same (condition, subject). ``until <= now`` cancels it. ``subject=None`` is every
    subject of the condition."""

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
    """The stream's epoch increased: a reconnect, or a restart from a snapshot."""

    stream_epoch: int
    restart: bool


Input = Union[Observed, TimerFired, Resolve, FalseAlarm, Snooze, SceneReturn, ConditionChanged, EpochStarted]


# --------------------------------------------------------------------------- effects


@dataclass(frozen=True)
class ScheduleTimer:
    """Fire ``TimerFired(condition_id, subject, kind)`` at wall time ``at``. Replaces any
    pending timer with the same (condition_id, subject, kind)."""

    condition_id: str
    subject: str
    kind: TimerKind
    at: float


@dataclass(frozen=True)
class CancelTimer:
    condition_id: str
    subject: str
    kind: TimerKind


@dataclass(frozen=True)
class RequestVerify:
    """Judge this condition again with ``purpose="verify"``.

    ``mode="original"`` re-judges the evidence ``interval`` from ``stream_epoch``;
    ``mode="fresh"`` judges a new frame (``interval`` is None). The observation that answers it
    carries this request's ``stream_epoch``, ``occurrence_gen`` and ``condition_version``. The
    runtime may retry until ``deadline``; the reducer resolves the candidate itself when the
    deadline passes.
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


@dataclass(frozen=True)
class CancelVerify:
    """Drop a queued or in-flight verification: its occurrence stopped verifying for a reason
    other than a verify answer (resolve, false alarm, scene return, condition change, deadline,
    restart). An answer that still arrives is rejected as stale."""

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
class EmitHealth:
    health: Health


@dataclass(frozen=True)
class StoreCounterExample:
    """Keep a 'this is not it' example for the condition. The runtime attaches the thumbnail."""

    condition_id: str
    occurrence_id: str
    note: str


@dataclass(frozen=True)
class Trace:
    """A trace record. Every ``Observed`` input yields exactly one Trace with a ``fate``."""

    t: float
    event: str
    condition_id: str
    subject: Optional[str] = None
    fate: Optional[Fate] = None
    occurrence_id: Optional[str] = None
    detail: str = ""


Effect = Union[ScheduleTimer, CancelTimer, RequestVerify, CancelVerify, RequestRecheck, EmitHealth,
               StoreCounterExample, Trace]


# --------------------------------------------------------------------------- state


@dataclass
class Key:
    """Decide state for one (condition_id, subject) on one stream. Every time is wall clock."""

    condition_id: str
    subject: str
    phase: str = "idle"
    gen: int = 0
    # the open occurrence (all reset when the key returns to idle)
    occurrence_id: Optional[str] = None
    decisions: int = 0                          # decisions emitted for this occurrence
    yes_count: int = 0                          # confident yes answers counted in the window
    window_start: Optional[float] = None        # evidence time of the earliest counted yes
    window_last: Optional[float] = None         # evidence time of the latest counted yes
    window_close_at: Optional[float] = None     # arrival time at which an unfinished window gives up
    interval: Optional[tuple[float, float]] = None   # hull of the counted yes evidence
    interval_epoch: Optional[int] = None
    yes_confidence: float = 0.0                 # confidence of the latest counted yes
    verify_deadline: Optional[float] = None
    no_count: int = 0                           # holding: consecutive confident no's
    held_since: Optional[float] = None
    expires_at: Optional[float] = None          # holding with max_open_s, until it passes
    expired: bool = False                       # expiry happened: no deferred met any more
    next_recheck: Optional[float] = None
    rearm_until: Optional[float] = None
    met_emitted: bool = False
    met_pending: bool = False                   # a state's met was suppressed; deliver it later
    met_confidence: float = 0.0
    # survives the return to idle
    last_met: Optional[float] = None
    last_uncertain: Optional[float] = None
    closed_evidence_end: Optional[float] = None  # evidence floor for detect answers
    # bookkeeping: never consulted by a lifecycle transition (last_touched only picks
    # which idle key to evict under capacity pressure)
    cost: Usage = field(default_factory=Usage)
    request_ids: tuple[str, ...] = ()
    last_touched: float = 0.0


BOOKKEEPING_FIELDS = ("cost", "request_ids", "last_touched")


@dataclass
class StreamDecideState:
    stream_id: str
    stream_epoch: int
    incarnation: str                            # minted once, at creation; part of every id
    max_keys: int
    next_occurrence: int = 1
    next_gen: int = 1
    versions: dict[str, int] = field(default_factory=dict)
    keys: dict[tuple[str, str], Key] = field(default_factory=dict)
    snoozes: dict[tuple[str, Optional[str]], float] = field(default_factory=dict)


@dataclass(frozen=True)
class StepResult:
    state: StreamDecideState
    decisions: tuple[Decision, ...]
    effects: tuple[Effect, ...]


def new_state(stream_id: str, *, created_at: float, stream_epoch: int = 0,
              max_keys: int = DEFAULT_MAX_KEYS) -> StreamDecideState:
    if not stream_id:
        raise ValueError("stream_id must not be empty")
    if max_keys < 1:
        raise ValueError("max_keys is >= 1")
    # The incarnation keeps ids unique even if a snapshot is lost and the stream starts over.
    return StreamDecideState(stream_id, stream_epoch, format(int(created_at * 1000), "x"), max_keys)


def gen_for(state: StreamDecideState, condition_id: str, subject: str = SCENE) -> int:
    key = state.keys.get((condition_id, subject))
    return 0 if key is None else key.gen


def phase_of(state: StreamDecideState, condition_id: str, subject: str = SCENE) -> str:
    key = state.keys.get((condition_id, subject))
    return "idle" if key is None else key.phase


def decision_view(state: StreamDecideState) -> dict:
    """The state without bookkeeping fields: what 'unchanged' means in the spec."""
    data = to_dict(state)
    for key in data["keys"]:
        for name in BOOKKEEPING_FIELDS:
            key.pop(name)
    return data


# --------------------------------------------------------------------------- serialisation


def to_dict(state: StreamDecideState) -> dict:
    def key_dict(key: Key) -> dict:
        data = asdict(key)
        data["interval"] = None if key.interval is None else list(key.interval)
        data["request_ids"] = list(key.request_ids)
        return data

    return {
        "stream_id": state.stream_id,
        "stream_epoch": state.stream_epoch,
        "incarnation": state.incarnation,
        "max_keys": state.max_keys,
        "next_occurrence": state.next_occurrence,
        "next_gen": state.next_gen,
        "versions": dict(sorted(state.versions.items())),
        "keys": [key_dict(state.keys[k]) for k in sorted(state.keys)],
        "snoozes": [[c, s, u] for (c, s), u in sorted(state.snoozes.items(), key=lambda i: (i[0][0], i[0][1] or ""))],
    }


def from_dict(data: Mapping) -> StreamDecideState:
    keys = {}
    for raw in data["keys"]:
        raw = dict(raw)
        raw["interval"] = None if raw["interval"] is None else tuple(raw["interval"])
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
        versions=dict(data["versions"]),
        keys=keys,
        snoozes={(c, s): u for c, s, u in data["snoozes"]},
    )


# --------------------------------------------------------------------------- the reducer


class _Out:
    """What one step produces, plus the conditions it runs under."""

    def __init__(self, conditions: Mapping[str, Condition]) -> None:
        self.conditions = conditions
        self.decisions: list[Decision] = []
        self.effects: list[Effect] = []

    def trace(self, now: float, event: str, condition_id: str, subject: Optional[str] = None, *,
              fate: Optional[Fate] = None, occurrence_id: Optional[str] = None, detail: str = "") -> None:
        self.effects.append(Trace(now, event, condition_id, subject, fate, occurrence_id, detail))


def step(state: StreamDecideState, inp: Input, now: float, conditions: Mapping[str, Condition]) -> StepResult:
    s = copy.deepcopy(state)
    out = _Out(conditions)
    if isinstance(inp, Observed):
        _on_observed(s, inp, now, conditions, out)
    elif isinstance(inp, TimerFired):
        _on_timer(s, inp, now, conditions, out)
    elif isinstance(inp, Resolve):
        _on_resolve(s, inp.occurrence_id, "resolve", "resolved by the application", now, conditions, out)
    elif isinstance(inp, FalseAlarm):
        _on_false_alarm(s, inp, now, conditions, out)
    elif isinstance(inp, Snooze):
        _on_snooze(s, inp, now, out)
    elif isinstance(inp, SceneReturn):
        _on_scene_return(s, inp, now, conditions, out)
    elif isinstance(inp, ConditionChanged):
        if not _sync_version(s, inp.condition_id, now, conditions, out):
            out.trace(now, "condition_unchanged", inp.condition_id)
    elif isinstance(inp, EpochStarted):
        _on_epoch(s, inp, now, conditions, out)
    else:
        raise TypeError(f"unknown input {inp!r}")
    return StepResult(s, tuple(out.decisions), tuple(out.effects))


# --------------------------------------------------------------------------- observations


def _classify(o: Observation, cond: Condition) -> str:
    """One of: failed, unknown, weak, yes, no. Confidence is symmetric."""
    if o.failed:
        return "failed"
    if o.answer == "unknown":
        return "unknown"
    if o.confidence < cond.min_confidence:
        return "weak"
    return o.answer


def _stale_reason(s: StreamDecideState, key: Optional[Key], cond: Condition, o: Observation) -> Optional[str]:
    if o.purpose == "verify" and key is not None and key.phase == "verifying" \
            and cond.verify_mode == "original":
        # An original-interval verification is about the candidate's own evidence, so it is
        # checked against the epoch that evidence came from.
        expected_epoch = key.interval_epoch
    else:
        expected_epoch = s.stream_epoch
    if o.stream_epoch != expected_epoch:
        return "epoch"
    if o.condition_version != cond.version:
        return "version"
    if o.occurrence_gen != (0 if key is None else key.gen):
        return "generation"
    if o.purpose == "verify" and (key is None or key.phase != "verifying"):
        return "purpose"
    if o.purpose == "detect" and key is not None and key.closed_evidence_end is not None \
            and o.evidence_t[1] <= key.closed_evidence_end:
        # judged on evidence from before an occurrence was declared over
        return "evidence"
    return None


def _on_observed(s: StreamDecideState, inp: Observed, now: float, conditions: Mapping[str, Condition],
                 out: _Out) -> None:
    o = inp.observation
    if o.stream_id != s.stream_id:
        raise ValueError(f"observation for stream {o.stream_id!r} sent to {s.stream_id!r}")
    cid, subject = o.condition_id, o.subject
    _sync_version(s, cid, now, conditions, out)
    cond = conditions.get(cid)
    if cond is None:
        out.trace(now, "stale_result", cid, subject, fate="stale", detail="unknown_condition")
        return

    key = s.keys.get((cid, subject))
    if key is not None:
        _advance(s, key, cond, now, out)

    reason = _stale_reason(s, key, cond, o)
    if reason is not None:
        out.trace(now, "stale_result", cid, subject, fate="stale", detail=reason)
        return

    if key is not None and key.phase != "idle":
        _charge(key, inp)  # every non-stale observation of an open occurrence is part of its cost

    answer = _classify(o, cond)
    occ = None if key is None else key.occurrence_id
    if answer == "failed":
        out.trace(now, "failure", cid, subject, fate="failed", occurrence_id=occ, detail=o.failure or "")
        return
    if answer == "weak":
        out.trace(now, "weak_answer", cid, subject, fate="judged_weak", occurrence_id=occ)
        return
    if answer in ("yes", "no") and key is not None:
        key.last_touched = now

    if o.purpose == "verify":
        _on_verify_answer(s, key, cond, inp, answer, now, out)
    elif cond.kind == "state":
        _on_state_answer(s, key, cond, inp, answer, now, out)
    else:
        _on_event_answer(s, key, cond, inp, answer, now, out)


def _on_state_answer(s, key, cond, inp, answer, now, out) -> None:
    o = inp.observation
    phase = "idle" if key is None else key.phase
    occ = None if key is None else key.occurrence_id
    if answer == "unknown":
        # The model can't tell: no information, in every phase.
        out.trace(now, f"unknown_while_{phase}", cond.id, o.subject, fate="judged_weak", occurrence_id=occ)
    elif phase == "idle":
        if answer == "yes":
            _open_with_yes(s, key, cond, inp, now, out)
        else:
            out.trace(now, "no_while_idle", cond.id, o.subject, fate="judged")
    elif phase == "confirming":
        if answer == "yes":
            _count_yes(s, key, cond, o, now, out)
        elif o.evidence_t[1] >= key.window_start:
            out.trace(now, "flicker", cond.id, o.subject, fate="judged", occurrence_id=occ,
                      detail="confident no while confirming")
            _to_idle(key, out, now)
        else:
            out.trace(now, "older_than_window", cond.id, o.subject, fate="judged", occurrence_id=occ)
    elif phase == "verifying":
        out.trace(now, "awaiting_verification", cond.id, o.subject, fate="judged", occurrence_id=occ)
    elif phase == "holding":
        if answer == "yes":
            key.no_count = 0
            out.trace(now, "held", cond.id, o.subject, fate="judged", occurrence_id=occ)
            if key.met_pending and not key.expired:
                _deliver_deferred_met(s, key, cond, o, now, out)
        else:
            key.no_count += 1
            out.trace(now, "no_while_holding", cond.id, o.subject, fate="judged", occurrence_id=occ,
                      detail=f"{key.no_count}/{cond.clear_after_no}")
            if key.no_count >= cond.clear_after_no:
                _emit(s, key, cond, "cleared", "no", o.reason, o.confidence, o.evidence_t[1], now, out)
                _raise_floor(key, o.evidence_t[1])
                _to_idle(key, out, now)
    else:
        raise AssertionError(phase)


def _on_event_answer(s, key, cond, inp, answer, now, out) -> None:
    o = inp.observation
    phase = "idle" if key is None else key.phase
    occ = None if key is None else key.occurrence_id
    if answer == "unknown":
        out.trace(now, f"unknown_while_{phase}", cond.id, o.subject, fate="judged_weak", occurrence_id=occ)
    elif phase == "idle":
        if answer == "yes":
            _open_with_yes(s, key, cond, inp, now, out)
        else:
            out.trace(now, "no_while_idle", cond.id, o.subject, fate="judged")
    elif phase == "candidate":
        if answer == "yes":
            _count_yes(s, key, cond, o, now, out)
        else:
            # A later "no" is not evidence against an event: the moment may simply have passed.
            out.trace(now, "ignored_while_candidate", cond.id, o.subject, fate="judged", occurrence_id=occ)
    elif phase == "verifying":
        # Only the verification of the original interval decides; later detections don't.
        out.trace(now, "awaiting_verification", cond.id, o.subject, fate="judged", occurrence_id=occ)
    elif phase == "reported":
        event = "repeat_absorbed" if answer == "yes" else "ignored_while_reported"
        out.trace(now, event, cond.id, o.subject, fate="judged", occurrence_id=occ)
    else:
        raise AssertionError(phase)


def _on_verify_answer(s, key, cond, inp, answer, now, out) -> None:
    o = inp.observation
    occ = key.occurrence_id
    if answer == "yes":
        out.trace(now, "verified", cond.id, o.subject, fate="judged", occurrence_id=occ)
        _accept(s, key, cond, o.confidence, o.reason, now, out)
    elif answer == "no":
        out.trace(now, "verify_rejected", cond.id, o.subject, fate="judged", occurrence_id=occ)
        _raise_floor(key, key.interval[1] if cond.verify_mode == "original" else o.evidence_t[1])
        _to_idle(key, out, now, verify_answered=True)
    else:  # the model, looking at the evidence again, can't tell
        out.trace(now, "verify_unknown", cond.id, o.subject, fate="judged_weak", occurrence_id=occ)
        _unverified(s, key, cond, "unverified", now, out, verify_answered=True)


def _open_with_yes(s, key, cond, inp, now, out) -> None:
    o = inp.observation
    if key is None:
        key = _ensure_key(s, cond.id, o.subject, now, out)
        if key is None:
            out.trace(now, "capacity_refused", cond.id, o.subject, fate="dropped_capacity")
            return
    key.last_touched = now
    key.occurrence_id = f"{s.stream_id}/{cond.id}/{s.incarnation}-{s.next_occurrence}"
    s.next_occurrence += 1
    key.gen = _bump(s)            # a new occurrence: results tagged for the idle period are stale
    key.decisions = 0
    key.yes_count = 0
    _charge(key, inp)
    out.trace(now, "opened", cond.id, o.subject, occurrence_id=key.occurrence_id)
    if cond.confirm > 1:
        key.phase = "confirming" if cond.kind == "state" else "candidate"
    _count_yes(s, key, cond, o, now, out)


def _count_yes(s, key, cond, o, now, out) -> None:
    """Count a confident yes into the confirm window. The window is on evidence time: every
    counted yes lies within ``confirm_window_s`` of every other. Emits the observation's fate."""
    e = o.evidence_t[1]
    occ = key.occurrence_id
    if key.yes_count == 0:
        _start_window(key, o)
        event = "counted"
    elif max(key.window_last, e) - min(key.window_start, e) <= cond.confirm_window_s:
        key.yes_count += 1
        key.window_start = min(key.window_start, e)
        key.window_last = max(key.window_last, e)
        key.interval = (min(key.interval[0], o.evidence_t[0]), max(key.interval[1], o.evidence_t[1]))
        event = "counted"
    elif e > key.window_last:
        _start_window(key, o)     # the earlier yeses were a flicker; this one starts again
        event = "window_restarted"
    else:
        out.trace(now, "older_than_window", cond.id, o.subject, fate="judged", occurrence_id=occ)
        return
    key.yes_confidence = o.confidence
    out.trace(now, event, cond.id, o.subject, fate="judged", occurrence_id=occ,
              detail=f"{key.yes_count}/{cond.confirm}")
    if key.yes_count < cond.confirm:
        # An unfinished window gives up once a yes inside it can no longer arrive.
        key.window_close_at = key.window_start + cond.confirm_window_s + cond.candidate_deadline_s
        out.effects.append(ScheduleTimer(cond.id, o.subject, "window", key.window_close_at))
        return
    if key.window_close_at is not None:
        key.window_close_at = None
        out.effects.append(CancelTimer(cond.id, o.subject, "window"))
    fast = cond.fast_path_confidence is not None and o.confidence >= cond.fast_path_confidence
    mode = cond.verify_mode
    if mode == "off" or fast:
        _accept(s, key, cond, o.confidence, o.reason, now, out)
        return
    key.phase = "verifying"
    key.verify_deadline = now + cond.candidate_deadline_s
    out.effects.append(RequestVerify(
        condition_id=cond.id, subject=o.subject, occurrence_id=occ, occurrence_gen=key.gen,
        condition_version=cond.version, mode=mode,
        interval=key.interval if mode == "original" else None,
        stream_epoch=key.interval_epoch if mode == "original" else s.stream_epoch,
        deadline=key.verify_deadline))
    out.effects.append(ScheduleTimer(cond.id, o.subject, "verify", key.verify_deadline))


def _start_window(key: Key, o: Observation) -> None:
    key.yes_count = 1
    key.window_start = key.window_last = o.evidence_t[1]
    key.interval = o.evidence_t
    key.interval_epoch = o.stream_epoch


def _accept(s, key, cond, confidence, reason, now, out) -> None:
    """The occurrence is real: hold the state or report the event, and emit met."""
    if key.verify_deadline is not None:
        key.verify_deadline = None
        out.effects.append(CancelTimer(key.condition_id, key.subject, "verify"))
    if cond.kind == "state":
        key.phase = "holding"
        key.held_since = now
        key.no_count = 0
        key.next_recheck = now + cond.recheck_every_s
        out.effects.append(ScheduleTimer(cond.id, key.subject, "recheck", key.next_recheck))
        if cond.max_open_s is not None:
            key.expires_at = now + cond.max_open_s
            out.effects.append(ScheduleTimer(cond.id, key.subject, "max_open", key.expires_at))
    else:
        key.phase = "reported"
        key.rearm_until = now + cond.rearm_s
        out.effects.append(ScheduleTimer(cond.id, key.subject, "rearm", key.rearm_until))
    key.met_confidence = confidence
    emitted = _emit(s, key, cond, "met", None, reason, confidence, key.window_start, now, out)
    if not emitted and cond.kind == "state":
        key.met_pending = True    # delivered at the first confident yes after the suppression lifts


def _deliver_deferred_met(s, key, cond, o, now, out) -> None:
    if _emit(s, key, cond, "met", "deferred", o.reason, o.confidence, key.window_start, now, out):
        key.met_pending = False


def _unverified(s, key, cond, cause, now, out, *, verify_answered: bool) -> None:
    """A candidate closes without a verified answer. Only a candidate that a confident yes
    supported is reported; it keeps its occurrence id."""
    if key.yes_count >= 1:
        if cond.on_unverified == "uncertain":
            _emit(s, key, cond, "uncertain", cause, _UNVERIFIED_REASONS[cause], key.yes_confidence,
                  key.window_start, now, out)
        else:
            out.effects.append(EmitHealth(Health(
                kind="unverified", stream_id=s.stream_id, t_wall=now, detail=_UNVERIFIED_REASONS[cause],
                condition_id=cond.id, subject=key.subject)))
    out.trace(now, "unverified", cond.id, key.subject, occurrence_id=key.occurrence_id, detail=cause)
    _to_idle(key, out, now, verify_answered=verify_answered)


_UNVERIFIED_REASONS = {
    "timeout": "a possible occurrence could not be verified before its deadline",
    "unverified": "the verifier could not tell from the original evidence",
    "restart": "verification was interrupted by a restart",
}


# --------------------------------------------------------------------------- emission


def _snoozed(s: StreamDecideState, condition_id: str, subject: str, now: float) -> bool:
    return any(now < s.snoozes.get((condition_id, who), float("-inf")) for who in (subject, None))


def _in_cooldown(cond: Condition, since: Optional[float], now: float) -> bool:
    # Clamped: a wall clock that steps backwards counts as no time elapsed, and a zero
    # cooldown never suppresses anything.
    return cond.cooldown_s > 0 and since is not None and max(0.0, now - since) < cond.cooldown_s


def _last_alert(key: Key) -> Optional[float]:
    times = [t for t in (key.last_met, key.last_uncertain) if t is not None]
    return max(times) if times else None


def _decision(s, key, type_, cause, reason, confidence, t_observed, now) -> Decision:
    key.decisions += 1
    request_ids, cost = key.request_ids, key.cost
    key.cost, key.request_ids = Usage(), ()
    return Decision(
        decision_id=f"{key.occurrence_id}/{key.decisions}", occurrence_id=key.occurrence_id, type=type_,
        condition_id=key.condition_id, stream_id=s.stream_id, subject=key.subject,
        t_observed=now if t_observed is None else t_observed, t_decided=now, confidence=confidence,
        reason=reason, cost=cost, trace_id=key.occurrence_id, request_ids=request_ids, cause=cause)


def _emit(s, key, cond, type_, cause, reason, confidence, t_observed, now, out) -> bool:
    """Emit a decision for the key's occurrence unless a rule suppresses it. Returns True if
    it was emitted."""
    why_not = None
    if type_ in ("met", "uncertain") and _snoozed(s, cond.id, key.subject, now):
        why_not = "snoozed"
    elif type_ == "met" and _in_cooldown(cond, key.last_met, now):
        why_not = "cooldown"
    elif type_ == "uncertain" and _in_cooldown(cond, _last_alert(key), now):
        why_not = "cooldown"
    elif type_ in ("cleared", "expired") and not key.met_emitted:
        why_not = "met_not_emitted"
    elif type_ == "cleared" and (cond.kind == "event" or not cond.emit_cleared):
        why_not = "emit_cleared_off"
    if why_not is not None:
        out.trace(now, "suppressed", cond.id, key.subject, occurrence_id=key.occurrence_id,
                  detail=f"{type_}: {why_not}")
        return False
    if type_ == "met":
        key.met_emitted = True
        key.last_met = now
    if type_ == "uncertain":
        key.last_uncertain = now
    out.decisions.append(_decision(s, key, type_, cause, reason, confidence, t_observed, now))
    return True


def _charge(key: Key, inp: Observed) -> None:
    key.cost = key.cost + inp.usage
    if inp.observation.request_id not in key.request_ids:
        key.request_ids = (key.request_ids + (inp.observation.request_id,))[-MAX_REQUEST_IDS:]


# --------------------------------------------------------------------------- keys, time, idle


def _timers_of(key: Key) -> dict[str, float]:
    """The live deadlines of a key, by timer kind. Each one has exactly one timer."""
    deadlines = {"window": key.window_close_at, "verify": key.verify_deadline, "recheck": key.next_recheck,
                 "max_open": key.expires_at, "rearm": key.rearm_until}
    return {kind: at for kind, at in deadlines.items() if at is not None}


def _raise_floor(key: Key, evidence_end: float) -> None:
    if key.closed_evidence_end is None or evidence_end > key.closed_evidence_end:
        key.closed_evidence_end = evidence_end


def _to_idle(key: Key, out: _Out, now: float, *, bump_to: Optional[int] = None,
             verify_answered: bool = False) -> None:
    """Close whatever is open: cancel its timers (and its verification, unless a verify answer
    closed it), and reset everything except the gen (unless bumped) and what survives idle."""
    if key.phase == "verifying" and not verify_answered:
        out.effects.append(CancelVerify(key.condition_id, key.subject, key.occurrence_id, key.gen))
    for kind in _timers_of(key):
        out.effects.append(CancelTimer(key.condition_id, key.subject, kind))
    fresh = Key(key.condition_id, key.subject, gen=key.gen if bump_to is None else bump_to,
                last_met=key.last_met, last_uncertain=key.last_uncertain,
                closed_evidence_end=key.closed_evidence_end, last_touched=now)
    key.__dict__.update(fresh.__dict__)


def _bump(s: StreamDecideState) -> int:
    gen = s.next_gen
    s.next_gen += 1
    return gen


def _advance(s, key, cond, now, out) -> None:
    """Apply every deadline that has passed. Deadlines are data, so a late or lost timer
    can't change the outcome: the next input that touches the key applies them."""
    if key.phase in ("confirming", "candidate") and now >= key.window_close_at:
        out.trace(now, "flicker", cond.id, key.subject, occurrence_id=key.occurrence_id,
                  detail="confirm window ended")
        _to_idle(key, out, now)
    if key.phase == "verifying" and now >= key.verify_deadline:
        _unverified(s, key, cond, "timeout", now, out, verify_answered=False)
    if key.phase == "reported" and now >= key.rearm_until:
        out.trace(now, "rearmed", cond.id, key.subject, occurrence_id=key.occurrence_id)
        _raise_floor(key, key.rearm_until)
        _to_idle(key, out, now)
    if key.phase == "holding" and key.expires_at is not None and now >= key.expires_at:
        key.expires_at = None
        key.expired = True
        key.met_pending = False   # never deliver a met for an occurrence that has expired
        _emit(s, key, cond, "expired", "max_open", "still true after max_open_s", key.met_confidence,
              None, now, out)


def _pinned(key: Key, cond: Optional[Condition], now: float) -> bool:
    """Open keys, and idle keys still inside their cooldown, are never evicted."""
    if key.phase != "idle":
        return True
    return cond is not None and _in_cooldown(cond, _last_alert(key), now)


def _ensure_key(s, condition_id, subject, now, out) -> Optional[Key]:
    if (condition_id, subject) in s.keys:
        return s.keys[(condition_id, subject)]
    if len(s.keys) >= s.max_keys:
        evictable = [k for k in s.keys.values() if not _pinned(k, out.conditions.get(k.condition_id), now)]
        if not evictable:
            out.effects.append(EmitHealth(Health(
                kind="capacity", stream_id=s.stream_id, t_wall=now,
                detail=f"{len(s.keys)} keys, none evictable", condition_id=condition_id, subject=subject)))
            return None
        victim = min(evictable, key=lambda k: (k.last_touched, k.condition_id, k.subject))
        del s.keys[(victim.condition_id, victim.subject)]
        out.trace(now, "evicted", victim.condition_id, victim.subject)
    key = Key(condition_id, subject, last_touched=now)
    s.keys[(condition_id, subject)] = key
    return key


# --------------------------------------------------------------------------- other inputs


def _on_timer(s, inp: TimerFired, now, conditions, out) -> None:
    _sync_version(s, inp.condition_id, now, conditions, out)
    cond = conditions.get(inp.condition_id)
    key = s.keys.get((inp.condition_id, inp.subject))
    if cond is None or key is None:
        out.trace(now, "timer_ignored", inp.condition_id, inp.subject, detail=inp.kind)
        return
    due_at = _timers_of(key).get(inp.kind)
    _advance(s, key, cond, now, out)
    if due_at is None:
        out.trace(now, "timer_ignored", cond.id, key.subject, detail=inp.kind)
    elif now < due_at:
        # fired early (for example after a wall-clock step): ask again for the right time
        out.effects.append(ScheduleTimer(cond.id, key.subject, inp.kind, due_at))
    elif inp.kind == "recheck" and key.phase == "holding":
        out.effects.append(RequestRecheck(cond.id, key.subject, key.occurrence_id, key.gen, cond.version))
        key.next_recheck = now + cond.recheck_every_s
        out.effects.append(ScheduleTimer(cond.id, key.subject, "recheck", key.next_recheck))
    # any other due timer was applied by _advance


def _find_open(s: StreamDecideState, occurrence_id: str) -> Optional[Key]:
    for key in s.keys.values():
        if key.phase != "idle" and key.occurrence_id == occurrence_id:
            return key
    return None


def _close_externally(s, key, cond, cause, reason, now, out) -> None:
    """Resolve, false alarm, scene return: the world was declared changed. Bumps the gen."""
    if cond.kind == "state" and key.phase == "holding":
        _emit(s, key, cond, "cleared", cause, reason, 1.0, None, now, out)
    out.trace(now, "closed", cond.id, key.subject, occurrence_id=key.occurrence_id, detail=cause)
    _to_idle(key, out, now, bump_to=_bump(s))


def _on_resolve(s, occurrence_id, cause, reason, now, conditions, out) -> None:
    key = _find_open(s, occurrence_id)
    if key is not None:
        _sync_version(s, key.condition_id, now, conditions, out)   # may close or remove it
        key = _find_open(s, occurrence_id)
    if key is not None:
        _advance(s, key, conditions[key.condition_id], now, out)  # may close it
        key = _find_open(s, occurrence_id)
    if key is None:
        out.trace(now, "control_ignored", _condition_of(s, occurrence_id) or "?",
                  occurrence_id=occurrence_id, detail=f"{cause}: not open")
        return
    _close_externally(s, key, conditions[key.condition_id], cause, reason, now, out)


def _condition_of(s: StreamDecideState, occurrence_id: str) -> Optional[str]:
    prefix = s.stream_id + "/"
    if not occurrence_id.startswith(prefix):
        return None
    rest = occurrence_id[len(prefix):]
    if "/" not in rest:
        return None
    return rest.rsplit("/", 1)[0]


def _on_false_alarm(s, inp: FalseAlarm, now, conditions, out) -> None:
    cid = _condition_of(s, inp.occurrence_id)
    if cid is None or cid not in conditions:
        out.trace(now, "control_ignored", cid or "?", occurrence_id=inp.occurrence_id, detail="false_alarm: unknown")
        return
    out.effects.append(StoreCounterExample(cid, inp.occurrence_id, inp.note))
    reason = "marked a false alarm" + (f": {inp.note}" if inp.note else "")
    _on_resolve(s, inp.occurrence_id, "false_alarm", reason, now, conditions, out)


def _on_snooze(s, inp: Snooze, now, out) -> None:
    s.snoozes = {k: u for k, u in s.snoozes.items() if u > now and k != (inp.condition_id, inp.subject)}
    if inp.until > now:
        s.snoozes[(inp.condition_id, inp.subject)] = inp.until
        out.trace(now, "snoozed", inp.condition_id, inp.subject, detail=f"until {inp.until}")
    else:
        out.trace(now, "snooze_cancelled", inp.condition_id, inp.subject)


def _on_scene_return(s, inp: SceneReturn, now, conditions, out) -> None:
    _sync_version(s, inp.condition_id, now, conditions, out)
    cond = conditions.get(inp.condition_id)
    if cond is None:
        out.trace(now, "control_ignored", inp.condition_id, inp.subject, detail="scene_return: unknown")
        return
    for (cid, subject) in sorted(s.keys):
        if cid != cond.id or (inp.subject is not None and subject != inp.subject):
            continue
        key = s.keys[(cid, subject)]
        _advance(s, key, cond, now, out)
        if key.phase == "idle":
            continue
        if cond.kind == "event":
            out.trace(now, "scene_return_ignored", cid, subject, occurrence_id=key.occurrence_id)
            continue
        _close_externally(s, key, cond, "scene_return", "the scene returned to its baseline", now, out)


def _sync_version(s, condition_id, now, conditions, out) -> bool:
    """Bring the recorded version of one condition in line with ``conditions``.

    Returns True when something changed (added, updated or removed).
    """
    cond = conditions.get(condition_id)
    recorded = s.versions.get(condition_id)
    if cond is not None and recorded == cond.version:
        return False
    if cond is None and recorded is None:
        return False
    for (cid, subject) in sorted(s.keys):
        if cid != condition_id:
            continue
        key = s.keys[(cid, subject)]
        if cond is None:
            out.trace(now, "condition_removed", cid, subject, occurrence_id=key.occurrence_id)
            _to_idle(key, out, now)
            del s.keys[(cid, subject)]
            continue
        if key.phase == "holding":
            _emit(s, key, cond, "cleared", "condition_changed", "the condition was changed", 1.0, None, now, out)
        out.trace(now, "condition_changed", cid, subject, occurrence_id=key.occurrence_id,
                  detail=f"version {recorded} -> {cond.version}")
        _to_idle(key, out, now, bump_to=_bump(s))
    if cond is None:
        s.versions.pop(condition_id, None)
        s.snoozes = {k: u for k, u in s.snoozes.items() if k[0] != condition_id}
    else:
        s.versions[condition_id] = cond.version
    return True


def _on_epoch(s, inp: EpochStarted, now, conditions, out) -> None:
    if inp.stream_epoch <= s.stream_epoch:
        out.trace(now, "epoch_ignored", "*", detail=f"{inp.stream_epoch} <= {s.stream_epoch}")
        return
    s.stream_epoch = inp.stream_epoch
    out.trace(now, "epoch_started", "*", detail=f"{inp.stream_epoch} restart={inp.restart}")
    for cid in sorted({c for c, _ in s.keys}):
        _sync_version(s, cid, now, conditions, out)
    for (cid, subject) in sorted(s.keys):
        key = s.keys[(cid, subject)]
        cond = conditions[cid]
        _advance(s, key, cond, now, out)
        if key.phase == "verifying" and cond.verify_mode == "fresh":
            # a fresh frame from the new epoch is as good as one from the old
            out.effects.append(RequestVerify(
                condition_id=cid, subject=subject, occurrence_id=key.occurrence_id, occurrence_gen=key.gen,
                condition_version=cond.version, mode="fresh", interval=None,
                stream_epoch=s.stream_epoch, deadline=key.verify_deadline))
        elif inp.restart and key.phase in ("candidate", "verifying"):
            # the original evidence (and any request in flight) died with the process
            _unverified(s, key, cond, "restart", now, out, verify_answered=False)
        if key.phase == "holding":
            out.effects.append(RequestRecheck(cid, subject, key.occurrence_id, key.gen, cond.version))
            key.next_recheck = now + cond.recheck_every_s
        for kind, at in sorted(_timers_of(key).items()):
            out.effects.append(ScheduleTimer(cid, subject, kind, at))
