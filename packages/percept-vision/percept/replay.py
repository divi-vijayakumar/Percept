"""Deterministic replay: feed a recorded trace back through the decision reducer.

A trace (see ``percept.trace``) records every input the runtime gave :func:`percept.decide.step`,
with the wall-clock ``now`` it was given and the condition set it was decided under, plus the
decide state each stream started from (a ``restore`` record). Because the reducer is pure, the
same inputs from the same state give the same decisions. :func:`replay` re-runs them and
compares the result with the ``decision`` records the runtime wrote.

This module also holds the plain-JSON encoding of decide inputs and conditions, so the writer and
the reader of a trace share one definition. Pure: no I/O, no clock.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Iterable, Mapping

from percept import decide
from percept.contracts import Condition, Decision, Observation, Usage

TRACE_FORMAT = 1

_INPUTS: Mapping[str, type] = {cls.__name__: cls for cls in (
    decide.Observed, decide.TimerFired, decide.Resolve, decide.FalseAlarm, decide.Snooze,
    decide.SceneReturn, decide.ConditionChanged, decide.EpochStarted)}
_SKIP_CONDITION_FIELDS = ("trigger",)       # runtime-only: decide never reads it


# --------------------------------------------------------------------------- encoding


def _plain(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_plain(v) for v in value]
    return value


def encode_input(inp: decide.Input) -> dict:
    """A decide input as plain JSON: ``{"type": <class name>, <fields>}``."""
    if type(inp) is decide.Observed:
        o = inp.observation
        return {"type": "Observed",
                "observation": {f.name: _plain(getattr(o, f.name)) for f in fields(Observation)},
                "usage": {f.name: getattr(inp.usage, f.name) for f in fields(Usage)}}
    if type(inp).__name__ not in _INPUTS:
        raise TypeError(f"unknown decide input {inp!r}")
    return {"type": type(inp).__name__, **{f.name: _plain(getattr(inp, f.name)) for f in fields(inp)}}


def decode_input(data: Mapping[str, Any]) -> decide.Input:
    """The inverse of :func:`encode_input`. Raises ``ValueError`` for anything else."""
    try:
        kind = data["type"]
        if kind == "Observed":
            o = dict(data["observation"])
            o["evidence_t"] = tuple(o["evidence_t"])
            return decide.Observed(Observation(**o), Usage(**data["usage"]))
        cls = _INPUTS[kind]
        return cls(**{k: v for k, v in data.items() if k != "type"})
    except (KeyError, TypeError) as exc:
        raise ValueError(f"malformed decide input: {exc!r}") from exc


def encode_condition(cond: Condition) -> dict:
    """A condition as plain JSON, without its trigger (the reducer never reads it)."""
    return {f.name: _plain(getattr(cond, f.name)) for f in fields(Condition)
            if f.name not in _SKIP_CONDITION_FIELDS}


def decode_condition(data: Mapping[str, Any]) -> Condition:
    raw = dict(data)
    for name in ("zones", "exclude_zones"):
        raw[name] = tuple(tuple(tuple(p) for p in poly) for poly in raw.get(name, ()))
    raw["not_when"] = tuple(raw.get("not_when", ()))
    return Condition(**raw)


def decision_summary(d: Decision) -> dict:
    """The fields of a decision that replay compares (plus when it was decided)."""
    return {"decision_id": d.decision_id, "occurrence_id": d.occurrence_id, "type": d.type,
            "cause": d.cause, "condition_id": d.condition_id, "subject": d.subject,
            "t_decided": d.t_decided}


_COMPARED = ("decision_id", "occurrence_id", "type", "cause", "condition_id", "subject")


# --------------------------------------------------------------------------- replay


@dataclass(frozen=True)
class ReplayResult:
    """What a replay produced. ``matches`` is True when, for every stream, the replayed
    decisions equal the recorded ones (ids, types, causes, conditions, subjects) in order;
    otherwise ``differences`` says where they first part, one line per problem."""

    decisions: tuple[Decision, ...]
    matches: bool
    differences: tuple[str, ...]
    inputs: int = 0


class _Replay:
    def __init__(self) -> None:
        self.states: dict[str, decide.StreamDecideState] = {}
        self.conditions: dict[str, dict[str, Condition]] = {}
        self.recorded: dict[str, list[dict]] = {}
        self.produced: dict[str, list[Decision]] = {}
        self.order: list[str] = []
        self.decisions: list[Decision] = []
        self.problems: list[str] = []
        self.inputs = 0
        self.skipped: dict[str, int] = {}

    def stream(self, sid: str) -> None:
        if sid not in self.recorded:
            self.order.append(sid)
            self.recorded[sid], self.produced[sid] = [], []

    def feed(self, n: int, rec: Mapping[str, Any]) -> None:
        kind, sid = rec.get("kind"), rec.get("stream_id")
        if rec.get("v") != TRACE_FORMAT:
            self.problems.append(f"record {n}: unsupported trace format {rec.get('v')!r}")
            return
        if kind not in ("restore", "conditions", "decide_input", "decision"):
            return
        self.stream(sid)
        if kind == "restore":
            self.states[sid] = decide.from_dict(rec["snapshot"])
        elif kind == "conditions":
            self.conditions[sid] = {c["id"]: decode_condition(c) for c in rec["conditions"]}
        elif kind == "decision":
            self.recorded[sid].append(dict(rec))
        else:
            self.step(n, sid, rec)

    def step(self, n: int, sid: str, rec: Mapping[str, Any]) -> None:
        state = self.states.get(sid)
        if state is None:
            self.skipped[sid] = self.skipped.get(sid, 0) + 1
            return
        conds = self.conditions.get(sid, {})
        versions = {k: int(v) for k, v in rec.get("versions", {}).items()}
        if versions != {c.id: c.version for c in conds.values()}:
            self.problems.append(f"record {n}: stream {sid!r}: input decided under condition versions "
                                 f"{versions}, but the trace defines {sorted((c.id, c.version) for c in conds.values())}")
        result = decide.step(state, decode_input(rec["input"]), float(rec["now"]), conds)
        self.inputs += 1
        self.states[sid] = result.state
        self.produced[sid].extend(result.decisions)
        self.decisions.extend(result.decisions)

    def compare(self) -> list[str]:
        out = list(self.problems)
        for sid in self.order:
            if self.skipped.get(sid):
                out.append(f"stream {sid!r}: {self.skipped[sid]} inputs before any restore record were not replayed")
            got = [{k: decision_summary(d)[k] for k in _COMPARED} for d in self.produced[sid]]
            want = [{k: r.get(k) for k in _COMPARED} for r in self.recorded[sid]]
            for i, (g, w) in enumerate(zip(got, want)):
                if g != w:
                    diff = {k: (w[k], g[k]) for k in _COMPARED if w[k] != g[k]}
                    out.append(f"stream {sid!r}: decision #{i + 1} differs (recorded, replayed): {diff}")
                    break
            if len(got) != len(want):
                extra = got[len(want):] if len(got) > len(want) else want[len(got):]
                which = "replay produced" if len(got) > len(want) else "the trace recorded"
                out.append(f"stream {sid!r}: {len(want)} decisions recorded, {len(got)} replayed; "
                           f"{which} extra: {[(e['decision_id'], e['type']) for e in extra[:3]]}")
        return out


def replay(records: Iterable[Mapping[str, Any]]) -> ReplayResult:
    """Re-run the recorded decide inputs, stream by stream, and compare the decisions.

    Each stream starts from its ``restore`` record (the snapshot the runtime restored, or the
    fresh state it created); a later ``restore`` record replaces the state, as a restart did.
    Records of kinds other than ``restore``, ``conditions``, ``decide_input`` and ``decision``
    are ignored. A trace whose oldest records were rotated away may lack a stream's ``restore``
    record: its inputs are then skipped and reported in ``differences``.
    """
    r = _Replay()
    for n, rec in enumerate(records, 1):
        try:
            r.feed(n, rec)
        except (KeyError, TypeError, ValueError) as exc:
            r.problems.append(f"record {n}: unreadable {rec.get('kind')!r} record: {exc!r:.200}")
    differences = tuple(r.compare())
    return ReplayResult(tuple(r.decisions), not differences, differences, r.inputs)

