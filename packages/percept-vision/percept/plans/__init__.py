"""Watch plans: a whole watch (sources, budget, judges, concerns) as one plain-data document.

Experimental: the schema may change before it is stable.

:func:`compile` checks a plan (a ``dict``, loaded from JSON or YAML by the caller) against
``SCHEMA_VERSION`` and turns it into the arguments of a :class:`~percept.Monitor`::

    compiled = percept.plans.compile(plan, {"fast": percept.judges.gemini(...)})
    monitor = percept.Monitor(**compiled.monitor_kwargs, store=percept.SqliteStore("percept.db"))

Each concern becomes one :class:`~percept.contracts.Condition`. :class:`Compiled` also says what
the plan will do: :meth:`~Compiled.explain` lists every effective setting and where it came from
(the plan, an SDK default, or a value the runtime fixes), :meth:`~Compiled.feasibility` checks the
periodic demand against the request budget, and :meth:`~Compiled.provenance` names the SDK, the
judges and the prompt a run used. ``plan_hash`` covers every effective setting, so two plans with
the same hash watch the same way; ``metadata`` is free-form and not hashed.

A concern whose attention is only ``external_event`` entries is judged only when the app calls
``monitor.poke(stream_id, condition_id)``; ``Compiled.events`` maps each event name to the
condition ids to poke.

Pure: no I/O, no clock. Any malformed plan raises :class:`PlanError` naming the field.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field, fields
from typing import Any, Mapping

from percept.contracts import Condition, Due, Frame, Judge
from percept.judges._cascade import cascade
from percept.judges.base import EVENT_RULE, STATE_RULE, SYSTEM, VERIFY_NOTE, max_attempts
from percept.replay import TRACE_FORMAT, encode_condition
from percept.triggers import Every, OnChange

__all__ = ["SCHEMA_VERSION", "Compiled", "PlanError", "compile"]

SCHEMA_VERSION = "percept.watch/v1alpha1"
MAX_SIDE = 640
# Runtime values a plan can't change (mirrors of percept.monitor and Monitor's defaults; a test
# pins them, so they can't drift apart).
EVENT_CLIP_S = 4.0
CLIP_FRAMES = 4
DEADLINES = {"event": 3.0, "state": 10.0}
BUDGET = {"max_requests_per_min": 6, "max_cost_per_hour": None, "max_concurrent_requests": 4}
DEFAULT_TRIGGERS = {"event": (OnChange(),), "state": (OnChange(), Every(300))}

# plan key -> (Condition field, value type)
_JUDGMENT = {"verification": ("verify", "str"), "min_confidence": ("min_confidence", "num"),
             "fast_path_confidence": ("fast_path_confidence", "num?"), "zones": ("zones", "polys"),
             "exclude_zones": ("exclude_zones", "polys"), "not_when": ("not_when", "strs")}
_DECISION = {"confirm": ("confirm", "int"), "confirm_window_s": ("confirm_window_s", "num"),
             "rearm_s": ("rearm_s", "num"), "recheck_every_s": ("recheck_every_s", "num"),
             "clear_after_no": ("clear_after_no", "int"), "max_open_s": ("max_open_s", "num?"),
             "cooldown_s": ("cooldown_s", "num"), "emit_cleared": ("emit_cleared", "bool"),
             "on_unverified": ("on_unverified", "str"), "candidate_deadline_s": ("candidate_deadline_s", "num")}


class PlanError(ValueError):
    """A plan that can't be compiled. The message starts with the dotted path of the field."""


def _fail(path: str, message: str) -> None:
    raise PlanError(f"{path or 'plan'}: {message}")


def _at(path: str, key: str) -> str:
    return f"{path}.{key}" if path else key


def _obj(value: Any, path: str, required: tuple[str, ...], optional: tuple[str, ...]) -> dict:
    if not isinstance(value, dict):
        _fail(path, f"expected an object, got {type(value).__name__}")
    for key in value:
        if key not in required and key not in optional:
            _fail(_at(path, str(key)), "unknown field")
    for key in required:
        if key not in value:
            _fail(_at(path, key), "required field is missing")
    return value


def _list(value: Any, path: str, nonempty: bool = False) -> list:
    if not isinstance(value, list):
        _fail(path, f"expected a list, got {type(value).__name__}")
    if nonempty and not value:
        _fail(path, "must not be empty")
    return value


def _str(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        _fail(path, "expected a non-empty string")
    return value


def _int(value: Any, path: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        _fail(path, f"expected an integer >= {minimum}")
    return value


def _num(value: Any, path: str) -> float:
    try:
        ok = isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
        number = float(value) if ok else 0.0
    except OverflowError:
        ok = False
    if not ok:
        _fail(path, "expected a finite number")
    return number


def _value(kind: str, value: Any, path: str) -> Any:
    """A plan value of type ``kind`` as the Condition field takes it."""
    if kind.endswith("?"):
        return None if value is None else _value(kind[:-1], value, path)
    if kind == "num":
        return _num(value, path)
    if kind == "int":
        return _int(value, path)
    if kind == "bool":
        if not isinstance(value, bool):
            _fail(path, "expected true or false")
        return value
    if kind == "str":
        return _str(value, path)
    if kind == "strs":
        return tuple(_str(s, f"{path}[{i}]") for i, s in enumerate(_list(value, path)))
    polys = []          # polys: a list of polygons, each a list of [x, y] points
    for i, poly in enumerate(_list(value, path)):
        points = []
        for j, point in enumerate(_list(poly, f"{path}[{i}]")):
            where = f"{path}[{i}][{j}]"
            if not isinstance(point, list) or len(point) != 2:
                _fail(where, "a point is an [x, y] pair")
            points.append((_num(point[0], where), _num(point[1], where)))
        polys.append(tuple(points))
    return tuple(polys)


# --------------------------------------------------------------------------- triggers


@dataclass(frozen=True)
class _External:
    """The trigger of a concern judged only on pokes: never due on its own."""

    def step(self, frame: Frame, conditions: tuple[Condition, ...], state: Any, now: float) -> tuple[Due, Any]:
        return {}, state

    def __repr__(self) -> str:
        return "external_event()"


def _describe(trigger: Any) -> dict:
    if isinstance(trigger, OnChange):
        return {"type": "on_change", "threshold": trigger.threshold, "exit_threshold": trigger.exit_threshold,
                "baseline_s": trigger.baseline_s, "grid": list(trigger.grid)}
    if isinstance(trigger, Every):
        return {"type": "every", "seconds": trigger.seconds}
    if isinstance(trigger, _External):
        return {"type": "external_event_only"}
    return {"type": "sdk_default"}


def _attention(value: Any, path: str) -> tuple[Any, tuple[str, ...]]:
    """(trigger, external event names) of one concern's ``attention``."""
    entries = _list(_obj(value, path, ("any",), ())["any"], _at(path, "any"), nonempty=True)
    triggers, events = [], []
    for i, entry in enumerate(entries):
        where = f"{path}.any[{i}]"
        if not isinstance(entry, dict) or len(entry) != 1:
            _fail(where, "expected an object with one of on_change, every, external_event")
        (key, arg), = entry.items()
        _obj(entry, where, (), ("on_change", "every", "external_event"))
        where = _at(where, key)
        try:
            if key == "external_event":
                events.append(_str(arg, where))
            elif key == "every":
                triggers.append(Every(_num(arg, where)))
            else:
                args = _obj(arg, where, (), ("threshold", "exit_threshold", "baseline_s"))
                kinds = {"threshold": "num", "exit_threshold": "num?", "baseline_s": "num"}
                triggers.append(OnChange(**{k: _value(kinds[k], v, _at(where, k)) for k, v in args.items()}))
        except PlanError:
            raise
        except ValueError as exc:
            _fail(where, str(exc))
    if len(triggers) > 1:
        _fail(_at(path, "any"), "at most one on_change or every trigger per concern; "
                                "combined triggers are not supported yet")
    trigger = triggers[0] if triggers else _External()
    return trigger, tuple(dict.fromkeys(events))


# --------------------------------------------------------------------------- compiled


@dataclass(frozen=True, eq=False)
class Compiled:
    """A compiled plan. ``Monitor(**monitor_kwargs, store=..., trace=...)`` runs it."""

    conditions: tuple[Condition, ...]
    monitor_kwargs: dict
    events: dict[str, tuple[str, ...]]
    plan_hash: str
    sources: tuple[dict, ...]
    plan_id: str
    revision: int
    _info: dict = field(repr=False)

    @property
    def _judge(self) -> Judge:
        return self.monitor_kwargs["judge"]

    def _verifier(self) -> str:
        return self._judge.members[-1].name if self._info["cascade"] else self._judge.name  # type: ignore[attr-defined]

    def explain(self) -> dict:
        """Every effective setting, each with its origin: ``plan``, ``sdk_default`` or ``sdk_fixed``."""
        info = self._info
        concerns = []
        for cond in self.conditions:
            c, encoded = info["concerns"][cond.id], encode_condition(cond)
            report = {f.name: {"value": encoded[f.name], "origin": "plan" if f.name in c["given"] else "sdk_default"}
                      for f in fields(Condition) if f.name not in ("trigger", "version")}
            triggers = DEFAULT_TRIGGERS[cond.kind] if cond.trigger is None else (cond.trigger,)
            if cond.kind == "event":
                evidence = {"clip_s": EVENT_CLIP_S, "frames": CLIP_FRAMES, "ends": "at the trigger"}
            else:
                evidence = {"frames": 1, "which": "the newest frame"}
            verify_mode = cond.verify_mode
            concerns.append({
                "id": cond.id, "source": c["source"], "route": c["route"], "judge": c["judge"],
                "fields": report,
                "trigger": {"value": [_describe(t) for t in triggers],
                            "origin": "sdk_default" if cond.trigger is None else "plan"},
                "external_events": list(c["events"]),
                "version": {"value": cond.version, "origin": "plan"},
                "verify_mode": {"value": verify_mode, "origin": "plan" if "verify" in c["given"] else "sdk_default"},
                "evidence": {"value": evidence, "origin": "sdk_fixed"},
                "detection_deadline_s": {"value": DEADLINES[cond.kind], "origin": "sdk_fixed"},
                "verification_judge": {"value": None if verify_mode == "off" else self._verifier(),
                                       "origin": "sdk_fixed"},
            })
        notes = []
        if info["cascade"]:
            notes.append(f"detection asks each concern's route; every verification request goes to the last "
                         f"judge in judges.order ({self._verifier()})")
        return {
            "schema_version": SCHEMA_VERSION, "plan_id": self.plan_id, "revision": self.revision,
            "plan_hash": self.plan_hash, "judge": self._judge.name, "judges": [j["name"] for j in info["judges"]],
            "budget": {k: {"value": self.monitor_kwargs[k], "origin": "plan" if k in info["budget_given"]
                           else "sdk_default"} for k in BUDGET},
            "sources": [dict(s) for s in self.sources], "concerns": concerns,
            "events": {k: list(v) for k, v in self.events.items()}, "notes": notes,
        }

    def feasibility(self) -> dict:
        """A static check of each source's periodic demand against the request budget.

        Concerns share one request only when the judge batches and they get identical evidence
        (same period, kind and zones). Change-triggered and poked demand depends on the footage
        and is not counted.
        """
        judge, info = self._judge, self._info
        attempts = max_attempts(judge)
        ceiling = self.monitor_kwargs["max_requests_per_min"]
        out: dict[str, dict] = {}
        rank = {"feasible": 0, "best_effort": 1, "infeasible": 2}
        for source in self.sources:
            sid = source["id"]
            groups: dict[Any, list[str]] = {}
            periods: dict[Any, float] = {}
            unknown = verifies = False
            for cond in self.conditions:
                c = info["concerns"][cond.id]
                if c["source"] != sid:
                    continue
                triggers = DEFAULT_TRIGGERS[cond.kind] if cond.trigger is None else (cond.trigger,)
                unknown = unknown or cond.trigger is None or bool(c["events"]) \
                    or any(isinstance(t, (OnChange, _External)) for t in triggers)
                verifies = verifies or (cond.kind == "event" and cond.verify_mode != "off")
                for t in triggers:
                    if isinstance(t, Every):
                        key = ((t.seconds, cond.kind, cond.zones, cond.exclude_zones) if judge.can_batch
                               else ("one", cond.id, t.seconds))
                        groups.setdefault(key, []).append(cond.id)
                        periods[key] = t.seconds
            requests = sum(60.0 / periods[k] for k in groups)
            demand = requests * attempts
            notes = [f"the budget is per stream: at most {ceiling} attempts a minute, retries included; "
                     f"one request may make up to {attempts}"]
            if unknown:
                notes.append("change-triggered demand is unknown until measured on representative video")
            if verifies:
                notes.append("every event candidate costs one more request to verify it")
            verdict = "infeasible" if demand > ceiling else ("best_effort" if unknown else "feasible")
            out[sid] = {"verdict": verdict, "ceiling_requests_per_min": ceiling, "max_attempts": attempts,
                        "periodic_requests_per_min": round(requests, 6),
                        "periodic_attempts_per_min": round(demand, 6),
                        "groups": [{"every_s": periods[k], "requests_per_min": round(60.0 / periods[k], 6),
                                    "conditions": ids} for k, ids in groups.items()],
                        "notes": notes}
        worst = max((s["verdict"] for s in out.values()), key=rank.__getitem__, default="feasible")
        return {"verdict": worst, "sources": out}

    def provenance(self) -> dict:
        """What a run used: SDK version, plan identity and hash, judges and prompt hash. No secrets."""
        from percept import __version__
        return {"sdk_version": __version__, "schema_version": SCHEMA_VERSION, "plan_id": self.plan_id,
                "revision": self.revision, "plan_hash": self.plan_hash,
                "judges": [dict(j) for j in self._info["judges"]],
                "prompt_sha256": hashlib.sha256(json.dumps([SYSTEM, EVENT_RULE, STATE_RULE, VERIFY_NOTE])
                                                .encode()).hexdigest(),
                "sources": [dict(s) for s in self.sources]}

    def trace_record(self, t_wall: float, stream_id: str | None = None) -> dict:
        """The provenance as a trace record, for ``TraceWriter.write``. ``t_wall`` is the caller's
        wall time; replay ignores ``plan`` records."""
        return {"v": TRACE_FORMAT, "t_wall": t_wall, "stream_id": stream_id, "kind": "plan", **self.provenance()}


# --------------------------------------------------------------------------- compile


def _judge_info(judge: Any, routes: Mapping[str, Any]) -> dict:
    price = judge.price
    model = getattr(judge, "model", None)
    return {"routes": sorted(r for r, j in routes.items() if j is judge), "name": judge.name,
            "model": model if isinstance(model, str) else None,
            "price": None if price is None else {f.name: getattr(price, f.name) for f in fields(price)}}


def _dedupe(judges: list) -> list:
    out: list = []
    for j in judges:
        if not any(j is k for k in out):
            out.append(j)
    return out


def compile(plan: Any, routes: Mapping[str, Judge]) -> Compiled:  # noqa: A001 - the verb is the API
    """Check ``plan`` and compile it; ``routes`` maps the plan's route names to judges."""
    if not isinstance(routes, Mapping) or not routes:
        _fail("routes", "expected a non-empty mapping of route name to judge")
    for name, judge in routes.items():
        if not isinstance(name, str) or not isinstance(judge, Judge):
            _fail(f"routes[{name!r}]", "expected a route name mapped to a Judge")
    _obj(plan, "", ("schema_version", "plan_id", "revision", "sources", "concerns"),
         ("budget", "judges", "metadata"))
    if plan["schema_version"] != SCHEMA_VERSION:
        _fail("schema_version", f"unsupported {plan['schema_version']!r}; this SDK reads {SCHEMA_VERSION!r}")
    plan_id = _str(plan["plan_id"], "plan_id")
    revision = _int(plan["revision"], "revision", 1)
    if "metadata" in plan and not isinstance(plan["metadata"], dict):
        _fail("metadata", "expected an object")

    sources: dict[str, dict] = {}
    for i, raw in enumerate(_list(plan["sources"], "sources", nonempty=True)):
        path = f"sources[{i}]"
        _obj(raw, path, ("id", "kind"), ("max_side",))
        sid = _str(raw["id"], f"{path}.id")
        if sid in sources:
            _fail(f"{path}.id", f"duplicate source id {sid!r}")
        if raw["kind"] != "video":
            _fail(f"{path}.kind", "only 'video' sources are supported")
        sources[sid] = {"id": sid, "kind": "video", "max_side": _int(raw.get("max_side", MAX_SIDE),
                                                                     f"{path}.max_side", 1)}

    concerns: dict[str, dict] = {}
    for i, raw in enumerate(_list(plan["concerns"], "concerns", nonempty=True)):
        path = f"concerns[{i}]"
        _obj(raw, path, ("id", "source", "description", "judgment", "decision"), ("attention",))
        cid = _str(raw["id"], f"{path}.id")
        if cid in concerns:
            _fail(f"{path}.id", f"duplicate concern id {cid!r}")
        sid = _str(raw["source"], f"{path}.source")
        if sid not in sources:
            _fail(f"{path}.source", f"no source with id {sid!r}")
        judgment = _obj(raw["judgment"], f"{path}.judgment", ("route",), (*_JUDGMENT, "evidence"))
        decision = _obj(raw["decision"], f"{path}.decision", ("kind",), tuple(_DECISION))
        route = _str(judgment["route"], f"{path}.judgment.route")
        if route not in routes:
            _fail(f"{path}.judgment.route", f"unknown route {route!r}; routes are {sorted(routes)}")
        kind = decision["kind"]
        if kind not in ("event", "state"):
            _fail(f"{path}.decision.kind", "expected 'event' or 'state'")
        kwargs: dict[str, Any] = {"id": cid, "text": _str(raw["description"], f"{path}.description"),
                                  "kind": kind, "version": revision}
        for section, table in (("judgment", _JUDGMENT), ("decision", _DECISION)):
            for key, value in raw[section].items():
                if key in table:
                    name, typ = table[key]
                    kwargs[name] = _value(typ, value, f"{path}.{section}.{key}")
        if "evidence" in judgment:
            where = f"{path}.judgment.evidence"
            evidence = _obj(judgment["evidence"], where, (), ("pre_seconds", "post_seconds", "max_side"))
            if "post_seconds" in evidence:
                _fail(f"{where}.post_seconds", "not supported: evidence ends at the trigger")
            if "max_side" in evidence and evidence["max_side"] != sources[sid]["max_side"]:
                _fail(f"{where}.max_side", f"resolution is set per source: sources {sid!r} has max_side "
                                           f"{sources[sid]['max_side']}")
            if "pre_seconds" in evidence:
                if kind == "state":
                    _fail(f"{where}.pre_seconds", "a state is judged on the newest frame, not a clip")
                if _num(evidence["pre_seconds"], f"{where}.pre_seconds") != EVENT_CLIP_S:
                    _fail(f"{where}.pre_seconds", f"must be {EVENT_CLIP_S}: per-condition evidence windows "
                                                  "are not supported yet")
        events: tuple[str, ...] = ()
        if "attention" in raw:
            kwargs["trigger"], events = _attention(raw["attention"], f"{path}.attention")
        concerns[cid] = {"kwargs": kwargs, "route": route, "source": sid, "events": events,
                         "given": set(kwargs) - {"id", "text", "version"} | {"id", "text", "kind"}, "path": path}

    # Judges: one distinct judge is used as is; more need an order and become a cascade.
    used = _dedupe([routes[c["route"]] for c in concerns.values()])
    order: list = []
    if "judges" in plan:
        names = _list(_obj(plan["judges"], "judges", ("order",), ())["order"], "judges.order", nonempty=True)
        for k, name in enumerate(names):
            if not isinstance(name, str) or name not in routes:
                _fail(f"judges.order[{k}]", f"unknown route {name!r}")
            order.append(routes[name])
        order = _dedupe(order)
        for judge in used:
            if not any(judge is j for j in order):
                _fail("judges.order", f"judge {judge.name!r} is used by a concern but not in the order")
    members = order or used
    is_cascade = len(members) > 1
    if is_cascade and not order:
        _fail("judges.order", "routes map to more than one judge: list them in judges.order, cheapest "
                              "first; verification uses the last")
    if is_cascade:
        try:
            judge: Any = cascade(*members)
        except ValueError as exc:
            raise PlanError(f"judges.order: {exc}") from None
    else:
        judge = members[0]

    budget = dict(BUDGET)
    budget_given: tuple[str, ...] = ()
    if "budget" in plan:
        raw = _obj(plan["budget"], "budget", (), tuple(BUDGET))
        budget_given = tuple(raw)
        for key in ("max_requests_per_min", "max_concurrent_requests"):
            if key in raw:
                budget[key] = _int(raw[key], f"budget.{key}", 1)
        if raw.get("max_cost_per_hour") is not None:
            cost = _num(raw["max_cost_per_hour"], "budget.max_cost_per_hour")
            if cost <= 0:
                _fail("budget.max_cost_per_hour", "must be positive, or null for no cap")
            budget["max_cost_per_hour"] = cost
    if budget["max_requests_per_min"] < max_attempts(judge):
        _fail("budget.max_requests_per_min", f"{budget['max_requests_per_min']} is below the "
              f"{max_attempts(judge)} attempts one judge call may make")

    conditions = []
    event_map: dict[str, list[str]] = {}
    for cid, c in concerns.items():
        c["judge"] = routes[c["route"]].name
        kwargs = dict(c["kwargs"], judge=c["judge"] if is_cascade else None)
        try:
            conditions.append(Condition(**kwargs))
        except (ValueError, TypeError) as exc:
            raise PlanError(f"{c['path']}: {exc}") from None
        for name in c["events"]:
            event_map.setdefault(name, []).append(cid)
    events = {k: tuple(v) for k, v in event_map.items()}

    judges_info = [_judge_info(j, routes) for j in members]
    effective = {
        "schema_version": SCHEMA_VERSION, "plan_id": plan_id, "revision": revision,
        "sources": list(sources.values()), "budget": budget, "judges": [j["name"] for j in judges_info],
        "conditions": [{**encode_condition(cond), "trigger": _describe(cond.trigger)} for cond in conditions],
        "events": {k: list(v) for k, v in events.items()},
    }
    canonical = json.dumps(effective, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    info = {"concerns": concerns, "judges": judges_info, "cascade": is_cascade, "budget_given": budget_given}
    return Compiled(
        conditions=tuple(conditions),
        monitor_kwargs={"judge": judge, "conditions": tuple(conditions), **budget},
        events=events,
        plan_hash=hashlib.sha256(canonical.encode()).hexdigest(),
        sources=tuple(sources.values()),
        plan_id=plan_id, revision=revision, _info=info,
    )
