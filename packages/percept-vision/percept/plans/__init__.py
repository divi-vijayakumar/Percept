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
judges and the prompt a run used. ``plan_hash`` covers every effective setting, the runtime's fixed
values and the judges' names, whatever the order of keys, concerns, sources or events, so two
plans with the same hash watch the same way with judges of the same names; ``metadata`` is
free-form and not hashed.

Nothing in ``monitor_kwargs`` sets a source's resolution: open each source with
``max_side=compiled.sources[i]["max_side"]`` yourself.

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

from percept.contracts import MAX_CLIP_FRAMES, Condition, Due, Frame, Judge
from percept.judges._cascade import cascade
from percept.judges.base import EVENT_RULE, STATE_RULE, SYSTEM, VERIFY_NOTE, max_attempts
from percept.replay import TRACE_FORMAT, encode_condition
from percept.plans.labels import LabelError, Labels, SplitManifest, load_labels, split
from percept.triggers import Every, OnChange

__all__ = ["SCHEMA_VERSION", "Compiled", "PlanError", "compile",
           # evaluation (experimental): labels, frozen splits
           "LabelError", "Labels", "SplitManifest", "load_labels", "split"]

SCHEMA_VERSION = "percept.watch/v1alpha1"
MAX_SIDE = 640
MAX_COUNT = 10**6           # largest integer a plan may give (counts, sizes)
MAX_SECONDS = 10**7         # longest duration a plan may give
# Runtime values a plan can't change (mirrors of percept.monitor and Monitor's defaults; a test
# pins them, so they can't drift apart).
EVENT_CLIP_S = 4.0          # an event's evidence window unless judgment.evidence sets it
CLIP_FRAMES = 4
PREROLL_S = 8.0             # the SDK's default pre-roll: the longest window a plan may ask for
PREROLL_FPS = 2.0           # ... sampled at this rate: a window holds at most floor(s * fps) + 2 frames
DEADLINES = {"event": 3.0, "state": 10.0}
BUDGET = {"max_requests_per_min": 6, "max_cost_per_hour": None, "max_concurrent_requests": 4}
DEFAULT_TRIGGERS = {"event": (OnChange(),), "state": (OnChange(), Every(300))}

# plan key -> (Condition field, value type)
_JUDGMENT = {"verification": ("verify", "str"), "min_confidence": ("min_confidence", "num"),
             "fast_path_confidence": ("fast_path_confidence", "num?"), "zones": ("zones", "polys"),
             "exclude_zones": ("exclude_zones", "polys"), "not_when": ("not_when", "strs")}
_DECISION = {"confirm": ("confirm", "int"), "confirm_window_s": ("confirm_window_s", "sec"),
             "rearm_s": ("rearm_s", "sec"), "recheck_every_s": ("recheck_every_s", "sec"),
             "clear_after_no": ("clear_after_no", "int"), "max_open_s": ("max_open_s", "sec?"),
             "cooldown_s": ("cooldown_s", "sec"), "emit_cleared": ("emit_cleared", "bool"),
             "on_unverified": ("on_unverified", "str"), "candidate_deadline_s": ("candidate_deadline_s", "sec")}


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


def _int(value: Any, path: str, minimum: int = 0, maximum: int | None = MAX_COUNT) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        _fail(path, f"expected an integer >= {minimum}")
    if maximum is not None and value > maximum:
        _fail(path, f"expected an integer <= {maximum}")
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
    if kind == "sec":
        seconds = _num(value, path)
        if abs(seconds) > MAX_SECONDS:
            _fail(path, f"expected at most {MAX_SECONDS} seconds")
        return seconds
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


def _bbox(zones: tuple) -> tuple[float, ...] | None:
    """The crop a judge sees for ``zones``: their bounding box (None: the whole frame)."""
    if not zones:
        return None
    xs, ys = [p[0] for poly in zones for p in poly], [p[1] for poly in zones for p in poly]
    return min(xs), min(ys), max(xs), max(ys)


def _describe(trigger: Any) -> dict:
    if isinstance(trigger, OnChange):
        return {"type": "on_change", "threshold": trigger.threshold, "exit_threshold": trigger.exit_threshold,
                "baseline_s": trigger.baseline_s, "grid": list(trigger.grid)}
    if isinstance(trigger, Every):
        return {"type": "every", "seconds": trigger.seconds}
    if isinstance(trigger, _External):
        return {"type": "external_event_only"}
    return {"type": "sdk_default"}


def _members(cond: Condition) -> tuple:
    """The triggers a condition is due on: its own, each member of a combined one, or the defaults."""
    if cond.trigger is None:
        return DEFAULT_TRIGGERS[cond.kind]
    return cond.trigger if isinstance(cond.trigger, tuple) else (cond.trigger,)


def _key(trigger: Any) -> str:
    return json.dumps(_describe(trigger), sort_keys=True, separators=(",", ":"))


def _describe_any(trigger: Any) -> Any:
    """A trigger's description; a combined trigger is the list of its members' (in canonical order)."""
    if isinstance(trigger, tuple):
        return [_describe(t) for t in sorted(trigger, key=_key)]
    return _describe(trigger)


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
                triggers.append(Every(_value("sec", arg, where)))
            else:
                args = _obj(arg, where, (), ("threshold", "exit_threshold", "baseline_s"))
                kinds = {"threshold": "num", "exit_threshold": "num?", "baseline_s": "sec"}
                triggers.append(OnChange(**{k: _value(kinds[k], v, _at(where, k)) for k, v in args.items()}))
        except PlanError:
            raise
        except ValueError as exc:
            _fail(where, str(exc))
    # several on_change/every entries combine: the concern is due when any of them fires. Members
    # are deduplicated and kept in a canonical order, so the entries' order changes nothing.
    unique = sorted({_key(t): t for t in triggers}.values(), key=_key)
    trigger = _External() if not unique else unique[0] if len(unique) == 1 else tuple(unique)
    return trigger, tuple(dict.fromkeys(events))


# --------------------------------------------------------------------------- compiled


@dataclass(frozen=True, eq=False)
class Compiled:
    """A compiled plan. ``Monitor(**monitor_kwargs, store=..., trace=...)`` runs it.

    ``sources`` holds each source's normalized settings. Open source ``i`` with
    ``max_side=sources[i]["max_side"]``: nothing in ``monitor_kwargs`` enforces it.
    """

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

    def _cost_notes(self) -> list[str]:
        if self.monitor_kwargs["max_cost_per_hour"] is not None and self._judge.price is None:
            return ["cost cap not enforced: judge has no price"]
        return []

    def explain(self) -> dict:
        """Every effective setting, each with its origin: ``plan``, ``sdk_default`` or ``sdk_fixed``."""
        info = self._info
        concerns = []
        for cond in self.conditions:
            c, encoded = info["concerns"][cond.id], encode_condition(cond)
            report = {f.name: {"value": encoded.get(f.name), "origin": "plan" if f.name in c["given"] else "sdk_default"}
                      for f in fields(Condition) if f.name not in ("trigger", "version")}
            triggers = _members(cond)
            if cond.kind == "event":
                evidence = {"clip_s": EVENT_CLIP_S if cond.clip_s is None else cond.clip_s,
                            "frames": CLIP_FRAMES if cond.clip_frames is None else cond.clip_frames,
                            "ends": "at the trigger"}
            else:
                evidence = {"frames": 1, "which": "the newest frame"}
            window_given = bool({"clip_s", "clip_frames"} & c["given"])
            verify_mode = cond.verify_mode
            concerns.append({
                "id": cond.id, "source": c["source"], "route": c["route"], "judge": c["judge"],
                "fields": report,
                "trigger": {"value": [_describe(t) for t in triggers],
                            "origin": "sdk_default" if cond.trigger is None else "plan"},
                "external_events": list(c["events"]),
                "version": {"value": cond.version, "origin": "plan"},
                "verify_mode": {"value": verify_mode, "origin": "plan" if "verify" in c["given"] else "sdk_default"},
                "evidence": {"value": evidence, "origin": "plan" if window_given
                             else "sdk_default" if cond.kind == "event" else "sdk_fixed"},
                "detection_deadline_s": {"value": DEADLINES[cond.kind], "origin": "sdk_fixed"},
                "verification_judge": {"value": None if verify_mode == "off" else self._verifier(),
                                       "origin": "sdk_fixed"},
            })
        notes = self._cost_notes()
        if info["cascade"]:
            notes.append(f"detection asks each concern's route; every verification request goes to the last "
                         f"judge in judges.order ({self._verifier()})")
        return {
            "schema_version": SCHEMA_VERSION, "plan_id": self.plan_id, "revision": self.revision,
            "plan_hash": self.plan_hash, "judge": self._judge.name, "judges": [j["name"] for j in info["judges"]],
            "budget": {k: {"value": self.monitor_kwargs[k], "origin": "plan" if k in info["budget_given"]
                           else "sdk_default"} for k in BUDGET},
            "sources": [{k: {"value": v, "origin": "plan" if k in info["sources_given"][s["id"]] else "sdk_default"}
                         for k, v in s.items()} for s in self.sources],
            "concerns": concerns,
            "events": {k: list(v) for k, v in self.events.items()}, "notes": notes,
        }

    def feasibility(self) -> dict:
        """A static check of each source's periodic demand against the request budget.

        Concerns share one request only when the judge batches and they get identical evidence
        (same period and kind, and the same crop: the bounding box of the include zones; exclude
        zones don't change the crop). Change-triggered and poked demand depends on the footage and
        is not counted, nor is verification: a source with verified events is at best
        ``best_effort``.
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
                triggers = _members(cond)       # each member of a combined trigger counts
                unknown = unknown or cond.trigger is None or bool(c["events"]) \
                    or any(isinstance(t, (OnChange, _External)) for t in triggers)
                verifies = verifies or (cond.kind == "event" and cond.verify_mode != "off")
                for t in triggers:
                    if isinstance(t, Every):
                        key = ((t.seconds, cond.kind, _bbox(cond.zones)) if judge.can_batch
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
                notes.append("every event candidate costs one more request to verify it, against the same "
                             "ceiling; the candidate rate is unknown until measured")
            notes += self._cost_notes()
            verdict = "infeasible" if demand > ceiling else ("best_effort" if unknown or verifies else "feasible")
            out[sid] = {"verdict": verdict, "ceiling_requests_per_min": ceiling, "max_attempts": attempts,
                        "periodic_requests_per_min": round(requests, 6),
                        "periodic_attempts_per_min": round(demand, 6),
                        "groups": [{"every_s": periods[k], "requests_per_min": round(60.0 / periods[k], 6),
                                    "conditions": ids} for k, ids in groups.items()],
                        "notes": notes}
        worst = max((s["verdict"] for s in out.values()), key=rank.__getitem__, default="feasible")
        return {"verdict": worst, "sources": out, "notes": self._cost_notes()}

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


def _effective(cond: Condition) -> dict:
    """A condition as hashed: an evidence window equal to the default is left out, so a plan that
    spells out the default hashes as one that doesn't."""
    data = encode_condition(cond)
    for name, default in (("clip_s", EVENT_CLIP_S), ("clip_frames", CLIP_FRAMES)):
        if data.get(name) == default:
            del data[name]
    return data


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
    revision = _int(plan["revision"], "revision", 1, None)
    if "metadata" in plan and not isinstance(plan["metadata"], dict):
        _fail("metadata", "expected an object")

    sources: dict[str, dict] = {}
    sources_given: dict[str, tuple[str, ...]] = {}
    for i, raw in enumerate(_list(plan["sources"], "sources", nonempty=True)):
        path = f"sources[{i}]"
        _obj(raw, path, ("id", "kind"), ("max_side",))
        sid = _str(raw["id"], f"{path}.id")
        if sid in sources:
            _fail(f"{path}.id", f"duplicate source id {sid!r}")
        if raw["kind"] != "video":
            _fail(f"{path}.kind", "only 'video' sources are supported")
        sources_given[sid] = ("id", "kind", *(("max_side",) if "max_side" in raw else ()))
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
            evidence = _obj(judgment["evidence"], where, (), ("pre_seconds", "frames", "post_seconds", "max_side"))
            if "post_seconds" in evidence:
                _fail(f"{where}.post_seconds", "not supported: evidence ends at the trigger")
            if "max_side" in evidence and evidence["max_side"] != sources[sid]["max_side"]:
                _fail(f"{where}.max_side", f"resolution is set per source: sources {sid!r} has max_side "
                                           f"{sources[sid]['max_side']}")
            for key in ("pre_seconds", "frames"):
                if key in evidence and kind == "state":
                    _fail(f"{where}.{key}", "a state is judged on the newest frame, not a clip")
            if "pre_seconds" in evidence:
                clip_s = _value("sec", evidence["pre_seconds"], f"{where}.pre_seconds")
                if not 0 < clip_s <= PREROLL_S:
                    _fail(f"{where}.pre_seconds", f"must be positive and at most the {PREROLL_S:g} s pre-roll "
                                                  "(a source with its own PreRoll needs one at least this long)")
                kwargs["clip_s"] = clip_s
            if "frames" in evidence:
                clip_frames = _int(evidence["frames"], f"{where}.frames", 1)
                if clip_frames > MAX_CLIP_FRAMES:
                    _fail(f"{where}.frames", f"expected an integer <= {MAX_CLIP_FRAMES}")
                kwargs["clip_frames"] = clip_frames
            window_s = kwargs.get("clip_s", EVENT_CLIP_S)
            reachable = math.floor(window_s * PREROLL_FPS) + 2
            if kwargs.get("clip_frames", CLIP_FRAMES) > reachable:
                _fail(f"{where}.frames", f"at most {reachable} frames fit {window_s:g} s of the default "
                                         f"{PREROLL_FPS:g} fps pre-roll")
        events: tuple[str, ...] = ()
        if "attention" in raw:
            kwargs["trigger"], events = _attention(raw["attention"], f"{path}.attention")
        concerns[cid] = {"kwargs": kwargs, "route": route, "source": sid, "events": events,
                         "given": set(kwargs) - {"version"} | {"judge"}, "path": path}

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
            if cost <= 0 or cost > MAX_COUNT:
                _fail("budget.max_cost_per_hour", f"must be positive and at most {MAX_COUNT}, or null for no cap")
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
        "sources": sorted(sources.values(), key=lambda s: s["id"]), "budget": budget,
        "judges": [j["name"] for j in judges_info],
        "conditions": sorted(({**_effective(cond), "trigger": _describe_any(cond.trigger)} for cond in conditions),
                             key=lambda c: c["id"]),
        "events": {k: sorted(v) for k, v in events.items()},
        "sdk_fixed": {"event_clip_s": EVENT_CLIP_S, "clip_frames": CLIP_FRAMES, "deadlines": DEADLINES,  # clip: defaults
                      "default_triggers": {k: [_describe(t) for t in v] for k, v in DEFAULT_TRIGGERS.items()}},
    }
    canonical = json.dumps(effective, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    info = {"concerns": concerns, "judges": judges_info, "cascade": is_cascade, "budget_given": budget_given,
            "sources_given": sources_given}
    return Compiled(
        conditions=tuple(conditions),
        monitor_kwargs={"judge": judge, "conditions": tuple(conditions), **budget},
        events=events,
        plan_hash=hashlib.sha256(canonical.encode()).hexdigest(),
        sources=tuple(sources.values()),
        plan_id=plan_id, revision=revision, _info=info,
    )
