"""Watch plans: validation, compilation, explain, feasibility, provenance, and a run end to end."""

from __future__ import annotations

import asyncio
import copy
import dataclasses
import inspect
import json
import time

import pytest

import percept
from percept import monitor as rt
from percept import plans
from percept import trace as tr
from percept.contracts import Condition
from percept.judges import FakeJudge
from percept.plans import PlanError
from percept.sources import PushSource
from percept.triggers import every, on_change
from test_monitor import Y, collect, image, run

DOORWAY = [[0.1, 0.1], [0.5, 0.1], [0.5, 0.9], [0.1, 0.9]]
DOOR = [[0.5, 0.1], [0.9, 0.1], [0.9, 0.9], [0.5, 0.9]]


def tup(poly):
    return tuple(tuple(float(v) for v in p) for p in poly)


def base(concerns, sources=None, **extra):
    return {"schema_version": plans.SCHEMA_VERSION, "plan_id": "test", "revision": 1,
            "sources": sources or [{"id": "lobby", "kind": "video"}], "concerns": concerns, **extra}


def doorway_plan():
    """(a) a person enters through a doorway and (b) a person opens a door: events on one source."""
    return base([
        {"id": "enters", "source": "lobby", "description": "a person enters through the doorway",
         "attention": {"any": [{"on_change": {"threshold": 0.25}}]},
         "judgment": {"route": "fast", "zones": [DOORWAY], "evidence": {"pre_seconds": 4.0, "max_side": 640}},
         "decision": {"kind": "event", "rearm_s": 30}},
        {"id": "opens", "source": "lobby", "description": "a person opens the door",
         "attention": {"any": [{"on_change": {}}]},
         "judgment": {"route": "fast", "zones": [DOOR], "not_when": ["the door is already open"],
                      "min_confidence": 0.7},
         "decision": {"kind": "event", "confirm": 1}},
    ], metadata={"owner": "someone", "note": "not hashed"})


def state_plan():
    """(c) a periodic state concern."""
    return base([{"id": "lamp_on", "source": "dock", "description": "the lamp is on",
                  "attention": {"any": [{"every": 0.2}]},
                  "judgment": {"route": "fast", "verification": "off"},
                  "decision": {"kind": "state", "clear_after_no": 1, "emit_cleared": True, "recheck_every_s": 0.2}}],
                sources=[{"id": "dock", "kind": "video", "max_side": 320}],
                budget={"max_requests_per_min": 600, "max_concurrent_requests": 2})


def poke_plan():
    """(d) a concern judged only when the app pokes it."""
    return base([{"id": "package_fell", "source": "lobby", "description": "a package fell off the shelf",
                  "attention": {"any": [{"external_event": "shelf_sensor"}]},
                  "judgment": {"route": "fast"}, "decision": {"kind": "event"}}])


def fake(**kw) -> FakeJudge:
    return FakeJudge({}, **kw)


def compile_(plan, **routes):
    return plans.compile(plan, routes or {"fast": fake()})


# --------------------------------------------------------------------------- golden fixtures


def test_doorway_events_compile_to_their_conditions():
    c = compile_(doorway_plan())
    assert c.conditions == (
        Condition("enters", "a person enters through the doorway", kind="event", trigger=on_change(0.25),
                  zones=(tup(DOORWAY),), rearm_s=30.0),
        Condition("opens", "a person opens the door", kind="event", trigger=on_change(), zones=(tup(DOOR),),
                  not_when=("the door is already open",), min_confidence=0.7),
    )
    kw = c.monitor_kwargs
    assert kw["judge"].name == "fake" and kw["conditions"] == c.conditions
    assert (kw["max_requests_per_min"], kw["max_cost_per_hour"], kw["max_concurrent_requests"]) == (6, None, 4)
    assert c.events == {} and c.sources == ({"id": "lobby", "kind": "video", "max_side": 640},)
    assert c.feasibility()["sources"]["lobby"]["verdict"] == "best_effort"
    percept.Monitor(**kw, store=percept.MemoryStore())     # accepted as is


def test_periodic_state_compiles():
    c = compile_(state_plan())
    assert c.conditions == (Condition("lamp_on", "the lamp is on", trigger=every(0.2), verify="off",
                                      clear_after_no=1, emit_cleared=True, recheck_every_s=0.2),)
    assert c.monitor_kwargs["max_requests_per_min"] == 600 and c.monitor_kwargs["max_concurrent_requests"] == 2
    f = c.feasibility()["sources"]["dock"]
    assert f["verdict"] == "feasible" and f["periodic_requests_per_min"] == 300 and f["periodic_attempts_per_min"] == 600


def test_poke_only_concern_gets_a_trigger_that_never_fires_and_an_event_map():
    c = compile_(poke_plan())
    (cond,) = c.conditions
    assert c.events == {"shelf_sensor": ("package_fell",)}
    assert cond.trigger is not None and isinstance(cond.trigger, percept.Trigger)
    assert cond.trigger.step(None, c.conditions, "s", 0.0) == ({}, "s")
    assert cond.trigger == compile_(poke_plan()).conditions[0].trigger            # groups by value
    assert hash(cond.trigger) == hash(compile_(poke_plan()).conditions[0].trigger)
    assert c.explain()["concerns"][0]["trigger"]["value"] == [{"type": "external_event_only"}]


def test_external_event_alongside_a_trigger_keeps_the_trigger():
    plan = poke_plan()
    plan["concerns"][0]["attention"]["any"].append({"every": 30})
    c = compile_(plan)
    assert c.conditions[0].trigger == every(30) and c.events == {"shelf_sensor": ("package_fell",)}


def test_absent_attention_uses_the_sdk_default():
    plan = poke_plan()
    del plan["concerns"][0]["attention"]
    c = compile_(plan)
    assert c.conditions[0].trigger is None
    assert c.explain()["concerns"][0]["trigger"]["origin"] == "sdk_default"


def test_revision_is_the_condition_version():
    plan = state_plan()
    plan["revision"] = 7
    assert compile_(plan).conditions[0].version == 7


# --------------------------------------------------------------------------- invalid plans


def mutate(fn):
    plan = doorway_plan()
    fn(plan)
    return plan


def j0(plan):
    return plan["concerns"][0]["judgment"]


INVALID = [
    (lambda p: p.update(extra=1), "extra: unknown field"),
    (lambda p: j0(p)["evidence"].update(fps=2), "concerns[0].judgment.evidence.fps: unknown field"),
    (lambda p: p.update(revision="1"), "revision: expected an integer"),
    (lambda p: p.update(revision=0), "revision: expected an integer >= 1"),
    (lambda p: j0(p).update(min_confidence="high"), "concerns[0].judgment.min_confidence: expected a finite number"),
    (lambda p: p["concerns"][0]["decision"].pop("kind"), "concerns[0].decision.kind: required field is missing"),
    (lambda p: p.pop("plan_id"), "plan_id: required field is missing"),
    (lambda p: p.update(schema_version="percept.watch/v2"), "schema_version: unsupported"),
    (lambda p: p["sources"].append({"id": "lobby", "kind": "video"}), "sources[1].id: duplicate source id"),
    (lambda p: p["concerns"][1].update(id="enters"), "concerns[1].id: duplicate concern id"),
    (lambda p: p["concerns"][1].update(source="garage"), "concerns[1].source: no source with id 'garage'"),
    (lambda p: j0(p).update(route="slow"), "concerns[0].judgment.route: unknown route 'slow'"),
    (lambda p: p["sources"][0].update(kind="audio"), "sources[0].kind: only 'video'"),
    (lambda p: j0(p)["evidence"].update(post_seconds=2), "concerns[0].judgment.evidence.post_seconds: not supported"),
    (lambda p: j0(p)["evidence"].update(max_side=320), "concerns[0].judgment.evidence.max_side: resolution is set per source"),
    (lambda p: j0(p)["evidence"].update(pre_seconds=2.0), "concerns[0].judgment.evidence.pre_seconds: must be 4.0: "
                                                          "per-condition evidence windows are not supported yet"),
    (lambda p: p["concerns"][0]["attention"]["any"].append({"sometimes": 5}), "concerns[0].attention.any[1].sometimes: unknown"),
    (lambda p: p["concerns"][0]["attention"].update(any=[{"every": -1}]), "concerns[0].attention.any[0].every:"),
    (lambda p: p["concerns"][0]["attention"].update(any=[{"on_change": {"threshold": 2}}]),
     "concerns[0].attention.any[0].on_change: threshold is in (0, 1]"),
    (lambda p: j0(p).update(zones=[[[0, 0], [1, 0]]]), "concerns[0]: zones: a polygon needs at least 3 points"),
    (lambda p: j0(p).update(zones=[[[0, 0], [1]]]), "concerns[0].judgment.zones[0][1]: a point is an [x, y] pair"),
    (lambda p: p["concerns"][0].update(id="a/b"), "concerns[0]: Condition.id may not contain '/'"),
    (lambda p: p["concerns"][0]["decision"].update(kind="moment"), "concerns[0].decision.kind: expected 'event' or 'state'"),
    (lambda p: p.update(budget={"max_requests_per_min": 1}), "budget.max_requests_per_min: 1 is below the 2 attempts"),
    (lambda p: p.update(budget={"max_cost_per_hour": 0}), "budget.max_cost_per_hour: must be positive"),
    (lambda p: p.update(judges={"order": ["fast", "nope"]}), "judges.order[1]: unknown route 'nope'"),
    (lambda p: p.update(metadata=[]), "metadata: expected an object"),
    (lambda p: p.update(concerns=[]), "concerns: must not be empty"),
]


@pytest.mark.parametrize("change, message", INVALID, ids=[m.split(":")[0] + f"#{i}" for i, (_, m) in enumerate(INVALID)])
def test_invalid_plan_raises_plan_error_naming_the_field(change, message):
    with pytest.raises(PlanError) as err:
        compile_(mutate(change))
    assert message in str(err.value), str(err.value)


def test_condition_errors_carry_the_concern_path():
    plan = state_plan()
    plan["concerns"][0]["judgment"]["verification"] = "original"
    with pytest.raises(PlanError, match=r"^concerns\[0\]: states verify on a fresh frame or not at all"):
        compile_(plan)


def test_pre_seconds_on_a_state_is_refused():
    plan = state_plan()
    plan["concerns"][0]["judgment"]["evidence"] = {"pre_seconds": 4.0}
    with pytest.raises(PlanError, match=r"evidence\.pre_seconds: a state is judged on the newest frame"):
        compile_(plan)


def test_a_plan_that_is_not_an_object_and_bad_routes_are_plan_errors():
    with pytest.raises(PlanError, match="^plan: expected an object"):
        plans.compile([], {"fast": fake()})
    with pytest.raises(PlanError, match="^routes"):
        plans.compile(doorway_plan(), {})
    with pytest.raises(PlanError, match=r"^routes\['fast'\]"):
        plans.compile(doorway_plan(), {"fast": object()})


def test_plan_error_is_a_value_error():
    assert issubclass(PlanError, ValueError)


# --------------------------------------------------------------------------- judges and routes


def test_two_routes_to_one_judge_are_aliases_not_a_cascade():
    judge = fake()
    plan = doorway_plan()
    plan["concerns"][1]["judgment"]["route"] = "cheap"
    c = plans.compile(plan, {"fast": judge, "cheap": judge})
    assert c.monitor_kwargs["judge"] is judge
    assert all(cond.judge is None for cond in c.conditions)
    assert c.provenance()["judges"] == [{"routes": ["cheap", "fast"], "name": "fake", "model": None, "price": None}]


def test_two_judges_need_an_order():
    plan = doorway_plan()
    plan["concerns"][1]["judgment"]["route"] = "strong"
    with pytest.raises(PlanError, match=r"^judges\.order: routes map to more than one judge"):
        plans.compile(plan, {"fast": fake(name="a"), "strong": fake(name="b")})


def test_an_order_builds_a_cascade_and_pins_each_condition():
    plan = doorway_plan()
    plan["concerns"][1]["judgment"]["route"] = "strong"
    plan["judges"] = {"order": ["fast", "strong"]}
    c = plans.compile(plan, {"fast": fake(name="a"), "strong": fake(name="b")})
    judge = c.monitor_kwargs["judge"]
    assert judge.name == "cascade(a>b)" and [m.name for m in judge.members] == ["a", "b"]
    assert [cond.judge for cond in c.conditions] == ["a", "b"]
    explained = c.explain()
    assert any("verification request goes to the last judge in judges.order (b)" in n for n in explained["notes"])
    assert {x["verification_judge"]["value"] for x in explained["concerns"]} == {"b"}
    percept.Monitor(**c.monitor_kwargs, store=percept.MemoryStore())      # routes check out


def test_an_order_missing_a_used_judge_is_refused():
    plan = doorway_plan()
    plan["concerns"][1]["judgment"]["route"] = "strong"
    plan["judges"] = {"order": ["fast"]}
    with pytest.raises(PlanError, match=r"^judges\.order: judge 'b' is used by a concern but not in the order"):
        plans.compile(plan, {"fast": fake(name="a"), "strong": fake(name="b")})


def test_same_named_distinct_judges_are_a_plan_error():
    plan = doorway_plan()
    plan["concerns"][1]["judgment"]["route"] = "strong"
    plan["judges"] = {"order": ["fast", "strong"]}
    with pytest.raises(PlanError, match=r"^judges\.order: cascade members need distinct names"):
        plans.compile(plan, {"fast": fake(), "strong": fake()})


# --------------------------------------------------------------------------- explain


def test_explain_reports_every_condition_field_with_an_origin():
    c = compile_(doorway_plan())
    report = c.explain()
    json.dumps(report)
    expected = {f.name for f in dataclasses.fields(Condition)} - {"trigger", "version"}
    for concern in report["concerns"]:
        assert set(concern["fields"]) == expected
        assert all(v["origin"] in ("plan", "sdk_default") for v in concern["fields"].values())
        assert concern["trigger"]["origin"] == "plan" and concern["version"] == {"value": 1, "origin": "plan"}
        assert concern["evidence"] == {"value": {"clip_s": rt.EVENT_CLIP_S, "frames": rt.CLIP_FRAMES,
                                                 "ends": "at the trigger"}, "origin": "sdk_fixed"}
        assert concern["detection_deadline_s"] == {"value": rt.DEADLINES[rt.EVENT], "origin": "sdk_fixed"}
        assert concern["verification_judge"] == {"value": "fake", "origin": "sdk_fixed"}
    enters = report["concerns"][0]["fields"]
    assert enters["rearm_s"] == {"value": 30.0, "origin": "plan"}
    assert enters["min_confidence"] == {"value": 0.6, "origin": "sdk_default"}
    assert enters["zones"]["origin"] == "plan" and enters["kind"]["origin"] == "plan"
    assert report["budget"]["max_requests_per_min"] == {"value": 6, "origin": "sdk_default"}
    assert compile_(state_plan()).explain()["budget"]["max_requests_per_min"] == {"value": 600, "origin": "plan"}


def test_mirrored_runtime_values_match_the_monitor():
    assert plans.EVENT_CLIP_S == rt.EVENT_CLIP_S and plans.CLIP_FRAMES == rt.CLIP_FRAMES
    assert plans.DEADLINES == {"event": rt.DEADLINES[rt.EVENT], "state": rt.DEADLINES[rt.STATE]}
    params = inspect.signature(percept.Monitor.__init__).parameters
    assert plans.BUDGET == {k: params[k].default for k in plans.BUDGET}
    m = percept.Monitor(fake(), store=percept.MemoryStore())
    assert plans.DEFAULT_TRIGGERS["event"] == (m._default_change,)
    assert plans.DEFAULT_TRIGGERS["state"] == (m._default_change, m._default_heartbeat)
    assert inspect.signature(PushSource).parameters["max_side"].default == plans.MAX_SIDE


# --------------------------------------------------------------------------- hash


def shuffled(value):
    if isinstance(value, dict):
        return {k: shuffled(value[k]) for k in reversed(list(value))}
    if isinstance(value, list):
        return [shuffled(v) for v in value]
    return value


def test_hash_ignores_key_order_whitespace_metadata_and_explicit_defaults():
    h = compile_(doorway_plan()).plan_hash
    assert len(h) == 64
    assert compile_(shuffled(doorway_plan())).plan_hash == h
    assert compile_(json.loads(json.dumps(doorway_plan(), indent=4))).plan_hash == h
    assert compile_(mutate(lambda p: p.update(metadata={"x": 1}))).plan_hash == h
    assert compile_(mutate(lambda p: j0(p).update(min_confidence=0.6, verification="auto"))).plan_hash == h
    assert compile_(mutate(lambda p: p.update(budget={"max_requests_per_min": 6}))).plan_hash == h
    assert compile_(mutate(lambda p: p["concerns"][0]["decision"].update(rearm_s=30.0))).plan_hash == h


@pytest.mark.parametrize("change", [
    lambda p: p.update(revision=2),
    lambda p: p.update(plan_id="other"),
    lambda p: j0(p).update(min_confidence=0.61),
    lambda p: p["concerns"][0]["attention"]["any"][0]["on_change"].update(threshold=0.3),
    lambda p: p["concerns"][0].update(attention={"any": [{"every": 5}]}),
    lambda p: p["concerns"][0].pop("attention"),
    lambda p: p.update(budget={"max_concurrent_requests": 1}),
    lambda p: p["sources"][0].update(max_side=320) or j0(p)["evidence"].pop("max_side"),
    lambda p: p["concerns"][1]["attention"]["any"].append({"external_event": "bell"}),
    lambda p: p["concerns"][0].update(description="someone enters"),
])
def test_hash_changes_with_any_effective_setting(change):
    assert compile_(mutate(change)).plan_hash != compile_(doorway_plan()).plan_hash


def test_hash_changes_with_the_judge():
    assert compile_(doorway_plan(), fast=fake(name="other")).plan_hash != compile_(doorway_plan()).plan_hash


# --------------------------------------------------------------------------- fuzz


def test_fuzzed_plans_raise_only_plan_errors():
    hypothesis = pytest.importorskip("hypothesis")
    st = hypothesis.strategies
    leaf = st.one_of(st.none(), st.booleans(), st.integers(min_value=-10**400, max_value=10**400),
                     st.floats(allow_nan=True, allow_infinity=True), st.text(max_size=6),
                     st.sampled_from(["video", "event", "state", "fast", "lobby", "auto", "off", "original", "fresh"]))
    values = st.recursive(leaf, lambda inner: st.one_of(st.lists(inner, max_size=3),
                                                        st.dictionaries(st.text(max_size=6), inner, max_size=3)),
                          max_leaves=8)
    seeds = [doorway_plan(), state_plan(), poke_plan()]

    def slots(value, out):
        if isinstance(value, dict):
            out.append(value)
            for v in value.values():
                slots(v, out)
        elif isinstance(value, list):
            out.append(value)
            for v in value:
                slots(v, out)
        return out

    @hypothesis.settings(max_examples=1000, deadline=None)
    @hypothesis.given(st.data())
    def check(data):
        plan = copy.deepcopy(data.draw(st.sampled_from(seeds)))
        for _ in range(data.draw(st.integers(1, 3))):
            target = data.draw(st.sampled_from(slots(plan, [])))
            op = data.draw(st.sampled_from(["drop", "insert", "replace"]))
            if isinstance(target, dict):
                keys = list(target)
                if op == "insert" or not keys:
                    target[data.draw(st.sampled_from(["id", "kind", "route", "any", "every", "x", "zones"]))] = \
                        data.draw(values)
                else:
                    key = data.draw(st.sampled_from(keys))
                    if op == "drop":
                        del target[key]
                    else:
                        target[key] = data.draw(values)
            elif op == "insert" or not target:
                target.append(data.draw(values))
            else:
                i = data.draw(st.integers(0, len(target) - 1))
                if op == "drop":
                    del target[i]
                else:
                    target[i] = data.draw(values)
        try:
            c = plans.compile(plan, {"fast": fake()})
        except PlanError:
            return
        json.dumps(c.explain())
        json.dumps(c.feasibility())
        json.dumps(c.provenance())

    check()


# --------------------------------------------------------------------------- feasibility


def ten_every_five(zones_differ=False):
    concerns = [{"id": f"c{i}", "source": "lobby", "description": f"thing {i} happens",
                 "attention": {"any": [{"every": 5}]},
                 "judgment": {"route": "fast",
                              "zones": [[[0.0, 0.0], [0.1 + i * 0.05 if zones_differ else 0.5, 0.0], [0.0, 0.5]]]},
                 "decision": {"kind": "event"}} for i in range(10)]
    return base(concerns, budget={"max_requests_per_min": 6})


def test_feasibility_golden_ten_events_one_zone_batching_judge():
    c = plans.compile(ten_every_five(), {"fast": FakeJudge({}, can_batch=True, retries=1)})
    f = c.feasibility()
    s = f["sources"]["lobby"]
    assert (s["periodic_requests_per_min"], s["periodic_attempts_per_min"], s["ceiling_requests_per_min"]) == (12, 24, 6)
    assert s["max_attempts"] == 2 and len(s["groups"]) == 1 and len(s["groups"][0]["conditions"]) == 10
    assert s["verdict"] == "infeasible" and f["verdict"] == "infeasible"
    assert any("verify" in n for n in s["notes"])


def test_feasibility_ten_zones_are_ten_requests():
    s = plans.compile(ten_every_five(zones_differ=True), {"fast": fake()}).feasibility()["sources"]["lobby"]
    assert s["periodic_requests_per_min"] == 120 and len(s["groups"]) == 10


def test_feasibility_non_batching_judge_sends_each_alone():
    s = plans.compile(ten_every_five(), {"fast": fake(can_batch=False)}).feasibility()["sources"]["lobby"]
    assert s["periodic_requests_per_min"] == 120 and len(s["groups"]) == 10 and s["verdict"] == "infeasible"


def test_feasibility_change_triggered_and_poked_demand_is_best_effort():
    for plan in (doorway_plan(), poke_plan()):
        s = compile_(plan).feasibility()["sources"]["lobby"]
        assert s["verdict"] == "best_effort"
        assert "change-triggered demand is unknown until measured on representative video" in s["notes"]


def test_feasibility_counts_the_default_state_heartbeat():
    plan = state_plan()
    del plan["concerns"][0]["attention"]
    s = compile_(plan).feasibility()["sources"]["dock"]
    assert s["groups"][0]["every_s"] == 300.0 and s["verdict"] == "best_effort"


# --------------------------------------------------------------------------- provenance


def test_provenance_never_carries_secrets(monkeypatch):
    secret = "AIzaSyD-not-a-real-key-0123456789abcdefgh"
    monkeypatch.setenv("GEMINI_API_KEY", secret)
    monkeypatch.setenv("GOOGLE_API_KEY", secret)
    judge = fake(price=percept.PriceModel(0.1, 0.4, 258))
    judge.api_key = secret
    judge.model = "some-model"
    c = plans.compile(doorway_plan(), {"fast": judge})
    p = c.provenance()
    for out in (p, c.explain(), c.trace_record(1.0), c.feasibility()):
        assert secret not in json.dumps(out)
    assert p["sdk_version"] == percept.__version__ and p["plan_hash"] == c.plan_hash
    assert p["judges"] == [{"routes": ["fast"], "name": "fake", "model": "some-model",
                            "price": {"usd_per_mtok_in": 0.1, "usd_per_mtok_out": 0.4, "tokens_per_image": 258}}]
    assert len(p["prompt_sha256"]) == 64


def test_prompt_hash_follows_the_prompt(monkeypatch):
    before = compile_(poke_plan()).provenance()["prompt_sha256"]
    monkeypatch.setattr(plans, "SYSTEM", plans.SYSTEM + " ")
    assert compile_(poke_plan()).provenance()["prompt_sha256"] != before


# --------------------------------------------------------------------------- end to end


def test_compiled_plan_runs_and_a_plan_record_still_replays(tmp_path):
    c = plans.compile(state_plan(), {"fast": FakeJudge({"lamp_on": Y})})
    path = tmp_path / "trace.jsonl"
    writer = tr.TraceWriter(path)
    writer.write(c.trace_record(time.time(), "dock"))

    async def main():
        async with percept.Monitor(**c.monitor_kwargs, store=percept.MemoryStore(), trace=writer) as m:
            src = PushSource("dock", max_side=c.sources[0]["max_side"])
            watch = m.watch(src, fps=None)
            for i in range(4):
                src.push(image(i))
                await asyncio.sleep(0.05)
            src.close()
            return await collect(watch)

    got = run(main())
    assert [d.decision.type for d in got] == ["met"] and got[0].decision.condition_id == "lamp_on"
    writer.close()
    records = list(tr.read(path))
    assert records[0]["kind"] == "plan" and records[0]["plan_hash"] == c.plan_hash
    result = tr.replay(path)
    assert result.matches, result.differences


# --------------------------------------------------------------------------- review follow-ups


def two_events(zones_a, zones_b, exclude_a=None):
    def concern(cid, zones, exclude):
        judgment = {"route": "fast", "zones": zones}
        if exclude:
            judgment["exclude_zones"] = exclude
        return {"id": cid, "source": "lobby", "description": f"{cid} happens",
                "attention": {"any": [{"every": 5}]}, "judgment": judgment, "decision": {"kind": "event"}}
    return base([concern("a", zones_a, exclude_a), concern("b", zones_b, None)])


def test_feasibility_groups_on_the_crop_not_on_exclude_zones():
    corner = [[[0.6, 0.6], [0.7, 0.6], [0.6, 0.7]]]
    s = compile_(two_events([DOOR], [DOOR], exclude_a=corner)).feasibility()["sources"]["lobby"]
    assert len(s["groups"]) == 1


def test_feasibility_groups_polygons_with_the_same_bounding_box():
    square = [[0.5, 0.1], [0.9, 0.1], [0.9, 0.9], [0.5, 0.9]]
    triangle = [[0.5, 0.1], [0.9, 0.9], [0.5, 0.9], [0.9, 0.1]]
    s = compile_(two_events([square], [triangle])).feasibility()["sources"]["lobby"]
    assert len(s["groups"]) == 1


def test_verified_events_make_a_source_at_best_best_effort():
    plan = base([{"id": "fell", "source": "lobby", "description": "a person falls",
                  "attention": {"any": [{"every": 60}]}, "judgment": {"route": "fast"}, "decision": {"kind": "event"}}],
                budget={"max_requests_per_min": 6})
    s = compile_(plan).feasibility()["sources"]["lobby"]
    assert s["periodic_attempts_per_min"] == 2 and s["verdict"] == "best_effort"
    assert any("verify" in n and "unknown" in n for n in s["notes"])
    plan["concerns"][0]["judgment"]["verification"] = "off"
    assert compile_(plan).feasibility()["sources"]["lobby"]["verdict"] == "feasible"


def test_a_cost_cap_without_a_price_is_noted_not_refused():
    plan = mutate(lambda p: p.update(budget={"max_cost_per_hour": 1.5}))
    c = compile_(plan)
    note = "cost cap not enforced: judge has no price"
    assert note in c.explain()["notes"] and note in c.feasibility()["notes"]
    assert note in c.feasibility()["sources"]["lobby"]["notes"]
    priced = plans.compile(plan, {"fast": fake(price=percept.PriceModel(0.1, 0.4, 258))})
    assert note not in priced.explain()["notes"]


def test_explain_reports_source_origins_and_the_route_as_plan():
    report = compile_(state_plan()).explain()
    assert report["sources"] == [{"id": {"value": "dock", "origin": "plan"}, "kind": {"value": "video", "origin": "plan"},
                                  "max_side": {"value": 320, "origin": "plan"}}]
    assert compile_(poke_plan()).explain()["sources"][0]["max_side"] == {"value": 640, "origin": "sdk_default"}
    assert report["concerns"][0]["fields"]["judge"]["origin"] == "plan"


def test_hash_ignores_the_order_of_concerns_sources_and_events():
    plan = doorway_plan()
    plan["sources"].append({"id": "yard", "kind": "video"})
    plan["concerns"][0]["attention"]["any"].append({"external_event": "bell"})
    plan["concerns"][1]["attention"]["any"].append({"external_event": "bell"})
    plan["concerns"][1]["attention"]["any"].append({"external_event": "alarm"})
    h = compile_(plan).plan_hash
    reordered = copy.deepcopy(plan)
    reordered["sources"].reverse()
    reordered["concerns"].reverse()
    reordered["concerns"][0]["attention"]["any"].reverse()
    assert compile_(reordered).plan_hash == h


@pytest.mark.parametrize("name, value", [("EVENT_CLIP_S", 5.0), ("CLIP_FRAMES", 6),
                                         ("DEADLINES", {"event": 1.0, "state": 10.0}),
                                         ("DEFAULT_TRIGGERS", {"event": (on_change(),), "state": (on_change(),)})])
def test_hash_covers_the_runtime_fixed_values(monkeypatch, name, value):
    before = compile_(poke_plan()).plan_hash
    monkeypatch.setattr(plans, name, value)
    assert compile_(poke_plan()).plan_hash != before


@pytest.mark.parametrize("change, message", [
    (lambda p: p.update(budget={"max_requests_per_min": 10**6 + 1}),
     "budget.max_requests_per_min: expected an integer <= 1000000"),
    (lambda p: p.update(budget={"max_concurrent_requests": 10**9}), "budget.max_concurrent_requests: expected an integer <="),
    (lambda p: p["concerns"][0]["decision"].update(rearm_s=1e8), "concerns[0].decision.rearm_s: expected at most"),
    (lambda p: p["concerns"][0]["decision"].update(rearm_s=float("inf")), "concerns[0].decision.rearm_s: expected a finite"),
    (lambda p: p["concerns"][0].update(attention={"any": [{"every": 1e9}]}),
     "concerns[0].attention.any[0].every: expected at most"),
    (lambda p: p["sources"][0].update(max_side=10**7), "sources[0].max_side: expected an integer <="),
])
def test_absurd_numbers_are_refused(change, message):
    with pytest.raises(PlanError) as err:
        compile_(mutate(change))
    assert message in str(err.value), str(err.value)


def test_a_poked_plan_runs_end_to_end():
    c = plans.compile(poke_plan(), {"fast": FakeJudge({"package_fell": Y})})

    async def main():
        async with percept.Monitor(**dict(c.monitor_kwargs, max_requests_per_min=600),
                                   store=percept.MemoryStore()) as m:
            src = PushSource("lobby")
            watch = m.watch(src, fps=None)
            for i in range(10):
                src.push(image(i))
                await asyncio.sleep(0.03)
                if i == 5:
                    m.poke("lobby", c.events["shelf_sensor"], reason="shelf sensor tripped")
            await asyncio.sleep(0.3)
            src.close()
            return await collect(watch)

    got = run(main())
    assert [(d.decision.type, d.decision.condition_id) for d in got] == [("met", "package_fell")]


def test_a_cascade_pinned_plan_sends_each_concern_to_its_judge():
    plan = state_plan()
    second = copy.deepcopy(plan["concerns"][0])
    second.update(id="door_open", description="the door is open")
    second["judgment"]["route"] = "strong"
    plan["concerns"].append(second)
    plan["judges"] = {"order": ["fast", "strong"]}
    a, b = FakeJudge({"lamp_on": Y, "door_open": Y}, name="a"), FakeJudge({"lamp_on": Y, "door_open": Y}, name="b")
    c = plans.compile(plan, {"fast": a, "strong": b})

    async def main():
        async with percept.Monitor(**c.monitor_kwargs, store=percept.MemoryStore()) as m:
            src = PushSource("dock")
            watch = m.watch(src, fps=None)
            for i in range(4):
                src.push(image(i))
                await asyncio.sleep(0.05)
            src.close()
            return await collect(watch)

    got = run(main())
    assert sorted(d.decision.condition_id for d in got) == ["door_open", "lamp_on"]

    def asked(judge):
        return {i.condition.id for *_, r in judge.sent if r.purpose == "detect" for i in r.items}

    assert asked(a) == {"lamp_on"} and asked(b) == {"door_open"}
