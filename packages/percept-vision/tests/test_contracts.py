import dataclasses
import json
from importlib import resources

import numpy as np
import pytest

from percept.contracts import (
    CONTRACT_VERSION,
    BlindInterval,
    Condition,
    Decision,
    Delivery,
    EvidenceRef,
    Frame,
    Health,
    Judge,
    JudgeItem,
    JudgeRequest,
    JudgeResult,
    Observation,
    PriceModel,
    RequestUsage,
    Source,
    Store,
    Trigger,
    Usage,
)

SQUARE = ((0.1, 0.1), (0.9, 0.1), (0.9, 0.9), (0.1, 0.9))


def obs(**kw):
    base = dict(condition_id="fall", subject="scene", answer="yes", confidence=0.8, reason="r",
                request_id="r-1", stream_id="cam-1", stream_epoch=0, condition_version=1, occurrence_gen=0,
                evidence_t=(1.0, 2.0), judge="fake")
    return Observation(**{**base, **kw})


def decision(**kw):
    base = dict(decision_id="d-1", occurrence_id="fall/scene/1", type="met", condition_id="fall",
                stream_id="cam-1", subject="scene", t_observed=100.0, t_decided=101.5,
                confidence=0.9, reason="a person fell", cost=Usage(attempts=2, tokens_in=900,
                tokens_out=40, cost_usd=0.0012, latency_s=2.2), trace_id="t-1",
                request_ids=("r-17", "r-18"),
                evidence=(EvidenceRef("cam-1", 3, 1841, 100.0),), keyframe=b"\xff\xd8jpeg")
    return Decision(**{**base, **kw})


# --- Condition


def test_condition_defaults_resolve_verification_by_kind():
    assert Condition("fall", "A person falls", kind="event").verify_mode == "original"
    assert Condition("door", "The door is open").verify_mode == "off"
    assert Condition("door", "The door is open", verify="fresh").verify_mode == "fresh"


def test_first_yes_counts_by_default():
    assert Condition("c", "x").confirm == 1


@pytest.mark.parametrize("kw, message", [
    (dict(id=""), "id"),
    (dict(id="a/b"), "/"),
    (dict(text="  "), "text"),
    (dict(kind="maybe"), "kind"),
    (dict(kind="event", verify="fresh"), "fresh frame"),
    (dict(verify="original"), "states verify"),
    (dict(min_confidence=1.5), "min_confidence"),
    (dict(fast_path_confidence=0.5), "fast_path"),
    (dict(confirm=0), "confirm"),
    (dict(clear_after_no=0), "clear_after_no"),
    (dict(confirm_window_s=0), "confirm_window_s"),
    (dict(rearm_s=float("nan")), "rearm_s"),
    (dict(cooldown_s=-1), "cooldown_s"),
    (dict(max_open_s=0), "max_open_s"),
    (dict(zones=(((0, 0), (1, 0)),)), "3 points"),
    (dict(exclude_zones=(((0, 0), (1.2, 0), (1, 1)),)), "normalised"),
    (dict(not_when=("",)), "not_when"),
    (dict(on_unverified="silent"), "on_unverified"),
])
def test_condition_rejects_invalid_values(kw, message):
    base = dict(id="c", text="something")
    with pytest.raises(ValueError, match=message):
        Condition(**{**base, **kw})


def test_condition_accepts_zones_and_is_frozen():
    c = Condition("c", "x", zones=(SQUARE,), exclude_zones=(SQUARE,), not_when=("a sofa",))
    with pytest.raises(dataclasses.FrozenInstanceError):
        c.text = "y"


# --- Frame


def test_frame_validates_and_ignores_image_in_equality():
    a = Frame("cam-1", 0, 5, 1.0, 1000.0, np.zeros((36, 48), np.uint8))
    b = Frame("cam-1", 0, 5, 1.0, 1000.0, np.ones((36, 48), np.uint8))
    assert a == b
    Frame("cam-1", 0, 5, 1.0, 1000.0, np.zeros((36, 48, 3), np.uint8))


@pytest.mark.parametrize("args", [
    ("", 0, 5, 1.0, 1000.0, np.zeros((2, 2), np.uint8)),
    ("cam", 0, 5, 1.0, 1000.0, np.zeros(4, np.uint8)),
    ("cam", 0, 5, 1.0, 1000.0, np.zeros((2, 2), np.float32)),
    ("cam", 0, 5, 1.0, 1000.0, np.zeros((2, 2, 4), np.uint8)),
    ("cam", -1, 5, 1.0, 1000.0, np.zeros((2, 2), np.uint8)),
    ("cam", 0, 5.5, 1.0, 1000.0, np.zeros((2, 2), np.uint8)),
])
def test_frame_rejects_bad_images_and_counters(args):
    with pytest.raises(ValueError):
        Frame(*args)


def test_evidence_ref_validates():
    with pytest.raises(ValueError):
        EvidenceRef("", 0, 0, 1.0)
    with pytest.raises(ValueError):
        EvidenceRef("cam", -1, 0, 1.0)
    with pytest.raises(ValueError):
        EvidenceRef("cam", 0, 0, float("nan"))


# --- Observation


def test_failed_observation_must_answer_unknown():
    assert obs(answer="unknown", confidence=0.0, failure="timeout").failed
    with pytest.raises(ValueError, match="unknown"):
        obs(answer="no", failure="timeout")


@pytest.mark.parametrize("kw", [
    dict(answer="maybe"), dict(confidence=1.1), dict(failure="oops", answer="unknown"),
    dict(stream_id=""), dict(request_id=""), dict(occurrence_gen=-1), dict(stream_epoch=1.5),
    dict(evidence_t=(5.0, 1.0)), dict(purpose="guess"),
])
def test_observation_rejects_invalid(kw):
    with pytest.raises(ValueError):
        obs(**kw)


# --- Usage and judging


def test_usage_adds_and_rejects_bad_counts():
    total = Usage(1, 10, 2, 0.001, 0.5) + Usage(2, 20, 4, 0.002, 0.7)
    assert (total.attempts, total.tokens_in, total.tokens_out) == (3, 30, 6)
    assert total.cost_usd == pytest.approx(0.003) and total.latency_s == pytest.approx(1.2)
    for bad in (dict(attempts=-1), dict(attempts=2.0), dict(tokens_in=1.5), dict(tokens_out=True)):
        with pytest.raises(ValueError):
            Usage(**bad)


@pytest.mark.parametrize("n", [1, 2, 3, 7])
def test_usage_split_is_deterministic_and_sums_exactly(n):
    whole = Usage(attempts=2, tokens_in=1001, tokens_out=40, cost_usd=0.0013, latency_s=2.2)
    parts = whole.split(n)
    assert len(parts) == n and parts == whole.split(n)
    assert sum(p.attempts for p in parts) == 2
    assert sum(p.tokens_in for p in parts) == 1001
    assert sum(p.tokens_out for p in parts) == 40
    assert sum(p.cost_usd for p in parts) == pytest.approx(0.0013)
    assert parts[0].tokens_in >= parts[-1].tokens_in


def test_usage_split_rejects_zero():
    with pytest.raises(ValueError):
        Usage().split(0)


def test_judge_request_validates_items():
    item = JudgeItem(Condition("a", "x"))
    img = (np.zeros((4, 4, 3), np.uint8),)
    JudgeRequest("r-1", "cam", 0, (item,), img, (1.0, 2.0))
    with pytest.raises(ValueError, match="at most once"):
        JudgeRequest("r-1", "cam", 0, (item, item), img, (1.0, 2.0))
    with pytest.raises(ValueError, match="image"):
        JudgeRequest("r-1", "cam", 0, (item,), (), (1.0, 2.0))
    with pytest.raises(ValueError):
        JudgeRequest("r-1", "cam", 0, (item,), img, (2.0, 1.0))


def test_price_model_rejects_negative():
    with pytest.raises(ValueError):
        PriceModel(-1, 0, 0)


# --- Decision


def test_decision_round_trips_through_json():
    d = decision()
    again = Decision.from_dict(json.loads(json.dumps(d.to_dict())))
    assert again == d
    assert again.keyframe == d.keyframe


def test_decision_from_dict_raises_value_error_on_malformed_input():
    data = decision().to_dict()
    del data["trace_id"]
    with pytest.raises(ValueError, match="malformed"):
        Decision.from_dict(data)
    data = decision().to_dict()
    data["cost"]["surprise"] = 1
    with pytest.raises(ValueError, match="malformed"):
        Decision.from_dict(data)


def test_decision_readers_ignore_new_top_level_fields():
    data = decision().to_dict()
    data["contract_version"] = "1.4"
    data["added_in_1_4"] = "x"
    assert Decision.from_dict(data) == decision()


def test_decision_requires_non_empty_ids():
    for name in ("condition_id", "stream_id", "subject", "trace_id"):
        with pytest.raises(ValueError):
            decision(**{name: ""})


def test_decision_rejects_unknown_type_and_large_keyframe():
    with pytest.raises(ValueError):
        decision(type="maybe")
    with pytest.raises(ValueError, match="40 KB"):
        decision(keyframe=b"x" * 40_001)


def test_decision_rejects_other_major_versions():
    data = decision().to_dict()
    data["contract_version"] = "2.0"
    with pytest.raises(ValueError, match="version"):
        Decision.from_dict(data)


def _validate(value, schema, path="$"):
    """A minimal JSON-schema check covering the keywords decision.schema.json uses."""
    t = schema.get("type")
    if "enum" in schema:
        assert value in schema["enum"], path
        return
    types = t if isinstance(t, list) else [t]
    py = {"object": dict, "array": list, "string": str, "boolean": bool, "null": type(None)}
    ok = any(
        (tt == "integer" and isinstance(value, int) and not isinstance(value, bool))
        or (tt == "number" and isinstance(value, (int, float)) and not isinstance(value, bool))
        or (tt in py and isinstance(value, py[tt]))
        for tt in types
    )
    assert ok, f"{path}: {value!r} is not {types}"
    if isinstance(value, dict):
        props = schema.get("properties", {})
        for req in schema.get("required", []):
            assert req in value, f"{path}: missing {req}"
        if schema.get("additionalProperties") is False:
            extra = set(value) - set(props)
            assert not extra, f"{path}: unexpected {extra}"
        for k, v in value.items():
            _validate(v, props[k], f"{path}.{k}")
    if isinstance(value, list) and "items" in schema:
        for i, v in enumerate(value):
            _validate(v, schema["items"], f"{path}[{i}]")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        assert value >= schema.get("minimum", value), path
        assert value <= schema.get("maximum", value), path


def test_decision_matches_the_published_schema():
    schema = json.loads(resources.files("percept").joinpath("decision.schema.json").read_text())
    _validate(decision().to_dict(), schema)
    _validate(decision(keyframe=None, evidence=()).to_dict(), schema)
    assert schema["properties"]["contract_version"]["pattern"] == "^" + CONTRACT_VERSION.split(".")[0] + "\\."


def test_schema_allows_new_top_level_fields_but_not_nested_ones():
    schema = json.loads(resources.files("percept").joinpath("decision.schema.json").read_text())
    assert "additionalProperties" not in schema
    assert schema["properties"]["cost"]["additionalProperties"] is False
    assert schema["properties"]["evidence"]["items"]["additionalProperties"] is False


def test_schema_lists_every_field_to_dict_writes():
    schema = json.loads(resources.files("percept").joinpath("decision.schema.json").read_text())
    assert set(schema["properties"]) == set(decision().to_dict())
    assert set(schema["required"]) == set(decision().to_dict())


# --- Health


def test_health_kinds_are_closed():
    h = Health("verify_deferred", "cam", 1.0, condition_id="fall", subject="scene")
    assert h.condition_id == "fall"
    with pytest.raises(ValueError):
        Health("on_fire", "cam", 1.0)


def test_blind_interval_validates():
    assert BlindInterval("link_down", "cam", 10.0).t_end_wall is None
    BlindInterval("scene_reset", "cam", 10.0, 12.0)
    with pytest.raises(ValueError):
        BlindInterval("link_down", "cam", 10.0, 9.0)
    with pytest.raises(ValueError):
        BlindInterval("nap", "cam", 10.0)


def test_condition_trigger_must_be_a_trigger():
    class T:
        def step(self, frame, conditions, state, now): return {}, state

    Condition("c", "x", trigger=T())
    with pytest.raises(ValueError, match="Trigger"):
        Condition("c", "x", trigger="nonsense")


# --- Protocols are structural


def test_protocols_accept_structural_implementations():
    class S:
        stream_id = "cam"
        def frames(self): ...

    class T:
        def step(self, frame, conditions, state, now): return {}, state

    class J:
        name = "fake"
        can_batch = True
        price = None
        async def judge(self, request): return JudgeResult((), ())

    class D:
        decision = None
        attempt = 1
        async def ack(self): ...

    class St:
        def load(self, key): return None
        def commit(self, writes): ...

    assert isinstance(S(), Source) and isinstance(T(), Trigger) and isinstance(J(), Judge)
    assert isinstance(D(), Delivery) and isinstance(St(), Store)
    assert RequestUsage("r", "fake", Usage()).failure is None


# --- D5 contract additions


def test_decision_cause_round_trips_and_defaults_to_none():
    assert decision().cause is None
    d = decision(type="cleared", cause="no")
    assert Decision.from_dict(json.loads(json.dumps(d.to_dict()))).cause == "no"
    older = decision().to_dict()
    del older["cause"]                      # a 1.0 payload written before `cause` existed
    assert Decision.from_dict(older).cause is None


def test_unverified_health_and_capacity_fate_exist():
    from percept.contracts import FATES, HEALTH_KINDS
    assert "unverified" in HEALTH_KINDS and "dropped_capacity" in FATES
    Health("unverified", "cam", 1.0, condition_id="fall", subject="scene")
