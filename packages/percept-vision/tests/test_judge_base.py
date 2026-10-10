import json

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from percept.contracts import (
    Condition,
    CounterExample,
    JudgeItem,
    JudgeRequest,
    JudgeResult,
    PriceModel,
    RequestUsage,
    Usage,
)
from percept.judges.base import (
    EVENT_RULE,
    VERIFY_NOTE,
    attribute_usage,
    build_prompt,
    cost_of,
    extract_json,
    failed_observations,
    parse_reply,
    worst_case_cost,
    worst_case_tokens,
)

PRICE = PriceModel(usd_per_mtok_in=1.0, usd_per_mtok_out=2.0, tokens_per_image=100)  # synthetic, not a vendor price


def make_request(n_images=2, purpose="detect"):
    items = (
        JudgeItem(Condition("door", "The door is open")),
        JudgeItem(Condition("fall", "A person falls", kind="event", not_when=("lying on a sofa",), version=3),
                  subject="person-7", occurrence_gen=2,
                  counter_examples=(CounterExample("a coat dropped on the floor"),)),
    )
    images = tuple(np.zeros((4, 4, 3), np.uint8) for _ in range(n_images))
    return JudgeRequest("r-9", "cam-2", 5, items, images, (100.0, 104.0), purpose=purpose)


def reply(*entries):
    return json.dumps({"results": list(entries)})


def entry(cid, answer="yes", confidence=0.8, reason="ok"):
    return {"id": cid, "answer": answer, "confidence": confidence, "reason": reason}


REQ = make_request()
GOOD = reply(entry("door"), entry("fall", "no", 0.7))


def failures(text):
    return [o.failure for o in parse_reply(text, REQ, "j")]


# --------------------------------------------------------------------------- prompt


def test_prompt_states_schema_and_unknown():
    p = build_prompt(REQ)
    assert '"results"' in p.system and '"unknown"' in p.system and "independently" in p.system


def test_prompt_event_wording_only_for_events():
    user = build_prompt(REQ).user
    door, fall = user.split("- id: ")[1:]
    assert EVENT_RULE in fall and EVENT_RULE not in door


def test_prompt_not_when_and_counter_examples():
    user = build_prompt(REQ).user
    assert 'FALSE case, answer no: "lying on a sofa"' in user
    assert 'previously mistaken; this is NOT it: "a coat dropped on the floor"' in user
    assert '"person-7"' in user


def test_prompt_orders_images_with_interval():
    user = build_prompt(make_request(n_images=3)).user
    assert "image 1 is the oldest and image 3 the newest" in user and "4.0 s" in user
    assert "1 image" in build_prompt(make_request(n_images=1)).user


def test_prompt_verify_is_strict_without_anchoring():
    user = build_prompt(make_request(purpose="verify")).user
    assert VERIFY_NOTE in user and "earlier" not in user
    assert VERIFY_NOTE not in build_prompt(REQ).user


def test_prompt_defines_confidence_as_certainty_of_the_answer():
    assert "how sure you are of your answer" in build_prompt(REQ).system


def test_prompt_keeps_non_ascii_text_readable():
    req = JudgeRequest("r", "c", 0, (JudgeItem(Condition("door", "ドアが開いている")),),
                       (np.zeros((4, 4, 3), np.uint8),), (0.0, 0.0))
    assert "ドアが開いている" in build_prompt(req).user


# --------------------------------------------------------------------------- parse


def test_parse_good_reply_and_tags():
    door, fall = parse_reply(GOOD, REQ, "judge-x")
    assert (door.answer, door.confidence, door.failure) == ("yes", 0.8, None)
    assert (fall.answer, fall.subject, fall.occurrence_gen, fall.condition_version) == ("no", "person-7", 2, 3)
    for o in (door, fall):
        assert (o.request_id, o.stream_id, o.stream_epoch, o.evidence_t, o.judge, o.purpose) == \
            ("r-9", "cam-2", 5, (100.0, 104.0), "judge-x", "detect")


@pytest.mark.parametrize("text", [
    "```json\n" + GOOD + "\n```",
    "```\n" + GOOD + "```",
    "Sure! Here you go: " + GOOD + " Hope that helps {not json}",
    "{broken " + GOOD,
])
def test_parse_tolerates_wrapping(text):
    assert failures(text) == [None, None]


@pytest.mark.parametrize("text", ["", "   ", "no idea", "{}", '{"results": 3}', "[1, 2]", '{"results": [}'])
def test_unusable_reply_is_all_malformed(text):
    assert failures(text) == ["malformed", "malformed"]


@pytest.mark.parametrize("depth", [1500, 100_000])
def test_deeply_nested_reply_is_malformed_not_a_crash(depth):
    assert failures('{"results":' + "[" * depth) == ["malformed", "malformed"]
    assert failures('{"results": [' + "[" * depth + "]" * depth + "]}") in (
        ["missing", "missing"], ["malformed", "malformed"])


def test_non_string_reply_is_malformed():
    assert failures(None) == ["malformed", "malformed"]


def test_missing_id():
    assert failures(reply(entry("door"))) == [None, "missing"]
    assert failures(reply()) == ["missing", "missing"]


def test_extra_ids_ignored():
    assert failures(reply(entry("door"), entry("fall"), entry("cat"), {"id": ["x"]}, "junk")) == [None, None]


def test_duplicate_ids():
    assert failures(reply(entry("door"), entry("door"), entry("fall"))) == [None, None]
    assert failures(reply(entry("door"), entry("door", "no"), entry("fall"))) == ["malformed", None]


@pytest.mark.parametrize("bad", [
    {"answer": True}, {"answer": "maybe"}, {"answer": None}, {"confidence": "0.8"}, {"confidence": True},
    {"confidence": None}, {"confidence": 1.01}, {"confidence": -0.1}, {"reason": 5},
])
def test_wrong_types_and_ranges_are_malformed(bad):
    assert failures(reply({**entry("door"), **bad}, entry("fall"))) == ["malformed", None]


def test_nan_confidence_is_malformed():
    text = '{"results": [{"id": "door", "answer": "yes", "confidence": NaN, "reason": ""}]}'
    assert failures(text) == ["malformed", "missing"]


def test_lenient_on_case_missing_reason_and_long_reason():
    door, fall = parse_reply(reply({"id": "door", "answer": " YES ", "confidence": 1},
                                   entry("fall", reason="x" * 500)), REQ, "j")
    assert (door.answer, door.confidence, door.reason) == ("yes", 1.0, "")
    assert len(fall.reason) == 160


def test_low_confidence_is_passed_through():
    door, _ = parse_reply(reply(entry("door", confidence=0.05), entry("fall")), REQ, "j")
    assert (door.answer, door.confidence, door.failure) == ("yes", 0.05, None)


@given(st.text(max_size=300))
def test_parse_never_raises_and_answers_every_id(text):
    obs = parse_reply(text, REQ, "j")
    assert [o.condition_id for o in obs] == ["door", "fall"]


def test_failed_observations():
    obs = failed_observations(REQ, "budget", "j")
    assert [(o.failure, o.answer, o.confidence) for o in obs] == [("budget", "unknown", 0.0)] * 2


def test_extract_json_skips_objects_without_results():
    assert extract_json('{"a": 1} then {"results": []}') == {"results": []}
    assert extract_json("nothing") is None


# --------------------------------------------------------------------------- cost


def test_cost_of():
    assert cost_of(PRICE, 1_000_000, 1_000_000) == pytest.approx(3.0)
    assert cost_of(PRICE, 0, 0) == 0


def test_worst_case_cost_covers_a_typical_reply():
    req = make_request(n_images=4)
    p = build_prompt(req)
    typical_in = len(req.images) * PRICE.tokens_per_image + (len(p.system) + len(p.user)) // 4
    typical_out = len(GOOD) // 4
    assert worst_case_cost(req, PRICE) >= cost_of(PRICE, typical_in, typical_out)
    assert worst_case_cost(req, PRICE) >= cost_of(PRICE, typical_in, req.max_output_tokens)
    one = worst_case_cost(req, PRICE, retries=0)
    assert worst_case_cost(req, PRICE) == pytest.approx(2 * one)  # default mirrors policy.call(retries=1)
    assert worst_case_cost(req, PRICE, retries=2) == pytest.approx(3 * one)


def test_worst_case_counts_non_ascii_as_one_token_per_character():
    def tokens(text):
        req = JudgeRequest("r", "c", 0, (JudgeItem(Condition("x", text)),), (np.zeros((4, 4, 3), np.uint8),),
                           (0.0, 0.0))
        return worst_case_tokens(req, PRICE)[0]
    assert tokens("火" * 300) - tokens("a") >= 299
    assert tokens("a" * 301) - tokens("a") <= 100


# --------------------------------------------------------------------------- usage


@given(st.integers(0, 5), st.integers(0, 10**6), st.integers(0, 10**5), st.integers(0, 3))
def test_attribute_usage_sums_exactly(attempts, tin, tout, extra):
    obs = parse_reply(GOOD, REQ, "j")
    usage = (RequestUsage("r-9", "j", Usage(attempts, tin, tout, 0.01, 1.5)),
             RequestUsage("r-9", "j", Usage(extra, 7, 3, 0.002, 0.5)))
    shares = attribute_usage(JudgeResult(obs, usage))
    assert len(shares) == 2
    total = sum(shares, Usage())
    assert (total.attempts, total.tokens_in, total.tokens_out) == (attempts + extra, tin + 7, tout + 3)
    assert total.cost_usd == pytest.approx(0.012)
    assert shares == attribute_usage(JudgeResult(obs, usage))


def test_attribute_usage_without_usage_is_zero():
    obs = parse_reply(GOOD, REQ, "j")
    assert attribute_usage(JudgeResult(obs, ())) == (Usage(), Usage())
