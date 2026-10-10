import asyncio

import numpy as np
import pytest

from percept.contracts import Condition, Judge, JudgeItem, JudgeRequest, JudgeResult, PriceModel
from percept.judges import FakeJudge, cascade
from percept.judges.base import RetryableError, worst_case_cost
from judge_conformance import assert_conformant
from test_fake_judge import fake_for

CHEAP = PriceModel(usd_per_mtok_in=0.1, usd_per_mtok_out=0.4, tokens_per_image=258)   # test values only
STRONG = PriceModel(usd_per_mtok_in=3.0, usd_per_mtok_out=15.0, tokens_per_image=1600)


def make_request(*conditions, purpose="detect", deadline_s=8.0, request_id="r1"):
    conditions = conditions or (Condition("a", "cond a"), Condition("b", "cond b"), Condition("c", "cond c"))
    images = (np.zeros((24, 32, 3), np.uint8),)
    return JudgeRequest(request_id, "cam", 2, tuple(JudgeItem(c) for c in conditions), images, (5.0, 6.0),
                        purpose=purpose, deadline_s=deadline_s)


def asked(judge):
    """The condition ids of each provider request a fake judge made."""
    return [tuple(i.condition.id for i in request.items) for _, _, _, request in judge.sent]


def run(judge, request):
    return asyncio.run(judge.judge(request))


def test_settled_answers_stand_and_the_rest_escalate():
    fast = FakeJudge({"a": ("yes", 0.9, "sure"), "b": ("unknown", 0.2, "?"), "c": ("no", 0.4, "weak")}, name="fast")
    strong = FakeJudge({"b": ("no", 0.95, "clear"), "c": ("yes", 0.85, "clear")}, name="strong")
    result = run(cascade(fast, strong), make_request())
    assert [(o.condition_id, o.answer, o.judge) for o in result.observations] == [
        ("a", "yes", "fast"), ("b", "no", "strong"), ("c", "yes", "strong")]
    assert asked(fast) == [("a", "b", "c")] and asked(strong) == [("b", "c")]


def test_every_hop_reports_usage():
    fast = FakeJudge({"a": ("unknown", 0.1, "")}, name="fast", price=CHEAP)
    flaky = FakeJudge(RetryableError("429"), name="mid", price=CHEAP)
    strong = FakeJudge({"a": ("yes", 0.9, "")}, name="strong", price=STRONG)
    result = run(cascade(fast, flaky, strong), make_request(Condition("a", "cond a")))
    assert [ru.judge for ru in result.usage] == ["fast", "mid", "strong"]
    assert [ru.usage.attempts for ru in result.usage] == [1, 2, 1]
    assert sum(ru.usage.attempts for ru in result.usage) == len(fast.sent) + len(flaky.sent) + len(strong.sent)
    assert result.observations[0].answer == "yes"


def test_failures_escalate_and_a_real_answer_beats_a_later_failure():
    fast = FakeJudge({"a": ("no", 0.3, "weak")}, name="fast")       # "b" missing: a failure
    strong = FakeJudge(TimeoutError(), name="strong")
    result = run(cascade(fast, strong), make_request(Condition("a", "cond a"), Condition("b", "cond b")))
    a, b = result.observations
    assert (a.answer, a.confidence, a.failure, a.judge) == ("no", 0.3, None, "fast")   # the last real answer
    assert (b.failure, b.judge) == ("timeout", "strong")                              # the last failure
    assert asked(strong) == [("a", "b")]


def test_thresholds_are_per_judge_and_default_to_min_confidence():
    def fast():
        return FakeJudge({"a": ("yes", 0.7, ""), "b": ("yes", 0.7, "")}, name="fast")
    strong = FakeJudge({"a": ("no", 0.9, ""), "b": ("no", 0.9, "")}, name="strong")
    conditions = (Condition("a", "cond a", min_confidence=0.6), Condition("b", "cond b", min_confidence=0.8))
    by_default = run(cascade(fast(), strong), make_request(*conditions))
    assert [o.judge for o in by_default.observations] == ["fast", "strong"]
    strict = run(cascade(fast(), strong, thresholds=[0.75, None]), make_request(*conditions))
    assert [o.judge for o in strict.observations] == ["strong", "strong"]


def test_a_pinned_condition_goes_only_to_its_judge():
    fast = FakeJudge({"a": ("unknown", 0.1, ""), "b": ("unknown", 0.1, "")}, name="fast")
    strong = FakeJudge({"a": ("yes", 0.9, ""), "b": ("yes", 0.9, "")}, name="strong")
    conditions = (Condition("a", "cond a", judge="strong"), Condition("b", "cond b", judge="fast"),
                  Condition("c", "cond c", judge="nobody"))
    result = run(cascade(fast, strong), make_request(*conditions))
    a, b, c = result.observations
    assert (a.answer, a.judge) == ("yes", "strong")
    assert (b.answer, b.judge, b.failure) == ("unknown", "fast", None)   # pinned: no escalation
    assert c.failure == "unavailable" and "nobody" in c.reason
    assert asked(fast) == [("b",)] and asked(strong) == [("a",)]


def test_verification_goes_straight_to_the_strongest_judge():
    fast = FakeJudge({"a": ("yes", 0.9, "")}, name="fast")
    strong = FakeJudge({"a": ("yes", 0.95, ""), "b": ("no", 0.9, "")}, name="strong")
    conditions = (Condition("a", "cond a", kind="event"), Condition("b", "cond b", judge="strong"))
    result = run(cascade(fast, strong), make_request(*conditions, purpose="verify"))
    assert [o.judge for o in result.observations] == ["strong", "strong"]
    assert all(o.purpose == "verify" for o in result.observations)
    assert fast.sent == [] and asked(strong) == [("a", "b")]


def test_hops_share_the_deadline():
    fast = FakeJudge({"a": ("unknown", 0.1, "")}, name="fast")
    strong = FakeJudge({"a": ("yes", 0.9, "")}, name="strong")
    clock_t = [0.0]  # the cascade's clock; the first hop takes 0.15 s of a 0.1 s deadline

    async def slow_judge(request):
        clock_t[0] += 0.15
        return await fast.judge(request)
    fast_proxy = type("P", (), {"name": "fast", "can_batch": True, "price": None, "judge": staticmethod(slow_judge)})()
    judge = cascade(fast_proxy, strong)
    judge._clock = lambda: clock_t[0]
    result = run(judge, make_request(Condition("a", "cond a"), deadline_s=0.1))
    ob = result.observations[0]
    assert (ob.answer, ob.judge, ob.failure) == ("unknown", "fast", None)
    assert strong.sent == []   # no time left for the second hop
    _, _, _, sent = fast.sent[0]
    assert sent.deadline_s == pytest.approx(0.1)


def test_one_observation_per_item_even_when_a_member_misbehaves():
    class Broken:
        name, can_batch, price = "broken", True, None

        async def judge(self, request):
            raise RuntimeError("bug")

    class Short:
        name, can_batch, price = "short", True, None

        async def judge(self, request):
            return JudgeResult((), ())

    for member in (Broken(), Short()):
        result = run(cascade(member), make_request())
        assert [o.condition_id for o in result.observations] == ["a", "b", "c"]
        assert all(o.failed for o in result.observations)


def test_price_batching_and_name():
    fast = FakeJudge({}, name="fast", price=CHEAP)
    strong = FakeJudge({}, name="strong", price=STRONG, can_batch=False)
    both = cascade(fast, strong)
    assert isinstance(both, Judge) and both.name == "cascade(fast>strong)" and both.hops == 2
    assert both.can_batch is False and cascade(fast).can_batch is True
    assert cascade(fast, FakeJudge({}, name="free")).price is None
    request = make_request()
    assert worst_case_cost(request, both.price) >= worst_case_cost(request, CHEAP) + worst_case_cost(request, STRONG)


def test_spend_stays_within_the_reservation():
    fast = FakeJudge({"a": ("unknown", 0.1, "")}, name="fast", price=CHEAP)
    strong = FakeJudge(RetryableError("429"), name="strong", price=STRONG)
    judge = cascade(fast, strong)
    request = make_request(Condition("a", "cond a"))
    result = run(judge, request)
    assert sum(ru.usage.cost_usd for ru in result.usage) <= worst_case_cost(request, judge.price)


@pytest.mark.parametrize("bad", [dict(judges=()), dict(judges=("x", "x")), dict(thresholds=[0.5]),
                                 dict(thresholds=[0.5, 1.5])])
def test_bad_construction(bad):
    judges = [FakeJudge({}, name=n) for n in bad.get("judges", ("x", "y"))]
    with pytest.raises(ValueError):
        cascade(*judges, thresholds=bad.get("thresholds"))


@pytest.mark.parametrize("price", [None, CHEAP])
def test_a_single_member_cascade_is_conformant(price):
    # The suite requires observations and usage to carry the judge's own name, so the cascade
    # takes its member's name here.
    assert_conformant(lambda scenario: cascade(fake_for(scenario, price=price), name="fake"))
