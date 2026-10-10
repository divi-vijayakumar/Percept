import asyncio
import dataclasses

import numpy as np
import pytest
from judge_conformance import ANSWERS, SCENARIOS, assert_conformant

import percept
from percept.contracts import Condition, Judge, JudgeItem, JudgeRequest, JudgeResult, PriceModel, Usage
from percept.judges import FakeJudge, policy
from percept.judges.base import Prompt, RefusalError, RetryableError, attribute_usage, worst_case_cost

PRICE = PriceModel(usd_per_mtok_in=0.3, usd_per_mtok_out=2.5, tokens_per_image=258)


def fake_for(scenario, *, can_batch=True, price=None):
    steps = iter(scenario.steps)
    last = [None]

    def script(request):
        kind, payload = next(steps, last[0])
        last[0] = (kind, payload)
        return {"reply": payload, "retryable": RetryableError("429"), "refusal": RefusalError("blocked"),
                "error": ValueError("400 bad request"), "hang": payload}[kind]

    hang = scenario.steps[0][0] == "hang"
    return FakeJudge(script, can_batch=can_batch, price=price, latency_s=1e9 if hang else 0.0)


def test_fake_judge_is_conformant():
    assert_conformant(fake_for)


def test_fake_judge_without_batching_is_conformant():
    assert_conformant(lambda s: fake_for(s, can_batch=False))


def test_fake_judge_with_price_is_conformant():
    assert_conformant(lambda s: fake_for(s, price=PRICE))


def test_scenarios_cover_every_failure_kind_but_budget():
    kinds = {f for s in SCENARIOS for f in s.expect.values()}
    assert kinds == {None, "timeout", "unavailable", "malformed", "missing", "refused"}


def request(*ids, deadline_s=8.0):
    items = tuple(JudgeItem(Condition(i, f"condition {i}")) for i in ids)
    return JudgeRequest("r-1", "cam", 0, items, (np.zeros((8, 8, 3), np.uint8),), (0.0, 1.0), deadline_s=deadline_s)


def test_satisfies_protocol_and_is_reachable_from_percept():
    assert isinstance(FakeJudge({}), Judge)
    assert percept.judges.FakeJudge is FakeJudge


def test_dict_script_answers_and_missing_ids():
    judge = FakeJudge({"a": ("yes", 0.9, "seen")})
    result = asyncio.run(judge.judge(request("a", "b")))
    a, b = result.observations
    assert (a.answer, a.confidence, a.reason, a.failure) == ("yes", 0.9, "seen", None)
    assert b.failure == "missing"
    assert result.usage[0].failure is None


def test_unbatched_judge_sends_one_request_per_item():
    judge = FakeJudge({"a": ("yes", 0.9, ""), "b": ("no", 0.8, "")}, can_batch=False)
    result = asyncio.run(judge.judge(request("a", "b")))
    assert [len(sent[3].items) for sent in judge.sent] == [1, 1]
    assert [o.answer for o in result.observations] == ["yes", "no"]
    assert [ru.request_id for ru in result.usage] == ["r-1/a", "r-1/b"]


def test_unbatched_items_after_the_deadline_time_out_without_sending():
    judge = FakeJudge({"a": ("yes", 0.9, ""), "b": ("no", 0.8, "")}, can_batch=False, latency_s=5.0)
    result = asyncio.run(judge.judge(request("a", "b", deadline_s=8.0)))
    assert [o.failure for o in result.observations] == [None, "timeout"]
    assert len(judge.sent) == 2  # the second item was sent and its reply came back too late
    judge = FakeJudge({"a": ("yes", 0.9, "")}, can_batch=False, latency_s=9.0)
    result = asyncio.run(judge.judge(request("a", "b", deadline_s=8.0)))
    assert [o.failure for o in result.observations] == ["timeout", "timeout"]
    assert len(judge.sent) == 1 and len(result.usage) == 1


def test_latency_past_deadline_is_a_timeout_and_virtual():
    judge = FakeJudge({"a": ("yes", 0.9, "")}, latency_s=30.0)
    result = asyncio.run(judge.judge(request("a")))
    assert result.observations[0].failure == "timeout"
    assert result.usage[0].failure == "timeout"


def test_retryable_then_success():
    replies = iter([RetryableError("503"), {"a": ("yes", 0.7, "")}])
    judge = FakeJudge(lambda r: next(replies))
    result = asyncio.run(judge.judge(request("a")))
    assert result.observations[0].answer == "yes"
    assert result.usage[0].usage.attempts == 2


def test_deterministic():
    def run():
        judge = FakeJudge(lambda r: RetryableError(), price=PRICE)
        return asyncio.run(judge.judge(request("a")))
    assert run() == run()


def test_reported_cost_is_within_the_worst_case_reservation():
    req = JudgeRequest("r-1", "cam", 0, tuple(JudgeItem(Condition(c, ANSWERS[c][2])) for c in ANSWERS),
                       tuple(np.zeros((8, 8, 3), np.uint8) for _ in range(4)), (0.0, 1.0))
    replies = iter([RetryableError("429"), dict(ANSWERS)])
    judge = FakeJudge(lambda r: next(replies), price=PRICE, retries=1)
    result = asyncio.run(judge.judge(req))
    actual = result.usage[0].usage
    assert actual.attempts == 2 and result.observations[0].failure is None
    assert 0 < actual.cost_usd <= worst_case_cost(req, PRICE, retries=judge.retries)
    shares = attribute_usage(result)
    assert sum(s.tokens_in for s in shares) == actual.tokens_in
    assert sum(s.cost_usd for s in shares) == pytest.approx(actual.cost_usd)


# --------------------------------------------------------------------------- broken adapters fail the suite


class NoImages(FakeJudge):
    async def _send(self, prompt, images, max_output_tokens, request):
        return await super()._send(prompt, (), max_output_tokens, request)


class ReversedImages(FakeJudge):
    async def _send(self, prompt, images, max_output_tokens, request):
        return await super()._send(prompt, images[::-1], max_output_tokens, request)


class OwnTokenLimit(FakeJudge):
    async def _send(self, prompt, images, max_output_tokens, request):
        return await super()._send(prompt, images, 4096, request)


class OwnPrompt(FakeJudge):
    async def _send(self, prompt, images, max_output_tokens, request):
        return await super()._send(Prompt("judge these", prompt.user), images, max_output_tokens, request)


class ZeroUsage(FakeJudge):
    async def judge(self, request):
        result = await super().judge(request)
        usage = tuple(dataclasses.replace(ru, usage=Usage(attempts=ru.usage.attempts)) for ru in result.usage)
        return JudgeResult(result.observations, usage)


class OwnDeadline(FakeJudge):
    """Runs on real time with its own deadline, longer than the request's."""

    hang = False

    async def judge(self, request):
        async def send(prompt, images, max_output_tokens, sub):
            if self.hang:
                await asyncio.sleep(60)
            return await self._send(prompt, images, max_output_tokens, sub)
        return await policy.call(send, request, judge_name=self.name, deadline_s=request.deadline_s + 0.6)


def broken(cls, scenario, **kw):
    judge = fake_for(scenario, **kw)
    bad = cls(judge._script, price=judge.price, latency_s=judge.latency_s)
    if cls is OwnDeadline:
        bad.latency_s, bad.hang = 0.0, judge.latency_s > 0
    return bad


@pytest.mark.parametrize("cls, why", [(NoImages, "every image"), (ReversedImages, "request order"),
                                      (OwnTokenLimit, "max_output_tokens"), (OwnPrompt, "build_prompt"),
                                      (OwnDeadline, "took")])
def test_broken_adapters_fail_the_suite(cls, why):
    with pytest.raises(AssertionError, match=why):
        assert_conformant(lambda s: broken(cls, s))


def test_zero_usage_fails_the_suite_when_priced():
    with pytest.raises(AssertionError, match="billed tokens"):
        assert_conformant(lambda s: broken(ZeroUsage, s, price=PRICE))
