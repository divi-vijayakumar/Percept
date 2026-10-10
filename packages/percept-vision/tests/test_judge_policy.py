import asyncio
import io
import json
import time

import numpy as np
import pytest
from PIL import Image

from percept.contracts import Condition, JudgeItem, JudgeRequest, PriceModel
from percept.judges import policy
from percept.judges.base import RawReply, RefusalError, RetryableError, attribute_usage, cost_of
from percept.judges.encode import to_jpeg
from percept.judges.fake import VirtualClock

PRICE = PriceModel(usd_per_mtok_in=1.0, usd_per_mtok_out=4.0, tokens_per_image=100)
OK = json.dumps({"results": [{"id": "a", "answer": "yes", "confidence": 0.9, "reason": "r"},
                             {"id": "b", "answer": "no", "confidence": 0.8, "reason": "r"}]})


def make_request(deadline_s=8.0, image=None):
    items = (JudgeItem(Condition("a", "cond a")), JudgeItem(Condition("b", "cond b")))
    image = np.zeros((480, 640, 3), np.uint8) if image is None else image
    return JudgeRequest("r-1", "cam", 1, items, (image, image), (0.0, 1.0), deadline_s=deadline_s)


class Provider:
    """A scripted ``send``: each step is a RawReply, an exception, or a (delay, step) pair."""

    def __init__(self, *steps, clock=None):
        self.steps = list(steps)
        self.clock = clock or VirtualClock()
        self.calls = []

    async def __call__(self, prompt, images, max_output_tokens, request):
        self.calls.append((prompt, images, max_output_tokens, request))
        step = self.steps.pop(0) if len(self.steps) > 1 else self.steps[0]
        if isinstance(step, tuple):
            delay, step = step
            await self.clock.sleep(delay)
        if isinstance(step, BaseException):
            raise step
        return step


def run(provider, req=None, **kw):
    req = req or make_request()
    kw.setdefault("sleep", provider.clock.sleep)
    kw.setdefault("clock", provider.clock)
    kw.setdefault("rng", lambda: 0.0)
    return asyncio.run(policy.call(provider, req, judge_name="j", **kw))


def kinds(result):
    return [o.failure for o in result.observations]


def test_success_passes_prompt_jpegs_and_usage():
    provider = Provider((1.5, RawReply(OK, 1000, 50)))
    result = run(provider, price=PRICE)
    assert kinds(result) == [None, None]
    prompt, images, max_tokens, request = provider.calls[0]
    assert "cond a" in prompt.user and max_tokens == 400 and request.request_id == "r-1"
    assert len(images) == 2 and all(i[:2] == b"\xff\xd8" for i in images)
    (ru,) = result.usage
    assert (ru.request_id, ru.judge, ru.failure) == ("r-1", "j", None)
    assert ru.usage.attempts == 1 and (ru.usage.tokens_in, ru.usage.tokens_out) == (1000, 50)
    assert ru.usage.cost_usd == pytest.approx(cost_of(PRICE, 1000, 50))
    assert ru.usage.latency_s == 1.5


def test_no_price_means_zero_cost():
    assert run(Provider(RawReply(OK, 1000, 50))).usage[0].usage.cost_usd == 0


def test_retryable_then_success_counts_both_attempts():
    provider = Provider(RetryableError("429"), RawReply(OK, 10, 5))
    result = run(provider)
    assert kinds(result) == [None, None]
    assert result.usage[0].usage.attempts == 2
    assert provider.clock() == pytest.approx(policy.BACKOFF_S)


def test_backoff_is_jittered():
    provider = Provider(RetryableError(), RawReply(OK))
    run(provider, rng=lambda: 1.0)
    assert provider.clock() == pytest.approx(2 * policy.BACKOFF_S)


def test_retryable_exhausted_is_unavailable():
    result = run(Provider(RetryableError("503")))
    assert kinds(result) == ["unavailable"] * 2
    assert result.usage[0].usage.attempts == 2 and result.usage[0].failure == "unavailable"


def test_retries_zero_means_one_attempt():
    result = run(Provider(RetryableError()), retries=0)
    assert result.usage[0].usage.attempts == 1 and kinds(result) == ["unavailable"] * 2


def test_no_retry_past_the_deadline():
    provider = Provider(RetryableError("429", retry_after_s=30.0), RawReply(OK))
    result = run(provider)
    assert kinds(result) == ["unavailable"] * 2 and len(provider.calls) == 1


def test_retry_skipped_for_the_deadline_says_so():
    result = run(Provider(RetryableError("429", retry_after_s=30.0)))
    assert result.observations[0].reason.startswith("no time to retry before the deadline")
    assert run(Provider(RetryableError("503"))).observations[0].reason.startswith("retries exhausted")


def test_provider_retry_after_is_honoured():
    provider = Provider(RetryableError("429", retry_after_s=2.0), RawReply(OK))
    assert kinds(run(provider)) == [None, None] and provider.clock() == 2.0


def test_refusal():
    result = run(Provider(RefusalError("safety")))
    assert kinds(result) == ["refused"] * 2 and result.usage[0].failure == "refused"
    assert "safety" in result.observations[0].reason


@pytest.mark.parametrize("exc", [ValueError("400"), KeyError("x"), RuntimeError()])
def test_other_errors_are_unavailable(exc):
    result = run(Provider(exc))
    assert kinds(result) == ["unavailable"] * 2 and result.usage[0].usage.attempts == 1


def test_transport_timeout_is_timeout_not_retried():
    provider = Provider(TimeoutError("read timed out"), RawReply(OK))
    assert kinds(run(provider)) == ["timeout"] * 2 and len(provider.calls) == 1


def test_late_reply_is_a_timeout_with_usage_kept():
    result = run(Provider((9.0, RawReply(OK, 100, 10))), price=PRICE)
    assert kinds(result) == ["timeout"] * 2
    assert result.usage[0].usage.tokens_in == 100 and result.usage[0].usage.cost_usd > 0


def test_real_hang_is_cut_at_the_deadline():
    async def hang(*_):
        await asyncio.sleep(60)

    start = time.monotonic()
    result = asyncio.run(policy.call(hang, make_request(deadline_s=0.05), judge_name="j"))
    assert kinds(result) == ["timeout"] * 2
    assert time.monotonic() - start < 1.0


def test_deadline_override():
    assert kinds(run(Provider((3.0, RawReply(OK))), deadline_s=2.0)) == ["timeout"] * 2


def test_bad_reply_object_is_unavailable():
    assert kinds(run(Provider(None))) == ["unavailable"] * 2


def test_deeply_nested_reply_does_not_raise():
    result = run(Provider(RawReply('{"results":' + "[" * 100_000)))
    assert kinds(result) == ["malformed"] * 2


def test_a_parser_crash_is_malformed(monkeypatch):
    def boom(*_):
        raise RecursionError("deep")
    monkeypatch.setattr(policy, "parse_reply", boom)
    result = run(Provider(RawReply(OK)))
    assert kinds(result) == ["malformed"] * 2 and result.usage[0].failure == "malformed"


def test_unbatched_sub_requests_have_their_own_ids():
    provider = Provider(RawReply(OK, 10, 4))
    result = run(provider, batch=False)
    assert [c[3].request_id for c in provider.calls] == ["r-1/a", "r-1/b"]
    assert [o.request_id for o in result.observations] == ["r-1/a", "r-1/b"]
    assert [ru.request_id for ru in result.usage] == ["r-1/a", "r-1/b"]
    assert [s.tokens_in for s in attribute_usage(result)] == [10, 10]


def test_malformed_reply_marks_request_failure():
    result = run(Provider(RawReply("sorry")))
    assert kinds(result) == ["malformed"] * 2 and result.usage[0].failure == "malformed"


def test_partial_reply_has_no_request_failure():
    text = json.dumps({"results": [{"id": "a", "answer": "yes", "confidence": 0.9, "reason": ""}]})
    result = run(Provider(RawReply(text)))
    assert kinds(result) == [None, "missing"] and result.usage[0].failure is None


def test_unencodable_image_fails_without_sending():
    provider = Provider(RawReply(OK))
    result = run(provider, make_request(image=np.zeros((4, 4, 3), np.float32)))
    assert kinds(result) == ["unavailable"] * 2 and result.usage == () and not provider.calls


def test_cancellation_propagates():
    async def main():
        task = asyncio.create_task(policy.call(Provider(RawReply(OK)), make_request(), judge_name="j"))
        task.cancel()
        await task
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(main())


# --------------------------------------------------------------------------- encode


def decode(data):
    return Image.open(io.BytesIO(data))


def test_encode_caps_longest_side_and_converts_bgr():
    bgr = np.zeros((960, 1280, 3), np.uint8)
    bgr[..., 0] = 255  # pure blue in BGR
    img = decode(to_jpeg(bgr))
    assert img.size == (640, 480) and img.mode == "RGB"
    r, g, b = img.getpixel((320, 240))
    assert b > 200 and r < 50


def test_encode_keeps_small_images_and_grayscale():
    img = decode(to_jpeg(np.full((30, 40), 128, np.uint8), max_side=640, quality=90))
    assert img.size == (40, 30) and img.mode == "L"
    assert decode(to_jpeg(np.zeros((100, 50, 3), np.uint8), max_side=20)).size == (10, 20)


def test_encode_rejects_bad_arrays():
    with pytest.raises(ValueError):
        to_jpeg(np.zeros((4, 4, 4), np.uint8))
