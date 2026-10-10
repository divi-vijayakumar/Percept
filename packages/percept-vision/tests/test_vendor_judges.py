"""The Gemini, Claude and OpenAI-compatible adapters, offline.

Each adapter gets an injected client whose one method replays wire responses: handwritten
objects shaped like the SDK's (so these tests need no SDK), plus, when the SDK is installed,
the SDK's own response models and exception classes.
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
from types import SimpleNamespace as NS

import numpy as np
import pytest
from PIL import Image

from percept.contracts import Condition, JudgeItem, JudgeRequest, PriceModel
from percept.judges import claude, gemini, openai_compatible
from percept.judges.adapter import map_error, reply_schema, require, retry_after
from percept.judges import policy
from percept.judges.base import RetryableError, build_prompt, max_attempts
from judge_conformance import assert_conformant

PRICE = PriceModel(usd_per_mtok_in=1.0, usd_per_mtok_out=2.0, tokens_per_image=100)  # synthetic, not a vendor price
TOKENS_IN, TOKENS_OUT = 3 * 100 + 100, 80


class StatusError(Exception):
    """A provider HTTP error: ``status_code`` like anthropic/openai, or ``code`` like google-genai."""

    def __init__(self, status: int, headers: dict | None = None, attr: str = "status_code") -> None:
        super().__init__(f"HTTP {status}")
        setattr(self, attr, status)
        self.response = NS(headers=headers or {})


# --------------------------------------------------------------------------- wire formats


class GeminiWire:
    attr = "code"

    @staticmethod
    def reply(text, finish="STOP", block=None):
        return NS(usage_metadata=NS(prompt_token_count=TOKENS_IN, candidates_token_count=TOKENS_OUT - 20,
                                    thoughts_token_count=20),
                  prompt_feedback=NS(block_reason=block) if block else None,
                  candidates=[] if block else [NS(finish_reason=finish, content=NS(parts=[
                      NS(text="thinking...", thought=True), NS(text=text, thought=None)]))])

    @classmethod
    def refusal(cls):
        return cls.reply("", finish="SAFETY")

    @staticmethod
    def client(method):
        return NS(aio=NS(models=NS(generate_content=method)))

    @staticmethod
    def make(**kw):
        return gemini("test-model", **kw)


class ClaudeWire:
    attr = "status_code"

    @staticmethod
    def reply(text, stop="end_turn"):
        return NS(content=[NS(type="text", text=text)], stop_reason=stop,
                  usage=NS(input_tokens=TOKENS_IN - 10, cache_creation_input_tokens=None,
                           cache_read_input_tokens=10, output_tokens=TOKENS_OUT))

    @classmethod
    def refusal(cls):
        return cls.reply("", stop="refusal")

    @staticmethod
    def client(method):
        return NS(messages=NS(create=method))

    @staticmethod
    def make(**kw):
        return claude("test-model", **kw)


class OpenAIWire:
    attr = "status_code"

    @staticmethod
    def reply(text, refusal=None, finish="stop"):
        return NS(choices=[NS(message=NS(content=None if refusal else text, refusal=refusal), finish_reason=finish)],
                  usage=NS(prompt_tokens=TOKENS_IN, completion_tokens=TOKENS_OUT))

    @classmethod
    def refusal(cls):
        return cls.reply("", refusal="I can't help with that.")

    @staticmethod
    def client(method):
        return NS(chat=NS(completions=NS(create=method)))

    @staticmethod
    def make(**kw):
        return openai_compatible("test-model", **kw)


WIRES = [GeminiWire, ClaudeWire, OpenAIWire]


class Provider:
    """Plays conformance steps, one per call, the last repeating."""

    def __init__(self, wire, steps):
        self.wire, self.steps, self.calls = wire, steps, []

    async def __call__(self, **kwargs):
        self.calls.append(kwargs)
        kind, payload = self.steps[min(len(self.calls), len(self.steps)) - 1]
        if kind == "reply":
            return self.wire.reply(payload)
        if kind == "refusal":
            return self.wire.refusal()
        if kind == "retryable":
            raise StatusError(429, {"retry-after": "0"}, self.wire.attr)
        if kind == "error":
            raise StatusError(400, attr=self.wire.attr)
        await asyncio.Event().wait()  # hang
        raise AssertionError("unreachable")


def build(wire, steps, **kw):
    provider = Provider(wire, steps)
    judge = wire.make(client=wire.client(provider), **kw)
    judge.sent = []
    return judge, provider


# --------------------------------------------------------------------------- conformance


@pytest.mark.parametrize("wire", WIRES, ids=lambda w: w.__name__)
@pytest.mark.parametrize("batch", [True, False], ids=["batched", "unbatched"])
def test_adapters_are_conformant(wire, batch):
    assert_conformant(lambda scenario: build(wire, scenario.steps, price=PRICE, can_batch=batch)[0])


@pytest.mark.parametrize("wire", WIRES, ids=lambda w: w.__name__)
def test_adapters_are_conformant_without_a_price(wire):
    assert_conformant(lambda scenario: build(wire, scenario.steps)[0])


# --------------------------------------------------------------------------- outbound request


def request(n_images=3, max_output_tokens=123) -> JudgeRequest:
    items = (JudgeItem(Condition("red", "The image is mostly red")), JudgeItem(Condition("door", "A door is open")))
    images = tuple(np.full((32, 32, 3), 60 * i, dtype=np.uint8) for i in range(n_images))
    return JudgeRequest("r1", "cam", 0, items, images, (10.0, 12.0), max_output_tokens=max_output_tokens)


def brightness(jpeg: bytes) -> float:
    assert jpeg[:2] == b"\xff\xd8"
    return float(np.asarray(Image.open(io.BytesIO(jpeg)).convert("L")).mean())


def run(judge, req):
    return asyncio.run(judge.judge(req))


OK = json.dumps({"results": [{"id": "red", "answer": "yes", "confidence": 0.9, "reason": "red"},
                             {"id": "door", "answer": "no", "confidence": 0.8, "reason": "closed"}]})


def test_gemini_request():
    judge, provider = build(GeminiWire, [("reply", OK)])
    req = request()
    result = run(judge, req)
    assert [o.answer for o in result.observations] == ["yes", "no"]
    assert result.usage[0].usage.tokens_in == TOKENS_IN and result.usage[0].usage.tokens_out == TOKENS_OUT
    call = provider.calls[0]
    prompt = build_prompt(req)
    assert call["model"] == "test-model"
    (content,) = call["contents"]
    assert content["role"] == "user"
    images, text = content["parts"][:-1], content["parts"][-1]
    assert text == {"text": prompt.user}
    assert [i["inline_data"]["mime_type"] for i in images] == ["image/jpeg"] * 3
    levels = [brightness(i["inline_data"]["data"]) for i in images]
    assert levels == sorted(levels) and levels[0] < levels[-1]
    config = call["config"]
    assert config["system_instruction"] == prompt.system and config["max_output_tokens"] == 123
    assert config["response_mime_type"] == "application/json"
    assert config["response_json_schema"] == reply_schema(req, closed=False)
    assert config["response_json_schema"]["properties"]["results"]["items"]["properties"]["id"]["enum"] == ["red", "door"]
    assert config["thinking_config"] == {"thinking_budget": 0}   # thinking off by default


@pytest.mark.parametrize("budget,expected", [(None, None), (128, {"thinking_budget": 128})])
def test_gemini_thinking_budget(budget, expected):
    judge, provider = build(GeminiWire, [("reply", OK)], thinking_budget=budget)
    run(judge, request())
    assert provider.calls[0]["config"].get("thinking_config") == expected


def test_claude_request():
    judge, provider = build(ClaudeWire, [("reply", OK)])
    req = request()
    result = run(judge, req)
    assert [o.answer for o in result.observations] == ["yes", "no"]
    assert result.usage[0].usage.tokens_in == TOKENS_IN  # cache reads count as input
    call = provider.calls[0]
    prompt = build_prompt(req)
    assert (call["model"], call["max_tokens"], call["system"]) == ("test-model", 123, prompt.system)
    (message,) = call["messages"]
    blocks = message["content"]
    assert message["role"] == "user" and blocks[-1] == {"type": "text", "text": prompt.user}
    assert all(b["type"] == "image" and b["source"]["media_type"] == "image/jpeg" for b in blocks[:-1])
    levels = [brightness(base64.b64decode(b["source"]["data"])) for b in blocks[:-1]]
    assert len(levels) == 3 and levels == sorted(levels)
    fmt = call["output_config"]["format"]
    assert fmt["type"] == "json_schema" and fmt["schema"] == reply_schema(req, closed=True)
    assert fmt["schema"]["additionalProperties"] is False


@pytest.mark.parametrize("base_url,token_param,param", [
    (None, "auto", "max_completion_tokens"),
    ("https://api.openai.com/v1", "auto", "max_completion_tokens"),
    ("https://my-team.openai.azure.com/openai/v1", "auto", "max_completion_tokens"),
    ("https://my-team.cognitiveservices.azure.com/openai/v1", "auto", "max_completion_tokens"),
    ("http://localhost:8000/v1", "auto", "max_tokens"),
    ("http://127.0.0.1:11434/v1", "auto", "max_tokens"),
    ("http://localhost:8000/v1", "max_completion_tokens", "max_completion_tokens"),
    (None, "max_tokens", "max_tokens"),
])
def test_openai_request(base_url, token_param, param):
    provider = Provider(OpenAIWire, [("reply", OK)])
    judge = openai_compatible("m", base_url=base_url, client=OpenAIWire.client(provider), token_param=token_param)
    req = request()
    assert [o.answer for o in run(judge, req).observations] == ["yes", "no"]
    call = provider.calls[0]
    prompt = build_prompt(req)
    other = ({"max_tokens", "max_completion_tokens"} - {param}).pop()
    assert call[param] == 123 and other not in call
    system, user = call["messages"]
    assert system == {"role": "system", "content": prompt.system}
    assert user["role"] == "user" and user["content"][-1] == {"type": "text", "text": prompt.user}
    urls = [b["image_url"]["url"] for b in user["content"][:-1]]
    assert all(u.startswith("data:image/jpeg;base64,") for u in urls)
    levels = [brightness(base64.b64decode(u.split(",", 1)[1])) for u in urls]
    assert len(levels) == 3 and levels == sorted(levels)
    fmt = call["response_format"]
    assert fmt["type"] == "json_schema" and fmt["json_schema"]["strict"] is True
    assert fmt["json_schema"]["schema"] == reply_schema(req, closed=True)


@pytest.mark.parametrize("wire", WIRES, ids=lambda w: w.__name__)
def test_structured_output_can_be_turned_off(wire):
    judge, provider = build(wire, [("reply", OK)], structured_output=False)
    run(judge, request())
    call = provider.calls[0]
    assert "output_config" not in call and "response_format" not in call
    assert "response_json_schema" not in call.get("config", {})


@pytest.mark.parametrize("wire", WIRES, ids=lambda w: w.__name__)
def test_name_defaults_to_the_model_and_sent_is_off_by_default(wire):
    judge = wire.make(client=wire.client(Provider(wire, [("reply", OK)])))
    assert judge.name == "test-model" and judge.sent is None
    run(judge, request())
    assert judge.sent is None
    assert wire.make(client=object(), name="fast").name == "fast"
    assert max_attempts(judge) == judge.max_attempts == 2


def test_openai_rejects_an_unknown_token_param():
    with pytest.raises(ValueError):
        openai_compatible("m", client=object(), token_param="max_output_tokens")


# --------------------------------------------------------------------------- replies and refusals


def outcome(wire, response):
    async def method(**_):
        return response
    judge = wire.make(client=wire.client(method), price=PRICE)
    return run(judge, request())


@pytest.mark.parametrize("wire,response", [
    (GeminiWire, GeminiWire.reply("", finish="SAFETY")),
    (GeminiWire, GeminiWire.reply("", finish="PROHIBITED_CONTENT")),
    (GeminiWire, GeminiWire.reply("", block="SAFETY")),
    (GeminiWire, GeminiWire.reply("", block="OTHER")),
    (ClaudeWire, ClaudeWire.refusal()),
    (OpenAIWire, OpenAIWire.refusal()),
    (OpenAIWire, OpenAIWire.reply("", finish="content_filter")),
])
def test_refusals_fail_as_refused_and_keep_their_usage(wire, response):
    result = outcome(wire, response)
    assert {o.failure for o in result.observations} == {"refused"}
    (ru,) = result.usage
    assert ru.failure == "refused" and ru.usage.attempts == 1
    assert ru.usage.tokens_in > 0 and ru.usage.tokens_out > 0 and ru.usage.cost_usd > 0


@pytest.mark.parametrize("wire,response", [
    (GeminiWire, NS(usage_metadata=NS(prompt_token_count=50, candidates_token_count=None), prompt_feedback=None,
                    candidates=[])),
    (OpenAIWire, NS(choices=[], usage=NS(prompt_tokens=50, completion_tokens=0))),
    (OpenAIWire, OpenAIWire.reply("not json at all")),
])
def test_empty_or_truncated_replies_are_malformed_with_usage(wire, response):
    result = outcome(wire, response)
    assert {o.failure for o in result.observations} == {"malformed"}
    assert result.usage[0].usage.tokens_in > 0


@pytest.mark.parametrize("wire,response", [
    (GeminiWire, GeminiWire.reply("", finish="MAX_TOKENS")),          # thinking spent the limit
    (GeminiWire, GeminiWire.reply('{"results": [', finish="MAX_TOKENS")),
    (ClaudeWire, ClaudeWire.reply('{"results": [', stop="max_tokens")),
    (OpenAIWire, OpenAIWire.reply('{"results": [', finish="length")),
    (OpenAIWire, OpenAIWire.reply(OK, finish="length")),               # parseable, still cut off
])
def test_truncated_replies_say_so(wire, response):
    result = outcome(wire, response)
    assert {(o.failure, o.reason) for o in result.observations} == {("malformed", policy.TRUNCATED)}
    assert result.usage[0].usage.tokens_out > 0


def test_gemini_skips_thought_parts():
    result = outcome(GeminiWire, GeminiWire.reply(OK))
    assert [o.answer for o in result.observations] == ["yes", "no"]


# --------------------------------------------------------------------------- error mapping


class APITimeoutError(Exception):
    pass


class APIConnectionError(Exception):
    pass


class ReadTimeout(APIConnectionError):
    pass


@pytest.mark.parametrize("exc,kind", [
    (StatusError(429), RetryableError),
    (StatusError(408), RetryableError),
    (StatusError(500), RetryableError),
    (StatusError(503, attr="code"), RetryableError),
    (StatusError(529), RetryableError),
    (APIConnectionError("reset"), RetryableError),
    (APITimeoutError("slow"), TimeoutError),
    (ReadTimeout("slow"), TimeoutError),
    (TimeoutError(), TimeoutError),
    (StatusError(400), StatusError),
    (StatusError(401), StatusError),
    (StatusError(403, attr="code"), StatusError),
    (StatusError(404), StatusError),
    (ValueError("bad"), ValueError),
])
def test_error_mapping(exc, kind):
    assert type(map_error(exc)) is kind


@pytest.mark.parametrize("exc,seconds", [
    (StatusError(429, {"retry-after": "2"}), 2.0),
    (StatusError(429, {"retry-after-ms": "1500", "retry-after": "9"}), 1.5),
    (StatusError(429, {"retry-after": "Wed, 21 Oct 2026 07:28:00 GMT"}), None),
    (StatusError(429), None),
])
def test_retry_after_headers(exc, seconds):
    assert retry_after(exc) == seconds
    assert map_error(exc).retry_after_s == seconds


def test_retry_after_from_a_google_retry_info():
    exc = StatusError(429, attr="code")
    exc.response = None
    exc.details = {"error": {"code": 429, "details": [
        {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "13s"}]}}
    assert map_error(exc).retry_after_s == 13.0


def test_a_permanent_error_fails_without_a_retry():
    judge, provider = build(ClaudeWire, [("error", "")])
    result = run(judge, request())
    assert {o.failure for o in result.observations} == {"unavailable"}
    assert len(provider.calls) == 1 and result.usage[0].usage.attempts == 1


def test_a_transport_timeout_fails_as_timeout():
    async def method(**_):
        raise APITimeoutError("read timed out")
    judge = gemini("m", client=GeminiWire.client(method))
    assert {o.failure for o in run(judge, request()).observations} == {"timeout"}


def test_a_missing_sdk_names_the_extra():
    with pytest.raises(ImportError, match=r"percept-vision\[gemini\]"):
        require("percept_no_such_sdk", "gemini")


# --------------------------------------------------------------------------- with the real SDKs


def _httpx_for(sdk):
    """The HTTP library the SDK's exceptions are built on (newer SDKs moved to ``httpx2``)."""
    import importlib
    return importlib.import_module("httpx2" if "httpx2" in vars(sdk._exceptions) else "httpx")


@pytest.mark.parametrize("module", ["anthropic", "openai"])
def test_real_sdk_errors_map(module):
    sdk = pytest.importorskip(module)
    httpx = _httpx_for(sdk)
    req = httpx.Request("POST", "https://api.example/v1")
    limited = sdk.RateLimitError("rate limited", response=httpx.Response(429, headers={"retry-after": "3"}, request=req),
                                 body=None)
    assert isinstance(map_error(limited), RetryableError) and map_error(limited).retry_after_s == 3.0
    server = sdk.InternalServerError("boom", response=httpx.Response(500, request=req), body=None)
    assert isinstance(map_error(server), RetryableError)
    bad = sdk.BadRequestError("bad", response=httpx.Response(400, request=req), body=None)
    assert map_error(bad) is bad
    assert isinstance(map_error(sdk.APITimeoutError(request=req)), TimeoutError)
    assert isinstance(map_error(sdk.APIConnectionError(request=req)), RetryableError)


def test_real_genai_errors_map():
    errors = pytest.importorskip("google.genai.errors")
    body = {"error": {"code": 429, "status": "RESOURCE_EXHAUSTED", "message": "quota",
                      "details": [{"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "7s"}]}}
    mapped = map_error(errors.ClientError(429, body))
    assert isinstance(mapped, RetryableError) and mapped.retry_after_s == 7.0
    assert isinstance(map_error(errors.ServerError(503, {"error": {"code": 503}})), RetryableError)
    bad = errors.ClientError(400, {"error": {"code": 400, "message": "bad"}})
    assert map_error(bad) is bad


def _replay(wire, response):
    async def method(**_):
        return response
    return run(wire.make(client=wire.client(method), price=PRICE), request())


def test_real_gemini_response_model():
    types = pytest.importorskip("google.genai.types")
    response = types.GenerateContentResponse.model_validate({
        "candidates": [{"content": {"role": "model", "parts": [{"text": OK}]}, "finishReason": "STOP"}],
        "usageMetadata": {"promptTokenCount": 900, "candidatesTokenCount": 40, "thoughtsTokenCount": 15}})
    result = _replay(GeminiWire, response)
    assert [o.answer for o in result.observations] == ["yes", "no"]
    assert (result.usage[0].usage.tokens_in, result.usage[0].usage.tokens_out) == (900, 55)
    blocked = types.GenerateContentResponse.model_validate({
        "promptFeedback": {"blockReason": "SAFETY"}, "usageMetadata": {"promptTokenCount": 900}})
    assert {o.failure for o in _replay(GeminiWire, blocked).observations} == {"refused"}


def test_real_claude_response_model():
    types = pytest.importorskip("anthropic.types")
    wire = {"id": "msg_1", "type": "message", "role": "assistant", "model": "test-model",
            "content": [{"type": "text", "text": OK}], "stop_reason": "end_turn", "stop_sequence": None,
            "usage": {"input_tokens": 900, "output_tokens": 40}}
    result = _replay(ClaudeWire, types.Message.model_validate(wire))
    assert [o.answer for o in result.observations] == ["yes", "no"]
    assert (result.usage[0].usage.tokens_in, result.usage[0].usage.tokens_out) == (900, 40)
    refused = types.Message.model_validate({**wire, "content": [], "stop_reason": "refusal"})
    assert {o.failure for o in _replay(ClaudeWire, refused).observations} == {"refused"}


def test_real_openai_response_model():
    chat = pytest.importorskip("openai.types.chat")
    wire = {"id": "c1", "object": "chat.completion", "created": 0, "model": "test-model",
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": OK, "refusal": None}}],
            "usage": {"prompt_tokens": 900, "completion_tokens": 40, "total_tokens": 940}}
    result = _replay(OpenAIWire, chat.ChatCompletion.model_validate(wire))
    assert [o.answer for o in result.observations] == ["yes", "no"]
    assert (result.usage[0].usage.tokens_in, result.usage[0].usage.tokens_out) == (900, 40)


def test_real_clients_are_built_without_sdk_retries(monkeypatch):
    pytest.importorskip("anthropic")
    pytest.importorskip("openai")
    pytest.importorskip("google.genai")
    assert claude("m", api_key="test").client.max_retries == 0
    local = openai_compatible("m", base_url="http://localhost:11434/v1")
    assert local.client.max_retries == 0 and local.token_param == "max_tokens"
    import httpx
    pinned = gemini("m", api_key="test").client
    assert isinstance(pinned._api_client._async_httpx_client, httpx.AsyncClient)  # not aiohttp
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    with pytest.raises(ValueError, match="GEMINI_API_KEY"):
        gemini("m")
