"""A judge backed by Google's Gemini API, through the ``google-genai`` SDK.

Install with ``pip install 'percept-vision[gemini]'``. The SDK is imported only when a client is
built, so ``import percept`` works without it.

Example::

    judge = percept.judges.gemini("<model id>", price=PriceModel(...))  # prices are yours to set
"""

from __future__ import annotations

import os
from typing import Any

from percept.contracts import JudgeRequest, PriceModel
from percept.judges.adapter import Adapter, reply_schema, require, usage_int
from percept.judges.base import Prompt, RawReply, RefusalError

# Finish reasons that mean the provider declined to answer (any prompt block reason does too).
REFUSED = frozenset({"SAFETY", "RECITATION", "BLOCKLIST", "PROHIBITED_CONTENT", "SPII", "IMAGE_SAFETY",
                     "IMAGE_PROHIBITED_CONTENT", "IMAGE_RECITATION"})


def _reason(value: Any) -> str:
    """An enum or string finish/block reason as its upper-case name ('' when unset)."""
    if value is None:
        return ""
    name = getattr(value, "name", None) or str(value)
    name = name.rsplit(".", 1)[-1].upper()
    return "" if name.endswith("UNSPECIFIED") else name


class GeminiJudge(Adapter):
    """One ``generate_content`` call per request: system instruction, the JPEGs in order, then
    the conditions; JSON output constrained to the requested ids."""

    def __init__(self, model: str, *, name: str, price: PriceModel | None, can_batch: bool, client: Any,
                 structured_output: bool, thinking_budget: int | None) -> None:
        super().__init__(name, price=price, can_batch=can_batch, client=client)
        self.model = model
        self.structured_output = structured_output
        self.thinking_budget = thinking_budget

    def build(self, prompt: Prompt, images: tuple[bytes, ...], max_output_tokens: int,
              request: JudgeRequest) -> dict:
        """The keyword arguments of the ``generate_content`` call."""
        parts: list[dict] = [{"inline_data": {"mime_type": "image/jpeg", "data": image}} for image in images]
        parts.append({"text": prompt.user})
        config: dict[str, Any] = {"system_instruction": prompt.system, "max_output_tokens": max_output_tokens,
                                  "response_mime_type": "application/json"}
        if self.structured_output:
            config["response_json_schema"] = reply_schema(request, closed=False)
        if self.thinking_budget is not None:
            config["thinking_config"] = {"thinking_budget": self.thinking_budget}
        return {"model": self.model, "contents": [{"role": "user", "parts": parts}], "config": config}

    async def _request(self, prompt: Prompt, images: tuple[bytes, ...], max_output_tokens: int,
                       request: JudgeRequest) -> RawReply:
        response = await self.client.aio.models.generate_content(
            **self.build(prompt, images, max_output_tokens, request))
        meta = getattr(response, "usage_metadata", None)
        tokens_in = usage_int(meta, "prompt_token_count")
        tokens_out = usage_int(meta, "candidates_token_count", "thoughts_token_count")  # both billed as output

        blocked = _reason(getattr(getattr(response, "prompt_feedback", None), "block_reason", None))
        if blocked:
            raise RefusalError(f"prompt blocked: {blocked}", tokens_in, tokens_out)
        candidates = getattr(response, "candidates", None) or []
        if not candidates:
            return RawReply("", tokens_in, tokens_out)
        finish = _reason(getattr(candidates[0], "finish_reason", None))
        if finish in REFUSED:
            raise RefusalError(f"finish reason {finish}", tokens_in, tokens_out)
        parts = getattr(getattr(candidates[0], "content", None), "parts", None) or []
        text = "".join(p.text for p in parts if getattr(p, "text", None) and not getattr(p, "thought", False))
        return RawReply(text, tokens_in, tokens_out, truncated=finish == "MAX_TOKENS")


def gemini(model: str, *, api_key: str | None = None, price: PriceModel | None = None, can_batch: bool = True,
           client: Any = None, name: str | None = None, structured_output: bool = True,
           thinking_budget: int | None = 0) -> GeminiJudge:
    """A Gemini judge for ``model``.

    ``api_key`` defaults to ``GEMINI_API_KEY``, then ``GOOGLE_API_KEY``. ``client`` replaces the SDK
    client (it needs ``client.aio.models.generate_content``). ``name`` (default: the model id) is
    the judge's name in observations and the route name a ``Condition.judge`` pins to.
    ``structured_output=False`` drops the response schema for models that reject one; JSON is
    still requested.

    ``thinking_budget`` caps the model's thinking tokens, which count against the request's
    ``max_output_tokens``: a model that thinks freely can spend the whole limit and return no
    answer. The default 0 turns thinking off where the model allows it. Some models cannot
    turn it off and reject 0 (they need at least 128, or a thinking level instead); pass a
    budget they accept and raise ``max_output_tokens`` to match, or None to send no setting.

    Retries are the policy's: the SDK client gets its own ``httpx`` client, which keeps the SDK
    off its ``aiohttp`` transport and that transport's internal, uncounted reconnect retry.
    """
    if client is None:
        genai = require("google.genai", "gemini")
        key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not key:
            raise ValueError("gemini judge needs an API key: pass api_key= or set GEMINI_API_KEY")
        httpx = require("httpx", "gemini")
        client = genai.Client(api_key=key, http_options={"httpx_async_client": httpx.AsyncClient()})
    return GeminiJudge(model, name=name or model, price=price, can_batch=can_batch, client=client,
                       structured_output=structured_output, thinking_budget=thinking_budget)
