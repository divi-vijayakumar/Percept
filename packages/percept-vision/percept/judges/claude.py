"""A judge backed by Anthropic's Messages API, through the ``anthropic`` SDK.

Install with ``pip install 'percept-vision[claude]'``. The SDK is imported only when a client is
built, so ``import percept`` works without it. There is no default model: pass the model id.

Example::

    judge = percept.judges.claude("<model id>", price=PriceModel(...))  # prices are yours to set
"""

from __future__ import annotations

import base64
from typing import Any

from percept.contracts import JudgeRequest, PriceModel
from percept.judges.adapter import Adapter, reply_schema, require, usage_int
from percept.judges.base import Prompt, RawReply, RefusalError


class ClaudeJudge(Adapter):
    """One ``messages.create`` call per request: the system prompt, then a user turn holding the
    JPEGs in order followed by the conditions; JSON output constrained to the requested ids."""

    def __init__(self, model: str, *, name: str, price: PriceModel | None, can_batch: bool, client: Any,
                 structured_output: bool) -> None:
        super().__init__(name, price=price, can_batch=can_batch, client=client)
        self.model = model
        self.structured_output = structured_output

    def build(self, prompt: Prompt, images: tuple[bytes, ...], max_output_tokens: int,
              request: JudgeRequest) -> dict:
        """The keyword arguments of the ``messages.create`` call."""
        content: list[dict] = [
            {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                         "data": base64.b64encode(image).decode("ascii")}}
            for image in images]
        content.append({"type": "text", "text": prompt.user})
        kwargs: dict[str, Any] = {"model": self.model, "max_tokens": max_output_tokens, "system": prompt.system,
                                  "messages": [{"role": "user", "content": content}]}
        if self.structured_output:
            kwargs["output_config"] = {"format": {"type": "json_schema",
                                                  "schema": reply_schema(request, closed=True)}}
        return kwargs

    async def _request(self, prompt: Prompt, images: tuple[bytes, ...], max_output_tokens: int,
                       request: JudgeRequest) -> RawReply:
        response = await self.client.messages.create(**self.build(prompt, images, max_output_tokens, request))
        usage = getattr(response, "usage", None)
        tokens_in = usage_int(usage, "input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")
        tokens_out = usage_int(usage, "output_tokens")
        if getattr(response, "stop_reason", None) == "refusal":
            raise RefusalError("stop reason refusal", tokens_in, tokens_out)
        blocks = getattr(response, "content", None) or []
        text = "".join(b.text for b in blocks if getattr(b, "type", None) == "text" and getattr(b, "text", None))
        return RawReply(text, tokens_in, tokens_out)


def claude(model: str, *, api_key: str | None = None, price: PriceModel | None = None, can_batch: bool = True,
           client: Any = None, name: str | None = None, structured_output: bool = True) -> ClaudeJudge:
    """A Claude judge for ``model`` (required; there is no default model).

    ``api_key`` defaults to whatever the SDK resolves (``ANTHROPIC_API_KEY`` and its other
    credential sources). ``client`` replaces the SDK client (it needs ``client.messages.create``).
    ``name`` (default: the model id) is the judge's name in observations and the route name a
    ``Condition.judge`` pins to. ``structured_output=False`` drops the output schema for models
    that reject one; the prompt still asks for JSON. Retries are the policy's: the SDK client is
    built with ``max_retries=0``.
    """
    if client is None:
        anthropic = require("anthropic", "claude")
        client = anthropic.AsyncAnthropic(api_key=api_key, max_retries=0)
    return ClaudeJudge(model, name=name or model, price=price, can_batch=can_batch, client=client,
                       structured_output=structured_output)
