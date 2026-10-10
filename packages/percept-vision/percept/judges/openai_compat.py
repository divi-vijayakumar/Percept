"""A judge for any OpenAI-compatible chat completions API, through the ``openai`` SDK.

Covers OpenAI itself and local servers that speak the same API (vLLM, Ollama, LM Studio): point
``base_url`` at the server. Install with ``pip install 'percept-vision[openai]'``. The SDK is
imported only when a client is built, so ``import percept`` works without it.

Example::

    judge = percept.judges.openai_compatible("<model id>", base_url="http://localhost:8000/v1")
"""

from __future__ import annotations

import base64
import os
from typing import Any

from percept.contracts import JudgeRequest, PriceModel
from percept.judges.adapter import Adapter, reply_schema, require, usage_int
from percept.judges.base import Prompt, RawReply, RefusalError

LOCAL_KEY = "unused"  # local servers ignore the key, but the SDK insists on one


class OpenAICompatibleJudge(Adapter):
    """One ``chat.completions.create`` call per request: a system message, then a user message
    holding the JPEGs (as data URIs, in order) followed by the conditions."""

    def __init__(self, model: str, *, name: str, price: PriceModel | None, can_batch: bool, client: Any,
                 structured_output: bool, token_param: str) -> None:
        super().__init__(name, price=price, can_batch=can_batch, client=client)
        self.model = model
        self.structured_output = structured_output
        self.token_param = token_param

    def build(self, prompt: Prompt, images: tuple[bytes, ...], max_output_tokens: int,
              request: JudgeRequest) -> dict:
        """The keyword arguments of the ``chat.completions.create`` call."""
        content: list[dict] = [
            {"type": "image_url",
             "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(image).decode("ascii")}}
            for image in images]
        content.append({"type": "text", "text": prompt.user})
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "system", "content": prompt.system}, {"role": "user", "content": content}],
            self.token_param: max_output_tokens,
        }
        if self.structured_output:
            kwargs["response_format"] = {"type": "json_schema", "json_schema": {
                "name": "judge_reply", "strict": True, "schema": reply_schema(request, closed=True)}}
        return kwargs

    async def _request(self, prompt: Prompt, images: tuple[bytes, ...], max_output_tokens: int,
                       request: JudgeRequest) -> RawReply:
        response = await self.client.chat.completions.create(
            **self.build(prompt, images, max_output_tokens, request))
        usage = getattr(response, "usage", None)
        tokens_in, tokens_out = usage_int(usage, "prompt_tokens"), usage_int(usage, "completion_tokens")
        choices = getattr(response, "choices", None) or []
        if not choices:
            return RawReply("", tokens_in, tokens_out)
        message = getattr(choices[0], "message", None)
        refusal = getattr(message, "refusal", None)
        if refusal or getattr(choices[0], "finish_reason", None) == "content_filter":
            raise RefusalError(str(refusal or "finish reason content_filter")[:160], tokens_in, tokens_out)
        return RawReply(str(getattr(message, "content", None) or ""), tokens_in, tokens_out)


def openai_compatible(model: str, *, base_url: str | None = None, api_key: str | None = None,
                      price: PriceModel | None = None, can_batch: bool = True, client: Any = None,
                      name: str | None = None, structured_output: bool = True) -> OpenAICompatibleJudge:
    """A judge for ``model`` on OpenAI (``base_url=None``) or an OpenAI-compatible server.

    ``api_key`` defaults to ``OPENAI_API_KEY``; with a ``base_url`` and no key, a placeholder is
    sent, as local servers expect. The output limit is sent as ``max_completion_tokens`` to OpenAI
    and as ``max_tokens`` to other servers, which is what each accepts. ``client`` replaces the SDK
    client (it needs ``client.chat.completions.create``). ``name`` (default: the model id) is the
    judge's name in observations and the route name a ``Condition.judge`` pins to.
    ``structured_output=False`` drops the JSON-schema response format for servers that reject it.
    Retries are the policy's: the SDK client is built with ``max_retries=0``.
    """
    if client is None:
        openai = require("openai", "openai")
        key = api_key or os.environ.get("OPENAI_API_KEY") or (LOCAL_KEY if base_url else None)
        client = openai.AsyncOpenAI(api_key=key, base_url=base_url, max_retries=0)
    return OpenAICompatibleJudge(model, name=name or model, price=price, can_batch=can_batch, client=client,
                                 structured_output=structured_output,
                                 token_param="max_completion_tokens" if base_url is None else "max_tokens")
