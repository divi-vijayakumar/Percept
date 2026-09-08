"""Paid raw-video comparator adapter built on Percept's shipped Gemini transport.

There is deliberately no HTTP implementation here.  The wrapper reuses ``GeminiVision`` for
credential discovery, exact model selection, native media encoding, timeout/retry behavior, and
the request-budget seam.  Construction remains lazy, and the baseline CLI imports this module only
after an explicit paid-run confirmation.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

from percept.backends.gemini import GeminiVision, _media_part

from raw_vlm_baseline import ProviderResponse, RawVlmRequest, RESPONSE_SCHEMA


MAX_OUTPUT_TOKENS = 4096


class _RawGeminiVision(GeminiVision):
    """Narrow raw completion surface that preserves GeminiVision's metered transport."""

    async def complete_raw(self, media_bytes: bytes, prompt: str) -> tuple[str, dict[str, Any]]:
        types = self._transport_namespace()
        client = self._ensure_client()
        parts = [_media_part(types, ("video/mp4", media_bytes)), prompt]

        async def call():
            return await client.aio.models.generate_content(
                model=self._model,
                contents=parts,
                config=types.GenerateContentConfig(max_output_tokens=MAX_OUTPUT_TOKENS),
            )

        response = await self._metered_completion(call)
        return response.text or "", self._budget_usage(response)


class PerceptBackendProvider:
    """Explicit Gemini/model binding for the strict whole-video comparator."""

    supports_native_video = True
    budget_reservation_before_transport = True

    def __init__(
        self,
        *,
        backend: str,
        model: str,
        vision: Any | None = None,
    ) -> None:
        if backend != "gemini":
            raise ValueError(
                "strict paid raw-video comparison currently supports only Gemini: "
                "OpenRouter lacks native-video transport and Anthropic lacks the "
                "request-budget seam"
            )
        if not isinstance(model, str) or not model.startswith("gemini"):
            raise ValueError("an explicit Gemini model id is required")
        self.backend = backend
        self.model = model
        self._vision = vision or _RawGeminiVision(
            settings=SimpleNamespace(models=SimpleNamespace(vision_model=model))
        )
        actual_model = getattr(self._vision, "_model", model)
        if actual_model != model:
            raise ValueError(
                f"shipped backend resolved model {actual_model!r}, expected explicit {model!r}"
            )
        if getattr(self._vision, "SUPPORTS_NATIVE_VIDEO_TRANSPORT", True) is not True:
            raise ValueError("configured Percept backend does not support native video")

    def configure_request_budget(self, request_budget: Any) -> None:
        configure = getattr(self._vision, "configure_request_budget", None)
        if not callable(configure):
            raise ValueError("configured Percept backend has no request-budget seam")
        configure(request_budget)

    def complete(self, request: RawVlmRequest) -> ProviderResponse:
        if request.backend != self.backend or request.model != self.model:
            raise ValueError("request provider identity differs from configured adapter")
        raw, usage = asyncio.run(
            self._vision.complete_raw(request.media_bytes, request.prompt)
        )
        return ProviderResponse(
            raw_response=raw,
            input_tokens=usage.get("input_tokens"),
            output_tokens=usage.get("output_tokens"),
            response_schema=RESPONSE_SCHEMA,
            reported_cost_usd=usage.get("cost_usd"),
        )
