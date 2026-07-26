"""OpenRouter vision adapter — OpenAI-compatible multi-provider VLM access (model sweep + cross-provider).

OpenRouter exposes Qwen-VL, InternVL, Molmo, GPT, Llama-Vision, … behind ONE OpenAI-compatible
``chat/completions`` API. This adapter implements the same narrow-grounded ``Vision`` contract as the
Gemini/Claude backends (one firing path, one gate), so ANY OpenRouter model is a drop-in judge — that is
exactly what the cost/latency/outcome model sweep needs.

**Lane discipline.** L2-only: needs ``OPENROUTER_API_KEY`` + the ``openai`` SDK. The module imports
cleanly with neither present (the SDK is lazy-imported inside ``_ensure_client``; invariant 7). The L1
lane uses ``FakeVision``, never this.

The model id is taken from ``settings.models.vision_model`` (rule 5: no literal in logic), e.g.
``"qwen/qwen2.5-vl-72b-instruct"``, ``"opengvlab/internvl3-78b"``, ``"openai/gpt-5-mini"``. Percept already
builds multi-frame clips (strip/sprite/window); this adapter sends them as ordered ``image_url`` blocks, so
multi-image models work and the current frame is read last (time = image order).
"""

from __future__ import annotations

import base64
import os
from typing import Any

from ..adapters import Box, Goal, Observation, Verdict
from ._resilience import CircuitBreaker, with_retry
# Reuse the validated, well-tested response parsing from the Claude adapter (same strict-JSON verdict
# shape, same defensive "unparsed -> low-confidence unknown" fallback that keeps the gate at ASK, and
# the same claim-mode correction contract so a correction parses identically on every backend).
from .anthropic import _CLAIM_MODE_CONTRACT, _JUDGE_SYSTEM, _coerce_json, _parse_observations, _parse_verdict

_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
_PROCESS_QA_SYSTEM = (
    "You answer a standing question by watching a process unfold. Each turn you get your running answer so "
    "far and new frames; integrate any new step and rewrite the FULL ordered answer (keep what holds, add "
    "what's new). Reply STRICT JSON only."
)
_DEFAULT_TIMEOUT_S = 120.0  # generous headroom for a slow open-weight model; a stall surfaces as retryable
# The OpenRouter default model — used ONLY when ``settings.models.vision_model`` is not an explicit
# ``vendor/model`` id (see ``_resolve_model``). Keeps a shared cross-backend setting from silently
# leaking a Claude/Gemini model id (e.g. "claude-opus-4-8") into an OpenRouter call, which then fails.
_DEFAULT_MODEL = "qwen/qwen3-vl-32b-instruct"
# OpenRouter now includes usage/cost automatically. Keep the former include request for compatibility
# with older gateways; current OpenRouter treats it as a deprecated no-op.
_USAGE_EXTRA_BODY = {"usage": {"include": True}}


class _BudgetTerminalError(Exception):
    """Internal marker: budget admission/reconciliation failures must never retry."""

    def __init__(self, original: Exception) -> None:
        super().__init__(str(original))
        self.original = original


def _timeout_s() -> float:
    try:
        return float(os.environ.get("PERCEPT_VISION_TIMEOUT_S", _DEFAULT_TIMEOUT_S))
    except (TypeError, ValueError):
        return _DEFAULT_TIMEOUT_S


def _require_openai():
    """Lazy-import the OpenAI SDK with a clear error if it is absent (L2 boundary)."""
    try:
        from openai import AsyncOpenAI  # noqa: PLC0415 — lazy by design (invariant 7)
    except ModuleNotFoundError as exc:  # pragma: no cover — exercised only without the extra
        raise RuntimeError(
            "OpenRouterVision needs the 'openai' SDK — install percept-vision[openrouter]. "
            "(L2 only; the L1 lane uses the fake vision backend.)"
        ) from exc
    return AsyncOpenAI


def _settings_of(arg: Any):
    if arg is None:
        from ..config import load_settings

        return load_settings()
    return arg


def _resolve_model(settings: Any) -> str:
    """Resolve the OpenRouter model id from ``settings.models.vision_model``.

    ``vision_model`` is a SHARED setting across vision backends (rule 5: no literal in logic), so its
    default/value may be a bare Claude/Gemini id like ``"claude-opus-4-8"`` meant for a different
    backend — sending that straight to OpenRouter silently fails (it is not a ``vendor/model`` id
    OpenRouter recognizes). An explicit ``vendor/model`` id (contains ``"/"``, e.g.
    ``"qwen/qwen3-vl-32b-instruct"``, ``"anthropic/claude-3-opus"``) IS respected — OpenRouter can route
    to any vendor. Anything else (absent, or a bare non-namespaced id) falls back to
    :data:`_DEFAULT_MODEL`.
    """
    raw = getattr(getattr(settings, "models", settings), "vision_model", None)
    if isinstance(raw, str) and "/" in raw:
        return raw
    return _DEFAULT_MODEL


def _data_url(frame: Any) -> str:
    """Render an opaque frame to a base64 ``data:`` URL (the OpenAI/OpenRouter image format).

    Accepts an Anthropic-style ``{"type": "image", "source": {...}}`` block, a ``(media_type, bytes)``
    pair, or raw JPEG/PNG bytes — the same frame shapes the other backends accept, so the bench's clips
    flow through unchanged."""
    if isinstance(frame, dict) and frame.get("type") == "image":
        src = frame.get("source", {})
        return f"data:{src.get('media_type', 'image/jpeg')};base64,{src.get('data', '')}"
    if isinstance(frame, tuple) and len(frame) == 2 and isinstance(frame[1], (bytes, bytearray)):
        media, raw = str(frame[0]), bytes(frame[1])
        return f"data:{media};base64,{base64.b64encode(raw).decode('ascii')}"
    if isinstance(frame, (bytes, bytearray)):
        return f"data:image/jpeg;base64,{base64.b64encode(bytes(frame)).decode('ascii')}"
    raise TypeError(
        "OpenRouterVision needs image bytes / a (media_type, bytes) pair / an image block as the frame; "
        f"got {type(frame).__name__}. Pre-encode the frame before judging."
    )


def _image_block(frame: Any) -> dict:
    return {"type": "image_url", "image_url": {"url": _data_url(frame)}}


def _usage_of(resp: Any) -> dict:
    """Pull token usage and billed cost off an
    OpenAI/OpenRouter chat completion for ``Verdict.usage`` (read by ``observability.judge_usage`` →
    the cost/token funnel). Defensive: no ``usage`` → ``{}`` — the metering hook stays a no-op.

    Current OpenRouter responses include this accounting automatically. We still send
    :data:`_USAGE_EXTRA_BODY` for compatibility with older gateways.
    """
    usage = getattr(resp, "usage", None)
    if usage is None and isinstance(resp, dict):
        usage = resp.get("usage")
    if usage is None:
        return {}
    get = usage.get if isinstance(usage, dict) else lambda k: getattr(usage, k, None)
    out: dict[str, float] = {}
    pt, ct = get("prompt_tokens"), get("completion_tokens")
    if isinstance(pt, int):
        out["input_tokens"] = pt
    if isinstance(ct, int):
        out["output_tokens"] = ct
    cost = get("cost")
    if isinstance(cost, (int, float)) and not isinstance(cost, bool):
        out["cost_usd"] = float(cost)
    return out


def _text_of(resp: Any) -> str:
    """Pull the assistant text out of an OpenAI/OpenRouter chat completion (object or dict)."""
    choices = getattr(resp, "choices", None)
    if choices is None and isinstance(resp, dict):
        choices = resp.get("choices")
    if not choices:
        return ""
    first = choices[0]
    msg = getattr(first, "message", None)
    if msg is None and isinstance(first, dict):
        msg = first.get("message", {})
    content = getattr(msg, "content", None)
    if content is None and isinstance(msg, dict):
        content = msg.get("content")
    return str(content or "")


class OpenRouterVision:
    """``Vision`` adapter backed by any OpenRouter model. Narrow grounded VQA, same firing contract.

    Constructed by the registry as ``OpenRouterVision(settings)`` (``vision="openrouter"``). The client is
    built lazily on first call so the module imports with no key present (L2 boundary)."""

    RUNTIME_EXECUTION_BOUNDARY = "process"

    #: PARSES the claim-mode correction channel (see :class:`~percept.backends.anthropic.AnthropicVision`).
    PARSES_CLAIM_CHANNEL = True

    def __init__(
        self,
        settings: Any = None,
        *,
        client: Any = None,
        request_budget: Any = None,
    ) -> None:
        self._settings = _settings_of(settings)
        self._client = client  # injectable for L2 fixtures; built lazily otherwise
        self._model = _resolve_model(self._settings)
        self._breaker = CircuitBreaker()
        self._timeout_s = _timeout_s()
        # Optional request admission/reconciliation seam used by paid benchmark runners.
        # It is deliberately duck-typed so this product adapter does not import evaluation code.
        self._request_budget = request_budget

    def _ensure_client(self):
        if self._client is None:
            AsyncOpenAI = _require_openai()
            key = (
                os.environ.get("OPENROUTER_API_KEY")
                or os.environ.get("OPENROUTER_KEY")
            )
            if not key:
                raise RuntimeError("OpenRouterVision needs OPENROUTER_API_KEY in the environment.")
            client_options = {
                "base_url": _OPENROUTER_BASE_URL,
                "api_key": key,
                "timeout": self._timeout_s,
                # OpenRouter attribution headers (optional but recommended; harmless if ignored).
                "default_headers": {
                    "HTTP-Referer": "https://github.com/divi-vijayakumar/Percept",
                    "X-Title": "Percept",
                },
            }
            if self._request_budget is not None:
                # Paid benchmarks own the retry loop so each physical attempt can be
                # admitted and accounted exactly once. Generic adapter construction
                # preserves the OpenAI SDK's historical/default retry behavior.
                client_options["max_retries"] = 0
            self._client = AsyncOpenAI(
                **client_options,
            )
        return self._client

    async def _completion(self, **kwargs: Any) -> Any:
        """Execute one metered completion with exact physical-attempt accounting."""
        client = self._ensure_client()

        async def _call():
            try:
                reservation = (
                    self._request_budget.begin_attempt()
                    if self._request_budget is not None
                    else None
                )
            except Exception as exc:
                raise _BudgetTerminalError(exc) from exc
            try:
                response = await client.chat.completions.create(**kwargs)
            except BaseException:
                if reservation is not None:
                    self._request_budget.fail_attempt(reservation)
                raise
            if reservation is not None:
                try:
                    self._request_budget.complete_attempt(
                        reservation,
                        _usage_of(response),
                    )
                except Exception as exc:
                    raise _BudgetTerminalError(exc) from exc
            return response

        attempts = (
            self._request_budget.attempts_per_call
            if self._request_budget is not None
            else 3
        )
        timeout_s = (
            min(self._timeout_s, self._request_budget.request_timeout_s)
            if self._request_budget is not None
            else self._timeout_s
        )
        try:
            return await with_retry(
                _call,
                attempts=attempts,
                breaker=self._breaker,
                timeout_s=timeout_s,
                non_retryable=(_BudgetTerminalError,),
            )
        except _BudgetTerminalError as exc:
            raise exc.original from exc

    def budget_snapshot(self) -> dict[str, int | float] | None:
        """Return the benchmark admission ledger without exposing credential data."""
        if self._request_budget is None:
            return None
        return self._request_budget.snapshot()

    def configure_request_budget(self, request_budget: Any) -> None:
        """Attach a request admission controller before the first completion is sent."""
        self._request_budget = request_budget

    async def judge(
        self,
        condition: str,
        frame: Any,
        *,
        roi: Box | None = None,
        exemplar: Any = None,
        corrections: dict | None = None,
        frames: list | None = None,
        audio: Any = None,
        context: dict | None = None,
    ) -> Verdict:
        """Ask the selected OpenRouter model a narrow grounded yes/no about ``condition``.

        ``frames`` is the temporal strip (earlier context, oldest→newest); the current ``frame`` is sent
        last so the model reads time from image order. ``audio`` is IGNORED (these models are vision-only
        here; a ``fused`` concern honestly degrades to the visual strip)."""
        del audio, roi  # accepted for protocol parity; ROI cropping is the caller's job (real pixels)
        ctx = context or {}
        if ctx.get("process_qa"):  # accumulating-answer question (mirror GeminiVision._judge_process_qa)
            return await self._judge_process_qa(condition, frame, frames, str(ctx.get("running_answer") or ""))
        claim_mode = bool(ctx.get("claim_mode"))  # claim-slot concern -> parse the correction channel
        content = self._build_content(condition, frame, exemplar=exemplar,
                                      corrections=corrections, frames=frames, context=context,
                                      claim_mode=claim_mode)
        messages = [{"role": "system", "content": _JUDGE_SYSTEM},
                    {"role": "user", "content": content}]

        resp = await self._completion(
            model=self._model,
            messages=messages,
            max_tokens=1024,  # headroom: a thinking/open model can spend tokens before the JSON verdict
            extra_body=_USAGE_EXTRA_BODY,  # compatibility with older OpenRouter gateways
        )
        verdict = _parse_verdict(_text_of(resp), claim_mode=claim_mode)
        verdict.usage = _usage_of(resp)  # 1.3 metering: token/cost funnel (was always {})
        return verdict

    async def _judge_process_qa(self, condition: str, frame: Any, frames: list | None, prior: str) -> Verdict:
        """One process-QA step: integrate this clip into the running answer (Item 2). Returns the rewritten
        ordered answer on ``Verdict.running_answer``; on a parse miss the prior answer is preserved."""
        strip = [f for f in (frames or []) if f is not frame]
        content: list = [_image_block(f) for f in strip]
        content.append(_image_block(frame))
        content.append({"type": "text", "text": (
            f"Standing question: {condition}\nRUNNING ANSWER SO FAR: {prior or '(none yet)'}\n"
            f"{len(strip) + 1} frame(s) in time order (oldest first). Integrate any NEW step and rewrite the "
            'full ordered answer. STRICT JSON {"updated_summary": "<answer>", "satisfied": true|false, '
            '"confidence": 0..1}.')})
        messages = [{"role": "system", "content": _PROCESS_QA_SYSTEM}, {"role": "user", "content": content}]

        resp = await self._completion(
            model=self._model,
            messages=messages,
            max_tokens=1024,
            extra_body=_USAGE_EXTRA_BODY,
        )
        try:
            data = _coerce_json(_text_of(resp))
        except (ValueError, TypeError):
            data = {}
        summary = str(data.get("updated_summary", "") or "").strip() or prior
        return Verdict(satisfied=bool(data.get("satisfied")) and bool(summary),
                       confidence=float(data.get("confidence", 0.5) or 0.5),
                       evidence=summary or "not enough evidence", running_answer=summary,
                       usage=_usage_of(resp))

    async def detect(self, frame: Any, goals: list[Goal]) -> list[Observation]:
        labels = ", ".join(sorted({g.condition for g in goals})) or "any salient object"
        prompt = (
            "Detect goal-relevant objects. For each, return a JSON list item "
            '{"label": str, "box": [x,y,w,h] normalized 0..1, "confidence": 0..1}. '
            f"Goals of interest: {labels}. Return STRICT JSON list (possibly empty)."
        )
        content = [_image_block(frame), {"type": "text", "text": prompt}]

        resp = await self._completion(
            model=self._model,
            messages=[{"role": "user", "content": content}],
            max_tokens=1024,
            extra_body=_USAGE_EXTRA_BODY,
        )
        return _parse_observations(_text_of(resp))

    async def detect_events(self, frames: list, goals: list[Goal]) -> list:
        return []

    def _build_content(
        self,
        condition: str,
        frame: Any,
        *,
        exemplar: Any,
        corrections: dict | None,
        frames: list | None = None,
        context: dict | None = None,
        claim_mode: bool = False,
    ) -> list:
        strip = [f for f in (frames or []) if f is not frame]  # earlier context, oldest->newest
        content: list = [_image_block(f) for f in strip]
        content.append(_image_block(frame))
        if exemplar is not None:
            content.append(_image_block(exemplar))
            question = (
                f'Looking only at the region of interest, and comparing to the reference exemplar: is this '
                f'true — "{condition}" — about THIS specific object? '
            )
        elif strip:
            question = (
                f'These {len(strip) + 1} images are consecutive moments in TIME ORDER (oldest first). '
                f'Considering how the scene CHANGES across them, is this true by the latest moment — '
                f'"{condition}"? '
            )
        else:
            question = f'Looking only at the image: is this true right now — "{condition}"? '
        if corrections:
            facts = "; ".join(f"{k}={v}" for k, v in corrections.items())
            question += f"Trust these authoritative facts over your own read: {facts}. "
        if context and isinstance(context, dict):
            # ``process_qa``/``claim_mode`` are RESERVED control keys, never surfaced as specialist facts.
            spec = "; ".join(f"{k}={v}" for k, v in context.items() if k not in ("process_qa", "claim_mode"))
            if spec:
                question += f"A specialist model reported these structured facts (trust them): {spec}. "
        question += 'Answer as STRICT JSON {"satisfied", "confidence", "evidence"}.'
        if claim_mode:
            question += _CLAIM_MODE_CONTRACT  # advertise the OPTIONAL correction fields (byte-for-byte off)
        content.append({"type": "text", "text": question})
        return content


__all__ = ["OpenRouterVision"]
