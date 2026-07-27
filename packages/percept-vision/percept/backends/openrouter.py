"""OpenRouter vision adapter — OpenAI-compatible multi-provider VLM access (model sweep + cross-provider).

OpenRouter exposes Qwen-VL, InternVL, Molmo, GPT, Llama-Vision, … behind ONE OpenAI-compatible
``chat/completions`` API. This adapter implements the same narrow-grounded ``Vision`` contract as the
Gemini/Claude backends (one firing path, one gate), so ANY OpenRouter model is a drop-in judge — that is
exactly what the cost/latency/outcome model sweep needs.

**Lane discipline.** L2-only: needs ``OPENROUTER_API_KEY`` + the ``openai`` SDK. The module imports
cleanly with neither present (the SDK is lazy-imported inside ``_ensure_client``; invariant 7). The L1
lane uses ``FakeVision``, never this.

The model id is taken from ``settings.models.vision_model`` (rule 5: no literal in logic), e.g.
``"qwen/qwen2.5-vl-72b-instruct"``, ``"opengvlab/internvl3-78b"``, ``"openai/gpt-5-mini"``. Ordered
images remain ordered ``image_url`` blocks. The benchmark's temporal-image route supplies an explicit,
bounded timestamped image sequence for models that do not accept native video.
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
from .anthropic import (
    _CLAIM_MODE_CONTRACT,
    _JUDGE_SYSTEM,
    _JUDGE_SYSTEM_REFLEX,
    _bounded_strip,
    _clip_timeline_sentence,
    _coerce_json,
    _is_reflex,
    _judge_max_frames,
    _judge_max_tokens,
    _parse_observations,
    _parse_verdict,
    _terse_reply,
)

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

# The DELIBERATE output ceiling — unchanged headroom, because an open/thinking model on a route
# that is not latency-bound may legitimately spend tokens before its JSON verdict, and truncating
# it there turns a real read into a spurious abstain. A REFLEX route is budgeted instead by the
# shared, measured policy in :func:`percept.backends.anthropic._judge_max_tokens`.
_DELIBERATE_MAX_TOKENS = 1024


class _BudgetTerminalError(Exception):
    """Internal marker: budget admission/reconciliation failures must never retry."""

    def __init__(self, original: Exception) -> None:
        super().__init__(str(original))
        self.original = original


class _PermanentProviderError(Exception):
    """Internal marker for a provider 4xx that cannot succeed on retry."""

    def __init__(self, original: BaseException) -> None:
        super().__init__(str(original))
        self.original = original


def _permanent_client_error(exc: BaseException) -> bool:
    """Return whether an OpenAI-compatible HTTP failure is permanently invalid.

    Timeouts, conflicts, and rate limits may recover; other 4xx responses such
    as unsupported media must consume exactly one physical attempt.
    """
    status = getattr(exc, "status_code", None)
    return (
        isinstance(status, int)
        and 400 <= status < 500
        and status not in {408, 409, 429}
    )


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
    """Render opaque media to a base64 ``data:`` URL.

    Accepts an Anthropic-style ``{"type": "image", "source": {...}}`` block, a ``(media_type, bytes)``
    pair, or raw JPEG/PNG bytes."""
    if isinstance(frame, dict) and frame.get("type") == "image":
        src = frame.get("source", {})
        return f"data:{src.get('media_type', 'image/jpeg')};base64,{src.get('data', '')}"
    if isinstance(frame, tuple) and len(frame) == 2 and isinstance(frame[1], (bytes, bytearray)):
        media, raw = str(frame[0]), bytes(frame[1])
        return f"data:{media};base64,{base64.b64encode(raw).decode('ascii')}"
    if isinstance(frame, (bytes, bytearray)):
        return f"data:image/jpeg;base64,{base64.b64encode(bytes(frame)).decode('ascii')}"
    raise TypeError(
        "OpenRouterVision needs media bytes / a (media_type, bytes) pair / an image block as the frame; "
        f"got {type(frame).__name__}. Pre-encode the frame before judging."
    )


def _media_type(frame: Any) -> str:
    if isinstance(frame, dict) and frame.get("type") == "image":
        return str(frame.get("source", {}).get("media_type", "image/jpeg"))
    if isinstance(frame, tuple) and len(frame) == 2:
        return str(frame[0])
    if isinstance(frame, (bytes, bytearray)):
        return "image/jpeg"
    raise TypeError(f"unsupported OpenRouter media value: {type(frame).__name__}")


def _is_video(frame: Any) -> bool:
    return _media_type(frame).lower().startswith("video/")


def _image_block(frame: Any) -> dict:
    if _is_video(frame):
        raise TypeError("video evidence must use a native video_url block")
    return {"type": "image_url", "image_url": {"url": _data_url(frame)}}


def _media_block(frame: Any) -> dict:
    if _is_video(frame):
        return {"type": "video_url", "video_url": {"url": _data_url(frame)}}
    return _image_block(frame)


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

    #: PARSES the claim-mode correction channel (see :class:`~percept.backends.anthropic.AnthropicVision`).
    PARSES_CLAIM_CHANNEL = True
    # Qwen's OpenRouter route accepts ordered image blocks but rejects native
    # ``video_url``. The runtime therefore selects its explicit one-request
    # temporal-image transport instead of attempting an unsupported MP4 call.
    SUPPORTS_NATIVE_VIDEO_TRANSPORT = False

    def __init__(
        self,
        settings: Any = None,
        *,
        client: Any = None,
        request_budget: Any = None,
        api_key: str | None = None,
        timeout_s: float | None = None,
    ) -> None:
        self._settings = _settings_of(settings)
        self._client = client  # injectable for L2 fixtures; built lazily otherwise
        # An EXPLICIT caller-supplied credential (hosted BYOK): when set, the environment is never
        # consulted for this adapter instance. ``None`` keeps the historical env-backed registry path.
        self._api_key = api_key
        self._model = _resolve_model(self._settings)
        self._breaker = CircuitBreaker()
        self._timeout_s = _timeout_s() if timeout_s is None else float(timeout_s)
        # Optional request admission/reconciliation seam used by paid benchmark runners.
        # It is deliberately duck-typed so this product adapter does not import evaluation code.
        self._request_budget = request_budget

    def _ensure_client(self):
        if self._client is None:
            AsyncOpenAI = _require_openai()
            key = (
                self._api_key
                or os.environ.get("OPENROUTER_API_KEY")
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
            except BaseException as exc:
                if reservation is not None:
                    self._request_budget.fail_attempt(reservation)
                if _permanent_client_error(exc):
                    raise _PermanentProviderError(exc) from exc
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
                non_retryable=(_BudgetTerminalError, _PermanentProviderError),
            )
        except _BudgetTerminalError as exc:
            raise exc.original from exc
        except _PermanentProviderError as exc:
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

        ``frames`` is the lower-fidelity ordered-image representation (earlier context,
        oldest→newest); the current ``frame`` is sent last. For motion-sensitive watches, pass one
        native clip as ``("video/mp4", bytes)`` and no ``frames``. ``audio`` is IGNORED (these models
        are vision-only here; a ``fused`` concern honestly degrades to visual evidence)."""
        del audio, roi  # accepted for protocol parity; ROI cropping is the caller's job (real pixels)
        ctx = context or {}
        if _is_video(frame):
            raise ValueError(
                "OpenRouterVision does not accept native video; use the runtime "
                "timestamped temporal-image transport"
            )
        if ctx.get("process_qa"):  # accumulating-answer question (mirror GeminiVision._judge_process_qa)
            return await self._judge_process_qa(condition, frame, frames, str(ctx.get("running_answer") or ""))
        claim_mode = bool(ctx.get("claim_mode"))  # claim-slot concern -> parse the correction channel
        event_chain = bool(ctx.get("event_chain"))
        # Verbosity follows the ROUTE (see ``_is_reflex``): a reflex route asks for the two fields the
        # gate actually reads; a deliberate route is byte-for-byte the historical request.
        reflex = _is_reflex(ctx)
        terse = _terse_reply(reflex=reflex, claim_mode=claim_mode)
        content = self._build_content(condition, frame, exemplar=exemplar,
                                      corrections=corrections, frames=frames, context=context,
                                      claim_mode=claim_mode, reflex=reflex)
        system = _JUDGE_SYSTEM_REFLEX if terse and not event_chain else _JUDGE_SYSTEM

        resp = await self._completion(
            model=self._model,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": content}],
            max_tokens=_judge_max_tokens(
                reflex=reflex,
                # An occurrence reply still owes cognition phase/action_present/events.
                structured=event_chain,
                claim_mode=claim_mode,
                deliberate=_DELIBERATE_MAX_TOKENS,
            ),
            extra_body=_USAGE_EXTRA_BODY,  # compatibility with older OpenRouter gateways
        )
        verdict = _parse_verdict(_text_of(resp), claim_mode=claim_mode)
        verdict.usage = _usage_of(resp)  # 1.3 metering: token/cost funnel (was always {})
        return verdict

    async def _judge_process_qa(self, condition: str, frame: Any, frames: list | None, prior: str) -> Verdict:
        """One process-QA step: integrate this clip into the running answer (Item 2). Returns the rewritten
        ordered answer on ``Verdict.running_answer``; on a parse miss the prior answer is preserved."""
        strip = [f for f in (frames or []) if f is not frame]
        content: list = [_media_block(f) for f in strip]
        content.append(_media_block(frame))
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
        reflex: bool = False,
    ) -> list:
        strip = [f for f in (frames or []) if f is not frame]  # earlier context, oldest->newest
        event_chain = bool((context or {}).get("event_chain"))
        # ``reflex`` narrows the REPLY SHAPE and, via ``_judge_max_frames``, the number of frames the
        # request carries. Every grounding instruction below (time order, exemplar deixis,
        # corrections, specialist facts) is unchanged: a reflex judge is asked the same question over
        # a shorter, more recent window, just not asked to write an essay about the answer.
        terse = _terse_reply(reflex=reflex, claim_mode=claim_mode)
        native_video = _is_video(frame)
        if native_video and strip:
            raise ValueError(
                "native video evidence cannot be mixed with ordered images; "
                "select one exact evidence shape"
            )
        # Same shared bound as the Gemini adapter, so the two providers ingest the same evidence on
        # the same route. Only the OLDEST context frames are dropped.
        bounded_strip = _bounded_strip(
            strip, max_frames=_judge_max_frames(reflex=reflex)
        )
        dropped_frames = len(strip) - len(bounded_strip)
        strip = bounded_strip
        temporal_timestamps = (context or {}).get(
            "_percept_temporal_image_timestamps_s"
        )
        ordered_images = [*strip, frame]
        # Whether each image ends up carrying its own ``t=`` label below. When it does, the clip's
        # timeline is pinned per frame and the occurrence question needs no prose about where the
        # retained window starts; when it does not, that origin has to be stated.
        per_frame_labels = False
        if (
            isinstance(temporal_timestamps, (tuple, list))
            and len(temporal_timestamps) == len(ordered_images) + dropped_frames
            and all(
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                for value in temporal_timestamps
            )
        ):
            # Timestamps follow the frames they label, so the bound slices them in lockstep. They
            # KEEP THE CLIP'S ORIGINAL ORIGIN rather than being re-based onto the retained window:
            # ``agent._fire_clip_occurrences`` decodes a reported ``t_start`` against
            # ``_percept_temporal_span_s`` (the full captured span), so re-basing here would shift
            # every occurrence earlier by exactly the dropped prefix.
            temporal_timestamps = list(temporal_timestamps)[dropped_frames:]
            per_frame_labels = True
            content = []
            for index, (image, timestamp_s) in enumerate(
                zip(ordered_images, temporal_timestamps, strict=True), start=1
            ):
                content.append(
                    {
                        "type": "text",
                        "text": f"Temporal frame {index}/{len(ordered_images)} at t={float(timestamp_s):.6g}s.",
                    }
                )
                content.append(_image_block(image))
        else:
            content = [_media_block(f) for f in strip]
            content.append(_media_block(frame))
        if event_chain:
            if exemplar is not None:
                content.append(_image_block(exemplar))
            representation = (
                "This is a short native video clip with continuous temporal order."
                if native_video
                else (
                    f"These {len(strip) + 1} images are consecutive moments in TIME ORDER "
                    "(oldest first)."
                )
            )
            question = (
                f'{representation} Find every distinct occurrence of "{condition}" in the '
                "evidence; do not merge adjacent repetitions. Classify the "
                "LATEST moment as pre (the occurrence has not begun), during (it is visibly in "
                "progress), or post (it just completed within this clip). Do not infer an action "
                "from a static hand/object pose; require visible motion/change in the evidence. "
            )
            if exemplar is not None:
                question += (
                    "The final image is a reference exemplar. Report occurrences "
                    "only for that specific object. "
                )
        elif exemplar is not None:
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
            spec = "; ".join(
                f"{k}={v}"
                for k, v in context.items()
                if k not in ("process_qa", "claim_mode", "event_chain")
                and not k.startswith("_percept_")
            )
            if spec:
                question += f"A specialist model reported these structured facts (trust them): {spec}. "
        if event_chain:
            question += _clip_timeline_sentence(
                (context or {}).get("_percept_temporal_span_s"),
                dropped_frames=dropped_frames,
                total_frames=len(ordered_images) + dropped_frames,
                offsets=(context or {}).get("_percept_temporal_offsets_s"),
                per_frame_labels=per_frame_labels,
            )
            # The occurrence fields stay REQUIRED on both routes — cognition abstains without a
            # consistent satisfied/action_present/phase tuple. A reflex occurrence request drops only
            # the free-text ``evidence`` field and caps label length.
            question += (
                (
                    'Answer as STRICT JSON {"satisfied", "confidence", "action_present", '
                    '"phase", "events": [{"t_start": seconds, "t_end": seconds|null, '
                    '"label": "at most 4 words"}]} and nothing else — omit "evidence", write no '
                    "prose and no explanation. "
                )
                if terse
                else (
                    'Answer as STRICT JSON {"satisfied", "confidence", "evidence", '
                    '"action_present", "phase", "events": [{"t_start": seconds, '
                    '"t_end": seconds|null, "label": "short description"}]}. '
                )
            )
            question += (
                "List every grounded occurrence in chronological order with times "
                "relative to the beginning of the evidence. Set action_present=true only for "
                '"during". '
                "Use satisfied=true when this clip grounds a during or just-completed post "
                "occurrence; otherwise false."
            )
        elif terse:
            question += (
                'Answer as STRICT JSON {"satisfied", "confidence"} and nothing else — no '
                '"evidence" field, no prose, no explanation.'
            )
        else:
            question += 'Answer as STRICT JSON {"satisfied", "confidence", "evidence"}.'
        if claim_mode:
            question += _CLAIM_MODE_CONTRACT  # advertise the OPTIONAL correction fields (byte-for-byte off)
        content.append({"type": "text", "text": question})
        return content


__all__ = ["OpenRouterVision"]
