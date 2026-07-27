"""Gemini (Google) backends — vision judge + conductor + specialist (the cheap/fast perception lane).

A faithful mirror of the Anthropic backends: the SAME firing contract (``Verdict``), the SAME conductor
``Decision`` (with the guarded behavior-block authoring), the SAME tuned prompts and JSON parsers —
only the wire is ``google.genai`` instead of ``anthropic``. Gemini Flash's cost/latency-to-intelligence
curve is the reason to run the high-frequency perception calls here.

Lane discipline (same as the Anthropic adapter): **L2-only**. ``google.genai`` is lazy-imported inside
methods, so this module imports with neither the SDK nor a key present (invariant 7). The L1 lane uses
``FakeVision``/``FakeConductor``, never this.

Model id comes from ``settings.models.*`` (rule 5) when it names a Gemini model; otherwise it falls back
to :data:`GEMINI_DEFAULT` so a Claude-pinned Settings still works with ``vision="gemini"``. The
behavior-block authoring firewall and the firing logic live UPSTREAM (``concern``/``behavior``/``gate``)
and are model-agnostic — switching perception to Gemini changes none of it.
"""

from __future__ import annotations

import base64
import json
import os
from types import ModuleType
from typing import Any, AsyncIterator

from ..adapters import AudioFrame, Box, Decision, Goal, Observation, Verdict
from ._resilience import CircuitBreaker, with_retry

# Reuse the provider-agnostic prompts + JSON-contract parsers (the contract is identical, incl. the
# claim-mode correction channel — one contract string, one parser, every parsing backend).
from .anthropic import (
    _CLAIM_MODE_CONTRACT,
    _JUDGE_SYSTEM,
    _JUDGE_SYSTEM_REFLEX,
    _bounded_strip,
    _is_reflex,
    _judge_max_frames,
    _judge_max_tokens,
    _parse_observations,
    _parse_verdict,
    _terse_reply,
)
from .anthropic_text import _CONDUCTOR_SYSTEM, _coerce_decision, _extract_json

GEMINI_DEFAULT = "gemini-3.5-flash"  # GA flash (structured output + function calling); 2.5-flash hit
# rate limits. Override via PERCEPT_*_MODEL=gemini-...

# The DELIBERATE judge output ceiling — unchanged headroom for a route that is not latency-bound.
# A REFLEX route is budgeted instead by the shared, measured policy in
# :func:`percept.backends.anthropic._judge_max_tokens`.
_DELIBERATE_MAX_TOKENS = 1024

# Process-QA uses the eval-validated running-summary prompt: append-only and specificity-preserving,
# it beats a holistic single answer on temporally spread process questions.
_PROCESS_QA_SYSTEM = (
    "You are the perception backend for a proactive, always-on video assistant answering a standing "
    "PROCESS question (e.g. 'what are the steps to do X?'). The video arrives as a stream of short clips "
    "over time; the process CONTINUES across the whole video, so new steps keep appearing late as well as "
    "early. You are given the RUNNING ANSWER built from earlier clips. From THIS clip (frames in time "
    "order, oldest first) and how the scene changes, identify any NEW step it shows, then return the "
    "running answer. RULES: (1) APPEND-ONLY — never drop a step you already listed; carry ALL prior steps "
    "forward and add any new one at the end. (2) BE SPECIFIC — name the exact object/ingredient/tool; "
    "treat two steps as the same only if it is literally the same action on the same object. (3) KEEP "
    "WATCHING for later steps even when you already have several. Reply STRICT JSON "
    '{"satisfied": true|false, "confidence": 0..1, "new_step": "<the specific step this clip adds, or '
    'empty>", "updated_summary": "<the full ordered list of ALL distinct steps so far>"}. '
    "satisfied=true once the running answer names at least one concrete step."
)

# Eval-validated process-QA lifecycle: log -> summarize -> wipe.
# The model is NEVER shown the accumulated memory: per clip it reports ONE short delta (cheap, constant
# output, no anchoring/truncation), appended to a per-concern LOG; a SINGLE summarize pass consolidates
# the whole log (global dedup -> tight answer) on boundary/on-demand, then the log is wiped.
_OBSERVE_STEP_SYSTEM = (
    "You are the perception backend for a proactive video assistant watching a PROCESS ('what are the "
    "steps to do X'). Look ONLY at THIS short clip (frames in TIME ORDER, oldest first) and report the "
    "single concrete step/action being performed RIGHT NOW, naming the SPECIFIC object/ingredient/tool. "
    "Do not summarize the whole video — just this moment. If nothing clear, leave step empty. Reply STRICT "
    'JSON {"confidence": 0..1, "step": "<the specific step happening now, or empty>"}.'
)
_SUMMARIZE_LOG_SYSTEM = (
    "You are consolidating a TIME-ORDERED log of per-moment step-observations from a SINGLE video into the "
    "answer to a process question. Merge duplicates and paraphrases (same action on the same object = ONE "
    "step) but keep genuinely distinct steps separate (different object/tool = different step). Drop noise. "
    'Output the distinct ordered steps. Reply STRICT JSON {"steps": ["<short step>", ...]}.'
)


def _require_genai():
    """Lazy-import google-genai with a clear error if absent (L2 boundary)."""
    try:
        from google import genai
        from google.genai import types
    except ModuleNotFoundError as exc:  # pragma: no cover — only without the extra
        raise RuntimeError(
            "Gemini backends need 'google-genai' — install percept-vision[gemini]. "
            "(L2 only; the L1 lane uses the fake backends.)"
        ) from exc
    return genai, types


def _gemini_key() -> str | None:
    return (
        os.environ.get("GEMINI_API_KEY")
        or os.environ.get("GOOGLE_API_KEY")
        or os.environ.get("GOOGLE_AISTUDIO_API_KEY")
    )


def _settings_of(arg: Any):
    if arg is None:
        from ..config import load_settings

        return load_settings()
    return arg


def _model_for(settings: Any, attr: str) -> str:
    """A Gemini model id from settings if it names one, else the Gemini default (so a Claude-pinned
    Settings still works with ``vision='gemini'``)."""
    m = getattr(getattr(settings, "models", settings), attr, None)
    return m if isinstance(m, str) and m.startswith("gemini") else GEMINI_DEFAULT


def _media_value(frame: Any) -> tuple[str, bytes]:
    if isinstance(frame, tuple) and len(frame) == 2 and isinstance(frame[1], (bytes, bytearray)):
        media, raw = str(frame[0]), bytes(frame[1])
    elif isinstance(frame, (bytes, bytearray)):
        media, raw = "image/png", bytes(frame)
    elif isinstance(frame, dict) and frame.get("type") == "image":
        src = frame.get("source", {})
        media = src.get("media_type", "image/png")
        raw = base64.b64decode(src.get("data", ""))
    else:
        raise TypeError(
            "GeminiVision needs media bytes / a (media_type, bytes) pair / an image dict as the frame; "
            f"got {type(frame).__name__}. Pre-encode the frame (and crop the ROI) before judging."
        )
    return media, raw


def _is_video(frame: Any) -> bool:
    media, _ = _media_value(frame)
    return media.lower().startswith("video/")


def _media_part(types: Any, frame: Any):
    """Render encoded image or native-video evidence to a Gemini ``Part``."""
    media, raw = _media_value(frame)
    return types.Part.from_bytes(data=raw, mime_type=media)


def _image_part(types: Any, frame: Any):
    """Render image-only evidence, rejecting accidental native-video use."""
    media, raw = _media_value(frame)
    if media.lower().startswith("video/"):
        raise TypeError("video evidence is not valid in an image-only Gemini path")
    return types.Part.from_bytes(data=raw, mime_type=media)


def _wrap_wav(pcm: bytes, sample_rate: int, channels: int) -> bytes:
    """Wrap raw s16le PCM in a real WAV container. The API rejects headerless PCM labeled
    ``audio/wav`` (400 INVALID_ARGUMENT), so a header is mandatory, not cosmetic."""
    import io
    import wave

    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(max(1, int(channels or 1)))
        w.setsampwidth(2)
        w.setframerate(int(sample_rate or 16000))
        w.writeframes(pcm)
    return buf.getvalue()


def _audio_part(types: Any, clip: Any):
    """Render an audio clip to a Gemini audio ``Part``: typed :class:`AudioFrame` | ``(mime, bytes)`` | wav bytes."""
    if isinstance(clip, AudioFrame):
        if clip.encoded is not None and clip.mime:  # pre-encoded container: send as-is
            return types.Part.from_bytes(data=bytes(clip.encoded), mime_type=str(clip.mime))
        # raw-PCM AudioFrame: build a genuine WAV from its own rate/layout
        raw = _wrap_wav(bytes(clip.pcm), clip.sample_rate, clip.channels)
        return types.Part.from_bytes(data=raw, mime_type="audio/wav")
    if isinstance(clip, tuple) and len(clip) == 2 and isinstance(clip[1], (bytes, bytearray)):
        media, raw = str(clip[0]), bytes(clip[1])
    elif isinstance(clip, (bytes, bytearray)):
        media, raw = "audio/wav", bytes(clip)
    else:
        raise TypeError(
            "GeminiVision audio needs raw bytes or a (mime_type, bytes) pair; "
            f"got {type(clip).__name__}. Pre-encode the audio clip before judging."
        )
    if media == "audio/wav" and not raw.startswith(b"RIFF"):
        raw = _wrap_wav(raw, 16000, 1)  # headerless PCM claiming wav: wrap with the runtime default layout
    return types.Part.from_bytes(data=raw, mime_type=media)


def _usage_of(resp: Any) -> dict:
    """Pull token usage off a google-genai response for ``Verdict.usage`` (read by
    ``observability.judge_usage`` → the cost/token funnel). Defensive: a response without
    ``usage_metadata`` (fakes/tests) yields ``{}`` and the metering hook stays a no-op."""
    um = getattr(resp, "usage_metadata", None)
    if um is None:
        return {}
    out: dict[str, int] = {}
    prompt = sum(
        value for value in (getattr(um, "prompt_token_count", None), getattr(um, "tool_use_prompt_token_count", None))
        if isinstance(value, int)
    )
    output = sum(
        value for value in (getattr(um, "candidates_token_count", None), getattr(um, "thoughts_token_count", None))
        if isinstance(value, int)
    )
    if prompt:
        out["input_tokens"] = prompt
    if output:
        out["output_tokens"] = output
    return out


# Per-request timeout (ms): a stalled socket ABORTS and surfaces to with_retry instead of hanging the whole
# run forever (the N=4 PVQA 64-minute hang). 120s is generous headroom for a thinking model.
_REQUEST_TIMEOUT_MS = 120_000


def _provider_http_options(provider_types: Any) -> Any:
    """Build timeout options with provider-SDK retries pinned to one attempt.

    ``google-genai`` versions used in production expose ``HttpRetryOptions``.
    Small import doubles used by the offline suite intentionally implement only
    ``HttpOptions``; those modules have neither an import spec nor a file and
    cannot perform hidden transport retries. A real SDK missing the retry
    control fails closed instead of silently invalidating the request ledger.
    """
    http_options = getattr(provider_types, "HttpOptions", None)
    if not callable(http_options):
        raise RuntimeError("google-genai does not expose HttpOptions")
    retry_options = getattr(provider_types, "HttpRetryOptions", None)
    if callable(retry_options):
        return http_options(
            timeout=_REQUEST_TIMEOUT_MS,
            retry_options=retry_options(attempts=1),
        )
    if (
        isinstance(provider_types, ModuleType)
        and getattr(provider_types, "__spec__", None) is None
        and getattr(provider_types, "__file__", None) is None
    ):
        return http_options(timeout=_REQUEST_TIMEOUT_MS)
    raise RuntimeError(
        "installed google-genai cannot pin provider retries; upgrade to a version "
        "that exposes google.genai.types.HttpRetryOptions"
    )


class _BudgetTerminalError(Exception):
    """Prevent budget admission/reconciliation failures from being retried."""

    def __init__(self, original: Exception) -> None:
        super().__init__(str(original))
        self.original = original


class _GeminiBase:
    def __init__(self, settings: Any = None, *, client: Any = None, model_attr: str = "vision_model") -> None:
        self._settings = _settings_of(settings)
        self._client = client
        self._model = _model_for(self._settings, model_attr)

    def _ensure_client(self):
        if self._client is None:
            genai, types = _require_genai()
            self._client = genai.Client(
                api_key=_gemini_key(),
                http_options=_provider_http_options(types),
            )
        return self._client


class GeminiVision(_GeminiBase):
    """``Vision`` adapter backed by Gemini Flash — narrow grounded VQA, same firing contract as Claude."""

    #: PARSES the claim-mode correction channel (see :class:`~percept.backends.anthropic.AnthropicVision`).
    PARSES_CLAIM_CHANNEL = True
    # Narrow, enforced capability marker used by WatchPlan apply. This adapter attaches ``audio`` as
    # a native Gemini part below; adapters that merely accept-and-discard the protocol argument do
    # not set the marker.
    SUPPORTS_FUSED_AUDIO_INPUT = True
    # Transport capability only: actual model quality/latency remains a
    # qualification result, never something inferred from this marker.
    SUPPORTS_NATIVE_VIDEO_TRANSPORT = True

    def __init__(self, settings: Any = None, *, client: Any = None, request_budget: Any = None) -> None:
        super().__init__(settings, client=client, model_attr="vision_model")
        self._breaker = CircuitBreaker()  # transient-error resilience for batch captures
        self._request_budget = request_budget

    def configure_request_budget(self, request_budget: Any) -> None:
        self._request_budget = request_budget

    def budget_snapshot(self) -> dict[str, int | float] | None:
        return self._request_budget.snapshot() if self._request_budget is not None else None

    def _budget_usage(self, response: Any) -> dict[str, Any]:
        usage = _usage_of(response)
        if self._request_budget is not None:
            price = self._request_budget.plan["price_snapshot"]
            usage["estimated_usd"] = (
                usage.get("input_tokens", 0) * float(price["input_usd_per_million_tokens"])
                + usage.get("output_tokens", 0) * float(price["output_usd_per_million_tokens"])
            ) / 1_000_000
        return usage

    async def _metered_completion(self, call):
        async def attempt():
            try:
                reservation = self._request_budget.begin_attempt() if self._request_budget is not None else None
            except Exception as exc:
                raise _BudgetTerminalError(exc) from exc
            try:
                response = await call()
            except BaseException:
                if reservation is not None:
                    self._request_budget.fail_attempt(reservation)
                raise
            if reservation is not None:
                try:
                    usage = self._budget_usage(response)
                    if "cost_usd" in usage:
                        self._request_budget.complete_attempt(reservation, usage)
                    else:
                        self._request_budget.complete_unpriced_attempt(reservation, usage)
                except Exception as exc:
                    raise _BudgetTerminalError(exc) from exc
            return response

        attempts = self._request_budget.attempts_per_call if self._request_budget is not None else 3
        timeout_s = min(_REQUEST_TIMEOUT_MS / 1000, self._request_budget.request_timeout_s) if self._request_budget is not None else _REQUEST_TIMEOUT_MS / 1000
        try:
            return await with_retry(
                attempt,
                attempts=attempts,
                breaker=self._breaker,
                timeout_s=timeout_s,
                non_retryable=(_BudgetTerminalError,),
            )
        except _BudgetTerminalError as exc:
            raise exc.original from exc

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
        if context and context.get("process_qa"):
            # Process-QA goal: build the incremental running answer instead of a one-shot verdict. The
            # gate still reads satisfied/confidence; running_answer rides alongside (Item 2).
            return await self._judge_process_qa(
                condition, frame, frames, str(context.get("running_answer") or "")
            )
        if context and context.get("observe_step"):
            # Item 4 logsum: report ONLY this clip's step delta (no memory shown). evidence=the step.
            return await self._observe_step(condition, frame, frames)
        claim_mode = bool(context and context.get("claim_mode"))  # claim-slot: parse the correction channel
        # Reply verbosity follows the ROUTE, not the provider: a reflex route (stamped by
        # ``JudgePool``) asks for the fields the gate reads and stops; a deliberate route is
        # byte-for-byte the historical request. Same policy and same prompts as the OpenRouter
        # adapter, so the two providers stay coherent.
        reflex = _is_reflex(context)
        terse = _terse_reply(reflex=reflex, claim_mode=claim_mode)
        if frame is None and audio is None:
            raise TypeError("GeminiVision requires image evidence, audio evidence, or both")
        if frame is None and (frames or exemplar is not None):
            raise TypeError("audio-only Gemini judgment cannot carry image strip/exemplar inputs")
        native_video = frame is not None and _is_video(frame)
        if native_video and frames:
            raise ValueError(
                "native video evidence cannot be mixed with ordered images; "
                "select one exact evidence shape"
            )
        _, types = _require_genai()
        client = self._ensure_client()
        # temporal strip: earlier context frames (oldest->newest), then the current frame last, so the
        # model reads time from Part order. A single multimodal call — Gemini's native modality is why
        # the strip + audio are token-cheap (one call, not N).
        strip = list(frames or [])
        if (
            strip
            and not (context or {}).get("_percept_frames_are_earlier")
            and strip[-1] is frame
        ):
            strip.pop()
        # A reflex route is bounded in FRAME COUNT as well as in output tokens — same shared policy
        # (:func:`_judge_max_frames`), same rationale, so both providers ingest the same evidence on
        # the same route. Only the OLDEST context is dropped; the current frame and the newest
        # context frames survive in order, and every ``len(strip) + 1`` count below follows.
        strip = _bounded_strip(
            strip,
            max_frames=_judge_max_frames(reflex=reflex),
            has_current_frame=frame is not None,
        )
        parts: list[Any] = [_media_part(types, f) for f in strip]
        if frame is not None:
            parts.append(_media_part(types, frame))
        if frame is None:
            question = f'Listening only to the attached audio: is this true right now — "{condition}"? '
        elif context and context.get("event_chain"):
            if exemplar is not None:
                parts.append(_image_part(types, exemplar))
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
                "LATEST moment as pre (not begun), during (visibly in progress), or post (just "
                "completed within this clip). Do not infer the action from a static pose; require "
                "visible motion/change in the evidence. "
            )
            if exemplar is not None:
                question += (
                    "The final attached image is a reference exemplar. Track "
                    "the occurrence only for that specific object. "
                )
        elif exemplar is not None:
            parts.append(_image_part(types, exemplar))
            question = (
                f'Looking only at the image (and comparing to the reference exemplar): is this true — '
                f'"{condition}" — about THIS specific object? '
            )
        elif strip:
            if os.environ.get("PERCEPT_STRIP_PROMPT", "change") == "neutral":
                # ablation variant: do NOT invite the model to weigh the change — just judge the latest
                # state, with the earlier frames as plain context (isolates the prompt-framing confound).
                question = (
                    f'Here are {len(strip) + 1} recent frames of the same scene. Looking especially at '
                    f'the most recent frame, is this true right now — "{condition}"? '
                )
            else:
                question = (
                    f'These {len(strip) + 1} images are consecutive moments in TIME ORDER (oldest first). '
                    f'Considering how the scene CHANGES across them, is this true by the latest moment — '
                    f'"{condition}"? '
                )
        else:
            question = (
                f'Looking only at the image: is this true right now — "{condition}"? '
                if audio is None
                else f'Looking at the image AND listening to the attached audio: is this true right now — "{condition}"? '
            )
        if audio is not None:
            parts.append(_audio_part(types, audio))
            question += (
                "An audio clip recorded from this same scene over the last few seconds is attached. "
                "LISTEN to it: for a condition about sound (a device running or switched on, an alarm, "
                "speech), the soundtrack is often the decisive evidence — a device can be ON while "
                "sitting stationary, and the image alone cannot settle it. "
            )
        if corrections:
            facts = "; ".join(f"{k}={v}" for k, v in corrections.items())
            question += (
                f"The user has told you these authoritative facts (trust them over your own read): "
                f"{facts}. "
            )
        if context:
            # ``claim_mode`` is a RESERVED control key (not a specialist fact) — never surface it as one.
            spec_facts = "; ".join(
                f"{k}={v}"
                for k, v in context.items()
                if k not in ("claim_mode", "event_chain") and not k.startswith("_percept_")
            )
            if spec_facts:
                question += (
                    f"A specialist model has reported these structured facts (trust them over your own "
                    f"read): {spec_facts}. "
                )
        if context and context.get("event_chain"):
            span_s = context.get("_percept_temporal_span_s")
            if isinstance(span_s, (int, float)) and not isinstance(span_s, bool):
                question += f"The evidence spans {float(span_s):.6g} seconds. "
            # The occurrence fields stay REQUIRED on both routes — cognition abstains without a
            # consistent satisfied/action_present/phase tuple. A reflex occurrence request drops
            # only the free-text ``evidence`` field and caps label length.
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
                "Use satisfied=true for a grounded during or just-completed post occurrence."
            )
        elif strip:
            # clip judging: ask for the temporal structure too (parsed into Verdict.phase/action_present;
            # single-frame replies omit these and default cleanly). A reflex route keeps the two
            # structured fields — they are cheap and cognition reads them — and drops the prose.
            question += (
                ('Answer as STRICT JSON {"satisfied", "confidence", '
                 '"action_present" (true if the action/event is visibly occurring across these '
                 'frames), "phase" (one of "pre","during","post")} and nothing else — no '
                 '"evidence" field, no prose, no explanation.')
                if terse
                else ('Answer as STRICT JSON {"satisfied", "confidence", "evidence", '
                      '"action_present" (true if the action/event is visibly occurring across these '
                      'frames), "phase" (one of "pre","during","post")}.')
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
        parts.append(question)

        # A reply that still owes cognition phase/action_present(/events) cannot shrink to two
        # scalars; only the plain single-moment reply can.
        structured = bool(strip) or bool(context and context.get("event_chain"))
        max_output_tokens = _judge_max_tokens(
            reflex=reflex,
            structured=structured,
            claim_mode=claim_mode,
            # 1024 (not 512) on the DELIBERATE route: gemini-3-flash is a thinking model whose
            # reasoning tokens can otherwise truncate the STRICT-JSON verdict mid-string (parse
            # fails -> false ASK/abstain). A reflex route does not ask for the prose that made
            # that headroom necessary, so it does not need to pay for it.
            deliberate=_DELIBERATE_MAX_TOKENS,
        )
        system_instruction = (
            _JUDGE_SYSTEM_REFLEX if terse and not structured else _JUDGE_SYSTEM
        )

        async def _call():
            return await client.aio.models.generate_content(
                model=self._model,
                contents=parts,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    max_output_tokens=max_output_tokens,
                ),
            )

        resp = await self._metered_completion(_call)  # exact per-attempt admission/reconciliation
        verdict = _parse_verdict(resp.text or "", claim_mode=claim_mode)
        verdict.usage = self._budget_usage(resp)  # 1.3 metering: light up the token/cost funnel (was always {})
        return verdict

    async def _judge_process_qa(
        self, condition: str, frame: Any, frames: list | None, prior: str
    ) -> Verdict:
        """Item 2: one judge step of an incremental process-QA running answer.

        Carries the prior running answer in, asks the model to integrate THIS clip's new step (if any)
        and rewrite the full ordered answer, and returns it on ``Verdict.running_answer`` (also mirrored
        to ``evidence``). On a parse miss the prior answer is preserved (never regress the accumulation).
        """
        _, types = _require_genai()
        client = self._ensure_client()
        strip = list(frames or [])
        if strip and strip[-1] is frame:
            strip.pop()
        if _is_video(frame) and strip:
            raise ValueError(
                "native video evidence cannot be mixed with ordered images; "
                "select one exact evidence shape"
            )
        parts: list[Any] = [_media_part(types, f) for f in strip]
        parts.append(_media_part(types, frame))
        parts.append(
            f"Standing question: {condition}\n"
            f"RUNNING ANSWER SO FAR: {prior or '(none yet)'}\n"
            f"{len(strip) + 1} frame(s) in time order (oldest first). JSON only."
        )

        async def _call():
            return await client.aio.models.generate_content(
                model=self._model,
                contents=parts,
                config=types.GenerateContentConfig(
                    system_instruction=_PROCESS_QA_SYSTEM, max_output_tokens=1024
                ),
            )

        resp = await self._metered_completion(_call)
        try:
            data = json.loads(_extract_json(resp.text or ""))
        except Exception:
            data = {}
        summary = str(data.get("updated_summary", "") or "").strip() or prior
        satisfied = bool(data.get("satisfied")) and bool(summary)
        confidence = float(data.get("confidence", 0.5) or 0.5)
        return Verdict(
            satisfied=satisfied,
            confidence=confidence,
            evidence=summary or "not enough evidence",
            running_answer=summary,
            usage=self._budget_usage(resp),
        )

    async def _observe_step(self, condition: str, frame: Any, frames: list | None) -> Verdict:
        """Item 4 logsum: report ONLY this clip's step delta. The model is NOT shown the accumulated log.

        Returns ``Verdict(satisfied=False, evidence=<the step, or "">)`` — satisfied is False BY DESIGN so a
        process goal never fires per-clip; the agent appends ``evidence`` to ``ConcernContext.process_log``
        and the ANSWER is produced later by :meth:`summarize_process` at the episode boundary / on demand.
        """
        _, types = _require_genai()
        client = self._ensure_client()
        strip = list(frames or [])
        if strip and strip[-1] is frame:
            strip.pop()
        if _is_video(frame) and strip:
            raise ValueError(
                "native video evidence cannot be mixed with ordered images; "
                "select one exact evidence shape"
            )
        parts: list[Any] = [_media_part(types, f) for f in strip]
        parts.append(_media_part(types, frame))
        parts.append(
            f"Process question: {condition}\n{len(strip) + 1} frame(s) in time order (oldest first). JSON only."
        )

        async def _call():
            return await client.aio.models.generate_content(
                model=self._model, contents=parts,
                config=types.GenerateContentConfig(system_instruction=_OBSERVE_STEP_SYSTEM, max_output_tokens=1024),
            )

        resp = await self._metered_completion(_call)
        try:
            data = json.loads(_extract_json(resp.text or ""))
        except Exception:
            data = {}
        step = str(data.get("step", "") or "").strip()
        confidence = float(data.get("confidence", 0.5) or 0.5)
        return Verdict(satisfied=False, confidence=confidence, evidence=step, usage=self._budget_usage(resp))

    async def summarize_process(self, condition: str, log: list[str]) -> str:
        """Item 4: ONE consolidation pass over the whole per-concern log -> the ordered distinct steps.

        Text-only (no frames). Returns the consolidated answer; the caller stores it and wipes the log.
        ``max_output_tokens=4096`` — a thinking model summarizing a long log truncates at 1024 (validated).
        """
        _, types = _require_genai()
        if not log:
            return ""
        client = self._ensure_client()
        logged = "\n".join(f"- {s}" for s in log)
        prompt = (
            f"Process question: {condition}\nTIME-ORDERED OBSERVATION LOG:\n{logged}\nJSON only."
        )

        async def _call():
            return await client.aio.models.generate_content(
                model=self._model, contents=[prompt],
                config=types.GenerateContentConfig(system_instruction=_SUMMARIZE_LOG_SYSTEM, max_output_tokens=4096),
            )

        resp = await self._metered_completion(_call)
        try:
            data = json.loads(_extract_json(resp.text or ""))
        except Exception:
            data = {}
        steps = [str(s).strip() for s in (data.get("steps") or []) if str(s).strip()]
        return "; ".join(steps)

    async def detect(self, frame: Any, goals: list[Goal]) -> list[Observation]:
        _, types = _require_genai()
        client = self._ensure_client()
        labels = ", ".join(sorted({g.condition for g in goals})) or "any salient object"
        prompt = (
            "Detect goal-relevant objects. For each, return JSON list items "
            '{"label": str, "box": [x,y,w,h] normalized 0..1, "confidence": 0..1}. '
            f"Goals of interest: {labels}. Return STRICT JSON list (possibly empty)."
        )
        async def _call():
            return await client.aio.models.generate_content(
                model=self._model,
                contents=[_image_part(types, frame), prompt],
                config=types.GenerateContentConfig(max_output_tokens=512),
            )

        resp = await self._metered_completion(_call)
        return _parse_observations(resp.text or "")

    async def detect_events(self, frames: list, goals: list[Goal]) -> list:
        return []


class GeminiConductor(_GeminiBase):
    """Gemini conductor: routes + authors goals (incl. the guarded behavior block). Mirrors Claude."""

    def __init__(self, settings: Any = None, *, client: Any = None) -> None:
        super().__init__(settings, client=client, model_attr="conductor_model")
        self._breaker = CircuitBreaker()

    async def run(self, event: str, utterance: str, context: dict) -> Decision:
        _, types = _require_genai()
        client = self._ensure_client()
        user = json.dumps({"event": event, "utterance": utterance, "context": context}, ensure_ascii=False)

        async def _call() -> Decision:
            resp = await client.aio.models.generate_content(
                model=self._model,
                contents=[user],
                config=types.GenerateContentConfig(
                    system_instruction=_CONDUCTOR_SYSTEM, max_output_tokens=512
                ),
            )
            try:
                payload = json.loads(_extract_json(resp.text or "{}"))
            except json.JSONDecodeError:
                # Never confidently mis-route on a malformed reply — degrade to a grounded reply.
                return Decision(route="respond_with_vision", look="request_frame")
            return _coerce_decision(payload)

        return await with_retry(_call, breaker=self._breaker, timeout_s=_REQUEST_TIMEOUT_MS / 1000)


class GeminiSpecialist(_GeminiBase):
    """Streams the reply token-by-token, optionally grounded on a frame. Mirrors Claude."""

    def __init__(self, settings: Any = None, *, client: Any = None) -> None:
        super().__init__(settings, client=client, model_attr="specialist_model")

    async def stream(self, prompt: str, *, frame: Any = None, fast: bool = False) -> AsyncIterator[str]:
        _, types = _require_genai()
        client = self._ensure_client()
        parts: list[Any] = []
        if frame is not None:
            try:
                parts.append(_image_part(types, frame))
            except TypeError:
                pass  # a non-image frame descriptor — reply text-only
        parts.append(prompt)
        async for chunk in await client.aio.models.generate_content_stream(
            model=self._model, contents=parts
        ):
            text = getattr(chunk, "text", None)
            if text:
                yield text


__all__ = ["GeminiVision", "GeminiConductor", "GeminiSpecialist", "GEMINI_DEFAULT"]
