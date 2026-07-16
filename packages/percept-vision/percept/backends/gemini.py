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
from typing import Any, AsyncIterator

from ..adapters import AudioFrame, Box, Decision, Goal, Observation, Verdict
from ._resilience import CircuitBreaker, with_retry

# Reuse the provider-agnostic prompts + JSON-contract parsers (the contract is identical).
from .anthropic import _JUDGE_SYSTEM, _parse_observations, _parse_verdict
from .anthropic_text import _CONDUCTOR_SYSTEM, _coerce_decision, _extract_json

GEMINI_DEFAULT = "gemini-3.5-flash"  # GA flash (structured output + function calling); 2.5-flash hit
# rate limits. Override via PERCEPT_*_MODEL=gemini-...

# Item 2 process-QA: the eval-validated running-summary prompt (append-only + specificity-preserving —
# beats the holistic single answer on temporally-spread process questions; see scratch/pvqa_process_ab).
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

# Item 4 process-QA = log -> summarize -> wipe (eval-validated; scratch/pvqa_process_ab FINDINGS Run E).
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


def _image_part(types: Any, frame: Any):
    """Render an opaque frame to a Gemini image ``Part``: ``(media_type, bytes)`` | bytes | image dict."""
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
            "GeminiVision needs image bytes / a (media_type, bytes) pair / an image dict as the frame; "
            f"got {type(frame).__name__}. Pre-encode the frame (and crop the ROI) before judging."
        )
    return types.Part.from_bytes(data=raw, mime_type=media)


def _audio_part(types: Any, clip: Any):
    """Render an audio clip to a Gemini audio ``Part``: typed :class:`AudioFrame` | ``(mime, bytes)`` | wav bytes."""
    if isinstance(clip, AudioFrame):  # typed audio modality (B2): render via its legacy-pair view
        clip = clip.as_pair()
    if isinstance(clip, tuple) and len(clip) == 2 and isinstance(clip[1], (bytes, bytearray)):
        media, raw = str(clip[0]), bytes(clip[1])
    elif isinstance(clip, (bytes, bytearray)):
        media, raw = "audio/wav", bytes(clip)
    else:
        raise TypeError(
            "GeminiVision audio needs raw bytes or a (mime_type, bytes) pair; "
            f"got {type(clip).__name__}. Pre-encode the audio clip before judging."
        )
    return types.Part.from_bytes(data=raw, mime_type=media)


def _usage_of(resp: Any) -> dict:
    """Pull token usage off a google-genai response for ``Verdict.usage`` (read by
    ``observability.judge_usage`` → the cost/token funnel). Defensive: a response without
    ``usage_metadata`` (fakes/tests) yields ``{}`` and the metering hook stays a no-op."""
    um = getattr(resp, "usage_metadata", None)
    if um is None:
        return {}
    out: dict[str, int] = {}
    pt = getattr(um, "prompt_token_count", None)
    ct = getattr(um, "candidates_token_count", None)
    if isinstance(pt, int):
        out["input_tokens"] = pt
    if isinstance(ct, int):
        out["output_tokens"] = ct
    return out


# Per-request timeout (ms): a stalled socket ABORTS and surfaces to with_retry instead of hanging the whole
# run forever (the N=4 PVQA 64-minute hang). 120s is generous headroom for a thinking model.
_REQUEST_TIMEOUT_MS = 120_000


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
                http_options=types.HttpOptions(timeout=_REQUEST_TIMEOUT_MS),
            )
        return self._client


class GeminiVision(_GeminiBase):
    """``Vision`` adapter backed by Gemini Flash — narrow grounded VQA, same firing contract as Claude."""

    def __init__(self, settings: Any = None, *, client: Any = None) -> None:
        super().__init__(settings, client=client, model_attr="vision_model")
        self._breaker = CircuitBreaker()  # transient-error resilience for batch captures

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
        _, types = _require_genai()
        client = self._ensure_client()
        # temporal strip: earlier context frames (oldest->newest), then the current frame last, so the
        # model reads time from Part order. A single multimodal call — Gemini's native modality is why
        # the strip + audio are token-cheap (one call, not N).
        strip = [f for f in (frames or []) if f is not frame]
        parts: list[Any] = [_image_part(types, f) for f in strip]
        parts.append(_image_part(types, frame))
        if exemplar is not None:
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
            question = f'Looking only at the image: is this true right now — "{condition}"? '
        if audio is not None:
            parts.append(_audio_part(types, audio))
            question += "You may also use the attached audio clip as evidence. "
        if corrections:
            facts = "; ".join(f"{k}={v}" for k, v in corrections.items())
            question += (
                f"The user has told you these authoritative facts (trust them over your own read): "
                f"{facts}. "
            )
        if context:
            spec_facts = "; ".join(f"{k}={v}" for k, v in context.items())
            question += (
                f"A specialist model has reported these structured facts (trust them over your own "
                f"read): {spec_facts}. "
            )
        if strip:
            # clip judging: ask for the temporal structure too (parsed into Verdict.phase/action_present;
            # single-frame replies omit these and default cleanly).
            question += ('Answer as STRICT JSON {"satisfied", "confidence", "evidence", '
                         '"action_present" (true if the action/event is visibly occurring across these '
                         'frames), "phase" (one of "pre","during","post")}.')
        else:
            question += 'Answer as STRICT JSON {"satisfied", "confidence", "evidence"}.'
        parts.append(question)

        async def _call():
            return await client.aio.models.generate_content(
                model=self._model,
                contents=parts,
                # 1024 (not 512): gemini-3-flash is a thinking model whose reasoning tokens can otherwise
                # truncate the STRICT-JSON verdict mid-string (parse fails -> false ASK/abstain).
                config=types.GenerateContentConfig(system_instruction=_JUDGE_SYSTEM, max_output_tokens=1024),
            )

        resp = await with_retry(_call, breaker=self._breaker)  # survive transient resets in batch captures
        verdict = _parse_verdict(resp.text or "")
        verdict.usage = _usage_of(resp)  # 1.3 metering: light up the token/cost funnel (was always {})
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
        strip = [f for f in (frames or []) if f is not frame]
        parts: list[Any] = [_image_part(types, f) for f in strip]
        parts.append(_image_part(types, frame))
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

        resp = await with_retry(_call, breaker=self._breaker)
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
            usage=_usage_of(resp),
        )

    async def _observe_step(self, condition: str, frame: Any, frames: list | None) -> Verdict:
        """Item 4 logsum: report ONLY this clip's step delta. The model is NOT shown the accumulated log.

        Returns ``Verdict(satisfied=False, evidence=<the step, or "">)`` — satisfied is False BY DESIGN so a
        process goal never fires per-clip; the agent appends ``evidence`` to ``ConcernContext.process_log``
        and the ANSWER is produced later by :meth:`summarize_process` at the episode boundary / on demand.
        """
        _, types = _require_genai()
        client = self._ensure_client()
        strip = [f for f in (frames or []) if f is not frame]
        parts: list[Any] = [_image_part(types, f) for f in strip]
        parts.append(_image_part(types, frame))
        parts.append(
            f"Process question: {condition}\n{len(strip) + 1} frame(s) in time order (oldest first). JSON only."
        )

        async def _call():
            return await client.aio.models.generate_content(
                model=self._model, contents=parts,
                config=types.GenerateContentConfig(system_instruction=_OBSERVE_STEP_SYSTEM, max_output_tokens=1024),
            )

        resp = await with_retry(_call, breaker=self._breaker)
        try:
            data = json.loads(_extract_json(resp.text or ""))
        except Exception:
            data = {}
        step = str(data.get("step", "") or "").strip()
        confidence = float(data.get("confidence", 0.5) or 0.5)
        return Verdict(satisfied=False, confidence=confidence, evidence=step, usage=_usage_of(resp))

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

        resp = await with_retry(_call, breaker=self._breaker)
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

        resp = await with_retry(_call, breaker=self._breaker)
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

        return await with_retry(_call, breaker=self._breaker)


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
