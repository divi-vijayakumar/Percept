"""Gemini Live vision adapter — the interactivity model via the Live API (``bidiGenerateContent``).

The Gemini *live* models (`gemini-3.1-flash-live-preview`, native-audio) are **audio-output only** — they
reject `response_modalities=["TEXT"]`. To make them usable as a perception judge we run them the only way
they speak: **audio out + `output_audio_transcription`**. The model SPEAKS its answer; the Live API returns
the transcript; we parse a spoken YES/NO + confidence into the same firing-contract ``Verdict`` every other
backend returns (one firing path, one gate). This is the realtime seam validated end-to-end against the
perception benchmark — a stepping stone to the v3 interaction seam, not the final voice-dialog shape.

**Lane discipline.** L2-only (Gemini key + `google-genai`); imports clean without either (lazy in
`_ensure_client`). Model id from `settings.models.vision_model` (rule 5). A clean per-call session avoids
context bleed between independent clip judgments.
"""

from __future__ import annotations

import asyncio
import base64
import os
import re
from typing import Any

from ..adapters import Box, Goal, Observation, Verdict
from ._resilience import CircuitBreaker, with_retry
from .gemini import _gemini_key, _require_genai

# A SPOKEN-verdict instruction: the model can only talk, so ask for a parseable spoken form.
_SPOKEN_JUDGE = (
    "You are a perception module answering ONE narrow yes/no question about the image(s). "
    "Speak your answer in EXACTLY this form and nothing else: the single word YES or NO, then the word "
    "CONFIDENCE followed by a number between 0 and 1, then the word BECAUSE followed by a short reason. "
    "Example: 'NO confidence 0.8 because the person is standing.'"
)
# process_qa (Item 2) for the live model: the spoken equivalent of GeminiVision._judge_process_qa — the
# model SPEAKS its updated full answer-so-far, integrating any new step it sees; the transcript IS the
# running answer the core accumulates and resolves.
_SPOKEN_PROCESS_QA = (
    "You are watching a process to answer a standing question. Each turn you are given your ANSWER SO FAR "
    "and new frames. Speak your UPDATED best answer: keep what still holds, add any new step you now see, "
    "in order. Speak ONLY the answer itself, as a short ordered description — no preamble."
)
_DEFAULT_TIMEOUT_S = 60.0


def _timeout_s() -> float:
    try:
        return float(os.environ.get("PERCEPT_VISION_TIMEOUT_S", _DEFAULT_TIMEOUT_S))
    except (TypeError, ValueError):
        return _DEFAULT_TIMEOUT_S


def _settings_of(arg: Any):
    if arg is None:
        from ..config import load_settings

        return load_settings()
    return arg


def _part_from_frame(frame: Any, types_mod):
    if isinstance(frame, dict) and frame.get("type") == "image":
        src = frame.get("source", {})
        return types_mod.Part.from_bytes(data=base64.b64decode(src.get("data", "")),
                                         mime_type=src.get("media_type", "image/jpeg"))
    if isinstance(frame, tuple) and len(frame) == 2 and isinstance(frame[1], (bytes, bytearray)):
        return types_mod.Part.from_bytes(data=bytes(frame[1]), mime_type=str(frame[0]))
    if isinstance(frame, (bytes, bytearray)):
        return types_mod.Part.from_bytes(data=bytes(frame), mime_type="image/jpeg")
    raise TypeError(f"GeminiLiveVision needs image bytes / (media,bytes) / an image block; got {type(frame).__name__}.")


def parse_spoken_verdict(transcript: str) -> Verdict:
    """Parse a SPOKEN verdict transcript into a firing-contract Verdict (defensive: unparsed -> low-conf).

    The model talks, so we read intent, not JSON: YES/NO (negation-aware) -> satisfied; the first 0..1 number
    -> confidence; the rest -> evidence. A blank/garbled transcript degrades to satisfied=False, conf=0.0 so
    the gate defers to ASK rather than risk a confident-wrong fire (invariant 1)."""
    text = (transcript or "").strip()
    if not text:
        return Verdict(satisfied=False, confidence=0.0, evidence="unparsed: empty transcript")
    low = text.lower()
    # satisfied: first explicit yes/no token wins; default False on absence
    yes_at = next((m.start() for m in re.finditer(r"\byes\b", low)), None)
    no_at = next((m.start() for m in re.finditer(r"\b(no|not|isn't|aren't|negative)\b", low)), None)
    if yes_at is None and no_at is None:
        satisfied = False
    elif no_at is None:
        satisfied = True
    elif yes_at is None:
        satisfied = False
    else:
        satisfied = yes_at < no_at
    # confidence: first number in [0,1]
    conf = 0.5
    for tok in re.findall(r"\d*\.?\d+", low):
        try:
            v = float(tok)
        except ValueError:
            continue
        if 0.0 <= v <= 1.0:
            conf = v
            break
    reason = text.split("because", 1)[1].strip() if "because" in low else text
    return Verdict(satisfied=satisfied, confidence=conf, evidence=reason[:240])


class GeminiLiveVision:
    """``Vision`` adapter backed by a Gemini *live* model: audio out, transcribed back to a Verdict."""

    def __init__(self, settings: Any = None, *, client: Any = None) -> None:
        self._settings = _settings_of(settings)
        self._client = client
        self._model = getattr(getattr(self._settings, "models", self._settings),
                              "vision_model", "gemini-3.1-flash-live-preview")
        self._breaker = CircuitBreaker()
        self._timeout_s = _timeout_s()

    def _ensure_client(self):
        if self._client is None:
            genai, _ = _require_genai()
            self._client = genai.Client(api_key=_gemini_key())
        return self._client

    async def judge(self, condition: str, frame: Any, *, roi: Box | None = None, exemplar: Any = None,
                    corrections: dict | None = None, frames: list | None = None, audio: Any = None,
                    context: dict | None = None) -> Verdict:
        del audio, roi, exemplar, corrections
        ctx = context or {}
        if ctx.get("process_qa"):  # accumulating-answer question — the spoken twin of GeminiVision._judge_process_qa
            return await self._judge_process_qa(condition, frame, frames, str(ctx.get("running_answer") or ""))
        _, types = _require_genai()
        parts = self._build_parts(condition, frame, types, frames=frames)
        transcript = await self._speak(parts, _SPOKEN_JUDGE, types)
        return parse_spoken_verdict(transcript)

    async def _judge_process_qa(self, condition: str, frame: Any, frames: list | None, prior: str) -> Verdict:
        """One process-QA step on the live model: the model SPEAKS its updated full answer-so-far; the
        transcript IS the running answer the core accumulates + resolves. satisfied stays False (resolution
        is the answer-path's job, not the satisfied gate); a blank transcript preserves the prior answer."""
        _, types = _require_genai()
        strip = [f for f in (frames or []) if f is not frame]
        parts: list = [_part_from_frame(f, types) for f in strip]
        parts.append(_part_from_frame(frame, types))
        parts.append(types.Part(text=(f"Standing question: {condition}\nYOUR ANSWER SO FAR: {prior or '(none yet)'}"
                                       f"\nSpeak your updated full answer.")))
        transcript = (await self._speak(parts, _SPOKEN_PROCESS_QA, types)).strip()
        answer = transcript or prior
        return Verdict(satisfied=False, confidence=0.7 if transcript else 0.0,
                       evidence=answer or "not enough evidence", running_answer=answer)

    async def _speak(self, parts: list, system_text: str, types) -> str:
        """Run one clean Live session: send the turn, collect the spoken transcript. A per-call timeout means
        a hung session (stalled WS / no turn_complete) aborts and retries instead of stalling the run."""
        client = self._ensure_client()
        config = types.LiveConnectConfig(
            response_modalities=["AUDIO"],                                  # live models only speak
            output_audio_transcription=types.AudioTranscriptionConfig(),    # ...so transcribe the speech
            system_instruction=types.Content(parts=[types.Part(text=system_text)]),
        )

        async def _session() -> str:
            said: list[str] = []
            async with client.aio.live.connect(model=self._model, config=config) as session:
                await session.send_client_content(turns=types.Content(role="user", parts=parts), turn_complete=True)
                async for msg in session.receive():
                    sc = getattr(msg, "server_content", None)
                    ot = getattr(sc, "output_transcription", None) if sc else None
                    if ot and getattr(ot, "text", None):
                        said.append(ot.text)
                    if sc and getattr(sc, "turn_complete", False):
                        break
            return "".join(said)

        async def _call() -> str:
            return await asyncio.wait_for(_session(), timeout=self._timeout_s)

        return await with_retry(_call, breaker=self._breaker)

    async def detect(self, frame: Any, goals: list[Goal]) -> list[Observation]:
        return []

    async def detect_events(self, frames: list, goals: list[Goal]) -> list:
        return []

    def _build_parts(self, condition: str, frame: Any, types_mod, *, frames: list | None = None) -> list:
        strip = [f for f in (frames or []) if f is not frame]
        parts: list = [_part_from_frame(f, types_mod) for f in strip]
        parts.append(_part_from_frame(frame, types_mod))
        if strip:
            q = (f'These {len(strip) + 1} images are consecutive moments in TIME ORDER (oldest first). '
                 f'Considering how the scene changes, is this true by the latest moment: "{condition}"? ')
        else:
            q = f'Is this true in the image right now: "{condition}"? '
        parts.append(types_mod.Part(text=q))
        return parts


__all__ = ["GeminiLiveVision", "parse_spoken_verdict"]
