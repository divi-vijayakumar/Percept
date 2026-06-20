"""Anthropic text backends — the Conductor (Haiku) and the Specialist (streaming reply).

These are **L2/L3-only**: deterministic L1 tests use the fakes, never these (standing rule 2 —
never assert backend intelligence with a fake). The Anthropic SDK is **lazy-imported inside
methods** so the core never pulls a vendor SDK at module import (invariant 7); a missing SDK or
key raises a clear, actionable error only when a real turn is attempted.

Split of responsibility (the seams):
  * :class:`AnthropicConductor` (Haiku, ``settings.models.conductor_model``) — the **only tier
    that writes standing state**: it routes a turn / scene delta and authors goal ops. It emits
    a :class:`~percept.adapters.Decision` (route + optional ``goal_ops`` + optional ``say``).
  * :class:`AnthropicSpecialist` (``settings.models.specialist_model``) — streams the reply
    token-by-token, optionally grounded on a frame.

The conductor lives behind the same registry name (``"claude"``) as the rest; ``registry.py``
points the conductor/specialist seams at this module.
"""

from __future__ import annotations

import json
from typing import Any, AsyncIterator

from ..adapters import Decision, Goal
from ..goals import GoalOp
from ._resilience import CircuitBreaker, with_retry

# The Haiku conductor's contract, stated to the model. Kept here (not in logic) so the prompt
# is reviewable; the model returns strict JSON we parse into a Decision + GoalOps.
_CONDUCTOR_SYSTEM = """You are Percept's conductor: a fast router for an always-on vision assistant.
Given a user utterance OR a scene-delta event plus the current standing state, decide what to do.
You are the ONLY component allowed to author standing goals (watch/count/remind).

Return STRICT JSON, no prose, with this shape:
{
  "route": one of ["respond_direct","respond_with_vision","task_with_tools","act","proactive_speak","ignore"],
  "look": "none" | "request_frame",
  "say": optional short string to speak (for proactive_speak or a canned reply),
  "goal_ops": [ {"op":"create","goal":{"id":"...","condition":"plain language","say":"what to speak when it fires"}} ]
}
Rules:
- Author a goal when the user wants you to WATCH/COUNT/REMIND about something over time
  ("watch my coffee", "tell me when the water boils"). Use a plain-language condition; never
  compile it to attributes.
- When in doubt, prefer respond_with_vision (look=request_frame) over guessing.
- For a scene-delta event you may proactively speak (route=proactive_speak) or stay quiet
  (route=ignore). Only speak when it is genuinely useful and in-envelope."""


def _coerce_decision(payload: dict) -> Decision:
    """Parse the model's JSON into a frozen :class:`Decision` (+ :class:`GoalOp` list)."""
    route = payload.get("route", "respond_direct")
    look = payload.get("look", "none")
    say = payload.get("say")
    ops: list = []
    for raw in payload.get("goal_ops", []) or []:
        op = raw.get("op", "create")
        if op == "create":
            g = raw.get("goal", {}) or {}
            goal = Goal(
                id=g.get("id") or f"g-{abs(hash(g.get('condition', ''))) % 10_000}",
                condition=g.get("condition", ""),
                say=g.get("say", ""),
                source=g.get("source", "camera"),
            )
            ops.append(GoalOp(op="create", goal=goal))
        else:
            ops.append(
                GoalOp(op=op, goal_id=raw.get("goal_id"), fields=raw.get("fields", {}) or {})
            )
    return Decision(route=route, look=look, say=say, goal_ops=ops)


class AnthropicConductor:
    """Haiku conductor: routes + authors goals. The only tier that writes standing state."""

    def __init__(self, settings: Any = None) -> None:
        self._settings = settings
        self._model = getattr(
            getattr(settings, "models", None), "conductor_model", "claude-haiku-4-5"
        )
        self._client: Any = None
        self._breaker = CircuitBreaker()

    def _ensure_client(self) -> Any:
        if self._client is None:
            try:
                import anthropic  # lazy: vendor SDK never imported at module top level
            except ImportError as exc:  # pragma: no cover — L2/L3 only
                raise RuntimeError(
                    "AnthropicConductor needs the 'anthropic' SDK (optional extra). "
                    "Install it or use the 'fake' conductor for offline runs."
                ) from exc
            self._client = anthropic.AsyncAnthropic()
        return self._client

    async def run(self, event: str, utterance: str, context: dict) -> Decision:
        client = self._ensure_client()
        user = json.dumps(
            {"event": event, "utterance": utterance, "context": context}, ensure_ascii=False
        )

        async def _call() -> Decision:
            resp = await client.messages.create(
                model=self._model,
                max_tokens=512,
                system=[
                    # prompt-cached: the stable system block is cacheable across turns
                    {
                        "type": "text",
                        "text": _CONDUCTOR_SYSTEM,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                messages=[{"role": "user", "content": user}],
            )
            text = "".join(
                getattr(block, "text", "") for block in resp.content
            ).strip()
            try:
                payload = json.loads(text)
            except json.JSONDecodeError:
                # Never confidently mis-route on a malformed reply: degrade to a grounded reply.
                return Decision(route="respond_with_vision", look="request_frame")
            return _coerce_decision(payload)

        return await with_retry(_call, breaker=self._breaker)


class AnthropicSpecialist:
    """Streams the reply token-by-token, optionally grounded on a frame."""

    def __init__(self, settings: Any = None) -> None:
        self._settings = settings
        self._model = getattr(
            getattr(settings, "models", None), "specialist_model", "claude-haiku-4-5"
        )
        self._client: Any = None

    def _ensure_client(self) -> Any:
        if self._client is None:
            try:
                import anthropic  # lazy
            except ImportError as exc:  # pragma: no cover — L2/L3 only
                raise RuntimeError(
                    "AnthropicSpecialist needs the 'anthropic' SDK (optional extra). "
                    "Install it or use the 'fake' specialist for offline runs."
                ) from exc
            self._client = anthropic.AsyncAnthropic()
        return self._client

    async def stream(
        self, prompt: str, *, frame: Any = None, fast: bool = False
    ) -> AsyncIterator[str]:
        client = self._ensure_client()
        content: list[dict] = []
        if frame is not None:
            # frame is expected to be a base64 image dict or a media block prepared upstream;
            # we pass it through as an image block when it looks like one, else stringify.
            block = _frame_block(frame)
            if block is not None:
                content.append(block)
        content.append({"type": "text", "text": prompt})

        async with client.messages.stream(
            model=self._model,
            max_tokens=1024,
            messages=[{"role": "user", "content": content}],
        ) as stream:
            async for text in stream.text_stream:
                yield text


def _frame_block(frame: Any) -> dict | None:
    """Best-effort: turn a frame into an Anthropic image content block, else None."""
    if isinstance(frame, dict) and frame.get("type") == "image":
        return frame
    if isinstance(frame, (bytes, bytearray)):
        import base64

        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": base64.b64encode(bytes(frame)).decode("ascii"),
            },
        }
    return None


__all__ = ["AnthropicConductor", "AnthropicSpecialist"]
