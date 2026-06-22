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
from ..behavior import coerce_authored_behavior, coerce_authored_perception
from ..goals import GoalOp
from ._resilience import CircuitBreaker, with_retry

# The Haiku conductor's contract, stated to the model. Kept here (not in logic) so the prompt
# is reviewable; the model returns strict JSON we parse into a Decision + GoalOps.
_CONDUCTOR_SYSTEM = """You are Percept's conductor: the router for an always-on vision assistant,
and the ONLY component that may author standing goals (watch / count / remind).

Return STRICT JSON, no prose:
{
  "route": "respond_direct" | "respond_with_vision" | "task_with_tools" | "act" | "proactive_speak" | "ignore",
  "look": "none" | "request_frame",
  "say": optional short string to speak,
  "goal_ops": [ {"op":"create","goal":{"id":"g-<short>","condition":"<plain language>","say":"<what to speak when it fires>","behavior":<optional, see BEHAVIOR>}} ]
}

FIRST decide STANDING-GOAL vs ONE-SHOT — this is the most important call:
- A STANDING GOAL is anything you must remember and act on LATER: the user wants to be
  notified / reminded / kept accountable OVER TIME. Trigger phrases include: "watch ...",
  "nudge me ...", "tell me when ...", "let me know when ...", "remind me ...", "alert me ...",
  "keep me ...", "count ...", "every N minutes". For ANY of these you MUST emit a goal_ops
  "create" with a plain-language condition, and set route="respond_direct" with a one-line
  confirmation in "say". Do NOT answer these with a one-shot look — the entire point is that it
  fires later, not now.
- A ONE-SHOT question about the current moment ("what's in my cup?", "where is he?") gets
  route="respond_with_vision", look="request_frame", and an EMPTY goal_ops.

Examples:
utterance: "watch my coffee and nudge me if I start drinking it"
-> {"route":"respond_direct","look":"none","say":"Watching your coffee — I'll nudge you if you start drinking it.","goal_ops":[{"op":"create","goal":{"id":"g-coffee","condition":"the user is drinking coffee","say":"Heads up — you're drinking coffee."}}]}
utterance: "while I'm at my desk, remind me to drink water every 10 minutes"
-> {"route":"respond_direct","look":"none","say":"I'll remind you to drink water every 10 minutes while you're at your desk.","goal_ops":[{"op":"create","goal":{"id":"g-water","condition":"the user is at their desk","say":"Time to drink some water.","behavior":{"rearm":"interval","cadence_s":600}}}]}
utterance: "what's in my cup right now?"
-> {"route":"respond_with_vision","look":"request_frame","goal_ops":[]}

BEHAVIOR (optional) — add a "behavior" block to a created goal ONLY when the request clearly implies HOW
to track it; OMIT it when unsure (the safe default is a plain watch that fires on the rising edge):
- fire ONCE then never again ("tell me once when ..."): {"rearm":"never"}
- fire if something is NOT done by a deadline ("remind me to X in N minutes", "... before it boils"):
  {"rearm":"deadline","deadline_s":<seconds>} — phrase the condition as the thing that SHOULD happen.
- a recurring nudge WHILE a condition holds ("every N minutes while I'm at my desk", "every N min when X"):
  {"rearm":"interval","cadence_s":<seconds>} — fires at most once per interval while the condition is true.
- a state CHANGE ("when the baby wakes up", "when it goes from A to B"):
  {"type":"transition","from":"<state A in plain language>","to":"<state B in plain language>"}
- high stakes ("warn me", safety): {"consequence":{"miss_cost":"severe"}}; for an irreversible action also
  "false_fire_cost":"irreversible". Never lower stakes you are unsure about.
- the Nth distinct time / a count threshold ("alert me the 3rd time ...", "the 2nd time ..."):
  {"accumulate":{"fire_when":"n == 3"}} — each rising edge of the condition is one distinct time; the
  expression may compare the count n (and time t in seconds) to numbers, joined by and/or — nothing else.
Only these keys are honored; anything else is ignored and the goal stays a safe watch.
utterance: "alert me the 3rd time the dog gets on the couch"
-> {"route":"respond_direct","look":"none","say":"I'll alert you the 3rd time the dog gets on the couch.","goal_ops":[{"op":"create","goal":{"id":"g-dog","condition":"the dog is on the couch","say":"That's the 3rd time the dog got on the couch.","behavior":{"accumulate":{"fire_when":"n == 3"}}}}]}
utterance: "tell me when the baby wakes up"
-> {"route":"respond_direct","look":"none","say":"I'll let you know when the baby wakes up.","goal_ops":[{"op":"create","goal":{"id":"g-wake","condition":"the baby is awake","say":"The baby is waking up.","behavior":{"type":"transition","from":"the baby is asleep","to":"the baby is awake"}}}]}

Other rules:
- Keep conditions in plain language — never compile them to attributes (one firing path).
- For a scene-delta event, proactively speak (route=proactive_speak) only when genuinely useful
  and in-envelope; otherwise route="ignore".
- Stay in the envelope: assistant-class, the user's own world, the user as beneficiary. No
  surveillance of non-consenting third parties."""


def _extract_json(text: str) -> str:
    """Pull the JSON object out of a model reply that may be fenced or wrapped in prose.

    Haiku often returns ```json ... ``` or a sentence around the object; a bare ``json.loads`` on
    the whole string then fails and the caller would silently degrade. Strip code fences, then take
    the outermost ``{ ... }`` span.
    """
    text = text.strip()
    if text.startswith("```"):
        # drop the opening fence (``` or ```json) and any closing fence
        text = text[3:]
        if text[:4].lower() == "json":
            text = text[4:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]
    return text


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
            # Validated authored behavior (interval/never/deadline/transition/consequence) -> goal.state.
            # Untrusted: every value is checked against a closed vocab, malformed entries are dropped.
            goal.state.update(coerce_authored_behavior(g.get("behavior")))
            # Validated authored perception (HOW to look: modality/temporal/roi/...) -> goal.state["perception"].
            # Same firewall discipline; an un-authored or malformed block leaves the safe single-frame default.
            goal.state.update(coerce_authored_perception(g.get("perception")))
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
                payload = json.loads(_extract_json(text))
            except json.JSONDecodeError:
                # Never confidently mis-route on a genuinely malformed reply: degrade to a
                # grounded reply (the safe default — but only after robust extraction failed).
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
