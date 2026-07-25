"""Claude (Anthropic) vision adapter for narrow grounded visual questions.

Call the cloud VLM as a **narrow grounded question about a cropped ROI**, rather than a generic
scene description — *"is the user drinking
from THIS mug <exemplar>?"* — and attach the deictic exemplar image when the user demonstrated
("this mug"). The judge returns the firing-contract ``Verdict{satisfied, confidence, evidence}``;
it never compiles the goal to ``attr=value`` (one firing path, one gate).

**Lane discipline.** This is an **L2-only** backend: it needs a key and the ``anthropic`` SDK.
The module imports cleanly with neither present (the core never imports a vendor SDK at module
top level — invariant 7; the SDK is **lazy-imported inside methods**). Calling a method without
the key/SDK raises a clear, actionable error. The L1 lane uses ``FakeVision``, never this.

The model id is taken from ``settings.models.vision_model`` — no literal in logic (rule 5).
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import re
from typing import Any

from ..adapters import Box, Goal, Observation, Verdict
from ._resilience import CircuitBreaker, with_retry

# A tight system prompt that pins the contract: narrow grounded yes/no + calibrated confidence
# + a one-line evidence string, as strict JSON. Kept here (a prompt is data for this adapter,
# not a tunable of the cognition logic) — the firing decision still lives in the gate.
#
# This is the STABLE per-concern instruction prefix: it never varies frame-to-frame. At 24/7
# capture volume it is re-sent on every judge call, so it is the natural prompt-caching breakpoint
# — see ``_system_blocks`` below, which attaches an ephemeral ``cache_control`` so the cloud only
# re-bills the variable per-frame suffix (the images + the narrow question) on each call.
_JUDGE_SYSTEM = (
    "You are a perception module answering ONE narrow grounded question about a specific "
    "region of an image. Answer only about what is visible in the cropped region (and the "
    "reference exemplar if one is attached). Do NOT describe the whole scene. Return STRICT "
    'JSON: {"satisfied": <true|false>, "confidence": <0..1>, "evidence": "<short reason>"}. '
    "Set confidence low when the region is ambiguous, occluded, or you are unsure — a wrong "
    "confident yes is the worst outcome."
)

# CLAIM-MODE contract addendum (1.5). Appended to the per-frame question ONLY when the caller marks
# the goal claim-mode (``context["claim_mode"]`` — set in ``agent.perceive_judged`` when the concern's
# resolved identity policy is ``claim_slot``). It advertises the OPTIONAL mutation-channel fields the
# claim ledger folds (:func:`_apply_claim_fields` maps them onto the existing ``Verdict`` fields). A
# non-claim goal never sees this text, so its prompt is byte-for-byte unchanged. Provider-agnostic:
# gemini/openrouter import it so the correction contract is identical on every parsing backend.
_CLAIM_MODE_CONTRACT = (
    " This watch tracks a CLAIMED VALUE a person may later CORRECT. In ADDITION to the fields above, "
    "you MAY include these OPTIONAL fields, but ONLY when the evidence plainly supports them: "
    '"operation" — one of "assert", "replace", "retract", "confirm" (DEFAULT "assert"). Use "replace" '
    "ONLY for an explicit correction speech-act — \"actually it's X\", \"correction: ...\", a visible "
    "reversal of a value stated earlier. A question, hypothetical, or proposal (\"could we do X?\") is "
    'NEVER a correction — leave "operation" at "assert". Use "retract" when a value is explicitly '
    'withdrawn ("never mind", "forget that"). Use "confirm" for a restatement of the SAME value. '
    '"value" — the asserted or corrected value exactly as heard/seen (e.g. "$25K"). "normalized_value" '
    "— a canonical form ONLY when you are confident (e.g. 25000); otherwise OMIT it. Never invent an "
    "operation you are unsure of: when in doubt omit all three and the default assert applies. Do NOT "
    "state who spoke."
)

# Outbound-call timeout (seconds) for the Anthropic client. A hung socket on the hot 24/7 path
# must surface as a retryable failure, not a stall — the SDK raises ``APITimeoutError`` past this
# and ``with_retry`` re-attempts. Overridable via ``PERCEPT_VISION_TIMEOUT_S`` (rule 5: no literal
# in logic). Kept here (a transport knob for this L2 adapter) rather than in the core Settings tree.
_DEFAULT_TIMEOUT_S = 30.0


def _timeout_s() -> float:
    try:
        return float(os.environ.get("PERCEPT_VISION_TIMEOUT_S", _DEFAULT_TIMEOUT_S))
    except (TypeError, ValueError):
        return _DEFAULT_TIMEOUT_S


def _system_blocks(text: str) -> list[dict]:
    """Render a stable instruction prefix as a cacheable Anthropic ``system`` block list.

    The ``cache_control: {"type": "ephemeral"}`` breakpoint marks this prefix as a cache write
    point: subsequent judge calls on the same per-zone/per-concern prefix are served from cache
    (~0.1x input cost) instead of re-billing the full prefix every frame. Anthropic only caches a
    prefix above a model-dependent minimum (~1024–4096 tokens), so on short prefixes this is a
    correct no-op (it silently won't cache) — the win scales with prefix size at volume.
    """
    return [{"type": "text", "text": text, "cache_control": {"type": "ephemeral"}}]


def _require_anthropic():
    """Lazy-import the Anthropic SDK with a clear error if it is absent (L2 boundary)."""
    try:
        import anthropic  # noqa: PLC0415 — lazy by design (invariant 7)
    except ModuleNotFoundError as exc:  # pragma: no cover — exercised only without the extra
        raise RuntimeError(
            "AnthropicVision needs the 'anthropic' SDK — install percept-vision[claude]. "
            "(L2 only; the L1 lane uses the fake vision backend.)"
        ) from exc
    return anthropic


def _settings_of(arg: Any):
    """Accept either a full Settings tree or a ModelSettings; tolerate None for late binding."""
    if arg is None:
        from ..config import load_settings

        return load_settings()
    return arg


class AnthropicVision:
    """``Vision`` adapter backed by Claude. Narrow grounded VQA on a cropped ROI.

    Constructed by the registry as ``AnthropicVision(settings)``. The client is built lazily on
    first call so the module imports with no key present (L2 boundary).
    """

    #: This backend PARSES the claim-mode correction channel (operation/value) — read by
    #: ``watchplan_apply`` to decide whether a ``claim_slot`` plan needs the "backend unverified"
    #: warning. A client backend that emits the channel declares the same marker.
    PARSES_CLAIM_CHANNEL = True

    def __init__(self, settings: Any = None, *, client: Any = None) -> None:
        self._settings = _settings_of(settings)
        self._client = client  # injectable for L2 fixtures; built lazily otherwise
        self._model = getattr(
            getattr(self._settings, "models", self._settings), "vision_model", "claude-opus-4-8"
        )
        # Per-instance breaker so a string of transient cloud failures trips OPEN and stops
        # hammering the hot path (mirrors gemini.py); ``with_retry`` adds exponential backoff.
        self._breaker = CircuitBreaker()
        self._timeout_s = _timeout_s()

    def _with_timeout(self, client):
        """Apply the explicit per-call timeout if the SDK supports ``with_options`` (real client);
        fall back to the bare client for L2 fakes that don't model client options."""
        with_options = getattr(client, "with_options", None)
        if callable(with_options):
            return with_options(timeout=self._timeout_s)
        return client

    # -- client ------------------------------------------------------------
    def _ensure_client(self):
        if self._client is None:
            anthropic = _require_anthropic()
            self._client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the env
        return self._client

    # -- the firing-contract judge ----------------------------------------
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
        """Ask Claude a narrow grounded yes/no about ``condition`` over the cropped ROI.

        The prompt grounds on the ROI crop; when an ``exemplar`` is present it is attached as a
        second image and the question is phrased deictically ("...from THIS one <exemplar>?").
        Any user ``corrections`` are surfaced as authoritative context (asserted beats observed), as is
        structured specialist ``context`` the caller supplied (e.g. an LPR plate
        read). A temporal ``frames`` strip is attached as extra images; ``audio`` is IGNORED — Claude has
        no audio modality, so a ``fused`` concern degrades to the visual strip (``fused``/``audio`` need
        ``vision="gemini"``).
        """
        del audio  # accepted for protocol parity; Claude is vision-only (honest degrade)
        client = self._with_timeout(self._ensure_client())
        claim_mode = bool(context and context.get("claim_mode"))  # claim-slot concern -> parse the correction channel
        content = self._build_content(condition, frame, roi=roi, exemplar=exemplar,
                                      corrections=corrections, frames=frames, context=context,
                                      claim_mode=claim_mode)

        async def _call():
            # The SDK call is synchronous — run it off the event loop so the explicit timeout and
            # retries don't block other captures. ``system`` carries the cached per-concern prefix.
            return await asyncio.to_thread(
                client.messages.create,
                model=self._model,
                max_tokens=512,  # 256 truncated verbose verdicts mid-JSON -> spurious "unparsed" misses
                system=_system_blocks(_JUDGE_SYSTEM),
                messages=[{"role": "user", "content": content}],
            )

        # Survive transient resets/timeouts in batch captures; trip the breaker on a run of failures.
        resp = await with_retry(_call, breaker=self._breaker)
        verdict = _parse_verdict(_text_of(resp), claim_mode=claim_mode)
        verdict.usage = _usage_of(resp)  # 1.3 metering: token/cost funnel (was always {})
        return verdict

    async def detect(self, frame: Any, goals: list[Goal]) -> list[Observation]:
        """Detect goal-relevant things in a frame (the entity-attribute substrate).

        Kept deliberately thin: the entity graph's continuity needs boxes + labels, not a scene
        description. Returns ``[]`` on a parse miss rather than fabricating detections.
        """
        client = self._with_timeout(self._ensure_client())
        labels = ", ".join(sorted({g.condition for g in goals})) or "any salient object"
        prompt = (
            "Detect goal-relevant objects. For each, return JSON list items "
            '{"label": str, "box": [x,y,w,h] normalized 0..1, "confidence": 0..1}. '
            f"Goals of interest: {labels}. Return STRICT JSON list (possibly empty)."
        )
        content = [_image_block(frame), {"type": "text", "text": prompt}]

        async def _call():
            return await asyncio.to_thread(
                client.messages.create,
                model=self._model,
                max_tokens=512,
                messages=[{"role": "user", "content": content}],
            )

        resp = await with_retry(_call, breaker=self._breaker)
        return _parse_observations(_text_of(resp))

    async def detect_events(self, frames: list, goals: list[Goal]) -> list:
        """Temporal events land in Phase 5; this adapter defers (returns ``[]``)."""
        return []

    # -- prompt construction ----------------------------------------------
    def _build_content(
        self,
        condition: str,
        frame: Any,
        *,
        roi: Box | None,
        exemplar: Any,
        corrections: dict | None,
        frames: list | None = None,
        context: dict | None = None,
        claim_mode: bool = False,
    ) -> list:
        """Build the multimodal user turn: the temporal strip + ROI crop, the exemplar, and the
        narrow question. ``frames`` are earlier-context images (oldest→newest); the current frame is
        last so the model reads time from image order.

        ``claim_mode`` (default ``False``) appends the correction-channel contract
        (:data:`_CLAIM_MODE_CONTRACT`) so a claim-slot concern can emit ``operation``/``value``; a
        non-claim goal leaves it ``False`` and the assembled prompt is byte-for-byte unchanged."""
        strip = [f for f in (frames or []) if f is not frame]  # earlier context, oldest->newest
        content: list = [_image_block(f, roi=roi, label="earlier moment") for f in strip]
        content.append(_image_block(frame, roi=roi, label="region of interest"))
        if exemplar is not None:
            content.append(_image_block(exemplar, label="reference exemplar"))
            question = (
                f'Looking only at the region of interest, and comparing to the reference '
                f'exemplar: is this true — "{condition}" — about THIS specific object? '
            )
        elif strip:
            question = (
                f'These {len(strip) + 1} region images are consecutive moments in TIME ORDER (oldest '
                f'first). Considering how the scene CHANGES across them, is this true by the latest '
                f'moment — "{condition}"? '
            )
        else:
            question = (
                f'Looking only at the region of interest: is this true right now — '
                f'"{condition}"? '
            )
        if corrections:
            facts = "; ".join(f"{k}={v}" for k, v in corrections.items())
            question += (
                f"The user has told you these authoritative facts (trust them over your own "
                f"read): {facts}. "
            )
        if context:
            # ``claim_mode`` is a RESERVED control key (not a specialist fact) — never surface it as one.
            spec_facts = "; ".join(f"{k}={v}" for k, v in context.items() if k != "claim_mode")
            if spec_facts:
                question += (
                    f"A specialist model has reported these structured facts (trust them over your own "
                    f"read): {spec_facts}. "
                )
        question += 'Answer as STRICT JSON {"satisfied", "confidence", "evidence"}.'
        if claim_mode:
            question += _CLAIM_MODE_CONTRACT
        content.append({"type": "text", "text": question})
        return content


# ---------------------------------------------------------------------------
# Image + response helpers (stdlib only; no vendor types leak into the core)
# ---------------------------------------------------------------------------


def _image_block(frame: Any, *, roi: Box | None = None, label: str | None = None) -> dict:
    """Render an opaque frame to an Anthropic image content block.

    Accepts already-encoded ``{"type": "image", ...}`` blocks (pass-through), raw bytes (assumed
    PNG/JPEG; base64-encoded), or a ``(media_type, bytes)`` pair. ROI cropping of real pixels is
    the caller's job (the narrow-VQA win is *what we ask*, grounded by the crop); we pass the
    ``roi`` through as a hint only when the frame is a plain descriptor.
    """
    if isinstance(frame, dict) and frame.get("type") == "image":
        return frame
    media_type = "image/png"
    raw: bytes
    if isinstance(frame, tuple) and len(frame) == 2 and isinstance(frame[1], (bytes, bytearray)):
        media_type, raw = str(frame[0]), bytes(frame[1])
    elif isinstance(frame, (bytes, bytearray)):
        raw = bytes(frame)
    else:
        # A non-image descriptor (e.g. a path or token) — let the SDK caller pre-encode in L2.
        raise TypeError(
            "AnthropicVision needs image bytes / an Anthropic image block as the frame; got "
            f"{type(frame).__name__}. Pre-encode the frame (and crop the ROI) before judging."
        )
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": media_type,
            "data": base64.b64encode(raw).decode("ascii"),
        },
    }


def _usage_of(resp: Any) -> dict:
    """Pull token usage off an Anthropic Messages response for ``Verdict.usage`` (read by
    ``observability.judge_usage`` → the cost/token funnel). Defensive: a response/mapping without
    ``usage`` (fakes/tests) yields ``{}`` and the metering hook stays a no-op."""
    usage = getattr(resp, "usage", None)
    if usage is None and isinstance(resp, dict):
        usage = resp.get("usage")
    if usage is None:
        return {}
    out: dict[str, int] = {}
    it = getattr(usage, "input_tokens", None) if not isinstance(usage, dict) else usage.get("input_tokens")
    ot = getattr(usage, "output_tokens", None) if not isinstance(usage, dict) else usage.get("output_tokens")
    if isinstance(it, int):
        out["input_tokens"] = it
    if isinstance(ot, int):
        out["output_tokens"] = ot
    return out


def _text_of(resp: Any) -> str:
    """Pull the concatenated text out of an Anthropic Messages response."""
    blocks = getattr(resp, "content", None)
    if blocks is None and isinstance(resp, dict):
        blocks = resp.get("content")
    parts: list[str] = []
    for b in blocks or []:
        text = getattr(b, "text", None)
        if text is None and isinstance(b, dict):
            text = b.get("text")
        if text:
            parts.append(text)
    return "".join(parts)


def _coerce_json(text: str) -> Any:
    """Extract the first JSON object/array from a possibly-fenced model reply."""
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
    start = min(
        [i for i in (text.find("{"), text.find("[")) if i != -1] or [-1]
    )
    if start == -1:
        raise ValueError("no JSON found in model reply")
    # bound to the LAST closing brace/bracket so a trailing ``` fence or prose can't break json.loads
    # (the verdict parser previously took text[start:] unbounded -> a fenced reply parsed as "unparsed").
    end = max(text.rfind("}"), text.rfind("]"))
    return json.loads(text[start : end + 1] if end > start else text[start:])


# Field-level salvage for a TRUNCATED verdict. A thinking model (gemini-2.5/3-flash) can spend its
# output budget on reasoning and cut the STRICT-JSON reply mid-``evidence`` string — but ``satisfied``
# and ``confidence`` come FIRST and are intact. Recovering them beats discarding a real read as
# ``unparsed`` -> confidence 0.0 -> a spurious ASK (the false-abstain the token bump was meant to avoid).
_SATISFIED_RE = re.compile(r'"satisfied"\s*:\s*(true|false)', re.IGNORECASE)
_CONFIDENCE_RE = re.compile(r'"confidence"\s*:\s*(-?\d*\.?\d+)')
_EVIDENCE_RE = re.compile(r'"evidence"\s*:\s*"([^"]*)')  # may be unterminated (truncation point)


def _salvage_verdict(text: str) -> Verdict | None:
    """Recover ``satisfied``/``confidence`` from a truncated/malformed reply, else ``None``.

    Requires BOTH fields (a verdict is meaningless without them); if either is missing the caller
    keeps the conservative 0.0 default and the gate still defers to ASK (invariant 1 preserved)."""
    m_sat, m_conf = _SATISFIED_RE.search(text), _CONFIDENCE_RE.search(text)
    if m_sat is None or m_conf is None:
        return None
    try:
        conf = max(0.0, min(1.0, float(m_conf.group(1))))
    except ValueError:
        return None
    m_ev = _EVIDENCE_RE.search(text)
    return Verdict(satisfied=m_sat.group(1).lower() == "true", confidence=conf,
                   evidence=(m_ev.group(1).strip() if m_ev else ""))


# The closed set of operations a CLAIM-MODE judge may emit (a strict subset of the contract's
# ``Operation`` vocabulary — ``close``/``mention`` are ledger-internal, never judge-authored). Anything
# outside this set degrades to ``"assert"``: a parser never INVENTS a mutation the model did not clearly
# request (invariant 1 — a value DIFFERENCE must never be mistaken for a correction).
_CLAIM_OPERATIONS: frozenset[str] = frozenset({"assert", "replace", "retract", "confirm"})


def _apply_claim_fields(verdict: Verdict, data: Any) -> None:
    """Map the OPTIONAL claim-mode channel fields off a parsed judge reply onto ``verdict`` in place.

    Strict validation, never fabrication: an unknown/malformed/absent ``operation`` degrades to
    ``"assert"`` (the safe default — asserts can only ESTABLISH an empty slot, never overwrite);
    ``value`` is taken verbatim as the judge emitted it; ``normalized_value`` is folded only when
    present. ``asserted_by`` is NEVER populated — there is no diarization, so a speaker is never
    fabricated (authority defaults to unrestricted, so revision chains still form)."""
    if not isinstance(data, dict):
        return
    op = data.get("operation", "assert")
    op = op.strip().lower() if isinstance(op, str) else "assert"
    verdict.operation = op if op in _CLAIM_OPERATIONS else "assert"
    if "value" in data:
        verdict.value = data.get("value")  # verbatim as heard/seen; may be None
    nv = data.get("normalized_value")
    if nv is not None:
        verdict.normalized_value = nv
    # asserted_by intentionally left at its default (None): no speaker attribution is ever invented.


def _parse_verdict(text: str, *, claim_mode: bool = False) -> Verdict:
    """Parse the strict-JSON judge reply into a ``Verdict`` (defensive defaults on a bad read).

    A parse failure is treated as *low-confidence unknown* (satisfied=False, confidence=0.0) so
    the gate defers to ASK rather than risking a confident-wrong fire (invariant 1) — UNLESS the
    decisive fields survive a truncation, in which case :func:`_salvage_verdict` recovers them.

    ``claim_mode`` (default ``False``) additionally folds the OPTIONAL correction channel
    (:func:`_apply_claim_fields`) onto the verdict; a non-claim call sets none of those fields and is
    byte-for-byte the pre-claim parser (the malformed-JSON salvage path is shared, unchanged)."""
    try:
        data = _coerce_json(text)
        ap = data.get("action_present")
        phase = str(data.get("phase", "")).lower()
        verdict = Verdict(
            satisfied=bool(data.get("satisfied", False)),
            confidence=float(data.get("confidence", 0.0)),
            evidence=str(data.get("evidence", "")),
            # optional structured fields (clip judging); absent in single-frame replies -> defaults
            phase=phase if phase in ("pre", "during", "post") else "",
            action_present=(bool(ap) if isinstance(ap, bool) else None),
        )
        if claim_mode:
            _apply_claim_fields(verdict, data)
        return verdict
    except (ValueError, TypeError, AttributeError, KeyError):
        salvaged = _salvage_verdict(text)
        if salvaged is not None:
            return salvaged
        return Verdict(satisfied=False, confidence=0.0, evidence=f"unparsed: {text[:120]}")


def _parse_observations(text: str) -> list[Observation]:
    try:
        data = _coerce_json(text)
    except ValueError:
        return []
    out: list[Observation] = []
    for item in data if isinstance(data, list) else []:
        try:
            box = item["box"]
            out.append(
                Observation(
                    label=str(item["label"]),
                    box=Box(float(box[0]), float(box[1]), float(box[2]), float(box[3])),
                    confidence=float(item.get("confidence", 0.5)),
                )
            )
        except (KeyError, TypeError, ValueError, IndexError):
            continue
    return out


# ---------------------------------------------------------------------------
# Reference "production judge"
# ---------------------------------------------------------------------------


def production_judge(settings: Any = None, *, client: Any = None) -> AnthropicVision:
    """Reference constructor documenting the hardened settings of the Claude vision judge.

    ``AnthropicVision`` is already production-hardened — this is a single, discoverable entry point
    that names the levers for a 24/7 deployment, all of which live on the adapter:

    * **Retry + backoff** — every ``judge``/``detect`` cloud call goes through
      :func:`._resilience.with_retry` (3 attempts, exponential backoff: 0.2s, 0.4s).
    * **Circuit breaker** — a per-instance :class:`._resilience.CircuitBreaker` (opens after 5
      failures, half-opens after 30s) short-circuits the hot path during a sustained outage instead
      of hammering it.
    * **Explicit timeout** — ``PERCEPT_VISION_TIMEOUT_S`` (default 30s) is applied via the SDK's
      ``with_options(timeout=...)`` so a hung socket surfaces as a *retryable* failure, not a stall.
    * **Prompt caching** — the stable per-zone/per-concern instruction prefix is sent as a ``system``
      block carrying an ephemeral ``cache_control`` breakpoint, so the per-frame call only re-bills
      the variable suffix (the images + the narrow question) — a large cost lever at 24/7 volume.

    Returns the same ``AnthropicVision`` the registry builds for ``vision="claude"``; prefer this
    when constructing the judge directly and you want the hardening choices spelled out at the call
    site.
    """
    return AnthropicVision(settings, client=client)


__all__ = ["AnthropicVision", "production_judge"]
