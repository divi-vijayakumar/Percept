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
import math
import os
import re
from typing import Any

from ..adapters import Box, Goal, Observation, Verdict
from ..contracts import ClipEvent
from ._resilience import CircuitBreaker, with_retry
# The router publishes the latency mode of the route it selected; this module owns the
# provider-agnostic reply CONTRACT (prompt + parser + output budget) that reads it, so every
# parsing backend asks for the identical shape on the identical route. No cycle: the pool knows
# nothing about backends.
from .judge_pool import LATENCY_MODE_CONTEXT_KEY, REFLEX_LATENCY_MODE

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

# The REFLEX variant of the same contract, used when a route declares itself latency-bound (see
# ``percept.backends.judge_pool.LATENCY_MODE_CONTEXT_KEY``). Provider-agnostic and kept beside
# :data:`_JUDGE_SYSTEM` so every parsing backend asks for the identical shape.
#
# Why it exists: streaming decomposition of the reflex gate showed the miss was in the GENERATION
# half, not ingestion — a route spent seconds emitting the ``evidence`` prose *after* it already
# knew the answer. ``satisfied``/``confidence`` are the only two fields the gate reads (``Gate``
# compares ``confidence`` against ``Settings.gate.threshold``), so a reflex judge is asked for
# exactly those two and nothing else. ``confidence`` is NOT optional: without it the parser
# defaults to 0.0 and every verdict fails closed to ASK, which is safe but useless.
#
# What is lost: the one-line free-text ``evidence`` string. It is GATE-IRRELEVANT (nothing in
# cognition branches on it) but it is carried on ``Verdict.evidence`` into ``Incident.verdict``
# and the ``judgment``/transition trace rows, so a reflex fire is explainable by its frames,
# confidence and receipt rather than by a provider sentence.
_JUDGE_SYSTEM_REFLEX = (
    "You are a perception module answering ONE narrow grounded question about a specific "
    "region of an image. Answer only about what is visible in the cropped region (and the "
    "reference exemplar if one is attached). Do NOT describe the scene and do NOT explain "
    'yourself. Return STRICT JSON and nothing else: {"satisfied": <true|false>, '
    '"confidence": <0..1>}. Both fields are REQUIRED. Set confidence low when the region is '
    "ambiguous, occluded, or you are unsure — a wrong confident yes is the worst outcome."
)


# --- reflex output budget: a ROUTE property, never a global downgrade ----------------------------
# WHAT A REFLEX REPLY IS: exactly the two fields the gate reads, and nothing else. That is a
# statement about the SHAPE of the answer, not about a clock — it holds whatever the deadline is.
#
# Measured on the benchmark's exact evidence payload with real streaming calls: a 1024-token
# "headroom" invited a route to spend 3.13s emitting 91 tokens it did not need; asking for two
# scalars instead of a prose verdict dropped generation to 0.03-0.19s and total latency from 2.80s
# to 0.91s median. That saving is real and is why the bound is kept — it buys cost and speed on
# every call. TTFT (~0.5-1.0s) is the ingestion floor and is NOT addressed here.
#
# (The measurement above was originally framed as clearing a "sub-1.5s reflex gate". That gate is
# gone — it sat below the median provider call time and cancelled the majority of judgments; see
# ``percept.backends.judge_pool.REFLEX_PROVIDER_DEADLINE_S`` and ``percept.slo`` §5. The token
# bound survives the gate's removal on its own merits: a terse verdict is cheaper and faster, and
# nothing in cognition reads the prose it drops.)
#
# So a reflex budget is sized to the VERDICT, not to a model's thinking habit. A deliberate call
# keeps each provider's historical ceiling: an open/thinking model on a non-latency-bound route may
# still spend tokens before its JSON, and truncating it there turns a real read into a false abstain.
_REFLEX_MAX_TOKENS = 32  # measured: {"satisfied": false, "confidence": 0.87} costs 13 output tokens
# A reflex request that must ALSO carry structured temporal fields (``phase``/``action_present``/
# ``events``) cannot go down to two scalars: cognition reads those fields —
# ``behavior.normalize_event_chain_verdict`` abstains outright on an inconsistent
# satisfied/action_present/phase tuple, and the occurrence ledger counts ``events``. Such a request
# drops only the free-text ``evidence`` field and takes a BOUNDED budget instead of open headroom.
_REFLEX_STRUCTURED_MAX_TOKENS = 320


def _is_reflex(context: Any) -> bool:
    """Whether the ROUTE that dispatched this judge call declared itself latency-bound.

    Stamped by :class:`~percept.backends.judge_pool.JudgePool` when ``mode='reflex'``. ABSENT means
    deliberate, so a directly-constructed adapter — or one behind a deliberate pool — is unchanged
    by this whole mechanism.
    """
    return (
        isinstance(context, dict)
        and context.get(LATENCY_MODE_CONTEXT_KEY) == REFLEX_LATENCY_MODE
    )


def _terse_reply(*, reflex: bool, claim_mode: bool) -> bool:
    """Whether this request may ask for a stripped-down reply.

    CLAIM MODE opts out even on a reflex route: the claim ledger reads ``operation``/``value``/
    ``normalized_value`` off the reply (:func:`_apply_claim_fields`), so trimming the correction
    channel to save latency or tokens would silently drop the mutation the watch exists to catch.
    Neither is ever worth losing a correction.
    """
    return reflex and not claim_mode


def _judge_max_tokens(
    *,
    reflex: bool,
    structured: bool,
    claim_mode: bool,
    deliberate: int,
) -> int:
    """The output budget for ONE judge request, decided by route mode and requested reply shape.

    ``structured`` marks a reply that must still carry temporal/occurrence fields beyond the two
    the gate reads. ``deliberate`` is the provider's own historical ceiling, returned unchanged
    whenever this is not a terse reflex request.
    """
    if not _terse_reply(reflex=reflex, claim_mode=claim_mode):
        return deliberate
    return _REFLEX_STRUCTURED_MAX_TOKENS if structured else _REFLEX_MAX_TOKENS


# --- reflex INPUT bound: the same route policy, applied to the temporal strip ---------------------
# The budget above bounds the GENERATION half of the reflex call. This bounds the INGESTION half,
# which the output work explicitly left open ("TTFT is the ingestion floor and is NOT addressed
# here — frame count is that lever"). It is the SAME mechanism: a property of the ROUTE, read off
# ``LATENCY_MODE_CONTEXT_KEY``, never a global downgrade.
#
# WHAT IT IS: reflex sends the THREE most recent frames — a small, fixed, cheap evidence window.
# Together with the token bound above, that is what makes reflex a distinct mode: bounded evidence
# in, bounded verdict out. Neither half depends on a deadline, and both survived the deadline's
# removal unchanged.
#
# THE ORIGINAL SWEEP, AND WHAT IS NOW KNOWN ABOUT IT. 3 was selected from a live sweep over
# identical 40 cases with frame count the only variable:
#
#     frames | recall | precision |  ASK  |  FIRE
#        2   | 0.449  |   0.354   | 30.4% |   8.9%
#        3   | 0.653  |   0.291   | 25.3% |  14.2%   <- retained
#        4   | 0.314  |   0.312   | 69.0% |   6.2%
#        5   | 0.093  |   0.175   | 83.8% |   2.3%
#
# THE 4- AND 5-FRAME ROWS ARE NOT USABLE EVIDENCE ANY MORE, and this comment will not pretend
# otherwise. Every row was collected under the 1.5s provider deadline that has since been replaced
# (see ``percept.backends.judge_pool.REFLEX_PROVIDER_DEADLINE_S``). The sweep's own reading of the
# upper rows was that "the answer is fine but arrives after the deadline, so the ASK rate is
# dominated by expiry rather than by uncertainty" — i.e. 4 and 5 were penalised by CANCELLATION,
# the exact artefact the deadline change removes. With calls now allowed to complete, the collapse
# at 4-5 may simply not reproduce.
#
# So: 3 is RETAINED, not re-confirmed, and it is not re-tuned here — moving it now, with no
# post-change measurement, would be picking a number to suit a metric. What holds independently of
# the deadline is the 2-frame row: at 2 the strip under-grounds the CHANGE the question asks about,
# so the model abstains more. That is a claim about evidence sufficiency, and it still stands.
# The upper bound is now a COST and EVIDENCE-SHAPE choice: three frames is the cheap lane, and
# widening it must be justified by a re-measured sweep, not inherited from a censored one.
#
# A DELIBERATE route stays unbounded — there the extra frames are free recall.
_REFLEX_MAX_FRAMES = 3


def _judge_max_frames(*, reflex: bool) -> int | None:
    """The temporal frame ceiling for ONE judge request, decided by route mode.

    Counts the WHOLE strip actually sent — the earlier context frames plus the current frame — so
    a bound of 3 means the current frame and the two most recent context frames. ``None`` means
    unbounded: a deliberate route, a directly-constructed adapter, or anything behind a deliberate
    pool keeps its evidence byte-for-byte.

    Deliberately narrower than :func:`_terse_reply` / :func:`_judge_max_tokens`, which carve out
    claim mode: those trim the REPLY, and the claim ledger reads correction fields off the reply.
    This trims the OLDEST evidence only, keeping the most recent frames, so the mutation channel is
    untouched and no carve-out is warranted.
    """
    return _REFLEX_MAX_FRAMES if reflex else None


def _bounded_strip(strip: list, *, max_frames: int | None, has_current_frame: bool = True) -> list:
    """Trim the earlier-context strip so ``strip + current frame`` fits ``max_frames``.

    KEEPS THE MOST RECENT frames and preserves their order: the strip is promised to the model as
    "consecutive moments in TIME ORDER (oldest first)", and recency is where the change signal
    lives, so the OLDEST context is what gets dropped. ``max_frames=None`` returns the strip
    unchanged, and a strip already inside the bound is returned unchanged (never padded).
    """
    if max_frames is None:
        return strip
    keep = max(0, max_frames - (1 if has_current_frame else 0))
    if len(strip) <= keep:
        return strip
    return strip[len(strip) - keep :]


def _bounded_window_origin_s(
    *,
    dropped_frames: int,
    total_frames: int,
    span_s: float,
    offsets: Any = None,
) -> float | None:
    """Where the RETAINED evidence begins on the CLIP'S OWN timeline, in seconds.

    ``None`` whenever the question does not arise: nothing was dropped, the clip is degenerate, or
    the origin cannot be placed inside the announced span.

    EXACT WHEN IT CAN BE. ``_percept_temporal_offsets_s`` is the agent's per-frame source
    provenance for this exact selection (``agent._temporal_evidence_context``), one value per
    ordered image, so the retained origin is simply the offset of the first frame that survived the
    bound. When it is absent the fallback is the sampling the PROMPT ITSELF already describes to
    the model — N consecutive moments spread across the announced span — which is what the model
    would assume anyway.
    """
    if dropped_frames <= 0 or total_frames <= 1 or dropped_frames >= total_frames:
        return None
    span = float(span_s)
    if not math.isfinite(span) or span <= 0.0:
        return None
    if (
        isinstance(offsets, (tuple, list))
        and len(offsets) == total_frames
        and all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            for value in offsets
        )
    ):
        origin = float(offsets[dropped_frames]) - float(offsets[0])
    else:
        origin = span * dropped_frames / (total_frames - 1)
    if not math.isfinite(origin) or origin <= 0.0 or origin >= span:
        return None
    return origin


def _clip_timeline_sentence(
    span_s: Any,
    *,
    dropped_frames: int = 0,
    total_frames: int = 0,
    offsets: Any = None,
    per_frame_labels: bool = False,
) -> str:
    """Tell an occurrence request WHICH TIMELINE its reported times live on.

    An ``event_chain`` reply is decoded downstream against the FULL captured window:
    ``agent._fire_clip_occurrences`` derives ``clip_start_t = source_t - _percept_temporal_span_s``
    and ``behavior.fire_clip_events`` reports each occurrence at ``clip_start_t + t_start``. So the
    model's ``t=0`` must be the start of that window and nothing else.

    :func:`_bounded_strip` breaks that on a request carrying NO per-frame labels: the span sentence
    still describes the whole window while the images now cover only its tail, so an occurrence in
    the retained tail is reported against a timeline that begins earlier than the evidence does and
    is decoded EARLY by up to the dropped prefix. This restates the origin instead of restating the
    span — re-basing the span onto the retained window would not fix it, it would make the error a
    CONSTANT shift of exactly the dropped prefix, because the decoder's ``clip_start_t`` is not
    ours to move.

    ``per_frame_labels=True`` means every image already carries its own ``t=`` label in clip
    coordinates (the OpenRouter timestamped transport), which anchors the timeline exactly and
    needs no prose. Nothing dropped -> the historical sentence, byte for byte, so a deliberate
    route is untouched.
    """
    if isinstance(span_s, bool) or not isinstance(span_s, (int, float)):
        return ""
    span = float(span_s)
    plain = f"The evidence spans {span:.6g} seconds. "
    if per_frame_labels:
        return plain
    origin = _bounded_window_origin_s(
        dropped_frames=dropped_frames,
        total_frames=total_frames,
        span_s=span,
        offsets=offsets,
    )
    if origin is None:
        return plain
    return plain + (
        f"The images shown cover only the later part of that window: the first image is at "
        f"t={origin:.6g}s and the last at t={span:.6g}s. Report every time on the full "
        f"{span:.6g}-second timeline — t=0 is the START of the window, not the first image shown. "
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
        elif context and context.get("event_chain"):
            question = (
                f'These {len(strip) + 1} region images are consecutive moments in TIME ORDER '
                f'(oldest first). Track one discrete occurrence of "{condition}". Classify the '
                "LATEST moment as pre (not begun), during (visibly in progress), or post (just "
                "completed within this clip). Do not infer the action from a static pose; require "
                "visible change across the ordered images. "
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
            spec_facts = "; ".join(
                f"{k}={v}"
                for k, v in context.items()
                if k not in ("claim_mode", "event_chain")
                and not k.startswith("_percept_")
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
            question += (
                'Answer as STRICT JSON {"satisfied", "confidence", "evidence", '
                '"action_present", "phase", "events": [{"t_start": seconds, '
                '"t_end": seconds|null, "label": "short description"}]}. '
                "List every grounded distinct occurrence in chronological order "
                "with times relative to the beginning of the evidence; do not merge "
                "adjacent repetitions. Set action_present=true only for \"during\". "
                "Use satisfied=true for a grounded during or just-completed post occurrence."
            )
        else:
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


def _parse_clip_events(value: Any) -> list[ClipEvent]:
    """Strictly parse a bounded, clip-relative occurrence list."""
    if value is None:
        return []
    if not isinstance(value, list) or len(value) > 64:
        raise ValueError("events must be a bounded list")
    events: list[ClipEvent] = []
    prior_start = -1.0
    for item in value:
        if not isinstance(item, dict):
            raise ValueError("event must be an object")
        start = float(item["t_start"])
        raw_end = item.get("t_end")
        end = None if raw_end is None else float(raw_end)
        if (
            not math.isfinite(start)
            or start < 0.0
            or start < prior_start
            or (
                end is not None
                and (
                    not math.isfinite(end)
                    or end < start
                )
            )
        ):
            raise ValueError("event interval is invalid or unordered")
        prior_start = start
        events.append(
            ClipEvent(
                t_start=start,
                t_end=end,
                label=str(item.get("label", ""))[:160],
            )
        )
    return events


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
        if not isinstance(data, dict) or "satisfied" not in data:
            # A reply that never states the ANSWER is not a verdict. Without this guard a stray
            # ``{"confidence": 0.99}`` parsed as a CONFIDENT NO — which does not merely fail to
            # fire, it RE-ARMS the gate's rising edge and so can mint a duplicate fire on the next
            # true read. Fall through to the salvage/unknown path instead: confidence 0.0 -> ASK,
            # which changes no edge state. (The mirror case — ``satisfied`` with no ``confidence``
            # — is already conservative: it keeps the 0.0 default and defers to ASK.)
            raise ValueError("judge reply carries no 'satisfied' field")
        try:
            clip_events = _parse_clip_events(data.get("events"))
        except ValueError:
            return Verdict(
                satisfied=False,
                confidence=0.0,
                evidence="event-chain abstention: invalid clip events",
            )
        ap = data.get("action_present")
        phase = str(data.get("phase", "")).lower()
        verdict = Verdict(
            satisfied=bool(data.get("satisfied", False)),
            confidence=float(data.get("confidence", 0.0)),
            evidence=str(data.get("evidence", "")),
            # optional structured fields (clip judging); absent in single-frame replies -> defaults
            phase=phase if phase in ("pre", "during", "post") else "",
            action_present=(bool(ap) if isinstance(ap, bool) else None),
            events=clip_events,
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
