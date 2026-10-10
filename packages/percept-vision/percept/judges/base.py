"""The pure part of every judge: the prompt, the reply schema, and the accounting.

A vendor adapter only moves bytes. Everything that decides what is asked and what an answer
means lives here, so every provider is held to the same contract:

* ``build_prompt`` turns a ``JudgeRequest`` into a system and a user text;
* ``parse_reply`` turns the model's text into exactly one ``Observation`` per requested
  condition, marking anything unusable as a failure instead of guessing;
* ``worst_case_cost`` / ``cost_of`` price a request before and after it is sent;
* ``attribute_usage`` shares one request's usage among the observations it produced.

No I/O, no clock, no asyncio.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass

from percept.contracts import (
    ANSWERS,
    Failure,
    JudgeItem,
    JudgeRequest,
    JudgeResult,
    Observation,
    PriceModel,
    Usage,
)

MAX_REASON = 160
CHARS_PER_TOKEN_UPPER = 3     # ASCII text; real tokenisers average closer to 4 chars per token
                              # (non-ASCII characters are counted as one token each)
PROMPT_OVERHEAD_TOKENS = 64   # message framing a provider adds around the texts


@dataclass(frozen=True)
class Prompt:
    """What is sent to the model, besides the images."""

    system: str
    user: str


@dataclass(frozen=True)
class RawReply:
    """What an adapter's ``send`` returns: the model's text and the tokens it was billed for."""

    text: str
    tokens_in: int = 0
    tokens_out: int = 0


class RetryableError(Exception):
    """Raised by ``send`` for a transient failure worth one more attempt (rate limit, 5xx).

    ``retry_after_s`` is the provider's hint, when it gives one.
    """

    def __init__(self, message: str = "", retry_after_s: float | None = None) -> None:
        super().__init__(message)
        self.retry_after_s = retry_after_s


class RefusalError(Exception):
    """Raised by ``send`` when the provider declined to answer (safety block, refusal)."""


# --------------------------------------------------------------------------- prompt

SYSTEM = (
    "You check plain-language conditions against images from a camera.\n"
    "Judge each listed condition independently, using only what is visible in the images. "
    "Do not guess: answer \"unknown\" when the images do not show enough to decide.\n"
    "Reply with JSON only, matching exactly this schema:\n"
    '{"results": [{"id": "<condition id>", "answer": "yes" | "no" | "unknown", '
    '"confidence": <0 to 1: how sure you are of your answer>, "reason": "<at most 160 characters>"}]}\n'
    "Include exactly one result for every listed condition id, and no other ids."
)

EVENT_RULE = "answer yes only if the event itself is visible in these images, not merely its end state"
STATE_RULE = "answer yes if this is true in the images"
VERIFY_NOTE = ("Be strict: answer yes only if the images clearly show it, no only if they clearly "
               "show it did not happen, otherwise unknown.")


def _images_line(request: JudgeRequest) -> str:
    n = len(request.images)
    t0, t1 = request.evidence_t
    if n == 1:
        return f"There is 1 image, taken at wall-clock time {t0:.3f}."
    return (f"There are {n} images in time order: image 1 is the oldest and image {n} the newest. "
            f"They cover {t1 - t0:.1f} s, from wall-clock time {t0:.3f} to {t1:.3f}.")


def _quote(text: str) -> str:
    return json.dumps(text, ensure_ascii=False)


def _item_block(item: JudgeItem) -> str:
    # Counter-examples are sent as their notes only; ``CounterExample.thumbnail_jpeg`` is not
    # sent yet, and so is not counted by ``worst_case_tokens`` either.
    c = item.condition
    lines = [f"- id: {_quote(c.id)}",
             f"  condition: {_quote(c.text)}",
             f"  kind: {c.kind} ({EVENT_RULE if c.kind == 'event' else STATE_RULE})"]
    if item.subject != "scene":
        lines.append(f"  about: {_quote(item.subject)}")
    lines += [f"  FALSE case, answer no: {_quote(s)}" for s in c.not_when]
    lines += [f"  previously mistaken; this is NOT it: {_quote(ce.note)}"
              for ce in item.counter_examples if ce.note.strip()]
    return "\n".join(lines)


def build_prompt(request: JudgeRequest) -> Prompt:
    """The system and user texts for ``request``. Images are sent in request order."""
    parts = [_images_line(request)]
    if request.purpose == "verify":
        parts.append(VERIFY_NOTE)
    parts.append("Conditions:")
    parts += [_item_block(item) for item in request.items]
    return Prompt(SYSTEM, "\n".join(parts))


# --------------------------------------------------------------------------- reply


def _observation(request: JudgeRequest, item: JudgeItem, judge_name: str, *, answer: str = "unknown",
                 confidence: float = 0.0, reason: str = "", failure: Failure | None = None) -> Observation:
    return Observation(
        condition_id=item.condition.id,
        subject=item.subject,
        answer=answer,  # type: ignore[arg-type]
        confidence=confidence,
        reason=reason,
        request_id=request.request_id,
        stream_id=request.stream_id,
        stream_epoch=request.stream_epoch,
        condition_version=item.condition.version,
        occurrence_gen=item.occurrence_gen,
        evidence_t=request.evidence_t,
        judge=judge_name,
        purpose=request.purpose,
        failure=failure,
    )


def failed_observations(request: JudgeRequest, failure: Failure, judge_name: str,
                        reason: str = "") -> tuple[Observation, ...]:
    """One failed observation per requested condition, in request order."""
    return tuple(_observation(request, item, judge_name, reason=reason or failure, failure=failure)
                 for item in request.items)


def extract_json(text: str) -> dict | None:
    """The first JSON object in ``text`` that has a ``results`` key.

    Tolerates code fences and prose around the object; returns None when there is none.
    """
    decoder = json.JSONDecoder()
    start = text.find("{")
    tries = 0
    while start != -1 and tries < 64:
        try:
            value, _ = decoder.raw_decode(text, start)
        except (ValueError, RecursionError):  # deep nesting overflows the decoder
            value = None
        if isinstance(value, dict) and "results" in value:
            return value
        start = text.find("{", start + 1)
        tries += 1
    return None


def _read_entry(entry: dict) -> tuple[str, float, str] | None:
    """(answer, confidence, reason) from one result entry, or None if it breaks the schema."""
    answer, confidence, reason = entry.get("answer"), entry.get("confidence"), entry.get("reason", "")
    if not isinstance(answer, str) or answer.strip().lower() not in ANSWERS:
        return None
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        return None
    if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        return None
    if not isinstance(reason, str):
        return None
    return answer.strip().lower(), float(confidence), reason.strip()[:MAX_REASON]


def parse_reply(text: str, request: JudgeRequest, judge_name: str) -> tuple[Observation, ...]:
    """Exactly one observation per requested condition, in request order. Never raises.

    A reply with no usable ``results`` list fails every id as ``malformed``. Otherwise each id
    is judged on its own entry: absent is ``missing``; a wrong type, an unknown answer, an
    out-of-range confidence or conflicting duplicates are ``malformed``. Ids that were not
    requested are ignored. Confidence is passed through, never clamped or thresholded.
    """
    payload = extract_json(text) if isinstance(text, str) else None
    results = payload.get("results") if payload else None
    if not isinstance(results, list):
        return failed_observations(request, "malformed", judge_name, "reply is not the expected JSON")

    wanted = {item.condition.id for item in request.items}
    found: dict[str, tuple[str, float, str] | None] = {}
    for entry in results:
        if not isinstance(entry, dict) or not isinstance(entry.get("id"), str) or entry["id"] not in wanted:
            continue
        cid, read = entry["id"], _read_entry(entry)
        if cid in found and found[cid] != read:
            read = None  # two different answers for one id: trust neither
        found[cid] = read

    out = []
    for item in request.items:
        cid = item.condition.id
        if cid not in found:
            out.append(_observation(request, item, judge_name, reason="no result for this id", failure="missing"))
        elif found[cid] is None:
            out.append(_observation(request, item, judge_name, reason="result breaks the schema",
                                    failure="malformed"))
        else:
            answer, confidence, reason = found[cid]  # type: ignore[misc]
            out.append(_observation(request, item, judge_name, answer=answer, confidence=confidence, reason=reason))
    return tuple(out)


# --------------------------------------------------------------------------- cost


def cost_of(price: PriceModel, tokens_in: int, tokens_out: int) -> float:
    """Dollars for one request's billed tokens."""
    return (tokens_in * price.usd_per_mtok_in + tokens_out * price.usd_per_mtok_out) / 1_000_000


def _text_tokens(text: str) -> int:
    ascii_chars = sum(1 for ch in text if ord(ch) < 128)
    return math.ceil(ascii_chars / CHARS_PER_TOKEN_UPPER) + (len(text) - ascii_chars)


def worst_case_tokens(request: JudgeRequest, price: PriceModel) -> tuple[int, int]:
    """Upper estimate of (tokens in, tokens out) for one attempt at ``request``.

    Counter-example thumbnails are not sent, so they are not counted.
    """
    prompt = build_prompt(request)
    text_tokens = _text_tokens(prompt.system) + _text_tokens(prompt.user)
    tokens_in = len(request.images) * price.tokens_per_image + text_tokens + PROMPT_OVERHEAD_TOKENS
    return tokens_in, request.max_output_tokens


def worst_case_cost(request: JudgeRequest, price: PriceModel, *, retries: int = 1) -> float:
    """The dollar amount to reserve before sending ``request``.

    ``retries`` mirrors ``policy.call``: up to ``retries + 1`` attempts may be billed, and a
    timed-out attempt is billed without reporting its tokens, so every attempt is reserved.
    """
    return attempts_for(retries) * cost_of(price, *worst_case_tokens(request, price))


def attempts_for(retries: int) -> int:
    """The most provider requests one ``policy.call`` with ``retries`` can make."""
    return retries + 1


def attribute_usage(result: JudgeResult) -> tuple[Usage, ...]:
    """Each observation's share of the usage of the request that produced it.

    Aligned with ``result.observations``. All usage recorded under one request id is summed and
    divided equally among that request's observations with ``Usage.split``, so the integer
    counts of the shares add up exactly to the request totals.
    """
    totals: dict[str, Usage] = {}
    for ru in result.usage:
        totals[ru.request_id] = totals.get(ru.request_id, Usage()) + ru.usage
    groups: dict[str, list[int]] = {}
    for index, ob in enumerate(result.observations):
        groups.setdefault(ob.request_id, []).append(index)
    shares: list[Usage] = [Usage()] * len(result.observations)
    for request_id, indexes in groups.items():
        for index, share in zip(indexes, totals.get(request_id, Usage()).split(len(indexes))):
            shares[index] = share
    return tuple(shares)
