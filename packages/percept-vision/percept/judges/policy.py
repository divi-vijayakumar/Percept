"""How one judge request is sent: deadline, retry, failure mapping and usage.

Every adapter runs its requests through :func:`call`, so all providers share one behaviour.
An adapter supplies only ``send``::

    async def send(prompt: Prompt, images: tuple[bytes, ...], max_output_tokens: int,
                   request: JudgeRequest) -> RawReply

``images`` are JPEG bytes in request order (oldest first). ``send`` makes exactly one provider
request and returns the reply text with the billed token counts, or raises:

* ``RetryableError`` for a transient failure (rate limit, 5xx, dropped connection);
* ``RefusalError`` when the provider declined to answer (with any billed tokens);
* ``TimeoutError`` when its own transport timed out;
* anything else for a permanent failure (bad key, bad request).

``call`` never raises (cancellation excepted): every outcome comes back as observations.
"""

from __future__ import annotations

import asyncio
import random
import time
from dataclasses import replace
from typing import Awaitable, Callable

from percept.contracts import Failure, JudgeRequest, JudgeResult, Observation, PriceModel, RequestUsage, Usage
from percept.judges.base import (
    Prompt,
    RawReply,
    RefusalError,
    RetryableError,
    build_prompt,
    cost_of,
    failed_observations,
    parse_reply,
)
from percept.judges.encode import MAX_SIDE, to_jpeg

Send = Callable[[Prompt, "tuple[bytes, ...]", int, JudgeRequest], Awaitable[RawReply]]

TRUNCATED = "reply truncated at max_output_tokens"
BACKOFF_S = 0.25  # the retry waits BACKOFF_S to 2 * BACKOFF_S, unless the provider says otherwise


async def call(
    send: Send,
    request: JudgeRequest,
    *,
    judge_name: str,
    price: PriceModel | None = None,
    deadline_s: float | None = None,
    retries: int = 1,
    batch: bool = True,
    max_side: int = MAX_SIDE,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    clock: Callable[[], float] = time.monotonic,
    rng: Callable[[], float] = random.random,
) -> JudgeResult:
    """Judge ``request`` through ``send`` within ``deadline_s`` (default: the request's).

    With ``batch=False`` each item is sent as its own single-item request, one after another,
    under the same overall deadline; items not sent before it passes fail as ``timeout``. Each
    sub-request has ``request_id`` ``"<parent id>/<condition id>"`` (condition ids never contain
    ``/``), so its observations and usage stay attributable to the request that was sent.
    Returns one observation per item in request order and one ``RequestUsage`` per request sent.
    """
    deadline_s = request.deadline_s if deadline_s is None else deadline_s
    start = clock()
    # Encoding runs synchronously on the event loop; whether to move it to a thread is left to
    # the runtime that calls this.
    try:
        images = tuple(to_jpeg(image, max_side=max_side) for image in request.images)
    except Exception as exc:  # noqa: BLE001 - a bad frame is an outage of this request, not a crash
        return JudgeResult(failed_observations(request, "unavailable", judge_name, _describe(exc)), ())

    parts = [request] if batch or len(request.items) == 1 else [
        replace(request, request_id=f"{request.request_id}/{i.condition.id}", items=(i,)) for i in request.items]
    observations: list[Observation] = []
    usage: list[RequestUsage] = []
    for part in parts:
        remaining = deadline_s - (clock() - start)
        if remaining <= 0:
            observations += failed_observations(part, "timeout", judge_name, "deadline passed before sending")
            continue
        obs, ru = await _call_one(send, part, images, judge_name=judge_name, price=price, deadline_s=remaining,
                                  retries=retries, sleep=sleep, clock=clock, rng=rng)
        observations += obs
        usage.append(ru)
    return JudgeResult(tuple(observations), tuple(usage))


async def _call_one(send: Send, request: JudgeRequest, images: tuple[bytes, ...], *, judge_name: str,
                    price: PriceModel | None, deadline_s: float, retries: int,
                    sleep: Callable[[float], Awaitable[None]], clock: Callable[[], float],
                    rng: Callable[[], float]) -> tuple[tuple[Observation, ...], RequestUsage]:
    prompt = build_prompt(request)
    start = clock()
    attempts = tokens_in = tokens_out = 0
    out_of_time = False  # a retry was skipped because it could not finish before the deadline

    async def attempt() -> RawReply:
        nonlocal attempts, tokens_in, tokens_out, out_of_time
        while True:
            attempts += 1
            try:
                reply = await send(prompt, images, request.max_output_tokens, request)
            except RetryableError as exc:
                delay = exc.retry_after_s if exc.retry_after_s is not None else BACKOFF_S * (1 + rng())
                if attempts > retries:
                    raise
                if clock() - start + delay >= deadline_s:
                    out_of_time = True
                    raise
                await sleep(delay)
                continue
            except RefusalError as exc:  # a declined request is still billed
                tokens_in += max(0, int(exc.tokens_in))
                tokens_out += max(0, int(exc.tokens_out))
                raise
            tokens_in += max(0, int(reply.tokens_in))
            tokens_out += max(0, int(reply.tokens_out))
            return reply

    failure: Failure | None = None
    reason = ""
    try:
        reply = await asyncio.wait_for(attempt(), timeout=deadline_s)
        if clock() - start > deadline_s:
            failure, reason = "timeout", f"reply arrived after the {deadline_s:.1f} s deadline"
    except (asyncio.TimeoutError, TimeoutError):
        failure, reason = "timeout", f"no reply within {deadline_s:.1f} s"
    except RetryableError as exc:
        cause = "no time to retry before the deadline" if out_of_time else "retries exhausted"
        failure, reason = "unavailable", f"{cause}: {_describe(exc)}"
    except RefusalError as exc:
        failure, reason = "refused", _describe(exc)
    except Exception as exc:  # noqa: BLE001 - failures are reported, never raised
        failure, reason = "unavailable", _describe(exc)

    if failure is None and getattr(reply, "truncated", False):
        observations = failed_observations(request, "malformed", judge_name, TRUNCATED)
        request_failure: Failure | None = "malformed"
    elif failure is None:
        try:
            observations = parse_reply(reply.text, request, judge_name)
        except Exception as exc:  # noqa: BLE001 - "never raises" must not depend on the parser
            observations = failed_observations(request, "malformed", judge_name, _describe(exc))
        kinds = {o.failure for o in observations}
        request_failure = kinds.pop() if len(kinds) == 1 else None
    else:
        observations = failed_observations(request, failure, judge_name, reason)
        request_failure = failure

    cost = cost_of(price, tokens_in, tokens_out) if price is not None else 0.0
    used = Usage(attempts, tokens_in, tokens_out, cost, max(0.0, clock() - start))
    return observations, RequestUsage(request.request_id, judge_name, used, request_failure)


def _describe(exc: BaseException) -> str:
    text = f"{type(exc).__name__}: {exc}" if str(exc) else type(exc).__name__
    return text[:160]
