"""How one judge request is sent: deadline, retry, failure mapping and usage.

Every adapter runs its requests through :func:`call`, so all providers share one behaviour.
An adapter supplies only ``send``::

    async def send(prompt: Prompt, images: tuple[bytes, ...], max_output_tokens: int,
                   request: JudgeRequest) -> RawReply

``images`` are JPEG bytes in request order (oldest first). ``send`` makes exactly one provider
request and returns the reply text with the billed token counts, or raises (it must tolerate
cancellation: an attempt that hangs is cancelled so its retry can run):

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
    worst_case_tokens,
)
from percept.judges.encode import MAX_SIDE, to_jpeg

Send = Callable[[Prompt, "tuple[bytes, ...]", int, JudgeRequest], Awaitable[RawReply]]

TRUNCATED = "reply truncated at max_output_tokens"
BACKOFF_S = 0.25  # the retry waits BACKOFF_S to 2 * BACKOFF_S, unless the provider says otherwise
ATTEMPT_SHARE = 0.6  # with a retry left, an attempt may use this share of the time remaining: one
                     # that hangs is abandoned in time for the retry to run (on an 8 s deadline the
                     # cut is 4.8 s and the retry keeps 3.2 s)


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
    # Encoding runs in a worker thread, so large frames never stall the event loop.
    try:
        images = await asyncio.to_thread(lambda: tuple(to_jpeg(image, max_side=max_side) for image in request.images))
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
    unreplied = 0  # attempts sent that never replied: possibly billed, tokens unknown
    in_flight = False  # an attempt is waiting for its reply (not a backoff sleep)

    async def attempt() -> RawReply:
        nonlocal attempts, tokens_in, tokens_out, out_of_time, unreplied, in_flight
        while True:
            attempts += 1
            sent = asyncio.ensure_future(send(prompt, images, request.max_output_tokens, request))
            in_flight = True
            try:
                if attempts <= retries:     # a retry is left: don't let this attempt use all the time
                    limit = max(0.0, deadline_s - (clock() - start)) * ATTEMPT_SHARE
                    await asyncio.wait({sent}, timeout=limit)
                    if not sent.done():
                        unreplied += 1
                        continue            # it hung: abandoned (finally cancels it), the retry runs
                reply = await sent
            except RetryableError as exc:
                in_flight = False       # it answered with an error: a cut during the backoff charges nothing
                delay = exc.retry_after_s if exc.retry_after_s is not None else BACKOFF_S * (1 + rng())
                if attempts > retries:
                    raise
                if clock() - start + delay >= deadline_s:
                    out_of_time = True
                    raise
                await sleep(delay)
                continue
            except RefusalError as exc:  # a declined request is still billed
                in_flight = False
                tokens_in += max(0, int(exc.tokens_in))
                tokens_out += max(0, int(exc.tokens_out))
                raise
            except (asyncio.TimeoutError, TimeoutError):
                unreplied += 1          # the transport gave up waiting: possibly billed, no tokens
                raise
            finally:
                if sent.done() and not sent.cancelled():
                    in_flight = False   # it answered or failed; a cut or cancelled one stays counted
                if not sent.done():     # abandoned, or the whole request was cancelled
                    sent.cancel()
                    sent.add_done_callback(_settled)   # whatever it ends with is not an outcome
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
        unreplied += in_flight      # cut mid-attempt, not during a backoff sleep
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

    cost = 0.0
    if price is not None:   # an attempt with no reply may still be billed: charged at its worst case
        cost = cost_of(price, tokens_in, tokens_out) + unreplied * cost_of(price, *worst_case_tokens(request, price))
    used = Usage(attempts, tokens_in, tokens_out, cost, max(0.0, clock() - start))
    return observations, RequestUsage(request.request_id, judge_name, used, request_failure,
                                      tuple(item.condition.id for item in request.items))


def _settled(task: asyncio.Future) -> None:
    """Retrieve an abandoned attempt's outcome, so a late failure is never logged as unhandled."""
    if not task.cancelled():
        task.exception()


def _describe(exc: BaseException) -> str:
    text = f"{type(exc).__name__}: {exc}" if str(exc) else type(exc).__name__
    return text[:160]
