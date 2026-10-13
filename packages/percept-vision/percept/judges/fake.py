"""A scripted judge for tests and offline examples.

``FakeJudge`` runs the same ``policy.call`` as a real adapter, so its replies go through the
real prompt, parsing, retry, deadline and failure paths. Only the provider is scripted.

The script is called once per provider request with the request being sent, and returns:

* a ``dict`` mapping condition id to ``(answer, confidence, reason)``: a well-formed reply
  with those results (ids left out of the dict are missing from the reply);
* a ``str``: the raw reply text, for malformed or oddly wrapped replies;
* an exception instance: raised by the provider, e.g. ``RetryableError()``,
  ``RefusalError()`` or ``TimeoutError()``.

Time is virtual by default: ``latency_s`` advances a ``VirtualClock`` instead of sleeping, so
a reply slower than the deadline fails as a timeout without the test waiting for it.
"""

from __future__ import annotations

import asyncio
import json
import math
from typing import Any, Callable, Mapping, Union

from percept.contracts import JudgeRequest, JudgeResult, PriceModel
from percept.judges import policy
from percept.judges.base import Prompt, RawReply, attempts_for

ScriptReply = Union[str, BaseException, Mapping[str, tuple]]
Script = Callable[[JudgeRequest], ScriptReply]


class VirtualClock:
    """A clock that moves only when something sleeps on it."""

    def __init__(self, t: float = 0.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t

    async def sleep(self, seconds: float) -> None:
        self.t += max(0.0, seconds)
        await asyncio.sleep(0)


def reply_json(results: Mapping[str, tuple], request: JudgeRequest | None = None) -> str:
    """A well-formed reply text; with ``request``, limited to its ids."""
    ids = None if request is None else {item.condition.id for item in request.items}
    return json.dumps({"results": [
        {"id": cid, "answer": a, "confidence": c, "reason": r}
        for cid, (a, c, r) in results.items() if ids is None or cid in ids
    ]})


class FakeJudge:
    """A deterministic ``Judge`` driven by ``script``, for tests and offline demos.

    ``script`` is either a callable, called once per provider request (retries included) with
    the :class:`~percept.contracts.JudgeRequest` being sent, or a fixed reply returned for every
    request. A reply is one of:

    * ``{condition_id: (answer, confidence, reason)}``, ``answer`` being ``"yes"``, ``"no"`` or
      ``"unknown"``: a well-formed reply. Ids the request did not ask about are ignored; ids it
      asked about and the dict leaves out fail as missing;
    * a ``str``: the raw reply text (to test malformed replies);
    * an exception instance, raised by the provider: ``base.RetryableError()`` is retried,
      ``base.RefusalError()`` and ``TimeoutError()`` fail the request.

    In the request, ``images`` are the BGR ``uint8`` arrays a real judge would be sent (cropped
    to the conditions' zones; a state's newest frame, an event's short clip, oldest first),
    ``items[i].condition`` the conditions asked about and ``purpose`` ``"detect"`` or
    ``"verify"``. A script that inspects the images behaves like a model watching the scene, so
    states can clear.

    ``sent`` records every provider request. A record holds the request, and a request's images
    are views of the source frames (a zone crop is a slice, not a copy), so each record keeps
    whole frames alive. For a long run, such as an evaluation over hours of footage, pass
    ``record_requests=False``: nothing is kept and ``sent`` is None, as on the vendor judges.
    """

    def __init__(self, script: Script | ScriptReply, *, name: str = "fake", can_batch: bool = True,
                 price: PriceModel | None = None, latency_s: float = 0.0, retries: int = 1,
                 clock: VirtualClock | None = None, record_requests: bool = True) -> None:
        self.name = name
        self.can_batch = can_batch
        self.price = price
        self.latency_s = latency_s
        self.retries = retries
        self.max_attempts = attempts_for(retries)
        self.clock = clock or VirtualClock()
        # Every provider request, retries included: (prompt, jpeg images, max_output_tokens, request).
        self.sent: list[tuple[Prompt, tuple[bytes, ...], int, JudgeRequest]] | None = (
            [] if record_requests else None)
        self._script = script if callable(script) else (lambda _request: script)

    async def judge(self, request: JudgeRequest) -> JudgeResult:
        return await policy.call(self._send, request, judge_name=self.name, price=self.price,
                                 retries=self.retries, batch=self.can_batch,
                                 sleep=self.clock.sleep, clock=self.clock, rng=lambda: 0.5)

    async def _send(self, prompt: Prompt, images: tuple[bytes, ...], max_output_tokens: int,
                    request: JudgeRequest) -> RawReply:
        if self.sent is not None:
            self.sent.append((prompt, images, max_output_tokens, request))
        reply: Any = self._script(request)
        await self.clock.sleep(self.latency_s)
        if isinstance(reply, BaseException):
            raise reply
        text = reply if isinstance(reply, str) else reply_json(reply, request)
        image_tokens = len(images) * (self.price.tokens_per_image if self.price else 0)
        tokens_in = math.ceil((len(prompt.system) + len(prompt.user)) / 4) + image_tokens
        return RawReply(text, tokens_in, min(max_output_tokens, math.ceil(len(text) / 4)))
