"""A judge that asks its member judges in order, escalating what the cheaper ones can't settle.

Pure orchestration over member judges; it talks to no provider itself.

Per condition, members are asked in order. A ``yes`` or ``no`` at or above that member's
threshold stands. An ``unknown``, a weaker answer or a failure escalates the condition to the
next member, together with every other unsettled condition of the request. When no member
settles a condition, the last real answer stands (a weak answer beats a later failure); with no
real answer at all, the last failure stands.

* Detection: ``Condition.judge`` names a member, and the condition is asked of that member
  only. An unknown name fails the condition as ``unavailable``.
* Verification requests (``purpose="verify"``) go straight to the last member, the strongest,
  for every condition; routes are not applied.
* All hops share the request's deadline. Every hop's ``RequestUsage`` is returned, so each
  attempt is visible to the budget. Hops reuse the request id; ``RequestUsage.judge`` and
  ``Observation.judge`` name the member that made the request. Failures the cascade makes up
  itself (an unknown route, a condition the deadline left unasked) carry the cascade's name.
"""

from __future__ import annotations

import time
from dataclasses import replace
from typing import Callable, Sequence

from percept.contracts import Failure, Judge, JudgeItem, JudgeRequest, JudgeResult, Observation, PriceModel, RequestUsage
from percept.judges.base import failed_observations, max_attempts


def _combined_price(judges: Sequence[Judge]) -> PriceModel | None:
    """A price whose worst case covers asking every member: rates summed, the largest image cost.

    None unless every member has a price.
    """
    prices = [j.price for j in judges]
    if any(p is None for p in prices):
        return None
    return PriceModel(sum(p.usd_per_mtok_in for p in prices), sum(p.usd_per_mtok_out for p in prices),  # type: ignore[union-attr]
                      max(p.tokens_per_image for p in prices))  # type: ignore[union-attr]


class Cascade:
    """See the module docstring. Build with :func:`cascade`."""

    def __init__(self, judges: Sequence[Judge], thresholds: Sequence[float | None] | None, name: str | None,
                 clock: Callable[[], float] = time.monotonic) -> None:
        if not judges:
            raise ValueError("cascade needs at least one judge")
        names = [j.name for j in judges]
        if len(set(names)) != len(names):
            raise ValueError(f"cascade members need distinct names, got {names}")
        thresholds = tuple(thresholds) if thresholds is not None else (None,) * len(judges)
        if len(thresholds) != len(judges):
            raise ValueError("give one threshold per judge (None: the condition's min_confidence)")
        if any(t is not None and not 0.0 <= t <= 1.0 for t in thresholds):
            raise ValueError("thresholds are in [0, 1]")
        self.members = tuple(judges)
        self.thresholds = thresholds
        self.name = name or "cascade(" + ">".join(names) + ")"
        self.can_batch = all(j.can_batch for j in judges)
        self.price = _combined_price(judges)
        self.max_attempts = sum(max_attempts(j) for j in judges)  # every member, every retry
        self._index = {n: k for k, n in enumerate(names)}
        self._clock = clock

    def _settles(self, k: int, item: JudgeItem, ob: Observation) -> bool:
        threshold = self.thresholds[k]
        threshold = item.condition.min_confidence if threshold is None else threshold
        return ob.failure is None and ob.answer != "unknown" and ob.confidence >= threshold

    async def _ask(self, member: Judge, request: JudgeRequest) -> JudgeResult:
        try:
            return await member.judge(request)
        except Exception as exc:  # noqa: BLE001 - a judge that breaks its contract is an outage
            return JudgeResult(failed_observations(request, "unavailable", member.name, f"judge raised {type(exc).__name__}"), ())

    async def judge(self, request: JudgeRequest) -> JudgeResult:
        start = self._clock()
        last = len(self.members) - 1
        standing: dict[str, Observation] = {}
        route: dict[str, tuple[int, bool]] = {}   # condition id -> (first member, pinned)
        for item in request.items:
            cid, pin = item.condition.id, item.condition.judge
            if request.purpose == "verify":
                route[cid] = (last, False)
            elif pin is None:
                route[cid] = (0, False)
            elif pin in self._index:
                route[cid] = (self._index[pin], True)
            else:
                standing[cid] = self._failed(request, item, "unavailable", f"no judge named {pin!r} in this cascade")
        open_ids = set(route)
        usage: list[RequestUsage] = []

        for k, member in enumerate(self.members):
            items = tuple(i for i in request.items if i.condition.id in open_ids
                          and (route[i.condition.id][0] == k if route[i.condition.id][1]
                               else route[i.condition.id][0] <= k))
            if not items:
                continue
            remaining = request.deadline_s - (self._clock() - start)
            if remaining <= 0:
                break
            result = await self._ask(member, replace(request, items=items, deadline_s=remaining))
            usage += result.usage
            got = {ob.condition_id: ob for ob in result.observations}
            for item in items:
                cid = item.condition.id
                ob = got.get(cid) or self._failed(request, item, "missing", f"{member.name} returned no observation",
                                                  member.name)
                prior = standing.get(cid)
                if ob.failure is None or prior is None or prior.failure is not None:
                    standing[cid] = ob
                if route[cid][1] or k == last or self._settles(k, item, ob):
                    open_ids.discard(cid)

        observations = tuple(standing.get(i.condition.id) or self._failed(request, i, "timeout", "deadline passed "
                                                                          "before this condition was asked")
                             for i in request.items)
        return JudgeResult(observations, tuple(usage))

    def _failed(self, request: JudgeRequest, item: JudgeItem, failure: Failure, reason: str,
                judge_name: str | None = None) -> Observation:
        return failed_observations(replace(request, items=(item,)), failure, judge_name or self.name, reason)[0]


def cascade(*judges: Judge, thresholds: Sequence[float | None] | None = None, name: str | None = None) -> Cascade:
    """A judge that asks ``judges`` in order, cheapest first, strongest last.

    ``thresholds`` gives, per judge, the confidence at or above which its yes/no stands; None
    (the default, or an entry) uses each condition's ``min_confidence``. The last judge's answer
    always stands. ``name`` defaults to ``"cascade(<member>><member>...)"``. ``price`` is set only
    when every member has one, and then covers asking every member; ``max_attempts`` is the sum
    of the members' (see ``base.max_attempts``).
    """
    return Cascade(judges, thresholds, name)
