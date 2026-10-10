"""Judges for the harness: the offline oracle, the hard spend cap, and building real judges.

``OracleJudge`` answers from ground truth with configurable, deterministic noise and latency.
It runs through Percept's own ``FakeJudge``, so its replies take the real prompt, parsing and
usage path, and its token counts follow the ``PriceModel`` it is given. Its numbers measure
the harness, never a model.

``CostGuard`` wraps any judge and enforces ``--max-cost-usd`` across every arm of a run: each
request reserves the judge's worst case (``percept.judges.base.worst_case_cost``, every attempt)
before it is sent, and a request that would not fit is refused (failure ``budget``) and trips
the guard, which the runner treats as a clean abort.
"""

from __future__ import annotations

import asyncio
import hashlib
from typing import Optional

from percept import JudgeRequest, JudgeResult, PriceModel
from percept.judges import FakeJudge, base, gemini

ONSET_BEFORE_S, ONSET_AFTER_S = 0.5, 2.0    # images showing [onset - 0.5, onset + 2] show the beginning


def parse_price(text: str) -> PriceModel:
    """``"usd_per_mtok_in,usd_per_mtok_out,tokens_per_image"``, e.g. ``"0.10,0.40,258"``."""
    a, b, c = text.split(",")
    return PriceModel(float(a), float(b), int(c))


class Timelines:
    """Maps a stream's evidence times to its case timeline: ``t = (wall - anchor) * speed``."""

    def __init__(self) -> None:
        self.anchors: dict[str, tuple[float, float]] = {}

    def set(self, stream_id: str, anchor: float, speed: float = 1.0) -> None:
        self.anchors[stream_id] = (anchor, speed)

    def media(self, stream_id: str, wall: float) -> float:
        anchor, speed = self.anchors.get(stream_id, (0.0, 1.0))
        return (wall - anchor) * speed


class OracleJudge:
    """Answers "does this condition begin in these images" from ground truth, with noise."""

    def __init__(self, cases, timelines: Timelines, *, fn_rate: float = 0.1, fp_rate: float = 0.02,
                 unknown_rate: float = 0.0, latency_s: float = 1.0, seed: int = 0,
                 price: Optional[PriceModel] = PriceModel(0.10, 0.40, 258)) -> None:
        self.cases = {c.case_id: c for c in cases}
        self.timelines, self.seed, self.latency_s = timelines, seed, latency_s
        self.fn_rate, self.fp_rate, self.unknown_rate = fn_rate, fp_rate, unknown_rate
        self.realtime = False           # sleep the latency for real (the Percept arm runs in real time)
        self._fake = FakeJudge(self._answer, name="oracle", price=price, latency_s=latency_s)
        self.name, self.can_batch, self.price = "oracle", True, price
        self.max_attempts = self._fake.max_attempts

    def _draw(self, *key) -> float:
        digest = hashlib.sha256(repr((self.seed,) + key).encode()).digest()
        return int.from_bytes(digest[:8], "big") / 2 ** 64

    def truth(self, case_id: str, label: str, a: float, b: float) -> bool:
        return any(o.label == label and a <= o.start + ONSET_AFTER_S and b >= o.start - ONSET_BEFORE_S
                   for o in self.cases[case_id].occurrences)

    def _answer(self, request: JudgeRequest) -> dict:
        case = self.cases[request.stream_id]
        a = self.timelines.media(request.stream_id, request.evidence_t[0])
        b = self.timelines.media(request.stream_id, request.evidence_t[1])
        labels = {c.id: c.label for c in case.conditions}
        out = {}
        for item in request.items:
            cid = item.condition.id
            truth = self.truth(case.case_id, labels[cid], a, b)
            u = self._draw(case.case_id, cid, round(a, 1), round(b, 1), request.purpose)
            if u < self.unknown_rate:
                out[cid] = ("unknown", 0.5, "oracle: can't tell")
            else:
                flip = self._draw("flip", case.case_id, cid, round(a, 1), round(b, 1), request.purpose)
                said = (not truth) if flip < (self.fn_rate if truth else self.fp_rate) else truth
                out[cid] = ("yes", 0.9, "oracle: begins") if said else ("no", 0.9, "oracle: not seen")
        return out

    async def judge(self, request: JudgeRequest) -> JudgeResult:
        if self.realtime:
            await asyncio.sleep(self.latency_s)
        return await self._fake.judge(request)


class CostCapReached(Exception):
    pass


class CostGuard:
    """A judge wrapper holding a run-wide dollar cap (see the module docstring)."""

    def __init__(self, inner, cap_usd: Optional[float]) -> None:
        if cap_usd is not None and getattr(inner, "price", None) is None:
            raise ValueError("a spend cap needs a judge with a PriceModel (pass --price)")
        self.inner, self.cap = inner, cap_usd
        self.name, self.can_batch, self.price = inner.name, inner.can_batch, inner.price
        self.max_attempts = base.max_attempts(inner)
        self.spent = self.held = 0.0
        self.requests = self.refused = 0
        self.tripped = asyncio.Event()

    def reservation(self, request: JudgeRequest) -> float:
        if self.price is None:
            return 0.0
        calls = 1 if self.can_batch else len(request.items)
        return base.worst_case_cost(request, self.price, retries=self.max_attempts - 1) * calls

    async def judge(self, request: JudgeRequest) -> JudgeResult:
        reserve = self.reservation(request)
        if self.cap is not None and self.spent + self.held + reserve > self.cap:
            self.refused += 1
            self.tripped.set()
            return JudgeResult(base.failed_observations(request, "budget", self.name, "harness spend cap reached"), ())
        self.held += reserve
        result = None
        try:
            result = await self.inner.judge(request)
            return result
        finally:
            self.held -= reserve
            self.requests += 1
            self.spent += reserve if result is None else sum(u.usage.cost_usd for u in result.usage)


def build(spec: str, *, cases, timelines: Timelines, price: Optional[PriceModel], oracle: dict) -> object:
    """``fake`` (the oracle) or ``gemini:<model>``. Real judges need ``price``."""
    if spec == "fake":
        return OracleJudge(cases, timelines, price=price or PriceModel(0.10, 0.40, 258), **oracle)
    vendor, _, model = spec.partition(":")
    if vendor == "gemini" and model:
        if price is None:
            raise SystemExit("a real judge needs --price in,out,tokens_per_image (the cap reserves with it)")
        return gemini(model, price=price)
    raise SystemExit(f"unknown judge {spec!r}: use fake or gemini:<model>")
