"""Judges for the harness: the offline oracle, the hard spend cap, and building real judges.

``OracleJudge`` answers from ground truth with configurable, deterministic noise and latency.
It sees what the images show: each image is traced back to its sampled frame's case time, and a
condition is true when a frame falls just after an onset. Noise is keyed on the evidence, so a
verification of the same frames answers the same way.
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

import numpy as np

from percept import JudgeRequest, JudgeResult, PriceModel
from percept.judges import FakeJudge, gemini
from percept.plans.run import CachingJudge

ONSET_BEFORE_S, ONSET_AFTER_S = 0.5, 2.0    # images showing [onset - 0.5, onset + 2] show the beginning


def parse_price(text: str) -> PriceModel:
    """``"usd_per_mtok_in,usd_per_mtok_out,tokens_per_image"``, e.g. ``"0.10,0.40,258"``."""
    a, b, c = text.split(",")
    return PriceModel(float(a), float(b), int(c))


def frame_key(image) -> bytes:
    return hashlib.blake2b(np.ascontiguousarray(image).tobytes(), digest_size=12).digest()


class Timelines:
    """Maps a stream's evidence times to its case timeline (``t = (wall - anchor) * speed``), and
    remembers each fed frame's case time by content, so the oracle can tell which sampled frames
    a request actually shows."""

    def __init__(self) -> None:
        self.anchors: dict[str, tuple[float, float]] = {}
        self.frames: dict[str, dict[bytes, list[float]]] = {}

    def set(self, stream_id: str, anchor: float, speed: float = 1.0) -> None:
        self.anchors[stream_id] = (anchor, speed)
        self.frames[stream_id] = {}

    def media(self, stream_id: str, wall: float) -> float:
        anchor, speed = self.anchors.get(stream_id, (0.0, 1.0))
        return (wall - anchor) * speed

    def note(self, stream_id: str, t: float, image) -> None:
        self.frames.setdefault(stream_id, {}).setdefault(frame_key(image), []).append(t)

    def frame_times(self, request: JudgeRequest) -> tuple[float, ...]:
        """The case time of each image in ``request``, oldest first. A content match can repeat
        (a static scene); the earliest candidate inside the evidence interval, not before the
        previous image, is taken. Unknown images (not from the feed) fall back to the interval."""
        sid = request.stream_id
        a, b = (self.media(sid, w) for w in request.evidence_t)
        known, out = self.frames.get(sid, {}), []
        for image in request.images:
            floor = out[-1] if out else a - 0.01
            cands = [t for t in known.get(frame_key(image), ()) if floor - 1e-6 <= t <= b + 0.01]
            out.append(min(cands) if cands else (b if out else a))
        return tuple(out)


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

    def truth(self, case_id: str, label: str, times) -> bool:
        """Yes when a sampled frame falls in ``[onset - 0.5, onset + 2]``: the images show the
        beginning. An onset that happens between two frames is not seen."""
        return any(o.label == label and o.start - ONSET_BEFORE_S <= t <= o.start + ONSET_AFTER_S
                   for o in self.cases[case_id].occurrences for t in times)

    def _answer(self, request: JudgeRequest) -> dict:
        case = self.cases[request.stream_id]
        times = self.timelines.frame_times(request)
        evidence = tuple(round(t, 2) for t in times)      # the noise depends on the evidence only:
        labels = {c.id: c.label for c in case.conditions}   # a second look at it answers the same
        out = {}
        for item in request.items:
            cid = item.condition.id
            truth = self.truth(case.case_id, labels[cid], times)
            u = self._draw(case.case_id, cid, evidence)
            if u < self.unknown_rate:
                out[cid] = ("unknown", 0.5, "oracle: can't tell")
            else:
                flip = self._draw("flip", case.case_id, cid, evidence)
                said = (not truth) if flip < (self.fn_rate if truth else self.fp_rate) else truth
                out[cid] = ("yes", 0.9, "oracle: begins") if said else ("no", 0.9, "oracle: not seen")
        return out

    async def judge(self, request: JudgeRequest) -> JudgeResult:
        if self.realtime:
            await asyncio.sleep(self.latency_s)
        return await self._fake.judge(request)


class CostCapReached(Exception):
    pass


class CostGuard(CachingJudge):
    """A judge wrapper holding a run-wide dollar cap (see the module docstring): Percept's
    ``CachingJudge`` with its reply cache off, so every request is sent."""

    def __init__(self, inner, cap_usd: Optional[float], spend_log=None) -> None:
        super().__init__(inner, cap_usd=cap_usd, spend_log=spend_log, cache=None)


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
