"""Evaluation driver: run configurations you declare over your labelled clips, and score them.

Experimental: names and outputs may change before they are stable.

Every function takes a :class:`~percept.plans.Compiled` plan with one source, the
:class:`~percept.plans.labels.Labels`, the clip files to run (for example
``manifest.clips["dev"]``) and ``video_root``, the directory that holds them. Each clip is
opened with :func:`percept.open_source` at the plan source's ``max_side``. ``fps`` is the
processing rate (None: the Monitor's default). ``label_of`` maps a condition id to its label in
the labels (default: the id itself); a condition it leaves out is not scored.

* :func:`attention_only` runs only the attention step (:mod:`percept.attention`, the one the
  Monitor runs): no judge calls, no cost. Its :class:`~percept.plans.score.Fire` records are the
  looks made due, so :func:`~percept.plans.score.evaluate` reports the candidate stage of the
  funnel: how many occurrences any look would have seen.
* :func:`run` runs the full runtime, one :class:`~percept.Monitor` per clip, with the plan's judge
  wrapped in :class:`CachingJudge` (a hard dollar cap and a reply cache).
* :func:`sweep` runs each configuration you declare on the dev or validation split and returns
  their scores side by side, in your order. It does not rank or choose between them.
* :func:`register` freezes a plan before the test split is run, and :func:`score_split` scores
  the test split only for a registered plan whose hashes still match.

Shell module: it reads video and runs the Monitor.
"""

from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import json
import math
import os
import tempfile
import time
import weakref
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Optional, Sequence, Union

import numpy as np

from percept import attention
from percept.contracts import JudgeRequest, JudgeResult, Observation
from percept.delivery import MemoryStore
from percept.judges import base
from percept.monitor import DEFAULT_FPS, Monitor
from percept.plans import EVENT_CLIP_S, Compiled
from percept.plans.labels import SPLITS, Labels, SplitManifest, _basename, _sha
from percept.plans.score import MIN_CONFIDENCE, ClipRun, Drop, Fire, Look, Prediction, Score, evaluate
from percept.sources import open_source
from percept.trace import read as read_trace
from percept.triggers import QualityGate, SceneMonitor, every, on_change

REGISTRATION_FORMAT = "percept.registration/v1alpha1"


def _setup(compiled: Compiled, labels: Labels, clips: Sequence[str], fps: Optional[float],
           label_of: Optional[Mapping[str, str]]) -> tuple[float, int, dict[str, str]]:
    """The processing rate, the source's max_side and condition id -> label; checks the clips."""
    if len(compiled.sources) != 1:
        raise ValueError(f"evaluation runs a plan with one source; this plan has {len(compiled.sources)}")
    if fps is not None and not (math.isfinite(fps) and fps > 0):
        raise ValueError(f"fps must be a positive number or None, got {fps!r}")
    for file in clips:
        labels.clip(file)       # KeyError: a clip that is not in the labels
    ids = [c.id for c in compiled.conditions]
    names = {i: i for i in ids} if label_of is None else {i: label_of[i] for i in ids if i in label_of}
    return fps or DEFAULT_FPS, compiled.sources[0]["max_side"], names


def _cooldown(compiled: Compiled) -> float:
    """The shortest rearm of the plan's conditions: the recall ceiling is then an upper bound."""
    return min((c.rearm_s for c in compiled.conditions), default=0.0)


# --------------------------------------------------------------------------- attention only


def attention_only(compiled: Compiled, labels: Labels, clips: Sequence[str], *, video_root: Union[str, os.PathLike],
                   fps: Optional[float] = None, label_of: Optional[Mapping[str, str]] = None) -> list[ClipRun]:
    """The trigger stage alone: one :class:`ClipRun` per clip whose ``fires`` are the looks the
    attention step makes due (trigger looks and the settled look after a scene reset), at the clip
    time of their frame. No judge is called; ``cost_usd`` is 0 and ``compute_s`` is process CPU
    seconds. Score it with :func:`~percept.plans.score.evaluate`: ``funnel["candidate"]`` counts
    the occurrences a look would have seen."""
    rate, max_side, names = _setup(compiled, labels, clips, fps, label_of)
    root = os.fspath(video_root)

    async def attend(file: str) -> ClipRun:
        conds = {c.id: c for c in compiled.conditions}
        # The Monitor's defaults for a condition without a trigger, built the way it builds them.
        groups, tlabels = attention.build(conds.values(), default_change=on_change(), default_heartbeat=every(300))
        quality, scene, state = QualityGate(), SceneMonitor(), attention.State()
        fires: list[Fire] = []
        anchor, t, cpu = None, 0.0, time.process_time()
        async with open_source(os.path.join(root, file), fps=rate, max_side=max_side) as source:
            async for frame in source.frames():
                anchor = frame.t_wall if anchor is None else anchor
                t = frame.t_wall - anchor
                s, state = attention.step(frame, state, groups=groups, conditions=conds, labels=tlabels,
                                          quality=quality, scene=scene, period=1.0 / rate)
                due = [lk.condition_id for lk in s.looks] + (list(conds) if s.settled else [])
                fires += [Fire(names[cid], t) for cid in due if cid in names]
        return ClipRun(_basename(file), t + 1.0 / rate, fires=tuple(fires), compute_s=time.process_time() - cpu,
                       cooldown_s=_cooldown(compiled))

    async def main() -> list[ClipRun]:
        return [await attend(f) for f in clips]

    return asyncio.run(main())


# --------------------------------------------------------------------------- the judge


class CachingJudge:
    """Wraps a judge with a hard dollar cap, a spend log and a reply cache. Implements the Judge
    protocol.

    Cap: each request reserves the judge's worst case (every attempt, see
    :func:`percept.judges.base.worst_case_cost_for`) before it is sent. A request whose reservation,
    added to what was spent and what requests in flight hold, would pass ``cap_usd`` is not sent:
    it fails with ``budget`` (never an answer) and sets ``tripped``. ``cap_usd=None``: no cap.

    Spend log: with ``spend_log`` (a path), one JSON line per request sent, so spend survives a
    killed run.

    Cache: replies are kept by the images (sha256 of their pixels), the conditions (id, text,
    version, kind, subject, counter-example notes), the purpose and the judge's name; an identical
    request is answered from the cache, sends nothing and costs $0 (and uses no request attempts).
    Failed replies are not kept. ``cache``: True for a new cache, a mapping to share one across
    runs, None for none.
    """

    def __init__(self, inner: Any, *, cap_usd: Optional[float], spend_log: Union[None, str, os.PathLike] = None,
                 cache: Union[bool, None, MutableMapping[str, tuple[Observation, ...]]] = True) -> None:
        if cap_usd is not None and (not math.isfinite(cap_usd) or cap_usd < 0):
            raise ValueError(f"cap_usd must be a finite number >= 0 or None, got {cap_usd!r}")
        if cap_usd is not None and getattr(inner, "price", None) is None:
            raise ValueError("a spend cap needs a judge with a PriceModel")
        self.inner, self.cap, self.spend_log = inner, cap_usd, spend_log
        self.name, self.can_batch, self.price = inner.name, inner.can_batch, inner.price
        self.max_attempts = base.max_attempts(inner)
        self.cache: Optional[MutableMapping] = {} if cache is True else None if cache in (None, False) else cache
        self.spent = self.held = 0.0
        self.requests = self.refused = self.hits = 0
        self.tripped = asyncio.Event()

    def max_attempts_for(self, request: JudgeRequest) -> int:
        return base.max_attempts_for(self.inner, request)

    def worst_case_cost_for(self, request: JudgeRequest) -> Optional[float]:
        return base.worst_case_cost_for(self.inner, request)

    def reservation(self, request: JudgeRequest) -> float:
        """The dollars ``request`` reserves: the worst case, per item for a judge that can't batch."""
        calls = 1 if self.can_batch else len(request.items)
        return (base.worst_case_cost_for(self.inner, request) or 0.0) * calls

    def key(self, request: JudgeRequest) -> str:
        h = hashlib.sha256()
        for image in request.images:
            a = np.ascontiguousarray(image)
            h.update(f"{a.shape}{a.dtype.str}".encode())
            h.update(a.tobytes())
        items = [[i.condition.id, i.condition.text, i.condition.version, i.condition.kind, i.subject,
                  [c.note for c in i.counter_examples]] for i in request.items]
        h.update(json.dumps([items, request.purpose, self.name]).encode())
        return h.hexdigest()

    async def judge(self, request: JudgeRequest) -> JudgeResult:
        key = None if self.cache is None else self.key(request)
        cached = None if key is None else self.cache.get(key)  # type: ignore[union-attr]
        if cached is not None:
            self.hits += 1
            gen = {i.condition.id: i.occurrence_gen for i in request.items}
            return JudgeResult(tuple(dataclasses.replace(
                o, request_id=request.request_id, stream_id=request.stream_id, stream_epoch=request.stream_epoch,
                evidence_t=request.evidence_t, occurrence_gen=gen[o.condition_id]) for o in cached), ())
        reserve = self.reservation(request)
        if self.cap is not None and self.spent + self.held + reserve > self.cap:
            self.refused += 1
            self.tripped.set()
            return JudgeResult(base.failed_observations(request, "budget", self.name, "spend cap reached"), ())
        self.held += reserve
        result = None
        try:
            result = await self.inner.judge(request)
            if key is not None and not any(o.failed for o in result.observations):
                self.cache[key] = result.observations  # type: ignore[index]
            return result
        finally:
            self.held -= reserve
            self.requests += 1
            cost = reserve if result is None else sum(u.usage.cost_usd for u in result.usage)
            self.spent += cost
            if self.spend_log is not None:
                with open(self.spend_log, "a") as f:
                    f.write(json.dumps({"t": round(time.time(), 3), "request_id": request.request_id,
                                        "cost_usd": cost, "settled": result is not None}) + "\n")
