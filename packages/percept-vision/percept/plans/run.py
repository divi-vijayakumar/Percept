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
import warnings
import weakref
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Optional, Sequence, Union

import numpy as np

from percept import attention
from percept.contracts import JudgeRequest, JudgeResult
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


def _rebase(old: str, new: str) -> str:
    """``old`` request id moved onto ``new``, keeping a per-condition suffix (``id/condition``)."""
    return new + old[len(old.split("/")[0]):]


def _identity(judge: Any) -> Any:
    """What makes a judge answer as it does: its ``cache_key`` if it has one, else its name, model,
    price and attempts, and for a cascade its members and thresholds."""
    if getattr(judge, "cache_key", None) is not None:
        return judge.cache_key
    model = getattr(judge, "model", None)
    members = getattr(judge, "members", None)
    return [judge.name, model if isinstance(model, str) else None, repr(judge.price), base.max_attempts(judge),
            [_identity(m) for m in members] if members else None, list(getattr(judge, "thresholds", ()) or ())]


class CachingJudge:
    """Wraps a judge with a hard dollar cap, a spend log and a reply cache. Implements the Judge
    protocol.

    Cap: each request reserves the judge's worst case (every attempt, see
    :func:`percept.judges.base.worst_case_cost_for`) before it is sent. A request whose reservation,
    added to what was spent and what requests in flight hold, would pass ``cap_usd`` is not sent:
    it fails with ``budget`` (never an answer) and sets ``tripped``. ``cap_usd=None``: no cap.

    Spend log: with ``spend_log`` (a path), one JSON line per request sent, so spend survives a
    killed run.

    Cache: replies are kept with their usage, by the images (sha256 of their pixels), the prompt
    as rendered (every condition's wording, ``not_when`` and counter-examples, the verify note; the
    evidence's wall-clock times made relative), each item's route, ``max_output_tokens`` and the
    judge's identity (its ``cache_key``, else name, model, price, attempts and cascade members). An identical
    request is answered from the cache and sends nothing, but is replayed as it happened: the
    original usage (attempts, tokens, cost) is returned, so the Monitor's budget and cost settle as
    they did live, and with ``replay_latency`` the reply waits the original latency divided by
    ``speed``. ``spent`` is what was actually paid; ``hits`` and ``saved_usd`` count the replays.
    Failed replies are not kept. ``cache``: True for a new cache, a mapping to share one across
    runs, None for none.
    """

    def __init__(self, inner: Any, *, cap_usd: Optional[float], spend_log: Union[None, str, os.PathLike] = None,
                 cache: Union[bool, None, MutableMapping[str, tuple]] = True, replay_latency: bool = True,
                 speed: Optional[float] = 1.0) -> None:
        self.replay_latency, self.speed, self.saved_usd = replay_latency, speed, 0.0
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
        t0, t1 = request.evidence_t     # the prompt as sent, its wall-clock times made relative
        prompt = base.build_prompt(dataclasses.replace(request, evidence_t=(0.0, round(t1 - t0, 1))))
        h.update(json.dumps([prompt.system, prompt.user, [i.condition.judge for i in request.items],
                             request.purpose, request.max_output_tokens, _identity(self.inner)], default=repr).encode())
        return h.hexdigest()

    async def judge(self, request: JudgeRequest) -> JudgeResult:
        key = None if self.cache is None else self.key(request)
        cached = None if key is None else self.cache.get(key)  # type: ignore[union-attr]
        if cached is not None:      # replayed as it happened live: same usage, same latency
            observations, usage = cached
            self.hits += 1
            self.saved_usd += sum(u.usage.cost_usd for u in usage)
            latency = sum(u.usage.latency_s for u in usage)
            if self.replay_latency and self.speed and latency:
                await asyncio.sleep(latency / self.speed)
            gen = {i.condition.id: i.occurrence_gen for i in request.items}
            rid = request.request_id
            return JudgeResult(
                tuple(dataclasses.replace(o, request_id=_rebase(o.request_id, rid), stream_id=request.stream_id,
                                          stream_epoch=request.stream_epoch, evidence_t=request.evidence_t,
                                          occurrence_gen=gen[o.condition_id]) for o in observations),
                tuple(dataclasses.replace(u, request_id=_rebase(u.request_id, rid)) for u in usage))
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
                self.cache[key] = (result.observations, result.usage)  # type: ignore[index]
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


# --------------------------------------------------------------------------- the full runtime


class _Shown:
    """Passes requests to ``judge`` and records the clip times of the frames each one showed,
    tracing every image to the frame it is (or is a view of) by identity."""

    def __init__(self, judge: CachingJudge) -> None:
        self._judge, self.frames, self._times = judge, {}, {}
        self.name, self.can_batch, self.price, self.max_attempts = judge.name, judge.can_batch, judge.price, judge.max_attempts
        self.max_attempts_for, self.worst_case_cost_for = judge.max_attempts_for, judge.worst_case_cost_for

    def note(self, image: np.ndarray, t: float) -> None:
        if len(self._times) > 512:      # forget frames nothing holds any more
            self._times = {k: v for k, v in self._times.items() if v[0]() is not None}
        self._times[id(image)] = (weakref.ref(image), t)

    def _t(self, image: np.ndarray) -> Optional[float]:
        while isinstance(image.base, np.ndarray):
            image = image.base
        entry = self._times.get(id(image))
        return entry[1] if entry is not None and entry[0]() is image else None

    async def judge(self, request: JudgeRequest) -> JudgeResult:
        self.frames[request.request_id] = tuple(t for t in map(self._t, request.images) if t is not None)
        return await self._judge.judge(request)


def _base_id(request_id: str) -> str:
    return request_id.split("/")[0]         # a non-batching judge splits a request per condition


async def _watch(compiled: Compiled, path: str, rate: float, max_side: int, names: Mapping[str, str],
                 guard: CachingJudge, trace_path: Path, speed: Optional[float]) -> ClipRun:
    shown, clock, start = _Shown(guard), [], time.time()    # clock: [anchor, last clip time]

    def on_frame(frame: Any) -> None:     # on the capture thread: pacing slows only decoding
        clock[:] = [frame.t_wall if not clock else clock[0], frame.t_wall - (clock[0] if clock else frame.t_wall)]
        shown.note(frame.image, clock[1])
        if speed:
            time.sleep(max(0.0, start + clock[1] / speed - time.time()))

    met, cpu, paid, refused = [], time.process_time(), guard.spent, guard.refused
    source = open_source(path, fps=rate, max_side=max_side, on_frame=on_frame)
    async with Monitor(**dict(compiled.monitor_kwargs, judge=shown), store=MemoryStore(), trace=trace_path) as m:
        async for delivery in m.watch(source, fps=rate):
            if delivery.decision.type == "met" and delivery.decision.condition_id in names:
                met.append(delivery.decision)
        cost = sum(s.get("cost_usd", 0.0) for s in m.stats()["streams"].values())
    anchor = clock[0] if clock else 0.0
    looks: list[tuple[str, str, Look]] = []
    drops: list[Drop] = []
    for rec in read_trace(trace_path):
        if rec["kind"] == "decide_input" and rec["input"]["type"] == "Observed":
            o = rec["input"]["observation"]
            if o["condition_id"] in names:
                looks.append((_base_id(o["request_id"]), o["condition_id"], Look(
                    names[o["condition_id"]], o["evidence_t"][0] - anchor, o["evidence_t"][1] - anchor, o["answer"],
                    o["failure"], o["purpose"], o.get("confidence") or 0.0,
                    shown.frames.get(_base_id(o["request_id"]), ()))))
        elif rec["kind"] == "drop" and rec.get("reason") in ("budget", "deadline") and rec.get("condition_id") in names:
            t1 = (rec["t_wall"] if rec.get("t_trigger") is None else rec["t_trigger"]) - anchor
            drops.append(Drop(names[rec["condition_id"]], t1 - EVENT_CLIP_S, t1, rec["reason"]))
    preds = []
    for d in met:       # scored by the frames of the looks it cites that said yes
        ids = {_base_id(r) for r in d.request_ids}
        ev = sorted(e.t_wall - anchor for e in d.evidence) or [d.t_observed - anchor]
        yes = sorted({t for rid, cid, lk in looks if cid == d.condition_id and rid in ids and not lk.failed
                      and lk.answer == "yes" and lk.confidence >= MIN_CONFIDENCE for t in lk.frames})
        preds.append(Prediction(names[d.condition_id], ev[0], ev[-1], ev[-1], d.confidence, tuple(yes)))
    return ClipRun(_basename(path), (clock[1] if clock else 0.0) + 1.0 / rate, tuple(preds),
                   tuple(lk for _, _, lk in looks), tuple(drops), cost_usd=cost, paid_usd=guard.spent - paid,
                   refused=guard.refused - refused,
                   compute_s=time.process_time() - cpu, cooldown_s=_cooldown(compiled))


def run(compiled: Compiled, labels: Labels, clips: Sequence[str], *, video_root: Union[str, os.PathLike],
        judge_cap_usd: float, cache: Optional[MutableMapping] = None, label_of: Optional[Mapping[str, str]] = None,
        fps: Optional[float] = None, spend_log: Union[None, str, os.PathLike] = None,
        speed: Optional[float] = 1.0) -> list[ClipRun]:
    """The full runtime over each clip: a :class:`~percept.Monitor` built from ``monitor_kwargs``
    (a fresh ``MemoryStore`` and a temporary trace), its judge wrapped in one :class:`CachingJudge`
    for the whole call (``judge_cap_usd`` in dollars; pass the same ``cache`` mapping to reuse
    replies across calls). Decisions become predictions, and the trace's looks and dropped looks
    the records misses are attributed with. ``cost_usd``: what the run would have cost live (a
    cached reply counts its original cost); ``paid_usd``: what was actually paid; ``compute_s``:
    CPU seconds of the whole process while the clip ran (decoding and the runtime included).
    ``refused``: requests the cap refused (they failed with ``budget``); a run that hit the cap
    warns, and its :class:`~percept.plans.score.Score` says ``capped``.

    Frames are fed at ``speed`` times real time (None: as fast as the Monitor takes them, which
    leaves it little time to look). The Monitor's request budget, deadlines and timers run on the
    real clock, so at any speed but 1 budget drops differ from a live run. A prediction's
    ``t_decided`` is the end of its evidence: latency from this mode is not a wall-clock latency."""
    if speed is not None and not (math.isfinite(speed) and speed > 0):
        raise ValueError(f"speed must be a positive number or None, got {speed!r}")
    if not (isinstance(judge_cap_usd, (int, float)) and math.isfinite(judge_cap_usd) and judge_cap_usd >= 0):
        raise ValueError(f"judge_cap_usd must be a finite number >= 0, got {judge_cap_usd!r}")
    rate, max_side, names = _setup(compiled, labels, clips, fps, label_of)
    guard = CachingJudge(compiled.monitor_kwargs["judge"], cap_usd=judge_cap_usd, spend_log=spend_log,
                         cache=True if cache is None else cache, speed=speed)

    async def main(tmp: str) -> list[ClipRun]:
        return [await _watch(compiled, os.path.join(os.fspath(video_root), f), rate, max_side, names, guard,
                             Path(tmp) / f"trace-{i}.jsonl", speed) for i, f in enumerate(clips)]

    with tempfile.TemporaryDirectory() as tmp:
        runs = asyncio.run(main(tmp))
    if guard.refused:
        warnings.warn(f"the ${judge_cap_usd:g} judge cap refused {guard.refused} request(s): "
                      "the score is of a capped run", RuntimeWarning, stacklevel=2)
    return runs


# --------------------------------------------------------------------------- freeze before test


def register(compiled: Compiled, manifest: SplitManifest, labels: Labels, *, split: str = "test", created: str,
             judge_versions: Optional[Mapping[str, str]] = None,
             ledger: Union[None, str, os.PathLike] = None) -> dict:
    """A registration record freezing ``compiled`` before ``split`` is run: the plan hash, its
    provenance (SDK version, prompt hash, the judges and the model ids requested, and
    ``judge_versions``, the model versions your judges reported, if any), the manifest and labels
    hashes, the split and ``created`` (your timestamp; no clock is read). ``registration_hash``
    covers the rest. No path is recorded, so the record is the same in every checkout. Keep it
    with your results and pass it to :func:`score_split`. With ``ledger`` (a path), the record is
    also appended to it as a JSON line."""
    if split not in SPLITS:
        raise ValueError(f"split must be one of {', '.join(SPLITS)}, got {split!r}")
    if manifest.labels_hash != labels.labels_hash:
        raise ValueError("the manifest was split from other labels")
    record = {"format": REGISTRATION_FORMAT, "plan_hash": compiled.plan_hash,
              "provenance": dict(compiled.provenance(), judge_versions=dict(judge_versions or {})),
              "manifest_hash": manifest.manifest_hash, "labels_hash": labels.labels_hash, "split": split,
              "created": created}
    record = dict(record, registration_hash=_sha(record))
    if ledger is not None:
        _append(ledger, {"kind": "registration", **record})
    return record


def _append(ledger: Union[str, os.PathLike], line: dict) -> None:
    with open(ledger, "a", encoding="utf-8") as f:
        f.write(json.dumps(line, sort_keys=True) + "\n")


def score_split(compiled: Compiled, labels: Labels, manifest: SplitManifest, *, split: str,
                video_root: Union[str, os.PathLike], judge_cap_usd: float, registration: Optional[Mapping] = None,
                ledger: Union[None, str, os.PathLike] = None, rescore: bool = False, **options: Any) -> Score:
    """:func:`run` on the clips of ``split`` and :func:`~percept.plans.score.evaluate` them.
    ``options`` go to :func:`run`.

    The test split is scored only with a ``registration`` from :func:`register` for this split
    whose plan, manifest and labels hashes match the current ones: freeze the plan before you look
    at the test split, so the test score is not chosen after the fact. Anything else raises
    ValueError. The returned score carries the ``registration_hash``.

    With ``ledger`` (the path given to :func:`register`), the registration must be in it, the
    test scoring is appended to it, and a second test scoring of the same registration is refused
    unless ``rescore=True`` (then it is recorded as a rescore). Without a ledger, scoring the
    test split once is the caller's responsibility."""
    if split not in SPLITS:
        raise ValueError(f"split must be one of {', '.join(SPLITS)}, got {split!r}")
    if manifest.labels_hash != labels.labels_hash:
        raise ValueError("the manifest was split from other labels")
    if split == "test":
        want = {"plan_hash": compiled.plan_hash, "manifest_hash": manifest.manifest_hash,
                "labels_hash": labels.labels_hash, "split": "test"}
        reg = dict(registration or {})
        body = {k: v for k, v in reg.items() if k != "registration_hash"}
        if not reg or any(reg.get(k) != v for k, v in want.items()) or reg.get("registration_hash") != _sha(body):
            raise ValueError("the test split is scored only for a frozen plan: call register(compiled, manifest, "
                             "labels, split='test', created=...) before running it and pass the record as "
                             "registration=; a record whose plan, manifest or labels hash differs from the current "
                             "ones (the plan, clips or labels changed since) does not count")
        rh = reg["registration_hash"]
        if ledger is not None:
            lines = []
            if os.path.exists(ledger):
                with open(ledger, encoding="utf-8") as f:
                    lines = [json.loads(x) for x in f if x.strip()]
            if not any(x.get("kind") == "registration" and x.get("registration_hash") == rh for x in lines):
                raise ValueError(f"registration {rh[:12]} is not in the ledger {_basename(os.fspath(ledger))}")
            if not rescore and any(x.get("kind") == "test_scored" and x.get("registration_hash") == rh for x in lines):
                raise ValueError(f"the test split was already scored for registration {rh[:12]}; "
                                 "pass rescore=True to score it again (recorded as a rescore)")
    score = evaluate(labels, run(compiled, labels, manifest.clips[split], video_root=video_root,
                                 judge_cap_usd=judge_cap_usd, **options))
    if split != "test":
        return score
    if ledger is not None:
        _append(ledger, {"kind": "test_scored", "registration_hash": rh, "rescore": rescore,
                         "recall": score.recall, "precision": score.precision, "capped": score.capped})
    return dataclasses.replace(score, registration_hash=rh)


# --------------------------------------------------------------------------- declared configurations


def sweep(configs: Mapping[str, Compiled], labels: Labels, manifest: SplitManifest, *, split: str,
          video_root: Union[str, os.PathLike], judge_cap_usd: float, cache: Optional[MutableMapping] = None,
          **options: Any) -> dict[str, Score]:
    """:func:`run` each configuration you declare (name -> compiled plan) on the ``dev`` or
    ``validation`` split and return their scores side by side, in your order. ``judge_cap_usd``
    caps the whole sweep (a configuration that runs out is scored ``capped``,
    with its refused requests counted); replies are shared through one cache, so a request two configurations
    send alike is paid once. It neither ranks nor picks: comparing the scores is yours. The test
    split is not swept: it is scored once, with :func:`score_split`, for a registered plan."""
    if split not in ("dev", "validation"):
        raise ValueError(f"sweep runs on dev or validation, not {split!r}")
    cache = {} if cache is None else cache
    out: dict[str, Score] = {}
    left = judge_cap_usd
    for name, compiled in configs.items():
        runs = run(compiled, labels, manifest.clips[split], video_root=video_root, judge_cap_usd=left,
                   cache=cache, **options)
        left = max(0.0, left - sum(r.cost_usd for r in runs))
        out[name] = evaluate(labels, runs)
    return out
