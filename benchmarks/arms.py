"""The arms: ways to turn the same footage and condition text into decisions.

* ``percept``: a ``Monitor`` with its defaults (change trigger, confirm/verify, request budget),
  fed in real time through a ``PushSource``. Public API only.
* ``chunk<T>``: judge the last ``T`` seconds every ``T`` seconds with every condition batched,
  then a plain cooldown per condition: what a developer writes in an afternoon.
* ``detector_debounce``: an OpenCV detector (HOG people detector or MOG2 motion) with a
  cooldown; only for conditions a detector can express, the others are reported unsupported.
* ``detector_then_vlm``: the same detectors gate judge calls; a condition no detector expresses
  is gated by motion.

Baseline arms run on the case timeline (no waiting); their decision time is the end of the
evidence plus the judge's reported latency. Every judged look is recorded, so misses can be
attributed to a stage in every arm.
"""

from __future__ import annotations

import asyncio
import time
import tracemalloc
import warnings
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from percept import Condition, JudgeItem, JudgeRequest, MemoryStore, Monitor, PushSource
from percept import trace as ptrace

from benchmarks.cases import Feed, VideoCase

CLIP_FRAMES, CLIP_S = 4, 4.0        # the evidence format Percept uses for events, shared by every arm
MIN_CONFIDENCE = 0.6                # Percept's default, applied by the baselines too
COOLDOWN_S = 30.0


@dataclass
class Pred:
    condition_id: str
    t0: float               # evidence interval, case timeline
    t1: float
    t_decided: float
    confidence: float


@dataclass
class Look:
    condition_id: str
    t0: float
    t1: float
    answer: str
    failed: Optional[str]   # the failure, if the look got no answer
    purpose: str = "detect"


@dataclass
class ArmResult:
    arm: str
    case_id: str
    preds: list = field(default_factory=list)
    looks: list = field(default_factory=list)
    budget_drops: list = field(default_factory=list)    # (condition_id, t0, t1): a look the budget dropped
    unsupported: list = field(default_factory=list)     # condition ids this arm can't express
    uncertain: int = 0
    digest: str = ""
    cost_usd: float = 0.0
    requests: int = 0
    aborted: bool = False


# --------------------------------------------------------------------------- detectors


class Detectors:
    """Rising edges of cheap OpenCV detectors, computed once per frame and shared."""

    def __init__(self, kinds: set) -> None:
        import cv2
        self.cv2, self.kinds = cv2, kinds & self.available()
        self.mog = cv2.createBackgroundSubtractorMOG2(history=150, detectShadows=False)
        if "person" in self.kinds:
            self.hog = cv2.HOGDescriptor()
            self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        self.active = {k: False for k in kinds}
        self.hits, self.next_hog, self.frames = 0, 0.0, 0

    @staticmethod
    def available() -> set:
        """OpenCV 5 dropped the built-in HOG people detector; then nothing expresses "person"."""
        import cv2
        return {"motion"} | ({"person"} if hasattr(cv2, "HOGDescriptor") else set())

    def step(self, t: float, image) -> set:
        """The detector kinds that turned on at this frame."""
        cv2, fired = self.cv2, set()
        small = cv2.resize(image, (320, max(1, round(image.shape[0] * 320 / image.shape[1]))))
        self.frames += 1
        if "motion" in self.kinds:
            fg = (self.mog.apply(small) > 0).mean()
            on = fg > (0.01 if self.active["motion"] else 0.02)    # hysteresis
            if on and not self.active["motion"] and self.frames > 5:
                fired.add("motion")
            self.active["motion"] = on
        if "person" in self.kinds and t >= self.next_hog:
            self.next_hog = t + 0.5                                  # 2 detections a second
            boxes, _ = self.hog.detectMultiScale(small, winStride=(8, 8))
            self.hits = self.hits + 1 if len(boxes) else 0
            on = self.hits >= 2
            if on and not self.active["person"]:
                fired.add("person")
            self.active["person"] = on
        return fired


# --------------------------------------------------------------------------- baselines


async def frames_of(feed: Feed):
    """The feed's frames, decoded on a worker thread so other cases keep running."""
    frames = iter(feed)
    while (item := await asyncio.to_thread(next, frames, None)) is not None:
        yield item


class Baseline:
    """Shared plumbing: a short frame history, judge calls, looks and the cooldown."""

    def __init__(self, arm: str, case: VideoCase, judge, timelines, cooldown_s: float = COOLDOWN_S) -> None:
        timelines.set(case.case_id, 0.0, 1.0)      # evidence times are case times
        self.case, self.judge, self.cooldown = case, judge, cooldown_s
        self.result = ArmResult(arm, case.case_id)
        self.history: deque = deque()
        self.last_emit: dict[str, float] = {}
        self.conds = {c.id: Condition(c.id, c.text, kind=c.kind) for c in case.conditions}

    def remember(self, t: float, image, keep_s: float) -> None:
        if not self.history or t - self.history[-1][0] >= 0.5:      # 2 fps, as Percept's pre-roll
            self.history.append((t, image))
        while self.history and self.history[0][0] < t - keep_s:
            self.history.popleft()

    def clip(self, t0: float, t1: float) -> list:
        held = [f for f in self.history if t0 <= f[0] <= t1]
        if len(held) <= CLIP_FRAMES:
            return held
        n = len(held) - 1
        return [held[round(i * n / (CLIP_FRAMES - 1))] for i in range(CLIP_FRAMES)]

    async def ask(self, ids: list, frames: list) -> None:
        if not ids or not frames or self.judge.tripped.is_set():
            return
        t0, t1 = frames[0][0], frames[-1][0]
        r = self.result
        request = JudgeRequest(f"{r.arm}-{r.case_id}-{r.requests}", r.case_id, 0,
                               tuple(JudgeItem(self.conds[i]) for i in ids), tuple(f[1] for f in frames), (t0, t1))
        result = await self.judge.judge(request)
        r.requests += 1
        r.cost_usd += sum(u.usage.cost_usd for u in result.usage)
        latency = sum(u.usage.latency_s for u in result.usage)
        for ob in result.observations:
            r.looks.append(Look(ob.condition_id, t0, t1, ob.answer, ob.failure))
            if ob.failure == "budget":
                r.aborted = True
            if ob.answer == "yes" and not ob.failed and ob.confidence >= MIN_CONFIDENCE:
                self.emit(ob.condition_id, t0, t1, t1 + latency, ob.confidence)

    def emit(self, cid: str, t0: float, t1: float, t_decided: float, confidence: float) -> None:
        last = self.last_emit.get(cid)
        if last is None or t1 - last >= self.cooldown:
            self.last_emit[cid] = t1
            self.result.preds.append(Pred(cid, t0, t1, t_decided, confidence))


async def run_chunk(case: VideoCase, judge, timelines, period_s: float, *, cooldown_s: float = COOLDOWN_S,
                    fps: float = 5.0) -> ArmResult:
    b = Baseline(f"chunk{period_s:g}", case, judge, timelines, cooldown_s)
    feed, due, t = Feed(case, fps), period_s, 0.0
    async for t, image in frames_of(feed):
        b.remember(t, image, period_s + 1.0)
        if t >= due:
            await b.ask(list(b.conds), b.clip(due - period_s, t))
            due += period_s
        if judge.tripped.is_set():
            break
    if t > due - period_s + 1.0:                    # the last, shorter chunk
        await b.ask(list(b.conds), b.clip(due - period_s, t))
    b.result.digest = feed.digest
    return b.result


async def run_detector(case: VideoCase, judge, timelines, *, then_vlm: bool, cooldown_s: float = COOLDOWN_S,
                       gate_s: float = 5.0, fps: float = 5.0) -> ArmResult:
    b = Baseline("detector_then_vlm" if then_vlm else "detector_debounce", case, judge, timelines, cooldown_s)
    have = Detectors.available()
    gates = {c.id: (c.detector if c.detector in have else None) or ("motion" if then_vlm else None)
             for c in case.conditions}
    b.result.unsupported = [cid for cid, g in gates.items() if g is None]
    det = Detectors({g for g in gates.values() if g})
    feed, gate_until = Feed(case, fps), {}
    async for t, image in frames_of(feed):
        b.remember(t, image, CLIP_S + 1.0)
        fired = det.step(t, image) if det.kinds else set()
        ids = [cid for cid, g in gates.items() if g in fired and t >= gate_until.get(cid, -1.0)]
        for cid in ids:
            gate_until[cid] = t + gate_s
        if not then_vlm:
            for cid in ids:
                b.emit(cid, t, t, t, 1.0)
        elif ids:
            await b.ask(ids, b.clip(t - CLIP_S, t))
        if judge.tripped.is_set():
            break
    b.result.digest = feed.digest
    return b.result


# --------------------------------------------------------------------------- Percept


async def run_percept(case: VideoCase, judge, timelines, trace_path: Path, *, speed: float = 1.0,
                      fps: float = 5.0, stats: Optional[dict] = None) -> ArmResult:
    """One ``Monitor`` per case, frames pushed at ``speed`` times real time (1.0 is the faithful
    setting: Percept's budget and timers run on the real clock)."""
    result = ArmResult("percept", case.case_id)
    conditions = [Condition(c.id, c.text, kind=c.kind) for c in case.conditions]
    source = PushSource(case.case_id)
    feed, anchor = Feed(case, fps), time.time() + 0.2
    timelines.set(case.case_id, anchor, speed)
    peak_preroll, lag = 0, 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")     # MemoryStore warns that it isn't crash-safe: fine here
        monitor = Monitor(judge, conditions, store=MemoryStore(), trace=trace_path)
    async with monitor:
        decisions = monitor.watch(source, ack="auto", fps=fps * speed)

        async def play() -> None:
            nonlocal peak_preroll, lag
            try:
                async for t, image in frames_of(feed):
                    if judge.tripped.is_set():
                        break
                    wall = anchor + t / speed
                    await asyncio.sleep(max(0.0, wall - time.time()))
                    lag = max(lag, time.time() - wall)          # how far pacing fell behind real time
                    source.push(image, t_wall=wall)
                    ring = getattr(source, "preroll", None)
                    peak_preroll = max(peak_preroll, len(ring) if ring is not None else 0)
            finally:
                source.close()

        async def stop_on_cap() -> None:
            await judge.tripped.wait()
            result.aborted = True
            await monitor.aclose()

        player, stopper = asyncio.create_task(play()), asyncio.create_task(stop_on_cap())
        async for delivery in decisions:
            d = delivery.decision
            ev = [e.t_wall for e in d.evidence] or [d.t_observed]
            media = lambda w: timelines.media(case.case_id, w)    # noqa: E731
            if d.type == "met":
                result.preds.append(Pred(d.condition_id, media(min(ev)), media(max(ev)), media(d.t_decided),
                                         d.confidence))
            elif d.type == "uncertain":
                result.uncertain += 1
        await player
        stopper.cancel()
        s = monitor.stats()["streams"].get(case.case_id, {})
        result.cost_usd, result.requests = s.get("cost_usd", 0.0), s.get("requests", 0)
    result.digest = feed.digest
    if stats is not None:
        stats["preroll_frames"] = max(stats.get("preroll_frames", 0), peak_preroll)
        stats["max_push_lag_s"] = round(max(stats.get("max_push_lag_s", 0.0), lag), 3)
    _read_trace(result, trace_path, lambda w: timelines.media(case.case_id, w))
    return result


def _read_trace(result: ArmResult, path: Path, media) -> None:
    """Looks and budget drops from the Monitor's trace, for stage attribution."""
    for rec in ptrace.read(path):
        if rec["kind"] == "decide_input" and rec["input"]["type"] == "Observed":
            o = rec["input"]["observation"]
            result.looks.append(Look(o["condition_id"], media(o["evidence_t"][0]), media(o["evidence_t"][1]),
                                     o["answer"], o["failure"], o["purpose"]))
        elif rec["kind"] == "fate" and rec.get("event") == "runtime" and rec["fate"] == "dropped_budget":
            t0 = media(rec["bucket_wall"])          # counted per minute: the drop is somewhere in it
            result.budget_drops.append((rec["condition_id"], t0, t0 + 60.0 * (media(1.0) - media(0.0))))


class Footprint:
    """CPU seconds and peak traced memory over a block of work."""

    def __enter__(self) -> "Footprint":
        if not tracemalloc.is_tracing():
            tracemalloc.start()
        tracemalloc.reset_peak()
        self.cpu0 = time.process_time()
        return self

    def __exit__(self, *exc) -> None:
        self.cpu_s = time.process_time() - self.cpu0
        self.peak_mb = tracemalloc.get_traced_memory()[1] / 1e6
