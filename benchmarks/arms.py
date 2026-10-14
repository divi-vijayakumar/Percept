"""The arms: ways to turn the same footage and condition text into decisions.

* ``percept``: a ``Monitor`` with the product defaults (change trigger, confirm/verify, request
  budget of 6 attempts a minute: each request reserves 2 and is charged the attempts it used, so
  at most 5 requests a minute when each succeeds first time) EXCEPT the rearm, which a run may
  standardise (E1: 30 s, the baselines' cooldown; the product default ``rearm_s`` is 120 s), fed
  in real time through a ``PushSource``.
* ``percept-matched``: the same with ``max_requests_per_min`` raised (``--percept-budget``,
  default 13 attempts a minute: at most 12 requests when each succeeds first time, ``chunk5``'s
  rate; retries and verification draw on the same budget), so budget is not the only difference.
  Both use only Percept's public API.
* ``chunk<T>``: judge the last ``T`` seconds every ``T`` seconds with every condition batched,
  then a plain cooldown per condition: what a developer writes in an afternoon.
* ``detector_debounce``: an OpenCV detector (MOG2 motion, or OpenCV 4's HOG people detector run
  on a 320-pixel frame: a weak detector) with a cooldown; only for conditions a detector can
  express, the others are reported unsupported.
* ``detector_then_vlm``: the same detectors gate judge calls; a condition no detector expresses
  is gated by motion.

Baseline arms run on the case timeline (no waiting); their decision time is the end of the
evidence plus the judge's reported latency. Every judged look is recorded, so misses can be
attributed to a stage in every arm.

Every decision and look records ``frames``: the case times of the frames actually shown to the
judge (a detector-only decision: the frame it fired on). Scoring uses those frames, not the
interval they span.

Each Percept case runs its own ``Monitor`` with the product's per-Monitor concurrency, and each
case decodes on its own thread, so cases run at once never compete for a decode thread or a
request slot (they do share the process, the event loop and the judge's HTTP client).
"""

from __future__ import annotations

import asyncio
import resource
import sys
import time
import weakref
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from percept import Condition, JudgeItem, JudgeRequest, MemoryStore, Monitor, PushSource
from percept import trace as ptrace
from percept.clips import PreRoll
from percept.judges import base
from percept.monitor import _max_output_tokens   # the Monitor's reply sizing, held fixed across arms

from benchmarks.cases import Feed, VideoCase

CLIP_FRAMES, CLIP_S = 4, 4.0        # the evidence format Percept uses for events, shared by every arm
from benchmarks.metrics import MIN_CONFIDENCE   # noqa: E402  Percept's default, applied by the baselines too
COOLDOWN_S = 30.0
HISTORY_SPACING_S = 0.5             # baselines keep at most 2 frames a second, as Percept's pre-roll
MAX_LAG_S = 1.0                     # a pushed frame later than this behind real time counts as lost


@dataclass
class Pred:
    condition_id: str
    t0: float               # evidence interval, case timeline
    t1: float
    t_decided: float
    confidence: float
    frames: tuple = ()      # case times of the frames the judge was shown (empty: the interval ends)
    evidence_frames: tuple = ()     # display only: every frame the decision cites (Percept: every request's)
    request_ids: tuple = ()         # Percept: the requests the decision cites


@dataclass
class Look:
    condition_id: str
    t0: float
    t1: float
    answer: str
    failed: Optional[str]   # the failure, if the look got no answer
    purpose: str = "detect"
    confidence: float = 0.0
    frames: tuple = ()      # case times of the frames the judge was shown (empty: the interval ends)
    request_id: str = ""    # Percept: the request that produced it


@dataclass
class ArmResult:
    arm: str
    case_id: str
    preds: list = field(default_factory=list)
    looks: list = field(default_factory=list)
    budget_drops: list = field(default_factory=list)    # (condition_id, t0, t1, reason): a look dropped unjudged
    fires: list = field(default_factory=list)           # (condition_id, t): a detector gate opened
    unsupported: list = field(default_factory=list)     # condition ids this arm can't express
    uncertain: int = 0
    digest: str = ""
    cost_usd: float = 0.0
    requests: int = 0
    aborted: bool = False
    cooldown_s: float = 0.0     # how long one decision suppresses the next for its condition
    seconds: float = 0.0        # footage actually watched (less than the case when aborted)
    # Frame accounting. Intentional: the configured sampling skipped it. Unintended: decoding,
    # a queue or processing couldn't keep up. Only unintended loss disqualifies a run.
    intentional_sampling: dict = field(default_factory=dict)  # harness_fps, monitor_fps_cap
    unintended_loss: dict = field(default_factory=dict)       # decode_gaps, handoff_dropped, late_frames
    budget_dropped: int = 0     # looks the request budget dropped (exact count)
    unverified: int = 0         # event candidates closed without a verified answer
    empty_clips: int = 0        # judge calls skipped because the clip held no frame
    preds_without_yes_frames: int = 0   # Percept decisions whose yes-look frames weren't found
    error: str = ""             # the case failed: recorded, scored as skipped
    compute: dict = field(default_factory=dict)     # detector arms: per-case CPU and wall seconds


def _finish(result: ArmResult, feed: Feed, complete: bool) -> ArmResult:
    result.digest = feed.digest
    result.seconds = feed.case.duration if complete else feed.frames / feed.fps
    result.aborted = result.aborted or not complete
    result.intentional_sampling["harness_fps"] = feed.sampled_out
    result.unintended_loss["decode_gaps"] = feed.decode_gaps
    return result


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


def _feed(case: VideoCase, fps: float, timelines) -> Feed:
    """The case's frames, each one noted with its case time (the oracle reads that)."""
    return Feed(case, fps, note=lambda t, image: timelines.note(case.case_id, t, image))


async def frames_of(feed: Feed, pace: Optional[float] = None):
    """The feed's frames, decoded on the case's own thread: other cases keep running, and
    decoding never takes a thread from the shared pool (JPEG encoding, trace flushes). With
    ``pace``, a frame is not yielded before its time at ``pace`` times real time (a caller that
    falls behind, e.g. waiting on the judge, catches up without sleeping), so a baseline calls the
    provider at the rate it would live, as Percept does; None yields as fast as decoding allows."""
    frames, loop = iter(feed), asyncio.get_running_loop()
    start = loop.time()
    with ThreadPoolExecutor(1, thread_name_prefix=f"decode-{feed.case.case_id}") as pool:
        while (item := await loop.run_in_executor(pool, next, frames, None)) is not None:
            if pace:
                wait = start + item[0] / pace - loop.time()
                if wait > 0:
                    await asyncio.sleep(wait)
            yield item


class Baseline:
    """Shared plumbing: a short frame history, judge calls, looks and the cooldown."""

    def __init__(self, arm: str, case: VideoCase, judge, timelines, cooldown_s: float = COOLDOWN_S) -> None:
        timelines.set(case.case_id, 0.0, 1.0)      # evidence times are case times
        self.case, self.judge, self.cooldown, self.timelines = case, judge, cooldown_s, timelines
        self.result = ArmResult(arm, case.case_id, cooldown_s=cooldown_s)
        self.history: deque = deque()
        self.last_emit: dict[str, float] = {}
        self.conds = {c.id: Condition(c.id, c.text, kind=c.kind) for c in case.conditions}

    def remember(self, t: float, image, keep_s: float) -> None:
        if not self.history or t - self.history[-1][0] >= HISTORY_SPACING_S:
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
        if not ids or self.judge.tripped.is_set():
            return
        r = self.result
        if not frames:              # nothing held in the window: counted, never silent
            r.empty_clips += 1
            return
        t0, t1, times = frames[0][0], frames[-1][0], tuple(f[0] for f in frames)
        items = tuple(JudgeItem(self.conds[i]) for i in ids)
        request = JudgeRequest(f"{r.arm}-{r.case_id}-{r.requests}", r.case_id, 0, items,
                               tuple(f[1] for f in frames), (t0, t1),
                               max_output_tokens=_max_output_tokens(len(items),
                                                                    sum(len(i.condition.id) for i in items)))
        result = await self.judge.judge(request)
        r.requests += 1
        r.cost_usd += sum(u.usage.cost_usd for u in result.usage)
        latency = sum(u.usage.latency_s for u in result.usage)
        for ob in result.observations:
            r.looks.append(Look(ob.condition_id, t0, t1, ob.answer, ob.failure, "detect", ob.confidence, times))
            if ob.failure == "budget":
                r.aborted = True
            if ob.answer == "yes" and not ob.failed and ob.confidence >= MIN_CONFIDENCE:
                self.emit(ob.condition_id, t0, t1, t1 + latency, ob.confidence, times)

    def emit(self, cid: str, t0: float, t1: float, t_decided: float, confidence: float, frames: tuple) -> None:
        """A decision, unless the condition's cooldown (counted from the end of the last emitted
        decision's evidence) still runs."""
        last = self.last_emit.get(cid)
        if last is None or t1 - last >= self.cooldown:
            self.last_emit[cid] = t1
            self.result.preds.append(Pred(cid, t0, t1, t_decided, confidence, frames))


async def run_chunk(case: VideoCase, judge, timelines, period_s: float, *, cooldown_s: float = COOLDOWN_S,
                    fps: float = 5.0, pace: Optional[float] = None) -> ArmResult:
    b = Baseline(f"chunk{period_s:g}", case, judge, timelines, cooldown_s)
    feed, due, t, complete = _feed(case, fps, timelines), period_s, 0.0, False
    async for t, image in frames_of(feed, pace):
        b.remember(t, image, period_s + 1.0)
        if t >= due:
            await b.ask(list(b.conds), b.clip(due - period_s, t))
            due += period_s
        if judge.tripped.is_set():
            break
    else:
        complete = True
        if t > due - period_s + 1.0:                # the last, shorter chunk
            await b.ask(list(b.conds), b.clip(due - period_s, t))
    return _finish(b.result, feed, complete and not judge.tripped.is_set())


async def run_detector(case: VideoCase, judge, timelines, *, then_vlm: bool, cooldown_s: float = COOLDOWN_S,
                       gate_s: float = 5.0, fps: float = 5.0, pace: Optional[float] = None) -> ArmResult:
    b = Baseline("detector_then_vlm" if then_vlm else "detector_debounce", case, judge, timelines, cooldown_s)
    have = Detectors.available()
    gates = {c.id: (c.detector if c.detector in have else None) or ("motion" if then_vlm else None)
             for c in case.conditions}
    b.result.unsupported = [cid for cid, g in gates.items() if g is None]
    det = Detectors({g for g in gates.values() if g})
    feed, gate_until, complete = _feed(case, fps, timelines), {}, False
    async for t, image in frames_of(feed, pace):
        b.remember(t, image, CLIP_S + 1.0)
        fired = det.step(t, image) if det.kinds else set()
        ids = [cid for cid, g in gates.items() if g in fired and t >= gate_until.get(cid, -1.0)]
        for cid in ids:
            gate_until[cid] = t + gate_s
            b.result.fires.append((cid, t))
        if not then_vlm:
            for cid in ids:
                b.emit(cid, t, t, t, 1.0, (t,))
        elif ids:
            await b.ask(ids, b.clip(t - CLIP_S, t))
        if judge.tripped.is_set():
            break
    else:
        complete = True
    return _finish(b.result, feed, complete and not judge.tripped.is_set())


# --------------------------------------------------------------------------- Percept


async def run_percept(case: VideoCase, judge, timelines, trace_path: Path, *, speed: float = 1.0,
                      fps: float = 5.0, max_requests_per_min: Optional[int] = None,
                      rearm_s: Optional[float] = None, arm: str = "percept",
                      stats: Optional[dict] = None, conditions: Optional[list] = None,
                      monitor_judge=None, on_frame=None) -> ArmResult:
    """One ``Monitor`` per case, frames pushed at ``speed`` times real time (1.0 is the faithful
    setting: Percept's budget and timers run on the real clock). ``None`` keeps a default. Frames
    are already sampled to ``fps``, so the Monitor processes every one (``watch(fps=None)``).
    ``conditions`` (a compiled plan's) replace the ones built from the case's condition text.
    ``monitor_judge`` replaces ``judge`` as the Monitor's judge (``judge`` still carries the run's
    spend cap), and ``await on_frame(monitor, t, image, wall)`` runs after each frame is pushed
    (a detector that pokes the Monitor)."""
    extra = {} if rearm_s is None else {"rearm_s": rearm_s}
    if conditions is None:
        conditions = [Condition(c.id, c.text, kind=c.kind, **extra) for c in case.conditions]
    result = ArmResult(arm, case.case_id, cooldown_s=conditions[0].rearm_s if conditions else 0.0)
    preroll = PreRoll()
    source = PushSource(case.case_id, preroll=preroll)
    feed, anchor, complete = _feed(case, fps, timelines), time.time() + 0.2, False
    timelines.set(case.case_id, anchor, speed)
    peak_preroll, lag, late = 0, 0.0, 0
    budget = {} if max_requests_per_min is None else {"max_requests_per_min": max_requests_per_min}
    shown = FrameRecorder(judge if monitor_judge is None else monitor_judge, timelines)
    monitor = Monitor(shown, conditions, store=MemoryStore(), trace=trace_path, **budget)
    async with monitor:
        decisions = monitor.watch(source, ack="auto", fps=None)

        async def play() -> None:
            nonlocal peak_preroll, lag, late, complete
            try:
                async for t, image in frames_of(feed):
                    if judge.tripped.is_set():
                        break
                    wall = anchor + t / speed
                    await asyncio.sleep(max(0.0, wall - time.time()))
                    if judge.tripped.is_set():                  # the cap closed the monitor meanwhile
                        break
                    behind = time.time() - wall                 # how far pacing fell behind real time
                    lag, late = max(lag, behind), late + (behind > MAX_LAG_S)
                    shown.pushed(image, t)
                    source.push(image, t_wall=wall)
                    if on_frame is not None:
                        await on_frame(monitor, t, image, wall)
                    peak_preroll = max(peak_preroll, len(preroll))
                else:
                    complete = True
            finally:
                source.close()

        async def stop_on_cap() -> None:
            await judge.tripped.wait()
            await monitor.aclose()

        player, stopper = asyncio.create_task(play()), asyncio.create_task(stop_on_cap())
        media = lambda w: timelines.media(case.case_id, w)    # noqa: E731
        async for delivery in decisions:
            d = delivery.decision
            ev = sorted(media(e.t_wall) for e in d.evidence) or [media(d.t_observed)]
            if d.type == "met":
                result.preds.append(Pred(d.condition_id, ev[0], ev[-1], media(d.t_decided), d.confidence,
                                         (), tuple(ev), tuple(d.request_ids)))
            elif d.type == "uncertain":
                result.uncertain += 1
        await player
        stopper.cancel()
        s = monitor.stats()["streams"].get(case.case_id, {})
        result.cost_usd, result.requests = s.get("cost_usd", 0.0), s.get("requests", 0)
        result.intentional_sampling["monitor_fps_cap"] = s.get("frames_capped", 0)
        result.unintended_loss.update(handoff_dropped=source.dropped, late_frames=late)
        result.budget_dropped = s.get("fates", {}).get("dropped_budget", 0)
    _finish(result, feed, complete and not judge.tripped.is_set())
    if stats is not None:
        stats["preroll_frames"] = max(stats.get("preroll_frames", 0), peak_preroll)
        stats["max_push_lag_s"] = round(max(stats.get("max_push_lag_s", 0.0), lag), 3)
        stats["requests_timed_by_content"] = stats.get("requests_timed_by_content", 0) + shown.by_content
    _read_trace(result, trace_path, media, shown.frames)
    yes_frames(result)
    return result


def _base(request_id: str) -> str:
    return request_id.split("/")[0]         # a non-batching judge splits a request per condition


def yes_frames(result: ArmResult) -> None:
    """A Percept decision's ``frames``: the frames of the looks it cites that said ``yes`` (at
    ``MIN_CONFIDENCE`` or more, detect or verify) for its condition. Its evidence also cites the
    requests that said no; those frames were not the reason for the decision, so, as for the
    baselines (whose decision frames are the yes-request's), they earn no credit. A decision whose
    yes-looks can't be found keeps no frames and is scored at its evidence's ends, and counted."""
    for p in result.preds:
        ids = {_base(r) for r in p.request_ids}
        times = sorted({t for lk in result.looks if lk.condition_id == p.condition_id and _base(lk.request_id) in ids
                        and not lk.failed and lk.answer == "yes" and lk.confidence >= MIN_CONFIDENCE
                        for t in lk.frames})
        if times:
            p.frames = tuple(times)
        else:
            result.preds_without_yes_frames += 1


class FrameRecorder:
    """Passes every request to ``judge`` unchanged and remembers the case times of the images it
    carried, by request id, so Percept's looks can be scored by the frames actually judged.
    Its name, price, batching and attempts are the wrapped judge's.

    A request's images are the pushed arrays themselves, or views of them (the source keeps a
    frame that needs no downscaling as is; a crop is a view), so each is traced to the frame
    pushed by identity. When that fails for any image, the request falls back to matching
    images by content (``Timelines.frame_times``), which is ambiguous only for byte-identical
    frames."""

    def __init__(self, judge, timelines) -> None:
        self._judge, self._timelines, self.frames = judge, timelines, {}
        self.name, self.can_batch, self.price = judge.name, judge.can_batch, judge.price
        self.max_attempts = base.max_attempts(judge)
        self._pushed: dict[int, tuple] = {}         # id(image) -> (weak reference, case time)
        self.exact = self.by_content = 0

    def pushed(self, image, t: float) -> None:
        if len(self._pushed) > 512:                 # forget frames nothing holds any more
            self._pushed = {k: v for k, v in self._pushed.items() if v[0]() is not None}
        self._pushed[id(image)] = (weakref.ref(image), t)

    def _time(self, image) -> Optional[float]:
        root = image
        while isinstance(root.base, np.ndarray):
            root = root.base
        entry = self._pushed.get(id(root))
        return entry[1] if entry is not None and entry[0]() is root else None

    async def judge(self, request: JudgeRequest):
        times = [self._time(image) for image in request.images]
        if None in times:
            self.by_content += 1
            self.frames[request.request_id] = self._timelines.frame_times(request)
        else:
            self.exact += 1
            self.frames[request.request_id] = tuple(times)
        return await self._judge.judge(request)


def _read_trace(result: ArmResult, path: Path, media, frames: Optional[dict] = None) -> None:
    """Looks and dropped looks from the Monitor's trace, for stage attribution. ``frames`` maps a
    request id to the case times of the images it carried (see ``FrameRecorder``)."""
    frames = frames or {}
    for rec in ptrace.read(path):
        if rec["kind"] == "decide_input" and rec["input"]["type"] == "Observed":
            o = rec["input"]["observation"]
            shown = frames.get(_base(o["request_id"]), ())
            result.looks.append(Look(o["condition_id"], media(o["evidence_t"][0]), media(o["evidence_t"][1]),
                                     o["answer"], o["failure"], o["purpose"], o.get("confidence") or 0.0,
                                     tuple(shown), o["request_id"]))
        elif rec["kind"] == "drop" and rec.get("reason") in ("budget", "deadline"):
            # One record per dropped look: the request budget refused it (``budget``), or it waited
            # past its deadline for a request slot or evidence (``deadline``). Its evidence would
            # have been the clip ending at the trigger (or, without one, when it was dropped).
            t = rec.get("t_trigger")
            t1 = media(rec["t_wall"] if t is None else t)
            result.budget_drops.append((rec["condition_id"], t1 - CLIP_S, t1, rec["reason"]))
        elif rec["kind"] == "fate" and rec.get("event") == "unverified":
            result.unverified += 1


def _rusage() -> tuple[float, float]:
    """(CPU seconds, peak resident set in MB) of this process so far."""
    r = resource.getrusage(resource.RUSAGE_SELF)
    scale = 1e6 if sys.platform == "darwin" else 1e3      # ru_maxrss: bytes on macOS, KiB on Linux
    return r.ru_utime + r.ru_stime, r.ru_maxrss / scale


def _asleep_s() -> Optional[float]:
    """Seconds the machine has spent asleep or suspended since boot: a clock that keeps counting
    through sleep minus one that stops. None where the platform has no such pair."""
    if sys.platform == "darwin":
        return time.clock_gettime(time.CLOCK_MONOTONIC_RAW) - time.clock_gettime(time.CLOCK_UPTIME_RAW)
    if hasattr(time, "CLOCK_BOOTTIME"):
        return time.clock_gettime(time.CLOCK_BOOTTIME) - time.clock_gettime(time.CLOCK_MONOTONIC)
    return None


class Footprint:
    """CPU seconds over a block of work, the process's peak resident memory when it ended (a
    high-water mark for the whole run so far: run one arm per process for a per-arm peak), and
    the seconds the machine slept during it (None where unmeasurable). A sleep stops the clock the
    footage is paced by, so a run that slept is not a real-time run."""

    def __enter__(self) -> "Footprint":
        self.cpu0, _ = _rusage()
        self.asleep0 = _asleep_s()
        return self

    def __exit__(self, *exc) -> None:
        cpu, self.peak_rss_mb = _rusage()
        self.cpu_s = cpu - self.cpu0
        asleep = _asleep_s()
        self.slept_s = None if asleep is None or self.asleep0 is None else round(max(0.0, asleep - self.asleep0), 3)
