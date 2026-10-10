"""The runtime: ``Monitor`` watches streams, decides when to look, judges within a budget, feeds
the answers to the reducer and delivers its decisions durably.

One event loop owns all decision state; nothing here takes a lock. Per stream there is one
source (with its own capture thread), a runner task (frame -> quality -> scene -> triggers ->
due work), a dispatcher task (budget, priority, single flight), a committer task (keyframes,
then one atomic outbox write per emitting step) and one decide state. Shared across streams:
the judge, limited by ``max_concurrent_requests``, and the store's writer thread.

Budget (per stream). A request reserves ``max_attempts(judge)`` attempts (per item for a judge
that can't batch) before it is sent and holds them until it finishes; then its actual attempts
stay charged for the window. A request is admitted only while everything charged in that way
plus its reservation fits ``max_requests_per_min``, so no 60 s span ever holds more outbound
attempts than the limit (for judges that report their attempts truthfully). Dollars work the
same way over an hour when the judge has a ``price``; without one, only the reported cost counts.
Verification uses the same budget and the same concurrency limit: its own queue, not a bypass.

Priority when constrained: verify, event detect, state detect, heartbeat, recheck. Event and
state detections wait at most ``DEADLINES`` seconds, then are dropped with fate
``dropped_budget``; heartbeats and rechecks wait for the next free slot.

Time: decide sees wall-clock time; timers run on the loop's monotonic clock, converted from
decide's wall-clock deadlines when they are scheduled (a timer that fires early is rescheduled
by decide). Triggers see each frame's stream clock ``Frame.t``.
"""

from __future__ import annotations

import asyncio
import itertools
import json
import logging
import time
from collections import Counter, OrderedDict, deque
from dataclasses import dataclass, replace
from typing import Any, Iterable, Optional, Sequence, Union

import numpy as np

from percept import decide
from percept.clips import PreRoll
from percept.contracts import (
    BlindInterval,
    Condition,
    CounterExample,
    Decision,
    EvidenceRef,
    Frame,
    Health,
    Judge,
    JudgeItem,
    JudgeRequest,
    JudgeResult,
    Source,
)
from percept.delivery import Outbox, SqliteStore, Watch, Writer, store_key
from percept.judges.base import attribute_usage, failed_observations, max_attempts, worst_case_cost
from percept.judges.encode import to_jpeg
from percept.sources import PushSource, open_source
from percept.triggers import Every, QualityGate, SceneMonitor, every, on_change
from percept.zones import crop_box

log = logging.getLogger("percept")

VERIFY, EVENT, STATE, HEARTBEAT, RECHECK = 1, 2, 3, 4, 5      # priorities, highest first
DEADLINES = {EVENT: 3.0, STATE: 10.0}   # seconds a detection may wait before dropped_budget
BUDGET_WINDOW_S = 60.0
COST_WINDOW_S = 3600.0
EVENT_CLIP_S = 4.0          # an event is judged on the pre-roll from this long before its trigger
CLIP_FRAMES = 4
VERIFY_RETRY_S = 1.0        # a failed or weak verification is asked again after this, until its deadline
JUDGE_DOWN_AFTER = 3        # consecutive fully failed requests that open a judge_down blind interval
HEALTH_RING = 256
MAX_BLIND = 1024            # closed blind intervals kept per stream
SNAPSHOT_EVERY_S = 1.0      # a snapshot with no decision in it is written at most this often
STORE_LAG_S = 1.0           # a commit slower than this raises store_lag
STORE_LAG_DEPTH = 64        # so does a writer queue deeper than this
FLUSH_TIMEOUT_S = 10.0      # aclose waits this long for queued commits
KEEP_REQUEST_FRAMES = 8     # recent requests whose frames are kept for keyframes
KEYFRAME_STEPS = ((320, 70), (240, 55), (160, 40))   # (max side, JPEG quality) until <= 40 KB
COVERAGE_BUCKET_S = 60.0
COVERAGE_BUCKETS = 1440


def _spread(frames: tuple[Frame, ...], n: int) -> tuple[Frame, ...]:
    """At most ``n`` frames: the first, the last and evenly spaced ones between (as PreRoll.select)."""
    m = len(frames)
    if m <= n:
        return frames
    return tuple(frames[round(i * (m - 1) / (n - 1))] for i in range(n))


def _keyframe_jpeg(image: np.ndarray) -> Optional[bytes]:
    for side, quality in KEYFRAME_STEPS:
        data = to_jpeg(image, max_side=side, quality=quality)
        if len(data) <= 40_000:
            return data
    return None


class _HealthRing:
    """Bounded, aggregated health: a repeat of an unread (kind, stream, condition, subject)
    raises its count instead of taking a slot. ``counts`` stay exact when entries are dropped."""

    def __init__(self, size: int = HEALTH_RING) -> None:
        self.items: OrderedDict[tuple, Health] = OrderedDict()
        self.size, self.dropped, self.counts = size, 0, Counter()
        self.ready = asyncio.Event()
        self.closed = False

    def put(self, h: Health) -> None:
        self.counts[h.kind] += h.count
        key = (h.kind, h.stream_id, h.condition_id, h.subject)
        old = self.items.get(key)
        if old is not None:
            self.items[key] = replace(h, count=old.count + h.count)
        else:
            if len(self.items) >= self.size:
                self.items.popitem(last=False)
                self.dropped += 1
            self.items[key] = h
        self.ready.set()

    async def drain(self):
        while True:
            self.ready.clear()
            if self.items:
                yield self.items.popitem(last=False)[1]
            elif self.closed:
                return
            else:
                await self.ready.wait()


class _Budget:
    """One stream's spend over sliding windows; see the module docstring."""

    def __init__(self, limit: int, max_cost: Optional[float]) -> None:
        self.limit, self.max_cost = limit, max_cost
        self.held = 0                    # attempts reserved by requests in flight
        self.held_cost = 0.0
        self.spent: deque[tuple[float, int]] = deque()        # (expires, attempts)
        self.spent_cost: deque[tuple[float, float]] = deque()

    def _used(self, now: float) -> tuple[int, float]:
        while self.spent and self.spent[0][0] <= now:
            self.spent.popleft()
        while self.spent_cost and self.spent_cost[0][0] <= now:
            self.spent_cost.popleft()
        return (self.held + sum(a for _, a in self.spent),
                self.held_cost + sum(c for _, c in self.spent_cost))

    def fits(self, attempts: int, cost: float, now: float) -> bool:
        used, used_cost = self._used(now)
        return used + attempts <= self.limit and (self.max_cost is None or used_cost + cost <= self.max_cost)

    def take(self, attempts: int, cost: float) -> None:
        self.held += attempts
        self.held_cost += cost

    def settle(self, attempts: int, cost: float, used: int, spent: float, now: float) -> None:
        self.held -= attempts
        self.held_cost -= cost
        self.spent.append((now + BUDGET_WINDOW_S, used))
        self.spent_cost.append((now + COST_WINDOW_S, spent))

    def next_change(self, now: float) -> Optional[float]:
        return self.spent[0][0] - now if self.spent else None


@dataclass
class _Work:
    """A pending look at one (condition, subject). Detections coalesce per key."""

    priority: int
    condition_id: str
    subject: str
    created: float
    deadline: Optional[float] = None            # monotonic; None slides
    t_trigger: Optional[float] = None           # wall time of the trigger (events: the clip's anchor)
    frame: Optional[Frame] = None               # fixed evidence (the settled look after a scene reset)
    images: Optional[tuple] = None              # evidence given to poke()
    verify: Optional[decide.RequestVerify] = None
    not_before: float = 0.0                     # verify retries
    blocked: bool = False                       # waited on the budget
    gone: bool = False                          # the original evidence is no longer held
    frames: tuple = ()                          # an original-interval verification's frames


class _Stream:
    """Everything the runtime keeps for one stream. Runs on the monitor's loop."""

    def __init__(self, m: Monitor, source: Any) -> None:
        self.m, self.source, self.id = m, source, source.stream_id
        self.outbox = Outbox(m._writer, self.id)
        self.budget = _Budget(m.max_requests_per_min, m.max_cost_per_hour)
        self.own_ring = getattr(source, "preroll", None) is None
        self.ring = PreRoll() if self.own_ring else source.preroll
        self.state: Optional[decide.StreamDecideState] = None
        self.backlog: list = []
        self.epoch, self.src_epoch, self.epoch_map = 0, None, {}
        self.latest: Optional[Frame] = None
        self.qstate = self.sstate = None
        self.tstates: dict[int, Any] = {}
        self.degraded = False
        self.pending: dict[tuple[str, str], _Work] = {}
        self.verifies: dict[tuple[str, str], _Work] = {}
        self.detect_task: Optional[asyncio.Task] = None
        self.verify_task: Optional[asyncio.Task] = None
        self.verify_work: Optional[_Work] = None
        self.granted = False                 # a concurrency slot handed over by another request
        self.dispatcher: Optional[asyncio.Task] = None
        self.timers: dict[tuple[str, str, str], asyncio.TimerHandle] = {}
        self.wake = asyncio.Event()
        self.commits: asyncio.Queue = asyncio.Queue()
        self.dirty = False
        self.req_frames: OrderedDict[str, tuple[Frame, ...]] = OrderedDict()
        self.req_refs: OrderedDict[str, tuple[EvidenceRef, ...]] = OrderedDict()
        self.keyframes: OrderedDict[str, bytes] = OrderedDict()   # occurrence id -> latest keyframe
        self.counter: dict[str, list[CounterExample]] = {}
        self.blind_open: dict[tuple[str, str], float] = {}
        self.blind: deque[BlindInterval] = deque(maxlen=MAX_BLIND)
        self.blind_dropped = 0
        self.fates: deque[tuple[int, Counter]] = deque(maxlen=COVERAGE_BUCKETS)
        self.totals: Counter = Counter()
        self.judge_failures = 0
        self.budget_blocked = False
        self.eof = False
        self.tasks: set[asyncio.Task] = set()
        self.committer = self.spawn(self._commit_loop())
        self.runner = self.spawn(self._run())

    # ----------------------------------------------------------------- plumbing

    def spawn(self, coro) -> asyncio.Task:
        task = asyncio.get_running_loop().create_task(coro)
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)
        return task

    def health(self, kind: str, detail: str = "", condition_id: Optional[str] = None) -> None:
        self.m._health.put(Health(kind, self.id, self.m._wall(), detail, condition_id=condition_id))  # type: ignore[arg-type]

    def blind_set(self, kind: str, on: bool, detail: str = "") -> None:
        key, now = (kind, detail), self.m._wall()
        if on and key not in self.blind_open:
            self.blind_open[key] = now
        elif not on and key in self.blind_open:
            start = self.blind_open.pop(key)
            if len(self.blind) == self.blind.maxlen:
                self.blind_dropped += 1
            self.blind.append(BlindInterval(kind, self.id, start, max(start, now), detail))  # type: ignore[arg-type]

    def fate(self, condition_id: str, fate: str, n: int = 1) -> None:
        bucket = int(self.m._wall() // COVERAGE_BUCKET_S)
        if not self.fates or self.fates[-1][0] != bucket:
            self.fates.append((bucket, Counter()))
        self.fates[-1][1][(condition_id, fate)] += n
        self.totals[fate] += n

    # ----------------------------------------------------------------- capture

    async def _run(self) -> None:
        try:
            await self._restore()
            self.dispatcher = self.spawn(self._dispatch_loop())
            self.spawn(self._tick_loop())
            frames = self.source.frames()
            try:
                async for frame in frames:
                    if self.m._closing:
                        break
                    self.on_frame(frame)
            finally:
                close = getattr(frames, "aclose", None)
                if close is not None:
                    await close()
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - a broken source ends its stream, it doesn't crash the monitor
            log.exception("percept: stream %s stopped", self.id)
            self.health("link_down", f"source failed: {type(exc).__name__}: {exc}"[:200])
        finally:
            self.eof = True
            self.wake.set()

    async def _restore(self) -> None:
        m, wall = self.m, self.m._wall()
        snap = await self.outbox.restore()
        state = None
        if snap is not None:
            try:
                state = decide.from_dict(snap["decide"])
            except (KeyError, TypeError, ValueError):
                self.health("store_lag", "the stored snapshot is unreadable; this stream starts fresh")
        if state is not None and state.stream_id == self.id:
            self.state, self.epoch = state, state.stream_epoch + 1
            self.step(decide.EpochStarted(self.epoch, restart=True))
        else:
            self.state = decide.new_state(self.id, created_at=wall)
        ids = sorted(m._conds)
        for cid, raw in zip(ids, await m._writer.load([store_key("counterexamples", self.id, "/" + c) for c in ids])):
            if raw is not None:
                self.counter[cid] = [CounterExample(e["note"], None if e["jpeg"] is None else bytes.fromhex(e["jpeg"]))
                                     for e in json.loads(raw)]
        for (cid, subject), until in m._snoozes.items():
            if until > wall:
                self.step(decide.Snooze(cid, until, subject))
                self.mute_timer(cid, subject, until)
        backlog, self.backlog = self.backlog, []
        for inp in backlog:
            self.step(inp)

    def on_frame(self, frame: Frame) -> None:
        m = self.m
        self.totals["frames"] += 1
        if frame.stream_epoch != self.src_epoch:
            if self.src_epoch is not None:  # a reconnect
                self.epoch += 1
                self.step(decide.EpochStarted(self.epoch, restart=False))
            self.src_epoch = frame.stream_epoch
            self.epoch_map[frame.stream_epoch] = self.epoch
            while len(self.epoch_map) > 16:
                del self.epoch_map[next(iter(self.epoch_map))]
        self.latest = frame
        if self.own_ring:
            self.ring.add(frame)
        verdict, self.qstate = m._quality.check(frame, self.qstate, frame.t)
        if verdict.reseeded:
            self.health("degraded", "a lasting blur became the new normal (refocused, or a lens out of focus)")
        if verdict.degraded != self.degraded:
            self.degraded = verdict.degraded
            self.blind_set("degraded_video", verdict.degraded)
            if verdict.degraded:
                self.health("degraded", "too many corrupt or blurry frames")
        if not verdict.ok:
            self.totals["frames_vetoed"] += 1
            for cid in m._active():
                self.fate(cid, "dropped_quality")
            return
        signal, self.sstate = m._scene.step(frame, self.sstate, frame.t)
        if signal.reset:
            self.blind_set("scene_reset", True)
            self.health("scene_reset", f"{signal.changed:.0%} of the frame changed at once")
        if signal.suppressed:
            return
        if signal.ended:
            self.blind_set("scene_reset", False)
            self.tstates = {}
        for trigger, conds in m._groups:
            try:
                due, self.tstates[id(trigger)] = trigger.step(frame, conds, self.tstates.get(id(trigger)), frame.t)
            except Exception:  # noqa: BLE001 - a broken custom trigger skips this frame
                log.exception("percept: trigger %r failed", trigger)
                continue
            for cid in due:
                cond = m._conds.get(cid)
                if cond is not None:
                    prio = EVENT if cond.kind == "event" else HEARTBEAT if isinstance(trigger, Every) else STATE
                    self.enqueue(cid, decide.SCENE, prio, t_trigger=frame.t_wall)
        if signal.ended:  # one batched "settled look" on this frame for every condition
            for cid in m._active():
                self.enqueue(cid, decide.SCENE, STATE, frame=frame)

    # ----------------------------------------------------------------- work

    def enqueue(self, cid: str, subject: str, prio: int, *, t_trigger: Optional[float] = None,
                frame: Optional[Frame] = None, images: Optional[tuple] = None) -> None:
        if cid in self.m._paused or cid not in self.m._conds:
            return
        now = self.m._mono()
        deadline = now + DEADLINES[prio] if prio in DEADLINES else None
        w = self.pending.get((cid, subject))
        if w is None:
            self.pending[(cid, subject)] = _Work(prio, cid, subject, now, deadline, t_trigger, frame, images)
        else:  # merged into the look already waiting
            self.fate(cid, "coalesced")
            w.priority = min(w.priority, prio)
            if deadline is not None:
                w.deadline = deadline if w.deadline is None else min(w.deadline, deadline)
            if t_trigger is not None:
                w.t_trigger = t_trigger if w.t_trigger is None else min(w.t_trigger, t_trigger)
            w.frame, w.images = frame or w.frame, images or w.images
        self.wake.set()

    def frames_of(self, interval: tuple[float, float], epoch: int) -> tuple[Frame, ...]:
        """Every pre-roll frame of ``epoch`` inside ``interval``, oldest first."""
        held = self.ring.within(interval) if self.own_ring else self.source.clip(interval, 1024)
        return tuple(f for f in held if self.epoch_map.get(f.stream_epoch) == epoch)

    def original(self, v: decide.RequestVerify) -> tuple[Frame, ...]:
        """The frames a verification re-judges: exactly those of the request that produced the
        interval when there was one, else a spread over every frame still held inside it.
        Empty when the start of the interval is gone."""
        recent = [fs for fs in self.req_frames.values() if fs]
        for frames in reversed(recent):
            if (frames[0].t_wall, frames[-1].t_wall) == tuple(v.interval) \
                    and self.epoch_map.get(frames[0].stream_epoch) == v.stream_epoch:
                return frames
        held = {(f.stream_epoch, f.seq): f for fs in recent for f in fs
                if v.interval[0] <= f.t_wall <= v.interval[1] and self.epoch_map.get(f.stream_epoch) == v.stream_epoch}
        held.update({(f.stream_epoch, f.seq): f for f in self.frames_of(v.interval, v.stream_epoch)})
        frames = sorted(held.values(), key=lambda f: (f.t_wall, f.seq))
        return _spread(tuple(frames), CLIP_FRAMES) if frames and frames[0].t_wall <= v.interval[0] else ()

    def evidence(self, w: _Work) -> Optional[tuple]:
        """(frames, images, evidence_t, crop) for a look, or None when there is nothing to judge."""
        cond = self.m._conds.get(w.condition_id)
        latest = self.latest
        if cond is None:
            return None
        if w.images is not None:
            t = w.t_trigger if w.t_trigger is not None else self.m._wall()
            return (), w.images, (t, t), ("given", id(w.images))
        if w.verify is not None:
            v = w.verify
            if v.mode == "original":
                frames = w.frames
                if not frames:
                    return None  # the original evidence is gone
            elif latest is None or v.stream_epoch != self.epoch:
                return None
            else:
                frames = (latest,)
        elif w.frame is not None:
            frames = (w.frame,)
        elif latest is None:
            return None
        elif cond.kind == "event":
            anchor = latest.t_wall if w.t_trigger is None else min(w.t_trigger, latest.t_wall)
            start = max(anchor - EVENT_CLIP_S, latest.t_wall - self.ring.seconds)
            held = tuple(f for f in self.frames_of((start, latest.t_wall), self.epoch) if f.seq != latest.seq)
            frames = _spread(held + (latest,), CLIP_FRAMES)    # the newest frame is always judged
        else:
            frames = (latest,)
        h, w_ = frames[0].image.shape[:2]
        box = crop_box(cond.zones, w_, h)
        images = tuple(f.image[box[1]:box[3], box[0]:box[2]] for f in frames)
        return frames, images, (frames[0].t_wall, frames[-1].t_wall), (tuple((f.stream_epoch, f.seq) for f in frames), box)

    async def _dispatch_loop(self) -> None:
        while True:
            self.wake.clear()
            try:
                timeout = self.dispatch(self.m._mono())
            except Exception:  # noqa: BLE001 - a bug here must not silently stop the stream
                log.exception("percept: stream %s: dispatch failed", self.id)
                timeout = 1.0
            if self.eof and not self.pending and not self.verifies and self.detect_task is None \
                    and self.verify_task is None:
                break
            try:
                await asyncio.wait_for(self.wake.wait(), timeout)
            except asyncio.TimeoutError:
                pass
        await self.commits.join()
        self.outbox.finish()

    def dispatch(self, now: float) -> float:
        """Start what the budget, the concurrency limit and single flight allow; return how long
        to sleep before looking again (new work or a finished request wakes the loop sooner)."""
        m, wait = self.m, 1.0
        for key, w in list(self.pending.items()):
            if w.deadline is not None and now > w.deadline:
                del self.pending[key]
                self.fate(w.condition_id, "dropped_budget")
            elif self.eof and self.latest is None and w.images is None:
                del self.pending[key]
                self.fate(w.condition_id, "dropped_budget")
        refused = False
        if self.verify_task is None and self.verifies:
            for key in [k for k in self.verifies if k[0] not in m._conds]:
                del self.verifies[key]
            ready = [w for w in self.verifies.values() if w.not_before <= now and not w.gone]
            for w in sorted(ready, key=lambda w: (w.verify.deadline, w.created)):
                why = self.start([w], "verify", now)
                if why == "gone":
                    w.gone = True
                    self.health("verify_deferred", "the original evidence left the pre-roll", w.condition_id)
                    continue
                if why == "budget":
                    refused = True
                    if not w.blocked:
                        w.blocked = True
                        self.health("verify_deferred", "waiting for the request budget", w.condition_id)
                break
            later = [w.not_before - now for w in self.verifies.values() if w.not_before > now]
            wait = min([wait] + later)
        if self.detect_task is None and self.pending:
            group = self.group()
            if group:
                refused |= self.start(group, "detect", now) == "budget"
            deadlines = [w.deadline - now for w in self.pending.values() if w.deadline is not None]
            wait = min([wait] + deadlines)
        if refused != self.budget_blocked:
            self.budget_blocked = refused
            self.blind_set("budget_exhausted", refused)
            if refused:
                self.health("budget_exhausted", f"more than {m.max_requests_per_min} attempts a minute are due")
        if refused and (nxt := self.budget.next_change(now)) is not None:
            wait = min(wait, nxt)
        if self.granted:            # handed a slot it can't use now: pass it on
            self.granted = False
            m._release()
        return max(0.005, wait)

    def group(self) -> list[_Work]:
        """The most urgent look, plus every waiting look with identical evidence."""
        order = sorted(self.pending.values(), key=lambda w: (w.priority, w.created))
        top_ev = self.evidence(order[0])
        if top_ev is None:
            return []
        cap = len(order) if self.m.judge.can_batch else max(1, self.m.max_requests_per_min // self.m._attempts)
        group = [order[0]]
        for w in order[1:]:
            if len(group) >= cap:
                break
            ev = self.evidence(w)
            if ev is not None and ev[3] == top_ev[3]:
                group.append(w)
        return group

    def start(self, works: list[_Work], purpose: str, now: float) -> Optional[str]:
        """Send one request, or say why not: "pool", "budget" or "gone"."""
        m = self.m
        if not self.granted and m._slots >= m.max_concurrent_requests:
            m._waiters[self.id] = self      # first come, first served when a slot frees
            return "pool"
        ev = self.evidence(works[0])
        if ev is None:
            return "gone"
        frames, images, evidence_t, _ = ev
        items = []
        for w in works:
            cond = m._conds[w.condition_id]
            if w.verify is not None:
                if cond.version != w.verify.condition_version:
                    self.verifies.pop((w.condition_id, w.subject), None)
                    return "gone"
                gen = w.verify.occurrence_gen
            else:
                gen = decide.gen_for(self.state, w.condition_id, w.subject)
            items.append(JudgeItem(cond, w.subject, gen, tuple(self.counter.get(w.condition_id, ()))))
        epoch = works[0].verify.stream_epoch if works[0].verify is not None else self.epoch
        request = JudgeRequest(f"{m._incarnation}-{next(m._ids)}", self.id, epoch, tuple(items), images,
                               evidence_t, purpose)  # type: ignore[arg-type]
        calls = 1 if m.judge.can_batch else len(items)
        reserve = m._attempts * calls
        price = getattr(m.judge, "price", None)
        cost = worst_case_cost(request, price, retries=m._attempts - 1) * calls if price is not None else 0.0
        if not self.budget.fits(reserve, cost, now):
            return "budget"
        self.budget.take(reserve, cost)
        if self.granted:
            self.granted = False
        else:
            m._slots += 1
        for w in works:
            if w.verify is None:
                del self.pending[(w.condition_id, w.subject)]
        refs = tuple(EvidenceRef(self.id, self.epoch_map.get(f.stream_epoch, epoch), f.seq, f.t_wall) for f in frames)
        self.remember(self.req_refs, request.request_id, refs, 256)
        self.remember(self.req_frames, request.request_id, frames, KEEP_REQUEST_FRAMES)
        task = self.spawn(self.call(request, works, reserve, cost))
        if purpose == "verify":
            self.verify_task, self.verify_work = task, works[0]
        else:
            self.detect_task = task
        return None

    @staticmethod
    def remember(lru: OrderedDict, key: str, value: Any, size: int) -> None:
        lru[key] = value
        while len(lru) > size:
            lru.popitem(last=False)

    async def call(self, request: JudgeRequest, works: list[_Work], reserve: int, cost: float) -> None:
        m, result = self.m, None
        try:
            result = await m.judge.judge(request)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - a judge that raises broke its contract: an outage
            result = JudgeResult(failed_observations(request, "unavailable", m.judge.name, f"judge raised {exc!r}"[:160]), ())
        finally:  # a cancelled request keeps its whole reservation charged
            used = reserve if result is None else sum(u.usage.attempts for u in result.usage)
            spent = cost if result is None else sum(u.usage.cost_usd for u in result.usage)
            self.budget.settle(reserve, cost, used, spent, m._mono())
            self.totals["requests"] += 1
            self.totals["attempts"] += used
            self.totals["cost_usd_micros"] += round(spent * 1e6)
            if request.purpose == "verify":
                self.verify_task = None
            else:
                self.detect_task = None
            m._release()
        self.absorb(request, result, works)

    def absorb(self, request: JudgeRequest, result: JudgeResult, works: list[_Work]) -> None:
        failed = [u for u in result.usage if u.failure is not None]
        for u in failed:
            self.health("judge_unavailable", f"{u.judge}: {u.failure}")
        if not result.usage and any(o.failed for o in result.observations):
            self.health("judge_unavailable", result.observations[0].reason)
        if result.observations and all(o.failed for o in result.observations):
            self.judge_failures += 1
        else:
            self.judge_failures = 0
        self.blind_set("judge_down", self.judge_failures >= JUDGE_DOWN_AFTER)
        for ob, share in zip(result.observations, attribute_usage(result)):
            if ob.stream_id == self.id:
                self.step(decide.Observed(ob, share))
        if request.purpose == "verify":
            w = works[0]
            key = (w.condition_id, w.subject)
            k = self.state.keys.get(key)
            if self.verifies.get(key) is w:
                if k is not None and k.phase == decide.VERIFYING and k.gen == w.verify.occurrence_gen:
                    w.not_before = self.m._mono() + VERIFY_RETRY_S   # no answer yet: ask again
                else:
                    del self.verifies[key]
        self.wake.set()

    # ----------------------------------------------------------------- decide

    def step(self, inp: Any) -> None:
        if self.state is None:
            self.backlog.append(inp)
            return
        r = decide.step(self.state, inp, self.m._wall(), self.m._conds)
        self.state = r.state
        loop = asyncio.get_running_loop()
        for e in r.effects:
            t = type(e)
            if t is decide.Trace:
                if e.fate is not None:
                    self.fate(e.condition_id, e.fate)
            elif t is decide.ScheduleTimer:
                key = (e.condition_id, e.subject, e.kind)
                old = self.timers.pop(key, None)
                if old is not None:
                    old.cancel()
                self.timers[key] = loop.call_later(max(0.0, e.at - self.m._wall()), self.fire, key)
            elif t is decide.CancelTimer:
                old = self.timers.pop((e.condition_id, e.subject, e.kind), None)
                if old is not None:
                    old.cancel()
            elif t is decide.RequestVerify:
                w = _Work(VERIFY, e.condition_id, e.subject, self.m._mono(), verify=e)
                if e.mode == "original":    # pinned now: frames are ephemeral
                    w.frames = self.original(e)
                self.verifies[(e.condition_id, e.subject)] = w
            elif t is decide.CancelVerify:
                w = self.verifies.get((e.condition_id, e.subject))
                if w is not None and w.verify.occurrence_gen == e.occurrence_gen:
                    del self.verifies[(e.condition_id, e.subject)]
                    if w.blocked:
                        self.fate(e.condition_id, "dropped_budget")
                    if self.verify_task is not None and self.verify_work is w:
                        self.verify_task.cancel()
            elif t is decide.RequestRecheck:
                self.enqueue(e.condition_id, e.subject, RECHECK)
            elif t is decide.EmitHealth:
                self.m._health.put(e.health)
            elif t is decide.StoreCounterExample:
                self.spawn(self.store_counter(e))
        if r.decisions:
            self.commits.put_nowait((json.dumps({"decide": decide.to_dict(r.state)}).encode(),
                                     tuple((d, *self.evidence_for(d)) for d in r.decisions)))
            self.dirty = False
        else:
            self.dirty = True

    def flush_snapshot(self) -> None:
        if self.dirty and self.state is not None:
            self.dirty = False
            self.commits.put_nowait((json.dumps({"decide": decide.to_dict(self.state)}).encode(), ()))

    def fire(self, key: tuple[str, str, str]) -> None:
        self.timers.pop(key, None)
        if not self.m._closing:
            self.step(decide.TimerFired(*key))

    def mute_timer(self, cid: str, subject: Optional[str], until: float) -> None:
        handle = asyncio.get_running_loop().call_later(max(0.0, until - self.m._wall()), self.mute_ended, cid, subject)
        self.m._handles.append(handle)

    def mute_ended(self, cid: str, subject: Optional[str]) -> None:
        """A snooze ended: look again at every held state it may have deferred a met for (D2)."""
        if self.m._closing or self.state is None or self.m._snoozes.get((cid, subject), 0.0) > self.m._wall():
            return
        for (kcid, ksubject), k in self.state.keys.items():
            if kcid == cid and subject in (None, ksubject) and k.met_pending and k.phase == decide.HOLDING:
                self.enqueue(cid, ksubject, STATE)

    def evidence_for(self, d: Decision) -> tuple[tuple[EvidenceRef, ...], Optional[np.ndarray]]:
        """Evidence refs and the keyframe image for a decision, chosen now (frames are ephemeral)."""
        refs: list[EvidenceRef] = []
        image = None
        for rid in reversed(d.request_ids):
            base = rid.split("/")[0]
            refs += [r for r in self.req_refs.get(base, ()) if r not in refs]
            frames = self.req_frames.get(base)
            if image is None and frames:
                image = min(frames, key=lambda f: abs(f.t_wall - d.t_observed)).image
        if image is None:
            interval = (min(d.t_observed, d.t_decided) - EVENT_CLIP_S, max(d.t_observed, d.t_decided))
            f = (self.ring.keyframe(interval, d.t_observed) if self.own_ring else self.source.keyframe(interval, d.t_observed))
            f = f or self.latest
            image = None if f is None else f.image
        return tuple(sorted(refs, key=lambda r: (r.t_wall, r.seq))[-8:]), image

    async def store_counter(self, e: decide.StoreCounterExample) -> None:
        examples = (self.counter.get(e.condition_id, []) + [CounterExample(e.note, self.keyframes.get(e.occurrence_id))])[-3:]
        self.counter[e.condition_id] = examples
        data = json.dumps([{"note": c.note, "jpeg": None if c.thumbnail_jpeg is None else c.thumbnail_jpeg.hex()}
                           for c in examples]).encode()
        await self.m._writer.commit({store_key("counterexamples", self.id, "/" + e.condition_id): data})

    # ----------------------------------------------------------------- commits

    async def _commit_loop(self) -> None:
        while True:
            snapshot, decisions = await self.commits.get()
            try:
                out = []
                for d, refs, image in decisions:
                    jpeg = None if image is None else await asyncio.to_thread(_keyframe_jpeg, image)
                    if jpeg is not None:
                        self.remember(self.keyframes, d.occurrence_id, jpeg, 64)
                    out.append(replace(d, evidence=refs, keyframe=jpeg))
                t0 = self.m._mono()
                await self.outbox.commit(snapshot, tuple(out))
                if self.m._mono() - t0 > STORE_LAG_S:
                    self.health("store_lag", f"a commit took {self.m._mono() - t0:.1f} s")
            except Exception:  # noqa: BLE001 - only when the writer gave up at shutdown
                log.exception("percept: stream %s: a commit was abandoned", self.id)
            finally:
                self.commits.task_done()

    async def _tick_loop(self) -> None:
        failures = 0
        while True:
            await asyncio.sleep(SNAPSHOT_EVERY_S)
            if self.dirty and self.commits.empty():
                self.flush_snapshot()
            w = self.m._writer
            if w.pending > STORE_LAG_DEPTH or w.failures > failures:
                self.health("store_lag", f"{w.pending} store writes waiting, {w.failures} failed")
            failures = w.failures

    # ----------------------------------------------------------------- reports

    def coverage(self, since_wall: float) -> dict:
        fates: dict[str, Counter] = {}
        first = int(since_wall // COVERAGE_BUCKET_S)
        for bucket, counts in self.fates:
            if bucket >= first:
                for (cid, fate), n in counts.items():
                    fates.setdefault(cid, Counter())[fate] += n
        now = self.m._wall()
        blind = [b for b in self.blind if b.t_end_wall is None or b.t_end_wall >= since_wall]
        blind += [BlindInterval(kind, self.id, start, None, detail)  # type: ignore[arg-type]
                  for (kind, detail), start in self.blind_open.items()]
        return {"stream_id": self.id, "since_wall": since_wall, "until_wall": now,
                "fates": {cid: dict(c) for cid, c in sorted(fates.items())},
                "blind_intervals": sorted(blind, key=lambda b: b.t_start_wall),
                "blind_intervals_dropped": self.blind_dropped}

    def stats(self) -> dict:
        t, ob = self.totals, self.outbox
        counts = {k: t[k] for k in ("frames", "frames_vetoed", "requests", "attempts")}
        fates = {k: v for k, v in t.items() if k not in counts and k != "cost_usd_micros"}
        return dict(counts, cost_usd=t["cost_usd_micros"] / 1e6, fates=fates, decisions=ob.committed,
                    delivered=ob.delivered, redelivered=ob.redelivered, acked=ob.acked, unacked=len(ob.entries),
                    pending_looks=len(self.pending), pending_verifications=len(self.verifies))

    def stop(self) -> list[asyncio.Task]:
        """Cancel capture, dispatch and in-flight judging (the committer is drained separately)."""
        for handle in self.timers.values():
            handle.cancel()
        self.timers.clear()
        tasks = [t for t in self.tasks if not t.done() and t is not self.committer]
        for task in tasks:
            task.cancel()
        return tasks


class Monitor:
    """Watches streams for conditions and delivers one decision per real occurrence.

    ``judge`` is any :class:`~percept.contracts.Judge` (``percept.judges.cascade(...)`` to ask
    cheaper judges first). ``store`` defaults to ``SqliteStore("percept.db")`` in the working directory;
    ``percept.MemoryStore()`` is the explicit, not crash-safe alternative.
    ``max_requests_per_min`` is a hard per-stream limit on outbound attempts over any 60 s;
    ``max_cost_per_hour`` (per stream) is reserved before each request and reconciled after,
    guaranteed for judges with a ``price`` and best-effort otherwise;
    ``max_concurrent_requests`` limits requests in flight across every stream.

    Configuration errors raise here, in :meth:`add` and in :meth:`watch`. Nothing raises
    during operation: outages, failures, budget exhaustion and store lag are reported by
    :meth:`health` and :meth:`coverage`. Methods other than :meth:`push` are called from the
    monitor's event loop thread.
    """

    def __init__(self, judge: Judge, conditions: Iterable[Condition] = (), *, store: Any = None,
                 max_requests_per_min: int = 6, max_cost_per_hour: Optional[float] = None,
                 max_concurrent_requests: int = 4) -> None:
        if not isinstance(judge, Judge):
            raise TypeError("judge must implement the Judge protocol (name, can_batch, price, async judge)")
        for name, value in (("max_requests_per_min", max_requests_per_min),
                            ("max_concurrent_requests", max_concurrent_requests)):
            if not (isinstance(value, int) and not isinstance(value, bool) and value >= 1):
                raise ValueError(f"{name} is an integer >= 1")
        if max_cost_per_hour is not None and not max_cost_per_hour > 0:
            raise ValueError("max_cost_per_hour is positive or None")
        self.judge = judge
        self._attempts = max_attempts(judge)
        if max_requests_per_min < self._attempts:
            raise ValueError(f"one judge call may make {self._attempts} attempts; "
                             f"max_requests_per_min must be at least that")
        self.max_requests_per_min, self.max_cost_per_hour = max_requests_per_min, max_cost_per_hour
        self.max_concurrent_requests = max_concurrent_requests
        self._owns_store = store is None
        self.store = SqliteStore("percept.db") if store is None else store
        if not (callable(getattr(self.store, "load", None)) and callable(getattr(self.store, "commit", None))):
            raise TypeError("store must implement load(key) and commit(writes)")
        self._writer = Writer(self.store)
        self._conds: dict[str, Condition] = {}
        for cond in conditions:
            self._check(cond)
            if cond.id in self._conds:
                raise ValueError(f"duplicate condition id {cond.id!r}")
            self._conds[cond.id] = cond
        self._paused: set[str] = set()
        self._snoozes: dict[tuple[str, Optional[str]], float] = {}
        self._streams: dict[str, _Stream] = {}
        self._health = _HealthRing()
        self._handles: list[asyncio.TimerHandle] = []
        self._slots = 0
        self._waiters: OrderedDict[str, _Stream] = OrderedDict()
        self._quality, self._scene = QualityGate(), SceneMonitor()
        self._default_change, self._default_heartbeat = on_change(), every(300)
        self._wall, self._mono = time.time, time.monotonic
        self._ids = itertools.count(1)
        self._incarnation = format(int(time.time() * 1000), "x")
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._closing = self._closed = False
        self._regroup()

    # ----------------------------------------------------------------- internals

    def _check(self, cond: Condition) -> None:
        if not isinstance(cond, Condition):
            raise TypeError(f"expected a Condition, got {type(cond).__name__}")
        if cond.judge is not None:
            routes = {j.name for j in getattr(self.judge, "members", ())}
            if cond.judge not in routes:
                raise ValueError(f"condition {cond.id!r}: unknown judge route {cond.judge!r}")

    def _active(self) -> list[str]:
        return [cid for cid in self._conds if cid not in self._paused]

    def _regroup(self) -> None:
        groups: dict[int, tuple[Any, list[Condition]]] = {}
        for cond in self._conds.values():
            if cond.id in self._paused:
                continue
            if cond.trigger is not None:
                triggers: tuple = (cond.trigger,)
            elif cond.kind == "event":
                triggers = (self._default_change,)
            else:
                triggers = (self._default_change, self._default_heartbeat)
            for trigger in triggers:
                groups.setdefault(id(trigger), (trigger, []))[1].append(cond)
        self._groups = [(t, tuple(cs)) for t, cs in groups.values()]

    def _bind(self) -> None:
        loop = asyncio.get_running_loop()
        if self._loop is None:
            self._loop = loop
        elif self._loop is not loop:
            raise RuntimeError("a Monitor runs on one event loop")
        if self._closing:
            raise RuntimeError("the monitor is closed")

    def _on_loop(self) -> None:
        if self._loop is None:
            return
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        if running is not self._loop:
            raise RuntimeError("call Monitor methods from its event loop thread")

    def _release(self) -> None:
        """A request finished: hand its slot to the stream that has waited longest, so no
        stream can starve another by taking the slot straight back."""
        while self._waiters:
            _, stream = self._waiters.popitem(last=False)
            if stream.dispatcher is not None and not stream.dispatcher.done():
                stream.granted = True
                stream.wake.set()
                return
        self._slots -= 1

    def _source_health(self, h: Health) -> None:
        self._health.put(h)
        stream = self._streams.get(h.stream_id)
        if stream is not None and h.kind in ("link_down", "link_up"):
            stream.blind_set("link_down", h.kind == "link_down")

    def _stream_for(self, occurrence_id: str) -> Optional[_Stream]:
        matches = [s for sid, s in self._streams.items() if occurrence_id.startswith(sid + "/")]
        return max(matches, key=lambda s: len(s.id), default=None)

    def _each(self, inp: Any) -> None:
        for stream in self._streams.values():
            stream.step(inp)

    # ----------------------------------------------------------------- streams

    def watch(self, source: Union[str, Source], *, stream_id: Optional[str] = None, ack: str = "auto",
              max_unacked: int = 100) -> Watch:
        """Decisions for one stream, in order, as ``Delivery`` envelopes (``async for``).

        ``source`` is a URI (see ``percept.open_source``), a ``Source``, or the id of a stream
        this monitor already has (pushed to, or watched before). Watching a stream again takes
        it over: every unacknowledged decision is delivered again. Ends at ``aclose()``, or
        once a source that ends (a file, a closed ``PushSource``) is fully judged and delivered.
        """
        self._bind()
        if isinstance(source, str):
            stream = self._streams.get(stream_id or source)
            if stream is None:
                opened = open_source(source, stream_id=stream_id, on_health=self._source_health, preroll=PreRoll())
                stream = self._streams.get(opened.stream_id) or self._add(opened)
        elif isinstance(source, Source):
            if stream_id is not None and stream_id != source.stream_id:
                raise ValueError("stream_id must match source.stream_id")
            stream = self._streams.get(source.stream_id)
            if stream is not None and stream.source is not source:
                raise ValueError(f"stream {source.stream_id!r} is already watched from another source")
            stream = stream or self._add(source)
        else:
            raise TypeError("watch() takes a URI or a Source")
        return Watch(stream.outbox, ack, max_unacked,
                     lambda n: stream.health("store_lag", f"delivery paused: {n} decisions unacknowledged"))

    def _add(self, source: Any) -> _Stream:
        if getattr(source, "preroll", 0) is None:
            source.preroll = PreRoll()  # sources that accept one are fed at capture rate
        stream = self._streams[source.stream_id] = _Stream(self, source)
        return stream

    def push(self, stream_id: str, image: np.ndarray, t_wall: Optional[float] = None) -> None:
        """Feed a frame from app-owned capture. A new ``stream_id`` starts a stream (on the
        loop thread); frames for an existing push stream may come from any thread."""
        stream = self._streams.get(stream_id)
        if stream is None:
            self._bind()
            stream = self._add(PushSource(stream_id, preroll=PreRoll()))
        if not isinstance(stream.source, PushSource):
            raise ValueError(f"stream {stream_id!r} is not a push stream")
        stream.source.push(image, t_wall)

    def poke(self, stream_id: str, condition_ids: Union[str, Sequence[str]], *, subject: Optional[str] = None,
             observed_at: Optional[float] = None, reason: str, evidence: Any = None) -> None:
        """Ask for a look now, from your own detector or tracker (the same path as a trigger).

        ``observed_at`` (wall time) anchors an event's clip; ``evidence`` (an image or a
        sequence of images) is judged instead of the stream's frames.
        """
        self._on_loop()
        stream = self._streams.get(stream_id)
        if stream is None:
            raise ValueError(f"unknown stream {stream_id!r}")
        ids = [condition_ids] if isinstance(condition_ids, str) else list(condition_ids)
        unknown = [c for c in ids if c not in self._conds]
        if unknown:
            raise ValueError(f"unknown condition ids {unknown}")
        images = None
        if evidence is not None:
            images = tuple(np.asarray(i) for i in ((evidence,) if isinstance(evidence, np.ndarray) else evidence))
            if not images or any(i.dtype != np.uint8 for i in images):
                raise ValueError("poke evidence is one or more uint8 images")
        for cid in ids:
            prio = EVENT if self._conds[cid].kind == "event" else STATE
            stream.enqueue(cid, subject or decide.SCENE, prio, t_trigger=observed_at, images=images)

    # ----------------------------------------------------------------- conditions and feedback

    def add(self, condition: Condition) -> None:
        """Add a condition, or replace one with the same id (its version is bumped if needed)."""
        self._on_loop()
        self._check(condition)
        old = self._conds.get(condition.id)
        if old == condition:
            return
        if old is not None and condition.version <= old.version:
            condition = replace(condition, version=old.version + 1)
        self._conds[condition.id] = condition
        self._regroup()
        self._each(decide.ConditionChanged(condition.id))

    def remove(self, condition_id: str) -> None:
        self._on_loop()
        if self._conds.pop(condition_id, None) is None:
            return
        self._paused.discard(condition_id)
        self._regroup()
        for stream in self._streams.values():
            stream.pending = {k: w for k, w in stream.pending.items() if k[0] != condition_id}
            stream.blind_set("paused", False, condition_id)
            stream.step(decide.ConditionChanged(condition_id))

    def pause(self, condition_id: str) -> None:
        """Stop judging a condition (verifications already requested still finish). Every stream
        records a ``paused`` blind interval until :meth:`resume`."""
        self._on_loop()
        if condition_id not in self._conds:
            raise ValueError(f"unknown condition {condition_id!r}")
        self._paused.add(condition_id)
        self._regroup()
        for stream in self._streams.values():
            stream.pending = {k: w for k, w in stream.pending.items() if k[0] != condition_id}
            stream.blind_set("paused", True, condition_id)

    def resume(self, condition_id: str) -> None:
        self._on_loop()
        if condition_id not in self._paused:
            return
        self._paused.discard(condition_id)
        self._regroup()
        for stream in self._streams.values():
            stream.blind_set("paused", False, condition_id)
            stream.enqueue(condition_id, decide.SCENE, HEARTBEAT)   # one look at what was missed

    def snooze(self, condition_id: str, until_wall: float, subject: Optional[str] = None) -> None:
        """Mute met and uncertain until ``until_wall``; judging continues. A state that is still
        true when the mute ends is looked at again, and its deferred met is delivered then."""
        self._on_loop()
        if condition_id not in self._conds:
            raise ValueError(f"unknown condition {condition_id!r}")
        self._snoozes[(condition_id, subject)] = until_wall
        for stream in self._streams.values():
            stream.step(decide.Snooze(condition_id, until_wall, subject))
            if stream.state is not None:
                stream.mute_timer(condition_id, subject, until_wall)

    def resolve(self, occurrence_id: str) -> None:
        """The application says this occurrence is over."""
        self._on_loop()
        stream = self._stream_for(occurrence_id)
        if stream is not None:
            stream.step(decide.Resolve(occurrence_id))

    def feedback(self, occurrence_id: str, label: str, note: Optional[str] = None) -> None:
        """Label an occurrence. ``"false_alarm"`` closes it and keeps it as a counter-example."""
        self._on_loop()
        if label != "false_alarm":
            raise ValueError("the only feedback label is 'false_alarm'")
        stream = self._stream_for(occurrence_id)
        if stream is not None:
            stream.step(decide.FalseAlarm(occurrence_id, note or ""))

    # ----------------------------------------------------------------- reports

    def health(self):
        """Operational events, aggregated, from a bounded ring (oldest first). Ends at ``aclose()``."""
        return self._health.drain()

    def stats(self) -> dict:
        """Exact counters, per stream and for health, available at any time."""
        return {"streams": {sid: s.stats() for sid, s in self._streams.items()},
                "health": dict(self._health.counts), "health_dropped": self._health.dropped,
                "store_pending": self._writer.pending, "store_failures": self._writer.failures}

    def coverage(self, stream_id: str, since_wall: float = 0.0) -> dict:
        """Fates per condition (counted per minute, from the minute holding ``since_wall``) and
        the blind intervals that overlap it: what was evaluated, what wasn't, and why."""
        stream = self._streams.get(stream_id)
        if stream is None:
            raise ValueError(f"unknown stream {stream_id!r}")
        return stream.coverage(since_wall)

    # ----------------------------------------------------------------- shutdown

    async def aclose(self) -> None:
        """Stop frames, cancel judge calls, flush the store, then close the sources."""
        if self._closed or self._closing:
            return
        self._closing = True
        streams = list(self._streams.values())
        for handle in self._handles:
            handle.cancel()
        stopping = [t for s in streams for t in s.stop()]
        await asyncio.gather(*stopping, return_exceptions=True)
        for s in streams:
            s.flush_snapshot()       # state changes since the last snapshot (no decision in them)
        drains = [s.commits.join() for s in streams]
        try:
            await asyncio.wait_for(asyncio.gather(*drains), FLUSH_TIMEOUT_S)
        except asyncio.TimeoutError:
            log.warning("percept: store flush timed out; unwritten decisions are lost")
        rest = [t for s in streams for t in s.tasks if not t.done()]
        for task in rest:
            task.cancel()
        await asyncio.gather(*rest, return_exceptions=True)
        for s in streams:
            s.outbox.close()
        await self._writer.close(FLUSH_TIMEOUT_S)
        for s in streams:
            close = getattr(s.source, "aclose", None)
            if close is not None:
                try:
                    await asyncio.wait_for(close(), FLUSH_TIMEOUT_S)
                except Exception:  # noqa: BLE001 - shutdown goes on
                    log.exception("percept: closing stream %s failed", s.id)
        if self._owns_store:
            self.store.close()
        self._health.closed = True
        self._health.ready.set()
        self._closed = True

    async def __aenter__(self) -> Monitor:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()
