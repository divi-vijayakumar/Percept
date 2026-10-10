"""The runtime: ``Monitor`` watches streams, decides when to look, judges within a budget, feeds
the answers to the reducer and delivers its decisions durably.

One event loop owns all decision state; nothing here takes a lock. Per stream there is one
source (with its own capture thread), a runner task (frame -> quality -> scene -> triggers ->
due work), a dispatcher task (budget, priority, single flight), a committer task (keyframes,
then one atomic outbox write per emitting step) and one decide state. Shared across streams:
the judge, limited by ``max_concurrent_requests``, and the store's writer thread.

Budget. A request reserves ``max_attempts_for(judge, request)`` attempts (per item for a judge
that can't batch; a cascade counts only the members the request can reach) against its stream's
``max_requests_per_min`` before it is sent and holds them until it finishes; then its actual
attempts stay charged for the window. A request is admitted only while everything charged in that
way plus its reservation fits, so no 60 s span ever holds more outbound attempts than the limit
(for judges that report their attempts truthfully). Dollars work the same way over an hour, on one
ledger for the whole monitor: every request reserves its worst case (``worst_case_cost_for``:
every attempt of every member it can reach) when the judge has a price; without one, only the
reported cost counts (best effort). A cancelled request reports no usage, so it stays charged at
its whole reservation for the window. Verification uses the same budgets and the same concurrency
limit: its own queue, not a bypass.

Reporting is separate from that safety ledger: ``stats()`` counts ``cost_usd`` as the cost the
judge reported (settled; a cancelled request settles at 0) and ``cost_reserved_usd`` as the
dollars still held (requests in flight, and cancelled requests' reservations in their window).
Each request's settled usage is also split evenly (``Usage.split``) among the conditions it
carried, for the per-condition totals.

Priority when constrained: verify, event detect, state detect, heartbeat, recheck. Event and
state detections wait at most ``DEADLINES`` seconds, then are dropped with fate
``dropped_budget``; heartbeats and rechecks wait for the next free slot.

Frames are processed at most ``fps`` per second (default 5, on the stream clock); capture is
never slowed, the pre-roll still sees every frame, and skipped frames are counted in coverage. A
look that needs the current frame waits while the stream has no recent frame (an open
``link_down`` interval, or no frame for ``STALE_FRAME_S``) rather than judging an old one as
current; such waits are counted as ``evidence_unavailable``.

Scene return (decide's ``SceneReturn`` input, cause ``"scene_return"``) is reserved in 1.0: no
trigger reports it, so the runtime never feeds it.

Time: decide sees wall-clock time; timers run on the loop's monotonic clock, converted from
decide's wall-clock deadlines when they are scheduled (a timer that fires early is rescheduled
by decide). Triggers see each frame's stream clock ``Frame.t``.
"""

from __future__ import annotations

import asyncio
import itertools
import json
import logging
import math
import os
import time
from collections import Counter, OrderedDict, deque
from dataclasses import asdict, dataclass, replace
from typing import Any, Callable, Iterable, Optional, Sequence, Union

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
    Usage,
)
from percept.delivery import Outbox, OutboxUnreadable, SqliteStore, Watch, Writer, store_key
from percept.judges.base import (
    attribute_usage,
    failed_observations,
    item_usage,
    max_attempts,
    max_attempts_for,
    worst_case_cost_for,
)
from percept.judges.encode import to_jpeg
from percept.replay import decision_record, encode_condition, encode_input
from percept.sources import PushSource, open_source
from percept.trace import FORMAT as TRACE_FORMAT, TraceWriter, scrub
from percept.triggers import Every, QualityGate, SceneMonitor, every, on_change
from percept.zones import crop_box

log = logging.getLogger("percept")

VERIFY, EVENT, STATE, HEARTBEAT, RECHECK = 1, 2, 3, 4, 5      # priorities, highest first
_PRIORITY_NAMES = {VERIFY: "verify", EVENT: "event", STATE: "state", HEARTBEAT: "heartbeat", RECHECK: "recheck"}
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
DEFAULT_FPS = 5.0           # frames processed per second, per stream, unless watch(fps=...) says otherwise
STALE_FRAME_S = 10.0        # a frame older than this is not judged as the current one
MIN_OUTPUT_TOKENS = 400     # a request's reply limit: at least this, more for more conditions ...
OUTPUT_TOKENS_BASE = 64     # ... the reply's frame ({"results": [...]}) ...
OUTPUT_TOKENS_PER_ITEM = 80 # ... plus one full answer each (id, answer, confidence, 160-char reason) ...
MAX_OUTPUT_TOKENS = 4096    # ... up to this ceiling


def _max_output_tokens(n_items: int, id_chars: int = 0) -> int:
    """The reply limit for a request judging ``n_items`` conditions whose ids total ``id_chars``
    characters: room for every schema-conformant answer (counted at 3 characters a token, like
    ``judges.base.worst_case_tokens``), so a batched reply is not cut off. The cost reservation
    reads the same number from the request."""
    want = OUTPUT_TOKENS_BASE + n_items * OUTPUT_TOKENS_PER_ITEM + math.ceil(id_chars / 3)
    return max(MIN_OUTPUT_TOKENS, min(MAX_OUTPUT_TOKENS, want))


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


class _Window:
    """Spend over a sliding window (see the module docstring): what requests in flight reserved,
    plus what finished requests actually used, each kept for ``window()`` seconds after it ended.
    A ``limit`` of None admits everything but still keeps the account."""

    def __init__(self, limit: Optional[float], window: Callable[[], float]) -> None:
        self.limit, self.window = limit, window
        self.held: float = 0
        self.spent: deque[tuple[float, float]] = deque()        # (expires, amount)

    def used(self, now: float) -> float:
        while self.spent and self.spent[0][0] <= now:
            self.spent.popleft()
        return self.held + sum(a for _, a in self.spent)

    def fits(self, amount: float, now: float) -> bool:
        return self.limit is None or self.used(now) + amount <= self.limit

    def take(self, amount: float) -> None:
        self.held += amount

    def settle(self, reserved: float, actual: float, now: float) -> None:
        self.held -= reserved
        self.spent.append((now + self.window(), actual))

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
    starved: bool = False                       # counted once as evidence_unavailable
    frames: tuple = ()                          # an original-interval verification's frames


class _Stream:
    """Everything the runtime keeps for one stream. Runs on the monitor's loop."""

    def __init__(self, m: Monitor, source: Any, fps: Optional[float]) -> None:
        self.m, self.source, self.id = m, source, source.stream_id
        self.outbox = Outbox(m._writer, self.id, lambda detail: self.health("store_lag", detail))
        self.budget = _Window(m.max_requests_per_min, lambda: BUDGET_WINDOW_S)
        self.held_cost = _Window(None, lambda: COST_WINDOW_S)   # this stream's share of m._cost's reservations
        self.cost_usd = 0.0                                     # settled: what the judge reported
        self.per_condition: dict[str, list] = {}                # id -> [requests, attempts, cost_usd]
        self.period, self.due = (1.0 / fps if fps else 0.0), None
        self.latest_mono = self.latest_rx = 0.0
        self.halted = False
        self.own_ring = getattr(source, "preroll", None) is None
        self.ring = PreRoll() if self.own_ring else source.preroll
        self.state: Optional[decide.StreamDecideState] = None
        self.backlog: list = []
        self.epoch, self.src_epoch, self.epoch_map = 0, None, {}
        self.latest: Optional[Frame] = None
        self.qstate = self.sstate = None
        self.tstates: dict[Any, Any] = {}       # trigger key (see Monitor._trigger_key) -> its state
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
        self.blind_open: dict[tuple[str, Optional[str]], tuple[float, str]] = {}
        self.blind: deque[BlindInterval] = deque(maxlen=MAX_BLIND)
        self.blind_dropped = 0
        self.fates: deque[tuple[int, Counter]] = deque(maxlen=COVERAGE_BUCKETS)
        self.totals: Counter = Counter()
        self.judge_failures = 0
        self.budget_blocked = False
        self.eof = False
        self.traced_versions: Optional[dict] = None     # the condition set last written to the trace
        self.traced_fates: Counter = Counter()          # runtime fates of this minute, not yet traced
        self.traced_bucket: Optional[int] = None
        self.tasks: set[asyncio.Task] = set()
        for cid in m._paused:           # paused before this stream existed
            self.blind_set("paused", True, "paused by the application", cid)
        self.committer = self.spawn(self._commit_loop())
        self.spawn(self._tick_loop())   # reports store trouble from the start, restore included
        self.runner = self.spawn(self._run())

    # ----------------------------------------------------------------- plumbing

    def spawn(self, coro) -> asyncio.Task:
        task = asyncio.get_running_loop().create_task(coro)
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)
        return task

    def health(self, kind: str, detail: str = "", condition_id: Optional[str] = None) -> None:
        self.m._report(Health(kind, self.id, self.m._wall(), detail, condition_id=condition_id))  # type: ignore[arg-type]

    def blind_set(self, kind: str, on: bool, detail: str = "", condition_id: Optional[str] = None) -> None:
        key, now = (kind, condition_id), self.m._wall()
        if on and key not in self.blind_open:
            self.blind_open[key] = (now, detail)
            self.m._rec("blind", self.id, blind_kind=kind, state="open", detail=detail, condition_id=condition_id)
        elif not on and key in self.blind_open:
            start, detail = self.blind_open.pop(key)
            self.m._rec("blind", self.id, blind_kind=kind, state="close", detail=detail, condition_id=condition_id)
            if len(self.blind) == self.blind.maxlen:
                self.blind_dropped += 1
            self.blind.append(BlindInterval(kind, self.id, start, max(start, now), detail,  # type: ignore[arg-type]
                                            condition_id))

    def fate(self, condition_id: str, fate: str, n: int = 1, *, record: bool = True) -> None:
        bucket = int(self.m._wall() // COVERAGE_BUCKET_S)
        if record and self.m._trace is not None:    # one record per minute, condition and fate
            if bucket != self.traced_bucket:
                self.trace_fates()
                self.traced_bucket = bucket
            self.traced_fates[(condition_id, fate)] += n
        if not self.fates or self.fates[-1][0] != bucket:
            self.fates.append((bucket, Counter()))
        self.fates[-1][1][(condition_id, fate)] += n
        self.totals[fate] += n

    def trace_fates(self) -> None:
        """Write the runtime fates counted in the current minute."""
        for (cid, fate), n in sorted(self.traced_fates.items()):
            self.m._rec("fate", self.id, event="runtime", fate=fate, condition_id=cid, n=n,
                        bucket_wall=(self.traced_bucket or 0) * COVERAGE_BUCKET_S)
        self.traced_fates.clear()

    def checkpoint_records(self) -> list[dict]:
        """This stream's restore and conditions records, to open a new trace file."""
        if self.state is None:
            return []
        m = self.m
        self.traced_versions = {cid: c.version for cid, c in m._conds.items()}
        return [m._record("restore", self.id, restored=False, checkpoint=True, snapshot=decide.to_dict(self.state)),
                m._record("conditions", self.id, conditions=[encode_condition(c) for c in m._conds.values()])]

    # ----------------------------------------------------------------- capture

    async def _run(self) -> None:
        try:
            try:
                await self._restore()
            except OutboxUnreadable as exc:
                self.halted = True      # never renumber or overwrite: report until someone looks
                self.health("store_lag", f"{exc}; this stream is halted and nothing was overwritten")
                return
            self.dispatcher = self.spawn(self._dispatch_loop())
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
            self.health("link_down", f"source failed: {type(exc).__name__}: {scrub(str(exc))[:80]}")
        finally:
            self.eof = True
            self.wake.set()
            if self.dispatcher is None:     # never started: let watch() end instead of hanging
                self.outbox.finish()

    async def _restore(self) -> None:
        m, wall = self.m, self.m._wall()
        failures = m._writer.failures
        raw = await self.outbox.restore()
        if m._writer.failures > failures:
            self.health("store_lag", f"{m._writer.failures - failures} store calls failed while restoring; retried")
        state = None
        if raw is not None:
            try:
                state = decide.from_dict(json.loads(raw)["decide"])
                if state.stream_id != self.id:
                    raise ValueError(f"snapshot of stream {state.stream_id!r}")
            except (KeyError, TypeError, ValueError) as exc:
                state = None    # kept aside, never overwritten in place; new ids can't collide (§5.1)
                await m._writer.commit({store_key("snapshot-unreadable", self.id, f"#{int(wall * 1000)}"): raw})
                self.health("store_lag", f"the decide snapshot was unreadable ({exc!r:.80}); it was kept "
                                         "aside and decide state starts over under a new incarnation")
        if state is not None:
            self.state, self.epoch = state, state.stream_epoch + 1
            if m._trace is not None:
                m._rec("restore", self.id, restored=True, snapshot=decide.to_dict(state))
            self.step(decide.EpochStarted(self.epoch, restart=True))
        else:
            self.state = decide.new_state(self.id, created_at=wall)
            if m._trace is not None:
                m._rec("restore", self.id, restored=False, snapshot=decide.to_dict(self.state))
        ids = sorted(m._conds)
        for cid, raw in zip(ids, await m._writer.load([store_key("counterexamples", self.id, "/" + c) for c in ids])):
            if raw is not None:
                self.counter[cid] = [CounterExample(e["note"], None if e["jpeg"] is None else bytes.fromhex(e["jpeg"]))
                                     for e in json.loads(raw)]
        for (cid, subject), until in m._snoozes.items():
            if until > wall:
                self.step(decide.Snooze(cid, until, subject))
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
        if self.own_ring:
            self.ring.add(frame)
        if self.period:     # the processing cap: capture and the pre-roll are not slowed
            if self.due is not None and frame.t < self.due - 0.1 * self.period:
                self.fate("*", "frames_capped")
                return
            self.due = (frame.t if self.due is None or frame.t - self.due > self.period else self.due) + self.period
        self.latest, self.latest_mono, self.latest_rx = frame, m._mono(), m._wall()
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
        for key, trigger, conds in m._groups:
            try:
                due, self.tstates[key] = trigger.step(frame, conds, self.tstates.get(key), frame.t)
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

    def current(self) -> Optional[Frame]:
        """The newest processed frame, unless the stream has gone quiet since: a frame from before
        an open ``link_down`` interval, or one older than ``STALE_FRAME_S``, is not current."""
        f, m = self.latest, self.m
        if f is None:
            return None
        down = self.blind_open.get(("link_down", None))
        if down is not None and self.latest_rx <= down[0]:
            return None
        if m._mono() - self.latest_mono > STALE_FRAME_S:
            return None
        if getattr(self.source, "live", True) and m._wall() - f.t_wall > STALE_FRAME_S:
            return None
        return f

    def drop(self, w: _Work, reason: str, now: float) -> None:
        """A look given up without being judged: counted as ``dropped_budget`` and, when tracing,
        recorded one by one (reason ``budget``: the budget refused it; ``deadline``: it waited
        past its deadline for a concurrency slot or for evidence; ``ended``: the stream ended)."""
        self.fate(w.condition_id, "dropped_budget")
        self.m._rec("drop", self.id, condition_id=w.condition_id, subject=w.subject, reason=reason,
                    purpose="verify" if w.verify is not None else "detect", priority=_PRIORITY_NAMES[w.priority],
                    t_trigger=w.t_trigger, waited_s=max(0.0, now - w.created))

    def starve(self, w: _Work) -> None:
        if not w.starved:
            w.starved = True
            self.fate(w.condition_id, "evidence_unavailable")

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
        latest = self.current()
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
                    self.starve(w)
                    return None  # the original evidence is gone
            elif latest is None or v.stream_epoch != self.epoch:
                if latest is None:
                    self.starve(w)
                return None
            else:
                frames = (latest,)
        elif w.frame is not None:
            frames = (w.frame,)
        elif latest is None:
            self.starve(w)
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
        self.m._cost_waiters.pop(self.id, None)
        await self.commits.join()
        self.outbox.finish()

    def dispatch(self, now: float) -> float:
        """Start what the budget, the concurrency limit and single flight allow; return how long
        to sleep before looking again (new work or a finished request wakes the loop sooner)."""
        m, wait = self.m, 1.0
        for key, w in list(self.pending.items()):
            if w.deadline is not None and now > w.deadline:
                del self.pending[key]
                self.drop(w, "budget" if w.blocked else "deadline", now)
            elif self.eof and self.current() is None and w.images is None and w.frame is None:
                del self.pending[key]               # the stream ended: no frame will come
                self.starve(w)
                self.drop(w, "ended", now)
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
            if group and self.start(group, "detect", now) == "budget":
                refused = True
                for w in group:
                    w.blocked = True
            deadlines = [w.deadline - now for w in self.pending.values() if w.deadline is not None]
            wait = min([wait] + deadlines)
        if refused != self.budget_blocked:
            self.budget_blocked = refused
            self.blind_set("budget_exhausted", refused)
            if refused:
                self.health("budget_exhausted", f"more than {m.max_requests_per_min} attempts a minute are due")
        changes = [c for c in (self.budget.next_change(now), m._cost.next_change(now)) if c is not None]
        if refused and changes:
            wait = min([wait] + changes)
        if not refused:             # not waiting for money: don't hold the line for others
            m._cost_waiters.pop(self.id, None)
        if self.granted:            # handed a slot it can't use now: pass it on
            self.granted = False
            m._release()
        return max(0.005, wait)

    def group(self) -> list[_Work]:
        """The most urgent look that can be judged now, plus every waiting look with identical
        evidence (a look whose evidence is unavailable doesn't hold up the others)."""
        order = sorted(self.pending.values(), key=lambda w: (w.priority, w.created))
        top, top_ev = next(((w, ev) for w in order if (ev := self.evidence(w)) is not None), (None, None))
        if top is None:
            return []
        cap = len(order) if self.m.judge.can_batch else max(1, self.m.max_requests_per_min // self.m._attempts)
        group = [top]
        for w in order[order.index(top) + 1:]:
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
                               evidence_t, purpose,  # type: ignore[arg-type]
                               max_output_tokens=_max_output_tokens(len(items),
                                                                    sum(len(i.condition.id) for i in items)))
        calls = 1 if m.judge.can_batch else len(items)
        reserve = max_attempts_for(m.judge, request) * calls
        cost = (worst_case_cost_for(m.judge, request) or 0.0) * calls
        cost_ok = m._cost.fits(cost, now)
        if cost_ok and cost > 0 and m._cost_waiters and next(iter(m._cost_waiters)) != self.id:
            cost_ok = False             # another stream has waited longer for the shared dollar cap
        if not cost_ok:
            m._cost_waiters.setdefault(self.id, None)
        if not (cost_ok and self.budget.fits(reserve, now)):
            return "budget"
        m._cost_waiters.pop(self.id, None)
        self.budget.take(reserve)       # both ledgers are checked and taken in one step on the loop:
        m._cost.take(cost)              # atomic across every stream
        self.held_cost.take(cost)
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
        m, result, t0 = self.m, None, self.m._mono()
        unknown = True      # cancelled, or the judge raised: what it sent and spent is unknown
        try:
            result = await m.judge.judge(request)
            unknown = False
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - a judge that raises broke its contract: an outage
            result = JudgeResult(failed_observations(request, "unavailable", m.judge.name, f"judge raised {type(exc).__name__}"), ())
        finally:  # unknown usage keeps the whole reservation charged for the window, and reports 0 spent
            usage = Usage() if unknown or result is None else sum((u.usage for u in result.usage), Usage())
            used = reserve if unknown else usage.attempts
            self.budget.settle(reserve, used, m._mono())
            m._cost.settle(cost, cost if unknown else usage.cost_usd, m._mono())
            self.held_cost.settle(cost, cost if unknown else 0.0, m._mono())
            self.totals["requests"] += 1
            self.totals["attempts"] += used
            self.cost_usd += usage.cost_usd
            shares = (item_usage(request, result) if not unknown and result is not None
                      else (Usage(),) * len(request.items))
            for item, share in zip(request.items, shares):
                row = self.per_condition.setdefault(item.condition.id, [0, 0, 0.0])
                row[0] += 1
                row[1] += share.attempts
                row[2] += share.cost_usd
            if m._trace is not None:
                self.trace_request(request, result, m._mono() - t0)
            if request.purpose == "verify":
                self.verify_task = None
            else:
                self.detect_task = None
            m._release()
        self.absorb(request, result, works)

    def trace_request(self, request: JudgeRequest, result: Optional[JudgeResult], latency: float) -> None:
        if result is None:
            usage, failure = [], "cancelled"
        else:
            usage = [dict(asdict(u.usage), judge=u.judge, failure=u.failure) for u in result.usage]
            failed = [o.failure for o in result.observations if o.failed]
            failure = failed[0] if failed and len(failed) == len(result.observations) else None
        self.m._rec("request", self.id, request_id=request.request_id, judge=self.m.judge.name,
                    purpose=request.purpose, stream_epoch=request.stream_epoch,
                    condition_ids=[i.condition.id for i in request.items], usage=usage,
                    latency_s=latency, failure=failure)

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
        m, now = self.m, self.m._wall()
        if m._trace is not None:
            m._checkpoint_if_due()
            versions = {cid: c.version for cid, c in m._conds.items()}
            if versions != self.traced_versions:
                self.traced_versions = versions
                m._rec("conditions", self.id, conditions=[encode_condition(c) for c in m._conds.values()])
            m._rec("decide_input", self.id, input=encode_input(inp), now=now, versions=versions)
        r = decide.step(self.state, inp, now, m._conds)
        self.state = r.state
        loop = asyncio.get_running_loop()
        for e in r.effects:
            t = type(e)
            if t is decide.Trace:
                m._rec("fate", self.id, event=e.event, condition_id=e.condition_id, subject=e.subject,
                       fate=e.fate, occurrence_id=e.occurrence_id, detail=e.detail)
                if e.fate is not None:
                    self.fate(e.condition_id, e.fate, record=False)
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
                        self.drop(w, "budget", self.m._mono())
                    if self.verify_task is not None and self.verify_work is w:
                        self.verify_task.cancel()
            elif t is decide.RequestRecheck:
                self.enqueue(e.condition_id, e.subject, RECHECK)
            elif t is decide.EmitHealth:
                m._report(e.health)
            elif t is decide.StoreCounterExample:
                self.spawn(self.store_counter(e))
        for d in r.decisions:
            fields = decision_record(d)
            del fields["stream_id"]         # the record's own stream_id
            m._rec("decision", self.id, **fields)
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

    def mute_ended(self, cid: str, subject: Optional[str]) -> None:
        """A snooze ended: look again at every held state it may have deferred a met for (D2)."""
        if self.m._closing or self.state is None:
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
            if self.dirty and self.commits.empty() and not self.halted:
                self.flush_snapshot()
            w = self.m._writer
            if w.pending > STORE_LAG_DEPTH or w.failures > failures or w.retrying:
                self.health("store_lag", f"{w.pending} store calls waiting, {w.failures} failed"
                                         + (", retrying" if w.retrying else ""))
            if self.halted:
                self.health("store_lag", "this stream is halted: its outbox record is unreadable")
            failures = w.failures
            trace = self.m._trace
            if trace is not None:
                if self.traced_fates and self.traced_bucket != int(self.m._wall() // COVERAGE_BUCKET_S):
                    self.trace_fates()
                self.m._checkpoint_if_due()
                if trace.dropped > self.m._trace_reported:
                    self.health("trace_dropped", f"trace: {trace.dropped - self.m._trace_reported} records dropped"
                                                 + (f" ({trace.error})" if trace.error else ""))
                    self.m._trace_reported = trace.dropped

    # ----------------------------------------------------------------- reports

    def coverage(self, since_wall: float) -> dict:
        fates: dict[str, Counter] = {}
        unavailable: Counter = Counter()
        capped = 0
        first = int(since_wall // COVERAGE_BUCKET_S)
        for bucket, counts in self.fates:
            if bucket >= first:
                for (cid, fate), n in counts.items():
                    if fate == "frames_capped":
                        capped += n
                    elif fate == "evidence_unavailable":
                        unavailable[cid] += n
                    else:
                        fates.setdefault(cid, Counter())[fate] += n
        now = self.m._wall()
        blind = [b for b in self.blind if b.t_end_wall is None or b.t_end_wall >= since_wall]
        blind += [BlindInterval(kind, self.id, start, None, detail, cid)  # type: ignore[arg-type]
                  for (kind, cid), (start, detail) in self.blind_open.items()]
        return {"stream_id": self.id, "since_wall": since_wall, "until_wall": now,
                "fates": {cid: dict(c) for cid, c in sorted(fates.items())},
                "frames_capped": capped, "evidence_unavailable": dict(sorted(unavailable.items())),
                "blind_intervals": sorted(blind, key=lambda b: b.t_start_wall),
                "blind_intervals_dropped": self.blind_dropped}

    def stats(self) -> dict:
        t, ob = self.totals, self.outbox
        counts = {k: t[k] for k in ("frames", "frames_capped", "frames_vetoed", "requests", "attempts",
                                    "evidence_unavailable")}
        fates = {k: v for k, v in t.items() if k not in counts}
        conditions = {cid: {"requests": r, "attempts": a, "cost_usd": c}
                      for cid, (r, a, c) in sorted(self.per_condition.items())}
        return dict(counts, cost_usd=self.cost_usd, cost_reserved_usd=self.held_cost.used(self.m._mono()),
                    conditions=conditions, fates=fates, decisions=ob.committed,
                    delivered=ob.delivered, redelivered=ob.redelivered, acked=ob.acked, unacked=ob.pending(),
                    pending_looks=len(self.pending), pending_verifications=len(self.verifies),
                    halted=self.halted)

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
    ``max_cost_per_hour`` is one cap for the whole monitor, reserved (worst case) before each
    request and reconciled after, guaranteed for judges with a ``price`` and best-effort otherwise;
    ``max_concurrent_requests`` limits requests in flight across every stream.
    ``trace`` (a path or a :class:`~percept.trace.TraceWriter`) records every runtime event as
    JSON lines, without images or credentials, so a run can be inspected and replayed
    (``percept.replay``); None (the default) records nothing.

    Configuration errors raise here, in :meth:`add` and in :meth:`watch`. Nothing raises
    during operation: outages, failures, budget exhaustion and store lag are reported by
    :meth:`health` and :meth:`coverage`. Methods other than :meth:`push` are called from the
    monitor's event loop thread.
    """

    def __init__(self, judge: Judge, conditions: Iterable[Condition] = (), *, store: Any = None,
                 max_requests_per_min: int = 6, max_cost_per_hour: Optional[float] = None,
                 max_concurrent_requests: int = 4, trace: Union[None, str, os.PathLike, TraceWriter] = None) -> None:
        if not isinstance(judge, Judge):
            raise TypeError("judge must implement the Judge protocol (name, can_batch, price, async judge)")
        for name, value in (("max_requests_per_min", max_requests_per_min),
                            ("max_concurrent_requests", max_concurrent_requests)):
            if not (isinstance(value, int) and not isinstance(value, bool) and value >= 1):
                raise ValueError(f"{name} is an integer >= 1")
        if max_cost_per_hour is not None and not max_cost_per_hour > 0:
            raise ValueError("max_cost_per_hour is positive or None")
        if not (trace is None or isinstance(trace, (str, os.PathLike, TraceWriter))):
            raise TypeError("trace is None, a path or a TraceWriter")
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
        self._mutes: dict[tuple[str, Optional[str]], asyncio.TimerHandle] = {}
        self._cost = _Window(max_cost_per_hour, lambda: COST_WINDOW_S)
        self._slots = 0
        self._waiters: OrderedDict[str, _Stream] = OrderedDict()
        self._cost_waiters: OrderedDict[str, None] = OrderedDict()   # streams waiting on the dollar cap, in order
        self._quality, self._scene = QualityGate(), SceneMonitor()
        self._default_change, self._default_heartbeat = on_change(), every(300)
        self._wall, self._mono = time.time, time.monotonic
        self._ids = itertools.count(1)
        self._incarnation = format(int(time.time() * 1000), "x")
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._closing = self._closed = False
        self._regroup()
        self._owns_trace = isinstance(trace, (str, os.PathLike))
        self._trace: Optional[TraceWriter] = TraceWriter(trace) if self._owns_trace else trace  # type: ignore[arg-type]
        if self._trace is not None:
            self._trace.checkpoints = True     # rotate where the runtime can open the file with a checkpoint
        self._trace_reported = 0    # trace records whose loss health has reported

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

    @staticmethod
    def _trigger_key(trigger: Any) -> Any:
        """Triggers group (and keep their state) by value when hashable, else by identity: two
        conditions with equal triggers share one step and one state."""
        try:
            hash(trigger)
        except TypeError:
            return ("id", id(trigger))
        return trigger

    def _regroup(self) -> None:
        groups: dict[Any, tuple[Any, list[Condition]]] = {}
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
                groups.setdefault(self._trigger_key(trigger), (trigger, []))[1].append(cond)
        self._groups = [(k, t, tuple(cs)) for k, (t, cs) in groups.items()]
        for stream in getattr(self, "_streams", {}).values():    # forget replaced triggers' state
            stream.tstates = {k: v for k, v in stream.tstates.items() if k in groups}

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

    def _record(self, kind: str, stream_id: str, **fields: Any) -> dict:
        return {"v": TRACE_FORMAT, "t_wall": self._wall(), "stream_id": stream_id, "kind": kind, **fields}

    def _rec(self, kind: str, stream_id: str, **fields: Any) -> None:
        """Write one trace record, when tracing. Never blocks and never raises."""
        if self._trace is not None:
            self._trace.write(self._record(kind, stream_id, **fields))

    def _checkpoint_if_due(self) -> None:
        """The trace file is full: start the next one with every stream's state (on the loop)."""
        if self._trace is not None and self._trace.rotate_due:
            self._trace.checkpoint([r for s in self._streams.values() for r in s.checkpoint_records()])

    def _report(self, h: Health) -> None:
        self._health.put(h)
        self._rec("health", h.stream_id, health_kind=h.kind, detail=h.detail, count=h.count,
                  condition_id=h.condition_id, subject=h.subject)

    def _source_health(self, h: Health) -> None:
        """A source's link health. Sources may call it from another thread (before ``frames()``
        runs, or an app's worker thread): it is then handed to the loop, which owns this state."""
        if self._loop is not None and not self._loop.is_closed():
            try:
                running = asyncio.get_running_loop()
            except RuntimeError:
                running = None
            if running is not self._loop:
                self._loop.call_soon_threadsafe(self._source_health, h)
                return
        self._report(h)
        stream = self._streams.get(h.stream_id)
        if stream is not None and h.kind in ("link_down", "link_up"):
            stream.blind_set("link_down", h.kind == "link_down", h.detail)

    def _stream_for(self, occurrence_id: str) -> Optional[_Stream]:
        matches = [s for sid, s in self._streams.items() if occurrence_id.startswith(sid + "/")]
        return max(matches, key=lambda s: len(s.id), default=None)

    def _each(self, inp: Any) -> None:
        for stream in self._streams.values():
            stream.step(inp)

    # ----------------------------------------------------------------- streams

    def watch(self, source: Union[str, Source], *, stream_id: Optional[str] = None, ack: str = "auto",
              max_unacked: int = 100, fps: Optional[float] = DEFAULT_FPS) -> Watch:
        """Decisions for one stream, in order, as ``Delivery`` envelopes (``async for``).

        ``source`` is a URI (see ``percept.open_source``), a ``Source``, or the id of a stream
        this monitor already has (pushed to, or watched before). Watching a stream again takes
        it over: every unacknowledged decision is delivered again. Ends at ``aclose()``, or
        once a source that ends (a file, a closed ``PushSource``) is fully judged and delivered.

        ``fps`` (default 5) caps how many frames per second, on the stream clock, go through the
        quality gate, scene check and triggers; ``None`` processes every frame. Capture and the
        pre-roll are not slowed, and skipped frames are counted in ``coverage()``. It applies
        when the stream starts; ``push()`` starts streams at the default.

        A URI is opened with ``open_source``'s defaults. To configure capture (its rate, frame
        size, reconnection, or ``on_frame`` / ``on_health`` callbacks), open the source yourself
        and watch it: ``open_source`` owns the source, ``watch`` owns monitoring and delivery::

            source = percept.open_source(uri, on_frame=show, fps=5)
            async for delivery in monitor.watch(source): ...

        Link health: the monitor subscribes to any source with ``add_health_listener`` (every
        source from ``open_source`` and every ``PushSource``, including ones the app built with
        its own ``on_frame`` / ``on_health``), so an outage is a ``link_down`` blind interval and
        health event without further wiring.
        """
        if fps is not None and not fps > 0:
            raise ValueError("fps is positive or None")
        self._bind()
        if isinstance(source, str):
            stream = self._streams.get(stream_id or source)
            if stream is None:
                opened = open_source(source, stream_id=stream_id, preroll=PreRoll())
                stream = self._streams.get(opened.stream_id) or self._add(opened, fps)
        elif isinstance(source, Source):
            if stream_id is not None and stream_id != source.stream_id:
                raise ValueError("stream_id must match source.stream_id")
            stream = self._streams.get(source.stream_id)
            if stream is not None and stream.source is not source:
                raise ValueError(f"stream {source.stream_id!r} is already watched from another source")
            stream = stream or self._add(source, fps)
        else:
            raise TypeError("watch() takes a URI or a Source")
        return Watch(stream.outbox, ack, max_unacked,
                     lambda n: stream.health("store_lag", f"delivery paused: {n} decisions unacknowledged"))

    def _add(self, source: Any, fps: Optional[float] = DEFAULT_FPS) -> _Stream:
        lock = getattr(self.store, "lock", None)
        if callable(lock):
            lock()      # one owner per database: raises StoreLocked if another process has it
        if getattr(source, "preroll", 0) is None:
            source.preroll = PreRoll()  # sources that accept one are fed at capture rate
        stream = self._streams[source.stream_id] = _Stream(self, source, fps)
        listen = getattr(source, "add_health_listener", None)
        if callable(listen):
            listen(self._source_health)
            if getattr(source, "link", None) == "down":     # already down when handed over
                self._source_health(Health("link_down", source.stream_id, self._wall(),
                                           "the link was down when the stream was added"))
        for key in self._snoozes:       # snoozes set before the loop was known
            if key not in self._mutes:
                self._schedule_mute(key)
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

    def set_link(self, stream_id: str, up: bool, detail: str = "") -> None:
        """Report a stream's link as down or up, for capture the app owns. Down opens a
        ``link_down`` blind interval and raises health ``link_down``; looks that need a current
        frame wait until it is up again. Repeating the current state reports nothing.

        This is the way to reach a stream the monitor created from :meth:`push` (the app never
        holds its ``PushSource``) or a custom ``Source``; an app that built its own
        ``PushSource`` can call ``PushSource.set_link`` directly, from any thread. For a push
        stream it is recorded on the source, so a frame pushed later marks the link up, and
        coming back up starts a new epoch (see ``PushSource.set_link``)."""
        self._on_loop()
        stream = self._streams.get(stream_id)
        if stream is None:
            raise ValueError(f"unknown stream {stream_id!r}")
        own = getattr(stream.source, "set_link", None)
        if callable(own):
            own(up, detail)
            return
        if (("link_down", None) in stream.blind_open) != (not up):
            self._source_health(Health("link_up" if up else "link_down", stream_id, self._wall(),  # type: ignore[arg-type]
                                       detail or ("reported up" if up else "reported down by the application")))

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
        self._rec("poke", stream_id, condition_ids=ids, subject=subject, observed_at=observed_at, reason=reason,
                  evidence_images=0 if images is None else len(images))
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
            stream.blind_set("paused", False, condition_id=condition_id)
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
            stream.blind_set("paused", True, "paused by the application", condition_id)

    def resume(self, condition_id: str) -> None:
        self._on_loop()
        if condition_id not in self._paused:
            return
        self._paused.discard(condition_id)
        self._regroup()
        for stream in self._streams.values():
            stream.blind_set("paused", False, condition_id=condition_id)
            stream.enqueue(condition_id, decide.SCENE, HEARTBEAT)   # one look at what was missed

    def snooze(self, condition_id: str, until_wall: float, subject: Optional[str] = None) -> None:
        """Mute met and uncertain until ``until_wall``; judging continues. A state that is still
        true when the mute ends is looked at again, and its deferred met is delivered then."""
        self._on_loop()
        if condition_id not in self._conds:
            raise ValueError(f"unknown condition {condition_id!r}")
        key = (condition_id, subject)
        self._snoozes[key] = until_wall
        for stream in self._streams.values():
            stream.step(decide.Snooze(condition_id, until_wall, subject))
        self._schedule_mute(key)

    def _schedule_mute(self, key: tuple[str, Optional[str]]) -> None:
        """One timer per (condition, subject), replaced by a new snooze, removed when it fires."""
        old = self._mutes.pop(key, None)
        if old is not None:
            old.cancel()
        until = self._snoozes.get(key)
        if until is None or self._loop is None:
            return
        if until <= self._wall():       # a snooze that ends now or earlier cancels the mute
            del self._snoozes[key]
            self._mute_ended(key)
            return
        self._mutes[key] = self._loop.call_later(until - self._wall(), self._mute_ended, key)

    def _mute_ended(self, key: tuple[str, Optional[str]]) -> None:
        self._mutes.pop(key, None)
        until = self._snoozes.get(key)
        if until is not None and until > self._wall():     # fired early: wait for the rest
            self._schedule_mute(key)
            return
        self._snoozes.pop(key, None)
        for stream in self._streams.values():
            stream.mute_ended(*key)

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
        """Exact counters, per stream and for health, available at any time.

        Cost: ``cost_usd`` is what the judge reported (settled), per stream and in total, with a
        per-condition split under ``streams[id]["conditions"]``; ``cost_reserved_usd`` is what is
        still held against ``max_cost_per_hour`` beyond that (requests in flight, and cancelled
        requests, which report no usage, for the rest of their hour); ``cost_usd_in_window`` is
        everything the hourly cap counts now."""
        return {"streams": {sid: s.stats() for sid, s in self._streams.items()},
                "health": dict(self._health.counts), "health_dropped": self._health.dropped,
                "store_pending": self._writer.pending, "store_failures": self._writer.failures,
                "store_retrying": self._writer.retrying, "cost_usd_in_window": self._cost.used(self._mono()),
                "cost_usd": sum(s.cost_usd for s in self._streams.values()),
                "cost_reserved_usd": sum(s.held_cost.used(self._mono()) for s in self._streams.values()),
                "trace_written": 0 if self._trace is None else self._trace.written,
                "trace_dropped": 0 if self._trace is None else self._trace.dropped}

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
        for handle in self._mutes.values():
            handle.cancel()
        self._mutes.clear()
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
            unlisten = getattr(s.source, "remove_health_listener", None)
            if callable(unlisten):
                unlisten(self._source_health)
            close = getattr(s.source, "aclose", None)
            if close is not None:
                try:
                    await asyncio.wait_for(close(), FLUSH_TIMEOUT_S)
                except Exception:  # noqa: BLE001 - shutdown goes on
                    log.exception("percept: closing stream %s failed", s.id)
        if self._owns_store:
            self.store.close()
        if self._trace is not None:     # last: everything above may still write records
            for s in streams:
                s.trace_fates()
            await asyncio.to_thread(self._trace.close if self._owns_trace else self._trace.flush, FLUSH_TIMEOUT_S)
        self._health.closed = True
        self._health.ready.set()
        self._closed = True

    async def __aenter__(self) -> Monitor:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()
