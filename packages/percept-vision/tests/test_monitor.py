"""The runtime end to end, offline: PushSource or fake sources, FakeJudge, injected faults."""

from __future__ import annotations

import asyncio
import threading
import time

import numpy as np
import pytest

import percept
from percept import monitor as rt
from percept.contracts import Condition, Health, JudgeResult
from percept.delivery import MemoryStore, SqliteStore, Writer, store_key
from percept.judges import FakeJudge, cascade
from percept.judges.base import RetryableError
from percept.sources import CaptureSource, PushSource, Reader

Y, N = ("yes", 0.9, "seen"), ("no", 0.9, "not seen")


def run(coro, timeout: float = 20.0):
    return asyncio.run(asyncio.wait_for(coro, timeout))


def image(i: int = 0, h: int = 96, w: int = 128) -> np.ndarray:
    """A textured frame (no smear, no blur); ``i`` shifts it slightly so frames differ."""
    y, x = np.mgrid[0:h, 0:w]
    gray = ((x * 7 + y * 13 + i) % 200 + 20).astype(np.uint8)
    return np.dstack([gray, gray, gray])


class Always:
    """A trigger that makes every condition due on every frame."""

    def step(self, frame, conditions, state, now):
        return {c.id: "always" for c in conditions}, None


class Never:
    def step(self, frame, conditions, state, now):
        return {}, None


class At:
    """Due on the frame with this ``seq`` only."""

    def __init__(self, seq: int) -> None:
        self.seq = seq

    def step(self, frame, conditions, state, now):
        return ({c.id: "at" for c in conditions} if frame.seq == self.seq else {}), None


def answers(per_purpose):
    """A FakeJudge script: ``per_purpose[purpose]`` is a list of answers used in order (the last repeats),
    or a callable(request) returning the reply."""
    calls = {"detect": 0, "verify": 0}

    def script(request):
        spec = per_purpose[request.purpose]
        if callable(spec):
            return spec(request)
        reply = spec[min(calls[request.purpose], len(spec) - 1)]
        calls[request.purpose] += 1
        if isinstance(reply, BaseException):
            return reply
        return {item.condition.id: reply for item in request.items}

    return script


async def push_frames(src: PushSource, n: int, *, t0: float | None = None, step: float = 0.5, pause: float = 0.02,
                      start: int = 0):
    for i in range(start, start + n):
        src.push(image(i), t_wall=None if t0 is None else t0 + i * step)
        await asyncio.sleep(pause)


async def collect(watch, limit: int | None = None):
    out = []
    async for d in watch:
        out.append(d)
        if limit is not None and len(out) >= limit:
            break
    return out


def mem() -> MemoryStore:
    return MemoryStore()


# --------------------------------------------------------------------------- end to end


def test_state_is_met_then_cleared_with_keyframe_and_evidence():
    door = Condition("door", "the door is open", trigger=Always(), clear_after_no=2, emit_cleared=True)
    judge = FakeJudge(answers({"detect": [Y, Y, N]}))

    async def main():
        async with percept.Monitor(judge, [door], store=mem(), max_requests_per_min=1000) as m:
            src = PushSource("cam")
            watch = m.watch(src)
            await push_frames(src, 6)
            src.close()
            got = await collect(watch)
            return got, m.stats(), m.coverage("cam")

    got, stats, cov = run(main())
    assert [(d.decision.type, d.decision.cause) for d in got] == [("met", None), ("cleared", "no")]
    met = got[0].decision
    assert met.keyframe and len(met.keyframe) <= 40_000 and met.keyframe[:2] == b"\xff\xd8"
    assert met.evidence and all(e.stream_id == "cam" and e.ephemeral for e in met.evidence)
    assert met.request_ids and met.cost.attempts >= 1
    assert stats["streams"]["cam"]["decisions"] == 2 and stats["streams"]["cam"]["acked"] == 2
    assert cov["fates"]["door"]["judged"] >= 3


def test_default_triggers_heartbeat_a_state_and_fire_an_event_on_change():
    door = Condition("door", "the door is open")                              # on_change + every(300)
    fall = Condition("fall", "a person falls", kind="event", zones=(((0.0, 0.0), (0.5, 0.0), (0.5, 0.5), (0.0, 0.5)),))
    judge = FakeJudge(answers({"detect": [Y], "verify": [Y]}))

    async def main():
        async with percept.Monitor(judge, [door, fall], store=mem(), max_requests_per_min=1000) as m:
            src = PushSource("cam")
            watch = m.watch(src)
            base = image(0)
            for _ in range(4):
                src.push(base)
                await asyncio.sleep(0.02)
            changed = base.copy()
            changed[5:40, 5:55] = 255   # inside the fall zone, a small part of the frame
            for _ in range(3):
                src.push(changed)
                await asyncio.sleep(0.05)
            src.close()
            return await collect(watch)

    got = run(main())
    assert sorted((d.decision.condition_id, d.decision.type) for d in got) == [("door", "met"), ("fall", "met")]
    first_door = next(r for _, _, _, r in judge.sent if r.items[0].condition.id == "door")
    assert first_door.purpose == "detect"


def test_an_event_is_verified_on_its_original_clip():
    fall = Condition("fall", "a person falls", kind="event", trigger=At(6))
    judge = FakeJudge(answers({"detect": [Y], "verify": [Y]}))

    async def main():
        async with percept.Monitor(judge, [fall], store=mem(), max_requests_per_min=1000) as m:
            src = PushSource("cam")
            watch = m.watch(src)
            await push_frames(src, 10, t0=time.time() - 10, pause=0.03)
            src.close()
            return await collect(watch)

    got = run(main())
    detect = [s for s in judge.sent if s[3].purpose == "detect"]
    verify = [s for s in judge.sent if s[3].purpose == "verify"]
    assert len(detect) == 1 and len(verify) == 1
    assert len(detect[0][1]) >= 3                              # a clip, not a single frame
    assert verify[0][1] == detect[0][1]                        # the same JPEG frames, in order
    assert verify[0][3].evidence_t == detect[0][3].evidence_t
    assert [(d.decision.type, d.decision.cause) for d in got] == [("met", None)]
    assert len(got[0].decision.evidence) == len(detect[0][1])


def test_a_dead_verifier_ends_in_uncertain_at_the_deadline(monkeypatch):
    monkeypatch.setattr(rt, "VERIFY_RETRY_S", 0.05)
    fall = Condition("fall", "a person falls", kind="event", trigger=At(2), candidate_deadline_s=0.6)
    judge = FakeJudge(answers({"detect": [Y], "verify": [RetryableError("down")]}))

    async def main():
        async with percept.Monitor(judge, [fall], store=mem(), max_requests_per_min=1000) as m:
            src = PushSource("cam")
            watch = m.watch(src)
            await push_frames(src, 4, t0=time.time() - 4)
            src.close()
            got = await collect(watch)
            return got, m.stats()["health"], m.coverage("cam")

    got, health, cov = run(main())
    assert [(d.decision.type, d.decision.cause) for d in got] == [("uncertain", "timeout")]
    assert health.get("judge_unavailable", 0) >= 2               # retried until the deadline
    assert sum(1 for s in judge.sent if s[3].purpose == "verify") >= 4
    assert any(b.kind == "judge_down" for b in cov["blind_intervals"])


# --------------------------------------------------------------------------- budget


def test_budget_property_no_window_ever_holds_more_attempts_than_the_limit():
    import hypothesis
    st = hypothesis.strategies

    @hypothesis.settings(max_examples=150, deadline=None)
    @hypothesis.given(limit=st.integers(4, 12), reserve=st.integers(1, 4),
                      arrivals=st.lists(st.tuples(st.floats(0, 5), st.floats(0, 20), st.floats(0, 1)), max_size=60))
    def check(limit, reserve, arrivals):
        reserve = min(reserve, limit)
        budget = rt._Budget(limit, None)
        now, attempts, running = 0.0, [], []
        for gap, duration, frac in arrivals:
            now += gap
            for end, used, start in sorted(running):
                if end <= now:
                    budget.settle(reserve, 0.0, used, 0.0, end)
                    running.remove((end, used, start))
            if budget.fits(reserve, 0.0, now):
                budget.take(reserve, 0.0)
                used = max(0, min(reserve, round(frac * reserve + 0.49)))
                attempts += [now + duration * k / max(1, used) for k in range(used)]
                running.append((now + duration, used, now))
        attempts.sort()
        for i, t in enumerate(attempts):
            assert sum(1 for u in attempts[i:] if u < t + rt.BUDGET_WINDOW_S) <= limit

    check()


def test_a_trigger_storm_with_a_cascade_and_retries_never_exceeds_the_window(monkeypatch):
    monkeypatch.setattr(rt, "BUDGET_WINDOW_S", 1.0)
    monkeypatch.setattr(rt, "DEADLINES", {rt.EVENT: 0.3, rt.STATE: 0.5})
    monkeypatch.setattr(rt, "VERIFY_RETRY_S", 0.05)
    attempts: list[tuple[float, str]] = []
    rng = np.random.default_rng(7)

    def lite(request):
        attempts.append((time.monotonic(), request.purpose))
        if rng.random() < 0.4:
            return RetryableError("busy")
        return {i.condition.id: ("unknown", 0.3, "?") for i in request.items}   # always escalates

    def pro(request):
        attempts.append((time.monotonic(), request.purpose))
        if rng.random() < 0.3:
            return RetryableError("busy")
        return {i.condition.id: Y for i in request.items}

    judge = cascade(FakeJudge(lite, name="lite"), FakeJudge(pro, name="pro"))
    assert judge.max_attempts == 4
    conds = [Condition("fall", "a person falls", kind="event", trigger=Always(), rearm_s=0.2),
             Condition("door", "the door is open", trigger=Always(), clear_after_no=1)]
    limit = 8

    async def main():
        async with percept.Monitor(judge, conds, store=mem(), max_requests_per_min=limit) as m:
            src = PushSource("cam")
            watch = m.watch(src)
            await push_frames(src, 110, pause=0.02)
            src.close()
            await collect(watch)               # drained: nothing is in flight any more
            return m.coverage("cam"), m.stats()

    cov, stats = run(main())
    times = sorted(t for t, _ in attempts)
    assert any(p == "verify" for _, p in attempts)                     # verification ran, in the budget
    for i, t in enumerate(times):
        assert sum(1 for u in times[i:] if u < t + 1.0) <= limit
    assert sum(c.get("dropped_budget", 0) for c in cov["fates"].values()) > 0
    assert any(b.kind == "budget_exhausted" for b in cov["blind_intervals"])
    assert stats["health"]["budget_exhausted"] >= 1
    assert stats["streams"]["cam"]["attempts"] == len(attempts)


def test_verification_waits_for_the_budget_like_everything_else(monkeypatch):
    monkeypatch.setattr(rt, "BUDGET_WINDOW_S", 1.0)
    sent: list[tuple[float, str]] = []

    def script(request):
        sent.append((time.monotonic(), request.purpose))
        return {i.condition.id: Y for i in request.items}

    fall = Condition("fall", "a person falls", kind="event", trigger=At(1))
    judge = FakeJudge(script)      # max_attempts 2: one request per window at max_requests_per_min=2

    async def main():
        async with percept.Monitor(judge, [fall], store=mem(), max_requests_per_min=2) as m:
            src = PushSource("cam")
            watch = m.watch(src)
            await push_frames(src, 3, t0=time.time() - 3)
            got = await asyncio.wait_for(watch.__anext__(), 3.0)
            return got, m.stats(), m.coverage("cam")

    got, stats, cov = run(main())
    (t_detect, p1), (t_verify, p2) = sent
    assert (p1, p2) == ("detect", "verify") and got.decision.type == "met"
    assert t_verify - t_detect >= 0.9            # queued until the detection's attempt left the window
    assert stats["health"]["verify_deferred"] == 1
    assert [b.kind for b in cov["blind_intervals"]] == ["budget_exhausted"]


def test_configuration_errors_raise_at_construction_and_watch(tmp_path):
    judge = FakeJudge({})
    with pytest.raises(ValueError, match="at least"):
        percept.Monitor(judge, [], store=mem(), max_requests_per_min=1)      # one call may make 2 attempts
    with pytest.raises(ValueError, match="route"):
        percept.Monitor(judge, [Condition("a", "x", judge="pro")], store=mem())
    with pytest.raises(ValueError, match="duplicate"):
        percept.Monitor(judge, [Condition("a", "x"), Condition("a", "y")], store=mem())

    async def main():
        async with percept.Monitor(judge, [], store=mem()) as m:
            with pytest.raises(FileNotFoundError):
                m.watch(str(tmp_path / "missing.mp4"))

    run(main())


# --------------------------------------------------------------------------- concurrency


class SlowJudge:
    """Sleeps for real: a 10 s provider."""

    name, can_batch, price, max_attempts = "slow", True, None, 1

    def __init__(self, seconds: float) -> None:
        self.seconds, self.started, self.cancelled = seconds, 0, 0

    async def judge(self, request):
        self.started += 1
        try:
            await asyncio.sleep(self.seconds)
        except asyncio.CancelledError:
            self.cancelled += 1
            raise
        return JudgeResult((), ())


class SlowStore(MemoryStore):
    def __init__(self, seconds: float) -> None:
        super().__init__()
        self.seconds = seconds

    def commit(self, writes):
        time.sleep(self.seconds)
        super().commit(writes)


def test_a_ten_second_judge_never_stalls_capture():
    judge = SlowJudge(10.0)
    door = Condition("door", "the door is open", trigger=Always())

    async def main():
        m = percept.Monitor(judge, [door], store=mem(), max_requests_per_min=100)
        src = PushSource("cam")
        m.watch(src)
        t0 = time.monotonic()
        await push_frames(src, 40, pause=0.01)
        await asyncio.sleep(0.05)
        elapsed = time.monotonic() - t0
        stats = m.stats()["streams"]["cam"]
        ring = len(src.preroll)
        await m.aclose()
        return elapsed, stats, ring, src.dropped

    elapsed, stats, ring, dropped = run(main())
    assert judge.started == 1 and judge.cancelled == 1          # single flight; cancelled at shutdown
    assert stats["frames"] == 40 and dropped == 0 and ring >= 1
    assert elapsed < 2.0


def test_a_slow_store_delays_delivery_but_not_capture_or_judging():
    door = Condition("door", "the door is open", trigger=Always(), clear_after_no=1, emit_cleared=True)
    judge = FakeJudge(answers({"detect": lambda r: {i.condition.id: (Y if len(judge.sent) % 2 else N)
                                                    for i in r.items}}))

    async def main():
        async with percept.Monitor(judge, [door], store=SlowStore(0.2), max_requests_per_min=1000) as m:
            src = PushSource("cam")
            m.watch(src)
            await push_frames(src, 20, pause=0.01)
            stats = m.stats()["streams"]["cam"]
            return stats, src.dropped

    stats, dropped = run(main(), timeout=40)
    assert stats["frames"] == 20 and dropped == 0
    assert stats["requests"] >= 8                      # judging went on while commits waited
    assert stats["decisions"] < stats["requests"] / 2  # most decisions were still waiting for the disk


# --------------------------------------------------------------------------- delivery


class CrashingStore(MemoryStore):
    """Applies the first ``crash_at`` commits, then fails every later one (the disk is gone)."""

    def __init__(self, crash_at: int) -> None:
        super().__init__()
        self.crash_at, self.commits = crash_at, 0
        self.committed: list[str] = []      # decision ids whose commit was applied
        self.acked: list[str] = []          # ... and whose deletion was applied
        self.records: dict[str, str] = {}   # store key -> decision id

    def commit(self, writes):
        self.commits += 1
        if self.commits > self.crash_at:
            raise OSError("crashed")
        for key, value in writes.items():
            if key.startswith("outbox:"):
                if value is None:
                    self.acked.append(self.records[key])
                else:
                    did = percept.Decision.from_dict(__import__("json").loads(value)["decision"]).decision_id
                    if key not in self.records:
                        self.committed.append(did)
                    self.records[key] = did
        super().commit(writes)

    def durable(self) -> MemoryStore:
        copy = MemoryStore()
        copy._data = dict(self._data)
        return copy


def _crash_run(crash_at: int):
    door = Condition("door", "the door is open", trigger=Always(), clear_after_no=1, emit_cleared=True)
    flip = iter(range(10**6))
    judge = FakeJudge(answers({"detect": lambda r: {i.condition.id: (Y if next(flip) % 2 == 0 else N)
                                                    for i in r.items}}))
    store = CrashingStore(crash_at)
    seen: list[tuple[str, int]] = []

    async def before():
        m = percept.Monitor(judge, [door], store=store, max_requests_per_min=1000)
        src = PushSource("cam")
        watch = m.watch(src)

        async def consume():
            async for d in watch:
                seen.append((d.decision.decision_id, d.attempt))

        consumer = asyncio.ensure_future(consume())
        await push_frames(src, 6, pause=0.02)
        await asyncio.sleep(0.05)
        consumer.cancel()
        await m.aclose()

    async def after():
        m = percept.Monitor(FakeJudge({}), [door], store=store.durable(), max_requests_per_min=1000)
        src = PushSource("cam")
        watch = m.watch(src)
        src.close()
        got = [(d.decision.decision_id, d.attempt) async for d in watch]
        await m.aclose()
        return got

    run(before())
    return store, seen, run(after())


def test_crash_at_every_commit_boundary_loses_and_duplicates_nothing(monkeypatch):
    monkeypatch.setattr(rt, "FLUSH_TIMEOUT_S", 0.05)
    monkeypatch.setattr(Writer, "RETRY_FIRST_S", 0.01)
    boundaries = 0
    for crash_at in range(0, 60):
        store, seen, redelivered = _crash_run(crash_at)
        if store.commits <= crash_at:   # the run finished before this boundary: all covered
            break
        boundaries += 1
        durable = store.durable()
        pending = [k for k in sorted(durable._data) if k.startswith("outbox:")]
        stored = {store.records[k]: __import__("json").loads(durable._data[k])["attempt"] for k in pending}
        expected = [store.records[k] for k in pending]
        assert [d for d, _ in redelivered] == expected                      # in order, same ids
        assert all(a == stored[d] + 1 for d, a in redelivered)              # attempt + 1
        lost = set(store.committed) - set(store.acked) - {d for d, _ in redelivered}
        assert not lost, (crash_at, lost)                                   # nothing committed is lost
        assert not set(store.acked) & {d for d, _ in redelivered}           # nothing acked comes back
        assert {d for d, _ in seen} <= set(store.committed)                 # committed before it was yielded
        for did, attempt in seen:                                           # delivered before: a higher attempt
            assert all(a > attempt for d, a in redelivered if d == did)
    assert boundaries >= 10


def test_auto_ack_only_after_the_loop_body_succeeds():
    door = Condition("door", "the door is open", trigger=Never())
    judge = FakeJudge(answers({"detect": [Y]}))

    async def main():
        async with percept.Monitor(judge, [door], store=mem()) as m:
            src = PushSource("cam")
            watch = m.watch(src)
            src.push(image())
            await asyncio.sleep(0.05)
            m.poke("cam", ["door"], reason="test")
            with pytest.raises(RuntimeError):
                async for d in watch:
                    first = (d.decision.decision_id, d.attempt)
                    raise RuntimeError("handler failed")
            src.close()
            again = [(d.decision.decision_id, d.attempt) async for d in m.watch("cam")]
            third = [d async for d in m.watch("cam")]
            return first, again, third

    first, again, third = run(main())
    assert again == [(first[0], 2)] and third == []


def test_max_unacked_blocks_delivery_only_and_raises_store_lag():
    door = Condition("door", "the door is open", trigger=Always(), clear_after_no=1, emit_cleared=True)
    flip = iter(range(10**6))
    judge = FakeJudge(answers({"detect": lambda r: {i.condition.id: (Y if next(flip) % 2 == 0 else N)
                                                    for i in r.items}}))
    store = mem()

    async def main():
        async with percept.Monitor(judge, [door], store=store, max_requests_per_min=1000) as m:
            src = PushSource("cam")
            watch = m.watch(src, ack="manual", max_unacked=2)
            await push_frames(src, 3, pause=0.03)
            held = [await watch.__anext__(), await watch.__anext__()]
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(watch.__anext__(), 0.3)
            requests = m.stats()["streams"]["cam"]["requests"]
            await push_frames(src, 6, pause=0.03, start=3)
            stats = m.stats()
            durable = sum(1 for k in store._data if k.startswith("outbox:"))
            await held[0].ack()
            third = await asyncio.wait_for(watch.__anext__(), 1.0)
            return held, third, requests, stats, durable

    held, third, requests, stats, durable = run(main())
    s = stats["streams"]["cam"]
    assert s["requests"] > requests and s["frames"] == 9           # capture and judging went on
    assert s["unacked"] > 2 and durable == s["unacked"]           # buffered durably, none dropped
    assert stats["health"]["store_lag"] >= 1
    assert third.decision.decision_id not in {d.decision.decision_id for d in held}


def test_resolve_while_a_request_is_in_flight_does_not_resurrect_the_occurrence():
    door = Condition("door", "the door is open", trigger=Never())
    gate = asyncio.Event()

    class Gated:
        name, can_batch, price, max_attempts = "gated", True, None, 1

        def __init__(self):
            self.inner, self.calls = FakeJudge(answers({"detect": [Y]})), 0

        async def judge(self, request):
            self.calls += 1
            if self.calls > 1:
                await gate.wait()
            return await self.inner.judge(request)

    async def main():
        async with percept.Monitor(Gated(), [door], store=mem()) as m:
            src = PushSource("cam")
            watch = m.watch(src)
            src.push(image())
            await asyncio.sleep(0.05)
            m.poke("cam", "door", reason="first look")
            met = await asyncio.wait_for(watch.__anext__(), 2)
            m.poke("cam", "door", reason="second look")     # in flight, held at the gate
            await asyncio.sleep(0.05)
            m.resolve(met.decision.occurrence_id)
            gate.set()
            await asyncio.sleep(0.1)
            src.close()
            rest = await collect(watch)
            return met, rest, m.coverage("cam")

    met, rest, cov = run(main())
    assert met.decision.type == "met" and rest == []
    assert cov["fates"]["door"]["stale"] == 1


def test_pause_is_an_accounted_blind_interval_and_resume_closes_it():
    door = Condition("door", "the door is open", trigger=Always())
    judge = FakeJudge(answers({"detect": [N]}))

    async def main():
        async with percept.Monitor(judge, [door], store=mem(), max_requests_per_min=1000) as m:
            src = PushSource("cam")
            m.watch(src)
            await push_frames(src, 2)
            m.pause("door")
            sent = len(judge.sent)
            await push_frames(src, 5, start=2)
            paused = m.coverage("cam")
            m.resume("door")
            await push_frames(src, 2, start=7)
            return sent, paused, m.coverage("cam"), len(judge.sent)

    sent, paused, after, total = run(main())
    open_ = [b for b in paused["blind_intervals"] if b.kind == "paused"]
    assert len(open_) == 1 and open_[0].t_end_wall is None and open_[0].detail == "door"
    closed = [b for b in after["blind_intervals"] if b.kind == "paused"]
    assert len(closed) == 1 and closed[0].t_end_wall is not None
    assert total > sent


def test_snooze_keeps_judging_and_delivers_the_deferred_met_when_the_mute_ends():
    door = Condition("door", "the door is open", trigger=Never())
    judge = FakeJudge(answers({"detect": [Y]}))

    async def main():
        async with percept.Monitor(judge, [door], store=mem()) as m:
            src = PushSource("cam")
            watch = m.watch(src)
            src.push(image())
            await asyncio.sleep(0.05)
            m.snooze("door", time.time() + 0.4)
            m.poke("cam", "door", reason="look")
            await asyncio.sleep(0.1)
            looks_while_muted = len(judge.sent)
            assert m.stats()["streams"]["cam"]["decisions"] == 0      # suppressed, deferred
            got = await asyncio.wait_for(watch.__anext__(), 2.0)
            return looks_while_muted, got, len(judge.sent)

    looks, got, total = run(main())
    assert looks == 1 and total == 2                     # one more look when the mute ended
    assert (got.decision.type, got.decision.cause) == ("met", "deferred")


def test_default_store_is_sqlite_in_the_working_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    door = Condition("door", "the door is open", trigger=Never())

    async def main():
        m = percept.Monitor(FakeJudge(answers({"detect": [Y]})), [door])
        assert isinstance(m.store, SqliteStore) and m.store.path == str(tmp_path / "percept.db")
        src = PushSource("cam")
        watch = m.watch(src, ack="manual")
        src.push(image())
        await asyncio.sleep(0.05)
        m.poke("cam", "door", reason="look")
        d = await asyncio.wait_for(watch.__anext__(), 2)
        await m.aclose()
        return d

    d = run(main())
    store = SqliteStore(tmp_path / "percept.db")
    assert store.load(store_key("snapshot", "cam")) is not None
    assert any(store.load(store_key("outbox", "cam", f"#{n:012d}")) for n in range(3))
    assert d.attempt == 1
    store.close()


def test_health_stays_bounded_through_a_simulated_hour_long_outage():
    async def main():
        m = percept.Monitor(FakeJudge({}), [Condition("door", "x", trigger=Never())], store=mem())
        clock = [1_000_000.0]
        m._wall = lambda: clock[0]
        m.push("cam", image())
        for second in range(3600):
            clock[0] += 1.0
            m._source_health(Health("link_down" if second % 2 == 0 else "link_up", "cam", clock[0]))
            m._health.put(Health("judge_unavailable", "cam", clock[0], "timeout", condition_id=f"c{second}"))
        stats, cov = m.stats(), m.coverage("cam", 0.0)
        drained = []
        async def read():
            async for h in m.health():
                drained.append(h)
        reader = asyncio.ensure_future(read())
        await asyncio.sleep(0.01)
        await m.aclose()
        await reader
        return stats, cov, drained, m._health.items

    stats, cov, drained, left = run(main())
    assert stats["health"]["link_down"] == 1800 and stats["health"]["link_up"] == 1800
    assert stats["health"]["judge_unavailable"] == 3600
    assert len(drained) <= rt.HEALTH_RING and not left
    assert stats["health_dropped"] >= 3600 - rt.HEALTH_RING
    assert len(cov["blind_intervals"]) <= rt.MAX_BLIND + 1
    assert cov["blind_intervals_dropped"] == 1800 - rt.MAX_BLIND


class TickingReader(Reader):
    def __init__(self) -> None:
        self.i = 0

    def read(self):
        time.sleep(0.01)
        self.i += 1
        return image(self.i), None


def test_shutdown_order_and_no_leaked_threads_or_tasks(tmp_path):
    before = {t.name for t in threading.enumerate()}
    judge = SlowJudge(10.0)
    conds = [Condition("door", "the door is open", trigger=Always())]

    async def main():
        m = percept.Monitor(judge, conds, store=SqliteStore(tmp_path / "s.db"), max_requests_per_min=100)
        src = CaptureSource("live", TickingReader, live=True, stall_timeout=None)
        m.watch(src)
        await asyncio.sleep(0.3)
        frames = m.stats()["streams"]["live"]["frames"]
        await m.aclose()
        tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
        return frames, tasks, src

    frames, tasks, src = asyncio.run(main())
    assert frames > 5 and judge.cancelled == 1 and tasks == []
    assert src.link == "ended"
    after = {t.name for t in threading.enumerate()}
    assert not {n for n in after - before if n.startswith("percept")}


def test_exports_stay_within_budget():
    assert {"Monitor", "SqliteStore", "MemoryStore"} <= set(percept.__all__)
    assert len(percept.__all__) <= 30


# --------------------------------------------------------------------------- scheduling details


def test_priority_is_verify_then_event_then_state(monkeypatch):
    monkeypatch.setattr(rt, "BUDGET_WINDOW_S", 0.4)
    left = ((0.0, 0.0), (0.5, 0.0), (0.5, 1.0), (0.0, 1.0))
    right = ((0.5, 0.0), (1.0, 0.0), (1.0, 1.0), (0.5, 1.0))
    conds = [Condition("door", "the door is open", trigger=At(0), zones=(right,)),
             Condition("fall", "a person falls", kind="event", trigger=At(0), zones=(left,))]
    judge = FakeJudge(answers({"detect": [Y], "verify": [Y]}))   # max_attempts 2: one request per window

    async def main():
        async with percept.Monitor(judge, conds, store=mem(), max_requests_per_min=2) as m:
            src = PushSource("cam")
            watch = m.watch(src)
            src.push(image())
            await collect(watch, limit=2)

    run(main())
    order = [(r.items[0].condition.id, r.purpose) for _, _, _, r in judge.sent]
    assert order == [("fall", "detect"), ("fall", "verify"), ("door", "detect")]


def test_the_concurrency_limit_covers_every_stream_and_both_lanes():
    state = {"now": 0, "max": 0}

    class Counting:
        name, can_batch, price, max_attempts = "counting", True, None, 1

        async def judge(self, request):
            state["now"] += 1
            state["max"] = max(state["max"], state["now"])
            await asyncio.sleep(0.03)
            state["now"] -= 1
            return await FakeJudge(answers({"detect": [Y], "verify": [Y]})).judge(request)

    conds = [Condition("door", "the door is open", trigger=Always()),
             Condition("fall", "a person falls", kind="event", trigger=Always(), rearm_s=0.05)]

    async def main():
        async with percept.Monitor(Counting(), conds, store=mem(), max_requests_per_min=1000,
                                   max_concurrent_requests=1) as m:
            a, b = PushSource("a"), PushSource("b")
            m.watch(a), m.watch(b)
            for i in range(8):
                a.push(image(i)), b.push(image(i))
                await asyncio.sleep(0.03)
            return m.stats()

    stats = run(main())
    assert state["max"] == 1
    assert stats["streams"]["a"]["requests"] >= 2 and stats["streams"]["b"]["requests"] >= 2


def test_the_cost_budget_is_reserved_before_a_request_is_sent(monkeypatch):
    monkeypatch.setattr(rt, "DEADLINES", {rt.EVENT: 0.1, rt.STATE: 0.1})
    price = percept.PriceModel(usd_per_mtok_in=1000.0, usd_per_mtok_out=1000.0, tokens_per_image=1000)
    judge = FakeJudge(answers({"detect": [N]}), price=price)
    door = Condition("door", "the door is open", trigger=Always())

    async def main():
        async with percept.Monitor(judge, [door], store=mem(), max_requests_per_min=1000,
                                   max_cost_per_hour=4.0) as m:
            src = PushSource("cam")
            m.watch(src)
            await push_frames(src, 6, pause=0.05)
            return m.stats(), m.coverage("cam")

    stats, cov = run(main())
    assert len(judge.sent) == 1                         # a second worst case no longer fits the hour
    assert 0 < stats["streams"]["cam"]["cost_usd"] < 4.0
    assert cov["fates"]["door"]["dropped_budget"] >= 1
    assert any(b.kind == "budget_exhausted" for b in cov["blind_intervals"])


class EpochSource:
    """A custom Source (no pre-roll of its own) that reconnects once: epochs 0, 0, 1, 1."""

    stream_id = "ep"

    async def frames(self):
        for epoch, seq in ((0, 0), (0, 1), (1, 0), (1, 1)):
            yield percept.Frame("ep", epoch, seq, time.monotonic(), time.time(), image(seq))
            await asyncio.sleep(0.05)


def test_a_source_epoch_change_starts_a_new_decide_epoch():
    judge = FakeJudge(answers({"detect": [Y]}))
    door = Condition("door", "the door is open", trigger=Always())

    async def main():
        async with percept.Monitor(judge, [door], store=mem(), max_requests_per_min=1000) as m:
            watch = m.watch(EpochSource())
            got = await collect(watch)
            return got, m._streams["ep"].state.stream_epoch

    got, epoch = run(main())
    assert epoch == 1
    assert {r.stream_epoch for _, _, _, r in judge.sent} == {0, 1}
    assert [d.decision.type for d in got] == ["met"]             # held across the reconnect, no repeat


def test_a_restart_redelivers_pending_decisions_first_then_resolves_interrupted_verification():
    store = mem()
    door = Condition("door", "the door is open", trigger=Never())
    fall = Condition("fall", "a person falls", kind="event", trigger=Never())

    class Hanging:
        name, can_batch, price, max_attempts = "hang", True, None, 1

        async def judge(self, request):
            if request.purpose == "verify":
                await asyncio.sleep(3600)
            return await FakeJudge(answers({"detect": [Y]})).judge(request)

    async def before():
        m = percept.Monitor(Hanging(), [door, fall], store=store)
        src = PushSource("cam")
        m.watch(src)
        src.push(image())
        await asyncio.sleep(0.05)
        m.poke("cam", ["door"], reason="look")
        await asyncio.sleep(0.05)
        m.poke("cam", ["fall"], reason="look")
        await asyncio.sleep(0.1)          # the verification hangs
        await m.aclose()

    async def after():
        m = percept.Monitor(FakeJudge({}), [door, fall], store=store)
        src = PushSource("cam")
        watch = m.watch(src)
        src.close()
        got = [(d.decision.condition_id, d.decision.type, d.decision.cause, d.attempt) async for d in watch]
        await m.aclose()
        return got

    run(before())
    assert run(after()) == [("door", "met", None, 2), ("fall", "uncertain", "restart", 1)]


def test_a_scene_reset_is_a_blind_interval_followed_by_one_settled_look():
    from percept.triggers import SceneMonitor

    judge = FakeJudge(answers({"detect": [N]}))
    conds = [Condition("door", "the door is open", trigger=Never()),
             Condition("fall", "a person falls", kind="event", trigger=Never())]

    async def main():
        async with percept.Monitor(judge, conds, store=mem(), max_requests_per_min=1000) as m:
            m._scene = SceneMonitor(suppress_s=0.15)
            src = PushSource("cam")
            m.watch(src)
            for _ in range(3):
                src.push(image())
                await asyncio.sleep(0.03)
            for _ in range(8):
                src.push(255 - image())          # the camera was bumped: everything changed
                await asyncio.sleep(0.05)
            return m.coverage("cam"), m.stats()

    cov, stats = run(main())
    assert [b.kind for b in cov["blind_intervals"]] == ["scene_reset"]
    assert cov["blind_intervals"][0].t_end_wall is not None
    assert stats["health"]["scene_reset"] == 1
    assert len(judge.sent) == 1 and {i.condition.id for i in judge.sent[0][3].items} == {"door", "fall"}
