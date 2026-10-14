"""The runtime end to end, offline: PushSource or fake sources, FakeJudge, injected faults."""

from __future__ import annotations

import asyncio
import os
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
            watch = m.watch(src, fps=None)
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
            watch = m.watch(src, fps=None)
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
            watch = m.watch(src, fps=None)
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
            watch = m.watch(src, fps=None)
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
        budget = rt._Window(limit, lambda: rt.BUDGET_WINDOW_S)
        now, attempts, running = 0.0, [], []
        for gap, duration, frac in arrivals:
            now += gap
            for end, used, start in sorted(running):
                if end <= now:
                    budget.settle(reserve, used, end)
                    running.remove((end, used, start))
            if budget.fits(reserve, now):
                budget.take(reserve)
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
            watch = m.watch(src, fps=None)
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
            watch = m.watch(src, fps=None)
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
        m.watch(src, fps=None)
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
            m.watch(src, fps=None)
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
        watch = m.watch(src, fps=None)

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
        watch = m.watch(src, fps=None)
        src.close()
        got = [(d.decision.decision_id, d.attempt) async for d in watch]
        await m.aclose()
        return got

    run(before())
    return store, seen, run(after())


def last_attempt(store: MemoryStore, key: str) -> int:
    """The highest attempt number a decision may have been delivered with before the crash:
    its attempt record if it was redelivered, else 1, plus every restart since (meta ``inc``)."""
    import json
    stream = key[len("outbox:"):key.rindex("#")]
    meta = json.loads(store._data[f"outbox-meta:{stream}"])
    attempt = store._data.get("outbox-attempt:" + key[len("outbox:"):])
    if attempt is not None:
        att = json.loads(attempt)
        return att["attempt"] + meta.get("inc", 0) - att["inc"]
    return 1 + meta.get("inc", 0) - json.loads(store._data[key])["inc"]


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
        stored = {store.records[k]: last_attempt(durable, k) for k in pending}
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
            watch = m.watch(src, fps=None)
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
            watch = m.watch(src, fps=None, ack="manual", max_unacked=2)
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
            watch = m.watch(src, fps=None)
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
            m.watch(src, fps=None)
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
    assert len(open_) == 1 and open_[0].t_end_wall is None and open_[0].condition_id == "door"
    closed = [b for b in after["blind_intervals"] if b.kind == "paused"]
    assert len(closed) == 1 and closed[0].t_end_wall is not None
    assert total > sent


def test_snooze_keeps_judging_and_delivers_the_deferred_met_when_the_mute_ends():
    door = Condition("door", "the door is open", trigger=Never())
    judge = FakeJudge(answers({"detect": [Y]}))

    async def main():
        async with percept.Monitor(judge, [door], store=mem()) as m:
            src = PushSource("cam")
            watch = m.watch(src, fps=None)
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
        watch = m.watch(src, fps=None, ack="manual")
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
        m.watch(src, fps=None)
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
            watch = m.watch(src, fps=None)
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
            m.watch(a, fps=None), m.watch(b, fps=None)
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
            m.watch(src, fps=None)
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
            watch = m.watch(EpochSource(), fps=None)
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
        m.watch(src, fps=None)
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
        watch = m.watch(src, fps=None)
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
            m.watch(src, fps=None)
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


# --------------------------------------------------------------------------- review findings (PR7)


class LoadFails(MemoryStore):
    def __init__(self, failures: int) -> None:
        super().__init__()
        self.failures = failures

    def load(self, key):
        if self.failures:
            self.failures -= 1
            raise OSError("disk busy")
        return super().load(key)


def test_f1_a_transient_store_failure_at_restore_recovers_and_is_reported(monkeypatch):
    monkeypatch.setattr(rt, "SNAPSHOT_EVERY_S", 0.05)
    door = Condition("door", "the door is open", trigger=Always())

    async def main():
        async with percept.Monitor(FakeJudge(answers({"detect": [Y]})), [door], store=LoadFails(1),
                                   max_requests_per_min=1000) as m:
            src = PushSource("cam")
            watch = m.watch(src, fps=None)
            await push_frames(src, 4, pause=0.03)
            got = await asyncio.wait_for(watch.__anext__(), 2.0)
            await asyncio.sleep(0.1)
            return got, m.stats()

    got, stats = run(main())
    assert got.decision.type == "met"
    assert stats["store_failures"] == 1 and stats["health"]["store_lag"] >= 1
    assert stats["health"].get("link_down", 0) == 0          # a store problem is not a source failure


def test_f1_r1_a_permanent_store_failure_keeps_reporting_and_aclose_returns(monkeypatch):
    monkeypatch.setattr(rt, "SNAPSHOT_EVERY_S", 0.05)
    monkeypatch.setattr(Writer, "RETRY_FIRST_S", 0.01)
    monkeypatch.setattr(Writer, "RETRY_MAX_S", 0.05)
    door = Condition("door", "the door is open", trigger=Always())

    async def main():
        m = percept.Monitor(FakeJudge(answers({"detect": [Y]})), [door], store=LoadFails(10**9),
                            max_requests_per_min=1000)
        src = PushSource("cam")
        watch = m.watch(src, fps=None)
        waiting = asyncio.ensure_future(watch.__anext__())
        await push_frames(src, 3, pause=0.03)
        await asyncio.sleep(0.3)
        first = m.stats()
        await asyncio.sleep(0.3)
        second = m.stats()
        t0 = time.monotonic()
        await m.aclose()
        closed_in = time.monotonic() - t0
        with pytest.raises(StopAsyncIteration):
            await asyncio.wait_for(waiting, 1.0)              # ends with the monitor, never hangs
        return first, second, closed_in

    first, second, closed_in = run(main())
    assert first["store_retrying"] and second["store_failures"] > first["store_failures"]
    assert second["health"]["store_lag"] > first["health"]["store_lag"] >= 1     # keeps reporting
    assert closed_in < 2.0


def _seed_pending(store):
    door = Condition("door", "the door is open", trigger=Never())

    async def seed():
        m = percept.Monitor(FakeJudge(answers({"detect": [Y]})), [door], store=store)
        src = PushSource("cam")
        watch = m.watch(src, ack="manual")
        src.push(image())
        await asyncio.sleep(0.05)
        m.poke("cam", "door", reason="look")
        d = await asyncio.wait_for(watch.__anext__(), 2)
        await m.aclose()
        return d.decision.decision_id

    return run(seed())


async def _restart_first(store, timeout=1.0):
    m = percept.Monitor(FakeJudge({}), [Condition("door", "the door is open", trigger=Never())], store=store)
    watch = m.watch(PushSource("cam"))
    t0 = time.monotonic()
    try:
        d = await asyncio.wait_for(watch.__anext__(), timeout)
        return d.decision.decision_id, d.attempt, time.monotonic() - t0, m.stats()
    finally:
        await m.aclose()


def test_f5_pending_decisions_reach_a_waiting_consumer_promptly_after_a_restart():
    store = mem()
    did = _seed_pending(store)
    got, attempt, latency, _ = run(_restart_first(store))
    assert (got, attempt) == (did, 2) and latency < 0.5


def test_f1_a_corrupt_snapshot_is_kept_aside_and_pending_decisions_still_arrive():
    store = mem()
    did = _seed_pending(store)
    store.commit({store_key("snapshot", "cam"): b"{not json"})
    got, attempt, _, stats = run(_restart_first(store))
    assert (got, attempt) == (did, 2)
    assert stats["health"]["store_lag"] >= 1
    kept = [k for k in store._data if k.startswith("snapshot-unreadable:")]
    assert len(kept) == 1 and store._data[kept[0]] == b"{not json"


def test_f1_an_unreadable_outbox_record_halts_the_stream_without_overwriting_anything():
    store = mem()
    _seed_pending(store)
    meta = store_key("outbox-meta", "cam")
    store.commit({meta: b"{broken"})
    before = dict(store._data)

    async def main():
        async with percept.Monitor(FakeJudge({}), [Condition("door", "x", trigger=Never())], store=store) as m:
            src = PushSource("cam")
            watch = m.watch(src)
            src.push(image())
            got = await asyncio.wait_for(collect(watch), 2.0)     # ends: never hangs
            return got, m.stats()

    got, stats = run(main())
    assert got == [] and stats["streams"]["cam"]["halted"] and stats["health"]["store_lag"] >= 1
    assert store._data == before


def test_f2_a_stream_added_during_a_pause_records_it_and_resume_closes_it():
    door = Condition("door", "the door is open", trigger=Always())

    async def main():
        async with percept.Monitor(FakeJudge(answers({"detect": [N]})), [door], store=mem(),
                                   max_requests_per_min=1000) as m:
            a = PushSource("a")
            m.watch(a, fps=None)
            a.push(image())
            await asyncio.sleep(0.05)
            m.pause("door")
            b = PushSource("b")                 # added while paused
            m.watch(b, fps=None)
            await push_frames(b, 3, pause=0.03)
            during = m.coverage("b")["blind_intervals"]
            m.resume("door")
            return during, m.coverage("b")["blind_intervals"]

    during, after = run(main())
    assert [(x.kind, x.condition_id, x.t_end_wall) for x in during] == [("paused", "door", None)]
    assert [(x.kind, x.condition_id) for x in after] == [("paused", "door")] and after[0].t_end_wall is not None


class OldFrame:
    """A custom source whose only frame is 100 s old, then silence."""

    stream_id = "ep"

    async def frames(self):
        yield percept.Frame("ep", 0, 0, time.monotonic(), time.time() - 100.0, image())
        await asyncio.sleep(3600)


def test_f3_an_old_frame_is_never_judged_as_the_current_one():
    door = Condition("door", "the door is open", trigger=Always(), recheck_every_s=0.2)
    judge = FakeJudge(answers({"detect": [Y]}))

    async def main():
        async with percept.Monitor(judge, [door], store=mem(), max_requests_per_min=1000) as m:
            m.watch(OldFrame(), fps=None)
            await asyncio.sleep(0.5)
            return m.coverage("ep")

    cov = run(main())
    assert judge.sent == []
    assert cov["evidence_unavailable"]["door"] >= 1


def test_f3_rechecks_wait_while_the_link_is_down_and_resume_when_frames_return(monkeypatch):
    monkeypatch.setattr(rt, "DEADLINES", {rt.EVENT: 0.2, rt.STATE: 0.2})
    door = Condition("door", "the door is open", trigger=Never(), recheck_every_s=0.15)
    judge = FakeJudge(answers({"detect": [Y]}))

    async def main():
        async with percept.Monitor(judge, [door], store=mem(), max_requests_per_min=1000) as m:
            src = PushSource("cam")
            m.watch(src, fps=None)
            src.push(image())
            await asyncio.sleep(0.05)
            m.poke("cam", "door", reason="look")
            await asyncio.sleep(0.1)
            m._source_health(Health("link_down", "cam", time.time(), "gone"))
            sent = len(judge.sent)
            m.poke("cam", "door", reason="during the outage")
            await asyncio.sleep(0.6)                           # several rechecks fall due
            during = len(judge.sent)
            m._source_health(Health("link_up", "cam", time.time(), "back"))
            src.push(image(1))
            await asyncio.sleep(0.3)
            return sent, during, len(judge.sent), m.coverage("cam")

    sent, during, after, cov = run(main())
    assert during == sent                                  # nothing judged on the pre-outage frame
    assert after > during                                  # the waiting recheck went out
    assert cov["evidence_unavailable"]["door"] >= 1
    assert cov["fates"]["door"]["dropped_budget"] >= 1    # the outage poke expired, accounted


def test_f4_one_owner_per_database_and_private_files(tmp_path):
    path = tmp_path / "percept.db"
    door = Condition("door", "the door is open", trigger=Never())

    async def main():
        async with percept.Monitor(FakeJudge(answers({"detect": [Y]})), [door], store=SqliteStore(path)) as m1:
            src = PushSource("cam")
            watch = m1.watch(src, ack="manual")
            src.push(image())
            await asyncio.sleep(0.05)
            m1.poke("cam", "door", reason="look")
            await asyncio.wait_for(watch.__anext__(), 2)
            m2 = percept.Monitor(FakeJudge({}), [door], store=SqliteStore(path))
            with pytest.raises(RuntimeError, match="in use"):
                m2.watch(PushSource("cam"))
            await m2.aclose()

    run(main())
    assert oct(os.stat(path).st_mode & 0o777) == oct(0o600)


def test_f6_snooze_timers_and_trigger_states_do_not_grow():
    async def main():
        door = Condition("door", "the door is open", trigger=Never())
        async with percept.Monitor(FakeJudge({}), [door], store=mem()) as m:
            src = PushSource("cam")
            m.watch(src, fps=None)
            src.push(image())
            await asyncio.sleep(0.05)
            for k in range(500):
                m.snooze("door", time.time() + 0.001, subject=f"person-{k}")
            await asyncio.sleep(0.1)
            sizes = (len(m._mutes), len(m._snoozes))
            for k in range(50):
                m.add(Condition("door", "the door is open", trigger=percept.on_change(), version=k + 2))
                src.push(image(k))
                await asyncio.sleep(0.005)
            return sizes, len(m._streams["cam"].tstates), len(m._groups)

    (mutes, snoozes), tstates, groups = run(main())
    assert mutes == 0 and snoozes == 0
    assert tstates <= groups


def test_f7_r3_frames_are_capped_at_5_fps_by_default_and_the_skips_are_counted():
    door = Condition("door", "the door is open", trigger=Never())

    async def main():
        async with percept.Monitor(FakeJudge({}), [door], store=mem()) as m:
            src = PushSource("cam")
            m.watch(src)                                    # the default cap
            await push_frames(src, 40, pause=0.01)
            return m.stats()["streams"]["cam"], m.coverage("cam"), len(src.preroll)

    stats, cov, ring = run(main())
    processed = stats["frames"] - cov["frames_capped"]
    assert stats["frames"] == 40 and 1 <= processed <= 6       # ~0.5 s at 5 fps
    assert cov["frames_capped"] == stats["frames_capped"] == 40 - processed
    assert ring >= 1                                            # capture and the pre-roll unaffected


def test_f7_judge_images_are_encoded_off_the_event_loop(monkeypatch):
    from percept.judges import policy

    threads = []
    original = policy.to_jpeg

    def recording(image, **kw):
        threads.append(threading.current_thread() is threading.main_thread())
        return original(image, **kw)

    monkeypatch.setattr(policy, "to_jpeg", recording)
    door = Condition("door", "the door is open", trigger=Never())

    async def main():
        async with percept.Monitor(FakeJudge(answers({"detect": [N]})), [door], store=mem()) as m:
            src = PushSource("cam")
            m.watch(src)
            src.push(image())
            await asyncio.sleep(0.05)
            m.poke("cam", "door", reason="look")
            await asyncio.sleep(0.2)

    run(main())
    assert threads and not any(threads)


def test_r4_a_monitor_whose_consumer_never_reads_keeps_its_outbox_bounded(tmp_path):
    door = Condition("door", "the door is open", trigger=Always(), clear_after_no=1, emit_cleared=True)
    flip = iter(range(10**6))
    judge = FakeJudge(answers({"detect": lambda r: {i.condition.id: (Y if next(flip) % 2 == 0 else N)
                                                    for i in r.items}}))

    async def main():
        async with percept.Monitor(judge, [door], store=SqliteStore(tmp_path / "s.db"),
                                   max_requests_per_min=10_000) as m:
            src = PushSource("cam")
            m.watch(src, fps=None)                          # attached, never read
            for _ in range(300):                            # a still scene: no scene resets
                src.push(image())
                await asyncio.sleep(0.004)
            stream = m._streams["cam"]
            while not stream.commits.empty() or stream.commits._unfinished_tasks:
                await asyncio.sleep(0.02)
            ob = stream.outbox
            return len(ob.ahead), len(ob.held), m.stats()["streams"]["cam"]

    ahead, held, stats = run(main(), timeout=60)
    assert stats["decisions"] >= 3 * percept.delivery.READ_AHEAD and stats["unacked"] == stats["decisions"]
    assert ahead <= percept.delivery.READ_AHEAD and held == 0
    import sqlite3
    with sqlite3.connect(tmp_path / "s.db") as conn:
        on_disk = conn.execute("SELECT COUNT(*) FROM percept_kv WHERE key LIKE 'outbox:%'").fetchone()[0]
    assert on_disk == stats["decisions"]


def test_r2_the_dollar_cap_holds_for_the_whole_monitor_across_streams_and_cascades(monkeypatch):
    monkeypatch.setattr(rt, "COST_WINDOW_S", 1.0)
    monkeypatch.setattr(rt, "DEADLINES", {rt.EVENT: 0.3, rt.STATE: 0.3})
    price = percept.PriceModel(usd_per_mtok_in=1000.0, usd_per_mtok_out=1000.0, tokens_per_image=100)
    rng = np.random.default_rng(3)

    def lite(request):
        if rng.random() < 0.3:
            return RetryableError("busy")
        return {i.condition.id: ("unknown", 0.3, "?") for i in request.items}

    inner = cascade(FakeJudge(lite, name="lite", price=price),
                    FakeJudge(answers({"detect": [N], "verify": [N]}), name="pro", price=price))
    calls: list[tuple[float, float]] = []

    class Recording:
        name, can_batch, price, max_attempts = inner.name, inner.can_batch, inner.price, inner.max_attempts

        async def judge(self, request):
            t0 = time.monotonic()
            await asyncio.sleep(0.03)
            result = await inner.judge(request)
            calls.append((t0, sum(u.usage.cost_usd for u in result.usage)))
            return result

    cap = 12.0
    conds = [Condition("door", "the door is open", trigger=Always())]

    async def main():
        async with percept.Monitor(Recording(), conds, store=mem(), max_requests_per_min=10_000,
                                   max_cost_per_hour=cap, max_concurrent_requests=4) as m:
            srcs = [PushSource(f"cam{i}") for i in range(4)]
            for s in srcs:
                m.watch(s, fps=None)
            for k in range(90):
                for s in srcs:
                    s.push(image())
                await asyncio.sleep(0.025)
            return m.stats()

    stats = run(main())
    starts = sorted(calls)
    assert sum(c for _, c in calls) > cap                       # the cap binds over the run
    for i, (t, _) in enumerate(starts):
        assert sum(c for u, c in starts[i:] if u < t + 1.0) <= cap + 1e-9
    assert len({sid for sid, s in stats["streams"].items() if s["requests"]}) == 4


def test_unavailable_original_evidence_resolves_through_the_unverified_path_without_a_judgment(monkeypatch):
    fall = Condition("fall", "a person falls", kind="event", trigger=At(2), candidate_deadline_s=0.5)
    judge = FakeJudge(answers({"detect": [Y], "verify": [Y]}))

    async def main():
        async with percept.Monitor(judge, [fall], store=mem(), max_requests_per_min=1000) as m:
            src = PushSource("cam")
            watch = m.watch(src, fps=None)
            monkeypatch.setattr(m._streams["cam"], "original", lambda v: ())   # the frames are gone
            await push_frames(src, 4, t0=time.time() - 4)
            src.close()
            return await collect(watch), m.coverage("cam"), m.stats()["health"]

    got, cov, health = run(main())
    assert [(d.decision.type, d.decision.cause) for d in got] == [("uncertain", "timeout")]
    assert not [s for s in judge.sent if s[3].purpose == "verify"]     # nothing fabricated, nothing sent
    assert cov["evidence_unavailable"]["fall"] == 1 and health["verify_deferred"] >= 1


# --------------------------------------------------------------------------- link health of app-owned sources


class OutageReader(Reader):
    """A live camera: ``frames`` frames, then the connection drops."""

    def __init__(self, frames: int) -> None:
        self.left = frames

    def read(self):
        if self.left <= 0:
            raise ConnectionError("connection reset by peer")
        self.left -= 1
        time.sleep(0.01)
        return image(self.left), None


def outage_opener(down_for: int):
    """Connection 1 delivers frames then drops; the next ``down_for`` opens fail (the outage);
    then the camera is back for good."""
    script = iter([OutageReader(5)] + ["down"] * down_for)

    def opener():
        step = next(script, None)
        if step == "down":
            raise ConnectionError("no route to host")
        return step or OutageReader(10**6)

    return opener


async def wait_for(predicate, limit_s: float = 5.0):
    t0 = time.monotonic()
    while not predicate() and time.monotonic() - t0 < limit_s:
        await asyncio.sleep(0.02)
    assert predicate()


def link_intervals(cov):
    return [b for b in cov["blind_intervals"] if b.kind == "link_down"]


def test_d1_an_app_owned_source_reports_its_outage_to_the_monitor_without_private_calls(monkeypatch):
    from percept import sources

    monkeypatch.setattr(sources, "_cv2", lambda: None)              # no OpenCV needed: a fake camera
    opener = outage_opener(down_for=4)
    monkeypatch.setattr(sources, "Cv2Reader", lambda target, **kw: opener())
    door = Condition("door", "the door is open", trigger=Never())
    app_frames, app_health = [], []

    async def main():
        async with percept.Monitor(FakeJudge({}), [door], store=mem()) as m:
            # The app owns the source: it needs every frame for its own live view and clip ring.
            src = percept.open_source("rtsp://cam-1/stream", on_frame=app_frames.append,
                                      on_health=app_health.append, backoff=(0.05, 0.05), stall_timeout=None)
            m.watch(src)
            await wait_for(lambda: [h.kind for h in app_health] == ["link_down", "link_up"])
            await asyncio.sleep(0.05)
            return m.coverage(src.stream_id), m.stats()

    cov, stats = run(main())
    (gap,) = link_intervals(cov)
    assert gap.t_end_wall is not None and gap.t_end_wall - gap.t_start_wall >= 4 * 0.05 * 0.9
    assert stats["health"]["link_down"] == 1 and stats["health"]["link_up"] == 1
    assert app_frames and [h.kind for h in app_health] == ["link_down", "link_up"]   # the app still hears it


def test_d1_capture_is_configured_on_open_source_and_watch_reports_its_outages(monkeypatch):
    from percept import sources

    monkeypatch.setattr(sources, "_cv2", lambda: None)
    opener = outage_opener(down_for=2)
    monkeypatch.setattr(sources, "Cv2Reader", lambda target, **kw: opener())
    door = Condition("door", "the door is open", trigger=Never())
    app_frames, app_health = [], []

    async def main():
        async with percept.Monitor(FakeJudge({}), [door], store=mem()) as m:
            # The canonical form: open_source owns capture, watch owns monitoring and delivery.
            source = percept.open_source("rtsp://user:pw@cam-2/live", on_frame=app_frames.append,
                                         on_health=app_health.append, fps=50, max_side=32,
                                         stall_timeout=None, backoff=(0.05, 0.05))
            m.watch(source)
            with pytest.raises(TypeError):
                m.watch(source, on_frame=print)           # watch() takes no source options
            await wait_for(lambda: [h.kind for h in app_health] == ["link_down", "link_up"])
            await asyncio.sleep(0.05)
            return source, m.coverage(source.stream_id), m.stats()

    src, cov, stats = run(main())
    assert src.stream_id == "rtsp://cam-2/live"                   # credentials never in the id
    assert app_frames and max(app_frames[-1].image.shape) <= 32
    assert len(link_intervals(cov)) == 1 and stats["health"]["link_down"] == 1
    assert stats["streams"][src.stream_id]["frames"] > 0


def test_d1_a_uri_watched_with_open_source_defaults_also_reports_outages(monkeypatch):
    from percept import sources

    monkeypatch.setattr(sources, "_cv2", lambda: None)
    opener = outage_opener(down_for=1)
    monkeypatch.setattr(sources, "Cv2Reader", lambda target, **kw: opener())
    real = sources.CaptureSource.__init__

    def quick_backoff(self, *a, **kw):                  # the defaults back off 2 s; keep the test short
        kw["backoff"] = (0.05, 0.05)
        real(self, *a, **kw)

    monkeypatch.setattr(sources.CaptureSource, "__init__", quick_backoff)
    door = Condition("door", "the door is open", trigger=Never())

    async def main():
        async with percept.Monitor(FakeJudge({}), [door], store=mem()) as m:
            m.watch("rtsp://cam-3/live")
            await wait_for(lambda: m.stats()["health"].get("link_up") == 1)
            return m.coverage("rtsp://cam-3/live")

    cov = run(main())
    assert len(link_intervals(cov)) == 1


def test_d1b_a_push_stream_outage_is_a_blind_interval_and_health():
    door = Condition("door", "the door is open", trigger=Never(), recheck_every_s=0.1)

    async def main():
        async with percept.Monitor(FakeJudge({}), [door], store=mem()) as m:
            src = PushSource("conn")
            m.watch(src, fps=None)
            src.push(image())
            await asyncio.sleep(0.05)
            src.set_link(False, "connector gone")           # the app's own connector died
            await asyncio.sleep(0.15)
            src.push(image(1))                              # frames again: the link is back
            await asyncio.sleep(0.05)
            m.push("other", image())                        # a monitor-owned push stream
            m.set_link("other", False, "camera unplugged")
            await asyncio.sleep(0.1)
            m.set_link("other", True)
            await asyncio.sleep(0.02)
            return m.coverage("conn"), m.coverage("other"), m.stats()

    conn, other, stats = run(main())
    (gap,) = link_intervals(conn)
    assert gap.detail == "connector gone" and gap.t_end_wall - gap.t_start_wall >= 0.1
    (gap,) = link_intervals(other)
    assert gap.t_end_wall is not None and gap.detail == "camera unplugged"
    assert stats["health"]["link_down"] == 2 and stats["health"]["link_up"] == 2


# --------------------------------------------------------------------------- cost accounting


PRICE = percept.PriceModel(usd_per_mtok_in=100.0, usd_per_mtok_out=400.0, tokens_per_image=300)


def test_d5_a_cancelled_request_is_reported_at_its_settled_cost_not_its_reservation():
    door = Condition("door", "the door is open", trigger=Always())
    inner = FakeJudge(answers({"detect": [N]}), price=PRICE)
    reported: list[float] = []
    hang = asyncio.Event

    class SecondHangs:
        name, can_batch, price, max_attempts = "fake", True, PRICE, 2

        def __init__(self):
            self.calls = 0

        async def judge(self, request):
            self.calls += 1
            if self.calls > 1:
                await hang().wait()                     # in flight when the monitor stops
            result = await inner.judge(request)
            reported.append(sum(u.usage.cost_usd for u in result.usage))
            return result

    async def main():
        m = percept.Monitor(SecondHangs(), [door], store=mem(), max_requests_per_min=1000, max_cost_per_hour=50.0)
        src = PushSource("cam")
        m.watch(src, fps=None)
        for i in range(4):
            src.push(image(i))
            await asyncio.sleep(0.05)
        in_flight = m.stats()
        await m.aclose()
        return in_flight, m.stats()

    in_flight, after = run(main())
    (spent,) = reported
    cam = after["streams"]["cam"]
    assert cam["cost_usd"] == pytest.approx(spent)                   # what the provider reported
    assert in_flight["streams"]["cam"]["cost_reserved_usd"] > 0      # the one in flight holds its worst case
    assert cam["cost_reserved_usd"] > 2 * spent                      # ...and keeps it for the window
    assert after["cost_usd_in_window"] == pytest.approx(spent + cam["cost_reserved_usd"])   # the cap still counts it
    assert after["cost_usd"] == pytest.approx(spent)


def test_d2_cost_attempts_and_requests_per_condition():
    conds = [Condition("door", "the door is open", trigger=At(0)),
             Condition("gate", "the gate is open", trigger=At(0)),
             Condition("fire", "there is fire", trigger=At(1))]

    def script(request):
        if request.items[0].condition.id == "fire" and not script.retried:
            script.retried = True
            return RetryableError("busy")
        return {i.condition.id: N for i in request.items}

    script.retried = False
    judge = FakeJudge(script, price=PRICE)

    async def main():
        async with percept.Monitor(judge, conds, store=mem(), max_requests_per_min=1000) as m:
            src = PushSource("cam")
            m.watch(src, fps=None)
            src.push(image(0))
            await asyncio.sleep(0.1)
            src.push(image(1))
            await asyncio.sleep(0.1)
            return m.stats()["streams"]["cam"]

    cam = run(main())
    per = cam["conditions"]
    assert set(per) == {"door", "gate", "fire"}
    assert per["door"]["requests"] == per["gate"]["requests"] == per["fire"]["requests"] == 1
    assert per["fire"]["attempts"] == 2 and per["door"]["attempts"] + per["gate"]["attempts"] == 1
    assert per["door"]["cost_usd"] == pytest.approx(per["gate"]["cost_usd"]) and per["door"]["cost_usd"] > 0
    assert sum(p["cost_usd"] for p in per.values()) == pytest.approx(cam["cost_usd"])
    assert sum(p["attempts"] for p in per.values()) == cam["attempts"]


# --------------------------------------------------------------------------- reservation per reachable model


def three_members(price=None, retries=1):
    """A three-member cascade whose members answer confidently (nothing escalates)."""
    reply = answers({"detect": [N], "verify": [N]})
    return cascade(*(FakeJudge(reply, name=n, price=price, retries=retries) for n in ("lite", "flash", "pro")))


def test_a_request_pinned_to_one_route_reserves_only_that_members_attempts(monkeypatch):
    monkeypatch.setattr(rt, "BUDGET_WINDOW_S", 1.0)
    judge = three_members()
    assert judge.max_attempts == 6
    door = Condition("door", "the door is open", trigger=Always(), judge="lite")

    async def main():
        async with percept.Monitor(judge, [door], store=mem(), max_requests_per_min=6) as m:
            src = PushSource("cam")
            m.watch(src, fps=None)
            await push_frames(src, 10, pause=0.03)
            return m.stats(), m.coverage("cam")

    stats, cov = run(main())
    sent = [s for member in judge.members for s in member.sent]
    assert {s[3].items[0].condition.judge for s in sent} == {"lite"}
    assert len(sent) >= 4                       # a request reserves lite's 2 attempts, not all 6
    assert stats["streams"]["cam"]["requests"] == stats["streams"]["cam"]["attempts"] == len(sent)


def test_reservation_hooks_cover_every_route_and_fall_back_for_plain_judges():
    from percept.contracts import JudgeItem, JudgeRequest
    from percept.judges.base import max_attempts_for, worst_case_cost, worst_case_cost_for

    lite, flash, pro = (FakeJudge({}, name=n, price=PRICE, retries=r) for n, r in (("lite", 0), ("flash", 1), ("pro", 2)))
    judge = cascade(lite, flash, pro)
    img = (np.zeros((8, 8, 3), np.uint8),)

    def req(*pins, purpose="detect"):
        items = tuple(JudgeItem(Condition(f"c{k}", "x", judge=p)) for k, p in enumerate(pins))
        return JudgeRequest("r", "cam", 0, items, img, (0.0, 0.0), purpose)

    assert max_attempts_for(judge, req("lite")) == 1
    assert max_attempts_for(judge, req("lite", "pro")) == 4
    assert max_attempts_for(judge, req(None)) == 6 == judge.max_attempts
    assert max_attempts_for(judge, req(None, "lite", purpose="verify")) == 3     # the last member only
    assert max_attempts_for(judge, req("nobody")) == 0                           # fails without asking
    r = req("flash")
    assert worst_case_cost_for(judge, r) == pytest.approx(worst_case_cost(r, PRICE, retries=1))
    assert worst_case_cost_for(lite, r) == pytest.approx(worst_case_cost(r, PRICE, retries=0))
    assert worst_case_cost_for(FakeJudge({}), r) is None                         # no price: best effort
    assert max_attempts_for(SlowJudge(0), r) == 1                                # no hook: max_attempts
    assert worst_case_cost_for(cascade(lite, FakeJudge({}, name="free")), req("lite")) == \
        pytest.approx(worst_case_cost(req("lite"), PRICE, retries=0))            # unreachable member unpriced


def test_reservations_bound_every_cascade_request_with_escalation_and_retries():
    import hypothesis
    from percept.contracts import JudgeItem, JudgeRequest
    from percept.judges.base import max_attempts_for, worst_case_cost_for

    st = hypothesis.strategies
    img = (np.zeros((8, 8, 3), np.uint8),)
    replies = st.sampled_from(["yes", "no", "unknown", "weak", "busy", "refused", "junk"])

    @hypothesis.settings(max_examples=200, deadline=None)
    @hypothesis.given(retries=st.lists(st.integers(0, 2), min_size=1, max_size=3),
                      pins=st.lists(st.sampled_from([None, 0, 1, 2, "nobody"]), min_size=1, max_size=4),
                      verify=st.booleans(), batch=st.booleans(), script=st.lists(replies, min_size=1, max_size=30))
    def check(retries, pins, verify, batch, script):
        steps = iter(script)

        def reply(request):
            r = next(steps, "yes")
            if r == "busy":
                return RetryableError("busy")
            if r == "refused":
                return percept.judges.base.RefusalError("no", tokens_in=50)
            if r == "junk":
                return "not json"
            answer, conf = ("yes", 0.3) if r == "weak" else (r, 0.9)
            return {i.condition.id: (answer, conf, "r") for i in request.items}

        names = [f"m{k}" for k in range(len(retries))]
        judge = cascade(*(FakeJudge(reply, name=n, price=PRICE, retries=r, can_batch=batch or k > 0)
                          for k, (n, r) in enumerate(zip(names, retries))))
        items = tuple(JudgeItem(Condition(f"c{k}", "x",
                                          judge=p if p in (None, "nobody") else names[min(p, len(names) - 1)]))
                      for k, p in enumerate(pins))
        request = JudgeRequest("r", "cam", 0, items, img, (0.0, 0.0), "verify" if verify else "detect")
        result = asyncio.run(judge.judge(request))
        calls = 1 if judge.can_batch else len(items)
        assert sum(u.usage.attempts for u in result.usage) <= max_attempts_for(judge, request) * calls
        assert sum(u.usage.cost_usd for u in result.usage) <= worst_case_cost_for(judge, request) * calls + 1e-12
        assert max_attempts_for(judge, request) <= judge.max_attempts

    check()


def test_a_busy_pinned_cascade_never_exceeds_the_window_or_the_dollar_cap(monkeypatch):
    monkeypatch.setattr(rt, "BUDGET_WINDOW_S", 1.0)
    monkeypatch.setattr(rt, "COST_WINDOW_S", 1.0)
    monkeypatch.setattr(rt, "DEADLINES", {rt.EVENT: 0.3, rt.STATE: 0.3})
    monkeypatch.setattr(rt, "VERIFY_RETRY_S", 0.05)
    rng = np.random.default_rng(11)
    attempts: list[tuple[float, str]] = []
    cheap = percept.PriceModel(usd_per_mtok_in=10.0, usd_per_mtok_out=40.0, tokens_per_image=300)

    def member(confident):
        def script(request):
            attempts.append((time.monotonic(), request.stream_id))
            if rng.random() < 0.3:
                return RetryableError("busy")
            return {i.condition.id: (Y if confident else ("unknown", 0.2, "?")) for i in request.items}
        return script

    chain = cascade(FakeJudge(member(False), name="lite", price=cheap),
                    FakeJudge(member(False), name="flash", price=cheap),
                    FakeJudge(member(True), name="pro", price=cheap))
    spent: list[tuple[float, float]] = []

    class Recording:
        name, can_batch, price, max_attempts = chain.name, chain.can_batch, chain.price, chain.max_attempts
        max_attempts_for, worst_case_cost_for = chain.max_attempts_for, chain.worst_case_cost_for
        members = chain.members

        async def judge(self, request):
            t0 = time.monotonic()
            result = await chain.judge(request)
            spent.append((t0, sum(u.usage.cost_usd for u in result.usage)))
            return result

    conds = [Condition("door", "the door is open", trigger=Always(), judge="lite", clear_after_no=1),
             Condition("smoke", "there is smoke", trigger=Always(), judge="pro"),
             Condition("fall", "a person falls", kind="event", trigger=Always(), rearm_s=0.2),     # escalates
             Condition("gate", "the gate is open", trigger=Always())]
    limit, cap = 8, 0.3

    async def main():
        async with percept.Monitor(Recording(), conds, store=mem(), max_requests_per_min=limit,
                                   max_cost_per_hour=cap) as m:
            srcs = [PushSource("a"), PushSource("b")]
            watches = [m.watch(s, fps=None) for s in srcs]
            for i in range(90):
                for s in srcs:
                    s.push(image(i))
                await asyncio.sleep(0.02)
            for s in srcs:
                s.close()
            for w in watches:
                await collect(w)
            return m.stats()

    stats = run(main(), timeout=60)
    for sid in ("a", "b"):
        times = sorted(t for t, s in attempts if s == sid)
        for i, t in enumerate(times):
            assert sum(1 for u in times[i:] if u < t + 1.0) <= limit
    assert stats["health"]["budget_exhausted"] >= 1
    starts = sorted(spent)
    for i, (t, _) in enumerate(starts):
        assert sum(c for u, c in starts[i:] if u < t + 1.0) <= cap + 1e-12
    assert sum(s["attempts"] for s in stats["streams"].values()) == len(attempts)


# --------------------------------------------------------------------------- review follow-ups


def test_d2_routed_conditions_are_charged_for_the_member_they_were_asked():
    cheap = percept.PriceModel(usd_per_mtok_in=1.0, usd_per_mtok_out=1.0, tokens_per_image=300)
    dear = percept.PriceModel(usd_per_mtok_in=100.0, usd_per_mtok_out=100.0, tokens_per_image=300)
    reply = answers({"detect": [Y], "verify": [Y]})
    judge = cascade(FakeJudge(reply, name="lite", price=cheap), FakeJudge(reply, name="pro", price=dear))
    conds = [Condition("door", "the door is open", trigger=At(0), judge="lite"),
             Condition("fire", "there is fire", trigger=At(0), judge="pro")]

    async def main():
        async with percept.Monitor(judge, conds, store=mem(), max_requests_per_min=1000) as m:
            src = PushSource("cam")
            watch = m.watch(src, fps=None)
            src.push(image())
            got = await collect(watch, limit=2)
            return got, m.stats()["streams"]["cam"]

    got, cam = run(main())
    (lite_sent,), (pro_sent,) = judge.members[0].sent, judge.members[1].sent   # one request, one hop each
    per = cam["conditions"]
    assert per["fire"]["cost_usd"] > 50 * per["door"]["cost_usd"] > 0          # pro is nearly all of it
    assert per["door"]["cost_usd"] + per["fire"]["cost_usd"] == pytest.approx(cam["cost_usd"])
    by_condition = {d.decision.condition_id: d.decision.cost for d in got}       # decide sees the same split
    assert by_condition["fire"].cost_usd == pytest.approx(per["fire"]["cost_usd"])
    assert by_condition["door"].cost_usd == pytest.approx(per["door"]["cost_usd"])


def test_link_health_from_a_worker_thread_is_handled_on_the_loop():
    door = Condition("door", "the door is open", trigger=Never())
    threads = []

    async def main():
        m = percept.Monitor(FakeJudge({}), [door], store=mem())
        report = m._report
        m._report = lambda h: (threads.append(threading.current_thread().name), report(h))
        src = PushSource("conn")
        m.watch(src, fps=None)                  # its runner has not iterated frames() yet
        worker = threading.Thread(target=src.set_link, args=(False, "gone"), name="worker")
        worker.start()
        worker.join()
        await asyncio.sleep(0.05)
        cov = m.coverage("conn")
        await m.aclose()
        return cov

    cov = run(main())
    assert threads == ["MainThread"] and len(link_intervals(cov)) == 1


def test_a_source_that_is_already_down_when_watched_reports_link_down():
    async def main():
        async with percept.Monitor(FakeJudge({}), [Condition("door", "x", trigger=Never())], store=mem()) as m:
            src = PushSource("conn")
            src.set_link(False, "connector not ready")
            m.watch(src, fps=None)
            await asyncio.sleep(0.02)
            return m.stats(), m.coverage("conn")

    stats, cov = run(main())
    assert stats["health"].get("link_down") == 1 and len(link_intervals(cov)) == 1


def test_a_judge_that_raises_keeps_its_reservation_charged_and_costs_nothing(monkeypatch):
    monkeypatch.setattr(rt, "BUDGET_WINDOW_S", 1.0)
    calls = []

    class Raises:
        name, can_batch, price, max_attempts = "raises", True, PRICE, 2

        async def judge(self, request):
            calls.append(request.request_id)
            raise RuntimeError("broken adapter")

    door = Condition("door", "the door is open", trigger=Always())

    async def main():
        async with percept.Monitor(Raises(), [door], store=mem(), max_requests_per_min=4,
                                   max_cost_per_hour=1000.0) as m:
            src = PushSource("cam")
            m.watch(src, fps=None)
            await push_frames(src, 10, pause=0.03)
            return m.stats()

    stats = run(main())
    cam = stats["streams"]["cam"]
    assert len(calls) == 2                           # 2 x 2 reserved attempts fill the window
    assert cam["attempts"] == 4 and cam["cost_usd"] == 0.0 and cam["cost_reserved_usd"] > 0
    assert stats["health"]["judge_unavailable"] >= 1


def test_the_output_token_limit_grows_with_the_conditions_in_a_request():
    import json
    import math

    from percept.judges.base import worst_case_cost_for

    conds = [Condition(f"check_{k}", f"condition number {k} holds", trigger=At(0)) for k in range(10)]
    reserved, box = [], []

    class Recording:
        name, can_batch, price, max_attempts = "rec", True, PRICE, 2

        def __init__(self):
            self.inner = FakeJudge(answers({"detect": [N]}), price=PRICE)
            self.requests = []

        async def judge(self, request):
            self.requests.append(request)
            reserved.append(box[0].stats()["cost_reserved_usd"])
            return await self.inner.judge(request)

    judge = Recording()

    async def main():
        async with percept.Monitor(judge, conds, store=mem(), max_requests_per_min=1000) as m:
            box.append(m)
            src = PushSource("cam")
            m.watch(src, fps=None)
            src.push(image())
            await asyncio.sleep(0.1)

    run(main())
    (request,) = judge.requests
    assert len(request.items) == 10
    full = json.dumps({"results": [{"id": c.id, "answer": "unknown", "confidence": 0.95, "reason": "r" * 160}
                                   for c in conds]})
    assert request.max_output_tokens >= math.ceil(len(full) / 3)     # ten full answers fit
    assert reserved == [pytest.approx(worst_case_cost_for(judge, request))]
    one = rt._max_output_tokens(1)
    assert one >= 400 and rt._max_output_tokens(1000) <= rt.MAX_OUTPUT_TOKENS


def test_every_dropped_look_is_traced_with_its_condition_trigger_time_and_reason(monkeypatch, tmp_path):
    import json

    monkeypatch.setattr(rt, "DEADLINES", {rt.EVENT: 0.1, rt.STATE: 0.1})
    price = percept.PriceModel(usd_per_mtok_in=1000.0, usd_per_mtok_out=1000.0, tokens_per_image=1000)
    judge = FakeJudge(answers({"detect": [N]}), price=price)
    door = Condition("door", "the door is open", trigger=Always())
    path = tmp_path / "trace.jsonl"

    async def main():
        async with percept.Monitor(judge, [door], store=mem(), max_requests_per_min=1000,
                                   max_cost_per_hour=4.0, trace=path) as m:
            src = PushSource("cam")
            m.watch(src, fps=None)
            await push_frames(src, 6, pause=0.05)
            return m.coverage("cam")

    cov = run(main())
    records = [json.loads(line) for line in path.read_text().splitlines()]
    drops = [r for r in records if r["kind"] == "drop"]
    assert len(drops) == cov["fates"]["door"]["dropped_budget"] >= 1
    for r in drops:
        assert r["stream_id"] == "cam" and r["condition_id"] == "door" and r["subject"] == "scene"
        assert r["reason"] == "budget" and r["purpose"] == "detect" and isinstance(r["t_trigger"], float)
        assert r["waited_s"] >= 0.1
    assert any(r["kind"] == "fate" and r.get("event") == "runtime" and r["fate"] == "dropped_budget"
               for r in records)                                      # the per-minute counts stay


class From:
    """Due on every frame from this ``seq`` on."""

    def __init__(self, seq: int) -> None:
        self.seq = seq

    def step(self, frame, conditions, state, now):
        return ({c.id: "from" for c in conditions} if frame.seq >= self.seq else {}), None


def _verify_retry_scenario(monkeypatch, deadlines=None):
    """A fall whose first verification fails and is asked again after VERIFY_RETRY_S, while two
    busy state conditions become due (from frame 3) and compete for the same request window."""
    monkeypatch.setattr(rt, "BUDGET_WINDOW_S", 1.0)
    monkeypatch.setattr(rt, "VERIFY_RETRY_S", 0.3)
    if deadlines:
        monkeypatch.setattr(rt, "DEADLINES", deadlines)
    thirds = [((x, 0.0), (x + 1 / 3, 0.0), (x + 1 / 3, 1.0), (x, 1.0)) for x in (0.0, 1 / 3, 2 / 3)]
    fall = Condition("fall", "a person falls", kind="event", trigger=At(1), zones=(thirds[0],),
                     candidate_deadline_s=0.9)
    busy = [Condition(c, f"state {c}", trigger=From(3), zones=(z,)) for c, z in zip("bc", thirds[1:])]
    base = answers({"detect": [N], "verify": [RetryableError("down"), Y]})

    def script(request):     # the fall's detection says yes; the busy states say no
        if request.purpose == "detect" and request.items[0].condition.id == "fall":
            return {"fall": Y}
        return base(request)
    judge = FakeJudge(script, latency_s=0.05, retries=0)
    seen = []

    async def main():
        async with percept.Monitor(judge, [fall, *busy], store=mem(), max_requests_per_min=4) as m:
            src = PushSource("cam")
            watch = m.watch(src, fps=None)
            feeder = asyncio.create_task(push_frames(src, 60, pause=0.05))

            async def health():
                async for h in m.health():
                    seen.append(h)
            ht = asyncio.create_task(health())
            got = None
            async for d in watch:
                if d.decision.condition_id == "fall":
                    got = d.decision
                    break
            await feeder
            ht.cancel()
            return got, m.coverage("cam")

    got, cov = run(main())
    return got, cov, seen, judge


def test_a_waiting_verification_keeps_its_attempts_from_new_detection(monkeypatch):
    # Without the kept room, the busy detections take the attempts in the window while the
    # failed verification waits to retry, and the fall closes uncertain at its deadline.
    got, cov, seen, judge = _verify_retry_scenario(monkeypatch)
    assert got.type == "met", (got.type, got.cause, got.reason)
    assert any(h.kind == "budget_exhausted" and "held for a waiting verification" in h.detail for h in seen)
    busy_detects = [r for _, _, _, r in judge.sent if r.purpose == "detect" and r.items[0].condition.id != "fall"]
    assert len(busy_detects) >= 3                    # detection kept running: held, never starved


def test_held_detections_past_their_deadline_count_as_dropped_for_verification():
    cond = Condition("door", "the door is open", trigger=Never())

    async def main():
        async with percept.Monitor(FakeJudge(answers({"detect": [N], "verify": [Y]})), [cond], store=mem()) as m:
            src = PushSource("cam")
            m.watch(src, fps=None)
            src.push(image())
            await asyncio.sleep(0.1)
            stream = m._streams["cam"]
            now = m._mono()
            for held, key in ((True, "s1"), (False, "s2")):
                w = rt._Work(rt.STATE, "door", key, now - 5.0, deadline=now - 1.0)
                w.blocked, w.held = True, held
                stream.pending[("door", key)] = w
            stream.dispatch(now)
            return m.coverage("cam")["fates"]["door"]

    fates = run(main())
    assert fates.get("dropped_for_verification") == 1 and fates.get("dropped_budget") == 1

# A verification task, however it ends, returns its slot and reservations exactly once
# (test_a_dead_verifier_ends_in_uncertain_at_the_deadline used to hang about once in 50 runs under
# load: the deadline timer cancelled a verification task before its first step, so its finally,
# the only cleanup, never ran and the stream never finished). Each case below forces one ordering.

class Gated(FakeJudge):
    """A FakeJudge whose verify requests wait on ``gate`` (when set) and are counted."""

    def __init__(self, script) -> None:
        super().__init__(script)
        self.gate: asyncio.Event | None = None
        self.started = {"detect": 0, "verify": 0}

    async def judge(self, request):
        self.started[request.purpose] += 1
        if request.purpose == "verify" and self.gate is not None:
            await self.gate.wait()
        return await super().judge(request)


FALL = Condition("fall", "a person falls", kind="event", trigger=At(2), candidate_deadline_s=60)
TIMER = ("fall", "scene", "verify")


async def _parked_verification(m, judge, stream_id="cam"):
    """A stream in VERIFYING whose verification failed once and waits to retry (not in flight)."""
    src = PushSource(stream_id)
    watch = m.watch(src, fps=None)
    await push_frames(src, 4)
    stream = m._streams[stream_id]
    for _ in range(500):
        if stream.verifies and stream.verify_task is None and judge.started["verify"]:
            break
        await asyncio.sleep(0.01)
    (w,) = stream.verifies.values()
    return src, watch, stream, w


def _settled(m, stream):
    """No slot, attempt or dollar is still held for a request in flight."""
    return m._slots == 0 and stream.budget.held == 0 and stream.held_cost.held == 0 and m._cost.held == 0


async def _ends_and_frees_capacity(m, src, watch, judge):
    """The watch ends with the timeout, and a second stream then gets the (single) slot."""
    src.close()
    got = await asyncio.wait_for(collect(watch), 5)
    assert [(d.decision.type, d.decision.cause) for d in got] == [("uncertain", "timeout")]
    before = judge.started["detect"]
    src2 = PushSource("cam2")
    m.watch(src2, fps=None)
    await push_frames(src2, 4, t0=m._wall())        # on the monitor's clock, which the test moved
    for _ in range(500):
        if judge.started["detect"] > before:
            break
        await asyncio.sleep(0.01)
    assert judge.started["detect"] > before


def test_a_verification_cancelled_before_its_first_step_is_settled(monkeypatch):
    monkeypatch.setattr(rt, "VERIFY_RETRY_S", 1000)
    judge = Gated(answers({"detect": [Y], "verify": [RetryableError("down")]}))

    async def main():
        async with percept.Monitor(judge, [FALL], store=mem(), max_requests_per_min=1000,
                                   max_concurrent_requests=1) as m:
            src, watch, stream, w = await _parked_verification(m, judge)
            requests, verifies = stream.totals["requests"], judge.started["verify"]
            w.not_before = 0.0
            stream.dispatch(m._mono())                  # the retry starts a task...
            task = stream.verify_task
            assert task is not None and not task.done()
            wall = m._wall
            m._wall = lambda: wall() + 120
            stream.fire(TIMER)                          # ...the deadline cancels it before it runs
            for _ in range(5):
                await asyncio.sleep(0)
            assert task.cancelled() and judge.started["verify"] == verifies       # it never ran
            assert stream.verify_task is None and _settled(m, stream)
            assert stream.totals["requests"] == requests + 1       # charged once, at its reservation
            await _ends_and_frees_capacity(m, src, watch, judge)

    run(main())


def test_a_verification_cancelled_during_the_call_is_settled_once(monkeypatch):
    monkeypatch.setattr(rt, "VERIFY_RETRY_S", 1000)
    judge = Gated(answers({"detect": [Y], "verify": [RetryableError("down")]}))

    async def main():
        async with percept.Monitor(judge, [FALL], store=mem(), max_requests_per_min=1000,
                                   max_concurrent_requests=1) as m:
            src, watch, stream, w = await _parked_verification(m, judge)
            requests, verifies = stream.totals["requests"], judge.started["verify"]
            judge.gate = asyncio.Event()                # the next verification hangs at the provider
            w.not_before = 0.0
            stream.wake.set()
            for _ in range(500):
                if judge.started["verify"] > verifies:
                    break
                await asyncio.sleep(0.01)
            task = stream.verify_task
            assert task is not None and not task.done()
            wall = m._wall
            m._wall = lambda: wall() + 120
            stream.fire(TIMER)
            for _ in range(5):
                await asyncio.sleep(0)
            assert task.cancelled() and stream.verify_task is None and _settled(m, stream)
            assert stream.totals["requests"] == requests + 1
            await _ends_and_frees_capacity(m, src, watch, judge)

    run(main())


def test_a_verification_past_its_deadline_is_never_started(monkeypatch):
    monkeypatch.setattr(rt, "VERIFY_RETRY_S", 1000)
    judge = Gated(answers({"detect": [Y], "verify": [RetryableError("down")]}))

    async def main():
        async with percept.Monitor(judge, [FALL], store=mem(), max_requests_per_min=1000,
                                   max_concurrent_requests=1) as m:
            src, watch, stream, w = await _parked_verification(m, judge)
            requests, verifies = stream.totals["requests"], judge.started["verify"]
            w.not_before = 0.0
            wall = m._wall
            m._wall = lambda: wall() + 120              # the deadline passed; its timer is late
            stream.dispatch(m._mono())
            assert stream.verify_task is None and _settled(m, stream)      # nothing taken
            for _ in range(5):
                await asyncio.sleep(0)
            assert judge.started["verify"] == verifies and stream.totals["requests"] == requests
            stream.fire(TIMER)
            await _ends_and_frees_capacity(m, src, watch, judge)

    run(main())


def test_a_verification_that_finished_is_not_settled_again(monkeypatch):
    judge = Gated(answers({"detect": [Y], "verify": [Y]}))
    calls = []
    settle = rt._Stream.settle

    def spy(self, f, task=None):
        calls.append((self, f))
        settle(self, f, task)

    monkeypatch.setattr(rt._Stream, "settle", spy)

    async def main():
        async with percept.Monitor(judge, [FALL], store=mem(), max_requests_per_min=1000,
                                   max_concurrent_requests=1) as m:
            src = PushSource("cam")
            watch = m.watch(src, fps=None)
            await push_frames(src, 4)
            src.close()
            got = await asyncio.wait_for(collect(watch), 5)
            for _ in range(5):                          # let the last done-callbacks run
                await asyncio.sleep(0)
            flights = {id(f): (stream, f) for stream, f in calls}
            # each request settles from its finally and again from its done-callback: counted once
            assert len(calls) == 2 * len(flights) == 2 * sum(judge.started.values())
            for stream, f in flights.values():          # a late cancel or settle changes nothing
                f.task.cancel()
                settle(stream, f, f.task)
            return got, m._slots, m.stats()["streams"]["cam"]["requests"], len(flights)

    got, slots, requests, flights = run(main())
    assert [d.decision.type for d in got] == ["met"] and slots == 0 and requests == flights == 2


def test_a_cancel_racing_the_answer_settles_once_and_ends_in_the_timeout(monkeypatch):
    monkeypatch.setattr(rt, "VERIFY_RETRY_S", 1000)
    judge = Gated(answers({"detect": [Y], "verify": [Y]}))

    async def main():
        async with percept.Monitor(judge, [FALL], store=mem(), max_requests_per_min=1000,
                                   max_concurrent_requests=1) as m:
            judge.gate = asyncio.Event()                # the verification waits at the provider
            src = PushSource("cam")
            watch = m.watch(src, fps=None)
            await push_frames(src, 4)
            stream = m._streams["cam"]
            for _ in range(500):
                if stream.verify_task is not None and judge.started["verify"]:
                    break
                await asyncio.sleep(0.01)
            task, requests = stream.verify_task, stream.totals["requests"]
            judge.gate.set()                            # the answer is ready...
            wall = m._wall
            m._wall = lambda: wall() + 120
            stream.fire(TIMER)                          # ...and the deadline cancels it in the same turn
            for _ in range(5):
                await asyncio.sleep(0)
            assert task.done() and stream.verify_task is None and _settled(m, stream)
            assert stream.totals["requests"] == requests + 1
            await _ends_and_frees_capacity(m, src, watch, judge)

    run(main())


def test_an_error_after_the_answer_is_logged_and_its_request_settled(monkeypatch, caplog):
    def absorb(self, request, result, works):
        raise RuntimeError("absorb broke")

    monkeypatch.setattr(rt._Stream, "absorb", absorb)
    judge = Gated(answers({"detect": [Y]}))

    async def main():
        async with percept.Monitor(judge, [FALL], store=mem(), max_requests_per_min=1000,
                                   max_concurrent_requests=1) as m:
            src = PushSource("cam")
            watch = m.watch(src, fps=None)
            await push_frames(src, 4)
            src.close()
            await asyncio.wait_for(collect(watch), 5)
            return m._slots, judge.started["detect"]

    with caplog.at_level("ERROR", logger="percept"):
        slots, detects = run(main())
    assert slots == 0 and detects == 1
    # reported when the task ends, naming its request, not only when the monitor closes
    failed = [r for r in caplog.records if r.getMessage().startswith("percept: stream cam: request ")]
    assert len(failed) == 1 and "absorb broke" in str(failed[0].exc_info[1])


def _bare_stream(m):
    stream = rt._Stream.__new__(rt._Stream)
    stream.m, stream.id, stream.verify_task, stream.verify_work = m, "cam", None, None
    stream.verifies, stream.pending = {}, {}
    return stream


def test_a_ceiling_too_small_for_both_keeps_first_come_order():
    m = percept.Monitor(FakeJudge(answers({"detect": [Y], "verify": [Y]})), [], store=mem(), max_requests_per_min=3)
    assert _bare_stream(m).verify_headroom() == 0          # 3 < 2 x 2 attempts: no reserve


def test_no_attempts_are_held_while_the_stream_waits_on_the_dollar_cap():
    fall = Condition("fall", "a person falls", kind="event")
    m = percept.Monitor(FakeJudge(answers({"detect": [Y], "verify": [Y]})), [fall], store=mem(), max_requests_per_min=6)
    stream = _bare_stream(m)
    stream.verifies[("fall", "scene")] = rt._Work(rt.VERIFY, "fall", "scene", 0.0)    # waiting to verify
    assert stream.verify_headroom() == m._attempts
    m._cost_waiters["cam"] = None                          # a verification refused for money
    assert stream.verify_headroom() == 0


def test_a_full_group_for_a_non_batching_judge_still_fits_with_the_verification_room(monkeypatch):
    # Without the group cap a full group (limit // attempts items) plus the kept room could never
    # fit, even in an empty window, and detection stopped for good.
    monkeypatch.setattr(rt, "BUDGET_WINDOW_S", 1.0)
    conds = [Condition(c, f"event {c}", kind="event", trigger=At(1), rearm_s=60.0) for c in "abcd"]
    judge = FakeJudge(answers({"detect": [N], "verify": [Y]}), can_batch=False)    # max_attempts 2

    async def main():
        async with percept.Monitor(judge, conds, store=mem(), max_requests_per_min=6) as m:
            src = PushSource("cam")
            m.watch(src, fps=None)
            await push_frames(src, 30, pause=0.05)
            return m.stats(), m.coverage("cam")

    stats, cov = run(main())
    looked = {r.items[0].condition.id for _, _, _, r in judge.sent if r.purpose == "detect"}
    assert looked == set("abcd")                         # every look was judged, none starved
    assert all(len(r.items) <= 2 for _, _, _, r in judge.sent)   # (6 - 2) // 2 items a request


# --------------------------------------------------------------------------- combined triggers


class Counting:
    """Due on the given seqs; counts its steps (shared by value: equal instances are one trigger)."""

    steps: list = []

    def __init__(self, *seqs: int) -> None:
        self.seqs = seqs

    def __eq__(self, other):
        return isinstance(other, Counting) and other.seqs == self.seqs

    def __hash__(self):
        return hash(("Counting", self.seqs))

    def step(self, frame, conditions, state, now):
        Counting.steps.append((frame.seq, tuple(c.id for c in conditions)))
        return ({c.id: "at" for c in conditions} if frame.seq in self.seqs else {}), None


async def _drive(conds, n, judge=None, limit=1000, pause=0.0):
    """Push ``n`` frames and drain; returns (coverage, stats, judge)."""
    judge = judge or FakeJudge(answers({"detect": [N], "verify": [Y]}))
    async with percept.Monitor(judge, conds, store=mem(), max_requests_per_min=limit) as m:
        src = PushSource("cam")
        watch = m.watch(src, fps=None)
        await push_frames(src, n, pause=pause)
        src.close()
        await collect(watch)
        return m.coverage("cam"), m.stats(), judge


def test_a_tuple_trigger_is_due_when_any_member_fires():
    cond = Condition("door", "the door is open", trigger=(At(2), At(5)))
    cov, stats, judge = run(_drive([cond], 8, pause=0.02))
    counts = cov["triggers"]["door"]["At"]
    assert counts["due"] == 2 and counts["admitted"] + counts["coalesced"] == 2
    assert counts["dropped_budget"] == counts["dropped_for_verification"] == 0
    assert cov["fates"]["door"].get("judged", 0) + cov["fates"]["door"].get("coalesced", 0) == 2


def test_both_on_change_and_every_members_make_a_look_due():
    zone = (((0.0, 0.0), (0.5, 0.0), (0.5, 0.5), (0.0, 0.5)),)
    cond = Condition("door", "the door is open", trigger=(percept.on_change(), percept.every(1000)), zones=zone)

    async def main():
        judge = FakeJudge(answers({"detect": [N], "verify": [Y]}))
        async with percept.Monitor(judge, [cond], store=mem(), max_requests_per_min=1000) as m:
            src = PushSource("cam")
            watch = m.watch(src, fps=None)
            base = image(0)
            for _ in range(4):
                src.push(base)
                await asyncio.sleep(0.02)
            changed = base.copy()
            changed[5:40, 5:55] = 255   # inside the zone, a small part of the frame
            for _ in range(3):
                src.push(changed)
                await asyncio.sleep(0.05)
            src.close()
            await collect(watch)
            return m.coverage("cam"), m.stats()

    cov, stats = run(main())
    per = cov["triggers"]["door"]
    assert per["every(1000)"]["due"] == 1 and per["on_change(0.2)"]["due"] >= 1
    assert set(per["every(1000)"]) == {"due", "admitted", "coalesced", "dropped_budget", "dropped_for_verification"}
    assert stats["streams"]["cam"]["triggers"]["every(1000)"]["due"] == 1


def test_equal_triggers_across_conditions_share_one_state_and_step():
    Counting.steps = []
    conds = [Condition("door", "the door is open", trigger=(Counting(1), Never())),
             Condition("gate", "the gate is open", trigger=Counting(1)),
             Condition("fire", "there is fire", trigger=(Counting(1), Counting(3)))]
    m = percept.Monitor(FakeJudge({}), conds, store=mem())
    keyed = {k: cs for k, _, cs in m._groups}
    assert [c.id for c in keyed[Counting(1)]] == ["door", "gate", "fire"]
    assert [c.id for c in keyed[Counting(3)]] == ["fire"]
    cov, _, _ = run(_drive(conds, 5))
    assert sum(1 for seq, ids in Counting.steps if ids == ("door", "gate", "fire")) == 5   # once a frame
    assert cov["triggers"]["fire"]["Counting"]["due"] == 2
    assert cov["triggers"]["door"]["Counting"]["due"] == cov["triggers"]["gate"]["Counting"]["due"] == 1


def test_a_trigger_listed_twice_in_one_tuple_counts_once():
    m = percept.Monitor(FakeJudge({}), [Condition("door", "the door is open", trigger=(At(1),) * 2)], store=mem())
    assert [len(cs) for _, _, cs in m._groups] == [1]


def test_coalesced_looks_keep_their_first_label_and_fates_match_a_single_trigger():
    # deterministic: enqueue directly, then let the drop path count them
    cond = Condition("door", "the door is open", trigger=Never())

    async def main():
        async with percept.Monitor(FakeJudge(answers({"detect": [N]})), [cond], store=mem()) as m:
            src = PushSource("cam")
            m.watch(src, fps=None)
            src.push(image())
            await asyncio.sleep(0.1)
            stream = m._streams["cam"]
            before = dict(m.coverage("cam")["fates"].get("door", {}))
            stream.detect_task = asyncio.get_running_loop().create_future()   # nothing dispatches
            stream.enqueue("door", "s", rt.STATE, trigger="every(5)")
            stream.enqueue("door", "s", rt.STATE, trigger="on_change(0.2)")
            w = stream.pending[("door", "s")]
            assert w.trigger == "every(5)"
            w.deadline, w.blocked = m._mono() - 1.0, True
            stream.dispatch(m._mono())
            stream.detect_task = None
            cov = m.coverage("cam")
            return before, cov

    before, cov = run(main())
    per = cov["triggers"]["door"]
    assert per["every(5)"] == {"due": 1, "admitted": 0, "coalesced": 0, "dropped_budget": 1,
                               "dropped_for_verification": 0}
    assert per["on_change(0.2)"] == {"due": 1, "admitted": 0, "coalesced": 1, "dropped_budget": 0,
                                     "dropped_for_verification": 0}
    fates = cov["fates"]["door"]
    assert fates.get("coalesced", 0) - before.get("coalesced", 0) == 1
    assert fates.get("dropped_budget", 0) - before.get("dropped_budget", 0) == 1


def test_poke_looks_are_labelled_poke():
    cond = Condition("door", "the door is open", trigger=Never())

    async def main():
        async with percept.Monitor(FakeJudge(answers({"detect": [N]})), [cond], store=mem()) as m:
            src = PushSource("cam")
            watch = m.watch(src, fps=None)
            src.push(image())
            await asyncio.sleep(0.05)
            m.poke("cam", "door", reason="sensor")
            await asyncio.sleep(0.1)
            src.close()
            await collect(watch)
            return m.coverage("cam")

    per = run(main())["triggers"]["door"]["poke"]
    assert per["due"] == 1 and per["admitted"] == 1


def test_fates_with_a_tuple_of_one_equal_those_of_the_bare_trigger():
    single = run(_drive([Condition("door", "the door is open", trigger=At(2))], 5))[0]
    as_tuple = run(_drive([Condition("door", "the door is open", trigger=(At(2),))], 5))[0]
    assert single["fates"] == as_tuple["fates"] and single["triggers"] == as_tuple["triggers"]


def test_a_storm_of_tuple_triggers_never_exceeds_the_window(monkeypatch):
    monkeypatch.setattr(rt, "BUDGET_WINDOW_S", 1.0)
    monkeypatch.setattr(rt, "DEADLINES", {rt.EVENT: 0.3, rt.STATE: 0.5})
    monkeypatch.setattr(rt, "VERIFY_RETRY_S", 0.05)
    attempts: list[float] = []
    rng = np.random.default_rng(11)

    def script(request):
        attempts.append(time.monotonic())
        if rng.random() < 0.3:
            return RetryableError("busy")
        return {i.condition.id: (Y if rng.random() < 0.5 else N) for i in request.items}

    judge = FakeJudge(script, can_batch=False)
    conds = [Condition("fall", "a person falls", kind="event", trigger=(Always(), At(3), percept.every(0.05)),
                       rearm_s=0.2),
             Condition("door", "the door is open", trigger=(percept.every(0.05), Always()), clear_after_no=1),
             Condition("gate", "the gate is open", trigger=(Always(), percept.on_change()), clear_after_no=1)]
    limit = 8
    cov, stats, _ = run(_drive(conds, 110, judge=judge, limit=limit, pause=0.02))
    attempts.sort()
    for i, t in enumerate(attempts):
        assert sum(1 for u in attempts[i:] if u < t + 1.0) <= limit
    assert stats["streams"]["cam"]["attempts"] == len(attempts)
    dropped = sum(per.get("dropped_budget", 0) for c in cov["triggers"].values() for per in c.values())
    assert dropped == sum(c.get("dropped_budget", 0) for c in cov["fates"].values())
    for cid, per in cov["triggers"].items():
        coalesced = sum(p["coalesced"] for p in per.values())
        assert coalesced == cov["fates"][cid].get("coalesced", 0)


def test_each_tuple_member_keeps_its_own_priority(monkeypatch):
    # A state with (every, change-like) members: the heartbeat member's looks are HEARTBEAT, the
    # other member's are STATE, exactly as the 0.9 default pair (on_change, every(300)) is.
    seen = []
    original = rt._Stream.enqueue

    def spy(self, cid, subject, prio, **kw):
        seen.append((kw.get("trigger"), prio))
        return original(self, cid, subject, prio, **kw)

    monkeypatch.setattr(rt._Stream, "enqueue", spy)
    door = Condition("door", "the door is open", trigger=(percept.every(100), At(3)))
    judge = FakeJudge(answers({"detect": [N], "verify": [N]}))

    async def main():
        async with percept.Monitor(judge, [door], store=mem()) as m:
            src = PushSource("cam")
            m.watch(src, fps=None)
            await push_frames(src, 5)

    run(main())
    prios = {label: prio for label, prio in seen if label is not None}
    assert prios["every(100)"] == rt.HEARTBEAT and prios["At"] == rt.STATE, seen


def test_a_retried_verification_counts_as_one_due_and_one_admitted_look(monkeypatch):
    monkeypatch.setattr(rt, "VERIFY_RETRY_S", 0.05)
    fall = Condition("fall", "a person falls", kind="event", trigger=At(1))
    judge = FakeJudge(answers({"detect": [Y], "verify": [RetryableError("down"), RetryableError("down"), Y]}),
                      retries=0)

    async def main():
        async with percept.Monitor(judge, [fall], store=mem()) as m:
            src = PushSource("cam")
            watch = m.watch(src, fps=None)
            await push_frames(src, 3)
            got = await asyncio.wait_for(watch.__anext__(), 5.0)
            return got, m.coverage("cam")

    got, cov = run(main())
    assert got.decision.type == "met"
    assert sum(1 for *_, r in judge.sent if r.purpose == "verify") == 3        # sent three times
    assert cov["triggers"]["fall"]["verify"]["due"] == 1
    assert cov["triggers"]["fall"]["verify"]["admitted"] == 1


# ----------------------------------------------------------------- ready()

def test_a_detector_started_before_the_watch_waits_for_ready_and_never_pokes_an_unknown_stream():
    fell = Condition("fell", "a box fell", kind="event", trigger=Never())

    async def once(i: int) -> list:
        async with percept.Monitor(FakeJudge(answers({"detect": [N], "verify": [N]})), [fell], store=mem(),
                                   max_requests_per_min=1000) as m:
            src = PushSource("cam")

            async def detector() -> None:
                try:
                    await m.ready("cam")
                    m.poke("cam", "fell", reason="sensor")
                finally:
                    src.close()                             # a failure fails the test, not hangs it

            async def consume() -> None:
                await asyncio.sleep(0.001 * (i % 3))        # the watch starts after the detector
                async for _ in m.watch(src):
                    pass

            return await asyncio.gather(asyncio.create_task(detector()), consume(), return_exceptions=True)

    async def main() -> int:
        results = [await once(i) for i in range(100)]
        return sum(isinstance(r, BaseException) for rs in results for r in rs)

    assert run(main(), timeout=60) == 0


def test_ready_returns_at_once_for_a_registered_stream():
    async def main():
        async with percept.Monitor(FakeJudge(answers({"detect": [N]})), [], store=mem()) as m:
            m.watch(PushSource("cam"))
            await asyncio.wait_for(m.ready("cam"), 0.05)
            assert m._ready == {}

    run(main())


def test_ready_raises_when_the_monitor_closes_before_the_stream_registers():
    async def main():
        m = percept.Monitor(FakeJudge(answers({"detect": [N]})), [], store=mem())
        waiter = asyncio.create_task(m.ready("cam"))
        await asyncio.sleep(0.01)
        await m.aclose()
        with pytest.raises(RuntimeError, match="closed"):
            await waiter
        with pytest.raises(RuntimeError, match="closed"):
            await m.ready("cam")
        assert m._ready == {}

    run(main())


def test_ready_times_out_and_leaves_no_waiter_behind():
    async def main():
        async with percept.Monitor(FakeJudge(answers({"detect": [N]})), [], store=mem()) as m:
            with pytest.raises(asyncio.TimeoutError):
                await m.ready("cam", timeout=0.01)
            assert m._ready == {}
            later = asyncio.create_task(m.ready("cam", timeout=5))
            await asyncio.sleep(0.01)
            m.watch(PushSource("cam"))
            await later                                     # a later waiter is still woken
            assert m._ready == {}

    run(main())


def test_ready_from_another_thread_raises_like_other_methods():
    async def main():
        async with percept.Monitor(FakeJudge(answers({"detect": [N]})), [], store=mem()) as m:
            m.watch(PushSource("cam"))
            errors = []

            def other() -> None:
                try:
                    asyncio.run(m.ready("cam"))
                except RuntimeError as e:
                    errors.append(str(e))

            worker = threading.Thread(target=other)
            worker.start()
            worker.join()
            assert errors and "event loop" in errors[0]

    run(main())
