"""The trace writer and deterministic replay, end to end through the runtime."""

from __future__ import annotations

import asyncio
import json
import threading
import time

import pytest

import percept
from percept import monitor as rt
from percept import trace as tr
from percept.contracts import Condition
from percept.delivery import MemoryStore
from percept.judges import FakeJudge
from percept.judges.base import RetryableError
from percept.replay import decode_input, encode_input, replay
from percept.sources import PushSource
from test_monitor import N, Y, Never, collect, image, run

KINDS = {"restore", "conditions", "decide_input", "decision", "request", "fate", "health", "blind", "poke"}


async def settle(seconds: float = 0.08) -> None:
    await asyncio.sleep(seconds)


def scripted():
    """A judge whose answers the test sets as it goes: ``plan[condition][purpose]``."""
    plan = {"door": {"detect": Y}, "fall": {"detect": Y, "verify": Y}}

    def script(request):
        out = {}
        for item in request.items:
            reply = plan[item.condition.id][request.purpose]
            if isinstance(reply, BaseException):
                return reply
            out[item.condition.id] = reply
        return out

    return FakeJudge(script), plan


DOOR = Condition("door", "the door is open", trigger=Never(), clear_after_no=1, emit_cleared=True, cooldown_s=30)
FALL = Condition("fall", "a person falls", kind="event", trigger=Never(), rearm_s=0.15, candidate_deadline_s=5)


async def first_run(store, path, stream_id="cam", reason="look"):
    judge, plan = scripted()
    m = percept.Monitor(judge, [DOOR, FALL], store=store, max_requests_per_min=1000, trace=path)
    src = PushSource(stream_id)
    watch = m.watch(src, fps=None)
    got = []

    async def consume():
        async for d in watch:
            got.append(d.decision)

    consumer = asyncio.create_task(consume())
    frames = iter(range(100))

    async def look(cid, why="look", wait=0.08):    # a new frame each time: older evidence is stale
        src.push(image(next(frames)))
        await settle(0.03)
        m.poke(stream_id, cid, reason=why)
        await settle(wait)

    await look("door", reason)                                  # met
    await look("fall", "fall detector", 0.25)                   # detect, verify on the clip, met; rearms
    m.snooze("fall", time.time() + 30)
    await look("fall", "again", 0.15)                           # verified, but snoozed: suppressed
    m.snooze("fall", 0)                                         # cancelled
    plan["door"]["detect"] = N
    await look("door")                                          # cleared
    plan["door"]["detect"] = RetryableError("down")
    await look("door", wait=0.2)                                # a failure: health, no change
    plan["door"]["detect"] = Y
    await look("door")                                          # met again, inside the cooldown: deferred
    m.pause("door")
    m.resume("door")
    m.add(Condition("fall", "a person falls down", kind="event", trigger=Never(), rearm_s=0.15))
    await settle()
    for _ in range(3):
        src.push(image(next(frames)))
        await settle(0.02)
    await settle(0.2)
    await m.aclose()
    consumer.cancel()
    return got


async def second_run(store, path):
    judge, plan = scripted()
    plan["fall"]["verify"] = ("unknown", 0.5, "can't tell")
    m = percept.Monitor(judge, [DOOR, FALL], store=store, max_requests_per_min=1000, trace=path)
    src = PushSource("cam")
    watch = m.watch(src, fps=None)
    src.push(image(9))
    await settle()
    m.poke("cam", "fall", reason="look")                        # verify can't tell: uncertain
    await settle(0.2)
    src.close()
    got = [d.decision for d in await collect(watch)]
    await m.aclose()
    return got


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("trace")
    store = MemoryStore()
    first = run(first_run(store, tmp / "one.jsonl"))
    second = run(second_run(store, tmp / "two.jsonl"))
    return tmp, first, second


def test_every_record_kind_is_written(runs):
    tmp, _, _ = runs
    records = list(tr.read(tmp / "one.jsonl"))
    assert KINDS <= {r["kind"] for r in records}
    assert all(r["v"] == 1 and isinstance(r["t_wall"], float) and r["stream_id"] == "cam" for r in records)
    poke = next(r for r in records if r["kind"] == "poke")
    assert poke["reason"] == "look" and poke["condition_ids"] == ["door"] and poke["evidence_images"] == 0
    request = next(r for r in records if r["kind"] == "request" and r["purpose"] == "verify")
    assert request["judge"] == "fake" and request["condition_ids"] == ["fall"] and request["usage"]
    assert request["latency_s"] >= 0
    assert any(r["kind"] == "request" and r["failure"] == "unavailable" for r in records)
    assert any(r["kind"] == "fate" and r["fate"] == "failed" for r in records)
    assert {r["state"] for r in records if r["kind"] == "blind" and r["blind_kind"] == "paused"} == {"open", "close"}
    assert len([r for r in records if r["kind"] == "conditions"]) == 2         # the start, and the change
    restores = [r for r in tr.read(tmp / "two.jsonl") if r["kind"] == "restore"]
    assert [r["restored"] for r in restores] == [True]


def test_replay_reproduces_every_decision(runs):
    tmp, first, second = runs
    assert {d.type for d in first} >= {"met", "cleared"}
    assert ("uncertain", "unverified") in {(d.type, d.cause) for d in second}
    one, two = list(tr.read(tmp / "one.jsonl")), list(tr.read(tmp / "two.jsonl"))
    for records, delivered in ((one, first), (two, None), (one + two, None)):
        result = replay(records)
        assert result.matches is True, result.differences
        assert result.inputs == sum(r["kind"] == "decide_input" for r in records)
        if delivered is not None:
            assert [d.decision_id for d in result.decisions] == [d.decision_id for d in delivered]
    # what made this run interesting actually happened
    events = {r.get("event") for r in one if r["kind"] == "fate"}
    assert {"verified", "suppressed", "rearmed", "snoozed", "snooze_cancelled", "condition_changed"} <= events
    assert any(r["kind"] == "fate" and "cooldown" in r["detail"] for r in one)
    assert any(r["input"]["type"] == "EpochStarted" and r["input"]["restart"] for r in two if r["kind"] == "decide_input")


def test_an_altered_record_is_reported_as_a_difference(runs):
    tmp, _, _ = runs
    records = list(tr.read(tmp / "one.jsonl"))
    for r in records:     # the judge's "no" that cleared the door becomes a confident "yes"
        if r["kind"] == "decide_input" and r["input"]["type"] == "Observed" \
                and r["input"]["observation"]["condition_id"] == "door" and r["input"]["observation"]["answer"] == "no":
            r["input"]["observation"]["answer"] = "yes"
            break
    result = replay(records)
    assert result.matches is False
    assert [d.type for d in result.decisions] == ["met", "met"]
    assert any("3 decisions recorded, 2 replayed" in d and "cleared" in d for d in result.differences), \
        result.differences

    dropped = [r for r in tr.read(tmp / "one.jsonl") if r["kind"] != "restore"]
    result = replay(dropped)
    assert not result.matches and any("before any restore record" in d for d in result.differences)


def test_inputs_round_trip_through_json():
    from percept import decide
    from percept.contracts import Observation, Usage

    o = Observation("door", "scene", "yes", 0.9, "seen", "r-1", "cam", 2, 3, 4, (1.5, 2.5), "fake", "verify")
    for inp in (decide.Observed(o, Usage(1, 2, 3, 0.25, 0.5)), decide.TimerFired("door", "scene", "verify"),
                decide.Resolve("cam/door/x-1"), decide.FalseAlarm("cam/door/x-1", "a cat"),
                decide.Snooze("door", float("inf"), None), decide.SceneReturn("door"),
                decide.ConditionChanged("door"), decide.EpochStarted(3, True)):
        assert decode_input(json.loads(json.dumps(encode_input(inp)))) == inp


# --------------------------------------------------------------------------- what is never written


def test_no_images_keyframes_secrets_or_raw_uris_reach_the_trace(tmp_path):
    secret_uri = "rtsp://admin:hunter2@cam.local/live"
    door = Condition("door", "the door is open", trigger=Never())
    judge = FakeJudge(lambda r: {"door": ("yes", 0.9, "Bearer abcdefghijklmnop0123, https://u:pw@x.io/a?k=s3cr3t, "
                                                       "rtsp://cam.local/live?token=abc123")})

    async def main():
        m = percept.Monitor(judge, [door], store=MemoryStore(), trace=tmp_path / "t.jsonl")
        src = PushSource(secret_uri)
        watch = m.watch(src)
        src.push(image())
        await settle()
        m.poke(secret_uri, "door", reason="api_key=sk-ABCDEFGHIJKLMNOPQRSTUV", evidence=image(1))
        got = await asyncio.wait_for(watch.__anext__(), 5)
        await m.aclose()
        return got.decision

    decision = run(main())
    assert decision.keyframe                                   # the decision has one; the trace must not
    text = (tmp_path / "t.jsonl").read_text()
    for leaked in ("hunter2", "admin", "abc123", "sk-ABCDEFGHIJKLMNOPQRSTUV", "abcdefghijklmnop0123", "s3cr3t",
                   "u:pw", "keyframe", "/9j/"):
        assert leaked not in text, leaked
    records = list(tr.read(tmp_path / "t.jsonl"))
    assert {r["stream_id"] for r in records} == {"rtsp://cam.local/live"}
    assert max(len(json.dumps(r)) for r in records) < 8000
    poke = next(r for r in records if r["kind"] == "poke")
    assert poke["reason"] == "api_key=<redacted>" and poke["evidence_images"] == 1
    assert replay(records).matches                             # scrubbed consistently, still replayable


def test_scrub():
    assert tr.scrub("see https://user:pw@h.io/p?q=1#f now") == "see https://h.io/p now"
    assert tr.scrub("password: hunter2, token=abc") == "password: <redacted>, token=<redacted>"
    assert tr.scrub("Authorization: Bearer abcdefghijkl") == "Authorization: Bearer <redacted>"
    assert tr.scrub("cam/door/18f-2 judged yes") == "cam/door/18f-2 judged yes"


# --------------------------------------------------------------------------- the writer


def test_rotation_keeps_at_most_backups_plus_one_files_and_reads_oldest_first(tmp_path):
    path = tmp_path / "t.jsonl"
    w = tr.TraceWriter(path, max_bytes=2_000, backups=2)
    for i in range(300):
        w.write({"v": 1, "kind": "x", "i": i, "pad": "." * 40})
        if i % 10 == 9:
            assert w.flush(5)
    w.close()
    assert sorted(p.name for p in tmp_path.iterdir()) == ["t.jsonl", "t.jsonl.1", "t.jsonl.2"]
    assert all(p.stat().st_size <= 2_000 for p in tmp_path.iterdir())
    seq = [r["i"] for r in tr.read(path)]
    assert seq == sorted(seq) and seq[-1] == 299 and len(seq) < 300
    assert w.written == 300 and w.dropped == 0


class SlowWriter(tr.TraceWriter):
    delay = 0.3

    def _write(self, batch):
        time.sleep(self.delay)
        super()._write(batch)


def test_a_slow_disk_never_blocks_the_caller_or_the_event_loop(tmp_path):
    w = SlowWriter(tmp_path / "slow.jsonl")
    t0 = time.perf_counter()
    for i in range(5_000):
        w.write({"v": 1, "i": i})
    assert time.perf_counter() - t0 < 0.5
    w.close()

    door = Condition("door", "the door is open", trigger=Never())
    judge = FakeJudge(lambda r: {"door": Y})
    lags = []

    async def main():
        async def ticker():
            while True:
                t = time.perf_counter()
                await asyncio.sleep(0.01)
                lags.append(time.perf_counter() - t - 0.01)

        tick = asyncio.create_task(ticker())
        slow = SlowWriter(tmp_path / "slow2.jsonl")
        m = percept.Monitor(judge, [door], store=MemoryStore(), trace=slow)
        src = PushSource("cam")
        watch = m.watch(src, fps=None)
        for i in range(10):
            src.push(image(i))
            m.poke("cam", "door", reason="look")
            await settle(0.02)
        got = await asyncio.wait_for(watch.__anext__(), 5)
        tick.cancel()
        await m.aclose()
        slow.close()
        return got.decision, slow

    decision, slow = run(main())
    assert decision.type == "met"
    assert max(lags) < 0.2
    assert slow.dropped == 0 and replay(tr.read(tmp_path / "slow2.jsonl")).matches


def test_a_failing_disk_is_reported_on_health_and_never_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(rt, "SNAPSHOT_EVERY_S", 0.05)
    (tmp_path / "not-a-dir").write_text("")
    door = Condition("door", "the door is open", trigger=Never())

    async def main():
        m = percept.Monitor(FakeJudge(lambda r: {"door": Y}), [door], store=MemoryStore(),
                            trace=tmp_path / "not-a-dir" / "t.jsonl")
        src = PushSource("cam")
        watch = m.watch(src, fps=None)
        src.push(image())
        await settle()
        m.poke("cam", "door", reason="look")
        got = await asyncio.wait_for(watch.__anext__(), 5)
        await settle(0.3)
        details = [h.detail for h in m._health.items.values() if h.kind == "store_lag"]
        stats = m.stats()
        await m.aclose()
        return got.decision, details, stats

    decision, details, stats = run(main())
    assert decision.type == "met"
    assert stats["trace_dropped"] > 0 and stats["trace_written"] == 0
    assert any(d.startswith("trace:") and "records dropped" in d for d in details), details


def test_memory_stays_bounded_under_a_write_storm(tmp_path):
    gate = threading.Event()

    class Stuck(tr.TraceWriter):
        def _write(self, batch):
            gate.wait(10)
            super()._write(batch)

    w = Stuck(tmp_path / "t.jsonl", max_pending=1_000)
    for i in range(50_000):
        w.write({"v": 1, "i": i})
        assert w.pending <= 1_000
    assert w.dropped >= 50_000 - 2_000          # at most one batch in the writer's hands
    gate.set()
    w.close()
    assert w.written + w.dropped == 50_000
    w.write({"v": 1})                           # after close: dropped, never raised
    assert w.written + w.dropped == 50_001


def test_monitor_rejects_a_bad_trace_argument():
    with pytest.raises(TypeError):
        percept.Monitor(FakeJudge({}), store=MemoryStore(), trace=42)
