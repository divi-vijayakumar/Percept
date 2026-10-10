"""Stores, the writer thread and the outbox, below the monitor."""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
import threading

import pytest

from percept import delivery
from percept.contracts import Decision, Usage
from percept.delivery import MemoryStore, Outbox, SqliteStore, Watch, Writer, store_key


def decision(n: int, stream: str = "cam") -> Decision:
    occ = f"{stream}/door/abc-{n}"
    return Decision(f"{occ}/1", occ, "met", "door", stream, "scene", 1.0 + n, 2.0 + n, 0.9, "seen",
                    Usage(1, 10, 5, 0.0, 0.1), occ)


def run(coro, timeout: float = 10.0):
    return asyncio.run(asyncio.wait_for(coro, timeout))


def test_memory_store_commits_atomically_and_deletes_with_none():
    store = MemoryStore()
    store.commit({"a": b"1", "b": b"2"})
    store.commit({"a": None, "c": b"3"})
    assert (store.load("a"), store.load("b"), store.load("c")) == (None, b"2", b"3")
    with pytest.raises(TypeError):
        store.commit({"d": b"4", "e": "not bytes"})  # type: ignore[dict-item]
    assert store.load("d") is None  # nothing of a bad commit is applied


def test_memory_store_logs_once_that_it_is_not_crash_safe(monkeypatch, caplog):
    monkeypatch.setattr(delivery, "_warned_memory", False)
    with caplog.at_level(logging.WARNING, logger="percept"):
        MemoryStore()
        MemoryStore()
    assert [r.message for r in caplog.records].count(
        "percept.MemoryStore keeps decisions in memory only: delivery is not crash-safe") == 1


def test_sqlite_store_is_wal_atomic_and_durable(tmp_path):
    path = tmp_path / "state.db"
    store = SqliteStore(path)
    store.commit({"a": b"1", "b": b"2"})
    with pytest.raises(Exception):
        store.commit({"a": None, "b": object()})  # type: ignore[dict-item]  # fails mid-transaction
    assert store.load("a") == b"1" and store.load("b") == b"2"  # rolled back as a whole
    store.commit({"a": None})
    store.close()
    again = SqliteStore(path)
    assert again.load("a") is None and again.load("b") == b"2"
    again.close()
    with sqlite3.connect(path) as conn:
        assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"


class FlakyStore(MemoryStore):
    def __init__(self, fail_first: int) -> None:
        super().__init__()
        self.fail_first, self.calls, self.threads = fail_first, [], set()

    def commit(self, writes):
        self.threads.add(threading.current_thread().name)
        self.calls.append(dict(writes))
        if len(self.calls) <= self.fail_first:
            raise OSError("disk busy")
        super().commit(writes)


def test_writer_runs_every_call_on_one_thread_in_order_and_retries_a_failed_commit(monkeypatch):
    monkeypatch.setattr(Writer, "RETRY_FIRST_S", 0.01)
    store = FlakyStore(fail_first=2)

    async def main():
        writer = Writer(store)
        first = asyncio.ensure_future(writer.commit({"k": b"1"}))
        second = asyncio.ensure_future(writer.commit({"k": b"2"}))   # waits behind the failing one
        await asyncio.gather(first, second)
        assert writer.failures == 2 and writer.pending == 0
        assert await writer.load(["k"]) == [b"2"]
        await writer.close(1.0)
        with pytest.raises(RuntimeError):
            await writer.commit({"k": b"3"})

    run(main())
    assert store.threads == {"percept-store-writer"}
    assert [c["k"] for c in store.calls] == [b"1", b"1", b"1", b"2"]


def test_writer_close_abandons_a_commit_that_never_succeeds(monkeypatch):
    monkeypatch.setattr(Writer, "RETRY_FIRST_S", 0.01)

    async def main():
        writer = Writer(FlakyStore(fail_first=10**9))
        pending = asyncio.ensure_future(writer.commit({"k": b"1"}))
        await asyncio.sleep(0.05)
        await writer.close(0.1)
        with pytest.raises(OSError):
            await pending
        assert not writer._thread.is_alive()

    run(main())


def test_outbox_restores_pending_decisions_in_order_with_attempt_plus_one():
    store = MemoryStore()

    async def first():
        ob = Outbox(Writer(store), "cam")
        assert await ob.restore() is None
        await ob.commit(b'{"decide": 1}', (decision(1), decision(2), decision(3)))
        await ob.commit(None, (decision(4),))
        entries = list(ob.entries.values())
        await ob.ack(entries[1])         # out of order
        await ob.ack(entries[0])
        await ob.writer.close(1.0)

    async def second():
        ob = Outbox(Writer(store), "cam")
        assert await ob.restore() == {"decide": 1}
        got = [(e.decision.decision_id, e.attempt) for e in ob.entries.values()]
        await ob.writer.close(1.0)
        return got

    run(first())
    assert json.loads(store.load(store_key("outbox-meta", "cam"))) == {"head": 2, "next": 4}
    assert run(second()) == [(decision(3).decision_id, 2), (decision(4).decision_id, 2)]
    assert run(second()) == [(decision(3).decision_id, 3), (decision(4).decision_id, 3)]  # never repeats


def test_store_keys_of_different_streams_never_collide():
    assert store_key("outbox", "a#000000000001") != store_key("outbox", "a", "#000000000001")


def test_watch_takeover_redelivers_unacknowledged_with_a_persisted_attempt():
    store = MemoryStore()

    async def main():
        ob = Outbox(Writer(store), "cam")
        await ob.restore()
        await ob.commit(None, (decision(1), decision(2)))
        ob.finish()
        first = Watch(ob, "manual", 10, lambda n: None)
        got = [(d.decision.decision_id, d.attempt) async for d in first]
        second = Watch(ob, "auto", 10, lambda n: None)   # takes over: nothing was acknowledged
        again = [(d.decision.decision_id, d.attempt) async for d in second]
        await ob.writer.close(1.0)
        return got, again, len(ob.entries)

    got, again, left = run(main())
    assert [a for _, a in got] == [1, 1] and [a for _, a in again] == [2, 2]
    assert [i for i, _ in got] == [i for i, _ in again]
    assert left == 0  # auto-acknowledged after the loop asked for more
    assert json.loads(store.load(store_key("outbox-meta", "cam"))) == {"head": 2, "next": 2}


class GatedStore(MemoryStore):
    """Holds the commit that writes ``hold_key`` until ``release`` is set."""

    def __init__(self, hold_key: str) -> None:
        super().__init__()
        self.hold_key, self.release, self.holding = hold_key, threading.Event(), threading.Event()

    def commit(self, writes):
        if self.hold_key in writes:
            self.holding.set()
            self.release.wait(5)
        super().commit(writes)


def test_an_ack_while_a_later_commit_is_in_flight_never_hides_that_decision():
    store = GatedStore(delivery._entry_key("cam", 1))

    async def main():
        ob = Outbox(Writer(store), "cam")
        await ob.restore()
        await ob.commit(None, (decision(1),))
        later = asyncio.ensure_future(ob.commit(None, (decision(2),)))   # held in the writer
        await asyncio.to_thread(store.holding.wait, 5)
        acked = asyncio.ensure_future(ob.ack(ob.entries[0]))             # queued behind it
        await asyncio.sleep(0.05)
        store.release.set()
        await asyncio.gather(later, acked)
        await ob.writer.close(1.0)

    async def restored():
        ob = Outbox(Writer(store), "cam")
        await ob.restore()
        await ob.writer.close(1.0)
        return [e.decision.decision_id for e in ob.entries.values()]

    run(main())
    assert run(restored()) == [decision(2).decision_id]
