"""Stores, the writer thread and the outbox, below the monitor."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sqlite3
import stat
import threading
import time
import tracemalloc
from dataclasses import replace

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




async def drain(ob: Outbox, ack: str = "manual") -> list:
    ob.finish()
    return [d async for d in Watch(ob, ack, 10**6, lambda n: None)]


class CountingStore(MemoryStore):
    def __init__(self) -> None:
        super().__init__()
        self.writes: list[dict] = []

    def commit(self, writes):
        self.writes.append(dict(writes))
        super().commit(writes)


def test_outbox_restores_pending_decisions_in_order_with_attempt_plus_one_and_rewrites_nothing():
    store = CountingStore()

    async def first():
        ob = Outbox(Writer(store), "cam")
        assert await ob.restore() is None
        await ob.commit(b'{"decide": 1}', (decision(1), decision(2), decision(3)))
        await ob.commit(None, (decision(4),))
        got = await drain(ob)
        await got[1].ack()         # out of order
        await got[0].ack()
        await ob.writer.close(1.0)

    async def second():
        ob = Outbox(Writer(store), "cam")
        before = len(store.writes)
        assert await ob.restore() == b'{"decide": 1}'
        restore_writes = store.writes[before:]
        got = [(d.decision.decision_id, d.attempt) for d in await drain(ob)]
        await ob.writer.close(1.0)
        return got, restore_writes

    run(first())
    assert json.loads(store.load(store_key("outbox-meta", "cam")))["head"] == 2
    got, writes = run(second())
    assert got == [(decision(3).decision_id, 2), (decision(4).decision_id, 2)]
    assert [list(w) for w in writes] == [[store_key("outbox-meta", "cam")]]     # one small write
    assert run(second())[0] == [(decision(3).decision_id, 3), (decision(4).decision_id, 3)]  # never repeats


def test_store_keys_of_different_streams_never_collide():
    assert store_key("outbox", "a#000000000001") != store_key("outbox", "a", "#000000000001")


def test_watch_takeover_redelivers_unacknowledged_with_a_persisted_attempt():
    store = MemoryStore()

    async def main():
        ob = Outbox(Writer(store), "cam")
        await ob.restore()
        await ob.commit(None, (decision(1), decision(2)))
        got = [(d.decision.decision_id, d.attempt) for d in await drain(ob)]
        again = [(d.decision.decision_id, d.attempt) for d in await drain(ob, "auto")]  # takes over
        await ob.writer.close(1.0)
        return got, again, ob.pending()

    got, again, left = run(main())
    assert [a for _, a in got] == [1, 1] and [a for _, a in again] == [2, 2]
    assert [i for i, _ in got] == [i for i, _ in again]
    assert left == 0  # auto-acknowledged after the loop asked for more
    meta = json.loads(store.load(store_key("outbox-meta", "cam")))
    assert (meta["head"], meta["next"]) == (2, 2)


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
        watch = Watch(ob, "manual", 10, lambda n: None)
        first = await watch.__anext__()
        later = asyncio.ensure_future(ob.commit(None, (decision(2),)))   # held in the writer
        await asyncio.to_thread(store.holding.wait, 5)
        acked = asyncio.ensure_future(first.ack())                       # queued behind it
        await asyncio.sleep(0.05)
        store.release.set()
        await asyncio.gather(later, acked)
        await ob.writer.close(1.0)

    async def restored():
        ob = Outbox(Writer(store), "cam")
        await ob.restore()
        got = [d.decision.decision_id for d in await drain(ob)]
        await ob.writer.close(1.0)
        return got

    run(main())
    assert run(restored()) == [decision(2).decision_id]


# --------------------------------------------------------------------------- review findings


class LoadFails(MemoryStore):
    def __init__(self, failures: int) -> None:
        super().__init__()
        self.failures = failures

    def load(self, key):
        if self.failures:
            self.failures -= 1
            raise OSError("disk busy")
        return super().load(key)


def test_writer_retries_a_failed_load_visibly_and_close_stops_a_failing_one_promptly(monkeypatch):
    monkeypatch.setattr(Writer, "RETRY_FIRST_S", 0.01)

    async def transient():
        writer = Writer(LoadFails(2))
        assert await writer.load(["k"]) == [None]
        assert writer.failures == 2 and not writer.retrying
        await writer.close(1.0)

    async def permanent():
        writer = Writer(LoadFails(10**9))
        pending = asyncio.ensure_future(writer.load(["k"]))
        await asyncio.sleep(0.1)
        assert writer.retrying and writer.failures >= 2
        t0 = time.monotonic()
        await writer.close(5.0)
        with pytest.raises(OSError):
            await pending
        return time.monotonic() - t0

    run(transient())
    assert run(permanent()) < 1.5


def test_an_unreadable_meta_record_is_never_renumbered_or_overwritten():
    store = MemoryStore()
    meta_key = store_key("outbox-meta", "cam")
    store.commit({meta_key: b"{not json", delivery._entry_key("cam", 7): b"{}"})

    async def main():
        ob = Outbox(Writer(store), "cam")
        with pytest.raises(delivery.OutboxUnreadable):
            await ob.restore()
        await ob.writer.close(1.0)

    run(main())
    assert store.load(meta_key) == b"{not json" and store.load(delivery._entry_key("cam", 7)) == b"{}"


def test_restore_wakes_a_consumer_already_waiting():
    store = MemoryStore()

    async def seed():
        ob = Outbox(Writer(store), "cam")
        await ob.restore()
        await ob.commit(None, (decision(1),))
        await ob.writer.close(1.0)

    async def main():
        ob = Outbox(Writer(store), "cam")
        waiting = asyncio.ensure_future(Watch(ob, "auto", 10, lambda n: None).__anext__())
        await asyncio.sleep(0.05)
        await ob.restore()
        got = await asyncio.wait_for(waiting, 0.5)
        await ob.writer.close(1.0)
        return got.decision.decision_id

    run(seed())
    assert run(main()) == decision(1).decision_id


def test_sqlite_store_has_one_owner_and_private_files(tmp_path):
    path = tmp_path / "percept.db"
    a, b = SqliteStore(path), SqliteStore(path)
    a.lock()
    with pytest.raises(RuntimeError, match="in use"):
        b.lock()
    a.commit({"k": b"v"})
    for name in ("percept.db", "percept.db-wal", "percept.db.lock"):
        assert stat.S_IMODE(os.stat(tmp_path / name).st_mode) == 0o600, name
    with sqlite3.connect(path) as conn:
        assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    a.close()
    b.lock()                      # released with the first owner
    assert b.load("k") == b"v"
    b.close()


def test_a_consumer_that_never_reads_keeps_memory_bounded_while_everything_stays_durable(tmp_path):
    """R4: thousands of decisions with keyframes, nobody reading: memory stays flat, disk holds all."""
    store = SqliteStore(tmp_path / "s.db")
    big = [replace(decision(n), keyframe=os.urandom(8_000)) for n in range(3000)]

    async def main():
        ob = Outbox(Writer(store), "cam")
        await ob.restore()
        Watch(ob, "manual", 100, lambda n: None)             # attached, never iterated
        tracemalloc.start()
        before = tracemalloc.get_traced_memory()[0]
        for i in range(0, len(big), 100):
            await ob.commit(b'{"decide": 1}', tuple(big[i:i + 100]))
        grown = tracemalloc.get_traced_memory()[0] - before
        tracemalloc.stop()
        sizes = (len(getattr(ob, "ahead", ())), len(getattr(ob, "held", ())), getattr(ob, "entries", None))
        first = [(await Watch(ob, "auto", 10, lambda n: None).__anext__()).decision.decision_id]
        await ob.writer.close(5.0)
        return grown, sizes, first

    grown, (ahead, held, entries), first = run(main(), timeout=60)
    assert entries is None                                    # no full in-memory mirror
    assert ahead <= delivery.READ_AHEAD and held == 0
    assert grown < 2_000_000, grown            # 3000 x 8 KB keyframes would be 24 MB in memory
    assert first == [big[0].decision_id]
    with sqlite3.connect(tmp_path / "s.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM percept_kv WHERE key LIKE 'outbox:%'").fetchone()[0] == 3000
    store.close()
