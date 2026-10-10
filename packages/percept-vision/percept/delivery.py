"""Durable delivery: the stores, the store's writer thread, the per-stream outbox and the
``Delivery`` envelope.

A decision is committed to its stream's outbox in the same atomic write as the decide snapshot
that produced it, before it is handed to a consumer, and stays there until it is acknowledged.
After a crash, :meth:`Outbox.restore` returns the last snapshot and every unacknowledged
decision, in order, so they are delivered again first with the same ``decision_id``.

``attempt`` counts deliveries that may have happened. It is persisted when it grows (on a
restore, and on a redelivery within one process), so it never repeats; after a crash it may skip
a number, because a committed decision may or may not have reached the consumer before the crash.

Store calls are synchronous (:class:`~percept.contracts.Store`). One :class:`Writer` thread makes
every call, in submission order, and the event loop awaits a future: a slow disk delays delivery,
never capture or judging. A commit that fails is retried until it succeeds or the writer is
closed; later commits wait behind it, so a decision is never overtaken by a snapshot that already
assumes it was delivered.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import queue
import sqlite3
import threading
from collections import OrderedDict
from typing import Any, Callable, Mapping, Optional

from percept.contracts import Decision

log = logging.getLogger("percept")
_warned_memory = False


# --------------------------------------------------------------------------- stores


class MemoryStore:
    """A :class:`~percept.contracts.Store` in this process's memory.

    For demos, tests and ephemeral workloads: decisions survive a consumer error (they are
    redelivered on the next ``watch()``), not a crash. Logs that once per process.
    """

    def __init__(self) -> None:
        global _warned_memory
        self._data: dict[str, bytes] = {}
        self._lock = threading.Lock()
        if not _warned_memory:
            _warned_memory = True
            log.warning("percept.MemoryStore keeps decisions in memory only: delivery is not crash-safe")

    def load(self, key: str) -> bytes | None:
        with self._lock:
            return self._data.get(key)

    def commit(self, writes: Mapping[str, bytes | None]) -> None:
        if not all(v is None or isinstance(v, bytes) for v in writes.values()):
            raise TypeError("store values are bytes or None")
        with self._lock:
            for key, value in writes.items():
                if value is None:
                    self._data.pop(key, None)
                else:
                    self._data[key] = value


class SqliteStore:
    """A durable :class:`~percept.contracts.Store` in one SQLite file, in WAL mode.

    ``commit`` is one transaction. The connection opens on first use; inside a ``Monitor`` the
    store's writer thread is its only user. ``synchronous=FULL``: a committed decision survives
    a process crash and a power loss.
    """

    def __init__(self, path: str | os.PathLike[str] = "percept.db") -> None:
        self.path = os.path.abspath(os.fspath(path))
        self._conn: sqlite3.Connection | None = None
        self._lock = threading.Lock()

    def __repr__(self) -> str:
        return f"SqliteStore({self.path!r})"

    def _db(self) -> sqlite3.Connection:
        if self._conn is None:
            conn = sqlite3.connect(self.path, isolation_level=None, check_same_thread=False)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=FULL")
            conn.execute("CREATE TABLE IF NOT EXISTS percept_kv (key TEXT PRIMARY KEY, value BLOB NOT NULL)")
            self._conn = conn
        return self._conn

    def load(self, key: str) -> bytes | None:
        with self._lock:
            row = self._db().execute("SELECT value FROM percept_kv WHERE key = ?", (key,)).fetchone()
        return None if row is None else bytes(row[0])

    def commit(self, writes: Mapping[str, bytes | None]) -> None:
        with self._lock:
            db = self._db()
            db.execute("BEGIN IMMEDIATE")
            try:
                for key, value in writes.items():
                    if value is None:
                        db.execute("DELETE FROM percept_kv WHERE key = ?", (key,))
                    else:
                        db.execute("INSERT INTO percept_kv (key, value) VALUES (?, ?) "
                                   "ON CONFLICT(key) DO UPDATE SET value = excluded.value", (key, bytes(value)))
                db.execute("COMMIT")
            except BaseException:
                db.execute("ROLLBACK")
                raise

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None


# --------------------------------------------------------------------------- the writer thread


def _settle(writer: Writer, fut: asyncio.Future, result: Any, error: BaseException | None) -> None:
    writer.pending -= 1
    if fut.done():
        return
    if error is None:
        fut.set_result(result)
    else:
        fut.set_exception(error)


class Writer:
    """One thread makes every store call, in submission order; the loop awaits futures.

    ``pending`` (calls submitted and not finished) and ``failures`` (failed commit attempts)
    are read by the runtime for ``store_lag`` health.
    """

    RETRY_FIRST_S, RETRY_MAX_S = 0.05, 1.0

    def __init__(self, store: Any) -> None:
        self.store = store
        self.pending = 0
        self.failures = 0
        self._queue: queue.SimpleQueue = queue.SimpleQueue()
        self._thread: threading.Thread | None = None
        self._abort = threading.Event()
        self._closed = False

    def _submit(self, fn: Callable[..., Any], *args: Any) -> asyncio.Future:
        loop = asyncio.get_running_loop()
        fut = loop.create_future()
        if self._closed:
            fut.set_exception(RuntimeError("the store writer is closed"))
            return fut
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, name="percept-store-writer", daemon=True)
            self._thread.start()
        self.pending += 1
        self._queue.put((fn, args, fut, loop))
        return fut

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                return
            fn, args, fut, loop = item
            try:
                result, error = fn(*args), None
            except BaseException as exc:  # noqa: BLE001 - handed to the awaiting coroutine
                result, error = None, exc
            try:
                loop.call_soon_threadsafe(_settle, self, fut, result, error)
            except RuntimeError:  # the loop is gone: nobody is waiting
                pass

    def _commit(self, writes: dict[str, bytes | None]) -> None:
        delay = self.RETRY_FIRST_S
        while True:
            try:
                self.store.commit(writes)
                return
            except Exception:
                self.failures += 1
                if self._abort.is_set():
                    raise
                self._abort.wait(delay)
                delay = min(self.RETRY_MAX_S, 2 * delay)

    async def commit(self, writes: Mapping[str, bytes | None]) -> None:
        await self._submit(self._commit, dict(writes))

    async def load(self, keys: list[str]) -> list[bytes | None]:
        return await self._submit(lambda ks: [self.store.load(k) for k in ks], list(keys))

    async def close(self, timeout: float) -> None:
        """Finish what is queued (up to ``timeout``), then stop; a failing commit is abandoned."""
        self._closed = True
        thread = self._thread
        if thread is None:
            return
        self._queue.put(None)
        await asyncio.to_thread(thread.join, timeout)
        if thread.is_alive():
            self._abort.set()
            await asyncio.to_thread(thread.join, timeout)


# --------------------------------------------------------------------------- the outbox


def store_key(kind: str, stream_id: str, suffix: str = "") -> str:
    """A store key; the stream id is JSON-quoted, so no stream id can collide with another."""
    return f"{kind}:{json.dumps(stream_id)}{suffix}"


def _entry_key(stream_id: str, n: int) -> str:
    return store_key("outbox", stream_id, f"#{n:012d}")


def _encode(decision: Decision, attempt: int) -> bytes:
    return json.dumps({"decision": decision.to_dict(), "attempt": attempt}).encode()


class _Entry:
    __slots__ = ("n", "decision", "attempt", "delivered", "acked")

    def __init__(self, n: int, decision: Decision, attempt: int) -> None:
        self.n, self.decision, self.attempt = n, decision, attempt
        self.delivered = self.acked = False


class Envelope:
    """A delivered decision (the :class:`~percept.contracts.Delivery` protocol).

    ``await ack()`` removes it from the outbox durably; acknowledging twice is harmless. After
    the monitor is closed an acknowledgement is dropped and the decision is redelivered later.
    """

    __slots__ = ("decision", "attempt", "_outbox", "_entry")

    def __init__(self, outbox: Outbox, entry: _Entry) -> None:
        self.decision, self.attempt = entry.decision, entry.attempt
        self._outbox, self._entry = outbox, entry

    def __repr__(self) -> str:
        return f"Delivery({self.decision.decision_id!r}, attempt={self.attempt})"

    async def ack(self) -> None:
        await self._outbox.ack(self._entry)


class Outbox:
    """One stream's committed, unacknowledged decisions, and its decide snapshot.

    Owned by the event loop. Durable layout: the snapshot, one record per decision numbered
    ``n``, and a meta record ``{"head", "next"}``: every unacknowledged decision has
    ``head <= n < next``.
    """

    def __init__(self, writer: Writer, stream_id: str) -> None:
        self.writer, self.stream_id = writer, stream_id
        self.entries: OrderedDict[int, _Entry] = OrderedDict()   # committed, not acknowledged
        self._live: OrderedDict[int, None] = OrderedDict()       # allocated, not acknowledged
        self.head = self.next = 0
        self.cursor = 0                       # the next ``n`` the current watcher receives
        self.watcher: Optional[Watch] = None
        self.finished = self.closed = False
        self.changed = asyncio.Event()
        self.committed = self.delivered = self.redelivered = self.acked = 0

    def _meta(self) -> bytes:
        return json.dumps({"head": self.head, "next": self.next}).encode()

    def _wake(self) -> None:
        self.changed.set()

    async def restore(self) -> Optional[dict]:
        """Load the snapshot and the unacknowledged decisions; bump and persist their attempts."""
        snap_key, meta_key = store_key("snapshot", self.stream_id), store_key("outbox-meta", self.stream_id)
        snap, meta = await self.writer.load([snap_key, meta_key])
        if meta is not None:
            info = json.loads(meta)
            head, self.next = int(info["head"]), int(info["next"])
            raws = await self.writer.load([_entry_key(self.stream_id, n) for n in range(head, self.next)])
            writes: dict[str, bytes | None] = {}
            for n, raw in zip(range(head, self.next), raws):
                if raw is None:
                    continue  # acknowledged out of order
                record = json.loads(raw)
                entry = _Entry(n, Decision.from_dict(record["decision"]), int(record["attempt"]) + 1)
                self.entries[n] = entry
                self._live[n] = None
                writes[_entry_key(self.stream_id, n)] = _encode(entry.decision, entry.attempt)
            self.head = next(iter(self._live), self.next)
            self.cursor = self.head
            if writes:
                await self.writer.commit(writes)
        return None if snap is None else json.loads(snap)

    async def commit(self, snapshot: bytes | None, decisions: tuple[Decision, ...] = ()) -> None:
        """Commit the snapshot and new decisions in one atomic write; then they can be delivered."""
        writes: dict[str, bytes | None] = {}
        if snapshot is not None:
            writes[store_key("snapshot", self.stream_id)] = snapshot
        numbered = []
        for decision in decisions:
            n, self.next = self.next, self.next + 1
            self._live[n] = None
            numbered.append((n, decision))
            writes[_entry_key(self.stream_id, n)] = _encode(decision, 1)
        if decisions:
            writes[store_key("outbox-meta", self.stream_id)] = self._meta()
        if not writes:
            return
        await self.writer.commit(writes)
        for n, decision in numbered:
            self.entries[n] = _Entry(n, decision, 1)
        self.committed += len(numbered)
        self._wake()

    async def ack(self, entry: _Entry) -> None:
        if entry.acked or self.closed:
            return
        entry.acked = True
        self.entries.pop(entry.n, None)
        self._live.pop(entry.n, None)
        writes: dict[str, bytes | None] = {_entry_key(self.stream_id, entry.n): None}
        head = next(iter(self._live), self.next)
        if head != self.head:
            self.head = head
            writes[store_key("outbox-meta", self.stream_id)] = self._meta()
        self.acked += 1
        self._wake()
        try:
            await self.writer.commit(writes)
        except Exception:  # noqa: BLE001 - the writer closed meanwhile: delivered again after a restart
            pass

    def attach(self, watcher: Watch) -> None:
        """A new consumer takes over: every unacknowledged decision is delivered to it again."""
        self.watcher = watcher
        self.cursor = next(iter(self.entries), self.next)
        self._wake()

    def unacked(self) -> int:
        """Decisions delivered to the current watcher and not acknowledged."""
        count = 0
        for n in self.entries:
            if n >= self.cursor:
                break
            count += 1
        return count

    def upcoming(self) -> Optional[_Entry]:
        for n, entry in self.entries.items():
            if n >= self.cursor:
                return entry
        return None

    def finish(self) -> None:
        self.finished = True
        self._wake()

    def close(self) -> None:
        self.closed = True
        self._wake()


class Watch:
    """The async iterator ``Monitor.watch()`` returns: one stream's decisions, in order.

    ``ack="auto"`` acknowledges a decision when the consumer asks for the next one, that is
    after the loop body returned without raising. If the body raises, nothing is acknowledged
    and the decision is delivered again by the next ``watch()`` of the stream (or after a
    restart). ``ack="manual"``: call ``await delivery.ack()``; at most ``max_unacked``
    decisions are held unacknowledged, and past that delivery waits (capture and judging don't).
    """

    def __init__(self, outbox: Outbox, ack: str, max_unacked: int, on_blocked: Callable[[int], None]) -> None:
        if ack not in ("auto", "manual"):
            raise ValueError("ack is 'auto' or 'manual'")
        if not (isinstance(max_unacked, int) and max_unacked >= 1):
            raise ValueError("max_unacked is an integer >= 1")
        self.outbox, self.ack, self.max_unacked, self.on_blocked = outbox, ack, max_unacked, on_blocked
        self._last: Optional[Envelope] = None
        outbox.attach(self)

    def __aiter__(self) -> Watch:
        return self

    async def __anext__(self) -> Envelope:
        ob = self.outbox
        if self._last is not None:
            last, self._last = self._last, None
            if ob.watcher is self:
                await last.ack()
        blocked = False
        while True:
            ob.changed.clear()
            if ob.watcher is not self or ob.closed:
                raise StopAsyncIteration
            entry = ob.upcoming()
            if entry is None:
                if ob.finished:
                    raise StopAsyncIteration
            elif self.ack == "auto" or ob.unacked() < self.max_unacked:
                break
            elif not blocked:
                blocked = True
                self.on_blocked(ob.unacked())
            await ob.changed.wait()
        ob.cursor = entry.n + 1
        if entry.delivered:  # delivered before in this process: persist the new attempt first
            entry.attempt += 1
            ob.redelivered += 1
            try:
                await ob.writer.commit({_entry_key(ob.stream_id, entry.n): _encode(entry.decision, entry.attempt)})
            except Exception:  # noqa: BLE001 - the writer closed meanwhile
                raise StopAsyncIteration from None
        entry.delivered = True
        ob.delivered += 1
        envelope = Envelope(ob, entry)
        if self.ack == "auto":
            self._last = envelope
        return envelope
