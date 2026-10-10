"""Durable delivery: the stores, the store's writer thread, the per-stream outbox and the
``Delivery`` envelope.

A decision is committed to its stream's outbox in the same atomic write as the decide snapshot
that produced it, before it is handed to a consumer, and stays there until it is acknowledged.
After a restart the stream's unacknowledged decisions are delivered again first, in order, with
the same ``decision_id``.

Memory stays bounded however far the consumer falls behind: the outbox keeps in memory only the
decisions delivered and not yet acknowledged (at most ``max_unacked``) and a short read-ahead of
new ones; everything else is read back from the store when its turn comes.

``attempt`` counts deliveries that may have happened, and never repeats: it grows by one on every
redelivery within a process and on every restart (a restart counts as a possible delivery,
because a committed decision may or may not have reached the consumer before it). Restarts are
counted once, in the outbox's meta record, so a restore rewrites nothing per decision.

Store calls are synchronous (:class:`~percept.contracts.Store`). One :class:`Writer` thread makes
every call, in submission order, and the event loop awaits a future: a slow disk delays delivery,
never capture or judging. A load or commit that fails is retried until it succeeds or the writer
is closed; later calls wait behind it, so a decision is never overtaken by a snapshot that already
assumes it was delivered. ``Writer.failures`` and ``Writer.retrying`` make the retries visible.
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

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows: no advisory lock
    fcntl = None  # type: ignore[assignment]

log = logging.getLogger("percept")
_warned_memory = False
READ_AHEAD = 16        # committed decisions kept in memory ahead of the consumer


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


class StoreLocked(RuntimeError):
    """Another process (or another store object) already owns this database."""


class SqliteStore:
    """A durable :class:`~percept.contracts.Store` in one SQLite file, in WAL mode.

    ``commit`` is one transaction with ``synchronous=FULL``: a committed decision survives a
    process crash and a power loss. The file (and its ``-wal``/``-shm`` companions) is created
    readable by its owner only, since it holds keyframes. One owner at a time: :meth:`lock` takes
    an exclusive advisory lock on ``<path>.lock`` (on platforms with ``fcntl``) and raises
    :class:`StoreLocked` if another process holds it; the monitor takes it when its first stream
    starts. Inside a ``Monitor`` the store's writer thread is its only user.
    """

    def __init__(self, path: str | os.PathLike[str] = "percept.db") -> None:
        self.path = os.path.abspath(os.fspath(path))
        self._conn: sqlite3.Connection | None = None
        self._lockfd: int | None = None
        self._lock = threading.Lock()

    def __repr__(self) -> str:
        return f"SqliteStore({self.path!r})"

    def lock(self) -> None:
        """Become the database's only owner (idempotent)."""
        with self._lock:
            self._acquire()

    def _acquire(self) -> None:
        if self._lockfd is not None:
            return
        fd = os.open(self.path + ".lock", os.O_CREAT | os.O_RDWR, 0o600)
        if fcntl is not None:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                os.close(fd)
                raise StoreLocked(f"{self.path} is in use by another Monitor or process "
                                  f"(lock file {self.path}.lock); give each one its own store path") from None
        self._lockfd = fd

    def _db(self) -> sqlite3.Connection:
        if self._conn is None:
            self._acquire()
            os.close(os.open(self.path, os.O_CREAT | os.O_RDWR, 0o600))  # -wal/-shm inherit this mode
            conn = sqlite3.connect(self.path, isolation_level=None, check_same_thread=False)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=FULL")
            conn.execute("PRAGMA busy_timeout=5000")
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
        """Close the connection and give up ownership."""
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None
            if self._lockfd is not None:
                os.close(self._lockfd)   # releases the flock
                self._lockfd = None


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

    ``pending`` (calls submitted and not finished), ``failures`` (failed store calls) and
    ``retrying`` (a call is failing right now) are read by the runtime for ``store_lag`` health
    and ``stats()``.
    """

    RETRY_FIRST_S, RETRY_MAX_S = 0.05, 1.0

    def __init__(self, store: Any) -> None:
        self.store = store
        self.pending = 0
        self.failures = 0
        self.retrying = False
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
                result, error = self._retried(fn, *args), None
            except BaseException as exc:  # noqa: BLE001 - handed to the awaiting coroutine
                result, error = None, exc
            try:
                loop.call_soon_threadsafe(_settle, self, fut, result, error)
            except RuntimeError:  # the loop is gone: nobody is waiting
                pass

    def _retried(self, fn: Callable[..., Any], *args: Any) -> Any:
        """Run ``fn`` until it succeeds, backing off; give up only when aborted."""
        delay = self.RETRY_FIRST_S
        while True:
            try:
                result = fn(*args)
                self.retrying = False
                return result
            except StoreLocked:
                raise
            except Exception:
                self.failures += 1
                self.retrying = True
                if self._abort.is_set():
                    raise
                self._abort.wait(delay)
                delay = min(self.RETRY_MAX_S, 2 * delay)

    async def commit(self, writes: Mapping[str, bytes | None]) -> None:
        await self._submit(self.store.commit, dict(writes))

    async def load(self, keys: list[str]) -> list[bytes | None]:
        return await self._submit(lambda ks: [self.store.load(k) for k in ks], list(keys))

    async def close(self, timeout: float) -> None:
        """Finish what is queued (up to ``timeout``), then stop. A call that is failing is
        abandoned at once: retrying against a broken store would only delay shutdown."""
        self._closed = True
        thread = self._thread
        if thread is None:
            return
        self._queue.put(None)
        if self.retrying:
            self._abort.set()
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


def _attempt_key(stream_id: str, n: int) -> str:
    return store_key("outbox-attempt", stream_id, f"#{n:012d}")


class OutboxUnreadable(ValueError):
    """The outbox's meta record can't be read: its numbering is unknown, so nothing may be written."""


class _Entry:
    __slots__ = ("n", "decision", "attempt", "acked")

    def __init__(self, n: int, decision: Decision, attempt: int) -> None:
        self.n, self.decision, self.attempt, self.acked = n, decision, attempt, False


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

    Owned by the event loop. Durable layout: the snapshot; one record per decision numbered
    ``n`` (with the incarnation that committed it); an optional attempt record per decision,
    written when it is redelivered within a process; and a meta record ``{head, next, inc}``:
    every unacknowledged decision has ``head <= n < next``, and ``inc`` counts restores.

    In memory: ``held`` (delivered, not acknowledged) and ``ahead`` (a read-ahead of committed
    decisions not yet delivered, at most ``READ_AHEAD``). Every live ``n`` is in ``held`` or at
    or after ``cursor``, so ``head = min(held, cursor)``.
    """

    def __init__(self, writer: Writer, stream_id: str, on_error: Callable[[str], None] = lambda detail: None) -> None:
        self.writer, self.stream_id, self.on_error = writer, stream_id, on_error
        self.held: OrderedDict[int, _Entry] = OrderedDict()
        self.ahead: OrderedDict[int, _Entry] = OrderedDict()
        self.head = self.next = self.committed_next = self.cursor = 0
        self.inc = 0
        self.watcher: Optional[Watch] = None
        self.finished = self.closed = False
        self.changed = asyncio.Event()
        self.committed = self.delivered = self.redelivered = self.acked = 0

    def _meta(self) -> bytes:
        return json.dumps({"head": self.head, "next": self.next, "inc": self.inc}).encode()

    def _wake(self) -> None:
        self.changed.set()

    async def restore(self) -> Optional[bytes]:
        """Read the meta record and return the raw snapshot (None for a new stream).

        Pending decisions stay on disk and are read back one by one as they are delivered.
        Raises :class:`OutboxUnreadable` when the meta record is corrupt: nothing is renumbered
        or overwritten then.
        """
        snap, meta = await self.writer.load([store_key("snapshot", self.stream_id),
                                             store_key("outbox-meta", self.stream_id)])
        if meta is not None:
            try:
                info = json.loads(meta)
                head, nxt, inc = int(info["head"]), int(info["next"]), int(info.get("inc", 0))
            except (ValueError, TypeError, KeyError) as exc:
                raise OutboxUnreadable(f"outbox meta of {self.stream_id!r} is unreadable: {exc!r}") from None
            self.head = self.cursor = head
            self.next = self.committed_next = nxt
            self.inc = inc + 1
            await self.writer.commit({store_key("outbox-meta", self.stream_id): self._meta()})
        self._wake()
        return snap

    async def commit(self, snapshot: bytes | None, decisions: tuple[Decision, ...] = ()) -> None:
        """Commit the snapshot and new decisions in one atomic write; then they can be delivered."""
        writes: dict[str, bytes | None] = {}
        if snapshot is not None:
            writes[store_key("snapshot", self.stream_id)] = snapshot
        numbered = []
        for decision in decisions:
            n, self.next = self.next, self.next + 1
            numbered.append((n, decision))
            writes[_entry_key(self.stream_id, n)] = json.dumps({"decision": decision.to_dict(),
                                                                "inc": self.inc}).encode()
        if decisions:
            writes[store_key("outbox-meta", self.stream_id)] = self._meta()
        if not writes:
            return
        await self.writer.commit(writes)
        for n, decision in numbered:
            self.committed_next = n + 1
            if len(self.ahead) < READ_AHEAD:
                self.ahead[n] = _Entry(n, decision, 1)
        self.committed += len(numbered)
        self._wake()

    async def take(self, n: int) -> Optional[_Entry]:
        """Decision ``n`` for delivery, with its attempt; None when it was acknowledged (or the
        watcher was replaced meanwhile). ``cursor`` moves past ``n`` only once ``n`` is held, so
        ``head`` never passes a live decision."""
        watcher = self.watcher
        entry = self.held.get(n)
        if entry is not None:                       # delivered before in this process
            entry.attempt += 1
            self.redelivered += 1
            await self.writer.commit({_attempt_key(self.stream_id, n):
                                      json.dumps({"attempt": entry.attempt, "inc": self.inc}).encode()})
        else:
            entry = self.ahead.pop(n, None)
            if entry is None:
                record, attempt = await self.writer.load([_entry_key(self.stream_id, n),
                                                          _attempt_key(self.stream_id, n)])
                if watcher is not self.watcher:
                    return None
                if record is None:
                    self.cursor = n + 1
                    return None
                try:
                    rec = json.loads(record)
                    decision = Decision.from_dict(rec["decision"])
                    if attempt is not None:
                        att = json.loads(attempt)
                        number = int(att["attempt"]) + self.inc - int(att["inc"])
                    else:
                        number = 1 + self.inc - int(rec["inc"])
                except (ValueError, TypeError, KeyError) as exc:
                    self.on_error(f"decision record {n} is unreadable and stays on disk: {exc!r}")
                    self.cursor = n + 1
                    return None
                entry = _Entry(n, decision, number)
            self.held[n] = entry
        if watcher is not self.watcher:
            return None
        self.cursor = n + 1
        for stale in [k for k in self.ahead if k < self.cursor]:
            del self.ahead[stale]
        return entry

    async def ack(self, entry: _Entry) -> None:
        if entry.acked or self.closed:
            return
        entry.acked = True
        self.held.pop(entry.n, None)
        writes: dict[str, bytes | None] = {_entry_key(self.stream_id, entry.n): None,
                                           _attempt_key(self.stream_id, entry.n): None}
        head = min([*self.held, self.cursor])
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
        self.cursor = self.head
        self._wake()

    def unacked(self) -> int:
        """Decisions delivered to the current watcher and not acknowledged."""
        return sum(1 for n in self.held if n < self.cursor)

    def pending(self) -> int:
        """Unacknowledged decisions (exact within a process; after a restart an upper bound until
        the ones acknowledged out of order before it have been passed)."""
        return len(self.held) + max(0, self.committed_next - self.cursor)

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
    restart). The same holds for a ``break``: the decision the body just processed was not
    acknowledged, so after ``break`` and ``aclose()`` it is delivered again on the next start;
    consumers dedupe on ``decision_id``. ``ack="manual"``: call ``await delivery.ack()``; at most
    ``max_unacked`` decisions are held unacknowledged, and past that delivery waits (capture and
    judging don't).
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
            if ob.cursor < ob.committed_next:
                if self.ack == "auto" or ob.unacked() < self.max_unacked:
                    try:
                        entry = await ob.take(ob.cursor)
                    except Exception:  # noqa: BLE001 - the writer closed meanwhile
                        raise StopAsyncIteration from None
                    if entry is None:
                        continue
                    break
                if not blocked:
                    blocked = True
                    self.on_blocked(ob.unacked())
            elif ob.finished:
                raise StopAsyncIteration
            await ob.changed.wait()
        ob.delivered += 1
        envelope = Envelope(ob, entry)
        if self.ack == "auto":
            self._last = envelope
        return envelope
