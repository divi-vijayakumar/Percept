"""The trace: one JSON line per runtime event, for debugging and deterministic replay.

``Monitor(..., trace="percept-trace.jsonl")`` (or a :class:`TraceWriter`) records what the runtime
saw and did; :func:`replay` re-runs a trace through the decision reducer. Every record has ``v``
(the format, 1), ``t_wall``, ``stream_id`` and ``kind``:

=============  ===================================================================================
kind           fields
=============  ===================================================================================
restore        ``restored`` (a snapshot was loaded), ``checkpoint`` (re-recorded at the top of a
               new file), ``snapshot`` (the decide state the stream continues from)
conditions     ``conditions``: the full condition set (without triggers), written before the first
               input, whenever a condition is added, changed or removed, and at each checkpoint
decide_input   ``input`` (the exact decide input), ``now`` (the wall time it was decided at),
               ``versions`` (condition id -> version it was decided under)
decision       the decision's wire form (``Decision.to_dict()``) without evidence and keyframe
request        ``request_id``, ``judge``, ``purpose``, ``stream_epoch``, ``condition_ids``,
               ``usage`` (one entry per provider request), ``latency_s``, ``failure``
fate           a decide trace (``event``, ``condition_id``, ``subject``, ``fate``,
               ``occurrence_id``, ``detail``), or runtime fates counted per minute (``event:
               "runtime"``, ``fate``, ``condition_id``, ``n``, ``bucket_wall``)
health         ``health_kind``, ``detail``, ``count``, ``condition_id``, ``subject``
blind          ``blind_kind``, ``state`` (``open`` or ``close``), ``detail``, ``condition_id``
poke           ``condition_ids``, ``subject``, ``observed_at``, ``reason``, ``evidence_images``
=============  ===================================================================================

Files rotate by size. Under a ``Monitor`` each new file starts with a checkpoint (every stream's
``restore`` and ``conditions``), so the newest file replays on its own.

What is never written: images, keyframes or any other bytes, or raw source URIs. Every string is
scrubbed: URLs lose credentials, query and fragment, and values that look like keys, tokens or
passwords are replaced. Scrubbing is a pattern match, not a guarantee. What you write yourself is
recorded in clear: condition text and ``not_when``, poke reasons, feedback notes, and the judge's
reasons.

Writing never blocks the caller: records go to a bounded queue drained by one thread. When the
queue is full, the disk fails or another writer holds the path, records are dropped and counted
(``dropped``, ``failures``, ``error``); the runtime reports that as ``trace_dropped`` health. One
process writes one trace path: a second writer on the same path (an advisory lock on
``<path>.lock``) drops its records rather than interleave them.
"""

from __future__ import annotations

import glob
import json
import math
import os
import re
import threading
from collections import deque
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional, Union

import numpy as np

from percept.replay import ReplayResult, replay as _replay
from percept.sources import scrub as _scrub_urls

try:
    import fcntl
except ImportError:  # pragma: no cover - not on Windows: no lock, one writer is up to you
    fcntl = None  # type: ignore[assignment]

FORMAT = 1
MAX_BYTES = 50_000_000
BACKUPS = 4                     # path.1 ... path.4, plus the live file: five files at most
MAX_PENDING = 10_000            # records queued for the writer thread; more are dropped
BATCH = 1_000                   # records scrubbed and written per turn of the writer thread

_KEY_NAME = r"[\w.-]*(?:api[_-]?key|apikey|token|secret|passw(?:or)?d|pwd|credential|subscription[_-]?key)[\w.-]*"
_SECRETS = (
    (re.compile(rf"(?i)\b({_KEY_NAME})([\"']?\s*[=:]\s*)([\"']?)[^\s,;&\"'}}]+"), r"\1\2\3<redacted>"),
    (re.compile(r"(?i)\b(bearer|basic)\s+[A-Za-z0-9._~+/=-]{8,}"), r"\1 <redacted>"),
    (re.compile(r"\beyJ[A-Za-z0-9_-]{4,}\.[A-Za-z0-9_-]{4,}(?:\.[A-Za-z0-9_-]+)?"), "<redacted>"),
    (re.compile(r"\b(?:sk-[A-Za-z0-9_-]{16,}|AIza[0-9A-Za-z_-]{30,}|xox[abprs]-[A-Za-z0-9-]{10,}"
                r"|gh[pousr]_[A-Za-z0-9]{20,}|(?:AKIA|ASIA)[0-9A-Z]{16})\b"), "<redacted>"),
    (re.compile(r"\b[0-9a-fA-F]{32}\b"), "<redacted>"),                     # 32-hex keys
)
_SECRET_40 = re.compile(r"(?<![A-Za-z0-9/+=])[A-Za-z0-9/+=]{40}(?![A-Za-z0-9/+=])")   # cloud secret keys


def _secret_40(m: re.Match) -> str:
    s = m.group(0)
    mixed = any(c.islower() for c in s) and any(c.isupper() for c in s) and any(c in "/+0123456789" for c in s)
    return "<redacted>" if mixed else s


def scrub(text: str) -> str:
    """``text`` without credentials: URLs are redacted, key-, token- and password-shaped values
    replaced. A best effort: it recognises common shapes, not every secret."""
    text = _scrub_urls(text)
    for pattern, repl in _SECRETS:
        text = pattern.sub(repl, text)
    return _SECRET_40.sub(_secret_40, text)


def _clean(value: Any) -> Any:
    """A JSON-safe copy: strings scrubbed, numpy scalars as numbers, non-finite floats as strings,
    bytes and arrays never written."""
    if isinstance(value, np.generic) and not isinstance(value, (np.str_, np.bytes_, np.void)):
        value = value.item()
    if isinstance(value, str):
        return scrub(value)
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return "NaN" if math.isnan(value) else ("Infinity" if value > 0 else "-Infinity")
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return f"<{type(value).__name__} omitted>"


class _Checkpoint:
    """A rotation point in the queue: start a new file and write ``records`` at its top."""

    def __init__(self, records: list) -> None:
        self.records = records


class TraceWriter:
    """Appends trace records to ``path`` from a background thread, started on the first write.

    :meth:`write` is safe from any thread, never blocks on the disk and never raises. Files
    rotate at ``max_bytes`` into ``path.1`` .. ``path.<backups>``. ``written``, ``dropped``,
    ``failures``, ``rotations`` and ``error`` (the last problem, cleared once writing works again)
    say how it is going. With ``checkpoints`` on (a ``Monitor`` sets it), a full file only sets
    ``rotate_due``; the rotation happens at the next :meth:`checkpoint`, in queue order.
    """

    def __init__(self, path: Union[str, os.PathLike], *, max_bytes: int = MAX_BYTES, backups: int = BACKUPS,
                 max_pending: int = MAX_PENDING) -> None:
        if max_bytes < 1 or backups < 0 or max_pending < 1:
            raise ValueError("max_bytes and max_pending are >= 1, backups >= 0")
        self.path, self.max_bytes, self.backups, self.max_pending = Path(path), max_bytes, backups, max_pending
        self.written = self.dropped = self.failures = self.rotations = 0
        self.error: Optional[str] = None
        self.checkpoints = False
        self.rotate_due = False
        self._queue: deque = deque()
        self._cv = threading.Condition()
        self._busy = self._closed = False
        self._file: Any = None
        self._lock: Any = None
        self._base = 0                  # bytes of the checkpoint at the top of the current file
        self._thread: Optional[threading.Thread] = None

    def _enqueue(self, item: Any, bounded: bool) -> None:
        with self._cv:
            if self._closed or (bounded and len(self._queue) >= self.max_pending):
                self.dropped += 1 if bounded else len(item.records)
                return
            self._queue.append(item)
            if self._thread is None:
                self._thread = threading.Thread(target=self._run, name="percept-trace", daemon=True)
                self._thread.start()
            self._cv.notify()

    def write(self, record: dict) -> None:
        """Queue one record (a dict of plain values). Dropped and counted when the queue is full."""
        self._enqueue(record, True)

    def checkpoint(self, records: Iterable[dict]) -> None:
        """Rotate here, in queue order, and open the new file with ``records``."""
        self.rotate_due = False
        self._enqueue(_Checkpoint(list(records)), False)

    @property
    def pending(self) -> int:
        return len(self._queue)

    def flush(self, timeout: Optional[float] = None) -> bool:
        """Wait until every queued record has been written (or dropped). False on timeout."""
        with self._cv:
            return self._cv.wait_for(lambda: not self._queue and not self._busy, timeout)

    def close(self, timeout: Optional[float] = 10.0) -> None:
        """Write what is queued (waiting at most ``timeout``), then stop. Later writes are dropped."""
        with self._cv:
            self._closed = True
            self._cv.notify_all()
            thread = self._thread
        if thread is not None:
            thread.join(timeout)

    # ----------------------------------------------------------------- writer thread

    def _run(self) -> None:
        while True:
            with self._cv:
                self._cv.wait_for(lambda: self._queue or self._closed)
                batch = [self._queue.popleft() for _ in range(min(BATCH, len(self._queue)))]
                self._busy = bool(batch)
                if not batch and self._closed:
                    break
            error, n = None, sum(len(i.records) if isinstance(i, _Checkpoint) else 1 for i in batch)
            try:
                self._write(batch)
            except Exception as exc:  # noqa: BLE001 - a failing disk drops records, never raises
                error = scrub(f"{type(exc).__name__}: {exc}")[:200]
                self._close_file()
            with self._cv:
                if error is None:
                    self.written, self.error = self.written + n, None
                else:
                    self.failures, self.dropped, self.error = self.failures + 1, self.dropped + n, error
                self._busy = False
                self._cv.notify_all()
        self._close_file()
        if self._lock is not None:
            self._lock.close()          # releases the advisory lock

    def _open(self) -> None:
        if self._lock is None and fcntl is not None:
            lock = open(self.path.with_name(self.path.name + ".lock"), "a")
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                lock.close()
                raise RuntimeError(f"another writer holds {self.path.name}; one process per trace path") from None
            self._lock = lock
        self._file = open(self.path, "ab")
        if self._file.tell() > 0:           # a line cut short by a crash must not swallow ours
            with open(self.path, "rb") as f:
                f.seek(-1, os.SEEK_END)
                if f.read(1) != b"\n":
                    self._file.write(b"\n")

    def _write(self, batch: list) -> None:
        if self._file is None:
            self._open()
        chunk: list[str] = []
        for item in batch:
            if isinstance(item, _Checkpoint):
                self._put("".join(chunk))
                chunk = []
                self._rotate()
                self._file.write(self._encode(item.records).encode())
                self._base = self._file.tell()
            else:
                chunk.append(self._encode([item]))
        self._put("".join(chunk))
        self._file.flush()

    @staticmethod
    def _encode(records: list) -> str:
        return "".join(json.dumps(_clean(r), separators=(",", ":"), allow_nan=False) + "\n" for r in records)

    def _put(self, text: str) -> None:
        if not text:
            return
        data = text.encode()
        pos, limit = self._file.tell(), max(self.max_bytes, 2 * self._base)
        if pos > 0 and pos + len(data) > limit:
            if self.checkpoints and pos <= 2 * limit:
                self.rotate_due = True      # the runtime checkpoints soon; until then, go on
            else:
                self._rotate()
                self._base = 0
        self._file.write(data)

    def _rotate(self) -> None:
        self._close_file()
        if self.backups == 0:
            self.path.unlink(missing_ok=True)
        else:
            for i in range(self.backups - 1, 0, -1):
                older = self.path.with_name(f"{self.path.name}.{i}")
                if older.exists():
                    os.replace(older, self.path.with_name(f"{self.path.name}.{i + 1}"))
            if self.path.exists():
                os.replace(self.path, self.path.with_name(f"{self.path.name}.1"))
        self._file = open(self.path, "ab")
        self.rotations += 1

    def _close_file(self) -> None:
        if self._file is not None:
            try:
                self._file.close()
            except OSError:
                pass
            self._file = None


def read(path: Union[str, os.PathLike]) -> Iterator[dict]:
    """Every record of a trace, oldest first: the rotated files (``path.N`` .. ``path.1``), then
    ``path``. A line that isn't JSON (a write cut short by a crash) is skipped."""
    path = Path(path)
    rotated = sorted((p for p in path.parent.glob(f"{glob.escape(path.name)}.*") if p.suffix[1:].isdigit()),
                     key=lambda p: -int(p.suffix[1:]))
    for p in rotated + ([path] if path.exists() else []):
        with open(p, encoding="utf-8") as f:
            for line in f:
                try:
                    yield json.loads(line)
                except ValueError:
                    continue


def replay(path: Union[str, os.PathLike]) -> ReplayResult:
    """Replay the trace at ``path`` (see :func:`percept.replay.replay`)."""
    return _replay(read(path))
