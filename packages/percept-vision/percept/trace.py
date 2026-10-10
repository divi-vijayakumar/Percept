"""The trace: one JSON line per runtime event, for debugging and deterministic replay.

``Monitor(..., trace="percept-trace.jsonl")`` (or a :class:`TraceWriter`) records what the runtime
saw and did. Every record has ``v`` (the format, 1), ``t_wall``, ``stream_id`` and ``kind``:

=============  ===================================================================================
kind           fields
=============  ===================================================================================
restore        ``restored`` (a snapshot was loaded), ``snapshot`` (the decide state the stream
               starts from; a fresh state when nothing was restored)
conditions     ``conditions``: the full condition set (without triggers), written before the first
               input and whenever a condition is added, changed or removed
decide_input   ``input`` (the exact decide input), ``now`` (the wall time it was decided at),
               ``versions`` (condition id -> version it was decided under)
decision       ``decision_id``, ``occurrence_id``, ``type``, ``cause``, ``condition_id``,
               ``subject``, ``t_decided``
request        ``request_id``, ``judge``, ``purpose``, ``stream_epoch``, ``condition_ids``,
               ``usage`` (one entry per provider request), ``latency_s``, ``failure``
fate           a decide trace (``event``, ``condition_id``, ``subject``, ``fate``,
               ``occurrence_id``, ``detail``) or a runtime fate (``event: "runtime"``, ``fate``
               such as ``dropped_budget``, ``frames_capped``, ``evidence_unavailable``, ``n``)
health         ``health_kind``, ``detail``, ``count``, ``condition_id``, ``subject``
blind          ``blind_kind``, ``state`` (``open`` or ``close``), ``detail``, ``condition_id``
poke           ``condition_ids``, ``subject``, ``observed_at``, ``reason``, ``evidence_images``
=============  ===================================================================================

What is never written: images, keyframes or any other bytes, API keys, credentials, or raw source
URIs. Stream ids are the redacted ones, and every string is scrubbed: URLs lose credentials, query
and fragment, and anything shaped like a key or token is replaced.

Writing never blocks the caller: records go to a bounded queue drained by one thread, which
rotates files by size. When the queue is full or the disk fails, records are dropped and counted
(``dropped``, ``failures``, ``error``); the runtime reports that on the health channel.
"""

from __future__ import annotations

import glob
import json
import os
import re
import threading
from collections import deque
from pathlib import Path
from typing import Any, Iterator, Optional, Union

from percept.sources import scrub as _scrub_urls

FORMAT = 1
MAX_BYTES = 50_000_000
BACKUPS = 4                     # path.1 ... path.4, plus the live file: five files at most
MAX_PENDING = 10_000            # records queued for the writer thread; more are dropped

_SECRETS = (
    (re.compile(r"(?i)\b(api[_-]?key|access[_-]?token|auth[_-]?token|token|secret|password|passwd|pwd)"
                r"(\s*[=:]\s*)[\"']?[^\s,;&\"']+"), r"\1\2<redacted>"),
    (re.compile(r"(?i)\b(bearer|basic)\s+[A-Za-z0-9._~+/=-]{8,}"), r"\1 <redacted>"),
    (re.compile(r"\b(sk-[A-Za-z0-9_-]{16,}|AIza[0-9A-Za-z_-]{30,}|xox[abprs]-[A-Za-z0-9-]{10,})"), "<redacted>"),
)


def scrub(text: str) -> str:
    """``text`` without credentials: URLs are redacted, key- and token-shaped values replaced."""
    text = _scrub_urls(text)
    for pattern, repl in _SECRETS:
        text = pattern.sub(repl, text)
    return text


def _clean(value: Any) -> Any:
    """A JSON-safe copy: strings scrubbed, bytes and arrays never written."""
    if isinstance(value, str):
        return scrub(value)
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return f"<{type(value).__name__} omitted>"


class TraceWriter:
    """Appends trace records to ``path`` from a background thread.

    :meth:`write` is safe from any thread, never blocks on the disk and never raises. Files
    rotate at ``max_bytes`` into ``path.1`` .. ``path.<backups>``. ``written``, ``dropped``,
    ``failures`` and ``error`` say how it is going.
    """

    def __init__(self, path: Union[str, os.PathLike], *, max_bytes: int = MAX_BYTES, backups: int = BACKUPS,
                 max_pending: int = MAX_PENDING) -> None:
        if max_bytes < 1 or backups < 0 or max_pending < 1:
            raise ValueError("max_bytes and max_pending are >= 1, backups >= 0")
        self.path, self.max_bytes, self.backups, self.max_pending = Path(path), max_bytes, backups, max_pending
        self.written = self.dropped = self.failures = 0
        self.error: Optional[str] = None
        self._queue: deque[dict] = deque()
        self._cv = threading.Condition()
        self._busy = self._closed = False
        self._file: Any = None
        self._thread = threading.Thread(target=self._run, name="percept-trace", daemon=True)
        self._thread.start()

    def write(self, record: dict) -> None:
        """Queue one record (a dict of plain values). Dropped and counted when the queue is full."""
        with self._cv:
            if self._closed or len(self._queue) >= self.max_pending:
                self.dropped += 1
                return
            self._queue.append(record)
            self._cv.notify()

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
        self._thread.join(timeout)

    # ----------------------------------------------------------------- writer thread

    def _run(self) -> None:
        while True:
            with self._cv:
                self._cv.wait_for(lambda: self._queue or self._closed)
                batch = list(self._queue)
                self._queue.clear()
                self._busy = bool(batch)
                if not batch and self._closed:
                    break
            error = None
            try:
                self._write(batch)
            except Exception as exc:  # noqa: BLE001 - a failing disk drops records, never raises
                error = scrub(f"{type(exc).__name__}: {exc}")[:200]
                self._close_file()
            with self._cv:
                if error is None:
                    self.written += len(batch)
                else:
                    self.failures, self.dropped, self.error = self.failures + 1, self.dropped + len(batch), error
                self._busy = False
                self._cv.notify_all()
        self._close_file()

    def _write(self, batch: list[dict]) -> None:
        data = "".join(json.dumps(_clean(r), separators=(",", ":")) + "\n" for r in batch).encode()
        if self._file is None:
            self._file = open(self.path, "ab")
        if self._file.tell() > 0 and self._file.tell() + len(data) > self.max_bytes:
            self._rotate()
        self._file.write(data)
        self._file.flush()

    def _rotate(self) -> None:
        self._close_file()
        if self.backups == 0:
            self.path.unlink(missing_ok=True)
        else:
            for i in range(self.backups - 1, 0, -1):
                older = self.path.with_name(f"{self.path.name}.{i}")
                if older.exists():
                    os.replace(older, self.path.with_name(f"{self.path.name}.{i + 1}"))
            os.replace(self.path, self.path.with_name(f"{self.path.name}.1"))
        self._file = open(self.path, "ab")

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

