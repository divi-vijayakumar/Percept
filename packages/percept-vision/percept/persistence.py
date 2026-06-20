"""Persistence — durable goals / entities / fire-log so the session survives disconnect.

This is what turns the *architectural* endurance property (bounded structured state) into a
deployable claim: kill the process, restart, rehydrate. Two impls sit behind the frozen
``Persistence`` protocol (``save(key, value)`` / ``load(key)``):

* :class:`InMemoryPersistence` — the default; a dict (lost on exit).
* :class:`SqlitePersistence` — the durable reference, **stdlib ``sqlite3``**, a JSON key/value
  table. A file-backed store survives a restart; ``":memory:"`` is handy for tests.

Values are whatever JSON-able snapshot the core hands down (see ``Percept.snapshot``). The
persistence layer is dumb on purpose — it stores opaque blobs; the core owns the schema.
"""

from __future__ import annotations

import json
import sqlite3
from typing import Any


class InMemoryPersistence:
    """Non-durable default — a plain dict. Useful when endurance isn't needed (or in tests)."""

    def __init__(self) -> None:
        self._store: dict[str, Any] = {}

    def save(self, key: str, value: Any) -> None:
        # round-trip through JSON so behaviour matches the durable backend (no aliasing surprises)
        self._store[key] = json.loads(json.dumps(value))

    def load(self, key: str) -> Any:
        return self._store.get(key)


class SqlitePersistence:
    """Durable reference store — a single JSON key/value table in SQLite (stdlib)."""

    def __init__(self, path: str = ":memory:") -> None:
        self._conn = sqlite3.connect(path)
        self._conn.execute("CREATE TABLE IF NOT EXISTS kv (key TEXT PRIMARY KEY, value TEXT)")
        self._conn.commit()

    def save(self, key: str, value: Any) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO kv (key, value) VALUES (?, ?)", (key, json.dumps(value))
        )
        self._conn.commit()

    def load(self, key: str) -> Any:
        row = self._conn.execute("SELECT value FROM kv WHERE key = ?", (key,)).fetchone()
        return json.loads(row[0]) if row is not None else None

    def keys(self) -> list[str]:
        return [r[0] for r in self._conn.execute("SELECT key FROM kv").fetchall()]

    def close(self) -> None:
        self._conn.close()


__all__ = ["InMemoryPersistence", "SqlitePersistence"]
