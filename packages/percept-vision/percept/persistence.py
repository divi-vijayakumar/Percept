"""Persistence — durable goals / entities / fire-log so the session survives disconnect.

This is what turns the *architectural* endurance property (bounded structured state) into a
deployable claim: kill the process, restart, rehydrate. Two impls sit behind the frozen
``Persistence`` protocol (``save(key, value)`` / ``load(key)``):

* :class:`InMemoryPersistence` — the default; a dict (lost on exit).
* :class:`SqlitePersistence` — the durable reference, **stdlib ``sqlite3``**, a JSON key/value
  table. A file-backed store survives a restart; ``":memory:"`` is handy for tests.

Values are whatever JSON-able snapshot the core hands down (see ``Percept.snapshot``). The
persistence layer is dumb on purpose — it stores opaque blobs; the core owns the schema.

``SqlitePersistence`` additionally advertises the optional
``TransactionalOutboxPersistence`` capability. It does not change ``save``/``load``; the runtime
uses the capability only for a bare reference ``WebhookSink``, leaving legacy stores and other
sinks on their existing paths.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
from dataclasses import dataclass, field
from typing import Any, Iterable, Protocol, runtime_checkable


TRANSACTIONAL_OUTBOX_CAPABILITY = "percept.persistence.transactional_outbox.v1"
_OUTBOX_TABLE = "percept_delivery_outbox"
_SINK_REF = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


class OutboxError(RuntimeError):
    """Base error for the optional durable-delivery capability."""


class OutboxValidationError(OutboxError, ValueError):
    """A pending delivery cannot be represented by the canonical outbox schema."""


class OutboxEventCollision(OutboxError):
    """One stable event ID was reused for a different payload, schema, or sink route."""


class OutboxCorruptError(OutboxError):
    """A durable outbox row failed its canonical payload hash check."""


def _finite_timestamp(value: object, field_name: str, *, optional: bool = False) -> float | None:
    if value is None and optional:
        return None
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
        or value < 0
    ):
        raise OutboxValidationError(f"{field_name} must be a finite non-negative number")
    return float(value)


def _canonical_payload(payload: object) -> tuple[Any, str, str]:
    """Return a detached JSON value, its canonical encoding, and its SHA-256."""

    try:
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        detached = json.loads(encoded)
    except (TypeError, ValueError) as exc:
        raise OutboxValidationError(f"delivery payload must be strict JSON: {exc}") from exc
    digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    return detached, encoded, digest


def _validate_event_id(event_id: object) -> str:
    if not isinstance(event_id, str) or not event_id or len(event_id) > 512:
        raise OutboxValidationError("event_id must be a non-empty string of at most 512 characters")
    return event_id


def _validate_sink_ref(sink_config_ref: object) -> str:
    if not isinstance(sink_config_ref, str) or not _SINK_REF.fullmatch(sink_config_ref):
        raise OutboxValidationError(
            "sink_config_ref must be an opaque identifier, not a URL, header, or credential"
        )
    return sink_config_ref


@dataclass(frozen=True)
class PendingDelivery:
    """Canonical, credential-free durable webhook envelope.

    ``sink_config_ref`` is an opaque lookup key.  The URL, authorization headers, signing key, and
    other credentials remain in hosted configuration and are never accepted by this record.
    ``payload_sha256`` covers the strict canonical JSON bytes returned by :attr:`payload_json`.
    """

    event_id: str
    payload: Any
    sink_config_ref: str
    created_at: float
    schema_version: int = 1
    attempts: int = 0
    updated_at: float | None = None
    last_attempt_at: float | None = None
    next_attempt_at: float | None = None
    payload_sha256: str = field(init=False)
    _payload_json: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        event_id = _validate_event_id(self.event_id)
        sink_ref = _validate_sink_ref(self.sink_config_ref)
        if type(self.schema_version) is not int or self.schema_version < 1:
            raise OutboxValidationError("schema_version must be a positive integer")
        if type(self.attempts) is not int or self.attempts < 0:
            raise OutboxValidationError("attempts must be a non-negative integer")
        created_at = _finite_timestamp(self.created_at, "created_at")
        updated_at = _finite_timestamp(
            self.created_at if self.updated_at is None else self.updated_at,
            "updated_at",
        )
        last_attempt_at = _finite_timestamp(
            self.last_attempt_at, "last_attempt_at", optional=True
        )
        next_attempt_at = _finite_timestamp(
            self.next_attempt_at, "next_attempt_at", optional=True
        )
        assert created_at is not None and updated_at is not None
        if updated_at < created_at:
            raise OutboxValidationError("updated_at cannot precede created_at")
        if last_attempt_at is not None and last_attempt_at < created_at:
            raise OutboxValidationError("last_attempt_at cannot precede created_at")
        if next_attempt_at is not None and next_attempt_at < created_at:
            raise OutboxValidationError("next_attempt_at cannot precede created_at")

        payload, payload_json, payload_sha256 = _canonical_payload(self.payload)
        if not isinstance(payload, dict):
            raise OutboxValidationError("delivery payload must be a JSON object")
        wire_ids = [payload[key] for key in ("id", "event_id") if key in payload]
        if not wire_ids or any(value != event_id for value in wire_ids):
            raise OutboxValidationError(
                "delivery payload must carry the same stable event ID in `id` or `event_id`"
            )

        object.__setattr__(self, "event_id", event_id)
        object.__setattr__(self, "payload", payload)
        object.__setattr__(self, "sink_config_ref", sink_ref)
        object.__setattr__(self, "created_at", created_at)
        object.__setattr__(self, "updated_at", updated_at)
        object.__setattr__(self, "last_attempt_at", last_attempt_at)
        object.__setattr__(self, "next_attempt_at", next_attempt_at)
        object.__setattr__(self, "payload_sha256", payload_sha256)
        object.__setattr__(self, "_payload_json", payload_json)

    @property
    def payload_json(self) -> str:
        """The canonical strict-JSON payload persisted and hashed by the outbox."""

        return self._payload_json


@runtime_checkable
class TransactionalOutboxPersistence(Protocol):
    """Optional persistence capability required for crash-safe hosted webhook delivery."""

    outbox_capability: str

    def commit_state_and_enqueue(
        self,
        state: Any,
        deliveries: Iterable[PendingDelivery],
        *,
        state_key: str = "percept",
    ) -> None: ...

    def load_pending_deliveries(self, *, limit: int | None = None) -> list[PendingDelivery]: ...

    def note_delivery_attempt(
        self,
        event_id: str,
        *,
        attempted_at: float,
        next_attempt_at: float | None = None,
    ) -> bool: ...

    def ack_delivery(
        self,
        event_id: str,
        delivered_ledger_state: Any,
        *,
        delivered_at: float,
        delivered_ledger_key: str = "percept_delivery_ledger",
    ) -> bool: ...


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
    """Durable reference store with an optional transactional delivery-outbox capability.

    Existing ``save``/``load`` behavior remains the base contract.  Hosted runtimes can detect the
    additive capability with ``isinstance(store, TransactionalOutboxPersistence)`` or by comparing
    :attr:`outbox_capability` to :data:`TRANSACTIONAL_OUTBOX_CAPABILITY`.
    """

    outbox_capability = TRANSACTIONAL_OUTBOX_CAPABILITY

    def __init__(self, path: str = ":memory:") -> None:
        self._conn = sqlite3.connect(path)
        self._conn.execute("CREATE TABLE IF NOT EXISTS kv (key TEXT PRIMARY KEY, value TEXT)")
        self._ensure_outbox_schema()
        self._conn.commit()

    def _ensure_outbox_schema(self) -> None:
        """Create or additively migrate the outbox without disturbing an existing ``kv`` store."""

        self._conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {_OUTBOX_TABLE} (
                event_id TEXT PRIMARY KEY,
                payload_json TEXT NOT NULL,
                schema_version INTEGER NOT NULL DEFAULT 1,
                payload_sha256 TEXT NOT NULL DEFAULT '',
                sink_config_ref TEXT NOT NULL DEFAULT 'legacy-default',
                attempts INTEGER NOT NULL DEFAULT 0,
                created_at REAL NOT NULL DEFAULT 0,
                updated_at REAL NOT NULL DEFAULT 0,
                last_attempt_at REAL,
                next_attempt_at REAL
            )
            """
        )
        columns = {
            row[1] for row in self._conn.execute(f"PRAGMA table_info({_OUTBOX_TABLE})")
        }
        additions = {
            "schema_version": "INTEGER NOT NULL DEFAULT 1",
            "payload_sha256": "TEXT NOT NULL DEFAULT ''",
            "sink_config_ref": "TEXT NOT NULL DEFAULT 'legacy-default'",
            "attempts": "INTEGER NOT NULL DEFAULT 0",
            "created_at": "REAL NOT NULL DEFAULT 0",
            "updated_at": "REAL NOT NULL DEFAULT 0",
            "last_attempt_at": "REAL",
            "next_attempt_at": "REAL",
        }
        required_base = {"event_id", "payload_json"}
        if not required_base.issubset(columns):
            missing = ", ".join(sorted(required_base - columns))
            raise OutboxCorruptError(f"outbox table is missing required columns: {missing}")
        for name, declaration in additions.items():
            if name not in columns:
                self._conn.execute(
                    f"ALTER TABLE {_OUTBOX_TABLE} ADD COLUMN {name} {declaration}"
                )

        # A v0 row had only event_id + payload_json. Canonicalize and hash it during the additive
        # migration so every row read through the v1 API has the same integrity check.
        for event_id, payload_json, payload_sha256 in self._conn.execute(
            f"SELECT event_id, payload_json, payload_sha256 FROM {_OUTBOX_TABLE}"
        ).fetchall():
            if payload_sha256:
                continue
            try:
                payload, canonical, digest = _canonical_payload(json.loads(payload_json))
                if not isinstance(payload, dict):
                    raise OutboxValidationError("legacy payload is not an object")
            except (json.JSONDecodeError, OutboxValidationError) as exc:
                raise OutboxCorruptError(
                    f"cannot migrate outbox event {event_id!r}: {exc}"
                ) from exc
            self._conn.execute(
                f"UPDATE {_OUTBOX_TABLE} SET payload_json = ?, payload_sha256 = ? WHERE event_id = ?",
                (canonical, digest, event_id),
            )
        self._conn.execute(
            f"CREATE INDEX IF NOT EXISTS idx_{_OUTBOX_TABLE}_order "
            f"ON {_OUTBOX_TABLE} (created_at, event_id)"
        )

    def save(self, key: str, value: Any) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO kv (key, value) VALUES (?, ?)", (key, json.dumps(value))
        )
        self._conn.commit()

    def load(self, key: str) -> Any:
        row = self._conn.execute("SELECT value FROM kv WHERE key = ?", (key,)).fetchone()
        return json.loads(row[0]) if row is not None else None

    @staticmethod
    def _encode_value(value: Any) -> str:
        try:
            return json.dumps(value)
        except (TypeError, ValueError) as exc:
            raise OutboxValidationError(f"persistence value must be JSON-serializable: {exc}") from exc

    def _begin(self) -> None:
        self._conn.execute("BEGIN IMMEDIATE")

    def commit_state_and_enqueue(
        self,
        state: Any,
        deliveries: Iterable[PendingDelivery],
        *,
        state_key: str = "percept",
    ) -> None:
        """Atomically persist post-fire state and insert its pending delivery envelopes.

        Re-enqueuing an identical event is idempotent and never resets attempt state. Reusing one
        event ID for different content or routing aborts and rolls back the whole state transaction.
        """

        state_json = self._encode_value(state)
        pending = list(deliveries)
        if any(not isinstance(item, PendingDelivery) for item in pending):
            raise OutboxValidationError("deliveries must contain PendingDelivery records")
        try:
            self._begin()
            self._conn.execute(
                "INSERT OR REPLACE INTO kv (key, value) VALUES (?, ?)",
                (state_key, state_json),
            )
            for item in pending:
                existing = self._conn.execute(
                    f"SELECT payload_sha256, schema_version, sink_config_ref "
                    f"FROM {_OUTBOX_TABLE} WHERE event_id = ?",
                    (item.event_id,),
                ).fetchone()
                identity = (item.payload_sha256, item.schema_version, item.sink_config_ref)
                if existing is not None:
                    if tuple(existing) != identity:
                        raise OutboxEventCollision(
                            f"event_id {item.event_id!r} already names a different pending delivery"
                        )
                    continue
                self._conn.execute(
                    f"""
                    INSERT INTO {_OUTBOX_TABLE} (
                        event_id, payload_json, schema_version, payload_sha256, sink_config_ref,
                        attempts, created_at, updated_at, last_attempt_at, next_attempt_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        item.event_id,
                        item.payload_json,
                        item.schema_version,
                        item.payload_sha256,
                        item.sink_config_ref,
                        item.attempts,
                        item.created_at,
                        item.updated_at,
                        item.last_attempt_at,
                        item.next_attempt_at,
                    ),
                )
            self._conn.commit()
        except BaseException:
            self._conn.rollback()
            raise

    def load_pending_deliveries(self, *, limit: int | None = None) -> list[PendingDelivery]:
        """Load pending rows oldest-first with event ID as the deterministic tie-breaker."""

        if limit is not None and (type(limit) is not int or limit < 0):
            raise OutboxValidationError("limit must be a non-negative integer or None")
        sql = (
            f"SELECT event_id, payload_json, sink_config_ref, created_at, schema_version, attempts, "
            f"updated_at, last_attempt_at, next_attempt_at, payload_sha256 "
            f"FROM {_OUTBOX_TABLE} ORDER BY created_at ASC, event_id ASC"
        )
        params: tuple[Any, ...] = ()
        if limit is not None:
            sql += " LIMIT ?"
            params = (limit,)
        rows = self._conn.execute(sql, params).fetchall()
        pending = []
        for row in rows:
            (
                event_id,
                payload_json,
                sink_ref,
                created_at,
                schema_version,
                attempts,
                updated_at,
                last_attempt_at,
                next_attempt_at,
                stored_hash,
            ) = row
            try:
                payload = json.loads(payload_json)
                item = PendingDelivery(
                    event_id=event_id,
                    payload=payload,
                    sink_config_ref=sink_ref,
                    created_at=created_at,
                    schema_version=schema_version,
                    attempts=attempts,
                    updated_at=updated_at,
                    last_attempt_at=last_attempt_at,
                    next_attempt_at=next_attempt_at,
                )
            except (json.JSONDecodeError, OutboxValidationError) as exc:
                raise OutboxCorruptError(f"invalid pending delivery {event_id!r}: {exc}") from exc
            if item.payload_sha256 != stored_hash:
                raise OutboxCorruptError(
                    f"payload hash mismatch for pending delivery {event_id!r}"
                )
            pending.append(item)
        return pending

    def note_delivery_attempt(
        self,
        event_id: str,
        *,
        attempted_at: float,
        next_attempt_at: float | None = None,
    ) -> bool:
        """Atomically increment attempt metadata without storing an error, URL, or credential."""

        event_id = _validate_event_id(event_id)
        attempted = _finite_timestamp(attempted_at, "attempted_at")
        next_attempt = _finite_timestamp(next_attempt_at, "next_attempt_at", optional=True)
        assert attempted is not None
        if next_attempt is not None and next_attempt < attempted:
            raise OutboxValidationError("next_attempt_at cannot precede attempted_at")
        try:
            self._begin()
            existing = self._conn.execute(
                f"SELECT created_at FROM {_OUTBOX_TABLE} WHERE event_id = ?", (event_id,)
            ).fetchone()
            if existing is None:
                self._conn.rollback()
                return False
            if attempted < float(existing[0]):
                raise OutboxValidationError("attempted_at cannot precede created_at")
            result = self._conn.execute(
                f"""
                UPDATE {_OUTBOX_TABLE}
                SET attempts = attempts + 1, updated_at = ?, last_attempt_at = ?, next_attempt_at = ?
                WHERE event_id = ?
                """,
                (attempted, attempted, next_attempt, event_id),
            )
            self._conn.commit()
            return result.rowcount == 1
        except BaseException:
            self._conn.rollback()
            raise

    @staticmethod
    def _ledger_timestamp(delivered_ledger_state: Any, event_id: str) -> float | None:
        if isinstance(delivered_ledger_state, dict):
            value = delivered_ledger_state.get(event_id)
            return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None
        if isinstance(delivered_ledger_state, list):
            for row in delivered_ledger_state:
                if (
                    isinstance(row, (list, tuple))
                    and len(row) >= 2
                    and row[0] == event_id
                    and isinstance(row[1], (int, float))
                    and not isinstance(row[1], bool)
                ):
                    return float(row[1])
        return None

    def ack_delivery(
        self,
        event_id: str,
        delivered_ledger_state: Any,
        *,
        delivered_at: float,
        delivered_ledger_key: str = "percept_delivery_ledger",
    ) -> bool:
        """Atomically remove one pending row and persist the delivered-id ledger snapshot."""

        event_id = _validate_event_id(event_id)
        delivered = _finite_timestamp(delivered_at, "delivered_at")
        assert delivered is not None
        ledger_timestamp = self._ledger_timestamp(delivered_ledger_state, event_id)
        if ledger_timestamp is None:
            raise OutboxValidationError(
                "delivered ledger state must contain event_id before its pending row can be ACKed"
            )
        if not math.isfinite(ledger_timestamp) or ledger_timestamp != delivered:
            raise OutboxValidationError(
                "delivered_at must match the event timestamp persisted in the delivered ledger"
            )
        ledger_json = self._encode_value(delivered_ledger_state)
        try:
            self._begin()
            deleted = self._conn.execute(
                f"DELETE FROM {_OUTBOX_TABLE} WHERE event_id = ?", (event_id,)
            )
            self._conn.execute(
                "INSERT OR REPLACE INTO kv (key, value) VALUES (?, ?)",
                (delivered_ledger_key, ledger_json),
            )
            self._conn.commit()
            return deleted.rowcount == 1
        except BaseException:
            self._conn.rollback()
            raise

    def keys(self) -> list[str]:
        return [r[0] for r in self._conn.execute("SELECT key FROM kv").fetchall()]

    def close(self) -> None:
        self._conn.close()


__all__ = [
    "InMemoryPersistence",
    "OutboxCorruptError",
    "OutboxError",
    "OutboxEventCollision",
    "OutboxValidationError",
    "PendingDelivery",
    "SqlitePersistence",
    "TRANSACTIONAL_OUTBOX_CAPABILITY",
    "TransactionalOutboxPersistence",
]
