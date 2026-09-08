"""Durable, append-only audit archive primitives (not wired into the live runtime).

This module is the storage half of a future bounded 24/7 write-through design.  It archives three
immutable record families in SQLite without changing :class:`EpisodeLog`, :class:`EventLog`,
:class:`ActionLog`, ``Percept.snapshot()``, or runtime eviction:

* emitted episode/Incident revisions, keyed by ``<episode_id>#r<n>``;
* complete EventLog history/count checkpoints, keyed by a caller-owned checkpoint ID; and
* immutable ActionLog records, keyed by a caller-owned record ID.

Rows use strict canonical JSON plus SHA-256, an auto-increment cursor, and a unique
``(stream, stable_key)`` identity.  Re-appending identical bytes is an idempotent no-op.  Reusing a
key for different bytes is a collision and rolls back the whole batch.  Reads are cursor-ordered and
verify the canonical encoding, hash, and indexed identity before returning data.

``HotTailPolicy`` is persisted policy *metadata only*.  This module deliberately does not evict or
restore live memory.  Runtime integration still requires all of the following, in this order:

1. give each ActionLog append and EventLog checkpoint a stable ID at its mutation source;
2. define an owner-thread write-through seam that commits the archive row before the corresponding
   in-memory record becomes evictable (and atomically with any cognition snapshot it depends on);
3. archive every emitted episode revision together with its frozen Incident payload before trimming
   either the episode or its revision tail;
4. add restore/reconciliation rules for EventLog counters and episode stream-time fields, including
   explicit discontinuity/rebase handling rather than replaying old monotonic timestamps blindly;
5. surface archive failures as fatal/health events, qualify crash boundaries, and only then enable
   record/byte hot-tail eviction using the declared policy.

Until those seams exist, this is a standalone durable archive capability, not episode continuity and
not a hosted qualification claim.
"""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import time
from collections import Counter
from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any, Callable, Iterable, Literal, Protocol, runtime_checkable


AUDIT_ARCHIVE_CAPABILITY = "percept.archive.sqlite_audit.v1"

EPISODE_REVISION_STREAM = "episode_incident_revision"
EVENT_CHECKPOINT_STREAM = "event_log_checkpoint"
ACTION_RECORD_STREAM = "action_log_record"
ARCHIVE_STREAMS = (
    EPISODE_REVISION_STREAM,
    EVENT_CHECKPOINT_STREAM,
    ACTION_RECORD_STREAM,
)
ArchiveStream = Literal[
    "episode_incident_revision",
    "event_log_checkpoint",
    "action_log_record",
]

_ARCHIVE_TABLE = "percept_audit_archive"
_POLICY_TABLE = "percept_archive_hot_tail_policy"


class ArchiveError(RuntimeError):
    """Base error for the standalone durable audit archive."""


class ArchiveValidationError(ArchiveError, ValueError):
    """A record or policy cannot be represented by the archive contract."""


class ArchiveCollisionError(ArchiveError):
    """A stable key was reused for different immutable content."""


class ArchiveCorruptError(ArchiveError):
    """A persisted archive row failed canonical JSON, hash, or identity verification."""


class ArchivePolicyConflict(ArchiveError):
    """Hot-tail policy revision ordering or immutable revision identity was violated."""


def _stable_key(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 512 or "\x00" in value:
        raise ArchiveValidationError(
            f"{field_name} must be a non-empty, NUL-free string of at most 512 characters"
        )
    return value


def _timestamp(value: object, field_name: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
        or value < 0
    ):
        raise ArchiveValidationError(f"{field_name} must be a finite non-negative number")
    return float(value)


def _positive_int(value: object, field_name: str) -> int:
    if type(value) is not int or value < 1:
        raise ArchiveValidationError(f"{field_name} must be a positive integer")
    return value


def _non_negative_int(value: object, field_name: str) -> int:
    if type(value) is not int or value < 0:
        raise ArchiveValidationError(f"{field_name} must be a non-negative integer")
    return value


def _canonical_json(value: object, field_name: str) -> tuple[Any, str, str]:
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        detached = json.loads(encoded)
    except (TypeError, ValueError) as exc:
        raise ArchiveValidationError(f"{field_name} must be strict JSON: {exc}") from exc
    digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    return detached, encoded, digest


def _json_object(value: object, field_name: str) -> dict[str, Any]:
    detached, _, _ = _canonical_json(value, field_name)
    if not isinstance(detached, dict):
        raise ArchiveValidationError(f"{field_name} must be a JSON object")
    return detached


@dataclass(frozen=True)
class EpisodeIncidentRevisionRecord:
    """One immutable emitted episode revision and its frozen Incident wire payload."""

    revision_id: str
    episode_id: str
    incident_id: str
    occurred_at: float
    episode: dict[str, Any]
    incident: dict[str, Any]
    schema_version: int = 1

    def __post_init__(self) -> None:
        episode_id = _stable_key(self.episode_id, "episode_id")
        revision_id = _stable_key(self.revision_id, "revision_id")
        incident_id = _stable_key(self.incident_id, "incident_id")
        prefix = f"{episode_id}#r"
        suffix = revision_id[len(prefix):] if revision_id.startswith(prefix) else ""
        if not suffix.isdigit() or int(suffix) < 1:
            raise ArchiveValidationError(
                "revision_id must be the stable <episode_id>#r<n> handle with n >= 1"
            )
        occurred_at = _timestamp(self.occurred_at, "occurred_at")
        schema_version = _positive_int(self.schema_version, "schema_version")
        episode = _json_object(self.episode, "episode")
        incident = _json_object(self.incident, "incident")
        if episode.get("id") != episode_id:
            raise ArchiveValidationError("episode payload id is required and must match episode_id")
        if incident.get("id") != incident_id:
            raise ArchiveValidationError("incident payload id is required and must match incident_id")
        if incident.get("revision_id") != revision_id:
            raise ArchiveValidationError(
                "incident payload revision_id is required and must match the archive revision_id"
            )
        object.__setattr__(self, "episode_id", episode_id)
        object.__setattr__(self, "revision_id", revision_id)
        object.__setattr__(self, "incident_id", incident_id)
        object.__setattr__(self, "occurred_at", occurred_at)
        object.__setattr__(self, "schema_version", schema_version)
        object.__setattr__(self, "episode", episode)
        object.__setattr__(self, "incident", incident)

    @property
    def stable_key(self) -> str:
        return self.revision_id

    @property
    def stream(self) -> ArchiveStream:
        return EPISODE_REVISION_STREAM

    def payload(self) -> dict[str, Any]:
        return {
            "record_type": self.stream,
            "schema_version": self.schema_version,
            "revision_id": self.revision_id,
            "episode_id": self.episode_id,
            "incident_id": self.incident_id,
            "occurred_at": self.occurred_at,
            "episode": self.episode,
            "incident": self.incident,
        }


@dataclass(frozen=True)
class EventLogCheckpointRecord:
    """A complete immutable EventLog history plus its verified distinct-event counts."""

    checkpoint_id: str
    occurred_at: float
    events: tuple[dict[str, Any], ...]
    counts_by_type: dict[str, int]
    total_count: int
    counter: int
    schema_version: int = 1

    def __post_init__(self) -> None:
        checkpoint_id = _stable_key(self.checkpoint_id, "checkpoint_id")
        occurred_at = _timestamp(self.occurred_at, "occurred_at")
        schema_version = _positive_int(self.schema_version, "schema_version")
        total_count = _non_negative_int(self.total_count, "total_count")
        counter = _non_negative_int(self.counter, "counter")
        raw_events, _, _ = _canonical_json(list(self.events), "events")
        if not isinstance(raw_events, list) or any(
            not isinstance(item, dict) for item in raw_events
        ):
            raise ArchiveValidationError("events must contain JSON objects")
        event_ids: set[str] = set()
        observed_counts: Counter[str] = Counter()
        for event in raw_events:
            event_id = _stable_key(event.get("id"), "event.id")
            event_type = _stable_key(event.get("type"), "event.type")
            if event_id in event_ids:
                raise ArchiveValidationError(f"duplicate event id {event_id!r} in checkpoint")
            event_ids.add(event_id)
            observed_counts[event_type] += 1
        counts = _json_object(self.counts_by_type, "counts_by_type")
        if any(
            not isinstance(key, str)
            or type(value) is not int
            or value < 0
            for key, value in counts.items()
        ):
            raise ArchiveValidationError(
                "counts_by_type must map event type strings to non-negative integers"
            )
        if total_count != len(raw_events):
            raise ArchiveValidationError("total_count does not match the event history length")
        if dict(sorted(counts.items())) != dict(sorted(observed_counts.items())):
            raise ArchiveValidationError("counts_by_type does not match the event history")
        if counter < total_count:
            raise ArchiveValidationError("counter cannot be lower than total_count")
        object.__setattr__(self, "checkpoint_id", checkpoint_id)
        object.__setattr__(self, "occurred_at", occurred_at)
        object.__setattr__(self, "schema_version", schema_version)
        object.__setattr__(self, "events", tuple(raw_events))
        object.__setattr__(self, "counts_by_type", counts)
        object.__setattr__(self, "total_count", total_count)
        object.__setattr__(self, "counter", counter)

    @classmethod
    def capture(
        cls,
        checkpoint_id: str,
        event_log: Any,
        *,
        occurred_at: float,
    ) -> "EventLogCheckpointRecord":
        """Freeze the current EventLog without modifying or promising to restore it."""

        events = []
        for event in event_log.events():
            if is_dataclass(event):
                events.append(asdict(event))
            elif isinstance(event, dict):
                events.append(event)
            else:
                raise ArchiveValidationError(
                    f"EventLog event must be a dataclass or dict, got {type(event).__name__}"
                )
        counts = dict(Counter(str(event["type"]) for event in events))
        counter = getattr(event_log, "_counter", len(events))
        return cls(
            checkpoint_id=checkpoint_id,
            occurred_at=occurred_at,
            events=tuple(events),
            counts_by_type=counts,
            total_count=len(events),
            counter=counter,
        )

    @property
    def stable_key(self) -> str:
        return self.checkpoint_id

    @property
    def stream(self) -> ArchiveStream:
        return EVENT_CHECKPOINT_STREAM

    def payload(self) -> dict[str, Any]:
        return {
            "record_type": self.stream,
            "schema_version": self.schema_version,
            "checkpoint_id": self.checkpoint_id,
            "occurred_at": self.occurred_at,
            "events": list(self.events),
            "counts_by_type": self.counts_by_type,
            "total_count": self.total_count,
            "counter": self.counter,
        }


@dataclass(frozen=True)
class ImmutableActionRecord:
    """One immutable copy of an ActionLog record under an explicit stable archive key."""

    record_id: str
    occurred_at: float
    action: dict[str, Any]
    schema_version: int = 1

    def __post_init__(self) -> None:
        record_id = _stable_key(self.record_id, "record_id")
        occurred_at = _timestamp(self.occurred_at, "occurred_at")
        schema_version = _positive_int(self.schema_version, "schema_version")
        action = _json_object(self.action, "action")
        _stable_key(action.get("kind"), "action.kind")
        _stable_key(action.get("goal_id"), "action.goal_id")
        action_t = _timestamp(action.get("t"), "action.t")
        if action_t != occurred_at:
            raise ArchiveValidationError("occurred_at must match action.t")
        if not isinstance(action.get("detail", {}), dict):
            raise ArchiveValidationError("action.detail must be a JSON object")
        if type(action.get("allowed", True)) is not bool:
            raise ArchiveValidationError("action.allowed must be a boolean")
        object.__setattr__(self, "record_id", record_id)
        object.__setattr__(self, "occurred_at", occurred_at)
        object.__setattr__(self, "schema_version", schema_version)
        object.__setattr__(self, "action", action)

    @classmethod
    def capture(cls, record_id: str, action: Any) -> "ImmutableActionRecord":
        """Detach an existing ActionRecord; later caller mutation cannot change archived bytes."""

        if is_dataclass(action):
            payload = asdict(action)
        elif isinstance(action, dict):
            payload = action
        else:
            raise ArchiveValidationError(
                f"action must be an ActionRecord/dataclass or dict, got {type(action).__name__}"
            )
        return cls(record_id=record_id, occurred_at=payload.get("t"), action=payload)

    @property
    def stable_key(self) -> str:
        return self.record_id

    @property
    def stream(self) -> ArchiveStream:
        return ACTION_RECORD_STREAM

    def payload(self) -> dict[str, Any]:
        return {
            "record_type": self.stream,
            "schema_version": self.schema_version,
            "record_id": self.record_id,
            "occurred_at": self.occurred_at,
            "action": self.action,
        }


ArchiveRecord = (
    EpisodeIncidentRevisionRecord
    | EventLogCheckpointRecord
    | ImmutableActionRecord
)


@dataclass(frozen=True, order=True)
class ArchiveCursor:
    """Opaque monotonic read position within one SQLite archive file."""

    value: int

    def __post_init__(self) -> None:
        _non_negative_int(self.value, "cursor")


@dataclass(frozen=True)
class ArchiveEntry:
    cursor: ArchiveCursor
    stream: ArchiveStream
    stable_key: str
    occurred_at: float
    archived_at: float
    schema_version: int
    payload: dict[str, Any]
    payload_sha256: str


@dataclass(frozen=True)
class ArchivePage:
    entries: tuple[ArchiveEntry, ...]
    next_cursor: ArchiveCursor | None


@dataclass(frozen=True)
class ArchiveAppendResult:
    inserted: int
    existing: int
    cursors: tuple[ArchiveCursor, ...]


@dataclass(frozen=True)
class HotTailPolicy:
    """Versioned record/byte bounds for a future live tail; persisted but not enforced here."""

    stream: ArchiveStream
    revision: int
    max_records: int
    max_bytes: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.stream not in ARCHIVE_STREAMS:
            raise ArchiveValidationError(f"unsupported archive stream {self.stream!r}")
        revision = _positive_int(self.revision, "revision")
        max_records = _positive_int(self.max_records, "max_records")
        max_bytes = _positive_int(self.max_bytes, "max_bytes")
        metadata = _json_object(self.metadata, "metadata")
        object.__setattr__(self, "revision", revision)
        object.__setattr__(self, "max_records", max_records)
        object.__setattr__(self, "max_bytes", max_bytes)
        object.__setattr__(self, "metadata", metadata)

    def payload(self) -> dict[str, Any]:
        return {
            "stream": self.stream,
            "revision": self.revision,
            "max_records": self.max_records,
            "max_bytes": self.max_bytes,
            "enforcement": "metadata_only",
            "eviction_wired": False,
            "metadata": self.metadata,
        }


@runtime_checkable
class DurableAuditArchive(Protocol):
    """Optional standalone archive capability; no live runtime integration is implied."""

    archive_capability: str
    enforces_hot_tail: bool

    def append_batch(self, records: Iterable[ArchiveRecord]) -> ArchiveAppendResult: ...

    def read(
        self,
        *,
        stream: ArchiveStream | None = None,
        after: ArchiveCursor | None = None,
        limit: int = 100,
    ) -> ArchivePage: ...

    def declare_hot_tail_policy(self, policy: HotTailPolicy) -> bool: ...

    def read_hot_tail_policy(self, stream: ArchiveStream) -> HotTailPolicy | None: ...


class SqliteAuditArchive:
    """Stdlib SQLite implementation of :class:`DurableAuditArchive`."""

    archive_capability = AUDIT_ARCHIVE_CAPABILITY
    enforces_hot_tail = False

    def __init__(self, path: str = ":memory:", *, clock: Callable[[], float] = time.time) -> None:
        self._conn = sqlite3.connect(path)
        self._clock = clock
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        self._conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {_ARCHIVE_TABLE} (
                cursor INTEGER PRIMARY KEY AUTOINCREMENT,
                stream TEXT NOT NULL,
                stable_key TEXT NOT NULL,
                occurred_at REAL NOT NULL,
                archived_at REAL NOT NULL,
                schema_version INTEGER NOT NULL,
                payload_json TEXT NOT NULL,
                payload_sha256 TEXT NOT NULL,
                UNIQUE(stream, stable_key)
            )
            """
        )
        self._conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {_POLICY_TABLE} (
                stream TEXT NOT NULL,
                revision INTEGER NOT NULL,
                declared_at REAL NOT NULL,
                policy_json TEXT NOT NULL,
                policy_sha256 TEXT NOT NULL,
                PRIMARY KEY(stream, revision)
            )
            """
        )
        self._verify_columns(
            _ARCHIVE_TABLE,
            {
                "cursor", "stream", "stable_key", "occurred_at", "archived_at",
                "schema_version", "payload_json", "payload_sha256",
            },
        )
        self._verify_columns(
            _POLICY_TABLE,
            {"stream", "revision", "declared_at", "policy_json", "policy_sha256"},
        )
        self._conn.execute(
            f"CREATE INDEX IF NOT EXISTS idx_{_ARCHIVE_TABLE}_stream_cursor "
            f"ON {_ARCHIVE_TABLE} (stream, cursor)"
        )
        self._conn.commit()

    def _verify_columns(self, table: str, required: set[str]) -> None:
        columns = {row[1] for row in self._conn.execute(f"PRAGMA table_info({table})")}
        if not required.issubset(columns):
            missing = ", ".join(sorted(required - columns))
            raise ArchiveCorruptError(f"{table} is missing required columns: {missing}")

    @staticmethod
    def _prepared(record: ArchiveRecord) -> tuple[ArchiveStream, str, float, int, str, str]:
        if not isinstance(
            record,
            (
                EpisodeIncidentRevisionRecord,
                EventLogCheckpointRecord,
                ImmutableActionRecord,
            ),
        ):
            raise ArchiveValidationError(f"unsupported archive record {type(record).__name__}")
        payload, payload_json, digest = _canonical_json(record.payload(), "archive payload")
        assert isinstance(payload, dict)
        return (
            record.stream,
            record.stable_key,
            record.occurred_at,
            record.schema_version,
            payload_json,
            digest,
        )

    def append_batch(self, records: Iterable[ArchiveRecord]) -> ArchiveAppendResult:
        """Append a batch atomically; identical keys are no-ops and collisions roll it all back."""

        prepared = [self._prepared(record) for record in records]
        inserted = 0
        existing_count = 0
        cursors: list[ArchiveCursor] = []
        try:
            self._conn.execute("BEGIN IMMEDIATE")
            for stream, key, occurred_at, schema_version, payload_json, digest in prepared:
                existing = self._conn.execute(
                    f"SELECT cursor, schema_version, payload_json, payload_sha256 "
                    f"FROM {_ARCHIVE_TABLE} WHERE stream = ? AND stable_key = ?",
                    (stream, key),
                ).fetchone()
                if existing is not None:
                    cursor, old_schema, old_json, old_digest = existing
                    if (
                        old_schema != schema_version
                        or old_json != payload_json
                        or old_digest != digest
                    ):
                        raise ArchiveCollisionError(
                            f"{stream} stable key {key!r} already names different immutable bytes"
                        )
                    existing_count += 1
                    cursors.append(ArchiveCursor(int(cursor)))
                    continue
                archived_at = _timestamp(self._clock(), "archive clock")
                result = self._conn.execute(
                    f"""
                    INSERT INTO {_ARCHIVE_TABLE} (
                        stream, stable_key, occurred_at, archived_at, schema_version,
                        payload_json, payload_sha256
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        stream,
                        key,
                        occurred_at,
                        archived_at,
                        schema_version,
                        payload_json,
                        digest,
                    ),
                )
                inserted += 1
                cursors.append(ArchiveCursor(int(result.lastrowid)))
            self._conn.commit()
        except BaseException:
            self._conn.rollback()
            raise
        return ArchiveAppendResult(inserted, existing_count, tuple(cursors))

    def append_episode_revision(
        self, record: EpisodeIncidentRevisionRecord
    ) -> ArchiveAppendResult:
        return self.append_batch([record])

    def append_event_checkpoint(
        self, record: EventLogCheckpointRecord
    ) -> ArchiveAppendResult:
        return self.append_batch([record])

    def append_action_record(self, record: ImmutableActionRecord) -> ArchiveAppendResult:
        return self.append_batch([record])

    @staticmethod
    def _after_value(after: ArchiveCursor | None) -> int:
        if after is None:
            return 0
        if type(after) is not ArchiveCursor:
            raise ArchiveValidationError("after must be an ArchiveCursor or None")
        return after.value

    @staticmethod
    def _stream(stream: ArchiveStream | None) -> ArchiveStream | None:
        if stream is not None and stream not in ARCHIVE_STREAMS:
            raise ArchiveValidationError(f"unsupported archive stream {stream!r}")
        return stream

    def _entry(self, row: tuple[Any, ...]) -> ArchiveEntry:
        (
            cursor,
            stream,
            stable_key,
            occurred_at,
            archived_at,
            schema_version,
            payload_json,
            stored_digest,
        ) = row
        try:
            parsed = json.loads(payload_json)
            payload, canonical, digest = _canonical_json(parsed, "stored archive payload")
        except (json.JSONDecodeError, ArchiveValidationError) as exc:
            raise ArchiveCorruptError(
                f"invalid archive payload at cursor {cursor}: {exc}"
            ) from exc
        if canonical != payload_json or digest != stored_digest:
            raise ArchiveCorruptError(f"archive payload hash/canonical mismatch at cursor {cursor}")
        if not isinstance(payload, dict):
            raise ArchiveCorruptError(f"archive payload at cursor {cursor} is not an object")
        key_field = {
            EPISODE_REVISION_STREAM: "revision_id",
            EVENT_CHECKPOINT_STREAM: "checkpoint_id",
            ACTION_RECORD_STREAM: "record_id",
        }.get(stream)
        if (
            key_field is None
            or payload.get("record_type") != stream
            or payload.get(key_field) != stable_key
            or payload.get("schema_version") != schema_version
            or payload.get("occurred_at") != occurred_at
        ):
            raise ArchiveCorruptError(f"archive indexed identity mismatch at cursor {cursor}")
        try:
            if stream == EPISODE_REVISION_STREAM:
                EpisodeIncidentRevisionRecord(
                    revision_id=payload["revision_id"],
                    episode_id=payload["episode_id"],
                    incident_id=payload["incident_id"],
                    occurred_at=payload["occurred_at"],
                    episode=payload["episode"],
                    incident=payload["incident"],
                    schema_version=payload["schema_version"],
                )
            elif stream == EVENT_CHECKPOINT_STREAM:
                EventLogCheckpointRecord(
                    checkpoint_id=payload["checkpoint_id"],
                    occurred_at=payload["occurred_at"],
                    events=tuple(payload["events"]),
                    counts_by_type=payload["counts_by_type"],
                    total_count=payload["total_count"],
                    counter=payload["counter"],
                    schema_version=payload["schema_version"],
                )
            else:
                ImmutableActionRecord(
                    record_id=payload["record_id"],
                    occurred_at=payload["occurred_at"],
                    action=payload["action"],
                    schema_version=payload["schema_version"],
                )
        except (ArchiveValidationError, KeyError, TypeError) as exc:
            raise ArchiveCorruptError(
                f"archive record schema mismatch at cursor {cursor}: {exc}"
            ) from exc
        return ArchiveEntry(
            cursor=ArchiveCursor(int(cursor)),
            stream=stream,
            stable_key=stable_key,
            occurred_at=float(occurred_at),
            archived_at=float(archived_at),
            schema_version=int(schema_version),
            payload=payload,
            payload_sha256=stored_digest,
        )

    def read(
        self,
        *,
        stream: ArchiveStream | None = None,
        after: ArchiveCursor | None = None,
        limit: int = 100,
    ) -> ArchivePage:
        """Read one deterministic cursor page, verifying every row before returning it."""

        stream = self._stream(stream)
        after_value = self._after_value(after)
        if type(limit) is not int or limit < 1 or limit > 10_000:
            raise ArchiveValidationError("limit must be an integer from 1 through 10000")
        where = "cursor > ?"
        params: list[Any] = [after_value]
        if stream is not None:
            where += " AND stream = ?"
            params.append(stream)
        params.append(limit)
        rows = self._conn.execute(
            f"""
            SELECT cursor, stream, stable_key, occurred_at, archived_at, schema_version,
                   payload_json, payload_sha256
            FROM {_ARCHIVE_TABLE}
            WHERE {where}
            ORDER BY cursor ASC
            LIMIT ?
            """,
            tuple(params),
        ).fetchall()
        entries = tuple(self._entry(row) for row in rows)
        return ArchivePage(
            entries=entries,
            next_cursor=entries[-1].cursor if entries else None,
        )

    def read_episode_revisions(
        self, *, after: ArchiveCursor | None = None, limit: int = 100
    ) -> ArchivePage:
        return self.read(stream=EPISODE_REVISION_STREAM, after=after, limit=limit)

    def read_event_checkpoints(
        self, *, after: ArchiveCursor | None = None, limit: int = 100
    ) -> ArchivePage:
        return self.read(stream=EVENT_CHECKPOINT_STREAM, after=after, limit=limit)

    def read_action_records(
        self, *, after: ArchiveCursor | None = None, limit: int = 100
    ) -> ArchivePage:
        return self.read(stream=ACTION_RECORD_STREAM, after=after, limit=limit)

    def latest_cursor(self, *, stream: ArchiveStream | None = None) -> ArchiveCursor | None:
        stream = self._stream(stream)
        if stream is None:
            row = self._conn.execute(f"SELECT MAX(cursor) FROM {_ARCHIVE_TABLE}").fetchone()
        else:
            row = self._conn.execute(
                f"SELECT MAX(cursor) FROM {_ARCHIVE_TABLE} WHERE stream = ?", (stream,)
            ).fetchone()
        return ArchiveCursor(int(row[0])) if row and row[0] is not None else None

    def declare_hot_tail_policy(self, policy: HotTailPolicy) -> bool:
        """Persist a policy revision; identical is idempotent and revisions only move forward."""

        if type(policy) is not HotTailPolicy:
            raise ArchiveValidationError("policy must be an exact HotTailPolicy")
        _, policy_json, digest = _canonical_json(policy.payload(), "hot-tail policy")
        try:
            self._conn.execute("BEGIN IMMEDIATE")
            existing = self._conn.execute(
                f"SELECT policy_json, policy_sha256 FROM {_POLICY_TABLE} "
                "WHERE stream = ? AND revision = ?",
                (policy.stream, policy.revision),
            ).fetchone()
            if existing is not None:
                if existing != (policy_json, digest):
                    raise ArchivePolicyConflict(
                        f"policy {policy.stream}@{policy.revision} has different immutable bytes"
                    )
                self._conn.commit()
                return False
            latest = self._conn.execute(
                f"SELECT MAX(revision) FROM {_POLICY_TABLE} WHERE stream = ?",
                (policy.stream,),
            ).fetchone()[0]
            if latest is not None and policy.revision <= int(latest):
                raise ArchivePolicyConflict(
                    f"policy revision must advance beyond {latest} for {policy.stream}"
                )
            declared_at = _timestamp(self._clock(), "archive clock")
            self._conn.execute(
                f"INSERT INTO {_POLICY_TABLE} "
                "(stream, revision, declared_at, policy_json, policy_sha256) "
                "VALUES (?, ?, ?, ?, ?)",
                (policy.stream, policy.revision, declared_at, policy_json, digest),
            )
            self._conn.commit()
            return True
        except BaseException:
            self._conn.rollback()
            raise

    def read_hot_tail_policy(self, stream: ArchiveStream) -> HotTailPolicy | None:
        stream = self._stream(stream)
        assert stream is not None
        row = self._conn.execute(
            f"SELECT revision, policy_json, policy_sha256 FROM {_POLICY_TABLE} "
            "WHERE stream = ? ORDER BY revision DESC LIMIT 1",
            (stream,),
        ).fetchone()
        if row is None:
            return None
        revision, policy_json, stored_digest = row
        try:
            payload = json.loads(policy_json)
            detached, canonical, digest = _canonical_json(payload, "stored hot-tail policy")
        except (json.JSONDecodeError, ArchiveValidationError) as exc:
            raise ArchiveCorruptError(f"invalid hot-tail policy for {stream}: {exc}") from exc
        if canonical != policy_json or digest != stored_digest:
            raise ArchiveCorruptError(f"hot-tail policy hash/canonical mismatch for {stream}")
        if (
            not isinstance(detached, dict)
            or detached.get("stream") != stream
            or detached.get("revision") != revision
            or detached.get("enforcement") != "metadata_only"
            or detached.get("eviction_wired") is not False
        ):
            raise ArchiveCorruptError(f"hot-tail policy identity mismatch for {stream}")
        try:
            return HotTailPolicy(
                stream=stream,
                revision=revision,
                max_records=detached.get("max_records"),
                max_bytes=detached.get("max_bytes"),
                metadata=detached.get("metadata", {}),
            )
        except ArchiveValidationError as exc:
            raise ArchiveCorruptError(
                f"hot-tail policy schema mismatch for {stream}: {exc}"
            ) from exc

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "SqliteAuditArchive":
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()


__all__ = [
    "ACTION_RECORD_STREAM",
    "ARCHIVE_STREAMS",
    "AUDIT_ARCHIVE_CAPABILITY",
    "ArchiveAppendResult",
    "ArchiveCollisionError",
    "ArchiveCorruptError",
    "ArchiveCursor",
    "ArchiveEntry",
    "ArchiveError",
    "ArchivePage",
    "ArchivePolicyConflict",
    "ArchiveValidationError",
    "DurableAuditArchive",
    "EPISODE_REVISION_STREAM",
    "EVENT_CHECKPOINT_STREAM",
    "EpisodeIncidentRevisionRecord",
    "EventLogCheckpointRecord",
    "HotTailPolicy",
    "ImmutableActionRecord",
    "SqliteAuditArchive",
]
