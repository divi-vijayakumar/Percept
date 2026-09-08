"""Transactional outbox storage invariants beneath the runtime webhook integration."""
from __future__ import annotations

import hashlib
import json
import sqlite3

import pytest

from percept.persistence import (
    InMemoryPersistence,
    OutboxCorruptError,
    OutboxEventCollision,
    OutboxValidationError,
    PendingDelivery,
    SqlitePersistence,
    TRANSACTIONAL_OUTBOX_CAPABILITY,
    TransactionalOutboxPersistence,
)


def _pending(
    event_id: str,
    *,
    created_at: float = 10.0,
    message: str = "door open",
    attempts: int = 0,
) -> PendingDelivery:
    return PendingDelivery(
        event_id=event_id,
        payload={"message": message, "id": event_id, "concern_id": "door"},
        sink_config_ref="enterprise-primary",
        created_at=created_at,
        attempts=attempts,
    )


def test_sqlite_exposes_optional_capability_without_changing_base_persistence() -> None:
    sqlite = SqlitePersistence(":memory:")
    try:
        assert isinstance(sqlite, TransactionalOutboxPersistence)
        assert sqlite.outbox_capability == TRANSACTIONAL_OUTBOX_CAPABILITY
        assert not isinstance(InMemoryPersistence(), TransactionalOutboxPersistence)

        sqlite.save("legacy-key", {"still": "works"})
        assert sqlite.load("legacy-key") == {"still": "works"}
    finally:
        sqlite.close()


def test_commit_reopen_and_load_are_canonical_and_deterministically_ordered(tmp_path) -> None:
    path = tmp_path / "outbox.sqlite3"
    store = SqlitePersistence(str(path))
    store.commit_state_and_enqueue(
        {"gate": "consumed"},
        [
            _pending("evt-b", created_at=10.0),
            _pending("evt-old", created_at=5.0, attempts=2),
            _pending("evt-a", created_at=10.0),
        ],
    )
    assert store.note_delivery_attempt("evt-a", attempted_at=11.0, next_attempt_at=20.0)
    store.close()

    reopened = SqlitePersistence(str(path))
    try:
        assert reopened.load("percept") == {"gate": "consumed"}
        rows = reopened.load_pending_deliveries()
        assert [row.event_id for row in rows] == ["evt-old", "evt-a", "evt-b"]
        assert rows[1].attempts == 1
        assert rows[1].last_attempt_at == 11.0
        assert rows[1].next_attempt_at == 20.0
        expected_json = json.dumps(
            rows[1].payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        assert rows[1].payload_json == expected_json
        assert rows[1].payload_sha256 == hashlib.sha256(expected_json.encode()).hexdigest()
        assert reopened.load_pending_deliveries(limit=1)[0].event_id == "evt-old"

        columns = {
            row[1]
            for row in reopened._conn.execute("PRAGMA table_info(percept_delivery_outbox)")
        }
        assert {"payload_json", "payload_sha256", "sink_config_ref", "attempts"} <= columns
        assert not ({"url", "headers", "authorization", "signing_secret"} & columns)
    finally:
        reopened.close()


def test_duplicate_enqueue_is_idempotent_but_collision_rolls_back_state_and_batch(tmp_path) -> None:
    store = SqlitePersistence(str(tmp_path / "atomic.sqlite3"))
    original = _pending("evt-1")
    store.commit_state_and_enqueue({"revision": 1}, [original])
    store.note_delivery_attempt("evt-1", attempted_at=12.0)

    # Same stable identity is a no-op for the row and does not reset its attempt metadata.
    store.commit_state_and_enqueue({"revision": 2}, [_pending("evt-1")])
    assert store.load("percept") == {"revision": 2}
    assert store.load_pending_deliveries()[0].attempts == 1

    # State is written first inside the transaction and evt-new is inserted before the collision.
    # Both must roll back when evt-1 is reused for different bytes.
    with pytest.raises(OutboxEventCollision):
        store.commit_state_and_enqueue(
            {"revision": 3},
            [_pending("evt-new"), _pending("evt-1", message="different payload")],
        )
    assert store.load("percept") == {"revision": 2}
    assert [row.event_id for row in store.load_pending_deliveries()] == ["evt-1"]
    store.close()


def test_ack_removes_pending_and_persists_delivered_ledger_in_one_transaction(tmp_path) -> None:
    path = tmp_path / "ack.sqlite3"
    store = SqlitePersistence(str(path))
    store.commit_state_and_enqueue({"gate": "spent"}, [_pending("evt-1")])
    ledger = [["evt-1", 42.0]]

    assert store.ack_delivery("evt-1", ledger, delivered_at=42.0)
    assert store.load_pending_deliveries() == []
    assert store.load("percept_delivery_ledger") == ledger
    # Idempotent re-ACK still leaves the authoritative delivered ledger persisted.
    assert not store.ack_delivery("evt-1", ledger, delivered_at=42.0)
    store.close()

    reopened = SqlitePersistence(str(path))
    try:
        assert reopened.load_pending_deliveries() == []
        assert reopened.load("percept_delivery_ledger") == ledger
    finally:
        reopened.close()


def test_ack_database_failure_rolls_back_pending_delete(tmp_path) -> None:
    store = SqlitePersistence(str(tmp_path / "ack-rollback.sqlite3"))
    store.commit_state_and_enqueue({"gate": "spent"}, [_pending("evt-rollback")])
    store._conn.execute(
        """
        CREATE TRIGGER reject_delivery_ledger
        BEFORE INSERT ON kv
        WHEN NEW.key = 'percept_delivery_ledger'
        BEGIN
            SELECT RAISE(ABORT, 'injected ledger write failure');
        END
        """
    )
    store._conn.commit()

    with pytest.raises(sqlite3.IntegrityError, match="injected ledger write failure"):
        store.ack_delivery(
            "evt-rollback", [["evt-rollback", 50.0]], delivered_at=50.0
        )
    assert [row.event_id for row in store.load_pending_deliveries()] == ["evt-rollback"]
    assert store.load("percept_delivery_ledger") is None
    store.close()


def test_ack_refuses_to_drop_row_when_ledger_does_not_contain_event(tmp_path) -> None:
    store = SqlitePersistence(str(tmp_path / "ack-validation.sqlite3"))
    store.commit_state_and_enqueue({"gate": "spent"}, [_pending("evt-1")])
    with pytest.raises(OutboxValidationError, match="must contain event_id"):
        store.ack_delivery("evt-1", [], delivered_at=42.0)
    assert [row.event_id for row in store.load_pending_deliveries()] == ["evt-1"]
    store.close()


def test_payload_hash_detects_database_tampering(tmp_path) -> None:
    store = SqlitePersistence(str(tmp_path / "tamper.sqlite3"))
    store.commit_state_and_enqueue({"gate": "spent"}, [_pending("evt-1")])
    store._conn.execute(
        "UPDATE percept_delivery_outbox SET payload_json = ? WHERE event_id = 'evt-1'",
        (json.dumps({"id": "evt-1", "message": "tampered"}),),
    )
    store._conn.commit()

    with pytest.raises(OutboxCorruptError, match="payload hash mismatch"):
        store.load_pending_deliveries()
    store.close()


def test_v0_outbox_table_is_additively_migrated_without_losing_kv_or_rows(tmp_path) -> None:
    path = tmp_path / "legacy.sqlite3"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE kv (key TEXT PRIMARY KEY, value TEXT)")
    conn.execute("INSERT INTO kv VALUES ('existing', '{\"value\": 1}')")
    conn.execute(
        "CREATE TABLE percept_delivery_outbox "
        "(event_id TEXT PRIMARY KEY, payload_json TEXT NOT NULL)"
    )
    legacy_payload = json.dumps({"id": "evt-legacy", "message": "already pending"})
    conn.execute(
        "INSERT INTO percept_delivery_outbox VALUES (?, ?)",
        ("evt-legacy", legacy_payload),
    )
    conn.commit()
    conn.close()

    migrated = SqlitePersistence(str(path))
    try:
        assert migrated.load("existing") == {"value": 1}
        [row] = migrated.load_pending_deliveries()
        assert row.event_id == "evt-legacy"
        assert row.sink_config_ref == "legacy-default"
        assert len(row.payload_sha256) == 64
        columns = {
            item[1]
            for item in migrated._conn.execute("PRAGMA table_info(percept_delivery_outbox)")
        }
        assert {
            "schema_version",
            "payload_sha256",
            "sink_config_ref",
            "attempts",
            "created_at",
            "updated_at",
            "last_attempt_at",
            "next_attempt_at",
        } <= columns
    finally:
        migrated.close()


def test_pending_delivery_rejects_transport_credentials_and_mismatched_ids() -> None:
    with pytest.raises(OutboxValidationError, match="opaque identifier"):
        PendingDelivery(
            event_id="evt-1",
            payload={"id": "evt-1"},
            sink_config_ref="https://user:secret@example.test/hook",
            created_at=1.0,
        )
    with pytest.raises(OutboxValidationError, match="same stable event ID"):
        PendingDelivery(
            event_id="evt-1",
            payload={"id": "evt-other"},
            sink_config_ref="enterprise-primary",
            created_at=1.0,
        )
