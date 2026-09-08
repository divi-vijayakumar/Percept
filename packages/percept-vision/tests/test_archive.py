"""Standalone durable audit archive qualification; no live-runtime eviction is enabled."""
from __future__ import annotations

import hashlib
import json
import sqlite3

import pytest

from percept.archive import (
    ACTION_RECORD_STREAM,
    AUDIT_ARCHIVE_CAPABILITY,
    EPISODE_REVISION_STREAM,
    EVENT_CHECKPOINT_STREAM,
    ArchiveCollisionError,
    ArchiveCorruptError,
    ArchiveCursor,
    ArchivePolicyConflict,
    ArchiveValidationError,
    DurableAuditArchive,
    EpisodeIncidentRevisionRecord,
    EventLogCheckpointRecord,
    HotTailPolicy,
    ImmutableActionRecord,
    SqliteAuditArchive,
)
from percept.consent import ActionRecord
from percept.events import EventLog


def _episode_revision(
    revision: int = 1,
    *,
    incident_id: str | None = None,
    value: str = "open",
) -> EpisodeIncidentRevisionRecord:
    episode_id = "door:ep1"
    revision_id = f"{episode_id}#r{revision}"
    incident_id = incident_id or f"inc-door-{revision}"
    return EpisodeIncidentRevisionRecord(
        revision_id=revision_id,
        episode_id=episode_id,
        incident_id=incident_id,
        occurred_at=10.0 + revision,
        episode={"id": episode_id, "status": value, "revisions": []},
        incident={
            "id": incident_id,
            "revision_id": revision_id,
            "message": value,
        },
    )


def _action(record_id: str, *, detail: str = "sent", t: float = 20.0) -> ImmutableActionRecord:
    return ImmutableActionRecord.capture(
        record_id,
        ActionRecord(
            kind="fire",
            goal_id="door",
            detail={"result": detail},
            t=t,
            allowed=True,
        ),
    )


def _event_checkpoint(checkpoint_id: str = "events@1") -> EventLogCheckpointRecord:
    log = EventLog(merge_gap_s=1.0)
    log.observe("wave", 0.0, participants=["alice"], confidence=0.7)
    log.observe("wave", 0.5, participants=["alice"], confidence=0.9)
    log.observe("wave", 3.0, participants=["bob"], confidence=0.8)
    log.observe("jump", 4.0, participants=["bob"], confidence=0.95)
    return EventLogCheckpointRecord.capture(checkpoint_id, log, occurred_at=5.0)


def test_archive_capability_is_explicit_and_does_not_enforce_hot_tail() -> None:
    archive = SqliteAuditArchive(":memory:")
    try:
        assert isinstance(archive, DurableAuditArchive)
        assert archive.archive_capability == AUDIT_ARCHIVE_CAPABILITY
        assert archive.enforces_hot_tail is False
    finally:
        archive.close()


def test_atomic_append_is_canonical_idempotent_cursor_ordered_and_reopenable(tmp_path) -> None:
    path = tmp_path / "audit.sqlite3"
    archive = SqliteAuditArchive(str(path), clock=lambda: 100.0)
    records = [_action("action-1"), _episode_revision(), _event_checkpoint()]
    first = archive.append_batch(records)
    assert first.inserted == 3
    assert first.existing == 0
    assert [cursor.value for cursor in first.cursors] == [1, 2, 3]

    duplicate = archive.append_batch(records)
    assert duplicate.inserted == 0
    assert duplicate.existing == 3
    assert duplicate.cursors == first.cursors

    raw = archive._conn.execute(
        "SELECT payload_json, payload_sha256 FROM percept_audit_archive WHERE cursor = 1"
    ).fetchone()
    payload_json, digest = raw
    assert payload_json == json.dumps(
        json.loads(payload_json),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    assert digest == hashlib.sha256(payload_json.encode()).hexdigest()

    first_page = archive.read(limit=2)
    assert [entry.stable_key for entry in first_page.entries] == [
        "action-1",
        "door:ep1#r1",
    ]
    assert first_page.next_cursor == ArchiveCursor(2)
    second_page = archive.read(after=first_page.next_cursor, limit=2)
    assert [entry.stable_key for entry in second_page.entries] == ["events@1"]
    assert archive.latest_cursor() == ArchiveCursor(3)
    assert archive.latest_cursor(stream=EPISODE_REVISION_STREAM) == ArchiveCursor(2)
    archive.close()

    reopened = SqliteAuditArchive(str(path), clock=lambda: 200.0)
    try:
        page = reopened.read()
        assert [entry.stream for entry in page.entries] == [
            ACTION_RECORD_STREAM,
            EPISODE_REVISION_STREAM,
            EVENT_CHECKPOINT_STREAM,
        ]
        assert reopened.read_action_records().entries[0].payload["action"]["goal_id"] == "door"
        assert reopened.read_episode_revisions().entries[0].stable_key == "door:ep1#r1"
        assert reopened.read_event_checkpoints().entries[0].stable_key == "events@1"
    finally:
        reopened.close()


def test_collision_rolls_back_every_new_row_in_the_batch(tmp_path) -> None:
    archive = SqliteAuditArchive(str(tmp_path / "collision.sqlite3"))
    archive.append_action_record(_action("action-1", detail="original"))

    with pytest.raises(ArchiveCollisionError, match="different immutable bytes"):
        archive.append_batch(
            [
                _episode_revision(),
                _action("action-1", detail="conflict"),
            ]
        )
    page = archive.read()
    assert [(entry.stream, entry.stable_key) for entry in page.entries] == [
        (ACTION_RECORD_STREAM, "action-1")
    ]
    archive.close()


def test_sqlite_failure_in_middle_of_batch_rolls_back_prior_insert(tmp_path) -> None:
    archive = SqliteAuditArchive(str(tmp_path / "failure.sqlite3"))
    archive._conn.execute(
        """
        CREATE TRIGGER reject_archive_row
        BEFORE INSERT ON percept_audit_archive
        WHEN NEW.stable_key = 'action-fail'
        BEGIN
            SELECT RAISE(ABORT, 'injected archive write failure');
        END
        """
    )
    archive._conn.commit()

    with pytest.raises(sqlite3.IntegrityError, match="injected archive write failure"):
        archive.append_batch([_action("action-before"), _action("action-fail")])
    assert archive.read().entries == ()
    archive.close()


def test_read_detects_payload_hash_and_index_identity_corruption(tmp_path) -> None:
    archive = SqliteAuditArchive(str(tmp_path / "corrupt.sqlite3"))
    archive.append_action_record(_action("action-1"))
    archive._conn.execute(
        "UPDATE percept_audit_archive SET payload_json = ? WHERE stable_key = 'action-1'",
        ('{"record_id":"action-1","tampered":true}',),
    )
    archive._conn.commit()
    with pytest.raises(ArchiveCorruptError, match="hash/canonical mismatch"):
        archive.read()
    archive.close()

    path = tmp_path / "identity-corrupt.sqlite3"
    second = SqliteAuditArchive(str(path))
    second.append_action_record(_action("action-2"))
    second._conn.execute(
        "UPDATE percept_audit_archive SET stable_key = 'action-other' WHERE stable_key = 'action-2'"
    )
    second._conn.commit()
    with pytest.raises(ArchiveCorruptError, match="indexed identity mismatch"):
        second.read()
    second.close()

    semantic = SqliteAuditArchive(str(tmp_path / "semantic-corrupt.sqlite3"))
    semantic.append_event_checkpoint(_event_checkpoint())
    payload = semantic.read_event_checkpoints().entries[0].payload
    payload["counts_by_type"] = {"wave": 3}
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    semantic._conn.execute(
        "UPDATE percept_audit_archive SET payload_json = ?, payload_sha256 = ?",
        (encoded, hashlib.sha256(encoded.encode()).hexdigest()),
    )
    semantic._conn.commit()
    with pytest.raises(ArchiveCorruptError, match="record schema mismatch"):
        semantic.read()
    semantic.close()


def test_schema_corruption_is_rejected_at_open(tmp_path) -> None:
    path = tmp_path / "bad-schema.sqlite3"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE percept_audit_archive (cursor INTEGER PRIMARY KEY)")
    conn.commit()
    conn.close()
    with pytest.raises(ArchiveCorruptError, match="missing required columns"):
        SqliteAuditArchive(str(path))


def test_event_checkpoint_freezes_history_counter_and_verified_counts() -> None:
    checkpoint = _event_checkpoint()
    assert checkpoint.total_count == 3
    assert checkpoint.counter == 3
    assert checkpoint.counts_by_type == {"wave": 2, "jump": 1}
    assert [event["id"] for event in checkpoint.events] == ["ev1", "ev2", "ev3"]
    assert checkpoint.events[0]["t"] == 0.5
    assert checkpoint.events[0]["confidence"] == 0.9

    with pytest.raises(ArchiveValidationError, match="does not match"):
        EventLogCheckpointRecord(
            checkpoint_id="bad-count",
            occurred_at=5.0,
            events=checkpoint.events,
            counts_by_type={"wave": 3},
            total_count=3,
            counter=3,
        )


def test_action_capture_is_detached_and_archive_rows_are_immutable(tmp_path) -> None:
    live = ActionRecord(
        kind="act",
        goal_id="door",
        detail={"command": "lock"},
        t=30.0,
        allowed=True,
    )
    frozen = ImmutableActionRecord.capture("action-lock-1", live)
    live.detail["command"] = "unlock"

    archive = SqliteAuditArchive(str(tmp_path / "action.sqlite3"))
    archive.append_action_record(frozen)
    payload = archive.read_action_records().entries[0].payload
    assert payload["action"]["detail"] == {"command": "lock"}
    with pytest.raises(ArchiveCollisionError):
        archive.append_action_record(
            ImmutableActionRecord.capture("action-lock-1", live)
        )
    archive.close()


def test_episode_revision_requires_stable_relationships_and_strict_json() -> None:
    with pytest.raises(ArchiveValidationError, match=r"<episode_id>#r<n>"):
        EpisodeIncidentRevisionRecord(
            revision_id="unrelated#r1",
            episode_id="door:ep1",
            incident_id="inc-1",
            occurred_at=1.0,
            episode={"id": "door:ep1"},
            incident={"id": "inc-1"},
        )
    with pytest.raises(ArchiveValidationError, match="strict JSON"):
        EpisodeIncidentRevisionRecord(
            revision_id="door:ep1#r1",
            episode_id="door:ep1",
            incident_id="inc-1",
            occurred_at=1.0,
            episode={"id": "door:ep1", "value": float("nan")},
            incident={"id": "inc-1", "revision_id": "door:ep1#r1"},
        )
    with pytest.raises(ArchiveValidationError, match="revision_id is required"):
        EpisodeIncidentRevisionRecord(
            revision_id="door:ep1#r1",
            episode_id="door:ep1",
            incident_id="inc-1",
            occurred_at=1.0,
            episode={"id": "door:ep1"},
            incident={"id": "inc-1"},
        )


def test_hot_tail_policy_is_versioned_canonical_metadata_not_eviction(tmp_path) -> None:
    path = tmp_path / "policy.sqlite3"
    archive = SqliteAuditArchive(str(path), clock=lambda: 100.0)
    v1 = HotTailPolicy(
        stream=EPISODE_REVISION_STREAM,
        revision=1,
        max_records=128,
        max_bytes=1_048_576,
        metadata={"reason": "operator-declared live tail"},
    )
    assert archive.declare_hot_tail_policy(v1)
    assert not archive.declare_hot_tail_policy(v1)
    with pytest.raises(ArchivePolicyConflict, match="different immutable bytes"):
        archive.declare_hot_tail_policy(
            HotTailPolicy(
                stream=EPISODE_REVISION_STREAM,
                revision=1,
                max_records=64,
                max_bytes=1_048_576,
            )
        )
    v2 = HotTailPolicy(
        stream=EPISODE_REVISION_STREAM,
        revision=2,
        max_records=64,
        max_bytes=524_288,
    )
    assert archive.declare_hot_tail_policy(v2)
    assert archive.read_hot_tail_policy(EPISODE_REVISION_STREAM) == v2
    assert archive.enforces_hot_tail is False
    raw = archive._conn.execute(
        "SELECT policy_json FROM percept_archive_hot_tail_policy "
        "WHERE stream = ? AND revision = 2",
        (EPISODE_REVISION_STREAM,),
    ).fetchone()[0]
    assert json.loads(raw)["enforcement"] == "metadata_only"
    assert json.loads(raw)["eviction_wired"] is False
    archive.close()

    reopened = SqliteAuditArchive(str(path))
    try:
        assert reopened.read_hot_tail_policy(EPISODE_REVISION_STREAM) == v2
    finally:
        reopened.close()


def test_cursor_policy_and_record_validation_are_fail_loud() -> None:
    with pytest.raises(ArchiveValidationError, match="positive integer"):
        HotTailPolicy(
            stream=ACTION_RECORD_STREAM,
            revision=1,
            max_records=0,
            max_bytes=1024,
        )
    archive = SqliteAuditArchive(":memory:")
    try:
        with pytest.raises(ArchiveValidationError, match="ArchiveCursor"):
            archive.read(after=1)  # type: ignore[arg-type]
        with pytest.raises(ArchiveValidationError, match="unsupported archive record"):
            archive.append_batch([object()])  # type: ignore[list-item]
    finally:
        archive.close()
