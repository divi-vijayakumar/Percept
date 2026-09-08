"""Crash qualification for transactional runtime webhook delivery.

The durable handoff has two independently testable boundaries:

* the consumed cognition state and frozen Incident payload must both exist before HTTP exposure;
* a successful HTTP send followed by a failed durable ACK must leave the same stable event ID
  pending, so a fresh runtime replays it before consuming its source.

The downstream contract is at-least-once, not exactly-once: a process death after the receiver saw
the body but before Percept committed the ACK can produce a duplicate delivery attempt.  Both
attempts carry the identical Incident ID so the receiver can deduplicate the logical event.
"""
from __future__ import annotations

import asyncio
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

import pytest

from percept import Goal, Percept
from percept.adapters import Verdict
from percept.backends.fake import FakeVision
from percept.harness.tier0 import gray_frame
from percept.incidents import make_incident
from percept.persistence import InMemoryPersistence, PendingDelivery, SqlitePersistence
from percept.runtime import run
from percept.runtime.sinks import WebhookSink, incident_payload


EVENT_ID = "inc-crash-boundary-0001"
YES = Verdict(satisfied=True, confidence=0.95, evidence="scripted: door open")
STILL = gray_frame(8, 8, 0)
WEBHOOK_URL = "https://hooks.example.test/incidents"


class _CrashBeforeWebhookAck(BaseException):
    """A process death, not a catchable delivery failure or terminal sink result."""


class _Response:
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


class _FailingAckSqlitePersistence(SqlitePersistence):
    def __init__(self, path: str) -> None:
        super().__init__(path)
        self.fail_ack = True

    def ack_delivery(self, *args, **kwargs):
        if self.fail_ack:
            raise sqlite3.OperationalError("injected ACK commit failure")
        return super().ack_delivery(*args, **kwargs)


@dataclass
class _CrashCheckpoint:
    path: Path
    attempted_payloads: list[dict]


def _core() -> Percept:
    return Percept.create(
        vision=FakeVision(script={"open": YES}),
        discover_plugins=False,
    )


def _goal() -> Goal:
    # Fixed cadence makes the still restart frame reach the real worker judge path.  Restored
    # refractory state must suppress it; the only webhook on restart must be the pending row.
    return Goal(
        id="door",
        condition="the door is open",
        say="door open",
        state={"type": "watch", "cadence": {"gate_on": "fixed", "min_judge_interval": 1.0}},
    )


def _checkpoint_after_state_commit_before_webhook_ack(path: Path) -> _CrashCheckpoint:
    """Create the exact durable state an abrupt worker death leaves in the crash window."""

    core = _core()
    core.add_goal(_goal())
    [fire] = asyncio.run(core.perceive_judged("open", source="camera", t=0.0))
    incident = make_incident(fire, ts=0.0, id=EVENT_ID)

    store = SqlitePersistence(str(path))
    webhook = WebhookSink(WEBHOOK_URL, retries=0)
    pending = PendingDelivery(
        event_id=incident.id,
        payload=incident_payload(incident, include_media=webhook.include_media),
        sink_config_ref=webhook.outbox_sink_config_ref,
        created_at=100.0,
    )
    store.commit_state_and_enqueue(core.snapshot(), [pending])
    assert store.note_delivery_attempt(EVENT_ID, attempted_at=101.0)

    attempted_payloads: list[dict] = []

    def crash_after_request_bytes(req, timeout):
        del timeout
        attempted_payloads.append(json.loads(req.data))
        raise _CrashBeforeWebhookAck("power loss before HTTP response")

    with patch(
        "percept.runtime.sinks.socket.getaddrinfo",
        side_effect=OSError("offline qualification"),
    ), patch("percept.runtime.sinks._send", side_effect=crash_after_request_bytes):
        with pytest.raises(_CrashBeforeWebhookAck):
            webhook(pending.payload)

    # A hard crash discards core, incident, and sink here.  Only the SQLite file survives.
    store.close()
    return _CrashCheckpoint(path=path, attempted_payloads=attempted_payloads)


async def _restart(checkpoint: _CrashCheckpoint) -> tuple[list[dict], list]:
    payloads: list[dict] = []

    def accept(req, timeout):
        del timeout
        payload = json.loads(req.data)
        observer = sqlite3.connect(checkpoint.path)
        try:
            [attempts] = observer.execute(
                "SELECT attempts FROM percept_delivery_outbox WHERE event_id = ?",
                (payload["id"],),
            ).fetchone()
            assert attempts == 2
        finally:
            observer.close()
        payloads.append(payload)
        return _Response()

    fresh = _core()
    reopened = SqlitePersistence(str(checkpoint.path))
    yielded = []
    try:
        with patch(
            "percept.runtime.sinks.socket.getaddrinfo",
            side_effect=OSError("offline qualification"),
        ), patch("percept.runtime.sinks._send", side_effect=accept):
            async for incident in run(
                fresh,
                source=[(5.0, STILL, "open")],
                persistence=reopened,
                restore=True,
                on_incident=WebhookSink(WEBHOOK_URL, retries=0),
                judge_dispatch="inline",
            ):
                yielded.append(incident)
        assert reopened.load_pending_deliveries() == []
        assert dict(reopened.load("percept_delivery_ledger"))[EVENT_ID] >= 0
    finally:
        reopened.close()
    return payloads, yielded


def test_crash_checkpoint_contains_consumed_state_and_replayable_payload(tmp_path: Path) -> None:
    checkpoint = _checkpoint_after_state_commit_before_webhook_ack(
        tmp_path / "crash.sqlite3"
    )

    assert [row["id"] for row in checkpoint.attempted_payloads] == [EVENT_ID]
    reopened = SqlitePersistence(str(checkpoint.path))
    try:
        assert reopened.load("percept") is not None
        assert reopened.load("percept_delivery_ledger") is None
        [pending] = reopened.load_pending_deliveries()
        assert pending.event_id == EVENT_ID
        assert pending.payload["id"] == EVENT_ID
        assert pending.attempts == 1
        assert WEBHOOK_URL not in pending.sink_config_ref
    finally:
        reopened.close()


def test_unacked_webhook_is_replayed_after_restart_with_the_same_event_id(
    tmp_path: Path,
) -> None:
    """A crash after network exposure replays once with the original stable event ID."""

    checkpoint = _checkpoint_after_state_commit_before_webhook_ack(
        tmp_path / "replay.sqlite3"
    )
    payloads, yielded = asyncio.run(_restart(checkpoint))

    assert [payload["id"] for payload in payloads] == [EVENT_ID]
    # The continuing true frame cannot mint a replacement Incident: restored cognition consumed the
    # edge.  Startup replay is a delivery action, so it is not yielded as a newly observed incident.
    assert yielded == []


def test_runtime_commits_state_and_payload_before_network_then_acks(tmp_path: Path) -> None:
    path = tmp_path / "ordering.sqlite3"
    store = SqlitePersistence(str(path))
    core = _core()
    core.add_goal(_goal())
    observed: list[str] = []

    def inspect_durable_handoff(req, timeout):
        del timeout
        payload = json.loads(req.data)
        event_id = payload["id"]
        observer = sqlite3.connect(path)
        try:
            assert observer.execute(
                "SELECT value FROM kv WHERE key = 'percept'"
            ).fetchone() is not None
            [row] = observer.execute(
                "SELECT event_id, payload_json FROM percept_delivery_outbox"
            ).fetchall()
            assert row[0] == event_id
            assert json.loads(row[1]) == payload
            assert observer.execute(
                "SELECT value FROM kv WHERE key = 'percept_delivery_ledger'"
            ).fetchone() is None
        finally:
            observer.close()
        observed.append(event_id)
        return _Response()

    async def exercise() -> list:
        with patch(
            "percept.runtime.sinks.socket.getaddrinfo",
            side_effect=OSError("offline qualification"),
        ), patch("percept.runtime.sinks._send", side_effect=inspect_durable_handoff):
            return [
                incident
                async for incident in run(
                    core,
                    source=[(0.0, STILL, "open")],
                    persistence=store,
                    on_incident=WebhookSink(WEBHOOK_URL, retries=0),
                    judge_dispatch="inline",
                )
            ]

    try:
        [incident] = asyncio.run(exercise())
        assert observed == [incident.id]
        assert store.load_pending_deliveries() == []
        assert dict(store.load("percept_delivery_ledger"))[incident.id] >= 0
    finally:
        store.close()


def test_send_before_failed_ack_remains_pending_and_replays_even_in_ignore_mode(
    tmp_path: Path,
) -> None:
    path = tmp_path / "ack-crash.sqlite3"
    store = _FailingAckSqlitePersistence(str(path))
    core = _core()
    core.add_goal(_goal())
    first_attempt: list[str] = []

    def accept_then_fail_ack(req, timeout):
        del timeout
        first_attempt.append(json.loads(req.data)["id"])
        return _Response()

    async def crash_at_ack() -> None:
        with patch(
            "percept.runtime.sinks.socket.getaddrinfo",
            side_effect=OSError("offline qualification"),
        ), patch("percept.runtime.sinks._send", side_effect=accept_then_fail_ack):
            async for _ in run(
                core,
                source=[(0.0, STILL, "open")],
                persistence=store,
                on_incident=WebhookSink(WEBHOOK_URL, retries=0),
                sink_errors="ignore",
                judge_dispatch="inline",
            ):
                pass

    with pytest.raises(RuntimeError, match="durable ACK failed"):
        asyncio.run(crash_at_ack())
    [event_id] = first_attempt
    [pending] = store.load_pending_deliveries()
    assert pending.event_id == event_id
    assert pending.attempts == 1
    assert store.load("percept_delivery_ledger") == []
    store.close()

    replayed: list[str] = []

    def accept_replay(req, timeout):
        del timeout
        replayed.append(json.loads(req.data)["id"])
        return _Response()

    reopened = SqlitePersistence(str(path))
    try:
        with patch(
            "percept.runtime.sinks.socket.getaddrinfo",
            side_effect=OSError("offline qualification"),
        ), patch("percept.runtime.sinks._send", side_effect=accept_replay):
            asyncio.run(
                _drain(
                    _core(),
                    reopened,
                    WebhookSink(WEBHOOK_URL, retries=0),
                )
            )
        assert replayed == [event_id]
        assert reopened.load_pending_deliveries() == []
        assert event_id in dict(reopened.load("percept_delivery_ledger"))
    finally:
        reopened.close()


async def _drain(core: Percept, store: SqlitePersistence, sink: WebhookSink) -> None:
    async for _ in run(
        core,
        source=[(5.0, STILL, "open")],
        persistence=store,
        restore=True,
        on_incident=sink,
        judge_dispatch="inline",
    ):
        pass


def test_transactional_store_keeps_non_webhook_sink_on_legacy_path(tmp_path: Path) -> None:
    store = SqlitePersistence(str(tmp_path / "callback.sqlite3"))
    core = _core()
    core.add_goal(_goal())
    callbacks = []
    try:
        yielded = asyncio.run(
            _collect(
                core,
                store,
                callbacks.append,
            )
        )
        assert [incident.id for incident in callbacks] == [incident.id for incident in yielded]
        assert store.load_pending_deliveries() == []
    finally:
        store.close()


def test_legacy_store_keeps_webhook_sink_on_existing_idempotent_path() -> None:
    store = InMemoryPersistence()
    core = _core()
    core.add_goal(_goal())
    payloads: list[dict] = []

    def accept(req, timeout):
        del timeout
        payloads.append(json.loads(req.data))
        return _Response()

    with patch(
        "percept.runtime.sinks.socket.getaddrinfo",
        side_effect=OSError("offline qualification"),
    ), patch("percept.runtime.sinks._send", side_effect=accept):
        yielded = asyncio.run(
            _collect(
                core,
                store,
                WebhookSink(WEBHOOK_URL, retries=0),
            )
        )
    assert [payload["id"] for payload in payloads] == [incident.id for incident in yielded]
    assert list(dict(store.load("percept_delivery_ledger"))) == [yielded[0].id]


async def _collect(core: Percept, store, sink) -> list:
    return [
        incident
        async for incident in run(
            core,
            source=[(0.0, STILL, "open")],
            persistence=store,
            on_incident=sink,
            judge_dispatch="inline",
        )
    ]


def test_startup_replays_pending_rows_in_storage_order_before_source_iteration(
    tmp_path: Path,
) -> None:
    path = tmp_path / "ordered-replay.sqlite3"
    store = SqlitePersistence(str(path))
    webhook = WebhookSink(WEBHOOK_URL, retries=0)
    core = _core()
    store.commit_state_and_enqueue(
        core.snapshot(),
        [
            PendingDelivery(
                event_id="evt-b",
                payload={"id": "evt-b"},
                sink_config_ref=webhook.outbox_sink_config_ref,
                created_at=10.0,
            ),
            PendingDelivery(
                event_id="evt-old",
                payload={"id": "evt-old"},
                sink_config_ref=webhook.outbox_sink_config_ref,
                created_at=5.0,
            ),
            PendingDelivery(
                event_id="evt-a",
                payload={"id": "evt-a"},
                sink_config_ref=webhook.outbox_sink_config_ref,
                created_at=10.0,
            ),
        ],
    )
    order: list[str] = []

    class _Source:
        def __aiter__(self):
            async def iterate():
                order.append("source")
                if False:
                    yield None

            return iterate()

    def accept(req, timeout):
        del timeout
        order.append(json.loads(req.data)["id"])
        return _Response()

    async def exercise() -> None:
        async for _ in run(
            _core(),
            source=_Source(),
            persistence=store,
            restore=True,
            on_incident=webhook,
            judge_dispatch="inline",
        ):
            pass

    try:
        with patch(
            "percept.runtime.sinks.socket.getaddrinfo",
            side_effect=OSError("offline qualification"),
        ), patch("percept.runtime.sinks._send", side_effect=accept):
            asyncio.run(exercise())
        assert order == ["evt-old", "evt-a", "evt-b", "source"]
        assert store.load_pending_deliveries() == []
    finally:
        store.close()
