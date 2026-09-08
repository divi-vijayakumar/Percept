"""Real loopback webhook receiver for offline delivery probes.

``LoopbackWebhookProbe`` owns a stdlib HTTP server and exposes an actual
:class:`~percept.runtime.sinks.WebhookSink` aimed at it.  Receiver observations are keyed by the
Incident ``id`` carried in the POST body and stamped with ``time.monotonic()`` at handler arrival
and successful ACK emission.  Lookup and waits are protected by one condition, so a runtime test or
evaluation harness can safely observe delivery from another thread without mocking the transport.

The probe measures the local runtime/delivery seam only.  It does not simulate provider latency,
network distance, or a receiver's durable commit.
"""

from __future__ import annotations

import copy
import json
import threading
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from percept.runtime.sinks import UrlPolicy, WebhookSink


@dataclass(frozen=True)
class WebhookReceiverObservation:
    """Thread-safe snapshot of one event ID at the loopback receiver.

    ``arrival_at`` is the first handler arrival for the event. ``ack_at`` is the first successful
    HTTP 204 response emission, or ``None`` while the receiver is holding/not yet sending its ACK.
    ``attempts`` counts POSTs carrying the same event ID, which keeps retry probes observable without
    changing the primary first-arrival/first-ACK latency anchors.
    """

    event_id: str
    payload: dict[str, Any]
    arrival_at: float
    ack_at: float | None
    attempts: int


class _ReceiverState:
    def __init__(self, *, hold_acks: bool, ack_hold_timeout_s: float) -> None:
        self.hold_acks = hold_acks
        self.ack_hold_timeout_s = ack_hold_timeout_s
        self._condition = threading.Condition()
        self._observations: dict[str, WebhookReceiverObservation] = {}
        self._released_event_ids: set[str] = set()
        self._release_all = False

    @staticmethod
    def _snapshot(observation: WebhookReceiverObservation) -> WebhookReceiverObservation:
        return WebhookReceiverObservation(
            event_id=observation.event_id,
            payload=copy.deepcopy(observation.payload),
            arrival_at=observation.arrival_at,
            ack_at=observation.ack_at,
            attempts=observation.attempts,
        )

    def record_arrival(self, payload: dict[str, Any]) -> str:
        event_id = str(payload["id"])
        observed_at = time.monotonic()
        with self._condition:
            existing = self._observations.get(event_id)
            if existing is None:
                observation = WebhookReceiverObservation(
                    event_id=event_id,
                    payload=copy.deepcopy(payload),
                    arrival_at=observed_at,
                    ack_at=None,
                    attempts=1,
                )
            else:
                observation = WebhookReceiverObservation(
                    event_id=event_id,
                    payload=copy.deepcopy(payload),
                    arrival_at=existing.arrival_at,
                    ack_at=existing.ack_at,
                    attempts=existing.attempts + 1,
                )
            self._observations[event_id] = observation
            self._condition.notify_all()
        return event_id

    def record_ack(self, event_id: str) -> None:
        acknowledged_at = time.monotonic()
        with self._condition:
            observation = self._observations[event_id]
            if observation.ack_at is None:
                self._observations[event_id] = WebhookReceiverObservation(
                    event_id=observation.event_id,
                    payload=observation.payload,
                    arrival_at=observation.arrival_at,
                    ack_at=acknowledged_at,
                    attempts=observation.attempts,
                )
            self._condition.notify_all()

    def lookup(self, event_id: str) -> WebhookReceiverObservation | None:
        with self._condition:
            observation = self._observations.get(str(event_id))
            return None if observation is None else self._snapshot(observation)

    def wait_for_arrival(
        self, event_id: str, *, timeout_s: float | None
    ) -> WebhookReceiverObservation | None:
        event_id = str(event_id)
        with self._condition:
            ready = self._condition.wait_for(
                lambda: event_id in self._observations,
                timeout=timeout_s,
            )
            if not ready:
                return None
            return self._snapshot(self._observations[event_id])

    def wait_for_ack(
        self, event_id: str, *, timeout_s: float | None
    ) -> WebhookReceiverObservation | None:
        event_id = str(event_id)
        with self._condition:
            ready = self._condition.wait_for(
                lambda: (
                    event_id in self._observations
                    and self._observations[event_id].ack_at is not None
                ),
                timeout=timeout_s,
            )
            if not ready:
                return None
            return self._snapshot(self._observations[event_id])

    def release_ack(self, event_id: str) -> None:
        with self._condition:
            self._released_event_ids.add(str(event_id))
            self._condition.notify_all()

    def release_all_acks(self) -> None:
        with self._condition:
            self._release_all = True
            self._condition.notify_all()

    def wait_until_ack_released(self, event_id: str) -> bool:
        if not self.hold_acks:
            return True
        with self._condition:
            return self._condition.wait_for(
                lambda: self._release_all or event_id in self._released_event_ids,
                timeout=self.ack_hold_timeout_s,
            )


def _handler_for(state: _ReceiverState) -> type[BaseHTTPRequestHandler]:
    class _WebhookProbeHandler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            try:
                body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
                payload = json.loads(body)
                if not isinstance(payload, dict) or not payload.get("id"):
                    raise ValueError("webhook payload must be an object with a non-empty id")
            except (json.JSONDecodeError, TypeError, ValueError):
                self.send_error(400, "invalid webhook payload")
                return

            event_id = state.record_arrival(payload)
            if not state.wait_until_ack_released(event_id):
                self.send_error(504, "probe did not release the held ACK")
                return

            self.send_response(204)
            self.end_headers()
            state.record_ack(event_id)

        def log_message(self, _format: str, *args: Any) -> None:
            # Tests inspect receiver state directly; suppress BaseHTTPRequestHandler's stderr log.
            del args

    return _WebhookProbeHandler


class LoopbackWebhookProbe:
    """Context-managed loopback receiver plus a real :class:`WebhookSink`.

    Set ``hold_acks=True`` to separate SDK publication from webhook acknowledgement.  After an event
    arrives, call :meth:`release_ack` with its event ID; teardown releases all outstanding handlers
    so a failed assertion cannot strand the process.

    The blocking ``wait_for_*`` methods are safe from any thread.  An asyncio caller should use
    ``await asyncio.to_thread(probe.wait_for_arrival, event_id, timeout_s=...)`` so it does not block
    the event loop that is driving delivery.
    """

    def __init__(
        self,
        *,
        hold_acks: bool = False,
        ack_hold_timeout_s: float = 5.0,
        sink_timeout_s: float = 5.0,
    ) -> None:
        if ack_hold_timeout_s <= 0:
            raise ValueError("ack_hold_timeout_s must be positive")
        if sink_timeout_s <= 0:
            raise ValueError("sink_timeout_s must be positive")
        self._state = _ReceiverState(
            hold_acks=bool(hold_acks),
            ack_hold_timeout_s=float(ack_hold_timeout_s),
        )
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), _handler_for(self._state))
        self._thread: threading.Thread | None = None
        self._closed = False
        self.sink = WebhookSink(
            self.url,
            timeout_s=float(sink_timeout_s),
            url_policy=UrlPolicy.dev(),
        )

    @property
    def url(self) -> str:
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}/incidents"

    def start(self) -> "LoopbackWebhookProbe":
        if self._closed:
            raise RuntimeError("a closed LoopbackWebhookProbe cannot be restarted")
        if self._thread is None:
            self._thread = threading.Thread(
                target=self._server.serve_forever,
                name="percept-webhook-probe",
                daemon=True,
            )
            self._thread.start()
        return self

    def close(self) -> None:
        if self._closed:
            return
        self._state.release_all_acks()
        if self._thread is not None:
            self._server.shutdown()
        self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._closed = True

    def __enter__(self) -> "LoopbackWebhookProbe":
        return self.start()

    def __exit__(self, _exc_type, _exc, _tb) -> None:
        self.close()

    @staticmethod
    def _validate_timeout(timeout_s: float | None) -> float | None:
        if timeout_s is None:
            return None
        timeout_s = float(timeout_s)
        if timeout_s < 0:
            raise ValueError("timeout_s must be non-negative or None")
        return timeout_s

    def lookup(self, event_id: str) -> WebhookReceiverObservation | None:
        """Return the latest immutable snapshot for ``event_id``, or ``None`` if unseen."""
        return self._state.lookup(event_id)

    def wait_for_arrival(
        self, event_id: str, *, timeout_s: float | None = None
    ) -> WebhookReceiverObservation | None:
        """Block until ``event_id`` reaches the HTTP handler, returning ``None`` on timeout."""
        return self._state.wait_for_arrival(
            event_id,
            timeout_s=self._validate_timeout(timeout_s),
        )

    def wait_for_ack(
        self, event_id: str, *, timeout_s: float | None = None
    ) -> WebhookReceiverObservation | None:
        """Block until a successful ACK is sent for ``event_id``, returning ``None`` on timeout."""
        return self._state.wait_for_ack(
            event_id,
            timeout_s=self._validate_timeout(timeout_s),
        )

    def release_ack(self, event_id: str) -> None:
        """Allow a held handler for ``event_id`` to emit its successful HTTP response."""
        self._state.release_ack(event_id)


__all__ = ["LoopbackWebhookProbe", "WebhookReceiverObservation"]
