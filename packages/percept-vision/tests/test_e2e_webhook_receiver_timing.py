"""Receiver-observed timing for the real runtime -> webhook delivery path.

This is deliberately an offline wiring probe, not a provider-latency benchmark.  ``FakeVision``
drives one rising-edge incident through the public ``percept.runtime.run`` entrypoint and a real
``WebhookSink`` POSTs it to a loopback HTTP server.  The receiver holds its response until after the
SDK consumer has observed the Incident, making one important clock distinction deterministic:

* ``sdk_yield_at`` is when ``async for incident in run(...)`` publishes the Incident;
* ``receiver_arrival_at`` is when the HTTP handler receives the JSON carrying that same event id;
* ``receiver_ack_at`` is when that handler sends its successful response.

Arrival may race the SDK yield because delivery runs off the perception critical path.  The held
ACK proves the stronger contract boundary without a wall-clock threshold: SDK publication does not
mean that the webhook has already been acknowledged.  All timestamps use ``time.monotonic()`` in
this process, so a production/staging probe can calculate the same stage deltas without clock skew.
"""

from __future__ import annotations

import asyncio
import time
import unittest
from dataclasses import dataclass
from typing import Any

from percept import Goal, Percept
from percept.adapters import Verdict
from percept.backends.fake import FakeVision
from percept.edge.tier0 import gray_frame
from percept.harness.webhook_probe import LoopbackWebhookProbe
from percept.runtime import run


W, H = 8, 8
STILL = gray_frame(W, H, 0)
MOVED = gray_frame(W, H, 255)
SOURCE = [
    (0.0, STILL, "clear"),
    (1.0, MOVED, "intruder"),  # still -> moved admits the scripted true verdict and fires once
]


@dataclass(frozen=True)
class _DeliveryTiming:
    event_id: str
    sdk_yield_at: float
    receiver_arrival_at: float
    receiver_ack_at: float
    run_completed_at: float

    @property
    def sdk_to_receiver_arrival_s(self) -> float:
        """Signed because receiver arrival and SDK publication are intentionally concurrent."""
        return self.receiver_arrival_at - self.sdk_yield_at

    @property
    def sdk_to_receiver_ack_s(self) -> float:
        return self.receiver_ack_at - self.sdk_yield_at


def _core() -> Percept:
    vision = FakeVision(
        script={
            "clear": Verdict(False, 0.95, "the zone is empty"),
            "intruder": Verdict(True, 0.95, "a person is in the zone"),
        }
    )
    core = Percept.create(vision=vision, discover_plugins=False)
    core.add_goal(
        Goal(
            id="zone",
            condition="a person is in the restricted zone",
            say="Alert: person in zone",
        )
    )
    return core


async def _probe(receiver: LoopbackWebhookProbe) -> _DeliveryTiming:
    yielded: list[tuple[Any, float]] = []

    async for incident in run(
        _core(),
        source=SOURCE,
        judge_dispatch="inline",
        on_incident=receiver.sink,
    ):
        sdk_yield_at = time.monotonic()
        yielded.append((incident, sdk_yield_at))

        # The probe's blocking wait must run off-loop: the delivery drain needs the event loop in
        # order to hand WebhookSink's blocking urllib work to its worker thread.
        arrival = await asyncio.to_thread(
            receiver.wait_for_arrival,
            incident.id,
            timeout_s=2.0,
        )
        if arrival is None:
            raise AssertionError("loopback receiver did not observe the webhook")

        if arrival.payload["id"] != incident.id:
            raise AssertionError("receiver payload id does not match the yielded Incident")
        if arrival.ack_at is not None:
            raise AssertionError("receiver ACK escaped before the test released it")

        receiver.release_ack(incident.id)
        acknowledged = await asyncio.to_thread(
            receiver.wait_for_ack,
            incident.id,
            timeout_s=2.0,
        )
        if acknowledged is None:
            raise AssertionError("loopback receiver did not send its ACK")

    run_completed_at = time.monotonic()
    if len(yielded) != 1:
        raise AssertionError(f"expected one rising-edge Incident, got {len(yielded)}")
    incident, sdk_yield_at = yielded[0]
    observation = receiver.lookup(incident.id)
    if observation is None or observation.ack_at is None:
        raise AssertionError("receiver timing record has no ACK timestamp")
    return _DeliveryTiming(
        event_id=incident.id,
        sdk_yield_at=sdk_yield_at,
        receiver_arrival_at=observation.arrival_at,
        receiver_ack_at=observation.ack_at,
        run_completed_at=run_completed_at,
    )


class ReceiverObservedWebhookTimingTests(unittest.TestCase):
    def test_sdk_yield_is_not_receiver_ack_for_the_same_event(self) -> None:
        with LoopbackWebhookProbe(hold_acks=True, sink_timeout_s=2.0) as receiver:
            timing = asyncio.run(_probe(receiver))

        # The body observed by the real receiver, the SDK publication, and the ACK all refer to the
        # same event id.  ACK is held until after SDK publication, so this ordering is causal rather
        # than a brittle latency threshold.  Arrival may be on either side of SDK publication.
        observation = receiver.lookup(timing.event_id)
        self.assertIsNotNone(observation)
        self.assertEqual(observation.event_id, timing.event_id)
        self.assertLessEqual(timing.receiver_arrival_at, timing.receiver_ack_at)
        self.assertLess(timing.sdk_yield_at, timing.receiver_ack_at)
        self.assertGreater(timing.sdk_to_receiver_ack_s, 0.0)
        self.assertLessEqual(timing.receiver_ack_at, timing.run_completed_at)


if __name__ == "__main__":
    unittest.main()
