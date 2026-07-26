"""P-DELIVERY-1 (the release criteria; ADR-001 §6) — offline scaffold probe.

Property under test: a slow / intermittently-failing / unavailable notification SINK stays
ISOLATED from capture+judging (a dead sink must never block or corrupt the perception pipeline);
the delivery queue and its retry age are BOUNDED (never grows/retries without limit); every event
carries a stable idempotency key (``event_id`` = ``Incident.id``) so a duplicate delivery ATTEMPT
(e.g. redelivery after a process restart) reuses that key and does NOT create a second logical
Incident; and a TERMINAL delivery failure is surfaced EXPLICITLY (dead-lettered with a reason),
never silently swallowed.

This is a probe/test harness, not a for-record benchmark, and it changes no production behavior.
It drives already-landed, unedited production code:

* ``percept.runtime.sinks.WebhookSink`` / ``BoundedDeadLetterQueue`` — the bounded-retry,
  bounded-queue delivery primitive. ``retries`` bounds the ATTEMPT COUNT,
  ``max_attempt_age_s`` bounds elapsed WALL-CLOCK retry time; either bound tripping is TERMINAL
  and dead-letters with an explicit ``reason`` (``"retries_exhausted"`` /
  ``"max_attempt_age_exceeded"``). ``BoundedDeadLetterQueue`` caps the dead-letter QUEUE itself by
  count/bytes, evicting oldest-first with a traced ``on_drop``.
* ``percept.incidents.make_incident`` — the fire -> Incident producer. ``Incident.id`` is the
  stable, durable handle the rest of the system references (see its docstring: feedback /
  dashboards / redaction all key off it) — this probe treats it as the delivery layer's
  idempotency key ("event_id"), since no separate ``event_id`` field exists on the wire today.
* ``percept.harness.metrics.MetricsCollector`` — reused, NOT reimplemented, as the
  delivery-queue-depth instrument: its generic ``record_admitted`` / ``record_admission_resolved``
  admit/resolve tracker (designed for the judge admission funnel) doubles cleanly as a "delivery
  enqueued -> delivery terminal (success or dead-lettered)" tracker keyed by ``event_id``, giving
  ``max_queue_depth()`` / ``max_oldest_pending_age_s()`` for free.
* ``percept.clock.FakeClock`` — deterministic time for the metrics collector; ``WebhookSink``'s own
  internal wall-clock accounting is exercised via a patched ``time.monotonic`` (same technique
  ``tests/test_runtime_sinks.py`` already uses for its ``max_attempt_age_s`` test), so this probe
  never sleeps for real and stays fast + deterministic.

``percept/incidents.py``, ``percept/harness/metrics.py``, and ``percept/clock.py`` are unedited;
this file only observes those existing seams. This is a standalone file (no shared
``__init__``/``conftest`` addition): every helper it needs (a duck-typed ``_Fire``, a fake HTTP
response, a fake flaky ``_send``) is defined locally, mirroring how ``tests/probes/test_p_judge_safe.py``
keeps its own ``_ProbeVision`` self-contained.

**Idempotency ledger (the implementation — tightened from the original A21 scaffold):** the original A21 probe
stood in a MINIMAL local dict-based ledger here because no production "delivery ledger" keyed by
``event_id`` existed yet — a real redelivery-after-restart would call ``WebhookSink.__call__`` again
with the same (persisted/replayed) ``Incident`` object and nothing downstream deduped on ``.id``
before creating a record. That gap is now closed: ``percept.runtime.sinks.IdempotencyLedger`` (a
bounded, restart-surviving ledger keyed by ``event_id`` = ``Incident.id``) and
``percept.runtime.sinks.IdempotentSink`` (the wrapper ``percept.runtime.worker.run()`` installs
around ``on_incident``) are real production code, wired into the delivery path. This file now
imports and drives THOSE classes directly — see ``TestIdempotentRedeliveryAcrossRestart`` below —
instead of a local test double.
"""

from __future__ import annotations

import unittest
from unittest.mock import patch

from percept.clock import FakeClock
from percept.contracts import Incident, Verdict
from percept.harness.metrics import MetricsCollector
from percept.incidents import make_incident
from percept.runtime.sinks import BoundedDeadLetterQueue, IdempotencyLedger, IdempotentSink, WebhookSink

# ---------------------------------------------------------------------------
# Local, self-contained test doubles (nothing outside this file is touched).
# ---------------------------------------------------------------------------


class _Fire:
    """Duck-typed fire stand-in — ``make_incident`` only reads ``.goal_id`` / ``.text`` /
    ``.verdict`` (see ``percept/incidents.py`` docstring: "duck-typed ... this module never
    imports the agent")."""

    def __init__(self, goal_id: str, text: str, verdict: Verdict) -> None:
        self.goal_id = goal_id
        self.text = text
        self.verdict = verdict


class _Resp:
    """Minimal context-manager stand-in for ``http.client.HTTPResponse`` (matches the fake used in
    ``tests/test_runtime_sinks.py``)."""

    status = 200

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def _fire(i: int, *, goal_id: str = "door") -> _Fire:
    return _Fire(goal_id, f"alert {i}", Verdict(True, 0.9, f"evidence {i}"))


# ---------------------------------------------------------------------------
# 1. Capture + judging stay ISOLATED from a permanently-unavailable sink.
# ---------------------------------------------------------------------------


class TestCaptureJudgingIsolatedFromSinkHealth(unittest.TestCase):
    """A dead sink must never block or corrupt capture+judging. This drives a mixed stream of
    repeated FIRE and ASK decisions (the gate's two "something happened" outcomes) through the REAL
    ``make_incident`` producer; only FIRE decisions produce an ``Incident`` and attempt delivery —
    ASK decisions never touch the sink at all. Capture/judging (the per-tick decision count) must
    reach the full tick count regardless of how badly the sink is failing."""

    def test_unavailable_sink_never_blocks_capture_or_judging_across_fire_and_ask_ticks(self) -> None:
        # A realistic judged stream: mostly ASK, with FIRE on the rising edges — repeated, as the
        # contract specifies ("produce repeated FIRE + ASK").
        actions = ["ask", "fire", "ask", "ask", "fire", "fire", "ask", "fire", "ask", "ask"]
        dlq = BoundedDeadLetterQueue(max_records=50)
        sink = WebhookSink(
            "https://hooks.example.test/incidents", retries=1, retry_backoff_s=0, dead_letter=dlq
        )

        capture_judge_count = 0
        delivery_attempts = 0
        with patch("percept.runtime.sinks._send", side_effect=ConnectionError("sink unavailable")):
            for i, action in enumerate(actions):
                # capture+judging: a decision is produced for EVERY tick, fire or ask, unconditionally.
                capture_judge_count += 1
                if action != "fire":
                    continue
                incident = make_incident(_fire(i), ts=float(i))
                delivery_attempts += 1
                try:
                    sink(incident)  # delivery is allowed to fail; it must not propagate into the loop
                except Exception:
                    pass  # the runtime's delivery path catches this — isolation, not silence (see DLQ below)

        self.assertEqual(
            capture_judge_count, len(actions),
            "capture/judging must complete for every tick regardless of sink health",
        )
        fire_count = sum(1 for a in actions if a == "fire")
        self.assertEqual(delivery_attempts, fire_count, "only FIRE ticks enter the delivery path, never ASK")
        self.assertEqual(len(dlq), fire_count, "every failed delivery is dead-lettered — none silently dropped")
        for rec in dlq.read_all():
            self.assertIn(rec["reason"], ("retries_exhausted", "max_attempt_age_exceeded"))


# ---------------------------------------------------------------------------
# 2. Delivery queue + retry age are BOUNDED.
# ---------------------------------------------------------------------------


class TestBoundedDeliveryQueueAndRetryAge(unittest.TestCase):
    def test_dead_letter_queue_stays_bounded_under_sustained_outage(self) -> None:
        dropped_reasons: list[str] = []
        dlq = BoundedDeadLetterQueue(
            max_records=5, on_drop=lambda rec, reason: dropped_reasons.append(reason)
        )
        sink = WebhookSink(
            "https://hooks.example.test/incidents", retries=0, retry_backoff_s=0, dead_letter=dlq
        )
        with patch("percept.runtime.sinks._send", side_effect=ConnectionError("sink down")):
            for i in range(20):
                incident = make_incident(_fire(i), ts=float(i))
                with self.assertRaises(ConnectionError):
                    sink(incident)
        self.assertEqual(len(dlq), 5, "the DLQ never exceeds max_records regardless of sustained failure volume")
        self.assertEqual(len(dropped_reasons), 15, "every eviction beyond the cap is traced, not silent")

    def test_retry_wall_clock_age_is_bounded_even_with_retries_remaining_in_budget(self) -> None:
        # Mirrors tests/test_runtime_sinks.py's max_attempt_age test, driven through THIS probe's
        # fire->Incident->sink path: a "slow" sink that eats 5s of wall clock per attempt must be
        # cut off by max_attempt_age_s well short of the full retries=10 budget.
        dead: list[dict] = []
        clock = [0.0]

        def slow_send(req, timeout):
            clock[0] += 5.0
            raise TimeoutError("slow sink")

        sink = WebhookSink(
            "https://hooks.example.test/incidents", retries=10, retry_backoff_s=0,
            max_attempt_age_s=12.0, dead_letter=dead,
        )
        incident = make_incident(_fire(0), ts=1.0)
        with patch("percept.runtime.sinks._send", slow_send), \
             patch("percept.runtime.sinks.time.monotonic", side_effect=lambda: clock[0]):
            with self.assertRaises(TimeoutError):
                sink(incident)
        self.assertLess(
            dead[0]["attempts"], 11,
            "the wall-clock age bound must cut retries short of the full attempt-count budget",
        )
        self.assertEqual(dead[0]["reason"], "max_attempt_age_exceeded")

    def test_delivery_queue_depth_and_oldest_pending_age_stay_bounded_via_metrics_collector(self) -> None:
        # Reuses percept.harness.metrics.MetricsCollector's generic admit/resolve tracker
        # as the delivery-queue-depth instrument: record_admitted(event_id) when a delivery is
        # enqueued, record_admission_resolved(event_id) the moment it goes terminal (delivered OR
        # dead-lettered) — a delivery is never left "pending" forever.
        clock = FakeClock(0.0)
        collector = MetricsCollector(clock=clock)
        dlq = BoundedDeadLetterQueue(max_records=100)
        sink = WebhookSink(
            "https://hooks.example.test/incidents", retries=0, retry_backoff_s=0, dead_letter=dlq
        )
        with patch("percept.runtime.sinks._send", side_effect=ConnectionError("sink down")):
            for i in range(8):
                incident = make_incident(_fire(i), ts=float(i))
                collector.record_admitted(incident.id, at=clock())
                try:
                    sink(incident)
                except Exception:
                    pass
                collector.record_admission_resolved(incident.id, at=clock())
                clock.advance(1000)  # 1s between fires

        self.assertEqual(
            collector.current_pending_count(), 0,
            "no delivery is left pending forever — every attempt resolves to success or dead-letter",
        )
        self.assertLessEqual(
            collector.max_queue_depth(), 1,
            "deliveries resolve (dead-letter) before the next is admitted — depth stays bounded",
        )
        self.assertEqual(
            collector.max_oldest_pending_age_s(), 0.0,
            "nothing is ever left pending across a clock tick — bounded retry age",
        )


# ---------------------------------------------------------------------------
# 3. Stable idempotency key: duplicate delivery ATTEMPTS (incl. across a simulated restart) reuse
#    event_id and do not create a duplicate logical Incident.
# ---------------------------------------------------------------------------


class TestIdempotentRedeliveryAcrossRestart(unittest.TestCase):
    """Every event carries a stable idempotency key (``event_id`` = ``Incident.id``, per
    ``percept/incidents.py``: "the ``id`` is the stable handle the rest of the system references
    later"). A duplicate delivery ATTEMPT for the SAME logical fire — e.g. redelivered after a
    process restart — must reuse that ``event_id`` and must NOT create a second logical Incident nor
    re-deliver it as a new logical event. the implementation: this now drives the REAL production ledger
    (``percept.runtime.sinks.IdempotencyLedger`` + ``IdempotentSink``) instead of a local test
    double — ``IdempotentSink`` is the exact wrapper ``percept.runtime.worker.run()`` installs around
    ``on_incident``, and ``IdempotencyLedger.snapshot()``/``.restore()`` are the exact hooks ``run()``
    wires through its ``persistence`` autosave/restore path."""

    def test_retry_of_a_failed_attempt_is_not_deduped_reaches_the_sink_again(self) -> None:
        # A retry of an IN-FLIGHT/FAILED attempt (never reached terminal-OK) must still reuse the
        # same event_id and still be DELIVERED — only a terminal-OK delivery is ever remembered, per
        # the contract ("a retry of an in-flight/failed one still reuses the same event_id").
        ledger = IdempotencyLedger()
        calls: list[Incident] = []

        def flaky(inc: Incident) -> str:
            calls.append(inc)
            if len(calls) == 1:
                raise ConnectionError("sink down")
            return "delivered"

        sink = IdempotentSink(flaky, ledger, clock=lambda: 100.0)
        incident = make_incident(_fire(0), ts=5.0)  # ONE fire -> ONE Incident -> ONE stable event_id

        with self.assertRaises(ConnectionError):
            sink(incident)  # attempt 1: fails
        self.assertFalse(
            ledger.is_delivered(incident.id), "a failed attempt must NOT be remembered as terminal-OK"
        )

        result = sink(incident)  # attempt 2: a retry/redelivery of the SAME event_id, sink now healthy
        self.assertEqual(result, "delivered")
        self.assertEqual(len(calls), 2, "a retry of a not-yet-delivered event_id must still reach the sink")
        self.assertTrue(ledger.is_delivered(incident.id), "the successful attempt is now terminal-OK")

    def test_redelivery_after_terminal_ok_is_deduped_sink_not_called_again(self) -> None:
        # Once an event_id reaches terminal-OK, a REDELIVERY of
        # the SAME event_id (e.g. a persisted pending-delivery record replayed post-restart) is
        # deduped Percept-side — the wrapped sink is NOT invoked again, so no duplicate logical
        # Incident reaches a downstream consumer.
        ledger = IdempotencyLedger()
        calls: list[Incident] = []

        def always_ok(inc: Incident) -> str:
            calls.append(inc)
            return "delivered"

        sink = IdempotentSink(always_ok, ledger, clock=lambda: 100.0)
        incident = make_incident(_fire(0), ts=5.0)

        self.assertEqual(sink(incident), "delivered")
        self.assertEqual(len(calls), 1)

        # A second attempt for the SAME event_id (a redelivery) must be a no-op, not a second delivery.
        result = sink(incident)
        self.assertIsNone(result, "a deduped redelivery is a no-op, not a second delivery")
        self.assertEqual(
            len(calls), 1,
            "the underlying sink must NOT be called again for an event_id already delivered to terminal-OK",
        )

    def test_ledger_survives_a_simulated_restart_via_snapshot_restore(self) -> None:
        # the implementation's restart-durability requirement: the SAME redelivery-dedupe decision holds across
        # a process restart, exercised through the exact snapshot()/restore() hooks
        # percept.runtime.worker.run() wires through its `persistence` autosave/restore path — no
        # secrets, no raw media, just [[event_id, ts], ...] (ADR-001 §1 lifecycle).
        ledger_before_restart = IdempotencyLedger()
        sink_before_restart = IdempotentSink(lambda inc: "delivered", ledger_before_restart, clock=lambda: 100.0)
        incident = make_incident(_fire(0), ts=5.0)
        self.assertEqual(sink_before_restart(incident), "delivered")
        self.assertTrue(ledger_before_restart.is_delivered(incident.id))

        # Simulate a RESTART: a FRESH ledger + a FRESH sink instance (as a real process restart would
        # construct), rehydrated ONLY from the persisted snapshot — the runtime's persisted
        # pending-delivery record then hands back the SAME Incident object/id for redelivery.
        persisted_snapshot = ledger_before_restart.snapshot()
        ledger_after_restart = IdempotencyLedger()
        ledger_after_restart.restore(persisted_snapshot)

        calls: list[Incident] = []
        sink_after_restart = IdempotentSink(
            lambda inc: calls.append(inc) or "delivered", ledger_after_restart, clock=lambda: 200.0,
        )
        result = sink_after_restart(incident)  # post-restart redelivery of the SAME event_id
        self.assertIsNone(result, "a redelivery recognized via the RESTORED ledger must still be deduped")
        self.assertEqual(calls, [], "the sink must never be reached — dedupe survives the simulated restart")

    def test_distinct_fires_get_distinct_event_ids_even_with_identical_content_and_timestamp(self) -> None:
        # Guards the OTHER direction: two genuinely distinct fires must never collide on event_id,
        # even if their content and ts happen to match (make_incident's default id includes a random
        # suffix precisely so independent fires can't be mistaken for retries of one another).
        inc_a = make_incident(_fire(0), ts=5.0)
        inc_b = make_incident(_fire(0), ts=5.0)
        self.assertNotEqual(inc_a.id, inc_b.id)


# ---------------------------------------------------------------------------
# 4. Terminal delivery failure is surfaced EXPLICITLY, never silently swallowed.
# ---------------------------------------------------------------------------


class TestTerminalFailureSurfacedExplicitly(unittest.TestCase):
    def test_terminal_failure_dead_letters_with_an_explicit_reason(self) -> None:
        dead: list[dict] = []
        sink = WebhookSink(
            "https://hooks.example.test/incidents", retries=2, retry_backoff_s=0, dead_letter=dead
        )
        incident = make_incident(_fire(0), ts=1.0)
        with patch("percept.runtime.sinks._send", side_effect=ConnectionError("sink unavailable")):
            with self.assertRaises(ConnectionError):
                sink(incident)
        self.assertEqual(len(dead), 1, "a terminal failure MUST produce exactly one dead-letter record")
        self.assertEqual(dead[0]["reason"], "retries_exhausted")
        self.assertEqual(dead[0]["attempts"], 3)  # 1 initial + 2 retries, all exhausted
        # The failure also re-raises rather than being swallowed (assertRaises above) — "surfaced
        # explicitly" means BOTH a traced DLQ record AND a caller-visible signal, never a silent no-op.

    def test_intermittent_failures_that_eventually_succeed_do_not_dead_letter(self) -> None:
        # The complement of the always-down case: an intermittently-failing sink that recovers WITHIN
        # its retry budget must deliver cleanly and never reach the DLQ at all.
        calls = []

        def flaky_send(req, timeout):
            calls.append(req)
            if len(calls) < 3:
                raise TimeoutError("intermittent")
            return _Resp()

        dead: list[dict] = []
        sink = WebhookSink(
            "https://hooks.example.test/incidents", retries=3, retry_backoff_s=0, dead_letter=dead
        )
        incident = make_incident(_fire(0), ts=1.0)
        with patch("percept.runtime.sinks._send", flaky_send):
            sink(incident)  # must not raise — recovers on the 3rd attempt, within budget
        self.assertEqual(len(calls), 3)
        self.assertEqual(dead, [], "a delivery that succeeds within its retry budget is never dead-lettered")


# ---------------------------------------------------------------------------
# the implementation (closed): the production idempotency ledger keyed by Incident.id now lives in
# percept/runtime/sinks.py (IdempotencyLedger + IdempotentSink) and is wired into the delivery path
# in percept/runtime/worker.py (run() wraps on_incident in an IdempotentSink and threads the ledger's
# snapshot()/restore() through the existing `persistence`/`restore`/`autosave_every_s` knobs).
# TestIdempotentRedeliveryAcrossRestart above drives that real production code directly.
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    unittest.main()
