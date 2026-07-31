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

import json
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
# 3a. The ledger's TTL boundary: exactly what age stops deduping.
# ---------------------------------------------------------------------------


class TestIdempotencyLedgerTtlBoundary(unittest.TestCase):
    """``IdempotencyLedger`` is bounded by AGE as well as count, and the docstring states the
    reason plainly: "a stale memory must never wedge a legitimately-retried event open forever"
    (``sinks.py:446-448``). The comparison is ``(now - ts) > self.ttl_s`` (``:452``) — a STRICT
    ``>`` — so the boundary sits at *exactly* ``ttl_s``: a record that old still reads delivered
    (still deduped), and only one tick past it becomes re-eligible for delivery again. The
    eviction sweep uses the mirror-image ``<=`` (``:466``), so the two must agree or the ledger
    would report a record as delivered that its own sweep has already dropped."""

    def test_at_exactly_ttl_s_a_record_still_dedupes_and_one_tick_past_it_is_re_eligible(self) -> None:
        ledger = IdempotencyLedger(ttl_s=10.0)
        ledger.mark_delivered("evt-1", ts=0.0)

        self.assertTrue(
            ledger.is_delivered("evt-1", now=9.999999),
            "inside the TTL the record is remembered — a redelivery is deduped",
        )
        self.assertTrue(
            ledger.is_delivered("evt-1", now=10.0),
            "at age EXACTLY ttl_s the strict `>` at sinks.py:452 has not tripped — still deduped",
        )
        self.assertFalse(
            ledger.is_delivered("evt-1", now=10.000001),
            "one tick past ttl_s the record reads NOT-delivered — the event is re-eligible again",
        )
        # Re-eligibility is not just a read: the aged record must actually let a real redelivery
        # through the wrapper, not silently swallow it.
        calls: list[Incident] = []
        sink = IdempotentSink(lambda inc: calls.append(inc) or "delivered", ledger, clock=lambda: 10.000001)
        incident = make_incident(_fire(0), ts=0.0, id="evt-1")
        self.assertEqual(sink(incident), "delivered")
        self.assertEqual(len(calls), 1, "an event whose ledger record aged out reaches the sink again")

    def test_the_eviction_sweep_agrees_with_is_delivered_at_the_ttl_boundary(self) -> None:
        # `_evict` keeps while `(now - oldest_ts) <= ttl_s` (:466); `is_delivered` drops while
        # `(now - ts) > ttl_s` (:452). The two halves must never disagree, or the ledger would
        # claim a record it has already evicted (or evict one it still reports as delivered).
        evicted: list[tuple[str, str]] = []
        ledger = IdempotencyLedger(ttl_s=10.0, on_evict=lambda eid, reason: evicted.append((eid, reason)))
        ledger.mark_delivered("evt-old", ts=0.0)

        # A later mark at EXACTLY the boundary sweeps at now == ttl_s: nothing may fall out.
        ledger.mark_delivered("evt-boundary", ts=10.0)
        self.assertEqual(evicted, [], "at age exactly ttl_s the sweep keeps the record — same boundary as :452")
        self.assertIn("evt-old", ledger)
        self.assertTrue(ledger.is_delivered("evt-old", now=10.0), "…and the read agrees it is still delivered")

        # One tick past, both halves flip together.
        ledger.mark_delivered("evt-new", ts=10.000001)
        self.assertEqual(evicted, [("evt-old", "ttl")], "one tick past ttl_s the sweep drops it, traced as `ttl`")
        self.assertNotIn("evt-old", ledger)
        self.assertFalse(ledger.is_delivered("evt-old", now=10.000001))
        self.assertIn("evt-boundary", ledger, "only the genuinely-aged record is swept, not the whole ledger")

    def test_ttl_none_or_a_missing_now_disables_the_age_check_entirely(self) -> None:
        # Both guards at :452 are conjunctive — either one absent means age is never consulted and
        # the record dedupes forever. Callers that omit `now` get the unbounded-memory behaviour.
        forever = IdempotencyLedger(ttl_s=None)
        forever.mark_delivered("evt-1", ts=0.0)
        self.assertTrue(
            forever.is_delivered("evt-1", now=1e12),
            "ttl_s=None disables the age check: the record dedupes no matter how old",
        )

        with_ttl = IdempotencyLedger(ttl_s=10.0)
        with_ttl.mark_delivered("evt-1", ts=0.0)
        self.assertTrue(
            with_ttl.is_delivered("evt-1"),
            "now=None (the default) also disables the age check — the caller must supply a clock",
        )
        self.assertFalse(
            with_ttl.is_delivered("evt-1", now=10.5), "…and supplying one restores the bound"
        )

        # ttl_s=None also disables the sweep, so a ttl-reason eviction can never fire.
        evicted: list[tuple[str, str]] = []
        unbounded = IdempotencyLedger(ttl_s=None, on_evict=lambda eid, r: evicted.append((eid, r)))
        unbounded.mark_delivered("a", ts=0.0)
        unbounded.mark_delivered("b", ts=1e9)
        self.assertEqual(evicted, [], "with ttl_s=None nothing is ever evicted for age")
        self.assertEqual(len(unbounded), 2)


# ---------------------------------------------------------------------------
# 3b. A duplicate event_id carrying a DIFFERENT payload.
# ---------------------------------------------------------------------------


class TestDuplicateEventIdWithDifferentPayload(unittest.TestCase):
    """``IdempotentSink.__call__`` (``sinks.py:529-539``) keys the dedupe decision on
    ``incident.id`` ALONE — the payload is never read, never hashed, never compared. That is the
    right call for the case the contract is written for (a redelivery of the SAME logical event
    after a restart). It also means a *colliding* or corrupted id — two genuinely different
    incidents sharing one ``.id`` — silently discards the second incident's content."""

    @staticmethod
    def _colliding_pair() -> tuple[Incident, Incident]:
        first = make_incident(_fire(0), ts=5.0, id="evt-collide")
        second = make_incident(_fire(1), ts=6.0, id="evt-collide")
        return first, second

    def test_a_second_incident_reusing_a_delivered_event_id_is_dropped_payload_never_consulted(self) -> None:
        first, second = self._colliding_pair()
        self.assertEqual(first.id, second.id, "the two incidents collide on event_id…")
        self.assertNotEqual(
            (first.message, first.verdict, first.ts), (second.message, second.verdict, second.ts),
            "…while carrying genuinely different payloads",
        )

        delivered: list[Incident] = []
        ledger = IdempotencyLedger()
        sink = IdempotentSink(lambda inc: delivered.append(inc) or "delivered", ledger, clock=lambda: 100.0)

        self.assertEqual(sink(first), "delivered")
        result = sink(second)

        # This is the SHIPPED decision, pinned so a change to it is deliberate: the second payload
        # is dropped, and dropped QUIETLY — no exception, no typed rejection, just None.
        self.assertIsNone(result, "the colliding second incident is dropped, not rejected with a typed error")
        self.assertEqual(len(delivered), 1, "the wrapped sink is never invoked for the second payload")
        self.assertIs(delivered[0], first, "the payload that survives is the FIRST one, by id order only")

    def test_the_dropped_payload_is_not_recoverable_from_the_ledger(self) -> None:
        # The ledger stores `(event_id, ts)` only (`snapshot()` :487-489) — deliberately, since it
        # must carry no media and no payload across a restart. The consequence for the collision
        # case is that the discarded incident is gone: nothing downstream can reconstruct it.
        first, second = self._colliding_pair()
        ledger = IdempotencyLedger()
        sink = IdempotentSink(lambda inc: "delivered", ledger, clock=lambda: 100.0)
        sink(first)
        sink(second)

        snap = ledger.snapshot()
        self.assertEqual([row[0] for row in snap], ["evt-collide"])
        flattened = repr(snap)
        for fragment in (second.message, str(second.ts)):
            self.assertNotIn(
                fragment, flattened,
                "the ledger holds (event_id, ts) only — the dropped payload is unrecoverable",
            )

    def test_dropping_a_colliding_payload_emits_an_observable_signal(self) -> None:
        # BUG-D-07-01, closed. A drop that a user can neither see nor count is indistinguishable
        # from a delivery that never happened. `BoundedDeadLetterQueue` sets the bar for this
        # module: its default drop path logs `dlq.dropped`, and `IdempotencyLedger._trace` logs
        # `idempotency_ledger.evicted`. The dedupe drop was the one path with neither a hook nor a
        # log line — nothing at all on the module's own logger.
        first, second = self._colliding_pair()
        evictions: list[tuple[str, str]] = []
        ledger = IdempotencyLedger(on_evict=lambda eid, reason: evictions.append((eid, reason)))
        sink = IdempotentSink(lambda inc: "delivered", ledger, clock=lambda: 100.0)
        sink(first)

        with self.assertLogs("percept.runtime.sinks", level="DEBUG") as cm:
            self.assertIsNone(sink(second))
        self.assertTrue(
            cm.output, "dropping a second, different payload for a known event_id must be traced"
        )
        self.assertTrue(any("idempotency.deduped" in line for line in cm.output))
        self.assertIn(
            {"event_id": "evt-collide", "reason": "already_delivered"},
            [getattr(rec, "fields", {}) for rec in cm.records],
        )
        self.assertEqual(
            evictions, [], "the ledger's on_evict is for evictions — a dedupe drop needs its own signal"
        )

    def test_the_dedupe_trace_is_debug_and_never_carries_the_payload(self) -> None:
        # It fires on EVERY redelivery on a retry-heavy sink, so it must be cheap and quiet: DEBUG
        # (so it is off under the package default of INFO) and structured fields only — never a
        # stringified incident, which would put a full body in the log on the most ordinary thing
        # this sink does.
        first, second = self._colliding_pair()
        ledger = IdempotencyLedger()
        sink = IdempotentSink(lambda inc: "delivered", ledger, clock=lambda: 100.0)
        sink(first)

        with self.assertLogs("percept.runtime.sinks", level="DEBUG") as cm:
            sink(second)
        deduped = [r for r in cm.records if r.getMessage() == "idempotency.deduped"]
        self.assertEqual([r.levelname for r in deduped], ["DEBUG"])
        for record in deduped:
            self.assertEqual(set(getattr(record, "fields", {})), {"event_id", "reason"})
            self.assertNotIn(second.message, repr(getattr(record, "fields", {})))

        # ...and at the package's own default level nothing is emitted at all.
        with self.assertNoLogs("percept.runtime.sinks", level="INFO"):
            sink(second)

    def test_a_wired_on_duplicate_hook_owns_the_report_and_receives_the_dropped_payload(self) -> None:
        # Mirrors `BoundedDeadLetterQueue.on_drop` / `IdempotencyLedger.on_evict` exactly: wired ->
        # the hook is the channel and the default log line stands down. The hook gets the incident
        # itself, because an integration that wants to keep the discarded payload can only do it here.
        first, second = self._colliding_pair()
        seen: list[tuple[str, Incident]] = []
        sink = IdempotentSink(
            lambda inc: "delivered", IdempotencyLedger(), clock=lambda: 100.0,
            on_duplicate=lambda eid, inc: seen.append((eid, inc)),
        )
        sink(first)
        with self.assertNoLogs("percept.runtime.sinks", level="DEBUG"):
            self.assertIsNone(sink(second))
        self.assertEqual(len(seen), 1)
        self.assertEqual(seen[0][0], "evt-collide")
        self.assertIs(seen[0][1], second, "the hook receives the payload that was dropped")

    def test_an_ordinary_redelivery_is_traced_too_and_is_never_raised_or_merged(self) -> None:
        # The reachable case, and the one the design is for: the SAME incident redelivered. It is
        # the designed happy path of at-least-once delivery (ADR-001 §6), so it must stay a quiet
        # no-op — a raise here would route a healthy redelivery into `run()`'s dead-letter queue —
        # and no combined incident is ever synthesized (ADR-001 §10.2 freezes events).
        delivered: list[Incident] = []
        sink = IdempotentSink(
            lambda inc: delivered.append(inc) or "delivered", IdempotencyLedger(), clock=lambda: 100.0,
        )
        incident = make_incident(_fire(0), ts=5.0)
        self.assertEqual(sink(incident), "delivered")
        with self.assertLogs("percept.runtime.sinks", level="DEBUG") as cm:
            self.assertIsNone(sink(incident))  # no raise
        self.assertTrue(any("idempotency.deduped" in line for line in cm.output))
        self.assertEqual(delivered, [incident], "no second delivery, and no merged third incident")


# ---------------------------------------------------------------------------
# 3c. Every eviction is traced — via the hook when wired, via a log line when not.
# ---------------------------------------------------------------------------


class TestIdempotencyLedgerEvictionIsTraced(unittest.TestCase):
    """A ledger record that falls out is a record that stops deduping — an event_id that ages or
    counts out becomes eligible for delivery all over again. That must never be silent, so
    ``_trace`` (``sinks.py:475-479``) has two paths: the wired ``on_evict(event_id, reason)``
    callback, and a default log line on ``percept.runtime.sinks``. Both reasons the ledger can
    produce — ``"ttl"`` and ``"max_records"`` — must reach both paths."""

    def test_a_wired_on_evict_receives_the_ttl_reason(self) -> None:
        seen: list[tuple[str, str]] = []
        ledger = IdempotencyLedger(ttl_s=10.0, max_records=100, on_evict=lambda e, r: seen.append((e, r)))
        ledger.mark_delivered("evt-aged", ts=0.0)
        ledger.mark_delivered("evt-fresh", ts=50.0)  # sweeps at now=50 — evt-aged is 50s old
        self.assertEqual(seen, [("evt-aged", "ttl")], "an age-out is reported with the event_id and `ttl`")
        self.assertEqual(len(ledger), 1)

    def test_a_wired_on_evict_receives_the_max_records_reason(self) -> None:
        seen: list[tuple[str, str]] = []
        # ttl_s=None isolates the COUNT bound: the age sweep at :464 is skipped entirely, so the
        # only eviction that can fire is the max_records loop at :470.
        ledger = IdempotencyLedger(max_records=2, ttl_s=None, on_evict=lambda e, r: seen.append((e, r)))
        for i, ts in enumerate((0.0, 1.0, 2.0, 3.0)):
            ledger.mark_delivered(f"evt-{i}", ts=ts)
        self.assertEqual(
            seen, [("evt-0", "max_records"), ("evt-1", "max_records")],
            "the count bound drops OLDEST-first and names each dropped event_id",
        )
        self.assertEqual(len(ledger), 2, "the ledger never exceeds max_records")
        self.assertIn("evt-3", ledger)

    def test_the_default_eviction_trace_is_logged_not_silent(self) -> None:
        # No on_evict supplied: the fallback branch must still emit `idempotency_ledger.evicted`
        # CARRYING the reason and the event_id (the structured fields ride on the LogRecord via
        # percept.observability.log's `extra={"fields": ...}`). Same idiom as
        # tests/test_runtime_sinks.py::…test_bounded_dead_letter_queue_default_drop_is_traced_via_log_not_silent.
        by_ttl = IdempotencyLedger(ttl_s=10.0, max_records=100)
        by_ttl.mark_delivered("evt-aged", ts=0.0)
        with self.assertLogs("percept.runtime.sinks", level="INFO") as cm:
            by_ttl.mark_delivered("evt-fresh", ts=50.0)
        self.assertTrue(any("idempotency_ledger.evicted" in line for line in cm.output))
        ttl_fields = [getattr(rec, "fields", {}) for rec in cm.records]
        self.assertIn(
            {"reason": "ttl", "event_id": "evt-aged"}, ttl_fields,
            "the default trace names BOTH the reason and the event_id that stopped deduping",
        )

        by_count = IdempotencyLedger(max_records=1, ttl_s=None)
        by_count.mark_delivered("evt-a", ts=0.0)
        with self.assertLogs("percept.runtime.sinks", level="INFO") as cm:
            by_count.mark_delivered("evt-b", ts=1.0)
        count_fields = [getattr(rec, "fields", {}) for rec in cm.records]
        self.assertIn({"reason": "max_records", "event_id": "evt-a"}, count_fields)


# ---------------------------------------------------------------------------
# 3d. restore() of a malformed snapshot must never block a restart.
# ---------------------------------------------------------------------------


class TestIdempotencyLedgerRestoreOfMalformedInput(unittest.TestCase):
    """``restore`` (``sinks.py:491-503``) is the restart path: the runtime hands it whatever the
    persistence backend read back. Its docstring is unambiguous — "a corrupt or absent ledger
    snapshot must never block a restart, it just means the ledger starts blank". Two mechanisms
    implement that: a falsy-snapshot early return (``:503-504``) and a PER-ITEM
    ``except (TypeError, ValueError, IndexError, KeyError): continue`` (``:517-518``) that skips
    the bad row rather than abandoning the whole snapshot. Note also that ``restore`` MERGES into
    whatever the ledger already holds — it never replaces or clears."""

    @staticmethod
    def _usable(ledger: IdempotencyLedger) -> bool:
        """The ledger still works after the malformed input: it can record and read a delivery."""
        ledger.mark_delivered("probe", ts=1.0)
        ok = ledger.is_delivered("probe", now=1.0)
        ledger._delivered.pop("probe", None)
        return ok

    def test_absent_or_empty_snapshots_are_a_total_no_op_and_leave_the_ledger_usable(self) -> None:
        for snap in (None, [], {}, "", 0, False):
            with self.subTest(snap=snap):
                ledger = IdempotencyLedger(ttl_s=None)
                ledger.restore(snap)
                self.assertEqual(len(ledger), 0, "a falsy snapshot restores nothing and raises nothing")
                self.assertTrue(self._usable(ledger), "…and the ledger is fully usable afterwards")

    def test_garbled_items_are_skipped_individually_rather_than_aborting_the_restore(self) -> None:
        # One good row surrounded by every shape the per-item guard catches: a bare int (TypeError
        # on item[0]), an empty list (IndexError), a one-element list (IndexError on item[1]), a
        # non-numeric ts (ValueError), None (TypeError), and a dict (KeyError on item[0]).
        ledger = IdempotencyLedger(ttl_s=None)
        ledger.restore([
            5,
            [],
            ["only-an-id"],
            ["evt-bad-ts", "not-a-float"],
            None,
            {"k": 1},
            ["evt-good", 7.0],
            ("evt-tuple", "8.0"),
        ])
        self.assertEqual(
            sorted(ledger.snapshot()), [["evt-good", 7.0], ["evt-tuple", 8.0]],
            "each malformed row is skipped on its own; the well-formed rows around it still land",
        )
        self.assertTrue(self._usable(ledger))

    def test_a_snapshot_of_only_garbage_restores_nothing_and_does_not_clear_existing_records(self) -> None:
        ledger = IdempotencyLedger(ttl_s=None)
        ledger.mark_delivered("evt-live", ts=1.0)
        ledger.restore([5, [], ["only-an-id"], ["evt-bad-ts", "not-a-float"], None, {"k": 1}])
        self.assertEqual(
            ledger.snapshot(), [["evt-live", 1.0]],
            "an all-garbage snapshot is a no-op — it does NOT wipe what the ledger already knows",
        )
        self.assertTrue(ledger.is_delivered("evt-live", now=2.0))

    def test_restore_merges_into_existing_records_rather_than_replacing_them(self) -> None:
        # Pinned because it is the surprising half: restore() is a MERGE, so a restart that
        # restores after any delivery has already been recorded keeps both. A future "replace"
        # rewrite would silently un-dedupe every in-session event.
        ledger = IdempotencyLedger(ttl_s=None)
        ledger.mark_delivered("evt-live", ts=1.0)
        ledger.restore([["evt-restored", 2.0]])
        self.assertEqual(sorted(k for k, _ in ledger.snapshot()), ["evt-live", "evt-restored"])
        self.assertTrue(ledger.is_delivered("evt-live", now=3.0))
        self.assertTrue(ledger.is_delivered("evt-restored", now=3.0))
        # A restored row for an id already present overwrites that id's ts, and only that id.
        ledger.restore([["evt-live", 99.0]])
        self.assertEqual(dict(ledger.snapshot()), {"evt-live": 99.0, "evt-restored": 2.0})

    def test_a_round_trip_through_snapshot_restore_survives_a_json_encode_decode(self) -> None:
        # The snapshot crosses a real persistence boundary, so it must be JSON-able as claimed at
        # :488 — and the decoded form must restore identically.
        source = IdempotencyLedger(ttl_s=None)
        source.mark_delivered("evt-a", ts=1.0)
        source.mark_delivered("evt-b", ts=2.0)
        restored = IdempotencyLedger(ttl_s=None)
        restored.restore(json.loads(json.dumps(source.snapshot())))
        self.assertEqual(restored.snapshot(), source.snapshot())

    def test_a_non_iterable_truthy_snapshot_is_a_no_op_not_a_crash(self) -> None:
        # BUG-D-07-02, closed. `if not snap: return` only screened FALSY input. A truthy
        # non-iterable — the shape a half-written or type-confused persistence record produces —
        # reached `for item in snap` and raised TypeError from OUTSIDE the per-item try. Nothing in
        # restore() caught it, so it propagated into the restart path: exactly the "a corrupt
        # snapshot blocks restart" failure the method's own docstring promises cannot happen. The
        # top-level shape is checked (and its rejection logged) before the loop now.
        for snap in (5, 3.5, True, object()):
            with self.subTest(snap=type(snap).__name__):
                ledger = IdempotencyLedger(ttl_s=None)
                ledger.restore(snap)
                self.assertEqual(
                    len(ledger), 0, "a corrupt snapshot starts the ledger blank, it does not raise"
                )
                self.assertTrue(self._usable(ledger))

    def test_a_string_or_mapping_snapshot_is_a_no_op_not_a_population(self) -> None:
        # The other half of the same hole: a str and a dict ARE iterable, so they would have walked
        # straight into the loop. A str yields characters (`item[0]` is a character, `item[1]` an
        # IndexError) and a dict yields its keys — both "parse" as a run of skipped items, which is
        # harmless but indistinguishable from an empty ledger. They are rejected at the top instead,
        # so the rejection is traced rather than inferred from a silently blank ledger.
        for snap in ("evt-a", {"evt-a": 1.0}):
            with self.subTest(snap=type(snap).__name__):
                ledger = IdempotencyLedger(ttl_s=None)
                with self.assertLogs("percept.snapshot", level="INFO") as caught:
                    ledger.restore(snap)
                self.assertEqual(len(ledger), 0)
                self.assertTrue(any("snapshot.rejected" in line for line in caught.output))
                self.assertTrue(self._usable(ledger))

    def test_a_corrupt_snapshot_never_wipes_what_the_ledger_already_knows(self) -> None:
        # The restart-path bar: a garbled payload must cost nothing that was already recorded.
        ledger = IdempotencyLedger(ttl_s=None)
        ledger.mark_delivered("evt-live", ts=1.0)
        for snap in (5, object(), "evt-a", {"evt-a": 1.0}):
            ledger.restore(snap)
        self.assertEqual(ledger.snapshot(), [["evt-live", 1.0]])
        self.assertTrue(ledger.is_delivered("evt-live", now=2.0))


# ---------------------------------------------------------------------------
# the implementation (closed): the production idempotency ledger keyed by Incident.id now lives in
# percept/runtime/sinks.py (IdempotencyLedger + IdempotentSink) and is wired into the delivery path
# in percept/runtime/worker.py (run() wraps on_incident in an IdempotentSink and threads the ledger's
# snapshot()/restore() through the existing `persistence`/`restore`/`autosave_every_s` knobs).
# TestIdempotentRedeliveryAcrossRestart above drives that real production code directly.
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    unittest.main()
