"""Delivery and observation must not sit on the perception critical path.

The defect these tests close: notification sinks were awaited INLINE in the frame loop
(``percept/runtime/worker.py`` ``_deliver_materialized``, the ``on_frame`` hook, every ``on_health``
emission) and ``WebhookSink.__call__`` was fully synchronous — a blocking ``urllib`` POST plus a
``time.sleep`` retry backoff — executed on the event-loop thread. One unreachable subscriber
therefore stalled gate evaluation, the audio pump and decode consumption, and the bounded capture
queue then shed frames on its behalf under a generic ``frames_dropped`` counter that named no cause.

NO WALL-CLOCK TIMING IS ASSERTED ANYWHERE HERE. Cadence is measured with two instruments that a
loaded CI box cannot perturb:

* ``_CountingSource.pulled`` — the worker pulls exactly one frame per tick, so this IS the tick
  counter, and
* a bounded **scheduler-turn budget** (``asyncio.sleep(0)`` turns) standing in for elapsed time.

A stalled sink that really blocked the loop exhausts the budget and fails the assertion; it never
turns into a flaky timing test that gets skipped a month later.
"""
from __future__ import annotations

import ast
import asyncio
import inspect
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from percept import Goal, Percept
from percept.adapters import Verdict
from percept.backends.fake import FakeVision
from percept.edge.tier0 import gray_frame
from percept.incidents import make_incident
from percept.runtime import RuntimeStats, run
from percept.runtime.sinks import BoundedDeadLetterQueue, UrlPolicy, WebhookSink
from percept.runtime.worker import (
    _DELIVERY_QUEUE_MAX,
    _DeliveryItem,
    _DeliverySpool,
    _sink_dead_letter,
    _sink_name,
)

W, H = 8, 8
STILL = gray_frame(W, H, 0)
MOVED = gray_frame(W, H, 255)

# One rising edge (STILL->MOVED) then a long quiet tail: exactly one Incident, then many more ticks
# the loop must keep taking while that Incident's delivery is stuck.
QUIET_TAIL = 18
FRAMES = (
    [(0.0, STILL, "clear"), (1.0, MOVED, "intruder")]
    + [(2.0 + i, STILL, "clear") for i in range(QUIET_TAIL)]
)

# Bounded stand-in for elapsed time. Generous enough that a healthy loop finishes the stream inside
# it, finite so a genuinely stalled loop FAILS instead of hanging the suite.
TURN_BUDGET = 200_000


def _core() -> Percept:
    vision = FakeVision(
        script={
            "intruder": Verdict(satisfied=True, confidence=0.95, evidence="a person is in the zone"),
            "clear": Verdict(satisfied=False, confidence=0.95, evidence="the zone is empty"),
        }
    )
    core = Percept.create(vision=vision, discover_plugins=False)
    core.add_goal(Goal(id="zone", condition="a person is in the restricted zone", say="Alert"))
    return core


class _CountingSource:
    """A real async source that counts the frames the frame loop has PULLED.

    The worker pulls one frame per tick, so ``pulled`` is the loop's tick counter — the instrumented
    counter every cadence assertion below reads, in place of a clock.
    """

    def __init__(self, frames) -> None:
        self._frames = list(frames)
        self.pulled = 0

    def __aiter__(self):
        async def _gen():
            for item in self._frames:
                await asyncio.sleep(0)  # cooperate, exactly as a real capture adapter does
                self.pulled += 1
                yield item

        return _gen()


async def _turns_until(predicate, *, budget: int = TURN_BUDGET) -> bool:
    """Spend up to ``budget`` scheduler turns waiting for ``predicate``. True if it came true.

    This is the virtual clock: one ``asyncio.sleep(0)`` is one tick of it. A loop wedged behind a
    sink cannot advance ``predicate`` no matter how many turns it is given, so exhausting the budget
    is a real failure signal rather than a slow machine.
    """
    for _ in range(budget):
        if predicate():
            return True
        await asyncio.sleep(0)
    return False


def _incident(i: int = 0):
    core = _core()
    goal = core.goals.get("zone")

    class _Fire:
        action = "fire"
        goal_id = "zone"
        concern_id = "zone"
        message = f"Alert {i}"
        verdict = Verdict(satisfied=True, confidence=0.9, evidence="seen")
        judgment = None
        entity_id = None
        goal = None

    fire = _Fire()
    fire.goal = goal
    return make_incident(fire, ts=float(i))


class FrameLoopCadenceUnderAStalledSinkTests(unittest.TestCase):
    """Acceptance 1 + 2: a stalled subscriber must not cost the loop a single tick."""

    def test_the_loop_takes_every_remaining_tick_while_an_incident_sink_is_stalled(self):
        async def scenario():
            source = _CountingSource(FRAMES)
            release = asyncio.Event()
            entered_at: list[int] = []

            async def stalled_incident_sink(_incident) -> None:
                entered_at.append(source.pulled)
                await release.wait()

            async def watchdog() -> bool:
                # The loop must reach the END of the stream while the sink above is still parked.
                advanced = await _turns_until(
                    lambda: bool(entered_at) and source.pulled >= len(FRAMES)
                )
                release.set()  # always release, so a failure reports rather than hangs
                return advanced

            task = asyncio.create_task(watchdog())
            stats = RuntimeStats()
            incidents = [
                incident
                async for incident in run(
                    _core(),
                    source=source,
                    on_incident=stalled_incident_sink,
                    stats=stats,
                )
            ]
            return await task, source, stats, incidents, entered_at

        advanced, source, stats, incidents, entered_at = asyncio.run(scenario())

        self.assertTrue(
            advanced,
            "the frame loop stopped advancing while an incident sink was stalled — delivery is "
            "still on the perception critical path",
        )
        self.assertEqual(len(incidents), 1)
        self.assertEqual(len(entered_at), 1)
        # The stall really did overlap the stream: the sink was entered with ticks still to come.
        self.assertLess(entered_at[0], len(FRAMES))
        self.assertEqual(
            len(FRAMES) - entered_at[0],
            source.pulled - entered_at[0],
            "every tick after the sink stalled was still taken",
        )
        self.assertEqual(stats.frames, len(FRAMES))
        self.assertEqual(stats.frames_dropped, 0)
        self.assertEqual(stats.deliveries_dropped, 0)

    def test_a_stalled_frame_sink_costs_the_loop_no_ticks_either(self):
        # on_frame is the per-frame subscriber (the live camera panel); it fires on EVERY tick, so a
        # stalled one used to halve the capture rate outright.
        async def scenario():
            source = _CountingSource(FRAMES)
            release = asyncio.Event()
            entered_at: list[int] = []

            async def stalled_frame_sink(_t, _frame) -> None:
                entered_at.append(source.pulled)
                await release.wait()

            async def watchdog() -> bool:
                advanced = await _turns_until(lambda: source.pulled >= len(FRAMES))
                release.set()
                return advanced

            task = asyncio.create_task(watchdog())
            stats = RuntimeStats()
            [
                incident
                async for incident in run(
                    _core(), source=source, on_frame=stalled_frame_sink, stats=stats
                )
            ]
            return await task, source, stats, entered_at

        advanced, source, stats, entered_at = asyncio.run(scenario())
        self.assertTrue(advanced, "a stalled on_frame sink still stalls the capture loop")
        self.assertEqual(entered_at[0], 1, "the very first frame's delivery is what stalled")
        self.assertEqual(source.pulled, len(FRAMES))
        self.assertEqual(stats.frames, len(FRAMES))

    def test_a_two_second_sink_costs_no_ticks_and_no_capture_queue_drops(self):
        """Acceptance 2, with a deliberately slow (>=2s) sink.

        The 2s is the STIMULUS, never the assertion: everything asserted is a counter. The sink is
        entered on tick 2 and is still inside its 2s sleep when the assertion about ticks is made.
        """

        async def scenario():
            source = _CountingSource(FRAMES)
            entered_at: list[int] = []
            finished_at: list[int] = []

            async def slow_sink(_incident) -> None:
                entered_at.append(source.pulled)
                await asyncio.sleep(2.0)
                finished_at.append(source.pulled)

            stats = RuntimeStats()
            incidents = [
                incident
                async for incident in run(
                    _core(), source=source, on_incident=slow_sink, stats=stats
                )
            ]
            return source, stats, incidents, entered_at, finished_at

        source, stats, incidents, entered_at, finished_at = asyncio.run(scenario())
        self.assertEqual(len(incidents), 1)
        self.assertEqual(stats.frames, len(FRAMES))
        self.assertEqual(
            stats.frames_dropped, 0, "a slow subscriber must not make the capture queue shed frames"
        )
        self.assertEqual(stats.deliveries_dropped, 0)
        # The whole stream was consumed BEFORE the 2s delivery came back.
        self.assertEqual(finished_at, [len(FRAMES)])
        self.assertLess(entered_at[0], len(FRAMES))

    def test_a_sink_that_does_not_suspend_still_runs_before_the_loop_advances(self):
        # The no-regression half. Decoupling delivery must not silently defer the ordinary sink by a
        # tick: a sink that never actually suspends still runs on the frame that produced it.
        async def scenario():
            source = _CountingSource(FRAMES)
            seen: list[int] = []

            def fast_sink(_incident) -> None:
                seen.append(source.pulled)

            incidents = [
                incident
                async for incident in run(_core(), source=source, on_incident=fast_sink)
            ]
            return seen, incidents

        seen, incidents = asyncio.run(scenario())
        self.assertEqual(len(incidents), 1)
        # 3, not 2: the worker prefetches the next envelope while it works the current one
        # (``_owner_driven_video``), so ``pulled`` runs exactly one ahead of the tick being handled.
        # The point is that it is the FIRING tick and not ``len(FRAMES)`` — the sink was not deferred.
        self.assertEqual(seen, [3], "the sink ran on the tick that produced the incident, not later")


class WebhookSinkIsOffTheEventLoopThreadTests(unittest.TestCase):
    """Acceptance 5: no ``time.sleep`` — indeed no blocking at all — on the loop thread."""

    def setUp(self):
        p = patch(
            "percept.runtime.sinks.socket.getaddrinfo", side_effect=OSError("no DNS in tests")
        )
        p.start()
        self.addCleanup(p.stop)

    def test_the_blocking_post_runs_on_a_worker_thread_and_the_loop_keeps_turning(self):
        loop_thread: list[int] = []
        send_thread: list[int] = []
        proceed = threading.Event()
        turns = [0]

        class _Resp:
            status = 200

            def __enter__(self):
                return self

            def __exit__(self, *_a):
                return False

        def blocking_send(_req, timeout):
            send_thread.append(threading.get_ident())
            proceed.wait()  # blocks its thread until the loop has demonstrably kept turning
            return _Resp()

        async def scenario():
            loop_thread.append(threading.get_ident())
            sink = WebhookSink("https://example.test/hook", timeout_s=1.0)
            with patch("percept.runtime.sinks._send", blocking_send):
                result = sink(_incident())
                self.assertTrue(
                    inspect.isawaitable(result),
                    "inside a running loop the webhook must hand back an awaitable, not block",
                )
                delivery = asyncio.ensure_future(result)

                async def ticker():
                    # The instrumented loop counter: real turns of the event loop, taken while the
                    # POST is blocked on its own thread.
                    while not delivery.done():
                        turns[0] += 1
                        if turns[0] > 500:
                            proceed.set()
                        await asyncio.sleep(0)

                await asyncio.gather(ticker(), delivery)

        asyncio.run(scenario())
        self.assertTrue(send_thread, "the webhook never reached its transport")
        self.assertNotEqual(
            send_thread[0],
            loop_thread[0],
            "the blocking webhook POST ran on the event-loop thread",
        )
        self.assertGreater(
            turns[0], 500, "the event loop did not keep turning while the webhook was blocked"
        )

    def test_retry_backoff_sleeps_on_the_worker_thread_not_the_loop_thread(self):
        loop_thread: list[int] = []
        sleep_threads: list[int] = []
        real_sleep = __import__("time").sleep

        def recording_sleep(seconds):
            sleep_threads.append(threading.get_ident())
            real_sleep(0)

        async def scenario():
            loop_thread.append(threading.get_ident())
            dead: list[dict] = []
            sink = WebhookSink(
                "https://example.test/hook", retries=2, retry_backoff_s=0.01, dead_letter=dead
            )
            with (
                patch("percept.runtime.sinks._send", side_effect=TimeoutError("down")),
                patch("percept.runtime.sinks.time.sleep", recording_sleep),
            ):
                with self.assertRaises(TimeoutError):
                    await sink(_incident())
            return dead

        dead = asyncio.run(scenario())
        self.assertEqual(len(sleep_threads), 2, "both backoffs ran")
        for ident in sleep_threads:
            self.assertNotEqual(
                ident, loop_thread[0], "the retry backoff slept on the event-loop thread"
            )
        # ...and the existing terminal-failure contract is untouched by the move.
        self.assertEqual(len(dead), 1)
        self.assertEqual(dead[0]["reason"], "retries_exhausted")
        self.assertEqual(dead[0]["attempts"], 3)

    def test_without_a_running_loop_the_sink_is_still_fully_synchronous(self):
        # The other half of the contract: a caller on its own thread still gets today's blocking
        # call and today's raise — no event loop is required to use a WebhookSink.
        dead: list[dict] = []
        sink = WebhookSink(
            "https://example.test/hook", retries=0, retry_backoff_s=0, dead_letter=dead
        )
        with patch("percept.runtime.sinks._send", side_effect=TimeoutError("down")):
            with self.assertRaises(TimeoutError):
                self.assertIsNone(sink(_incident()))
        self.assertEqual(len(dead), 1)

    def test_no_time_sleep_in_sinks_is_reachable_from_the_event_loop_thread(self):
        """Acceptance 5, statically: every ``time.sleep`` in sinks.py is inside the one method that
        is only ever entered through ``asyncio.to_thread``."""
        path = Path(inspect.getsourcefile(WebhookSink))
        self.assertEqual(path.name, "sinks.py")
        tree = ast.parse(path.read_text(encoding="utf-8"))

        enclosing: dict[int, str] = {}
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for child in ast.walk(node):
                    enclosing.setdefault(id(child), node.name)

        sleeps = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "sleep"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "time"
        ]
        self.assertEqual(len(sleeps), 1, "sinks.py grew another time.sleep — re-audit its thread")
        self.assertEqual(enclosing[id(sleeps[0])], "_deliver_blocking")

        # ...and _deliver_blocking is CALLED from exactly two places, both inside __call__: the
        # to_thread hand-off, and the no-running-loop fallback. Anything else would be a new way
        # onto the loop thread.
        callers = {
            enclosing[id(node)]
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "_deliver_blocking"
        }
        to_thread_args = [
            node.args[0].attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "to_thread"
            and node.args
            and isinstance(node.args[0], ast.Attribute)
        ]
        self.assertEqual(callers, {"__call__"})
        self.assertEqual(to_thread_args, ["_deliver_blocking"])

    def test_a_webhook_sink_does_not_stall_the_frame_loop(self):
        """The two halves together, through the real runtime: a dead webhook, a live loop."""
        proceed = threading.Event()

        def blocking_send(_req, _timeout):
            proceed.wait()
            raise ConnectionError("subscriber is down")

        async def scenario():
            source = _CountingSource(FRAMES)
            dead = BoundedDeadLetterQueue(max_records=10)
            sink = WebhookSink(
                "http://127.0.0.1:9/hook",
                retries=0,
                retry_backoff_s=0,
                dead_letter=dead,
                url_policy=UrlPolicy.dev(),
            )

            async def watchdog() -> bool:
                advanced = await _turns_until(lambda: source.pulled >= len(FRAMES))
                proceed.set()
                return advanced

            task = asyncio.create_task(watchdog())
            stats = RuntimeStats()
            [
                incident
                async for incident in run(
                    _core(),
                    source=source,
                    on_incident=sink,
                    sink_errors="ignore",
                    stats=stats,
                )
            ]
            return await task, source, stats, dead

        advanced, source, stats, dead = asyncio.run(scenario())
        self.assertTrue(advanced, "a dead webhook still stalls the frame loop")
        self.assertEqual(source.pulled, len(FRAMES))
        self.assertEqual(stats.frames, len(FRAMES))
        self.assertEqual(stats.frames_dropped, 0)
        # The failure is still surfaced, not swallowed: DLQ record + counted sink error.
        self.assertEqual(len(dead), 1)
        self.assertEqual(dead.read_all()[0]["reason"], "retries_exhausted")
        self.assertEqual(stats.sink_errors, 1)


class BoundedDeliveryQueueTests(unittest.TestCase):
    """Acceptance 3 + 4: drop-oldest, a typed receipt naming the sink, and no vanished Incident."""

    def _spool(self, *, maxsize, dlq=None, receipts=None):
        received = receipts if receipts is not None else []
        return _DeliverySpool(
            on_health=received.append, stats=RuntimeStats(), dead_letter=dlq, maxsize=maxsize
        ), received

    def _item(self, spool, incident, *, channel="incident", sink="webhook_sink"):
        async def _never(_payload):
            await asyncio.Event().wait()

        return _DeliveryItem(
            fn=_never,
            args=(incident,),
            channel=channel,
            sink=sink,
            errors="ignore",
            health_kind="incident_sink_error",
            incident=incident if channel == "incident" else None,
        )

    def test_the_default_bound_matches_the_capture_queue(self):
        # Sized to a structure, not tuned to a metric: one capture queue's worth of pending work.
        self.assertEqual(_DELIVERY_QUEUE_MAX, inspect.signature(run).parameters["source_queue_size"].default)
        self.assertEqual(
            inspect.signature(run).parameters["delivery_queue_size"].default, _DELIVERY_QUEUE_MAX
        )

    def test_drop_oldest_emits_a_typed_receipt_naming_the_sink(self):
        async def scenario():
            dlq = BoundedDeadLetterQueue(max_records=50)
            spool, receipts = self._spool(maxsize=2, dlq=dlq)
            incidents = [_incident(i) for i in range(5)]
            for incident in incidents:
                await spool.deliver(self._item(spool, incident))
            # let the lane emit its queued receipts without waiting on the (never-returning) sink
            for _ in range(50):
                await asyncio.sleep(0)
            return spool, receipts, dlq, incidents

        spool, receipts, dlq, incidents = asyncio.run(scenario())

        # 5 offered: 1 entered inline and suspended, 2 fit the queue, 2 evicted the oldest.
        self.assertEqual(spool.dropped, 2)
        drops = [e for e in receipts if e.kind == "delivery_dropped"]
        self.assertEqual(len(drops), 2)
        for event in drops:
            self.assertEqual(event.data["sink"], "webhook_sink")  # NAMES THE SINK
            self.assertEqual(event.data["channel"], "incident")
            self.assertEqual(event.data["reason"], "delivery_queue_overflow")
            self.assertEqual(event.data["queue_max"], 2)
            self.assertTrue(event.data["dead_lettered"])
        self.assertEqual([e.data["dropped_total"] for e in drops], [1, 2])
        # Drop-OLDEST: the two earliest queued deliveries went, the newest survived.
        self.assertEqual(
            [e.data["incident_id"] for e in drops], [incidents[1].id, incidents[2].id]
        )

    def test_a_dropped_incident_is_dead_lettered_never_merely_counted(self):
        async def scenario():
            dlq = BoundedDeadLetterQueue(max_records=50)
            spool, receipts = self._spool(maxsize=1, dlq=dlq)
            incidents = [_incident(i) for i in range(4)]
            for incident in incidents:
                await spool.deliver(self._item(spool, incident))
            for _ in range(50):
                await asyncio.sleep(0)
            return spool, dlq, incidents

        spool, dlq, incidents = asyncio.run(scenario())
        records = dlq.read_all()
        self.assertEqual(len(records), spool.dropped)
        self.assertEqual(len(records), 2)
        for record in records:
            self.assertEqual(record["reason"], "delivery_queue_overflow")
            self.assertEqual(record["channel"], "incident")
            self.assertEqual(record["sink"], "webhook_sink")
            self.assertEqual(record["attempts"], 0)
        self.assertEqual(
            [r["incident_id"] for r in records], [incidents[1].id, incidents[2].id]
        )

    def test_a_dropped_frame_is_receipted_but_not_dead_lettered(self):
        # The deliberate asymmetry: a frame or a health event is best-effort by nature, an Incident
        # is the product. Only the product gets a dead-letter record.
        async def scenario():
            dlq = BoundedDeadLetterQueue(max_records=50)
            spool, receipts = self._spool(maxsize=1, dlq=dlq)
            for i in range(4):
                await spool.deliver(
                    self._item(spool, _incident(i), channel="frame", sink="panel_feed")
                )
            for _ in range(50):
                await asyncio.sleep(0)
            return spool, receipts, dlq

        spool, receipts, dlq = asyncio.run(scenario())
        drops = [e for e in receipts if e.kind == "delivery_dropped"]
        self.assertEqual(len(drops), 2)
        self.assertEqual(len(dlq), 0)
        for event in drops:
            self.assertEqual(event.data["sink"], "panel_feed")
            self.assertFalse(event.data["dead_lettered"])

    def test_the_receipt_names_the_users_sink_through_the_delivery_wrappers(self):
        from percept.runtime.sinks import IdempotencyLedger, IdempotentSink, SupersessionSink

        def my_webhook(_incident):
            return None

        wrapped = SupersessionSink(IdempotentSink(my_webhook, IdempotencyLedger()))
        self.assertEqual(_sink_name(wrapped), "my_webhook")

        webhook = WebhookSink("https://example.test/hook", dead_letter=[])
        self.assertEqual(_sink_name(IdempotentSink(webhook, IdempotencyLedger())), "WebhookSink")
        self.assertIsNotNone(_sink_dead_letter(IdempotentSink(webhook, IdempotencyLedger())))
        self.assertIsNone(_sink_dead_letter(my_webhook))

    def test_run_dead_letters_every_incident_the_bound_evicts(self):
        """The invariant end to end through the real ``run()``: receipts and DLQ records reconcile."""

        async def scenario():
            source = _CountingSource(FRAMES)
            release = asyncio.Event()
            dlq: list[dict] = []
            events = []

            async def stalled(_payload) -> None:
                await release.wait()

            async def watchdog() -> bool:
                advanced = await _turns_until(lambda: source.pulled >= len(FRAMES))
                release.set()
                return advanced

            task = asyncio.create_task(watchdog())
            stats = RuntimeStats()
            [
                incident
                async for incident in run(
                    _core(),
                    source=source,
                    on_incident=stalled,
                    on_frame=lambda t, f: None,
                    on_health=events.append,
                    delivery_queue_size=2,
                    delivery_dead_letter=dlq,
                    stats=stats,
                )
            ]
            return await task, stats, events, dlq

        advanced, stats, events, dlq = asyncio.run(scenario())
        self.assertTrue(advanced)
        drops = [e for e in events if e.kind == "delivery_dropped"]
        self.assertGreater(stats.deliveries_dropped, 0, "a 2-deep lane behind a stalled sink must overflow")
        self.assertEqual(len(drops), stats.deliveries_dropped, "every drop is receipted")
        incident_drops = [e for e in drops if e.data["channel"] == "incident"]
        self.assertEqual(
            len(dlq),
            len(incident_drops),
            "every evicted Incident has a dead-letter record — none vanished",
        )
        for event in drops:
            self.assertTrue(event.data["sink"], "a receipt must name the sink it dropped for")

    def test_a_raising_sink_still_takes_the_run_down_under_sink_errors_raise(self):
        # Fail-loud survives the move off the critical path: it is latched by the lane and re-raised
        # on the frame loop, so it still routes through the worker's runtime_error path.
        def broken(_incident):
            raise RuntimeError("sink down")

        events = []

        async def scenario():
            return [
                incident
                async for incident in run(
                    _core(),
                    source=_CountingSource(FRAMES),
                    on_incident=broken,
                    on_health=events.append,
                    sink_errors="raise",
                )
            ]

        with self.assertRaises(RuntimeError):
            asyncio.run(scenario())
        self.assertTrue([e for e in events if e.kind == "incident_sink_error"])
        self.assertTrue([e for e in events if e.kind == "runtime_error"])

    def test_every_queued_delivery_is_flushed_before_the_run_reports_done(self):
        # Decoupling must not lose deliveries at end-of-source: a subscriber sees exactly what an
        # inline-awaited run showed it.
        seen: list = []

        async def slow_but_finite(incident):
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            seen.append(incident.id)

        async def scenario():
            return [
                incident
                async for incident in run(
                    _core(), source=_CountingSource(FRAMES), on_incident=slow_but_finite
                )
            ]

        incidents = asyncio.run(scenario())
        self.assertEqual(seen, [incident.id for incident in incidents])

    def test_the_delivery_lane_rejects_a_nonsense_bound(self):
        with self.assertRaises(ValueError):
            _DeliverySpool(on_health=None, maxsize=0)
        with self.assertRaises(ValueError):
            _DeliverySpool(on_health=None, maxsize=True)


class ObservationDispatchIsSchedulableTests(unittest.TestCase):
    """The third seam: the observation provider must not be awaited inline on the frame loop."""

    def test_the_default_is_still_inline_and_the_knob_is_validated(self):
        # Scheduling the detector re-prices the single-flight judge lane's supersession, which the
        # occurrence fold turns into a different fire count on the shipped native-video fixture. So
        # the mechanism ships enabled-on-request and the default stays byte-for-byte. See run()'s
        # docstring for the numbers and whose decision the default is.
        self.assertEqual(
            inspect.signature(run).parameters["observation_dispatch"].default, "inline"
        )
        async def bad_knob():
            return [
                incident
                async for incident in run(
                    _core(),
                    source=_CountingSource(FRAMES),
                    observation_dispatch="eventually",
                )
            ]

        with self.assertRaises(ValueError):
            asyncio.run(bad_knob())

    def test_a_slow_observation_provider_costs_the_loop_no_ticks(self):
        # `_prepare_split_jobs` used to `await core.dispatch_observation_candidate(candidate)` on the
        # frame loop even under judge_dispatch="async" — the DETECTOR call, a second provider round
        # trip, sitting on the critical path in the very mode whose whole promise is that provider
        # latency does not.
        async def scenario():
            core = _core()
            source = _CountingSource(FRAMES)
            release = asyncio.Event()
            entered_at: list[int] = []
            original = core.vision.detect

            async def slow_detect(*args, **kwargs):
                entered_at.append(source.pulled)
                await release.wait()
                return await original(*args, **kwargs)

            core.vision.detect = slow_detect

            async def watchdog() -> bool:
                advanced = await _turns_until(
                    lambda: bool(entered_at) and source.pulled >= len(FRAMES)
                )
                release.set()
                return advanced

            task = asyncio.create_task(watchdog())
            stats = RuntimeStats()
            incidents = [
                incident
                async for incident in run(
                    core,
                    source=source,
                    judge_dispatch="async",
                    observation_dispatch="async",
                    stats=stats,
                )
            ]
            return await task, source, stats, incidents, entered_at

        advanced, source, stats, incidents, entered_at = asyncio.run(scenario())
        self.assertTrue(
            advanced, "a slow provider still holds the frame loop under judge_dispatch='async'"
        )
        self.assertLess(entered_at[0], len(FRAMES))
        self.assertEqual(source.pulled, len(FRAMES))
        self.assertEqual(stats.frames, len(FRAMES))
        self.assertEqual(len(incidents), 1)


if __name__ == "__main__":
    unittest.main()
