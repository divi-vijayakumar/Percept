"""The shared backend resilience wrapper: breaker telemetry, ``non_retryable``, half-open, attempts.

`percept/backends/_resilience.py` is the single retry/breaker seam every real judge backend wraps its
network call in (anthropic `:594`, gemini `:382`, gemini_live `:184`, openrouter `:344`,
anthropic_text `:195`). A defect here is a defect in every provider at once, and it is paid in real
latency and real spend, so the knobs are pinned as *contracts*, not as descriptions:

* ``non_retryable`` must cost exactly one attempt and must propagate the caller's own exception
  object, so an openrouter `_PermanentProviderError` (a 4xx that will never succeed) is not paid for
  three times — and it must NOT move the breaker, which is a PROVIDER-HEALTH signal and may only be
  moved by evidence about the provider (BUG-C-05-02).
* the breaker must let a recovered provider back in after ``reset_after_s``, as exactly ONE probe,
  and must re-open at once if that probe fails (BUG-C-05-01).
* ``attempts=1`` must mean one attempt and no backoff sleep.

Every bar here is now met by the shipped code; the two characterization pins that encoded the
pre-fix ordering as the contract were rewritten with the fix rather than preserved.
"""

from __future__ import annotations

import asyncio
import time
import unittest
from unittest import mock

from percept.backends._resilience import CircuitBreaker, CircuitOpen, with_retry


class _PermanentProviderError(Exception):
    """Stand-in for openrouter's `_PermanentProviderError` — a 4xx that retrying cannot fix."""


class _PermanentSubclassError(_PermanentProviderError):
    """A narrower permanent error a backend might raise instead of the declared base."""


class _BudgetTerminalError(Exception):
    """Stand-in for openrouter's `_BudgetTerminalError` — raised by our OWN token ledger in
    `begin_attempt()` (`openrouter.py:310`), before a socket is touched. It is the sharpest example
    of an exception that says nothing whatsoever about the provider's health."""


class _TransientProviderError(Exception):
    """A sibling of the permanent error that is NOT in the ``non_retryable`` tuple."""


def _no_sleep():
    """Patch ``asyncio.sleep`` with a recorder so the backoff schedule is observable, not waited on.

    ``_resilience`` holds the module object, so patching the attribute on ``asyncio`` reaches it.
    Returns ``(patcher, slept)`` — ``slept`` is the list of delays with_retry asked for.
    """
    slept: list[float] = []

    async def _record(delay, *args, **kwargs):
        slept.append(delay)

    return mock.patch.object(asyncio, "sleep", _record), slept


class CircuitBreakerTelemetryTests(unittest.IsolatedAsyncioTestCase):
    async def test_snapshot_tracks_state_failures_and_opens(self):
        b = CircuitBreaker(fail_threshold=2, reset_after_s=999.0)
        self.assertEqual(b.snapshot(), {"state": "closed", "failures": 0, "opens": 0})

        b.record_failure()
        self.assertEqual(b.state, "closed")  # below threshold
        b.record_failure()  # trips closed -> open
        snap = b.snapshot()
        self.assertEqual(snap["state"], "open")
        self.assertEqual(snap["opens"], 1)
        self.assertFalse(b.allow())  # open short-circuits

        b.record_success()  # recovery
        self.assertEqual(b.snapshot(), {"state": "closed", "failures": 0, "opens": 1})

    async def test_opens_counts_distinct_trips_not_every_failure(self):
        b = CircuitBreaker(fail_threshold=1, reset_after_s=0.0)
        b.record_failure()  # opens (1)
        self.assertEqual(b.snapshot()["opens"], 1)
        self.assertTrue(b.allow())  # reset_after_s=0 -> half-open immediately
        b.record_failure()  # opens again (2)
        self.assertEqual(b.snapshot()["opens"], 2)


class NonRetryableShortCircuitTests(unittest.IsolatedAsyncioTestCase):
    """§3.8 Resilience gaps — 'non_retryable short-circuit'.

    The failure mode the row names is spend, not correctness: a permanent 4xx retried three times
    triples the latency AND the token bill of a call that could never have succeeded.
    """

    async def test_a_permanent_error_costs_exactly_one_attempt(self):
        calls: list[int] = []
        permanent = _PermanentProviderError("400 invalid_request_error")

        async def call():
            calls.append(1)
            raise permanent

        patcher, slept = _no_sleep()
        with patcher:
            with self.assertRaises(_PermanentProviderError) as caught:
                await with_retry(
                    call,
                    attempts=3,
                    base_delay_s=0.0,
                    non_retryable=(_PermanentProviderError,),
                )

        self.assertEqual(len(calls), 1, "a non_retryable error was paid for more than once")
        # Propagated unchanged: the bare `raise` re-raises the caller's own object, not a wrapper.
        # A backend that keys on `except _PermanentProviderError` upstream depends on this.
        self.assertIs(caught.exception, permanent)
        self.assertEqual(slept, [], "backoff was paid on an error that will never succeed")

    async def test_a_sibling_error_not_in_the_tuple_still_gets_every_attempt(self):
        """The positive control: the short-circuit is scoped to the declared tuple only.

        Also pins the deterministic backoff schedule (0.2 x 2**i, no jitter) that keeps L1
        reproducible, and that the exception which escapes is the LAST attempt's.
        """
        raised: list[_TransientProviderError] = []

        async def call():
            exc = _TransientProviderError(f"attempt {len(raised) + 1}")
            raised.append(exc)
            raise exc

        patcher, slept = _no_sleep()
        with patcher:
            with self.assertRaises(_TransientProviderError) as caught:
                await with_retry(
                    call,
                    attempts=3,
                    base_delay_s=0.1,
                    non_retryable=(_PermanentProviderError,),
                )

        self.assertEqual(len(raised), 3)
        self.assertIs(caught.exception, raised[-1])
        self.assertEqual(slept, [0.1, 0.2], "backoff is 0.1 x 2**i between attempts, and only between")

    async def test_a_subclass_of_a_declared_permanent_error_also_short_circuits(self):
        """Matching is isinstance, not `type(exc) in non_retryable`.

        A backend declares the base class; the SDK raises a narrower subclass. If this ever regressed
        to an identity test the short-circuit would silently stop applying to real provider errors.
        """
        calls: list[int] = []

        async def call():
            calls.append(1)
            raise _PermanentSubclassError("403 permission_error")

        patcher, _slept = _no_sleep()
        with patcher:
            with self.assertRaises(_PermanentSubclassError):
                await with_retry(
                    call,
                    attempts=3,
                    base_delay_s=0.0,
                    non_retryable=(_PermanentProviderError,),
                )
        self.assertEqual(len(calls), 1)

    async def test_a_non_retryable_error_records_no_breaker_failure(self):
        """Ordering contract (BUG-C-05-02): `isinstance(exc, non_retryable)` is checked BEFORE
        `record_failure()`, so a non-retryable error never moves the breaker.

        The breaker is a PROVIDER-HEALTH signal. Both things a backend puts in `non_retryable` are
        non-evidence about provider health: openrouter's `_BudgetTerminalError` is raised by our own
        local token ledger (`openrouter.py:310`) before a socket is touched, and a permanent 4xx is
        proof the provider is UP and the request is bad. Counting either one made the breaker report
        on the wrong subject.

        REWRITTEN from `test_a_permanent_error_still_records_a_breaker_failure_before_re_raising`,
        which pinned the defect (`failures: 1`) as the contract.
        """
        breaker = CircuitBreaker(fail_threshold=5, reset_after_s=30.0)

        async def call():
            raise _PermanentProviderError("400 invalid_request_error")

        patcher, _slept = _no_sleep()
        with patcher:
            with self.assertRaises(_PermanentProviderError):
                await with_retry(
                    call,
                    attempts=3,
                    base_delay_s=0.0,
                    breaker=breaker,
                    non_retryable=(_PermanentProviderError,),
                )

        self.assertEqual(breaker.snapshot(), {"state": "closed", "failures": 0, "opens": 0})

    async def test_a_retryable_error_does_still_record_a_breaker_failure(self):
        """The control for the row above: the reorder scopes the breaker to real provider evidence,
        it does not disconnect the breaker. An error OUTSIDE `non_retryable` is exactly the kind of
        signal the breaker exists for, and each attempt still counts."""
        breaker = CircuitBreaker(fail_threshold=5, reset_after_s=30.0)

        async def call():
            raise _TransientProviderError("503 upstream unavailable")

        patcher, _slept = _no_sleep()
        with patcher:
            with self.assertRaises(_TransientProviderError):
                await with_retry(
                    call,
                    attempts=3,
                    base_delay_s=0.0,
                    breaker=breaker,
                    non_retryable=(_PermanentProviderError,),
                )

        self.assertEqual(breaker.snapshot(), {"state": "closed", "failures": 3, "opens": 0})

    async def test_permanent_errors_alone_never_open_the_shipped_breaker(self):
        """The second-order consequence, asserted as the intended trade (BUG-C-05-02).

        A provider answering a permanent error on EVERY call — a stale key, a 401 — will now never
        open the breaker, so there is no backoff and no short-circuit. That is correct: each call
        costs exactly ONE attempt, so there is no retry storm to protect against, and the caller
        receives the real exception naming the actual problem every time. Before the fix the sixth
        call raised `CircuitOpen` instead, which was one attempt cheaper and destroyed the reason.

        REWRITTEN from `test_the_consequence_permanent_errors_alone_open_the_shipped_breaker`, which
        pinned the sixth-call `CircuitOpen` as the contract.
        """
        breaker = CircuitBreaker()  # shipped defaults: fail_threshold=5, reset_after_s=30.0
        self.assertEqual(breaker.fail_threshold, 5)
        self.assertEqual(breaker.reset_after_s, 30.0)

        calls: list[int] = []
        raised: list[Exception] = []

        async def call():
            calls.append(1)
            exc = _PermanentProviderError("400 invalid_request_error")
            raised.append(exc)
            raise exc

        patcher, slept = _no_sleep()
        with patcher:
            for _ in range(6):  # one past the old fail_threshold, where CircuitOpen used to appear
                with self.assertRaises(_PermanentProviderError) as caught:
                    await with_retry(
                        call,
                        attempts=3,
                        base_delay_s=0.0,
                        breaker=breaker,
                        non_retryable=(_PermanentProviderError,),
                    )
                # The provider's own exception, every time — never a CircuitOpen standing in for it.
                self.assertIs(caught.exception, raised[-1])

        self.assertEqual(len(calls), 6, "one attempt per call, six calls — no retry storm")
        self.assertEqual(slept, [], "no backoff is paid on an error that can never succeed")
        self.assertEqual(breaker.snapshot(), {"state": "closed", "failures": 0, "opens": 0})

    async def test_a_locally_raised_refusal_never_blames_the_provider(self):
        """The bug's motivating case, stated in its own terms.

        `_BudgetTerminalError` is raised by our token ledger before any socket is touched. Five of
        them used to open the shared breaker, and from the sixth call `with_retry` raised
        `CircuitOpen` — so a run that hit its own budget ceiling reported a dead provider. The
        ledger's own exception must reach the caller on every call, unchanged.
        """
        breaker = CircuitBreaker()  # shipped defaults

        async def call():
            raise _BudgetTerminalError("request budget exhausted before the attempt")

        patcher, _slept = _no_sleep()
        with patcher:
            for _ in range(6):
                with self.assertRaises(_BudgetTerminalError):
                    await with_retry(
                        call,
                        attempts=3,
                        base_delay_s=0.0,
                        breaker=breaker,
                        non_retryable=(_BudgetTerminalError, _PermanentProviderError),
                    )
        self.assertEqual(breaker.state, "closed", "a local budget refusal opened the breaker")


class HalfOpenTransitionTests(unittest.IsolatedAsyncioTestCase):
    """§3.8 Resilience gaps — 'Half-open transition after `reset_after_s`'.

    Spec contract: `CircuitBreaker` is documented as `fail_threshold=5, reset_after_s=30,
    half-open single probe`, and `allow()` carries the comment "half-open: let one through".
    """

    @staticmethod
    def _fake_clock(breaker: CircuitBreaker, start: float = 1_000.0) -> dict:
        """Drive `_now` from a dict so the transition is exercised without wall-clock waiting."""
        clock = {"t": start}
        breaker._now = lambda: clock["t"]  # type: ignore[method-assign]
        return clock

    async def test_a_recovered_provider_is_readmitted_once_reset_after_s_has_elapsed(self):
        """The row's stated failure mode: 'A recovered provider stays circuit-open forever'.

        Uses the real running-loop clock (`_now` reads `asyncio.get_event_loop().time()`), so this
        also pins that the breaker is driven by the loop clock and not by a never-advancing source.
        """
        breaker = CircuitBreaker(fail_threshold=2, reset_after_s=0.02)
        calls: list[int] = []

        async def call():
            calls.append(1)
            return "ok"

        breaker.record_failure()
        breaker.record_failure()
        self.assertEqual(breaker.state, "open")

        with self.assertRaises(CircuitOpen):
            await with_retry(call, attempts=3, base_delay_s=0.0, breaker=breaker)
        self.assertEqual(calls, [], "an open breaker must not reach the provider")

        await asyncio.sleep(0.05)  # > reset_after_s on the running loop's clock

        self.assertEqual(await with_retry(call, attempts=3, base_delay_s=0.0, breaker=breaker), "ok")
        self.assertEqual(len(calls), 1)
        self.assertEqual(breaker.snapshot(), {"state": "closed", "failures": 0, "opens": 1})

    async def test_the_transition_needs_the_full_reset_window_not_a_partial_one(self):
        """Boundary: strictly-less-than `reset_after_s` stays open; exactly `reset_after_s` admits.

        The comparison is `>=`, so the window is inclusive at its far edge.
        """
        breaker = CircuitBreaker(fail_threshold=1, reset_after_s=30.0)
        clock = self._fake_clock(breaker)

        breaker.record_failure()
        self.assertEqual(breaker.state, "open")

        clock["t"] += 29.999
        self.assertFalse(breaker.allow(), "readmitted before the reset window elapsed")

        clock["t"] += 0.001  # exactly reset_after_s since it opened
        self.assertTrue(breaker.allow())

    async def test_exactly_one_probe_is_admitted_at_the_half_open_transition(self):
        """The spec's contract is a SINGLE probe; `allow()` says "half-open: let one through".

        A dead-but-restarting provider must receive one call, not the whole backlog. (Fixed in
        BUG-C-05-01: the transition used to set `_opened_at = None`, fully re-closing the breaker,
        so every subsequent caller was admitted too.)
        """
        breaker = CircuitBreaker(fail_threshold=3, reset_after_s=30.0)
        clock = self._fake_clock(breaker)

        for _ in range(3):
            breaker.record_failure()
        self.assertFalse(breaker.allow())

        clock["t"] += 30.0
        self.assertTrue(breaker.allow(), "the probe must be admitted")
        self.assertEqual(breaker.state, "half_open")

        # No outcome has been recorded for the probe yet, so the breaker still knows nothing new
        # about the provider. Everyone behind the probe must keep waiting.
        self.assertFalse(breaker.allow(), "a second caller was admitted alongside the probe")
        self.assertFalse(breaker.allow(), "a third caller was admitted alongside the probe")

    async def test_a_failed_probe_re_opens_the_breaker_immediately(self):
        """A probe that fails proves the provider is still dead — the breaker must re-open at once.

        Re-opening is unconditional on `fail_threshold`: the probe existed precisely to answer this
        question and it answered it. (Fixed in BUG-C-05-01: the transition used to zero `_failures`,
        so a still-dead provider was handed `fail_threshold` more calls before tripping again.)
        """
        breaker = CircuitBreaker(fail_threshold=3, reset_after_s=30.0)
        clock = self._fake_clock(breaker)

        for _ in range(3):
            breaker.record_failure()
        clock["t"] += 30.0
        self.assertTrue(breaker.allow())  # the probe goes out

        breaker.record_failure()  # ...and the provider is still dead

        self.assertEqual(breaker.state, "open", "a failed probe left the breaker closed")
        self.assertFalse(breaker.allow())
        self.assertEqual(breaker.snapshot()["opens"], 2)

    async def test_a_successful_probe_closes_the_breaker_for_everyone(self):
        """The other half of the transition: one good probe re-admits the whole backlog at once.

        Half-open is a question, not a rate limit — once it is answered the breaker stops holding
        anyone out, and the failure history that opened it is discarded.
        """
        breaker = CircuitBreaker(fail_threshold=3, reset_after_s=30.0)
        clock = self._fake_clock(breaker)

        for _ in range(3):
            breaker.record_failure()
        clock["t"] += 30.0
        self.assertTrue(breaker.allow())
        self.assertFalse(breaker.allow())  # held out while the probe is in flight

        breaker.record_success()

        self.assertEqual(breaker.snapshot(), {"state": "closed", "failures": 0, "opens": 1})
        self.assertTrue(breaker.allow())
        self.assertTrue(breaker.allow())

    async def test_an_abandoned_probe_does_not_wedge_the_breaker_forever(self):
        """`_probe_in_flight` is released in a `finally`, NOT on the success/failure paths.

        A caller that neither returns nor raises through `with_retry` — a cancellation is the real
        case — records no outcome, so the breaker learns nothing. If the reservation leaked with it
        the breaker would sit half-open holding a probe that is never coming back, and NO call would
        ever be admitted again: strictly worse than the open state it came from.
        """
        breaker = CircuitBreaker(fail_threshold=1, reset_after_s=30.0)
        clock = self._fake_clock(breaker)
        breaker.record_failure()
        clock["t"] += 30.0

        started = asyncio.Event()

        async def wedged():
            started.set()
            await asyncio.Event().wait()  # never completes

        task = asyncio.create_task(with_retry(wedged, attempts=3, base_delay_s=0.0, breaker=breaker))
        await started.wait()
        self.assertEqual(breaker.state, "half_open")
        self.assertFalse(breaker.allow(), "the probe reservation was not taken")

        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task

        # No outcome was recorded, so the breaker is still half-open — but it is willing to lend the
        # probe out again rather than refusing every caller for the rest of the process.
        self.assertEqual(breaker.state, "half_open")
        self.assertTrue(breaker.allow(), "the breaker wedged half-open on an abandoned probe")

    async def test_a_probe_that_ends_in_non_evidence_leaves_the_breaker_still_asking(self):
        """Where the two fixes compose. A half-open probe that dies on a `non_retryable` error
        recorded neither a success nor a failure — by design, since neither is evidence about the
        provider — so the breaker must stay HALF-OPEN rather than guess. It keeps lending the slot
        one call at a time until something real answers the question.
        """
        breaker = CircuitBreaker(fail_threshold=1, reset_after_s=30.0)
        clock = self._fake_clock(breaker)
        breaker.record_failure()
        clock["t"] += 30.0

        async def call():
            raise _BudgetTerminalError("request budget exhausted before the attempt")

        patcher, _slept = _no_sleep()
        with patcher:
            for _ in range(3):
                with self.assertRaises(_BudgetTerminalError):
                    await with_retry(
                        call,
                        attempts=3,
                        base_delay_s=0.0,
                        breaker=breaker,
                        non_retryable=(_BudgetTerminalError,),
                    )
                # Never re-opened (that would blame the provider) and never closed (nothing proved
                # it healthy) — and never wedged, since the next caller still gets the slot.
                self.assertEqual(breaker.state, "half_open")
        self.assertEqual(breaker.snapshot()["opens"], 1)


class AttemptsBoundTests(unittest.IsolatedAsyncioTestCase):
    """§3.8 Resilience gaps — '`attempts=1`'. Degenerate config must not silently retry anyway."""

    async def test_attempts_one_calls_once_and_re_raises_without_backoff(self):
        calls: list[int] = []
        failure = _TransientProviderError("503 upstream unavailable")

        async def call():
            calls.append(1)
            raise failure

        patcher, slept = _no_sleep()
        started = time.monotonic()
        with patcher:
            with self.assertRaises(_TransientProviderError) as caught:
                # base_delay_s is deliberately large: if a backoff were paid the elapsed guard below
                # catches it even if the sleep recorder above were somehow bypassed.
                await with_retry(call, attempts=1, base_delay_s=5.0)
        elapsed = time.monotonic() - started

        self.assertEqual(len(calls), 1, "attempts=1 made more than one provider call")
        self.assertIs(caught.exception, failure)
        self.assertEqual(slept, [], "attempts=1 paid a backoff it can never use")
        self.assertLess(elapsed, 1.0)

    async def test_attempts_one_records_exactly_one_breaker_failure(self):
        breaker = CircuitBreaker(fail_threshold=3, reset_after_s=30.0)

        async def call():
            raise _TransientProviderError("503 upstream unavailable")

        patcher, _slept = _no_sleep()
        with patcher:
            with self.assertRaises(_TransientProviderError):
                await with_retry(call, attempts=1, base_delay_s=0.0, breaker=breaker)

        # Not 3: a degenerate attempts=1 must not burn the breaker budget three times faster.
        self.assertEqual(breaker.snapshot(), {"state": "closed", "failures": 1, "opens": 0})

    async def test_attempts_one_success_returns_and_records_a_breaker_success(self):
        breaker = CircuitBreaker(fail_threshold=3, reset_after_s=30.0)
        breaker.record_failure()
        breaker.record_failure()
        self.assertEqual(breaker.snapshot()["failures"], 2)  # one short of opening

        calls: list[int] = []

        async def call():
            calls.append(1)
            return "verdict"

        patcher, slept = _no_sleep()
        with patcher:
            self.assertEqual(await with_retry(call, attempts=1, base_delay_s=5.0, breaker=breaker), "verdict")

        self.assertEqual(len(calls), 1)
        self.assertEqual(slept, [])
        self.assertEqual(breaker.snapshot(), {"state": "closed", "failures": 0, "opens": 0})

    async def test_attempts_zero_never_calls_the_provider_and_raises_no_provider_error(self):
        """Documented side of the other degenerate config, so the boundary is not a surprise.

        The spec says nothing about `attempts=0`. The loop body never runs, so `last_exc` is None and
        the `assert last_exc is not None` guard fires: the caller gets an `AssertionError` naming
        nothing, and zero spend. Pinned so a change here is deliberate.
        """
        calls: list[int] = []

        async def call():
            calls.append(1)
            return "never reached"

        with self.assertRaises(AssertionError):
            await with_retry(call, attempts=0, base_delay_s=0.0)
        self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()
