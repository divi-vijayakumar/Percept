"""Retry + backoff + a tiny circuit breaker for outbound backend calls.

Deterministic backoff (no jitter RNG so L1 stays reproducible). Real backends wrap their
network calls with :func:`with_retry`; fakes never need it.

Two invariants hold this seam together, and both are about keeping the breaker HONEST rather than
merely defensive: the breaker is a three-state machine whose reset window lends out exactly ONE
probe (:class:`CircuitBreaker`), and only evidence ABOUT THE PROVIDER is allowed to move it
(:func:`with_retry`). Everything else here is mechanism.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Awaitable, Callable, TypeVar

T = TypeVar("T")


class CircuitOpen(RuntimeError):
    """Raised when the breaker is open and a call is short-circuited."""


@dataclass
class CircuitBreaker:
    """A three-state breaker: ``closed`` -> ``open`` -> ``half_open`` -> ``closed``/``open``.

    HALF-OPEN IS THE WHOLE POINT OF THE RESET WINDOW. When ``reset_after_s`` elapses the breaker
    does not know the provider recovered — it knows only that it is worth ASKING. So it admits
    exactly ONE probe and holds everyone else out until that probe reports back:

    * ``allow()`` in ``half_open`` with a probe already in flight returns ``False``. Without this,
      the window's expiry re-admits the entire backlog at once against a provider that may still be
      dead — unbounded admission, not a probe.
    * the transition does NOT zero ``_failures`` and does NOT clear ``_opened_at``. The breaker is
      still open; it is lending out one call. Zeroing the count means a FAILED probe starts again
      from one and needs ``fail_threshold`` fresh failures to re-open, which is a
      ``fail_threshold``-times multiplier on wasted spend against a provider already proven dead.
    * a failure recorded while half-open re-opens IMMEDIATELY, regardless of threshold: the probe
      existed precisely to answer that question, and it answered it.

    ``_probe_in_flight`` is released by :func:`with_retry` in a ``finally``, never on the
    success/failure paths — a caller that neither returns nor raises through the wrapper (a
    cancellation, a ``GeneratorExit``) would otherwise wedge the breaker half-open with a probe that
    is never coming back, and no call would ever be admitted again.
    """

    fail_threshold: int = 5
    reset_after_s: float = 30.0
    _failures: int = field(default=0, init=False)
    _opened_at: float | None = field(default=None, init=False)
    _opens: int = field(default=0, init=False)  # telemetry: total closed->open transitions
    _half_open: bool = field(default=False, init=False)
    _probe_in_flight: bool = field(default=False, init=False)

    def _now(self) -> float:
        return asyncio.get_event_loop().time()

    def allow(self) -> bool:
        if self._half_open:
            # One probe at a time. Until it reports back the breaker knows nothing new.
            if self._probe_in_flight:
                return False
            self._probe_in_flight = True
            return True
        if self._opened_at is None:
            return True
        if self._now() - self._opened_at >= self.reset_after_s:
            # Half-open: lend out exactly one call. Still open, still counting — the failure
            # history that opened it stays intact so a failed probe re-opens on the spot.
            self._half_open = True
            self._probe_in_flight = True
            return True
        return False

    def record_success(self) -> None:
        self._failures = 0
        self._opened_at = None
        self._half_open = False
        self._probe_in_flight = False

    def record_failure(self) -> None:
        if self._half_open:
            # The probe proved the provider is still dead. Re-open at once — the threshold is for
            # deciding whether a provider is unhealthy, not for re-deciding what a probe just showed.
            self._half_open = False
            self._probe_in_flight = False
            self._opened_at = self._now()
            self._opens += 1
            return
        self._failures += 1
        if self._failures >= self.fail_threshold and self._opened_at is None:
            self._opened_at = self._now()
            self._opens += 1

    def release_probe(self) -> None:
        """Drop the half-open probe reservation without deciding anything about the provider.

        Called from :func:`with_retry`'s ``finally`` so an abandoned probe (cancelled, or an
        exception path that never reaches record_*) cannot wedge the breaker half-open forever. The
        breaker stays half-open and simply becomes willing to lend out another probe."""
        self._probe_in_flight = False

    @property
    def state(self) -> str:
        """``"open"`` (short-circuiting), ``"half_open"`` (one probe admitted, everyone else held
        out) or ``"closed"`` (passing calls through)."""
        if self._half_open:
            return "half_open"
        return "open" if self._opened_at is not None else "closed"

    def snapshot(self) -> dict:
        """Telemetry surface — current state, the live failure count, and total opens.

        ``state`` is one of ``closed`` / ``open`` / ``half_open``. ``failures`` deliberately survives
        the half-open transition, so a snapshot taken during a probe still shows the history that
        opened the breaker rather than a misleading zero."""
        return {"state": self.state, "failures": self._failures, "opens": self._opens}


async def with_retry(
    call: Callable[[], Awaitable[T]],
    *,
    attempts: int = 3,
    base_delay_s: float = 0.2,
    breaker: CircuitBreaker | None = None,
    timeout_s: float | None = None,
    non_retryable: tuple[type[BaseException], ...] = (),
) -> T:
    """Call ``call`` with exponential backoff; honor an optional circuit breaker.

    ``timeout_s`` (opt) is a HARD per-attempt wall-clock bound via :func:`asyncio.wait_for`. It exists
    because a provider client's own request timeout does not always bound a WEDGED socket send/connect
    (observed: a live judge call hung for hours inside ``sock_send``). With it set, a stuck attempt is
    cancelled to ``TimeoutError`` and retried/failed like any other error — the run can never hang.
    ``None`` = byte-for-byte the historical behavior (no extra bound).

    THE BREAKER IS A PROVIDER-HEALTH SIGNAL, SO ONLY EVIDENCE ABOUT THE PROVIDER MAY MOVE IT.
    ``non_retryable`` is therefore checked BEFORE ``record_failure()``, not after. The two kinds of
    exception a backend puts in that tuple are both non-evidence:

    * a LOCALLY raised refusal never touched a socket — openrouter's ``_BudgetTerminalError`` comes
      out of our own token ledger declining to spend (``openrouter.py:310``, inside
      ``begin_attempt()``, before the request is built). Counting five of those opened the shared
      breaker, and from the sixth call this wrapper short-circuited with ``CircuitOpen`` INSTEAD of
      the ledger's own exception — destroying the real reason and blaming a provider that was never
      contacted.
    * a permanent 4xx is evidence the provider is UP and the REQUEST is bad. A breaker that opens on
      it takes a healthy provider offline because we sent it something malformed.

    Second-order consequence, and it is the correct trade: a provider that returns a permanent error
    on EVERY call (a stale key, a 401) will now never open the breaker, so there is no backoff and no
    short-circuit. Each call costs exactly ONE attempt — there is no retry storm to protect against —
    and the caller receives the real exception, which names the actual problem, on every call. A
    silent ``CircuitOpen`` after five of them was cheaper by one attempt and far more expensive to
    diagnose.

    The two rules compose: a HALF-OPEN probe that ends in a non-retryable error records neither a
    success nor a failure, so the breaker stays half-open — still not knowing — and lends the probe
    slot to the next caller. Calls trickle through one at a time until something real answers the
    question, which is the honest behavior for "no evidence yet"."""
    probing = False
    if breaker is not None:
        if not breaker.allow():
            raise CircuitOpen("circuit breaker open")
        # Only the call that was lent the half-open probe owns the reservation — a call admitted
        # through a CLOSED breaker must not release a probe some other task is holding.
        probing = breaker.state == "half_open"
    last_exc: Exception | None = None
    try:
        for i in range(attempts):
            try:
                result = await (asyncio.wait_for(call(), timeout_s) if timeout_s else call())
                if breaker is not None:
                    breaker.record_success()
                return result
            except Exception as exc:  # noqa: BLE001 — resilience wrapper re-raises after retries
                last_exc = exc
                # Non-evidence first: a locally-raised refusal or a permanent 4xx must never move a
                # PROVIDER-health signal. See the paragraph above.
                if isinstance(exc, non_retryable):
                    raise
                if breaker is not None:
                    breaker.record_failure()
                if i < attempts - 1:
                    await asyncio.sleep(base_delay_s * (2**i))
    finally:
        # Always, on every exit path including cancellation — a probe reservation that outlives the
        # call it was taken for wedges the breaker half-open and nothing is ever admitted again.
        if probing and breaker is not None:
            breaker.release_probe()
    assert last_exc is not None
    raise last_exc


__all__ = ["CircuitBreaker", "CircuitOpen", "with_retry"]
