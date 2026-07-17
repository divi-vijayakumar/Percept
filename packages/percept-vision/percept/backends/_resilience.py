"""Retry + backoff + a tiny circuit breaker for outbound backend calls.

Deterministic backoff (no jitter RNG so L1 stays reproducible). Real backends wrap their
network calls with :func:`with_retry`; fakes never need it.
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
    fail_threshold: int = 5
    reset_after_s: float = 30.0
    _failures: int = field(default=0, init=False)
    _opened_at: float | None = field(default=None, init=False)
    _opens: int = field(default=0, init=False)  # telemetry: total closed->open transitions

    def _now(self) -> float:
        return asyncio.get_event_loop().time()

    def allow(self) -> bool:
        if self._opened_at is None:
            return True
        if self._now() - self._opened_at >= self.reset_after_s:
            # half-open: let one through
            self._opened_at = None
            self._failures = 0
            return True
        return False

    def record_success(self) -> None:
        self._failures = 0
        self._opened_at = None

    def record_failure(self) -> None:
        self._failures += 1
        if self._failures >= self.fail_threshold and self._opened_at is None:
            self._opened_at = self._now()
            self._opens += 1

    @property
    def state(self) -> str:
        """``"open"`` (short-circuiting) or ``"closed"`` (passing calls through)."""
        return "open" if self._opened_at is not None else "closed"

    def snapshot(self) -> dict:
        """Telemetry surface — current state, the live failure count, and total opens."""
        return {"state": self.state, "failures": self._failures, "opens": self._opens}


async def with_retry(
    call: Callable[[], Awaitable[T]],
    *,
    attempts: int = 3,
    base_delay_s: float = 0.2,
    breaker: CircuitBreaker | None = None,
    timeout_s: float | None = None,
) -> T:
    """Call ``call`` with exponential backoff; honor an optional circuit breaker.

    ``timeout_s`` (opt) is a HARD per-attempt wall-clock bound via :func:`asyncio.wait_for`. It exists
    because a provider client's own request timeout does not always bound a WEDGED socket send/connect
    (observed: a live judge call hung for hours inside ``sock_send``). With it set, a stuck attempt is
    cancelled to ``TimeoutError`` and retried/failed like any other error — the run can never hang.
    ``None`` = byte-for-byte the historical behavior (no extra bound)."""
    if breaker is not None and not breaker.allow():
        raise CircuitOpen("circuit breaker open")
    last_exc: Exception | None = None
    for i in range(attempts):
        try:
            result = await (asyncio.wait_for(call(), timeout_s) if timeout_s else call())
            if breaker is not None:
                breaker.record_success()
            return result
        except Exception as exc:  # noqa: BLE001 — resilience wrapper re-raises after retries
            last_exc = exc
            if breaker is not None:
                breaker.record_failure()
            if i < attempts - 1:
                await asyncio.sleep(base_delay_s * (2**i))
    assert last_exc is not None
    raise last_exc


__all__ = ["CircuitBreaker", "CircuitOpen", "with_retry"]
