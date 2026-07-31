"""The Clock primitive — run-time cognition reads time ONLY through an injectable clock (v3 §9).

Run-time cognition's temporal operators (dwell, refractory, absence/deadline, episode boundary) are
functions of *time and order*, so they are only deterministic — and only **replayable** — if time enters
through one injectable seam instead of `time.time()` scattered across the code. That seam already exists:
``Percept(clock=...)`` and ``Scheduler(clock=...)`` take a ``Callable[[], float]`` returning **seconds**,
threaded through every hot-path stamp as ``self.clock() if t is None else t`` (the explicit ``t`` is the
stream timeline; the clock is the wall-clock fallback). This module names that contract and ships the two
fills it needs: a real :data:`WallClock` and a test-driven :class:`FakeClock`.

``FakeClock`` is a drop-in for the ``clock`` callable (it IS callable) **plus** an ``advance()`` the test or
the timeline-replay harness drives — so a deadline / absence fire can be exercised deterministically
with zero wall-clock waits and no change to any call site.

The *stream* timeline is the other half, and it is the untrusted one: the explicit ``t`` a caller passes
in comes from a file, a decoder, a socket or a host clock, none of which promise monotonicity or even
finiteness. :class:`StreamTimeGuard` names that boundary. It lives in :mod:`percept._streamtime`, not
here, and is only re-exported for the harness callers that already import it from this module — because
this module is **harness-only** (ADR-001 §5, enforced by
``tests/test_adr001_status_honesty.py::test_clock_module_is_harness_only``) while the untrusted stream
timeline is read by run-time cognition too. Importing the guard from here outside ``harness/`` would make
the ADR's implementation-status note stale; import it from :mod:`percept._streamtime` instead.
"""

from __future__ import annotations

import time
from typing import Callable

from ._streamtime import StreamTimeGuard, StreamTimeVerdict

# The Clock contract: a zero-arg callable returning the current time in SECONDS (monotonic in production).
# Kept as a structural type alias rather than a Protocol class so the existing ``Callable[[], float]``
# call sites and injections are already conformant — naming the contract, not changing it.
Clock = Callable[[], float]

# Production clock: monotonic seconds (matches the existing default in agent.py / scheduler.py).
WallClock: Clock = time.monotonic


class FakeClock:
    """A deterministic, manually-advanced clock for tests and timeline replay.

    Drop-in for the ``clock`` callable — ``Percept.create(clock=FakeClock())`` — so cognition's time advances
    only when the test says so. ``advance`` takes **milliseconds** to match the WatchSpec window units."""

    def __init__(self, start_s: float = 0.0) -> None:
        self._t = float(start_s)

    def __call__(self) -> float:
        """The Clock interface: current time in seconds."""
        return self._t

    def advance(self, ms: float) -> float:
        """Move the clock forward by ``ms`` milliseconds; returns the new time in seconds."""
        self._t += float(ms) / 1000.0
        return self._t

    def set(self, seconds: float) -> None:
        """Jump to an absolute time in seconds (test-only)."""
        self._t = float(seconds)


__all__ = ["Clock", "WallClock", "FakeClock", "StreamTimeGuard", "StreamTimeVerdict"]
