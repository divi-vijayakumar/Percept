"""Scheduler — time-triggered proactive fires for programmable Watches.

Timing lives in the clock; **content judgment lives in the model**. On a tick, ``due`` returns the
timers whose moment has arrived; the agent re-invokes the conductor with a ``scheduler_tick`` event
so it can ``proactive_speak`` or cancel. Repeating timers fire **once per interval** (a long stall
does not back-fill a burst — it advances past the missed slots and fires once).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable


@dataclass
class Timer:
    id: str
    goal_id: str
    interval_s: float
    next_fire: float
    repeat: bool


class Scheduler:
    """Monotonic-clock timers. ``every`` repeats; ``at`` is one-shot; ``due`` advances them."""

    def __init__(self, clock: Callable[[], float] | None = None) -> None:
        self._clock = clock or time.monotonic
        self._timers: dict[str, Timer] = {}
        self._counter = 0

    def _now(self, now: float | None) -> float:
        return self._clock() if now is None else now

    def every(self, interval_s: float, goal_id: str, *, now: float | None = None) -> Timer:
        start = self._now(now)
        self._counter += 1
        timer = Timer(
            id=f"t{self._counter}",
            goal_id=goal_id,
            interval_s=interval_s,
            next_fire=start + interval_s,
            repeat=True,
        )
        self._timers[timer.id] = timer
        return timer

    def at(self, when_s: float, goal_id: str) -> Timer:
        self._counter += 1
        timer = Timer(
            id=f"t{self._counter}",
            goal_id=goal_id,
            interval_s=0.0,
            next_fire=when_s,
            repeat=False,
        )
        self._timers[timer.id] = timer
        return timer

    def due(self, now: float | None = None) -> list[Timer]:
        """Return timers whose moment has arrived; advance repeaters, drop fired one-shots.

        A repeater fires **once** even if several intervals elapsed since the last tick — it skips
        forward to the next future slot rather than emitting a backlog.
        """
        t = self._now(now)
        fired: list[Timer] = []
        for timer in list(self._timers.values()):
            if timer.next_fire > t:
                continue
            fired.append(timer)
            if timer.repeat:
                # advance to the first slot strictly in the future (skip missed intervals).
                step = timer.interval_s if timer.interval_s > 0 else 1.0
                while timer.next_fire <= t:
                    timer.next_fire += step
            else:
                del self._timers[timer.id]
        return fired

    def cancel(self, timer_id: str) -> bool:
        return self._timers.pop(timer_id, None) is not None

    def active(self) -> list[Timer]:
        return list(self._timers.values())


__all__ = ["Timer", "Scheduler"]
