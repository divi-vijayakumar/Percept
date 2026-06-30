"""Developer debug helpers — turn the per-frame ``TraceEvent`` stream into readable output.

Wire :func:`print_trace` as the ``on_trace`` callback to watch a frame's decision unfold::

    from percept.debug import print_trace
    agent = Percept.create(on_trace=print_trace)         # or run(core, ..., on_trace=print_trace)

Prints one line per stage, e.g.::

    [trace=camera@12.0 goal=caffeine] judge.finished satisfied=True confidence=0.92
    [trace=camera@12.0 goal=caffeine] gate.classified truth=yes action=fire
"""

from __future__ import annotations

import sys
from typing import Any

from .observability import TraceEvent


def format_trace(ev: TraceEvent) -> str:
    head = f"[trace={ev.trace_id}" + (f" goal={ev.goal_id}" if ev.goal_id else "") + "]"
    body = " ".join(f"{k}={v}" for k, v in ev.data.items())
    return f"{head} {ev.stage}" + (f" {body}" if body else "")


def print_trace(ev: TraceEvent, *, file: Any = None) -> None:
    """An ``on_trace`` callback that prints each :class:`TraceEvent` (to stderr by default)."""
    print(format_trace(ev), file=file if file is not None else sys.stderr)


def collect_trace(into: list) -> Any:
    """Return an ``on_trace`` callback that appends each event to ``into`` (handy in tests)."""
    def _cb(ev: TraceEvent) -> None:
        into.append(ev)
    return _cb


__all__ = ["format_trace", "print_trace", "collect_trace"]
