"""The stream-timeline firewall — monotonicity and finiteness for a caller-supplied ``t``.

This is deliberately **not** in :mod:`percept.clock`, and the split is load-bearing. ``clock.py`` is the
*injectable clock seam* — ADR-001 §5 records it as harness-only, and
``tests/test_adr001_status_honesty.py::test_clock_module_is_harness_only`` enforces that: the moment
run-time cognition imports ``percept.clock``, the ADR's implementation-status note is stale and the
suite says so.

The guard below is the opposite half. The clock is *ours* and monotonic by construction; the ``t`` a
caller threads in comes from a file, a decoder, a socket or a host clock, none of which promise
monotonicity or even finiteness. That untrusted timeline is read by run-time cognition
(:mod:`percept.rhythm`, :mod:`percept.interest`) as well as by the harness, so its firewall cannot live
behind a harness-only seam. Pure stdlib, no clock dependency.
"""

from __future__ import annotations

from typing import Literal

from ._numeric import finite

#: What one stream timestamp is, relative to the last one this guard saw. ``ok`` is strictly forward
#: (including the very first sample); ``duplicate`` is the same instant again — two readings on one
#: frame's timeline, which is normal (a visual evaluation and an external wake can share a ``ts``);
#: ``rewind`` is strictly backwards, which no forward-running stream ever produces.
StreamTimeVerdict = Literal["ok", "duplicate", "rewind"]


class StreamTimeRewindWarning(UserWarning):
    """The caller's stream timeline stepped backwards and the reader absorbed it as a discontinuity.

    A warning, not an error, because a rewind is a recoverable source hiccup (a reconnect, a file
    seek, a corrected host clock) and a dead watch is a far worse answer than a dropped brake. It is
    a warning rather than nothing because the brake really was dropped: the affected goals/concerns
    can spend again immediately, and that has to be traceable to this moment rather than look like
    the cap misfiring.

    Shared by every reader that chose the RESET policy — :meth:`percept.harness.tier0.MotionGate._gate_ts`
    and :meth:`percept.sampling.SamplingController.should_process` — so a caller can catch one class
    and see every stream discontinuity, whichever seam noticed it first.
    """


class StreamTimeGuard:
    """Monotonicity firewall for the caller-supplied stream timeline.

    The injectable clock is *ours* and is monotonic by construction. The ``t`` a caller threads in is
    not: a file seek, a source reconnect on a restarted timebase, or a corrected host clock all step it
    backwards, and a broken decoder hands over a ``nan``. Both are silent killers of any operator
    written as ``t - last < window``:

    * **backwards** — ``t - last`` stays negative, so a "has the window elapsed?" test answers *no*
      forever and the operator is muted for the rest of the run, while frames keep flowing.
    * **not a number** — *every* comparison against NaN is ``False``, so the same test is not
      tightened but **bypassed**, and the NaN is then stored as the new ``last``, poisoning the state.

    :meth:`check` refuses the second case loudly (it is never recoverable data) and merely *names*
    the first. **Choosing what a rewind means is the caller's policy, and the live callers
    legitimately choose differently:**

    * :mod:`percept.interest` raises — its intervals are an evidence record, and a record whose
      timeline went backwards is not repairable, so the stream must stop.
    * :class:`~percept.harness.tier0.MotionGate` **resets** — a rewind there is a reconnect or a
      seek, a recoverable source hiccup, and raising would turn it into a dead watch. It forgets the
      per-goal state the old timeline was measured against, exactly as ``MotionGate.reset()`` forgets
      the previous frame, and carries on.
    * :mod:`percept.rhythm` **resets** for the same reason — a rewound media timebase is a source
      property, not a programming error, and raising would cost every rep in the rewound span.

    The guard therefore holds no policy of its own. It records, classifies, and returns.
    """

    __slots__ = ("_last", "_rewinds")

    def __init__(self) -> None:
        self._last: float | None = None
        self._rewinds = 0

    def check(self, name: str, t: object) -> StreamTimeVerdict:
        """Classify ``t`` against the last accepted timestamp, then accept it.

        Raises :class:`ValueError` naming ``name`` when ``t`` is not a real finite number — a NaN or
        an infinity is not a timeline position under any policy, so there is nothing to choose. A
        ``rewind`` verdict still *accepts* ``t`` as the new baseline: the stream has moved to a new
        timeline, and re-measuring every later frame against the abandoned one would report every
        one of them as a rewind too.
        """
        value = finite(name, t)
        last = self._last
        self._last = value
        if last is None or value > last:
            return "ok"
        if value == last:
            return "duplicate"
        self._rewinds += 1
        return "rewind"

    def reset(self) -> None:
        """Forget the baseline — the next :meth:`check` is a first sample and cannot be a rewind.

        For a caller that already knows the timeline restarted (a declared new epoch, a new source),
        so a deliberate restart is not reported as a discontinuity. The rewind tally is kept: it is
        a run-level diagnostic, not per-epoch state.
        """
        self._last = None

    @property
    def last(self) -> float | None:
        """The last accepted timestamp, or ``None`` before the first one."""
        return self._last

    @property
    def rewinds(self) -> int:
        """How many backwards steps this guard has seen — the number to surface, never to swallow."""
        return self._rewinds


__all__ = ["StreamTimeGuard", "StreamTimeVerdict", "StreamTimeRewindWarning"]
