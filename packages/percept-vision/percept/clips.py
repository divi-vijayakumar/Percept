"""Clips: the pre-roll ring, and which frames a judge sees for an interval.

Pure: numpy-free standard library plus the contract types, no I/O, no clock. Time comes from
the frames themselves (``Frame.t_wall``, Unix seconds), because evidence intervals in a
:class:`~percept.contracts.JudgeRequest` are wall-clock.

:class:`PreRoll` keeps the last few seconds of a stream, sampled to a low rate (by default
8 s at 2 fps, so about 16 frames). It is what makes "show the judge the moment it happened"
possible after a trigger fires.

Evidence is ephemeral. A frame that has aged out of the ring is gone for good; nothing here
writes to disk. A decision keeps exactly one durable image, its keyframe, which the runtime
picks with :meth:`PreRoll.keyframe` while the frame is still in the ring.
"""

from __future__ import annotations

import math
from collections import deque

from percept.contracts import Frame

Interval = tuple[float, float]  # (start, end), wall-clock seconds, inclusive


class PreRoll:
    """A time-bounded ring of recent frames, sampled to at most ``fps`` frames per second.

    Sampling is by wall-clock slot: a frame is kept when it is the first one seen in its
    ``1/fps``-second slot, so the ring never holds more than about ``seconds * fps + 1``
    frames however fast frames are offered. Frames older than ``seconds`` behind the newest
    one are dropped. If the wall clock steps backwards (an NTP correction), frames that are
    now in the future are dropped so the ring stays ordered.
    """

    def __init__(self, seconds: float = 8.0, fps: float = 2.0) -> None:
        if not (math.isfinite(seconds) and seconds > 0 and math.isfinite(fps) and fps > 0):
            raise ValueError("PreRoll seconds and fps must be positive")
        self.seconds = float(seconds)
        self.fps = float(fps)
        self._frames: deque[Frame] = deque()
        self._slot: int | None = None

    def __repr__(self) -> str:
        return f"PreRoll(seconds={self.seconds:g}, fps={self.fps:g}, frames={len(self._frames)})"

    def __len__(self) -> int:
        return len(self._frames)

    @property
    def frames(self) -> tuple[Frame, ...]:
        """Every frame in the ring, oldest first."""
        return tuple(self._frames)

    @property
    def oldest(self) -> float | None:
        """``t_wall`` of the oldest frame still held, or None when empty.

        An interval that starts before this is only partly covered: :meth:`select` returns
        what is left of it.
        """
        return self._frames[0].t_wall if self._frames else None

    def add(self, frame: Frame) -> bool:
        """Offer a frame. Returns True when it was kept."""
        frames = self._frames
        if frames and frame.t_wall < frames[-1].t_wall:
            while frames and frames[-1].t_wall > frame.t_wall:
                frames.pop()
            self._slot = math.floor(frames[-1].t_wall * self.fps) if frames else None
        slot = math.floor(frame.t_wall * self.fps)
        if slot == self._slot:
            return False
        self._slot = slot
        frames.append(frame)
        horizon = frame.t_wall - self.seconds
        while frames[0].t_wall < horizon:
            frames.popleft()
        return True

    def clear(self) -> None:
        self._frames.clear()
        self._slot = None

    def within(self, interval: Interval) -> tuple[Frame, ...]:
        """Frames whose ``t_wall`` lies in the closed interval, oldest first."""
        start, end = _check(interval)
        return tuple(f for f in self._frames if start <= f.t_wall <= end)

    def select(self, interval: Interval, n: int = 4) -> tuple[Frame, ...]:
        """Up to ``n`` frames spread evenly over ``interval``, oldest to newest.

        When the interval holds more than ``n`` frames, the first and last are always
        included and the rest are evenly spaced by index between them. When it holds ``n`` or
        fewer, all of them are returned. The part of an interval older than the ring is gone,
        so a partly covered interval yields only its covered frames, and an interval entirely
        older than the ring yields nothing.
        """
        if not (isinstance(n, int) and n >= 1):
            raise ValueError("select needs n >= 1")
        inside = self.within(interval)
        m = len(inside)
        if m <= n:
            return inside
        if n == 1:
            return (inside[-1],)
        return tuple(inside[round(i * (m - 1) / (n - 1))] for i in range(n))

    def keyframe(self, interval: Interval, at: float | None = None) -> Frame | None:
        """The frame a decision keeps: the one in ``interval`` nearest ``at``.

        ``at`` defaults to the end of the interval, the latest evidence for the answer. Ties go
        to the later frame. None when no frame of the interval is still in the ring.
        """
        inside = self.within(interval)
        if not inside:
            return None
        target = interval[1] if at is None else at
        return min(reversed(inside), key=lambda f: abs(f.t_wall - target))


def _check(interval: Interval) -> Interval:
    start, end = interval
    if not (math.isfinite(start) and math.isfinite(end) and start <= end):
        raise ValueError(f"interval is (start, end) with start <= end, got {interval!r}")
    return start, end
