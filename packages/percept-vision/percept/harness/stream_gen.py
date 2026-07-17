"""PROBE-HARNESS SCAFFOLDING — ``stream_gen`` (Viability §6, board card A10).

**Wiring-only, not a production connector.** The probe harness's frame-stream generator: it
replays a fixture clip (a recorded sequence of frames) OR synthesizes a parametric stream (motion
bursts, seeded pseudo-random bursts), can **fan out** one source to N concurrent consumer streams,
and reads time only through an **injected** :data:`~percept.clock.Clock` — never
``time.monotonic()``/``time.sleep`` directly — so a probe or soak run is fully deterministic and
driven by the test/soak harness advancing a :class:`~percept.clock.FakeClock`, with zero
wall-clock waits (same contract :mod:`percept.clock` names for run-time cognition, reused here for
the probe harness itself).

Standing rule 2 (see :mod:`percept.backends.fake`) still applies: this generates *frames and
timing*, never claims model intelligence. Frame payload is the existing tier-0
:class:`~percept.harness.tier0.Frame` (a coarse grayscale grid) so a generated stream drops
straight into :class:`~percept.harness.tier0.MotionGate` / ``GatedWatchLoop`` without adapting.

Not wired into ``percept/registry.py``. Downstream: board card A11 (``load_driver``) pulls frames
from :class:`StreamGen` to feed concurrent sessions against ``percept.backends.fake_judge``
(A12, already built) for the Phase-A probes (queueing, backpressure, latency, delivery).
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

from ..clock import Clock
from .tier0 import Frame, gray_frame

# ---------------------------------------------------------------------------
# Timeline construction — fixture replay (a) and synthetic streams (b)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TimedFrame:
    """One frame on a stream's own timeline: a scheduled emission offset ``t`` (seconds, >= 0,
    relative to the stream's start) plus the :class:`~percept.harness.tier0.Frame` payload."""

    t: float
    frame: Frame

    def __post_init__(self) -> None:
        if self.t < 0:
            raise ValueError(f"TimedFrame.t must be >= 0, got {self.t!r}")


Timeline = tuple[TimedFrame, ...]


def fixture_clip(frames: Sequence[Frame], *, dt: float = 1.0, start: float = 0.0) -> Timeline:
    """Build a fixed-cadence timeline from a plain fixture frame sequence — replay case (a): "play
    this recorded clip back at its assumed frame rate". For irregular fixture cadence (real
    per-frame capture timestamps), use :func:`timed_clip` instead."""
    if dt <= 0:
        raise ValueError(f"fixture_clip dt must be > 0, got {dt!r}")
    return tuple(TimedFrame(t=start + i * dt, frame=f) for i, f in enumerate(frames))


def timed_clip(frames: Sequence[Frame], timestamps: Sequence[float]) -> Timeline:
    """Build a timeline from explicit per-frame timestamps — replay case (a) with irregular
    cadence (e.g. a fixture captured with real, non-uniform inter-frame gaps)."""
    if len(frames) != len(timestamps):
        raise ValueError(
            f"frames ({len(frames)}) and timestamps ({len(timestamps)}) must be the same length"
        )
    return tuple(TimedFrame(t=t, frame=f) for t, f in zip(timestamps, frames))


def synthetic_clip(
    frame_fn: Callable[[int], Frame],
    n: int,
    *,
    dt: float = 1.0,
    start: float = 0.0,
) -> Timeline:
    """Build a timeline from a parametric frame function — replay case (b): synthetic streams.

    ``frame_fn(i)`` deterministically produces frame ``i`` of ``n`` (index-based, so pass a closure
    over a seeded ``random.Random`` for parametric randomness — determinism then follows from the
    seed, same reproducibility contract as ``percept.backends.fake_judge``'s seeded latency/error
    injection: never touch the global ``random`` module).
    """
    if n < 0:
        raise ValueError(f"synthetic_clip n must be >= 0, got {n!r}")
    if dt <= 0:
        raise ValueError(f"synthetic_clip dt must be > 0, got {dt!r}")
    return tuple(TimedFrame(t=start + i * dt, frame=frame_fn(i)) for i in range(n))


def parametric_motion_clip(
    *,
    w: int = 8,
    h: int = 8,
    n: int = 48,
    motion_frames: Iterable[int] = (),
    motion_pixels: int = 40,
    motion_value: int = 255,
    still_fill: int = 0,
    dt: float = 1.0,
    start: float = 0.0,
) -> Timeline:
    """A ready-made synthetic clip: a mostly-still scene with motion bursts at the explicit
    ``motion_frames`` indices (same shape as ``harness.bench_gating.synthetic_stream``, generalized
    into a timed :data:`Timeline`)."""
    motion_set = frozenset(motion_frames)

    def _frame(i: int) -> Frame:
        if i not in motion_set:
            return gray_frame(w, h, fill=still_fill)
        pixels = [still_fill] * (w * h)
        for p in range(min(motion_pixels, len(pixels))):
            pixels[p] = motion_value
        return Frame(w=w, h=h, pixels=tuple(pixels))

    return synthetic_clip(_frame, n, dt=dt, start=start)


def random_motion_clip(
    *,
    seed: int,
    w: int = 8,
    h: int = 8,
    n: int = 48,
    motion_prob: float = 0.15,
    motion_pixels: int = 40,
    motion_value: int = 255,
    still_fill: int = 0,
    dt: float = 1.0,
    start: float = 0.0,
) -> Timeline:
    """A synthetic clip with pseudo-random motion bursts, seeded for reproducibility: same
    ``seed`` (and other params) -> byte-identical :data:`Timeline`. Draws from a private
    ``random.Random(seed)``, never the global ``random`` module, matching ``fake_judge``'s seeding
    convention."""
    rng = random.Random(seed)
    motion_set = frozenset(i for i in range(n) if rng.random() < motion_prob)
    return parametric_motion_clip(
        w=w,
        h=h,
        n=n,
        motion_frames=motion_set,
        motion_pixels=motion_pixels,
        motion_value=motion_value,
        still_fill=still_fill,
        dt=dt,
        start=start,
    )


# ---------------------------------------------------------------------------
# StreamGen — one consumer's cursor over a timeline, driven by an injected clock
# ---------------------------------------------------------------------------


class StreamGen:
    """One stream consumer: replays a :data:`Timeline` driven by an injected :data:`Clock`.

    No wall-clock waits: :meth:`poll` is non-blocking — it compares elapsed clock time (since the
    stream's construction, i.e. since ``clock()`` was first read) against each remaining frame's
    scheduled offset and returns whatever has become due. Deterministic replay is a property of the
    *caller* driving the clock: feed a :class:`~percept.clock.FakeClock` and call ``.advance(ms)``
    between ``poll()`` calls to control exactly which frames become due — no sleeping, no thread
    scheduling nondeterminism. Two ``StreamGen`` instances built over the same timeline and driven
    by clocks that receive the same sequence of ``advance()`` calls emit byte-identical sequences.
    """

    def __init__(self, timeline: Timeline | Sequence[TimedFrame], *, clock: Clock) -> None:
        self._timeline: Timeline = tuple(timeline)
        self._clock = clock
        self._t0 = clock()
        self._idx = 0

    @property
    def timeline(self) -> Timeline:
        return self._timeline

    @property
    def emitted_count(self) -> int:
        """How many frames this consumer has emitted so far."""
        return self._idx

    def done(self) -> bool:
        """True once every frame in the timeline has been emitted via :meth:`poll`."""
        return self._idx >= len(self._timeline)

    def poll(self) -> list[TimedFrame]:
        """Return frames newly due, in order, without blocking. Reads ``clock()`` exactly once."""
        if self._idx >= len(self._timeline):
            return []
        now = self._clock() - self._t0
        due: list[TimedFrame] = []
        while self._idx < len(self._timeline) and self._timeline[self._idx].t <= now:
            due.append(self._timeline[self._idx])
            self._idx += 1
        return due

    def reset(self) -> None:
        """Rewind to the start of the timeline, re-anchored to the current clock reading."""
        self._t0 = self._clock()
        self._idx = 0


def fan_out(
    timeline: Timeline | Sequence[TimedFrame],
    n: int,
    *,
    clock: Clock | None = None,
    clocks: Sequence[Clock] | None = None,
) -> list[StreamGen]:
    """Fan one source ``timeline`` out to ``n`` independent, concurrent :class:`StreamGen`
    consumers (Viability §6 fan-out requirement).

    Exactly one of ``clock`` / ``clocks`` must be given:

    * ``clock`` — one :data:`Clock` shared by all ``n`` streams (the common case: N concurrent
      consumers of one source, e.g. N goals/sessions all watching the same camera, paced by the
      same simulated wall-clock).
    * ``clocks`` — a sequence of ``n`` distinct clocks, one per stream, so consumers can be
      desynchronized (e.g. to probe uneven backpressure across sessions).

    Each returned stream owns an independent cursor (``_idx``/``_t0``) over the SAME immutable
    timeline, so polling one stream never affects another's position — the fan-out consumers are
    independent even when they share a clock.
    """
    if n <= 0:
        raise ValueError(f"fan_out n must be > 0, got {n!r}")
    if (clock is None) == (clocks is None):
        raise ValueError("fan_out requires exactly one of `clock` (shared) or `clocks` (per-stream)")

    tl: Timeline = tuple(timeline)

    if clocks is not None:
        if len(clocks) != n:
            raise ValueError(f"clocks length ({len(clocks)}) must equal n ({n})")
        return [StreamGen(tl, clock=c) for c in clocks]

    assert clock is not None  # narrowed by the XOR check above
    return [StreamGen(tl, clock=clock) for _ in range(n)]


__all__ = [
    "TimedFrame",
    "Timeline",
    "fixture_clip",
    "timed_clip",
    "synthetic_clip",
    "parametric_motion_clip",
    "random_motion_clip",
    "StreamGen",
    "fan_out",
]
