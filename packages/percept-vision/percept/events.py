"""Events — temporal perception beside the object/attribute track.

A pass, a rep, a discrete action is **not** an entity-with-an-attribute — it's an event over a
span. This is an *additive* parallel track: a rolling frame window (replacing keep-latest),
``Vision.detect_events`` over that window, and **event identity/dedup** — the temporal analog of
IoU continuity — so a multi-frame event counts once. The event gate reuses the three-state
semantics (a new event is a rising edge; counting falls out as "distinct events of a type").
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Event:
    type: str
    t: float
    participants: list = field(default_factory=list)
    confidence: float = 1.0
    id: str | None = None  # assigned by EventLog when the event is first seen


def frame_nbytes(frame: Any) -> int:
    """Best-effort decoded size of ``frame`` in bytes — the unit a frame RING is actually bounded in.

    A count-bounded ring is only bounded in memory if every frame is the same size, and it is not: the
    SAME ``size=48`` ring costs ~0.4 MB of 8x8 test frames and ~300 MB of 1080p RGB. So the bound that
    matters is a byte budget. Recognises numpy-ish buffers (``nbytes``), raw ``bytes``/``bytearray``,
    the harness :class:`~percept.contracts.Frame` (``w*h``), and PIL-ish images; anything unrecognised
    reports ``0`` (unknown), which can never *cause* a drop — an unmeasurable frame is never evicted on
    a guess.
    """
    n = getattr(frame, "nbytes", None)  # numpy ndarray / memoryview-ish
    if isinstance(n, int) and n >= 0:
        return n
    if isinstance(frame, (bytes, bytearray, memoryview)):
        return len(frame)
    w, h = getattr(frame, "w", None), getattr(frame, "h", None)
    if isinstance(w, int) and isinstance(h, int):
        return max(0, w) * max(0, h)
    size = getattr(frame, "size", None)  # PIL Image.size == (w, h)
    if isinstance(size, tuple) and len(size) == 2 and all(isinstance(v, int) for v in size):
        return size[0] * size[1] * len(getattr(frame, "getbands", lambda: "RGB")())
    return 0


class FrameWindow:
    """A rolling ``(t, frame)`` ring buffer per source — temporal perception needs a span, not a frame.

    Bounded by COUNT (``size``) and, optionally, by BYTES (``max_bytes``). The byte bound exists because
    the count bound alone does not bound memory: this ring is the SDK's largest resident structure (see
    ``percept/runtime/worker.py`` — the incident snippet's pre-roll), and its cost scales with the
    source's resolution, not with anything the operator configured. Per ADR-001 §9 a bound-driven drop
    is never silent: each byte-bound eviction calls ``on_drop(t, frame, reason="max_bytes")``, which the
    runtime turns into a health event. The newest frame is never evicted (a ring of one always holds the
    current frame), so a single frame larger than the budget is kept and reported, never dropped blind.
    """

    def __init__(
        self,
        size: int = 12,
        *,
        max_bytes: int | None = None,
        on_drop: Any = None,
    ) -> None:
        self._size = max(1, size)
        self._max_bytes = int(max_bytes) if max_bytes and max_bytes > 0 else None
        self._on_drop = on_drop
        self._frames: list[tuple[float, Any]] = []
        self._bytes = 0

    def push(self, t: float, frame: Any) -> None:
        self._frames.append((t, frame))
        if self._max_bytes is not None:
            self._bytes += frame_nbytes(frame)
        if len(self._frames) > self._size:
            self._evict("max_frames")
        if self._max_bytes is not None:
            while len(self._frames) > 1 and self._bytes > self._max_bytes:
                self._evict("max_bytes")

    def _evict(self, reason: str) -> None:
        dt, dropped = self._frames.pop(0)
        if self._max_bytes is not None:
            self._bytes = max(0, self._bytes - frame_nbytes(dropped))
        if reason == "max_bytes" and self._on_drop is not None:
            self._on_drop(dt, dropped, reason)

    @property
    def nbytes(self) -> int:
        """Measured bytes currently retained (0 when no byte bound is configured)."""
        return self._bytes

    def frames(self) -> list[tuple[float, Any]]:
        return list(self._frames)

    def latest(self) -> tuple[float, Any] | None:
        return self._frames[-1] if self._frames else None

    def clear(self) -> None:
        """Drop retained frames at a source discontinuity."""
        self._frames.clear()
        self._bytes = 0

    def __len__(self) -> int:
        return len(self._frames)


class AudioWindow:
    """A rolling ``(t, chunk)`` audio buffer per source, time-windowed by seconds — the audio analog of
    :class:`FrameWindow`. Feeds the fused/acoustic perception path: the agent concatenates the last N
    seconds into one clip for ``Vision.judge(audio=...)``. Fed by the voice loop's mic stream (the same
    audio already flowing for STT) — no new capture path."""

    def __init__(self, seconds: float = 4.0) -> None:
        self._seconds = max(0.5, seconds)
        self._chunks: list[tuple[float, Any]] = []

    def push(self, t: float, chunk: Any) -> None:
        self._chunks.append((t, chunk))
        cutoff = t - self._seconds
        while self._chunks and self._chunks[0][0] < cutoff:
            self._chunks.pop(0)

    def clip(self, seconds: float, *, now: float | None = None) -> list[Any]:
        """The audio chunks within the last ``seconds`` (oldest→newest)."""
        if not self._chunks:
            return []
        end = self._chunks[-1][0] if now is None else now
        start = end - seconds
        return [c for t, c in self._chunks if t >= start]

    def clear(self) -> None:
        """Drop retained chunks at a source discontinuity."""
        self._chunks.clear()

    def __len__(self) -> int:
        return len(self._chunks)


class EventLog:
    """Dedups raw per-tick detections into **distinct** events (the temporal analog of continuity).

    A run of the same event type within ``merge_gap_s`` of the last sighting is one event
    (continuation — extend it); a longer gap opens a new one. So ``[hit, hit, hit]`` -> 1 event and
    ``[hit, miss, hit]`` (the miss opening a gap) -> 2. ``count`` is distinct events of a type.
    """

    def __init__(self, merge_gap_s: float = 1.5) -> None:
        self.merge_gap_s = merge_gap_s
        self._events: list[Event] = []
        self._counter = 0
        self._last_by_type: dict[str, Event] = {}

    def observe(
        self, type: str, t: float, *, participants: list | None = None, confidence: float = 1.0
    ) -> tuple[Event, bool]:
        """Fold a raw detection into the log. Returns ``(event, is_new)``."""
        last = self._last_by_type.get(type)
        if last is not None and t - last.t <= self.merge_gap_s:
            last.t = t  # continuation — extend the active event, don't recount
            last.confidence = max(last.confidence, confidence)
            return last, False
        self._counter += 1
        event = Event(
            type=type,
            t=t,
            participants=participants or [],
            confidence=confidence,
            id=f"ev{self._counter}",
        )
        self._events.append(event)
        self._last_by_type[type] = event
        return event, True

    def count(self, type: str | None = None) -> int:
        return len([e for e in self._events if type is None or e.type == type])

    def events(self, type: str | None = None) -> list[Event]:
        return [e for e in self._events if type is None or e.type == type]


__all__ = ["Event", "FrameWindow", "EventLog"]
