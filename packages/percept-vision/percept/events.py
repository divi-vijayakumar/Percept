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


class FrameWindow:
    """A rolling ``(t, frame)`` ring buffer per source — temporal perception needs a span, not a frame."""

    def __init__(self, size: int = 12) -> None:
        self._size = max(1, size)
        self._frames: list[tuple[float, Any]] = []

    def push(self, t: float, frame: Any) -> None:
        self._frames.append((t, frame))
        if len(self._frames) > self._size:
            self._frames.pop(0)

    def frames(self) -> list[tuple[float, Any]]:
        return list(self._frames)

    def latest(self) -> tuple[float, Any] | None:
        return self._frames[-1] if self._frames else None

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
