"""Events — temporal perception beside the object/attribute track.

A pass, a rep, a discrete action is **not** an entity-with-an-attribute — it's an event over a
span. This is an *additive* parallel track: a rolling frame window (replacing keep-latest),
``Vision.detect_events`` over that window, and **event identity/dedup** — the temporal analog of
IoU continuity — so a multi-frame event counts once. The event gate reuses the three-state
semantics (a new event is a rising edge; counting falls out as "distinct events of a type").
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field, is_dataclass, replace
from typing import Any, Literal


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


AudioDirection = Literal["input", "output", "mixed", "unknown"]
AudioBoundaryKind = Literal["onset", "vad_start", "endpoint"]


@dataclass(frozen=True)
class AudioBoundary:
    """A retained cheap-audio boundary on its :class:`AudioUnit`'s ``event_t`` clock."""

    kind: AudioBoundaryKind
    t: float

    def __post_init__(self) -> None:
        if self.kind not in ("onset", "vad_start", "endpoint"):
            raise ValueError("audio boundary kind is invalid")
        if not isinstance(self.t, (int, float)) or isinstance(self.t, bool) or not math.isfinite(self.t):
            raise ValueError("audio boundary time must be finite")


@dataclass(frozen=True)
class AudioUnit:
    """One ordered audio ingress unit, independent of how it arrived.

    ``chunk`` is the existing typed :class:`~percept.contracts.AudioFrame` (or a legacy opaque audio
    payload). ``t_start``/``t_end`` are source/media intervals; ``event_t`` is the capture/receipt
    clock used by the runtime gate (the two are identical for recorded media). ``seq`` makes missing,
    duplicate, and reordered pushes explicit. Capture, file replay, and application-pushed audio all
    use this same unit.
    """

    seq: int
    t_start: float
    t_end: float
    chunk: Any
    event_t: float | None = None
    energy: float | None = None
    sample_rate: int | None = None
    channels: int | None = None
    sample_format: str | None = None
    direction: AudioDirection = "unknown"
    participant_id: str | None = None
    boundaries: tuple[AudioBoundary, ...] = ()
    # Carrier provenance is retained on the immutable unit rather than inferred later from the
    # rolling window. Defaults preserve manually-pushed/legacy units; compiled runtime ingestion
    # fills every field before the unit can enter state.
    source_id: str | None = None
    source_epoch: int | None = None
    source_sequence: int | None = None
    observed_time: float | None = None
    source_time: float | None = None

    @property
    def t(self) -> float:
        return self.t_end if self.event_t is None else self.event_t

    def __iter__(self):
        """Keep the historical ``(t, chunk, energy)`` destructuring shape."""
        yield self.t
        yield self.chunk
        yield self.energy

    def __len__(self) -> int:
        return 3

    def __getitem__(self, index):
        return (self.t, self.chunk, self.energy)[index]


@dataclass(frozen=True)
class AudioPushResult:
    """What happened when an :class:`AudioUnit` entered an :class:`AudioWindow`."""

    accepted: bool
    duplicate: bool = False
    reordered: bool = False
    missing: int = 0
    gap_s: float = 0.0
    overlap_s: float = 0.0
    format_changed: bool = False
    expired: int = 0
    expired_bytes: int = 0
    evicted: int = 0
    evicted_bytes: int = 0

    @property
    def discontinuous(self) -> bool:
        return bool(
            self.duplicate
            or self.reordered
            or self.missing
            or self.gap_s
            or self.overlap_s
            or self.format_changed
            or self.evicted
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "duplicate": self.duplicate,
            "reordered": self.reordered,
            "missing": self.missing,
            "gap_s": self.gap_s,
            "overlap_s": self.overlap_s,
            "format_changed": self.format_changed,
            "expired": self.expired,
            "expired_bytes": self.expired_bytes,
            "evicted": self.evicted,
            "evicted_bytes": self.evicted_bytes,
        }


def _audio_nbytes(chunk: Any) -> int:
    pcm = getattr(chunk, "pcm", None)
    if isinstance(pcm, (bytes, bytearray, memoryview)) and pcm:
        return len(pcm)
    encoded = getattr(chunk, "encoded", None)
    if isinstance(encoded, (bytes, bytearray, memoryview)):
        return len(encoded)
    if isinstance(pcm, (bytes, bytearray, memoryview)):
        return 0
    if isinstance(chunk, (bytes, bytearray, memoryview)):
        return len(chunk)
    if (
        isinstance(chunk, tuple)
        and len(chunk) == 2
        and isinstance(chunk[1], (bytes, bytearray, memoryview))
    ):
        return len(chunk[1])
    return 0


_PCM_BYTES_PER_SAMPLE = {"s16le": 2, "f32le": 4, "u8": 1}


def _immutable_audio_chunk(chunk: Any) -> Any:
    """Detach retained PCM/encoded bytes from caller-owned mutable buffers."""
    if isinstance(chunk, (bytearray, memoryview)):
        return bytes(chunk)
    if (
        isinstance(chunk, tuple)
        and len(chunk) == 2
        and isinstance(chunk[1], (bytearray, memoryview))
    ):
        return (chunk[0], bytes(chunk[1]))
    if is_dataclass(chunk):
        updates = {}
        for name in ("pcm", "encoded"):
            value = getattr(chunk, name, None)
            if isinstance(value, (bytearray, memoryview)):
                updates[name] = bytes(value)
        if updates:
            return replace(chunk, **updates)
    return chunk


def _canonical_audio_unit(unit: AudioUnit) -> AudioUnit:
    """Detach mutable bytes and fill unit format metadata from a typed frame."""
    chunk = _immutable_audio_chunk(unit.chunk)
    if not all(hasattr(chunk, name) for name in ("sample_rate", "channels", "sample_format")):
        return replace(unit, chunk=chunk) if chunk is not unit.chunk else unit
    return replace(
        unit,
        chunk=chunk,
        sample_rate=unit.sample_rate if unit.sample_rate is not None else chunk.sample_rate,
        channels=unit.channels if unit.channels is not None else chunk.channels,
        sample_format=(
            unit.sample_format if unit.sample_format is not None else chunk.sample_format
        ),
    )


def validate_audio_unit(unit: AudioUnit) -> AudioUnit:
    """Validate and return the canonical unit; malformed audio fails before state mutation."""
    unit = _canonical_audio_unit(unit)
    if not isinstance(unit.seq, int) or isinstance(unit.seq, bool) or unit.seq < 0:
        raise ValueError("audio unit seq must be a non-negative integer")
    if (
        not isinstance(unit.t_start, (int, float))
        or isinstance(unit.t_start, bool)
        or not isinstance(unit.t_end, (int, float))
        or isinstance(unit.t_end, bool)
        or not math.isfinite(unit.t_start)
        or not math.isfinite(unit.t_end)
    ):
        raise ValueError("audio unit times must be finite")
    if unit.event_t is not None and (
        not isinstance(unit.event_t, (int, float))
        or isinstance(unit.event_t, bool)
        or not math.isfinite(unit.event_t)
    ):
        raise ValueError("audio unit event_t must be finite when known")
    if unit.t_end < unit.t_start:
        raise ValueError("audio unit t_end must be >= t_start")
    if unit.energy is not None and (
        not isinstance(unit.energy, (int, float))
        or isinstance(unit.energy, bool)
        or not math.isfinite(unit.energy)
        or not 0.0 <= unit.energy <= 1.0
    ):
        raise ValueError("audio unit energy must be finite and within [0, 1]")
    if unit.sample_rate is not None and (
        not isinstance(unit.sample_rate, int)
        or isinstance(unit.sample_rate, bool)
        or unit.sample_rate <= 0
    ):
        raise ValueError("audio unit sample_rate must be positive when known")
    if unit.channels is not None and (
        not isinstance(unit.channels, int)
        or isinstance(unit.channels, bool)
        or unit.channels <= 0
    ):
        raise ValueError("audio unit channels must be positive when known")
    if unit.sample_format is not None and unit.sample_format not in _PCM_BYTES_PER_SAMPLE:
        raise ValueError("audio unit sample_format is invalid")
    if unit.direction not in ("input", "output", "mixed", "unknown"):
        raise ValueError("audio unit direction is invalid")
    if unit.participant_id is not None and not isinstance(unit.participant_id, str):
        raise ValueError("audio unit participant_id must be a string when known")
    if not isinstance(unit.boundaries, tuple) or not all(
        isinstance(boundary, AudioBoundary) for boundary in unit.boundaries
    ):
        raise ValueError("audio unit boundaries must be an AudioBoundary tuple")

    raw_chunk = unit.chunk if isinstance(unit.chunk, bytes) else None
    if raw_chunk is not None and unit.sample_format is not None:
        bytes_per_frame = _PCM_BYTES_PER_SAMPLE[unit.sample_format] * (unit.channels or 1)
        if len(raw_chunk) % bytes_per_frame:
            raise ValueError("audio chunk PCM is not aligned to its declared format/channels")

    if all(hasattr(unit.chunk, name) for name in ("sample_rate", "channels", "sample_format", "pcm")):
        frame = unit.chunk
        if unit.sample_rate != frame.sample_rate:
            raise ValueError("audio unit sample_rate disagrees with its frame")
        if unit.channels != frame.channels:
            raise ValueError("audio unit channels disagrees with its frame")
        if unit.sample_format != frame.sample_format:
            raise ValueError("audio unit sample_format disagrees with its frame")
        if frame.t_start is not None and (
            not isinstance(frame.t_start, (int, float))
            or isinstance(frame.t_start, bool)
            or not math.isfinite(frame.t_start)
        ):
            raise ValueError("audio frame t_start must be finite when known")
        if frame.t_end is not None and (
            not isinstance(frame.t_end, (int, float))
            or isinstance(frame.t_end, bool)
            or not math.isfinite(frame.t_end)
        ):
            raise ValueError("audio frame t_end must be finite when known")
        if (
            frame.t_start is not None
            and frame.t_end is not None
            and frame.t_end < frame.t_start
        ):
            raise ValueError("audio frame t_end must be >= t_start")
        if frame.t_start is not None and not math.isclose(
            float(frame.t_start), float(unit.t_start), rel_tol=0.0, abs_tol=1e-9
        ):
            raise ValueError("audio unit t_start disagrees with its frame")
        if frame.t_end is not None and not math.isclose(
            float(frame.t_end), float(unit.t_end), rel_tol=0.0, abs_tol=1e-9
        ):
            raise ValueError("audio unit t_end disagrees with its frame")
        if frame.pcm:
            bytes_per_frame = _PCM_BYTES_PER_SAMPLE[frame.sample_format] * frame.channels
            if len(frame.pcm) % bytes_per_frame:
                raise ValueError("audio frame PCM is not aligned to its format/channels")
    return unit


class AudioWindow:
    """A bounded, ordered audio timeline used by fused/acoustic perception.

    Units are retained oldest→newest by ``(t_start, seq)`` regardless of arrival order. Pushes report
    sequence gaps, duplicates, reorder, temporal gaps/overlap, and format changes instead of silently
    flattening them. The rolling time bound is backed by count and byte caps, so a stalled consumer or
    hostile timestamp stream cannot grow memory without limit.
    """

    def __init__(
        self,
        seconds: float = 4.0,
        *,
        max_units: int = 4096,
        max_bytes: int = 16 * 1024 * 1024,
    ) -> None:
        if (
            not isinstance(seconds, (int, float))
            or isinstance(seconds, bool)
            or not math.isfinite(seconds)
            or seconds <= 0
        ):
            raise ValueError("seconds must be finite and positive")
        if not isinstance(max_units, int) or isinstance(max_units, bool) or max_units <= 0:
            raise ValueError("max_units must be a positive integer")
        if not isinstance(max_bytes, int) or isinstance(max_bytes, bool) or max_bytes <= 0:
            raise ValueError("max_bytes must be a positive integer")
        self._seconds = max(0.5, seconds)
        self._max_units = max_units
        self._max_bytes = max_bytes
        self._units: list[AudioUnit] = []
        self._bytes = 0
        self._max_seq: int | None = None
        self._seen_limit = max(4096, self._max_units * 2)
        self._seen_order: deque[int] = deque()
        self._seen_seq: set[int] = set()

    @property
    def _chunks(self) -> list[tuple[float, Any]]:
        """Compatibility view for callers that inspected the old private list."""
        return [(u.t, u.chunk) for u in self._units]

    @property
    def nbytes(self) -> int:
        return self._bytes

    @property
    def next_seq(self) -> int:
        return 0 if self._max_seq is None else self._max_seq + 1

    def push(self, t: float | AudioUnit, chunk: Any = None) -> AudioPushResult:
        if isinstance(t, AudioUnit):
            unit = t
        else:
            # Use the exact same duration/rate/channel/energy normalization as capture, replay, and
            # Percept.push_audio. In particular, a typed frame is never stored as a zero-duration
            # point with unknown energy.
            from percept.runtime.audio import normalize_audio_unit

            unit = normalize_audio_unit((float(t), chunk, None), seq=self.next_seq)
        return self.push_unit(unit)

    def _preview_unit(
        self, unit: AudioUnit
    ) -> tuple[AudioUnit, AudioPushResult, int, int]:
        """Validate and plan one insertion without mutating timeline or duplicate history."""
        unit = validate_audio_unit(unit)
        unit_bytes = _audio_nbytes(unit.chunk)
        if unit_bytes > self._max_bytes:
            raise ValueError("audio unit exceeds the window byte cap")
        if unit.seq in self._seen_seq:
            return unit, AudioPushResult(accepted=False, duplicate=True), 0, unit_bytes

        previous_max_seq = self._max_seq
        reordered = previous_max_seq is not None and unit.seq < previous_max_seq
        missing = (
            max(0, unit.seq - previous_max_seq - 1)
            if previous_max_seq is not None and unit.seq > previous_max_seq
            else 0
        )

        index = 0
        key = (unit.t_start, unit.seq)
        while index < len(self._units) and (self._units[index].t_start, self._units[index].seq) < key:
            index += 1
        previous = self._units[index - 1] if index else None
        following = self._units[index] if index < len(self._units) else None

        gap_s = 0.0
        overlap_s = 0.0
        if previous is not None:
            delta = unit.t_start - previous.t_end
            if delta > 1e-9:
                gap_s = delta
            elif delta < -1e-9:
                overlap_s = -delta
        if following is not None:
            delta = following.t_start - unit.t_end
            if delta > 1e-9:
                gap_s = max(gap_s, delta)
            elif delta < -1e-9:
                overlap_s = max(overlap_s, -delta)

        # Overlapping PCM has no implicit merge policy. Reject it before mutating the timeline; a
        # future caller can explicitly mix/trim and submit a non-overlapping replacement.
        if overlap_s:
            return (
                unit,
                AudioPushResult(
                    accepted=False,
                    reordered=reordered,
                    missing=missing,
                    gap_s=gap_s,
                    overlap_s=overlap_s,
                ),
                index,
                unit_bytes,
            )

        def _format_diff(other: AudioUnit | None) -> bool:
            return bool(
                other is not None
                and (
                    (
                        unit.sample_rate is not None
                        and other.sample_rate is not None
                        and unit.sample_rate != other.sample_rate
                    )
                    or (
                        unit.channels is not None
                        and other.channels is not None
                        and unit.channels != other.channels
                    )
                    or (
                        unit.sample_format is not None
                        and other.sample_format is not None
                        and unit.sample_format != other.sample_format
                    )
                )
            )

        format_changed = _format_diff(previous) or _format_diff(following)

        return (
            unit,
            AudioPushResult(
                accepted=True,
                reordered=reordered,
                missing=missing,
                gap_s=gap_s,
                overlap_s=overlap_s,
                format_changed=format_changed,
            ),
            index,
            unit_bytes,
        )

    def preview_unit(self, unit: AudioUnit, *, reset: bool = False) -> AudioPushResult:
        """Return the exact admission decision without mutating retained audio or sequence state."""
        if reset:
            empty = AudioWindow(
                self._seconds,
                max_units=self._max_units,
                max_bytes=self._max_bytes,
            )
            return empty.preview_unit(unit)
        return self._preview_unit(unit)[1]

    def push_unit(self, unit: AudioUnit) -> AudioPushResult:
        unit, preview, index, unit_bytes = self._preview_unit(unit)
        if not preview.accepted:
            return preview

        previous_max_seq = self._max_seq
        self._units.insert(index, unit)
        self._bytes += unit_bytes
        self._max_seq = unit.seq if previous_max_seq is None else max(previous_max_seq, unit.seq)
        self._seen_seq.add(unit.seq)
        self._seen_order.append(unit.seq)
        while len(self._seen_order) > self._seen_limit:
            forgotten = self._seen_order.popleft()
            self._seen_seq.discard(forgotten)

        newest_end = max(u.t_end for u in self._units)
        cutoff = newest_end - self._seconds
        expired = 0
        expired_bytes = 0
        while len(self._units) > 1:
            stale_index = next(
                (i for i, retained in enumerate(self._units) if retained.t_end < cutoff),
                None,
            )
            if stale_index is None:
                break
            dropped = self._units.pop(stale_index)
            size = _audio_nbytes(dropped.chunk)
            self._bytes = max(0, self._bytes - size)
            expired += 1
            expired_bytes += size

        evicted = 0
        evicted_bytes = 0
        while len(self._units) > 1 and (
            len(self._units) > self._max_units or self._bytes > self._max_bytes
        ):
            dropped = self._units.pop(0)
            size = _audio_nbytes(dropped.chunk)
            self._bytes = max(0, self._bytes - size)
            evicted += 1
            evicted_bytes += size

        return AudioPushResult(
            accepted=True,
            reordered=preview.reordered,
            missing=preview.missing,
            gap_s=preview.gap_s,
            overlap_s=preview.overlap_s,
            format_changed=preview.format_changed,
            expired=expired,
            expired_bytes=expired_bytes,
            evicted=evicted,
            evicted_bytes=evicted_bytes,
        )

    def annotate(self, seq: int, *boundaries: AudioBoundary) -> bool:
        """Attach boundary annotations to an already-retained unit without duplicating them."""
        if not isinstance(seq, int) or isinstance(seq, bool) or seq < 0:
            raise ValueError("audio annotation seq must be a non-negative integer")
        if not all(isinstance(boundary, AudioBoundary) for boundary in boundaries):
            raise ValueError("audio annotations must be AudioBoundary values")
        for index, unit in enumerate(self._units):
            if unit.seq != seq:
                continue
            merged = list(unit.boundaries)
            for boundary in boundaries:
                if boundary not in merged:
                    merged.append(boundary)
            merged.sort(key=lambda boundary: (boundary.t, boundary.kind))
            self._units[index] = replace(unit, boundaries=tuple(merged))
            return True
        return False

    def units(self) -> list[AudioUnit]:
        return list(self._units)

    def window(self, t_start: float, t_end: float) -> list[AudioUnit]:
        """Units intersecting the explicit half-open ``[t_start, t_end)`` window."""
        if (
            not isinstance(t_start, (int, float))
            or isinstance(t_start, bool)
            or not isinstance(t_end, (int, float))
            or isinstance(t_end, bool)
            or not math.isfinite(t_start)
            or not math.isfinite(t_end)
        ):
            raise ValueError("audio window times must be finite")
        if t_end < t_start:
            raise ValueError("audio window t_end must be >= t_start")
        return [
            unit
            for unit in self._units
            if (
                unit.t_end > t_start and unit.t_start < t_end
            ) or (
                unit.t_start == unit.t_end and t_start <= unit.t_start < t_end
            )
        ]

    def boundaries(
        self, t_start: float | None = None, t_end: float | None = None
    ) -> list[AudioBoundary]:
        for name, value in (("t_start", t_start), ("t_end", t_end)):
            if value is not None and (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(value)
            ):
                raise ValueError(f"audio boundary {name} must be finite")
        if t_start is not None and t_end is not None and t_end < t_start:
            raise ValueError("audio boundary t_end must be >= t_start")
        out = [boundary for unit in self._units for boundary in unit.boundaries]
        if t_start is not None:
            out = [boundary for boundary in out if boundary.t >= t_start]
        if t_end is not None:
            out = [boundary for boundary in out if boundary.t < t_end]
        return sorted(out, key=lambda boundary: (boundary.t, boundary.kind))

    def clip(self, seconds: float, *, now: float | None = None) -> list[Any]:
        """Latest contiguous, format-homogeneous audio segment within the requested duration."""
        if (
            not isinstance(seconds, (int, float))
            or isinstance(seconds, bool)
            or not math.isfinite(seconds)
            or seconds <= 0
        ):
            raise ValueError("audio clip seconds must be finite and positive")
        if now is not None and (
            not isinstance(now, (int, float))
            or isinstance(now, bool)
            or not math.isfinite(now)
        ):
            raise ValueError("audio clip now must be finite when known")
        if not self._units:
            return []
        end = max(unit.t_end for unit in self._units) if now is None else float(now)
        start = end - seconds
        # The logical clip includes audio ending exactly at ``end``. Express that as a true half-open
        # interval by advancing only its exclusive upper bound by one representable float. The lower
        # bound stays exact, so a chunk ending exactly at ``start`` is excluded.
        candidates = self.window(start, math.nextafter(end, math.inf))
        if not candidates:
            return []
        if all(
            unit.t_start == unit.t_end
            and unit.sample_rate is None
            and unit.channels is None
            and unit.sample_format is None
            for unit in candidates
        ):
            # Legacy opaque/encoded rows have no source duration to prove or disprove continuity.
            # Preserve their historical event-time clip behavior; typed PCM takes the strict path below.
            return [unit.chunk for unit in candidates]
        contiguous = [candidates[-1]]
        for unit in reversed(candidates[:-1]):
            newer = contiguous[0]
            same_format = (
                unit.sample_rate,
                unit.channels,
                unit.sample_format,
            ) == (
                newer.sample_rate,
                newer.channels,
                newer.sample_format,
            )
            if not same_format or not math.isclose(
                unit.t_end, newer.t_start, rel_tol=0.0, abs_tol=1e-9
            ):
                break
            contiguous.insert(0, unit)
        return [unit.chunk for unit in contiguous]

    def clip_units(self, seconds: float, *, now: float | None = None) -> tuple[AudioUnit, ...]:
        """The immutable units behind :meth:`clip`, retaining exact carrier provenance.

        Compiled audio dispatch uses this instead of rebuilding evidence from the rolling window
        after a provider call. The selected tuple is frozen into the judge job before dispatch.
        """
        if (
            not isinstance(seconds, (int, float))
            or isinstance(seconds, bool)
            or not math.isfinite(seconds)
            or seconds <= 0
        ):
            raise ValueError("audio clip seconds must be finite and positive")
        if now is not None and (
            not isinstance(now, (int, float))
            or isinstance(now, bool)
            or not math.isfinite(now)
        ):
            raise ValueError("audio clip now must be finite when known")
        if not self._units:
            return ()
        end = max(unit.t_end for unit in self._units) if now is None else float(now)
        candidates = self.window(end - seconds, math.nextafter(end, math.inf))
        if not candidates:
            return ()
        contiguous = [candidates[-1]]
        for unit in reversed(candidates[:-1]):
            newer = contiguous[0]
            if (
                (unit.sample_rate, unit.channels, unit.sample_format)
                != (newer.sample_rate, newer.channels, newer.sample_format)
                or unit.source_id != newer.source_id
                or unit.source_epoch != newer.source_epoch
                or not math.isclose(
                    unit.t_end, newer.t_start, rel_tol=0.0, abs_tol=1e-9
                )
            ):
                break
            contiguous.insert(0, unit)
        return tuple(contiguous)

    def clear(self) -> None:
        """Drop retained chunks at a source discontinuity."""
        self._units.clear()
        self._bytes = 0
        self._max_seq = None
        self._seen_order.clear()
        self._seen_seq.clear()

    def __len__(self) -> int:
        return len(self._units)


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


__all__ = [
    "Event",
    "FrameWindow",
    "AudioBoundary",
    "AudioUnit",
    "AudioPushResult",
    "AudioWindow",
    "validate_audio_unit",
    "EventLog",
]
