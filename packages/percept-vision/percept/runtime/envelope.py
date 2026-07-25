"""Versioned ingress contract for live multimodal source items.

The runtime used to accept only positional ``(t, ...)`` tuples.  That shape cannot say whether ``t``
came from the source or from the process observing it, and it cannot distinguish a queue drop from a
reconnect or a decoder reset.  ``SourceEnvelope`` keeps those facts separate while retaining a legacy
tuple view for one compatibility cycle.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal

SOURCE_ENVELOPE_VERSION = 1
SourceModality = Literal["video", "audio"]
CoverageKind = Literal["observed", "observed_after_discontinuity", "excluded"]


@dataclass(frozen=True)
class SourceFormat:
    """Stable payload format fields used to detect a decoder/device format change."""

    media_type: str
    width: int | None = None
    height: int | None = None
    sample_rate: int | None = None
    channels: int | None = None
    sample_format: str | None = None

    def __post_init__(self) -> None:
        if not self.media_type:
            raise ValueError("source format media_type must be non-empty")
        for name in ("width", "height", "sample_rate", "channels"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, int) or isinstance(value, bool) or value <= 0):
                raise ValueError(f"source format {name} must be a positive integer or None")


@dataclass(frozen=True)
class SourceEnvelope:
    """One captured or pushed source item with independent source and observation clocks.

    ``sequence`` is contiguous within an ``epoch`` and advances before a bounded transport queue is
    entered, so a queue eviction becomes an observable sequence gap.  A reconnect starts the next
    epoch at sequence zero.  ``source_time`` is optional: adapters must leave it ``None`` when the
    source did not provide or intrinsically measure a timestamp.  ``observed_time`` is always the
    local monotonic time at which the item was captured or received.

    ``payload`` is the judge input.  ``signal`` is the cheap admission input (a gate ``Frame`` for
    video, normalized energy for audio).  The envelope owns no media bytes beyond those existing
    runtime primitives.
    """

    source_id: str
    modality: SourceModality
    epoch: int
    sequence: int
    observed_time: float
    source_time: float | None
    format: SourceFormat
    payload: Any
    signal: Any = None
    version: int = SOURCE_ENVELOPE_VERSION
    # Transitional selection for ``as_legacy`` only.  It never changes either preserved clock.
    legacy_timebase: Literal["source", "observed"] = field(default="source", repr=False, compare=False)

    @property
    def event_time(self) -> float:
        """Timestamp selected by the source adapter's legacy timebase.

        A source-time view is unavailable when the source supplied no time; callers needing media time
        must fail closed rather than silently substitute observation time.
        """
        if self.legacy_timebase == "observed":
            return self.observed_time
        if self.source_time is None:
            raise ValueError("source time is unknown")
        return self.source_time

    def as_legacy(self) -> tuple:
        if self.modality == "video":
            return (self.event_time, self.signal, self.payload)
        return (self.event_time, self.payload, self.signal)

    def __iter__(self):
        return iter(self.as_legacy())

    def __getitem__(self, index):
        return self.as_legacy()[index]

    def __len__(self) -> int:
        return 3


@dataclass(frozen=True)
class SourceCoverageEffect:
    """Exact coverage consequence of inspecting one envelope."""

    kind: CoverageKind
    missing_sequences: int = 0


@dataclass(frozen=True)
class SourceInspection:
    """Fail-closed ingress decision plus every detected discontinuity."""

    accepted: bool
    events: tuple[str, ...]
    coverage: SourceCoverageEffect

    @property
    def resets_continuity(self) -> bool:
        return self.accepted and self.coverage.kind == "observed_after_discontinuity"


@dataclass
class SourceCoverage:
    """Run-local accounting of the exact effects returned by :class:`SourceTracker`."""

    observed: int = 0
    excluded: int = 0
    discontinuities: int = 0
    missing_sequences: int = 0

    def record(self, inspection: SourceInspection) -> None:
        if inspection.accepted:
            self.observed += 1
        else:
            self.excluded += 1
        if inspection.coverage.kind == "observed_after_discontinuity":
            self.discontinuities += 1
        self.missing_sequences += inspection.coverage.missing_sequences

    def as_dict(self) -> dict[str, int]:
        return {
            "observed": self.observed,
            "excluded": self.excluded,
            "discontinuities": self.discontinuities,
            "missing_sequences": self.missing_sequences,
        }


@dataclass
class _StreamState:
    epoch: int
    sequence: int
    observed_time: float
    source_time: float | None
    format: SourceFormat


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _video_payload_valid(envelope: SourceEnvelope) -> bool:
    payload = envelope.payload
    if envelope.format.media_type == "application/x-percept-opaque":
        if payload is None:
            return False
    else:
        if (
            not isinstance(payload, tuple)
            or len(payload) != 2
            or not isinstance(payload[0], str)
            or not payload[0]
            or not isinstance(payload[1], (bytes, bytearray, memoryview))
            or not payload[1]
        ):
            return False
        if payload[0] != envelope.format.media_type:
            return False
        if payload[0].lower() == "image/jpeg":
            encoded = bytes(payload[1])
            if len(encoded) < 4 or not encoded.startswith(b"\xff\xd8"):
                return False
    frame = envelope.signal
    width, height = getattr(frame, "w", None), getattr(frame, "h", None)
    pixels = getattr(frame, "pixels", None)
    return (
        isinstance(width, int)
        and width > 0
        and isinstance(height, int)
        and height > 0
        and pixels is not None
        and len(pixels) == width * height
    )


def _audio_payload_valid(envelope: SourceEnvelope) -> bool:
    frame = envelope.payload
    if envelope.format.media_type == "application/x-percept-opaque":
        if frame is None:
            return False
        energy = envelope.signal
        return energy is None or (_finite(energy) and 0.0 <= float(energy) <= 1.0)
    pcm, encoded = getattr(frame, "pcm", None), getattr(frame, "encoded", None)
    if not (isinstance(pcm, (bytes, bytearray, memoryview)) or pcm is None):
        return False
    if not (isinstance(encoded, (bytes, bytearray, memoryview)) or encoded is None):
        return False
    if not (pcm or encoded):
        return False
    for name in ("sample_rate", "channels"):
        value = getattr(frame, name, None)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            return False
    if getattr(frame, "sample_rate", None) != envelope.format.sample_rate:
        return False
    if getattr(frame, "channels", None) != envelope.format.channels:
        return False
    if getattr(frame, "sample_format", None) != envelope.format.sample_format:
        return False
    sample_width = {"s16le": 2, "f32le": 4, "u8": 1}.get(
        getattr(frame, "sample_format", None)
    )
    if pcm and (
        sample_width is None
        or len(pcm) % (getattr(frame, "channels") * sample_width) != 0
    ):
        return False
    energy = envelope.signal
    return energy is None or (_finite(energy) and 0.0 <= float(energy) <= 1.0)


def _validation_event(envelope: Any) -> str | None:
    if not isinstance(envelope, SourceEnvelope):
        return "source_corrupt_payload"
    if envelope.version != SOURCE_ENVELOPE_VERSION:
        return "source_version_unsupported"
    if (
        not isinstance(envelope.source_id, str)
        or not envelope.source_id
        or envelope.modality not in ("video", "audio")
        or not isinstance(envelope.format, SourceFormat)
    ):
        return "source_corrupt_payload"
    if (
        not isinstance(envelope.epoch, int)
        or isinstance(envelope.epoch, bool)
        or envelope.epoch < 0
        or not isinstance(envelope.sequence, int)
        or isinstance(envelope.sequence, bool)
        or envelope.sequence < 0
        or not _finite(envelope.observed_time)
        or float(envelope.observed_time) < 0.0
        or (envelope.source_time is not None and not _finite(envelope.source_time))
        or (envelope.source_time is not None and float(envelope.source_time) < 0.0)
    ):
        return "source_corrupt_payload"
    if envelope.modality == "video":
        return None if _video_payload_valid(envelope) else "source_corrupt_payload"
    return None if _audio_payload_valid(envelope) else "source_corrupt_payload"


class SourceTracker:
    """Validate ordering/clock/format continuity for all ingress streams.

    Invalid, duplicate, reordered and timestamp-reset items are excluded without advancing state.
    Sequence gaps, reconnects and format changes accept the current item but explicitly break temporal
    continuity.  A timestamp reset is accepted only when accompanied by a new epoch.
    """

    def __init__(self) -> None:
        self._streams: dict[tuple[str, SourceModality], _StreamState] = {}
        self.coverage = SourceCoverage()

    def inspect(self, envelope: Any, *, require_source_time: bool = False) -> SourceInspection:
        invalid = _validation_event(envelope)
        if invalid is not None:
            return self._record(False, (invalid,), "excluded")
        if require_source_time and envelope.source_time is None:
            return self._record(False, ("source_time_unknown",), "excluded")
        key = (envelope.source_id, envelope.modality)
        previous = self._streams.get(key)
        events: list[str] = []
        missing = 0

        if previous is None and envelope.sequence > 0:
            missing = envelope.sequence
            events.append("source_gap")
        elif previous is not None:
            if envelope.epoch < previous.epoch:
                return self._record(False, ("source_reorder",), "excluded")
            if envelope.epoch == previous.epoch:
                if envelope.sequence == previous.sequence:
                    return self._record(False, ("source_duplicate",), "excluded")
                if envelope.sequence < previous.sequence:
                    return self._record(False, ("source_reorder",), "excluded")
                if envelope.observed_time < previous.observed_time:
                    return self._record(False, ("source_observed_time_reset",), "excluded")
                if envelope.source_time is not None and previous.source_time is not None:
                    if envelope.source_time < previous.source_time:
                        return self._record(False, ("source_timestamp_reset",), "excluded")
                if envelope.sequence > previous.sequence + 1:
                    missing = envelope.sequence - previous.sequence - 1
                    events.append("source_gap")
            else:
                if envelope.sequence != 0:
                    return self._record(False, ("source_corrupt_payload",), "excluded")
                if envelope.observed_time < previous.observed_time:
                    return self._record(False, ("source_observed_time_reset",), "excluded")
                events.append("source_reconnect")
                if envelope.epoch > previous.epoch + 1:
                    events.append("source_epoch_gap")
            if envelope.format != previous.format:
                events.append("source_format_change")

        if envelope.source_time is None:
            events.append("source_time_unknown")

        self._streams[key] = _StreamState(
            epoch=envelope.epoch,
            sequence=envelope.sequence,
            observed_time=envelope.observed_time,
            source_time=envelope.source_time,
            format=envelope.format,
        )
        discontinuity = any(
            event in {
                "source_gap",
                "source_reconnect",
                "source_epoch_gap",
                "source_format_change",
            }
            for event in events
        )
        kind: CoverageKind = "observed_after_discontinuity" if discontinuity else "observed"
        return self._record(True, tuple(events), kind, missing)

    def exclude(self, *events: str) -> SourceInspection:
        """Record a consumer-level exclusion (for example, required media time was unknown)."""
        return self._record(False, tuple(events), "excluded")

    def _record(
        self,
        accepted: bool,
        events: tuple[str, ...],
        kind: CoverageKind,
        missing: int = 0,
    ) -> SourceInspection:
        result = SourceInspection(accepted, events, SourceCoverageEffect(kind, missing))
        self.coverage.record(result)
        return result


def legacy_video_envelope(
    item: tuple,
    *,
    source_id: str,
    sequence: int,
    observed_time: float,
    timebase: Literal["source", "observed"] = "source",
) -> SourceEnvelope:
    """Wrap a legacy video tuple without synthesizing a clock the tuple did not carry.

    ``timebase`` declares whether the tuple's explicit ``t`` is source or observation time.  The other
    clock remains independently unknown/local; it is never derived from a reported rate.
    """
    if not isinstance(item, tuple) or len(item) != 3:
        raise ValueError("legacy video source item must be a (t, gate_frame, judge_frame) tuple")
    item_time, gate_frame, judge_frame = item
    return SourceEnvelope(
        source_id=source_id,
        modality="video",
        epoch=0,
        sequence=sequence,
        observed_time=float(item_time) if timebase == "observed" else observed_time,
        source_time=None if timebase == "observed" else item_time,
        format=SourceFormat(media_type="application/x-percept-opaque"),
        payload=judge_frame,
        signal=gate_frame,
        legacy_timebase=timebase,
    )


def legacy_audio_envelope(
    item: tuple,
    *,
    source_id: str,
    sequence: int,
    observed_time: float,
    timebase: Literal["source", "observed"] = "source",
) -> SourceEnvelope:
    """Wrap a legacy audio tuple while preserving its declared clock and never inventing the other."""
    if not isinstance(item, tuple) or len(item) != 3:
        raise ValueError("legacy audio source item must be a (t, AudioFrame, energy) tuple")
    item_time, frame, energy = item
    return SourceEnvelope(
        source_id=source_id,
        modality="audio",
        epoch=0,
        sequence=sequence,
        observed_time=float(item_time) if timebase == "observed" else observed_time,
        source_time=None if timebase == "observed" else item_time,
        format=SourceFormat(media_type="application/x-percept-opaque"),
        payload=frame,
        signal=energy,
        legacy_timebase=timebase,
    )


__all__ = [
    "SOURCE_ENVELOPE_VERSION",
    "SourceCoverage",
    "SourceCoverageEffect",
    "SourceEnvelope",
    "SourceFormat",
    "SourceInspection",
    "SourceTracker",
    "legacy_audio_envelope",
    "legacy_video_envelope",
]
