"""Versioned ingress contract for live multimodal source items.

The runtime used to accept only positional ``(t, ...)`` tuples.  That shape cannot say whether ``t``
came from the source or from the process observing it, and it cannot distinguish a queue drop from a
reconnect or a decoder reset.  ``SourceEnvelope`` keeps those facts separate while retaining a legacy
tuple view for one compatibility cycle.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
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
        if not isinstance(self.media_type, str) or not self.media_type:
            raise ValueError("source format media_type must be non-empty")
        if self.sample_format is not None and not isinstance(self.sample_format, str):
            raise ValueError("source format sample_format must be a string or None")
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
    # An adapter-known break whose exact missing sample/frame count is unavailable.  The epoch must
    # still advance; this names why continuity broke without inventing a numeric gap.
    discontinuity: Literal["capture_overflow"] | None = None
    # Transitional selection for ``as_legacy`` only.  It never changes either preserved clock.
    legacy_timebase: Literal["source", "observed"] = field(default="source", repr=False, compare=False)
    # Audio provenance that must survive capture-envelope validation before ordered-unit ingestion.
    audio_direction: Literal["input", "output", "mixed", "unknown"] | None = None
    participant_id: str | None = None

    def __post_init__(self) -> None:
        """Own mutable media buffers even though callers may pass ``bytearray``/``memoryview``.

        A frozen envelope around a mutable byte buffer is not actually immutable: validation could
        approve one byte sequence and the judge could later consume another.  Normalize the media
        carriers we understand at construction time.  Opaque objects remain caller-owned by design.
        """
        payload = self.payload
        if isinstance(payload, (bytearray, memoryview)):
            object.__setattr__(self, "payload", bytes(payload))
            return
        if (
            isinstance(payload, tuple)
            and len(payload) == 2
            and isinstance(payload[1], (bytearray, memoryview))
        ):
            object.__setattr__(self, "payload", (payload[0], bytes(payload[1])))
            return
        # Avoid duck-typing arbitrary properties while normalizing an untrusted envelope.
        from percept.contracts import AudioFrame

        if isinstance(payload, AudioFrame):
            updates = {}
            if isinstance(payload.pcm, (bytearray, memoryview)):
                updates["pcm"] = bytes(payload.pcm)
            if isinstance(payload.encoded, (bytearray, memoryview)):
                updates["encoded"] = bytes(payload.encoded)
            if updates:
                try:
                    object.__setattr__(self, "payload", replace(payload, **updates))
                except (TypeError, ValueError):
                    # A duck-typed/non-dataclass payload is rejected later rather than mutated here.
                    pass

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


def _wav_container_valid(blob: bytes) -> bool:
    """Validate RIFF length and require structurally complete ``fmt `` + non-empty ``data`` chunks."""
    if len(blob) < 12 or not (blob.startswith(b"RIFF") and blob[8:12] == b"WAVE"):
        return False
    if int.from_bytes(blob[4:8], "little") + 8 != len(blob):
        return False
    offset = 12
    saw_format = False
    saw_audio = False
    while offset < len(blob):
        if offset + 8 > len(blob):
            return False
        chunk_id = blob[offset:offset + 4]
        chunk_size = int.from_bytes(blob[offset + 4:offset + 8], "little")
        content_start = offset + 8
        content_end = content_start + chunk_size
        if content_end > len(blob):
            return False
        if chunk_id == b"fmt ":
            if chunk_size < 16:
                return False
            saw_format = True
        elif chunk_id == b"data":
            if chunk_size <= 0:
                return False
            saw_audio = True
        offset = content_end + (chunk_size & 1)
    return offset == len(blob) and saw_format and saw_audio


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
        encoded = bytes(payload[1])
        media_type = payload[0].lower()
        # Validate complete container signatures, not merely a JPEG prefix.  Unknown encoded image
        # formats fail closed; callers with a specialist payload can use the explicit opaque format.
        if media_type in ("image/jpeg", "image/jpg"):
            if len(encoded) < 4 or not (
                encoded.startswith(b"\xff\xd8") and encoded.endswith(b"\xff\xd9")
            ):
                return False
        elif media_type == "image/png":
            if (
                len(encoded) < 20
                or not encoded.startswith(b"\x89PNG\r\n\x1a\n")
                or encoded[-12:-8] != b"\x00\x00\x00\x00"
                or encoded[-8:-4] != b"IEND"
            ):
                return False
        elif media_type == "image/webp":
            if len(encoded) < 12 or not (
                encoded.startswith(b"RIFF") and encoded[8:12] == b"WEBP"
            ):
                return False
            if int.from_bytes(encoded[4:8], "little") + 8 != len(encoded):
                return False
        elif media_type == "image/gif":
            gif_signatures = (
                bytes.fromhex("474946383761"),
                bytes.fromhex("474946383961"),
            )
            if not encoded.startswith(gif_signatures) or not encoded.endswith(b";"):
                return False
        else:
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
    sample_format = getattr(frame, "sample_format", None)
    if not isinstance(sample_format, str):
        return False
    sample_width = {"s16le": 2, "f32le": 4, "u8": 1}.get(sample_format)
    if pcm and (
        sample_width is None
        or len(pcm) % (getattr(frame, "channels") * sample_width) != 0
    ):
        return False
    media_type = envelope.format.media_type.lower()
    if media_type in ("audio/pcm", "audio/l16"):
        if not pcm:
            return False
    elif encoded:
        mime = getattr(frame, "mime", None)
        if not isinstance(mime, str) or mime.lower() != media_type:
            return False
        blob = bytes(encoded)
        if media_type in ("audio/wav", "audio/x-wav"):
            if not _wav_container_valid(blob):
                return False
        elif media_type == "audio/flac":
            if not blob.startswith(b"fLaC"):
                return False
        elif media_type in ("audio/ogg", "application/ogg"):
            if not blob.startswith(b"OggS"):
                return False
        else:
            return False
    else:
        return False
    energy = envelope.signal
    return energy is None or (_finite(energy) and 0.0 <= float(energy) <= 1.0)


def _validation_event(envelope: Any) -> str | None:
    if not isinstance(envelope, SourceEnvelope):
        return "source_corrupt_payload"
    if type(envelope.version) is not int or envelope.version != SOURCE_ENVELOPE_VERSION:
        return "source_version_unsupported"
    if (
        not isinstance(envelope.source_id, str)
        or not envelope.source_id
        or envelope.modality not in ("video", "audio")
        or not isinstance(envelope.format, SourceFormat)
        or envelope.discontinuity not in (None, "capture_overflow")
        or envelope.audio_direction not in (None, "input", "output", "mixed", "unknown")
        or (
            envelope.participant_id is not None
            and not isinstance(envelope.participant_id, str)
        )
        or (
            envelope.modality != "audio"
            and (envelope.audio_direction is not None or envelope.participant_id is not None)
        )
    ):
        return "source_corrupt_payload"
    fmt = envelope.format
    if (
        not isinstance(fmt.media_type, str)
        or not fmt.media_type
        or (
            fmt.sample_format is not None
            and not isinstance(fmt.sample_format, str)
        )
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
    try:
        if envelope.modality == "video":
            return None if _video_payload_valid(envelope) else "source_corrupt_payload"
        return None if _audio_payload_valid(envelope) else "source_corrupt_payload"
    except Exception:  # noqa: BLE001 - untrusted payload validation is a fail-closed boundary
        return "source_corrupt_payload"


class SourceTracker:
    """Validate ordering/clock/format continuity for all ingress streams.

    Invalid, duplicate, reordered and timestamp-reset items are excluded without advancing state.
    Sequence gaps, reconnects and format changes accept the current item but explicitly break temporal
    continuity.  A timestamp reset is accepted only when accompanied by a new epoch.
    """

    def __init__(self) -> None:
        self._streams: dict[tuple[str, SourceModality], _StreamState] = {}
        self.coverage = SourceCoverage()

    def preview(self, envelope: Any, *, require_source_time: bool = False) -> SourceInspection:
        """Return the exact ingress decision without mutating stream state or coverage."""
        probe = SourceTracker()
        probe._streams = dict(self._streams)
        return probe.inspect(envelope, require_source_time=require_source_time)

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
        if envelope.discontinuity == "capture_overflow":
            events.append("source_capture_overflow")

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
                if envelope.discontinuity != "capture_overflow":
                    events.append("source_reconnect")
                if envelope.epoch > previous.epoch + 1:
                    events.append("source_epoch_gap")
            if envelope.discontinuity is not None and envelope.epoch == previous.epoch:
                return self._record(False, ("source_corrupt_payload",), "excluded")
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
                "source_capture_overflow",
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


def audio_unit_envelope(
    unit: Any,
    *,
    source_id: str,
    sequence: int,
    observed_time: float,
) -> SourceEnvelope:
    """Wrap an ordered unit in an observation-only carrier envelope.

    The envelope ``sequence`` is arrival order and belongs to ``SourceTracker``. ``unit.seq`` and its
    source interval remain independent and belong to ``AudioWindow``; no transport source clock is
    fabricated from them.
    """
    from percept.events import AudioUnit

    if not isinstance(unit, AudioUnit):
        raise TypeError("audio unit envelope input must be an AudioUnit")
    return SourceEnvelope(
        source_id=source_id,
        modality="audio",
        epoch=0,
        sequence=sequence,
        observed_time=observed_time,
        source_time=None,
        # The carrier deliberately does not duplicate AudioUnit media/format authority. Ordered
        # interval, format, duplicate, and reorder decisions remain transactional in AudioWindow.
        format=SourceFormat(media_type="application/x-percept-opaque"),
        payload=unit.chunk,
        signal=unit.energy,
        audio_direction=unit.direction,
        participant_id=unit.participant_id,
        legacy_timebase="observed",
    )


__all__ = [
    "SOURCE_ENVELOPE_VERSION",
    "SourceCoverage",
    "SourceCoverageEffect",
    "SourceEnvelope",
    "SourceFormat",
    "SourceInspection",
    "SourceTracker",
    "audio_unit_envelope",
    "legacy_audio_envelope",
    "legacy_video_envelope",
]
