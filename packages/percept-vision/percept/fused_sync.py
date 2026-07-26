"""Internal event-time synchronization for compiled fused Watches.

The public fused surface is ``WatchPlan(sense="fused")`` driven by
``percept.run(..., source=..., audio_source=...)``.  This module deliberately
contains no public input DTO or extension registry.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from enum import Enum
from types import MappingProxyType
from typing import Any, Literal, Mapping


_FUSED_SYNC_POLICY_VERSION = "fused-sync-v1"


class _FusedDiscontinuity(str, Enum):
    NONE = "none"
    GAP = "gap"
    RECONNECT = "reconnect"
    EPOCH_GAP = "epoch_gap"
    FORMAT_CHANGE = "format_change"
    CAPTURE_OVERFLOW = "capture_overflow"
    PAUSE = "pause"
    RESUME = "resume"
    WATCH_REVISION = "watch_revision"
    CANCELLATION = "cancellation"
    WORKER_ERROR = "worker_error"


class _FusedCoverageUncertainty(str, Enum):
    MISSING_VIDEO = "missing_video"
    MISSING_AUDIO = "missing_audio"
    CLOCK_DOMAIN_MISMATCH = "clock_domain_mismatch"
    CLOCK_CONFIDENCE_INSUFFICIENT = "clock_confidence_insufficient"
    SKEW_EXCEEDED = "skew_exceeded"
    WATERMARK_PENDING = "watermark_pending"
    DISCONTINUITY = "discontinuity"
    BOUNDS_EXCEEDED = "bounds_exceeded"
    EOS_INCOMPLETE = "eos_incomplete"


@dataclass(frozen=True)
class _FusedBoundDropReceipt:
    dropped_count: int
    latest_anchor_identity: tuple[str, int, int, int]
    reason: _FusedCoverageUncertainty = _FusedCoverageUncertainty.BOUNDS_EXCEEDED


def _finite(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _immutable_payload(payload: Any) -> Any:
    if type(payload) is bytes:
        if not payload:
            raise ValueError("fused bytes payload must be non-empty")
        return payload
    if (
        type(payload) is tuple
        and len(payload) == 2
        and type(payload[0]) is str
        and bool(payload[0])
        and type(payload[1]) is bytes
        and bool(payload[1])
    ):
        return payload
    if isinstance(payload, (bytearray, memoryview)):
        frozen = bytes(payload)
        if not frozen:
            raise ValueError("fused bytes payload must be non-empty")
        return frozen
    if (
        isinstance(payload, tuple)
        and len(payload) == 2
        and isinstance(payload[1], (bytearray, memoryview))
    ):
        if type(payload[0]) is not str or not payload[0]:
            raise ValueError("fused encoded payload requires a non-empty media type")
        frozen = bytes(payload[1])
        if not frozen:
            raise ValueError("fused encoded payload must be non-empty")
        return (payload[0], frozen)
    if isinstance(payload, str):
        if not payload:
            raise ValueError("fused scripted payload must be non-empty")
        return payload
    from .contracts import AudioFrame
    from .events import AudioUnit, validate_audio_unit

    if type(payload) is AudioUnit:
        unit = validate_audio_unit(payload)
        if type(unit.chunk) is not AudioFrame:
            raise ValueError("fused audio requires an exact canonical AudioFrame carrier")
        for name in ("pcm", "encoded"):
            carrier = getattr(unit.chunk, name)
            if carrier is not None and type(carrier) is not bytes:
                raise ValueError("fused audio carriers must be exact immutable bytes")
        return unit
    raise ValueError("fused accepted payload is outside the qualified immutable carrier matrix")


def _payload_nbytes(payload: Any) -> int:
    if isinstance(payload, (bytes, bytearray, memoryview)):
        return memoryview(payload).nbytes
    if isinstance(payload, tuple) and len(payload) == 2:
        return _payload_nbytes(payload[1])
    chunk = getattr(payload, "chunk", None)
    if chunk is not None:
        return _payload_nbytes(chunk)
    total = 0
    for name in ("pcm", "encoded"):
        carrier = getattr(payload, name, None)
        if isinstance(carrier, (bytes, bytearray, memoryview)):
            total += memoryview(carrier).nbytes
    if total:
        return total
    if isinstance(payload, str):
        return len(payload.encode("utf-8"))
    raise ValueError("fused accepted payload has no exact byte measure")


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        if not all(type(key) is str and key for key in value):
            raise ValueError("fused provenance keys must be non-empty strings")
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_freeze(item) for item in value)
    if isinstance(value, (bytearray, memoryview)):
        return bytes(value)
    if value is None or type(value) in (str, bytes, bool, int):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    raise ValueError("fused provenance contains an unsupported mutable or opaque value")


@dataclass(frozen=True)
class _FusedSyncPolicy:
    """One compiled, immutable synchronization policy owned by one Watch."""

    window_s: float
    allowed_lateness_s: float = 0.25
    max_skew_s: float = 0.5
    min_clock_confidence: float = 1.0
    max_units_per_modality: int = 256
    max_bytes_per_modality: int = 16 * 1024 * 1024
    version: str = _FUSED_SYNC_POLICY_VERSION

    def __post_init__(self) -> None:
        for name in (
            "window_s",
            "allowed_lateness_s",
            "max_skew_s",
            "min_clock_confidence",
        ):
            value = getattr(self, name)
            if not _finite(value) or float(value) < 0.0:
                raise ValueError(f"fused policy {name} must be finite and non-negative")
        if not 0.0 <= self.min_clock_confidence <= 1.0:
            raise ValueError("fused policy min_clock_confidence must be within [0, 1]")
        for name in ("max_units_per_modality", "max_bytes_per_modality"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"fused policy {name} must be a positive integer")
        if type(self.version) is not str or not self.version:
            raise ValueError("fused policy version must be a non-empty string")


@dataclass(frozen=True)
class _AcceptedMediaUnit:
    """Immutable media accepted after adapter and carrier validation."""

    modality: Literal["video", "audio"]
    payload: Any
    source_id: str
    source_epoch: int
    source_sequence: int
    generation: int
    watch_application: str
    watch_revision: int
    event_start: float
    event_end: float
    observed_start: float
    observed_end: float
    clock_domain: Literal["source", "observed"]
    clock_authority: str
    clock_confidence: float
    provenance: Mapping[str, Any]
    discontinuity: _FusedDiscontinuity = _FusedDiscontinuity.NONE
    source_watermark: float | None = None
    byte_size: int | None = None

    def __post_init__(self) -> None:
        if self.modality not in ("video", "audio"):
            raise ValueError("fused unit modality must be video or audio")
        if type(self.source_id) is not str or not self.source_id:
            raise ValueError("fused unit source_id must be a non-empty string")
        if type(self.watch_application) is not str or not self.watch_application:
            raise ValueError("fused unit watch_application must be a non-empty string")
        if not isinstance(self.provenance, Mapping) or not self.provenance:
            raise ValueError("fused unit provenance must be a non-empty mapping")
        object.__setattr__(self, "provenance", _freeze(self.provenance))
        for name in (
            "source_epoch",
            "source_sequence",
            "generation",
            "watch_revision",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"fused unit {name} must be a non-negative integer")
        for name in ("event_start", "event_end", "observed_start", "observed_end"):
            if not _finite(getattr(self, name)):
                raise ValueError(f"fused unit {name} must be finite")
        if self.event_end < self.event_start:
            raise ValueError("fused unit event interval is reversed")
        if self.observed_end < self.observed_start:
            raise ValueError("fused unit observed interval is reversed")
        if self.clock_domain not in ("source", "observed"):
            raise ValueError("fused unit clock_domain must be source or observed")
        if type(self.clock_authority) is not str or not self.clock_authority:
            raise ValueError("fused unit clock_authority must be a non-empty string")
        if not _finite(self.clock_confidence) or not 0.0 <= self.clock_confidence <= 1.0:
            raise ValueError("fused unit clock_confidence must be within [0, 1]")
        if type(self.discontinuity) is not _FusedDiscontinuity:
            raise ValueError("fused unit discontinuity must be typed")
        if self.source_watermark is not None and not _finite(self.source_watermark):
            raise ValueError("fused unit source_watermark must be finite when present")
        payload = _immutable_payload(self.payload)
        object.__setattr__(self, "payload", payload)
        from .events import AudioUnit

        if self.modality == "audio":
            if type(payload) is not AudioUnit:
                raise ValueError("fused audio units require an exact canonical AudioUnit payload")
            if not (
                math.isclose(self.event_start, payload.t_start, abs_tol=1e-9)
                and math.isclose(self.event_end, payload.t_end, abs_tol=1e-9)
            ):
                raise ValueError("fused audio payload interval must match accepted event bounds")
            if (
                payload.source_id != self.source_id
                or payload.source_epoch != self.source_epoch
                or payload.source_sequence != self.source_sequence
                or payload.observed_time != self.observed_end
            ):
                raise ValueError("fused audio payload provenance must match its accepted carrier")
        elif type(payload) is AudioUnit:
            raise ValueError("fused video units cannot carry AudioUnit payloads")
        measured = _payload_nbytes(payload)
        if self.byte_size is None:
            object.__setattr__(self, "byte_size", measured)
        elif (
            not isinstance(self.byte_size, int)
            or isinstance(self.byte_size, bool)
            or self.byte_size < 0
            or self.byte_size != measured
        ):
            raise ValueError("fused unit byte_size must exactly match its immutable payload")

    @property
    def identity(self) -> tuple[str, str, str, int, int, int]:
        return (
            self.watch_application,
            self.modality,
            self.source_id,
            self.source_epoch,
            self.generation,
            self.watch_revision,
        )


@dataclass(frozen=True)
class _FusedEvidenceBundle:
    """Exact immutable evidence selected for one fused provider dispatch."""

    video_units: tuple[_AcceptedMediaUnit, ...]
    audio_units: tuple[_AcceptedMediaUnit, ...]
    video_sources: tuple[tuple[str, int, int], ...]
    audio_sources: tuple[tuple[str, int, int], ...]
    video_event_interval: tuple[float, float] | None
    audio_event_interval: tuple[float, float] | None
    video_observed_interval: tuple[float, float] | None
    audio_observed_interval: tuple[float, float] | None
    actual_skew_s: float | None
    video_watermark: float | None
    audio_watermark: float | None
    coverage_reasons: tuple[_FusedCoverageUncertainty, ...]
    watch_application: str
    watch_revision: int
    policy_version: str
    evidence_seq: int | None = None
    judgment_event_id: str | None = None

    def __post_init__(self) -> None:
        if type(self.video_units) is not tuple or type(self.audio_units) is not tuple:
            raise ValueError("fused evidence units must be exact immutable tuples")
        if any(
            type(unit) is not _AcceptedMediaUnit or unit.modality != modality
            for modality, units in (
                ("video", self.video_units),
                ("audio", self.audio_units),
            )
            for unit in units
        ):
            raise ValueError("fused evidence contains an invalid modality carrier")
        if any(
            unit.watch_application != self.watch_application
            or unit.watch_revision != self.watch_revision
            for unit in (*self.video_units, *self.audio_units)
        ):
            raise ValueError("fused evidence crosses a Watch application or revision")
        if type(self.coverage_reasons) is not tuple or any(
            type(reason) is not _FusedCoverageUncertainty
            for reason in self.coverage_reasons
        ):
            raise ValueError("fused coverage uncertainty must be an exact typed tuple")
        if len(set(self.coverage_reasons)) != len(self.coverage_reasons):
            raise ValueError("fused coverage uncertainty reasons must be unique")

    @property
    def complete(self) -> bool:
        return not self.coverage_reasons

    def with_judgment_identity(
        self, *, evidence_seq: int, judgment_event_id: str
    ) -> "_FusedEvidenceBundle":
        if (
            not isinstance(evidence_seq, int)
            or isinstance(evidence_seq, bool)
            or evidence_seq < 0
        ):
            raise ValueError("fused evidence_seq must be a non-negative integer")
        if type(judgment_event_id) is not str or not judgment_event_id:
            raise ValueError("fused judgment_event_id must be a non-empty string")
        if self.evidence_seq is not None or self.judgment_event_id is not None:
            if (
                self.evidence_seq == evidence_seq
                and self.judgment_event_id == judgment_event_id
            ):
                return self
            raise ValueError("fused bundle judgment identity is already frozen")
        return replace(
            self,
            evidence_seq=evidence_seq,
            judgment_event_id=judgment_event_id,
        )


@dataclass
class _Lane:
    units: list[_AcceptedMediaUnit]
    watermark: float | None = None
    max_event_end_seen: float | None = None
    held_bytes: int = 0
    evicted_through: float | None = None


class _FusedSynchronizer:
    """Bounded, per-Watch event-time synchronizer."""

    def __init__(
        self,
        policy: _FusedSyncPolicy,
        *,
        watch_application: str,
        watch_revision: int,
    ) -> None:
        if type(policy) is not _FusedSyncPolicy:
            raise ValueError("fused synchronizer requires an exact frozen policy")
        if type(watch_application) is not str or not watch_application:
            raise ValueError("fused synchronizer requires a Watch application")
        if (
            not isinstance(watch_revision, int)
            or isinstance(watch_revision, bool)
            or watch_revision < 0
        ):
            raise ValueError("fused synchronizer Watch revision must be non-negative")
        self.policy = policy
        self.watch_application = watch_application
        self.watch_revision = watch_revision
        self.generation = 0
        self._lanes: dict[tuple[str, str, str, int, int, int], _Lane] = {}
        self._source_identity: dict[str, tuple[str, int]] = {}
        self._accepted_sequence_frontier: dict[
            tuple[str, str, str, int, int, int], int
        ] = {}
        self._pending: list[_AcceptedMediaUnit] = []
        self._ready: list[_FusedEvidenceBundle] = []
        # Video acceptance is strictly increasing for the generation's single
        # (source_id, epoch) identity, so exact-once emission needs one scalar
        # high-water rather than an ever-growing identity set.
        self._emitted_anchor_sequence: int | None = None
        self._closed = False
        self._close_drained = False
        self._bound_drop_count = 0
        self._latest_bound_drop: tuple[str, int, int, int] | None = None
        self._generation_start_event_time: float | None = None

    def _record_bound_drop(self, bundle: _FusedEvidenceBundle) -> None:
        if bundle.video_units:
            self._bound_drop_count += 1
            self._latest_bound_drop = self._anchor_id(bundle.video_units[-1])

    def _forced_bounds_bundle(
        self,
        anchor: _AcceptedMediaUnit,
    ) -> _FusedEvidenceBundle:
        current = self._bundle_for(anchor)
        return replace(
            current,
            video_units=(anchor,),
            audio_units=(),
            video_sources=((anchor.source_id, anchor.source_epoch, anchor.generation),),
            audio_sources=(),
            video_event_interval=(anchor.event_start, anchor.event_end),
            audio_event_interval=None,
            video_observed_interval=(anchor.observed_start, anchor.observed_end),
            audio_observed_interval=None,
            coverage_reasons=tuple(
                dict.fromkeys(
                    (
                        *current.coverage_reasons,
                        _FusedCoverageUncertainty.BOUNDS_EXCEEDED,
                    )
                )
            ),
        )

    def _queue_forced_bounds(self, anchor: _AcceptedMediaUnit) -> None:
        """Retain a bounded typed degradation receipt; capture never blocks."""
        self._ready.append(self._forced_bounds_bundle(anchor))
        self._mark_anchor_emitted(anchor)
        while (
            len(self._ready) > self.policy.max_units_per_modality
            or sum(
                unit.byte_size or 0
                for bundle in self._ready
                for unit in (*bundle.video_units, *bundle.audio_units)
            )
            > self.policy.max_bytes_per_modality
        ):
            self._record_bound_drop(self._ready.pop(0))

    def _bound_anchor_retention(self) -> None:
        """Keep pending plus degradation receipts inside one compiled budget."""
        def retained_bytes() -> int:
            return sum(unit.byte_size or 0 for unit in self._pending) + sum(
                unit.byte_size or 0
                for bundle in self._ready
                for unit in (*bundle.video_units, *bundle.audio_units)
            )

        while self._ready and (
            len(self._pending) + len(self._ready)
            > self.policy.max_units_per_modality
            or retained_bytes() > self.policy.max_bytes_per_modality
        ):
            self._record_bound_drop(self._ready.pop(0))

    def drain_bound_drops(self) -> tuple[_FusedBoundDropReceipt, ...]:
        """Expose compact, typed telemetry for every bound-driven receipt drop."""
        if not self._bound_drop_count or self._latest_bound_drop is None:
            return ()
        receipt = _FusedBoundDropReceipt(
            dropped_count=self._bound_drop_count,
            latest_anchor_identity=self._latest_bound_drop,
        )
        self._bound_drop_count = 0
        self._latest_bound_drop = None
        return (receipt,)

    def accept(self, unit: _AcceptedMediaUnit) -> bool:
        """Accept an already-validated unit, or reject a late/foreign/bounded one."""
        if type(unit) is not _AcceptedMediaUnit:
            raise ValueError("fused synchronizer accepts exact _AcceptedMediaUnit values")
        if self._closed:
            return False
        if (
            unit.watch_application != self.watch_application
            or unit.watch_revision != self.watch_revision
            or unit.generation != self.generation
        ):
            return False
        source_identity = (unit.source_id, unit.source_epoch)
        prior_source = self._source_identity.get(unit.modality)
        if prior_source is not None and prior_source != source_identity:
            raise ValueError(
                f"fused synchronization supports one {unit.modality} source per generation"
            )
        if unit.byte_size is None or unit.byte_size > self.policy.max_bytes_per_modality:
            return False
        self._source_identity.setdefault(unit.modality, source_identity)
        lane = self._lanes.setdefault(unit.identity, _Lane([]))
        frontier = self._accepted_sequence_frontier.get(unit.identity)
        if frontier is not None and unit.source_sequence <= frontier:
            return False
        if lane.watermark is not None and unit.event_end < lane.watermark:
            return False
        if any(
            existing.source_sequence == unit.source_sequence for existing in lane.units
        ):
            return False
        lane.units.append(unit)
        self._accepted_sequence_frontier[unit.identity] = unit.source_sequence
        lane.units.sort(
            key=lambda item: (
                item.event_start,
                item.event_end,
                item.source_sequence,
            )
        )
        lane.held_bytes += unit.byte_size
        lane.max_event_end_seen = (
            unit.event_end
            if lane.max_event_end_seen is None
            else max(lane.max_event_end_seen, unit.event_end)
        )
        candidate_watermark = (
            unit.source_watermark
            if unit.source_watermark is not None
            else lane.max_event_end_seen - self.policy.allowed_lateness_s
        )
        lane.watermark = (
            candidate_watermark
            if lane.watermark is None
            else max(lane.watermark, candidate_watermark)
        )
        while (
            len(lane.units) > self.policy.max_units_per_modality
            or lane.held_bytes > self.policy.max_bytes_per_modality
        ):
            removed = lane.units.pop(0)
            lane.held_bytes -= removed.byte_size or 0
            lane.evicted_through = (
                removed.event_end
                if lane.evicted_through is None
                else max(lane.evicted_through, removed.event_end)
            )
        return unit in lane.units

    def set_generation_start(self, at: float) -> None:
        """Record a trusted runtime start/discontinuity coordinate exactly once."""
        if not _finite(at):
            raise ValueError("fused generation start must be finite")
        if self._generation_start_event_time is None:
            self._generation_start_event_time = float(at)
        elif not math.isclose(
            self._generation_start_event_time,
            float(at),
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError("fused generation start is already established")

    @property
    def generation_start_event_time(self) -> float | None:
        """Trusted lower coverage boundary for the current generation, if known."""
        return self._generation_start_event_time

    def advance_generation(
        self,
        reason: _FusedDiscontinuity,
        *,
        watch_revision: int | None = None,
    ) -> int:
        """Invalidate all pending evidence before an asynchronous boundary."""
        if type(reason) is not _FusedDiscontinuity or reason is _FusedDiscontinuity.NONE:
            raise ValueError("generation advance requires a typed discontinuity reason")
        if watch_revision is not None:
            if (
                not isinstance(watch_revision, int)
                or isinstance(watch_revision, bool)
                or watch_revision < 0
            ):
                raise ValueError("fused Watch revision must be non-negative")
            self.watch_revision = watch_revision
        self.generation += 1
        self._lanes.clear()
        self._source_identity.clear()
        self._accepted_sequence_frontier.clear()
        self._pending.clear()
        self._ready.clear()
        self._emitted_anchor_sequence = None
        self._closed = False
        self._close_drained = False
        self._bound_drop_count = 0
        self._latest_bound_drop = None
        self._generation_start_event_time = None
        return self.generation

    @staticmethod
    def _interval(
        units: tuple[_AcceptedMediaUnit, ...], start: str, end: str
    ) -> tuple[float, float] | None:
        if not units:
            return None
        return (min(getattr(unit, start) for unit in units), max(getattr(unit, end) for unit in units))

    @staticmethod
    def _skew(
        video: tuple[_AcceptedMediaUnit, ...],
        audio: tuple[_AcceptedMediaUnit, ...],
    ) -> float | None:
        if not video or not audio:
            return None
        anchor = video[-1]
        distances = []
        for unit in audio:
            if unit.event_start <= anchor.event_end and unit.event_end >= anchor.event_start:
                distances.append(0.0)
            elif unit.event_end < anchor.event_start:
                distances.append(anchor.event_start - unit.event_end)
            else:
                distances.append(unit.event_start - anchor.event_end)
        return min(distances)

    @staticmethod
    def _slice_audio(
        accepted: _AcceptedMediaUnit,
        start: float,
        end: float,
    ) -> _AcceptedMediaUnit | None:
        """Return the exact PCM samples in a fixed event interval."""
        unit = accepted.payload
        overlap_start = max(start, accepted.event_start)
        overlap_end = min(end, accepted.event_end)
        if overlap_end <= overlap_start:
            return None
        if overlap_start == accepted.event_start and overlap_end == accepted.event_end:
            return accepted
        from .contracts import AudioFrame
        from .events import AudioUnit

        if type(unit) is not AudioUnit or type(unit.chunk) is not AudioFrame:
            raise ValueError("fused rechunking requires exact canonical PCM AudioUnits")
        frame = unit.chunk
        if (
            type(frame.pcm) is not bytes
            or frame.encoded is not None
            or frame.sample_format != "s16le"
        ):
            raise ValueError("fused rechunking requires canonical s16le PCM")
        width = frame.channels * 2
        first = int(round((overlap_start - unit.t_start) * frame.sample_rate))
        last = int(round((overlap_end - unit.t_start) * frame.sample_rate))
        first = max(0, min(first, len(frame.pcm) // width))
        last = max(first, min(last, len(frame.pcm) // width))
        pcm = frame.pcm[first * width : last * width]
        if not pcm:
            return None
        exact_start = unit.t_start + first / frame.sample_rate
        exact_end = unit.t_start + last / frame.sample_rate
        sliced_frame = replace(
            frame,
            pcm=pcm,
            t_start=exact_start,
            t_end=exact_end,
        )
        sliced_unit = replace(
            unit,
            chunk=sliced_frame,
            t_start=exact_start,
            t_end=exact_end,
            event_t=exact_end,
        )
        return replace(
            accepted,
            payload=sliced_unit,
            event_start=exact_start,
            event_end=exact_end,
            byte_size=None,
        )

    @staticmethod
    def _continuous_through(
        units: tuple[_AcceptedMediaUnit, ...],
        start: float,
        end: float,
    ) -> bool:
        if not units:
            return False
        cursor = start
        for unit in units:
            rate = getattr(unit.payload, "sample_rate", None) or 1
            tolerance = 1.0 / float(rate)
            if unit.event_start > cursor + tolerance:
                return False
            cursor = max(cursor, unit.event_end)
            if cursor >= end - tolerance:
                return True
        return cursor >= end

    def _modality_lanes(
        self, modality: Literal["video", "audio"]
    ) -> tuple[_Lane, ...]:
        lanes = [
            lane
            for identity, lane in self._lanes.items()
            if identity[1] == modality
            and identity[4] == self.generation
            and identity[5] == self.watch_revision
        ]
        return tuple(lanes)

    @staticmethod
    def _anchor_id(anchor: _AcceptedMediaUnit) -> tuple[str, int, int, int]:
        return (
            anchor.source_id,
            anchor.source_epoch,
            anchor.generation,
            anchor.source_sequence,
        )

    def _anchor_was_emitted(self, anchor: _AcceptedMediaUnit) -> bool:
        high_water = self._emitted_anchor_sequence
        return high_water is not None and anchor.source_sequence <= high_water

    def _mark_anchor_emitted(self, anchor: _AcceptedMediaUnit) -> None:
        high_water = self._emitted_anchor_sequence
        if high_water is None or anchor.source_sequence > high_water:
            self._emitted_anchor_sequence = anchor.source_sequence

    def _bundle_for(
        self,
        anchor: _AcceptedMediaUnit,
    ) -> _FusedEvidenceBundle:
        """Build the current immutable view for one proven accepted anchor."""
        start = anchor.event_start - self.policy.window_s
        end = anchor.event_end + self.policy.max_skew_s
        video_lanes = self._modality_lanes("video")
        audio_lanes = self._modality_lanes("audio")
        video = tuple(
            sorted(
                (
                    unit
                    for lane in video_lanes
                    for unit in lane.units
                    if unit.event_end >= start and unit.event_end <= anchor.event_end
                ),
                key=lambda item: (item.event_start, item.event_end, item.source_sequence),
            )
        )
        all_audio = tuple(
            unit for lane in audio_lanes for unit in lane.units
        )
        audio_candidates = sorted(
            (
                unit
                for lane in audio_lanes
                for unit in lane.units
                if unit.event_end >= start and unit.event_start <= end
            ),
            key=lambda item: (item.event_start, item.event_end, item.source_sequence),
        )
        audio = tuple(
            sliced
            for unit in audio_candidates
            if (sliced := self._slice_audio(unit, start, end)) is not None
        )
        video_wm = min(
            (lane.watermark for lane in video_lanes if lane.watermark is not None),
            default=None,
        )
        audio_wm = min(
            (lane.watermark for lane in audio_lanes if lane.watermark is not None),
            default=None,
        )
        reasons: list[_FusedCoverageUncertainty] = []
        if not video:
            reasons.append(_FusedCoverageUncertainty.MISSING_VIDEO)
        required_audio_start = (
            max(start, self._generation_start_event_time)
            if self._generation_start_event_time is not None
            else start
        )
        required_audio_end = anchor.event_end - self.policy.max_skew_s
        audio_continuous = bool(audio) and self._continuous_through(
            audio,
            required_audio_start,
            required_audio_end,
        )
        if not audio:
            reasons.append(_FusedCoverageUncertainty.MISSING_AUDIO)
        elif not audio_continuous:
            reasons.append(_FusedCoverageUncertainty.MISSING_AUDIO)
        domains = {
            (unit.clock_domain, unit.clock_authority) for unit in (*video, *audio)
        }
        if len(domains) > 1:
            reasons.append(_FusedCoverageUncertainty.CLOCK_DOMAIN_MISMATCH)
        if any(
            unit.clock_confidence < self.policy.min_clock_confidence
            for unit in (*video, *audio)
        ):
            reasons.append(_FusedCoverageUncertainty.CLOCK_CONFIDENCE_INSUFFICIENT)
        skew = self._skew(video, all_audio)
        if skew is not None and skew > self.policy.max_skew_s:
            reasons.append(_FusedCoverageUncertainty.SKEW_EXCEEDED)
        if not self._closed and (
            video_wm is None
            or video_wm < anchor.event_end
            or audio_wm is None
            or audio_wm < end
        ):
            reasons.append(_FusedCoverageUncertainty.WATERMARK_PENDING)
        # ``advance_generation`` is the continuity proof: the first unit in the
        # new generation retains its typed reason, but no bundle can borrow media
        # from the prior generation and the marker therefore does not poison
        # otherwise complete post-boundary evidence.
        if any(
            lane.evicted_through is not None and lane.evicted_through >= start
            for lane in (*video_lanes, *audio_lanes)
        ):
            reasons.append(_FusedCoverageUncertainty.BOUNDS_EXCEEDED)
        if self._closed and (not video or not audio or not audio_continuous):
            reasons.append(_FusedCoverageUncertainty.EOS_INCOMPLETE)
        bundle = _FusedEvidenceBundle(
            video_units=video,
            audio_units=audio,
            video_sources=tuple(
                dict.fromkeys(
                    (unit.source_id, unit.source_epoch, unit.generation) for unit in video
                )
            ),
            audio_sources=tuple(
                dict.fromkeys(
                    (unit.source_id, unit.source_epoch, unit.generation) for unit in audio
                )
            ),
            video_event_interval=self._interval(video, "event_start", "event_end"),
            audio_event_interval=self._interval(audio, "event_start", "event_end"),
            video_observed_interval=self._interval(video, "observed_start", "observed_end"),
            audio_observed_interval=self._interval(audio, "observed_start", "observed_end"),
            actual_skew_s=skew,
            video_watermark=video_wm,
            audio_watermark=audio_wm,
            coverage_reasons=tuple(dict.fromkeys(reasons)),
            watch_application=self.watch_application,
            watch_revision=self.watch_revision,
            policy_version=self.policy.version,
        )
        return bundle

    def bundle_for(
        self,
        anchor: _AcceptedMediaUnit,
        *,
        evidence_seq: int | None = None,
        judgment_event_id: str | None = None,
    ) -> _FusedEvidenceBundle:
        """Freeze or pend the window around one exact accepted video anchor."""
        if type(anchor) is not _AcceptedMediaUnit or anchor.modality != "video":
            raise ValueError("fused bundle anchor must be an accepted video unit")
        if (
            anchor.watch_application != self.watch_application
            or anchor.watch_revision != self.watch_revision
            or anchor.generation != self.generation
        ):
            raise ValueError("fused bundle anchor is outside this Watch generation")
        video_lanes = self._modality_lanes("video")
        if not any(any(unit is anchor for unit in lane.units) for lane in video_lanes):
            raise ValueError("fused bundle anchor was not accepted by this synchronizer")
        bundle = self._bundle_for(anchor)
        if (
            _FusedCoverageUncertainty.WATERMARK_PENDING in bundle.coverage_reasons
            and not self._anchor_was_emitted(anchor)
            and not any(pending is anchor for pending in self._pending)
        ):
            if len(self._pending) >= self.policy.max_units_per_modality:
                oldest = self._pending.pop(0)
                self._queue_forced_bounds(oldest)
            self._pending.append(anchor)
            while sum(unit.byte_size or 0 for unit in self._pending) > self.policy.max_bytes_per_modality:
                oldest = self._pending.pop(0)
                self._queue_forced_bounds(oldest)
            self._bound_anchor_retention()
        if evidence_seq is not None or judgment_event_id is not None:
            if evidence_seq is None or judgment_event_id is None:
                raise ValueError("fused judgment identity fields must be supplied together")
            bundle = bundle.with_judgment_identity(
                evidence_seq=evidence_seq,
                judgment_event_id=judgment_event_id,
            )
        return bundle

    def drain_ready(self) -> tuple[_FusedEvidenceBundle, ...]:
        """Return newly watermark-closed pending windows exactly once."""
        ready: list[_FusedEvidenceBundle] = list(self._ready)
        self._ready.clear()
        pending: list[_AcceptedMediaUnit] = []
        for anchor in self._pending:
            bundle = self._bundle_for(anchor)
            if _FusedCoverageUncertainty.WATERMARK_PENDING in bundle.coverage_reasons:
                pending.append(anchor)
                continue
            self._mark_anchor_emitted(anchor)
            ready.append(bundle)
        self._pending = pending
        return tuple(ready)

    def close(self) -> tuple[_FusedEvidenceBundle, ...]:
        """Close all watermarks exactly once at overall end-of-stream."""
        if self._close_drained:
            return ()
        self._closed = True
        for lane in self._lanes.values():
            lane.watermark = math.inf
        bundles = tuple(self._ready) + tuple(
            self._bundle_for(anchor) for anchor in self._pending
        )
        for anchor in self._pending:
            self._mark_anchor_emitted(anchor)
        self._ready.clear()
        self._pending.clear()
        self._close_drained = True
        return bundles


__all__ = [
    "_AcceptedMediaUnit",
    "_FusedBoundDropReceipt",
    "_FusedCoverageUncertainty",
    "_FusedDiscontinuity",
    "_FusedEvidenceBundle",
    "_FusedSyncPolicy",
    "_FusedSynchronizer",
]
