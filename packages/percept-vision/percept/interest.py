"""Deterministic video-interest intervals for programmable Watches over live multimodal streams.

This module turns a scalar, timestamped interest signal into a small typed event stream:
``interest.entered`` -> zero or more ``interest.updated`` -> exactly one ``interest.exited``.
It is deliberately below semantic judgment.  A tier-0 motion score can describe *where the
admission signal was active*; it cannot claim that a Watch's plain-language condition was true.

The engine is stdlib-only, replay deterministic, and fail closed:

* interval ids are content-derived (never random or wall-clock-derived);
* input times are finite and monotonic, and interval bounds never run backwards;
* regions must be valid normalized boxes;
* updates/exits can only be emitted for an open interval;
* hysteresis and consecutive-sample debounce are explicit configuration;
* expiry, reconnect and end-of-stream close an open interval with distinct reasons.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Literal, Sequence

from .adapters import Box

InterestEventKind = Literal[
    "interest.entered",
    "interest.updated",
    "interest.exited",
]
InterestExitReason = Literal["signal_lost", "expired", "reconnect", "end_of_stream"]

INTEREST_ENTERED: InterestEventKind = "interest.entered"
INTEREST_UPDATED: InterestEventKind = "interest.updated"
INTEREST_EXITED: InterestEventKind = "interest.exited"


def _finite(name: str, value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return 0.0 if result == 0.0 else result


def _unit(name: str, value: Any) -> float:
    result = _finite(name, value)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {result}")
    return result


def _identity(name: str, value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _validated_regions(regions: Sequence[Box]) -> tuple[Box, ...]:
    clean: list[Box] = []
    for index, region in enumerate(regions):
        if not isinstance(region, Box):
            raise ValueError(f"regions[{index}] must be a Box")
        values = tuple(
            _finite(f"regions[{index}].{name}", getattr(region, name))
            for name in ("x", "y", "w", "h")
        )
        x, y, w, h = values
        if x < 0.0 or y < 0.0 or w <= 0.0 or h <= 0.0 or x + w > 1.0 or y + h > 1.0:
            raise ValueError(
                f"regions[{index}] must be a positive normalized box contained in [0, 1]"
            )
        clean.append(Box(x=x, y=y, w=w, h=h))
    # Canonical order means a detector returning the same regions in a different order replays
    # byte-for-byte.  Duplicates carry no information and are removed.
    return tuple(sorted(set(clean), key=lambda box: (box.x, box.y, box.w, box.h)))


def _validated_tracks(track_ids: Sequence[str]) -> tuple[str, ...]:
    clean = {_identity(f"track_ids[{index}]", value) for index, value in enumerate(track_ids)}
    return tuple(sorted(clean))


@dataclass(frozen=True)
class InterestIntervalConfig:
    """Mechanical interval policy; thresholds must be selected for the caller's signal.

    ``enter_threshold`` must be greater than ``exit_threshold`` (hysteresis).
    ``enter_debounce``/``exit_debounce`` are consecutive sample counts.  ``expiry_s`` splits an
    interval after that much *observed-clock* silence; ``None`` disables automatic expiry.
    No default quality threshold is embedded in this type.
    """

    enter_threshold: float
    exit_threshold: float
    enter_debounce: int = 1
    exit_debounce: int = 1
    expiry_s: float | None = None

    def __post_init__(self) -> None:
        enter = _unit("enter_threshold", self.enter_threshold)
        exit_ = _unit("exit_threshold", self.exit_threshold)
        if exit_ >= enter:
            raise ValueError("exit_threshold must be lower than enter_threshold")
        if (
            not isinstance(self.enter_debounce, int)
            or isinstance(self.enter_debounce, bool)
            or self.enter_debounce < 1
        ):
            raise ValueError("enter_debounce must be an integer >= 1")
        if (
            not isinstance(self.exit_debounce, int)
            or isinstance(self.exit_debounce, bool)
            or self.exit_debounce < 1
        ):
            raise ValueError("exit_debounce must be an integer >= 1")
        if self.expiry_s is not None and _finite("expiry_s", self.expiry_s) <= 0.0:
            raise ValueError("expiry_s must be > 0 when set")


@dataclass(frozen=True)
class InterestSample:
    """One interest observation on both the source and observer clocks.

    ``confidence=None`` means the producer has no calibrated confidence signal; the engine never
    manufactures one from a salience score.  ``coverage`` describes this observation's usable
    source coverage, not correctness.
    """

    source_at: float
    observed_at: float
    score: float
    reason: str
    confidence: float | None = None
    coverage: float = 1.0
    regions: tuple[Box, ...] = ()
    track_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_at", _finite("source_at", self.source_at))
        object.__setattr__(self, "observed_at", _finite("observed_at", self.observed_at))
        object.__setattr__(self, "score", _unit("score", self.score))
        object.__setattr__(self, "reason", _identity("reason", self.reason))
        if self.confidence is not None:
            object.__setattr__(self, "confidence", _unit("confidence", self.confidence))
        object.__setattr__(self, "coverage", _unit("coverage", self.coverage))
        object.__setattr__(self, "regions", _validated_regions(self.regions))
        object.__setattr__(self, "track_ids", _validated_tracks(self.track_ids))


@dataclass(frozen=True)
class InterestIntervalEvent:
    """A typed, immutable interval lifecycle event.

    Bounds are conservative observed evidence bounds.  ``event_observed_at`` is when the state
    transition became known, which can be later than ``observed_end`` after exit debounce, expiry,
    reconnect or EOS.  ``source_end`` is never extrapolated when the source is disconnected.
    """

    kind: InterestEventKind
    interval_id: str
    source: str
    source_start: float
    source_end: float
    observed_start: float
    observed_end: float
    event_observed_at: float
    reason: str
    score: float
    confidence: float | None
    coverage: float
    regions: tuple[Box, ...]
    track_ids: tuple[str, ...]
    admission_id: str
    detector_id: str

    def __post_init__(self) -> None:
        if self.kind not in (INTEREST_ENTERED, INTEREST_UPDATED, INTEREST_EXITED):
            raise ValueError(f"unknown interest event kind {self.kind!r}")
        object.__setattr__(self, "interval_id", _identity("interval_id", self.interval_id))
        object.__setattr__(self, "source", _identity("source", self.source))
        for name in (
            "source_start",
            "source_end",
            "observed_start",
            "observed_end",
            "event_observed_at",
        ):
            object.__setattr__(self, name, _finite(name, getattr(self, name)))
        if self.source_end < self.source_start:
            raise ValueError("source_end must be >= source_start")
        if self.observed_end < self.observed_start:
            raise ValueError("observed_end must be >= observed_start")
        if self.event_observed_at < self.observed_end:
            raise ValueError("event_observed_at must be >= observed_end")
        object.__setattr__(self, "reason", _identity("reason", self.reason))
        object.__setattr__(self, "score", _unit("score", self.score))
        if self.confidence is not None:
            object.__setattr__(self, "confidence", _unit("confidence", self.confidence))
        object.__setattr__(self, "coverage", _unit("coverage", self.coverage))
        object.__setattr__(self, "regions", _validated_regions(self.regions))
        object.__setattr__(self, "track_ids", _validated_tracks(self.track_ids))
        object.__setattr__(self, "admission_id", _identity("admission_id", self.admission_id))
        object.__setattr__(self, "detector_id", _identity("detector_id", self.detector_id))

    @property
    def duration(self) -> float:
        """Observed-clock interval duration, always non-negative."""
        return self.observed_end - self.observed_start

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-ready envelope with stable list ordering."""
        return {
            "kind": self.kind,
            "interval_id": self.interval_id,
            "source": self.source,
            "source_start": self.source_start,
            "source_end": self.source_end,
            "observed_start": self.observed_start,
            "observed_end": self.observed_end,
            "event_observed_at": self.event_observed_at,
            "reason": self.reason,
            "score": self.score,
            "confidence": self.confidence,
            "coverage": self.coverage,
            "regions": [
                {"x": box.x, "y": box.y, "w": box.w, "h": box.h}
                for box in self.regions
            ],
            "track_ids": list(self.track_ids),
            "admission_id": self.admission_id,
            "detector_id": self.detector_id,
        }


@dataclass
class _OpenInterval:
    interval_id: str
    source_start: float
    source_end: float
    observed_start: float
    observed_end: float
    last_sample: InterestSample
    regions: tuple[Box, ...]
    track_ids: tuple[str, ...]


class InterestIntervalEngine:
    """One-source deterministic hysteresis/debounce state machine.

    An engine is pinned to one ``(source, admission_id, detector_id)`` identity.  Construct one per
    scalar signal.  ``observe`` may emit an expiry exit followed by a new enter on the same call;
    that is the explicit split policy when a gap reaches ``expiry_s``.  Shorter gaps merge.
    """

    def __init__(
        self,
        *,
        source: str,
        admission_id: str,
        detector_id: str,
        config: InterestIntervalConfig,
    ) -> None:
        self.source = _identity("source", source)
        self.admission_id = _identity("admission_id", admission_id)
        self.detector_id = _identity("detector_id", detector_id)
        if not isinstance(config, InterestIntervalConfig):
            raise TypeError("config must be an InterestIntervalConfig")
        self.config = config
        self._open: _OpenInterval | None = None
        self._entering: list[InterestSample] = []
        self._exit_count = 0
        self._last_observed_at: float | None = None
        self._last_sample_observed_at: float | None = None
        self._last_source_at: float | None = None
        self._ordinal = 0
        self._ended = False

    @property
    def interval_id(self) -> str | None:
        """The currently open interval id, if any."""
        return None if self._open is None else self._open.interval_id

    @property
    def active(self) -> bool:
        return self._open is not None

    def observe(self, sample: InterestSample) -> list[InterestIntervalEvent]:
        """Consume one validated sample and return zero or more lifecycle events."""
        if not isinstance(sample, InterestSample):
            raise TypeError("sample must be an InterestSample")
        if self._ended:
            raise RuntimeError("cannot observe after end_of_stream; call reconnect() first")
        self._validate_monotonic(sample)

        events = self._expire_before(sample.observed_at)
        self._last_observed_at = sample.observed_at
        self._last_sample_observed_at = sample.observed_at
        self._last_source_at = sample.source_at

        if self._open is None:
            if sample.score >= self.config.enter_threshold:
                self._entering.append(sample)
                if len(self._entering) >= self.config.enter_debounce:
                    events.append(self._enter(sample))
            else:
                self._entering.clear()
            return events

        if sample.score <= self.config.exit_threshold:
            self._exit_count += 1
            if self._exit_count >= self.config.exit_debounce:
                events.append(
                    self._exit(
                        reason="signal_lost",
                        event_observed_at=sample.observed_at,
                        closing_sample=sample,
                    )
                )
            return events

        self._exit_count = 0
        self._extend(sample)
        events.append(self._event(INTEREST_UPDATED, sample.reason, sample.observed_at, sample))
        return events

    def advance(self, *, observed_at: float) -> list[InterestIntervalEvent]:
        """Advance only the observer clock, firing expiry exactly at/after its deadline."""
        now = _finite("observed_at", observed_at)
        if self._ended:
            return []
        if self._last_observed_at is not None and now < self._last_observed_at:
            raise ValueError("observed_at must be monotonic")
        events = self._expire_before(now)
        self._last_observed_at = now
        return events

    def reconnect(
        self,
        *,
        observed_at: float,
        source_at: float | None = None,
    ) -> list[InterestIntervalEvent]:
        """Close the current interval as ``reconnect`` and start a new source-time epoch.

        Reconnect is an explicit split: even an immediately recurring signal receives a new stable
        interval id.  Observer time must remain monotonic; source time may restart afterwards.
        """
        now = _finite("observed_at", observed_at)
        self._validate_control_time(now)
        events = self._close_control("reconnect", now, source_at)
        self._reset_candidates()
        self._last_observed_at = now
        self._last_sample_observed_at = None
        self._last_source_at = None
        self._ended = False
        return events

    def end_of_stream(
        self,
        *,
        observed_at: float | None = None,
        source_at: float | None = None,
    ) -> list[InterestIntervalEvent]:
        """Terminally close an open interval as ``end_of_stream``.

        Pending, not-yet-entered debounce is discarded without an orphan exit.  Repeated EOS is
        idempotent.  To consume a new stream epoch afterwards, call :meth:`reconnect`.
        """
        if self._ended:
            return []
        now = (
            self._last_observed_at
            if observed_at is None
            else _finite("observed_at", observed_at)
        )
        if now is None:
            self._ended = True
            self._reset_candidates()
            return []
        self._validate_control_time(now)
        events = self._close_control("end_of_stream", now, source_at)
        self._reset_candidates()
        self._last_observed_at = now
        self._ended = True
        return events

    def expire(self, *, observed_at: float) -> list[InterestIntervalEvent]:
        """Explicitly close an open interval as ``expired`` at ``observed_at``.

        Unlike :meth:`advance`, this method does not require ``config.expiry_s`` and is useful when
        an owning Watch expires or is disarmed.
        """
        now = _finite("observed_at", observed_at)
        self._validate_control_time(now)
        events = self._close_control("expired", now, None)
        self._reset_candidates()
        self._last_observed_at = now
        return events

    def _validate_monotonic(self, sample: InterestSample) -> None:
        if self._last_observed_at is not None and sample.observed_at < self._last_observed_at:
            raise ValueError("sample.observed_at must be monotonic")
        if self._last_source_at is not None and sample.source_at < self._last_source_at:
            raise ValueError("sample.source_at must be monotonic within a stream epoch")

    def _validate_control_time(self, observed_at: float) -> None:
        if self._last_observed_at is not None and observed_at < self._last_observed_at:
            raise ValueError("observed_at must be monotonic")

    def _expire_before(self, observed_at: float) -> list[InterestIntervalEvent]:
        if self.config.expiry_s is None or self._last_sample_observed_at is None:
            return []
        deadline = self._last_sample_observed_at + self.config.expiry_s
        if observed_at < deadline:
            return []
        events = []
        if self._open is not None:
            events.append(
                self._exit(
                    reason="expired",
                    event_observed_at=deadline,
                    closing_sample=None,
                )
            )
        # A partial enter debounce is evidence too.  It cannot bridge a gap that would have expired
        # a confirmed interval, or the eventual interval would claim continuous evidence it never saw.
        self._reset_candidates()
        return events

    def _enter(self, confirmation: InterestSample) -> InterestIntervalEvent:
        if self._open is not None:
            raise RuntimeError("duplicate interest.entered invariant violated")
        first = self._entering[0]
        interval_id = self._make_interval_id(first)
        self._open = _OpenInterval(
            interval_id=interval_id,
            source_start=first.source_at,
            source_end=confirmation.source_at,
            observed_start=first.observed_at,
            observed_end=confirmation.observed_at,
            last_sample=confirmation,
            regions=confirmation.regions or first.regions,
            track_ids=confirmation.track_ids or first.track_ids,
        )
        self._entering.clear()
        self._exit_count = 0
        return self._event(INTEREST_ENTERED, confirmation.reason, confirmation.observed_at, confirmation)

    def _extend(self, sample: InterestSample) -> None:
        if self._open is None:
            raise RuntimeError("orphan interest.updated invariant violated")
        if sample.source_at < self._open.source_end or sample.observed_at < self._open.observed_end:
            raise RuntimeError("negative interest interval duration invariant violated")
        self._open.source_end = sample.source_at
        self._open.observed_end = sample.observed_at
        self._open.last_sample = sample
        if sample.regions:
            self._open.regions = sample.regions
        if sample.track_ids:
            self._open.track_ids = sample.track_ids

    def _exit(
        self,
        *,
        reason: InterestExitReason,
        event_observed_at: float,
        closing_sample: InterestSample | None,
    ) -> InterestIntervalEvent:
        if self._open is None:
            raise RuntimeError("orphan interest.exited invariant violated")
        sample = closing_sample if closing_sample is not None else self._open.last_sample
        event = self._event(INTEREST_EXITED, reason, event_observed_at, sample)
        self._open = None
        self._reset_candidates()
        return event

    def _close_control(
        self,
        reason: InterestExitReason,
        observed_at: float,
        source_at: float | None,
    ) -> list[InterestIntervalEvent]:
        if source_at is not None:
            source_time = _finite("source_at", source_at)
            if self._last_source_at is not None and source_time < self._last_source_at:
                raise ValueError("source_at must be monotonic within a stream epoch")
        if self._open is None:
            return []
        # A disconnect/EOS timestamp is not evidence that interest remained active until then.
        # Preserve the last positive evidence bound and report the later transition time separately.
        return [
            self._exit(
                reason=reason,
                event_observed_at=observed_at,
                closing_sample=None,
            )
        ]

    def _event(
        self,
        kind: InterestEventKind,
        reason: str,
        event_observed_at: float,
        sample: InterestSample,
    ) -> InterestIntervalEvent:
        if self._open is None:
            raise RuntimeError(f"orphan {kind} invariant violated")
        if (
            self._open.source_end < self._open.source_start
            or self._open.observed_end < self._open.observed_start
        ):
            raise RuntimeError("negative interest interval duration invariant violated")
        return InterestIntervalEvent(
            kind=kind,
            interval_id=self._open.interval_id,
            source=self.source,
            source_start=self._open.source_start,
            source_end=self._open.source_end,
            observed_start=self._open.observed_start,
            observed_end=self._open.observed_end,
            event_observed_at=event_observed_at,
            reason=reason,
            score=sample.score,
            confidence=sample.confidence,
            coverage=sample.coverage,
            regions=sample.regions or self._open.regions,
            track_ids=sample.track_ids or self._open.track_ids,
            admission_id=self.admission_id,
            detector_id=self.detector_id,
        )

    def _make_interval_id(self, first: InterestSample) -> str:
        self._ordinal += 1
        payload = json.dumps(
            {
                "version": 1,
                "source": self.source,
                "admission_id": self.admission_id,
                "detector_id": self.detector_id,
                "ordinal": self._ordinal,
                "source_start": first.source_at,
                "observed_start": first.observed_at,
            },
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return "intv_" + hashlib.sha256(payload).hexdigest()[:24]

    def _reset_candidates(self) -> None:
        self._entering.clear()
        self._exit_count = 0


class Tier0MotionInterest:
    """Opt-in adapter from a ``GateDecision``'s reported mean-motion gauge to intervals.

    This adapter intentionally reads ``decision.signal.motion`` rather than ``decision.escalate``.
    The latter may be driven by a localized, structural, semantic, audio or fused signal that is not
    present in ``GateDecision`` with enough detail to reconstruct a truthful interval.  Confidence
    remains ``None`` because frame-diff salience is not a calibrated confidence.
    """

    def __init__(
        self,
        *,
        source: str,
        config: InterestIntervalConfig,
        admission_id: str = "tier0.signal.motion",
        detector_id: str = "motion_score.mean",
    ) -> None:
        self.engine = InterestIntervalEngine(
            source=source,
            admission_id=admission_id,
            detector_id=detector_id,
            config=config,
        )

    def observe(
        self,
        decision: Any,
        *,
        regions: Sequence[Box] = (),
        track_ids: Sequence[str] = (),
    ) -> list[InterestIntervalEvent]:
        signal = getattr(decision, "signal", None)
        if signal is None:
            raise TypeError("decision must expose a tier-0 signal")
        ts = _finite("decision.signal.ts", getattr(signal, "ts", None))
        return self.engine.observe(
            InterestSample(
                source_at=ts,
                observed_at=ts,
                score=getattr(signal, "motion", None),
                reason="tier0_motion",
                confidence=None,
                coverage=1.0,
                regions=tuple(regions),
                track_ids=tuple(track_ids),
            )
        )

    def end_of_stream(self) -> list[InterestIntervalEvent]:
        return self.engine.end_of_stream()


__all__ = [
    "INTEREST_ENTERED",
    "INTEREST_UPDATED",
    "INTEREST_EXITED",
    "InterestEventKind",
    "InterestExitReason",
    "InterestIntervalConfig",
    "InterestSample",
    "InterestIntervalEvent",
    "InterestIntervalEngine",
    "Tier0MotionInterest",
]
