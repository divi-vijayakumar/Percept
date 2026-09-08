"""Deterministic video-interest intervals for replay and live Watch admissions.

This module turns a scalar, timestamped interest signal into a small typed event stream:
``interest.entered`` -> zero or more ``interest.updated`` -> exactly one ``interest.exited``.
It is deliberately below semantic judgment.  A tier-0 motion score can describe *where the
admission signal was active*; it cannot claim that a Watch's plain-language condition was true.
The optional :class:`Tier0MotionInterest` integration is confined to finite in-memory replay.
:class:`LiveInterestTracker` records the same typed, per-Watch admission outcomes as they are made by
:func:`percept.run`.  Both recorders project a gate outcome through the one function
:func:`interest_admission_from_outcome`, so an offline replay and a live run cannot describe the same
admitted unit differently.

The engine is stdlib-only, replay deterministic, and fail closed:

* interval ids are content-derived (never random or wall-clock-derived);
* input times are finite and monotonic, and interval bounds never run backwards;
* regions must be valid normalized boxes;
* updates/exits can only be emitted for an open interval;
* hysteresis and consecutive-sample debounce are explicit configuration;
* expiry, reconnect and end-of-stream close an open interval with distinct reasons.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
from collections import deque
from dataclasses import dataclass
from typing import Any, Iterable, Literal, Sequence

from .adapters import Box

InterestEventKind = Literal[
    "interest.entered",
    "interest.updated",
    "interest.exited",
]
InterestExitReason = Literal["signal_lost", "expired", "reconnect", "end_of_stream"]
LiveInterestExitReason = Literal[
    "signal_lost",
    "not_admitted",
    "schedule_inactive",
    "cadence_limited",
    "fixed_cadence_unconfigured",
    "admission_changed",
    "watch_removed",
    "source_gap",
    "source_reconnect",
    "source_epoch_gap",
    "source_format_change",
    "source_capture_overflow",
    "end_of_stream",
    "cancelled",
    "worker_error",
    "paused",
]

INTEREST_ENTERED: InterestEventKind = "interest.entered"
INTEREST_UPDATED: InterestEventKind = "interest.updated"
INTEREST_EXITED: InterestEventKind = "interest.exited"


def _finite(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be an int or float")
    result = float(value)
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
        object.__setattr__(self, "enter_threshold", enter)
        object.__setattr__(self, "exit_threshold", exit_)
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
        if self.expiry_s is not None:
            expiry = _finite("expiry_s", self.expiry_s)
            if expiry <= 0.0:
                raise ValueError("expiry_s must be > 0 when set")
            object.__setattr__(self, "expiry_s", expiry)


@dataclass(frozen=True)
class InterestSample:
    """One interest observation on both the source and observer clocks.

    ``evidence_id`` is the producer's stable identity for the exact causal evidence item (the replay
    motion adapter hashes the ordered ``previous,current`` coarse-frame pair). ``confidence=None``
    means the producer has no calibrated confidence signal; the engine never manufactures one from
    a salience score. ``coverage`` describes this observation's usable source coverage, not
    correctness.
    """

    source_at: float
    observed_at: float
    score: float
    reason: str
    evidence_id: str
    confidence: float | None = None
    coverage: float = 1.0
    regions: tuple[Box, ...] = ()
    track_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_at", _finite("source_at", self.source_at))
        object.__setattr__(self, "observed_at", _finite("observed_at", self.observed_at))
        object.__setattr__(self, "score", _unit("score", self.score))
        object.__setattr__(self, "reason", _identity("reason", self.reason))
        object.__setattr__(self, "evidence_id", _identity("evidence_id", self.evidence_id))
        if self.confidence is not None:
            object.__setattr__(self, "confidence", _unit("confidence", self.confidence))
        object.__setattr__(self, "coverage", _unit("coverage", self.coverage))
        object.__setattr__(self, "regions", _validated_regions(self.regions))
        object.__setattr__(self, "track_ids", _validated_tracks(self.track_ids))


@dataclass(frozen=True)
class InterestIntervalEvent:
    """A typed, immutable interval lifecycle event.

    ``stream_epoch`` is caller-owned and ``evidence_id`` identifies the sample that triggered this
    event. Bounds are conservative observed evidence bounds. ``event_observed_at`` is when the state
    transition became known, which can be later than ``observed_end`` after exit debounce, expiry,
    reconnect or EOS. ``source_end`` is never extrapolated when the source is disconnected.
    """

    kind: InterestEventKind
    interval_id: str
    source: str
    stream_epoch: str
    source_start: float
    source_end: float
    observed_start: float
    observed_end: float
    event_observed_at: float
    reason: str
    evidence_id: str
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
        object.__setattr__(self, "stream_epoch", _identity("stream_epoch", self.stream_epoch))
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
        object.__setattr__(self, "evidence_id", _identity("evidence_id", self.evidence_id))
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
            "stream_epoch": self.stream_epoch,
            "source_start": self.source_start,
            "source_end": self.source_end,
            "observed_start": self.observed_start,
            "observed_end": self.observed_end,
            "event_observed_at": self.event_observed_at,
            "reason": self.reason,
            "evidence_id": self.evidence_id,
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

    An engine is pinned to one ``(source, stream_epoch, admission_id, detector_id)`` identity.
    Construct one per scalar signal epoch. ``observe`` may emit an expiry exit followed by a new
    enter on the same call; that is the explicit split policy when a gap reaches ``expiry_s``.
    Shorter gaps merge.
    """

    def __init__(
        self,
        *,
        source: str,
        stream_epoch: str,
        admission_id: str,
        detector_id: str,
        config: InterestIntervalConfig,
    ) -> None:
        self.source = _identity("source", source)
        self.stream_epoch = _identity("stream_epoch", stream_epoch)
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
        stream_epoch: str,
        observed_at: float,
        source_at: float | None = None,
    ) -> list[InterestIntervalEvent]:
        """Close the current interval as ``reconnect`` and start a new source-time epoch.

        Reconnect is an explicit split and requires a NEW caller-owned epoch: even an immediately
        recurring signal receives a new stable interval id. Observer time remains monotonic; source
        time may restart afterwards.
        """
        next_epoch = _identity("stream_epoch", stream_epoch)
        if next_epoch == self.stream_epoch:
            raise ValueError("reconnect stream_epoch must differ from the current epoch")
        now = _finite("observed_at", observed_at)
        self._validate_control_time(now)
        events = self._close_control("reconnect", now, source_at)
        self._reset_candidates()
        self._last_observed_at = now
        self._last_sample_observed_at = None
        self._last_source_at = None
        self._ended = False
        self._ordinal = 0
        self.stream_epoch = next_epoch
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
                    event_observed_at=observed_at,
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
            stream_epoch=self.stream_epoch,
            source_start=self._open.source_start,
            source_end=self._open.source_end,
            observed_start=self._open.observed_start,
            observed_end=self._open.observed_end,
            event_observed_at=event_observed_at,
            reason=reason,
            evidence_id=sample.evidence_id,
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
                "version": 2,
                "source": self.source,
                "stream_epoch": self.stream_epoch,
                "admission_id": self.admission_id,
                "detector_id": self.detector_id,
                "ordinal": self._ordinal,
                "source_start": first.source_at,
                "reason": first.reason,
                "evidence_id": first.evidence_id,
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
    """Replay adapter from a ``GateDecision`` to intervals, scored by the statistic that admitted.

    A decision built with ``include_outcomes=True`` carries one
    :class:`~percept.harness.tier0.WatchAdmissionOutcome` per Watch, holding that Watch's own
    ``score``, ``threshold``, ``admission_id`` and ``detector_id``.  This adapter projects those
    outcomes through :func:`interest_admission_from_outcome` — the SAME function the live worker uses
    to build its per-Watch record — so an offline replay and a live run cannot describe the same
    admitted unit differently.

    It used to score every interval from ``decision.signal.motion``, the reported whole-frame MEAN
    gauge, on the belief that per-Watch detail was not present in ``GateDecision``.  It is: the mean
    is simply not the statistic that admits a ``block_max`` / ``p95`` / ego-residual / structural /
    semantic / fused Watch, so a localized-salience Watch could spend a judge call on a frame that
    replayed as no interest at all.

    ``decision.outcomes`` is empty on a decision built WITHOUT ``include_outcomes=True``, and on a
    hand-constructed legacy one.  That is the documented fallback and the pre-existing behaviour: the
    interval is scored from the mean gauge under this adapter's constructor
    ``admission_id``/``detector_id``.  A caller who wants the admitting statistic must ask the gate
    for outcomes.

    **Cardinality — one interval timeline per admitting statistic, not per Watch.**  Outcomes are
    grouped by ``(admission_id, detector_id)``; each group owns an :class:`InterestIntervalEngine`
    created lazily inside the epoch, and Watches sharing one statistic are unioned by taking the
    highest score among them.  Per-Watch engines were considered and rejected:
    :class:`InterestIntervalEvent` carries no ``watch_id`` — its identity is exactly the engine's
    ``(source, stream_epoch, admission_id, detector_id)`` — so per-Watch intervals would reach a
    consumer with nothing to attribute them to, and two Watches sharing a statistic would derive the
    *same* ``interval_id``, whose payload has no watch field either.  The per-Watch record already
    exists and is the live one (:class:`LiveInterestTracker` / :class:`LiveInterestEvent`).  This
    timeline therefore stays source-keyed and splits only where the emitted record can express the
    difference: a ``block_max`` Watch and a ``mean`` Watch on one source now produce two correctly
    labelled timelines instead of one mislabelled one.  Engine count is bounded by the number of
    distinct statistics in the WatchSpec — never by the number of frames or of admissions.

    Confidence remains ``None`` because frame-diff salience is not a calibrated confidence.  Each
    finite :meth:`~percept.harness.tier0.GatedWatchLoop.run` supplies an explicit ``stream_epoch``
    and independent ``observed_times``; this adapter owns no live capture or reconnect integration.
    """

    def __init__(
        self,
        *,
        source: str,
        config: InterestIntervalConfig,
        admission_id: str = "tier0.signal.motion",
        detector_id: str = "motion_score.mean",
    ) -> None:
        self.source = _identity("source", source)
        self.admission_id = _identity("admission_id", admission_id)
        self.detector_id = _identity("detector_id", detector_id)
        if not isinstance(config, InterestIntervalConfig):
            raise TypeError("config must be an InterestIntervalConfig")
        self.config = config
        self._engine: InterestIntervalEngine | None = None
        self._stream_epoch: str | None = None
        # (admission_id, detector_id) -> engine.  Insertion order is the WatchSpec's condition order,
        # so the emitted event order is replay deterministic.
        self._outcome_engines: dict[tuple[str, str], InterestIntervalEngine] = {}

    def begin_epoch(self, stream_epoch: str) -> None:
        """Begin one finite replay epoch; an unfinished prior epoch is a lifecycle error."""
        if self._engine is not None:
            raise RuntimeError("an interest replay epoch is already active")
        self._engine = InterestIntervalEngine(
            source=self.source,
            stream_epoch=stream_epoch,
            admission_id=self.admission_id,
            detector_id=self.detector_id,
            config=self.config,
        )
        self._stream_epoch = self._engine.stream_epoch
        self._outcome_engines = {}

    @property
    def engine(self) -> InterestIntervalEngine | None:
        """The mean-gauge fallback engine, which is also this adapter's epoch lifecycle handle."""
        return self._engine

    @property
    def statistic_engines(self) -> dict[tuple[str, str], InterestIntervalEngine]:
        """A copy of the ``(admission_id, detector_id)`` -> engine map opened so far this epoch."""
        return dict(self._outcome_engines)

    def observe(
        self,
        decision: Any,
        *,
        observed_at: float,
        evidence_id: str,
        regions: Sequence[Box] = (),
        track_ids: Sequence[str] = (),
    ) -> list[InterestIntervalEvent]:
        signal = getattr(decision, "signal", None)
        if signal is None:
            raise TypeError("decision must expose a tier-0 signal")
        if self._engine is None:
            raise RuntimeError("begin_epoch() must be called before observe()")
        source_at = _finite("decision.signal.ts", getattr(signal, "ts", None))
        observed = _finite("observed_at", observed_at)
        evidence = _identity("evidence_id", evidence_id)
        outcomes = tuple(getattr(decision, "outcomes", ()) or ())
        if outcomes:
            return self._observe_outcomes(
                outcomes,
                source_at=source_at,
                observed_at=observed,
                evidence_id=evidence,
                regions=regions,
                track_ids=track_ids,
            )
        motion = _unit("decision.signal.motion", getattr(signal, "motion", None))
        return self._engine.observe(
            InterestSample(
                source_at=source_at,
                observed_at=observed,
                score=motion,
                reason="tier0_motion",
                evidence_id=evidence,
                confidence=None,
                coverage=1.0,
                regions=tuple(regions),
                track_ids=tuple(track_ids),
            )
        )

    def _observe_outcomes(
        self,
        outcomes: Sequence[Any],
        *,
        source_at: float,
        observed_at: float,
        evidence_id: str,
        regions: Sequence[Box],
        track_ids: Sequence[str],
    ) -> list[InterestIntervalEvent]:
        """One sample per admitting statistic, carrying the score the gate actually compared.

        Every condition yields an outcome on every evaluated unit, admitted or not, so each
        statistic's engine sees a continuous scalar and can still close an interval on
        ``signal_lost`` — feeding it only the admitted units would leave intervals open until
        end-of-stream.  A Watch denied by schedule or cadence reports ``0.0``, which is the same
        thing the live recorder does with a policy non-admission: not looking is not interest.
        Interval boundaries remain this adapter's ``config`` hysteresis, never the Watch's own
        threshold; only the *score* moves to the admitting statistic.
        """
        best_by_statistic: dict[tuple[str, str], LiveInterestAdmission] = {}
        for outcome in outcomes:
            admitted = bool(getattr(outcome, "admitted", False))
            admission = interest_admission_from_outcome(
                outcome,
                evidence_id=evidence_id,
                cause=getattr(outcome, "cause", None) if admitted else None,
            )
            key = (admission.admission_id, admission.detector_id)
            incumbent = best_by_statistic.get(key)
            if incumbent is None or admission.score > incumbent.score:
                best_by_statistic[key] = admission
        events: list[InterestIntervalEvent] = []
        for key, admission in best_by_statistic.items():
            engine = self._outcome_engines.get(key)
            if engine is None:
                assert self._stream_epoch is not None  # begin_epoch() set it with the engine
                engine = InterestIntervalEngine(
                    source=self.source,
                    stream_epoch=self._stream_epoch,
                    admission_id=admission.admission_id,
                    detector_id=admission.detector_id,
                    config=self.config,
                )
                self._outcome_engines[key] = engine
            events.extend(
                engine.observe(
                    interest_sample_from_admission(
                        admission,
                        source_at=source_at,
                        observed_at=observed_at,
                        regions=regions,
                        track_ids=track_ids,
                    )
                )
            )
        return events

    def end_of_stream(self, *, observed_at: float | None = None) -> list[InterestIntervalEvent]:
        """Finish the current finite replay epoch and become ready for the next one."""
        if self._engine is None:
            return []
        engines = [self._engine, *self._outcome_engines.values()]
        try:
            events: list[InterestIntervalEvent] = []
            for engine in engines:
                events.extend(engine.end_of_stream(observed_at=observed_at))
            return events
        finally:
            self._engine = None
            self._stream_epoch = None
            self._outcome_engines = {}

    def abort_epoch(self) -> None:
        """Discard an unreturned replay report after failure, emitting no orphan lifecycle event."""
        self._engine = None
        self._stream_epoch = None
        self._outcome_engines = {}


# ---------------------------------------------------------------------------
# Live per-Watch admission lifecycle
# ---------------------------------------------------------------------------

_LIVE_CAUSES = frozenset(
    ("motion", "structural", "semantic", "fixed", "fused", "coast", "audio", "candidate")
)
#: The two causes the *worker* owns rather than the gate: a coasted continuation and an external
#: audio wake.  Neither is a tier-0 statistic, so an outcome routed through one of them cannot keep
#: the gate's ``admission_id``/``detector_id`` — see :func:`interest_admission_from_outcome`.
_WORKER_ROUTE_CAUSES = frozenset(("coast", "audio", "candidate"))
#: Everything else: a cause the gate itself decided, whose detector identity is already truthful.
_GATE_ADMISSION_CAUSES = frozenset(_LIVE_CAUSES - _WORKER_ROUTE_CAUSES)
#: The active-interval reason each admission cause names.  One table, read by both recorders — the
#: live one through :func:`_live_active_reason`, the replay one through
#: :func:`interest_sample_from_admission` — so a motion admission reads ``tier0_motion`` on either
#: timeline and a fixed-cadence one reads ``fixed_cadence`` on either.
_LIVE_ACTIVE_REASONS = {
    "motion": "tier0_motion",
    "structural": "tier0_structural",
    "semantic": "tier0_semantic",
    "fixed": "fixed_cadence",
    # the noisy-OR over the cheap channels — a gate-owned cause like the three above it, but not a
    # single tier-0 statistic, so it names the fusion rather than a `tier0_*` channel
    "fused": "wake_fusion",
    "coast": "coast",
    "audio": "audio",
    "candidate": "candidate",
}
_LIVE_NON_ADMISSION_REASONS = frozenset(
    (
        "signal_below_threshold",
        "schedule_inactive",
        "cadence_limited",
        "fixed_cadence_unconfigured",
        "not_admitted",
    )
)
_LIVE_EXIT_REASONS = frozenset(
    (
        "signal_lost",
        "not_admitted",
        "schedule_inactive",
        "cadence_limited",
        "fixed_cadence_unconfigured",
        "admission_changed",
        "watch_removed",
        "source_gap",
        "source_reconnect",
        "source_epoch_gap",
        "source_format_change",
        "source_capture_overflow",
        "end_of_stream",
        "cancelled",
        "worker_error",
        "paused",
    )
)


@dataclass(frozen=True)
class LiveInterestAdmission:
    """One Watch's exact typed admission outcome for an accepted video unit.

    The live worker constructs this from the gate's per-Watch outcome after applying the worker-owned
    coast/audio routes.  A tracker therefore records policy; it never re-runs schedule, cadence, or
    threshold logic.
    """

    watch_id: str
    admitted: bool
    cause: str | None
    non_admission_reason: str | None
    score: float
    threshold: float
    evidence_id: str
    admission_id: str
    detector_id: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "watch_id", _identity("watch_id", self.watch_id))
        object.__setattr__(self, "score", _unit("score", self.score))
        object.__setattr__(self, "threshold", _unit("threshold", self.threshold))
        object.__setattr__(self, "evidence_id", _identity("evidence_id", self.evidence_id))
        object.__setattr__(self, "admission_id", _identity("admission_id", self.admission_id))
        object.__setattr__(self, "detector_id", _identity("detector_id", self.detector_id))
        if not isinstance(self.admitted, bool):
            raise TypeError("admitted must be a bool")
        if self.admitted:
            if self.cause not in _LIVE_CAUSES or self.non_admission_reason is not None:
                raise ValueError("an admitted outcome needs one cause and no non-admission reason")
        elif self.cause is not None or self.non_admission_reason not in _LIVE_NON_ADMISSION_REASONS:
            raise ValueError("a non-admitted outcome needs one reason and no admission cause")


def interest_admission_from_outcome(
    outcome: Any,
    *,
    evidence_id: str,
    cause: str | None = None,
) -> LiveInterestAdmission:
    """Project one gate ``WatchAdmissionOutcome`` onto the record BOTH interest recorders read.

    ``outcome`` is a :class:`~percept.harness.tier0.WatchAdmissionOutcome` — the gate's complete
    policy result for one Watch on one accepted unit, carrying the ``score`` it actually compared,
    that Watch's ``threshold``, and the ``admission_id``/``detector_id`` naming the statistic that
    did the comparing.  ``cause`` is the caller's FINAL admission cause after any caller-owned route
    (the worker's coast/audio), or ``None`` when this Watch was not admitted on this unit — which is
    how a Watch the gate admitted but the caller then dropped is recorded as ``not_admitted``.

    This function exists because there are two recorders and only one truth.  The live worker builds
    :class:`LiveInterestAdmission` for :class:`LiveInterestTracker`; :class:`Tier0MotionInterest`
    builds a replay :class:`InterestSample` from the same record via
    :func:`interest_sample_from_admission`.  They performed this projection independently once, and
    drifted: replay scored every interval from the reported whole-frame mean gauge while the live
    path used ``outcome.score``, so the two disagreed about the same admitted unit.  Neither call
    site may re-derive these fields.
    """
    goal_id = _identity("outcome.goal_id", getattr(outcome, "goal_id", None))
    score = _unit("outcome.score", getattr(outcome, "score", None))
    threshold = _unit("outcome.threshold", getattr(outcome, "threshold", None))
    gate_admission_id = _identity("outcome.admission_id", getattr(outcome, "admission_id", None))
    gate_detector_id = _identity("outcome.detector_id", getattr(outcome, "detector_id", None))
    evidence = _identity("evidence_id", evidence_id)
    if cause is None:
        return LiveInterestAdmission(
            watch_id=goal_id,
            admitted=False,
            cause=None,
            # A Watch the gate refused keeps the gate's own reason; a Watch the gate admitted and the
            # caller then dropped has no gate reason to keep, so it is the catch-all.
            non_admission_reason=(
                getattr(outcome, "non_admission_reason", None)
                if not getattr(outcome, "admitted", False)
                else None
            )
            or "not_admitted",
            score=score,
            threshold=threshold,
            evidence_id=evidence,
            admission_id=gate_admission_id,
            detector_id=gate_detector_id,
        )
    gate_owned = cause in _GATE_ADMISSION_CAUSES
    return LiveInterestAdmission(
        watch_id=goal_id,
        admitted=True,
        cause=cause,
        non_admission_reason=None,
        score=score,
        threshold=threshold,
        evidence_id=evidence,
        # A caller-owned route did not run the gate's detector, so it must not borrow its name.
        admission_id=gate_admission_id if gate_owned else f"runtime.admission.{cause}",
        detector_id=gate_detector_id if gate_owned else f"runtime.{cause}",
    )


def interest_sample_from_admission(
    admission: LiveInterestAdmission,
    *,
    source_at: float,
    observed_at: float,
    regions: Sequence[Box] = (),
    track_ids: Sequence[str] = (),
) -> InterestSample:
    """The replay recorder's half of the projection: one admission record -> one interest sample.

    The score is the admission's — the statistic the gate compared — never a separately reported
    gauge.  ``confidence`` stays ``None`` (tier-0 salience is not calibrated) and ``coverage`` stays
    ``1.0`` (the unit was observed whole; a partially-covered unit is a source-layer fact, not an
    admission one).

    The reason names the admission FAMILY, read out of the same :data:`_LIVE_ACTIVE_REASONS` table
    the live recorder uses, so a motion admission reads ``tier0_motion`` on either timeline.  It is
    taken from the admission id rather than from ``cause`` because a sample is produced for every
    evaluated Watch, admitted or not, and a below-cutoff motion reading is still a motion reading.
    """
    if not isinstance(admission, LiveInterestAdmission):
        raise TypeError("admission must be a LiveInterestAdmission")
    family = admission.admission_id.rsplit(".", 1)[-1]
    return InterestSample(
        source_at=source_at,
        observed_at=observed_at,
        score=admission.score,
        reason=_LIVE_ACTIVE_REASONS.get(family, f"tier0_{family}"),
        evidence_id=admission.evidence_id,
        confidence=None,
        coverage=1.0,
        regions=tuple(regions),
        track_ids=tuple(track_ids),
    )


@dataclass(frozen=True)
class LiveInterestEvent:
    """Immutable live VID lifecycle event for one Watch and one admitted interval.

    ``source_*`` and ``observed_*`` retain independent clocks.  Interval ends always name the last
    admitted unit; ``event_*`` names the later unit/control boundary that made an exit known.
    """

    kind: InterestEventKind
    interval_id: str
    watch_id: str
    source_id: str
    source_epoch: int
    sequence_start: int
    sequence_end: int
    source_start: float | None
    source_end: float | None
    observed_start: float
    observed_end: float
    event_source_epoch: int
    event_sequence: int
    event_source_time: float | None
    event_observed_at: float
    reason: str
    admission_cause: str
    evidence_id: str
    score: float
    threshold: float
    confidence: None
    regions: tuple[Box, ...]
    track_ids: tuple[str, ...]
    admission_id: str
    detector_id: str
    coverage_kind: str
    missing_sequences: int
    source_events: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.kind not in (INTEREST_ENTERED, INTEREST_UPDATED, INTEREST_EXITED):
            raise ValueError(f"unknown live interest event kind {self.kind!r}")
        for name in ("interval_id", "watch_id", "source_id", "reason", "evidence_id"):
            object.__setattr__(self, name, _identity(name, getattr(self, name)))
        if self.kind == INTEREST_EXITED and self.reason not in _LIVE_EXIT_REASONS:
            raise ValueError(f"unknown live interest exit reason {self.reason!r}")
        if self.admission_cause not in _LIVE_CAUSES:
            raise ValueError(f"unknown live interest admission cause {self.admission_cause!r}")
        for name in (
            "source_epoch",
            "sequence_start",
            "sequence_end",
            "event_source_epoch",
            "event_sequence",
            "missing_sequences",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be an integer >= 0")
        if self.sequence_end < self.sequence_start:
            raise ValueError("sequence_end must be >= sequence_start")
        for name in ("source_start", "source_end", "event_source_time"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _finite(name, value))
        if (
            self.source_start is not None
            and self.source_end is not None
            and self.source_end < self.source_start
        ):
            raise ValueError("source_end must be >= source_start")
        for name in ("observed_start", "observed_end", "event_observed_at"):
            object.__setattr__(self, name, _finite(name, getattr(self, name)))
        if self.observed_end < self.observed_start:
            raise ValueError("observed_end must be >= observed_start")
        if self.event_observed_at < self.observed_end:
            raise ValueError("event_observed_at must be >= observed_end")
        object.__setattr__(self, "score", _unit("score", self.score))
        object.__setattr__(self, "threshold", _unit("threshold", self.threshold))
        if self.confidence is not None:
            raise ValueError("tier-0 admission confidence must be None")
        object.__setattr__(self, "regions", _validated_regions(self.regions))
        object.__setattr__(self, "track_ids", _validated_tracks(self.track_ids))
        object.__setattr__(self, "admission_id", _identity("admission_id", self.admission_id))
        object.__setattr__(self, "detector_id", _identity("detector_id", self.detector_id))
        if self.coverage_kind not in ("observed", "observed_after_discontinuity"):
            raise ValueError("coverage_kind must describe an accepted source unit")
        object.__setattr__(
            self,
            "source_events",
            tuple(_identity(f"source_events[{i}]", event) for i, event in enumerate(self.source_events)),
        )

    @property
    def stream_epoch(self) -> str:
        return f"{self.source_id}:{self.source_epoch}"

    @property
    def coverage(self) -> float:
        return 1.0

    @property
    def duration(self) -> float:
        return self.observed_end - self.observed_start

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "interval_id": self.interval_id,
            "watch_id": self.watch_id,
            "source_id": self.source_id,
            "source_epoch": self.source_epoch,
            "sequence_start": self.sequence_start,
            "sequence_end": self.sequence_end,
            "source_start": self.source_start,
            "source_end": self.source_end,
            "observed_start": self.observed_start,
            "observed_end": self.observed_end,
            "event_source_epoch": self.event_source_epoch,
            "event_sequence": self.event_sequence,
            "event_source_time": self.event_source_time,
            "event_observed_at": self.event_observed_at,
            "reason": self.reason,
            "admission_cause": self.admission_cause,
            "evidence_id": self.evidence_id,
            "score": self.score,
            "threshold": self.threshold,
            "confidence": None,
            "regions": [
                {"x": box.x, "y": box.y, "w": box.w, "h": box.h}
                for box in self.regions
            ],
            "track_ids": list(self.track_ids),
            "admission_id": self.admission_id,
            "detector_id": self.detector_id,
            "coverage": self.coverage,
            "coverage_kind": self.coverage_kind,
            "missing_sequences": self.missing_sequences,
            "source_events": list(self.source_events),
        }


class InterestBackpressureError(RuntimeError):
    """The bounded lifecycle channel could not accept one whole producer batch."""


class InterestCallbackUnsupportedError(TypeError):
    """Callbacks are intentionally not a live lifecycle transport."""


class InterestRunError(RuntimeError):
    """Both the worker and lifecycle delivery failed during the same run."""

    def __init__(self, worker_error: BaseException, delivery_error: BaseException) -> None:
        self.worker_error = worker_error
        self.delivery_error = delivery_error
        super().__init__(
            f"runtime failed with {worker_error!r}; interest delivery also failed with "
            f"{delivery_error!r}"
        )


class InterestSubscription:
    """Single-use, bounded async subscription owned by the SDK producer.

    The producer only appends immutable atomic batches and signals the consumer task; it never calls
    subscriber code.  Capacity is measured in batches, so a single frame's multi-Watch transitions
    cannot be split.  One additional terminal batch is reserved to close already-published intervals
    even when regular capacity is exhausted.
    """

    __slots__ = (
        "_max_batches",
        "_batches",
        "_current",
        "_ready",
        "_loop",
        "_run_id",
        "_iterated",
        "_closed",
        "_rejected_batch",
    )

    def __init_subclass__(cls, **kwargs: Any) -> None:
        raise TypeError("InterestSubscription is a final SDK type and cannot be subclassed")

    def __init__(self, *, max_batches: int = 256) -> None:
        if (
            not isinstance(max_batches, int)
            or isinstance(max_batches, bool)
            or max_batches < 1
        ):
            raise ValueError("max_batches must be an integer >= 1")
        self._max_batches = max_batches
        self._batches: deque[tuple[LiveInterestEvent, ...]] = deque()
        self._current: deque[LiveInterestEvent] = deque()
        self._ready = asyncio.Event()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._run_id: str | None = None
        self._iterated = False
        self._closed = False
        self._rejected_batch: tuple[LiveInterestEvent, ...] | None = None

    @property
    def closed(self) -> bool:
        return self._closed

    def __aiter__(self) -> "InterestSubscription":
        if self._iterated:
            raise RuntimeError("InterestSubscription supports exactly one async iterator")
        self._iterated = True
        self._claim_loop()
        return self

    async def __anext__(self) -> LiveInterestEvent:
        self._claim_loop()
        while True:
            if self._current:
                return self._current.popleft()
            if self._batches:
                self._current.extend(self._batches.popleft())
                continue
            if self._closed:
                raise StopAsyncIteration
            self._ready.clear()
            if self._batches or self._closed:
                continue
            await self._ready.wait()

    def _claim_loop(self) -> None:
        loop = asyncio.get_running_loop()
        if self._loop is None:
            self._loop = loop
        elif self._loop is not loop:
            raise RuntimeError("InterestSubscription cannot cross event loops")

    def _bind(self, run_id: str) -> None:
        self._claim_loop()
        if self._run_id is not None:
            raise RuntimeError("InterestSubscription is single-use and is already bound")
        if self._closed:
            raise RuntimeError("InterestSubscription is already closed")
        self._run_id = _identity("interest_session_id", run_id)

    def _publish(self, events: Iterable[LiveInterestEvent]) -> None:
        if self._run_id is None or self._closed:
            raise RuntimeError("InterestSubscription is not open for publication")
        batch = self._validated_batch(events)
        if not batch:
            return
        if self._rejected_batch is not None:
            raise InterestBackpressureError(
                "interest subscription already rejected an atomic lifecycle batch"
            )
        if len(self._batches) >= self._max_batches:
            # The tracker has already advanced through this exact atomic transition. Preserve it in
            # the single terminal reserve so finalization can publish the rejected transition before
            # closing every interval it opened/updated. The original overflow still terminates run.
            self._rejected_batch = batch
            raise InterestBackpressureError(
                f"interest subscription capacity ({self._max_batches} batches) was exhausted"
            )
        self._batches.append(batch)
        self._ready.set()

    def _finish(self, events: Iterable[LiveInterestEvent]) -> None:
        if self._run_id is None:
            raise RuntimeError("InterestSubscription was not bound")
        if self._closed:
            return
        terminal = self._validated_batch(events)
        # The single reserve is one atomic batch in queue accounting. If ordinary publication
        # overflowed, its rejected transition comes first; terminal exits then close the exact state
        # produced by that transition. No exit can become orphaned and no published enter remains open.
        reserved = (self._rejected_batch or ()) + terminal
        if reserved:
            self._batches.append(reserved)
        self._rejected_batch = None
        self._closed = True
        self._ready.set()

    @staticmethod
    def _validated_batch(events: Iterable[LiveInterestEvent]) -> tuple[LiveInterestEvent, ...]:
        batch = tuple(events)
        if any(type(event) is not LiveInterestEvent for event in batch):
            raise TypeError("interest batches must contain exact LiveInterestEvent values")
        return batch


@dataclass(frozen=True)
class _LiveSourceUnit:
    raw_source_id: str
    source_epoch: int
    sequence: int
    source_time: float | None
    observed_time: float

    @classmethod
    def from_envelope(cls, envelope: Any) -> "_LiveSourceUnit":
        return cls(
            raw_source_id=_identity("source_id", envelope.source_id),
            source_epoch=envelope.epoch,
            sequence=envelope.sequence,
            source_time=envelope.source_time,
            observed_time=envelope.observed_time,
        )


@dataclass
class _LiveOpenInterval:
    interval_id: str
    admission: LiveInterestAdmission
    first: _LiveSourceUnit
    last: _LiveSourceUnit


def _digest(prefix: str, payload: dict[str, Any], *, width: int = 32) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return prefix + hashlib.sha256(encoded).hexdigest()[:width]


def _public_source_id(run_id: str, raw_source_id: str) -> str:
    return _digest("source_", {"version": 1, "run_id": run_id, "source": raw_source_id}, width=20)


class LiveInterestTracker:
    """Run-local state machine over typed per-Watch outcomes and accepted source units."""

    def __init__(self, *, run_id: str) -> None:
        self._run_id = _identity("run_id", run_id)
        self._open: dict[tuple[str, str], _LiveOpenInterval] = {}
        self._latest: dict[str, _LiveSourceUnit] = {}

    @property
    def active(self) -> tuple[tuple[str, str], ...]:
        return tuple(sorted(self._open))

    def record_accepted(self, envelope: Any) -> None:
        unit = _LiveSourceUnit.from_envelope(envelope)
        self._latest[unit.raw_source_id] = unit

    def observe(
        self,
        envelope: Any,
        admissions: Iterable[LiveInterestAdmission],
        *,
        coverage_kind: str = "observed",
        missing_sequences: int = 0,
        source_events: Sequence[str] = (),
    ) -> list[LiveInterestEvent]:
        unit = _LiveSourceUnit.from_envelope(envelope)
        by_watch: dict[str, LiveInterestAdmission] = {}
        for admission in admissions:
            if type(admission) is not LiveInterestAdmission:
                raise TypeError("admissions must contain exact LiveInterestAdmission values")
            if admission.watch_id in by_watch:
                raise ValueError(f"duplicate Watch outcome {admission.watch_id!r}")
            by_watch[admission.watch_id] = admission

        events: list[LiveInterestEvent] = []
        raw_source = unit.raw_source_id
        live_watch_ids = set(by_watch)
        for source_id, watch_id in sorted(self._open):
            if source_id == raw_source and watch_id not in live_watch_ids:
                events.append(
                    self._close(
                        (source_id, watch_id),
                        reason="watch_removed",
                        event_unit=unit,
                        coverage_kind=coverage_kind,
                        missing_sequences=missing_sequences,
                        source_events=tuple(source_events),
                    )
                )

        for watch_id in sorted(by_watch):
            admission = by_watch[watch_id]
            key = (raw_source, watch_id)
            opened = self._open.get(key)
            if not admission.admitted:
                if opened is not None:
                    events.append(
                        self._close(
                            key,
                            reason=_live_non_admission_reason(admission.non_admission_reason),
                            event_unit=unit,
                            coverage_kind=coverage_kind,
                            missing_sequences=missing_sequences,
                            source_events=tuple(source_events),
                        )
                    )
                continue
            if opened is not None and (
                opened.admission.cause != admission.cause
                or opened.admission.admission_id != admission.admission_id
                or opened.admission.detector_id != admission.detector_id
            ):
                events.append(
                    self._close(
                        key,
                        reason="admission_changed",
                        event_unit=unit,
                        coverage_kind=coverage_kind,
                        missing_sequences=missing_sequences,
                        source_events=tuple(source_events),
                    )
                )
                opened = None
            if opened is None:
                interval_id = _digest(
                    "intv_",
                    {
                        "version": 1,
                        "run_id": self._run_id,
                        "source": raw_source,
                        "epoch": unit.source_epoch,
                        "sequence": unit.sequence,
                        "watch_id": watch_id,
                        "evidence_id": admission.evidence_id,
                    },
                    width=24,
                )
                opened = _LiveOpenInterval(interval_id, admission, unit, unit)
                self._open[key] = opened
                kind: InterestEventKind = INTEREST_ENTERED
            else:
                opened.last = unit
                opened.admission = admission
                kind = INTEREST_UPDATED
            events.append(
                self._event(
                    opened,
                    kind=kind,
                    reason=_live_active_reason(admission.cause),
                    event_unit=unit,
                    coverage_kind=coverage_kind,
                    missing_sequences=missing_sequences,
                    source_events=tuple(source_events),
                )
            )
        self._latest[raw_source] = unit
        return events

    def discontinuity(self, envelope: Any, inspection: Any) -> list[LiveInterestEvent]:
        unit = _LiveSourceUnit.from_envelope(envelope)
        reason = _discontinuity_reason(inspection.events)
        return self._close_source(
            unit.raw_source_id,
            reason=reason,
            event_unit=unit,
            coverage_kind=inspection.coverage.kind,
            missing_sequences=inspection.coverage.missing_sequences,
            source_events=inspection.events,
        )

    def pause(self, envelope: Any, inspection: Any) -> list[LiveInterestEvent]:
        unit = _LiveSourceUnit.from_envelope(envelope)
        return self._close_source(
            unit.raw_source_id,
            reason="paused",
            event_unit=unit,
            coverage_kind=inspection.coverage.kind,
            missing_sequences=inspection.coverage.missing_sequences,
            source_events=inspection.events,
        )

    def close_all(self, reason: LiveInterestExitReason) -> list[LiveInterestEvent]:
        if reason not in _LIVE_EXIT_REASONS:
            raise ValueError(f"unknown terminal reason {reason!r}")
        events: list[LiveInterestEvent] = []
        for key in sorted(tuple(self._open)):
            opened = self._open[key]
            event_unit = self._latest.get(key[0], opened.last)
            events.append(
                self._close(
                    key,
                    reason=reason,
                    event_unit=event_unit,
                    coverage_kind="observed",
                    missing_sequences=0,
                    source_events=(),
                )
            )
        return events

    def _close_source(
        self,
        raw_source_id: str,
        *,
        reason: LiveInterestExitReason,
        event_unit: _LiveSourceUnit,
        coverage_kind: str,
        missing_sequences: int,
        source_events: tuple[str, ...],
    ) -> list[LiveInterestEvent]:
        return [
            self._close(
                key,
                reason=reason,
                event_unit=event_unit,
                coverage_kind=coverage_kind,
                missing_sequences=missing_sequences,
                source_events=source_events,
            )
            for key in sorted(tuple(self._open))
            if key[0] == raw_source_id
        ]

    def _close(
        self,
        key: tuple[str, str],
        *,
        reason: LiveInterestExitReason,
        event_unit: _LiveSourceUnit,
        coverage_kind: str,
        missing_sequences: int,
        source_events: tuple[str, ...],
    ) -> LiveInterestEvent:
        opened = self._open.pop(key)
        return self._event(
            opened,
            kind=INTEREST_EXITED,
            reason=reason,
            event_unit=event_unit,
            coverage_kind=coverage_kind,
            missing_sequences=missing_sequences,
            source_events=source_events,
        )

    def _event(
        self,
        opened: _LiveOpenInterval,
        *,
        kind: InterestEventKind,
        reason: str,
        event_unit: _LiveSourceUnit,
        coverage_kind: str,
        missing_sequences: int,
        source_events: tuple[str, ...],
    ) -> LiveInterestEvent:
        admission = opened.admission
        evidence_id = _digest(
            "evidence_",
            {
                "version": 1,
                "run_id": self._run_id,
                "source": opened.first.raw_source_id,
                "epoch": event_unit.source_epoch,
                "sequence": event_unit.sequence,
                "watch_id": admission.watch_id,
                "gate_evidence_id": admission.evidence_id,
                "kind": kind,
            },
        )
        return LiveInterestEvent(
            kind=kind,
            interval_id=opened.interval_id,
            watch_id=admission.watch_id,
            source_id=_public_source_id(self._run_id, opened.first.raw_source_id),
            source_epoch=opened.first.source_epoch,
            sequence_start=opened.first.sequence,
            sequence_end=opened.last.sequence,
            source_start=opened.first.source_time,
            source_end=opened.last.source_time,
            observed_start=opened.first.observed_time,
            observed_end=opened.last.observed_time,
            event_source_epoch=event_unit.source_epoch,
            event_sequence=event_unit.sequence,
            event_source_time=event_unit.source_time,
            event_observed_at=event_unit.observed_time,
            reason=reason,
            admission_cause=admission.cause or "motion",
            evidence_id=evidence_id,
            score=admission.score,
            threshold=admission.threshold,
            confidence=None,
            regions=(),
            track_ids=(),
            admission_id=admission.admission_id,
            detector_id=admission.detector_id,
            coverage_kind=coverage_kind,
            missing_sequences=missing_sequences,
            source_events=source_events,
        )


def _live_non_admission_reason(reason: str | None) -> LiveInterestExitReason:
    if reason == "signal_below_threshold":
        return "signal_lost"
    if reason in _LIVE_EXIT_REASONS:
        return reason  # type: ignore[return-value]
    return "not_admitted"


def _live_active_reason(cause: str | None) -> str:
    return _LIVE_ACTIVE_REASONS.get(cause, "not_admitted")


def _discontinuity_reason(events: Sequence[str]) -> LiveInterestExitReason:
    for reason in (
        "source_capture_overflow",
        "source_reconnect",
        "source_epoch_gap",
        "source_format_change",
        "source_gap",
    ):
        if reason in events:
            return reason  # type: ignore[return-value]
    raise ValueError("accepted discontinuity did not carry a recognized source event")


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
    "LiveInterestExitReason",
    "LiveInterestAdmission",
    "LiveInterestEvent",
    "interest_admission_from_outcome",
    "interest_sample_from_admission",
    "InterestBackpressureError",
    "InterestCallbackUnsupportedError",
    "InterestRunError",
    "InterestSubscription",
    "LiveInterestTracker",
]
