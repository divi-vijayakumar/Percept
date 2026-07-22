"""Event capsule — the ``event_capsule`` composite evidence shape (additive Temporal value).

A plain ``strip``/``window`` is ONE contiguous span. Some fires need a wider story: the lead-up (a
motion ring already buffers), how the episode itself opened and where it stands now, and — for an
episode that runs long — SOME evidence from near its start even after the lead-up ring has long since
aged past it. That composite is the capsule:

* **pre-roll** — the lead-up to the fire, read from the SAME rolling ``FrameWindow`` ring the incident
  snippet's pre-roll already uses (:func:`percept.incidents.snippet_from_window`). This module builds
  no new ring; :func:`assemble_event_capsule` only reads ``window.frames()``.
* **boundary** — the episode's own opening observation (:attr:`~percept.episodes.Episode.exemplar` at
  :attr:`~percept.episodes.Episode.started_at`) plus its most recent one
  (:attr:`~percept.episodes.Episode.representative` at :attr:`~percept.episodes.Episode.last_seen`) —
  "what opened this" and "what it looks like now", straight off the existing :class:`~percept.episodes.Episode`.
* **early snapshot** — ONE frame :class:`EarlySnapshotTimer` captures a fixed ``after_s`` seconds after
  the episode opened. It is TIMER-driven, not window-driven: once captured it is frozen and never
  overwritten, so a long episode (the ring having long since evicted its opening frames) still carries
  evidence from near its start.

Both :class:`EventCapsule` and :func:`assemble_event_capsule` are pure / side-effect free (mirroring the
fold functions in :mod:`percept.behavior`) — safe to call speculatively, from a shadow path, or from a
fixture test with no live agent loop. Storage reuses the EXISTING :class:`~percept.evidence.EvidenceStore`
retention mechanism (ADR-003) via :func:`store_event_capsule` — this module never builds a new store.

Standing rule 2 (see ``percept/harness/scripted.py``): this is scripted wiring, not intelligence — it
proves the composite evidence shape assembles correctly, never that a model would want it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

#: One captured observation: a stream-time stamp + the opaque frame payload (matches
#: ``percept.events.FrameWindow``'s ``(t, frame)`` tuple shape).
Frame = tuple[float, Any]


@dataclass
class EarlySnapshotTimer:
    """Per-episode timer: captures ONE frame a fixed ``after_s`` seconds after the episode opened.

    Timer-driven, not window-driven — :meth:`observe` is fed on every tick (whatever frame is current
    at that tick) and freezes the FIRST one seen at or after ``started_at + after_s``. Once captured it
    is never overwritten, so replaying more frames — or the pre-roll ring evicting its own early
    frames — can never lose the snapshot. A goal with many concurrent episodes (one timer instance,
    many ``episode_id`` keys) is supported; :meth:`forget` bounds memory for a long-running deployment
    by dropping an episode's state once it closes.
    """

    after_s: float = 5.0
    _armed_at: dict[str, float] = field(default_factory=dict)
    _captured: dict[str, Frame] = field(default_factory=dict)

    def arm(self, episode_id: str, started_at: float) -> None:
        """Register (or re-register) an episode's open time.

        Idempotent for the SAME ``started_at`` (a repeated arm on an already-open episode changes
        nothing). A DIFFERENT ``started_at`` for an already-known id (an episode id reused after its
        prior instance was evicted/closed) is treated as a fresh episode: it re-arms and drops any
        stale capture, so a new episode can never inherit an old one's snapshot.
        """
        if self._armed_at.get(episode_id) != started_at:
            self._armed_at[episode_id] = started_at
            self._captured.pop(episode_id, None)

    def observe(self, episode_id: str, t: float, frame: Any) -> None:
        """Feed one tick's ``(t, frame)`` for ``episode_id``.

        A no-op once this episode already has a captured snapshot, or if it was never armed (armed_at
        is what supplies "seconds since open" — observing before arming is defined as inert, never an
        implicit arm at the observed time, so a caller cannot accidentally shift the episode's origin).
        """
        if episode_id in self._captured:
            return
        started_at = self._armed_at.get(episode_id)
        if started_at is None:
            return
        if t - started_at >= self.after_s:
            self._captured[episode_id] = (t, frame)

    def snapshot(self, episode_id: str) -> Frame | None:
        """The captured ``(t, frame)`` for this episode, or ``None`` if not yet (or never) captured."""
        return self._captured.get(episode_id)

    def forget(self, episode_id: str) -> None:
        """Drop this episode's timer state (e.g. on close) so long-running memory stays bounded."""
        self._armed_at.pop(episode_id, None)
        self._captured.pop(episode_id, None)


@dataclass(frozen=True)
class EventCapsule:
    """The composite evidence assembled for one fired concern under ``Temporal == "event_capsule"``.

    All three components are OPTIONAL — an incomplete capsule degrades gracefully (empty tuple / ``None``)
    rather than raising, so a caller missing one input (e.g. no episode identity) still gets whatever the
    others supply. :attr:`is_complete` is the exit-criteria predicate: all three present.
    """

    episode_id: str | None
    pre_roll: tuple[Frame, ...] = ()
    boundary: tuple[Frame, ...] = ()
    early_snapshot: Frame | None = None

    def frames(self) -> list[Frame]:
        """All captured frames, oldest -> newest, de-duplicated by timestamp — a boundary frame that
        also falls inside the pre-roll span (or IS the early snapshot) is not doubled."""
        by_t: dict[float, Frame] = {}
        for f in self.pre_roll:
            by_t[f[0]] = f
        for f in self.boundary:
            by_t[f[0]] = f
        if self.early_snapshot is not None:
            by_t[self.early_snapshot[0]] = self.early_snapshot
        return [by_t[t] for t in sorted(by_t)]

    @property
    def has_pre_roll(self) -> bool:
        return bool(self.pre_roll)

    @property
    def has_boundary(self) -> bool:
        return bool(self.boundary)

    @property
    def has_early_snapshot(self) -> bool:
        return self.early_snapshot is not None

    @property
    def is_complete(self) -> bool:
        """All three components present — what a fixture test asserts to prove the capsule assembled."""
        return self.has_pre_roll and self.has_boundary and self.has_early_snapshot


def assemble_event_capsule(
    window: Any,
    episode: Any,
    timer: EarlySnapshotTimer,
    *,
    pre_s: float | None = None,
) -> EventCapsule:
    """Assemble the composite evidence for one fired concern. Pure / side-effect free.

    ``window`` is the existing rolling ring (:class:`percept.events.FrameWindow`, or any object /
    iterable yielding ``(t, frame)`` pairs via a ``.frames()`` method) — the SAME structure
    :func:`percept.incidents.snippet_from_window` already reads for the incident's pre-roll; this
    function builds no new ring. ``pre_s`` bounds the pre-roll by time exactly like that function's own
    ``pre_s`` (last ``pre_s`` seconds of the window); ``None`` takes the whole window.

    ``episode`` is a :class:`percept.episodes.Episode` (or any object duck-typing ``id`` /
    ``started_at`` / ``exemplar`` / ``last_seen`` / ``representative``) — ``None`` yields an empty
    boundary (an unbound observation, mirroring :func:`percept.behavior.episode_fold`'s ``unbound``
    case). A field that is itself ``None`` (e.g. no ``exemplar`` recorded) is skipped rather than
    fabricated.

    ``timer`` is the :class:`EarlySnapshotTimer` already fed via :meth:`~EarlySnapshotTimer.observe`
    on each tick; this function only READS it (:meth:`~EarlySnapshotTimer.snapshot`), so assembly never
    mutates timer state and is safe to call more than once (e.g. from a shadow path).
    """
    frames = window.frames() if hasattr(window, "frames") else list(window)
    pre_roll: tuple[Frame, ...] = ()
    if frames:
        if pre_s is not None:
            cutoff = frames[-1][0] - pre_s
            frames = [(t, f) for (t, f) in frames if t >= cutoff]
        pre_roll = tuple(frames)

    episode_id = getattr(episode, "id", None) if episode is not None else None
    boundary_frames: list[Frame] = []
    if episode is not None:
        started_at = getattr(episode, "started_at", None)
        exemplar = getattr(episode, "exemplar", None)
        if started_at is not None and exemplar is not None:
            boundary_frames.append((started_at, exemplar))
        last_seen = getattr(episode, "last_seen", None)
        representative = getattr(episode, "representative", None)
        if (
            last_seen is not None
            and representative is not None
            and (last_seen, representative) not in boundary_frames
        ):
            boundary_frames.append((last_seen, representative))

    early_snapshot = timer.snapshot(episode_id) if episode_id is not None else None

    return EventCapsule(
        episode_id=episode_id,
        pre_roll=pre_roll,
        boundary=tuple(boundary_frames),
        early_snapshot=early_snapshot,
    )


# --------------------------------------------------------------------------------------------------
# Trace — an ADDITIVE trace stage (percept.tracing.STAGE_CAPSULE_ASSEMBLED). Counts/flags only, no
# frame bytes, mirroring the media-off ``.ptrace`` discipline percept.behavior's shadow advisory
# already follows (``_fold_advisory``). This function GOVERNS NOTHING — assembly already happened.
# --------------------------------------------------------------------------------------------------


def capsule_trace_data(capsule: EventCapsule) -> dict:
    """JSON-safe advisory dict for the ``capsule.assembled`` trace stage. All scalars, so it survives a
    media-OFF ``.ptrace`` round-trip like every other advisory stage."""
    return {
        "episode_id": capsule.episode_id,
        "pre_roll_n": len(capsule.pre_roll),
        "boundary_n": len(capsule.boundary),
        "has_early_snapshot": capsule.has_early_snapshot,
        "is_complete": capsule.is_complete,
    }


# --------------------------------------------------------------------------------------------------
# Retention — capsule frames go through the EXISTING EvidenceStore retention mechanism (ADR-003).
# This module never builds a new store; it only calls ``EvidenceStore.put_bundle`` — the SAME seam
# incidents.py's clip snippets use for a heterogeneous multi-artifact capture.
# --------------------------------------------------------------------------------------------------


def _default_encode(frame: Any) -> tuple[bytes, str]:
    """Encode one opaque frame payload to ``(bytes, media_type)``.

    Recognises the ``(mime, bytes)`` tuple shape :func:`percept.incidents._encode_frame_bundle` already
    accepts; anything else (a test double, an opaque object) falls back to its ``repr()`` so assembly
    never raises on an unrecognised frame type — matching that function's own fallback."""
    if (
        isinstance(frame, tuple)
        and len(frame) == 2
        and isinstance(frame[0], str)
        and isinstance(frame[1], (bytes, bytearray))
    ):
        return bytes(frame[1]), frame[0]
    return repr(frame).encode("utf-8"), "application/x-percept-frame-repr"


def store_event_capsule(
    store: Any,
    capsule: EventCapsule,
    *,
    encode: "Callable[[Any], tuple[bytes, str]] | None" = None,
    retention_class: str = "fire",
) -> Any | None:
    """Persist a capsule's frames through the caller's :class:`percept.evidence.EvidenceStore`.

    Every component becomes one ordered child of a SINGLE ``put_bundle`` parent, so the pre-roll /
    boundary / early-snapshot spans expire and evict TOGETHER as one artifact under
    ``retention_class`` (default ``"fire"`` — the same class an incident snippet uses; ADR-003 §5 names
    the closed class vocabulary this module never extends). Returns ``None`` for an empty capsule
    (nothing to store); otherwise the parent :class:`~percept.evidence.EvidenceArtifact` handle.
    """
    enc = encode or _default_encode
    ordered = capsule.frames()
    if not ordered:
        return None
    items: Iterable[tuple[bytes, str]] = [enc(frame) for (_t, frame) in ordered]
    return store.put_bundle(
        items,
        capture_start_ts=ordered[0][0],
        capture_end_ts=ordered[-1][0],
        retention_class=retention_class,
        media_type="application/x-percept-event-capsule",
    )


__all__ = [
    "Frame",
    "EarlySnapshotTimer",
    "EventCapsule",
    "assemble_event_capsule",
    "capsule_trace_data",
    "store_event_capsule",
]
