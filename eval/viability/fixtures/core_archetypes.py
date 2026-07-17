"""Neutral archetype golden fixtures — CORE gating set only (Viability §5, board card A16).

Generates deterministic, synthetic :class:`~percept.harness.stream_gen.Timeline` clips for the
SIX core gating archetypes named in the Viability spec:

    1. state             — object present inside an ROI
    2. transition         — object placed and released
    3. brief_action        — hand wave / short gesture
    4. persistence         — door remains open for N seconds
    5. absence             — person leaves a marked workstation
    6. repeated_episode     — repeated object transfer (rearm + episode separation)

HARD SCOPE: the supplemental archetypes (audio-visual indicator+alarm, specialist-verification)
are OUT OF SCOPE for this module — they are non-gating and pull in audio/specialist modalities not
allowed in v0.1. This module only ever builds fixtures for the six names above (see
:data:`CORE_ARCHETYPES`); :data:`EXCLUDED_SUPPLEMENTAL_ARCHETYPES` documents the two names that
must never appear here, for the test suite to assert against.

Every fixture's `decision_eligible_ts` is produced BY CONSTRUCTION (Viability §5a, the PRIMARY
substrate): each generator places a scripted `event_start_ts` on the clip's own timeline and calls
`eval.viability.ground_truth.ground_truth_for_archetype(...)`, which turns that into a
deterministic `GroundTruth` — no pixels are read, no human annotator is involved, and two calls
with identical inputs always produce byte-identical output (see
`percept.contracts.GroundTruth.by_construction`).

Frame content is deliberately the same coarse grayscale grid tier-0 already uses
(`percept.harness.tier0.Frame`) built via `percept.harness.stream_gen`'s timeline helpers, so a
generated fixture drops straight into `MotionGate` / `load_driver` like any other `stream_gen`
clip. Two disjoint pixel regions carry the two signals a fixture needs:

  * `_ROI_INDEXES` — "the marked ROI / workstation" a state/transition/persistence/absence fixture
    watches (object present, door open, person present).
  * `_MOTION_INDEXES` — a disjoint quadrant used to render transient hand/gesture motion (entering,
    waving, releasing) that never overlaps the ROI, so a pure ROI-content check is unaffected by
    motion happening elsewhere in frame, while a whole-frame motion gate still sees it.

Each archetype has at least one POSITIVE fixture (the condition is scripted to actually resolve —
`ground_truth` is a real `GroundTruth`) and at least one NEGATIVE fixture (the condition never
resolves — e.g. a flicker, an early release, an unconfirmed absence — `ground_truth=None` signals
"no eligible judgment; a correct gate must never fire on this clip").
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

# `eval/viability` is not (yet) an installed package (see `ground_truth.py`'s own test) — make its
# directory importable the same way `eval/viability/test_ground_truth.py` does, so this module can
# `from ground_truth import ...` regardless of how it itself is imported/run.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ground_truth import (  # noqa: E402
    CORE_ARCHETYPE_ELIGIBLE_AFTER_S,
    ground_truth_for_archetype,
)
from percept.contracts import GroundTruth  # noqa: E402
from percept.harness.stream_gen import Timeline, fixture_clip  # noqa: E402
from percept.harness.tier0 import Frame  # noqa: E402

#: The six CORE gating archetypes this module builds fixtures for — must match
#: `ground_truth.CORE_ARCHETYPE_ELIGIBLE_AFTER_S` exactly (enforced by a test).
CORE_ARCHETYPES: frozenset[str] = frozenset(CORE_ARCHETYPE_ELIGIBLE_AFTER_S)

#: The two supplemental (non-gating) archetypes named in the Viability spec that are explicitly
#: OUT OF SCOPE for card A16 — audio/specialist modalities not allowed in v0.1. Named here only so
#: the test suite has something concrete to assert is absent from this module's output.
EXCLUDED_SUPPLEMENTAL_ARCHETYPES: frozenset[str] = frozenset(
    {"audio_visual_indicator_alarm", "specialist_verification"}
)

# ---------------------------------------------------------------------------
# Frame construction — two disjoint pixel regions on an 8x8 coarse grid
# ---------------------------------------------------------------------------

_W, _H = 8, 8
_ROI_INDEXES: tuple[int, ...] = tuple(r * _W + c for r in range(0, 4) for c in range(0, 4))
_MOTION_INDEXES: tuple[int, ...] = tuple(r * _W + c for r in range(4, 8) for c in range(4, 8))

_BG = 20
_ROI_FILL = 235
_MOTION_FILL = 235

_DT = 0.1  # 10 Hz sampling — fine enough to resolve the smallest offset (state: 0.1s)


def _frame(*, roi_on: bool, motion_on: bool = False) -> Frame:
    """One coarse grayscale frame: ROI quadrant on/off, disjoint motion quadrant on/off."""
    pixels = [_BG] * (_W * _H)
    if roi_on:
        for i in _ROI_INDEXES:
            pixels[i] = _ROI_FILL
    if motion_on:
        for i in _MOTION_INDEXES:
            pixels[i] = _MOTION_FILL
    return Frame(w=_W, h=_H, pixels=tuple(pixels))


def _segments_to_timeline(
    segments: Sequence[tuple[float, bool, bool]], *, dt: float = _DT, start: float = 0.0
) -> Timeline:
    """Expand ``[(duration_s, roi_on, motion_on), ...]`` into a fixed-cadence Timeline.

    Each segment holds its (roi_on, motion_on) state for ``duration_s`` seconds, sampled at
    ``dt``. Segment durations are expected to be (near-)exact multiples of ``dt``.
    """
    frames: list[Frame] = []
    for duration_s, roi_on, motion_on in segments:
        n = round(duration_s / dt)
        frames.extend(_frame(roi_on=roi_on, motion_on=motion_on) for _ in range(n))
    return fixture_clip(frames, dt=dt, start=start)


@dataclass(frozen=True)
class ArchetypeFixture:
    """One neutral archetype fixture: a deterministic clip plus its by-construction ground truth.

    ``ground_truth`` is ``None`` for NEGATIVE variants — the scripted condition never resolves
    (a flicker, an early release, an unconfirmed absence), so there is no eligible judgment instant
    and a correct gate must never fire on this clip.
    """

    fixture_id: str
    archetype: str
    positive: bool
    description: str
    timeline: Timeline
    ground_truth: GroundTruth | None
    meta: dict[str, object]

    def __post_init__(self) -> None:
        if self.archetype not in CORE_ARCHETYPES:
            raise ValueError(
                f"fixture {self.fixture_id!r} uses archetype {self.archetype!r}, "
                f"not one of the core six: {sorted(CORE_ARCHETYPES)}"
            )
        if self.positive and self.ground_truth is None:
            raise ValueError(f"positive fixture {self.fixture_id!r} must carry a GroundTruth")
        if not self.positive and self.ground_truth is not None:
            raise ValueError(f"negative fixture {self.fixture_id!r} must NOT carry a GroundTruth")


# ---------------------------------------------------------------------------
# 1. state — object present inside an ROI
# ---------------------------------------------------------------------------


def _build_state_fixtures() -> tuple[ArchetypeFixture, ...]:
    event_start_ts = 1.0

    positive = _segments_to_timeline(
        [
            (1.0, False, False),  # baseline: ROI empty
            (1.0, True, False),  # object present in ROI from event_start_ts onward
        ]
    )
    negative = _segments_to_timeline(
        [
            (2.0, False, False),  # ROI stays empty for the whole clip — no event
        ]
    )
    return (
        ArchetypeFixture(
            fixture_id="state_object_present",
            archetype="state",
            positive=True,
            description="Object placed inside the watched ROI at t=1.0s and held for the rest of the clip.",
            timeline=positive,
            ground_truth=ground_truth_for_archetype(
                "state", event_start_ts=event_start_ts, fixture_id="state_object_present"
            ),
            meta={"event_start_ts": event_start_ts},
        ),
        ArchetypeFixture(
            fixture_id="state_object_absent",
            archetype="state",
            positive=False,
            description="ROI stays empty for the entire clip — the state never becomes true.",
            timeline=negative,
            ground_truth=None,
            meta={},
        ),
    )


# ---------------------------------------------------------------------------
# 2. transition — object placed and released
# ---------------------------------------------------------------------------


def _build_transition_fixtures() -> tuple[ArchetypeFixture, ...]:
    event_start_ts = 1.3  # the placement/contact instant

    positive = _segments_to_timeline(
        [
            (1.0, False, False),  # baseline
            (0.3, False, True),  # hand enters (motion, ROI still empty)
            (0.1, True, True),  # contact: object placed, hand still present
            (0.2, True, True),  # hand lingers/retreats, object stays
            (0.9, True, False),  # hand fully released, object remains
        ]
    )
    negative = _segments_to_timeline(
        [
            (1.0, False, False),  # baseline
            (0.3, False, True),  # hand enters
            (0.1, True, True),  # brief touch — object momentarily appears
            (0.2, False, True),  # hand withdraws WITH the object — never released
            (0.9, False, False),  # settles back to empty — no lasting state change
        ]
    )
    return (
        ArchetypeFixture(
            fixture_id="transition_object_placed_and_released",
            archetype="transition",
            positive=True,
            description=(
                "Hand enters ROI, places an object at t=1.3s, then releases and withdraws; "
                "object remains — a resolved closed->open-style transition."
            ),
            timeline=positive,
            ground_truth=ground_truth_for_archetype(
                "transition",
                event_start_ts=event_start_ts,
                fixture_id="transition_object_placed_and_released",
            ),
            meta={"event_start_ts": event_start_ts},
        ),
        ArchetypeFixture(
            fixture_id="transition_touch_without_release",
            archetype="transition",
            positive=False,
            description=(
                "Hand touches the ROI but withdraws the object again instead of releasing it — "
                "no lasting state change, the transition never resolves."
            ),
            timeline=negative,
            ground_truth=None,
            meta={},
        ),
    )


# ---------------------------------------------------------------------------
# 3. brief_action — hand wave / short gesture
# ---------------------------------------------------------------------------


def _build_brief_action_fixtures() -> tuple[ArchetypeFixture, ...]:
    event_start_ts = 1.0

    # Three on/off toggles = a recognizable multi-cycle wave, never touching the ROI.
    positive = _segments_to_timeline(
        [
            (1.0, False, False),  # baseline, still
            (0.1, False, True),
            (0.1, False, False),
            (0.1, False, True),
            (0.1, False, False),
            (0.1, False, True),
            (0.1, False, False),
            (0.9, False, False),  # settle back to still
        ]
    )
    negative = _segments_to_timeline(
        [
            (1.0, False, False),  # baseline, still
            (0.1, False, True),  # a single flicker — not a deliberate gesture
            (1.5, False, False),  # settle back to still
        ]
    )
    return (
        ArchetypeFixture(
            fixture_id="brief_action_wave_recognized",
            archetype="brief_action",
            positive=True,
            description="A three-cycle hand wave starting at t=1.0s, resolving into a recognizable gesture.",
            timeline=positive,
            ground_truth=ground_truth_for_archetype(
                "brief_action",
                event_start_ts=event_start_ts,
                fixture_id="brief_action_wave_recognized",
            ),
            meta={"event_start_ts": event_start_ts},
        ),
        ArchetypeFixture(
            fixture_id="brief_action_single_flicker",
            archetype="brief_action",
            positive=False,
            description="A single one-frame motion flicker — too brief/un-repeated to count as a recognized gesture.",
            timeline=negative,
            ground_truth=None,
            meta={},
        ),
    )


# ---------------------------------------------------------------------------
# 4. persistence — door remains open for N seconds
# ---------------------------------------------------------------------------

_PERSISTENCE_HOLD_S = 3.0  # the "N seconds" this fixture set requires


def _build_persistence_fixtures() -> tuple[ArchetypeFixture, ...]:
    event_start_ts = 0.5  # door opens

    positive = _segments_to_timeline(
        [
            (0.5, False, False),  # door closed
            (3.5, True, False),  # door opens at t=0.5s and stays open through the hold + buffer
        ]
    )
    negative = _segments_to_timeline(
        [
            (0.5, False, False),  # door closed
            (1.5, True, False),  # door opens at t=0.5s ...
            (2.0, False, False),  # ... but closes again at t=2.0s, only 1.5s < required 3.0s hold
        ]
    )
    return (
        ArchetypeFixture(
            fixture_id="persistence_door_held_open",
            archetype="persistence",
            positive=True,
            description=f"Door opens at t=0.5s and stays open for the full required {_PERSISTENCE_HOLD_S}s hold.",
            timeline=positive,
            ground_truth=ground_truth_for_archetype(
                "persistence",
                event_start_ts=event_start_ts,
                fixture_id="persistence_door_held_open",
                eligible_after_s=_PERSISTENCE_HOLD_S,
            ),
            meta={"event_start_ts": event_start_ts, "hold_s": _PERSISTENCE_HOLD_S},
        ),
        ArchetypeFixture(
            fixture_id="persistence_door_closes_early",
            archetype="persistence",
            positive=False,
            description=f"Door opens at t=0.5s but closes again after 1.5s — short of the required {_PERSISTENCE_HOLD_S}s hold.",
            timeline=negative,
            ground_truth=None,
            meta={"hold_s": _PERSISTENCE_HOLD_S},
        ),
    )


# ---------------------------------------------------------------------------
# 5. absence — person leaves a marked workstation
# ---------------------------------------------------------------------------


def _build_absence_fixtures() -> tuple[ArchetypeFixture, ...]:
    event_start_ts = 1.0  # person leaves

    positive = _segments_to_timeline(
        [
            (1.0, True, False),  # person present at the workstation
            (2.5, False, False),  # leaves at t=1.0s and stays away through confirmation + buffer
        ]
    )
    negative = _segments_to_timeline(
        [
            (1.0, True, False),  # person present
            (0.8, False, False),  # leaves at t=1.0s ...
            (1.7, True, False),  # ... but returns at t=1.8s, before the 2.0s confirmation window
        ]
    )
    return (
        ArchetypeFixture(
            fixture_id="absence_person_leaves",
            archetype="absence",
            positive=True,
            description="Person leaves the marked workstation at t=1.0s and stays away past the confirmation window.",
            timeline=positive,
            ground_truth=ground_truth_for_archetype(
                "absence", event_start_ts=event_start_ts, fixture_id="absence_person_leaves"
            ),
            meta={"event_start_ts": event_start_ts},
        ),
        ArchetypeFixture(
            fixture_id="absence_person_returns_before_confirmation",
            archetype="absence",
            positive=False,
            description="Person leaves at t=1.0s but returns at t=1.8s — a flicker, never a confirmed absence.",
            timeline=negative,
            ground_truth=None,
            meta={},
        ),
    )


# ---------------------------------------------------------------------------
# 6. repeated_episode — repeated object transfer (rearm + episode separation)
# ---------------------------------------------------------------------------

_REPEAT_ON_S = 0.3
_REPEAT_PERIOD_S = 1.2  # start-to-start spacing: 0.9s clear gap between episodes, well clear of rearm
_REPEAT_STARTS = (0.5, 1.7, 2.9)  # 3 repetitions


def _repeated_episode_segments(starts: Sequence[float], *, tail_s: float) -> list[tuple[float, bool, bool]]:
    segments: list[tuple[float, bool, bool]] = []
    cursor = 0.0
    for s in starts:
        gap = s - cursor
        if gap > 0:
            segments.append((gap, False, False))
        segments.append((_REPEAT_ON_S, True, False))
        cursor = s + _REPEAT_ON_S
    segments.append((tail_s, False, False))
    return segments


def _build_repeated_episode_fixtures() -> tuple[ArchetypeFixture, ...]:
    final_event_start_ts = _REPEAT_STARTS[-1]

    positive = _segments_to_timeline(_repeated_episode_segments(_REPEAT_STARTS, tail_s=0.5))

    # Negative A: only 2 of the required 3 repetitions occur within the observed window.
    insufficient = _segments_to_timeline(
        _repeated_episode_segments(_REPEAT_STARTS[:2], tail_s=0.5)
    )

    # Negative B: repetitions are spaced too close together to be separable episodes (no clear
    # rearm gap) — a correctly-gated detector must not count these as 3 distinct episodes.
    close_starts = (0.5, 0.9, 1.3)  # on_s=0.3 -> only 0.1s gap between episodes
    blurred = _segments_to_timeline(_repeated_episode_segments(close_starts, tail_s=0.5))

    return (
        ArchetypeFixture(
            fixture_id="repeated_episode_object_transfer_x3",
            archetype="repeated_episode",
            positive=True,
            description=(
                f"Three clearly-separated object-transfer episodes at t={_REPEAT_STARTS}, "
                "each separated by a 0.9s gap well clear of rearm."
            ),
            timeline=positive,
            ground_truth=ground_truth_for_archetype(
                "repeated_episode",
                event_start_ts=final_event_start_ts,
                fixture_id="repeated_episode_object_transfer_x3",
            ),
            meta={"episode_starts": _REPEAT_STARTS, "event_start_ts": final_event_start_ts},
        ),
        ArchetypeFixture(
            fixture_id="repeated_episode_insufficient_repetitions",
            archetype="repeated_episode",
            positive=False,
            description="Only 2 of the required 3 repetitions occur before the clip ends.",
            timeline=insufficient,
            ground_truth=None,
            meta={"episode_starts": _REPEAT_STARTS[:2]},
        ),
        ArchetypeFixture(
            fixture_id="repeated_episode_episodes_too_close",
            archetype="repeated_episode",
            positive=False,
            description=(
                "Three transfers occur but only 0.1s apart — too close to separate into distinct "
                "episodes (rearm/episode-separation stress case)."
            ),
            timeline=blurred,
            ground_truth=None,
            meta={"episode_starts": close_starts},
        ),
    )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_BUILDERS = (
    _build_state_fixtures,
    _build_transition_fixtures,
    _build_brief_action_fixtures,
    _build_persistence_fixtures,
    _build_absence_fixtures,
    _build_repeated_episode_fixtures,
)


def build_core_archetype_fixtures() -> tuple[ArchetypeFixture, ...]:
    """Build every neutral CORE-archetype fixture (six archetypes, positive + negative variants).

    Pure and deterministic: repeated calls return equal (value-equal) fixtures — no randomness, no
    wall-clock, no I/O. Excludes the two supplemental (audio-visual, specialist-verification)
    archetypes by construction (see module docstring / :data:`EXCLUDED_SUPPLEMENTAL_ARCHETYPES`).
    """
    fixtures: list[ArchetypeFixture] = []
    for build in _BUILDERS:
        fixtures.extend(build())
    return tuple(fixtures)


__all__ = [
    "CORE_ARCHETYPES",
    "EXCLUDED_SUPPLEMENTAL_ARCHETYPES",
    "ArchetypeFixture",
    "build_core_archetype_fixtures",
]
