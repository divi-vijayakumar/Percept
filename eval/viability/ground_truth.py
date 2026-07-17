"""Fixture-facing helper for GroundTruth-by-construction (card A6; consumed by A14/A16/A27).

Card A6 (`percept.contracts.GroundTruth`) defines the EVALUATION-ONLY ground-truth record and its
two construction paths (see `percept.contracts.ANNOTATION_RUBRIC`). This module is the fixture
AUTHOR's side of the primary "by construction" path (Viability §5a): for the six CORE gating
archetypes, it documents — as a single named constant per archetype — the deterministic
eligibility offset a synthetic fixture generator uses, so every fixture in the neutral archetype
set (card A16) derives its `decision_eligible_ts` the same, reproducible way instead of each
fixture author picking an ad-hoc number.

This is scaffolding for A16, not A16 itself: A16 owns the actual clip generation / labeled fixture
set. Nothing here reads pixels or generates video; it only turns
`(archetype, event_start_ts, fixture_id)` into a `GroundTruth` record.
"""

from __future__ import annotations

from percept.contracts import GroundTruth

#: The six CORE gating archetypes named in the Viability spec's neutral fixture set (§5, card A16).
#: Each archetype's offset is the fixture author's DOCUMENTED reaction-time/resolution delay between
#: the scripted `event_start_ts` and the first frame at which a competent human (or, here, the
#: fixture's own deterministic generator) could confidently judge the condition — i.e. exactly what
#: `ANNOTATION_RUBRIC` describes, applied "by construction" instead of by a human annotator.
#:
#: These are PLACEHOLDER defaults (seconds) pending A16's actual fixture design; they exist so the
#: by-construction PATH is wired now, deterministically, rather than blocking on final numbers. A16
#: may override per-fixture via `by_construction(..., eligible_after_s=...)` directly.
CORE_ARCHETYPE_ELIGIBLE_AFTER_S: dict[str, float] = {
    # A state is true/false at every instant once established — eligibility is ~immediate (the
    # frame the state is visible in) plus a small margin for a full pixel/frame to register.
    "state": 0.1,
    # A transition (e.g. door closed -> open) needs the motion to resolve into an unambiguous new
    # state before a human would commit to a judgment — a slightly longer margin than "state".
    "transition": 0.5,
    # A brief action (a wave, a reach) is only judgeable once enough of the action has played out
    # to distinguish it from a similar-looking partial motion.
    "brief_action": 0.75,
    # Persistence (condition held for N seconds) is eligible only once the hold duration itself has
    # elapsed — this offset is typically SUPPLIED per-fixture (the hold length), this is a floor.
    "persistence": 1.0,
    # Absence (something that should be present is missing) needs a confirmation window long enough
    # to rule out "just out of frame for a moment" — the longest default offset of the six.
    "absence": 2.0,
    # A repeated episode (event recurs N times) is eligible once the Nth repetition itself resolves;
    # like persistence this is usually fixture-specific — this is a floor for a single repetition.
    "repeated_episode": 0.5,
}


def ground_truth_for_archetype(
    archetype: str,
    *,
    event_start_ts: float,
    fixture_id: str,
    eligible_after_s: float | None = None,
) -> GroundTruth:
    """Build a by-construction :class:`GroundTruth` for one of the six core gating archetypes.

    Looks up the archetype's documented default offset in
    :data:`CORE_ARCHETYPE_ELIGIBLE_AFTER_S` unless the caller supplies its own
    ``eligible_after_s`` (e.g. a persistence/repeated-episode fixture with a specific hold length).
    Raises :class:`KeyError` for an archetype name outside the core six — this helper intentionally
    does not guess an offset for an archetype it doesn't recognize.
    """
    if eligible_after_s is None:
        if archetype not in CORE_ARCHETYPE_ELIGIBLE_AFTER_S:
            raise KeyError(
                f"unknown core archetype {archetype!r} — expected one of "
                f"{sorted(CORE_ARCHETYPE_ELIGIBLE_AFTER_S)}, or pass eligible_after_s= explicitly"
            )
        eligible_after_s = CORE_ARCHETYPE_ELIGIBLE_AFTER_S[archetype]
    return GroundTruth.by_construction(
        event_start_ts=event_start_ts,
        eligible_after_s=eligible_after_s,
        fixture_id=fixture_id,
    )
