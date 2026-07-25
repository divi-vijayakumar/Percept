"""Concern-level SELECTION of the subject-identity resolver (ADR-005 open-question 1).

``SourceProfile.identity_strategy`` is too coarse to be the PRIMARY selector. One screen source
legitimately carries three different identity policies at once: a document content-key for a
purchase-order goal, a track handle for a cursor / physical-object goal, and NO subject at all for a
global workflow-state goal. One conversation carries claim-slot goals, speaker-specific goals and
session-wide goals. The source class cannot know which of these a given goal wants — only the goal
does.

So the resolver is chosen the way every other concern-level knob is chosen in percept
(``percept.episodes.episode_policy_of``, the authored ``goal.state`` channel): **the source profile
DECLARES a default/capability; the concern SELECTS.**

    resolver_name = concern.identity_policy  OR  source_profile.default_identity_policy
    #                (goal.state['identity_policy'])      (== SourceProfile.identity_strategy)
    #                 ^ concern wins when it declared one

Additive by construction: a concern that authors nothing gets ``source_profile.identity_strategy``,
which for every shipped profile is exactly today's value — unselected behaviour is byte-for-byte
unchanged.

Envelope relevance (coordinate with ADR-007 §``security_person_id``)
------------------------------------------------------------------------------

Making selection per-concern is also what stops a future biometric resolver from becoming GLOBALLY
active merely because the source happens to be a security camera. A source profile's default is a
capability a concern may decline, never a mandate the runtime imposes on every goal. And selection
re-checks the envelope: :func:`select_identity_policy` refuses to hand back a policy that resolves to
a person-identifying resolver — the same refusal :meth:`SubjectResolverRegistry.register` already
makes at registration time, re-applied at SELECTION time so a resolver that reached the registry by
some other door still cannot be switched on by naming it in a profile default.

Recognized no-subject scope (ADR-005 vocabulary)
------------------------------------------------------------------------------

``"scene"`` (:data:`SCENE_IDENTITY`) is the recognized name for the *global workflow-state* scope the
module preamble anticipates — "NO subject at all". It is a scope value, NOT a SubjectResolver (it is
deliberately never registered), and it is the stable value transition goals latch against when no
per-entity resolver applies (see :func:`transition_latch_key`).
"""

from __future__ import annotations

from typing import Any, Hashable

from .envelope import (
    DEFAULT_SUBJECT_RESOLVERS,
    OUT_OF_ENVELOPE,
    EnvelopeViolation,
    SubjectResolverRegistry,
)
from .sourceprofile import SourceProfile

#: The ``goal.state`` key a concern authors to SELECT its identity resolver — the same validated
#: state channel ``episode_policy_of`` / ``sense`` / ``perception`` already use. Absent (or blank /
#: non-string) means "declared nothing" and the source-profile default is inherited.
STATE_KEY = "identity_policy"

#: The recognized "no subject at all" scope (ADR-005) — a global workflow-state scope, NOT a
#: SubjectResolver (deliberately never registered). It is the stable value transition goals latch
#: against when no per-entity resolver applies.
SCENE_IDENTITY = "scene"

#: The shipped resolver names that scope to a per-entity handle (a scene-local OBJECT track). Only
#: these re-scope a transition latch to the in-focus entity; every other declared policy
#: (``content_key``, ``speaker_label``, an unknown name) latches on the stable scene scope, so a
#: volatile per-tick focus never splits the latch. Mirrors the shipped
#: :data:`~percept.envelope.DEFAULT_SUBJECT_RESOLVERS` object resolvers.
PER_ENTITY_IDENTITY_POLICIES: frozenset[str] = frozenset(
    {"track_iou_label", "appearance_recent_set"}
)


def transition_latch_key(
    declared_policy: str | None, focus_id: Hashable
) -> tuple[Hashable, str]:
    """The stable key a transition goal's rising-edge latch is scoped by, plus its provenance.

    A transition goal arms on state A and fires on a later state B; the two observations must latch
    against the SAME key or the edge is missed. Keying by the volatile in-focus ``focus.id`` splits
    that latch the moment the in-focus entity changes between arming and completing. So the latch is
    scoped by the concern's DECLARED identity policy, and the contract is **fail-toward-stable**: when
    no per-entity resolver was declared (or a per-entity resolver was declared but nothing is in
    focus), the key falls back to the literal stable :data:`SCENE_IDENTITY` (never a per-tick value),
    so a scene / workflow-state goal latches globally and fires across a focus flip.

    Returns ``(latch_key, latch_key_source)``. The mapping (consulting ONLY ``declared_policy``):

    * ``None`` -> ``("scene", "default_scene")`` — nothing declared, global scene scope.
    * ``"scene"`` / :data:`SCENE_IDENTITY` -> ``("scene", "declared")`` — explicit scene scope.
    * a per-entity policy (:data:`PER_ENTITY_IDENTITY_POLICIES`) WITH ``focus_id`` present ->
      ``(focus_id, "declared")`` — per-entity isolation holds.
    * a per-entity policy with ``focus_id`` ``None`` -> ``("scene", "fallback_scene")``.
    * anything else (``content_key``, ``speaker_label``, an unknown name) ->
      ``("scene", "fallback_scene")``.

    The fallback is ALWAYS the literal ``"scene"``, never a per-tick value.
    """
    if declared_policy is None:
        return SCENE_IDENTITY, "default_scene"
    if declared_policy == SCENE_IDENTITY:
        return SCENE_IDENTITY, "declared"
    if declared_policy in PER_ENTITY_IDENTITY_POLICIES:
        if focus_id is not None:
            return focus_id, "declared"
        return SCENE_IDENTITY, "fallback_scene"
    return SCENE_IDENTITY, "fallback_scene"


def identity_policy_of(goal: Any) -> str | None:
    """The identity policy a CONCERN declared, or ``None`` when it declared none.

    Reads ``goal.state['identity_policy']`` — a non-empty string is the resolver NAME this concern
    selects. Anything else (missing key, blank, non-string, no ``state`` at all) is "declared
    nothing", so :func:`select_identity_policy` falls back to the source-profile default and the
    concern is byte-for-byte unchanged. Mirrors :func:`percept.episodes.episode_policy_of`.
    """
    state = getattr(goal, "state", None)
    if not isinstance(state, dict):
        return None
    value = state.get(STATE_KEY)
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def watch_reducer_entity_id(goal: Any, focus_id: Hashable) -> Hashable:
    """Choose the ordinary judged-Watch reducer entity without changing legacy scope.

    Only a concern that explicitly declares ``identity_policy="scene"`` opts out of focus identity
    and reduces onto the existing non-entity key ``None``. Undeclared concerns — including those
    whose source profile happens to inherit a scene-like default — and every other declared policy
    keep the historical focus entity unchanged.
    """

    return None if identity_policy_of(goal) == SCENE_IDENTITY else focus_id


def _assert_within_envelope(
    policy: str,
    resolvers: SubjectResolverRegistry,
    *,
    selected_by: str,
) -> None:
    """Refuse to activate a person-identifying resolver, however it entered ``resolvers``.

    ``register()`` already refuses to register such a resolver; this is the SELECTION-time twin, so
    naming a person resolver as a profile default (or a concern's choice) is refused at the point of
    use even if the object reached the registry without going through the constructor's guard.
    """
    resolver = resolvers.get(policy)
    if resolver is not None and getattr(resolver, "identifies_persons", False):
        raise EnvelopeViolation(
            f"identity policy {policy!r} (selected by {selected_by}) resolves to a "
            f"person-identifying subject resolver. "
            + OUT_OF_ENVELOPE["security_person_id"].sentence()
        )


def select_identity_policy(
    goal: Any,
    source_profile: SourceProfile,
    *,
    resolvers: SubjectResolverRegistry = DEFAULT_SUBJECT_RESOLVERS,
) -> str:
    """Choose the identity resolver for THIS concern on THIS source. Concern wins.

    Returns ``goal.state['identity_policy']`` when the concern declared one, otherwise
    ``source_profile.default_identity_policy`` (i.e. the profile's ``identity_strategy``). The
    returned name is a resolver NAME — an open, free-text string (ADR-005): it need not be one of
    the shipped resolvers, so a client's registered custom resolver is selectable too. The one thing
    it may never be is a person-identifying resolver (:func:`_assert_within_envelope`).

    Purely additive: a concern that authored nothing gets the profile default, which for shipped
    profiles is today's ``identity_strategy`` unchanged.
    """
    declared = identity_policy_of(goal)
    if declared is not None:
        policy, selected_by = declared, "concern (goal.state['identity_policy'])"
    else:
        policy, selected_by = source_profile.default_identity_policy, "source-profile default"
    _assert_within_envelope(policy, resolvers, selected_by=selected_by)
    return policy


__all__ = [
    "STATE_KEY",
    "SCENE_IDENTITY",
    "PER_ENTITY_IDENTITY_POLICIES",
    "transition_latch_key",
    "identity_policy_of",
    "watch_reducer_entity_id",
    "select_identity_policy",
]
