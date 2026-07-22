"""Situation -> WatchProfile resolution rules.

``percept.watchprofile`` is schema + registry: :meth:`WatchProfileRegistry.resolve` walks an
``(environment, goal_type, state)`` key through an exact-match-then-wildcard chain, but it does not know
how to turn a plain-language situation into that key — that inference is deliberately left to a layer
above the registry (see the ``WatchProfileRegistry`` docstring: "this is deliberately just a fallback
CHAIN, not goal-temporality inference ... which sits on top of this registry"). This module IS that layer.

:func:`resolve_situation` is a pure function of three axes:

- **goal temporality** — ``"stative"`` (the goal's truth is a persistent condition, e.g. "the stove is on")
  or ``"dynamic"`` (the goal's truth is an event/transition, e.g. "someone sits down").
- **camera mount** — ``"fixed"`` (tripod / wall-mounted) or a moving mount (``"wearable"``, ``"glasses"``,
  ``"handheld"`` — smart glasses and body-cams are the same camera-motion problem as a handheld phone).
- **start-state** — ``"start_anchored"`` (the condition may already be true at t0, so the runtime must not
  demand a false->true edge it may never see) or ``"transition"`` (only a genuine edge counts).

It returns a :class:`SituationResolution`: a ``key`` — the ``(environment, goal_type, state)`` tuple to
feed :meth:`percept.watchprofile.WatchProfileRegistry.get`/``resolve`` — plus ``overrides``, a plain dict
of the knob values this situation implies (independent of whichever profile the registry happens to have
registered for that key). :func:`resolve_profile` is a convenience that also fetches the registry's
best-match profile for ``key`` and layers the axis-derived overrides on top of it.

Design rationale (grounded in this repo's measured tuning results — cited, not re-derived here):

- **salience_stat is NOT a situation axis (ADR-008 ownership transfer, card C2).** Localized
  (block-max) salience nearly doubled recall vs. a whole-frame mean (0.31 -> 0.59; see the project's
  viability-ladder findings), which is why every shipped :class:`~percept.watchprofile.WatchProfile`
  (``percept.profiles_default``) and both stable :class:`~percept.sourceprofile.SourceProfile`
  entries (``percept.source_profiles_default``) declare ``salience_stat="block_max"`` as THEIR OWN
  default. This module used to re-assert that choice unconditionally as a situation-axis override,
  which silently overrode a differently-tuned registered/source profile — ownership of "which
  statistic" belongs to the profile/source layer (:data:`percept.tunables.REGISTRY`:
  ``Settings.edge.salience_stat`` -> owner ``"SourceProfile"``), not to this axis-derived layer, so
  :func:`resolve_situation` no longer emits ``"salience_stat"``/``"stat_thresholds"`` and
  :func:`resolve_profile` passes the resolved profile's own value through untouched. A profile that
  declares nothing still gets the SDK default (``"mean"``, byte-for-byte legacy); every SHIPPED
  profile still resolves to ``block_max``/0.20 because that is what it declares, not because this
  module forces it.
- **ego-compensation follows camera motion.** A moving, uncompensated camera drove tier-0 sampler
  escalation to ~98% of frames (global motion never lets cheap-motion settle below the escalate threshold)
  vs. ~30% with ego-compensation on (see ``percept.profiles_default.EGO_COMPENSATION_NOTE``). Enabling it
  on a genuinely fixed camera is a no-op (no global motion to compensate), so when the mount is UNKNOWN
  this module defaults ``ego_compensate`` to ``True`` — the failure mode of guessing "fixed" and being
  wrong (98% escalation) is far worse than the failure mode of guessing "moving" and being wrong (a no-op).
- **window beats sprite, unconditionally.** An ordered-window plan (the judge sees ``frames=[...]``) beats
  a sprite montage; sprite is strictly worse and must never be selected. ``resolve_situation`` has no code
  path that can produce ``plan_mode="sprite"`` for any input, including unrecognized/wildcard axis values.
- **start-anchored relaxes the gate.** ``gate.require_false_first=False`` lets an already-true-at-start
  condition fire on first read; this is the gate-relaxation finding. A ``"transition"`` start-state does the
  opposite (``require_false_first=True``), matching :class:`percept.watchprofile.GateProfile`'s own
  documented semantics ("transition-only goals set it True"). An UNKNOWN start-state defaults to ``True``
  (demand a genuine edge) rather than ``False`` — assuming "already true" when that is not known risks an
  immediate, ungrounded fire; requiring an edge is the conservative failure mode.
- **unknown temporality defaults to window, not single.** This matches the already-shipped
  ``percept.profiles_default.WILDCARD_DEFAULT`` profile's own choice of the "safest known default" plan
  mode for a fully-unrecognized context.

This module does no I/O and reads no environment variables: every function here is a pure mapping from its
arguments to a return value, so the same inputs always produce the same outputs.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from .watchprofile import (
    DEFAULT_REGISTRY,
    WILDCARD,
    WatchProfile,
    WatchProfileRegistry,
)

# ---------------------------------------------------------------------------
# Recognized vocabulary for each axis. Anything else (including None / "" / typos) is treated as
# "unknown" for that axis and falls back to the safest default — see the per-axis resolvers below.
# ---------------------------------------------------------------------------

GOAL_TEMPORALITIES: tuple[str, ...] = ("stative", "dynamic")
CAMERA_MOUNTS: tuple[str, ...] = ("fixed", "wearable", "glasses", "handheld")
START_STATES: tuple[str, ...] = ("start_anchored", "transition")

# The plan_mode this module will select for an unrecognized/wildcard goal temporality: the safest known
# default (see module docstring), and NEVER "sprite" (see docstring: sprite is strictly worse and must
# never be selected by this module, for any input).
_WILDCARD_PLAN_MODE = "window"

def _normalize(value: Any) -> str:
    """Lowercase/strip a raw axis value for comparison; non-strings and falsy values normalize to ``""``
    (which never matches a recognized value, so they fall into the "unknown" branch below)."""
    if not isinstance(value, str):
        return ""
    return value.strip().lower()


def _resolve_environment(camera_mount: Any) -> tuple[str, bool]:
    """Camera mount -> ``(environment key segment, ego_compensate default)``.

    ``"fixed"`` -> ``"fixed_camera"``, ego-compensation off (nothing to compensate). ``"wearable"`` /
    ``"glasses"`` / ``"handheld"`` -> ``"handheld"`` (smart glasses and body-cams are the same
    moving-camera problem as a handheld phone; ``percept.profiles_default.HANDHELD`` registers exactly one
    profile for all of them), ego-compensation on. Anything unrecognized -> :data:`WILDCARD`
    (so the registry's own wildcard-fallback chain decides the profile) with ego-compensation defaulted ON
    -- see module docstring for why "guess moving" is the safer wrong guess than "guess fixed".
    """
    normalized = _normalize(camera_mount)
    if normalized == "fixed":
        return "fixed_camera", False
    if normalized in ("wearable", "glasses", "handheld"):
        return "handheld", True
    return WILDCARD, True


def _resolve_goal_type(goal_temporality: Any) -> tuple[str, str]:
    """Goal temporality -> ``(goal_type key segment, plan_mode)``.

    ``"stative"`` -> single-frame plan (``"single"``). ``"dynamic"`` -> ordered-window plan (``"window"``).
    Anything unrecognized -> :data:`WILDCARD` key segment, plan_mode falls back to ``"window"`` (the safest
    known default; see module docstring). ``"sprite"`` is never returned, for any input.
    """
    normalized = _normalize(goal_temporality)
    if normalized == "stative":
        return "stative", "single"
    if normalized == "dynamic":
        return "dynamic", "window"
    return WILDCARD, _WILDCARD_PLAN_MODE


def _resolve_state(start_state: Any) -> tuple[str, bool]:
    """Start-state -> ``(state key segment, gate.require_false_first)``.

    ``"start_anchored"`` (condition may already be true at t0) -> ``require_false_first=False``, so the
    gate fires on first read instead of demanding a false->true edge that may never come (the
    gate-relaxation finding). ``"transition"`` -> ``require_false_first=True`` (only a genuine edge counts).
    Anything unrecognized -> :data:`WILDCARD` key segment, and ``require_false_first`` defaults to ``True``
    -- the conservative choice: assuming "already true" when that is not actually known risks an
    ungrounded, immediate fire.
    """
    normalized = _normalize(start_state)
    if normalized in ("start_anchored", "start-anchored"):
        return "start_anchored", False
    if normalized in ("transition", "transition-only", "transition_only"):
        return "transition", True
    return WILDCARD, True


@dataclass(frozen=True)
class SituationResolution:
    """The result of resolving a situation triple: a registry key plus the overrides this situation
    implies, independent of whatever profile the registry happens to have registered for that key.

    - ``key`` — ``(environment, goal_type, state)``, feedable directly to
      :meth:`percept.watchprofile.WatchProfileRegistry.get` or ``.resolve``.
    - ``overrides`` — a plain, JSON-serializable dict:
        - ``"plan_mode"``: ``"single"`` or ``"window"`` (never ``"sprite"``).
        - ``"gate"``: ``{"require_false_first": bool}``.
        - ``"ego_compensate"``: bool. Not yet a typed :class:`~percept.watchprofile.WatchProfile` field
          (see ``percept.profiles_default.EGO_COMPENSATION_NOTE``), so it rides here as a plain override
          rather than being merged into a returned ``WatchProfile`` by :func:`resolve_profile`.

      ``"salience_stat"``/``"stat_thresholds"`` are deliberately NOT situation-axis overrides
      (ADR-008 ownership transfer, card C2): which statistic to use is owned by the profile/source
      layer, not by goal-temporality/camera-mount/start-state. :func:`resolve_profile` passes
      whatever the resolved registry profile (or resolved :class:`~percept.sourceprofile.SourceProfile`)
      already declares straight through, unmodified.
    """

    key: tuple[str, str, str]
    overrides: dict[str, Any]


def resolve_situation(goal_temporality: Any, camera_mount: Any, start_state: Any) -> SituationResolution:
    """Pure mapping: ``(goal temporality, camera mount, start-state)`` -> registry key + overrides.

    No I/O, no environment reads, no registry access -- calling this twice with the same three arguments
    always returns an equal :class:`SituationResolution`. Any axis value outside its recognized vocabulary
    (:data:`GOAL_TEMPORALITIES`, :data:`CAMERA_MOUNTS`, :data:`START_STATES` -- including ``None``, ``""``,
    and typos) falls back to that axis's safest default independently; the other two axes still resolve
    normally. See the per-axis resolvers (:func:`_resolve_environment`, :func:`_resolve_goal_type`,
    :func:`_resolve_state`) and the module docstring for what "safest" means for each axis.
    """
    environment, ego_compensate = _resolve_environment(camera_mount)
    goal_type, plan_mode = _resolve_goal_type(goal_temporality)
    state, require_false_first = _resolve_state(start_state)

    assert plan_mode != "sprite"  # defensive: no branch above may ever select it (see module docstring)

    return SituationResolution(
        key=(environment, goal_type, state),
        overrides={
            "plan_mode": plan_mode,
            "gate": {"require_false_first": require_false_first},
            "ego_compensate": ego_compensate,
        },
    )


def resolve_profile(
    goal_temporality: Any,
    camera_mount: Any,
    start_state: Any,
    *,
    registry: WatchProfileRegistry = DEFAULT_REGISTRY,
) -> tuple[WatchProfile, dict[str, Any]]:
    """Convenience wrapper: resolve the situation, look up the best-match profile in ``registry`` for the
    resulting key (exact match, then :data:`~percept.watchprofile.WILDCARD` fallback, per
    :meth:`~percept.watchprofile.WatchProfileRegistry.resolve`), and layer the situation's overrides
    (``plan_mode``, ``gate.require_false_first``) on top of it via :func:`dataclasses.replace`.
    ``salience_stat``/``stat_thresholds`` are NOT overridden here (ADR-008 ownership transfer, card
    C2) — the resolved profile's own declared value passes through untouched, whatever it is. Returns
    ``(profile, overrides)`` -- ``overrides`` is the same dict :func:`resolve_situation` produced,
    included because ``"ego_compensate"`` has no home on :class:`~percept.watchprofile.WatchProfile`
    yet (see :class:`SituationResolution`) and callers need it.

    ``registry`` defaults to :data:`percept.watchprofile.DEFAULT_REGISTRY`; this module does not import
    :mod:`percept.profiles_default` and so does not implicitly seed it. Raises
    :class:`~percept.watchprofile.ProfileNotFoundError` if ``registry`` has no profile registered for the
    resolved key, not even a wildcard fallback -- callers must seed the registry first (e.g. via
    ``percept.profiles_default.seed()``) or pass one that is already seeded.

    Purity note: this still does no I/O and reads no environment variables. Reading an in-memory
    ``WatchProfileRegistry`` supplied by the caller is not I/O; the same ``(situation, registry)`` pair
    always resolves to the same result for as long as ``registry``'s contents are unchanged.
    """
    situation = resolve_situation(goal_temporality, camera_mount, start_state)
    base = registry.resolve(*situation.key)

    overrides = situation.overrides
    merged_gate = replace(base.gate, require_false_first=overrides["gate"]["require_false_first"])
    merged = replace(
        base,
        plan_mode=overrides["plan_mode"],
        gate=merged_gate,
    )

    assert merged.plan_mode != "sprite"  # never returned, regardless of what `base` was

    return merged, overrides


__all__ = [
    "GOAL_TEMPORALITIES",
    "CAMERA_MOUNTS",
    "START_STATES",
    "SituationResolution",
    "resolve_situation",
    "resolve_profile",
]
