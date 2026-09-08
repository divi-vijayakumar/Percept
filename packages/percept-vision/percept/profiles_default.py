"""Seeded, empirically-derived default WatchProfiles (continuous-learning bridge).

``percept.watchprofile`` is schema + registry only — it defines the SHAPE of
``Q(p | environment, goal, state)`` but ships with an empty ``DEFAULT_REGISTRY``. This module is the DATA:
three named profiles read directly off a viability eval (the charades-e2e ablation ladder), plus a
WILDCARD fallback so :meth:`percept.watchprofile.WatchProfileRegistry.resolve` never raises
:class:`percept.watchprofile.ProfileNotFoundError` for a context nobody has named yet.

These are NOT guesses. Every profile's ``notes`` field stamps it to the eval run that produced it
(:data:`H0_PROVENANCE`) — that provenance is what makes a profile auditable config instead of a magic
number: a reader can trace ``block_max @ 0.20`` back to the ablation step that measured it, and a future
change that re-runs the eval and gets a different number knows exactly which profile to update and why.

Importing this module has the side effect of seeding :data:`percept.watchprofile.DEFAULT_REGISTRY` (see
:func:`seed`, called at the bottom of this file) — the same "seeded at import" contract this module implements.
Tests that want an isolated registry should call ``seed(WatchProfileRegistry())`` instead of touching the
process-wide default.
"""

from __future__ import annotations

from .watchprofile import (
    DEFAULT_REGISTRY,
    WILDCARD,
    GateProfile,
    WatchProfile,
    WatchProfileRegistry,
)

# ---------------------------------------------------------------------------
# Provenance — shared by every profile below; all four knob sets are readouts of the SAME eval run.
# ---------------------------------------------------------------------------

#: The charades-e2e ablation ladder:
#: starting from the legacy byte-for-byte defaults (``salience_stat="mean"``, ``plan_mode="single"``,
#: gate closed), each step below is one knob flip, measured independently, on the same eval track.
H0_PROVENANCE = (
    "Provenance: viability eval, charades-e2e ablation ladder (eval/bench/track_charades_e2e.py; "
    "docs/percept-mvp-implementation-plan.md). Recall climbed 0.39 (mean-stat baseline) -> 0.68 "
    "(+ salience_stat=\"block_max\" @ stat_thresholds[\"block_max\"]=0.20 -- localized salience beats a "
    "whole-frame mean) -> 0.73 (+ plan_mode=\"window\" -- the judge sees a short clip, not one frame) -> "
    "0.90 (+ gate-relax: gate.require_false_first=False, gate.refractory_s=15 -- an already-true state "
    "fires on first read instead of waiting for a false->true edge that may never come) at the operating "
    "point. This profile's knobs are a typed readout of a rung on that ladder, not a hand-picked guess."
)

#: Additional provenance specific to a MOVING camera (handheld / wearable / body-cam), from the same
#: eval run's ego-motion arm.
EGO_COMPENSATION_NOTE = (
    "environment=\"handheld\" additionally implies percept.config.EdgeSettings.ego_compensate=True "
    "(PERCEPT_EGO_COMPENSATE=1): without compensation, global camera motion can keep cheap-motion above "
    "the escalation threshold even when the scene has no semantic event. WatchProfile has no typed "
    "ego_compensate field; the situation resolver emits "
    "the SourceProfile-owned Settings.edge.ego_compensate binding from the camera_mount axis. Callers "
    "must pass situation=(goal_temporality, 'handheld', start_state) (or the equivalent deployment "
    "setting) rather than assuming the profile name alone flips the runtime bit. A three-clip EGO "
    "replay preserved event-window reachability but did not prove a universal admission-rate claim."
)


def _h0_gate(**overrides: float | bool) -> GateProfile:
    """A :class:`GateProfile` at the empirically-tuned gate-relax + refractory operating point (``refractory_s=15``),
    with ``require_false_first`` left at the schema default (``False``) unless overridden."""
    kwargs: dict[str, float | bool] = {"refractory_s": 15.0}
    kwargs.update(overrides)
    return GateProfile(**kwargs)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# The three named empirically-derived profiles.
# ---------------------------------------------------------------------------

#: FIXED camera watching a localized / low-motion action (e.g. "is the stove on", "is the door open") whose
#: truth can already hold at the start of a clip. The proven 0.90-recall stack: block_max localization +
#: window-mode judging + gate-relax (fire on an already-true state instead of demanding a false->true edge).
FIXED_CAM_LOCALIZED = WatchProfile(
    name="fixed-cam-localized",
    environment="fixed_camera",
    goal_type="stative",
    state="start_anchored",
    salience_stat="block_max",
    stat_thresholds={"block_max": 0.20},
    plan_mode="window",
    gate=_h0_gate(require_false_first=False),
    notes=(
        "The proven 0.90-recall stack for a FIXED camera watching a localized/low-motion, "
        "already-true-at-start action. " + H0_PROVENANCE
    ),
)

#: Dynamic goals (the action's truth changes mid-clip, not just present-at-start) on a FIXED camera. Same
#: tier-0 eye and window-mode judge as fixed-cam-localized; the eval run did not additionally credit this
#: exact (goal_type, state) combination with the 0.90 figure, so require_false_first rides the schema
#: default rather than restating the gate-relax claim verbatim.
DYNAMIC_ACTION = WatchProfile(
    name="dynamic-action",
    environment="fixed_camera",
    goal_type="dynamic",
    state="transition",
    salience_stat="block_max",
    stat_thresholds={"block_max": 0.20},
    plan_mode="window",
    gate=_h0_gate(),
    notes=(
        "Dynamic goals (truth changes over the clip, e.g. a false->true action transition) on a FIXED "
        "camera. Same block_max localization + window judging as fixed-cam-localized; refractory_s=15 caps "
        "re-fire chatter across an evolving action. " + H0_PROVENANCE
    ),
)

#: Smart glasses / body-cam: a MOVING camera, any goal/state (the key knob here is the camera mount, not
#: the goal shape) -- registered under a WILDCARD goal_type/state so it is the broad default for anything
#: shot handheld.
HANDHELD = WatchProfile(
    name="handheld",
    environment="handheld",
    goal_type=WILDCARD,
    state=WILDCARD,
    salience_stat="block_max",
    stat_thresholds={"block_max": 0.20},
    plan_mode="window",
    gate=_h0_gate(),
    notes=(
        "Smart glasses / body-cam: a MOVING camera. " + EGO_COMPENSATION_NOTE + " " + H0_PROVENANCE
    ),
)

# ---------------------------------------------------------------------------
# WILDCARD fallback -- so resolve() always returns something, even for a context nobody has named yet.
# ---------------------------------------------------------------------------

#: Fully-broad fallback (``environment=goal_type=state=WILDCARD``). Carries the same proven
#: block_max + window + refractory-15 stack as fixed-cam-localized/dynamic-action (the safest known
#: default) but does NOT assume ego-compensation, since an unrecognized environment may not be a moving
#: camera. A resolve() that lands here is a signal to author a more specific profile for that context, not
#: proof the defaults are a good fit.
WILDCARD_DEFAULT = WatchProfile(
    name="wildcard-default",
    environment=WILDCARD,
    goal_type=WILDCARD,
    state=WILDCARD,
    salience_stat="block_max",
    stat_thresholds={"block_max": 0.20},
    plan_mode="window",
    gate=_h0_gate(),
    notes=(
        "Global fallback so WatchProfileRegistry.resolve() never raises ProfileNotFoundError for an "
        "unrecognized (environment, goal_type, state) context. Reuses the proven block_max + window + "
        "refractory-15 stack as the safest known default; does NOT assume ego-compensation (an unknown "
        "environment may not be a moving camera) -- treat a resolve() landing here as a prompt to author a "
        "more specific profile, not as validated for that context. " + H0_PROVENANCE
    ),
)

#: Registration order matters for readability (most-specific first) but not for correctness --
#: WatchProfileRegistry.resolve() always tries exact-key before any wildcard, regardless of insertion order.
ALL_PROFILES: tuple[WatchProfile, ...] = (
    FIXED_CAM_LOCALIZED,
    DYNAMIC_ACTION,
    HANDHELD,
    WILDCARD_DEFAULT,
)


def seed(registry: WatchProfileRegistry = DEFAULT_REGISTRY, *, overwrite: bool = False) -> None:
    """Register every profile in :data:`ALL_PROFILES` onto ``registry`` (:data:`DEFAULT_REGISTRY` by
    default). Re-seeding the SAME registry with these same singleton profile objects is always a no-op
    (:meth:`WatchProfileRegistry.register` treats re-registering an identical object as a no-op)."""
    for profile in ALL_PROFILES:
        registry.register(profile, overwrite=overwrite)


# Seed the process-wide DEFAULT_REGISTRY as soon as this module is imported -- this module's whole job.
seed()


__all__ = [
    "H0_PROVENANCE",
    "EGO_COMPENSATION_NOTE",
    "FIXED_CAM_LOCALIZED",
    "DYNAMIC_ACTION",
    "HANDHELD",
    "WILDCARD_DEFAULT",
    "ALL_PROFILES",
    "seed",
]
