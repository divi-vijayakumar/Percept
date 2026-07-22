"""WatchProfile — ``Q(p | environment, goal, state)`` made concrete.

Precision and recall are not one fixed number; they are a function of the deployment context. Empirical
tuning found the free variables: the tier-0 eye's localization (``salience_stat`` / ``stat_thresholds``), how the judge sees
time (``plan_mode``), how densely the fovea samples (``cadence``), how the gate treats an already-true
state (``gate.require_false_first``) and repeat firings (``gate.refractory_s``), and how many goals a scene
can carry before cross-talk (``goal_density_guidance``).

A :class:`WatchProfile` is a named, typed binding of those knobs to an ``(environment, goal_type, state)``
context — the concrete shape of ``Q(p | environment, goal, state)``. It invents NO new knobs: every field
is a typed handle onto a setting that already exists and is read elsewhere in this package:

- ``salience_stat`` / ``stat_thresholds`` <-> :class:`percept.config.EdgeSettings` (``salience_stat``,
  ``motion_gate``/``motion_gate_block_max``/``motion_gate_p95``) and
  :func:`percept.harness.tier0.compile_watchspec`'s ``salience_stat=`` / ``stat_thresholds=`` kwargs.
- ``plan_mode`` <-> :data:`percept.behavior.Temporal` (``"single" | "strip" | "window" | "sprite"`` — the
  ``PerceptionSpec.temporal`` vocabulary; docs/how-to/authoring-goals-and-watch-plans.md Part 2.1).
- ``cadence`` <-> :class:`percept.sampling.SensePolicy` (``base_fps``, ``max_fps``, ``escalate_threshold``,
  ``hold_s`` — the fovea's per-concern processing-density envelope).
- ``gate.require_false_first`` <-> :data:`percept.config.GateSettings.require_false_first`.
- ``gate.refractory_s`` <-> the re-arm separation :func:`percept.individuation.associate` / the edge
  detectors' ``refractory_s`` read — a logical "gate" knob (not a literal ``GateSettings`` field) because
  the gate-relaxation finding tunes require-false-first and refractory together.
- ``goal_density_guidance`` <-> the authoring-time "keep the active goal set small and mutually distinct"
  guidance (docs/how-to/authoring-goals-and-watch-plans.md Part 1.1); advisory, nothing enforces it.

**This module is schema + registry ONLY.** No line in ``agent.py`` / ``runtime/worker.py`` reads a
WatchProfile yet — that wiring is separate, per the architectural rule: "the runtime reads its
behavior from a profile; the eval harness writes profiles. No knob is hardcoded in firing logic." Seeded,
empirically-derived default profiles (``fixed-cam-localized``, ``dynamic-action``, ``handheld``) ship in a follow-up
module; this module only makes the shape expressive enough to hold them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator

from .behavior import _VALID_TEMPORAL  # {"single", "strip", "window", "sprite"} — the plan_mode vocabulary
from .contracts import SalienceStat  # Literal["mean", "block_max", "p95"]

# ``plan_mode`` / ``salience_stat`` vocabularies, exposed as plain tuples for validation + introspection
# (error messages, tests) without re-exporting behavior's private Literal alias.
PLAN_MODES: tuple[str, ...] = tuple(sorted(_VALID_TEMPORAL))
SALIENCE_STATS: tuple[str, ...] = ("mean", "block_max", "p95")

WILDCARD = "*"  # a registry key segment that matches anything — see WatchProfileRegistry.resolve


class WatchProfileError(ValueError):
    """A WatchProfile (or a registry operation on one) failed validation."""


class ProfileNotFoundError(KeyError):
    """No WatchProfile matches a registry lookup (exact key, and no wildcard fallback matched either)."""


@dataclass(frozen=True)
class CadenceProfile:
    """The fovea's processing-density envelope — the profile's handle onto
    :class:`percept.sampling.SensePolicy`. ``base_fps`` is the quiet floor, ``max_fps`` the cost ceiling,
    ``escalate_threshold`` the cheap-motion level that triggers a saccade toward ``max_fps``; ``hold_s``
    keeps the escalation alive briefly after a spike — SensePolicy's fourth knob, included so a profile
    round-trips to a real ``SensePolicy`` (:meth:`to_sense_policy`) without guessing a default.
    """

    base_fps: float = 4.0
    max_fps: float = 15.0
    escalate_threshold: float = 0.15
    hold_s: float = 0.5

    def __post_init__(self) -> None:
        if self.base_fps <= 0 or self.max_fps <= 0:
            raise WatchProfileError(
                f"cadence fps must be > 0 (base_fps={self.base_fps}, max_fps={self.max_fps})"
            )
        if self.base_fps > self.max_fps:
            raise WatchProfileError(
                f"cadence.base_fps ({self.base_fps}) must be <= cadence.max_fps ({self.max_fps})"
            )
        if not (0.0 <= self.escalate_threshold <= 1.0):
            raise WatchProfileError(
                f"cadence.escalate_threshold must be in 0..1, got {self.escalate_threshold}"
            )
        if self.hold_s < 0:
            raise WatchProfileError(f"cadence.hold_s must be >= 0, got {self.hold_s}")

    def to_sense_policy(self):
        """Build the real :class:`percept.sampling.SensePolicy` this cadence describes."""
        from .sampling import SensePolicy

        return SensePolicy(
            base_fps=self.base_fps,
            max_fps=self.max_fps,
            escalate_threshold=self.escalate_threshold,
            hold_s=self.hold_s,
        )


@dataclass(frozen=True)
class GateProfile:
    """The gate's handle: whether an already-true-at-start condition should fire on first read
    (:data:`percept.config.GateSettings.require_false_first` — start-anchored persistent states relax this
    to ``False``; transition-only goals set it ``True``), and the minimum separation between two distinct
    firings of the same kind (``refractory_s``, the re-arm interval ``percept.individuation.associate`` /
    the edge rep-counters read)."""

    require_false_first: bool = False
    refractory_s: float = 0.5

    def __post_init__(self) -> None:
        if self.refractory_s < 0:
            raise WatchProfileError(f"gate.refractory_s must be >= 0, got {self.refractory_s}")


@dataclass(frozen=True)
class GoalDensityGuidance:
    """Advisory, NOT enforced by any runtime code: authoring guidance for how many concurrent goals a
    scene tolerates before cross-talk (docs/how-to/authoring-goals-and-watch-plans.md Part 1.1 — "keep the
    active goal set small and mutually distinct"). The empirically-tuned operating point held this at <=3 distinct goals.
    This rides on the profile so a derived profile can carry the finding that produced it; nothing reads
    it automatically (that's a separate authoring-time surface)."""

    max_recommended_goals: int | None = 3
    note: str = ""

    def __post_init__(self) -> None:
        if self.max_recommended_goals is not None and self.max_recommended_goals < 1:
            raise WatchProfileError(
                "goal_density_guidance.max_recommended_goals must be >= 1 or None, "
                f"got {self.max_recommended_goals}"
            )


@dataclass(frozen=True)
class WatchProfile:
    """A named binding of the runtime's tuning knobs to an ``(environment, goal_type, state)`` context —
    the concrete shape of ``Q(p | environment, goal, state)``. See the module docstring for
    which real setting each field is a typed handle onto.

    - ``environment`` — camera mount / scene motion, e.g. ``"fixed_camera"``, ``"handheld"``, ``"wearable"``.
    - ``goal_type`` — the goal's temporal character, e.g. ``"stative"``, ``"dynamic"``, ``"periodic"``.
    - ``state`` — already-true-at-start vs. transition-only, e.g. ``"start_anchored"``, ``"transition"``.

    These three are free-text (not a closed enum) deliberately: empirical tuning seeds the first vocabulary; a new
    environment/goal-type is added by deriving a profile (the continuous-learning bridge), not by
    editing an enum here.
    """

    name: str
    environment: str
    goal_type: str
    state: str
    salience_stat: SalienceStat = "mean"
    stat_thresholds: dict[str, float] = field(default_factory=dict)
    plan_mode: str = "single"
    cadence: CadenceProfile = field(default_factory=CadenceProfile)
    gate: GateProfile = field(default_factory=GateProfile)
    goal_density_guidance: GoalDensityGuidance = field(default_factory=GoalDensityGuidance)
    notes: str = ""

    def __post_init__(self) -> None:
        for label, value in (
            ("name", self.name),
            ("environment", self.environment),
            ("goal_type", self.goal_type),
            ("state", self.state),
        ):
            if not isinstance(value, str) or not value.strip():
                raise WatchProfileError(f"WatchProfile.{label} must be a non-empty string, got {value!r}")
        if self.salience_stat not in SALIENCE_STATS:
            raise WatchProfileError(
                f"WatchProfile.salience_stat must be one of {SALIENCE_STATS}, got {self.salience_stat!r}"
            )
        if self.plan_mode not in PLAN_MODES:
            raise WatchProfileError(
                f"WatchProfile.plan_mode must be one of {PLAN_MODES}, got {self.plan_mode!r}"
            )
        for stat, thr in self.stat_thresholds.items():
            if stat not in SALIENCE_STATS:
                raise WatchProfileError(
                    f"stat_thresholds key {stat!r} is not a known salience_stat, expected one of "
                    f"{SALIENCE_STATS}"
                )
            try:
                thr_f = float(thr)
            except (TypeError, ValueError) as exc:
                raise WatchProfileError(f"stat_thresholds[{stat!r}] must be a number, got {thr!r}") from exc
            if not (0.0 <= thr_f <= 1.0):
                raise WatchProfileError(f"stat_thresholds[{stat!r}] must be in 0..1, got {thr_f}")
        # CadenceProfile / GateProfile / GoalDensityGuidance validate themselves in their own
        # __post_init__ at construction time — nothing further to check about them here.

    @property
    def key(self) -> tuple[str, str, str]:
        """The registry key this profile registers under: ``(environment, goal_type, state)``."""
        return (self.environment, self.goal_type, self.state)


@dataclass
class WatchProfileRegistry:
    """A lookup table of :class:`WatchProfile`, keyed by ``(environment, goal_type, state)``.

    Registration always keys off the profile's OWN fields (:attr:`WatchProfile.key`) — you cannot register
    a profile under a key that disagrees with its ``environment``/``goal_type``/``state``, so a lookup can
    never return a profile whose fields don't match what was asked.

    :meth:`get` is an exact-key lookup (``None`` if absent). :meth:`resolve` additionally falls back
    through :data:`WILDCARD` (``"*"``) keys — ``(env, goal_type, "*")`` -> ``(env, "*", "*")`` ->
    ``("*", "*", "*")`` — so a handful of broad defaults can cover many contexts without one profile per
    exact combination. This is deliberately just a fallback CHAIN, not goal-temporality inference (mapping
    e.g. "dynamic" -> ``plan_mode="window"`` automatically is a separate resolution-rules layer, which sits
    on top of this registry).
    """

    _profiles: dict[tuple[str, str, str], WatchProfile] = field(default_factory=dict)

    def register(self, profile: WatchProfile, *, overwrite: bool = False) -> None:
        """Register ``profile`` under its own ``(environment, goal_type, state)`` key.

        Raises :class:`WatchProfileError` if a DIFFERENT profile already occupies that key, unless
        ``overwrite=True``. Re-registering the identical profile object is always a no-op.
        """
        key = profile.key
        existing = self._profiles.get(key)
        if existing is not None and existing is not profile and not overwrite:
            raise WatchProfileError(
                f"a profile is already registered for {key}: {existing.name!r}; "
                f"pass overwrite=True to replace it with {profile.name!r}"
            )
        self._profiles[key] = profile

    def unregister(self, environment: str, goal_type: str, state: str) -> None:
        """Drop whatever profile is registered for this exact key, if any (no-op if absent)."""
        self._profiles.pop((environment, goal_type, state), None)

    def get(self, environment: str, goal_type: str, state: str) -> WatchProfile | None:
        """Exact-key lookup only. ``None`` if nothing is registered for this exact context."""
        return self._profiles.get((environment, goal_type, state))

    def resolve(self, environment: str, goal_type: str, state: str) -> WatchProfile:
        """Exact lookup, then :data:`WILDCARD` fallback; raises :class:`ProfileNotFoundError` if nothing
        registered matches even the broadest (``"*", "*", "*"``) fallback."""
        for key in (
            (environment, goal_type, state),
            (environment, goal_type, WILDCARD),
            (environment, WILDCARD, WILDCARD),
            (WILDCARD, WILDCARD, WILDCARD),
        ):
            profile = self._profiles.get(key)
            if profile is not None:
                return profile
        raise ProfileNotFoundError(
            f"no WatchProfile registered for ({environment!r}, {goal_type!r}, {state!r}) and no "
            f"wildcard fallback matched; registered keys: {sorted(self._profiles)}"
        )

    def all(self) -> list[WatchProfile]:
        """Every registered profile, in registration order (Python dict order)."""
        return list(self._profiles.values())

    def __contains__(self, key: tuple[str, str, str]) -> bool:
        return key in self._profiles

    def __len__(self) -> int:
        return len(self._profiles)

    def __iter__(self) -> Iterator[WatchProfile]:
        return iter(self._profiles.values())


# A process-wide default registry: empirically-derived defaults seed this instance, and the runtime wiring
# reads from it by default. Nothing in this codebase writes to it yet. Tests and callers
# that want isolation should construct their own ``WatchProfileRegistry()`` instead of touching this one.
DEFAULT_REGISTRY = WatchProfileRegistry()


__all__ = [
    "WatchProfileError",
    "ProfileNotFoundError",
    "CadenceProfile",
    "GateProfile",
    "GoalDensityGuidance",
    "WatchProfile",
    "WatchProfileRegistry",
    "DEFAULT_REGISTRY",
    "PLAN_MODES",
    "SALIENCE_STATS",
    "WILDCARD",
]
