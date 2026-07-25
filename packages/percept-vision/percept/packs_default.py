"""The shipped :class:`~percept.domain_pack.DomainPack` set — ``facility-observation@1``.

One pack, four templates, built ONLY from senses that already exist in
:func:`percept.behavior.program_of` — no new temporal shape, no new sense, nothing invented:

- ``door_left_open`` — an ordinary scene Watch with sustained truth and an explicit fixed
  observation cadence.
- ``package_appeared`` — the **transition** sense (``type="transition"``, ``from``/``to``): a state
  change from "nothing" to "present".
- ``standing_water`` — also the **transition** sense: the same from/absent-to-present shape as a
  package, a different domain.
- ``zone_occupied`` — an ordinary Watch: one rising-edge incident per contiguous occupied interval.
  It does not identify, track, or count distinct people.

Every :class:`~percept.domain_pack.ComponentRecommendation` here is ``strength="preferred"`` —
this pack recommends, it never requires (ADR-008 Invariants: "templates never auto-arm"; no
identity/tracking calibration exists yet for a domain that would need a harder floor — a
``security@1`` pack is explicitly future work, not shipped here).

Importing this module seeds :data:`~percept.domain_pack.DEFAULT_DOMAIN_PACK_REGISTRY` (the same
import-for-its-side-effect contract :mod:`percept.execution_policies_default` /
:mod:`percept.profiles_default` / :mod:`percept.source_profiles_default` already use) —
:func:`~percept.domain_pack.get_domain_pack` does this for you. The pack is ALSO advertised under
the ``percept.packs`` entry-point group (see ``pyproject.toml``) via :func:`register_entrypoint_pack`
below, so a fresh install exercises the SAME pip-installed-pack discovery path
(:func:`~percept.domain_pack.discover_domain_pack_entrypoints`) a third-party pack would use — not
just the in-code default.
"""

from __future__ import annotations

from .direct_judge import DirectRouting
from .domain_pack import ComponentRecommendation, DomainPack, GoalTemplate, register_domain_pack
from .identity_policy import SCENE_IDENTITY

# ---------------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------------

#: Sustained scene Watch: the compatibility parameter ``deadline_after_s`` names the required-open
#: duration, while the executable Goal receives only hold + fixed-cadence state.
_DOOR_LEFT_OPEN = GoalTemplate(
    name="door_left_open",
    condition="the {door_name} is open",
    parameters={
        "door_name": "door",
        "deadline_after_s": 300.0,
        "observation_interval_s": 15.0,
    },
    target="door",
    identity_policy=SCENE_IDENTITY,
    hold_s="{deadline_after_s}",
    cadence={
        "min_judge_interval": "{observation_interval_s}",
        "gate_on": "fixed",
    },
    consequence={"miss_cost": "mild", "false_fire_cost": "trivial"},
    notes=(
        "fires once after the door has been continuously observed open for deadline_after_s "
        "(default 300s / 5min); a shorter open interval does not fire."
    ),
)

#: Transition sense: no package -> package present.
_PACKAGE_APPEARED = GoalTemplate(
    name="package_appeared",
    condition="a package has appeared at the {zone_name}",
    parameters={
        "zone_name": "doorstep",
        "from_state": "no package is present",
        "to_state": "a package is visible",
    },
    target="package",
    behavior={"type": "transition", "from": "{from_state}", "to": "{to_state}"},
    consequence={"miss_cost": "mild", "false_fire_cost": "trivial"},
    notes="fires once, on the rising edge of a package becoming visible at the named zone.",
)

#: Transition sense: floor dry -> standing water visible. Higher default miss_cost than a package
#: (a safety-relevant spill sits still, undetected, longer than most domains would tolerate).
_STANDING_WATER = GoalTemplate(
    name="standing_water",
    condition="standing water has appeared in the {zone_name}",
    parameters={
        "zone_name": "storage area",
        "from_state": "the floor is dry",
        "to_state": "standing water is visible on the floor",
    },
    target="floor",
    behavior={"type": "transition", "from": "{from_state}", "to": "{to_state}"},
    consequence={"miss_cost": "severe", "false_fire_cost": "mild"},
    notes="fires once, on the rising edge of standing water becoming visible in the named zone.",
)

#: Ordinary Watch: each confidently observed empty interval re-arms the next occupied rising edge.
_ZONE_OCCUPIED = GoalTemplate(
    name="zone_occupied",
    condition="a person is present in the {zone_name}",
    parameters={"zone_name": "restricted zone"},
    target="person",
    identity_policy=SCENE_IDENTITY,
    consequence={"miss_cost": "mild", "false_fire_cost": "trivial"},
    notes=(
        "fires once per contiguous occupied interval of the named zone. It does not identify, "
        "track, or count distinct people in a crowd."
    ),
)

_TEMPLATES = {
    t.name: t for t in (_DOOR_LEFT_OPEN, _PACKAGE_APPEARED, _STANDING_WATER, _ZONE_OCCUPIED)
}

# ---------------------------------------------------------------------------
# Component recommendations — all "preferred": this pack recommends, it never requires.
# ---------------------------------------------------------------------------

_COMPONENTS = (
    ComponentRecommendation(
        ref="sense:physical_scene",
        strength="preferred",
        reason="a facility camera is a fixed physical scene, not a screen or a conversation.",
    ),
    ComponentRecommendation(
        ref="execution_policy:low_cost@1",
        strength="preferred",
        reason=(
            "facility observation is typically a sparse-event, fixed-camera deployment: a "
            "quieter cadence + the cache cascade save real spend without missing the rare event."
        ),
    ),
)

# ---------------------------------------------------------------------------
# DirectJudge routes — a RESERVED namespace, no rule implementation ships here (see module
# docstring and percept.direct_judge). None of the four templates' rendered conditions are
# authored under this namespace prefix today, so DirectJudge falls through to the normal judge
# path for all of them until a real rule is registered under it — an honest, forward-declared seam,
# not a claim of present cost savings.
# ---------------------------------------------------------------------------

_DIRECT_JUDGE_ROUTES = (DirectRouting(condition_namespace="facility_observation."),)

# ---------------------------------------------------------------------------
# Fixtures + quality bars — plain data declarations. A template appears here only when the pack has
# a matching evaluation route for the behavior it executes. ``door_left_open`` deliberately does
# not: the sustained-duration runtime is executable, but a still-open interval cannot be scored by
# the opening-action MEVA labels and no matched duration corpus ships in this release. Consequently
# the SDK makes no accuracy or efficiency claim for that template.
# ---------------------------------------------------------------------------

_FIXTURES = (
    "facility_observation/package_appeared",
    "facility_observation/standing_water",
    "facility_observation/zone_occupied",
)

_QUALITY_BARS = {
    "package_appeared.recall_min": 0.60,
    "standing_water.recall_min": 0.55,
    "zone_occupied.recall_min": 0.60,
}

# ---------------------------------------------------------------------------
# SearchSpace — data only (percept.tunables.REGISTRY safe_to_optimize=True paths this domain cares
# about); an optimizer card consumes this later. Nothing here executes a search.
# ---------------------------------------------------------------------------

_SEARCH_SPACE = {
    "Settings.edge.motion_gate": {"min": 0.02, "max": 0.5},
    "WatchProfile.cadence.base_fps": {"min": 1.0, "max": 8.0},
    "WatchProfile.cadence.escalate_threshold": {"min": 0.05, "max": 0.4},
}

FACILITY_OBSERVATION_V1 = DomainPack(
    id="facility_observation",
    version="1",
    templates=_TEMPLATES,
    components=_COMPONENTS,
    policy_clamps={"false_negative_cost_min": "mild", "retention_max_days": 30.0},
    direct_judge_routes=_DIRECT_JUDGE_ROUTES,
    fixtures=_FIXTURES,
    quality_bars=_QUALITY_BARS,
    search_space=_SEARCH_SPACE,
    notes=(
        "Physical-scene Watch templates for a door left open, package appearance, standing water, "
        "and zone occupancy. door_left_open is executable but not quality-qualified: it has no "
        "fixture or quality bar until a matched sustained-duration evaluation ships. No security@1 "
        "pack ships because the SDK does not provide identity/tracking calibration."
    ),
)

register_domain_pack(FACILITY_OBSERVATION_V1)


def register_entrypoint_pack(register: "callable") -> None:
    """The ``percept.packs`` entry-point TARGET this package's own ``pyproject.toml`` advertises
    (mirrors ``percept.sourceprofile``'s ``discover_source_profile_entrypoints`` contract: an entry
    point's value is a callable taking the register function). Lets a fresh install exercise the
    SAME pip-installed-pack discovery path
    (:func:`percept.domain_pack.discover_domain_pack_entrypoints`) this pack's own in-code default
    already uses, so the two registration paths are provably consistent rather than one being
    untested."""
    register(FACILITY_OBSERVATION_V1, overwrite=True)


__all__ = ["FACILITY_OBSERVATION_V1", "register_entrypoint_pack"]
