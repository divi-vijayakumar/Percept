"""Official starter packs for programmable Watches over live video streams.

The SDK ships two official, mechanics-only starter packs:

- ``physical_scene_starter@1`` — four fixed-scene templates;
- ``ego_motion_starter@1`` — four moving-camera templates.

``facility_observation@1`` remains registered as the original compatibility surface for the
physical templates. It is not removed or silently redirected.

Every template is built only from behavior routes that :func:`percept.behavior.program_of` and the
runtime worker execute today — ordinary rising-edge Watches and two-state transitions:

- ``door_left_open`` — an ordinary scene Watch with sustained truth and an explicit fixed
  observation cadence.
- ``package_appeared`` — the **transition** sense (``type="transition"``, ``from``/``to``): a state
  change from "nothing" to "present".
- ``standing_water`` — also the **transition** sense: the same from/absent-to-present shape as a
  package, a different domain.
- ``zone_occupied`` — an ordinary Watch: one rising-edge incident per contiguous occupied interval.
  It does not identify, track, or count distinct people.
- ``camera_view_moving`` — an ordinary scene-scoped Watch over a moving-camera observation.
- ``scene_transition`` — a caller-authored visible scene A -> visible scene B transition.
- ``object_focus_gained`` — a category-level out-of-focus -> in-focus transition. It does not
  establish that a previously seen object was reacquired.
- ``task_step_reached`` — a caller-authored visible prior-state -> visible next-state transition.
  It does not understand or infer a task plan.

Every :class:`~percept.domain_pack.ComponentRecommendation` is ``strength="preferred"`` — these
packs recommend, they never require (ADR-008 Invariants: "templates never auto-arm"; no
identity/tracking calibration exists yet for a domain that would need a harder floor — a
``security@1`` pack is explicitly future work, not shipped here).

Importing this module seeds :data:`~percept.domain_pack.DEFAULT_DOMAIN_PACK_REGISTRY` (the same
import-for-its-side-effect contract :mod:`percept.execution_policies_default` /
:mod:`percept.profiles_default` / :mod:`percept.source_profiles_default` already use) —
:func:`~percept.domain_pack.get_domain_pack` does this for you. All three names are also advertised
under the ``percept.packs`` entry-point group (see ``pyproject.toml``), so a fresh install exercises
the same pip-installed-pack discovery path
(:func:`~percept.domain_pack.discover_domain_pack_entrypoints`) a third-party pack would use.

Availability is not qualification. Both official packs deliberately declare ``fixtures=()`` and
``quality_bars={}``: deterministic tests prove compilation, ambiguity handling, rising-edge state,
and replay mechanics, but no permission-safe held-out corpus currently supports an accuracy,
latency, or cost claim.
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
# Ego Motion Starter templates — scene-scoped mechanics only.
# ---------------------------------------------------------------------------

_CAMERA_VIEW_MOVING = GoalTemplate(
    name="camera_view_moving",
    condition="the camera viewpoint is moving because the wearer or device is moving",
    parameters={"observation_interval_s": 1.0},
    sense="vision",
    target="camera viewpoint",
    identity_policy=SCENE_IDENTITY,
    cadence={
        "min_judge_interval": "{observation_interval_s}",
        "gate_on": "fixed",
    },
    consequence={"miss_cost": "mild", "false_fire_cost": "trivial"},
    notes=(
        "fires on the ordinary scene-level rising edge of a judged moving-camera observation. "
        "It is an unqualified template, not a camera-motion accuracy claim."
    ),
)

_SCENE_TRANSITION = GoalTemplate(
    name="scene_transition",
    condition="the camera view transitioned from {from_scene} to {to_scene}",
    parameters={
        "from_scene": "the starting scene",
        "to_scene": "the destination scene",
        "observation_interval_s": 1.0,
    },
    sense="vision",
    target="visible scene",
    identity_policy=SCENE_IDENTITY,
    behavior={
        "type": "transition",
        "from": "{from_scene} is the current visible scene",
        "to": "{to_scene} is the current visible scene",
    },
    cadence={
        "min_judge_interval": "{observation_interval_s}",
        "gate_on": "fixed",
    },
    consequence={"miss_cost": "mild", "false_fire_cost": "trivial"},
    notes=(
        "fires only after the existing transition runtime observes the authored from-scene and then "
        "the authored to-scene. It does not detect a general revisit or establish location identity."
    ),
)

_OBJECT_FOCUS_GAINED = GoalTemplate(
    name="object_focus_gained",
    condition="{object_name} became centered and clearly visible in the camera view",
    parameters={
        "object_name": "the target object category",
        "observation_interval_s": 1.0,
    },
    sense="vision",
    target="{object_name}",
    identity_policy=SCENE_IDENTITY,
    behavior={
        "type": "transition",
        "from": "{object_name} is not centered and clearly visible in the camera view",
        "to": "{object_name} is centered and clearly visible in the camera view",
    },
    cadence={
        "min_judge_interval": "{observation_interval_s}",
        "gate_on": "fixed",
    },
    consequence={"miss_cost": "mild", "false_fire_cost": "trivial"},
    notes=(
        "category-level visible focus only. Scene scoping is deliberate: this template does not "
        "identify, track, or claim same-object reacquisition after a gap."
    ),
)

_TASK_STEP_REACHED = GoalTemplate(
    name="task_step_reached",
    condition="the visible work changed from {previous_state} to {completed_state}",
    parameters={
        "previous_state": "the visible prior step state",
        "completed_state": "the visible completed next-step state",
        "observation_interval_s": 1.0,
    },
    sense="vision",
    target="visible task state",
    identity_policy=SCENE_IDENTITY,
    behavior={
        "type": "transition",
        "from": "{previous_state}",
        "to": "{completed_state}",
    },
    cadence={
        "min_judge_interval": "{observation_interval_s}",
        "gate_on": "fixed",
    },
    consequence={"miss_cost": "mild", "false_fire_cost": "trivial"},
    notes=(
        "fires on one explicitly authored visible state transition. It does not infer a task plan, "
        "understand intent, or advance later steps automatically."
    ),
)

_EGO_TEMPLATES = {
    t.name: t
    for t in (
        _CAMERA_VIEW_MOVING,
        _SCENE_TRANSITION,
        _OBJECT_FOCUS_GAINED,
        _TASK_STEP_REACHED,
    )
}

# ---------------------------------------------------------------------------
# Component recommendations — all "preferred": this pack recommends, it never requires.
# ---------------------------------------------------------------------------

_LEGACY_COMPONENTS = (
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

_PHYSICAL_COMPONENTS = (
    ComponentRecommendation(
        ref="sense:physical_scene",
        strength="preferred",
        reason="a facility camera is a fixed physical scene, not a screen or a conversation.",
    ),
    ComponentRecommendation(
        ref="execution_policy:low_cost@1",
        strength="preferred",
        reason=(
            "low_cost@1 is an optional sparse-event starting posture; applications must validate "
            "its cadence and admission behavior on their own labeled deployment media."
        ),
    ),
)

_EGO_COMPONENTS = (
    ComponentRecommendation(
        ref="sense:physical_scene",
        strength="preferred",
        reason="the current video SourceProfile supports fixed, handheld, and wearable cameras.",
    ),
    ComponentRecommendation(
        ref="profile:handheld",
        strength="preferred",
        reason=(
            "the existing handheld WatchProfile carries the moving-camera admission posture; "
            "applications still select and validate it explicitly for their source."
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
# Fixtures + quality bars. The legacy facility pack preserves its original candidate declarations
# for data-contract compatibility; they are not achieved qualification results. The two official
# starters are deliberately empty. Mechanics fixtures live in tests and must never be presented as
# accuracy evidence.
# ---------------------------------------------------------------------------

_LEGACY_FIXTURES = (
    "facility_observation/package_appeared",
    "facility_observation/standing_water",
    "facility_observation/zone_occupied",
)

_LEGACY_QUALITY_BARS = {
    "package_appeared.recall_min": 0.60,
    "standing_water.recall_min": 0.55,
    "zone_occupied.recall_min": 0.60,
}

_UNQUALIFIED_FIXTURES: tuple[str, ...] = ()
_UNQUALIFIED_QUALITY_BARS: dict[str, float] = {}

# ---------------------------------------------------------------------------
# SearchSpace — data only (percept.tunables.REGISTRY safe_to_optimize=True paths this domain cares
# about); an optimizer consumes this later. Nothing here executes a search.
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
    components=_LEGACY_COMPONENTS,
    policy_clamps={"false_negative_cost_min": "mild", "retention_max_days": 30.0},
    direct_judge_routes=_DIRECT_JUDGE_ROUTES,
    fixtures=_LEGACY_FIXTURES,
    quality_bars=_LEGACY_QUALITY_BARS,
    search_space=_SEARCH_SPACE,
    notes=(
        "Compatibility pack for physical-scene Watch templates. Unmatched templates are not quality-qualified. Its retained fixtures and "
        "quality_bars are candidate qualification declarations, not achieved results; no template "
        "is qualified until permission-safe, independently reviewed, source-group-held-out "
        "Watch-semantic evaluation exists. No security@1 pack ships because the SDK does not "
        "provide identity/tracking calibration."
    ),
)

PHYSICAL_SCENE_STARTER_V1 = DomainPack(
    id="physical_scene_starter",
    version="1",
    templates=_TEMPLATES,
    components=_PHYSICAL_COMPONENTS,
    policy_clamps={"false_negative_cost_min": "mild"},
    fixtures=_UNQUALIFIED_FIXTURES,
    quality_bars=_UNQUALIFIED_QUALITY_BARS,
    search_space=_SEARCH_SPACE,
    notes=(
        "Official free Physical Scene Starter: door-left-open, package-appearance, standing-water, "
        "and scene-level zone-occupancy Watch mechanics. Availability is not qualification; the "
        "pack declares no fixtures or quality bars and makes no accuracy, latency, or cost claim. "
        "It does not identify, track, or count people or objects."
    ),
)

EGO_MOTION_STARTER_V1 = DomainPack(
    id="ego_motion_starter",
    version="1",
    templates=_EGO_TEMPLATES,
    components=_EGO_COMPONENTS,
    policy_clamps={"false_negative_cost_min": "mild"},
    fixtures=_UNQUALIFIED_FIXTURES,
    quality_bars=_UNQUALIFIED_QUALITY_BARS,
    search_space={},
    notes=(
        "Official free Ego Motion Starter: scene-scoped moving-view, visible scene-transition, "
        "category-level focus-gained, and visible task-step transition mechanics. Availability is "
        "not qualification; the pack declares no fixtures or quality bars and makes no accuracy, "
        "latency, or cost claim. Revisit, same-object reacquisition, identity, tracking, location "
        "authority, and task understanding are not supported claims."
    ),
)

for _pack in (
    FACILITY_OBSERVATION_V1,
    PHYSICAL_SCENE_STARTER_V1,
    EGO_MOTION_STARTER_V1,
):
    register_domain_pack(_pack)


def register_entrypoint_pack(register: "callable") -> None:
    """Register the legacy ``facility_observation@1`` entry-point target."""
    register(FACILITY_OBSERVATION_V1)


def register_physical_scene_entrypoint_pack(register: "callable") -> None:
    """Register the official ``physical_scene_starter@1`` entry-point target."""
    register(PHYSICAL_SCENE_STARTER_V1)


def register_ego_motion_entrypoint_pack(register: "callable") -> None:
    """Register the official ``ego_motion_starter@1`` entry-point target."""
    register(EGO_MOTION_STARTER_V1)


__all__ = [
    "FACILITY_OBSERVATION_V1",
    "PHYSICAL_SCENE_STARTER_V1",
    "EGO_MOTION_STARTER_V1",
    "register_entrypoint_pack",
    "register_physical_scene_entrypoint_pack",
    "register_ego_motion_entrypoint_pack",
]
