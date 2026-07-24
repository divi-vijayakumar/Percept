"""MEVA activity vocabulary -> ``facility_observation`` pack, plus optional pack scaffolding.

Two things live here, kept separate:

1. **The mapping** (:data:`MEVA_ACTIVITY_TO_CONDITION`) — which shipped ``facility_observation@1``
   template each mapped MEVA activity type resolves to, and the eval *goal shape*
   (:mod:`percept.eval.differential` scope vocabulary) a fixture of that condition carries. This is
   the transform's contract; it targets templates the shipped pack ALREADY defines, so no registry
   edit is required to run the transform.

2. **Optional, self-contained pack scaffolding** (:func:`build_meva_facility_pack`) — a
   ``facility_observation_meva@1`` :class:`~percept.domain_pack.DomainPack` whose templates are
   NAMED with MEVA's own activity vocabulary (``person_opens_facility_door`` etc.) and reuse the
   shipped senses. It is DELIBERATELY NOT registered at import time: registering it is a one-line
   call the human/Fable applies (see ``README.md`` and :func:`register`), so this dev-tooling module
   never mutates the shared :data:`percept.domain_pack.DEFAULT_DOMAIN_PACK_REGISTRY` that sibling
   pack cards also touch.

Mapping rationale (from the DS1 brief):

* ``person_opens_facility_door`` -> ``door_left_open`` (absence sense; fires if the door stays open
  past a deadline). Goal shape ``episode`` — a continuous true-episode with a rising edge.
* ``person_loiters`` / ``person_enters_scene_through_structure`` (person-enters-zone) ->
  ``zone_occupied`` (episode/count sense). Goal shape ``episode``.
* ``person_abandons_package`` / abandoned-object / drop-off -> ``package_appeared`` (transition
  sense; rising edge of an object becoming present). Goal shape ``point`` — one fire, one instant.
"""

from __future__ import annotations

from dataclasses import dataclass

from percept.domain_pack import (
    ComponentRecommendation,
    DomainPack,
    GoalTemplate,
    register_domain_pack,
)

__all__ = [
    "ConditionSpec",
    "MEVA_ACTIVITY_TO_CONDITION",
    "FACILITY_PACK_REF",
    "condition_for",
    "mapped_activity_types",
    "build_meva_facility_pack",
    "register",
]

#: The shipped pack the transform maps onto. ``watch_template(f"{FACILITY_PACK_REF}/door_left_open")``
#: renders a portable WatchPlan for the mapped condition without arming anything.
FACILITY_PACK_REF = "facility_observation@1"


@dataclass(frozen=True)
class ConditionSpec:
    """How one MEVA activity type projects onto ``facility_observation``.

    - ``template`` — the shipped ``facility_observation@1`` template name.
    - ``goal_shape`` — the :mod:`percept.eval.differential` scope shape a fixture of this condition
      declares (``point`` / ``episode`` — both SUPPORTED by the one-to-one event-level matcher).
    - ``pack_ref`` — the fully-qualified ``watch_template`` reference for the mapped condition.
    """

    template: str
    goal_shape: str
    pack_ref: str


def _spec(template: str, goal_shape: str) -> ConditionSpec:
    return ConditionSpec(template=template, goal_shape=goal_shape, pack_ref=f"{FACILITY_PACK_REF}/{template}")


#: MEVA activity type -> ``facility_observation`` condition. Aliases (``abandonment`` etc.) are
#: included so the transform tolerates the schema variants across MEVA annotation revisions. An
#: activity type absent from this map is intentionally DROPPED by the transform (not every MEVA
#: activity is a facility-observation condition) — see :func:`condition_for`.
MEVA_ACTIVITY_TO_CONDITION: dict[str, ConditionSpec] = {
    # snake_case KF1 / DIVA vocabulary (this repo's synthetic sample).
    "person_opens_facility_door": _spec("door_left_open", "episode"),
    "person_loiters": _spec("zone_occupied", "episode"),
    "person_enters_scene_through_structure": _spec("zone_occupied", "episode"),
    "person_enters_through_structure": _spec("zone_occupied", "episode"),
    "person_abandons_package": _spec("package_appeared", "point"),
    "person_drops_off_package": _spec("package_appeared", "point"),
    "abandonment": _spec("package_appeared", "point"),
    # CamelCase MEVA-KF1 vocabulary as it appears in the REAL public drops (act3 records). These are
    # the same three facility conditions under the names the shipped annotations actually use:
    #   Open_Facility_Door -> door_left_open (a person opens the facility door)
    #   Enter_Facility / Exit_Facility -> zone_occupied (a person is present in the doorway zone,
    #       transiting the structure — the person-in-monitored-zone occupancy episode)
    #   Abandon_Package -> package_appeared (an object is left and becomes present)
    "Open_Facility_Door": _spec("door_left_open", "episode"),
    "Enter_Facility": _spec("zone_occupied", "episode"),
    "Exit_Facility": _spec("zone_occupied", "episode"),
    "Abandon_Package": _spec("package_appeared", "point"),
}


def condition_for(activity_type: str) -> ConditionSpec | None:
    """Return the :class:`ConditionSpec` for ``activity_type``, or ``None`` if it does not map."""
    return MEVA_ACTIVITY_TO_CONDITION.get(activity_type)


def mapped_activity_types() -> tuple[str, ...]:
    """Every MEVA activity type the transform recognizes (sorted)."""
    return tuple(sorted(MEVA_ACTIVITY_TO_CONDITION))


# ---------------------------------------------------------------------------
# Optional MEVA-tuned pack scaffolding — NOT auto-registered (see module docstring).
# ---------------------------------------------------------------------------

#: Templates named with MEVA's own activity vocabulary, reusing the SHIPPED senses (absence /
#: transition / count) so a caller can arm a goal straight from a MEVA activity name. The behavior
#: dicts mirror ``percept.packs_default`` exactly — no new sense is invented here.
_MEVA_TEMPLATES = (
    GoalTemplate(
        name="person_opens_facility_door",
        condition="the {door_name} is open",
        parameters={"door_name": "facility door", "deadline_after_s": 300.0},
        target="door",
        behavior={"rearm": "deadline", "deadline_after_s": "{deadline_after_s}"},
        episode={"policy": "door_interval"},
        consequence={"miss_cost": "mild", "false_fire_cost": "trivial"},
        notes=(
            "MEVA person_opens_facility_door -> facility_observation door_left_open: fires once the "
            "door has stayed open past deadline_after_s."
        ),
    ),
    GoalTemplate(
        name="person_loiters",
        condition="a person is present in the {zone_name}",
        parameters={"zone_name": "monitored zone"},
        target="person",
        behavior={"type": "count"},
        consequence={"miss_cost": "mild", "false_fire_cost": "trivial"},
        notes=(
            "MEVA person_loiters / person_enters_scene_through_structure -> facility_observation "
            "zone_occupied: fires once per distinct occupancy episode."
        ),
    ),
    GoalTemplate(
        name="person_abandons_package",
        condition="a package has appeared at the {zone_name}",
        parameters={
            "zone_name": "monitored area",
            "from_state": "no package is present",
            "to_state": "a package is visible",
        },
        target="package",
        behavior={"type": "transition", "from": "{from_state}", "to": "{to_state}"},
        consequence={"miss_cost": "mild", "false_fire_cost": "trivial"},
        notes=(
            "MEVA person_abandons_package / drop-off -> facility_observation package_appeared: "
            "fires once on the rising edge of a package becoming visible."
        ),
    ),
)

_MEVA_COMPONENTS = (
    ComponentRecommendation(
        ref="sense:physical_scene",
        strength="preferred",
        reason="a MEVA fixed security camera is a physical scene, not a screen or a conversation.",
    ),
    ComponentRecommendation(
        ref="identity_resolver:appearance_recent_set",
        strength="preferred",
        reason="zone occupancy needs short-horizon person tracking across a few frames only.",
    ),
)


def build_meva_facility_pack() -> DomainPack:
    """Build (but do NOT register) the ``facility_observation_meva@1`` pack.

    A self-contained pack whose templates are named with MEVA's activity vocabulary and reuse the
    shipped senses. Returned as data so a caller decides whether/when to
    :func:`register` it — this module never mutates the shared registry at import.
    """
    return DomainPack(
        id="facility_observation_meva",
        version="1",
        templates={t.name: t for t in _MEVA_TEMPLATES},
        components=_MEVA_COMPONENTS,
        policy_clamps={"false_negative_cost_min": "mild", "retention_max_days": 30.0},
        notes=(
            "MEVA-vocabulary view of facility_observation for dataset->fixture eval. Templates are "
            "named after MEVA activity types and reuse the shipped absence/transition/count senses; "
            "no new sense is introduced. Not auto-registered — register() it explicitly."
        ),
    )


def register(*, overwrite: bool = False) -> DomainPack:
    """Register :func:`build_meva_facility_pack` onto the default registry and return it.

    THE ONE-LINE WIRING the human/Fable applies to make ``facility_observation_meva@1`` resolvable::

        from eval.datasets.meva.pack_ext import register; register()

    Left as an explicit opt-in (never called at import) so this dev-tooling module does not collide
    with sibling pack-registry work.
    """
    pack = build_meva_facility_pack()
    register_domain_pack(pack, overwrite=overwrite)
    return pack
