---
title: Official free DomainPacks
updated: 2026-07-28
applies_to: percept-vision 1.0.0
status: stable
---
# Official free DomainPacks

DomainPacks are versioned collections of Watch templates and advisory component recommendations.
They generate ordinary `WatchPlan` values; they do not arm a runtime, choose credentials, or deploy
anything.

```python
from percept import Percept
from percept.domain_pack import get_domain_pack, watch_template
from percept.watchplan_apply import apply_watch_plan

pack = get_domain_pack("physical_scene_starter@1")
plan = watch_template(
    "physical_scene_starter@1/package_appeared",
    {"zone_name": "loading dock"},
)

core = Percept.create(mode="offline", discover_plugins=False)
receipt = apply_watch_plan(core, plan, domain_pack=pack)
assert receipt.status in {"applied", "applied_with_warnings"}
```

Calling `get_domain_pack()` or `watch_template()` never auto-arms a Watch. Applying the returned plan
is a separate, explicit operation.

## Shipped pack names

| Pack | Templates | Qualification status |
|---|---|---|
| `physical_scene_starter@1` | `door_left_open`, `package_appeared`, `standing_water`, `zone_occupied` | Mechanics-only; `fixtures=()` and `quality_bars={}` |
| `ego_motion_starter@1` | `camera_view_moving`, `scene_transition`, `object_focus_gained`, `task_step_reached` | Mechanics-only; `fixtures=()` and `quality_bars={}` |
| `facility_observation@1` | Original four physical templates | Compatibility name; retained fixture/bar values are candidate declarations, not achieved results |

The deterministic test suite proves rendering, resolution, compilation, rising-edge behavior,
uncertainty handling, transition re-arming, and cognition replay. It does not establish real-media
accuracy, provider latency, or provider cost.

## Physical Scene Starter boundary

`zone_occupied` is scene-scoped. It fires once per contiguous confidently judged occupied interval;
it does not identify, track, or count people in a crowd. The other templates similarly describe
scene-level Watch mechanics. This pack is not medical, emergency, biometric, or security authority.

## Ego Motion Starter boundary

The Ego templates use two runtime primitives that already ship:

- an ordinary scene-level rising-edge Watch; and
- a two-state transition that must observe the authored `from` state before the authored `to` state.

`object_focus_gained` is category-level visible focus. It does not prove that a previously seen
object is the same object after a gap. `task_step_reached` compares two caller-authored visible
states; it does not infer a task plan or user intent.

General revisit, same-object reacquisition, person identity, tracking, location authority, and task
understanding are not supported claims of this starter pack.

## Discovery

All three built-in names resolve directly with `get_domain_pack()`. They are also installed as
separate `percept.packs` entry points, using the same discovery contract as third-party packs:

```python
from percept.domain_pack import discover_domain_pack_entrypoints

discover_domain_pack_entrypoints()
```

Applications import the installed `percept-vision` SDK. Do not copy or vendor its runtime,
subscriber, or pack implementation into an application.
