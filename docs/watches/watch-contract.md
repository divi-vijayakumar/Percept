---
title: `watch()` — the frozen public contract
updated: 2026-07-23
applies_to: percept-vision 1.0.0
status: stable
---
# `watch()` — the frozen public contract

`percept.watch(stream, goals, profile=None) -> AsyncIterator[WatchEvent]` is the stable, versioned
entrypoint for direct frame-in / event-out integrations. It is a thin seam over the existing
cognition core — `Percept.create()` + `Percept.perceive_judged()` — not a new firing path.
**No detection behavior is hardcoded in `watch()` itself.**

```python
from percept import Goal, watch

goals = [Goal(id="zone", condition="a person is in the restricted zone", say="Alert: person in zone")]

async for event in watch(camera_frames, goals):
    if event.action == "fire":
        notify(event.text)
    elif event.action == "ask":
        ask_user(event.text)
```

## Signature

```python
async def watch(
    stream: AsyncIterable[Any] | Iterable[Any],
    goals: list[Goal],
    profile: Any = None,
    *,
    percept: Percept | None = None,
    source: str = "camera",
) -> AsyncIterator[WatchEvent]: ...
```

- **`stream`** — an (async or sync) iterable of frames, oldest to newest. A frame is whatever the
  configured `vision` backend accepts.
- **`goals`** — the `Goal`s to watch; registered on the core before the first frame.
- **`profile`** — OPTIONAL and deliberately **opaque today**. `WatchProfile` will bind
  `(environment, goal-type, state)` to the runtime's knobs (`salience_stat`, `plan_mode`, cadence,
  gate config, ...) — the free variables of `Q(p | environment, goal, state)`. **The documented
  default (`profile=None`) is today's
  `Settings`-driven behavior** (`percept.load_settings()`), unchanged. A bare `Settings` instance
  is also accepted as a minimal, non-behavior-changing pass-through (see
  `percept.watch._settings_from_profile`) — unsupported profile objects resolve to the default.
- **`percept`** — OPTIONAL, a pre-built `Percept` core to reuse (custom backends, state carried
  across calls). Omit it to get one built with `Percept.create()`.
- **`source`** — the frame-source key; multi-camera setups pass distinct sources per `watch()` call.

Frames are consumed **in order**; each frame yields zero or more `WatchEvent`s (one per active goal
that fired or asked this tick).

## `WatchEvent` — the streamed contract

```python
@dataclass
class WatchEvent:
    goal_id: str
    action: Literal["fire", "ask", "silent"]
    t: float
    text: str = ""
    entity_id: str | None = None
    verdict: Verdict | None = None
```

`action` is the three-state gate's outward decision:

| `action`   | meaning                                                          |
|------------|-------------------------------------------------------------------|
| `"fire"`   | a rising-edge, user-facing alert (`text` is the goal's `say`)     |
| `"ask"`    | low-confidence or stale evidence — the gate refuses to guess      |
| `"silent"` | no user-facing change (part of the type for completeness — see below) |

Today's `watch()` wraps `Percept.perceive_judged()` byte-for-byte, and that internal wiring never
surfaces a `"silent"` action to callers — a quiet tick simply yields nothing. `"silent"` is in the
`Literal` for forward/observability completeness (a future consumer, e.g. a health dashboard, may
want the full three-state stream), not because `watch()` emits it today.

`WatchEvent` is defined in `percept.contracts` alongside the rest of the frozen wire DTOs (`Verdict`,
`Incident`, ...) and is exported from `percept` directly. It is **not** `percept.events.Event` — that
is the older, distinct *temporal-perception* event (a pass, a rep, a discrete action over a frame
window); the names were kept apart deliberately to avoid collision.

## Versioning

- **`percept.WATCH_VERSION`** (`percept.watch.WATCH_VERSION`) — semver of the `watch()` entrypoint
  itself (the function signature + streaming semantics). Bump MAJOR on a breaking signature/streaming
  change, MINOR for an additive optional kwarg, PATCH otherwise.
- **`percept.CONTRACT_VERSION`** — semver of the DTO wire shapes `watch()` returns (`WatchEvent` and
  friends). It is `1.10.0` at this release (`WATCH_VERSION` is `1.1.0`), and the two version
  independently. Same-MAJOR is wire-compatible (see `percept.contracts.negotiate`).

These are independent: a `watch()` behavior change (new kwarg) bumps `WATCH_VERSION`; a new/changed
DTO field bumps `CONTRACT_VERSION`. Most changes bump at most one.

## Relationship to `run()`

[`percept.runtime.run()`](creating-a-watch.md) is a separate, richer entrypoint: it owns live camera
capture, the tier-0 **motion gate** (the cheap "should we even look?" pre-filter), pre-roll snippets,
health events, and reconnect/backpressure handling, yielding `Incident`s. `watch()` is the minimal
seam underneath that richness — a direct frame-in / `WatchEvent`-out wrapper around the judge+gate
core, with no tier-0 gating or capture management of its own. Use `run()` for a production camera
worker today; `watch()` is the frozen contract simpler and non-camera integrations build on. A future
compatible extension may layer it on `run()`'s tier-0 path when that choice can be expressed as a
profile setting rather than a hardcoded default.
