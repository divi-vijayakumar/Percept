# Create a custom configuration for your use case

Percept configuration is layered and open. You rarely need to fork anything — you tune a shipped
profile, register a new one, or bind a `WatchProfile` to your `(environment, goal, state)`. This
guide shows three levels, from a one-line tweak to a fully registered profile.

Three axes, each a different question:

| Object | Answers | Keyed on |
| --- | --- | --- |
| `SourceProfile` | what **kind of world** is on the other end of the stream | an open, name-keyed registry |
| `WatchProfile` | how to tune the **eye within that world** | `(environment, goal_type, state)` |
| `Settings` env vars | one knob, right now, no code | `PERCEPT_*` |

`environment` on a `WatchProfile` is a camera **mount** (`fixed_camera`, `handheld`), not a source
class. The two axes are orthogonal: a screen recording (`SourceProfile`) can be watched with any
mount's tuning (`WatchProfile`).

For the full `Settings` tree and the layer precedence, see [../configuration.md](../configuration.md).
The lowest-effort change of all is an env var — `PERCEPT_SALIENCE_STAT=block_max`,
`PERCEPT_GATE_REFRACTORY_S=15`, `PERCEPT_MOTION_GATE_BLOCK_MAX=0.20`, `PERCEPT_COAST_ENABLED=0` —
no code, no registration.

## A. Tune a shipped SourceProfile

`with_parameters(**overrides)` returns a tuned copy of any shipped profile — the original is frozen
and untouched. Use it when the shipped strategy is right but a value or two is not.

```python
from percept.source_profiles_default import SCREEN_DOCUMENT

MY_SOURCE = SCREEN_DOCUMENT.with_parameters(change_signal="my-phash", coast_interval_s=1.0)
```

This keeps `SCREEN_DOCUMENT`'s strategies (content-keyed identity, screenshot-plus-changed-region
evidence, the mandatory coast) and overrides only the two parameters you named.

## B. Define and register a new SourceProfile

The source-profile registry is **open and name-keyed** — you extend Percept by adding to it from
your own package, never by editing a shipped tuple.

```python
from percept.sourceprofile import SourceProfile, register_source_profile

MY_SOURCE = SourceProfile(
    name="financial_dashboard",
    modalities=frozenset({"video", "text"}),
    temporal_unit="page",
    admission_strategy="tier0_localized_salience_or_coast",
    evidence_strategy="screenshot_plus_changed_region",
    identity_strategy="content_key",
    default_timebase="media",
    parameters={"change_signal": "my_phash", "salience_stat": "block_max",
                "salience_threshold": 0.20, "coast_interval_s": 1.0, "coast_required": True},
    stability="experimental",
)
register_source_profile(MY_SOURCE)   # -> DEFAULT_SOURCE_REGISTRY
```

`__post_init__` validates **shape only** (non-empty name, a real timebase, immutable parameters) —
never membership in a closed vocabulary, so your own strategy names are welcome.

**Gate fields are structurally forbidden in `parameters`.** `refractory_s`, `rising_edge`,
`require_false_first`, `fire`/`ask`/`silent`, `dwell_s`, `gate`, … each raise `SourceProfileError`
if they appear. A source profile has no firing rule by construction — the Gate is the sole firing
authority, and those knobs belong to the `WatchProfile` (level C). This is the "One Gate" invariant
enforced in code, not convention.

**Shipping from a third-party package.** Instead of calling `register_source_profile` at import,
advertise an entry point in the `percept.source_profiles` group. Each entry point is a callable
Percept invokes with `register_source_profile`:

```toml
# your package's pyproject.toml
[project.entry-points."percept.source_profiles"]
financial_dashboard = "my_pkg.profiles:register"
```

```python
# my_pkg/profiles.py
def register(register_source_profile):
    register_source_profile(MY_SOURCE)
```

## C. Define and register a custom WatchProfile

A `WatchProfile` binds the runtime's tuning knobs to a `(environment, goal_type, state)` context.
It invents no knobs — every field is a typed handle onto a setting read elsewhere.

```python
from percept.watchprofile import WatchProfile, CadenceProfile, GateProfile, DEFAULT_REGISTRY

MY_PROFILE = WatchProfile(
    name="warehouse-fixed",
    environment="fixed_camera", goal_type="stative", state="start_anchored",
    salience_stat="block_max", stat_thresholds={"block_max": 0.20},
    plan_mode="window",
    cadence=CadenceProfile(base_fps=2.0, max_fps=10.0),
    gate=GateProfile(require_false_first=False, refractory_s=15.0),
)
DEFAULT_REGISTRY.register(MY_PROFILE)
```

The registry keys off the profile's own `(environment, goal_type, state)` fields — you cannot
register under a disagreeing key — and `resolve()` falls back through `*` wildcards, so a broad
default always matches. There is **no `register_watch_profile()` helper and no entry-point group**
for watch profiles: register on `DEFAULT_REGISTRY` (or your own `WatchProfileRegistry`) directly.

## Apply it

**A WatchProfile** — pass it to `watch()`, which resolves it at core construction:

```python
from percept import watch

async for event in watch(stream, "the forklift entered the aisle", profile=MY_PROFILE):
    ...
```

For an embedding that builds the core itself, resolve the profile to concrete runtime knobs first:

```python
from percept import Percept
from percept.profile_wiring import resolve_watch_profile

r = resolve_watch_profile(MY_PROFILE)
core = Percept.create(settings=r.settings, sense_policy=r.sense_policy, plan_mode=r.plan_mode)
```

`resolve_watch_profile` returns a `ResolvedProfile(settings, sense_policy, plan_mode)` — every field
`Percept.create` needs. A profile only overrides the specific knobs it declares; it never resets an
operator's env-configured values it has no opinion on.

**A SourceProfile** — resolve its declared strategies into runtime settings through the
configuration boundary:

```python
core = Percept.configure(source_profile=MY_SOURCE)   # full layered resolution
# Percept.create(source_profile=MY_SOURCE) also accepts it.
```

`Percept.configure` runs the six-layer resolver (SDK < SourceProfile < DomainPack < ExecutionPolicy
< OptimizationLock < Deployment < WatchPlan), so a tuned admission signal and a declared coast floor
are bound onto the runtime with per-field provenance. See
[../configuration.md](../configuration.md) for the precedence and the provenance receipt.

## Prefer to derive a profile from data?

Hand-tuning is fine for a known environment. When you have labeled fixtures and want the numbers to
pick the winner, `percept-derive-profile` scores a declared ladder of candidate profiles and emits a
provenance-stamped, content-addressed profile — see [./derive-a-profile.md](./derive-a-profile.md).
To search safe numeric knobs within one fixed shape, see [./optimize.md](./optimize.md). Both refuse
to score if the configuration moved after your bars were declared.
