---
title: Configuration reference — everything you can tune
updated: 2026-07-28
applies_to: percept-vision 1.0.0
status: stable
---
# Configuration reference — everything you can tune

Every value on this page is declared in `percept/config.py`. The `PERCEPT_*` environment variables
are read **once**, by `load_settings()`, into an immutable `Settings` tree — nothing in the cognition
logic reads the environment directly, so a run is reproducible from its environment. Types and
defaults below are the field definitions in that module; cite the module, not a line number.

- Package `percept-vision` 1.0.0 · `pip install percept-vision` · `import percept` · Python >= 3.10
- Wire contract `CONTRACT_VERSION = "1.10.0"`, `WATCH_VERSION = "1.1.0"`,
  `CONFORMANCE_VERSION = "1.1.0"`
- Website: <https://percept.percia.ai>

---

## How configuration enters the system

There are three doors. They compose — a later door can override what an earlier one set.

**1. `Percept.create(..., settings=)`** — the direct constructor. If you pass no `settings=`, it calls
`load_settings()`, which reads the whole `PERCEPT_*` environment into a `Settings` tree. Pass a
pre-built `Settings` to bypass the environment entirely. Per-seam keyword overrides
(`vision=`, `conductor=`, …) win over what `settings.models.*` selected.

```python
import percept
core = percept.Percept.create(vision="claude")          # settings from env
core = percept.Percept.create(settings=my_settings)      # settings you built
```

**2. `Percept.configure(...)`** — the resolver door (the configuration-ownership boundary,
ADR-008). Instead of raw env, you hand it the six ordered configuration objects and it resolves them
through `ConfigResolver`, compiles the result to real runtime primitives, and calls `create()` for
you:

```python
core = percept.Percept.configure(
    source_profile=...,        # what KIND of world
    domain_pack=...,           # a bundle of goals/detectors for a domain
    execution_policy=...,      # deployment-wide policy (e.g. strict admission)
    optimization_lock=...,     # a validated, provenance-stamped tunable set
    watch_plan=...,            # the explicit, highest-precedence layer
    vision="claude",           # any create() kwarg passes straight through
)
```

Every keyword `configure()` does not itself consume passes through to `create()`; an explicit
`settings=`/`sense_policy=`/`plan_mode=` you pass wins over what resolution produced.
`Percept.from_bundle(mapping)` is exactly `Percept.configure(**mapping)`.

**3. `watch(stream, goals, profile=)`** — the streaming entry point. `profile=None` (the default) is
today's `Percept.create()` behaviour, byte-for-byte. Pass a `WatchProfile` and it is resolved for
real and threaded into the core that `watch()` builds (never a mid-run mutation).

Note on `run()`: `percept.run(...)` has **no** `settings=` keyword. Either pass a core you already
built (`percept.run(core, source=...)`) or let it build one from the environment
(`percept.run(source=..., goals=..., vision=...)`).

---

## The Settings tree

`Settings` is ten frozen dataclass groups. Each field maps to one `PERCEPT_*` variable.

Boolean parsing: a value is truthy iff it is one of `{1, true, yes, on}` (case-insensitive, trimmed).
Anything else — including `0`, `false`, `no`, `off`, empty — is false. An unparseable numeric value
falls back to the default rather than raising.

### `gate` — the consequence gate

| Field | Type | Default | Env var |
|---|---|---|---|
| `threshold` | float | `0.7` | `PERCEPT_GATE_THRESHOLD` |
| `validity_window_s` | float | `90.0` | `PERCEPT_GATE_VALIDITY_S` |
| `key_ttl_s` | float | `3600.0` | `PERCEPT_GATE_KEY_TTL_S` |
| `max_keys` | int | `0` | `PERCEPT_GATE_MAX_KEYS` |
| `stale_ask` | bool | `False` | `PERCEPT_GATE_STALE_ASK` |
| `require_false_first` | bool | `False` | `PERCEPT_GATE_REQUIRE_FALSE_FIRST` |
| `refractory_s` | float | `0.0` | `PERCEPT_GATE_REFRACTORY_S` |

`threshold` is the confident-yes cutoff; a verdict below it is not a fire. `key_ttl_s`/`max_keys`
bound a 24/7 gate's key set (`0` disables each). `require_false_first` demands a confident-false
baseline before a key can fire, so a condition already true at start is not a fire. `refractory_s`
suppresses a re-fire of the same key within that many seconds. `stale_ask` folds perception recency
into freshness so a goal not read within `validity_window_s` begins to ASK. The last three default
off, preserving the historical rising-edge gate exactly.

### `entity` — frame-to-frame continuity

| Field | Type | Default | Env var |
|---|---|---|---|
| `iou_match` | float | `0.5` | `PERCEPT_IOU_MATCH` |
| `max_entities` | int | `64` | `PERCEPT_MAX_ENTITIES` |

### `cache` — the frame-hash short-circuit

| Field | Type | Default | Env var |
|---|---|---|---|
| `enabled` | bool | `False` | `PERCEPT_CACHE_ENABLED` |
| `threshold` | float | `0.0` | `PERCEPT_CACHE_THRESHOLD` |
| `ttl_s` | float | `8.0` | `PERCEPT_CACHE_TTL_S` |
| `tau_bits` | int | `6` | `PERCEPT_CACHE_TAU_BITS` |
| `audit_rate` | float | `0.05` | `PERCEPT_CACHE_AUDIT_RATE` |

### `models` — backend selection and model ids

| Field | Type | Default | Env var |
|---|---|---|---|
| `vision` | str \| None | `None` | `PERCEPT_VISION_BACKEND` |
| `conductor` | str \| None | `None` | `PERCEPT_CONDUCTOR_BACKEND` |
| `specialist` | str \| None | `None` | `PERCEPT_SPECIALIST_BACKEND` |
| `stt` | str | `"fake"` | `PERCEPT_STT_BACKEND` |
| `tts` | str | `"fake"` | `PERCEPT_TTS_BACKEND` |
| `vision_model` | str | `"claude-opus-4-8"` | `PERCEPT_VISION_MODEL` |
| `conductor_model` | str | `"claude-haiku-4-5"` | `PERCEPT_CONDUCTOR_MODEL` |
| `specialist_model` | str | `"claude-haiku-4-5"` | `PERCEPT_SPECIALIST_MODEL` |

`vision`/`conductor`/`specialist` default to `None`, **not** `"fake"`: you do not land on a scripted
backend by not typing something. A bare LIVE run with `vision` unset raises `ModeError` rather than
silently faking a perception result — set a real backend and its key, or pass `mode="offline"` for
the keyless scripted path. See [reference/backends.md](../judges/backends.md).

### `routing` — local-router deferral

| Field | Type | Default | Env var |
|---|---|---|---|
| `confidence_threshold` | float | `0.6` | `PERCEPT_ROUTE_THRESHOLD` |

Below this confidence the local router defers to the conductor.

### `identity` — cross-gap re-identification

| Field | Type | Default | Env var |
|---|---|---|---|
| `appearance_threshold` | float | `0.8` | `PERCEPT_APPEARANCE_MATCH` |

### `episodes` — counting windows

| Field | Type | Default | Env var |
|---|---|---|---|
| `window_s` | float | `3600.0` | `PERCEPT_EPISODE_WINDOW_S` |
| `max_episodes` | int | `64` | `PERCEPT_MAX_EPISODES` |

### `events` — temporal perception and incident snippets

| Field | Type | Default | Env var |
|---|---|---|---|
| `window` | int | `12` | `PERCEPT_FRAME_WINDOW` |
| `preroll_max_bytes` | int | `134217728` (128 MB) | `PERCEPT_PREROLL_MAX_BYTES` |
| `merge_gap_s` | float | `1.5` | `PERCEPT_EVENT_MERGE_GAP_S` |
| `audio_window_s` | float | `4.0` | `PERCEPT_AUDIO_WINDOW_S` |

`preroll_max_bytes` is a byte budget for the incident pre-roll ring — a memory bound, not a frame
count. When exceeded, the oldest frames are evicted and every eviction is reported, never silent.

### `consent` — actuation gating

| Field | Type | Default | Env var |
|---|---|---|---|
| `allow_computer_use` | bool | `False` | `PERCEPT_ALLOW_COMPUTER_USE` |
| `allowed_actions` | frozenset | `frozenset()` | `PERCEPT_ACT_ALLOWED_ACTIONS` |
| `actuation` | str | `"none"` | `PERCEPT_ACTUATION` |

`allowed_actions` parses a comma-separated list. `actuation` is one of `none` (default),
`screen` (advisory software acts), or `physical` — a declaration that this deployment sits outside
Percept's envelope, after which every act is refused (see ADR-007).

### `edge` — the tier-0 salience gate

| Field | Type | Default | Env var |
|---|---|---|---|
| `motion_gate` | float | `0.10` | `PERCEPT_MOTION_GATE` |
| `salience_stat` | str | `"block_max"` | `PERCEPT_SALIENCE_STAT` |
| `ego_compensate` | bool | `False` | `PERCEPT_EGO_COMPENSATE` |
| `motion_gate_block_max` | float | `0.20` | `PERCEPT_MOTION_GATE_BLOCK_MAX` |
| `motion_gate_p95` | float | `0.15` | `PERCEPT_MOTION_GATE_P95` |
| `coast_enabled` | bool | `True` | `PERCEPT_COAST_ENABLED` |

`salience_stat` is one of `mean` (whole-frame average motion), `block_max` (the most active block —
localized, so small low-motion actions survive), or `p95` (95th percentile). Thresholds are
**per-statistic and not interchangeable**: `block_max`/`p95` sit on a higher value scale than the
mean, each with its own cutoff, so the mean-tuned `0.10` over-fires them. `salience_stat="mean"` with
`motion_gate=0.10` reproduces the legacy whole-frame gate byte-for-byte, but it is **not** the default
any more and is rarely what you want: a whole-frame average dilutes a small mover against a still
background, and on a 100-case corpus it admitted **zero frames on 24 cases** (0.572 recall) versus
`block_max@0.20`'s 2 (0.947). `ego_compensate` subtracts
global camera motion before diffing — for a **moving** camera only. See
[reference/edge-policy.md](../admission/wake-policy.md).

---

## Environment variables read outside the Settings tree

These are read directly where they are used, not through `load_settings()`.

| Variable | Default | Effect |
|---|---|---|
| `PERCEPT_MODE` | `live` | `offline` selects keyless scripted seams; anything else is live |
| `PERCEPT_JUDGE_CATALOG` | unset | path to a JSON judge catalog; unset ⇒ **no** judge authorized |
| `PERCEPT_NO_BANNER` | unset | suppress the scripted-backend startup banner |
| `PERCEPT_HOME` | `~/.percept` | on-box home for labels/dispositions |
| `PERCEPT_MAX_INCIDENTS` | (session default) | cap on retained session incidents |
| `PERCEPT_VISION_TIMEOUT_S` | `30` | per-call vision timeout, seconds |
| `PERCEPT_STRIP_PROMPT` | `change` | Gemini strip-prompt mode: `change` or `neutral` |
| `PERCEPT_POSE_PROVIDER` | `rtmo` | pose backend provider |
| `PERCEPT_RTMO_MODEL` | `rtmo` | RTMO model id |
| `PERCEPT_POSE_DEVICE` | `cpu` | pose inference device |
| `PERCEPT_POSE_MODEL` | unset | pose model path |
| `PERCEPT_POSE_CHANNELS` | unset | pose channel selection |
| `PERCEPT_SYNTHETIC_DIR` | unset | synthetic-fixture directory (tests) |

### Provider credentials (not `PERCEPT_*`)

A selected live backend with no key raises `ModeError` naming the exact variable.

| Backend | Credential (any one) |
|---|---|
| `claude` | `ANTHROPIC_API_KEY` |
| `gemini` | `GEMINI_API_KEY` / `GOOGLE_API_KEY` / `GOOGLE_AISTUDIO_API_KEY` |
| `openrouter` | `OPENROUTER_API_KEY` / `OPENROUTER_KEY` |
| `deepgram` | `DEEPGRAM_API_KEY` |

---

## The recommended watch defaults

The tier-0 **eye** is now tuned by default: `salience_stat="block_max"` with its own
`motion_gate_block_max=0.20` cutoff. A developer embedding via `run()` /
`Percept.create()` gets the localized gate without setting anything, so the historical failure mode —
a silent zero-admission because the whole-frame `mean` averaged the action away — no longer happens by
default.

What is still deliberately conservative is **delivery**, not admission: `refractory_s=0.0` means every
qualifying frame can fire. The `percept-watch` CLI sets these in the environment *before* importing
`percept`; an embedder should consider the same:

| Variable | Value | When |
|---|---|---|
| `PERCEPT_GATE_REFRACTORY_S` | `15` | one notification per episode instead of a stream of repeats |
| `PERCEPT_EGO_COMPENSATE` | `true` | **moving** camera only (glasses / body-cam) |
| `PERCEPT_SALIENCE_STAT` | `mean` | only to reproduce the pre-1.0 whole-frame gate — expect it to admit far less |

See [how-to/tune-admission.md](../admission/tuning-admission.md) and, for content
that has near-zero pixel motion (screenshares, documents), [how-to/screenshare.md](../watches/scenario-screenshare.md).

---

## SourceProfile vs WatchProfile

These are **orthogonal axes**. A `SourceProfile` says what KIND of world the stream is; a
`WatchProfile` says how to tune the eye within a camera-shaped situation. You can combine any of one
with any of the other.

| | SourceProfile | WatchProfile |
|---|---|---|
| Answers | what kind of world | how to tune the eye |
| Fields | `name`, `modalities`, `temporal_unit`, `admission_strategy`, `evidence_strategy`, `identity_strategy`, `default_timebase` | `salience_stat`, `stat_thresholds`, `plan_mode`, `cadence`, `gate.require_false_first`, `gate.refractory_s` |
| Gate fields | **forbidden** (a source profile never carries gate knobs) | allowed (`gate.*`) |

Built-in **SourceProfiles**: `physical_scene` (default, stable), `screen_document` (stable —
content-key admission plus a fixed low-rate coast floor), `conversation` (experimental).

Built-in **WatchProfiles**: `fixed-cam-localized`, `dynamic-action`, `handheld`, `wildcard-default`.

To author your own profile, see [how-to/custom-configuration.md](profiles.md) — that
recipe (define + register a `SourceProfile` / `WatchProfile`, and apply it) is not duplicated here.
For reading a `percept-doctor` report and tuning the admission signal for your content, see
[how-to/tune-admission.md](../admission/tuning-admission.md).

---

## The ConfigResolver precedence chain

`ConfigResolver` (the ADR-008 configuration-ownership boundary) layers six configuration objects. From
lowest to highest precedence — a higher layer wins a conflict at the same field:

```
SDK defaults < SourceProfile < DomainPack < ExecutionPolicy < OptimizationLock
             < Deployment (PERCEPT_* env) < WatchPlan (explicit)
```

**Ownership gates precedence, not the reverse.** A field write from a layer that does not own that
field is a configuration error, even though the layer sits high in the chain — a layer cannot reach
outside its ownership just because it is more specific.

**Provenance is recorded per field.** The resolved result carries, for every effective field, which
layer set it — inspectable via `percept resolve --explain` (see
[reference/cli.md](../resources/cli.md)).
