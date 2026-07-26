# ADR-008: Configuration ownership, resolution, and provenance

- **Status:** Accepted for the ownership/precedence model and the tunable inventory it is grounded
  in (the implementation — ships `percept/tunables.py`, the generated `tunables.json` artifact, and
  `tests/test_tunables.py`). The two-step binder this ADR specifies — `ConfigResolver` /
  `RuntimeCompiler` — and four of the six declared objects (`DomainPack`, `ExecutionPolicy`,
  `DeploymentConfig`, `WatchPlan`, `OptimizationLock`) do **not exist in code yet**; they ship in
  follow-on changes against the contract this document fixes. `SourceProfile` and `WatchProfile`
  already exist (ADR-005) and are extended, not replaced.
- **Depends on:** ADR-005 (Source profiles — the `FORBIDDEN_FIELDS` / structural-enforcement
  precedent this ADR reuses for ownership gating); ADR-002 (Runtime health & coverage — the
  "degraded is diagnostic, not automatic" rule §Corrections(b) restates for configuration);
  `percept/watch.py` (`WATCH_VERSION = "1.0.0"` — untouched); `percept/contracts/__init__.py`
  (`CONTRACT_VERSION`, `WatchSpec` — untouched).
- **Related:** ADR-001 (the Gate/judgment contract — one Gate, closed verdicts, restated in
  §Invariants); ADR-007 (Envelope enforcement — the same "refuse in the constructor, not in a
  review checklist" posture this ADR applies to out-of-ownership writes).
- **Source (read for this ADR):** `percept/config.py` (`Settings`, `load_settings`, 10 dataclasses),
  `percept/watch.py` (`watch()`, `_build_core`), `percept/agent.py` (`Percept.create`), the
  `Goal.state` sub-dict validators in `percept/behavior.py` (`coerce_authored_sense/episode/
  perception/rhythm`), `percept/watchprofile.py` (`WatchProfile`, `CadenceProfile`, `GateProfile`),
  `percept/sourceprofile.py` (`SourceProfile`, `FORBIDDEN_FIELDS`), `percept/harness/detectors.py`
  (`DetectorManifest`, `register_detector`), `percept/provider.py` (`ADAPTER_IDENTITY`, model id vs.
  provider name), `percept/doctor.py` (`EXIT_ZERO_COVERAGE`, the coverage-is-diagnostic rule),
  `percept/sprite.py` and `CHANGELOG.md` (the sprite→window measurement), `percept/gate.py`
  (`GATE_REASONS`).

## Context

The SDK has accumulated **six independent places a behavior-affecting value can be set**: the
`PERCEPT_*`-overridable `Settings` tree (`load_settings()`), keyword arguments to `watch()` /
`Percept.create()`, keys inside a `Goal`'s free-form `state` dict, a `WatchProfile`'s tuning fields,
a `SourceProfile`'s strategy fields, and a detector's registered `DetectorManifest`. Each was added
to solve one problem (tier-0 calibration, ADR-005's source-class axis, and clip-window
declaration) and each is internally coherent. None of them says, in code, **who is allowed to set
which field when two of them disagree**, or **why a value ended up what it did** when a client asks
"where did this threshold come from?" at 2am.

the implementation first did the accounting: `percept/tunables.py` enumerates every one of these knobs as a
typed `TunableSpec` and emits `tunables.json`, a generated artifact a CI parity test pins against
the schemas that produced it (`tests/test_tunables.py`). That inventory is this ADR's evidence base
— every claim below about what exists, what owns what, and what is missing is checked against it,
not asserted from memory.

Two concrete failures motivate fixing the ownership question now rather than after a seventh
configuration surface appears:

1. **Silent last-writer-wins is already possible.** `Percept.create()` accepts an explicit
   `settings=` alongside a `WatchProfile`-derived one; `resolve_watch_profile` picks a `base` and a
   caller can pass a different one, and nothing stops a `PERCEPT_*` env var from disagreeing with
   both without either party finding out.
2. **No caller can currently ask "which of these six places set this field, and did anything
   override anything?"** — the freshness/provenance model `percept.contracts.Attribute` already
   gives *perceived facts* (observed vs. asserted vs. cached) does not exist for *configuration
   facts*.

## I/O contract

This ADR is a contract about a boundary, so the boundary is stated first, before any of the six
objects are described individually.

**Input** — six independently constructible, independently versioned configuration objects (see
§Six declared objects). Each is ordinary data: no object reads another, no object talks to the
runtime, and constructing one performs no I/O and no side effect beyond shape validation (the same
discipline `SourceProfile.__post_init__` and `WatchProfile.__post_init__` already follow).

**The boundary is two functions, not one:**

```
(SourceProfile, DomainPack, ExecutionPolicy, DeploymentConfig, OptimizationLock, WatchPlan)
        │
        ▼
   ConfigResolver.resolve(...)   — validates ownership, applies precedence, records provenance
        │
        ▼
   ResolvedWatch                 — immutable; one concrete value per field + per-field Provenance
        │
        ▼
   RuntimeCompiler.compile(...)  — the ONLY place a ResolvedWatch becomes runtime primitives
        │
        ▼
(Goal / Goal.state, WatchSpec, SensePolicy, provider construction)
```

**Output** — the existing runtime primitives, unchanged: a `Goal` (with its `state` dict populated),
a `WatchSpec` for the edge wire, a `SensePolicy` for the fovea, and constructed provider adapters
(vision/conductor/specialist/stt/tts). `RuntimeCompiler` produces *only* things `percept/agent.py`
and `percept/runtime/worker.py` already know how to consume — it is explicitly not a seventh place
that invents a new runtime shape.

Nothing may skip step one and hand raw profile objects to step two: `RuntimeCompiler` accepts a
`ResolvedWatch`, never a `WatchPlan` or a `SourceProfile` directly. This is what "the resolver never
absorbs runtime compilation" means operationally — a call site cannot bypass provenance tracking by
constructing a `ResolvedWatch` field-by-field outside `ConfigResolver.resolve`, because every field
on it is written exactly once, at resolution, with its provenance attached in the same assignment.

## Six declared objects

| object | owns | status |
|---|---|---|
| `SourceProfile` | source-class defaults: admission/evidence/identity strategy, default timebase, coverage-expectation parameters (`parameters["coverage_floor"]`-shaped, see §Corrections(b)) | **exists** (`percept/sourceprofile.py`, ADR-005) — extended, not replaced: this ADR adds no field, it adds an ownership row for the fields it already has |
| `DomainPack` | domain vocabulary, phrase templates, per-domain recommended `WatchProfile`/`SourceProfile` pairings | **new** — not yet in code |
| `ExecutionPolicy` | resource/scheduling posture: judge-call budget, cadence ceilings, concurrency, degrade-under-load behavior | **new** — not yet in code |
| `DeploymentConfig` | which providers are AVAILABLE and how to reach them: credentials, endpoints, `PERCEPT_*` env — **env vars live here, nowhere else** | **new** — not yet in code (today `Settings`/`load_settings` plays this role ad hoc; this object is where that responsibility formally lives once it ships) |
| `WatchPlan` | the user's stated intent: goals, explicit per-field overrides, the `source=`/`profile=` selection a caller passes to `watch()` | **new** — not yet in code (today `watch()`'s own kwargs play this role ad hoc) |
| `OptimizationLock` | a validated, hashed selection of tunables an optimizer previously proved safe — never authored by hand | **new** — not yet in code |

Each object is independently versioned (see §Version ledger) and independently constructible with
no reference to the others. `ConfigResolver` is the first place any two of them are read together.

### Ownership table (per field class)

| field class | owning object | example |
|---|---|---|
| source-class strategy | `SourceProfile` | `admission_strategy`, `identity_strategy` |
| domain vocabulary/templates | `DomainPack` | phrase templates, recommended pairings |
| resource/scheduling posture | `ExecutionPolicy` | judge-call budget, concurrency ceiling |
| provider availability + env | `DeploymentConfig` | `PERCEPT_VISION_BACKEND`, credentials, endpoints |
| validated tunable selection | `OptimizationLock` | a `salience_stat` + `stat_thresholds` pair an optimizer proved |
| explicit user intent | `WatchPlan` | the goal text, an explicit `salience_stat="block_max"` override |

**Ownership gates precedence, not the reverse.** A field write from an object that does not own
that field's class is not "the last writer wins" — it is a `ConfigError`, raised by
`ConfigResolver.resolve` before any value is produced. A `DomainPack` cannot set
`PERCEPT_VISION_BACKEND`-shaped deployment fields no matter when it is applied in the precedence
chain, because deployment fields are not in its owned set. This mirrors `SourceProfile`'s
`FORBIDDEN_FIELDS` (ADR-005 §Decision 2): the refusal is structural, at construction/resolution
time, never a review-time convention.

## Two-step binding

**Step 1 — `ConfigResolver.resolve(...) -> ResolvedWatch`.** Takes some subset of the six objects
(any may be omitted; an omitted object contributes nothing, it does not error), validates that every
non-default field on every object belongs to that object's owned class, applies the precedence order
below field-by-field, and returns an **immutable** `ResolvedWatch` — one concrete value per
resolvable field, each carrying a `Provenance` record: which object supplied the winning value, and
whether it overrode a lower-precedence value that disagreed.

**Step 2 — `RuntimeCompiler.compile(resolved: ResolvedWatch) -> ...`.** Turns the immutable
`ResolvedWatch` into the runtime shapes `percept/agent.py` already knows: a `Goal` (state populated
from the resolved fields the state-key inventory in `tunables.json` names), a `WatchSpec` (the
existing low-level edge wire form — **untouched, not extended, not replaced**), a `SensePolicy`
(today's `CadenceProfile.to_sense_policy()` path), and constructed provider adapters via
`percept.provider` (name resolves to factory; a model id is configuration carried to that adapter,
never an adapter selector — see §Invariants).

The resolver never absorbs runtime compilation, and the compiler never re-derives ownership or
precedence — each does exactly one job, in one direction, once per `watch()` call. This mirrors the
existing `resolve_watch_profile` (`percept/profile_wiring.py`) → `Percept.create` split, generalized
from one profile type to six typed inputs with provenance tracking added.

### Precedence, low to high

```
SDK defaults  <  SourceProfile  <  DomainPack  <  ExecutionPolicy  <  OptimizationLock  <  Deployment (incl. PERCEPT_* env)  <  WatchPlan explicit
```

A higher-precedence object's value wins **only for fields it owns** (see the ownership table); a
`WatchPlan` explicit override on a field it does not own is itself a `ConfigError`, not a silent
promotion to the top of the chain.

**Two tested consequences**, both direct readings of the ordering above and both to be pinned by
tests in the `ConfigResolver` implementation:

1. **Env beats profile defaults.** A `DeploymentConfig`-owned value (including a `PERCEPT_*` env
   var) outranks the same field's `SourceProfile`/`DomainPack`/`ExecutionPolicy` default, because
   `Deployment` sits above all three in the chain. This is today's behavior for the fields
   `Settings`/`load_settings` already covers (env always wins over a dataclass default); the
   ordering formalizes it rather than changing it.
2. **Explicit `WatchPlan` fields beat env.** A caller who explicitly names `salience_stat=` (or any
   other `WatchPlan`-ownable field) on their watch plan outranks whatever `PERCEPT_*` env is set,
   because `WatchPlan` explicit sits above `Deployment` in the chain. An *unset* `WatchPlan` field
   does not participate — it is not "explicit," so the next value down the chain applies untouched.

**Loud diagnostic on profile override.** When a `Deployment`/env value overrides a field owned by an
**explicitly selected** profile (a `SourceProfile` or `WatchProfile` the caller named, not a
resolved default), `ConfigResolver` emits a diagnostic the caller cannot miss — the same posture
`SourceProfile.admission_note()` and `percept doctor`'s findings vocabulary already take for
"something is not what you might assume it is." Silence here is exactly the failure mode ADR-005
existed to close for gate semantics; this is its configuration-provenance analogue.

## Corrections

Four points where existing prose or intuition disagrees with what the code actually does. The ADR
follows the code.

**(a) Window, not sprite, for screen evidence defaults.** `percept/sprite.py` is **deprecated in the
plan path**: an ordered frame *window* beat a sprite montage unconditionally on the measured
dynamic-goal set — recall 0.68 → 0.73 (0.90 with gate-relax) — and `percept/profile_resolution.py`
can never select `plan_mode="sprite"`; a spec/profile that still names it is normalized to
`"window"` with a `DeprecationWarning` (`percept.agent._normalize_plan_mode`). Any configuration
object that defaults evidence-gathering shape for screen/document content must default to
`plan_mode="window"` (or `temporal="strip"`/`"window"` in `PerceptionSpec` terms), never `"sprite"`.
`percept.sprite.make_sprite` remains importable as a standalone primitive; nothing in the firing path
calls it.

**(b) Zero-admit is `'degraded'` only when a coverage floor was declared and unmet.** `percept/
doctor.py`'s own rule (module docstring): *"Zero admits is DIAGNOSTIC, not necessarily failure. A
quiet camera under a motion-gated watch correctly admits nothing... Runtime-side coverage
degradation is only justified when a DECLARED coverage expectation is violated, which is not
something a dry run knows."* A still camera legitimately admits 0 and is healthy. Whatever object
ends up owning a "coverage floor" field (a `SourceProfile` or `ExecutionPolicy` candidate — the
inventory in `tunables.json` flags this as unresolved, see the open-questions note at the end of
this ADR) must be **explicit, opt-in, and declared** — never inferred from an observed zero — for a
zero-admit run to be reported as degraded rather than quiet.

**(c) Keep existing names.** `offline`/`live` (not "simulated"/"production"), `AttentionIntent`
(the channel-neutral effective-attention DTO, `percept/contracts/__init__.py`), an opaque
`evidence_ref` (never a typed evidence blob at the contract layer), `WatchSpec` as the name of the
low-level edge wire form (untouched by this ADR — see §I/O contract), and `SourceProfile` as an
**extension** of the existing `percept/sourceprofile.py` dataclass, not a new type replacing it. No
renaming ships with this ADR or any card it authorizes.

**(d) A deployment change touching a locked field invalidates the `OptimizationLock` — loud, never
silent.** An `OptimizationLock` records a hash over the exact tunable values it validated. If a
`DeploymentConfig`/env change alters a field the lock covers, `ConfigResolver` must detect the hash
mismatch and refuse to apply the lock silently — it reports the mismatch (the same "loud" posture as
`ADMISSION_SIGNAL_UNTUNED` and `SOURCE_PROFILE_EXPERIMENTAL` in `percept/doctor.py`) rather than
letting the higher-precedence deployment value quietly outrank a validated selection. This is a
narrower rule than plain precedence: deployment still wins the *value*, but the lock's validity
claim is void and must say so, not persist as if nothing changed.

## Version ledger

Six version numbers already exist or are introduced by this ADR, and they mean six different things.
Conflating any two of them is a bug class this ledger exists to name:

| version | meaning | current value | bump rule |
|---|---|---|---|
| SDK package version | `packages/percept-vision/pyproject.toml` `version` | `0.7.0` (next PyPI release is `1.0.0` — local wheels only until then, per release policy) | semver over the whole package |
| `WATCH_VERSION` | `watch()`'s public API surface — signature + streaming semantics | `1.0.0` | MAJOR on a breaking signature/streaming change; MINOR for an additive kwarg (e.g. the goals-widening card held this at `1.0.0` because the change was additive) |
| `CONTRACT_VERSION` | DTO shapes across `percept.contracts` (the frozen membrane) | `1.6.0` | MAJOR on a breaking frozen-shape change; MINOR per additive batch (new optional field/DTO); PATCH for docs-only |
| `WatchSpec.version` | the edge wire schema integer | `1` | unchanged by this ADR; `WatchSpec` is explicitly out of scope (§I/O contract) |
| pack manifest schema | `DomainPack`'s on-disk/wire shape | `v1` (prospective — `DomainPack` does not exist in code yet) | set by the `DomainPack` implementation |
| lock schema | `OptimizationLock`'s on-disk/wire shape | `v1` (prospective — `OptimizationLock` does not exist in code yet) | set by the `OptimizationLock` implementation |
| tunables artifact schema | `tunables.json`'s `schema_version` | `1` | bump on any change to the `TunableSpec` shape itself |

## Invariants

Restated here because a configuration architecture that could violate any of these would be a
regression, whatever it resolves to:

- **One Gate.** `percept.gate.Gate` is the only firing authority; no configuration object may carry
  gate semantics (`SourceProfile.FORBIDDEN_FIELDS` is the existing precedent; the same set is
  forbidden on every one of the six objects).
- **Closed FIRE/ASK/SILENT + `GATE_REASONS`.** The verdict vocabulary and the reason-code vocabulary
  stay closed and unextended by configuration.
- **Model id ≠ adapter identity.** `percept.provider`: a provider **name** resolves to a factory; a
  model id is configuration carried to that adapter, never a selector (`ProviderError` if a model id
  is used where a name belongs). `DeploymentConfig` may own model-id fields; it may never let a model
  id substitute for a provider name.
- **No import-time semantic config.** Constructing any of the six objects, or importing the modules
  that define them, performs no I/O, reads no environment variable as a side effect of import, and
  changes no runtime behavior. Only `ConfigResolver.resolve(...)`, called explicitly, reads
  `DeploymentConfig`'s env layer.
- **Admission ≠ extraction.** A `SourceProfile`'s `admission_strategy` decides whether a frame is
  worth judging; it is never the judge itself, and no configuration object may collapse the two
  (ADR-005 §Decision 1).
- **Templates never auto-arm.** A `DomainPack` template is a suggestion surfaced to the author; no
  configuration resolution path may take a `DomainPack` template and register it as an active goal
  without an explicit `WatchPlan` selection.
- **The optimizer tunes only declared-safe paths.** An `OptimizationLock` may only cover fields
  `tunables.json` marks `safe_to_optimize: true`; a lock covering any other field is invalid to
  construct.

## Consequences

**Good**

- `tunables.json` gives every subsequent card a single, CI-checked place to look up "does
  this field exist, who owns it, is it safe to automate" instead of grepping six modules.
- Ownership-gated precedence makes "who set this" answerable in code, not folklore — the same
  upgrade ADR-005 made for gate semantics, applied to configuration broadly.
- No existing frozen surface moves: `WatchSpec`, `WATCH_VERSION`, `CONTRACT_VERSION`, and the
  `SourceProfile`/`WatchProfile` dataclasses are read, extended in ownership terms, and left
  otherwise untouched.

**Costs / accepted risks**

- Four of the six objects, `ConfigResolver`, and `RuntimeCompiler` are follow-on implementations of this
  contract; until then this ADR is a binding target, not a description of running code, and a reader
  of `percept/` alone will not find `DomainPack`/`ExecutionPolicy`/`DeploymentConfig`/`WatchPlan`/
  `OptimizationLock`.
- The "coverage floor" field's owning object is not fixed by this ADR (see §Corrections(b)) —
  left as an open question for the resolver implementation rather than guessed here.
- `Settings`/`load_settings()` continues to play `DeploymentConfig`'s role informally until that
  object ships; this ADR does not require an immediate migration, only that new deployment-shaped
  fields be added with this ownership model in mind.

## Open questions

1. **Which object owns a coverage-floor field?** `SourceProfile` (per-source-class default) and
   `ExecutionPolicy` (per-deployment posture) are both plausible owners; `tunables.json` marks this
   knob `owner: "unclassified"` pending an explicit ownership decision.
2. **Does `ConfigResolver` construct once per `watch()` call or once per process?** A per-call
   resolve keeps provenance fresh against a changing `DeploymentConfig`/env; a per-process resolve is
   cheaper. Deferred until `ConfigResolver` is implemented.
3. **How does an `OptimizationLock` hash mismatch surface to the caller** — an exception, a
   `WatchEvent`-adjacent diagnostic, or a `percept doctor` finding? `percept/doctor.py`'s findings
   vocabulary is the likely home (consistent with `ADMISSION_SIGNAL_UNTUNED`), but not fixed here.
4. **Does `DomainPack` recommend a `(SourceProfile, WatchProfile)` pair, or select one outright?**
   Recommending preserves the "templates never auto-arm" invariant; selecting outright would need an
   explicit `WatchPlan` acknowledgement to stay consistent with it. Deferred to the `DomainPack` card.
