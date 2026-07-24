# ADR-009: The operational judge request — deployment-owned availability, fail-closed resolution

- **Status:** Accepted. Motivated by a pre-1.10 gap: a `WatchPlan.judge` value was accepted and
  serialized but never consumed by resolution or runtime component selection, so a requested judge
  silently had no effect. The catalog half of this ADR ships in code
  (`percept/deployment_judges.py`, `tests/test_deployment_judges.py`): the deployment-owned judge
  availability object and its loaders. The resolver reconciliation and the dispatch-side re-check
  this ADR specifies ship alongside it (`percept/judge_registry.py`, resolver and dispatch wiring).
- **Depends on:** ADR-008 (Configuration ownership, resolution, and provenance — this ADR
  materializes the first concrete slice of its `DeploymentConfig` object, previously recorded there
  as "not yet in code"); ADR-005 (Source profiles — the `FORBIDDEN_FIELDS` / refuse-at-construction
  precedent this ADR reuses for fail-closed authorization); ADR-001 (the Gate/judgment contract —
  the judge serves the Gate, it does not become a second firing authority).
- **Related:** ADR-007 (Envelope enforcement — the same "refuse in the constructor, not in a review
  checklist" posture applied to an unauthorized judge request); `percept/registry.py` (the
  name → factory seam a `factory_ref` names).
- **Source (read for this ADR):** `percept/registry.py` (`register`, the `vision` seam,
  name-resolves-to-factory), `percept/config.py` (`Settings`, `Settings.models.vision`),
  ADR-008's `DeploymentConfig` ownership row and its "model id ≠ adapter identity" invariant.

## Context

`WatchPlan.judge` was accepted into the plan surface but **never consumed**: a caller could name a
specialist judge and nothing downstream read it. Its docstring told a reader to "defer to
`DeploymentConfig`" — an object ADR-008 recorded as "not yet in code." So the field was a promise
with no keeper: an intent expressed, silently dropped.

Two failure shapes follow from a judge request nobody validates:

1. **Silent downgrade.** A caller asks for `audit_specialist_judge@0.1`; the runtime, finding no
   wiring, quietly falls back to the generalist vision model and judges anyway. The caller believes
   a specialist reviewed the frame; a generalist did. For an audit/consequence judge, that is the
   exact failure the request existed to prevent.
2. **Ambient authorization.** With no explicit availability layer, "which judges may run here" is
   folklore — whatever adapters happen to be importable — rather than a decision a deployment made
   on purpose. A judge is a capability; capabilities must be granted, not stumbled into.

This ADR makes `WatchPlan.judge` an **operational request** and gives availability an explicit,
deployment-owned home.

## Decision

**`WatchPlan.judge` is a REQUEST, not a selection.** The user states which judge they want; the
user does not decide whether it is reachable. Availability is owned by the **deployment layer** — a
judge *catalog* that is the first materialized piece of ADR-008's `DeploymentConfig`.

**The resolver reconciles request against catalog in one atomic authorization step, pre-arm.** A
`WatchPlan.judge` ref is authorized iff its exact ref is a key in the deployment catalog. Unknown,
unapproved, or unavailable → **fail closed before the watch arms**. There is no "arm now, discover
the judge is missing at first dispatch" path.

**No fallback to the generalist when a specialist was requested.** If a caller named a judge and it
is not authorized, the watch does not silently degrade to the generalist vision model — it refuses.
This is enforced in **two** places, deliberately redundant:

- at **resolve** (pre-arm): an unauthorized ref never produces a `components["judge"]`, so the
  watch cannot arm;
- at **dispatch**: even a resolved watch re-checks that the judge it is about to call is the judge
  that was authorized, so a later catalog change or a hand-constructed component cannot smuggle a
  generalist in behind the request.

A request with **no** judge (the common case) is unaffected: the generalist path is the default
*only when nothing was requested*, never as a fallback from a *failed* specific request.

## Judge identity is a separate resolved component

The resolved judge is its own component — `components["judge"]`, **additive**. It is **never folded
into `Settings.models.vision`**. Judge-ref, provider, and model stay three independent provenance
dimensions:

- **judge-ref** — the opaque request/authorization key (e.g. `audit_specialist_judge@0.1`); this
  module never parses a version out of it or orders refs (`JudgeDeploymentSpec` keys are opaque);
- **provider** — who runs the judge (`provider_ref`);
- **model** — the model id carried to that provider's adapter (`model_ref`), configuration and not a
  selector, consistent with ADR-008's "model id ≠ adapter identity" invariant.

`factory_ref` names the **vision-seam factory** (the `name` half of
`percept.registry.register("vision", name, factory)`) that implements the judge — a name that
resolves to a factory, never a model id. Collapsing any of these into `Settings.models.vision` would
destroy the ability to answer "which judge, from which provider, on which model, and who authorized
it" — the provenance this ADR exists to keep.

## Immutable judge-context binding

A judge may be armed with a **static** context blob — reference material bound at resolve time, not
retrieved live. The binding is typed: a `ref` plus a `sha256` over the exact bytes it names. The
resolver verifies the hash over those exact bytes and injects the verified blob under a **reserved
context key**; there is **no live retrieval** at dispatch. A hash mismatch fails closed like any
other unauthorized condition. `JudgeDeploymentSpec.supports_static_context` records whether a given
judge accepts such a blob (default `True`); `DeploymentJudges.context_store_root` is the optional
root a binding resolves against — data, not an import-time read.

Immutability is the point: the frame a judge saw and the reference it judged against are both fixed
and reproducible, so a judgment can be re-derived from recorded inputs. Live retrieval would make a
judgment depend on state outside the recorded envelope.

## The catalog (what ships now)

`percept/deployment_judges.py`:

- `JudgeDeploymentSpec` — frozen; `factory_ref` / `provider_ref` / `model_ref` (each a validated
  non-empty string) + `supports_static_context: bool = True`.
- `DeploymentJudges` — `available_judges: Mapping[str, JudgeDeploymentSpec]` (opaque ref keys) +
  optional `context_store_root`. The mapping is frozen at construction; `authorizes(ref)` /
  `spec_for(ref)` are the fail-closed lookups the resolver calls.
- `EMPTY_DEPLOYMENT_JUDGES` — the default catalog: **no judge authorized**.
- Loaders: `DeploymentJudges.from_file(path)` (JSON; malformed JSON / missing / extra / wrong-typed
  keys → `DeploymentJudgesError`, never a partial load) and `load_deployment_judges(env)` (reads
  `PERCEPT_JUDGE_CATALOG`; unset → `EMPTY_DEPLOYMENT_JUDGES`).

**An empty catalog means every `WatchPlan.judge` request fails closed downstream — and that is
intended.** A deployment must *explicitly authorize* a judge before the resolver will arm it.
Silence is a refusal, not an implicit "use the generalist." This is the fail-closed default, not a
degraded state.

This module is the FIRST materialized piece of ADR-008's `DeploymentConfig` — the object that ADR
described as owning "which providers are AVAILABLE and how to reach them" while noting it did "not
yet exist in code." It **supersedes the old `WatchPlan.judge` docstring meaning** ("defer to
`DeploymentConfig`"): the deferral now has a concrete target.

## Versioning

Additive across the board — no frozen shape moves:

| version | change | rule |
|---|---|---|
| `CONTRACT_VERSION` | `1.9.0` → `1.10.0` | MINOR: an additive `components["judge"]` resolution component and the judge-request semantics; no existing DTO field changes type or meaning. |
| `WATCH_VERSION` | `1.0.0` → `1.1.0` | MINOR: `WatchPlan.judge` gains operational meaning; the public `watch()` surface is extended, not broken. |

**Incident, judgment, and evidence shapes stay 1.9-compatible.** This ADR adds a resolution
component and an availability object; it does not touch the incident/judgment/evidence envelope
shapes, which remain byte-compatible with contract 1.9.

## Invariants

- **One Gate.** The judge serves the Gate; it never becomes a second firing authority (ADR-001).
- **Fail closed, pre-arm.** An unknown/unapproved/unavailable judge request refuses before the watch
  arms — never a runtime surprise, never a silent generalist substitution.
- **No generalist fallback for a specific request.** Enforced at resolve *and* dispatch.
- **Model id ≠ adapter identity.** `factory_ref` names a factory; `model_ref` is configuration
  carried to it, never a selector (ADR-008).
- **Judge-ref is opaque.** No version parsing, no ordering, no relatedness between two refs; only an
  exact-key match authorizes.
- **No import-time I/O.** Constructing a catalog or importing the module reads no file and no env as
  a side effect; only `load_deployment_judges`/`from_file`, called explicitly, touch the
  environment or disk.

## Consequences

**Good**

- A named judge is honored or refused — never silently downgraded. The provenance question "which
  judge, provider, model, and who authorized it" is answerable in code.
- Deployment owns availability explicitly; "which judges run here" is a decision, not an accident of
  which adapters import.
- Additive versioning: no existing frozen surface (`Settings.models.vision`, incident/judgment/
  evidence shapes) moves.

**Costs / accepted risks**

- The resolver reconciliation and dispatch-side re-check are specified here but ship in follow-on
  cards; until then this ADR is a binding target for those, not a description of a fully wired path.
- A deployment that forgets to configure a catalog gets zero authorized judges. That is the
  intended fail-closed default, but it will surface as "my judge request refused" for a deployment
  that expected ambient availability — correctly, and loudly.

## Open questions

1. **How does a fail-closed judge refusal surface to the caller** — a resolve-time exception, a
   `WatchEvent`-adjacent diagnostic, or a `percept doctor` finding? Deferred to the resolver card.
2. **Does the catalog resolve once per `watch()` call or once per process?** Same trade-off ADR-008
   left open for `ConfigResolver`; deferred to the same card.
