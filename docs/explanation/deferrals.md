# Deferrals — corrected rationales

This page is an honesty pass on what Percept v0.1 does **not** ship, and why. Earlier deferral notes in
internal planning docs were sometimes wrong or overstated ("never," "not needed") in ways that don't
survive contact with what the runtime actually does. This page replaces those with the corrected
reasoning and — for every deferred item — the concrete condition that would promote it back into scope.

A deferral without a promotion trigger is just a reason not to look at something again. Every row below
has one.

## Judgment event bus / pub-sub

**Deferred.** The iterator + sinks pattern already gives one-producer/N-consumer delivery with
`event_id` idempotency. A bus would add its own start/subscription-race handling and a second drop point
that the delivery ledger can't see — i.e. it reintroduces the exact delivery-loss failure mode one layer
inward, not a net improvement.

**Promotion trigger:** the first consumer that must attach *after* emission (late subscriber needing
replay-from-offset), or a move to multi-process delivery.

## Capability manifests / registry

**Deferred.** Nothing in v0.1 reads latency, cost, or placement metadata — there is no consumer for a
manifest yet, so building one is speculative.

**Promotion trigger:** an optimizer or control-plane routing layer that actually needs to select among
backends by capability.

## FAST/RICH cascade

**Deferred — corrected framing.** The earlier framing conflated two different questions. **T0** (the
tier-0 edge/wake layer) asks *should semantic cognition run at all* — that's already built. A **cascade**
asks a different question: *which semantic evidence source is reliable enough to trust for this
judgment*. Those are not the same mechanism, and describing the cascade as an extension of T0 was the
error being corrected here.

The cascade itself is deferred because v0.1 has no demonstrated cost/assurance requirement that justifies
running two semantic judges, and no calibration data to route between them even if it did. Adding a
second adapter also means a second exposure under the fail-closed judge-injection contract, plus a
confident-and-wrong short-circuit failure path that hasn't been designed.

**Promotion trigger:** a demonstrated cost/assurance requirement for a second semantic judge, *and*
calibration data sufficient to route between the two reliably.

## General state memory

**Softened from "never."** The Gate already holds edge state — rising-edge is "previous truth value per
key," `rearm` is a reset, refractory is a suppression window — serialized per
`(source_id, plan_id, episode_or_entity_key)`. So "no state memory" was never literally true; the binding
rule was always narrower than that.

**The actual rule:** no independent store may become a second authority for Gate truth, edge state, rearm
state, or revision validity. Auxiliary procedure/domain state *may* exist alongside the Gate if it is
revision-bound, carried explicitly as evidence/context (not read back as ambient truth), and structurally
incapable of bypassing the Gate.

**Promotion trigger:** a Shape-2 `ProcedureState` design that satisfies those constraints.

## Baseline manager

**Deferred.** Learned suppression makes the system quieter under a bug, not louder — and a quiet failure
is invisible in a way a loud one isn't. Shipping this before the eval harness can catch a regression means
shipping a mechanism whose failure mode is silence.

**Promotion trigger:** replay + golden eval load-bearing in CI, then opt-in with a published suppression
report (so a regression in suppression behavior is visible, not silent).

## NL semantic goal compiler

**Deferred to H2.** v0.1's L1 string sugar is verbatim pass-through into `condition:` — zero inference,
deterministic desugaring. It is not compilation, and should not be described as if it were one.

## Domain packs

**Deferred to H2+, never clinical.** A domain pack today would be one hero demo's worth of validated
tuning, not a general capability. The eldercare / patient-monitoring pack in particular has been removed
from OSS-facing docs — it contradicted the strategy's clinical exclusion in more than one place. Domain
packs are H2+ scope, and clinical/patient-monitoring framing is out of scope permanently, not just
deferred.

## JS SDK · REST API · edge/hybrid/server modes · optimizer

**Deferred.** v0.1 is an embedded Python library only. These are separate surface-area commitments (a
second language runtime, a network API contract, deployment-topology support, and a routing/optimization
layer) that don't need to exist for the runtime and extension contract to be validated.

---

## Already-implemented surfaces must be declared, not silently deferred

A distinct problem from the deferrals above: some surfaces already exist in the codebase and are
reachable by users today — entity memory, re-identification, the edge twin. Calling these "deferred" is
false; they already shipped. "Deferred" cannot mean both "already publicly shipped" and "not in v0.1" —
that's a contradiction, not a scoping decision.

The resolution isn't silence. For each such surface the options are: remove it from the exported stable
surface, mark it **experimental** (no stability/support guarantee, excluded from the v0.1 compatibility
promise), or commit to a full support contract. The default is experimental-and-declared. See the
stability table in the root [`README.md`](../../README.md) for the current disposition of each surface.
