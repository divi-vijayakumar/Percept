# Percept documentation

Organized by [Diátaxis](https://diataxis.fr/): **tutorials** teach, **how-to** guides solve a task,
**reference** describes the surface, **explanation** builds understanding. Read online at
**[percept.percia.ai](https://percept.percia.ai)**.

The architecture diagram is **generated from the code** ([`architecture.svg`](architecture.svg) ·
interactive [`architecture.html`](../architecture.html)) — it cannot drift from what ships. The deep
per-module reference is the generated [CodeWiki](../codewiki/) (source of truth after the code itself).

## Tutorials — learning-oriented
- [Choose your path](tutorials/choose-your-path.md) — the narrow first path: pick one row, ignore the rest.
- [Quickstart](tutorials/quickstart.md) — install, run offline on the fake backend, then a real backend.

## How-to — task-oriented
- [Runtime](how-to/runtime.md) — drive a live source with `run(source) → incidents`, and build a custom run.
- [Authoring goals and watch plans](how-to/authoring-goals-and-watch-plans.md) — phrase goals and match the plan to the action so quality isn't lost before the judge runs.
- [Create a custom configuration](how-to/custom-configuration.md) — author a `SourceProfile` / `WatchProfile` for your use case.
- [Custom judge & backend](how-to/custom-judge.md) — bring your own vision model or author a specialist judge; the conformance suite.
- [Tune admission](how-to/tune-admission.md) — from "nothing fires" to a tuned eye: read a `percept-doctor` report correctly, name your source class, and plug in your own change signal. **Percept ships the harness; you tune the eye for your content.**
- [Run an optimization](how-to/optimize.md) — `percept-optimize` (safe-tunable search → `OptimizationLock`) and `percept-derive-profile`.
- [Evaluate](how-to/eval.md) — per-concern precision/recall on a manifest, the paired A/B differential, and the CI gate.
- [Derive a profile](how-to/derive-a-profile.md) — turn labelled fixtures + a ladder into a provenance-stamped profile.
- [Run benchmarks](how-to/benchmarks.md) — the scorecard, the audit gate, and the reproducible model-sweep runbook.
- [Porch-package scenario](how-to/porch-package.md) — a discrete physical-event watch recipe.
- [Screenshare scenario](how-to/screenshare.md) — a fixed-cadence content watch.
- [Voice and watchdog](how-to/voice-and-watchdog.md) — the voice-escalation seam and long-running watches.
- [Trace bundles and replay](how-to/trace-bundles-and-replay.md) — portable bug reports + deterministic replay.
- [Release readiness](how-to/release-readiness.md) — the pre-publish checklist, audited against the built artifact.

## Reference — information-oriented
- [Configuration reference](configuration.md) — **everything you can tune**: the `Settings` tree, every env var, `SourceProfile` vs `WatchProfile`, and the resolver precedence.
- [Command-line reference](reference/cli.md) — all nine console scripts, their flags, and install requirements.
- [Scope and persistence](reference/v1-4-scope-and-persistence.md) — **read first if you are designing around percept**: the subject kinds covered, what is excluded, why episodes do not survive a restart, and what `scope='durable'` does and does not mean.
- [`watch()` contract](reference/watch-contract.md) — the frozen public entrypoint + `WatchEvent` schema.
- [Consequence gate](reference/consequence-gate.md) — FIRE/ASK/SILENT, the rising edge, refractory, and the typed reason vocabulary.
- [Backends](reference/backends.md) — the seams, backend names, extras, custom backends.
- [Vision adapter interface](reference/vision-adapter.md) — the public extension point: bring your own model, fail-closed injection, and the runnable conformance suite.
- [Registration](reference/registration.md) — detector / concern / enricher registries.
- [Wake policy](reference/edge-policy.md) — the on-device "should we spend a VLM call?" layer.
- [Edge parity](reference/edge-parity.md) — JS ↔ Python edge conformance.
- [Sustained-duration Watches](reference/sustained-duration-watches.md) — Python support boundary,
  corrected door semantics, migration, and restart behavior.
- [Audio modality](reference/audio-modality.md) — fused vision + audio judging.
- [Command seam](reference/command-seam.md) — drive a running watch: arm by voice, ask, pause/mute.
- [Operating frontier](reference/operating-frontier.md) — the recall / false-fire trade-off surface.
- [Eval hours costing](reference/eval-hours-costing.md) — sizing a live evaluation run.
- [Benchmark methodology](reference/benchmark-methodology.md) — how the benchmark is designed (no numbers).
- [Publishable benchmark](reference/publishable-benchmark.md) — the runbook + external reference bars.

## Explanation — understanding-oriented
- [Architecture](explanation/architecture.md) — the four tiers, two-layer brain, the firing paths.
- [Concepts](explanation/concepts.md) — the cognition / perception split, the gate, rising-edge firing, memory.
- [E2E flow: milk boil-over](explanation/e2e-flow-milk-boilover.md) — a judged path, end to end.
- [Motion and strip](explanation/motion-and-strip.md) — the motion score and temporal strip.

> **Benchmark numbers** are intentionally absent from these docs until the model-sweep phase publishes a
> reproducible scorecard. Methodology and external reference bars are documented; Percept's own accuracy
> numbers are not claimed yet.
