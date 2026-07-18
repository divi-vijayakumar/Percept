# Percept documentation

Organized by [Diátaxis](https://diataxis.fr/): **tutorials** teach, **how-to** guides solve a task,
**reference** describes the surface, **explanation** builds understanding.

The architecture diagram is **generated from the code** ([`architecture.svg`](architecture.svg) ·
interactive [`architecture.html`](../architecture.html)) — it cannot drift from what ships.

## Tutorials — learning-oriented
- [Choose your path](tutorials/choose-your-path.md) — the narrow first path: pick one row, ignore the rest.
- [Quickstart](tutorials/quickstart.md) — install, run offline on the fake backend, then a real backend.

## How-to — task-oriented
- [Runtime](how-to/runtime.md) — drive a live source with `run(source) → incidents`.
- [Eval](how-to/eval.md) — the offline eval harness, manifests, and the CI gate.
- [Fall scenario](how-to/fall.md) — a high-consequence watch recipe.
- [Screenshare scenario](how-to/screenshare.md) — a fixed-cadence content watch.

## Reference — information-oriented
- [`watch()` contract](reference/watch-contract.md) — the frozen public entrypoint + `WatchEvent` schema.
- [Backends](reference/backends.md) — the five seams, backend names, extras, custom backends.
- [Registration](reference/registration.md) — detector / concern / enricher registries.
- [Wake policy](reference/edge-policy.md) — the on-device "should we spend a VLM call?" layer.
- [Edge parity](reference/edge-parity.md) — JS ↔ Python edge conformance.
- [Audio modality](reference/audio-modality.md) — fused vision + audio judging.
- [Benchmark methodology](reference/benchmark-methodology.md) — how the benchmark is designed (no numbers).
- [Publishable benchmark](reference/publishable-benchmark.md) — the runbook + external reference bars.

## Explanation — understanding-oriented
- [Architecture](explanation/architecture.md) — the four tiers, two-layer brain, the firing paths.
- [Concepts](explanation/concepts.md) — the cognition / perception split.
- [E2E flow: milk boil-over](explanation/e2e-flow-milk-boilover.md) — a judged path, end to end.
- [Motion and strip](explanation/motion-and-strip.md) — the motion score and temporal strip.

> **Benchmark numbers** are intentionally absent from these docs until the model-sweep phase publishes a
> reproducible scorecard. Methodology and external reference bars are documented; Percept's own accuracy
> numbers are not claimed yet.
