---
title: Benchmarks
updated: 2026-07-28
applies_to: percept-vision 1.0.0
status: stable
---
# Benchmarks

How Percept is measured, how to reproduce a measurement, and what it is honest to claim.

> **Status: no Percept accuracy numbers are published yet.**
> The benchmark is designed, the harness runs, and internal measurements exist. Publication is
> gated on [the publishing gate](#the-publishing-gate) below. Figures appearing in the methodology
> pages are **external reference bars or worked examples, not Percept claims**.

## Start here

| you want to | read |
|---|---|
| run a benchmark on your own clips | [Running benchmarks](running-benchmarks.md) |
| understand how the benchmark is designed | [Methodology](methodology.md) |
| measure *your* goals, not ours | [Evaluate](../how-to/eval.md) — per-concern precision/recall on your manifest |
| size a live run before spending on it | [Eval hours costing](../reference/eval-hours-costing.md) |

## Run one in two commands

The benchmark harness and the golden set are **repo/dev tooling, not packaged in the wheel** — the
wheel is just `import percept`. So benchmark commands run from a source checkout:

```bash
pip install -e "packages/percept-vision[bench]"

PYTHONPATH=packages/percept-vision percept-bench scorecard \
    --manifest eval/golden-v1/MANIFEST.json --out out/scorecard.txt
```

Point `--manifest` at your own labelled clips to measure your use case instead of ours.

## Reading a precision number without being misled

This is the single most important thing on this page.

**Precision is uninterpretable without ground-truth coverage.** A corpus where 85% of frames are
labelled action will hand almost any detector a high precision score; a corpus where 5% is labelled
will punish a good one. The comparable quantity is **chance-corrected lift** — precision divided by
the corpus's own GT coverage.

| corpus shape | raw precision | GT coverage | lift | honest reading |
|---|---|---|---|---|
| dense (most frames are the event) | high | high | ≈1× | near chance — the corpus did the work |
| sparse (the event is rare) | low | low | »1× | the detector did the work |

A raw precision figure quoted without its coverage is not a result. Always publish both.

## The publishing gate

A number does not become a claim by being measured. Before any accuracy figure ships:

| gate | why |
|---|---|
| `percept-bench audit --release` passes | the manifest and baseline registry are validated, not assumed |
| full-mode scorecard, **no modeled cells** | every cell came from a real run, not an extrapolation |
| content-hashed release manifest | the exact corpus is pinned, so the run is reproducible |
| ground-truth coverage published alongside precision | see the section above — precision without coverage is not a result |
| the run is reproducible from a named commit | not from someone's dirty working tree |

Single-run figures are not claims either. Recall on a fixed corpus moves run-to-run with provider
latency against the judge deadline, so a claim needs either repeated runs or a stated noise band.

## The quality rail around the benchmark

The benchmark is the outermost ring. Inside it sit cheaper, deterministic gates that catch
regressions long before a paid run:

```bash
make test              # unit + contract suites, no provider, no model
make parity            # JS ↔ Python edge conformance
make audit-benchmark   # validates the manifest + baseline registry
make scorecard         # the offline scorecard
```

`percept-bench audit --release` is the gate a published claim must pass.

## Related

- [Operating frontier](../reference/operating-frontier.md) — the recall / false-fire trade-off surface.
- [Tunables](../reference/tunables.md) — the 31 knobs an optimizer may search.
- [Run an optimization](../how-to/optimize.md) — search → validated `OptimizationLock`.
