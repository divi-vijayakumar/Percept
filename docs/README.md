---
title: Percept documentation
updated: 2026-07-28
applies_to: percept-vision 1.0.0
status: stable
generated_by: tools/build_docs_index.py
---

# Percept documentation

Everything you need to adopt Percept, for humans and for agents. Read online at
**[percept.percia.ai](https://percept.percia.ai)**.

**This index is generated** by `tools/build_docs_index.py` from each page's frontmatter, so
it cannot list a page that doesn't exist or miss one that does. Every page carries an
`updated:` date — a page older than the release it documents is visibly stale.

New here? → **[Quickstart](tutorials/quickstart.md)** · want the whole surface? →
**[Tunables](reference/tunables.md)** · using an agent? → **[Coding agents](agents/README.md)**

The architecture diagram is **generated from the code**
([`architecture.svg`](architecture.svg) · interactive [`architecture.html`](architecture.html))
— it cannot drift from what ships.

## Tutorials — start here

*Learning-oriented. Run something that works, then understand it.*

| page | what it covers | updated |
|---|---|---|
| [Quickstart](tutorials/quickstart.md) | Install, run offline on the scripted backend, then a real one. | 2026-07-28 |

## How-to — do a specific task

*Task-oriented. You know what you want; these get you there.*

| page | what it covers | updated |
|---|---|---|
| [Authoring goals and watch plans](how-to/authoring-goals-and-watch-plans.md) | Phrase goals and match the plan to the action, so quality isn't lost before the judge runs. | 2026-07-17 |
| [Create a custom configuration for your use case](how-to/custom-configuration.md) | Author a `SourceProfile` / `WatchProfile` for your use case. | 2026-07-23 |
| [Judging, and creating a custom judge or specialist](how-to/custom-judge.md) | Bring your own vision model or author a specialist judge; the conformance suite. | 2026-07-23 |
| [Derive a profile for a new environment](how-to/derive-a-profile.md) | Turn labelled fixtures + a ladder into a provenance-stamped profile. | 2026-07-23 |
| [Evaluating Percept on your own benchmark](how-to/eval.md) | Per-concern precision/recall on your manifest, the paired A/B differential, and the CI gate. | 2026-07-23 |
| [Run an optimization](how-to/optimize.md) | `percept-optimize` — safe-tunable search into a validated `OptimizationLock`. | 2026-07-23 |
| [Scenario: tell me when a package is on the porch](how-to/porch-package.md) | A discrete physical-event watch recipe. | 2026-07-20 |
| [The runtime worker — `run(source) → incidents`](how-to/runtime.md) | **Start here to embed Percept.** `run(source) → incidents`: sources, sinks, tuning, egomotion, cost, troubleshooting. | 2026-07-28 |
| [Scenario: help me by watching my screenshare](how-to/screenshare.md) | A fixed-cadence content watch. | 2026-07-16 |
| [Trace bundles & replay](how-to/trace-bundles-and-replay.md) | Portable bug reports and deterministic replay. | 2026-07-23 |
| [Tuning admission: from "nothing fires" to a tuned eye](how-to/tune-admission.md) | From "nothing fires" to a tuned eye. **Percept ships the harness; you tune the eye for your content.** | 2026-07-23 |
| [Two apps on one command seam — a silent watchdog and a voice assistant](how-to/voice-and-watchdog.md) | The voice-escalation seam and long-running watches. | 2026-07-16 |

## Reference — the surface

*Information-oriented. What exists, what it's called, what it defaults to.*

| page | what it covers | updated |
|---|---|---|
| [Audio modality — the acoustic-onset → judge wake path](reference/audio-modality.md) | Fused vision + audio judging. | 2026-07-23 |
| [Backends](reference/backends.md) | The seams, backend names, extras, and custom backends. | 2026-07-28 |
| [Command-line reference](reference/cli.md) | All nine console scripts, their flags, and install requirements. | 2026-07-23 |
| [The command seam — `RuntimeSession`](reference/command-seam.md) | Drive a running watch — arm by voice, ask, pause/mute. | 2026-07-16 |
| [Configuration reference — everything you can tune](reference/configuration.md) | **Everything you can tune** — the `Settings` tree, every env var, and resolver precedence. | 2026-07-28 |
| [The consequence gate — the stable decision surface](reference/consequence-gate.md) | FIRE/ASK/SILENT, the rising edge, refractory, and the typed reason vocabulary. | 2026-07-23 |
| [Edge parity](reference/edge-parity.md) | JS ↔ Python edge conformance. | 2026-07-16 |
| [Edge Wake Policy](reference/edge-policy.md) | The on-device "should we spend a VLM call?" layer. | 2026-07-23 |
| [Eval-Hours Costing for the Operating-Point Bars](reference/eval-hours-costing.md) | Sizing a live evaluation run before you spend on it. | 2026-07-23 |
| [The operating-frontier report](reference/operating-frontier.md) | The recall / false-fire trade-off surface. | 2026-07-20 |
| [Registration](reference/registration.md) | Detector / concern / enricher registration in detail. | 2026-07-20 |
| [Registries and adapters](reference/registries.md) | **Backend seams, detectors, and profile registries** — and how to add your own. Generated. | 2026-07-28 |
| [Scope and persistence (percept 1.0.0)](reference/scope-and-persistence.md) | **Read first if you are designing around Percept** — what's covered, what's excluded, why episodes don't survive a restart. | 2026-07-23 |
| [Tunables](reference/tunables.md) | **All 157 tunables, and the 31 an optimizer may search.** Generated from the registry. | 2026-07-28 |
| [The Vision adapter interface (public extension point)](reference/vision-adapter.md) | The public extension point — bring your own model, fail-closed injection, conformance suite. | 2026-07-23 |
| [`watch()` — the frozen public contract](reference/watch-contract.md) | The frozen public entrypoint and the `WatchEvent` schema. | 2026-07-23 |

## Benchmarks — how it's measured

*How the benchmark is designed, how to run it, and what may honestly be claimed.*

| page | what it covers | updated |
|---|---|---|
| [Benchmarks](benchmarks/README.md) | **Start here** — status, how to run one, and how to read a precision number without being misled. | 2026-07-28 |
| [Percept Benchmark Methodology Spec](benchmarks/methodology.md) | How the benchmark is designed (no numbers). | 2026-07-28 |
| [Run benchmarks](benchmarks/running-benchmarks.md) | The scorecard, the audit gate, and the reproducible model-sweep runbook. | 2026-07-28 |

## Explanation — why it works this way

*Understanding-oriented. The design and the reasoning behind it.*

| page | what it covers | updated |
|---|---|---|
| [Architecture: the four tiers and the two-layer brain](explanation/architecture.md) | The four tiers, the two-layer brain, the firing paths. | 2026-07-23 |
| [Core concepts: the cognition/perception split, the gate, rising-edge firing, memory](explanation/concepts.md) | The cognition / perception split, the gate, rising-edge firing, memory. | 2026-07-23 |
| [End to end: one judged request (milk boiling over)](explanation/e2e-flow-milk-boilover.md) | A judged path, end to end. | 2026-07-23 |
| [Motion score and the temporal strip](explanation/motion-and-strip.md) | The motion score and the temporal strip. | 2026-07-23 |

## Coding agents — onboarding

*Adopt Percept from Claude Code, Codex, Cursor, or Copilot.*

| page | what it covers | updated |
|---|---|---|
| [Coding agents](agents/README.md) | Install the `percept-agent` skill; the three things agents get wrong. | 2026-07-28 |

## Demo — see it run

*A browser harness driving the real runtime.*

| page | what it covers | updated |
|---|---|---|
| [Percept demo harness](demo/README.md) | Browser demos driving the **real** runtime — upload a video or watch a live stream. | 2026-07-28 |

---

## Not in these docs, on purpose

- **Architecture decision records** live in [`.adr/`](../.adr/) — tracked and citable from
  code, but internal reasoning rather than adoption material.
- **Benchmark numbers** are not published yet. See [Benchmarks](benchmarks/README.md) for
  the status and the gate a published claim must pass.
- **Per-module API reference** is the generated CodeWiki, regenerated from source.
