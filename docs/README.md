---
title: Percept documentation
updated: 2026-07-30
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

New here? → **[Quickstart](get-started/quickstart.md)** · want the whole surface? →
**[Tunables](configuration/tunables.md)** · using an agent? → **[Coding agents](get-started/coding-agents.md)**

The architecture diagram is **generated from the code**
([`architecture.svg`](architecture.svg) · interactive [`architecture.html`](architecture.html))
— it cannot drift from what ships.

## Get started

*Install it, run something that works, then build.*

| page | what it covers | updated |
|---|---|---|
| [Coding agents](get-started/coding-agents.md) | Install the `percept-agent` skill; the three things agents get wrong. | 2026-07-28 |
| [Quickstart](get-started/quickstart.md) | Install, run offline on the scripted backend, then a real one. | 2026-07-28 |

## Watches

*The core object: a goal armed against a source, firing on the rising edge.*

| page | what it covers | updated |
|---|---|---|
| [The runtime worker — `run(source) → incidents`](watches/creating-a-watch.md) | **Start here to embed Percept.** `run(source) → incidents`: sources, sinks, tuning, egomotion, cost, troubleshooting. | 2026-07-28 |
| [Official free DomainPacks](watches/domain-packs.md) | DomainPacks are versioned collections of Watch templates and advisory component recommendations. | 2026-07-28 |
| [The command seam — `RuntimeSession`](watches/driving-a-live-watch.md) | Drive a running watch — arm by voice, ask, pause/mute. | 2026-07-16 |
| [Scenario: tell me when a package is on the porch](watches/scenario-porch-package.md) | A discrete physical-event watch recipe. | 2026-07-20 |
| [Scenario: help me by watching my screenshare](watches/scenario-screenshare.md) | A fixed-cadence content watch. | 2026-07-16 |
| [Sustained-duration Watches: 1.0 release contract](watches/sustained-duration-watches.md) | A sustained-duration Watch fires only after a condition has remained continuously supported for an. | 2026-07-28 |
| [The consequence gate — the stable decision surface](watches/the-gate.md) | FIRE/ASK/SILENT, the rising edge, refractory, and the typed reason vocabulary. | 2026-07-23 |
| [`watch()` — the frozen public contract](watches/watch-contract.md) | The frozen public entrypoint and the `WatchEvent` schema. | 2026-07-23 |
| [Authoring goals and watch plans](watches/writing-goals.md) | Phrase goals and match the plan to the action, so quality isn't lost before the judge runs. | 2026-07-17 |

## Sources

*What Percept watches — cameras, files, streams, screens, audio.*

| page | what it covers | updated |
|---|---|---|
| [Audio modality — the acoustic-onset → judge wake path](sources/audio.md) | Fused vision + audio judging. | 2026-07-23 |
| [Derive a profile for a new environment](sources/source-profiles.md) | Turn labelled fixtures + a ladder into a provenance-stamped profile. | 2026-07-23 |
| [Two apps on one command seam — a silent watchdog and a voice assistant](sources/voice-and-watchdog.md) | The voice-escalation seam and long-running watches. | 2026-07-16 |

## Admission — the eye

*Everything upstream of a model call. This is what makes Percept cheap.*

| page | what it covers | updated |
|---|---|---|
| [Registration](admission/detectors.md) | Detector / concern / enricher registration in detail. | 2026-07-20 |
| [Edge parity](admission/edge-parity.md) | JS ↔ Python edge conformance. | 2026-07-16 |
| [Motion score and the temporal strip](admission/how-admission-works.md) | The motion score and the temporal strip. | 2026-07-23 |
| [Tuning admission: from "nothing fires" to a tuned eye](admission/tuning-admission.md) | From "nothing fires" to a tuned eye. **Percept ships the harness; you tune the eye for your content.** | 2026-07-23 |
| [Edge Wake Policy](admission/wake-policy.md) | The on-device "should we spend a VLM call?" layer. | 2026-07-23 |

## Judges

*The models that answer "is this condition true?" — and how to bring your own.*

| page | what it covers | updated |
|---|---|---|
| [Backends](judges/backends.md) | The seams, backend names, extras, and custom backends. | 2026-07-28 |
| [The Vision adapter interface (public extension point)](judges/bring-your-own-model.md) | The public extension point — bring your own model, fail-closed injection, conformance suite. | 2026-07-23 |
| [Judging, and creating a custom judge or specialist](judges/custom-specialist-judge.md) | Bring your own vision model or author a specialist judge; the conformance suite. | 2026-07-23 |
| [Hosted BYOK invocation contract](judges/hosted-byok.md) | Per-invocation provider credentials for a multi-tenant hosted service — routing, isolation, cancellation, deadlines, and the gated detection route. | 2026-07-26 |
| [Registries and adapters](judges/registries.md) | **Backend seams, detectors, and profile registries** — and how to add your own. Generated. | 2026-07-28 |

## Incidents and delivery

*What comes out, and how it reaches your app.*

| page | what it covers | updated |
|---|---|---|
| [Scope and persistence (percept 1.0.0)](incidents/scope-and-persistence.md) | **Read first if you are designing around Percept** — what's covered, what's excluded, why episodes don't survive a restart. | 2026-07-23 |

## Configuration

*Every knob, who owns it, and how to tune it safely.*

| page | what it covers | updated |
|---|---|---|
| [Episodes — one occurrence, one notification](configuration/episodes.md) | A Watch that fires every time a condition is *still* true is unusable. | 2026-07-29 |
| [Run an optimization](configuration/optimize.md) | `percept-optimize` — safe-tunable search into a validated `OptimizationLock`. | 2026-07-23 |
| [Configuration reference — everything you can tune](configuration/overview.md) | **Everything you can tune** — the `Settings` tree, every env var, and resolver precedence. | 2026-07-28 |
| [Create a custom configuration for your use case](configuration/profiles.md) | Author a `SourceProfile` / `WatchProfile` for your use case. | 2026-07-23 |
| [Tunables](configuration/tunables.md) | **All 157 tunables, and the 31 an optimizer may search.** Generated from the registry. | 2026-07-30 |

## Observability

*Traces, replay, cost, and the quality frontier.*

| page | what it covers | updated |
|---|---|---|
| [The operating-frontier report](observability/operating-frontier.md) | The recall / false-fire trade-off surface. | 2026-07-20 |
| [Trace bundles & replay](observability/traces-and-replay.md) | Portable bug reports and deterministic replay. | 2026-07-23 |

## Evaluation and benchmarks

*Measure it on your data, and read the numbers honestly.*

| page | what it covers | updated |
|---|---|---|
| [Benchmarks](evaluation/benchmarks.md) | **Start here** — status, how to run one, and how to read a precision number without being misled. | 2026-07-28 |
| [Evaluating Percept on your own benchmark](evaluation/evaluate-your-data.md) | Per-concern precision/recall on your manifest, the paired A/B differential, and the CI gate. | 2026-07-23 |
| [Percept Benchmark Methodology Spec](evaluation/methodology.md) | How the benchmark is designed (no numbers). | 2026-07-28 |
| [Run benchmarks](evaluation/running-benchmarks.md) | The scorecard, the audit gate, and the reproducible model-sweep runbook. | 2026-07-28 |
| [Eval-Hours Costing for the Operating-Point Bars](evaluation/sizing-a-run.md) | Sizing a live evaluation run before you spend on it. | 2026-07-23 |

## Resources

*How it works, the architecture, and the full CLI.*

| page | what it covers | updated |
|---|---|---|
| [Architecture: the four tiers and the two-layer brain](resources/architecture.md) | The four tiers, the two-layer brain, the firing paths. | 2026-07-23 |
| [Command-line reference](resources/cli.md) | All nine console scripts, their flags, and install requirements. | 2026-07-23 |
| [Core concepts: the cognition/perception split, the gate, rising-edge firing, memory](resources/how-it-works.md) | The cognition / perception split, the gate, rising-edge firing, memory. | 2026-07-23 |
| [End to end: one judged request (milk boiling over)](resources/worked-example-milk-boilover.md) | A judged path, end to end. | 2026-07-23 |

## Demo

*A browser harness driving the real runtime.*

| page | what it covers | updated |
|---|---|---|
| [Percept demo harness](demo/README.md) | Browser demos driving the **real** runtime — upload a video or watch a live stream. | 2026-07-28 |

---

## Not in these docs, on purpose

- **Architecture decision records** live in [`.adr/`](../.adr/) — tracked and citable from
  code, but internal reasoning rather than adoption material.
- **Benchmark numbers** are not published yet. See [Benchmarks](evaluation/benchmarks.md) for
  the status and the gate a published claim must pass.
- **Per-module API reference** is the generated CodeWiki, regenerated from source.
