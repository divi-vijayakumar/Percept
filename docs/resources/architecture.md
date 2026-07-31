---
title: Architecture: the four tiers and the two-layer brain
updated: 2026-07-23
applies_to: percept-vision 1.0.0
status: stable
---
# Architecture: the four tiers and the two-layer brain

Percept turns a goal stated in plain language — *"tell me when the kettle boils"* — into a proactive
agent over a live audio-visual stream. This page is the map: how signal flows up through four tiers, how
a two-layer brain keeps the common case cheap, and where each concern lives in code.

`pip install percept-vision`; `import percept`; Python >= 3.10. The core is pure standard library and
runs offline with deterministic fake backends (`mode="offline"`) — no keys required. The project site
is <https://percept.percia.ai>.

![Percept architecture](../architecture.svg)

> The diagram is **generated from the code** by
> [`tools/build_architecture.py`](../../tools/build_architecture.py). It is a build artifact, never
> hand-edited: a stale code-pointer or a claim that disagrees with the source fails the build, so the
> picture always matches what actually ships.

## The four tiers — signal flows up

Raw signal enters at the bottom and rises through four tiers; a compiled watch plan flows down. Each
tier does strictly less work than the one above it and hands up a smaller, more structured fact.

| tier | role | package |
|---|---|---|
| **Feeds** — devices | capture a source (camera · RTSP · video · mic · screen) into frames and audio | `percept-edge` |
| **Measures** — encoders | cheap high-rate detectors, pose channels, and the tier-0 motion/salience gate that decides *whether a VLM call is worth spending* | [`percept.harness`](../../packages/percept-vision/percept/harness) |
| **Cognition** — System 1 | senses, the three-state gate, routing, cache, entity graph; cheap, deterministic, always in the loop | [`percept`](../../packages/percept-vision/percept) |
| **VLM** — System 2 | general vision-language intelligence summoned behind vendor-neutral registry seams, paid per call | [`percept.backends`](../../packages/percept-vision/percept/backends) |

- **Feeds** own the `run(source) -> incidents` loop and turn a device into frames + audio.
- **Measures** turn raw signal into channels and timed boundaries and answer one cheap question per
  frame: *is anything worth a closer look?* See [Motion score and the temporal strip](../admission/how-admission-works.md).
- **Cognition (System 1)** is the brain. It **never sees pixels** — it reasons over a stream of
  `Verdict{satisfied, confidence, evidence}` and cheap tier-0 signals. It is stdlib-pure, deterministic,
  and always running.
- **VLM (System 2)** is a stateless per-call evaluator. `vision.judge(condition, frame) -> Verdict`
  answers "is this plain-language condition true of this frame, right now?" and nothing more. It is the
  only tier that costs money, and the brain summons it only when the cheaper tiers cannot resolve a goal.

The split is the whole design: a cheap, stateful **System 1** that reasons over small structured facts,
and an expensive, stateless **System 2** it calls only when it must.

## The per-decision lifecycle

Every frame (or utterance) flows through the same cost-avoiding, refuse-to-guess pipeline **in every
shipped configuration** — the four stages below, not five. The model is called only when the cheaper
tiers cannot resolve the goal first.

```
sample ──► route ──► judge ──► gate ──► deliver
 fovea    regex     authorize   FIRE     incident
 density  fast-path  → bind →   ASK      to sink
                     dispatch   SILENT
```

- **sample** — the fovea sampler ([`sampling.py`](../../packages/percept-vision/percept/sampling.py))
  coasts at a quiet floor and escalates toward a cost ceiling only while cheap motion is high. A still
  scene spends nothing.
- **route** — the local fast-path ([`routing/`](../../packages/percept-vision/percept/routing)) resolves
  a confident spoken turn with a regex lexicon and defers anything ambiguous or goal-authoring to the
  model. Higher threshold = quieter router.
- **judge** — a judge is never taken on trust. A `WatchPlan.judge` value is a *request*; the runtime
  **authorizes** it against a deployment-owned catalog, **binds** an immutable static context verified
  byte-for-byte against a pinned SHA-256, and only then **dispatches**. Every step is fail-closed: any
  mismatch refuses the arm rather than silently falling back to a generalist model. See
  [Judging, and creating a custom judge](../judges/custom-specialist-judge.md).
- **gate** — the three-state consequence gate
  ([`gate.py`](../../packages/percept-vision/percept/gate.py)) maps the verdict to **FIRE** (known true,
  rising edge), **ASK** (unknown), or **SILENT** (not true, or a duplicate). It fires once when a
  condition becomes true and re-arms on a confident-false. See
  [../reference/consequence-gate.md](../watches/the-gate.md).
- **deliver** — a FIRE or ASK becomes an `Incident` delivered to a sink; evidence is retained behind an
  opaque handle, the episode is updated, and a trace is recorded.

**Cache is shipped code, not a shipped stage.** The frame-hash cascade
([`cache/`](../../packages/percept-vision/percept/cache)) can splice a fifth stage in between **route**
and **judge** — reusing a prior judgment when a near-identical ROI recurs within a bounded time window, at
a decayed confidence — but `CacheSettings.enabled` defaults to `False`
([`config.py`](../../packages/percept-vision/percept/config.py)), and no shipped profile flips it on. An
unenabled cache is constructed empty and its lookup always defers, so it is a real, tested, opt-in
component with zero shipped consumers today — it is not part of the lifecycle above until a deployment
explicitly turns it on.

The wire contract crossing every seam is versioned: `CONTRACT_VERSION = "1.10.0"`, with `watch()` at
`WATCH_VERSION = "1.1.0"` and conformance at `CONFORMANCE_VERSION = "1.1.0"`.

## Why not a raw VLM-in-a-loop?

The obvious baseline — send every frame to a VLM and act on the answer — collapses System 1 and System 2
into one expensive call per frame. It pays for perception even when nothing has changed, has no memory of
what it already reported (so it re-alerts on every positive frame), and has no principled way to abstain
when it is unsure.

Percept keeps the two layers apart. The cheap, stateful brain absorbs the common case — a still scene, a
confident local route, a cached verdict, a duplicate the gate suppresses — so a paid model call happens
only on a genuine, novel, unresolved change. The same gate that decides *when* to fire also decides
*when to ask instead of guess*.

## Fourteen modules across five layers

Each module below links to the file or package directory it lives in, under
[`packages/percept-vision/percept/`](../../packages/percept-vision/percept). At a high level they
group into five layers:

- **Foundation** — [Contracts](../../packages/percept-vision/percept/contracts) (the frozen shared
  vocabulary: `Verdict`, `Decision`, `Goal`, `Incident`, the component `Protocol`s,
  `CONTRACT_VERSION`) and
  [Profiles & Configuration](../../packages/percept-vision/percept/config.py) (the env-driven
  settings tree and layered resolver).
- **Planning** — [WatchPlan & Goals](../../packages/percept-vision/percept/watch.py) (the frozen
  `watch()` seam, natural-language goal desugaring, plan compilation, prepared apply with cache
  preflight and bounded rollback of goal/binding/ledger state; cache eviction and revision
  notifications are not transactional).
- **Cognition core** — [Core & Agent](../../packages/percept-vision/percept/agent.py),
  [Routing](../../packages/percept-vision/percept/routing),
  [Cache](../../packages/percept-vision/percept/cache),
  [Judges](../../packages/percept-vision/percept/judge_registry.py),
  [Backends](../../packages/percept-vision/percept/backends),
  [Sensing & Signals](../../packages/percept-vision/percept/sampling.py), and
  [Episodes & Evidence](../../packages/percept-vision/percept/episodes.py).
- **Runtime** — [Runtime](../../packages/percept-vision/percept/runtime) (the `run` /
  `RuntimeSession` front door: open a source, drive the core, pace judge calls, deliver to safe
  sinks).
- **Quality & tooling** — [Observability](../../packages/percept-vision/percept/observability.py),
  [Evaluation](../../packages/percept-vision/percept/eval), and
  [Harness](../../packages/percept-vision/percept/harness).

OSS snapshots persist plain-goal cognition only. A core with an applied compiled Watch refuses
`snapshot()`/`save_state()` with `CompiledWatchSnapshotUnsupportedError` rather than writing an
artifact that cannot reconstruct its runtime, source-profile, judge, and episode state. Compiled Watch
restart recovery is therefore unsupported in the OSS 1.0 runtime; durable recovery belongs to the
hosted runtime contract.

## Known limitation: bounded cost means bounded throughput, not a dollar ceiling

"Bounded cost" throughout this page and the rest of Percept's docs describes a **rate** bound, not a
**spend** bound, and the two are easy to conflate. What is actually enforced, everywhere in the runtime:

- **`SensePolicy.max_fps`** (default `15.0`, [`sampling.py`](../../packages/percept-vision/percept/sampling.py))
  is the one real ceiling the loop obeys: the densest the fovea will ever ask a judge to run, per
  concern. This is what "bounded cost" cashes out to in code — a call-rate ceiling.
- **`percept.slo.DOLLARS_PER_STREAM_HOUR_CEILING_USD`** (`5.0`) is a pre-registered **measurement**
  threshold a benchmark or CI run is checked against after the fact (`percept eval`, `TraceSummary.
  dollars_per_stream_hour`) — it grades a completed run, it does not gate a live one.

There is no `max_usd_per_hour` (or equivalent) setting anywhere in `percept.config.Settings` or in the
runtime worker. Nothing in a live stream reads a dollar figure and refuses to place another judge call
once crossed — the runtime will keep calling the vision model at whatever the rate ceiling and the
active goal count allow, for as long as the stream runs, regardless of accumulated spend. A deployment's
real dollar exposure is `(call-rate ceiling) x (price per call) x (hours run)`, and bounding it in
dollars — not just in calls — is the operator's responsibility today: watch `TraceSummary.cost_usd` /
`dollars_per_stream_hour` from a running trace and stop the stream yourself, or enforce a cap at the
provider/billing layer. This is a stated boundary of the OSS 1.0 runtime, not a bug to route around.

## See also

- [Core concepts](how-it-works.md) — the cognition/perception split, the gate, rising-edge firing, memory.
- [End to end: one judged request](worked-example-milk-boilover.md) — a single goal traced through the pipeline.
- [Motion score and the temporal strip](../admission/how-admission-works.md) — the tier-0 wake layer.
- [Backends](../judges/backends.md) · [Edge policy](../admission/wake-policy.md) ·
  [Configuration](../configuration/overview.md) · [Watch contract](../watches/watch-contract.md)
- [README](../../README.md) · project site <https://percept.percia.ai>
