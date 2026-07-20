# Architecture

How the pieces fit. Percept turns a goal stated in plain language into a proactive agent over a live
audio-visual stream. This page is the **map**; every claim links to the code that implements it.

> The diagram on this page is **generated from the code** by [`tools/build_architecture.py`](../../tools/build_architecture.py)
> — see [`architecture.svg`](../architecture.svg) (static) or the interactive
> [`architecture.html`](../../architecture.html). It cannot drift: a stale code-pointer fails the build.

If you read one other page first, read [Concepts](concepts.md) — the cognition / perception split.

## The four tiers

Signal flows **up**; a compiled `WatchSpec` flows **down**. Each tier is a package or subpackage.

```
┌── VLM · System 2 — summoned ─────────────────────────────────────────────────┐  percept/backends
│  vision.judge(condition, frame) → Verdict   ·   conductor compiles goal→spec  │  Claude · Gemini · …
│  general intelligence, stateless, paid per call, behind vendor-neutral seams  │  (○ seam unless wired)
└───────────────────────────────────────────────────────────────────────────────┘
        ▲ Verdict{satisfied, confidence, evidence}   │ only when the brain decides to spend a call
┌── Cognition · System 1 — always in the loop ─────────────────────────────────┐  percept/
│  senses (rhythm·transition·judge·…) · the three-state gate (FIRE·SILENT·ASK)  │  agent · gate ·
│  · concern (count·escalate) · entity graph · deterministic · cheap · stdlib   │  behavior · rhythm
│  ★ the brain never sees pixels — it reasons over Verdict + Tier0Signal streams │  · concern
└───────────────────────────────────────────────────────────────────────────────┘
        ▲ channels · Tier0Signal{peak·onset·threshold}   │ every frame · no model call
┌── Measures (encoders) — raw → channels ──────────────────────────────────────┐  percept/harness
│  eye: pose-channels (RTMO) · ears: acoustic-onset / STT · motion gate         │  detectors · pose
│  turn raw signal into channels and timed boundaries; propose, never judge      │  · tier0
└───────────────────────────────────────────────────────────────────────────────┘
        ▲ frames · audio · scalar signals
┌── Feeds (devices) — raw signal ──────────────────────────────────────────────┐  percept-edge
│  camera · RTSP · video · mic · screen · GPS         run(source) → incidents    │  (on-device, always)
└───────────────────────────────────────────────────────────────────────────────┘
```

The split is the whole idea: a **cheap, stateful System-1 brain** that reasons over a stream of small
structured facts, and an **expensive, stateless System-2 perception** it summons only when it must. A raw
VLM-in-a-loop collapses these two; Percept keeps them apart so the common case costs nothing.

## The tiers in code

| tier | role | package | key symbols |
|---|---|---|---|
| **Feeds** | capture a source into frames/audio; own the `run(source) → incidents` loop | `percept-edge`, [`percept/runtime/worker.py`](../../packages/percept-vision/percept/runtime/worker.py) `run` | camera · mic · screen |
| **Measures** | high-rate encoders turn raw signal into channels + timed boundaries; the motion gate decides *whether to spend a VLM call* | [`percept/harness`](../../packages/percept-vision/percept/harness) | `make_detector` ([detectors.py](../../packages/percept-vision/percept/harness/detectors.py)) · `PoseChannels.measure` · `MotionGate.evaluate` ([tier0.py](../../packages/percept-vision/percept/harness/tier0.py)) |
| **Cognition** (System 1) | senses, the three-state gate, the concern primitive, entity graph — reasons over `Verdict` + `Tier0Signal`, never pixels | [`percept`](../../packages/percept-vision/percept) | `Percept` ([agent.py](../../packages/percept-vision/percept/agent.py)) · `Gate.evaluate` ([gate.py](../../packages/percept-vision/percept/gate.py)) · `RhythmTracker` ([rhythm.py](../../packages/percept-vision/percept/rhythm.py)) |
| **VLM** (System 2) | one stateless seam: `vision.judge(condition, frame) → Verdict`; the conductor compiles goals | [`percept/backends`](../../packages/percept-vision/percept/backends) | `AnthropicVision.judge` ([anthropic.py](../../packages/percept-vision/percept/backends/anthropic.py)) · `GeminiVision.judge` |
| **Wire contract** | the frozen types crossing every seam | [`percept/contracts`](../../packages/percept-vision/percept/contracts) | `Verdict` · `CONTRACT_VERSION` (1.3.0) |

## The three-state gate — the rising-edge armed-bit

The defining primitive is [`Gate.evaluate`](../../packages/percept-vision/percept/gate.py): it maps a
belief to one of three actions and **fires only on the rising edge** of a condition becoming true, not on
every frame it stays true.

- **FIRE** — known true *and* newly so → act once, then disarm until it re-arms.
- **SILENT** — known false → stay quiet.
- **ASK** — unknown → ask the human rather than deliver a confidently-wrong nag.

`FIRE`, `SILENT`, `ASK` are the three `GateAction` values in
[gate.py](../../packages/percept-vision/percept/gate.py); the rising-edge armed-bit is what separates
Percept from a per-frame classifier that re-alerts on every positive frame.

## The concern primitive — count · escalate

A **concern** is the brain's running stake in one watched thing. It carries a cost asymmetry
(`miss_cost` × `false_fire_cost`) that drives its **calibration-free** jobs —
[`attention_weight`, `is_hard_quadrant`, `requires_verification`](../../packages/percept-vision/percept/concern.py).
Its per-concern firing threshold ([`threshold_for`](../../packages/percept-vision/percept/concern.py)) is
today **pinned to the gate default (0.7)** and inert: it stays a no-op until the labeling pipeline shows
confidence discriminates. When evidence is ambiguous the concern **escalates** perception —
single frame → strip → +audio — via
[`escalate_perception`](../../packages/percept-vision/percept/escalation.py).

## The two firing paths

Both paths share the same gate and emit the same incident; they differ only in whether perception was paid
for. (Worked judged example: [the milk-boilover flow](e2e-flow-milk-boilover.md).)

**1. Edge-only — no VLM.** Detectors propose boundaries; the brain counts and gates. Repetition counting
(the **rhythm** sense) runs with no model call —
[`perceive_rhythm`](../../packages/percept-vision/percept/agent.py) folds pose channels through
`RhythmTracker` and emits a `FireEvent` on each causally-confirmed rep:

```
measure (pose-channels) → perceive_rhythm → RhythmTracker.observe → Rep → gate → incident
```

**2. Judged — VLM.** The motion gate clears, the brain asks perception one grounded question
([`perceive_judged`](../../packages/percept-vision/percept/agent.py)), and the `Verdict` flows through the
same three-state gate:

```
wake (MotionGate) → vision.judge(condition, frame) → Verdict → gate → FIRE · ASK · SILENT → incident
```

### Fire-before-confirm

For high-consequence goals the brain can **act first and verify after**: a `FIRE` can instantiate a
**verification concern** that confirms the action's effect on subsequent frames
([agent.py — "REQUIRE VERIFICATION (A3)"](../../packages/percept-vision/percept/agent.py)), rather than
blocking the alert on a slow confirmation. The edge path embodies the same idea —
*edge-proposes · brain-counts · VLM-confirms* — so the cheap signal carries the count and the model is
spent only to confirm, never to tally.

## Two-layer brain, vendor-neutral seams

System-1 cognition is **pure stdlib and always in the loop**; it never imports a vendor SDK (enforced by
[`test_core_no_vendor_imports.py`](../../packages/percept-vision/tests/test_core_no_vendor_imports.py)).
System-2 intelligence is summoned through five named seams — `vision · specialist · conductor · stt · tts`
— resolved by name in [`percept.registry`](../../packages/percept-vision/percept/registry.py)
(`register(seam, name, factory)` → `make(seam, name)`). Swap a backend by changing a string; the cognition
layer is unchanged.

```python
agent = Percept.create(mode="offline")    # scripted backends, no keys — proves WIRING, not intelligence
agent = Percept.create(vision="gemini")   # pip install "percept-vision[gemini]" + GEMINI_API_KEY
```

A backend is **● integrated** when its module imports *and* it is registered via `registry.make`;
otherwise it is a **○ seam** (a named slot with no wired implementation). The generated diagram marks each
adaptor ●/○ **from the code**, so the picture always matches what actually ships. See
[Backends](../reference/backends.md).

## What you author — the goal

You configure perception by declaring a `Goal`; an authored `behavior` block (or the conductor) is
firewalled before it reaches the engine. The tunable surface, grouped:

| group | fields | controls |
|---|---|---|
| **condition** | `condition`, `say`, `type` | what to watch, what to announce, the sense |
| **consequence** | `miss_cost`, `false_fire_cost`, `protected_floor` | how the gate weighs a miss vs a false alert |
| **behavior** | `rearm`, `fire_when`, `cadence_s`, `deadline_s` | when it re-arms; alert on the *n*-th; rate limits |
| **perception** | `modality`, `temporal`, `n_frames`, `roi`, `audio_s` | how the VLM looks (single · strip · window · audio) |
| **rhythm** | `channel`, `cadence_s`, `mode`, `window_s` | the rep prior: which body channel, discrete vs continuous |
| **sense** (fovea) | `base_fps`, `max_fps`, `escalate_threshold` | per-goal sampling density: coast, then escalate on motion |
| **spatial** | `region_hint`, `gate_on` | region of interest; which signal arms the goal |

## See also

- [Concepts](concepts.md) — the cognition / perception split in depth
- [Backends](../reference/backends.md) — the five seams and how to add one
- [Registration](../reference/registration.md) — the detector / concern / enricher registries
- [Wake policy](../reference/edge-policy.md) — the cheap "should we spend a VLM call?" layer
- [Runtime](../how-to/runtime.md) — `run(source) → incidents`
- [Audio modality](../reference/audio-modality.md) — fused vision + audio judging
