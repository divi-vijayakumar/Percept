# Architecture

How the pieces fit. Percept turns a goal stated in plain language into a proactive agent over a live
audio-visual stream. This page is the **map** — every layer links to its detailed doc.

If you read one other page first, read [Concepts](concepts.md) (the cognition / perception split). This
page zooms out to the whole stack and the two firing paths.

## The stack

```
  you author ──►  Goal / WatchSpec     condition · consequence · behavior · perception · rhythm
                       │   compiled by the conductor (optional LLM) or written by hand
                       ▼
┌──────────────────────────── COGNITION — "the brain" ──────────────────────────────────┐
│  senses: judge · transition · episode · absence · rhythm                                │
│  the three-state gate  (known → act · not → silent · unknown → ask)  · rising-edge      │
│  entity graph · concern / consequence · stateful · deterministic · cheap                │
│  ★ the brain never sees pixels — it reasons only over Verdict + Boundary streams         │
└─────────────────────────────────────────────────────────────────────────────────────────┘
   ▲ Verdict{satisfied, confidence, evidence}            ▲ Boundary{peak·onset·threshold·sample}
   │  only when the brain decides to spend a call        │  every frame · no model call
┌─────────── PERCEPTION — "the eyes" ───────────┐   ┌──────────── EDGE — cheap signal ──────────┐
│  vision.judge(condition, frame) → Verdict      │   │  detectors: motion · pose · acoustic       │
│  per-call VLM · stateless · vendor-neutral seam│   │  + the wake policy (motion gate)           │
│  gemini · claude · fake                        │   │  proposes WHERE an event is; never judges  │
└─────────────────────────────────────────────────┘   └─────────────────────────────────────────────┘
   ▲                                                       ▲
   │ frames                                                │ frames · audio · scalar signals
┌──────────────────────── DEVICES / RUNTIME — capture ────────────────────────────────────┐
│  camera · RTSP · video · glasses · mic · screen        run(source) → incidents            │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

The split is the whole idea: a **cheap, stateful brain** that reasons over a stream of small structured
facts, and an **expensive, stateless perception** it calls only when it has to. A raw VLM-in-a-loop
collapses these two; Percept keeps them apart so the common case costs nothing.

## The layers (bottom-up)

| layer | what it does | package | detail |
|---|---|---|---|
| **Devices / runtime** | capture a source into frames; own the `camera → incidents` loop | `percept-edge`, runtime | [runtime](runtime.md), [edge parity](edge-parity.md) |
| **Edge** | cheap high-fps detectors propose timed boundaries; the wake policy decides *whether to spend a VLM call* | `percept-harness` | [registration](registration.md), [wake policy](edge-policy.md) |
| **Cognition** ("the brain") | senses, the three-state gate, entity graph, concern/consequence — reasons over `Verdict` + `Boundary`, never pixels | `percept-vision` | [concepts](concepts.md) |
| **Perception** ("the eyes") | one stateless seam: `vision.judge(condition, frame) → Verdict` | `percept-vision` backends | [backends](backends.md) |
| **Wire contract** | the frozen types crossing every seam (`Verdict`, `Goal`, `Box`, protocols) | `percept-contracts` | — |

## The two firing paths

The defining decision in the engine is **when to spend a VLM call**. There are exactly two paths, and
they share the same gate and emit the same incident — they differ only in whether perception was paid for.

**1. Edge-only — no VLM.** Detectors propose boundaries; the brain counts and gates. Repetition counting
(the **rhythm** sense) runs entirely without a model call:

```
measure (pose-channels) → perceive_rhythm → RhythmTracker.observe → Rep → gate → incident
```

**2. Judged — VLM.** The wake policy clears, the brain asks perception one grounded question, and the
`Verdict` flows through the same three-state gate:

```
wake (motion gate) → vision.judge(condition, frame) → Verdict → gate → FIRE · ASK · SILENT → incident
```

Same gate, same `incident`. The edge path is what makes always-on watching affordable; the judged path is
reserved for the moments that actually need a frontier model. For a worked judged example end to end, see
[the milk-boilover flow](e2e-flow-milk-boilover.md).

## What you author — the goal

You configure perception by declaring a `Goal`; an authored `behavior` block (or the conductor) is
firewalled by `coerce_authored_*` before it touches the engine. The tunable surface, grouped:

| group | fields | controls |
|---|---|---|
| **condition** | `condition`, `say`, `type` | what to watch, what to announce, the sense |
| **consequence** | `miss_cost`, `false_fire_cost`, `protected_floor` | how the gate weighs a miss vs a false alert |
| **behavior** | `rearm`, `fire_when`, `cadence_s`, `deadline_s` | when it re-arms; alert on the *n*-th; rate limits |
| **perception** | `modality`, `temporal`, `n_frames`, `roi`, `audio_s` | how the VLM looks (single frame · strip · window · audio) |
| **rhythm** | `channel`, `cadence_s`, `mode`, `window_s` | the rep prior: which body channel, discrete vs continuous |
| **sense** (fovea) | `base_fps`, `max_fps`, `escalate_threshold` | per-goal sampling density: coast, then escalate on motion |
| **spatial** | `region_hint`, `gate_on` | region of interest; which signal arms the goal |

```python
from percept import Percept, Goal

agent = Percept.create()                                   # fakes, offline, no keys
agent.add_goal(Goal(
    id="coffee",
    condition="the person drinks from a coffee cup",
    say="that's a coffee",
    state={"type": "count", "fire_when": "n == 3"},        # alert on the 3rd cup
))
```

## Swap a backend — the seams

Percept never imports a vendor SDK directly. Five named seams — `vision · specialist · conductor · stt ·
tts` — resolve a backend by name through `percept.registry`. Swap one by changing a string; the cognition
layer is unchanged. Built in today: `fake` (offline, no deps), `gemini`, `claude`, `deepgram` (stt/tts).
Add your own with `registry.register(seam, name, factory)` or a `percept.backends` entry point. See
[Backends](backends.md).

```python
agent = Percept.create(vision="gemini")    # pip install "percept-vision[gemini]" + GEMINI_API_KEY
```

## See also

- [Concepts](concepts.md) — the cognition / perception split in depth
- [Backends](backends.md) — the five seams and how to add one
- [Registration](registration.md) — the detector / concern / enricher registries
- [Wake policy](edge-policy.md) — the cheap "should we spend a VLM call?" layer
- [Runtime](runtime.md) — `run(source) → incidents`
- [Audio modality](audio-modality.md) — fused vision + audio judging
