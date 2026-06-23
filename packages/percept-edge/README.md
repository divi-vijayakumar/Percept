# `@percept/edge`

> **⚠️ Research preview (v0.6.0).** Published for real-life testing + feedback, **not production**. APIs
> may change between `0.x` releases — pin a version.

The on-device **reactive edge** for the [Percept](https://pypi.org/project/percept-vision/) SDK — the
cheap, high-fps reflex that runs next to the camera. It does two jobs:

1. **gates** — decides *when* a frame is worth an expensive cloud vision call (motion + VAD salience), so
   the cloud VLM runs only on the frames that matter (~83% fewer calls in practice);
2. **runs the brain-selected edge skills** — cheap detectors that propose *timed events* the cognition
   core counts (reps, sound onsets) **without** a cloud call.

Pure ESM JavaScript, **zero runtime dependencies** (vitest is the only devDep). The DSP skills + the gate
need no model and no native deps; the one model-backed skill (pose) takes a pluggable adapter you supply.

## The wire contract (FROZEN — byte-for-byte with the Python SDK)

```text
down (cloud → edge):
  WatchSpec       { conditions: WatchCondition[], version }
  WatchCondition  { goal_id, anchor_text, threshold, region_hint: Box|null, detectors: string[] }
up   (edge → cloud):
  EdgeCapabilities { capabilities: string[] }                       # advertise, at connect
  Tier0Signal      { motion, structural, semantic, ts }             # per-frame salience GAUGE
  BoundaryReport   { boundaries: [{ goal_id, detector, kind, t, value }] }   # per-event, when a skill fires
```

`detectors` carries the brain's skill selection; `EdgeCapabilities` advertises what this edge can run so
the brain selects only within it (the capability handshake). Every shape has `make*`/`parse*` validators
in `src/contract.js`, kept identical to the Python `percept-harness` dataclasses — verified by a
cross-language parity harness (12/12 byte-for-byte; the JS and Python detectors + gate produce identical
output on identical input).

## The gate (the reflex)

```js
import { Edge, makeFrame } from "@percept/edge";

const edge = new Edge({ onSignal: (s) => console.log("salience", s) });
edge.applyWatchSpec(/* the brain's WatchSpec JSON */);

edge.processFrame(makeFrame(8, 8, 100)); // first frame: no history → no signal
edge.processFrame(makeFrame(8, 8, 100)); // still scene → gated out (null), no cloud call
edge.processFrame(makeFrame(8, 8, 240)); // motion clears threshold → Tier0Signal up
```

`motionScore(prev, frame)` (`src/motion.js`) is a deterministic mean-absolute pixel-diff over a minimal
grayscale frame (`{ width, height, data: Uint8Array }`), normalized to `[0,1]`. VAD lives in `src/vad.js`
(`EnergyVad` + a `SileroVadAdapter` slot).

## The skill registry (the on-device "skills")

Cheap, high-fps detectors the brain selects per goal. Each consumes a signal channel and emits timed
**boundaries** the brain counts. The JS twin of `percept_harness/detectors.py`:

| skill | channel | proposes | model? |
|---|---|---|---|
| `motion-periodicity` | motion | one `peak` per rep (counting) | no — pure DSP |
| `acoustic-onset` | energy | one `onset` per sound starting | no — pure DSP |
| `pose-openness` | frame | one `peak` per whole-body rep | yes — pluggable landmarker |

When a `WatchSpec` selects detectors, the edge runs them every frame and emits boundaries up a separate
channel (decoupled from the per-frame gauge):

```js
const edge = new Edge({ onBoundary: (batch) => batch.forEach(b => count(b.goal_id)) });
edge.applyWatchSpec(/* WatchSpec with detectors:["motion-periodicity"] */);
for (const frame of stream) edge.processFrame(frame, { energy });   // boundaries fire on the rising edges
```

Model-backed skills are **lazy** — `await edge.loadSkills()` loads them once on activation, never at
construct. `edge.capabilities()` / `edge.advertise()` report the registered set up the wire.

### Per-device customization — register only what you need

The same runtime, different skill sets (a drone, AR glasses, and a fitness coach each register what they
need). A skill is `registerSkill({ name, signal, manifest, factory })`, nothing more:

```js
import { registerPoseSkill } from "@percept/edge/pose";
registerPoseSkill(myLandmarker);   // opt-in: supply your pose model adapter (MediaPipe Tasks for Web, TF.js…)
```

## Deployment topologies

- **A — all server-side:** the Python `percept-harness` runs the gate + skills in-process with the brain.
  No `@percept/edge`.
- **B — on-device:** `@percept/edge` gates + runs skills on the device, advertises `EdgeCapabilities`,
  and sends `Tier0Signal` + `BoundaryReport` up; the server maps boundaries to `perceive_boundary`. Use
  this to keep frames on-device until something matters (bandwidth + privacy).

## Known tuning (it's a research preview)

The motion-gate threshold (default `0.10`) is a **cost↔recall knob**: well-calibrated for salient motion,
but it skips very-low-motion onsets (measured edge event-recall ≈ 0 on subtle actions — the gate never
escalates, so the VLM never sees them). Tune the threshold per motion regime for your use case. This is a
deliberate post-publish tuning item, not a blocker.

## Test

```bash
npm install
npm test    # 70 vitest: contract, motion, vad, skills, pose, edge
```

Apache-2.0.
