# `percept-edge`

> **percept-edge 1.0.0** · the on-device edge for [percept-vision](https://pypi.org/project/percept-vision/) ·
> docs: <https://percept.percia.ai> · Apache-2.0. The package version moves independently of the wire
> `CONTRACT_VERSION` in `src/contract.js` — pin a version.

The on-device **reactive edge** for the [Percept](https://pypi.org/project/percept-vision/) SDK — the
cheap, high-fps reflex that runs next to the camera. It does two jobs:

1. **gates** — decides *when* a frame is worth an expensive cloud vision call (motion + VAD salience), so
   the cloud VLM runs only on the frames that matter (a large reduction in cloud calls);
2. **runs the brain-selected edge skills** — cheap detectors that propose *timed events* the cognition
   core counts (reps, sound onsets) **without** a cloud call;
3. **screens egress for sensitive content** — a multi-modal, on-device privacy classifier on the
   *outbound* boundary (`PrivacyScreen`), so a frame/message can be redacted or held before it reaches a
   cloud judge. This layer can *never* be a cloud call — you can't ship sensitive data to a remote LLM to
   ask whether it's safe to ship — so it is an owned on-device artifact.

Pure ESM JavaScript, **zero runtime dependencies** (vitest is the only devDep). The DSP skills + the gate
need no model and no native deps; the one model-backed skill (pose) takes a pluggable adapter you supply.

## The wire contract (detector + gate OUTPUTS: byte-for-byte with the Python SDK)

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
in `src/contract.js`, kept identical to the Python `percept.edge` contract dataclasses — verified by a
cross-language parity harness (12/12 byte-for-byte; the JS and Python detectors + gate produce identical
output on identical input).

> **WatchSpec is a documented subset on-device.** The edge executes `cadence_s` (a per-condition
> escalation throttle) and `gate_on` (`motion`/`structural`/`semantic`/`fixed`) locally in
> `_shouldEscalate` (`src/edge.js`). The one server-only field is **`schedule`** — civil-time windows are
> enforced by the cognition layer when it gates, not on-device. The edge only *proposes* (Tier0Signal +
> boundaries), so the "byte-for-byte" guarantee is on the **detector + gate outputs** (verified 12/12),
> not full `WatchCondition` field parity — an edge never fires on its own.

## The gate (the reflex)

```js
import { Edge, makeFrame } from "percept-edge";

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
**boundaries** the brain counts. The JS twin of the Python `percept.edge.detectors` surface:

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
import { registerPoseSkill } from "percept-edge/pose";
registerPoseSkill(myLandmarker);   // opt-in: supply your pose model adapter (MediaPipe Tasks for Web, TF.js…)
```

## PrivacyScreen — on-device egress screening (`0.7`)

A multi-modal privacy classifier that sits on the **egress** boundary, sibling to `Edge`. Multi-modal in,
one verdict out, in the same `allow` / `redact` / `drop` / `confirm` vocabulary your gates already speak —
so it drops into existing egress paths without changing them. It fills the previously-inert `semantic`
Tier-0 channel with an owned, on-device *safety* judgment.

```js
import { PrivacyScreen } from "percept-edge/privacy";

const screen = new PrivacyScreen({ image: myPrivacyModel /* your ONNX adapter */ });
await screen.load();

// TEXT — synchronous, sub-millisecond, no model, no clock. The hot path on every message.
screen.screen({ modality: "text", text: "my ssn is 123-45-6789" });
//        → { sensitive: true, categories: ["pii-id"], score: 0.95, spans: [{start,end,category}], action: "redact" }

// IMAGE / AUDIO — async (model inference); image runs under a fail-closed timeout.
await screen.screenAsync({ modality: "image", frame /* egress-candidate JPEG, not the salience buffer */ });
await screen.screenAsync({ modality: "audio", pcm });
```

Three tiers behind one interface:

| tier | category coverage | how | ships in |
|---|---|---|---|
| **text** | `pii-id`, `payment`, `financial`, `credential` | deterministic regex + Luhn/ABA/SSN checksums — **no model, sub-ms** | `0.7` (full) |
| **image** | `document-id`, `sexual`, `credential`, … | a small quantized multi-label CNN behind an **adapter slot you supply** (`{ load, infer }`) — see `src/privacy-image.js` | `0.7` (seam); real model app-side |
| **audio** | *(none asserted)* | coarse acoustic-context signal (no on-device STT) — honestly bounded, only ever raises `confirm` | `0.7` (coarse, provisional) |

**Invariants.** *On-device only* — nothing here (nor any tier) makes a network call; the image model is a
product-supplied artifact behind an adapter, so `percept-edge` stays **zero-dependency**. *Deterministic* —
same input → same verdict; `clock` feeds only the image timeout, never a decision. *Fail-closed* — on
timeout/error/malformed model output for an egress candidate, the verdict is `sensitive: true` with a
conservative action (`drop` by default, tunable via `failAction`). *Recall-biased* — a masked non-secret
costs the judge a little context; a leaked secret is unrecoverable.

Ships **with an eval harness** (`eval/`, mirroring `parity/`) — a labeled multimodal fixture set scored on
per-category leak-rate (recall), over-mask rate (precision), and added-latency p50/p95, gated in CI:

```bash
npm run eval    # prints the scorecard + regression-checks it against eval/scorecard.baseline.json
```

## Deployment topologies

- **A — all server-side:** Python `percept.edge` runs the gate + skills in-process with the brain.
  No `percept-edge`.
- **B — on-device:** `percept-edge` gates + runs skills on the device, advertises `EdgeCapabilities`,
  and sends `Tier0Signal` + `BoundaryReport` up; the server maps boundaries to `perceive_boundary`. Use
  this to keep frames on-device until something matters (bandwidth + privacy).

## Known tuning

The motion-gate threshold (a `WatchCondition.threshold`, default `0.3`; the edge falls back to
`Edge.defaultThreshold = 0.05` when a condition sets none) is a **cost↔recall knob**: well-calibrated for salient motion,
but it skips very-low-motion onsets (measured edge event-recall ≈ 0 on subtle actions — the gate never
escalates, so the VLM never sees them). Tune the threshold per motion regime for your use case. This is a
deliberate post-publish tuning item, not a blocker.

## Test

```bash
npm install
npm test    # 124 vitest: contract, motion, vad, skills, pose, edge, privacy, privacy-eval
npm run eval  # the PrivacyScreen scorecard (leak-rate / over-mask / latency) + regression gate
```

Apache-2.0.
