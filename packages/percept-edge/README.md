# `@percept/edge`

On-device **reactive edge** for the Percept SDK (Phase 6 / V0.4): the reflex
gate that decides *when* a frame is worth an expensive cloud vision call. It is
the perception analog of voice VAD.

- **down channel** (cloud → edge): the brain compiles active goals into a
  `WatchSpec` (anchors + thresholds). The edge consumes it via
  `edge.applyWatchSpec(spec)`.
- **up channel** (edge → cloud): the edge reports a `Tier0Signal` (motion /
  structural / semantic salience). The expensive vision call runs **only when
  salience clears threshold**.

Pure ESM JavaScript. The **tested path needs no model downloads and no native
deps** — motion (pixel-diff) and an energy-threshold VAD work standalone. The
heavy detectors are documented adapter slots (see below).

## The wire-contract (FROZEN — must match the Python SDK byte-for-byte)

These JSON shapes are shared with the Python core; field names are frozen so
the two halves interoperate over the wire. A mismatch breaks interop.

```text
Box            = { x, y, w, h }                                # normalized 0..1
WatchCondition = { goal_id, anchor_text, threshold, region_hint: Box | null }
WatchSpec      = { conditions: WatchCondition[], version }     # version defaults to 1
Tier0Signal    = { motion, structural, semantic, ts }
```

Factories and validators live in `src/contract.js`:

- `makeWatchCondition({ goal_id, anchor_text, threshold?, region_hint? })`
- `makeWatchSpec({ conditions?, version? })`
- `parseWatchSpec(json)` — accepts a JSON string **or** object, validates every
  field, and returns a normalized `WatchSpec`. This is the cross-language entry
  point: feed it exactly what the Python side emits.
- `makeTier0Signal({ motion, structural, semantic, ts })` /
  `parseTier0Signal(json)`

Example `WatchSpec` exactly as the Python side emits (round-trips through
`parseWatchSpec`):

```json
{"conditions":[{"goal_id":"g1","anchor_text":"the user is drinking coffee","threshold":0.3,"region_hint":null}],"version":1}
```

## The motion-only proof point

The single wired loop — `WatchSpec` consumed by an on-device gate that reports
`Tier0Signal` back up — is the proof point. The wired, tested detector is
**motion only**:

```js
import { Edge, makeFrame } from "@percept/edge";

const signals = [];
const edge = new Edge({ onSignal: (s) => signals.push(s) });

// down: receive the brain's WatchSpec
edge.applyWatchSpec(
  '{"conditions":[{"goal_id":"g1","anchor_text":"the user is drinking coffee","threshold":0.3,"region_hint":null}],"version":1}'
);

const still = makeFrame(4, 4, 100); // constant grayscale frame
const moved = makeFrame(4, 4, 240); // a big change

edge.processFrame(still); // first frame: no history → no signal
edge.processFrame(still); // still scene → gated out (returns null), no cloud call
const sig = edge.processFrame(moved); // motion clears threshold → Tier0Signal up

// sig === { motion: ~0.55, structural: 0, semantic: 0, ts: <seconds> }
```

`motionScore(prevFrame, frame)` (in `src/motion.js`) is a deterministic
mean-absolute pixel-diff over a minimal grayscale frame
(`{ width, height, data: Uint8Array }`), normalized to `[0, 1]`: a still scene
scores ~0, a large change scores high.

VAD lives in `src/vad.js`: `EnergyVad` is a streaming energy-threshold
endpointer that fires an endpoint after a run of low-energy frames following
speech; `endpointEnergySequence(energies, opts)` runs it over a fixture
sequence and returns the endpoint frame indices.

## Where the heavy detectors plug in

All heavy detectors are **interface stubs / adapter slots** — documented, not
exercised by `npm test`, requiring no model download for the standalone path:

| Slot | Method | Intended detector |
|---|---|---|
| Voice activity | `SileroVadAdapter` (in `vad.js`) | Silero-VAD (ONNX via onnxruntime-web / transformers.js) |
| Structural salience | `edge.setStructuralAdapter(fn)` | pose / hands (MediaPipe) |
| Semantic salience | `edge.setSemanticAdapter(fn)` | CLIP zero-shot gate (transformers.js/ONNX) |
| Media publish | `edge.setPublisher(fn)` | LiveKit (publish gated media upstream) |

Each adapter returns a `[0, 1]` salience that folds into the same gate; the
overall salience is the strongest channel, so motion alone can escalate. With
no adapters wired, structural/semantic stay `0` and the loop runs motion-only.

## Test

```bash
npm install
npm test    # vitest
```

Covers: `motionScore` (still → ~0, motion → high), `parseWatchSpec` on a
cross-language fixture (the exact JSON the Python side emits, asserting
round-trip), the energy VAD endpointing a fixture sequence, and the `Edge`
emitting a well-formed `Tier0Signal` (still frames → no escalation, a moving
frame → escalation).
