# Changelog — `percept-edge`

Notable changes to the `percept-edge` npm package. From **1.0.0** the public surface follows semantic
versioning. The package version (this file) moves independently of the wire `CONTRACT_VERSION` in
`src/contract.js` (currently `1.10.0`, additive-compatible with the Python `percept.contracts` wire),
which changes only on wire-shape changes.

## Unreleased — numeric-seam hardening (**breaking**: warrants a MAJOR at release)

Three changes below alter observable behaviour of an exported surface, so this line cannot ship as a
minor. **The version number is deliberately not chosen here** — pick it at release. No wire-contract
change beyond the additive `gate_on: "fused"` member (see below); `CONTRACT_VERSION` moves additively.

### Fixed — the audio gate could not fire on the material it exists to hear

`AcousticOnset` shipped `high = 0.09` / `low = 0.0225` where the Python harness uses `0.001` /
`0.00025`. An energy of `0.09` is an RMS amplitude of ~0.3; the loudest chunk in the entire 987-second
annotated corpus reaches `0.0277`, so **the JS edge's shipped acoustic gate could wake zero times on
real material** while the Python twin woke normally on the identical stream. The bars are now aligned
and exported as `ACOUSTIC_ONSET_HIGH` / `ACOUSTIC_ONSET_LOW`.

This divergence survived because the parity harness's acoustic vectors all sat far above every
candidate threshold — it reported MATCH with the two runtimes an order of magnitude apart. Two
threshold-discriminating energy vectors now cover it.

### Fixed — `PrivacyScreen` threw on a valid in-band audio buffer

`PrivacyAudioHead`'s triangular ZCR membership was evaluated in float64 with no clamp and no edge
short-circuit, so at the lower band edge it produced a residual of the order of `1e-17`. For several
ordinary band tunings that residual is **negative**, and `makePrivacyVerdict` then raised
`TypeError: score must be a finite number in [0,1]` — an unhandled throw on the egress path for valid
PCM. A degenerate band (`lo === hi`) separately produced `1 - 0/0 = NaN`, silently disabling the tier;
the NaN was laundered twice before reaching the one guard that would have caught it.
`bandMembership()` now short-circuits both edges and clamps, and the constructor rejects a non-finite
`minEnergy` and a non-finite or non-increasing `zcrBand`.

### Fixed — other numeric seams

- `runImageTier` read the caller-supplied clock *above* its `try`, so a throwing clock escaped the
  fail-closed boundary and rejected the promise where the contract promises a verdict.
- `makeBox` validated nothing while its twin `parseBox` required all four fields finite — the only
  make/parse pair in `contract.js` that disagreed, so a `NaN` region hint serialized into a box Python
  could not read.
- `motionScore` returned `undefined` for a `NaN` `blockGrid` and `NaN` for a `NaN` region, against a
  documented `@returns {number} in [0,1]`.
- `conditionMotion`'s result reached the threshold comparison without the `clamp01` the structural and
  semantic adapters were given.
- `imageBudgetMs: NaN` made `setTimeout` fire at 0 ms, so **every image frame failed closed** and, with
  the default `failAction`, was dropped at egress. Now rejected at construction.
- The detector bank routed any unrecognised channel string to frame-diff motion instead of giving it no
  input, so a detector declaring `signal: "thermal"` — or a typo — silently emitted motion boundaries.
- `blockGrid` with a non-positive dimension scored `0` where the Python twin raises; the block
  partition padded empty blocks instead of fitting the grid to the active area.
- `egoResidualScore` measured the residual over the whole frame even when the condition carried a
  `region_hint`, so an ego-compensated goal silently lost its watch-out zone.

A single `src/numeric.js` (`finiteNumber`, `unitInterval`, `positiveDim`, `clamp01`, `nonEmptyBox`) now
backs these seams; it absorbed four separate copies of the finite-number predicate that had drifted.

### Changed — **breaking**

- **`makeFrame` saturates instead of wrapping.** A numeric fill outside `0..255` was masked to a byte
  (`300` → `44`, `-1` → `255`), where the Python `gray_frame` clamps (`300` → `255`, `-1` → `0`). Two
  helpers building the same `(w, h, fill)` produced different images. Now matches `_clamp_u8`.
- **`PRIVACY_CATEGORIES` is a frozen facade, not a `Set`.** Its docstring promised immutability while
  exporting a bare mutable `Set` (`Object.freeze` does not disable `add`/`delete`/`clear`). It now
  exposes `has`, `size` and `[Symbol.iterator]`. A consumer calling `.forEach`, `.values()`, `.keys()`,
  `.entries()`, the mutators, or testing `instanceof Set` breaks. **In-repo callers: zero** — every use
  is `.has()` or `for…of`. `PRIVACY_ACTIONS` is now genuinely frozen.
- **`AcousticOnset` rejects an inverted hysteresis band** (`low > high`) at construction, where it
  previously oscillated at half the sample rate and minted phantom onsets from a steady tone.
  `low === high` stays permitted — the oscillation is unreachable at a zero-width band.
- **A malformed `wake_policy` raises** instead of silently degrading to the legacy `max(signals)` while
  still wearing an authored threshold.

### Changed — wake fusion (additive wire member)

The fusion now runs over the **measured** channels with `weights` as per-channel priors defaulting to
`1.0`, so adding a weight can no longer score *below* the maximum it replaced, and a channel-name typo
no longer produces total deafness with every signal present. Each physical measurement populates
exactly one channel (three alias pairs removed). `top_channel` is derived from the same per-channel
contribution terms the score summed, and is `null` when nothing contributed. `GATE_ON` gains `"fused"`
so a wake policy fuses only for goals that asked to be fused, instead of silently overriding every
other channel — **a deployment relying on that override must now set `gate_on: "fused"` explicitly.**

## 1.0.0 — first stable release

First stable release of the on-device edge, published alongside `percept-vision 1.0.0`. Gate + DSP
skills, the capability handshake, and `PrivacyScreen` egress screening are stable; the JS detector/gate
outputs remain byte-for-byte with the Python `percept.edge` harness (parity-verified). No wire-contract
change from the previous line.

## 0.7.0 — PrivacyScreen (on-device egress screening)

New capability: **`PrivacyScreen`** — a multi-modal, on-device privacy classifier on the *egress*
boundary, sibling to `Edge`, filling the previously-inert `semantic` Tier-0 channel with an owned safety
verdict. Multi-modal in (`text` | `image` | `audio`), one `PrivacyVerdict` out, in the existing
`allow` / `redact` / `drop` / `confirm` action vocabulary. No wire-contract change — `CONTRACT_VERSION`
stays `1.0.0` (the verdict is a package-local shape, not a Python↔JS message).

### Added
- `PrivacyScreen` (`src/privacy.js`) — orchestrator with a synchronous `screen(input)` (text-only,
  sub-millisecond hot path) and `async screenAsync(input)` (all modalities); `load()` for lazy model
  init; a pure, recall-biased action ladder (`allow < redact < confirm < drop`).
- `makePrivacyVerdict` + `PRIVACY_CATEGORIES` / `PRIVACY_ACTIONS` / `PRIVACY_SCREEN_VERSION`.
- **Text tier** (`src/privacy-text.js`, `scanText`) — deterministic, no-model, no-clock detectors: SSN
  (structural validity), payment cards (Luhn), CVV-in-context, EIN/tax id, ABA routing (checksum),
  bank account, and vendor/`generic` credentials. Emits redaction spans (text offsets).
- **Image tier** (`src/privacy-image.js`) — an adapter **slot** (`PrivacyImageAdapter` + `runImageTier`),
  twin of `SileroVadAdapter`: the product supplies the ONNX model + onnxruntime-web; the SDK ships the
  interface, the fail-closed timeout, and result validation. **No model asset ships; zero deps preserved.**
- **Audio tier** (`src/privacy-audio.js`, `PrivacyAudioHead`) — a coarse, honestly-bounded acoustic-context
  head (no on-device STT); asserts no category and never escalates past `confirm`. *Provisional in 0.7.*
- **Eval harness** (`eval/`) — labeled multimodal fixtures scored on per-category leak-rate (recall),
  over-mask rate (precision), and added-latency p50/p95; `npm run eval` prints the scorecard and
  regression-gates it against `eval/scorecard.baseline.json`. Wired into CI via `test/privacy-eval.test.js`.
- Subpath exports `./privacy`, `./privacy-text`, `./privacy-image`, `./privacy-audio`.

### Invariants
- **On-device only** — no network call from `PrivacyScreen` or any tier; enforced by a source-scan test.
- **Deterministic** — same input → same verdict; `clock` feeds only the image timeout, never a decision.
- **Fail-closed** — timeout/error/malformed model output on an egress candidate ⇒ `sensitive: true` with
  a conservative action (`drop` default, tunable via `failAction`).

### Notes / follow-ups (0.7.x)
- Full acoustic head + real app-side ONNX integration (model-version handshake, real-frame fixtures).
- Optional read-only bridge surfacing the last image score into `Tier0Signal.semantic` for observability.
- Detector corpus to be extended with the app's handed-over patterns; per-category over-mask ceilings to
  be agreed with the consuming app.

## 0.6.0 — Research preview

Initial OSS research-preview: the motion/VAD **gate**, the brain-selected **skill registry**
(`motion-periodicity`, `acoustic-onset`, pluggable `pose-openness`), the `Edge` WatchSpec-down /
Tier0Signal-up loop, and the frozen JS wire-contract (`src/contract.js`) verified byte-for-byte against
the Python SDK by the `parity/` harness.
