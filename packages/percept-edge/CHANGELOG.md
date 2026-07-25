# Changelog — `percept-edge`

Notable changes to the `percept-edge` npm package. From **1.0.0** the public surface follows semantic
versioning. The package version (this file) moves independently of the wire `CONTRACT_VERSION` in
`src/contract.js` (currently `1.10.0`, additive-compatible with the Python `percept.contracts` wire),
which changes only on wire-shape changes.

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
