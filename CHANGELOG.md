# Changelog

Notable changes to Percept. This is a **research preview** — the SDK's public surface may change between
0.x releases.

## 0.6.0 — Research preview (2026-06)

### Packaging — one install
- `percept-contracts` and `percept-harness` are now **bundled inside `percept-vision`** as the
  `percept.contracts` and `percept.harness` subpackages. Install one package:
  `pip install percept-vision`. Heavy deps stay opt-in via extras:
  `[gemini] [claude] [openrouter] [deepgram] [runtime] [harness] [pose] [bench] [audio] [screen] [otel]`.
- **Migration:** `from percept_harness.X import Y` → `from percept.harness.X import Y`;
  `from percept_contracts import Z` → `from percept.contracts import Z`.
- **Deprecated:** the standalone `percept-harness` PyPI package is deprecated in favor of
  `percept-vision (>=0.6.0)`. `percept-contracts` was never published standalone.

### Versioning
- Package version is `0.6.0`. The **wire-contract** version is separate: `CONTRACT_VERSION = "1.2.0"`
  (additive-only). A `percept-vision 0.6.0` install exporting contract `1.2.0` is expected, not a bug.

### Added
- **OpenRouter** vision backend (`vision="openrouter"`) — cross-provider VLM judging (Qwen-VL, InternVL, GPT…).
- **Gemini Live** backend (`vision="gemini-live"`) — the realtime/interactivity model, via
  audio→transcript→verdict; `process_qa` runs over a spoken running answer.
- `process_qa` (accumulating-answer questions) implemented on all three judge backends; the proactiveqa
  bench track now measures the shipping SDK path.
- Runtime worker is **cadence-aware** for brief-action goals; injectable `Clock` (+ `FakeClock`) for
  deterministic, replayable temporal logic; CI import-lint enforcing a vendor-free pure-stdlib core.
- Per-request timeout on the Gemini client so a stalled socket can't hang a run.

### Benchmark fixes
- pt-action sensitive above-Nyquist cadence (edge recall 0→0.75); proactiveqa running-answer resolution
  (unresolved 1.0→0.29, PAUC 25→55 on gemini-3.5-flash); per-performer counting (HERO; pommel 9→33==GT);
  timespeak zero-width-window filter. See `docs/benchmark-methodology.md`.

`percept/gate.py` (the three-state firing gate) is **byte-for-byte stable** across this release.
