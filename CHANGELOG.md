# Changelog

Notable changes to Percept. This is a **research preview** — the SDK's public surface may change between
0.x releases.

## Unreleased

### Contract — wire version 1.2.0 → 1.3.0 (additive-only; same MAJOR, so 1.2 consumers negotiate cleanly)
- `Verdict.usage` — optional token/cost accounting for the judge call (`{input_tokens, output_tokens,
  cost_usd}`); gate-irrelevant, defaults `{}`.
- `Verdict.events` + new `ClipEvent` DTO — clip-relative event timestamps a video-native judge saw
  inside the judged window; defaults `[]`.
- `Concern.fps` / `Concern.verify_only` — native-clip judging knobs beside `clip`/`window_s`; inert
  unless the concern declares a clip.
- `Concern.version` — the authored concern's own version (a "versioned detection asset"), distinct
  from `CONTRACT_VERSION` and the `WatchSpec.version` wire integer.
- `Tier0Signal.energy` — the audio-energy channel for frame-free audio ingest; `null`/`None` means
  "not an audio-carrying signal" (distinct from `0.0` = measured silence). Mirrored in
  `@percept/edge` `makeTier0Signal`/`parseTier0Signal` (JS contract also bumped to 1.3.0); a 1.2-era
  payload without the key parses unchanged.
- JSON Schemas regenerated (`percept/contracts/schema/`), including the new `ClipEvent.schema.json`.

### `@percept/edge` (npm) — 0.6.0 → 0.7.0
- **`PrivacyScreen`** — a new on-device, multi-modal privacy classifier on the *egress* boundary (sibling
  to `Edge`), filling the previously-inert `semantic` Tier-0 channel with an owned safety verdict in the
  existing `allow`/`redact`/`drop`/`confirm` vocabulary. Text tier (deterministic regex + Luhn/ABA/SSN
  checksums) is full; image tier is an adapter **slot** (app supplies the ONNX model — zero-dep preserved);
  audio tier is a coarse, bounded acoustic head (provisional). Ships with an eval harness + CI-gated
  scorecard (`npm run eval`). No wire-contract change — `CONTRACT_VERSION` stays 1.3.0. See
  `packages/percept-edge/CHANGELOG.md`.

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
- The wheel ships **only the `percept` namespace** — the benchmark harness (top-level `eval`) is
  repo/dev tooling and is no longer installed (no `eval` name squat in site-packages). `percept-bench`
  from a bare wheel install prints a "run from a checkout" note instead of importing it.

### Versioning
- Package version is `0.6.0`. The **wire-contract** version is separate: `CONTRACT_VERSION = "1.2.0"`
  (additive-only). A `percept-vision 0.6.0` install exporting contract `1.2.0` is expected, not a bug.

### Security
- **`WebhookSink` is SSRF-safe by default** (`UrlPolicy`): https-only, and any non-globally-routable
  target (private, loopback, link-local incl. metadata `169.254.169.254`, CGNAT `100.64/10`) is
  rejected; redirects are not followed. **Breaking:** a local sink (`http://localhost:8080` …) now
  raises at construction — pass `url_policy=UrlPolicy.dev()` for it.
- **Dead-letter records are redacted** — `{url(scheme://host only), body_sha256, error, attempts}`;
  the raw body, signing HMAC, auth headers, and path/query secrets (Slack/Discord tokens, `?token=`)
  are never persisted.

### Observability
- **Per-frame decision trace** (`on_trace`): `Percept.create(on_trace=…)` / `run(…, on_trace=…)` emit
  `TraceEvent`s (`source.frame` · `judge.started/finished` · `gate.classified` · `incident.created`)
  correlated by one `trace_id` per frame — answers "why did THIS frame fire/ask/stay silent?".
  `percept.debug.print_trace` is a ready-made callback. Default off; a tracer that raises or is async
  can't break the pipeline.
- `RuntimeStats.frames_dropped` (bounded-queue drops) and `CircuitBreaker.snapshot()` telemetry.

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
