# Changelog

Notable changes to Percept. From **1.0.0** the public surface follows semantic versioning: the `watch()`
seam (`WATCH_VERSION`) and the wire contract (`CONTRACT_VERSION`) are versioned independently of the
package, and same-MAJOR changes are additive. See the README "What can you rely on?" table for the
per-surface stability guarantees.

## 1.0.0 — first stable release (2026-07-23)

The first stable public release. `pip install percept-vision` (`import percept`); the core is pure
standard library and runs offline with deterministic fakes. Highlights of the surface as of 1.0.0:

- **Wire contract at `CONTRACT_VERSION = "1.10.0"`** (additive-only across the pre-1.0 line). 1.10 makes
  the judge operational: `WatchPlan.judge` + `WatchPlan.judge_context` (an `ArtifactBinding` bound to an
  immutable, SHA-256-verified static context) and judge provenance. The `watch()` entrypoint is at
  `WATCH_VERSION = "1.1.0"`; the adapter conformance suite at `CONFORMANCE_VERSION = "1.1.0"`.
- **Batteries-included runtime** — `percept.run(source, goals, vision) → Incident` and
  `percept.RuntimeSession` (the command seam) open a camera/video/RTSP source, run the tier-0 gate, judge
  only on escalation, and deliver on the rising edge.
- **Layered configuration with provenance** — the `Settings` tree, versioned `SourceProfile` /
  `WatchProfile`s, and the `ConfigResolver` precedence chain (ADR-008).
- **Offline quality rail** — `percept-eval` (differential A/B), `percept-optimize` (locked-config
  search), `percept-derive-profile`, `percept-bench` (per-concern P/R + scorecard), and the model-free
  `percept-doctor` / `percept-replay` / trace visualizer. No Percept accuracy numbers are published yet
  (methodology + runbook only — see `docs/evaluation/running-benchmarks.md`).
- **Removed: the `[screen]` extra.** There is no built-in local screen-capture source in 1.0 —
  `screen://` is not a capture URI and opening one raises — so an extra that installed a screen grabber
  the runtime could not use was advertising a capability that does not exist. Watching a screen is
  supported through **client-pushed frames** (grab the display yourself, yield
  `(t, gate_frame, judge_frame)` into `source=`): see `docs/watches/scenario-screenshare.md`. `screen://` is
  unchanged as a **source-profile inference hint**.
- **`Development Status` is `4 - Beta`, not `5 - Production/Stable`.** The public API and the wire
  contract are stable and semver-governed from 1.0, and the deterministic mechanics are qualified; what
  is not qualified is provider-dependent behaviour — no reflex accuracy, recall, or end-to-end latency
  figure has passed its gate, and the audio-only and fused paths ship as **preview** with no
  provider-quality claim.

The detailed additive changes that accumulated on the pre-1.0 development line are recorded below.

### Contract — wire version 1.2.0 → 1.3.0 (additive; same MAJOR, wire-compatible)

### Contract — wire version 1.2.0 → 1.3.0 (additive; same MAJOR, wire-compatible)
- **`Incident.entity_id`** — the gated subject's correlation handle now SURVIVES the emission
  boundary. The rising-edge gate has always keyed on `(goal.id, focus.id)` and `FireEvent.entity_id`
  has always carried it, but `make_incident()` never read it: identity was computed upstream and
  dropped at emission. Downstream, two independent subjects tripping the same concern were
  indistinguishable, which reads as spurious out-of-order / duplicate divergence (a client saw three
  fabricated `out_of_order` divergences on a walkthrough with no defect).
  - `Incident.entity_id: str | None = None` — appended LAST on the dataclass, so every existing
    positional `Incident(...)` build keeps its meaning. `None` = not entity-scoped (single-subject
    streams, boundary fires) — the back-compatible default.
  - `make_incident(fire, ...)` reads `fire.entity_id` (duck-typed, like every other field it reads);
    an explicit `entity_id=` kwarg overrides. A legacy fire with no such attribute yields `None`.
  - Flows through the existing generic serializers unchanged: `incident_payload()` / the webhook sink
    emit `entity_id`. The `incident` health event and the `incident.created` trace now carry it too,
    and `percept watch` prints `entity=<id>` when the fire was entity-scoped.
  - **JS twin** (`packages/percept-edge/src/contract.js`): `makeIncident` defaults `entity_id` to
    `null` and validates string-or-null; `parseIncident` carries it off the wire and tolerates its
    absence on a pre-1.3 payload. The JS `CONTRACT_VERSION` now names the Python contract version it
    twins (`1.3.0`); it implements a subset of the DTOs — the version tracks the wire, not coverage.
  - JSON Schema regenerated (`percept/contracts/schema/*`): `entity_id` is optional + nullable, never
    `required`.
  - **Scope**: `entity_id` is a GENERIC correlation key. It makes no claim about *who* or *which
    document* the subject is — richer subject/incident relations are ADR-006, not this change.

### Contract — wire version 1.0.0 → 1.1.0 (additive; same MAJOR, wire-compatible)
- **`AttentionIntent`** — a new frozen DTO carrying the **effective** (post-policy), channel-neutral
  attention request a fired incident makes: `message`, `goal_id`, `incident_id`, `ts`, `preempt`
  (`interrupt`/`queue`/`drop`), `ttl_s`, `verbatim`, and an optional `preferred_channel` **hint** (not a
  modality enum — the SDK does not prescribe application channels; the app decides speech/UI/webhook/action).
- **`GoalOrigin`** enum + **`Goal.origin`** field — durable goal PROVENANCE
  (`structured_trusted`/`language_authored`/`restored`/`system_template`). Stamped by the SDK entry point
  that admits the goal (`Percept.add_goal` → `structured_trusted`; conductor GoalOps / `arm_from_language`
  → `language_authored`), **overwriting** any caller-supplied value — a caller cannot claim interrupt-trust
  by filling a field. Persisted with the goal, so trust survives `snapshot`/`restore` +
  `save_state`/`load_state`; a pre-provenance snapshot restores as neutral `restored`. Additive + defaulted
  (`language_authored`, the least-privileged), so an existing goal round-trips unchanged.
- **`AttentionPolicy`** (in `percept.incidents`, not the frozen membrane — it is behaviour, not a wire
  shape) + `attention_intent_for(incident, goal, policy)` + `parse_requested_delivery(delivery)`: the
  single, deterministic requested→effective clamp. A requested `preempt="interrupt"` is clamped to `queue`
  unless the goal's origin is trusted (`structured_trusted`/`system_template`), its consequence class is
  allowlisted, or an explicit per-goal allowlist entry matches. `parse_requested_delivery` is now the one
  interpreter of the legacy `goal.state['delivery']` dict (`preempt`/`ttl_s`/`verbatim` semantics unchanged
  from `RuntimeSession._speak_fire`, plus the additive `preferred_channel` hint).
- JSON Schemas regenerated (`percept/contracts/schema/`), including the new `AttentionIntent.schema.json`
  and `Goal.origin`; `CONTRACT_VERSION` bumped to `1.1.0`.

### Runtime — session seams (public, additive)
- **`RuntimeSession.on_goal_change(cb) → unsubscribe`** — fires on EVERY goal-set change, subscribed at
  the authoritative `GoalRegistry` revision-change point (not `arm`/`disarm`), so it catches the
  `arm_from_language`/`apply_goal_ops` command path, restore, and same-id replacement — the paths a
  callback wired to `arm` would silently miss. Purpose: grounding-snapshot invalidation. Callbacks are
  additive + isolated (one raising subscriber can't break another).
- **`RuntimeSession.list_goals()`** — a bounded public read of active goals; each entry now carries
  `{id, condition, status, origin, delivery}` where `origin` is the `GoalOrigin` wire string and
  `delivery` is the goal's parsed **requested** delivery (`{preempt, ttl_s, verbatim}`). Additive: the
  original `id`/`condition`/`status` keys are unchanged.
- **`RuntimeSession.goal_view(goal_id) → dict | None`** — resolve a single goal by id to that same
  bounded read-only view, so an application can map an `Incident` back to its goal's authored delivery
  (for `attention_intent_for`) without reaching into `core.goals`. `None` for an unknown id; never leaks
  the `Goal` object.
- **`percept.agent.guard_live_language_path`** — public rename of `_guard_live_language_path` (the
  underscore name is kept as an alias for one release), so applications import no underscore symbol.

### Runtime — SDK voice diet (the voice embodiment moves app-side) — BREAKING
The application embodiment plane (voice transport, floor arbitration, fire delivery, A2V metrics) left the
SDK. The perception package no longer owns the voice stack; the one working voice application now lives in
`templates/realtime_voice_watch_agent` (`cd templates/realtime_voice_watch_agent && pip install -e .`),
composed through the SDK's public seams only.
- **`RuntimeSession.attach_transport(...)` and `RuntimeSession.deliver_fire(...)` are now raising stubs**
  (`RuntimeError`) that point at the template — not delegating shims. **Migration:** copy the template;
  wire a `VoiceEngine` to the Floor via its `PerceptBridge.attach`, and route fires with
  `percept.start(..., on_incident=bridge.on_incident)` → `attention_intent_for` → the template's `Floor`.
  Also removed from the session: `detach_transport`, the `arm_goal`/`watch`/`stop_watching`/`whats_armed`
  tool schemas + handlers, the brain-state pulse, and the A2V-latency plumbing — all reproduced,
  app-side, in the template's `percept_bridge.py`/`floor.py`/`telemetry.py`.
- **`percept.realtime` is removed** — `percept/realtime/{transport,server}.py` are deleted and
  `import percept.realtime` now raises `ImportError` (for one release) naming the template. The Pipecat +
  serverless-WebRTC + Gemini-Live wire (`PipecatRealtimeTransport`, `serve_realtime`, `build_realtime_app`)
  moved verbatim into the template's `engine_adapter.py`/`media.py`, re-expressed against the template's
  own `VoiceEngine`/`SpeechRequest` seam.
- **The `percept-vision[realtime]` extra is emptied and RESERVED.** Its voice dependencies (`pipecat-ai`,
  `fastapi`, `uvicorn`, `google-genai`, …) now live in the template's own `pyproject.toml`; the SDK ships
  none of them, and nothing pulls a voice vendor at `import percept`/`import percept.runtime` (pinned by
  `tests/test_voice_diet_boundary.py`). The extra name is kept as a placeholder reserved for future
  **SDK-side streaming judgment**, not voice.
- **`VoiceTransport` / `UtteranceEvent` / `Reply` are deprecated** in `percept.contracts` (docstring
  deprecation notes; shapes unchanged, still functional this release as a migration seam). No SDK-core
  behavior consumes them after the diet; the template owns its own `VoiceEngine`/`SpeechRequest` types.
  Removal from core contracts is targeted at v0.7/v1.0 unless an SDK-level consumer appears.

### Runtime — judge dispatch moves off the capture loop (default `judge_dispatch="async"`)
- `percept.runtime.run` / `percept.runtime.start` no longer await the cloud judgment inline at the
  admit site. A single-flight scheduler (one judgment in flight + one latest-pending per source)
  drives `perceive_judged` off-loop, so a slow VLM turn no longer stalls capture, the tier-0 gate,
  frame subscribers, or the absence/deadline checks. The admit condition itself is unchanged.
- **Emission latency note:** fires can now trail their admitting frame by up to ~1 frame interval
  plus the judge duration (`incident.ts` remains the admission timestamp, not completion time).
  Because the pre-roll buffer keeps rolling while the judgment is in flight, incident snippets are
  assembled slightly *newer* than before (the pre-roll window ends nearer the emission instant).
- Under overload, newly admitted frames replace the queued candidate (latest wins —
  `RuntimeStats.judge_superseded`); a candidate admitted before a goal change is discarded rather
  than judged under the new watch plan (`RuntimeStats.judge_revision_dropped`). The budget
  invariant is `judge_calls + judge_superseded + judge_revision_dropped == escalations`.
  `perceive_judged(expected_goals_revision=…)` aborts a stale in-flight judgment before any
  gate/disposition mutation. A `RuntimeHealthEvent("judge_dispatch")` announces the mode once;
  escalation health events carry `judge_active`/`judge_pending` (each 0/1) for live observability.
- **Escape hatch (one release):** pass `judge_dispatch="inline"` to `run(...)` for the byte-for-byte
  historical loop (fires emitted before the next frame is read). The benchmark harness loop
  (`percept.harness.tier0.GatedWatchLoop`) intentionally stays inline; the divergence is filed in
  `tests/test_harden_e1_loop_parity.py`.

### Contract — wire version 1.0.0 (initial public contract; consolidates pre-release surface that iterated internally through 1.1–1.3 but never shipped)
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
  `@percept/edge` `makeTier0Signal`/`parseTier0Signal` (JS contract twin at 1.0.0); the field is
  optional, so a payload omitting the key parses unchanged.
- JSON Schemas regenerated (`percept/contracts/schema/`), including the new `ClipEvent.schema.json`.

### `@percept/edge` (npm) — 0.6.0 → 0.7.0
- **`PrivacyScreen`** — a new on-device, multi-modal privacy classifier on the *egress* boundary (sibling
  to `Edge`), filling the previously-inert `semantic` Tier-0 channel with an owned safety verdict in the
  existing `allow`/`redact`/`drop`/`confirm` vocabulary. Text tier (deterministic regex + Luhn/ABA/SSN
  checksums) is full; image tier is an adapter **slot** (app supplies the ONNX model — zero-dep preserved);
  audio tier is a coarse, bounded acoustic head (provisional). Ships with an eval harness + CI-gated
  scorecard (`npm run eval`). No wire-contract change — `CONTRACT_VERSION` stays 1.0.0. See
  `packages/percept-edge/CHANGELOG.md`.

## 0.6.0 — Research preview (2026-06)

### Packaging — one install
- `percept-contracts` and `percept-harness` are now **bundled inside `percept-vision`** as the
  `percept.contracts` and `percept.harness` subpackages. Install one package:
  `pip install percept-vision`. Heavy deps stay opt-in via extras:
  `[gemini] [claude] [openrouter] [deepgram] [runtime] [harness] [pose] [bench] [audio] [otel]`.
- **Migration:** `from percept_harness.X import Y` → `from percept.harness.X import Y`;
  `from percept_contracts import Z` → `from percept.contracts import Z`.
- **Deprecated:** the standalone `percept-harness` PyPI package is deprecated in favor of
  `percept-vision (>=0.6.0)`. `percept-contracts` was never published standalone.
- The wheel ships **only the `percept` namespace** — the benchmark harness (top-level `eval`) is
  repo/dev tooling and is no longer installed (no `eval` name squat in site-packages). `percept-bench`
  from a bare wheel install prints a "run from a checkout" note instead of importing it.

### Versioning
- Package version is `0.6.0`. The **wire-contract** version is separate: `CONTRACT_VERSION = "1.3.0"`
  (additive-only). A `percept-vision 0.6.0` install exporting contract `1.3.0` is expected, not a bug.

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
  timespeak zero-width-window filter. See `docs/evaluation/methodology.md`.

`percept/gate.py` (the three-state firing gate) is **byte-for-byte stable** across this release.
