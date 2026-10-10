# Changelog

Notable changes to Percept.

## 1.0.0 — unreleased

Percept 1.0 is a rewrite around one job: a decision layer that turns a live video stream into one
decision per real occurrence of a plain-language condition.

- New, small public API (`Monitor`, `Condition`, `Decision`), replacing the 0.x surface.
- The 0.x modules are removed from `main`. Releases `0.7.0` and the unreleased 0.8 previews came
  from an experimental line that is preserved in the project history but not continued.

### 1.0.0.dev1 — fixes found while migrating an application

- **Link health of app-owned sources.** `watch(source)` now subscribes the monitor to any source
  with `add_health_listener` (every `open_source` source and every `PushSource`), so a camera
  outage is a `link_down` blind interval and health event even when the app built the source with
  its own `on_frame` / `on_health`. Sources gain `add_health_listener` / `remove_health_listener`.
  `watch(uri, ...)` accepts `open_source` options (`on_frame`, `on_health`, `capture_fps`,
  `max_side`, `stall_timeout`, `backoff`, ...); its own `fps` stays the processing cap.
- **Push stream outages.** `PushSource.set_link(up, detail="")` and
  `Monitor.set_link(stream_id, up, detail="")` report an app-owned feed as down or up, with the
  same health and blind interval as a camera; a frame pushed while down marks it up again and
  starts a new epoch.
- **Cost reporting.** `stats()` `cost_usd` is now the settled cost the judge reported. A cancelled
  request (no usage reported) settles at 0 for reporting; its worst-case reservation still counts
  against `max_cost_per_hour` for the window, and is shown as `cost_reserved_usd` (per stream and
  in total). Previously the reservation was reported as spend.
- **Per-condition cost.** `stats()["streams"][id]["conditions"]` gives `requests`, `attempts` and
  settled `cost_usd` per condition, each request's usage split evenly (`Usage.split`) among the
  conditions it carried.
- **Reservations follow routes.** Judges may provide `max_attempts_for(request)` and
  `worst_case_cost_for(request)`; a cascade counts only the members a request can reach (a pinned
  condition its member, a verification the last member, an unpinned condition every member), each
  at its own price. Requests pinned to a cheap member no longer reserve every member's attempts.
  The per-stream attempt limit and the monitor-wide dollar cap hold as before.
- **Triggers are values.** `on_change(...)` and `every(n)` are frozen and compare and hash by
  configuration, so identical `Condition`s are equal and re-adding one is a no-op. Conditions with
  equal triggers share one trigger step.
- Validation builds the wheel from a clean copy, checks its files against the tracked tree, and
  runs the quickstart against the installed wheel.

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
