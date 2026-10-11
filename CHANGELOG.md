# Changelog

Notable changes to Percept.

## 1.0.0 — unreleased (pre-release builds: `1.0.0.dev1`)

Percept 1.0 is a rewrite around one job: a decision layer that turns a live video stream and
plain-language conditions into one decision per real occurrence, for an agent to act on. Cheap
triggers decide when to look, a judge (a vision-language model) decides what is true, and a pure
reducer decides whether the agent hears about it.

The 0.x modules are removed. Releases `0.7.0` and the unreleased 0.8 previews came from an
experimental line that is preserved in the project history but not continued.

### API

- `Monitor(judge, conditions, store=..., max_requests_per_min=6, max_cost_per_hour=None,
  max_concurrent_requests=4, trace=None)`; `watch(source, ack="auto" | "manual", max_unacked=100,
  fps=5)` yields `Delivery` envelopes; `poke`, `push`, `set_link`, `add`, `remove`, `pause`,
  `resume`, `snooze`, `resolve`, `feedback`, `health`, `stats`, `coverage`, `aclose`.
- `Condition` (state or event) with zones, `not_when`, confirmation, verification, rearm,
  recheck, clearing, `max_open_s`, cooldown and per-condition judge routes.
- `Decision`, the versioned wire contract (`contract_version` 1.0, `decision.schema.json`):
  `met`, `uncertain`, `cleared`, `expired`, with a stable `decision_id`, `occurrence_id`, `cause`,
  `reason`, `confidence`, `cost`, evidence references and a keyframe.
- Sources: `open_source(uri, ...)` for RTSP, webcams, HLS/DASH and files, with reconnection and
  stall detection; `PushSource` for app-owned capture. `open_source` owns capture and
  `watch(source)` owns monitoring; `watch(uri)` opens a URI with the defaults.
- Triggers: `on_change()` and `every(seconds)`, or any object with a pure `step()`.
- Judges: `percept.judges.gemini`, `claude`, `openai_compatible` (OpenAI and compatible local
  servers), `cascade`, and the scripted `FakeJudge`; `PriceModel` for cost reservation.
- Stores: `SqliteStore` (the default, `percept.db`) and `MemoryStore` (opt-in, not crash-safe).
- `percept.trace`: a JSON-lines trace of every runtime event, and `replay(path)`, which
  reproduces every decision without the video or the judge.

### Guarantees

- **Failures are never answers.** A timeout, outage, refusal or malformed reply changes no
  occurrence; it is reported as health and coverage. `min_confidence` applies to "yes" and "no"
  alike.
- **One decision per occurrence.** Confirm windows on evidence time; events verified on their
  original frames; stale answers (older epoch, condition version, occurrence or evidence)
  rejected; states held, rechecked and cleared on evidence only. The rules are in
  `docs/decision-semantics.md`.
- **At-least-once delivery.** Each decision is committed atomically with the state that produced
  it before delivery, and redelivered with the same `decision_id` until acknowledged.
- **Hard budgets.** No 60 s span holds more outbound attempts than `max_requests_per_min` per
  stream; the dollar cap reserves each request's worst case first (guaranteed for judges with a
  price). Reservations follow cascade routes. `stats()` reports settled `cost_usd`,
  `cost_reserved_usd` and per-condition `requests`, `attempts` and `cost_usd`.
- **Visible blindness.** Link outages (including app-reported ones via `set_link`), degraded
  video, scene resets, budget exhaustion, judge outages and paused conditions are blind intervals
  in `coverage()`.
- **Deterministic replay.** The decision reducer is pure; a trace replays exactly.

### Known limits

- **Static cameras only.** Moving cameras (PTZ, body-worn, vehicle, drone, phone) and screen
  content are out of scope; a camera that keeps moving fires change triggers on nearly every
  frame and the budget drops most looks.
- **No audio.** Video frames only.
- **No file lock on Windows.** The one-process-per-database and one-writer-per-trace locks use
  `fcntl`; on Windows that is up to you.
- **Blur detection on noisy, low-detail scenes.** A defocused lens with visible sensor noise can
  read as sharp, so such frames are judged rather than dropped.
- Trigger thresholds and condition defaults are starting points, not calibrated values.

### Changes during the 1.0.0.dev1 pre-release

- **Link health of app-owned sources.** `watch(source)` now subscribes the monitor to any source
  with `add_health_listener` (every `open_source` source and every `PushSource`), so a camera
  outage is a `link_down` blind interval and health event even when the app built the source with
  its own `on_frame` / `on_health`. Sources gain `add_health_listener` / `remove_health_listener`.
  One place configures a source: `open_source()` owns capture (rate, size, reconnection) and the
  source callbacks; `watch()` owns monitoring and delivery. To configure capture, write
  `source = open_source(uri, on_frame=..., fps=5)` then `monitor.watch(source)`; `watch(uri)`
  opens the source with its defaults.
- **Push stream outages.** `PushSource.set_link(up, detail="")` and
  `Monitor.set_link(stream_id, up, detail="")` report an app-owned feed as down or up, with the
  same health and blind interval as a camera; a frame pushed while down marks it up again and
  starts a new epoch.
- **Cost reporting.** `stats()` `cost_usd` is now the settled cost the judge reported. A cancelled
  request (no usage reported) settles at 0 for reporting; its worst-case reservation still counts
  against `max_cost_per_hour` for the window, and is shown as `cost_reserved_usd` (per stream and
  in total). Previously the reservation was reported as spend.
- **Per-condition cost.** `stats()["streams"][id]["conditions"]` gives `requests`, `attempts` and
  settled `cost_usd` per condition. Each provider request's usage goes to the conditions it
  carried (new `RequestUsage.condition_ids`, additive), so in a cascade a condition pinned to an
  expensive member is charged for that member; the same attribution feeds decision `cost`.
- A judge that raises is charged its whole reservation against the request limit and dollar cap
  for the window (it was charged nothing), and reports a settled cost of 0.
- A request's reply limit (`max_output_tokens`) grows with the conditions it carries (room for
  every full answer, from 400 up to 4096 tokens), so a batched reply is no longer cut off and
  failed as malformed; the cost reservation uses the same limit.
- The trace records each look given up unjudged (`kind: "drop"`: condition, subject, trigger
  time, reason `budget` / `deadline` / `ended`, time waited), besides the per-minute fate counts.
- `PushSource.set_link` and `push` are safe to call concurrently: link reports alternate strictly
  and arrive in order. Link health reported from another thread is handled on the monitor's loop.
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
- Documentation in `docs/` (quickstart, concepts, conditions, judges, sources, delivery and the
  decision semantics); examples for Frigate person events and a kitchen state on a heartbeat.
- A release workflow (wheel and sdist from a clean tree, `make validate`, PyPI Trusted
  Publishing on a `v*` tag) and a pre-publication scan, `make prepublish`.

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
- Fixes to the 0.x benchmark harness's cadence, running-answer resolution, per-performer counting
  and zero-width-window filtering. That harness and its results are not part of 1.0.

`percept/gate.py` (the three-state firing gate) is **byte-for-byte stable** across this release.
