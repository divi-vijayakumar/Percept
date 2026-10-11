# Changelog

Notable changes to Percept.

## 0.9.0 — Developer Preview (unreleased)

Percept 0.9.0, the Developer Preview, is a rewrite around one job: a decision layer that turns a
live video stream and plain-language conditions into one decision per real occurrence, for an
agent to act on. Cheap triggers decide when to look, a judge (a vision-language model) decides
what is true, and a pure reducer decides whether the agent hears about it. The API may change
before 1.0.

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
- `Sink`: an interface only; no sinks ship in this release.
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

### Details

- **One place configures a source.** `open_source()` owns capture (rate, size, reconnection) and
  the source callbacks; `watch()` owns monitoring and delivery. `watch(source)` subscribes the
  monitor to the source's link health (`add_health_listener` / `remove_health_listener`), so a
  camera outage is a `link_down` blind interval and health event even when the app built the
  source with its own `on_frame` / `on_health`.
- **App-owned feeds.** `PushSource.set_link(up, detail="")` and
  `Monitor.set_link(stream_id, up, detail="")` report a pushed feed as down or up, with the same
  health and blind interval as a camera; a frame pushed while down marks it up again and starts a
  new epoch. `set_link` and `push` are safe to call concurrently, and link health reported from
  another thread is handled on the monitor's loop.
- **Cost reporting.** `stats()` `cost_usd` is the settled cost the judge reported. A cancelled
  request (no usage reported) settles at 0 for reporting; its worst-case reservation still counts
  against `max_cost_per_hour` for the window and is shown as `cost_reserved_usd`. A judge that
  raises is charged its whole reservation for the window and reports a settled cost of 0.
- **Per-condition cost.** `stats()["streams"][id]["conditions"]` gives `requests`, `attempts` and
  settled `cost_usd` per condition. Each provider request's usage goes to the conditions it
  carried (`RequestUsage.condition_ids`), so in a cascade a condition pinned to an expensive
  member is charged for that member; the same attribution feeds decision `cost`.
- **Reservations follow routes.** Judges may provide `max_attempts_for(request)` and
  `worst_case_cost_for(request)`; a cascade counts only the members a request can reach, each at
  its own price.
- **Reply limit.** A request's `max_output_tokens` grows with the conditions it carries (from 400
  up to 4096 tokens), so a batched reply is not cut off; the cost reservation uses the same limit.
- **Triggers are values.** `on_change(...)` and `every(n)` compare and hash by configuration, so
  identical `Condition`s are equal and conditions with equal triggers share one trigger step.
- The trace records each look given up unjudged (`kind: "drop"`, with the reason `budget`,
  `deadline` or `ended`), besides the per-minute fate counts.
- Documentation in `docs/`; offline examples for a quickstart, Frigate person events and a
  kitchen state on a heartbeat; a benchmark harness in `benchmarks/` (no results are published
  for this release).
- Tooling: structural guardrails, `make validate` (clean-copy wheel build and an installed-wheel
  quickstart), a pre-publication scan (`make prepublish`) and a tag-triggered release workflow
  using PyPI Trusted Publishing.

### Known limits

- **Static cameras only.** Moving cameras (PTZ, body-worn, vehicle, drone, phone) and screen
  content are out of scope; a camera that keeps moving fires change triggers on nearly every
  frame and the budget drops most looks.
- **Small, distant or brief activity can be missed.** In busy surveillance scenes the change
  trigger often does not fire on such activity, so the judge is never asked. Pair these
  conditions with a heartbeat (`every`), a tighter zone or a lower `on_change` threshold, and
  check `coverage()`.
- **Slow providers can time out.** Each judge request has an 8 s deadline; sparse requests to a
  slow provider can exceed it. They fail as `timeout`, are reported on health and coverage, and
  change no decision.
- Trigger thresholds and condition defaults are starting points, not calibrated values;
  confidence is the judge's and is advisory.
- **No audio.** Video frames only.
- **No file lock on Windows.** The one-process-per-database and one-writer-per-trace locks use
  `fcntl`; on Windows that is up to you.
- **Blur detection on noisy, low-detail scenes.** A defocused lens with visible sensor noise can
  read as sharp, so such frames are judged rather than dropped.
- No performance numbers (accuracy, latency or cost) are published for this release.

## Earlier releases

0.7.0 and earlier: a research preview with a different API; not continued.
