# Changelog

Notable changes to Percept.

## Unreleased

### Fixed

- An event entry absorbed while an earlier candidate was verifying is no longer lost when that
  candidate closes without `met` (verify "no", verify "unknown" ending `uncertain`, or the verify
  deadline). The key keeps the absorbed detection (request id, exact evidence interval, stream epoch),
  and the runtime judges that detection again on exactly its own evidence: the frames its request
  judged (kept for one absorbed request per key), else the pre-roll's frames of exactly that interval.
  Never other frames, so recovery does not depend on the person still being in view, nor on the
  verification finishing within the pre-roll. At most one recovery per closed candidate, an ordinary
  event detect under the request budget (trigger label `relook`), in a slot of its own that an
  ordinary look cannot absorb or replace.
- When the recovery cannot review the entry (frames and exact interval gone, the epoch changed by a
  reconnect, budget, deadline, a stale or failed answer) it is reported as a `relook_dropped` blind
  interval over the absorbed interval, with the reason in `detail`, and no decision is emitted. The
  trace has a `relook` record.
- `BlindKind` gains `relook_dropped`: consumers that match blind-interval kinds exhaustively must
  handle it.
- Snapshots: a key carries `absorbed_request_id`, `absorbed_interval` and `absorbed_epoch` only while a
  verifying candidate holds an absorbed detection. `SNAPSHOT_FORMAT` stays 1. A 1.0.0 reader loads every
  such snapshot except one taken in that moment, which it rejects as malformed: do not downgrade from
  1.0.1 to 1.0.0 with persisted state.

## 1.0.0 — Developer Preview

Percept 1.0 is an open-source runtime for continuous vision applications: configure what to
inspect, how to escalate to vision models, how much inference to spend and when to deliver a
decision, then evaluate those choices on your own clips. The runtime API is the 1.0 contract;
watch plans and the evaluation toolkit (`percept.plans`) are experimental. Code written for 0.9.0
runs unchanged.

### API

- `percept.clips.PreRoll` is supported API, imported from `percept.clips` (not exported from
  `percept`): pass one as `preroll=` to `open_source` or `PushSource` to give event evidence a
  longer or denser window than the default 8 s at 2 frames a second.

- `FakeJudge(..., record_requests=False)` keeps no record of the requests it serves (`sent` is
  None, as on the vendor judges). Recording stays on by default. A request's images are views of
  the source frames, so each record kept whole frames alive: a long offline run grew without
  bound. The benchmark's offline oracle no longer records.
- Evaluation driver `percept.plans.run` (experimental; reached as a module, no new top-level
  names): `attention_only(compiled, labels, clips, video_root=)` runs the Monitor's attention step
  over labelled clips with no judge calls and records the looks made due as `Fire`s (the candidate
  stage); `run(...)` runs a Monitor per clip with the plan's judge behind `CachingJudge` (a hard
  dollar cap that reserves the worst case and refuses with `budget`, a per-request spend log, and
  a reply cache keyed on the rendered prompt, images and judge identity; a cached reply replays
  its original usage and latency, so an identical re-run decides the same and pays nothing:
  `cost_usd` stays the live-equivalent cost, `paid_usd` is what was paid; a run the cap cut short
  is scored `capped`) and turns decisions and the trace into a
  `ClipRun`; `sweep(configs, ...)` scores the configurations you declare on dev or validation in
  your order, without ranking them; `register(...)` freezes a plan (plan, manifest and labels
  hashes, provenance) and `score_split(split=test, ...)` refuses the test split without a
  matching registration; with an optional `ledger` file, a test split is scored once per
  registration unless `rescore=True`. `examples/plan_evaluation.py` runs the whole flow offline. The
  benchmark's `CostGuard` is now `CachingJudge` with the cache off; its behaviour is unchanged.
- Evaluation in `percept.plans` (experimental): `load_labels(doc)` checks ground truth for your
  own clips (the format of the benchmark's `local:` datasets, plus an optional recording `group`)
  and raises `LabelError` with the field path; overlapping occurrences of a label merge;
  `Labels.labels_hash` is the same across checkouts. `split(labels, durations)` freezes clips into
  dev, validation and test by `sha256(group) mod 100`, so adding clips never moves existing ones,
  and returns a `SplitManifest` with clips, occurrence counts, hours and a `manifest_hash`.
  `evaluate(labels, runs)` scores one run (`ClipRun` per clip) and returns `Score`: TP/FP/FN,
  recall, precision, false alerts per hour and per 100 occurrences, misses attributed to a stage,
  the funnel, API cost and local compute seconds per correct event, and seeded bootstrap 95%
  intervals. It reports; it never picks between runs. The benchmark harness now scores through it.
- `percept.attention.build(conditions, default_change=, default_heartbeat=, paused=)` returns the
  trigger groups and labels the Monitor uses, and `attention.PRIORITY` the queue priority of each
  kind of look, so an offline evaluator needs no private import. The Monitor calls it; behaviour
  is unchanged.
- `Monitor.ready(stream_id, timeout=None)`: wait until a stream is registered (by `watch()` or a
  first `push()`), so a detector in another task can `poke()` it without racing the watch.
  Returns at once for a registered stream; raises `RuntimeError` if the monitor closes first and
  `asyncio.TimeoutError` on `timeout`. `examples/frigate_semantic.py` uses it.
- `percept.plans` (experimental; the schema may change before it is stable): a watch plan, one
  plain-data document (`percept.watch/v1alpha1`) with sources, budget, judges and concerns.
  `plans.compile(plan, routes)` checks it, raising `PlanError` with the dotted path of the field,
  and returns `Compiled`: the conditions, `monitor_kwargs` for `Monitor(**...)`, the
  `external_event` map for `poke`, and a `plan_hash` of every effective setting. `explain()`
  lists each setting with its origin (plan, SDK default or fixed by the runtime),
  `feasibility()` checks periodic demand against the request budget, and `provenance()` /
  `trace_record()` name the SDK version, judges and prompt hash, without secrets. Example:
  `examples/watch_plan.py`.
- Combined triggers: `Condition.trigger` also takes a tuple of triggers, meaning any of them (the
  condition is due when any member fires), for example `(on_change(), every(60))`. Equal
  triggers still share one state across conditions, and every look draws on the stream's one
  request budget. In a watch plan, several `on_change`/`every` entries in `attention.any` now
  compile to a combined trigger (members in a canonical order, so the entries' order does not
  change `plan_hash`); `feasibility()` counts each `every` member.
- `Monitor.coverage()` adds `triggers`: per condition and per trigger label (`on_change(0.2)`,
  `every(5)`, a custom trigger's class name, or `poke`, `settled`, `recheck`, `verify`), the
  looks that were `due`, `admitted`, `coalesced`, `dropped_budget` and
  `dropped_for_verification`. `stats()` adds the same totals per label under each stream's
  `triggers`. The existing `fates` are unchanged.
- Per-condition evidence windows: `Condition.clip_s` and `Condition.clip_frames` (events only)
  set how many seconds before the trigger an event is judged on and how many frames (1 to 16)
  are spread over them, the newest always included. `None` keeps 4 seconds and 4 frames.
  Verification on the original interval re-judges exactly the frames the detection used. A
  window longer than the stream's pre-roll (8 seconds unless the source has its own `PreRoll`)
  is refused with `ValueError` when the condition or the stream is added. When the pre-roll
  holds less than the window (the stream just started or reconnected: under 90% of `clip_s`, or
  fewer frames than `clip_frames`), the event is judged on what is held and counted in
  `coverage()["evidence_short"]` per condition and in `stats()`, apart from `fates`; the trace's
  `request` record gains `evidence_s`, `clip_s`, `frames`, `clip_frames` and `evidence_short`.
  Trace `conditions` records carry the two fields only when set; older traces replay unchanged.
  In a watch plan, `judgment.evidence.pre_seconds` (at most 8) and `judgment.evidence.frames`
  compile to them (a state with either is a `PlanError`); plans without them keep their
  `plan_hash`.

### Changed

- Verification comes before new detection. While a verification waits (queued, or between its
  retries), a detection may not take the request attempts the verification needs; a detection
  held that way past its deadline gets the new fate `dropped_for_verification` (trace drop reason
  `verification`) instead of `dropped_budget`. Nothing is held when no verification waits.
- Per-attempt judge timeouts: with a retry left, an attempt may use at most 60% of the time
  remaining in the request's deadline (`ATTEMPT_SHARE` in `percept.judges.policy`), so a hung
  attempt is abandoned in time for the retry to run. An attempt that never replied is charged at
  its worst-case cost when the judge has a price.

### Benchmarks

- A run the machine slept through is not publishable: each arm records the seconds the machine
  slept (`slept_s`), and on macOS the runner keeps the machine awake with `caffeinate`.
- The harness stops an arm once its failed looks already exceed the publishable failure rate for
  the most looks it can make (status `aborted_failed_looks`), and stops the run on a sustained
  provider outage (status `aborted_provider_outage`). The manifest's `status` names how the run
  ended (`complete`, `aborted_spend_cap` or one of these); an aborted run is never complete.
- Detector-and-tracker baseline arms `detector_track_rule` (a zone-entry rule on person tracks)
  and `detector_track_decide` (the same tracks through Percept's decision rules), for entry
  conditions only.
- `percept-plan-<name>` arms run one compiled watch plan per source from `--percept-plans`.
- The offline oracle judge runs with `FakeJudge(record_requests=False)`.

### Fixed

- A verification cancelled by its deadline before its request task first ran (a scheduling race,
  seen under heavy load) never returned its request slot or budget reservation, and its stream
  never finished: the watch hung, and the leaked slot could hold back other streams. Every request
  is now settled exactly once however its task ends (finished, failed, cancelled mid-call or before
  it ran), charging unknown usage at its full reservation as before. A verification whose deadline
  has already passed is no longer started. Decisions are unchanged.

## 0.9.0 — Developer Preview (2026-10-11)

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
- Documentation in `docs/`, including an integration recipe (`docs/integration.md`) and an
  `llms.txt` index; offline examples for a quickstart, Frigate person events, a kitchen state on a
  heartbeat and a webhook integration (`examples/webhook_agent.py --offline`); a benchmark harness in `benchmarks/` (no results are published
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
