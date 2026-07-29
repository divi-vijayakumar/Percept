# Changelog

## Unreleased

### Fixed
- **Transition goals no longer miss the rising edge when the in-focus entity changes (P06-002).** An
  A→B transition goal's arm/fire latch was keyed by the volatile `focus.id`, so the moment the
  in-focus entity changed between the arming (A) observation and the completing (B) observation the
  two landed under different keys and the edge was silently dropped. The latch is now keyed by the
  concern's **declared identity policy**, defaulting to a stable `"scene"` scope — a global
  workflow-state goal latches globally and fires across a focus flip, while a goal that declares a
  per-entity track/appearance policy keeps per-entity isolation. Consequently the `entity_id` on the
  `transition.decision` trace stage (and on the resulting Incident) now reports the **latch key**; the
  raw in-focus id is preserved in a new additive `focus_entity_id` field, alongside `identity_policy`,
  `latch_key`, `latch_key_source`, and the `from_`/`to_` confidence+evidence behind the decision.
- **Transition judge calls are now metered and fail-closed (P06-002).** The transition path's two
  judge calls (`from`/`to`) went to `adapter.judge` directly, bypassing the `_timed_judge` cost/latency
  wire and the `_safe_timed_judge` fail-closed boundary. Both calls now route through that boundary, so
  they record the same `judge.latency`/usage telemetry as the watch path and a timeout/malformed
  verdict degrades the tick (`judge.degraded`, a new `judge_degraded` outcome on `transition.decision`)
  instead of crashing the run or phantom-firing. Contract (`CONTRACT_VERSION` 1.10.0) is unchanged —
  the new trace fields are purely additive observability.

### Added
- **`percept-watch --trace PATH` + a `percept-replay` command — ADR-004's normative replay is now
  REACHABLE (card Z5).** `percept.replay.replay_cognition` was good, tested code consuming an
  artifact **no user of the SDK could produce**: `Percept.on_trace` defaults to `None`,
  `percept.run(on_trace=None)`, `percept-watch` contained no trace reference at all, and
  `percept-doctor` builds a `TraceRecorder` that records only the tier-0 observation/admission
  stages — never `gate.classified`, the one stage `replay_inputs` consumes. So §2 of ADR-004
  defined replay normatively and nothing shipped could exercise it.
  - `percept-watch --trace out.json` builds a `TraceRecorder` whose `RunTraceManifest` carries the
    live Gate's knobs (`percept.replay.gate_config_of` → `GATE_CONFIG_KEYS`) and wires
    `core.on_trace = rec.tracer()` (plus `on_health=rec.health_sink()` on the live path). Recording
    the gate_config is the load-bearing part: without it the replay honestly reports
    `gate_config_recorded=False` and runs against library defaults instead of against *this* run.
    The trace is written even on Ctrl-C — a killed run is exactly the run you want the trace of.
  - `percept-watch --offline [--fixture NAME]` is the keyless acceptance path: no API key, no
    camera, the bundled deterministic fixture (`percept.offline`) driven through the frozen
    `watch()` seam. It proves WIRING, not intelligence.
  - `percept-replay TRACE` (new console script → `percept.replay:main`) loads a trace JSON **or** a
    portable `.ptrace` directory/archive (`load_replayable`) and prints the `ReplayReport` — digest,
    gate outcomes, incidents, whether the gate config was recorded, and every divergence. The **exit
    code is the contract**: `0` = every replayed outward action matched the recorded run, `1` = at
    least one diverged, `2` = the artifact could not be loaded. `--json` emits the report;
    `--escalates`/`--suppress-false` declare the per-goal behaviour a trace cannot know.
  - `tests/test_replay_cli.py` pins the whole round trip: `percept-watch --offline --trace t.json`
    then `percept-replay t.json` reports **zero divergences**, the trace actually contains
    `gate.classified` records carrying their verdict inputs, the manifest carries all four gate
    knobs, the digest is stable across the file round trip, a bundle replays to the same digest as
    its JSON, and a tampered gate config makes the CLI exit non-zero.
  `percept/watch.py`'s signature and `WATCH_VERSION` are untouched — all of this is additive wiring
  behind the existing `percept=` / `on_trace` seams.

### Fixed
- **The "~7 MB/s memory leak" is a BOUNDED plateau, and the bound is now in BYTES and loud (card
  Z1).** Two client teams measured RSS 92 → 342 MB (and 405 MB on a second clip, at HALF the judge
  calls) on 36-second clips and concluded percept could not run for an hour. Profiled, not reasoned
  about: `percept.runtime.worker.run` driven offline with a scripted judge over 8,000 frames, RSS +
  `tracemalloc` + a weakref census of live decoded frames. Result: the Python heap is **flat** at
  46.3 MB from frame ~200 to frame 8,000 and RSS **plateaus** (282.8 MB, unchanged over the last
  5,600 frames) — 0.00 KB/frame of heap growth. There is no leak. The retaining references are
  exactly two, and they are rings:
  `percept/runtime/worker.py` `clips = FrameWindow(size=snippet_size)` (48 decoded frames, the
  incident snippet's pre-roll) and `percept/agent.py:635` `FrameWindow(settings.events.window)`
  (12 frames). They hold the SAME objects, so steady state is `max(48, 12) + ~2 in flight` ≈ 50
  decoded frames — at 1080p RGB, ~300 MB. That is the clients' number, and it explains why the
  second clip cost MORE at half the judge calls: the plateau scales with frame SIZE, not with spend.
  The measurement error was reading a fixed high-water mark on a clip that ended before the plateau.
  The real defect it exposed: **a count bound is not a memory bound.** `FrameWindow` now takes an
  optional `max_bytes` budget (`percept/events.py`, with `frame_nbytes` measuring numpy/bytes/
  harness-`Frame`/PIL shapes), and the pre-roll ring is bounded by
  `settings.events.preroll_max_bytes` (`PERCEPT_PREROLL_MAX_BYTES`, default **128 MB** — still the
  full 5 s pre-roll at 4 fps / 1080p) or `run(snippet_max_bytes=…)`; `0` restores the count-only
  ring. Per ADR-001 §9 the drop is never silent: each byte-bound eviction emits
  `RuntimeHealthEvent("preroll_capped")` with `reason` / `frame_bytes` / `held_bytes` /
  `max_bytes` and bumps `stats.preroll_frames_dropped`. The newest frame is never evicted, and a
  frame whose size cannot be measured (`frame_nbytes → 0`) never causes a drop — no eviction on a
  guess. `tests/test_soak_memory.py` is the fence: it fails if per-frame retention grows across a
  run, if the budget stops bounding, or if an eviction goes unreported (verified to bite by
  injecting a real per-frame leak).
- **`golden_eval.score` was KEY-BLIND — every multi-goal recall/precision number is over-credited
  (card X1).** The one-to-one greedy §5a matcher selected a claimant episode purely by
  `ep.contains(decision_emitted_ts)` and never compared the prediction's `JudgmentEvent.key` to
  `GroundTruthEpisode.episode_id` — the correspondence its own docstring asserted. On a MULTI-GOAL
  stream a fire of the WRONG goal landing inside another goal's window was therefore scored as a
  MATCH: recall inflated, false fires hidden, misses erased. Both claimant predicates now carry a
  key-equality term (`str(pred.key) == ep.episode_id`); determinism, the sort order, and the
  duplicate / false-fire branches are otherwise untouched, and single-goal scoring is byte-for-byte
  unchanged. This instrument sits under `percept eval`, the L1 derivation ladder, RB-FRONTIER and
  the visualizer's drift lane, so **every previously reported multi-goal recall/precision figure
  must be re-measured** — including the internal H0 ablation numbers, which were measured on
  multi-goal streams with goal density as an ablation axis.
  Consequences of the fix: the visualizer's drift lane now keys predictions by the trace record's
  `goal_id` (a ground-truth `episode_id` must be the `goal_id` of the goal it labels) and says so
  in the lane note when no fired goal id intersects any episode id, instead of silently reading
  `recall 0.00`.
- **Cost was a structural FALSE ZERO — a traced run now reports what it actually spent (card X2).**
  `TraceSummary` reported `{"input_tokens": 0.0, "output_tokens": 0.0, "usd": 0.0,
  "usage_reported_calls": 0}` for **every** run, including one that measurably spent 134,946 input +
  5,231 output tokens across 108 judge calls: the token counts existed in the Funnel
  (`observability.judge_usage` → `Funnel.record_judge`) but nothing carried them to the trace/cost
  surface, and `TraceRecorder.record_usage` / `record_verdict_usage` / `slo.price_usage` had **zero
  callers**. A client with a `cost_usd` bar got a green light on a run that spent real money.
  `Percept._timed_judge` — the one site every judge call passes through, inline and async dispatch
  alike — now emits a `judge.usage` trace event alongside the funnel record, and `TraceRecorder`
  prices it via the new `slo.price_usage_or_none`. `TraceSummary.dollars_per_stream_hour` makes the
  rate readable off a real run.
  **Unknown is now typed as unknown.** `input_tokens` / `output_tokens` / `cost_usd` are
  `float | None`: `None` (rendered `—` by `cost_text()` / `tokens_text()`, by `percept doctor` and by
  the visualizer's cost lane) whenever no judge call reported usage, or when tokens cannot be priced
  honestly. A model absent from `slo.PRICE_TABLE` yields **no dollar figure at all** rather than a
  `DEFAULT_PRICE`-derived guess presented as a real price. The one licensed zero is a run that made
  no judge call (`percept doctor`, a fully gated stream), where `$0` is measured. `— (unknown)` is
  safe; `0.0` is a wrong number that a bar can pass. See `tests/test_trace_cost.py`.
- **`plan_hash` now covers the RESOLVED thresholds and policy, not just the goals (card X3).**
  A threshold A/B — the commonest paired eval on this harness — declares the *same goals* in both
  arms by construction, so both arms printed the **same** `plan_hash` in the "quote these identities"
  block. That defeated the hash's only job: letting a reader prove which configuration produced which
  numbers. `PlanIdentity.from_goals` takes a `policy=` mapping of the resolved knobs
  (`motion_gate` / `motion_gate_block_max` / `motion_gate_p95` / `salience_stat` / gate threshold /
  `require_false_first` / `refractory_s`, plus per-goal `dwell` from `state["hold_s"]` and the
  resolved `plan_mode`) and folds it into the hashed payload. `percept eval` fills it from each arm's
  `thresholds`/`policy` blocks and `percept doctor` from the same `Settings` fields the worker
  compiles with, so an eval's hash stays comparable with a production run's. The hash stays canonical
  and **order-independent** — sorted keys, `None`-pruned, ints and floats unified, nested blocks
  flattened to dotted keys — exactly the content-addressing contract
  `percept.profile_versioning.content_hash` already holds for profiles. The report prints the hashed
  policy under each arm's hash, and says so explicitly when two arms *do* share one identity.
  Additive: `PlanIdentity` gains a `policy` field (a sorted tuple, so it stays frozen and hashable)
  and `RunTraceManifest.create` a `plan_policy=` pass-through. `percept.watch` is untouched.

### Added
- **A live run now says what it covered — and says it out loud when it admitted nothing (card X4).**
  `RuntimeHealthEvent("runtime_ended")` used to carry an EMPTY payload, so a session that admitted
  **zero** frames was byte-for-byte indistinguishable from a healthy quiet one on the live path —
  `percept doctor` could report `ZERO_ADMITS_ON_FIXTURE`, but doctor is a DRY RUN. `runtime_ended`
  now carries this run's `frames_seen` / `admitted` / `judge_calls` / `incidents`, and a run that
  observed frames while admitting none of them emits a preceding
  `RuntimeHealthEvent("zero_admission_coverage")` with the same counts plus the `threshold` and
  salience `stat` those frames were judged against. **It is a fact, not an alarm.** A still camera
  under a motion-gated watch correctly admits nothing and saves every model call — that is the whole
  tier-0 economic claim — so the event is deliberately **not** in `tracing.DEGRADED_HEALTH_KINDS`,
  never latches an unhealthy interval, and records as `state="info"`. Read it the way doctor frames
  `ZERO_ADMITS_ON_FIXTURE`: diagnostic. If the scene was not quiet, the eye is mis-selected, and
  `percept doctor` will name the statistic that would have admitted. No public signature changed.
- **The runtime now constructs `JudgmentEvent` — ADR-001's centre governs a live object (card X7).**
  `contracts.JudgmentEvent` was the "frozen output of one pass through the judge funnel" that the hot
  path never built: the live route was `Verdict -> GateOutcome -> Fired -> FireEvent`, and every
  construction site was rendering/eval-side. `BehaviorExecutor.fire_judged` now freezes one alongside
  the existing `Fired` and hangs it on `Fired.judgment` / **`FireEvent.judgment`**, so it reaches the
  emission boundary (`runtime.worker`'s incident/ask fan-out) with the gate key, the per-key
  `evidence_seq` reserved at judge dispatch (ADR-001 §8), the action, and a real `JudgmentTiming`.
  **Additive only**: what fires, when it fires and every existing field are byte-for-byte unchanged;
  `judgment` is `None` on paths that take no judge-funnel stamps (episode/rhythm/boundary/transition/
  absence) and on any timing that cannot be honestly frozen (counted as `judgment.build_failed`) —
  an absent record, never an invented one. Stamps come from the core's injected clock only;
  `source_media_ts` carries the caller's stream timeline `t` when one was supplied and is left unset
  otherwise. Capture-side times upstream of the core (camera capture, the worker's tier-0 admission)
  are not threaded through this seam and are deliberately not approximated.
- **The operating-frontier report — the honest form of a quality claim (card RB-FRONTIER).**
  `percept-eval --frontier` emits the full §3.2 report from the eval harness: **exact one-sided 95%
  bounds** on FIRE recall and on false FIREs **per scheduled watch-hour**, plus ASK rate, coverage,
  assisted recall, degraded time, duplicate FIREs per episode, and a **per-archetype breakdown over
  all six core gating archetypes** — including the archetypes with no evidence, printed as
  `NO EVIDENCE` rather than omitted, because a partial frontier report invites a cherry-picked read.
  **The bound is the claim, never the point estimate.** A 29-for-29 held-out run is recall `1.00` as
  a point estimate and `>= 0.9019` as a claim; a 3-for-3 run is also `1.00` and supports only
  `>= 0.368`, so it FAILS a 0.90 bar and the report prints the 29 episodes it would take. Bounds
  come from `percept.eval.exact_ci` — Garwood/Poisson (`factor(k)/T`) for rates, Clopper–Pearson
  (`Beta⁻¹(0.05; x, n−x+1)`) for proportions — pure Python, no SciPy, and asserted cell-by-cell
  against every table in `docs/evaluation/sizing-a-run.md`. **No normal approximation**: at
  these counts a Wald interval on a perfect run returns zero width at 1.0, which is not a bound.
  **Both denominators, always.** Per scheduled watch-hour is the bar-eligible number (concurrency is
  already inside it; degraded time stays inside it — a watch blind for 20 minutes still owes them);
  per source-hour is published only **at a stated active-watch count**, and omitted otherwise rather
  than printed with an implied concurrency of 1. Degraded time is subtracted from coverage, never
  from a rate denominator. **Abstentions stay visible** — ASKs, SILENTs, typed suppressions, and
  judge work that entered the funnel and never resolved are itemised in an abstention ledger and
  netted out of nothing. Zero labeled episodes reports `NO_EVIDENCE`, never a vacuous recall of 1.0.
  **Nothing is reinvented**: every match/miss/false-FIRE/duplicate count comes from
  `percept.harness.golden_eval.score`, every funnel/degraded/unresolved figure from
  `percept.tracing.TraceSummary` (card D8a), and the whole thing hangs off the **existing**
  `percept eval` entry point (`--frontier`, `--frontier-bars`, `--frontier-arm`, `--frontier-json`)
  — no second CLI, no second entry-point family. See `docs/observability/operating-frontier.md`.

- **The native visualizer — WHERE a decision died, not just that nothing happened (card D9).**
  `percept.visualize` renders a **standalone, single-file HTML page** over the artifacts that
  already exist: `percept.tracing` (`RunTraceManifest` / `TraceRecord` / `TraceSummary`), the
  portable `.ptrace` bundle, and `percept.harness.golden_eval.score` for the drift lane. **Nothing
  is recomputed**: the metrics panel *is* `TraceSummary`, the same object `percept doctor` and
  `percept eval` print, so the three surfaces cannot disagree about the funnel; the §5a greedy
  matcher is called, never reimplemented.
  Lanes are **pipeline stages on one shared media-time (`source_ts`) axis** — source/media, health,
  admission (admitted vs dropped, with the motion trace and the threshold rule that decided each
  frame), judge calls (bar width = wall latency; tokens and cost in the detail), gate decisions
  (`FIRE`/`ASK`/`SILENT` with the O2 reason code), incidents, ground-truth drift when supplied, a
  context lane, and cost. **An empty lane is drawn, not omitted** — the gap IS the diagnosis.
  Click a gate decision (or use ← / → / Home / End) for the **decision chain**: tier-0 → sense →
  VLM (model, latency, tokens, cost, evidence, confidence) → gate, each with the **per-stat
  counterfactual** — what `mean` / `block_max` / `p95` would each have decided **against its own
  threshold**. On the real client screen fixture every statistic sat under half its own cutoff, and
  the panel says so in one line ("never close — peak is 20% of its own cutoff"): that is what turns
  "nothing fired" into "here is the knob". `percept-doctor` now stamps `stat_thresholds` and
  `salience_stat` into the trace manifest so the counterfactual is READ, never invented; a trace
  without them says "threshold not recorded" instead of guessing.
  **Security posture, non-negotiable — a trace is untrusted input.** Verdict evidence is
  provider-generated and a model can emit markup, so every trace-derived string is escaped, and
  **no trace text ever reaches JavaScript**: each panel is pre-rendered server-side and the single
  inline script is a fixed, data-free constant pinned in the CSP by `sha256`. **Media embedding is
  opt-in and OFF by default** (the first client workload is accounting screens — "standalone" must
  not mean "silently bundle the customer's video"); a source URI is printed as text and never
  placed in an attribute, so nothing is auto-fetched. `default-src 'none'`, no external script,
  style, font or asset, size limits on every axis, and the page opens from `file://` with **zero
  network requests** (asserted in the tests).
  Rich relations stay **deferred to ADR-006**: episode supersession, document semantic identity,
  speaker attribution and finding settlement are deliberately not drawn — the context lane carries
  only what the trace actually has (`entity_id` correlation from card D4, goal-revision drops).
  Shipped as flags on the existing entry point, not a new console-script family:
  `percept-doctor --visualize-out page.html`, `--visualize-from run.ptrace` (render an existing
  trace or bundle, opening no source), `--visualize-media` (the explicit media opt-in) and
  `--golden gt.json`.
- **`SourceProfile` — an OPEN registry of source classes, and the client-tunes-it admission loop
  (card D7, ADR-005).** `percept.sourceprofile` adds the axis `WatchProfile` cannot express:
  `WatchProfile`'s `environment` is a camera mount (`fixed_camera` / `handheld` / `wearable`), so
  "this stream is a screen recording" was unsayable. A `SourceProfile` names the source class and
  selects **admission strategy, sampling, evidence packaging, identity strategy, episode
  segmentation and default timebase — and nothing else**. It is a name-keyed registry, never a closed
  enum: a third party ships `industrial_line` / `medical_imaging` / `satellite_change` /
  `code_editor` / `financial_dashboard` with `register_source_profile(...)` or the
  `percept.source_profiles` entry-point group, without editing a percept literal (and
  `percept-doctor --source-profile` takes open text for the same reason).
  **A source profile cannot change gate semantics.** `FIRE`/`ASK`/`SILENT`, the rising edge,
  refractory and dwell mean the same thing on every source — one `Gate`, one incident truth. Enforced
  structurally: the dataclass has no gate field, `FORBIDDEN_FIELDS` are rejected inside `parameters`,
  and the tests assert both.
  Three profiles ship. `physical_scene` (the default, and the ONE class the stock tier-0 eye is
  calibrated on); `screen_document`; and `conversation`, which ships `stability="experimental"` in a
  typed field — speaker identity is off-contract (no diarization behind `speaker_label`) and goals
  judge serially at ~24 s/turn measured.
  **`screen_document` declares its admission signal UNTUNED, on purpose.** Measured with
  `percept-doctor` on the real client fixture `probes/01-screen-walkthrough/data/clean.mp4` (36 s,
  180 frames, 1920x1080, containing a hard cut between two documents): `mean` max 0.0163 / thr 0.100,
  `block_max` max 0.0919 / thr 0.200, `p95` max 0.0743 / thr 0.150 — **zero admits on every
  statistic**. `block_max` amplifies the localized signal 5.6x as designed and still lands under HALF
  its own cutoff, because the tier-0 gate frame is 48x36 greyscale over an 8x6 grid: the eye is
  under-resolved for screen content, not mis-selected. **No threshold was lowered** (that would wreck
  physical-scene precision). Instead the profile fails safe: a fixed, non-optional low-rate coast
  (2.0 s — a floor, not a tuned value; `coast_sense_prior()` arms it on the existing
  `_capture_plan`/`SamplingController` seam), `media` timebase, evidence = full screenshot +
  changed-region crop + optional previous screenshot, episode = page interval, and identity =
  **content key, never geometry** (IoU+label cannot separate INV-2291 from INV-2294).
- **`percept-doctor` now reports the active source profile** (`--source-profile`, defaulting to
  `physical_scene` and labelling the choice `explicit`/`default`), with two new findings:
  `ADMISSION_SIGNAL_UNTUNED` ("this profile is running on coast alone, supply a change signal") and
  `SOURCE_PROFILE_EXPERIMENTAL`. Both are **warns** — an untuned admission signal never invents an
  exit code. The `--json` report carries the full profile under `plan.source_profile.profile`.
- **`docs/admission/tuning-admission.md` — the tuning loop as a deliverable.** A worked, runnable guide
  from "nothing fires" to a tuned eye: how to read a doctor report correctly (using the `clean.mp4`
  numbers above, including why the answer is NOT to lower the threshold), how to name your source
  class, how to turn on the coast, how to register your own phash/OCR model as an **admission**
  detector (it runs at native fps on the full-resolution judge frame with no cloud judge call), and
  how to A/B it with `percept eval` on YOUR fixtures against YOUR bars. Stated plainly there and in
  ADR-005: percept's stock eye does not admit on this class of content, percept ships no OCR model
  and no calibrated screen threshold, and **the client supplies and tunes the admission signal**.
- **`.adr/ADR-005-source-profiles.md`** — the decision, the measurement, the gate-invariance
  argument, and an explicit constraint on card L1: the `WatchProfile` ontology is camera-shaped and
  L1 must not be built on it.
- **`percept check` widened past the Vision seam — conformance for the seams D6 kept (card D2).**
  `percept.conformance` (suite `1.1.0`) now covers all three surviving extension points through ONE
  reporting shape (`CheckResult` / `ConformanceReport`) and one severity vocabulary
  (`REQUIRED`/`ADVISORY`), and one CLI (`python -m percept.conformance TARGET --role …`):
  - **`check_detector(...)` — the bring-your-own ADMISSION detector** (`register_detector` +
    `DetectorManifest`), the seam a client plugs an OCR/phash model into for screen content. It
    checks manifest well-formedness, the `push(t, value) -> [Boundary]` / `reset()` call shape the
    `DetectorBank` actually uses, boundary types + attribution, survival of a degenerate stream
    (flat-zero, pinned-high, single-sample), **determinism on a fixed input**, that `reset()` really
    restores initial state, and that the name is **selectable end-to-end into a `WatchSpec`** through
    the real registry → firewall → `compile_watchspec` path (registry state is restored afterwards).
    The failure mode this exists to prevent is the silent one: an unroutable `signal` string is
    routed to the MOTION channel by `DetectorBank` without raising, so the only production symptom is
    a watch that quietly under-admits. A FRAME-backed detector's push checks **SKIP loudly** (the
    report reads `PARTIALLY CHECKED`) unless the caller supplies `samples=` — there is no universal
    probe frame and a fabricated one would prove nothing.
  - **`check_tier0_adapter(...)` — the `structural_adapter` / `semantic_adapter` callables `run()`
    now accepts (card D6a).** Bounded normalized score (a non-number, NaN or out-of-`[0,1]` value
    silently pins or mutes admission), survival of a degenerate frame (1×1, uniform black/white), and
    a real `MotionGate` gated on that channel driven over a frame stream **without crashing the
    gate**, with the adapter's value asserted to be the one reported on `Tier0Signal`.
  - **The honest scope disclaimer is now PRINTED on every report, for every role** (vision included).
    This suite proves a plug-in is WELL-FORMED — that percept can call it, route it and read what it
    returns. It is **not** a measure of the model's intelligence or quality, and a green report does
    **not** mean "my OCR model is good". That question is `percept eval`'s (card D3): a paired A/B on
    the client's fixtures against the client's pre-registered bars.
- **`percept-doctor` — a plan-aware tier-0 dry run that spends ZERO judge calls (card D1).** A new
  console entry point (alongside `percept-watch` / `percept-bench` / `percept-schema`) that answers,
  without an API key and at no cost: did the source open, did frames arrive, what timebase was
  resolved (through the runtime's own resolver), what plan compiled, which profile was active, which
  admission signals were even available, how many candidates each goal admitted, and **which named
  rule suppressed every non-admit** (`no_predecessor_frame` / `out_of_schedule` /
  `cadence_not_elapsed` / `signal_unavailable` / `below_threshold`). Admissions come from replaying
  the REAL `MotionGate`, so a doctor number cannot drift from what the runtime would have done, and
  the funnel panel is read straight off `percept.tracing.TraceSummary` (card D8a) so doctor, eval and
  the visualizer cannot invent three definitions of it. `--json` emits the whole report; exit codes
  are `0` completed / `2` zero semantic coverage / `3` source failure / `4` plan unobservable under
  the selected profile / `5` invalid configuration.
  **The three salience statistics are never compared to one threshold.** `mean`, `block_max` and
  `p95` live on different value scales and the SDK already carries separate calibrated cutoffs, so
  every row prints its own effective threshold and its own admitted count. **Zero admits is
  diagnostic, not automatically failure**: it is reported loudly (`ZERO_ADMITS_ON_FIXTURE`) but adds
  no runtime health alarm — a still scene that saves every model call is the product working.
- **`percept.harness.tier0.salience_stats()` — all three salience statistics from ONE pass.** The
  per-block score list is computed once (`_block_scores`) and `mean` / `block_max` / `p95` are pure
  selectors over it, each bit-identical to `motion_score(..., stat=…)` (test-enforced). This is what
  makes doctor's counterfactual table cheap and what guarantees it agrees with the live gate.
- **`percept eval` — paired differential A/B on client fixtures (card D3).** New
  `percept.eval.differential` (+ the `percept-eval` console script) answers the question a client
  with a new model actually has — *does my OCR detector actually help?* — as a PAIRED comparison,
  not two independent runs printed side by side:
  `percept-eval --baseline baseline-plan.yaml --candidate ocr-plan.yaml --fixtures ./fixtures --bars bars.yaml`.
  The report carries both **plan hashes** (`PlanIdentity.from_goals`, the same content hash the
  runtime stamps on a trace manifest), adapter/model versions, per-fixture results, per-fixture
  **paired deltas**, a **percentile-bootstrap confidence interval** on each delta (seeded from the
  registration hash, so two runs give the same interval), **cost deltas**, the **exact held-out
  split**, and the threshold/policy values selected before the run. Matching is NOT reimplemented —
  every recall/precision number comes from `percept.harness.golden_eval.score` (the Viability §5a
  one-to-one greedy matcher) and every cost/funnel number from card D8a's
  `percept.tracing.TraceSummary`, so eval cannot disagree with `doctor` or the visualizer.
  - **Bars are client-supplied.** With no `--bars`, eval reports the numbers and returns
    `NOT_ASSESSED`; it never invents a pass/fail threshold on a client's behalf.
  - **No tune-to-green.** Bars, split and thresholds are pre-registered into a
    `registration_hash` covering both plan hashes, both arms' components/thresholds, the split and
    the bars; `--expect-registration-hash` makes a configuration that moved after the bars were
    declared raise `ConfigDriftError` and score nothing. Bars are also refused (not silently
    applied) when the held-out split is undeclared.
  - **Scope is scored honestly (ADR-006).** Every fixture declares a `goal_shape` and gets one of
    `SUPPORTED` / `PARTIALLY_SUPPORTED` / `UNSUPPORTED_FOR_THIS_GOAL_SHAPE`. `point`/`episode` are
    supported; `recurrence` is partial; `supersession`, `correction`, `settled_interval`,
    `speaker_attributed_claim` and `document_identity_change` are UNSUPPORTED — those relations do
    not exist in the contract yet, so eval reports them rather than scoring a richer interaction
    with the wrong yardstick. An unknown shape is unsupported, never optimistically supported.
- **Portable trace bundle + deterministic cognition replay (card D8b).** A media *reference* is not
  self-contained, so `percept.trace_bundle` ships two explicit, self-describing modes: `local`
  (manifest + `events.jsonl` + `summary.json` + media **references**, uri + sha256 — the media stays
  on the client's machine) and `portable` (the same plus selected media **bytes**, packed as a
  `run-123.ptrace` directory or a single `run-123.ptrace.zip` a client attaches to a ticket).
  **Media inclusion is opt-in and defaults OFF** — the first client workload is accounting screens,
  and a default-on bundle is an exfiltration vector, not a convenience; evidence offered to a
  local-mode bundle is dropped and the omission is *counted* into the bundle manifest rather than
  being silent. Every file is content-hashed (`sha256` per file plus a `payload_sha256` over the
  ordered file list), every bundle carries a retention stamp (`retention_days`, default 30 d, and an
  `expires_wall_ts`), and `TraceBundle.verify_path()` catches a truncated or edited attachment
  instead of replaying half a run. `export_bug_report(recorder, path)` is the one-call client path.
- **`percept.replay` — COGNITION REPLAY, defined honestly (card D8b, ADR-004 §2).** Recorded
  normalized verdicts + timestamps are re-fed through the `Gate` and the incident rule, offline, with
  no model call and no wall clock: same trace in → **byte-identical** decisions out, asserted as a
  content-hash `digest` in `tests/test_cognition_replay.py`. MEDIA REEVALUATION (recorded media sent
  to a live model again) is *not* deterministic — sampling, model drift, provider changes — and is
  explicitly out of scope here; it belongs to `percept eval`. Nothing in this module or in a bundle
  manifest claims a trace "replays the run" on the strength of a media URI. What a trace cannot know
  (hard-quadrant escalation, `fire_when`, `suppress_false`) is declared per goal via
  `ReplayGoalConfig`, and any disagreement with the recorded run surfaces as a reported
  `Divergence`, never as a smoothed-over match. The gate's knobs are replayed from the new additive
  `RunTraceManifest.gate_config` (`gate_config_of(core.gate)`); when a trace lacks them the report
  says `gate_config_recorded=False` instead of pretending today's defaults were the run's.
  `gate.classified` trace records additionally carry the gate's INPUTS
  (`satisfied`/`confidence`/`freshness`/`entity_id`/`hold_s`) — additive trace data, no behaviour
  branches on it.
- **The tier-0 extension seam is reachable from `run()` (card D6a).** `percept.runtime.run()` now takes
  `structural_adapter` / `semantic_adapter` — `(frame) -> 0..1` callables that fill the `structural` /
  `semantic` channels of the reported `Tier0Signal`, exactly as `MotionGate` has always accepted them.
  Previously the only way to supplement tier 0 with a cheap local model (a CLIP drift score, a pose or
  privacy classifier, an OCR "the document changed" score) was to bypass `run()` and drive
  `GatedWatchLoop` yourself. A goal gated on `structural`/`semantic` (or a `wake_policy`) turns that
  score into admission. Both default `None` → the motion-only path is byte-for-byte unchanged.
- **`examples/byo_admission_detector.py` + "Admission is not extraction" (card D6b).**
  `register_detector` is now documented in the user-facing reference as what it actually is: the
  bring-your-own **admission detector** path (OCR-driven admission included). It is **not** a complete
  "bring your own OCR specialist" path — a boundary such as `document_changed at t=8.2` decides *when to
  look closely*; it does not deliver `invoice_number = "INV-2291"` to the semantic judge. Facts reach the
  judge only when the caller passes `Vision.judge(context={...})`. The example runs offline with no keys.
- **Effective run configuration is RECORDED, never silently substituted (card D5b/c).**
  `percept.runtime.run()` now retains what it was ASKED for alongside what it actually opened with:
  `stats.effective_config` (an `EffectiveConfig` of `EffectiveFps` + `EffectiveTimebase`), the same
  record mirrored on `core.effective_config`, and one informational
  `RuntimeHealthEvent("effective_config")` at start. `EffectiveFps` carries `requested_fps`,
  `effective_fps` and a `reason` from the closed `FPS_REASONS` vocabulary —
  `brief-window-coverage` (a brief-event plan needs the cadence), `source-limit` (a `str` source's
  rate is fixed at open, so the 8 fps headroom cap is taken up front), `profile-default` (nothing
  overrode the request). Capture behaviour is unchanged; an override is no longer *invisible*. An
  override on its own is not a warning — the loud channel stays reserved for a coverage requirement
  that is actually unmet (`capture_plan_unmet`).
- **`run(timebase="auto")` — media time for recorded sources, wall time for live ones (card D5c).**
  Every `str` source used to be stamped on the wall clock, so a 36 s clip decoded in 0.72 s gave every
  `incident.ts` the decode instant and the timestamps were useless as audit evidence. `"auto"` (the new
  default) resolves to `"media"` for recorded media (a file path/URI with a media suffix) and `"wall"`
  for live devices/streams (`webcam:0`, `mic:0`, `rtsp://…`, anything unrecognised — those must keep
  sharing the agent's deadline clock). `timebase="media"`/`"wall"` forces the choice; a non-`str`
  source still feeds its own explicit `t` and is never re-stamped. **Both** clocks are recorded per
  frame regardless (`stats.last_media_t` / `stats.last_wall_t`).

### Removed / Breaking
- **The enricher registry is REMOVED (card D6c).** `register_enricher` / `enrich_signal` /
  `enricher_for` / `Enrichment` / `NullEnricher` / `FakeEnricher` and the modules
  `percept.harness.enrichers` + `percept.edge.enrichers` are gone. They were documented as "add an OCR
  specialist without editing core" and had **zero callers** anywhere outside their own module: nothing
  in the runtime ever ran a registered enricher, so registering one changed nothing about any watch — a
  client went looking for the OCR seam, found this one, and correctly concluded it was dead. The
  screen/document workload needs OCR for **admission** (`register_detector` / the tier-0 adapters above
  already cover that), not for semantics — the client's judge read 7 px document text correctly with no
  OCR at all. The `percept.Enricher` protocol is retained as a DEPRECATED name for one contract MAJOR
  (nothing consumes it) and will be removed in the next MAJOR. `Vision.judge(context=...)` — the live
  way to hand the judge structured facts — is untouched.
- **`run(core=…, goals=…)` / `run(core=…, vision=…)` now raises `ValueError` (card D5a).** Supplying
  `goals=`/`vision=` alongside an explicit `core` used to silently DISCARD them: the watch ran
  byte-identically and simply never armed the caller's goals — a silent failure that cost two client
  teams a debugging cycle. A warning would still permit that run, so the call is refused before the
  first frame is consumed: *"goals= and vision= cannot be supplied when core= is provided; apply goals
  to the core or omit core."* Both single-argument forms (`run(core, source=…)` and
  `run(source=…, goals=[…])`) are unchanged.


- **Consequence gate GA (card O2)** — the FIRE/ASK/SILENT decision model is now a documented, stable
  surface: `docs/watches/the-gate.md`.
  - `Gate.decide(...) -> GateOutcome` — the typed outcome (action **plus** a reason code) alongside the
    unchanged `Gate.evaluate(...) -> GateAction`. `evaluate` is exactly `decide(...).action`.
  - A closed reason vocabulary — `GATE_REASONS`, `REASON_ACTION`, `SUPPRESSION_REASONS`,
    `GATE_ACTIONS`, `DELIVERED_ACTIONS` — exported from the top-level `percept` namespace.
  - **Gate suppression is observable**: a fire withheld by `require_false_first` (`start_anchored`) or
    `refractory_s` (`refractory`) now emits a `gate.suppressed` log line and, under
    `percept.runtime.worker.run`, a `RuntimeHealthEvent("gate_suppressed")` carrying the key, the
    reason code and the decision time. Previously a silent non-event.
  - `SILENT` is documented and asserted as **internal-only** — never the action on a delivered event.

- **Published Vision adapter interface + conformance suite** — `docs/judges/bring-your-own-model.md`
  documents the frozen `Vision` seam as the public extension point (every escalation kwarg, the
  `Verdict`/`Observation` invariants, and fail-closed injection: your instance reaches `core.vision`
  unwrapped, and a live core with no vision raises `ModeError` rather than falling back to a scripted
  double). `percept.conformance` ships the *runnable* version of that contract —
  `check_vision_adapter(adapter)` in your own tests, or
  `python -m percept.conformance mypkg.vision:MyVision` (`--static` for shape-only, no key needed).
  Required vs advisory severities; it checks SHAPE, never model accuracy (that stays an L2/L3 eval
  question). Versioned independently as `CONFORMANCE_VERSION`. Additive — no change to `watch()`,
  `WATCH_VERSION`, or `CONTRACT_VERSION`.

### Changed
- **Ordered window is the LIVE judge plan for dynamic goals (card J2).** A resolved `WatchProfile`'s
  `plan_mode` is no longer inert: `plan_mode="window"` now floors every concern's `PerceptionSpec` on the
  judging path, so the judge receives an ordered frame window (`judge(frames=[...])`, oldest→newest, the
  anchor frame passed separately) instead of one still frame. The seeded default profiles already carry
  `plan_mode="window"` for dynamic goals (recall 0.68 → 0.73; 0.90 with gate-relax on the H0 set). The
  profile's plan mode is a FLOOR — a concern that authored a richer `perception` block keeps it.

### Deprecated
- **`sprite` in the plan path.** `temporal="sprite"` / `plan_mode="sprite"` no longer builds a montage;
  it is normalized to `"window"` with a `DeprecationWarning` (an ordered window beats a montage
  unconditionally). `percept.sprite.make_sprite` remains as a standalone, importable primitive — nothing
  in the firing path calls it.

### Notes
- **No default behavior change.** Both precision guards remain off by default, so a default deployment
  suppresses nothing and emits no new health event. `watch()`'s signature and `WATCH_VERSION` are
  unchanged.

## 0.7.0 — research preview (live runtime)

The live **runtime** is now the front door: point Percept at a stream, give it plain-English goals, get a
notification once per event.

### Added
- **Fixture-bound offline mode** — `percept.offline_core()` builds a core in explicit `mode="offline"`
  bound to a **bundled deterministic fixture** that ships in the wheel (`percept/fixtures/*.json`), so a
  keyless first run replays the real fovea → judge → gate path end to end: `watch(offline_frames(),
  offline_goals(), percept=offline_core())`. The offline judge is a *replayer*, not a general fake — a
  frame the fixture does not script raises `OfflineFixtureError` (a `ModeError`) instead of inventing a
  plausible verdict, and every verdict's evidence says it is a replay, not perception. A LIVE run with no
  vision backend / no credential still raises the typed `ModeError` and never falls back to a fake judge
  (`assert_live_or_offline()` exposes that pre-flight to front doors).
- **`watch(stream, goals) → AsyncIterator[WatchEvent]`** — the frozen public entrypoint (versioned,
  `WATCH_VERSION`). `WatchEvent` + `WatchAction` added to the contract (`CONTRACT_VERSION` 1.1 → 1.2, additive).
- **`percept-watch` CLI** — `pip install "percept-vision[runtime,openrouter]"` then
  `percept-watch --goal "…"`. Webcam / video / RTSP; `--egomotion`, `--fps`, `--refractory`, `--vision`.
- **OpenRouter judge backend** (`vision="openrouter"`, extra `[openrouter]`) — cheap, fast, hosted VLMs
  (default `qwen/qwen3-vl-32b-instruct`).
- **Localized salience** `PERCEPT_SALIENCE_STAT=block_max` — catches low-motion actions (holding, sitting)
  a global-mean motion gate misses (nearly doubled recall on the H0 charades set).
- **Ego-motion compensation** `PERCEPT_EGO_COMPENSATE=true` — subtract global camera/head motion before
  gating; makes a **moving camera** (smart glasses / body-cam) viable (measured ~98% → ~30% escalation).
- **Gate time-refractory** `PERCEPT_GATE_REFRACTORY_S` — suppress duplicate notifications of the same goal;
  one fire per episode.
- **Production trace telemetry** — opt-in per-frame trace (edge decision · verdict · gate action · reason).
- **Docs:** `docs/watches/creating-a-watch.md` (embed the runtime), `docs/watches/writing-goals.md`,
  `examples/vision_assistant.py`, updated `percept-agent` Claude skill.

### Notes
- Research preview. Not yet built: WatchProfile resolution wired into the runtime, window/episode judge
  modes in the live path, the health/evidence/delivery surfaces (see the project roadmap).
- No published benchmark numbers; quality is workload-dependent (bring your own fixtures via the eval harness).

## 0.6.0 — OSS research preview
Initial public research-preview of the cognition core.
