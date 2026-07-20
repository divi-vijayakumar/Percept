# Changelog

## Unreleased

### Added
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
- **`run(core=…, goals=…)` / `run(core=…, vision=…)` now raises `ValueError` (card D5a).** Supplying
  `goals=`/`vision=` alongside an explicit `core` used to silently DISCARD them: the watch ran
  byte-identically and simply never armed the caller's goals — a silent failure that cost two client
  teams a debugging cycle. A warning would still permit that run, so the call is refused before the
  first frame is consumed: *"goals= and vision= cannot be supplied when core= is provided; apply goals
  to the core or omit core."* Both single-argument forms (`run(core, source=…)` and
  `run(source=…, goals=[…])`) are unchanged.


- **Consequence gate GA (card O2)** — the FIRE/ASK/SILENT decision model is now a documented, stable
  surface: `docs/reference/consequence-gate.md`.
  - `Gate.decide(...) -> GateOutcome` — the typed outcome (action **plus** a reason code) alongside the
    unchanged `Gate.evaluate(...) -> GateAction`. `evaluate` is exactly `decide(...).action`.
  - A closed reason vocabulary — `GATE_REASONS`, `REASON_ACTION`, `SUPPRESSION_REASONS`,
    `GATE_ACTIONS`, `DELIVERED_ACTIONS` — exported from the top-level `percept` namespace.
  - **Gate suppression is observable**: a fire withheld by `require_false_first` (`start_anchored`) or
    `refractory_s` (`refractory`) now emits a `gate.suppressed` log line and, under
    `percept.runtime.worker.run`, a `RuntimeHealthEvent("gate_suppressed")` carrying the key, the
    reason code and the decision time. Previously a silent non-event.
  - `SILENT` is documented and asserted as **internal-only** — never the action on a delivered event.

- **Published Vision adapter interface + conformance suite** — `docs/reference/vision-adapter.md`
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
- **Docs:** `docs/integration.md` (embed the runtime), `docs/how-to/authoring-goals-and-watch-plans.md`,
  `examples/vision_assistant.py`, updated `percept-agent` Claude skill.

### Notes
- Research preview. Not yet built: WatchProfile resolution wired into the runtime, window/episode judge
  modes in the live path, the health/evidence/delivery surfaces (see the project roadmap).
- No published benchmark numbers; quality is workload-dependent (bring your own fixtures via the eval harness).

## 0.6.0 — OSS research preview
Initial public research-preview of the cognition core.
