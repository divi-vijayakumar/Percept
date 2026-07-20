# Changelog

## Unreleased

### Added
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
