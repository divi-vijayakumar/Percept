# Changelog

## 0.7.0 — research preview (hackathon runtime)

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
  `examples/hackathon_watch.py`, updated `percept-agent` Claude skill.

### Notes
- Research preview. Not yet built: WatchProfile resolution wired into the runtime, window/episode judge
  modes in the live path, the health/evidence/delivery surfaces (see the project roadmap).
- No published benchmark numbers; quality is workload-dependent (bring your own fixtures via the eval harness).

## 0.6.0 — OSS research preview
Initial public research-preview of the cognition core.
