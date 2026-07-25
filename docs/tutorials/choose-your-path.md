# Choose your path

Percept has a lot of surface area — runtime, edge, harness, contracts, backends, benchmarks, process-QA,
rhythm, screen, audio. **You do not need most of it to start.** Pick a row, do that, ignore the rest
until you need it.

| I want to… | Start here | Install | Ignore for now |
|---|---|---|---|
| Build a simple watcher (no keys) | [quickstart](quickstart.md) (`mode="offline"`), or the `percept-agent` skill in Claude Code | `pip install percept-vision` | edge, harness, benchmarks |
| Watch a webcam / video / RTSP | **`percept.run(source="webcam:0", goals=["..."])`** — [runtime how-to](../how-to/runtime.md) | `[runtime]` + a backend (`[gemini]`/`[claude]`) | JS edge, rhythm |
| Fire only after a condition stays true for a duration | Build a `hold_s` Watch and use the [in-process Python duration route](../reference/sustained-duration-watches.md) | `[runtime]` + a backend | remote JS edge, restart continuity |
| Watch **and listen** (fused audio) | `run(source="webcam:0", goals=["..."], audio_source="mic:0")` — [audio modality](../reference/audio-modality.md) | `[runtime,mic]` + a backend | JS edge, rhythm |
| Drive a live watch (arm/ask/pause by voice or code) | [Command seam](../reference/command-seam.md) — `RuntimeSession` | `[runtime]` + a backend | — |
| Watch my screen | [Screenshare how-to](../how-to/screenshare.md) | `[runtime]` (+ `[screen]`) | camera/benchmark internals |
| Use a real VLM | [Backends reference](../reference/backends.md) | `[gemini]` or `[claude]` | harness internals |
| Write my own judge / vision backend | [Custom judge & backend how-to](../how-to/custom-judge.md) | your backend | harness internals |
| Author a profile for my use case | [Custom configuration how-to](../how-to/custom-configuration.md) | `pip install percept-vision` | edge, harness |
| Tune the eye for my content | [Tuning admission how-to](../how-to/tune-admission.md) | `pip install percept-vision` | edge, harness |
| Configure env vars / thresholds | [Configuration reference](../configuration.md) | core | — |
| Run on-device edge gating | [Edge parity reference](../reference/edge-parity.md) | `npm install percept-edge` | sustained-duration Watches, process-QA, benchmark methodology |
| Evaluate / optimize / benchmark | [Eval](../how-to/eval.md) · [Optimize](../how-to/optimize.md) · [Benchmarks](../how-to/benchmarks.md) | `[bench]` | runtime internals |
| Look up a CLI command | [Command-line reference](../reference/cli.md) | core | — |

## You probably don't need this yet

- **`percept.harness`** — only if you're tuning server-side edge detectors (pose, acoustic-onset, motion).
- **`percept.contracts`** — only if you're integrating another runtime against the wire contract.
- **`percept-bench`** — only if you're measuring quality against a manifest.
- **process-QA / rhythm** — only if your goal is a multi-step process or repetition counting.
- **`percept-edge` (npm)** — only if you want the gate to run on-device (browser/WASM) before any cloud call.

## The 10-minute path

1. `pip install percept-vision` (zero deps, no keys).
2. Try the `percept-agent` skill in Claude Code, or build a minimal script:
   ```python
   agent = percept.Percept.create(mode="offline", discover_plugins=False)
   agent.add_goal(Goal(id="kettle", condition="kettle boiled", say="Done!"))
   for ev in await agent.perceive_judged("boiling"): print(ev)
   ```
3. Read [Concepts](../explanation/concepts.md) (the eyes/brain split) and
   [Architecture](../explanation/architecture.md) (the four tiers).
4. When you want real eyes, add `[gemini]` or `[claude]` and a key — the cognition layer above is
   unchanged.

Full map: the [docs index](../README.md).
