# Percept demo harness

Two browser demos that drive the **real** `percept.watch()` runtime — this is how we dogfood the wheel
end-to-end before a client wheel ships. Nothing here is simulated: every timeline, gate decision,
context-ledger row, and `.ptrace` comes from the runtime, via the same seam as `percept.watch_cli --trace`.

- **Demo 01 — MP4 video watch** (`demo-01-video-watch.html`): upload a video, author a goal, run a watch.
- **Demo 02 — screen-share watch** (`demo-02-screen-watch.html`): record a screen, then the runtime judges the recording.

## Run it

```bash
python docs/demo/serve.py            # http://localhost:8788
```

Open the printed URL. With **no API key**, the demos run the bundled fixture through the real runtime
(scripted verdicts — proves wiring, not model intelligence). To run a **real judge over your own media**,
set a key first:

```bash
OPENROUTER_API_KEY=…  python docs/demo/serve.py     # or GEMINI_API_KEY / ANTHROPIC_API_KEY
```

Pick the matching backend in the page's "judge backend" selector. Real runs spend real tokens.

## What is real vs. what to know

Everything the pages render is real runtime output:

| Panel | Source (real) |
|---|---|
| Run manifest | `.ptrace` `manifest` — `percept_version`, `watch_version` 1.1.0, `contract_version` 1.10.0, plan hash |
| Primary metrics | `.ptrace` `summary` — the `observed → admitted → judge_started → judge_completed → gate` funnel, cost, savings |
| Timeline / lanes | trace `records` — `source.frame`, `judge.started/finished`, `gate.classified` |
| Context ledger | `shadow.fold` records — the **advisory** subject + `EpisodeFoldResult` (Y11 shadow mode) |
| Subscriber events | the real `WatchEvent`s the run emitted |
| `.ptrace` | the exact trace file the run wrote, downloadable |

Two honest caveats:
- The **context ledger is advisory**. Per Y11 shadow mode, the identity/episode fold is computed and
  written to the trace but governs nothing — the gate fires exactly as it would without it. It is shown
  because it is real trace data, not because it drives behaviour (it doesn't, yet).
- The **keyless fixture** is a scripted-verdict run. It exercises the real runtime and trace machinery,
  but the judgments are authored, not a model's — the page labels this on every fixture run.

## How it maps to live APIs

`serve.py` wraps the frozen public seam — `percept.run(core, source=…, on_trace=rec.tracer())` for uploads,
`percept.watch(offline_frames(fixture), …)` for the keyless path — the same calls `percept.watch_cli`
makes. The browser clients only render the returned trace JSON, so swapping in a hosted runtime later is a
change of endpoint, not a rebuild.

## Not shipped API

This harness is internal test tooling. It is **not** part of the published package and not a supported API.
`serve.py` binds `127.0.0.1` only and never egresses media — uploads are written to a temp file, judged,
and deleted; the screen recording stays in the browser until you stop and hand it to the local runtime.
