# Percept demo harness

Browser demos that drive the **real** `percept.watch()` runtime — this is how we dogfood the wheel
end-to-end before a client wheel ships. Nothing here is simulated: every timeline, gate decision,
context-ledger row, and `.ptrace` comes from the runtime, via the same seam as `percept.watch_cli --trace`.

- **Demo 01 — MP4 video watch** (`demo-01-video-watch.html`): upload a video, author a goal, run a watch.
- **Demo 02 — screen-share watch** (`demo-02-screen-watch.html`): record a screen, then the runtime judges the recording.
- **Demo 04 — live stream + replay** (`demo-04-stream-uploaded-video.html`): watch a **live feed** be
  judged as it happens, then replay every decision on the media timeline. See below.

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

## Demo 04 — live sources

Demo 04 feeds the runtime from **two source classes that reach it through the same call**,
`percept.run(core, source=<uri>)`. That is the point: the second is a rehearsal for the first, on the
same seam.

Both need a local RTSP server. A bare `mediamtx` refuses paths it was not configured for
(`path 'traffic' is not configured` → ffmpeg fails with `400 Bad Request`), so use the config here:

```bash
mediamtx docs/demo/mediamtx.yml                              # terminal 1
# or, equivalently:  docker run --rm -it -p 8554:8554 bluenviron/mediamtx
```

**Source 1 — a real live feed.** Publish it yourself, then point the page at the path:

```bash
pip install yt-dlp                                            # terminal 2
ffmpeg -i "$(yt-dlp -g 'https://www.youtube.com/watch?v=sTF-6_xinUU')" \
  -an -c:v copy -f rtsp -rtsp_transport tcp rtsp://localhost:8554/traffic
```

Pick **Live feed** in the page, use `rtsp://localhost:8554/traffic`, and press **Check** — the harness
probes it with the same OpenCV/FFMPEG stack the run will use, so an unreachable feed is a clear message
rather than a run that reads zero frames. (`-an` matters: a live source has no container to demux, so
the run is vision-only and says so.)

**Source 2 — your own video, restreamed through that same server.** Pick **Upload → restream** and
choose a file. The harness runs `ffmpeg -stream_loop -1 -re` into `rtsp://…/percept-<run_id>`, so the
file arrives in real time and on loop: no seeking, no rewind, no look-ahead. The loop is load-bearing —
a one-shot `-re` publish of a short clip can finish before the runtime has finished arming.

A third tab, **Upload → as file**, runs the demo 01 path (seekable, faster than real time) so you can
see what changes when the same clip becomes a stream you cannot outrun.

Live runs end on the **run for** ceiling or on **Stop run**; either way the `.ptrace` is written and the
page names the reason. Then **Replay** releases the same records on the media clock — driven by the
`<video>` element for a file run, and by the frames the runtime itself handed over for a live one.

Endpoints demo 04 adds: `POST /run/stream` (NDJSON: one line per trace record and per delivered
Incident, then a final `done` carrying exactly what `POST /run` would have returned),
`POST /stop/<run_id>`, and `GET /source-check?url=…`.

## How it maps to live APIs

`serve.py` wraps the frozen public seam — `percept.run(core, source=…, on_trace=rec.tracer())` for uploads,
`percept.watch(offline_frames(fixture), …)` for the keyless path — the same calls `percept.watch_cli`
makes. The browser clients only render the returned trace JSON, so swapping in a hosted runtime later is a
change of endpoint, not a rebuild.

## Not shipped API

This harness is internal test tooling. It is **not** part of the published package and not a supported API.
`serve.py` binds `127.0.0.1` only and never egresses media — uploads are written to a temp file, judged,
and deleted; the screen recording stays in the browser until you stop and hand it to the local runtime.
