# Realtime voice watch agent (Pipecat SmallWebRTC + Gemini Live)

One working, copyable application proving the whole system end-to-end **through public Percept seams
only**: a held speech-to-speech voice wire (Gemini Live over Pipecat P2P WebRTC, browser AEC, no SFU),
a proactive Percept watch on the same camera, and an application-owned **floor** that decides who may
speak into the room.

```text
browser (prebuilt WebRTC client)                     server (app.py)
  mic  ── AEC native ──►  SmallWebRTC P2P ──►  Gemini Live (S2S: hears+reasons+speaks)
  speaker  ◄──────────────────────────────────    │ watch / stop_watching / whats_armed tools
  camera ─────────────►  video frames ────────►   │ percept_state (grounding, PULL-only)
                                                percept.start(core, source=video tee)
                                                incidents → AttentionIntent → Floor → speech
```

## Layout

| File | Owns |
|---|---|
| `app.py` | composition root — `Percept.create` + `percept.start` + wiring; public seams only |
| `engine_adapter.py` | template-local `SpeechRequest`/`VoiceEngine` seam + the Pipecat/Gemini-Live engine (readiness kick, pinned live model, pull-grounding tool — every hard-won workaround documented inline) |
| `media.py` | WebRTC signaling + video tee + the **three-way audio split** (engine / local VAD / gated world-audio) |
| `percept_bridge.py` | the ONE place the app touches Percept: tools, transcript tee, snapshot invalidation, incident→intent→floor |
| `floor.py` | who speaks now: user > trusted interrupt > wire reply > deferred; TTL; coalescing |
| `fire_renderer.py` | channel selection (speech / webhook / log; `action` documented out of scope) |
| `engagement.py` | `always_open` demo flag + wake-word / push-to-talk / idle stubs |
| `telemetry.py` | honest A2V/V2V latency, correlated by incident ID, monotonic clocks |

## Run

```bash
# from the repo root, once (dev checkout):
pip install -e packages/percept-vision

cd templates/realtime_voice_watch_agent
pip install -e .
export GEMINI_API_KEY=...          # (or GOOGLE_API_KEY / GOOGLE_AISTUDIO_API_KEY)
cp config.example.yaml config.yaml # optional — defaults work
python app.py                      # → open http://127.0.0.1:8930, allow camera + mic
```

Set `PERCEPT_RT_DEBUG=1` to trace the mic→Gemini audio chain (frame counts, readiness gate, first
transcript) if the wire connects but stays silent.

## Tests

```bash
make test-templates                # from the repo root (keyless; pipecat tests skip if not installed)
```

`tests/test_app_offline.py` is the end-to-end offline proof: the real `build_agent` wiring + a fake
engine + `Percept.create(mode="offline")` + scripted frames → a user utterance gets a spoken reply, a
scripted FIRE is spoken through the floor, and barge-in cancels playback. `tests/test_boundary.py`
enforces the public-seam allowlist (no `core.goals`, no `session._*`).

## Live-test checklist (run before shipping changes to this template)

Conversation/floor:

- [ ] greeting within ~3s of connecting (the readiness kick worked)
- [ ] "what do you see?" answered from the live camera
- [ ] "tell me the second I raise my hand" → instant "On it." + `whats_armed` reflects it
- [ ] raising the hand fires a proactive spoken nudge while you are silent
- [ ] a fire during user speech queues behind the turn, speaks after it
- [ ] a conductor-authored `preempt: interrupt` is clamped to queue; a structured interrupt-armed
      goal (config.yaml `delivery: {preempt: interrupt}`) preempts
- [ ] barge-in during incident playback stops playout
- [ ] `PERCEPT_RT_DEBUG=1` shows mic frames flowing

Scheduler proof (non-blocking capture — NOT "judge_calls tracks frame cadence"): watch the
`[watch] …` heartbeat while covering/uncovering the camera:

- [ ] `frames` keeps increasing while a judge is in flight/blocked
- [ ] capture timestamps stay near source cadence
- [ ] `judge_active ≤ 1` and `judge_pending ≤ 1` at all times
- [ ] `superseded` increases under overload (slow judgments SHOULD supersede candidates)
- [ ] incidents still eventually emit

Record the model/version + date of the last successful run here:

| date | model | notes |
|---|---|---|
| _pending first live run of the template port_ | `gemini-3.1-flash-live-preview` (pinned) | ported from `percept/realtime/*` (verified live 2026-07-14 pre-port) |

## Design notes (the hard-won bits — do not "simplify" these away)

- **Grounding is PULL-only.** Goal changes/incidents only invalidate a local snapshot; the model
  fetches it via the `percept_state` tool. Pushing state into a Gemini Live session is answered OUT
  LOUD and steers later turns ("keeps talking about the armed goals" bug).
- **No local VAD gate on the engine branch.** Gemini's own auto-VAD owns turns; a Silero gate
  silenced the mic entirely in the first live attempt.
- **The readiness kick.** Gemini Live drops all mic audio until it processes an `LLMContextFrame`;
  the seeded greeting context at start opens that gate at t≈0.
- **Pinned live model.** Pipecat's silent default is a different (native-audio) model class that
  connects fine and hears nothing. `gemini-3.1-flash-live-preview` is the verified wire.
- **`user_speaking` comes from the local energy VAD**, which stays live during assistant playback —
  never from the gated Percept world-audio branch (that would blind barge-in during playout).
- **Trust is stamped, never claimed.** Voice-armed watches go through the conductor
  (`LANGUAGE_AUTHORED`, interrupt-clamped); only app/config goals armed in code are
  `STRUCTURED_TRUSTED`.
