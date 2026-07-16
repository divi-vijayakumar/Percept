# Percept Templates

Copyable reference applications demonstrating Percept for specific use cases. Install the SDK, copy a template directory, and adapt to your needs.

## Getting Started

Each template is a self-contained Python application:

```bash
cd templates/<template_name>
pip install -e .
python app.py
```

The SDK is a shared dependency — install it once, then each template installs its own extra requirements.

## Available Templates

### `realtime_voice_watch_agent/`

**Voice-native watch agent with live perception and real-time voice interaction.**

A complete working example combining:
- **Live vision**: a Gemini vision backend judging the same camera the wire sees
- **Voice I/O**: Pipecat SmallWebRTC for peer-to-peer audio (browser AEC — no echo loop)
- **Real-time voice model**: Gemini Live (pinned `gemini-3.1-flash-live-preview`) for natural conversation
- **Multi-signal floor**: `user_speaking` (local energy VAD) + `wire_speaking` (engine output edges) + `fire_playing` (renderer state); priority user > trusted interrupt > wire reply > deferred
- **Grounding**: pull-only state snapshots (the `percept_state` tool) — no sync push, which Gemini Live answers out loud
- **Trust**: voice-armed watches stay `LANGUAGE_AUTHORED` (interrupt-clamped); only code/config goals are `STRUCTURED_TRUSTED`

**Install & Run:**
```bash
pip install -e packages/percept-vision   # once, from the repo root (dev checkout)
cd templates/realtime_voice_watch_agent
pip install -e .
export GEMINI_API_KEY=<your_key>
python app.py
# Open browser to http://127.0.0.1:8930, allow camera + mic
```

**Tests:** `make test-templates` from the repo root (also part of `make test`). Keyless — the
pipecat-dependent engine tests skip cleanly when the template's deps aren't installed.

**Tested workflows** (see `tests/test_app_offline.py` — the end-to-end offline proof — plus the
live-test checklist in the template's own `README.md`):
- "What do you see?" → live camera feed analysis
- "Tell me when I raise my hand" → instant "On it." + spoken proactive nudge on the rising edge
- Fire during user speech queues behind the turn (priority ladder); TTL drop-if-stale
- Conductor-authored interrupt clamped to queue; structured-trusted interrupt preempts
- Barge-in during incident playback stops playout

**Key files:**
- `app.py` — composition root (public SDK seams only: `Percept.create` + `percept.start` + `RuntimeSession`)
- `engine_adapter.py` — template-local `VoiceEngine`/`SpeechRequest` seam + the Pipecat/Gemini-Live engine (readiness kick, model pin, pull-grounding tool)
- `media.py` — WebRTC signaling + video tee + the three-way audio split (engine / local VAD / gated world-audio)
- `percept_bridge.py` — tool schemas, watch commands, grounding snapshot invalidation, incident → `AttentionIntent` → floor
- `floor.py` — three-signal priority ladder (user, interrupt, reply, deferred) + TTL + coalescing
- `fire_renderer.py` — channel selection (speech, webhook, log; `action` documented out of scope)
- `engagement.py`, `telemetry.py` — always-open demo flag + stubs; A2V/V2V latency by incident ID
- `tests/test_boundary.py` — the public-seam allowlist scan (no `core.goals`, no `._*` on SDK objects)

**References:**
- See `.claude/skills/percept-agent/` for the lightweight offline watch-agent templates
- See `docs/how-to/runtime.md` for the perception runtime API
