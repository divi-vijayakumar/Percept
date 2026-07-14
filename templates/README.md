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
- **Live vision**: Gemini 2.0 Flash for real-time frame analysis
- **Voice I/O**: Pipecat SmallWebRTC for peer-to-peer audio
- **Real-time voice model**: Gemini Live for natural conversation
- **Multi-signal floor**: user speech + engine output + Percept incidents
- **Grounding**: pull-only state snapshots, no sync push to avoid re-answering

**Install & Run:**
```bash
cd templates/realtime_voice_watch_agent
pip install -e .
export GEMINI_API_KEY=<your_key>
python app.py
# Open browser to localhost:8000, allow camera + mic
```

**Tested workflows** (see `tests/test_app_offline.py` for verification):
- "What do you see?" → live camera feed analysis
- "Tell me when I raise my hand" → spoken proactive nudge on rising edge
- Fire during user speech queues behind the turn (priority ladder)
- Conductor-authored interrupt clamped to queue; structured-trusted interrupt preempts
- Barge-in during incident playback stops playout

**Key files:**
- `app.py` — composition root (SDK + Pipecat + Gemini Live)
- `engine_adapter.py` — Pipecat voice engine integration + pull-only grounding
- `bridge.py` — tool schemas, watch commands, grounding snapshot invalidation
- `floor.py` — three-signal priority ladder (user, interrupt, reply)
- `fire_renderer.py` — channel selection (speech, webhook, action)

**References:**
- See `.claude/skills/percept-agent/` for the lightweight offline watch-agent templates
- See `docs/how-to/runtime.md` for the perception runtime API
