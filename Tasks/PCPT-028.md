---
id: PCPT-028
title: templates/realtime_voice_watch_agent — voice embodiment template (Pipecat + Gemini Live)
status: done
priority: P0
owner: fable-5
wave: boundary-cleanup-2
pr: ""
human_test: pending
updated: 2026-07-14
---

## Scope (plan Step 4) — blocked on PCPT-025 + PCPT-026 merging
Full template: `app.py` composition root, `engine_adapter.py` (template-local `VoiceEngine`/`SpeechRequest` + verbatim port of `percept/realtime/transport.py` internals), `media.py` (server + video tee + three-way audio split), `percept_bridge.py` (tools, snapshot invalidation — NO sync_state push), `floor.py` (user > trusted interrupt > wire > deferred; local VAD stays live during playback), `fire_renderer.py`, `engagement.py`, `telemetry.py` (A2V/V2V by incident ID), own `pyproject.toml` (no SDK voice extra), tests incl. offline E2E + public-symbol allowlist boundary test. Makefile `test-templates`.

## Acceptance
`make test-templates` green keyless (pipecat tests skip cleanly); boundary allowlist test passes; offline E2E proves utterance reply + spoken fire + barge-in cancel.

Full spec: plan file Step 4.
