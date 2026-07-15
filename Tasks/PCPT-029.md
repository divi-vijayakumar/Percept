---
id: PCPT-029
title: SDK diet — strip session voice logic, delete percept/realtime, deprecate VoiceTransport
status: done
priority: P0
owner: opus
wave: boundary-cleanup-3
pr: ""
human_test: pending
updated: 2026-07-14
---

## Scope (plan Step 5) — blocked on PCPT-028 merging (template proven green first)
- Strip session voice/delivery members (`attach_transport`, `deliver_fire`, `_speak_fire`, `_pulse_state`, tool handlers, A2V metrics, `_transcribe`, …); `attach_transport`/`deliver_fire` become raising stubs pointing at the template.
- Delete `percept/realtime/{transport,server}.py`; `realtime/__init__.py` = ImportError stub.
- Strip pipecat deps from `[realtime]` extra (name reserved for future SDK streaming judgment).
- Deprecate `VoiceTransport`/`UtteranceEvent`/`Reply` in contracts (docstring + CHANGELOG).
- Delete/move template-owned SDK tests; keep `test_runtime_voice*.py`.

## Acceptance
Full `make test` green; installed-wheel smoke: no pipecat import at SDK import time; CHANGELOG migration section.

Full spec: plan file Step 5.
