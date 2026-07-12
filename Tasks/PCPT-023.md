---
id: PCPT-023
title: v2v watch-agent demo — verify harness boots + pipeline starts
status: ready
priority: P1
owner: claude
wave: demo
pr: overnight
human_test: required
updated: 2026-07-11
---

## Why
Plan item 2: demo of the v2v watch agent after docs are fixed. The OSS demo harness (Pipecat SmallWebRTC P2P + Gemini Live) lives in scratch/watch_demo/. Live browser test REQUIRES a human — autonomous scope is: confirm it boots, imports, and the pipeline starts without a paid key.

## Scope
- Locate + run scratch/watch_demo/{pipecat_voice_transport,percept_pipecat_server}.py; confirm boot + pipeline start (no live browser).
- Confirm the demo uses the fixed run() surface + arming "tell me when I'm drinking coffee" with the desk preset (block_max) is wired.
- Document exactly what a human must do for the live browser test.

## Acceptance criteria
- [ ] harness imports + boots headless; pipeline start logged; no crash
- [ ] a written run-book for the human live test (deps, ports, browser URL)
- [ ] blockers (paid key, browser) clearly flagged — NOT force-built

## Human test
Run the browser live test per the run-book; confirm a spoken fire on a real webcam.
