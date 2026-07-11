---
id: PCPT-012
title: Edge.processAudio + adaptive noise floor (sdk_request_001)
status: backlog
priority: P1
owner: unassigned
wave: A
pr: PR-8
human_test: pending
---

## Why
percept-edge has frameEnergy, EnergyVad (hangover + turn-finalization), AcousticOnset - but no frame-free ingest: the only path to a Tier0Signal rides a camera frame. PERCIA-031 re-implemented this in web-core.

## Scope
edge.processAudio(samples,{sampleRate,ts}) -> frameEnergy -> SkillBank energy channel + wake policy -> Tier0Signal|null. EnergyVad: opt-in adaptive noise floor (EMA of low-energy frames) + dBFS threshold option. Depends on PCPT-001 (Tier0Signal.energy).

## Acceptance criteria
- [ ] vad.test.js: existing hangover/turn-finalization goldens UNCHANGED; adaptive-floor convergence; dBFS<->linear
- [ ] edge.test.js: silence -> null; speech -> Tier0Signal with energy; processFrame snapshot-identical to before
- [ ] Withhold-rate golden: fixture PCM sequence withholds ~32% of chunks (exact count asserted)
- [ ] parity/compare.py drives same PCM energy through Python AcousticOnset/GatedWatchLoop and JS processAudio - identical to 1e-9

## Human test
Speak into the mic demo: silence chunks withheld, speech chunks emit signals; withhold % near web-core's measured 32%.
