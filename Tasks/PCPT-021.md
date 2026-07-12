---
id: PCPT-021
title: R-1 — sustained-duration (hold_s / dwell) compilation
status: ready
priority: P1
owner: claude
wave: r-track
pr: overnight
human_test: pending
updated: 2026-07-11
---

## Why
"without a hard hat for more than 10s" must compile+fire — blocks the PPE demo. NOTE: hold_s already exists as SaccadeHold.hold_s in sampling.py (sensor saccade) — do NOT conflate; this is a NEW gate-level dwell operator.

## Scope
Add hold_s/min_duration to WatchCondition/Concern (contracts); compile whitelist in behavior.py (compile_fire_when ~143); dwell operator wiring in gate.py + clock.py (clock.py documents dwell, implements no operator).

## Acceptance criteria
- [ ] "for more than 10s" compiles and fires deterministically on the replay harness
- [ ] gate byte-for-byte goldens stay green (adds a compile path, must NOT alter existing fires) — CRITICAL
- [ ] additive contract change only; make test test-js parity green

## Human test
Confirm on the replay harness that a 10s-dwell PPE goal fires only after sustained absence.
