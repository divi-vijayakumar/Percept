---
id: PCPT-021
title: R-1 — sustained-duration (hold_s / dwell) compilation
status: done
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
- [x] "for more than 10s" compiles and fires deterministically on the replay harness
- [x] gate byte-for-byte goldens stay green (adds a compile path, must NOT alter existing fires) — CRITICAL
- [x] additive contract change only; make test test-js parity green

## Human test
Confirm on the replay harness that a 10s-dwell PPE goal fires only after sustained absence.

## Feedback
- **2026-07-12 claude:** DONE (commit pending). Dwell implemented in the COGNITION gate (gate.py) — the natural home (per-key sustained-true timer), not tier-0. hold_s authored via goal.state['hold_s'] exactly like deadline_s/cadence_s -> **NO contract/WatchCondition/Concern/JS change** (lower risk than the spec's 'add to WatchCondition' framing; flag if you want it as a first-class Concern field for authorability). Gate goldens (33) BYTE-FOR-BYTE unchanged; py 961 (+8 dwell tests), parity 21/21, js 135. clock.py untouched — the gate reads `now` (the stream timeline, which IS the clock seam).
