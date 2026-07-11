---
id: PCPT-009
title: Per-goal detector fusion (WakePolicy on WatchCondition, AND/veto)
status: backlog
priority: P1
owner: unassigned
wave: C
pr: PR-7c
human_test: pending
---

## Why
WakePolicy noisy-OR is gate-wide and injection-only; multi-cue concerns (motion AND pose-immobility) can't express combination per goal.

## Scope
Carry WakePolicy (weights + AND/veto) on WatchCondition; evaluate per condition in MotionGate; mirror in skills.js wakeScore; extend parity to wakeScore (currently uncovered).

## Acceptance criteria
- [ ] Existing noisy-OR goldens unchanged in test_detectors.py
- [ ] AND/veto truth-table tests Python + JS
- [ ] parity covers wakeScore

## Human test
Author a fall concern (motion AND pose) and confirm it only escalates when both cues agree.
