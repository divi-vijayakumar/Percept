---
id: PCPT-025
title: Non-blocking judge scheduler — _JudgeScheduler + revision-abort in perceive_judged
status: done
priority: P0
owner: fable-5
wave: boundary-cleanup-1
pr: ""
human_test: pending
updated: 2026-07-14
---

## Scope (plan Steps 1+2)
Capture loop must never await a network judgment. `_JudgeScheduler` in `runtime/worker.py`: single-flight + one latest-pending per source, admission-revision pinning, cancellation-safe done-callback, flush-to-quiescence, `close(drain=)`. `expected_goals_revision` param on `core.perceive_judged()` aborts BEFORE Gate/entity mutation. Flag `judge_dispatch="inline"` first, then flip default to `"async"`. Update `test_harden_e1_loop_parity.py` divergence list (no XPASS). New stats: `judge_superseded`, `judge_revision_dropped`.

## Acceptance
`tests/test_runtime_judge_scheduler.py` (8 cases incl. revision-abort with latch-state assertion, pending-revision-drop, cancel race, flush-to-quiescence). Full `make test` green.

Full spec: plan file `consider-this-improved-architecture-elegant-balloon.md` Steps 1–2.
