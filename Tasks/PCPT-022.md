---
id: PCPT-022
title: Docs P1/P2 — runtime param list, motion-gate-vs-cadence callout, on_trace, README/quickstart
status: ready
priority: P1
owner: claude
wave: docs
pr: overnight
human_test: pending
updated: 2026-07-11
---

## Scope
- docs/how-to/runtime.md: complete the run() param list (add on_trace, wake_policy, clock, regate_on_goal_change, paused/muted, persistence/restore/autosave_every_s, audio_gate/audio_wake/on_utterance/vad).
- Add a named callout: static/low-motion subject -> whole-frame motion gate won't escalate -> use gate_on="fixed" + min_judge_interval (the decision that sank the realtime demo). ALSO surface the new salience modes: block_max for localized events, ego_compensate for wearables (docs/reference/edge-policy.md already has these — cross-link).
- Document on_trace + decision-trace stages (source.frame, judge.started/finished, gate.classified, incident.created).
- quickstart.md ~129: forward-pointer so the Gemini perceive_judged example doesn't read as "the real-backend way".
- README.md: add a short percept.run(agent, source="webcam:0") "watch a live source" snippet.
- choose-your-path.md:10-11: honest signature (depends on PCPT-017 outcome).

## Acceptance criteria
- [ ] All run() kwargs documented; cadence + salience-mode callout present; on_trace stages documented
- [ ] Docs snippets match the real signature (post PCPT-017)
