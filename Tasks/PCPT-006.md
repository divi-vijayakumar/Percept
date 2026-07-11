---
id: PCPT-006
title: Clip-verify escalation rung + track_clip_verify benchmark
status: backlog
priority: P0
owner: unassigned
wave: B
pr: PR-6
human_test: pending
---

## Why
High-consequence concerns need one richer look before FIRE. Runtime already holds the FrameWindow pre-roll; this judges the same bytes instead of only storing them.

## Scope
MAX_PERCEPTION_LEVEL 2->3: on candidate FIRE/ASK for a wants_clip/verify_only concern, judge the rolling pre-roll ONCE as a clip before the gate fires. New eval/bench/track_clip_verify.py + golden fixtures with strip-invisible events; baseline row in baselines.json.

## Acceptance criteria
- [ ] Ladder order single -> strip -> +audio -> clip-verify (test_escalation.py, test_escalation_policy_track.py)
- [ ] verify_only concerns never clip-judge on rungs 0-2; clip refusal downgrades FIRE -> ASK, never silently drops
- [ ] All escalation tests deterministic via FakeVision + FakeClock
- [ ] Existing offline tracks: ZERO delta vs baselines.json (clip rung is opt-in)
- [ ] track_clip_verify measures judge-calls-per-event, cost via Verdict.usage, timestamp accuracy
- [ ] One PERCEPT_LIVE benchmark-run TRACKS=stable before merge

## Human test
Review the new track's scorecard row: calls-per-event should be ~1 for clip vs N for strip on the golden fixtures.
