---
id: PCPT-019
title: R-3 — mute() alias on RuntimeSession
status: ready
priority: P1
owner: claude
wave: r-track
pr: overnight
human_test: pending
updated: 2026-07-11
---

## Why
API honesty vs the spec surface: session.py has detach_audio_source() + muted() but no mute(). Spec lists mute() as the verb.

## Scope
Add `def mute(self)` to RuntimeSession (packages/percept-vision/percept/runtime/session.py ~line 94) aliasing detach_audio_source() (sets _muted=True). Add/extend a unit test.

## Acceptance criteria
- [ ] RuntimeSession.mute() exists, sets _muted True, mirrors detach_audio_source()
- [ ] unit test covers mute()/muted() round-trip
- [ ] make test green
