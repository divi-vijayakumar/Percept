---
id: PCPT-005
title: Native video judging: _video_part + judge_clip(fps) in Gemini backend
status: backlog
priority: P0
owner: unassigned
wave: B
pr: PR-5
human_test: pending
---

## Why
judge_clip today sends N image Parts. IRA's eval: native video = one judgment per event, conf 0.9 with timestamps, ~10x cheaper on busy scenes. The single biggest genuine gap in their letter.

## Scope
_video_part() (inline video/mp4 + Files API above inline limit); judge_clip(condition, clip, *, fps=1.0) accepting encoded bytes OR frame sequence; frame->mp4 behind a [clip] extra (decide pyav vs opencv in a spike); graceful fallback to multi-frame strip; parse Verdict.events timestamps.

## Acceptance criteria
- [ ] FakeVision clip:: scripted key makes the clip path CI-testable
- [ ] test_clip.py: frame-sequence fallback without encoder; encoder selection; fps clamping; timestamps parsed; missing timestamps -> []
- [ ] Claude backend honestly degrades (same pattern as audio)
- [ ] [clip] extra keeps L1 vendor-free (test_core_no_vendor_imports.py green)

## Human test
Watch one live clip judgment on a temporal event and confirm the verdict carries sensible timestamps.
