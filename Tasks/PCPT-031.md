---
id: PCPT-031
title: async scheduler + slow deliberate judge supersedes the salient frame — live watches miss brief events
status: done
priority: P0
owner: ""
wave: post-live-test
pr: ""
human_test: verified-headless
updated: 2026-07-14
---

## Finding (headless video proof, 2026-07-14)
Video `video_6860.mp4` (35s, person makes tea, adds sugar at t≈26s). Synthetic goal armed at t0:
"the person adds sugar…". Funnel (`templates/realtime_voice_watch_agent/scratch_video_smoke.py`):

| dispatch | escalations | judge_calls | superseded | fires |
|---|---|---|---|---|
| async (default) | 56 | 6 | 50 | **0** |
| inline (escape hatch) | 56 | 56 | 0 | **3 ✓** |

Judge is CORRECT (direct probe: t=26.2s satisfied=True conf=0.9; all other sampled frames False).
Judge latency ~3-5s/call, floored by model reasoning (NOT image size: 384p still ~3s, still True).

## Root cause
`_JudgeScheduler` (PCPT-025) is single-flight active+latest-pending — right for a LIVE camera (drop
stale, judge freshest). But a ~4s request/response `GeminiVision.judge` under sustained motion
supersedes the mid-window PEAK: while judge N runs (t≈24→28s), the pour at t=26 is superseded; when
the judge frees, latest-pending hands it the t≈28s "done" frame. 50 of 56 escalations dropped; the
sugar frame among them. This is the SAME reason the live browser coffee/hand tests never fired.

## RESOLVED (2026-07-15) — C1 capsules wired into the live path
Fixed by making the judge frame a montage of the recent window (capsule.py::CapsuleBuilder →
media.VideoTee(capsule=True)), so the salient frame rides inside every judge frame and can't be
superseded. Video test through the REAL percept.run in ASYNC dispatch: single-frame fires=0 →
capsule fires=1 (incident at the pour, ts=26.3s), same 6 judged / 50 superseded, no inline blocking.
Latency floor (~3-5s) is unchanged and tracked separately in [[PCPT-032]].

## Why it mattered
The `async` default silently missed brief events whenever `judge_latency > event_duration`. For
request/response judges (~3-5s) that is MOST brief actions (a sip, a hand-raise, adding sugar).

## Options (pick with the template family in mind)
1. **Per-profile default**: recorded-clip / accuracy worker → `inline`; live low-latency → `async`.
   Ship the realtime voice template pointing docs at this knob (SHORT-TERM, cheap).
2. **Bounded judge concurrency** N>1 per goal (2-3 in flight) so a burst is covered without a full
   inline stall — needs care re: gate-state serialization.
3. **The real answer = the deferred vNext L1 rung**: a persistent StreamingJudgeSession that emits
   continuously instead of a 4s request/response per frame. Superseding is only pathological because
   the judge is slow-and-discrete. This is the strongest argument yet for building L1.
4. **Downsize judge frame** (1080p→512p): ~30% latency win, helps but does not close the gap alone.

## Immediate (shipped this session)
- Funnel instrumentation + honest authoring-failure nudge (PCPT-030 bridge).
- Headless smoke harness proving the fire end-to-end (scratch_video_smoke.py, inline).

## Related
[[PCPT-030]] conductor misroutes indirect phrasings. Live delivery (fire→voice through the held wire)
still unproven end-to-end — but fire→incident is now proven; delivery is the next isolated segment.
