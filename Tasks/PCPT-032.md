---
id: PCPT-032
title: Low-latency VLM judge — reduce 3–5s request/response floor toward sub-second
status: backlog
priority: P2
owner: ""
wave: research
pr: ""
human_test: n/a
updated: 2026-07-14
---

## Goal
Cut the judge stage from the measured ~3–5s request/response floor toward **sub-second**
detection latency, without giving up honest uncertainty, non-blocking capture, calibration, or the
downstream Gate. This is the research backlog behind the shipped C1 stopgap (see below).

## Status of the immediate unblock (SHIPPED as a probe)
C1 event capsules fix the COVERAGE bug at the current latency. Proven on the sugar clip
(`templates/realtime_voice_watch_agent/scratch_capsule_probe.py`):

| approach | judge calls | blocking | fires |
|---|---:|---|---:|
| async (latest-frame) | 6 | no | 0 |
| inline (every frame) | 56 | yes (415s) | 3–5 |
| **C1 capsule per event** | **7** | **no** | **2 ✓** |

One judge call per T0-opened event, over a `percept.sprite.make_sprite` montage of
{first, peak-salience, last} frames → the salient frame can't be superseded. Latency STILL ~3–5s
(request/response floor) — accepted for now; a 4s-late reliable nudge beats 0% fires.
**Next: fold C1 into the runtime proper** (reuse the existing snippet/pre-roll window + coast +
`judge(frames=…)`), not just the scratch probe.

## The latency research (this card)
Reduce time from *event-visible* → *valid verdict*. Approaches to visit, roughly increasing
sophistication / decreasing generality:

1. **Event capsules** (SHIPPED as probe) — guarantee preservation of the actual event window.
   Fixes coverage; does NOT reduce per-judge latency. The floor under everything below.
2. **Early recognition** — detect an action from its FIRST PARTIAL execution (reaching for the jar,
   hand rising through 30°) rather than the completed act. Trades a little precision for seconds of
   lead. Candidates: online action detection (TOAD/LSTR-style), pose-state machines feeding the
   judge only on partial-onset.
3. **Predictive admission** — use PRECURSORS to raise attention (lower T0 threshold / pre-warm a
   session / start a capsule) BEFORE the event completes, so the judge is already looking when it
   happens. Precursor = motion toward a known region, an object entering an ROI, an audio onset.
4. **Forecast warnings** — for CALIBRATED goal families only, notify on a forecast (P(event soon) >
   threshold) with explicit uncertainty, not on a confirmed fire. Requires the calibration registry
   (assurance work) so a forecast can't silently become a false alarm on a high-FP-cost watch.
5. **Predictive intervention** — domain-specific, AUTHORIZED, consequence-aware early action
   (composes with C2/C3 template families: streaming action session, domain packs). Highest
   assurance bar; interrupt/act authority must be structurally granted, never conductor-authored.

## Backend candidates (from the research review, UNVERIFIED — pull the papers before trusting)
- Cloud live tool-call judges (Gemini Live `report_goal_states` function call; OpenAI Realtime
  out-of-band `input_image` + function call) — NOTE the DOC-VERIFIED-but-recheck 1 FPS / image-only
  Gemini Live limit; may not beat C1's latency, only its connection overhead.
- Local recent-window VLMs (Qwen-family 2B/4B + 2–8 frame memory) — privacy + no round-trip; the
  most promising local route.
- Native streaming VLMs (StreamingVLM / LiveStar / Flash-VStream / VideoLLM-online) — adapt to a
  ternary state head, not commentary; H100-class caveats.
- Specialist online detectors (pose/action/audio/defect) routed by goal family; hybrid
  proposer→VLM-verifier is the leading product hypothesis.

## Guardrails (carry from the problem spec)
- Perception never blocks on the judge. `UNKNOWN ≠ NO` (silence/timeout → UNKNOWN via a report lease,
  never inferred NO). Verdicts are evidence-WINDOW stamped. Gate stays the only thing that fires.
  A session is a cache, not a store. No confidence is "calibrated by definition."
- Do NOT promote any `ContinuousJudgeSession` protocol into `percept.contracts` until ≥2 materially
  different backends satisfy it and the sugar regression passes (spec §14.2 no-speculative-seams).

## Reference material
- `STREAMING-JUDGE-SESSION-PROBLEM-SPEC.md` (repo root) — the problem + measured baselines.
- `CONTINUOUS-JUDGMENT-RESEARCH-SPEC-v2.md` — external research review (event-capsule reframe,
  coverage-vs-latency split, backend landscape, eval harness). Citations UNVERIFIED.
- Probes: `templates/realtime_voice_watch_agent/scratch_{video_smoke,judge_probe,judge_latency,capsule_probe}.py`.
- Related: [[PCPT-031]] (async scheduler supersedes salient frame — the bug this fixes/researches).
