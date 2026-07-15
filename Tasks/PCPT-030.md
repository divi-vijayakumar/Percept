---
id: PCPT-030
title: Conductor misroutes indirect watch phrasings — "if I'm drinking caffeine nudge me" authors nothing
status: todo
priority: P0
owner: ""
wave: post-live-test
pr: ""
human_test: pending
updated: 2026-07-14
---

## Live-test finding (2026-07-14, first browser run of the voice template)
Divi's natural phrasings never armed a watch — the Gemini text conductor routed them as scene chat
(`respond_with_vision`, zero goal_ops):
- "if I'm drinking caffeine just gently nudge me not to" → NOTHING
- "keep track of my caffeine intake and gently nudge me when I drink coffee" → NOTHING
- "nudge me to get back to work if I get distracted or drowsy" → NOTHING (live session)

Canonical imperative phrasings work: "tell me the second I raise my hand" and
"nudge me when I drink coffee" both author + arm correctly (verified in isolation).

## Why it matters
Natural speech IS indirect. The wire model had ALREADY classified intent (it called the `watch`
tool) — the conductor then re-classifies from scratch and loses. Double-classification with the
weaker classifier last.

## Candidate fixes (pick with evals, not vibes)
1. Conductor prompt: routing rules for conditional/desiderative watch phrasings ("if X … nudge me",
   "keep track of …", "help me not to …").
2. SDK seam: `arm_from_language(command, intent_hint="watch")` — caller-supplied classification bias
   for callers that already know (tool-call path). Deterministic clamp still owns everything else.
3. Both.

## Acceptance
- The three failing phrasings above author + arm.
- Routing eval set (add these as cases) shows no regression on question/chat utterances
  (must NOT start arming goals for "did you see me drink coffee?").
- Template E2E: mitigated user-visible half already shipped — bridge speaks up on authoring no-op
  (percept_bridge._authoring_failed, test_watch_that_authors_nothing_owns_up_through_the_floor).

## Related live-test observations (log 2026-07-14)
- Gemini Live refusal loop ("I'm just a language model…") after ~8 min session with audio-watch
  request — conversation session poisoned; reconnect recovers (state is SDK-side). Consider README
  known-issues note + auto-reconnect in engagement.
- Heartbeat proved the scheduler: frames=8542 flowing with judge_calls=0 (no goals armed) — capture
  loop healthy and non-blocking throughout.
