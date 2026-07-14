---
id: PCPT-026
title: AttentionIntent + GoalOrigin provenance + session seams (on_goal_change, list_goals)
status: doing
priority: P0
owner: opus
wave: boundary-cleanup-1
pr: ""
human_test: pending
updated: 2026-07-14
---

## Scope (plan Step 3)
- `AttentionIntent` contract (preempt/ttl_s/verbatim/preferred_channel — NO modality enum); CONTRACT_VERSION 1.0.0→1.1.0.
- `GoalOrigin` enum stamped by the SDK entry point (structured arm → STRUCTURED_TRUSTED, conductor GoalOps → LANGUAGE_AUTHORED), caller-supplied origin overwritten; survives snapshot/restore.
- `AttentionPolicy.clamp(requested, origin, consequence, incident)` — conductor-authored interrupt clamps to queue.
- `attention_intent_for(incident, goal, policy)` in `incidents.py`.
- `RuntimeSession.on_goal_change(cb)` subscribed at GoalRegistry revision change (fires for arm_from_language path too), `list_goals()`, public `guard_live_language_path`.

## Acceptance
`test_contracts_attention_intent.py` clamping matrix + origin save/restore + on_goal_change all-mutation-paths + list_goals bounds. Full `make test` green.

Full spec: plan file Step 3.
