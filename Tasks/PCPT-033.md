---
id: PCPT-033
title: Judge tail-latency mitigation — bounded concurrency + hedged calls + timeout→UNKNOWN
status: backlog
priority: P2
owner: ""
wave: research
pr: ""
human_test: pending
updated: 2026-07-15
---

## Why
Measured A→incident latency (capsule + async, real percept.run): **6.7/6.7/8.0/8.0/21.3/23.5s** over six
runs — median ~7s, P95 ~21–23s. The spikes are cloud-VLM TAIL latency: one judge call stalls
(API variance / rate-limit / retry-backoff), and because the scheduler is single-flight, the event's
capsule waits for the one slot to free → clears only at end-of-stream. This is the biggest lever on
the number that actually dominates today, and it needs no new backend (unlike the streaming judge in
[[PCPT-032]]).

## Scope (contained changes to the single-flight `_JudgeScheduler`, worker.py — PCPT-025)
1. **Bounded concurrency (N=2–3 in flight per source):** one stalled call no longer blocks the next
   capsule. HIGHEST leverage on the spike case. Care: the gate's rising-edge latch assumes ORDERED
   verdicts per (goal, entity); out-of-order completions must be reordered or the gate made
   order-tolerant. Revision/generation guards already exist (expected_goals_revision) — reuse them.
2. **Hedged judging:** if the active call hasn't returned in ~6s (≈ the median + margin), launch a
   second on the freshest capsule; first to return wins, cancel the loser. Collapses the tail toward
   the median at ~1.3× call cost. Gate on consequence (only hedge latency-critical goals).
3. **Per-goal judge timeout → UNKNOWN:** a stalled call becomes explicit uncertainty (report lease,
   [[PCPT-032]] semantics), never a silent multi-second hang or an inferred NO.

## Acceptance
- P95 A→incident on the sugar clip drops materially below 21s (target: within ~2× the median).
- No gate correctness regression: rising-edge/refractory still hold under out-of-order verdicts
  (add a test that completes judges out of submission order and asserts single-fire).
- Concurrency + hedge are bounded and observable (new stats: judge_inflight, judge_hedged,
  judge_timeout_unknown); default remains conservative (N configurable, hedge off unless consequence
  opts in).
- `make test` green; the 21s-tail scenario is reproduced in a test with a stalled fake judge.

## Non-goals
- The per-call latency floor (~3–5s) — that's the streaming/faster-judge research ([[PCPT-032]]).
- The delivery hop (incident→voice) — separate, still unmeasured.

## Related
[[PCPT-031]] (coverage — fixed by capsules), [[PCPT-032]] (latency floor / streaming judge),
[[PCPT-025]] (the `_JudgeScheduler` this modifies).
