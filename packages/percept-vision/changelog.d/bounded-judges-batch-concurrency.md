### Multi-goal judging — the batch period stops being the latency

**The defect.** A frame admits N goals and each spends its own judge call. `Percept.perceive_judged`
awaited them one after another, so a frame's makespan was the **sum** of N provider calls. Because
the runtime's judge scheduler is single-flight (one in flight, one latest-wins pending — ADR-001 §9)
and admits nothing new until that coroutine returns, the sum WAS the batch period and therefore WAS
the onset→incident latency.

An 82-minute run with 73 goals armed on one source measured it: 60 batches across 4,897s (81.6s per
batch), onset→incident p50 **79.8s** / max **94.0s** against a 2s product gate, and **16,571 of
16,631 admissions (99.6%) superseded**. Nothing was starved — every goal was judged exactly 60 times,
Gini 0.0. Fairness was never the problem; serialization was.

Separately, the shipped single-goal `decision_deadline_s` was applied unchanged to the whole batch,
so a 73-goal batch was cancelled **wholesale** and delivered nothing — including for goals that had
already answered. The long run only produced output because the deadline had been scaled 73x by hand.

**What changed.**

- `Percept.perceive_judged` now runs a **dispatch pass** and an **apply pass** over the same goal
  list, in the same order. Pass 1 prepares each watch goal and launches its judge call against a
  bounded slot pool; pass 2 applies each result. The sum becomes a max; evidence-sequence
  reservation, funnel accounting, gate/executor mutation and the order of the returned events all
  still follow goal order.
- New `Percept.judge_concurrency` / `Percept.create(judge_concurrency=…)` (tunable
  `Percept.create.judge_concurrency`, owner `ExecutionPolicy`), defaulting to
  `agent._DEFAULT_JUDGE_CONCURRENCY = 8` — sized from the measured single-call mean (~1.69s) so one
  source sustains ~4.7 judge calls/s against one credential.
- `JudgePool.max_inflight` publishes the pool's concurrent-decision capacity (the **minimum** route
  occupancy, because every eligible route is scheduled per decision).
  `Percept.effective_judge_concurrency()` floors the configured ceiling by every declared capacity in
  the batch, so the core can never fan out past what a deployment's routes declared. The bound is a
  queue, not a lottery: `asyncio.Semaphore` is FIFO, so a goal late in the list waits, never starves.
- The judge scheduler's absolute budget is now **derived**: `decision_deadline_s` stays the
  per-decision number every caller authors, and the batch gets that number times the concurrency
  waves it actually needs. Per-goal lateness remains a per-goal outcome (`JudgePool.deadline_s`,
  then the fail-closed degrade to ASK). One goal's budget can no longer cancel twenty goals'
  answers. The batch number is a **structural worst case used as a backstop**, not a latency budget:
  every call is already bounded above by the pool, so `ceil(N / K)` bounded waves is the ceiling.
  The multiplier is not a tax on batching — it collapses to 1 at `K >= N`, so widening
  `judge_concurrency` shrinks the batch's real makespan and its backstop together. At the shipped
  `REFLEX_DECISION_DEADLINE_S = 8.4` with K=8, a 73-goal batch backstops at 84s, deliberately above
  `stall_after_s = 20.0` so the loud health event fires before the silent cancellation ever could.

**What it does not buy.** Pooling converts a sum into a max, and the expected max of N draws sits
near the distribution's N/(N+1) quantile — a 73-goal batch lands near p99 no matter how many slots it
gets. Concurrency removes the waves; only hedging would move the tail. `tests/
test_judge_batch_concurrency.py` measures the whole N x K curve offline against a judge that sleeps
the real measured distribution, and its `--measure` mode reprints it at real speed.

**Not changed, deliberately.** `_JudgeScheduler._dispatch` (the split protocol's provider phase)
stays sequential: a route may declare `terminal_outcomes`, and "spend no further provider call after
one trips" is only enforceable if the next job has not started. No production `run()` path constructs
that scheduler today.
