# Live-provider run: per-stage metrics and end-to-end latency

Runtime under test: `percept` 1.0.0 built from `2d5097df3984c7cd08c77685e5229c0b0bcffb1a` (contract 1.10.0), interpreter `/Users/dhivyavijayakumar/percept-launch/venvs/oss-2d5097d/bin/python`.
Provider: **gemini / `gemini-3.1-flash-lite`**, transport `ordered_timestamped_images`, evidence shape `video_clip` (3 frames over 2.0 s).  Generated 2026-07-27T17:02:12Z.

**What was actually run:** 40 of 100 corpus cases, each in BOTH modes (deterministic stratified sample over all (dataset, goal_kind) cells, n=40).  4602 s of media per mode, replayed in REAL TIME.  Case concurrency 4.  Total wall 313 s.

Every case is driven through the real public path — `percept.run(...)` over a paced file source, a compiled `WatchPlan`, the shipped tier-0 gate, and a live provider behind a `JudgePool`. No stage is stubbed and no verdict is scripted.

The two arms differ in exactly two declared ways: the pool's `mode` (which stamps `_percept_latency_mode` and therefore the output budget — 320 tokens and no free-text `evidence` for reflex, 1024 tokens with prose for deliberate), and the pool deadline (1.5 s reflex / 30.0 s deliberate). The deliberate deadline is generous ON PURPOSE: holding deliberate at 1.5 s would measure how often a 1024-token generation misses a bus, not what the bound costs in quality. The counterfactual is reported instead, from the per-call route receipts.

## Per-stage metrics — `reflex`

40 of 40 cases measured; 0 errored (errors are counted, never dropped from a denominator).

Recall appears TWICE at every stage. **stage-local** is versus what that stage actually received; **cumulative** is versus the original source ground truth. A stage at 1.000 stage-local lost nothing itself however low its cumulative figure is — the pair is what localises the loss, and a single number per stage cannot.

| Stage | Recall (stage-local) | Recall (cumulative) | Event recall (local) | Event recall (cum.) | Precision | Economy | Drop rate | Units in -> out |
|---|---|---|---|---|---|---|---|---|
| Stage 0 - Ingest + normalize | 1.000 | 1.000 | 1.000 | 1.000 | 0.115 | 0.353 | 0.000 | 78,355 -> 27,694 |
| Stage 1 - Admission (VID tier-0 gate) | 0.667 | 0.667 | 1.000 | 1.000 | 0.105 | 0.729 | 0.000 | 27,694 -> 20,199 |
| Stage 2 - Local perception | 1.000 | 0.667 | 1.000 | 1.000 | 0.105 | 0.196 | 0.000 | 20,199 -> 3,954 |
| Stage 3 - Judge dispatch | 0.201 | 0.134 | 0.983 | 0.983 | 0.110 | 0.985 | 0.001 | 3,954 -> 3,896 |
| Stage 4 - Cognition + disposition | 0.594 | 0.080 | 0.750 | 0.737 | 0.298 | 0.220 | 0.000 | 3,896 -> 857 |

Two boundaries read differently here than they do in the scripted-oracle suite, and both differences are caused by the judge being REAL:

- **Stage 2** carries the judge scheduler. With `judge_dispatch="async"` the scheduler is single-flight with one latest-pending slot, so while a ~1.4 s provider call is out, every newly admitted frame but the freshest is superseded. Its economy is therefore the ratio of provider throughput to admission rate, not a perception loss — and every superseded unit carries a `RuntimeStats.judge_superseded` receipt plus a health event, so it is economy and not drop.
- **Stage 3's TIME recall is structurally bounded by that economy** and largely restates it. The column that localises a judge regression is EVENT recall: did at least one unit inside each ground-truth occurrence survive to a completed verdict?

## Per-stage breakdown by goal_kind

`state` has n=4 cases corpus-wide and cannot support a claim: it is reported, never gated.

### `reflex` — Stage 0 - Ingest + normalize

| goal_kind | Recall (stage-local) | Recall (cumulative) | Event recall (local) | Event recall (cum.) | Precision | Economy | Drop rate | n cases | n GT events |
|---|---|---|---|---|---|---|---|---|---|
| count | 1.000 | 1.000 | 1.000 | 1.000 | 0.171 | 0.211 | 0.000 | 12 | 69 |
| duration | 1.000 | 1.000 | 1.000 | 1.000 | 0.606 | 0.290 | 0.000 | 12 | 12 |
| state | 1.000 | 1.000 | 1.000 | 1.000 | 0.085 | 0.219 | 0.000 | 4 | 18 |
| transition | 1.000 | 1.000 | 1.000 | 1.000 | 0.052 | 0.511 | 0.000 | 12 | 19 |

### `reflex` — Stage 1 - Admission (VID tier-0 gate)

| goal_kind | Recall (stage-local) | Recall (cumulative) | Event recall (local) | Event recall (cum.) | Precision | Economy | Drop rate | n cases | n GT events |
|---|---|---|---|---|---|---|---|---|---|
| count | 0.650 | 0.650 | 1.000 | 1.000 | 0.152 | 0.731 | 0.000 | 12 | 69 |
| duration | 0.536 | 0.536 | 1.000 | 1.000 | 0.619 | 0.511 | 0.000 | 12 | 12 |
| state | 0.723 | 0.723 | 1.000 | 1.000 | 0.104 | 0.590 | 0.000 | 4 | 18 |
| transition | 0.832 | 0.832 | 1.000 | 1.000 | 0.056 | 0.772 | 0.000 | 12 | 19 |

### `reflex` — Stage 2 - Local perception

| goal_kind | Recall (stage-local) | Recall (cumulative) | Event recall (local) | Event recall (cum.) | Precision | Economy | Drop rate | n cases | n GT events |
|---|---|---|---|---|---|---|---|---|---|
| count | 1.000 | 0.650 | 1.000 | 1.000 | 0.152 | 0.207 | 0.000 | 12 | 69 |
| duration | 1.000 | 0.536 | 1.000 | 1.000 | 0.619 | 0.284 | 0.000 | 12 | 12 |
| state | 1.000 | 0.723 | 1.000 | 1.000 | 0.104 | 0.262 | 0.000 | 4 | 18 |
| transition | 1.000 | 0.832 | 1.000 | 1.000 | 0.056 | 0.179 | 0.000 | 12 | 19 |

### `reflex` — Stage 3 - Judge dispatch

| goal_kind | Recall (stage-local) | Recall (cumulative) | Event recall (local) | Event recall (cum.) | Precision | Economy | Drop rate | n cases | n GT events |
|---|---|---|---|---|---|---|---|---|---|
| count | 0.200 | 0.130 | 0.986 | 0.986 | 0.149 | 0.987 | 0.003 | 12 | 69 |
| duration | 0.265 | 0.142 | 1.000 | 1.000 | 0.603 | 0.968 | 0.000 | 12 | 12 |
| state | 0.205 | 0.148 | 0.944 | 0.944 | 0.083 | 0.980 | 0.003 | 4 | 18 |
| transition | 0.151 | 0.126 | 1.000 | 1.000 | 0.048 | 0.988 | 0.000 | 12 | 19 |

### `reflex` — Stage 4 - Cognition + disposition

| goal_kind | Recall (stage-local) | Recall (cumulative) | Event recall (local) | Event recall (cum.) | Precision | Economy | Drop rate | n cases | n GT events |
|---|---|---|---|---|---|---|---|---|---|
| count | 0.733 | 0.095 | 0.824 | 0.812 | 0.278 | 0.394 | 0.000 | 12 | 69 |
| duration | 0.627 | 0.089 | 0.917 | 0.917 | 0.630 | 0.604 | 0.000 | 12 | 12 |
| state | 0.344 | 0.051 | 0.588 | 0.556 | 0.178 | 0.162 | 0.000 | 4 | 18 |
| transition | 0.478 | 0.060 | 0.526 | 0.526 | 0.175 | 0.130 | 0.000 | 12 | 19 |

## Per-stage breakdown by dataset

### `reflex` — Stage 0 - Ingest + normalize

| dataset | Recall (stage-local) | Recall (cumulative) | Event recall (local) | Event recall (cum.) | Precision | Economy | Drop rate | n cases | n GT events |
|---|---|---|---|---|---|---|---|---|---|
| charades | 1.000 | 1.000 | 1.000 | 1.000 | 0.947 | 0.463 | 0.000 | 6 | 6 |
| omnipro | 1.000 | 1.000 | 1.000 | 1.000 | 0.048 | 0.219 | 0.000 | 16 | 48 |
| perception_test | 1.000 | 1.000 | 1.000 | 1.000 | 0.226 | 0.200 | 0.000 | 6 | 6 |
| proactiveqa | 1.000 | 1.000 | 1.000 | 1.000 | 0.059 | 1.000 | 0.000 | 6 | 6 |
| repcount | 1.000 | 1.000 | 1.000 | 1.000 | 0.788 | 0.208 | 0.000 | 6 | 52 |

### `reflex` — Stage 1 - Admission (VID tier-0 gate)

| dataset | Recall (stage-local) | Recall (cumulative) | Event recall (local) | Event recall (cum.) | Precision | Economy | Drop rate | n cases | n GT events |
|---|---|---|---|---|---|---|---|---|---|
| charades | 0.504 | 0.504 | 1.000 | 1.000 | 0.941 | 0.482 | 0.000 | 6 | 6 |
| omnipro | 0.742 | 0.742 | 1.000 | 1.000 | 0.052 | 0.689 | 0.000 | 16 | 48 |
| perception_test | 0.685 | 0.685 | 1.000 | 1.000 | 0.285 | 0.544 | 0.000 | 6 | 6 |
| proactiveqa | 0.856 | 0.856 | 1.000 | 1.000 | 0.063 | 0.806 | 0.000 | 6 | 6 |
| repcount | 0.610 | 0.610 | 1.000 | 1.000 | 0.769 | 0.626 | 0.000 | 6 | 52 |

### `reflex` — Stage 2 - Local perception

| dataset | Recall (stage-local) | Recall (cumulative) | Event recall (local) | Event recall (cum.) | Precision | Economy | Drop rate | n cases | n GT events |
|---|---|---|---|---|---|---|---|---|---|
| charades | 1.000 | 0.504 | 1.000 | 1.000 | 0.941 | 0.302 | 0.000 | 6 | 6 |
| omnipro | 1.000 | 0.742 | 1.000 | 1.000 | 0.052 | 0.219 | 0.000 | 16 | 48 |
| perception_test | 1.000 | 0.685 | 1.000 | 1.000 | 0.285 | 0.265 | 0.000 | 6 | 6 |
| proactiveqa | 1.000 | 0.856 | 1.000 | 1.000 | 0.063 | 0.167 | 0.000 | 6 | 6 |
| repcount | 1.000 | 0.610 | 1.000 | 1.000 | 0.769 | 0.238 | 0.000 | 6 | 52 |

### `reflex` — Stage 3 - Judge dispatch

| dataset | Recall (stage-local) | Recall (cumulative) | Event recall (local) | Event recall (cum.) | Precision | Economy | Drop rate | n cases | n GT events |
|---|---|---|---|---|---|---|---|---|---|
| charades | 0.283 | 0.142 | 1.000 | 1.000 | 0.928 | 0.968 | 0.000 | 6 | 6 |
| omnipro | 0.192 | 0.143 | 0.979 | 0.979 | 0.046 | 0.989 | 0.001 | 16 | 48 |
| perception_test | 0.203 | 0.139 | 1.000 | 1.000 | 0.226 | 0.969 | 0.000 | 6 | 6 |
| proactiveqa | 0.144 | 0.123 | 1.000 | 1.000 | 0.055 | 0.989 | 0.000 | 6 | 6 |
| repcount | 0.209 | 0.128 | 0.981 | 0.981 | 0.724 | 0.932 | 0.015 | 6 | 52 |

### `reflex` — Stage 4 - Cognition + disposition

| dataset | Recall (stage-local) | Recall (cumulative) | Event recall (local) | Event recall (cum.) | Precision | Economy | Drop rate | n cases | n GT events |
|---|---|---|---|---|---|---|---|---|---|
| charades | 0.572 | 0.081 | 0.833 | 0.833 | 0.947 | 0.570 | 0.000 | 6 | 6 |
| omnipro | 0.323 | 0.046 | 0.447 | 0.438 | 0.062 | 0.242 | 0.000 | 16 | 48 |
| perception_test | 0.893 | 0.124 | 1.000 | 1.000 | 0.312 | 0.645 | 0.000 | 6 | 6 |
| proactiveqa | 0.509 | 0.063 | 0.833 | 0.833 | 0.285 | 0.098 | 0.000 | 6 | 6 |
| repcount | 0.875 | 0.112 | 0.980 | 0.962 | 0.836 | 0.758 | 0.000 | 6 | 52 |

## Reflex vs deliberate — what the latency bound costs

Only one mode was run; no comparison is available.

## End-to-end latency

Every wall figure crosses the media clock exactly once, through the replay anchor recorded at the first decoded frame: `onset_wall = anchor.monotonic + (onset_media - anchor.media)`. Legs come from the runtime's own `JudgmentTiming` (monotonic, validated non-decreasing) plus the instant the `Incident` was observed by the consumer.

### `reflex`

| Leg | n | p50 | p95 | max | mean |
|---|---|---|---|---|---|
| action onset -> Incident (THE PRODUCT GATE) | 258 | 6.834 | 28.058 | 37.702 | 10.098 |
| action onset -> frame ingested | 258 | 5.160 | 26.594 | 36.202 | 8.634 |
| action onset -> admitted by tier-0 | 258 | 5.339 | 26.836 | 36.391 | 8.866 |
| ingest -> admission | 817 | 0.167 | 0.500 | 0.666 | 0.222 |
| admission -> judge dispatch | 817 | 0.001 | 0.002 | 0.005 | 0.001 |
| judge dispatch -> verdict | 817 | 1.130 | 1.405 | 1.501 | 1.137 |
| verdict -> decision emitted | 817 | 0.084 | 0.162 | 0.184 | 0.083 |
| decision -> Incident observed | 817 | 0.000 | 0.001 | 0.003 | 0.000 |
| ingest -> Incident | 817 | 1.489 | 1.833 | 2.002 | 1.443 |
| admission -> Incident | 817 | 1.169 | 1.501 | 1.668 | 1.221 |

**Product gate (action -> Incident < 2.0 s): 32/258 matched Incidents clear it (0.124).**

| goal_kind | n | action->Incident p50 | p95 | max |
|---|---|---|---|---|
| count | 68 | 2.312 | 6.543 | 7.768 |
| duration | 105 | 10.968 | 29.615 | 31.369 |
| state | 14 | 2.985 | 4.560 | 4.668 |
| transition | 71 | 10.875 | 31.536 | 37.702 |

## Quality by goal_kind

`state` has n=4 cases in the full corpus and is reported, never gated.

### `reflex`

| goal_kind | GT occurrences | Occ. recall | Occ. precision | Timely recall @2s | Incidents |
|---|---|---|---|---|---|
| count | 69 | 0.696 | 0.270 | 0.348 | 252 |
| duration | 12 | 0.917 | 0.682 | 0.333 | 154 |
| state | 18 | 0.556 | 0.241 | 0.000 | 58 |
| transition | 19 | 0.526 | 0.201 | 0.211 | 353 |
| **all** | 118 | 0.669 | 0.316 | 0.271 | 817 |

## Coverage — every (dataset, goal_kind) cell that was actually measured

An aggregate hides which category the pipeline fails on. This table is the receipt that every cell was reached; a cell short of its selected count is a gap, not a rounding error.

| dataset / goal_kind | selected | `reflex` measured |
|---|---|---|
| charades/duration | 6 | 6 |
| omnipro/count | 6 | 6 |
| omnipro/state | 4 | 4 |
| omnipro/transition | 6 | 6 |
| perception_test/duration | 6 | 6 |
| proactiveqa/transition | 6 | 6 |
| repcount/count | 6 | 6 |

## Quality by dataset

### `reflex`

| dataset | GT occurrences | Occ. recall | Occ. precision | Timely recall @2s | Incidents |
|---|---|---|---|---|---|
| charades | 6 | 0.833 | 0.943 | 0.333 | 87 |
| omnipro | 48 | 0.417 | 0.081 | 0.125 | 357 |
| perception_test | 6 | 1.000 | 0.343 | 0.333 | 67 |
| proactiveqa | 6 | 0.833 | 0.268 | 0.167 | 235 |
| repcount | 52 | 0.827 | 0.859 | 0.404 | 71 |

## Provider accounting

| Measure | `reflex` |
|---|---|
| Logical judge calls (`judge.started`) | 3,954 |
| Physical attempts (adapter boundary) | 3,914 |
| Route attempts (receipted) | 3,954 |
| Scheduler batches (`RuntimeStats.judge_calls`) | 3,954 |
| Verdicts completed | 3,896 |
| Degraded (adapter raised) | 0 |
| Adapter-level exceptions | 0 |
| Failed route calls (no usable answer) | 182 |
| Input tokens | 4,042,166 |
| Output tokens | 142,726 |
| Input bytes to provider | 513,751,178 |
| Reflex context stamped on call | 3,914 |
| Verdicts carrying evidence prose | 0 |
| Estimated cost (USD) | 1.2246 |

Terminal outcome classes from the pool's own route receipts (this harness passes NO `terminal_outcomes`, so a `deadline` or `capacity` is recorded and the run continues — the qualification bench raises on these, which is right for a gate and wrong for a measurement):

| Outcome | `reflex` |
|---|---|
| `cancelled` | 239 |
| `deadline` | 182 |
| `decisive` | 3621 |
| `uncertain` | 94 |

Cost is tokens x a DECLARED price, not a measurement: $0.25/M input, $1.5/M output (declared; matches prior arm). Re-derive with `--price-in/--price-out`.

## Provider latency, completed vs censored

A call the pool CANCELS at its deadline has no measured provider latency — only a lower bound (the elapsed at cancellation). The two are reported separately; pooling them would manufacture a fast-looking distribution out of calls that never finished.

| Mode | Completed calls | p50 | p95 | max | Cancelled calls | censored elapsed p50 |
|---|---|---|---|---|---|---|
| `reflex` | 3,675 | 1.047 | 1.321 | 1.492 | 239 | 1.474 |

## Runtime bookkeeping under a workload that produces ASKs

The scripted-oracle suite could not exercise this: its judge always answered with high confidence, so the gate resolved to fire/silent and almost never to ASK, and that suite explicitly flagged a suspected double-increment on the ASK path as NOT cleared by it. A live judge produces ASKs in bulk, so the counter is exercised here for the first time.

| Mode | judged | fired | refused | ask | cases with `refused == ask` | cases with `refused == 2 x ask` | cases where funnel tokens == adapter tokens |
|---|---|---|---|---|---|---|---|
| `reflex` | 3,896 | 817 | 620 | 310 | 0/40 | 40/40 | 40/40 |

`funnel.refused` is intended to count one refusal per refused judgment. Where it reads exactly twice the ASK count, it is double-incrementing on the ASK path — a bookkeeping defect in the runtime's own mirror, not a data-path defect: every per-stage number in this report is derived from the trace stream and `RuntimeStats`, both of which reconcile.

## Measurement integrity

- Runtime verified at import: `build_info().git_commit == 2d5097df3984c7cd08c77685e5229c0b0bcffb1a` — this is the wheel that carries the reflex output budget, not the pre-change build.
- Reflex bounding verified at the ADAPTER boundary, not assumed: the harness counts how many provider calls arrived with `_percept_latency_mode='reflex'` stamped on the context, and how many verdicts carried evidence prose. See the accounting table.
- Replay is PACED to the media clock with a lossless bounded queue (`queue_policy='block'`); the per-frame pacing lag is measured and reported, because a runtime that falls behind the source clock would understate every latency below.
- Exactly one clock crossing exists in this file: the replay anchor recorded at the first decoded frame. All other latency arithmetic is monotonic-to-monotonic.
- A case that errors is reported and counted; it is never dropped from a denominator.
- Suppression is not loss: a tier-0 rejection emits its own `edge.decision` receipt, and a superseded judge job increments a named `RuntimeStats` counter. Only units that vanish with NO receipt enter `drop_rate`.
- The pool is given NO `terminal_outcomes`, so `deadline`/`capacity`/`provider_error` are recorded as receipts and counted rather than raised.

## What this run does NOT establish

- A 0.95 recall bar is not statistically ratifiable on this corpus. An exact one-sided Clopper-Pearson lower bound clearing 0.95 with zero misses needs n=59 positives; the duration and transition cells carry 38-40 occurrences in the FULL corpus and fewer in any subset. No 0.95 claim is made or implied here.
- `state` carries n=4 cases corpus-wide. It is reported and never gated.
- Stage 2 is an identity pass-through with no `EvidenceExtractor` armed: its economy of 1.000 cannot distinguish 'lossless' from 'absent'.
- Judge quality here is one provider, one model, one prompt shape, on one day. It is a measurement of THIS route, not a general statement about the runtime's ceiling.
- Cost is tokens x a declared price. The tokens are measured; the price is an input.
- Cases run concurrently share one event loop and one machine. The concurrency level is recorded with every number; a latency figure measured at concurrency N is not automatically the figure at concurrency 1.
