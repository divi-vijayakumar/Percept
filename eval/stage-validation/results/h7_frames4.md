# Live-provider run: per-stage metrics and end-to-end latency

Runtime under test: `percept` 1.0.0 built from `2d5097df3984c7cd08c77685e5229c0b0bcffb1a` (contract 1.10.0), interpreter `/Users/dhivyavijayakumar/percept-launch/venvs/oss-2d5097d/bin/python`.
Provider: **gemini / `gemini-3.1-flash-lite`**, transport `ordered_timestamped_images`, evidence shape `video_clip` (4 frames over 2.0 s).  Generated 2026-07-27T12:23:51Z.

**What was actually run:** 40 of 100 corpus cases, each in BOTH modes (deterministic stratified sample over all (dataset, goal_kind) cells, n=40).  4602 s of media per mode, replayed in REAL TIME.  Case concurrency 4.  Total wall 1200 s.

Every case is driven through the real public path — `percept.run(...)` over a paced file source, a compiled `WatchPlan`, the shipped tier-0 gate, and a live provider behind a `JudgePool`. No stage is stubbed and no verdict is scripted.

The two arms differ in exactly two declared ways: the pool's `mode` (which stamps `_percept_latency_mode` and therefore the output budget — 320 tokens and no free-text `evidence` for reflex, 1024 tokens with prose for deliberate), and the pool deadline (1.5 s reflex / 30.0 s deliberate). The deliberate deadline is generous ON PURPOSE: holding deliberate at 1.5 s would measure how often a 1024-token generation misses a bus, not what the bound costs in quality. The counterfactual is reported instead, from the per-call route receipts.

## Per-stage metrics — `reflex`

40 of 40 cases measured; 0 errored (errors are counted, never dropped from a denominator).

Recall appears TWICE at every stage. **stage-local** is versus what that stage actually received; **cumulative** is versus the original source ground truth. A stage at 1.000 stage-local lost nothing itself however low its cumulative figure is — the pair is what localises the loss, and a single number per stage cannot.

| Stage | Recall (stage-local) | Recall (cumulative) | Event recall (local) | Event recall (cum.) | Precision | Economy | Drop rate | Units in -> out |
|---|---|---|---|---|---|---|---|---|
| Stage 0 - Ingest + normalize | 1.000 | 1.000 | 1.000 | 1.000 | 0.115 | 0.353 | 0.000 | 78,355 -> 27,694 |
| Stage 1 - Admission (VID tier-0 gate) | 0.667 | 0.667 | 1.000 | 1.000 | 0.105 | 0.729 | 0.000 | 27,694 -> 20,199 |
| Stage 2 - Local perception | 1.000 | 0.667 | 1.000 | 1.000 | 0.105 | 0.152 | 0.000 | 20,199 -> 3,079 |
| Stage 3 - Judge dispatch | 0.149 | 0.099 | 0.932 | 0.932 | 0.116 | 0.884 | 0.004 | 3,079 -> 2,723 |
| Stage 4 - Cognition + disposition | 0.189 | 0.019 | 0.327 | 0.305 | 0.342 | 0.065 | 0.000 | 2,723 -> 177 |

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
| count | 1.000 | 0.650 | 1.000 | 1.000 | 0.152 | 0.154 | 0.000 | 12 | 69 |
| duration | 1.000 | 0.536 | 1.000 | 1.000 | 0.619 | 0.231 | 0.000 | 12 | 12 |
| state | 1.000 | 0.723 | 1.000 | 1.000 | 0.104 | 0.204 | 0.000 | 4 | 18 |
| transition | 1.000 | 0.832 | 1.000 | 1.000 | 0.056 | 0.141 | 0.000 | 12 | 19 |

### `reflex` — Stage 3 - Judge dispatch

| goal_kind | Recall (stage-local) | Recall (cumulative) | Event recall (local) | Event recall (cum.) | Precision | Economy | Drop rate | n cases | n GT events |
|---|---|---|---|---|---|---|---|---|---|
| count | 0.137 | 0.089 | 0.913 | 0.913 | 0.154 | 0.882 | 0.007 | 12 | 69 |
| duration | 0.190 | 0.102 | 1.000 | 1.000 | 0.600 | 0.860 | 0.009 | 12 | 12 |
| state | 0.128 | 0.093 | 0.944 | 0.944 | 0.076 | 0.859 | 0.000 | 4 | 18 |
| transition | 0.130 | 0.108 | 0.947 | 0.947 | 0.058 | 0.892 | 0.003 | 12 | 19 |

### `reflex` — Stage 4 - Cognition + disposition

| goal_kind | Recall (stage-local) | Recall (cumulative) | Event recall (local) | Event recall (cum.) | Precision | Economy | Drop rate | n cases | n GT events |
|---|---|---|---|---|---|---|---|---|---|
| count | 0.268 | 0.024 | 0.333 | 0.304 | 0.305 | 0.135 | 0.000 | 12 | 69 |
| duration | 0.219 | 0.022 | 0.583 | 0.583 | 0.660 | 0.203 | 0.000 | 12 | 12 |
| state | 0.250 | 0.023 | 0.235 | 0.222 | 0.218 | 0.088 | 0.000 | 4 | 18 |
| transition | 0.080 | 0.009 | 0.222 | 0.211 | 0.186 | 0.025 | 0.000 | 12 | 19 |

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
| charades | 1.000 | 0.504 | 1.000 | 1.000 | 0.941 | 0.239 | 0.000 | 6 | 6 |
| omnipro | 1.000 | 0.742 | 1.000 | 1.000 | 0.052 | 0.165 | 0.000 | 16 | 48 |
| perception_test | 1.000 | 0.685 | 1.000 | 1.000 | 0.285 | 0.222 | 0.000 | 6 | 6 |
| proactiveqa | 1.000 | 0.856 | 1.000 | 1.000 | 0.063 | 0.133 | 0.000 | 6 | 6 |
| repcount | 1.000 | 0.610 | 1.000 | 1.000 | 0.769 | 0.195 | 0.000 | 6 | 52 |

### `reflex` — Stage 3 - Judge dispatch

| dataset | Recall (stage-local) | Recall (cumulative) | Event recall (local) | Event recall (cum.) | Precision | Economy | Drop rate | n cases | n GT events |
|---|---|---|---|---|---|---|---|---|---|
| charades | 0.203 | 0.102 | 1.000 | 1.000 | 0.949 | 0.861 | 0.008 | 6 | 6 |
| omnipro | 0.137 | 0.102 | 0.958 | 0.958 | 0.049 | 0.885 | 0.004 | 16 | 48 |
| perception_test | 0.145 | 0.100 | 1.000 | 1.000 | 0.217 | 0.860 | 0.009 | 6 | 6 |
| proactiveqa | 0.129 | 0.111 | 1.000 | 1.000 | 0.068 | 0.899 | 0.001 | 6 | 6 |
| repcount | 0.133 | 0.081 | 0.885 | 0.885 | 0.705 | 0.743 | 0.018 | 6 | 52 |

### `reflex` — Stage 4 - Cognition + disposition

| dataset | Recall (stage-local) | Recall (cumulative) | Event recall (local) | Event recall (cum.) | Precision | Economy | Drop rate | n cases | n GT events |
|---|---|---|---|---|---|---|---|---|---|
| charades | 0.233 | 0.024 | 0.833 | 0.833 | 1.000 | 0.229 | 0.000 | 6 | 6 |
| omnipro | 0.136 | 0.014 | 0.130 | 0.125 | 0.083 | 0.081 | 0.000 | 16 | 48 |
| perception_test | 0.150 | 0.015 | 0.333 | 0.333 | 0.188 | 0.174 | 0.000 | 6 | 6 |
| proactiveqa | 0.082 | 0.009 | 0.500 | 0.500 | 0.412 | 0.014 | 0.000 | 6 | 6 |
| repcount | 0.345 | 0.028 | 0.435 | 0.385 | 0.856 | 0.284 | 0.000 | 6 | 52 |

## Reflex vs deliberate — what the latency bound costs

Only one mode was run; no comparison is available.

## End-to-end latency

Every wall figure crosses the media clock exactly once, through the replay anchor recorded at the first decoded frame: `onset_wall = anchor.monotonic + (onset_media - anchor.media)`. Legs come from the runtime's own `JudgmentTiming` (monotonic, validated non-decreasing) plus the instant the `Incident` was observed by the consumer.

### `reflex`

| Leg | n | p50 | p95 | max | mean |
|---|---|---|---|---|---|
| action onset -> Incident (THE PRODUCT GATE) | 69 | 6.735 | 26.269 | 30.168 | 9.666 |
| action onset -> frame ingested | 69 | 4.922 | 24.619 | 28.568 | 8.068 |
| action onset -> admitted by tier-0 | 69 | 5.234 | 24.790 | 28.968 | 8.297 |
| ingest -> admission | 192 | 0.167 | 0.499 | 0.586 | 0.224 |
| admission -> judge dispatch | 192 | 0.001 | 0.002 | 0.008 | 0.001 |
| judge dispatch -> verdict | 192 | 1.356 | 1.492 | 1.502 | 1.322 |
| verdict -> decision emitted | 192 | 0.080 | 0.161 | 0.166 | 0.077 |
| decision -> Incident observed | 192 | 0.000 | 0.001 | 0.002 | 0.000 |
| ingest -> Incident | 192 | 1.666 | 1.999 | 2.002 | 1.625 |
| admission -> Incident | 192 | 1.469 | 1.551 | 1.668 | 1.401 |

**Product gate (action -> Incident < 2.0 s): 6/69 matched Incidents clear it (0.087).**

| goal_kind | n | action->Incident p50 | p95 | max |
|---|---|---|---|---|
| count | 22 | 2.949 | 6.731 | 7.635 |
| duration | 33 | 12.432 | 29.448 | 30.168 |
| state | 6 | 3.252 | 3.793 | 3.834 |
| transition | 8 | 11.395 | 24.847 | 28.756 |

## Quality by goal_kind

`state` has n=4 cases in the full corpus and is reported, never gated.

### `reflex`

| goal_kind | GT occurrences | Occ. recall | Occ. precision | Timely recall @2s | Incidents |
|---|---|---|---|---|---|
| count | 69 | 0.290 | 0.314 | 0.043 | 70 |
| duration | 12 | 0.583 | 0.673 | 0.167 | 49 |
| state | 18 | 0.333 | 0.222 | 0.000 | 27 |
| transition | 19 | 0.211 | 0.174 | 0.053 | 46 |
| **all** | 118 | 0.314 | 0.359 | 0.051 | 192 |

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
| charades | 6 | 0.833 | 1.000 | 0.167 | 30 |
| omnipro | 48 | 0.167 | 0.087 | 0.021 | 103 |
| perception_test | 6 | 0.333 | 0.158 | 0.167 | 19 |
| proactiveqa | 6 | 0.500 | 0.368 | 0.000 | 19 |
| repcount | 52 | 0.365 | 0.952 | 0.058 | 21 |

## Provider accounting

| Measure | `reflex` |
|---|---|
| Logical judge calls (`judge.started`) | 3,079 |
| Physical attempts (adapter boundary) | 3,026 |
| Route attempts (receipted) | 3,078 |
| Scheduler batches (`RuntimeStats.judge_calls`) | 3,079 |
| Verdicts completed | 2,723 |
| Degraded (adapter raised) | 0 |
| Adapter-level exceptions | 0 |
| Failed route calls (no usable answer) | 1,781 |
| Input tokens | 4,232,983 |
| Output tokens | 34,092 |
| Input bytes to provider | 539,992,807 |
| Reflex context stamped on call | 3,026 |
| Verdicts carrying evidence prose | 0 |
| Estimated cost (USD) | 1.1094 |

Terminal outcome classes from the pool's own route receipts (this harness passes NO `terminal_outcomes`, so a `deadline` or `capacity` is recorded and the run continues — the qualification bench raises on these, which is right for a gate and wrong for a measurement):

| Outcome | `reflex` |
|---|---|
| `cancelled` | 2115 |
| `capacity` | 13 |
| `deadline` | 1768 |
| `decisive` | 871 |
| `uncertain` | 79 |

Cost is tokens x a DECLARED price, not a measurement: $0.25/M input, $1.5/M output (ai.google.dev/gemini-api/docs/pricing). Re-derive with `--price-in/--price-out`.

## Provider latency, completed vs censored

A call the pool CANCELS at its deadline has no measured provider latency — only a lower bound (the elapsed at cancellation). The two are reported separately; pooling them would manufacture a fast-looking distribution out of calls that never finished.

| Mode | Completed calls | p50 | p95 | max | Cancelled calls | censored elapsed p50 |
|---|---|---|---|---|---|---|
| `reflex` | 912 | 1.308 | 1.465 | 4.136 | 2,110 | 1.477 |

## Runtime bookkeeping under a workload that produces ASKs

The scripted-oracle suite could not exercise this: its judge always answered with high confidence, so the gate resolved to fire/silent and almost never to ASK, and that suite explicitly flagged a suspected double-increment on the ASK path as NOT cleared by it. A live judge produces ASKs in bulk, so the counter is exercised here for the first time.

| Mode | judged | fired | refused | ask | cases with `refused == ask` | cases with `refused == 2 x ask` | cases where funnel tokens == adapter tokens |
|---|---|---|---|---|---|---|---|
| `reflex` | 2,723 | 192 | 3,756 | 1,878 | 0/40 | 40/40 | 38/40 |

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
