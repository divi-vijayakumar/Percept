# Live-provider run: per-stage metrics and end-to-end latency

Runtime under test: `percept` 1.0.0 built from `2d5097df3984c7cd08c77685e5229c0b0bcffb1a` (contract 1.10.0), interpreter `/Users/dhivyavijayakumar/percept-launch/venvs/oss-2d5097d/bin/python`.
Provider: **gemini / `gemini-3.1-flash-lite`**, transport `ordered_timestamped_images`, evidence shape `video_clip` (3 frames over 2.0 s).  Generated 2026-07-27T10:33:25Z.

**What was actually run:** 100 of 100 corpus cases, each in BOTH modes (full corpus).  11284 s of media per mode, replayed in REAL TIME.  Case concurrency 4.  Total wall 2871 s.

Every case is driven through the real public path — `percept.run(...)` over a paced file source, a compiled `WatchPlan`, the shipped tier-0 gate, and a live provider behind a `JudgePool`. No stage is stubbed and no verdict is scripted.

The two arms differ in exactly two declared ways: the pool's `mode` (which stamps `_percept_latency_mode` and therefore the output budget — 320 tokens and no free-text `evidence` for reflex, 1024 tokens with prose for deliberate), and the pool deadline (1.5 s reflex / 30.0 s deliberate). The deliberate deadline is generous ON PURPOSE: holding deliberate at 1.5 s would measure how often a 1024-token generation misses a bus, not what the bound costs in quality. The counterfactual is reported instead, from the per-call route receipts.

## Per-stage metrics — `reflex`

100 of 100 cases measured; 0 errored (errors are counted, never dropped from a denominator).

Recall appears TWICE at every stage. **stage-local** is versus what that stage actually received; **cumulative** is versus the original source ground truth. A stage at 1.000 stage-local lost nothing itself however low its cumulative figure is — the pair is what localises the loss, and a single number per stage cannot.

| Stage | Recall (stage-local) | Recall (cumulative) | Event recall (local) | Event recall (cum.) | Precision | Economy | Drop rate | Units in -> out |
|---|---|---|---|---|---|---|---|---|
| Stage 0 - Ingest + normalize | 1.000 | 1.000 | 1.000 | 1.000 | 0.165 | 0.437 | 0.000 | 155,189 -> 67,775 |
| Stage 1 - Admission (VID tier-0 gate) | 0.661 | 0.661 | 1.000 | 1.000 | 0.139 | 0.780 | 0.000 | 67,775 -> 52,868 |
| Stage 2 - Local perception | 1.000 | 0.661 | 1.000 | 1.000 | 0.139 | 0.160 | 0.000 | 52,868 -> 8,467 |
| Stage 3 - Judge dispatch | 0.177 | 0.117 | 0.964 | 0.964 | 0.161 | 0.958 | 0.002 | 8,467 -> 8,112 |
| Stage 4 - Cognition + disposition | 0.514 | 0.060 | 0.681 | 0.657 | 0.458 | 0.181 | 0.000 | 8,112 -> 1,470 |

Two boundaries read differently here than they do in the scripted-oracle suite, and both differences are caused by the judge being REAL:

- **Stage 2** carries the judge scheduler. With `judge_dispatch="async"` the scheduler is single-flight with one latest-pending slot, so while a ~1.4 s provider call is out, every newly admitted frame but the freshest is superseded. Its economy is therefore the ratio of provider throughput to admission rate, not a perception loss — and every superseded unit carries a `RuntimeStats.judge_superseded` receipt plus a health event, so it is economy and not drop.
- **Stage 3's TIME recall is structurally bounded by that economy** and largely restates it. The column that localises a judge regression is EVENT recall: did at least one unit inside each ground-truth occurrence survive to a completed verdict?

## Per-stage breakdown by goal_kind

`state` has n=4 cases corpus-wide and cannot support a claim: it is reported, never gated.

### `reflex` — Stage 0 - Ingest + normalize

| goal_kind | Recall (stage-local) | Recall (cumulative) | Event recall (local) | Event recall (cum.) | Precision | Economy | Drop rate | n cases | n GT events |
|---|---|---|---|---|---|---|---|---|---|
| count | 1.000 | 1.000 | 1.000 | 1.000 | 0.354 | 0.212 | 0.000 | 28 | 233 |
| duration | 1.000 | 1.000 | 1.000 | 1.000 | 0.554 | 0.231 | 0.000 | 40 | 40 |
| state | 1.000 | 1.000 | 1.000 | 1.000 | 0.085 | 0.219 | 0.000 | 4 | 18 |
| transition | 1.000 | 1.000 | 1.000 | 1.000 | 0.084 | 0.678 | 0.000 | 28 | 38 |

### `reflex` — Stage 1 - Admission (VID tier-0 gate)

| goal_kind | Recall (stage-local) | Recall (cumulative) | Event recall (local) | Event recall (cum.) | Precision | Economy | Drop rate | n cases | n GT events |
|---|---|---|---|---|---|---|---|---|---|
| count | 0.622 | 0.622 | 1.000 | 1.000 | 0.329 | 0.670 | 0.000 | 28 | 233 |
| duration | 0.516 | 0.516 | 1.000 | 1.000 | 0.528 | 0.535 | 0.000 | 40 | 40 |
| state | 0.723 | 0.723 | 1.000 | 1.000 | 0.104 | 0.590 | 0.000 | 4 | 18 |
| transition | 0.814 | 0.814 | 1.000 | 1.000 | 0.081 | 0.842 | 0.000 | 28 | 38 |

### `reflex` — Stage 2 - Local perception

| goal_kind | Recall (stage-local) | Recall (cumulative) | Event recall (local) | Event recall (cum.) | Precision | Economy | Drop rate | n cases | n GT events |
|---|---|---|---|---|---|---|---|---|---|
| count | 1.000 | 0.622 | 1.000 | 1.000 | 0.329 | 0.203 | 0.000 | 28 | 233 |
| duration | 1.000 | 0.516 | 1.000 | 1.000 | 0.528 | 0.258 | 0.000 | 40 | 40 |
| state | 1.000 | 0.723 | 1.000 | 1.000 | 0.104 | 0.244 | 0.000 | 4 | 18 |
| transition | 1.000 | 0.814 | 1.000 | 1.000 | 0.081 | 0.143 | 0.000 | 28 | 38 |

### `reflex` — Stage 3 - Judge dispatch

| goal_kind | Recall (stage-local) | Recall (cumulative) | Event recall (local) | Event recall (cum.) | Precision | Economy | Drop rate | n cases | n GT events |
|---|---|---|---|---|---|---|---|---|---|
| count | 0.191 | 0.119 | 0.948 | 0.948 | 0.327 | 0.948 | 0.001 | 28 | 233 |
| duration | 0.244 | 0.126 | 1.000 | 1.000 | 0.543 | 0.926 | 0.008 | 40 | 40 |
| state | 0.185 | 0.134 | 1.000 | 1.000 | 0.082 | 0.959 | 0.005 | 4 | 18 |
| transition | 0.133 | 0.108 | 1.000 | 1.000 | 0.078 | 0.965 | 0.001 | 28 | 38 |

### `reflex` — Stage 4 - Cognition + disposition

| goal_kind | Recall (stage-local) | Recall (cumulative) | Event recall (local) | Event recall (cum.) | Precision | Economy | Drop rate | n cases | n GT events |
|---|---|---|---|---|---|---|---|---|---|
| count | 0.587 | 0.070 | 0.688 | 0.652 | 0.566 | 0.339 | 0.000 | 28 | 233 |
| duration | 0.599 | 0.075 | 0.875 | 0.875 | 0.650 | 0.502 | 0.000 | 40 | 40 |
| state | 0.486 | 0.065 | 0.667 | 0.667 | 0.219 | 0.183 | 0.000 | 4 | 18 |
| transition | 0.369 | 0.040 | 0.447 | 0.447 | 0.280 | 0.103 | 0.000 | 28 | 38 |

## Per-stage breakdown by dataset

### `reflex` — Stage 0 - Ingest + normalize

| dataset | Recall (stage-local) | Recall (cumulative) | Event recall (local) | Event recall (cum.) | Precision | Economy | Drop rate | n cases | n GT events |
|---|---|---|---|---|---|---|---|---|---|
| charades | 1.000 | 1.000 | 1.000 | 1.000 | 0.851 | 0.262 | 0.000 | 20 | 20 |
| omnipro | 1.000 | 1.000 | 1.000 | 1.000 | 0.054 | 0.217 | 0.000 | 20 | 66 |
| perception_test | 1.000 | 1.000 | 1.000 | 1.000 | 0.182 | 0.200 | 0.000 | 20 | 20 |
| proactiveqa | 1.000 | 1.000 | 1.000 | 1.000 | 0.092 | 1.000 | 0.000 | 20 | 20 |
| repcount | 1.000 | 1.000 | 1.000 | 1.000 | 0.807 | 0.209 | 0.000 | 20 | 203 |

### `reflex` — Stage 1 - Admission (VID tier-0 gate)

| dataset | Recall (stage-local) | Recall (cumulative) | Event recall (local) | Event recall (cum.) | Precision | Economy | Drop rate | n cases | n GT events |
|---|---|---|---|---|---|---|---|---|---|
| charades | 0.495 | 0.495 | 1.000 | 1.000 | 0.826 | 0.502 | 0.000 | 20 | 20 |
| omnipro | 0.714 | 0.714 | 1.000 | 1.000 | 0.056 | 0.687 | 0.000 | 20 | 66 |
| perception_test | 0.635 | 0.635 | 1.000 | 1.000 | 0.199 | 0.576 | 0.000 | 20 | 20 |
| proactiveqa | 0.821 | 0.821 | 1.000 | 1.000 | 0.087 | 0.863 | 0.000 | 20 | 20 |
| repcount | 0.609 | 0.609 | 1.000 | 1.000 | 0.826 | 0.593 | 0.000 | 20 | 203 |

### `reflex` — Stage 2 - Local perception

| dataset | Recall (stage-local) | Recall (cumulative) | Event recall (local) | Event recall (cum.) | Precision | Economy | Drop rate | n cases | n GT events |
|---|---|---|---|---|---|---|---|---|---|
| charades | 1.000 | 0.495 | 1.000 | 1.000 | 0.826 | 0.277 | 0.000 | 20 | 20 |
| omnipro | 1.000 | 0.714 | 1.000 | 1.000 | 0.056 | 0.197 | 0.000 | 20 | 66 |
| perception_test | 1.000 | 0.635 | 1.000 | 1.000 | 0.199 | 0.239 | 0.000 | 20 | 20 |
| proactiveqa | 1.000 | 0.821 | 1.000 | 1.000 | 0.087 | 0.137 | 0.000 | 20 | 20 |
| repcount | 1.000 | 0.609 | 1.000 | 1.000 | 0.826 | 0.230 | 0.000 | 20 | 203 |

### `reflex` — Stage 3 - Judge dispatch

| dataset | Recall (stage-local) | Recall (cumulative) | Event recall (local) | Event recall (cum.) | Precision | Economy | Drop rate | n cases | n GT events |
|---|---|---|---|---|---|---|---|---|---|
| charades | 0.256 | 0.127 | 1.000 | 1.000 | 0.829 | 0.935 | 0.004 | 20 | 20 |
| omnipro | 0.195 | 0.139 | 0.985 | 0.985 | 0.057 | 0.970 | 0.001 | 20 | 66 |
| perception_test | 0.188 | 0.120 | 1.000 | 1.000 | 0.173 | 0.915 | 0.013 | 20 | 20 |
| proactiveqa | 0.129 | 0.106 | 1.000 | 1.000 | 0.086 | 0.963 | 0.001 | 20 | 20 |
| repcount | 0.191 | 0.116 | 0.946 | 0.946 | 0.749 | 0.917 | 0.002 | 20 | 203 |

### `reflex` — Stage 4 - Cognition + disposition

| dataset | Recall (stage-local) | Recall (cumulative) | Event recall (local) | Event recall (cum.) | Precision | Economy | Drop rate | n cases | n GT events |
|---|---|---|---|---|---|---|---|---|---|
| charades | 0.622 | 0.079 | 1.000 | 1.000 | 0.924 | 0.560 | 0.000 | 20 | 20 |
| omnipro | 0.292 | 0.041 | 0.400 | 0.394 | 0.109 | 0.154 | 0.000 | 20 | 66 |
| perception_test | 0.459 | 0.055 | 0.750 | 0.750 | 0.186 | 0.427 | 0.000 | 20 | 20 |
| proactiveqa | 0.386 | 0.041 | 0.650 | 0.650 | 0.321 | 0.103 | 0.000 | 20 | 20 |
| repcount | 0.635 | 0.074 | 0.740 | 0.700 | 0.830 | 0.574 | 0.000 | 20 | 203 |

## Reflex vs deliberate — what the latency bound costs

Only one mode was run; no comparison is available.

## End-to-end latency

Every wall figure crosses the media clock exactly once, through the replay anchor recorded at the first decoded frame: `onset_wall = anchor.monotonic + (onset_media - anchor.media)`. Legs come from the runtime's own `JudgmentTiming` (monotonic, validated non-decreasing) plus the instant the `Incident` was observed by the consumer.

### `reflex`

| Leg | n | p50 | p95 | max | mean |
|---|---|---|---|---|---|
| action onset -> Incident (THE PRODUCT GATE) | 739 | 11.569 | 84.503 | 172.719 | 22.453 |
| action onset -> frame ingested | 739 | 10.136 | 82.870 | 171.053 | 20.916 |
| action onset -> admitted by tier-0 | 739 | 10.435 | 83.336 | 171.455 | 21.143 |
| ingest -> admission | 1708 | 0.167 | 0.500 | 0.998 | 0.225 |
| admission -> judge dispatch | 1708 | 0.001 | 0.002 | 0.009 | 0.001 |
| judge dispatch -> verdict | 1708 | 1.255 | 1.475 | 1.503 | 1.224 |
| verdict -> decision emitted | 1708 | 0.082 | 0.159 | 0.180 | 0.082 |
| decision -> Incident observed | 1708 | 0.001 | 0.002 | 0.005 | 0.001 |
| ingest -> Incident | 1708 | 1.502 | 1.835 | 2.014 | 1.532 |
| admission -> Incident | 1708 | 1.334 | 1.506 | 1.669 | 1.308 |

**Product gate (action -> Incident < 2.0 s): 53/739 matched Incidents clear it (0.072).**

| goal_kind | n | action->Incident p50 | p95 | max |
|---|---|---|---|---|
| count | 189 | 2.769 | 6.527 | 14.043 |
| duration | 267 | 12.895 | 29.744 | 32.333 |
| state | 18 | 2.903 | 4.345 | 4.397 |
| transition | 265 | 35.869 | 135.053 | 172.719 |

## Quality by goal_kind

`state` has n=4 cases in the full corpus and is reported, never gated.

### `reflex`

| goal_kind | GT occurrences | Occ. recall | Occ. precision | Timely recall @2s | Incidents |
|---|---|---|---|---|---|
| count | 233 | 0.605 | 0.558 | 0.146 | 339 |
| duration | 40 | 0.900 | 0.658 | 0.350 | 406 |
| state | 18 | 0.667 | 0.231 | 0.056 | 78 |
| transition | 38 | 0.447 | 0.299 | 0.079 | 885 |
| **all** | 329 | 0.626 | 0.433 | 0.158 | 1708 |

## Coverage — every (dataset, goal_kind) cell that was actually measured

An aggregate hides which category the pipeline fails on. This table is the receipt that every cell was reached; a cell short of its selected count is a gap, not a rounding error.

| dataset / goal_kind | selected | `reflex` measured |
|---|---|---|
| charades/duration | 20 | 20 |
| omnipro/count | 8 | 8 |
| omnipro/state | 4 | 4 |
| omnipro/transition | 8 | 8 |
| perception_test/duration | 20 | 20 |
| proactiveqa/transition | 20 | 20 |
| repcount/count | 20 | 20 |

## Quality by dataset

### `reflex`

| dataset | GT occurrences | Occ. recall | Occ. precision | Timely recall @2s | Incidents |
|---|---|---|---|---|---|
| charades | 20 | 1.000 | 0.915 | 0.500 | 259 |
| omnipro | 66 | 0.379 | 0.115 | 0.076 | 287 |
| perception_test | 20 | 0.800 | 0.204 | 0.200 | 147 |
| proactiveqa | 20 | 0.650 | 0.324 | 0.100 | 805 |
| repcount | 203 | 0.650 | 0.848 | 0.153 | 210 |

## Provider accounting

| Measure | `reflex` |
|---|---|
| Logical judge calls (`judge.started`) | 8,467 |
| Physical attempts (adapter boundary) | 8,367 |
| Route attempts (receipted) | 8,463 |
| Scheduler batches (`RuntimeStats.judge_calls`) | 8,467 |
| Verdicts completed | 8,112 |
| Degraded (adapter raised) | 0 |
| Adapter-level exceptions | 0 |
| Failed route calls (no usable answer) | 2,111 |
| Input tokens | 20,944,240 |
| Output tokens | 240,103 |
| Input bytes to provider | 1,296,140,561 |
| Reflex context stamped on call | 8,367 |
| Verdicts carrying evidence prose | 0 |
| Estimated cost (USD) | 5.5962 |

Terminal outcome classes from the pool's own route receipts (this harness passes NO `terminal_outcomes`, so a `deadline` or `capacity` is recorded and the run continues — the qualification bench raises on these, which is right for a gate and wrong for a measurement):

| Outcome | `reflex` |
|---|---|
| `cancelled` | 2461 |
| `deadline` | 2111 |
| `decisive` | 5771 |
| `uncertain` | 231 |

Cost is tokens x a DECLARED price, not a measurement: $0.25/M input, $1.5/M output (ai.google.dev/gemini-api/docs/pricing). Re-derive with `--price-in/--price-out`.

## Provider latency, completed vs censored

A call the pool CANCELS at its deadline has no measured provider latency — only a lower bound (the elapsed at cancellation). The two are reported separately; pooling them would manufacture a fast-looking distribution out of calls that never finished.

| Mode | Completed calls | p50 | p95 | max | Cancelled calls | censored elapsed p50 |
|---|---|---|---|---|---|---|
| `reflex` | 5,906 | 1.171 | 1.443 | 1.496 | 2,461 | 1.484 |

## Runtime bookkeeping under a workload that produces ASKs

The scripted-oracle suite could not exercise this: its judge always answered with high confidence, so the gate resolved to fire/silent and almost never to ASK, and that suite explicitly flagged a suspected double-increment on the ASK path as NOT cleared by it. A live judge produces ASKs in bulk, so the counter is exercised here for the first time.

| Mode | judged | fired | refused | ask | cases with `refused == ask` | cases with `refused == 2 x ask` | cases where funnel tokens == adapter tokens |
|---|---|---|---|---|---|---|---|
| `reflex` | 8,112 | 1,708 | 5,090 | 2,545 | 0/100 | 100/100 | 97/100 |

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
