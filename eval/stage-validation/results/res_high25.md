# Live-provider run: per-stage metrics and end-to-end latency

Runtime under test: `percept` 1.0.0 built from `2d5097df3984c7cd08c77685e5229c0b0bcffb1a` (contract 1.10.0), interpreter `/Users/dhivyavijayakumar/percept-launch/venvs/oss-2d5097d/bin/python`.
Provider: **gemini / `gemini-3.1-flash-lite`**, transport `ordered_timestamped_images`, evidence shape `video_clip` (3 frames over 2.0 s).  Generated 2026-07-27T17:08:07Z.

**What was actually run:** 25 of 100 corpus cases, each in BOTH modes (explicit case list, n=25).  1071 s of media per mode, replayed in REAL TIME.  Case concurrency 4.  Total wall 314 s.

Every case is driven through the real public path — `percept.run(...)` over a paced file source, a compiled `WatchPlan`, the shipped tier-0 gate, and a live provider behind a `JudgePool`. No stage is stubbed and no verdict is scripted.

The two arms differ in exactly two declared ways: the pool's `mode` (which stamps `_percept_latency_mode` and therefore the output budget — 320 tokens and no free-text `evidence` for reflex, 1024 tokens with prose for deliberate), and the pool deadline (1.5 s reflex / 30.0 s deliberate). The deliberate deadline is generous ON PURPOSE: holding deliberate at 1.5 s would measure how often a 1024-token generation misses a bus, not what the bound costs in quality. The counterfactual is reported instead, from the per-call route receipts.

## Per-stage metrics — `reflex`

25 of 25 cases measured; 0 errored (errors are counted, never dropped from a denominator).

Recall appears TWICE at every stage. **stage-local** is versus what that stage actually received; **cumulative** is versus the original source ground truth. A stage at 1.000 stage-local lost nothing itself however low its cumulative figure is — the pair is what localises the loss, and a single number per stage cannot.

| Stage | Recall (stage-local) | Recall (cumulative) | Event recall (local) | Event recall (cum.) | Precision | Economy | Drop rate | Units in -> out |
|---|---|---|---|---|---|---|---|---|
| Stage 0 - Ingest + normalize | 1.000 | 1.000 | 1.000 | 1.000 | 0.338 | 0.227 | 0.000 | 28,792 -> 6,529 |
| Stage 1 - Admission (VID tier-0 gate) | 0.595 | 0.595 | 1.000 | 1.000 | 0.310 | 0.644 | 0.000 | 6,529 -> 4,205 |
| Stage 2 - Local perception | 1.000 | 0.595 | 1.000 | 1.000 | 0.310 | 0.185 | 0.000 | 4,205 -> 776 |
| Stage 3 - Judge dispatch | 0.164 | 0.098 | 0.967 | 0.967 | 0.320 | 0.865 | 0.010 | 776 -> 671 |
| Stage 4 - Cognition + disposition | 0.348 | 0.034 | 0.528 | 0.511 | 0.529 | 0.212 | 0.000 | 671 -> 142 |

Two boundaries read differently here than they do in the scripted-oracle suite, and both differences are caused by the judge being REAL:

- **Stage 2** carries the judge scheduler. With `judge_dispatch="async"` the scheduler is single-flight with one latest-pending slot, so while a ~1.4 s provider call is out, every newly admitted frame but the freshest is superseded. Its economy is therefore the ratio of provider throughput to admission rate, not a perception loss — and every superseded unit carries a `RuntimeStats.judge_superseded` receipt plus a health event, so it is economy and not drop.
- **Stage 3's TIME recall is structurally bounded by that economy** and largely restates it. The column that localises a judge regression is EVENT recall: did at least one unit inside each ground-truth occurrence survive to a completed verdict?

## Per-stage breakdown by goal_kind

`state` has n=4 cases corpus-wide and cannot support a claim: it is reported, never gated.

### `reflex` — Stage 0 - Ingest + normalize

| goal_kind | Recall (stage-local) | Recall (cumulative) | Event recall (local) | Event recall (cum.) | Precision | Economy | Drop rate | n cases | n GT events |
|---|---|---|---|---|---|---|---|---|---|
| count | 1.000 | 1.000 | 1.000 | 1.000 | 0.340 | 0.203 | 0.000 | 9 | 64 |
| duration | 1.000 | 1.000 | 1.000 | 1.000 | 0.606 | 0.290 | 0.000 | 12 | 12 |
| state | 1.000 | 1.000 | 1.000 | 1.000 | 0.126 | 0.200 | 0.000 | 3 | 15 |
| transition | 1.000 | 1.000 | 1.000 | 1.000 | 0.019 | 0.250 | 0.000 | 1 | 1 |

### `reflex` — Stage 1 - Admission (VID tier-0 gate)

| goal_kind | Recall (stage-local) | Recall (cumulative) | Event recall (local) | Event recall (cum.) | Precision | Economy | Drop rate | n cases | n GT events |
|---|---|---|---|---|---|---|---|---|---|
| count | 0.639 | 0.639 | 1.000 | 1.000 | 0.286 | 0.760 | 0.000 | 9 | 64 |
| duration | 0.536 | 0.536 | 1.000 | 1.000 | 0.619 | 0.511 | 0.000 | 12 | 12 |
| state | 0.738 | 0.738 | 1.000 | 1.000 | 0.147 | 0.632 | 0.000 | 3 | 15 |
| transition | 1.000 | 1.000 | 1.000 | 1.000 | 0.030 | 0.626 | 0.000 | 1 | 1 |

### `reflex` — Stage 2 - Local perception

| goal_kind | Recall (stage-local) | Recall (cumulative) | Event recall (local) | Event recall (cum.) | Precision | Economy | Drop rate | n cases | n GT events |
|---|---|---|---|---|---|---|---|---|---|
| count | 1.000 | 0.639 | 1.000 | 1.000 | 0.286 | 0.157 | 0.000 | 9 | 64 |
| duration | 1.000 | 0.536 | 1.000 | 1.000 | 0.619 | 0.232 | 0.000 | 12 | 12 |
| state | 1.000 | 0.738 | 1.000 | 1.000 | 0.147 | 0.193 | 0.000 | 3 | 15 |
| transition | 1.000 | 1.000 | 1.000 | 1.000 | 0.030 | 0.181 | 0.000 | 1 | 1 |

### `reflex` — Stage 3 - Judge dispatch

| goal_kind | Recall (stage-local) | Recall (cumulative) | Event recall (local) | Event recall (cum.) | Precision | Economy | Drop rate | n cases | n GT events |
|---|---|---|---|---|---|---|---|---|---|
| count | 0.153 | 0.098 | 0.953 | 0.953 | 0.310 | 0.905 | 0.014 | 9 | 64 |
| duration | 0.179 | 0.096 | 1.000 | 1.000 | 0.608 | 0.800 | 0.013 | 12 | 12 |
| state | 0.137 | 0.101 | 1.000 | 1.000 | 0.114 | 0.920 | 0.000 | 3 | 15 |
| transition | 0.160 | 0.160 | 1.000 | 1.000 | 0.034 | 0.776 | 0.013 | 1 | 1 |

### `reflex` — Stage 4 - Cognition + disposition

| goal_kind | Recall (stage-local) | Recall (cumulative) | Event recall (local) | Event recall (cum.) | Precision | Economy | Drop rate | n cases | n GT events |
|---|---|---|---|---|---|---|---|---|---|
| count | 0.458 | 0.045 | 0.541 | 0.516 | 0.535 | 0.265 | 0.000 | 9 | 64 |
| duration | 0.265 | 0.025 | 0.583 | 0.583 | 0.731 | 0.228 | 0.000 | 12 | 12 |
| state | 0.385 | 0.039 | 0.467 | 0.467 | 0.241 | 0.181 | 0.000 | 3 | 15 |
| transition | 0.000 | 0.000 | 0.000 | 0.000 | n/a | 0.000 | 0.000 | 1 | 1 |

## Per-stage breakdown by dataset

### `reflex` — Stage 0 - Ingest + normalize

| dataset | Recall (stage-local) | Recall (cumulative) | Event recall (local) | Event recall (cum.) | Precision | Economy | Drop rate | n cases | n GT events |
|---|---|---|---|---|---|---|---|---|---|
| charades | 1.000 | 1.000 | 1.000 | 1.000 | 0.947 | 0.463 | 0.000 | 6 | 6 |
| omnipro | 1.000 | 1.000 | 1.000 | 1.000 | 0.092 | 0.208 | 0.000 | 7 | 28 |
| perception_test | 1.000 | 1.000 | 1.000 | 1.000 | 0.226 | 0.200 | 0.000 | 6 | 6 |
| repcount | 1.000 | 1.000 | 1.000 | 1.000 | 0.788 | 0.208 | 0.000 | 6 | 52 |

### `reflex` — Stage 1 - Admission (VID tier-0 gate)

| dataset | Recall (stage-local) | Recall (cumulative) | Event recall (local) | Event recall (cum.) | Precision | Economy | Drop rate | n cases | n GT events |
|---|---|---|---|---|---|---|---|---|---|
| charades | 0.504 | 0.504 | 1.000 | 1.000 | 0.941 | 0.482 | 0.000 | 6 | 6 |
| omnipro | 0.765 | 0.765 | 1.000 | 1.000 | 0.097 | 0.719 | 0.000 | 7 | 28 |
| perception_test | 0.685 | 0.685 | 1.000 | 1.000 | 0.285 | 0.544 | 0.000 | 6 | 6 |
| repcount | 0.610 | 0.610 | 1.000 | 1.000 | 0.769 | 0.626 | 0.000 | 6 | 52 |

### `reflex` — Stage 2 - Local perception

| dataset | Recall (stage-local) | Recall (cumulative) | Event recall (local) | Event recall (cum.) | Precision | Economy | Drop rate | n cases | n GT events |
|---|---|---|---|---|---|---|---|---|---|
| charades | 1.000 | 0.504 | 1.000 | 1.000 | 0.941 | 0.241 | 0.000 | 6 | 6 |
| omnipro | 1.000 | 0.765 | 1.000 | 1.000 | 0.097 | 0.164 | 0.000 | 7 | 28 |
| perception_test | 1.000 | 0.685 | 1.000 | 1.000 | 0.285 | 0.222 | 0.000 | 6 | 6 |
| repcount | 1.000 | 0.610 | 1.000 | 1.000 | 0.769 | 0.199 | 0.000 | 6 | 52 |

### `reflex` — Stage 3 - Judge dispatch

| dataset | Recall (stage-local) | Recall (cumulative) | Event recall (local) | Event recall (cum.) | Precision | Economy | Drop rate | n cases | n GT events |
|---|---|---|---|---|---|---|---|---|---|
| charades | 0.193 | 0.097 | 1.000 | 1.000 | 0.933 | 0.829 | 0.008 | 6 | 6 |
| omnipro | 0.143 | 0.109 | 1.000 | 1.000 | 0.093 | 0.913 | 0.005 | 7 | 28 |
| perception_test | 0.131 | 0.090 | 1.000 | 1.000 | 0.219 | 0.766 | 0.019 | 6 | 6 |
| repcount | 0.155 | 0.095 | 0.942 | 0.942 | 0.738 | 0.811 | 0.027 | 6 | 52 |

### `reflex` — Stage 4 - Cognition + disposition

| dataset | Recall (stage-local) | Recall (cumulative) | Event recall (local) | Event recall (cum.) | Precision | Economy | Drop rate | n cases | n GT events |
|---|---|---|---|---|---|---|---|---|---|
| charades | 0.312 | 0.030 | 1.000 | 1.000 | 1.000 | 0.304 | 0.000 | 6 | 6 |
| omnipro | 0.218 | 0.024 | 0.286 | 0.286 | 0.133 | 0.151 | 0.000 | 7 | 28 |
| perception_test | 0.022 | 0.002 | 0.167 | 0.167 | 0.036 | 0.134 | 0.000 | 6 | 6 |
| repcount | 0.558 | 0.053 | 0.653 | 0.615 | 0.926 | 0.444 | 0.000 | 6 | 52 |

## Reflex vs deliberate — what the latency bound costs

Only one mode was run; no comparison is available.

## End-to-end latency

Every wall figure crosses the media clock exactly once, through the replay anchor recorded at the first decoded frame: `onset_wall = anchor.monotonic + (onset_media - anchor.media)`. Legs come from the runtime's own `JudgmentTiming` (monotonic, validated non-decreasing) plus the instant the `Incident` was observed by the consumer.

### `reflex`

| Leg | n | p50 | p95 | max | mean |
|---|---|---|---|---|---|
| action onset -> Incident (THE PRODUCT GATE) | 82 | 4.371 | 27.769 | 30.848 | 8.693 |
| action onset -> frame ingested | 82 | 2.801 | 25.901 | 29.091 | 7.040 |
| action onset -> admitted by tier-0 | 82 | 3.084 | 26.205 | 29.418 | 7.274 |
| ingest -> admission | 145 | 0.167 | 0.404 | 0.532 | 0.221 |
| admission -> judge dispatch | 145 | 0.001 | 0.002 | 0.004 | 0.001 |
| judge dispatch -> verdict | 145 | 1.357 | 1.486 | 1.502 | 1.337 |
| verdict -> decision emitted | 145 | 0.077 | 0.159 | 0.175 | 0.077 |
| decision -> Incident observed | 145 | 0.000 | 0.001 | 0.001 | 0.000 |
| ingest -> Incident | 145 | 1.666 | 1.867 | 2.001 | 1.636 |
| admission -> Incident | 145 | 1.440 | 1.597 | 1.669 | 1.415 |

**Product gate (action -> Incident < 2.0 s): 7/82 matched Incidents clear it (0.085).**

| goal_kind | n | action->Incident p50 | p95 | max |
|---|---|---|---|---|
| count | 35 | 2.630 | 7.194 | 8.268 |
| duration | 35 | 13.893 | 29.498 | 30.848 |
| state | 12 | 3.735 | 5.061 | 5.134 |

## Quality by goal_kind

`state` has n=4 cases in the full corpus and is reported, never gated.

### `reflex`

| goal_kind | GT occurrences | Occ. recall | Occ. precision | Timely recall @2s | Incidents |
|---|---|---|---|---|---|
| count | 64 | 0.406 | 0.530 | 0.094 | 66 |
| duration | 12 | 0.500 | 0.745 | 0.083 | 47 |
| state | 15 | 0.667 | 0.375 | 0.000 | 32 |
| transition | 1 | 0.000 | n/a | 0.000 | 0 |
| **all** | 92 | 0.457 | 0.566 | 0.076 | 145 |

## Coverage — every (dataset, goal_kind) cell that was actually measured

An aggregate hides which category the pipeline fails on. This table is the receipt that every cell was reached; a cell short of its selected count is a gap, not a rounding error.

| dataset / goal_kind | selected | `reflex` measured |
|---|---|---|
| charades/duration | 6 | 6 |
| omnipro/count | 3 | 3 |
| omnipro/state | 3 | 3 |
| omnipro/transition | 1 | 1 |
| perception_test/duration | 6 | 6 |
| repcount/count | 6 | 6 |

## Quality by dataset

### `reflex`

| dataset | GT occurrences | Occ. recall | Occ. precision | Timely recall @2s | Incidents |
|---|---|---|---|---|---|
| charades | 6 | 1.000 | 1.000 | 0.167 | 35 |
| omnipro | 28 | 0.429 | 0.230 | 0.000 | 61 |
| perception_test | 6 | 0.000 | 0.000 | 0.000 | 12 |
| repcount | 52 | 0.462 | 0.892 | 0.115 | 37 |

## Provider accounting

| Measure | `reflex` |
|---|---|
| Logical judge calls (`judge.started`) | 776 |
| Physical attempts (adapter boundary) | 751 |
| Route attempts (receipted) | 776 |
| Scheduler batches (`RuntimeStats.judge_calls`) | 776 |
| Verdicts completed | 671 |
| Degraded (adapter raised) | 0 |
| Adapter-level exceptions | 0 |
| Failed route calls (no usable answer) | 260 |
| Input tokens | 1,385,147 |
| Output tokens | 16,338 |
| Input bytes to provider | 75,827,806 |
| Reflex context stamped on call | 751 |
| Verdicts carrying evidence prose | 0 |
| Estimated cost (USD) | 0.3708 |

Terminal outcome classes from the pool's own route receipts (this harness passes NO `terminal_outcomes`, so a `deadline` or `capacity` is recorded and the run continues — the qualification bench raises on these, which is right for a gate and wrong for a measurement):

| Outcome | `reflex` |
|---|---|
| `cancelled` | 364 |
| `deadline` | 260 |
| `decisive` | 371 |
| `uncertain` | 41 |

Cost is tokens x a DECLARED price, not a measurement: $0.25/M input, $1.5/M output (declared; matches prior arm). Re-derive with `--price-in/--price-out`.

## Provider latency, completed vs censored

A call the pool CANCELS at its deadline has no measured provider latency — only a lower bound (the elapsed at cancellation). The two are reported separately; pooling them would manufacture a fast-looking distribution out of calls that never finished.

| Mode | Completed calls | p50 | p95 | max | Cancelled calls | censored elapsed p50 |
|---|---|---|---|---|---|---|
| `reflex` | 387 | 1.320 | 1.472 | 1.496 | 364 | 1.478 |

## Runtime bookkeeping under a workload that produces ASKs

The scripted-oracle suite could not exercise this: its judge always answered with high confidence, so the gate resolved to fire/silent and almost never to ASK, and that suite explicitly flagged a suspected double-increment on the ASK path as NOT cleared by it. A live judge produces ASKs in bulk, so the counter is exercised here for the first time.

| Mode | judged | fired | refused | ask | cases with `refused == ask` | cases with `refused == 2 x ask` | cases where funnel tokens == adapter tokens |
|---|---|---|---|---|---|---|---|
| `reflex` | 671 | 145 | 602 | 301 | 0/25 | 25/25 | 25/25 |

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
