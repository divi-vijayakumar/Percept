# Live-provider run: per-stage metrics and end-to-end latency

Runtime under test: `percept` 1.0.0 built from `2d5097df3984c7cd08c77685e5229c0b0bcffb1a` (contract 1.10.0), interpreter `/Users/dhivyavijayakumar/percept-launch/venvs/oss-2d5097d/bin/python`.
Provider: **gemini / `gemini-3.1-flash-lite`**, transport `ordered_timestamped_images`, evidence shape `video_clip` (3 frames over 2.0 s).  Generated 2026-07-27T17:13:20Z.

**What was actually run:** 18 of 100 corpus cases, each in BOTH modes (explicit case list, n=18).  460 s of media per mode, replayed in REAL TIME.  Case concurrency 4.  Total wall 156 s.

Every case is driven through the real public path — `percept.run(...)` over a paced file source, a compiled `WatchPlan`, the shipped tier-0 gate, and a live provider behind a `JudgePool`. No stage is stubbed and no verdict is scripted.

The two arms differ in exactly two declared ways: the pool's `mode` (which stamps `_percept_latency_mode` and therefore the output budget — 320 tokens and no free-text `evidence` for reflex, 1024 tokens with prose for deliberate), and the pool deadline (1.5 s reflex / 30.0 s deliberate). The deliberate deadline is generous ON PURPOSE: holding deliberate at 1.5 s would measure how often a 1024-token generation misses a bus, not what the bound costs in quality. The counterfactual is reported instead, from the per-call route receipts.

## Per-stage metrics — `reflex`

18 of 18 cases measured; 0 errored (errors are counted, never dropped from a denominator).

Recall appears TWICE at every stage. **stage-local** is versus what that stage actually received; **cumulative** is versus the original source ground truth. A stage at 1.000 stage-local lost nothing itself however low its cumulative figure is — the pair is what localises the loss, and a single number per stage cannot.

| Stage | Recall (stage-local) | Recall (cumulative) | Event recall (local) | Event recall (cum.) | Precision | Economy | Drop rate | Units in -> out |
|---|---|---|---|---|---|---|---|---|
| Stage 0 - Ingest + normalize | 1.000 | 1.000 | 1.000 | 1.000 | 0.665 | 0.258 | 0.000 | 11,007 -> 2,836 |
| Stage 1 - Admission (VID tier-0 gate) | 0.564 | 0.564 | 1.000 | 1.000 | 0.673 | 0.547 | 0.000 | 2,836 -> 1,551 |
| Stage 2 - Local perception | 1.000 | 0.564 | 1.000 | 1.000 | 0.673 | 0.231 | 0.000 | 1,551 -> 358 |
| Stage 3 - Judge dispatch | 0.189 | 0.107 | 0.922 | 0.922 | 0.635 | 0.874 | 0.003 | 358 -> 313 |
| Stage 4 - Cognition + disposition | 0.448 | 0.048 | 0.644 | 0.594 | 0.709 | 0.399 | 0.000 | 313 -> 125 |

Two boundaries read differently here than they do in the scripted-oracle suite, and both differences are caused by the judge being REAL:

- **Stage 2** carries the judge scheduler. With `judge_dispatch="async"` the scheduler is single-flight with one latest-pending slot, so while a ~1.4 s provider call is out, every newly admitted frame but the freshest is superseded. Its economy is therefore the ratio of provider throughput to admission rate, not a perception loss — and every superseded unit carries a `RuntimeStats.judge_superseded` receipt plus a health event, so it is economy and not drop.
- **Stage 3's TIME recall is structurally bounded by that economy** and largely restates it. The column that localises a judge regression is EVENT recall: did at least one unit inside each ground-truth occurrence survive to a completed verdict?

## Per-stage breakdown by goal_kind

`state` has n=4 cases corpus-wide and cannot support a claim: it is reported, never gated.

### `reflex` — Stage 0 - Ingest + normalize

| goal_kind | Recall (stage-local) | Recall (cumulative) | Event recall (local) | Event recall (cum.) | Precision | Economy | Drop rate | n cases | n GT events |
|---|---|---|---|---|---|---|---|---|---|
| count | 1.000 | 1.000 | 1.000 | 1.000 | 0.788 | 0.208 | 0.000 | 6 | 52 |
| duration | 1.000 | 1.000 | 1.000 | 1.000 | 0.606 | 0.290 | 0.000 | 12 | 12 |

### `reflex` — Stage 1 - Admission (VID tier-0 gate)

| goal_kind | Recall (stage-local) | Recall (cumulative) | Event recall (local) | Event recall (cum.) | Precision | Economy | Drop rate | n cases | n GT events |
|---|---|---|---|---|---|---|---|---|---|
| count | 0.610 | 0.610 | 1.000 | 1.000 | 0.769 | 0.626 | 0.000 | 6 | 52 |
| duration | 0.536 | 0.536 | 1.000 | 1.000 | 0.619 | 0.511 | 0.000 | 12 | 12 |

### `reflex` — Stage 2 - Local perception

| goal_kind | Recall (stage-local) | Recall (cumulative) | Event recall (local) | Event recall (cum.) | Precision | Economy | Drop rate | n cases | n GT events |
|---|---|---|---|---|---|---|---|---|---|
| count | 1.000 | 0.610 | 1.000 | 1.000 | 0.769 | 0.206 | 0.000 | 6 | 52 |
| duration | 1.000 | 0.536 | 1.000 | 1.000 | 0.619 | 0.245 | 0.000 | 12 | 12 |

### `reflex` — Stage 3 - Judge dispatch

| goal_kind | Recall (stage-local) | Recall (cumulative) | Event recall (local) | Event recall (cum.) | Precision | Economy | Drop rate | n cases | n GT events |
|---|---|---|---|---|---|---|---|---|---|
| count | 0.160 | 0.098 | 0.904 | 0.904 | 0.733 | 0.817 | 0.000 | 6 | 52 |
| duration | 0.209 | 0.112 | 1.000 | 1.000 | 0.593 | 0.901 | 0.004 | 12 | 12 |

### `reflex` — Stage 4 - Cognition + disposition

| goal_kind | Recall (stage-local) | Recall (cumulative) | Event recall (local) | Event recall (cum.) | Precision | Economy | Drop rate | n cases | n GT events |
|---|---|---|---|---|---|---|---|---|---|
| count | 0.582 | 0.057 | 0.617 | 0.558 | 0.893 | 0.479 | 0.000 | 6 | 52 |
| duration | 0.375 | 0.042 | 0.750 | 0.750 | 0.605 | 0.365 | 0.000 | 12 | 12 |

## Per-stage breakdown by dataset

### `reflex` — Stage 0 - Ingest + normalize

| dataset | Recall (stage-local) | Recall (cumulative) | Event recall (local) | Event recall (cum.) | Precision | Economy | Drop rate | n cases | n GT events |
|---|---|---|---|---|---|---|---|---|---|
| charades | 1.000 | 1.000 | 1.000 | 1.000 | 0.947 | 0.463 | 0.000 | 6 | 6 |
| perception_test | 1.000 | 1.000 | 1.000 | 1.000 | 0.226 | 0.200 | 0.000 | 6 | 6 |
| repcount | 1.000 | 1.000 | 1.000 | 1.000 | 0.788 | 0.208 | 0.000 | 6 | 52 |

### `reflex` — Stage 1 - Admission (VID tier-0 gate)

| dataset | Recall (stage-local) | Recall (cumulative) | Event recall (local) | Event recall (cum.) | Precision | Economy | Drop rate | n cases | n GT events |
|---|---|---|---|---|---|---|---|---|---|
| charades | 0.504 | 0.504 | 1.000 | 1.000 | 0.941 | 0.482 | 0.000 | 6 | 6 |
| perception_test | 0.685 | 0.685 | 1.000 | 1.000 | 0.285 | 0.544 | 0.000 | 6 | 6 |
| repcount | 0.610 | 0.610 | 1.000 | 1.000 | 0.769 | 0.626 | 0.000 | 6 | 52 |

### `reflex` — Stage 2 - Local perception

| dataset | Recall (stage-local) | Recall (cumulative) | Event recall (local) | Event recall (cum.) | Precision | Economy | Drop rate | n cases | n GT events |
|---|---|---|---|---|---|---|---|---|---|
| charades | 1.000 | 0.504 | 1.000 | 1.000 | 0.941 | 0.251 | 0.000 | 6 | 6 |
| perception_test | 1.000 | 0.685 | 1.000 | 1.000 | 0.285 | 0.238 | 0.000 | 6 | 6 |
| repcount | 1.000 | 0.610 | 1.000 | 1.000 | 0.769 | 0.206 | 0.000 | 6 | 52 |

### `reflex` — Stage 3 - Judge dispatch

| dataset | Recall (stage-local) | Recall (cumulative) | Event recall (local) | Event recall (cum.) | Precision | Economy | Drop rate | n cases | n GT events |
|---|---|---|---|---|---|---|---|---|---|
| charades | 0.226 | 0.114 | 1.000 | 1.000 | 0.933 | 0.922 | 0.008 | 6 | 6 |
| perception_test | 0.153 | 0.105 | 1.000 | 1.000 | 0.208 | 0.878 | 0.000 | 6 | 6 |
| repcount | 0.160 | 0.098 | 0.904 | 0.904 | 0.733 | 0.817 | 0.000 | 6 | 52 |

### `reflex` — Stage 4 - Cognition + disposition

| dataset | Recall (stage-local) | Recall (cumulative) | Event recall (local) | Event recall (cum.) | Precision | Economy | Drop rate | n cases | n GT events |
|---|---|---|---|---|---|---|---|---|---|
| charades | 0.355 | 0.040 | 0.833 | 0.833 | 0.921 | 0.356 | 0.000 | 6 | 6 |
| perception_test | 0.476 | 0.050 | 0.667 | 0.667 | 0.263 | 0.376 | 0.000 | 6 | 6 |
| repcount | 0.582 | 0.057 | 0.617 | 0.558 | 0.893 | 0.479 | 0.000 | 6 | 52 |

## Reflex vs deliberate — what the latency bound costs

Only one mode was run; no comparison is available.

## End-to-end latency

Every wall figure crosses the media clock exactly once, through the replay anchor recorded at the first decoded frame: `onset_wall = anchor.monotonic + (onset_media - anchor.media)`. Legs come from the runtime's own `JudgmentTiming` (monotonic, validated non-decreasing) plus the instant the `Incident` was observed by the consumer.

### `reflex`

| Leg | n | p50 | p95 | max | mean |
|---|---|---|---|---|---|
| action onset -> Incident (THE PRODUCT GATE) | 86 | 6.169 | 27.701 | 30.072 | 9.834 |
| action onset -> frame ingested | 86 | 4.537 | 26.099 | 28.764 | 8.251 |
| action onset -> admitted by tier-0 | 86 | 4.870 | 26.368 | 28.927 | 8.510 |
| ingest -> admission | 114 | 0.199 | 0.512 | 0.654 | 0.256 |
| admission -> judge dispatch | 114 | 0.000 | 0.001 | 0.001 | 0.000 |
| judge dispatch -> verdict | 114 | 1.258 | 1.468 | 1.499 | 1.239 |
| verdict -> decision emitted | 114 | 0.080 | 0.160 | 0.166 | 0.078 |
| decision -> Incident observed | 114 | 0.000 | 0.000 | 0.000 | 0.000 |
| ingest -> Incident | 114 | 1.602 | 1.868 | 2.004 | 1.574 |
| admission -> Incident | 114 | 1.333 | 1.581 | 1.656 | 1.318 |

**Product gate (action -> Incident < 2.0 s): 11/86 matched Incidents clear it (0.128).**

| goal_kind | n | action->Incident p50 | p95 | max |
|---|---|---|---|---|
| count | 35 | 2.668 | 7.558 | 7.935 |
| duration | 51 | 12.301 | 29.148 | 30.072 |

## Quality by goal_kind

`state` has n=4 cases in the full corpus and is reported, never gated.

### `reflex`

| goal_kind | GT occurrences | Occ. recall | Occ. precision | Timely recall @2s | Incidents |
|---|---|---|---|---|---|
| count | 52 | 0.481 | 0.875 | 0.135 | 40 |
| duration | 12 | 0.750 | 0.689 | 0.333 | 74 |
| **all** | 64 | 0.531 | 0.754 | 0.172 | 114 |

## Coverage — every (dataset, goal_kind) cell that was actually measured

An aggregate hides which category the pipeline fails on. This table is the receipt that every cell was reached; a cell short of its selected count is a gap, not a rounding error.

| dataset / goal_kind | selected | `reflex` measured |
|---|---|---|
| charades/duration | 6 | 6 |
| perception_test/duration | 6 | 6 |
| repcount/count | 6 | 6 |

## Quality by dataset

### `reflex`

| dataset | GT occurrences | Occ. recall | Occ. precision | Timely recall @2s | Incidents |
|---|---|---|---|---|---|
| charades | 6 | 0.833 | 0.976 | 0.333 | 41 |
| perception_test | 6 | 0.667 | 0.333 | 0.333 | 33 |
| repcount | 52 | 0.481 | 0.875 | 0.135 | 40 |

## Provider accounting

| Measure | `reflex` |
|---|---|
| Logical judge calls (`judge.started`) | 358 |
| Physical attempts (adapter boundary) | 340 |
| Route attempts (receipted) | 358 |
| Scheduler batches (`RuntimeStats.judge_calls`) | 358 |
| Verdicts completed | 313 |
| Degraded (adapter raised) | 0 |
| Adapter-level exceptions | 0 |
| Failed route calls (no usable answer) | 95 |
| Input tokens | 714,834 |
| Output tokens | 10,115 |
| Input bytes to provider | 39,699,200 |
| Reflex context stamped on call | 340 |
| Verdicts carrying evidence prose | 0 |
| Estimated cost (USD) | 0.1939 |

Terminal outcome classes from the pool's own route receipts (this harness passes NO `terminal_outcomes`, so a `deadline` or `capacity` is recorded and the run continues — the qualification bench raises on these, which is right for a gate and wrong for a measurement):

| Outcome | `reflex` |
|---|---|
| `cancelled` | 140 |
| `deadline` | 95 |
| `decisive` | 198 |
| `uncertain` | 20 |

Cost is tokens x a DECLARED price, not a measurement: $0.25/M input, $1.5/M output (declared on the command line). Re-derive with `--price-in/--price-out`.

## Provider latency, completed vs censored

A call the pool CANCELS at its deadline has no measured provider latency — only a lower bound (the elapsed at cancellation). The two are reported separately; pooling them would manufacture a fast-looking distribution out of calls that never finished.

| Mode | Completed calls | p50 | p95 | max | Cancelled calls | censored elapsed p50 |
|---|---|---|---|---|---|---|
| `reflex` | 200 | 1.237 | 1.455 | 1.484 | 140 | 1.485 |

## Runtime bookkeeping under a workload that produces ASKs

The scripted-oracle suite could not exercise this: its judge always answered with high confidence, so the gate resolved to fire/silent and almost never to ASK, and that suite explicitly flagged a suspected double-increment on the ASK path as NOT cleared by it. A live judge produces ASKs in bulk, so the counter is exercised here for the first time.

| Mode | judged | fired | refused | ask | cases with `refused == ask` | cases with `refused == 2 x ask` | cases where funnel tokens == adapter tokens |
|---|---|---|---|---|---|---|---|
| `reflex` | 313 | 114 | 232 | 116 | 0/18 | 18/18 | 18/18 |

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
