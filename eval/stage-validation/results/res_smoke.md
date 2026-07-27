# Live-provider run: per-stage metrics and end-to-end latency

Runtime under test: `percept` 1.0.0 built from `2d5097df3984c7cd08c77685e5229c0b0bcffb1a` (contract 1.10.0), interpreter `/Users/dhivyavijayakumar/percept-launch/venvs/oss-2d5097d/bin/python`.
Provider: **gemini / `gemini-3.1-flash-lite`**, transport `ordered_timestamped_images`, evidence shape `video_clip` (3 frames over 2.0 s).  Generated 2026-07-27T16:27:42Z.

**What was actually run:** 2 of 100 corpus cases, each in BOTH modes (deterministic stratified sample over all (dataset, goal_kind) cells, n=2).  149 s of media per mode, replayed in REAL TIME.  Case concurrency 2.  Total wall 121 s.

Every case is driven through the real public path — `percept.run(...)` over a paced file source, a compiled `WatchPlan`, the shipped tier-0 gate, and a live provider behind a `JudgePool`. No stage is stubbed and no verdict is scripted.

The two arms differ in exactly two declared ways: the pool's `mode` (which stamps `_percept_latency_mode` and therefore the output budget — 320 tokens and no free-text `evidence` for reflex, 1024 tokens with prose for deliberate), and the pool deadline (1.5 s reflex / 30.0 s deliberate). The deliberate deadline is generous ON PURPOSE: holding deliberate at 1.5 s would measure how often a 1024-token generation misses a bus, not what the bound costs in quality. The counterfactual is reported instead, from the per-call route receipts.

## Per-stage metrics — `reflex`

2 of 2 cases measured; 0 errored (errors are counted, never dropped from a denominator).

Recall appears TWICE at every stage. **stage-local** is versus what that stage actually received; **cumulative** is versus the original source ground truth. A stage at 1.000 stage-local lost nothing itself however low its cumulative figure is — the pair is what localises the loss, and a single number per stage cannot.

| Stage | Recall (stage-local) | Recall (cumulative) | Event recall (local) | Event recall (cum.) | Precision | Economy | Drop rate | Units in -> out |
|---|---|---|---|---|---|---|---|---|
| Stage 0 - Ingest + normalize | 1.000 | 1.000 | 1.000 | 1.000 | 0.210 | 0.216 | 0.000 | 4,081 -> 882 |
| Stage 1 - Admission (VID tier-0 gate) | 0.856 | 0.856 | 1.000 | 1.000 | 0.235 | 0.762 | 0.000 | 882 -> 672 |
| Stage 2 - Local perception | 1.000 | 0.856 | 1.000 | 1.000 | 0.235 | 0.183 | 0.000 | 672 -> 123 |
| Stage 3 - Judge dispatch | 0.174 | 0.149 | 1.000 | 1.000 | 0.225 | 0.992 | 0.008 | 123 -> 122 |
| Stage 4 - Cognition + disposition | 0.077 | 0.012 | 0.500 | 0.500 | 0.419 | 0.041 | 0.000 | 122 -> 5 |

Two boundaries read differently here than they do in the scripted-oracle suite, and both differences are caused by the judge being REAL:

- **Stage 2** carries the judge scheduler. With `judge_dispatch="async"` the scheduler is single-flight with one latest-pending slot, so while a ~1.4 s provider call is out, every newly admitted frame but the freshest is superseded. Its economy is therefore the ratio of provider throughput to admission rate, not a perception loss — and every superseded unit carries a `RuntimeStats.judge_superseded` receipt plus a health event, so it is economy and not drop.
- **Stage 3's TIME recall is structurally bounded by that economy** and largely restates it. The column that localises a judge regression is EVENT recall: did at least one unit inside each ground-truth occurrence survive to a completed verdict?

## Per-stage breakdown by goal_kind

`state` has n=4 cases corpus-wide and cannot support a claim: it is reported, never gated.

### `reflex` — Stage 0 - Ingest + normalize

| goal_kind | Recall (stage-local) | Recall (cumulative) | Event recall (local) | Event recall (cum.) | Precision | Economy | Drop rate | n cases | n GT events |
|---|---|---|---|---|---|---|---|---|---|
| count | 1.000 | 1.000 | 1.000 | 1.000 | 0.017 | 0.200 | 0.000 | 1 | 1 |
| duration | 1.000 | 1.000 | 1.000 | 1.000 | 0.998 | 0.333 | 0.000 | 1 | 1 |

### `reflex` — Stage 1 - Admission (VID tier-0 gate)

| goal_kind | Recall (stage-local) | Recall (cumulative) | Event recall (local) | Event recall (cum.) | Precision | Economy | Drop rate | n cases | n GT events |
|---|---|---|---|---|---|---|---|---|---|
| count | 1.000 | 1.000 | 1.000 | 1.000 | 0.022 | 0.743 | 0.000 | 1 | 1 |
| duration | 0.846 | 0.846 | 1.000 | 1.000 | 0.997 | 0.847 | 0.000 | 1 | 1 |

### `reflex` — Stage 2 - Local perception

| goal_kind | Recall (stage-local) | Recall (cumulative) | Event recall (local) | Event recall (cum.) | Precision | Economy | Drop rate | n cases | n GT events |
|---|---|---|---|---|---|---|---|---|---|
| count | 1.000 | 1.000 | 1.000 | 1.000 | 0.022 | 0.184 | 0.000 | 1 | 1 |
| duration | 1.000 | 0.846 | 1.000 | 1.000 | 0.997 | 0.181 | 0.000 | 1 | 1 |

### `reflex` — Stage 3 - Judge dispatch

| goal_kind | Recall (stage-local) | Recall (cumulative) | Event recall (local) | Event recall (cum.) | Precision | Economy | Drop rate | n cases | n GT events |
|---|---|---|---|---|---|---|---|---|---|
| count | 0.167 | 0.167 | 1.000 | 1.000 | 0.020 | 1.000 | 0.000 | 1 | 1 |
| duration | 0.174 | 0.148 | 1.000 | 1.000 | 1.000 | 0.960 | 0.040 | 1 | 1 |

### `reflex` — Stage 4 - Cognition + disposition

| goal_kind | Recall (stage-local) | Recall (cumulative) | Event recall (local) | Event recall (cum.) | Precision | Economy | Drop rate | n cases | n GT events |
|---|---|---|---|---|---|---|---|---|---|
| count | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.031 | 0.000 | 1 | 1 |
| duration | 0.083 | 0.012 | 1.000 | 1.000 | 1.000 | 0.083 | 0.000 | 1 | 1 |

## Per-stage breakdown by dataset

### `reflex` — Stage 0 - Ingest + normalize

| dataset | Recall (stage-local) | Recall (cumulative) | Event recall (local) | Event recall (cum.) | Precision | Economy | Drop rate | n cases | n GT events |
|---|---|---|---|---|---|---|---|---|---|
| charades | 1.000 | 1.000 | 1.000 | 1.000 | 0.998 | 0.333 | 0.000 | 1 | 1 |
| omnipro | 1.000 | 1.000 | 1.000 | 1.000 | 0.017 | 0.200 | 0.000 | 1 | 1 |

### `reflex` — Stage 1 - Admission (VID tier-0 gate)

| dataset | Recall (stage-local) | Recall (cumulative) | Event recall (local) | Event recall (cum.) | Precision | Economy | Drop rate | n cases | n GT events |
|---|---|---|---|---|---|---|---|---|---|
| charades | 0.846 | 0.846 | 1.000 | 1.000 | 0.997 | 0.847 | 0.000 | 1 | 1 |
| omnipro | 1.000 | 1.000 | 1.000 | 1.000 | 0.022 | 0.743 | 0.000 | 1 | 1 |

### `reflex` — Stage 2 - Local perception

| dataset | Recall (stage-local) | Recall (cumulative) | Event recall (local) | Event recall (cum.) | Precision | Economy | Drop rate | n cases | n GT events |
|---|---|---|---|---|---|---|---|---|---|
| charades | 1.000 | 0.846 | 1.000 | 1.000 | 0.997 | 0.181 | 0.000 | 1 | 1 |
| omnipro | 1.000 | 1.000 | 1.000 | 1.000 | 0.022 | 0.184 | 0.000 | 1 | 1 |

### `reflex` — Stage 3 - Judge dispatch

| dataset | Recall (stage-local) | Recall (cumulative) | Event recall (local) | Event recall (cum.) | Precision | Economy | Drop rate | n cases | n GT events |
|---|---|---|---|---|---|---|---|---|---|
| charades | 0.174 | 0.148 | 1.000 | 1.000 | 1.000 | 0.960 | 0.040 | 1 | 1 |
| omnipro | 0.167 | 0.167 | 1.000 | 1.000 | 0.020 | 1.000 | 0.000 | 1 | 1 |

### `reflex` — Stage 4 - Cognition + disposition

| dataset | Recall (stage-local) | Recall (cumulative) | Event recall (local) | Event recall (cum.) | Precision | Economy | Drop rate | n cases | n GT events |
|---|---|---|---|---|---|---|---|---|---|
| charades | 0.083 | 0.012 | 1.000 | 1.000 | 1.000 | 0.083 | 0.000 | 1 | 1 |
| omnipro | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.031 | 0.000 | 1 | 1 |

## Reflex vs deliberate — what the latency bound costs

Only one mode was run; no comparison is available.

## End-to-end latency

Every wall figure crosses the media clock exactly once, through the replay anchor recorded at the first decoded frame: `onset_wall = anchor.monotonic + (onset_media - anchor.media)`. Legs come from the runtime's own `JudgmentTiming` (monotonic, validated non-decreasing) plus the instant the `Incident` was observed by the consumer.

### `reflex`

| Leg | n | p50 | p95 | max | mean |
|---|---|---|---|---|---|
| action onset -> Incident (THE PRODUCT GATE) | 1 | 23.271 | 23.271 | 23.271 | 23.271 |
| action onset -> frame ingested | 1 | 21.649 | 21.649 | 21.649 | 21.649 |
| action onset -> admitted by tier-0 | 1 | 21.828 | 21.828 | 21.828 | 21.828 |
| ingest -> admission | 4 | 0.167 | 0.178 | 0.179 | 0.170 |
| admission -> judge dispatch | 4 | 0.001 | 0.002 | 0.002 | 0.001 |
| judge dispatch -> verdict | 4 | 1.051 | 1.346 | 1.397 | 1.131 |
| verdict -> decision emitted | 4 | 0.116 | 0.138 | 0.141 | 0.105 |
| decision -> Incident observed | 4 | 0.001 | 0.001 | 0.001 | 0.001 |
| ingest -> Incident | 4 | 1.336 | 1.580 | 1.623 | 1.407 |
| admission -> Incident | 4 | 1.169 | 1.402 | 1.443 | 1.237 |

**Product gate (action -> Incident < 2.0 s): 0/1 matched Incidents clear it (0.000).**

| goal_kind | n | action->Incident p50 | p95 | max |
|---|---|---|---|---|
| count | 0 | n/a | n/a | n/a |
| duration | 1 | 23.271 | 23.271 | 23.271 |

## Quality by goal_kind

`state` has n=4 cases in the full corpus and is reported, never gated.

### `reflex`

| goal_kind | GT occurrences | Occ. recall | Occ. precision | Timely recall @2s | Incidents |
|---|---|---|---|---|---|
| count | 1 | 0.000 | 0.000 | 0.000 | 3 |
| duration | 1 | 1.000 | 1.000 | 0.000 | 1 |
| **all** | 2 | 0.500 | 0.250 | 0.000 | 4 |

## Coverage — every (dataset, goal_kind) cell that was actually measured

An aggregate hides which category the pipeline fails on. This table is the receipt that every cell was reached; a cell short of its selected count is a gap, not a rounding error.

| dataset / goal_kind | selected | `reflex` measured |
|---|---|---|
| charades/duration | 1 | 1 |
| omnipro/count | 1 | 1 |

## Quality by dataset

### `reflex`

| dataset | GT occurrences | Occ. recall | Occ. precision | Timely recall @2s | Incidents |
|---|---|---|---|---|---|
| charades | 1 | 1.000 | 1.000 | 0.000 | 1 |
| omnipro | 1 | 0.000 | 0.000 | 0.000 | 3 |

## Provider accounting

| Measure | `reflex` |
|---|---|
| Logical judge calls (`judge.started`) | 123 |
| Physical attempts (adapter boundary) | 121 |
| Route attempts (receipted) | 123 |
| Scheduler batches (`RuntimeStats.judge_calls`) | 123 |
| Verdicts completed | 122 |
| Degraded (adapter raised) | 0 |
| Adapter-level exceptions | 0 |
| Failed route calls (no usable answer) | 8 |
| Input tokens | 211,877 |
| Output tokens | 3,607 |
| Input bytes to provider | 5,816,075 |
| Reflex context stamped on call | 121 |
| Verdicts carrying evidence prose | 0 |
| Estimated cost (USD) | 0.0584 |

Terminal outcome classes from the pool's own route receipts (this harness passes NO `terminal_outcomes`, so a `deadline` or `capacity` is recorded and the run continues — the qualification bench raises on these, which is right for a gate and wrong for a measurement):

| Outcome | `reflex` |
|---|---|
| `cancelled` | 9 |
| `deadline` | 8 |
| `decisive` | 108 |
| `uncertain` | 6 |

Cost is tokens x a DECLARED price, not a measurement: $0.25/M input, $1.5/M output (declared on the command line). Re-derive with `--price-in/--price-out`.

## Provider latency, completed vs censored

A call the pool CANCELS at its deadline has no measured provider latency — only a lower bound (the elapsed at cancellation). The two are reported separately; pooling them would manufacture a fast-looking distribution out of calls that never finished.

| Mode | Completed calls | p50 | p95 | max | Cancelled calls | censored elapsed p50 |
|---|---|---|---|---|---|---|
| `reflex` | 112 | 1.135 | 1.367 | 1.478 | 9 | 1.497 |

## Runtime bookkeeping under a workload that produces ASKs

The scripted-oracle suite could not exercise this: its judge always answered with high confidence, so the gate resolved to fire/silent and almost never to ASK, and that suite explicitly flagged a suspected double-increment on the ASK path as NOT cleared by it. A live judge produces ASKs in bulk, so the counter is exercised here for the first time.

| Mode | judged | fired | refused | ask | cases with `refused == ask` | cases with `refused == 2 x ask` | cases where funnel tokens == adapter tokens |
|---|---|---|---|---|---|---|---|
| `reflex` | 122 | 4 | 30 | 15 | 0/2 | 2/2 | 2/2 |

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
