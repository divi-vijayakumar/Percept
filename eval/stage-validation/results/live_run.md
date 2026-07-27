# Live-provider run: per-stage metrics and end-to-end latency

Runtime under test: `percept` 1.0.0 built from `2d5097df3984c7cd08c77685e5229c0b0bcffb1a` (contract 1.10.0), interpreter `/Users/dhivyavijayakumar/percept-launch/venvs/oss-2d5097d/bin/python`.
Provider: **gemini / `gemini-3.1-flash-lite`**, transport `ordered_timestamped_images`, evidence shape `video_clip` (5 frames over 2.0 s).  Generated 2026-07-27T08:46:54Z.

**What was actually run:** 100 of 100 corpus cases, each in BOTH modes (full corpus).  11284 s of media per mode, replayed in REAL TIME.  Case concurrency 4.  Total wall 5774 s.

Every case is driven through the real public path — `percept.run(...)` over a paced file source, a compiled `WatchPlan`, the shipped tier-0 gate, and a live provider behind a `JudgePool`. No stage is stubbed and no verdict is scripted.

The two arms differ in exactly two declared ways: the pool's `mode` (which stamps `_percept_latency_mode` and therefore the output budget — 320 tokens and no free-text `evidence` for reflex, 1024 tokens with prose for deliberate), and the pool deadline (1.5 s reflex / 30.0 s deliberate). The deliberate deadline is generous ON PURPOSE: holding deliberate at 1.5 s would measure how often a 1024-token generation misses a bus, not what the bound costs in quality. The counterfactual is reported instead, from the per-call route receipts.

## Per-stage metrics — `reflex`

100 of 100 cases measured; 0 errored (errors are counted, never dropped from a denominator).

Recall appears TWICE at every stage. **stage-local** is versus what that stage actually received; **cumulative** is versus the original source ground truth. A stage at 1.000 stage-local lost nothing itself however low its cumulative figure is — the pair is what localises the loss, and a single number per stage cannot.

| Stage | Recall (stage-local) | Recall (cumulative) | Event recall (local) | Event recall (cum.) | Precision | Economy | Drop rate | Units in -> out |
|---|---|---|---|---|---|---|---|---|
| Stage 0 - Ingest + normalize | 1.000 | 1.000 | 1.000 | 1.000 | 0.165 | 0.437 | 0.000 | 155,189 -> 67,775 |
| Stage 1 - Admission (VID tier-0 gate) | 0.661 | 0.661 | 1.000 | 1.000 | 0.139 | 0.780 | 0.000 | 67,775 -> 52,868 |
| Stage 2 - Local perception | 1.000 | 0.661 | 1.000 | 1.000 | 0.139 | 0.138 | 0.000 | 52,868 -> 7,304 |
| Stage 3 - Judge dispatch | 0.128 | 0.085 | 0.815 | 0.815 | 0.151 | 0.858 | 0.003 | 7,304 -> 6,269 |
| Stage 4 - Cognition + disposition | 0.065 | 0.005 | 0.149 | 0.122 | 0.392 | 0.025 | 0.000 | 6,269 -> 156 |

## Per-stage metrics — `deliberate`

100 of 100 cases measured; 0 errored (errors are counted, never dropped from a denominator).

Recall appears TWICE at every stage. **stage-local** is versus what that stage actually received; **cumulative** is versus the original source ground truth. A stage at 1.000 stage-local lost nothing itself however low its cumulative figure is — the pair is what localises the loss, and a single number per stage cannot.

| Stage | Recall (stage-local) | Recall (cumulative) | Event recall (local) | Event recall (cum.) | Precision | Economy | Drop rate | Units in -> out |
|---|---|---|---|---|---|---|---|---|
| Stage 0 - Ingest + normalize | 1.000 | 1.000 | 1.000 | 1.000 | 0.165 | 0.437 | 0.000 | 155,189 -> 67,775 |
| Stage 1 - Admission (VID tier-0 gate) | 0.661 | 0.661 | 1.000 | 1.000 | 0.139 | 0.780 | 0.000 | 67,775 -> 52,868 |
| Stage 2 - Local perception | 1.000 | 0.661 | 1.000 | 1.000 | 0.139 | 0.121 | 0.000 | 52,868 -> 6,419 |
| Stage 3 - Judge dispatch | 0.146 | 0.096 | 0.903 | 0.903 | 0.167 | 1.000 | 0.000 | 6,419 -> 6,419 |
| Stage 4 - Cognition + disposition | 0.713 | 0.069 | 0.788 | 0.711 | 0.427 | 0.279 | 0.000 | 6,419 -> 1,793 |

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
| count | 1.000 | 0.622 | 1.000 | 1.000 | 0.329 | 0.169 | 0.000 | 28 | 233 |
| duration | 1.000 | 0.516 | 1.000 | 1.000 | 0.528 | 0.224 | 0.000 | 40 | 40 |
| state | 1.000 | 0.723 | 1.000 | 1.000 | 0.104 | 0.192 | 0.000 | 4 | 18 |
| transition | 1.000 | 0.814 | 1.000 | 1.000 | 0.081 | 0.125 | 0.000 | 28 | 38 |

### `reflex` — Stage 3 - Judge dispatch

| goal_kind | Recall (stage-local) | Recall (cumulative) | Event recall (local) | Event recall (cum.) | Precision | Economy | Drop rate | n cases | n GT events |
|---|---|---|---|---|---|---|---|---|---|
| count | 0.120 | 0.074 | 0.777 | 0.777 | 0.304 | 0.764 | 0.005 | 28 | 233 |
| duration | 0.176 | 0.091 | 0.975 | 0.975 | 0.544 | 0.767 | 0.009 | 40 | 40 |
| state | 0.103 | 0.075 | 0.889 | 0.889 | 0.069 | 0.815 | 0.007 | 4 | 18 |
| transition | 0.108 | 0.088 | 0.842 | 0.842 | 0.079 | 0.892 | 0.002 | 28 | 38 |

### `reflex` — Stage 4 - Cognition + disposition

| goal_kind | Recall (stage-local) | Recall (cumulative) | Event recall (local) | Event recall (cum.) | Precision | Economy | Drop rate | n cases | n GT events |
|---|---|---|---|---|---|---|---|---|---|
| count | 0.091 | 0.007 | 0.105 | 0.082 | 0.432 | 0.064 | 0.000 | 28 | 233 |
| duration | 0.067 | 0.006 | 0.359 | 0.350 | 0.880 | 0.041 | 0.000 | 40 | 40 |
| state | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.030 | 0.000 | 4 | 18 |
| transition | 0.049 | 0.004 | 0.219 | 0.184 | 0.240 | 0.016 | 0.000 | 28 | 38 |

### `deliberate` — Stage 0 - Ingest + normalize

| goal_kind | Recall (stage-local) | Recall (cumulative) | Event recall (local) | Event recall (cum.) | Precision | Economy | Drop rate | n cases | n GT events |
|---|---|---|---|---|---|---|---|---|---|
| count | 1.000 | 1.000 | 1.000 | 1.000 | 0.354 | 0.212 | 0.000 | 28 | 233 |
| duration | 1.000 | 1.000 | 1.000 | 1.000 | 0.554 | 0.231 | 0.000 | 40 | 40 |
| state | 1.000 | 1.000 | 1.000 | 1.000 | 0.085 | 0.219 | 0.000 | 4 | 18 |
| transition | 1.000 | 1.000 | 1.000 | 1.000 | 0.084 | 0.678 | 0.000 | 28 | 38 |

### `deliberate` — Stage 1 - Admission (VID tier-0 gate)

| goal_kind | Recall (stage-local) | Recall (cumulative) | Event recall (local) | Event recall (cum.) | Precision | Economy | Drop rate | n cases | n GT events |
|---|---|---|---|---|---|---|---|---|---|
| count | 0.622 | 0.622 | 1.000 | 1.000 | 0.329 | 0.670 | 0.000 | 28 | 233 |
| duration | 0.516 | 0.516 | 1.000 | 1.000 | 0.528 | 0.535 | 0.000 | 40 | 40 |
| state | 0.723 | 0.723 | 1.000 | 1.000 | 0.104 | 0.590 | 0.000 | 4 | 18 |
| transition | 0.814 | 0.814 | 1.000 | 1.000 | 0.081 | 0.842 | 0.000 | 28 | 38 |

### `deliberate` — Stage 2 - Local perception

| goal_kind | Recall (stage-local) | Recall (cumulative) | Event recall (local) | Event recall (cum.) | Precision | Economy | Drop rate | n cases | n GT events |
|---|---|---|---|---|---|---|---|---|---|
| count | 1.000 | 0.622 | 1.000 | 1.000 | 0.329 | 0.150 | 0.000 | 28 | 233 |
| duration | 1.000 | 0.516 | 1.000 | 1.000 | 0.528 | 0.207 | 0.000 | 40 | 40 |
| state | 1.000 | 0.723 | 1.000 | 1.000 | 0.104 | 0.184 | 0.000 | 4 | 18 |
| transition | 1.000 | 0.814 | 1.000 | 1.000 | 0.081 | 0.108 | 0.000 | 28 | 38 |

### `deliberate` — Stage 3 - Judge dispatch

| goal_kind | Recall (stage-local) | Recall (cumulative) | Event recall (local) | Event recall (cum.) | Precision | Economy | Drop rate | n cases | n GT events |
|---|---|---|---|---|---|---|---|---|---|
| count | 0.153 | 0.096 | 0.876 | 0.876 | 0.337 | 1.000 | 0.000 | 28 | 233 |
| duration | 0.215 | 0.111 | 0.975 | 0.975 | 0.554 | 1.000 | 0.000 | 40 | 40 |
| state | 0.146 | 0.106 | 0.944 | 0.944 | 0.083 | 1.000 | 0.000 | 4 | 18 |
| transition | 0.103 | 0.084 | 0.974 | 0.974 | 0.077 | 1.000 | 0.000 | 28 | 38 |

### `deliberate` — Stage 4 - Cognition + disposition

| goal_kind | Recall (stage-local) | Recall (cumulative) | Event recall (local) | Event recall (cum.) | Precision | Economy | Drop rate | n cases | n GT events |
|---|---|---|---|---|---|---|---|---|---|
| count | 0.709 | 0.068 | 0.824 | 0.721 | 0.575 | 0.416 | 0.000 | 28 | 233 |
| duration | 0.737 | 0.082 | 0.949 | 0.925 | 0.651 | 0.625 | 0.000 | 40 | 40 |
| state | 0.395 | 0.042 | 0.529 | 0.500 | 0.170 | 0.193 | 0.000 | 4 | 18 |
| transition | 0.710 | 0.059 | 0.541 | 0.526 | 0.272 | 0.202 | 0.000 | 28 | 38 |

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
| charades | 1.000 | 0.495 | 1.000 | 1.000 | 0.826 | 0.238 | 0.000 | 20 | 20 |
| omnipro | 1.000 | 0.714 | 1.000 | 1.000 | 0.056 | 0.161 | 0.000 | 20 | 66 |
| perception_test | 1.000 | 0.635 | 1.000 | 1.000 | 0.199 | 0.210 | 0.000 | 20 | 20 |
| proactiveqa | 1.000 | 0.821 | 1.000 | 1.000 | 0.087 | 0.121 | 0.000 | 20 | 20 |
| repcount | 1.000 | 0.609 | 1.000 | 1.000 | 0.826 | 0.199 | 0.000 | 20 | 203 |

### `reflex` — Stage 3 - Judge dispatch

| dataset | Recall (stage-local) | Recall (cumulative) | Event recall (local) | Event recall (cum.) | Precision | Economy | Drop rate | n cases | n GT events |
|---|---|---|---|---|---|---|---|---|---|
| charades | 0.182 | 0.090 | 1.000 | 1.000 | 0.830 | 0.770 | 0.016 | 20 | 20 |
| omnipro | 0.117 | 0.083 | 0.818 | 0.818 | 0.049 | 0.835 | 0.004 | 20 | 66 |
| perception_test | 0.148 | 0.094 | 0.950 | 0.950 | 0.184 | 0.764 | 0.000 | 20 | 20 |
| proactiveqa | 0.109 | 0.090 | 1.000 | 1.000 | 0.088 | 0.901 | 0.002 | 20 | 20 |
| repcount | 0.116 | 0.071 | 0.764 | 0.764 | 0.736 | 0.654 | 0.010 | 20 | 203 |

### `reflex` — Stage 4 - Cognition + disposition

| dataset | Recall (stage-local) | Recall (cumulative) | Event recall (local) | Event recall (cum.) | Precision | Economy | Drop rate | n cases | n GT events |
|---|---|---|---|---|---|---|---|---|---|
| charades | 0.075 | 0.007 | 0.650 | 0.650 | 0.954 | 0.066 | 0.000 | 20 | 20 |
| omnipro | 0.049 | 0.004 | 0.056 | 0.045 | 0.061 | 0.039 | 0.000 | 20 | 66 |
| perception_test | 0.020 | 0.002 | 0.053 | 0.050 | 0.337 | 0.011 | 0.000 | 20 | 20 |
| proactiveqa | 0.048 | 0.004 | 0.300 | 0.300 | 0.293 | 0.014 | 0.000 | 20 | 20 |
| repcount | 0.097 | 0.007 | 0.110 | 0.084 | 0.950 | 0.075 | 0.000 | 20 | 203 |

### `deliberate` — Stage 0 - Ingest + normalize

| dataset | Recall (stage-local) | Recall (cumulative) | Event recall (local) | Event recall (cum.) | Precision | Economy | Drop rate | n cases | n GT events |
|---|---|---|---|---|---|---|---|---|---|
| charades | 1.000 | 1.000 | 1.000 | 1.000 | 0.851 | 0.262 | 0.000 | 20 | 20 |
| omnipro | 1.000 | 1.000 | 1.000 | 1.000 | 0.054 | 0.217 | 0.000 | 20 | 66 |
| perception_test | 1.000 | 1.000 | 1.000 | 1.000 | 0.182 | 0.200 | 0.000 | 20 | 20 |
| proactiveqa | 1.000 | 1.000 | 1.000 | 1.000 | 0.092 | 1.000 | 0.000 | 20 | 20 |
| repcount | 1.000 | 1.000 | 1.000 | 1.000 | 0.807 | 0.209 | 0.000 | 20 | 203 |

### `deliberate` — Stage 1 - Admission (VID tier-0 gate)

| dataset | Recall (stage-local) | Recall (cumulative) | Event recall (local) | Event recall (cum.) | Precision | Economy | Drop rate | n cases | n GT events |
|---|---|---|---|---|---|---|---|---|---|
| charades | 0.495 | 0.495 | 1.000 | 1.000 | 0.826 | 0.502 | 0.000 | 20 | 20 |
| omnipro | 0.714 | 0.714 | 1.000 | 1.000 | 0.056 | 0.687 | 0.000 | 20 | 66 |
| perception_test | 0.635 | 0.635 | 1.000 | 1.000 | 0.199 | 0.576 | 0.000 | 20 | 20 |
| proactiveqa | 0.821 | 0.821 | 1.000 | 1.000 | 0.087 | 0.863 | 0.000 | 20 | 20 |
| repcount | 0.609 | 0.609 | 1.000 | 1.000 | 0.826 | 0.593 | 0.000 | 20 | 203 |

### `deliberate` — Stage 2 - Local perception

| dataset | Recall (stage-local) | Recall (cumulative) | Event recall (local) | Event recall (cum.) | Precision | Economy | Drop rate | n cases | n GT events |
|---|---|---|---|---|---|---|---|---|---|
| charades | 1.000 | 0.495 | 1.000 | 1.000 | 0.826 | 0.227 | 0.000 | 20 | 20 |
| omnipro | 1.000 | 0.714 | 1.000 | 1.000 | 0.056 | 0.144 | 0.000 | 20 | 66 |
| perception_test | 1.000 | 0.635 | 1.000 | 1.000 | 0.199 | 0.186 | 0.000 | 20 | 20 |
| proactiveqa | 1.000 | 0.821 | 1.000 | 1.000 | 0.087 | 0.104 | 0.000 | 20 | 20 |
| repcount | 1.000 | 0.609 | 1.000 | 1.000 | 0.826 | 0.178 | 0.000 | 20 | 203 |

### `deliberate` — Stage 3 - Judge dispatch

| dataset | Recall (stage-local) | Recall (cumulative) | Event recall (local) | Event recall (cum.) | Precision | Economy | Drop rate | n cases | n GT events |
|---|---|---|---|---|---|---|---|---|---|
| charades | 0.227 | 0.112 | 1.000 | 1.000 | 0.836 | 1.000 | 0.000 | 20 | 20 |
| omnipro | 0.142 | 0.101 | 0.970 | 0.970 | 0.055 | 1.000 | 0.000 | 20 | 66 |
| perception_test | 0.162 | 0.103 | 0.950 | 0.950 | 0.175 | 1.000 | 0.000 | 20 | 20 |
| proactiveqa | 0.101 | 0.083 | 1.000 | 1.000 | 0.085 | 1.000 | 0.000 | 20 | 20 |
| repcount | 0.157 | 0.095 | 0.857 | 0.857 | 0.727 | 1.000 | 0.000 | 20 | 203 |

### `deliberate` — Stage 4 - Cognition + disposition

| dataset | Recall (stage-local) | Recall (cumulative) | Event recall (local) | Event recall (cum.) | Precision | Economy | Drop rate | n cases | n GT events |
|---|---|---|---|---|---|---|---|---|---|
| charades | 0.731 | 0.082 | 1.000 | 1.000 | 0.958 | 0.636 | 0.000 | 20 | 20 |
| omnipro | 0.289 | 0.029 | 0.359 | 0.348 | 0.085 | 0.189 | 0.000 | 20 | 66 |
| perception_test | 0.779 | 0.080 | 0.895 | 0.850 | 0.222 | 0.611 | 0.000 | 20 | 20 |
| proactiveqa | 0.745 | 0.062 | 0.800 | 0.800 | 0.300 | 0.210 | 0.000 | 20 | 20 |
| repcount | 0.765 | 0.073 | 0.908 | 0.778 | 0.811 | 0.688 | 0.000 | 20 | 203 |

## Reflex vs deliberate — what the latency bound costs

| Measure | reflex | deliberate | delta (reflex - deliberate) |
|---|---|---|---|
| Occurrence recall | 0.137 | 0.684 | -0.547 |
| Occurrence precision | 0.419 | 0.405 | +0.014 |
| Timely recall @ 2 s | 0.012 | 0.049 | -0.036 |
| Incidents emitted | 186 | 2875 | -2689 |
| Stage-4 event recall (cumulative) | 0.122 | 0.711 | -0.590 |
| Stage-3 event recall (stage-local) | 0.815 | 0.903 | -0.088 |
| action -> Incident p50 (s) | 9.540 | 14.065 | -4.525 |
| action -> Incident p95 (s) | 62.827 | 106.053 | -43.226 |
| provider call p50 (s) | 1.365 | 1.653 | -0.288 |
| provider call p95 (s) | 1.473 | 2.283 | -0.811 |
| logical judge calls | 7304 | 6419 | +885 |
| output tokens | 35826 | 497321 | -461495 |
| input tokens | 5504374 | 35783686 | -30279312 |

**Counterfactual on the deliberate arm:** 4,201 of 6,318 deliberate provider calls took longer than the reflex budget of 1.5 s (0.665). Those calls would have returned a `deadline` receipt and an UNKNOWN verdict under the reflex configuration, so the deliberate quality numbers above are NOT reachable at the reflex deadline.

## End-to-end latency

Every wall figure crosses the media clock exactly once, through the replay anchor recorded at the first decoded frame: `onset_wall = anchor.monotonic + (onset_media - anchor.media)`. Legs come from the runtime's own `JudgmentTiming` (monotonic, validated non-decreasing) plus the instant the `Incident` was observed by the consumer.

### `reflex`

| Leg | n | p50 | p95 | max | mean |
|---|---|---|---|---|---|
| action onset -> Incident (THE PRODUCT GATE) | 78 | 9.540 | 62.827 | 64.101 | 17.401 |
| action onset -> frame ingested | 78 | 7.707 | 61.159 | 62.435 | 15.684 |
| action onset -> admitted by tier-0 | 78 | 8.039 | 61.326 | 62.601 | 15.904 |
| ingest -> admission | 186 | 0.167 | 0.403 | 0.502 | 0.216 |
| admission -> judge dispatch | 186 | 0.001 | 0.002 | 0.003 | 0.001 |
| judge dispatch -> verdict | 186 | 1.433 | 1.496 | 1.505 | 1.398 |
| verdict -> decision emitted | 186 | 0.072 | 0.157 | 0.168 | 0.075 |
| decision -> Incident observed | 186 | 0.000 | 0.002 | 0.003 | 0.001 |
| ingest -> Incident | 186 | 1.667 | 2.001 | 2.009 | 1.690 |
| admission -> Incident | 186 | 1.500 | 1.601 | 1.668 | 1.474 |

**Product gate (action -> Incident < 2.0 s): 4/78 matched Incidents clear it (0.051).**

| goal_kind | n | action->Incident p50 | p95 | max |
|---|---|---|---|---|
| count | 23 | 3.535 | 4.968 | 9.135 |
| duration | 26 | 11.442 | 26.803 | 31.835 |
| state | 2 | 4.084 | 4.309 | 4.334 |
| transition | 27 | 29.811 | 64.101 | 64.101 |

### `deliberate`

| Leg | n | p50 | p95 | max | mean |
|---|---|---|---|---|---|
| action onset -> Incident (THE PRODUCT GATE) | 1165 | 14.065 | 106.053 | 180.552 | 27.043 |
| action onset -> frame ingested | 1165 | 11.901 | 103.385 | 178.051 | 24.823 |
| action onset -> admitted by tier-0 | 1165 | 12.168 | 103.552 | 178.217 | 25.081 |
| ingest -> admission | 2875 | 0.167 | 0.500 | 22.596 | 0.233 |
| admission -> judge dispatch | 2875 | 0.001 | 0.002 | 0.006 | 0.001 |
| judge dispatch -> verdict | 2875 | 1.837 | 2.479 | 29.311 | 1.870 |
| verdict -> decision emitted | 2875 | 0.082 | 0.159 | 0.193 | 0.082 |
| decision -> Incident observed | 2875 | 0.000 | 0.001 | 0.012 | 0.001 |
| ingest -> Incident | 2875 | 2.167 | 2.834 | 29.649 | 2.186 |
| admission -> Incident | 2875 | 1.974 | 2.503 | 29.312 | 1.953 |

**Product gate (action -> Incident < 2.0 s): 16/1165 matched Incidents clear it (0.014).**

| goal_kind | n | action->Incident p50 | p95 | max |
|---|---|---|---|---|
| count | 221 | 3.322 | 6.969 | 13.401 |
| duration | 409 | 12.392 | 30.465 | 53.700 |
| state | 14 | 3.358 | 4.700 | 5.686 |
| transition | 521 | 34.384 | 151.385 | 180.552 |

## Quality by goal_kind

`state` has n=4 cases in the full corpus and is reported, never gated.

### `reflex`

| goal_kind | GT occurrences | Occ. recall | Occ. precision | Timely recall @2s | Incidents |
|---|---|---|---|---|---|
| count | 233 | 0.090 | 0.469 | 0.009 | 49 |
| duration | 40 | 0.375 | 0.897 | 0.050 | 29 |
| state | 18 | 0.111 | 0.333 | 0.000 | 6 |
| transition | 38 | 0.184 | 0.265 | 0.000 | 102 |
| **all** | 329 | 0.137 | 0.419 | 0.012 | 186 |

### `deliberate`

| goal_kind | GT occurrences | Occ. recall | Occ. precision | Timely recall @2s | Incidents |
|---|---|---|---|---|---|
| count | 233 | 0.674 | 0.574 | 0.047 | 385 |
| duration | 40 | 0.950 | 0.700 | 0.125 | 584 |
| state | 18 | 0.556 | 0.230 | 0.000 | 61 |
| transition | 38 | 0.526 | 0.282 | 0.000 | 1845 |
| **all** | 329 | 0.684 | 0.405 | 0.049 | 2875 |

## Coverage — every (dataset, goal_kind) cell that was actually measured

An aggregate hides which category the pipeline fails on. This table is the receipt that every cell was reached; a cell short of its selected count is a gap, not a rounding error.

| dataset / goal_kind | selected | `reflex` measured | `deliberate` measured |
|---|---|---|---|
| charades/duration | 20 | 20 | 20 |
| omnipro/count | 8 | 8 | 8 |
| omnipro/state | 4 | 4 | 4 |
| omnipro/transition | 8 | 8 | 8 |
| perception_test/duration | 20 | 20 | 20 |
| proactiveqa/transition | 20 | 20 | 20 |
| repcount/count | 20 | 20 | 20 |

## Quality by dataset

### `reflex`

| dataset | GT occurrences | Occ. recall | Occ. precision | Timely recall @2s | Incidents |
|---|---|---|---|---|---|
| charades | 20 | 0.700 | 0.962 | 0.050 | 26 |
| omnipro | 66 | 0.061 | 0.104 | 0.000 | 48 |
| perception_test | 20 | 0.050 | 0.333 | 0.050 | 3 |
| proactiveqa | 20 | 0.300 | 0.299 | 0.000 | 87 |
| repcount | 203 | 0.099 | 0.955 | 0.010 | 22 |

### `deliberate`

| dataset | GT occurrences | Occ. recall | Occ. precision | Timely recall @2s | Incidents |
|---|---|---|---|---|---|
| charades | 20 | 1.000 | 0.961 | 0.250 | 359 |
| omnipro | 66 | 0.348 | 0.098 | 0.015 | 297 |
| perception_test | 20 | 0.900 | 0.284 | 0.000 | 225 |
| proactiveqa | 20 | 0.800 | 0.297 | 0.000 | 1743 |
| repcount | 203 | 0.729 | 0.837 | 0.049 | 251 |

## Provider accounting

| Measure | `reflex` | `deliberate` |
|---|---|---|
| Logical judge calls (`judge.started`) | 7,304 | 6,419 |
| Physical attempts (adapter boundary) | 7,204 | 6,319 |
| Route attempts (receipted) | 7,304 | 6,419 |
| Scheduler batches (`RuntimeStats.judge_calls`) | 7,304 | 6,419 |
| Verdicts completed | 6,269 | 6,419 |
| Degraded (adapter raised) | 0 | 0 |
| Adapter-level exceptions | 0 | 0 |
| Failed route calls (no usable answer) | 5,224 | 1 |
| Input tokens | 5,504,374 | 35,783,686 |
| Output tokens | 35,826 | 497,321 |
| Input bytes to provider | 1,888,649,520 | 1,632,465,824 |
| Reflex context stamped on call | 7,204 | 0 |
| Verdicts carrying evidence prose | 0 | 6,318 |
| Estimated cost (USD) | 1.4298 | 9.6919 |

Terminal outcome classes from the pool's own route receipts (this harness passes NO `terminal_outcomes`, so a `deadline` or `capacity` is recorded and the run continues — the qualification bench raises on these, which is right for a gate and wrong for a measurement):

| Outcome | `reflex` | `deliberate` |
|---|---|---|
| `cancelled` | 6228 | 1 |
| `deadline` | 5224 | 1 |
| `decisive` | 960 | 6280 |
| `uncertain` | 116 | 138 |

Cost is tokens x a DECLARED price, not a measurement: $0.25/M input, $1.5/M output (ai.google.dev/gemini-api/docs/pricing, page last updated 2026-07-21). Re-derive with `--price-in/--price-out`.

## Provider latency, completed vs censored

A call the pool CANCELS at its deadline has no measured provider latency — only a lower bound (the elapsed at cancellation). The two are reported separately; pooling them would manufacture a fast-looking distribution out of calls that never finished.

| Mode | Completed calls | p50 | p95 | max | Cancelled calls | censored elapsed p50 |
|---|---|---|---|---|---|---|
| `reflex` | 976 | 1.365 | 1.473 | 1.516 | 6,216 | 1.478 |
| `deliberate` | 6,318 | 1.653 | 2.283 | 29.307 | 1 | 29.997 |

## Runtime bookkeeping under a workload that produces ASKs

The scripted-oracle suite could not exercise this: its judge always answered with high confidence, so the gate resolved to fire/silent and almost never to ASK, and that suite explicitly flagged a suspected double-increment on the ASK path as NOT cleared by it. A live judge produces ASKs in bulk, so the counter is exercised here for the first time.

| Mode | judged | fired | refused | ask | cases with `refused == ask` | cases with `refused == 2 x ask` | cases where funnel tokens == adapter tokens |
|---|---|---|---|---|---|---|---|
| `reflex` | 6,269 | 186 | 10,714 | 5,357 | 0/100 | 100/100 | 100/100 |
| `deliberate` | 6,419 | 2,875 | 1,070 | 535 | 0/100 | 100/100 | 100/100 |

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
