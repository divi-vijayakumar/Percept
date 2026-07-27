# Does tier-0 admission cluster by goal shape?

A falsification run for the proposed `GoalShapeProfile` layer. 100-case internal validation
corpus, real tier-0 pixel math from the 1.0.0 wheel, no provider calls.

## Pre-registered decision rule

Fixed before any numbers were looked at. **BUILD** the goal-shape layer only if BOTH hold:

1. Conditioning on `goal_kind` reduces mean regret by **>= 5 recall points** versus the
   single best global configuration, at matched economy.
2. The per-kind optima are genuinely **different configurations**.

## Verdict

- Criterion 1 (regret reduction >= 0.05): **FAIL** — best observed delta = **0.048** recall (95% CI [0.005, 0.098]), at economy 0.10.
- Criterion 2 (per-kind optima differ): **PASS** — all four kinds select different configurations.

### DO NOT BUILD the goal-shape layer

Criterion 1 fails, so the conjunction fails. It fails **narrowly** on the headline number
(0.048 against a 0.05 bar), and that margin is well inside the
bootstrap interval — so the honest reading is *not* "goal shape is worth 4.8 points and we
were 0.2 short". Three independent checks say the 4.8 points are not real:

1. **It does not survive refitting.** Leave-one-case-out, the per-kind policy's advantage
   over global is at best 0.021 recall — it is *negative* at three of
   the four economy levels. The in-sample gap is the per-kind policy fitting 4 configs
   where the global policy fits 1, not a property of goal shape.
2. **The rival explains it better.** A per-`dataset` policy beats the per-kind policy at
   every economy level (best delta 0.058 vs
   0.048), and `SourceProfile` already exists to carry it.
3. **Goal shape adds nothing beyond source class.** Splitting a dataset further by
   `goal_kind` is worth at most -0.005 recall.

Criterion 2 passes — the four kinds do select different configs — but criterion 2 was
written as a guard against a spurious criterion-1 pass, not as evidence on its own. Given
the collinearity between `goal_kind` and `dataset` in this corpus, four different picks is
what five different *sources* look like when you relabel them by goal shape.

**What to do instead**, in descending order of measured value:

- **Change the shipped default statistic.** `mean@0.10` -> `block_max@0.20` is worth
  **+0.376 recall** (0.572 -> 0.947) and takes zero new configuration surface. That
  is 8x the largest goal-shape effect and
  is the single highest-value change this sweep found.
- **Fix the 24 starving cases.** Under the shipped default, 24/100 cases admit
  *zero* frames — the gate never opens, so no downstream judge quality can recover them.
- **If per-context defaults are wanted, key them on `SourceProfile`**, which already exists.

## Method

- Gate frame `48x36`, sampled at 4.0 fps (cv2 BGR2GRAY + cv2.resize INTER_LINEAR (percept/runtime/source.py)).
- Scoring: bit-identical vectorisation of percept.harness.tier0. A self-check asserts equality with `salience_stats` /
  `ego_residual_score` for every grid.
- **Recall** = fraction of ground-truth positives with >=1 admitted sample in window. Zero-width ground-truth instants use a +/- 1.0s
  tolerance window; positive-width intervals use +/- 0.25s (one sample period).
- **Economy** = admitted_samples / total_samples.
- Sweep: stats ['mean', 'block_max', 'p95', 'ego'], block grids [[4, 3], [6, 4], [8, 6], [12, 9], [16, 12], [20, 15]], enter_debounce [1, 2, 3],
  ~56 thresholds per stat = **2391 configurations** x 100 cases.
- All policies are compared at matched economy. The global policy is constrained on the
  corpus mean; per-kind / per-dataset policies are constrained *within each cell*, which is
  a strictly tighter budget — so any gain they show is not bought with extra spend.

The asymmetry runs *against* the null this report reaches, and it did not bind: at every
economy level the per-kind policy actually spent LESS than the global one while being scored
(e.g. at budget 0.10, per-kind mean economy 0.087 vs global 0.092). The goal-shape policy was
not handicapped into losing.

## Cell sizes and statistical power

| goal_kind | n | dataset composition |
|---|---|---|
| duration | 40 | charades=20, perception_test=20 |
| transition | 28 | omnipro=8, proactiveqa=20 |
| count | 28 | omnipro=8, repcount=20 |
| state | 4 | omnipro=4 |

`state` has n=4 and **cannot support a confident per-kind claim**. Any per-kind result is
also confounded with dataset: `duration` is charades+perception_test, `transition` is mostly
proactiveqa, `count` is mostly repcount. Goal kind and source class are not independent here.

## What the shipped defaults actually do

| config | mean recall | mean economy | cases admitting ZERO frames |
|---|---|---|---|
| `mean@0.1 grid=8x6 debounce=1` | 0.572 | 0.162 | 24/100 |
| `block_max@0.2 grid=8x6 debounce=1` | 0.947 | 0.442 | 2/100 |
| `p95@0.15 grid=8x6 debounce=1` | 0.914 | 0.429 | 4/100 |

## Regret by policy, at matched economy

Regret = per-case oracle recall (best config for that case within budget) minus the policy's
recall on that case. Lower is better; the oracle is 0 by construction.

| economy | global | per-goal_kind | per-dataset | oracle | delta (kind vs global) | delta (dataset vs global) | distinct per-kind configs |
|---|---|---|---|---|---|---|---|
| 0.10 | 0.324 | 0.276 | 0.266 | 0.000 | 0.048 | 0.058 | 4/4 |
| 0.20 | 0.198 | 0.186 | 0.153 | 0.000 | 0.011 | 0.045 | 4/4 |
| 0.35 | 0.097 | 0.093 | 0.074 | 0.000 | 0.004 | 0.024 | 4/4 |
| 0.50 | 0.030 | 0.033 | 0.031 | 0.000 | -0.003 | -0.001 | 4/4 |

Mean recall of each policy (same runs):

| economy | global | per-goal_kind | per-dataset | oracle |
|---|---|---|---|---|
| 0.10 | 0.568 | 0.616 | 0.626 | 0.892 |
| 0.20 | 0.744 | 0.755 | 0.789 | 0.942 |
| 0.35 | 0.894 | 0.899 | 0.918 | 0.992 |
| 0.50 | 0.968 | 0.965 | 0.967 | 0.998 |

### The same comparison, refitted leave-one-case-out

The table above is **in-sample**: every policy picked its configs on the same 100 cases it is
scored on. A per-kind policy fits 4 configs where the global policy fits 1, so it is
mechanically favoured. Refitting each policy without the case it scores removes that bias and
is the number that predicts behaviour on a goal the sweep never saw.

| economy | global (LOO) | per-goal_kind (LOO) | per-dataset (LOO) | delta kind-vs-global (LOO, 95% CI) |
|---|---|---|---|---|
| 0.10 | 0.560 | 0.581 | 0.574 | 0.021 [-0.012, 0.061] |
| 0.20 | 0.744 | 0.687 | 0.690 | -0.057 [-0.115, -0.009] |
| 0.35 | 0.894 | 0.873 | 0.871 | -0.022 [-0.038, -0.007] |
| 0.50 | 0.951 | 0.914 | 0.910 | -0.037 [-0.068, -0.012] |

### Does goal shape add anything beyond source class?

`goal_kind` and `dataset` are close to collinear in this corpus (see the cell table), so a
per-kind gain could be a per-dataset gain wearing a different label. The nested contrast
(`dataset x goal_kind` versus `dataset` alone) is the clean test: it asks whether splitting a
source class *by goal shape* buys anything. Only `omnipro` carries more than one goal kind,
so this is the only place the question can be asked at all.

| economy | delta nested-vs-per-dataset (LOO, 95% CI) |
|---|---|
| 0.10 | -0.007 [-0.023, 0.007] |
| 0.20 | -0.028 [-0.063, -0.001] |
| 0.35 | -0.005 [-0.017, 0.003] |
| 0.50 | -0.007 [-0.018, 0.000] |

## The per-kind optima

Read the last column with care. The global config is budgeted on the **corpus** mean, so on an
individual cell it may spend more than that cell's budget — it does so on `transition`,
`count` and `state` at most levels. Where it beats the per-kind pick on a cell, it is often
buying that with spend the per-kind pick was not allowed. In aggregate the direction is the
other way (per-kind spends less overall at every level), which is why the headline in-sample
delta is a generous estimate for the goal-shape policy — and it still misses the bar.

### economy <= 0.10

Global pick: `p95@0.268767 grid=6x4 deb=1`

| goal_kind | n | selected config | recall | economy | global recall here | global economy here |
|---|---|---|---|---|---|---|
| duration | 40 | `block_max@0.352458 grid=8x6 deb=1` | 0.650 | 0.092 | 0.525 | 0.053 |
| transition | 28 | `ego@0.144681 grid=- deb=1` | 0.711 | 0.075 | 0.738 | 0.132 *(over budget)* |
| count | 28 | `p95@0.268767 grid=4x3 deb=1` | 0.437 | 0.095 | 0.415 | 0.104 *(over budget)* |
| state | 4 | `ego@0.129223 grid=- deb=1` | 0.875 | 0.070 | 0.875 | 0.128 *(over budget)* |

### economy <= 0.20

Global pick: `block_max@0.490314 grid=16x12 deb=1`

| goal_kind | n | selected config | recall | economy | global recall here | global economy here |
|---|---|---|---|---|---|---|
| duration | 40 | `p95@0.172825 grid=4x3 deb=1` | 0.825 | 0.184 | 0.750 | 0.132 |
| transition | 28 | `block_max@0.282833 grid=6x4 deb=1` | 0.810 | 0.183 | 0.845 | 0.227 *(over budget)* |
| count | 28 | `p95@0.240676 grid=8x6 deb=1` | 0.585 | 0.195 | 0.616 | 0.250 *(over budget)* |
| state | 4 | `ego@0.129223 grid=- deb=1` | 0.875 | 0.070 | 0.875 | 0.202 *(over budget)* |

### economy <= 0.35

Global pick: `block_max@0.315732 grid=12x9 deb=1`

| goal_kind | n | selected config | recall | economy | global recall here | global economy here |
|---|---|---|---|---|---|---|
| duration | 40 | `block_max@0.352458 grid=16x12 deb=1` | 0.950 | 0.303 | 0.925 | 0.260 |
| transition | 28 | `block_max@0.490314 grid=20x15 deb=1` | 0.943 | 0.343 | 0.943 | 0.400 *(over budget)* |
| count | 28 | `block_max@0.253362 grid=8x6 deb=1` | 0.766 | 0.348 | 0.804 | 0.379 *(over budget)* |
| state | 4 | `ego@0.04674 grid=- deb=1` | 1.000 | 0.321 | 0.875 | 0.301 |

### economy <= 0.50

Global pick: `block_max@0.315732 grid=16x12 deb=1`

| goal_kind | n | selected config | recall | economy | global recall here | global economy here |
|---|---|---|---|---|---|---|
| duration | 40 | `block_max@0.315732 grid=16x12 deb=1` | 1.000 | 0.368 | 1.000 | 0.368 |
| transition | 28 | `p95@0.154762 grid=8x6 deb=2` | 0.973 | 0.442 | 0.952 | 0.551 *(over budget)* |
| count | 28 | `block_max@0.203312 grid=8x6 deb=1` | 0.901 | 0.476 | 0.939 | 0.531 *(over budget)* |
| state | 4 | `ego@0.04674 grid=- deb=1` | 1.000 | 0.321 | 0.958 | 0.371 |

## Recall / economy frontier per goal_kind

### duration (n=40)

| economy budget | best mean recall | achieved economy | config |
|---|---|---|---|
| 0.02 | 0.350 | 0.016 | `block_max@0.439224 grid=8x6 deb=2` |
| 0.05 | 0.525 | 0.048 | `block_max@0.352458 grid=6x4 deb=1` |
| 0.10 | 0.650 | 0.092 | `block_max@0.352458 grid=8x6 deb=1` |
| 0.20 | 0.825 | 0.184 | `p95@0.172825 grid=4x3 deb=1` |
| 0.35 | 0.950 | 0.303 | `block_max@0.352458 grid=16x12 deb=1` |
| 0.50 | 1.000 | 0.368 | `block_max@0.315732 grid=16x12 deb=1` |
| 0.75 | 1.000 | 0.368 | `block_max@0.315732 grid=16x12 deb=1` |
| 1.00 | 1.000 | 0.368 | `block_max@0.315732 grid=16x12 deb=1` |

### transition (n=28)

| economy budget | best mean recall | achieved economy | config |
|---|---|---|---|
| 0.02 | 0.402 | 0.016 | `ego@0.203061 grid=- deb=1` |
| 0.05 | 0.568 | 0.048 | `ego@0.161988 grid=- deb=1` |
| 0.10 | 0.711 | 0.075 | `ego@0.144681 grid=- deb=1` |
| 0.20 | 0.810 | 0.183 | `block_max@0.282833 grid=6x4 deb=1` |
| 0.35 | 0.943 | 0.343 | `block_max@0.490314 grid=20x15 deb=1` |
| 0.50 | 0.973 | 0.442 | `p95@0.154762 grid=8x6 deb=2` |
| 0.75 | 0.994 | 0.626 | `block_max@0.352458 grid=20x15 deb=1` |
| 1.00 | 1.000 | 0.957 | `block_max@0.084305 grid=12x9 deb=1` |

### count (n=28)

| economy budget | best mean recall | achieved economy | config |
|---|---|---|---|
| 0.02 | 0.138 | 0.019 | `ego@0.203061 grid=- deb=1` |
| 0.05 | 0.264 | 0.045 | `ego@0.129223 grid=- deb=1` |
| 0.10 | 0.437 | 0.095 | `p95@0.268767 grid=4x3 deb=1` |
| 0.20 | 0.585 | 0.195 | `p95@0.240676 grid=8x6 deb=1` |
| 0.35 | 0.766 | 0.348 | `block_max@0.253362 grid=8x6 deb=1` |
| 0.50 | 0.901 | 0.476 | `block_max@0.203312 grid=8x6 deb=1` |
| 0.75 | 1.000 | 0.724 | `block_max@0.163149 grid=12x9 deb=1` |
| 1.00 | 1.000 | 0.724 | `block_max@0.163149 grid=12x9 deb=1` |

### state (n=4)

| economy budget | best mean recall | achieved economy | config |
|---|---|---|---|
| 0.02 | 0.333 | 0.017 | `block_max@0.611016 grid=6x4 deb=1` |
| 0.05 | 0.625 | 0.045 | `block_max@0.393456 grid=4x3 deb=1` |
| 0.10 | 0.875 | 0.070 | `ego@0.129223 grid=- deb=1` |
| 0.20 | 0.875 | 0.070 | `ego@0.129223 grid=- deb=1` |
| 0.35 | 1.000 | 0.321 | `ego@0.04674 grid=- deb=1` |
| 0.50 | 1.000 | 0.321 | `ego@0.04674 grid=- deb=1` |
| 0.75 | 1.000 | 0.321 | `ego@0.04674 grid=- deb=1` |
| 1.00 | 1.000 | 0.321 | `ego@0.04674 grid=- deb=1` |

## Rival hypothesis: source class

`SourceProfile` already exists as a per-source-class layer. If clustering is real but
source-keyed, the correct answer is to use that layer, not to add a new one.

| economy | per-dataset selected configs |
|---|---|
| 0.10 | charades: `block_max@0.393456 grid=12x9 deb=2`; omnipro: `ego@0.129223 grid=- deb=1`; perception_test: `block_max@0.352458 grid=8x6 deb=1`; proactiveqa: `ego@0.144681 grid=- deb=1`; repcount: `p95@0.240676 grid=4x3 deb=1` |
| 0.20 | charades: `block_max@0.315732 grid=16x12 deb=3`; omnipro: `ego@0.092071 grid=- deb=1`; perception_test: `block_max@0.226962 grid=6x4 deb=1`; proactiveqa: `mean@0.103085 grid=- deb=3`; repcount: `p95@0.240676 grid=12x9 deb=1` |
| 0.35 | charades: `block_max@0.315732 grid=16x12 deb=1`; omnipro: `block_max@0.490314 grid=20x15 deb=1`; perception_test: `block_max@0.203312 grid=6x4 deb=1`; proactiveqa: `mean@0.103085 grid=- deb=3`; repcount: `block_max@0.182127 grid=6x4 deb=1` |
| 0.50 | charades: `block_max@0.315732 grid=16x12 deb=1`; omnipro: `block_max@0.352458 grid=16x12 deb=1`; perception_test: `block_max@0.203312 grid=6x4 deb=1`; proactiveqa: `p95@0.138587 grid=4x3 deb=3`; repcount: `block_max@0.203312 grid=8x6 deb=1` |

### Second source-class key: camera motion

| economy | n | single config | split on is_camera_moving | delta | distinct configs |
|---|---|---|---|---|---|
| 0.10 | 20 | 0.650 | 0.650 | 0.000 | 2 |
| 0.20 | 20 | 0.900 | 0.950 | 0.050 | 2 |
| 0.35 | 20 | 1.000 | 1.000 | 0.000 | 2 |
| 0.50 | 20 | 1.000 | 1.000 | 0.000 | 2 |

**Underpowered — not a finding.** Only `perception_test` carries the flag (is_camera_moving=0 n=17, is_camera_moving=1 n=3). A
three-case cell cannot support a claim in either direction; this is here so the reader knows
the axis was checked, not because the number means anything.

## Prior findings this run corroborates or contradicts

**Corroborated — `mean` is grid-invariant.** Not merely measured but true *by construction*:
`_block_scores` tiles the active area exactly with integer denominators, so summing block
numerators over the area denominator IS the whole-frame mean. The self-check confirms
bit-identity across all six grids. Sweeping `block_grid` for `stat=mean` is a no-op.

**Corroborated — the shipped `mean@0.10` starves.** The prior 25-case run saw zero admitted
frames on 7/25 cases (28%); this 100-case run sees 24/100 (24%), at 0.572 mean recall.

**Partially contradicted — the interior `12x9` optimum.** The interior-optimum half holds: the
best grid is never an endpoint at any economy level, for either block statistic. The magnitude
does not: the largest recall spread across grids at matched economy is **0.099**,
not the ~0.31 seen on 25 cases. Nor is the winner stably `12x9` — across levels and statistics
it lands on `12x9`, `16x12`, `6x4`, `8x6`.

| economy | stat | recall spread across grids | best grid | interior? |
|---|---|---|---|---|
| 0.10 | block_max | 0.035 | 16x12 | yes |
| 0.10 | p95 | 0.047 | 6x4 | yes |
| 0.20 | block_max | 0.099 | 16x12 | yes |
| 0.20 | p95 | 0.040 | 8x6 | yes |
| 0.35 | block_max | 0.053 | 12x9 | yes |
| 0.35 | p95 | 0.037 | 6x4 | yes |
| 0.50 | block_max | 0.055 | 16x12 | yes |
| 0.50 | p95 | 0.043 | 6x4 | yes |

At 100 cases the grid is a real but second-order lever, and which grid wins depends on the
economy budget. The first-order lever is the **statistic**, not the grid:
swapping `mean` for `block_max` is worth 0.376 recall — 4x the
largest grid effect.

