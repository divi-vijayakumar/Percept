# PIN REQUIRED — proposed bars and sweep matrix

Status: **proposed, awaiting founder ratification.** Nothing here is ratified. Per the repo's
pre-registration discipline (`percept/slo.py`, `DerivationPreRegistration`) every bar below must be
fixed by decision *before* the first SV run, not chosen after seeing results.

Candidate under test: `codex/launch-bounded-judges` @ `bba572a` (`percept` 1.0.0)
Companion artifact: [`sweep_matrix.json`](sweep_matrix.json) (machine-readable)
Covers: SV-1.1, SV-1.6, SV-2.1, SV-2.2, SV-2.3, SV-2.4, SV-4.2, SV-4.9

---

## 1. The decision this document makes

A pinned scalar silently asserts one configuration. That assertion is often false.

The test applied to every candidate bar was: **does a runtime parameter materially move this
outcome?** Where the answer is yes, the artifact is a **swept frontier** and the bar is a policy over
that frontier. Where the answer is no — and where that was *proved*, not assumed — a single
defensible value is correct, and sweeping it is waste that multiplies the run matrix for nothing.

Both answers were reached by measurement wherever measurement was cheap. Section 8 states plainly
which proposals are measured and which are judgement calls.

---

## 2. What was reused rather than reinvented

| Reused | Where | Used for |
|---|---|---|
| `FOVEA_CALL_REDUCTION_FLOOR = 0.50` | `percept/slo.py:243` | SV-1.3 economy bar, **verbatim** |
| `fovea_call_reduction(report) = report.reduction` | `percept/slo.py:246`, `percept/harness/tier0.py` `LoopReport.reduction` | the economy metric definition |
| `REFLEX_P95_THRESHOLD_S = 0.0100` | `percept/slo.py:91` | the budget SV-4.9's latency ceiling is derived *from* |
| `SensePolicy.max_fps = 15.0` | `percept/sampling.py:35` | the feed rate SV-4.9's throughput floor is derived *from* |
| `ArchetypePolicy.min_call_reduction` / `min_candidate_recall` / `min_event_recall` and the Pareto machinery | `percept/eval/viability_frontier.py` | **the home for every swept bar** — see §7 |
| `clopper_pearson_lower` / `required_positives` | `percept/eval/exact_ci.py` | the ceiling on what any bar can honestly claim — see §6 |
| OBO metric definition (`abs(pred - gt) <= 1`) | `docs/reference/benchmark-methodology.md:537` | SV-2.1's metric |
| External RepCount-A OBO reference `35.4` | `docs/reference/publishable-benchmark.md` | SV-2.1's bar value |
| Tier-0 calibration `block_max@0.20 -> 0.85`, `p95@0.15 -> 0.79`, `mean@0.10 -> 0.06` (PT action, N=80) | `percept/config.py:188-193`, `docs/reference/edge-policy.md` | prior evidence the new measurement is checked against |
| Archetype vocabulary `state / transition / brief_action / persistence / absence / repeated_episode` | `docs/reference/operating-frontier.md` | flagged — see §9.3 |

**No parallel machinery was invented.** The swept bars are `ArchetypePolicy` rows; the swept
configurations are `operating_points` with their `configuration_sha256`; the frontier is
`_mark_pareto`. All of it already exists.

---

## 3. Evidence base

Everything below was produced offline with **zero provider or network calls**.

| | |
|---|---|
| Interpreter | `/private/tmp/percept-launch/oss-final-venv-bba572a/bin/python` (the exact launch wheel) |
| Scorer | `percept.harness.tier0.salience_stats` / `ego_residual_score` from the wheel — **not** reimplemented |
| Frame pipeline | `cv2` grayscale + resize to `GATE_W, GATE_H = 48, 36` (`percept/runtime/source.py:26`), decimated to `SourceConfig.fps = 4.0` (`percept/runtime/source.py:43`) — the runtime's own front end, reproduced exactly |
| Corpus | `eval/stage-validation/corpus.json`, 100 cases |
| Subset | **25 cases**, 5 per dataset, `random.Random(20260726)` (the corpus `SELECTION_SEED`); the 3 `perception_test` camera-moving cases were force-included so both camera strata appear |
| Scale | 9,107 scored samples x 7 block grids x 3 statistics + ego residual (k=4) |
| Audio | `ffmpeg` -> f32 mono 16 kHz, framed at `AudioConfig.chunk_s = 0.25` (`percept/runtime/audio.py:56`), scored with `percept.runtime._vad.EnergyVad` from the wheel |
| Reducer | `SerializedDecisionReducer` + `Gate` + `EvidenceSequencer` from the wheel |

Ground-truth handling, stated because two datasets needed it:
`repcount` `positives` is empty and `cycle_bounds_frames` were converted with each file's own
measured fps; `omnipro` `positives` are unusable (see §9.1) so `provenance.ground_truth[*]
.trigger_time_sec` was used with a **declared +/-1.0 s instant tolerance** — itself a pin the founder
must ratify (§5.9).

---

## 4. The measurement that decides the sweep question

The founder's hypothesis was that `block_max` at grid size 20 behaves materially differently from
other grid sizes. **Confirmed, and sharpened in three ways.**

The real parameter is **`block_grid: tuple[int, int]`** in
`percept/harness/tier0.py::motion_score` / `_block_scores` / `salience_stats`. It is a call-site
argument with a `(8, 6)` default; it is *not* a `Settings` field and *not* on the wire, so a
deployment cannot currently change it — see §9.2.

### 4a. `mean` is grid-invariant — proved, so do not sweep it

Identical interval recall and identical call reduction to 4 decimal places across all 7 grids on all
5 datasets. `_block_scores` tiles the active area exactly, so summing the integer block numerators
over the integer area denominator *is* the whole-area mean. This is the code's documented invariant,
now confirmed empirically. Sweeping the grid for `mean` would multiply the matrix by 7 for nothing.

### 4b. At the shipped threshold, the grid does not move recall — it destroys economy

`block_max` @ the shipped `motion_gate_block_max = 0.20`. Cell = `interval_recall / call_reduction`:

| dataset | 4x3 | 6x4 | **8x6** | 12x9 | 16x12 | **20x15** | 24x18 |
|---|---|---|---|---|---|---|---|
| charades | 0.40/0.92 | 0.60/0.89 | 1.00/0.80 | 1.00/0.67 | 1.00/0.50 | 1.00/**0.34** | 1.00/0.23 |
| proactiveqa (ego) | 1.00/0.63 | 1.00/0.49 | 1.00/0.34 | 1.00/0.17 | 1.00/0.08 | 1.00/**0.02** | 1.00/0.01 |
| perception_test | 0.60/0.73 | 1.00/0.50 | 1.00/0.29 | 1.00/0.11 | 1.00/0.06 | 1.00/**0.04** | 1.00/0.02 |
| omnipro | 0.72/0.67 | 0.78/0.60 | 0.83/0.47 | 0.89/0.32 | 0.94/0.25 | 0.94/**0.20** | 1.00/0.16 |
| repcount | 0.70/0.78 | 0.86/0.68 | 0.86/0.61 | 0.91/0.52 | 1.00/0.39 | 1.00/**0.28** | 1.00/0.20 |

At **20x15 the shipped threshold admits 98% of frames on ego video** — no gating at all — and misses
`FOVEA_CALL_REDUCTION_FLOOR = 0.50` on 4 of 5 datasets. The mechanism is visible in the raw
distributions: `block_max` p50 on perception_test climbs monotonically **0.159 (4x3) -> 0.244 (8x6)
-> 0.433 (20x15) -> 0.466 (24x18)**. A fixed 0.20 sits above p75 at 4x3 and below p50 from 12x9 on.
**The threshold's percentile meaning slides with the grid, so grid and threshold cannot be pinned
independently.**

### 4c. At *matched* economy the grid genuinely moves recall — by up to 31 points

Bisecting the threshold to hold call reduction fixed, then reading interval recall (pooled, 25 cases):

| stat | at reduction | 4x3 | 6x4 | 8x6 | **12x9** | 16x12 | 20x15 | 24x18 |
|---|---|---|---|---|---|---|---|---|
| `block_max` | 0.50 | 0.76 | 0.84 | 0.87 | 0.86 | 0.88 | 0.87 | **0.89** |
| `block_max` | 0.70 | 0.67 | 0.71 | 0.76 | **0.79** | 0.76 | 0.76 | 0.78 |
| `block_max` | 0.85 | 0.39 | 0.45 | 0.58 | **0.70** | 0.67 | 0.57 | 0.51 |
| `p95` | 0.85 | 0.39 | 0.41 | 0.39 | 0.39 | 0.41 | 0.38 | 0.42 |

`block_max` swings **0.39 -> 0.70** at a tight economy target, with a clear interior optimum near
**12x9**. `p95` is flat (<= 4 points). So: **sweep `block_grid` for `block_max`; pin it at (8,6) for
`p95` and `mean`.**

### 4d. Two further material knobs found

**`salience_stat` is the largest single lever.** Pooled interval recall at the shipped per-stat
defaults: `mean@0.10 = 0.41`, `p95@0.15 = 0.80`, `block_max@0.20 = 0.88`. More seriously,
**`mean@0.10` admits zero frames on 7 of 25 cases** (2/5 charades, 2/5 perception_test, 3/5
repcount). The shipped default `salience_stat = "mean"` is structurally blind on ~28% of this corpus
— a stat-choice failure, not a threshold miss. Direction matches the in-tree PT calibration.

**`ego_compensate` is not free.** On `perception_test` camera-moving cases (n=3) at threshold 0.10,
`mean` gives recall 0.33 / reduction 0.76 while ego gives recall **0.00** / reduction 0.94. On
`proactiveqa` (egocentric, n=5): `mean` 1.00/0.52 vs ego 0.80/0.63. It buys economy by *spending*
recall; it does not dominate. **n=3 is far too small to ratify anything** — this establishes that it
must be swept and stratified by `provenance.is_camera_moving`, not what its value should be.

**`enter_debounce` costs recall on short positives.** At `block_max@0.20`, 8x6, going 1 -> 2 -> 3:
charades 1.00 / 0.60 / 0.60, repcount 0.86 / 0.74 / 0.72, omnipro 0.83 / 0.72 / 0.67, while
proactiveqa holds 1.00 / 1.00 / 1.00. Sparse, short positives lose 12-40 recall points to buy ~8
economy points; long positives lose nothing. Archetype-dependent, therefore swept.

---

## 5. Per-bar proposals

### 5.1 SV-1.1 — Admission recall (cumulative) — **SWEPT**

Not one number. A **frontier per (archetype x salience configuration)**, expressed as a
`viability_frontier` `ArchetypePolicy` with `min_candidate_recall` and `min_call_reduction`, over
`operating_points` drawn from:

```
salience_stat x block_grid x per-stat threshold x ego_compensate x enter_debounce
```

with `block_grid` swept **only** for `block_max` (proved immaterial for `mean`, near-immaterial for
`p95` — §4a, §4c).

**Proposed policy per archetype** (recall is interval-level: fraction of ground-truth positive
intervals with >= 1 admitted sample; the harness must report the stage-local/cumulative pair as the
spec's two-population rule requires):

| goal_kind | intervals in full corpus | `min_candidate_recall` | `min_call_reduction` | basis |
|---|---:|---|---|---|
| `duration` | 40 | **0.90** | 0.50 | measured 1.00 at `block_max@0.20`/8x6 on the subset; 0.90 is the ceiling the denominator can support (§6) |
| `transition` | 38 | **0.90** | **0.25** | measured 0.92 recall but only **0.36** reduction — long ego/YouTube sources are not still scenes and cannot meet the 0.50 floor |
| `count` | 233 | **0.85** | 0.50 | measured 0.88; the denominator supports up to 0.987, so 0.85 is a real (not denominator-limited) bar |
| `state` | 18 | **0.80** | 0.50 | measured 0.67 on n=1 case — see the warning below |

**Rationale for the values, and why they are not tuned to green.** Two are set by what the corpus can
*support*, not by what was observed: `duration` and `transition` have ~40 intervals each, whose
maximum ratifiable bar at a perfect run is 0.928 and 0.924 (§6), so 0.90 is the honest ceiling
regardless of the measurement. `count` at 0.85 sits *below* the observed 0.88 deliberately —
the observation is a 25-case subset and the bar must survive the other 75 cases. `state` at 0.80 sits
*above* the observed 0.67 deliberately, because that observation comes from a single case and is not
evidence of anything; 0.80 is the archetype's weakest-supportable value.

> **The `state` archetype cannot carry a gated bar.** 4 cases / 18 intervals; at case level the
> maximum 95% one-sided bound at a perfect run is **0.473**. Recommendation: **report `state`, do not
> gate it**, and record it as `NO EVIDENCE` in the operating-frontier sense until the corpus grows.

The `min_call_reduction` split is the honest reading of `slo.py`'s own caveat: 0.50 is a floor for a
*representative mostly-still deployment scene*, "not fit to any single measured track". Measured
per-track at `block_max@0.20`/8x6 it is missed by 3/5 proactiveqa, 4/5 perception_test, 3/5 omnipro
and 3/5 repcount clips. Gating every archetype at 0.50 would gate the still-scene claim on
non-still-scene footage.

### 5.2 SV-1.3 — Economy — **SINGLE, reused verbatim**

`fovea_call_reduction(report) >= 0.50` — `percept/slo.py:243`. Not restated, not re-derived, not
adjusted. Per-track and per-archetype reduction are **reported**; only the still-scene aggregate is
**gated**, exactly as the constant's own docstring scopes it.

### 5.3 SV-1.6 — VAD admission — **SWEPT**, and the recall half is currently **unpinnable**

Measured on real corpus audio at the shipped settings (`utterance_threshold = 0.01` amplitude ->
`1e-4` energy via `_amplitude_threshold_to_energy`, `hangover_frames = 8` = 2.0 s,
`min_speech_frames = 3` = 0.75 s):

| dataset | speech-classified chunk fraction @ 0.01 | utterances endpointed |
|---|---|---|
| omnipro (5 clips) | **0.96 – 1.00** | **0 on 4 of 5** |
| perception_test (5) | 0.00 – 0.33 | 0 – 1 |
| charades (5) | 0.03 – 1.00 | 0 – 2 |

**The shipped default delivers zero admission economy on continuous-audio content** and the
endpointer never fires there, because 8 consecutive silent 0.25 s chunks never occur. Across
`0.003 -> 0.05` the admitted fraction spans essentially the full 0..1 range on the *same clip*.
`utterance_threshold` is an absolute amplitude cutoff with nothing normalising it to a noise floor,
so its behaviour is a property of the recording gain, not of the speech. **No single value is
defensible.** Sweep `utterance_threshold` x `hangover_frames` x `min_speech_frames`.

> **Blocking honesty item.** The 100-case corpus contains **no VAD ground truth**. Only 1 of 20
> omnipro cases carries `audio_dependency = "required"`, and even that one has no annotated speech
> intervals; `proactiveqa` EGO videos have **no audio stream at all**. SV-1.6's *recall* and
> *precision* bars therefore cannot be measured on this corpus at any threshold. Either an external
> annotated VAD set is imported, or SV-1.6 is scoped to **economy + endpoint-behaviour only** and its
> recall bar is deferred. Proposing a recall number today would be a fabrication.

### 5.4 SV-2.1 — Counting / rhythm accuracy — **SINGLE bar over a SWEPT configuration**

- **Metric**: OBO, already defined at `docs/reference/benchmark-methodology.md:537` as
  `abs(predicted_count - gt_count) <= 1`. Report normalized and absolute MAE alongside.
- **Bar**: **OBO >= 0.354** at the best cell of the frontier — the tree's own recorded external
  reference (`docs/reference/publishable-benchmark.md`, Thinking Machines Lab, RepCount-A
  off-by-one `35.4`). Reusing the external number avoids inventing a threshold.
- **Swept because the count is configuration-dependent**: `observation_rate_fps` x `smooth_s` x
  `canonical_select`. The runtime's own docstrings record the interaction —
  `percept/rhythm.py:280-288`: smoothing takes real-pose fps12 `|err|` from **4.0 to 2.5** but
  *regresses* the cached fps=6 bench (primed OBO **0.50 -> 0.38**); `rhythm.py:291-296`:
  `canonical_select` takes real-pose cross-fps consistency from **0/4 to 2/4**. A counting bar that
  does not name (fps, smooth_s, canonical_select) is undefined.

> **Two shape corrections to the spec's suggestion.**
> 1. The spec proposes "+/-1 rep over >=10-rep sequences". `+/-1 rep` *is* OBO, but as a pass bar it
>    must be a **rate over sequences**, not a per-sequence guarantee — no published counter achieves
>    +/-1 on every sequence, and the tree's own external reference is 35.4%.
> 2. The `>=10-rep` qualifier leaves only **8 of 20** repcount cases (measured distribution:
>    4,4,5,5,5,5,7,7,8,9,9,11,12,12,14,14,17,17,19,19). At n=8 the maximum supportable bound at a
>    perfect run is **0.688**, so the qualifier makes the bar unratifiable. **Score OBO over all 20;
>    report the >=10-rep subset separately.**

### 5.5 SV-2.2 — Rhythm channel selection — **SINGLE bar over a SWEPT configuration**

- **Bar**: correct-channel rate **>= 0.80**, at the swept `(select_margin, select_dwell)` point
  reported as the default.
- **Rationale, anchored to the shipped defaults rather than to a result**: `select_dwell = 2` with
  `select_margin = 0.25` means the runtime is *designed* to hold a wrong incumbent for at least two
  reselects. A 1.00 bar would contradict the shipped policy. 0.80 permits exactly that designed
  transient while still failing a selector that is coin-flipping.
- **Swept**: `select_margin` [0.10, 0.25, 0.50] x `select_dwell` [1, 2, 4] x `select_band_s`. The
  first two compose multiplicatively — sweeping one alone reports a slice, not the frontier.

### 5.6 SV-2.3 — Re-id on unique items — **asymmetric: zero-tolerance gate + reported frontier**

- **Gate**: `false_merge_count == 0` on the unique-item fixtures, at every operating point reported
  as a recommended default.
- **Reported, never gated**: `fragment_rate` and `switch_rate` across the sweep.
- **Rationale**: the spec states the asymmetry itself — "a false merge is worse than a fragment". A
  false merge fuses two real identities and corrupts every downstream count and episode; a fragment
  only costs recall. Zero-tolerance on the harmful side plus a reported curve on the benign side is
  the shape that matches the asymmetry, and it requires no invented number.
- **Swept**: `appearance_threshold` [0.70, 0.80, 0.90, 0.95] (`percept/config.py`
  `IdentitySettings.appearance_threshold`, default 0.8) x `iou_match` [0.3, 0.5, 0.7]
  (`EntitySettings.iou_match`, default 0.5). These *are* the merge-vs-fragment trade; `switch`,
  `fragment` and `merge` are three points on one curve parameterised by them.

### 5.7 SV-2.4 — Entity extraction fidelity — **SINGLE, and currently unevaluable**

- **Bar**: per-attribute exact-match accuracy **>= 0.90**, with unknown/absent scored as a **miss**,
  never as a pass.
- **Single, not swept**: there is no knob in `percept/entity.py` or `percept/extraction.py` that
  trades attribute accuracy against cost, so a frontier would have exactly one point.
- **Blocking**: **no annotated attribute fixture exists.** The corpus carries `goal_text`,
  `positives` and `provenance` — no per-entity attribute ground truth. This bar cannot be evaluated
  until the fixture is built, and **0.90 must be re-ratified once its difficulty is known.** It is a
  judgement call with no measurement behind it.

### 5.8 SV-4.2 — Entity tracks — **SWEPT**, same shape as SV-2.3

Track lifecycle across occlusion/re-entry is governed by the same continuity threshold, so
switch/fragment/merge rates are three points on one curve parameterised by `iou_match`
[0.3, 0.5, 0.7] and bounded by `max_entities` [16, 64, 256] — an evicted entity re-enters as a *new*
track, and that eviction-induced fragment is indistinguishable from a matcher failure unless
`max_entities` is swept alongside. Same asymmetric policy as SV-2.3: **gate `false_merge == 0`,
report switch and fragment rates across the sweep.** No measurement was taken; the sweep axes are a
code-reading conclusion.

### 5.9 SV-4.9 — Temporal reducer performance — **SINGLE, derived, with measured headroom**

- **Bar**: reducer **p95 submit latency <= 1.0 ms** AND **sustained throughput >= 15,000 submits/s**.
- **Derivation (not invented, not fit):**
  - *Latency*: `percept/slo.py:91` already pins `REFLEX_P95_THRESHOLD_S = 0.0100 s`, itself derived
    as 15% of the `max_fps = 15` frame period. The reducer sits behind that gate, so it may consume
    only a small fixed fraction of it. **10% of the reflex budget = 1.0 ms** — the same "small fixed
    fraction of an already-committed budget" construction `slo.py` uses for its own constants.
  - *Throughput*: the fastest anything upstream can feed the reducer is
    `SensePolicy.max_fps = 15.0` (`percept/sampling.py:35`, the runtime's documented cost ceiling)
    times the concurrent Watch count. **15 fps x 1,000 concurrent Watches = 15,000/s**, so a reducer
    at that rate cannot be the bottleneck by construction.
- **Measured on the launch wheel** (offline, `SerializedDecisionReducer` + `Gate` +
  `EvidenceSequencer`):

  | shape | throughput | p50 | p95 | max |
  |---|---:|---:|---:|---:|
  | 1 key, serial | 364,144/s | 2.3 us | 3.0 us | 650 us |
  | 16 keys, 16-way | 332,523/s | 2.4 us | 3.0 us | 7,055 us |
  | 256 keys, 256-way | 234,180/s | 4.0 us | 5.5 us | 151 us |
  | 1024 keys, 1024-way | 110,309/s | 8.4 us | 14.1 us | 5,139 us |

  Worst measured p95 is **14.1 us = 1.4% of the proposed bar (71x headroom)**; worst throughput is
  **7.4x the proposed floor**. The bar is deliberately set far from the observation so it is a
  regression tripwire, not a fit to a passing run.
- **Gate on p95, report max.** Measured `max` reached 7.1 ms under contention — event-loop / GC tail,
  not reducer work. A max-based bar would be a flake generator.

---

## 6. The ceiling on any bar this corpus can carry

Using `percept/eval/exact_ci.py` (Clopper-Pearson, 95% one-sided lower bound — the tree's own
machinery, no normal approximation), the **maximum bar that a PERFECT run could support**:

| goal_kind | cases | GT intervals | max ratifiable bar (interval-level, 0 misses) |
|---|---:|---:|---:|
| `count` | 28 | 233 | **0.987** |
| `duration` | 40 | 40 | **0.928** |
| `transition` | 28 | 38 | **0.924** |
| `state` | 4 | 18 | **0.847** (case-level: **0.473**) |
| all | 100 | 329 | 0.991 |

Trials required for a bar to be *supported*: 0.80 needs n=14 at zero misses, 0.90 needs n=29,
**0.95 needs n=59**.

**Consequence: a 0.95 admission-recall bar is not ratifiable for `duration`, `transition` or `state`
on this corpus, at any configuration, even on a flawless run.** This is arithmetic, not pessimism,
and it is why the proposed bars in §5.1 top out at 0.90. It independently vindicates the spec's
refusal of a universal 0.95.

---

## 7. Where the swept bars live

`percept/eval/viability_frontier.py` already is this machinery, and it should be reused rather than
paralleled:

| Concept here | Existing type |
|---|---|
| one swept configuration | `_OperatingPointSpec` — `{id, configuration, configuration_sha256}` |
| per-archetype bars | `ArchetypePolicy` — `min_candidate_recall`, `min_event_recall`, `min_precision`, `min_call_reduction`, `max_event_abstention_rate`, `max_missing_rate` (no field has a default — every one must be pre-registered) |
| economy at an operating point | `CallReduction` — exact integer numerator/denominator, negative allowed so a regression is reported not clipped |
| the frontier | `_mark_pareto` — `pareto_frontier` and `policy_feasible_pareto` |

Three properties make it the right home rather than a coincidental fit: `configuration_sha256` is
verified against the canonical JSON of the configuration, so an operating point cannot silently
change identity; `ArchetypePolicy` has **no defaults**, so an unstated bar is a load error rather
than a silent zero; and `_mark_pareto` already computes exactly the recall-vs-call-reduction frontier
this document produces. It also fails closed on qualification (`launch_qualified = False` for raw
manifest-and-row input), which is correct for a pre-ratification artifact.

---

## 8. Measured vs judgement — the honest split

| Proposal | Basis | Confidence |
|---|---|---|
| `block_grid` is material for `block_max` (§4b, §4c) | **Measured** — 9,107 samples x 7 grids, wheel scorer | High. Effect is large (up to 31 recall points at matched economy) and mechanically explained. |
| `block_grid` is immaterial for `mean` | **Measured + proved by construction** | Certain. Bit-identical across all grids/datasets; follows from exact tiling. |
| `block_grid` is near-immaterial for `p95` | **Measured** | Medium-high. <= 4 points across the full range; single 25-case subset. |
| `salience_stat` is the largest lever; `mean` is blind on 28% of cases | **Measured** | High. Direction matches the independent in-tree PT calibration (N=80). |
| `enter_debounce` is material and archetype-dependent | **Measured** | Medium-high. Clear monotone effect; small per-dataset n. |
| `ego_compensate` trades recall for economy | **Measured, but n=3 moving-camera cases** | **Low.** Establishes it must be swept; does **not** establish a value. |
| `utterance_threshold` saturates at the shipped default | **Measured** on real audio | High for the finding. **Recall bar remains unpinnable — no VAD ground truth exists.** |
| SV-4.9 latency/throughput headroom | **Measured** on the launch wheel | High. 71x and 7.4x margins. |
| SV-4.9 bar *values* (1.0 ms, 15,000/s) | **Derived** from `slo.py:91` + `sampling.py:35` | High — a stated construction, not a fit. |
| SV-1.3 = 0.50 | **Existing code pin, reused verbatim** | Highest — nothing new asserted. |
| SV-2.1 bar = OBO >= 0.354 | **Existing external reference**, in-tree | Medium-high for the value; the frontier cells are unmeasured by me. |
| SV-2.1 fps / `smooth_s` / `canonical_select` are material | **Cited from in-tree docstrings**, not re-measured | Medium — the tree records the numbers; I did not reproduce them. |
| SV-1.1 archetype bars (0.90 / 0.90 / 0.85 / 0.80) | **Measured recall + exact-CI ceiling**, then a judgement call on the margin | Medium. The *ceilings* are arithmetic; the chosen margins below/above them are judgement. |
| SV-2.2 bar = 0.80 | **JUDGEMENT CALL** — reasoned from `select_dwell = 2` semantics | Low-medium. No measurement. Founder must ratify. |
| SV-2.3 / SV-4.2 `false_merge == 0` | **JUDGEMENT CALL** — derived from the spec's own stated asymmetry | Medium. The *shape* is well-grounded; no measurement of achievability. |
| SV-2.3 / SV-4.2 sweep axes | **Code reading only** | Medium. Axes are certain; ranges are guesses. |
| SV-2.4 bar = 0.90 | **JUDGEMENT CALL, and unevaluable** — no fixture exists | **Lowest.** Placeholder for a bar, not a bar. |
| Instant tolerance = +/-1.0 s (omnipro) | **JUDGEMENT CALL** made to enable measurement at all | Low. Must be ratified; it moves every omnipro number. |

**Nothing here was chosen because it made something pass.** Two proposals sit deliberately *below*
the observed value (SV-1.1 `count`, SV-4.9 both bars) and one deliberately *above* it (SV-1.1
`state`).

---

## 9. Items that block or distort the first run

### 9.1 Corpus defect — `omnipro` positives are all `[[0.0, 0.0]]`

All 20 `omnipro` cases carry `positives = [[0.0, 0.0]]` while the real trigger times sit unused in
`provenance.ground_truth[*].trigger_time_sec` (e.g. case `omnipro::TXeVf42sQdM::80` has
`trigger_time_sec = 29` but a positive at 0.0). Any harness reading `positives` directly will score
omnipro against an instant at t=0 and produce a **silent near-zero recall that reads as a runtime
failure** — the exact defect class the spec records finding three times already. `repcount`
`positives` is likewise empty (`[]`); it must be derived from `cycle_bounds_frames` using each file's
own fps. **Both need fixing in `build_corpus.py` before SV-1.1 runs.** All omnipro numbers in this
document use the `provenance` trigger times.

### 9.2 `block_grid` is not configurable

It is a function argument with an `(8, 6)` default, not a `Settings` field and not on the wire
(`WatchCondition` carries `salience_stat` but not the grid). If the sweep concludes that 12x9
outperforms 8x6, **acting on that requires a code change**, not configuration. The founder should
decide whether the sweep is diagnostic-only for 1.0.0, or whether an `EdgeSettings.block_grid` +
`WatchCondition.block_grid` field is in scope. Note the parity constraint: `_block_scores` is
bit-identical to the JS edge, so the grid must ride the wire or the two diverge.

### 9.3 Two archetype vocabularies exist

The corpus uses `goal_kind` in {`duration`, `transition`, `count`, `state`};
`docs/reference/operating-frontier.md` uses {`state`, `transition`, `brief_action`, `persistence`,
`absence`, `repeated_episode`}. These proposals use `goal_kind` because that is what the corpus
carries, but **a third vocabulary must not be created.** The founder should pick one and map the
other to it before these bars are pinned.

### 9.4 Instant-positive tolerance is itself a pin

`omnipro` positives are zero-width. Interval recall against a zero-width instant is undefined without
a tolerance. **+/-1.0 s** was declared to make measurement possible; it is arbitrary and it moves
every omnipro number in this document. It must be ratified alongside the bars.

### 9.5 Sample-size warnings that must survive into the scorecard

- `state`: 4 cases. **Report, do not gate.**
- `perception_test` camera-moving: 3 cases. The ego_compensate finding is directional only.
- `proactiveqa` has no audio stream — it contributes nothing to SV-1.6.
- Every §4 number comes from a **25-case subset**. They are sufficient to decide *which knobs need
  sweeping*; they are **not** sufficient to set a final value, which is what the full 100-case run is
  for.

---

## 10. Summary of the decision requested

| Bar | Proposal | Kind |
|---|---|---|
| SV-1.1 | per-archetype policy over a `salience_stat x block_grid x threshold x ego x debounce` frontier; recall 0.90 / 0.90 / 0.85 / *report-only*, reduction 0.50 / **0.25** / 0.50 / 0.50 | **swept** |
| SV-1.3 | `>= 0.50` — reused verbatim from `slo.py:243` | single (existing pin) |
| SV-1.6 | swept `utterance_threshold x hangover_frames x min_speech_frames`; **recall bar deferred — no ground truth** | **swept + deferred** |
| SV-2.1 | `OBO >= 0.354` over a swept `fps x smooth_s x canonical_select` frontier; score all 20 cases | single bar / swept config |
| SV-2.2 | `>= 0.80` over a swept `select_margin x select_dwell` frontier | single bar / swept config |
| SV-2.3 | `false_merge == 0`; fragment/switch reported over `appearance_threshold x iou_match` | asymmetric |
| SV-2.4 | `>= 0.90` per-attribute — **unevaluable, no fixture** | single (blocked) |
| SV-4.2 | `false_merge == 0`; rates reported over `iou_match x max_entities` | asymmetric |
| SV-4.9 | p95 `<= 1.0 ms`, throughput `>= 15,000/s` — derived from `slo.py:91` + `sampling.py:35` | single (derived) |

Plus four decisions that are not bars: fix the corpus positives (§9.1), rule on `block_grid`
configurability (§9.2), pick one archetype vocabulary (§9.3), ratify the instant tolerance (§9.4).
