---
title: Edge Wake Policy
updated: 2026-07-23
applies_to: percept-vision 1.0.0
status: stable
---
# Edge Wake Policy

The tier-0 gate decides **when percept spends a judge call**. It is pure arithmetic over downsampled
frames — no model, no network, no user data leaving the process. Everything it reads is small enough
to state exactly, so this page states it exactly: what the gate computes, which knobs change it, in
what order those knobs win, and how to tell whether a change helped.

This is the reference `percept/config.py` points at from `EdgeSettings`. For the *procedure* of
getting from "nothing fires" to a tuned eye on your own content, read
[how-to: tuning admission](tuning-admission.md) — this page is the contract underneath it.

> **Scope.** Everything here is about **admission** — whether a frame reaches the judge at all. It
> says nothing about whether the judge then answers correctly, and no number on this page is a
> product accuracy claim.

---

## 1. The gate, mechanically

### The gate frame

Capture downsamples every frame to a **48x36 greyscale** `Frame` before the gate ever sees it
(`GATE_W, GATE_H = 48, 36` in `percept/runtime/source.py`). That is 1,728 pixels — deliberately
tiny, because the gate's only job is *still vs. moved*, and a full-resolution diff would cost more
than the call it is trying to avoid.

Everything below operates on that 48x36 buffer, never on the judge frame.

### The diff

Consecutive gate frames are absolute-differenced per pixel. `prev_frame is None` (the first frame,
no predecessor) scores `0.0` — there is nothing to diff, so the very first frame is never motion.
Mismatched dimensions raise; they are a contract error, not a degraded case.

### The block partition

`block_grid` — a `(Gx, Gy)` pair — tiles the active area (the whole gate frame, or a `region_hint`
box) into `Gx * Gy` blocks. Block boundaries are **floor-of-product**:

```
bx(i) = x0 + (i * aw) // Gx
by(j) = y0 + (j * ah) // Gy
```

so adjacent blocks differ by at most one pixel and the partition is bit-identical in Python and in
the JS edge (`percept-edge`) — a release-gated parity invariant, see
[edge-parity](edge-parity.md). Each block scores

```
block_score = sum_abs_diff / (block_px * 255)
```

an integer sum over an integer denominator, so the numpy fast path and the pure-stdlib path return
the same float. A degenerate block (0 px, only reachable when the active area is smaller than the
grid) contributes `0.0`.

### One pass prices all three statistics

`_block_scores` (`percept/harness/tier0.py`) makes **one** pass over the pixels and returns
`(per_block_scores, total_abs_diff, area_px)`. Every salience statistic in the module is then a
**pure selector over that list**:

| stat | selector | note |
|---|---|---|
| `mean` | `total / (area_px * 255)` | the blocks tile the area exactly, so summing their integer numerators over the integer area denominator **is** the whole-area mean |
| `block_max` | `max(scores)` | the single loudest block |
| `p95` | nearest-rank, `idx = (95*n + 99) // 100 - 1` | integer rank only — no float percentile, which would drift between Python and JS |

This is why `salience_stats(prev, frame)` reports all three for the cost of one, and why
`percept doctor`'s counterfactual table **cannot disagree with the live gate**: the counterfactual
rows and the escalating row come from the same pass. `motion_score(..., stat=s)` and
`salience_stats(...)[s]` are asserted bit-identical in `tests/test_doctor.py`.

### What escalates, and what gets reported

Two different numbers, and conflating them is the most common misreading of a trace:

- **The escalate comparison** uses the condition's resolved `salience_stat` (and its `region_hint`,
  or the whole-frame ego residual when `ego_compensate` is on) — `_condition_motion`.
- **The reported gauge** `Tier0Signal.motion` is **always the whole-frame `mean`**, whichever
  statistic is active. It stays the plain mean so it is comparable across configs.

So a trace can show `motion = 0.004` on a frame that was admitted. That is not a bug; the gauge and
the decision answer different questions.

---

## 2. The three statistics

```
SALIENCE_STATS = ("mean", "block_max", "p95")
```

| stat | what it measures | blind to |
|---|---|---|
| `mean` | whole-frame average abs-diff | anything small. A mover covering 1% of the frame contributes 1% of its own intensity to the average |
| `block_max` | the loudest single block | nothing localized; it is the most sensitive of the three, and correspondingly the noisiest |
| `p95` | the 95th-percentile block (nearest-rank) | a *single* active block on a fine grid — with 48 blocks the 95th-percentile rank is index 45, so one hot block ranks below it and scores 0 |

### They live on different value scales

`block_max` and `p95` select an *individual block*; `mean` averages over all of them. A single active
block is by construction `>=` the frame average, so **the mean-tuned `0.10` cutoff over-fires the
other two**. Each statistic therefore carries its own threshold:

| stat | `Settings.edge` field | env var | default |
|---|---|---|---|
| `mean` | `motion_gate` | `PERCEPT_MOTION_GATE` | `0.10` |
| `block_max` | `motion_gate_block_max` | `PERCEPT_MOTION_GATE_BLOCK_MAX` | `0.20` |
| `p95` | `motion_gate_p95` | `PERCEPT_MOTION_GATE_P95` | `0.15` |

The stat -> threshold-field routing is `_SALIENCE_THRESHOLD_PATH` in `percept/resolver.py`, mirrored
by `compile_watchspec`'s `stat_thresholds` argument. An un-mapped stat falls back to the base
`threshold`.

**Comparing the three values against one cutoff is a category error**, and nothing in the codebase
ever returns a single "the" salience for exactly that reason.

### Why `mean` can admit nothing at all

A whole-frame average dilutes a small mover against a still background. Concretely — a 6x6-pixel
patch of a 48x36 gate frame going from mid-grey to white:

```python
from percept.harness.tier0 import salience_stats
salience_stats(still_frame, one_bright_block)
# {'mean': 0.0104, 'block_max': 0.498, 'p95': 0.0}
```

`mean` reads `0.0104` against a `0.10` cutoff — a factor of ten short, so it admits **nothing**, and
the judge never sees the event. `block_max` reads `0.498` against `0.20` and admits. `p95` reads
`0.0`, because one hot block out of 48 does not reach the 95th-percentile rank.

That failure is not "the gate was conservative." It is the gate being **structurally unable to see**
the event. A stat that admits zero frames produces zero evidence, and no downstream tuning recovers
it.

---

## 3. The shipped default, and its evidence

**`Settings.edge.salience_stat = "block_max"`.** This was `"mean"` for one reason — byte-for-byte
compatibility with the legacy whole-frame gate. That compatibility is not worth what it costs.

Two independent measurements, both recorded in `percept/config.py`'s `EdgeSettings` and in the
`tunables.json` note for `Settings.edge.salience_stat`:

| corpus | `mean@0.10` | `block_max@0.20` | `p95@0.15` |
|---|---|---|---|
| Perception Test action events (N=80, 7.8k frames) | 0.06 recall | 0.85 | 0.79 |
| 100-case stage-validation corpus, 2,391 configs, **matched economy** | 0.572 recall, **zero frames admitted on 24/100 cases** | 0.947 recall, zero on **2/100** | — |

A +0.376 recall move on the second corpus, with no new configuration surface. These are **tier-0
admission calibration figures on labeled corpora** — the admit/recall of the cheap wake channel, used
to set shipped threshold defaults. They are not an end-to-end percept accuracy claim; see the README
"Benchmark status".

Raising the `block_max` cutoff trades recall for cost (Perception Test action events: `0.94 @ 0.15`,
`0.85 @ 0.20`, `0.68 @ 0.25`).

### Two negative results, stated plainly

Both matter more than the positive one, because both close off tuning directions that look
attractive:

1. **Conditioning the statistic per goal shape was measured and rejected.** The same sweep found a
   regret reduction of **0.048** against a **pre-registered 0.05 bar**, and the effect was
   **negative under leave-one-out**. So `salience_stat` stays a single global default with a
   per-source-class override, and does *not* become a per-goal-shape lookup. Percept ships the
   measurement that says "don't", not a knob that looks like it works.

2. **`mean` is grid-invariant by construction.** Because the blocks tile the active area exactly,
   `total / (area_px * 255)` is the same number for every grid. Verified on the single-block fixture
   above:

   | `block_grid` | `mean` | `block_max` | `p95` |
   |---|---|---|---|
   | (4,3) | 0.0104 | 0.1245 | 0.1245 |
   | (8,6) | 0.0104 | 0.4980 | 0.0000 |
   | (12,9) | 0.0104 | 0.4980 | 0.0000 |
   | (16,12) | 0.0104 | 0.4980 | 0.0000 |
   | (24,18) | 0.0104 | 0.4980 | 0.0000 |

   **Under `mean`, changing `block_grid` is a no-op.** If you are running `mean` and reaching for the
   grid, you are tuning a variable that cannot move your result.

---

## 4. `block_grid` — not currently a declared tunable

**Say this part plainly: `block_grid` is not configurable.** It exists only as a function default
`(8, 6)` on `motion_score`, `salience_stats` and `_block_scores` in `percept/harness/tier0.py`. It
appears nowhere in `tunables.json`, nowhere in `Settings`, nowhere in `SourceProfile.parameters`, and
nowhere in `WatchProfile`. No env var reaches it. The only way to vary it today is to call
`percept.harness.tier0` directly from your own code — a harness experiment, not a deployment setting.

If it were promoted to a tunable, these are the grids that tile 48x36 into **square** blocks:

| `block_grid` | block size | blocks |
|---|---|---|
| (4,3) | 12x12 px | 12 |
| (8,6) | 6x6 px | 48 (the shipped default) |
| (12,9) | 4x4 px | 108 |
| (16,12) | 3x3 px | 192 |
| (24,18) | 2x2 px | 432 |

`(48,36)` also tiles exactly, at 1x1 px — degenerate, since `block_max` then reduces to the loudest
single pixel. And if blocks need not be square, 90 `(Gx, Gy)` pairs tile the frame with uniform
blocks (10 divisors of 48 x 9 divisors of 36); the floor-of-product boundaries mean *any* grid runs,
just with blocks differing by a pixel.

Measured spread in recall across grids at matched economy is roughly **0.099**, with **no stable
optimum** — no grid wins consistently, which is a large part of why it has not been promoted.

> **Provenance caveat.** That 0.099 spread figure comes from a sweep that is **not recorded anywhere
> in this repository** — unlike the `salience_stat` defaults, which are written into
> `percept/config.py` and `tunables.json`. Treat it as unverified against shipped code.

---

## 5. How to change it — the precedence chain

`percept/resolver.py` defines the layer order (`PRECEDENCE`), lowest to highest:

```
SDK -> SourceProfile -> DomainPack -> ExecutionPolicy -> OptimizationLock -> Deployment -> WatchPlan
```

For admission specifically that means:

| # | layer | how you set it | wins over |
|---|---|---|---|
| 1 | **SDK default** | `Settings.edge.salience_stat = "block_max"`, `motion_gate*` | nothing |
| 2 | **SourceProfile** | `parameters['salience_stat']` + `parameters['salience_threshold']`, bound onto `Settings.edge.*` via `_SALIENCE_THRESHOLD_PATH` | SDK |
| 3 | **Deployment / env** | `PERCEPT_SALIENCE_STAT`, `PERCEPT_MOTION_GATE`, `PERCEPT_MOTION_GATE_BLOCK_MAX`, `PERCEPT_MOTION_GATE_P95` | SourceProfile |
| 4 | **WatchPlan** | `WatchProfile.salience_stat` / `WatchProfile.stat_thresholds`, declared as an explicit plan override | everything |

A `SourceProfile`'s salience binding only fires when the profile is `admission_tuned` **and** declares
a `salience_stat` in the recognized vocabulary; a threshold that is not a real number is ignored
rather than coerced.

Note the two mechanisms that both mention `WatchProfile.salience_stat`. `ConfigResolver` treats it as
a `WatchPlan`-layer *explicit* override that expands onto `Settings.edge.salience_stat`. Separately,
`resolve_watch_profile` (`percept/profile_wiring.py`) copies a `WatchProfile`'s fields onto
`Settings.edge` **unconditionally** — and `WatchProfile.salience_stat`'s own field default is
`"mean"`, not `"block_max"`.

> **Footgun.** Resolving a `WatchProfile` you constructed without naming a statistic silently
> downgrades the deployment from `block_max` to `mean`:
>
> ```python
> WatchProfile(name="bare", environment="indoor", goal_type="event", state="dynamic")
> # resolve_watch_profile(...).settings.edge.salience_stat == "mean"
> ```
>
> If you build `WatchProfile`s by hand, set `salience_stat` explicitly.

### Which shipped profiles declare a statistic

`percept/source_profiles_default.py`:

| profile | `admission_tuned` | `salience_stat` | `salience_threshold` | coast |
|---|---|---|---|---|
| `physical_scene` | yes | `block_max` | `0.20` | none — motion *is* the signal |
| `screen_document` | yes | `block_max` | `0.20` | `2.0s`, `coast_required=True` |
| `conversation` | yes | *(none)* | — | none |

`conversation` is audio/text only: its tuned admission signal is `utterance_boundary`, not a pixel
statistic, so it declares no `salience_stat` and contributes nothing to the salience binding. It is
`admission_tuned=True` — tuned, just not on this axis — and it is marked `stability="experimental"`;
do not build on it.

### Worked example: a per-source override

```
$ percept resolve --condition "a person enters the loading bay" \
                  --source-profile physical_scene --explain

  Settings.edge.motion_gate           = 0.1         <- SDK (sdk_default) [registry default]
  Settings.edge.motion_gate_block_max = 0.2         <- SourceProfile (physical_scene)
  Settings.edge.motion_gate_p95       = 0.15        <- SDK (sdk_default) [registry default]
  Settings.edge.salience_stat         = 'block_max' <- SourceProfile (physical_scene)
```

The profile bound `salience_stat` and *only* the `block_max` threshold — the `mean` and `p95` cutoffs
stay at their SDK defaults, because the profile declared one statistic, not three.

Now an env override at the Deployment layer:

```
$ PERCEPT_SALIENCE_STAT=p95 PERCEPT_MOTION_GATE_P95=0.12 \
  percept resolve --condition "a person enters the loading bay" \
                  --source-profile physical_scene --explain

  Settings.edge.motion_gate_block_max = 0.2   <- SourceProfile (physical_scene)
  Settings.edge.motion_gate_p95       = 0.12  <- Deployment (env:PERCEPT_MOTION_GATE_P95)
  Settings.edge.salience_stat         = 'p95' <- Deployment (env:PERCEPT_SALIENCE_STAT)
```

`--explain` prints the owning layer and the reason for **every** resolved field, so "why is this value
what it is" is never guesswork.

---

## 6. What is tunable at all — the governance model

The registry (`percept/tunables.py`, mirrored to `tunables.json`) declares **160** tunables. Of those,
**31** carry `safe_to_optimize=True`, and **30** survive the optimizer's guardrail deny-list — so 30
fields are effectively sweepable.

### The deny-list is checked before the registry

`percept/optimizer.py` carries a hard-coded, data-independent `GUARDRAIL_FORBIDDEN_PREFIXES`:

```python
('Settings.gate.', 'WatchProfile.gate.', 'Goal.state.', 'Settings.consent.',
 'Percept.create.allow_scripted_brain', 'Settings.episodes.window_s',
 'Settings.events.preroll_max_bytes', 'SourceProfile.parameters.coverage_floor')
```

covering gate vocabulary (`percept.gate.Gate` is the sole firing authority), goal meaning (the whole
`Goal.state.` namespace — conditions, `fire_when`, miss/false-fire costs, rearm policy), consent and
the actuation envelope, and retention/egress ceilings.

It is checked **before** the registry is consulted, so flipping `safe_to_optimize` in `tunables.json`
cannot hand the optimizer a path it must never touch (ADR-008). A scored recall number is never
allowed to rewrite what a goal *means* or widen what the SDK is allowed to *do*.

### `salience_stat` is owner-selected, not optimizer-searched

```python
TunableSpec(path='Settings.edge.salience_stat', owner='SourceProfile',
            visibility='public', safe_to_optimize=False, default='block_max',
            value_type='enum', choices=('mean', 'block_max', 'p95'),
            env_var='PERCEPT_SALIENCE_STAT', requires_recompile=True, risk='medium')
```

`safe_to_optimize=False` with `owner='SourceProfile'` is deliberate. Which eye a source class needs is
a **declaration about the world** — static camera, moving camera, screen content — not a number a
search should discover per fixture. The measured rejection of per-goal-shape conditioning (§3) is the
evidence behind that stance.

### `visibility` and `safe_to_optimize` are orthogonal

They answer different questions, and conflating them is a common misreading:

- `visibility=public` — **may a user set this?**
- `safe_to_optimize` — **may an automated search move this?**

138 of 160 tunables are `public`; **107 of those are public but not sweepable**. Almost every knob you
can legitimately configure is one the optimizer is not allowed to touch.

---

## 7. How to measure whether a change helped

### `percept doctor` — the counterfactual table

`percept doctor` is a tier-0 dry run: it opens the source, compiles the plan, runs the **real**
`MotionGate`, and prints what admitted and which rule suppressed the rest. It spends **zero** judge
calls and needs no API key.

```
$ percept doctor --source clip.mp4 --goal "a small object moves across the lower left" --fps 10

── GOAL g0 ──────────────────────────────────────────────────────────────────
  active stat / gate_on   block_max / motion
  admitted                2
  SALIENCE STATS — each against ITS OWN calibrated threshold (different scales,
  never one cutoff): mean is whole-frame; block_max/p95 are localized blocks.
    stat             max    median      mean   threshold  admitted
    mean          0.0166    0.0000    0.0026       0.100         0
    block_max     0.3645    0.0000    0.0436       0.200         2 <- active
    p95           0.1819    0.0000    0.0284       0.150         1
  WHICH RULE SUPPRESSED THE NON-ADMITS
    no_predecessor_frame    1
    below_threshold         57
```

Every row is the real gate re-run under that statistic against **that statistic's own effective
threshold** (base cutoff plus the goal's `motion_bias`). `StatDiagnosis.headroom` is
`max_value - effective_threshold`: negative means nothing in the run cleared, and a *small* negative
is a near miss worth a second look.

### Reading a zero-admission result

When nothing admits across the whole run, doctor emits `ZERO_ADMITS_ON_FIXTURE` (warn) and — if the
run has no coverage at all — exits `2` (`EXIT_ZERO_COVERAGE`). **This is diagnostic, not automatically
a failure**: a quiet scene under a motion-gated watch correctly admits nothing and saves every model
call.

The discriminator is the counterfactual table:

| what the table shows | diagnosis |
|---|---|
| active stat admits 0, another stat **would** admit (`ALTERNATIVE_STAT_WOULD_ADMIT`) | the **eye is mis-selected** — change `salience_stat` |
| all stats admit 0, max is close to a cutoff | genuinely quiet, or a threshold worth measuring |
| all stats admit 0, max is a fraction of every cutoff, **on content you know changed** | the **eye is under-resolved** — you need a different admission signal, not a lower threshold |

Running the same clip under `PERCEPT_SALIENCE_STAT=mean` produces exactly the first case:

```
  [WARN ] ALTERNATIVE_STAT_WOULD_ADMIT
      g0: active stat 'mean' admitted 0; block_max would admit 2, p95 would
      admit 1 (each against its OWN calibrated threshold)
```

A fourth finding, `COAST_DISABLED_ON_REQUIRED`, fires when a profile declares `coast_required=True`
but a `strict_admission` ExecutionPolicy set `Settings.edge.coast_enabled=False` — the profile's only
non-signal admission mechanism was unbound, and the run then admitted zero. Both decisions are named
so you can see which to reverse.

### The two metrics

**Admission recall** — the fraction of ground-truth positive intervals with at least one admitted
sample. Two implementations, on different denominators:

- `percept.harness.golden_eval.score` — **episode-level**: `recall = matched_episodes / episodes`,
  one-to-one matched. A long episode gets no extra credit for extra frames, and a burst of duplicate
  predictions inside one episode does not inflate recall.
- `percept.eval.viability_frontier` — `candidate_recall = admitted_event_candidates /
  event_candidates`, i.e. **window-level within labeled event windows**. Reported alongside
  `event_recall = event_fires / event_windows`.

**Economy** — `admitted / total`. In the frontier evaluator this is `admitted_rate = admitted /
candidate_windows`. Its complement is `LoopReport.reduction`:

```python
reduction = 1.0 - (vision_calls / frames)
```

re-exported as `percept.slo.fovea_call_reduction(report)`, with `FOVEA_CALL_REDUCTION_FLOOR = 0.50` —
the architecture's minimum bar for tier-0 gating to be worth having at all. A gate that admits more
than half the frames is not paying for itself.

### The one methodological rule: compare at matched economy

**Always hold economy constant when comparing configs.** A config that admits more will trivially show
higher recall — at the limit, admitting every frame gives perfect admission recall and zero call
reduction, which is the ungated baseline wearing a tuning result as a disguise.

This is not a stylistic preference. The **grid conclusion inverts** when economy is not held constant:
an unmatched comparison makes finer grids look strictly better, because finer grids raise `block_max`
(more blocks, smaller denominators, a higher maximum) and therefore admit more. Match the economy and
the ~0.099 spread has no stable optimum.

`percept.eval.viability_frontier` exists to do this correctly: it compares a baseline arm against
adaptive operating points on **the same cases, the same candidate windows, the same event windows**,
reports rates with their exact integer denominators, and requires every archetype's operating points
to be pre-registered in the input manifest. It refuses to choose a universal threshold for you.

---

## 8. If your Watch never fires

Check admission **before** you blame the judge. A judge that is never called cannot be wrong, and
every prompt change you make is untestable until frames reach it.

1. **Run `percept doctor` on the same source.** Zero cost, no API key. Read `admitted` first.
   - `admitted > 0` -> admission is working. The problem is downstream: the judge, the condition
     wording, or the gate's rising-edge/refractory policy. Stop here and look there.
   - `admitted == 0` -> continue.

2. **Read the `WHICH RULE SUPPRESSED THE NON-ADMITS` block.** It attributes every non-admit to one of
   `SUPPRESSION_REASONS`:
   - `out_of_schedule` / `cadence_not_elapsed` — the gate never even looked. The eye is not the
     problem; fix the schedule or `min_judge_interval`.
   - `signal_unavailable` — the goal gates on a channel nothing feeds (e.g. an audio-gated goal with
     no audio source opened). Fix the plan.
   - `no_predecessor_frame` — the first frame of the run, which is never motion by definition.
   - `unexplained` — a doctor bug; it is reported as an ERROR finding, please file it with the JSON.
   - `below_threshold` — continue.

3. **Read the counterfactual table.** If `ALTERNATIVE_STAT_WOULD_ADMIT` fired, the eye is
   mis-selected: set that statistic (§5) and re-run.

4. **If every stat admits zero and the max is a fraction of every cutoff, on content you know
   changed** — the eye is under-resolved for this content. The 48x36 gate frame and its 6x6-pixel
   blocks dilute a document-sized edit away before any statistic sees it.

   **Do not lower the threshold to make the fixture admit.** Those cutoffs carry physical-scene
   precision, they are shared by every deployment reading the same settings, and dropping them buys
   admissions on compression noise rather than on real events. That is tune-to-green. Instead:
   - name the source class (`--source-profile screen_document`) so the coast floor binds — a fixed
     low-rate admission underneath the signal, which is why `screen_document` declares
     `coast_required=True`;
   - or use `gate_on="fixed"` for content that changes with ~zero pixel motion (see
     [screenshare](../watches/scenario-screenshare.md));
   - or supply your own change signal via `register_detector(...)` and prove it beats the baseline on
     your bars — the full loop is [how-to: tuning admission](tuning-admission.md).

5. **Confirm you resolved what you think you resolved.** `percept resolve --explain` prints the owning
   layer for every field. A hand-built `WatchProfile` silently resolving `salience_stat` to `"mean"`
   (§5) shows up here immediately.

---

## 9. Adjacent knobs

### `ego_compensate` — for a moving camera

`Settings.edge.ego_compensate` (default `False`, env `PERCEPT_EGO_COMPENSATE`) removes whole-frame
translation before gating. `ego_residual_score` searches integer shifts `(dx, dy)` in `[-k, k]`
(`k=4`) and takes the **residual** — the smallest mean-abs-diff between `prev` shifted by `(dx, dy)`
and `frame` over their overlap. A pure camera pan is explained by some shift, so its residual is
`~0` and does not escalate; a subject entering the frame cannot be shifted away, so its residual
stays high. The `(0,0)` shift equals the uncompensated mean, so the residual is always `<=` the
full-frame `motion_score`.

Pick it by **camera mount**, not by goal: wearable, glasses, drone, robot, handheld.

Two current limits, stated because they change what a trace means:

- **Ego and block do not compose.** `_condition_motion` checks `ego_compensate` first, so an
  ego-compensated goal gates on the whole-frame residual and the block statistics are not consulted.
  `percept doctor` says so explicitly with an `EGO_COMPENSATED_GATE` finding — the admitted counts are
  still the real gate's, but the value columns describe the *uncompensated* frames.
- **No IMU fast path.** Device angular velocity is not wired in; the shift search is the only
  estimator.

### The wake policy — fusing several weak cues

The gate can treat each detector output as a channel and fuse them into one cheap score
(`wake_score`, `percept/harness/detectors.py`):

```text
P(goal) = noisy_or(weight[channel] * signal[channel])
```

With **no** policy weights this is exactly `max(signals)` — the legacy path, unchanged. With weights
it becomes a noisy-OR over the named channels, so several moderate cues can wake the gate together
without any single one needing to be decisive. A policy `floor` raises the result to at least that
value.

Channels the reference gate populates:

| channel | meaning | source |
|---|---|---|
| `motion` / `change` | frame-diff magnitude | the tier-0 gate itself (always available) |
| `structural` | local geometry/layout salience | an optional structural adapter — `0.0` in the reference gate |
| `semantic` / `semantic_relevance` | goal-specific image relevance | an optional semantic adapter — `0.0` in the reference gate |
| `audio` / `audio_event` | acoustic energy | `Tier0Signal.energy`, when an audio source is opened |

> In the **reference** gate shipped here, `structural` and `semantic` are structurally `0.0` — no
> CLIP/pose adapter is wired. `percept doctor` reports them as `UNAVAILABLE` rather than as zeros you
> might mistake for measurements. The real `percept-edge` supplies them.

Audio is motion-invariant, so it is the most reliable wake when the visual channel is weak: weighting
`audio_event` makes a loud event escalate at zero motion. On Perception Test **sound** events whose
source is *not on screen* (`is_visible=0`), motion-only recall is **0.28**; adding the audio channel
lifts it to **0.70**.

### `gate_on` — which channel decides

Per goal, via `state["cadence"]["gate_on"]`. `"motion"` (the default) uses the salience statistic;
`"fixed"` ignores pixels entirely and escalates every `min_judge_interval` seconds — the right choice
for content that changes with ~zero pixel motion (reading, menus, screenshares, camera-covered
checks). `"structural"` / `"semantic"` gate on those channels, which means they can only admit if an
adapter feeds them.

---

## 10. Design intent: cheap channels per use case

Not measurements — the design table for which cheap channels a use case should lean on, and where the
judge and the gate divide the work.

| Use case | Cheap wake before VLM | Wake trigger | VLM confirms | Ask/refuse behavior |
| --- | --- | --- | --- | --- |
| Read a book with me | fixed cadence + OCR/text change + semantic relevance | stable readable page or new passage | notes, topics, summary | ask on unreadable text or uncertain section |
| Smoke or sparks | change + semantic relevance + optional audio event | plume, flicker, glow, alarm sound | smoke/sparks visible and severity | ask on haze, steam, glare |
| Flooding | fixed cadence + change + water/object presence | water-like region appears/grows | flooding/leak and visible source | ask on reflections or shadows |
| Breaking into car | person presence + change + audio event | person near car plus impact/open-door/glass | forced-entry indicators | ask unless clear; avoid identity claims |
| Speeding car ticketing | vehicle presence + tracking/change | vehicle crosses zone quickly | "vehicle passed quickly" only | refuse ticketing/enforcement claims |
| Somebody asking for help | audio event/STT + presence | distress phrase or scream | visible person/context | ask on ambiguous speech |
| Camera covered | fixed cadence + dark/blur/low-texture change | persistent occlusion-like frame | lens covered vs lights off | ask on natural darkness |
| Correct my posture | pose + fixed cadence + presence | poor posture sustained | posture relative to goal | ask when body is partly visible |
| Count jumping jacks | pose-openness + motion-periodicity | rep boundary peaks | activity type periodically | count boundaries, not per-frame VLM |
| Translate my menu | fixed cadence + OCR/text | readable menu region | translation/layout context | ask on glare, low OCR confidence |
| Tell me about pointed menu item | hand/pose + OCR + semantic relevance | pointer near an item | referenced item | ask if multiple nearby items |
| How do I peel this fruit | object presence + semantic relevance + hand motion | fruit-like object in hand | fruit/action state | ask if fruit is uncertain |
| Drink-water reminder | schedule/fixed cadence + presence/object/action | reminder time plus user context | whether drinking happened if needed | use rising edge; avoid repeated nagging |

Cheap local channels maximize recall and cost control; the judge does semantic confirmation and
explanation; the cognition gate still fires only on the rising edge and asks when the evidence is not
trustworthy.

---

## See also

- [how-to: tuning admission](tuning-admission.md) — the procedure this page is the reference for
- [how-to: the runtime worker](../watches/creating-a-watch.md) — where the gate sits in `run()`
- [how-to: screenshare](../watches/scenario-screenshare.md) — `gate_on="fixed"` for zero-motion content
- [reference: edge parity](edge-parity.md) — the Python/JS bit-identity invariants
- [explanation: motion and strip](how-admission-works.md)
- [ADR-005: source profiles](../../.adr/ADR-005-source-profiles.md) — why the statistic is owned by the source class
- [ADR-008: configuration ownership, resolution, provenance](../../.adr/ADR-008-configuration-ownership-resolution-provenance.md) — the precedence chain and the guardrails
