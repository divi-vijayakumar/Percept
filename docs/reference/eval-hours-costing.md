# Eval-Hours Costing for the Operating-Point Bars (PRD Revision B §3.5)

Status: reference — computed once, to be re-run whenever a candidate bar or the archetype list changes.

> This document exists to answer one question before anyone types a number into `false_fire_max` or
> `recall_min`: **how many held-out hours / labeled episodes does that number actually cost to defend?**
> PRD Revision B §3.5 is explicit that the bar must not be set before this arithmetic is done. This doc does
> the arithmetic, in the open, so §14.2 becomes an informed tradeoff instead of an aspirational guess.

All numbers below are exact (`scipy.stats.chi2` / `scipy.stats.beta`), not the `≈3/T` rule-of-thumb rounded
by hand. The rule-of-thumb is correct to two significant figures for the zero-event case and is kept as the
memorable form; the tables use the exact values.

## 1. The math

### 1.1 Zero-event case: the "rule of three"

A watch that's live for `T` scheduled watch-hours and produces a false FIRE at some true (unknown) rate `λ`
per hour is a Poisson process. If you observe **zero** false FIREs over `T` hours, the data doesn't prove
`λ = 0` — it bounds `λ` from above. The exact upper one-sided 95% confidence bound, for zero observed events,
is:

```
λ_upper = -ln(0.05) / T  ≈  2.996 / T  ≈  3 / T     (per hour)
```

This is the "rule of three": observing zero events in `T` hours supports a claim of "rate ≤ 3/T", not
tighter, no matter how clean the run felt. It does not get better with a "well-behaved" run — it is a
property of `T` alone.

### 1.2 General case: k observed events

If `k` false FIREs occur during the `T`-hour held-out run (not just zero), the exact upper one-sided 95%
bound comes from the standard Poisson–chi-square (Garwood 1936) relation:

```
λ_upper(k, T) = χ²_{0.95, 2(k+1)} / (2T)     (per hour)
```

where `χ²_{0.95, df}` is the 95th percentile of the chi-square distribution with `df` degrees of freedom.
Define `factor(k) = χ²_{0.95, 2(k+1)} / 2` so `λ_upper = factor(k) / T`, and invert for the hours a target
bar `b` requires:

```
T_required(b, k) = factor(k) / b
```

Exact `factor(k)` values (k = observed false FIREs during the run):

| k (observed false FIREs) | df = 2(k+1) | χ²₀.₉₅ | factor(k) = χ²₀.₉₅ / 2 |
|---|---|---|---|
| 0 | 2 | 5.9915 | **2.996** |
| 1 | 4 | 9.4877 | **4.744** |
| 2 | 6 | 12.5916 | **6.296** |

**Worked example** (this is the exact case PRD §3.5 cites): target `false_fire_max = 0.5` per scheduled
watch-hour, zero false FIREs observed. `T = factor(0) / b = 2.996 / 0.5 = 5.99 ≈ 6.0` clean held-out hours —
**per archetype**, matching the PRD's "~6 clean hours" claim exactly. If even one false FIRE happens during
that run, the same bar needs `4.744 / 0.5 ≈ 9.5` hours instead — the bound does not "reset," it gets
materially more expensive per additional event.

### 1.3 Recall: the mirror-image problem

`recall_min` is a *lower* one-sided 95% bound on a proportion (FIRE recall over labeled positive episodes),
not a rate — so it uses the exact binomial (Clopper–Pearson) interval, not Poisson. For `n` labeled positive
episodes with `x` correctly detected (FIRE) and `n − x` missed, the lower one-sided 95% bound is:

```
recall_lower = Beta⁻¹(0.05; x, n − x + 1)
```

(the `α`-quantile of a Beta(x, n−x+1) distribution — the standard exact/Clopper-Pearson form). For the
all-hits case (`x = n`, zero misses) this collapses to `recall_lower = 0.05^(1/n)`, which is why a small
`n` with a perfect record still doesn't clear a tight bar: a single held-out set that happened to go 10-for-10
only supports `recall ≥ 0.741`, not `≥ 0.90`.

**Worked example**: target `recall_min = 0.90`, zero misses. Smallest `n` such that `0.05^(1/n) ≥ 0.90` is
`n = 29` (`0.05^(1/29) = 0.9019`; `n = 28` gives `0.8985`, which fails). So **29 labeled positive episodes
per archetype**, all correctly caught, is the minimum defensible evidence for a 0.90 recall claim — one miss
in that set and the requirement jumps to 46.

## 2. Decision table — `false_fire_max`

Required **clean held-out hours per archetype**, by candidate bar and by how many false FIREs actually occur
during the run (a run isn't disqualified by a false FIRE — the required hours just go up):

| candidate `false_fire_max` (per scheduled watch-hour) | hours needed, 0 false FIREs (k=0) | hours needed, 1 false FIRE (k=1) | hours needed, 2 false FIREs (k=2) |
|---|---|---|---|
| 2.0 | 1.5 h | 2.4 h | 3.2 h |
| 1.0 | 3.0 h | 4.7 h | 6.3 h |
| **0.5** | **6.0 h** | **9.5 h** | **12.6 h** |
| 0.2 | 15.0 h | 23.7 h | 31.5 h |

All figures are **per archetype**. With Percept's six core gating archetypes (§5 below), multiply by 6 for
the total held-out-hour budget the bar commits the team to. E.g. `false_fire_max = 0.5` with a clean run
(k=0 across the board) is `6.0 × 6 = 36` total held-out hours; if one archetype throws a single false FIRE
during its run, that archetype alone costs 9.5 h instead of 6.0 h and the total budget grows accordingly.

## 3. Decision table — `recall_min`

Required **labeled positive episodes per archetype**, by candidate bar and by how many are missed (Clopper–
Pearson exact, one-sided 95% lower bound):

| candidate `recall_min` | episodes needed, 0 misses | episodes needed, 1 miss | episodes needed, 2 misses |
|---|---|---|---|
| 0.95 | 59 | 93 | 124 |
| **0.90** | **29** | 46 | 61 |
| 0.85 | 19 | 30 | 40 |
| 0.80 | 14 | 22 | 30 |
| 0.75 | 11 | 18 | 23 |

Same rule: per archetype, multiply by 6 for the total labeled-positive-episode budget. `recall_min = 0.90`
with a clean pass is `29 × 6 = 174` positive episodes; `0.95` nearly quadruples that to `59 × 6 = 354`.

## 4. Denominator discipline (§3.3) — why "per stream-hour" alone is incomplete

§3.2 requires rates reported per **scheduled watch-hour**; §3.3 additionally requires per **source-hour at a
stated number of active watches**. A bare "per stream-hour" figure is not wrong, it's *underspecified*, for
two independent reasons that both bite in production, not just in eval:

1. **Concurrency is invisible in a stream-hour.** One camera stream can host N concurrently armed watches
   (e.g. "package on porch" + "door left open" + "person loitering" all watching the same feed). "0.5 false
   FIREs per stream-hour" could mean 0.5 per watch with N=1, or a much worse 5.0 per watch-hour hiding behind
   an N=10 average — the two are wildly different operational realities and the stream-hour number alone
   cannot distinguish them. Reporting *scheduled watch-hours* forces the concurrency count into the
   denominator explicitly.
2. **Coverage is invisible in a stream-hour.** §3.4 requires degraded intervals to count against coverage and
   recall rather than silently shrinking the denominator. A "stream-hour" is ambiguous about whether it means
   raw ingested video (including time the watch was degraded/failed and structurally couldn't have fired) or
   only healthy observation time. Scheduled watch-hours is the honest denominator — a watch that was
   *scheduled* to be armed for an hour but was blind for 20 minutes of it still owes that 20 minutes against
   its coverage number, not a quiet subtraction from the reporting base.

Practical rule for this doc's tables and for §3.2 reporting: **always publish both** — per scheduled
watch-hour (the bar-eligible number) and per source-hour at a stated active-watch count (the deployment-
economics number a founder or customer will actually reason about, e.g. "how many false alerts will one
camera generate per day with our typical 3-watch bundle"). Neither one substitutes for the other.

## 5. A concrete capture schedule

Percept's six core gating archetypes (`eval/viability/ground_truth.py::CORE_ARCHETYPE_ELIGIBLE_AFTER_S`):
`state`, `transition`, `brief_action`, `persistence`, `absence`, `repeated_episode`.

Using the §2 worked bar (`false_fire_max = 0.5`, budgeting for k=1 per archetype as a realistic margin — a
plan that only survives a perfectly clean run is not a plan) and the §3 worked bar (`recall_min = 0.90`,
budgeting for 1 miss per archetype for the same reason):

| archetype | held-out clean hours needed (false-fire, k≤1 margin) | positive episodes needed (recall, ≤1 miss margin) | rough capture time |
|---|---|---|---|
| state | 9.5 h | 46 | ~1 day of scheduled recording + ~2–3 h staging/labeling the 46 episodes (short, few-second events) |
| transition | 9.5 h | 46 | same — transitions are quick to stage back-to-back |
| brief_action | 9.5 h | 46 | same |
| persistence | 9.5 h | 46 | slower to stage — each episode needs its hold duration to actually elapse; budget more wall-clock per episode |
| absence | 9.5 h | 46 | slowest to stage — each episode needs the full confirmation window (§ground_truth: longest of the six offsets) before it's labelable |
| repeated_episode | 9.5 h | 46 | each episode is itself multi-occurrence; budget more wall-clock per episode |
| **total** | **57 h** | **276** | |

Two independent capture tracks, not one:

- **Negative/clean footage** (the false-fire hours): this is just camera-on time with no true positive of
  that archetype in frame. It can run unattended (a porch camera recording overnight, an empty hallway) and
  overlaps across archetypes for free — one hour of quiet porch footage is simultaneously "clean" evidence
  for every archetype watching that scene, so the 57 held-out hours do **not** all need dedicated separate
  recording sessions if watches run concurrently on shared footage. Realistic elapsed time: **1–2 weeks** of
  passive recording across a small number of cameras, run in parallel with other work.
- **Positive/labeled episodes** (the recall episodes): these must be staged (something actually happens) and
  then hand-labeled against `ANNOTATION_RUBRIC` (`percept.contracts`). Staging is fast for `state`/
  `transition`/`brief_action` (seconds per episode, can batch many per session) and slower for `persistence`/
  `absence`/`repeated_episode` (each episode's ground-truth eligibility requires the actual hold/confirmation/
  repetition window to elapse — see the offsets in `CORE_ARCHETYPE_ELIGIBLE_AFTER_S`). Realistic elapsed time
  for 276 labeled positive episodes across six archetypes, including labeling: **3–5 days** of dedicated
  staging/labeling effort, not calendar time — it parallelizes across archetypes and people.

## 6. What this means for the decision (§14.2)

The founder's actual choice is a tradeoff, not a number picked in the abstract:

- **`false_fire_max = 0.5`** (the PRD's illustrative bar) with a 1-false-FIRE margin costs **~57 held-out
  hours total**, collectible passively in 1–2 weeks since clean footage overlaps across archetypes for free.
  This is affordable and is the recommended launch bar *if* that footage is actually scheduled and run before
  the bar is locked — not assumed.
- **`false_fire_max = 1.0`** halves the held-out-hour requirement (~3.0–4.7 h/archetype, ~18–28 h total) at
  the cost of a materially looser false-FIRE guarantee. Reasonable as an interim locked bar if the 1–2 week
  capture window isn't available before the release date the PRD is gating — but it should be stated as
  interim, with the tighter 0.5 as a documented follow-up once more hours are banked, not silently kept.
- **`false_fire_max = 0.2`** is expensive (~15–24 h/archetype, ~90–142 h total) and is not recommended as a
  v0.1 launch bar — flag as a post-launch tightening target once production watch-hours accumulate as
  additional held-out evidence.
- **`recall_min = 0.90`** with a 1-miss margin costs **276 labeled positive episodes total**, achievable in
  3–5 days of staging/labeling effort. This is the recommended launch bar.
- **`recall_min = 0.95`** more than triples the episode requirement per archetype (59→93 for 0-to-1 miss,
  354–558 total) — defensible only if the team has a genuinely larger labeling budget than a v0.1 timeline
  usually allows; treat as a stretch target, not the locked bar.

**The bar that costs nothing is the one nobody defends.** Whatever `false_fire_max` / `recall_min` values are
written into §14.2, PRD §3.5 requires the PRD to also record: the implied hours from the tables above, and
the hours actually run before the bar is called met. If the actual held-out run falls short of the table's
requirement for the chosen bar, the honest move is to report the bar the run *actually* supports (compute it
from §1's formulas with the real `T` and `k`) rather than the aspirational one.
