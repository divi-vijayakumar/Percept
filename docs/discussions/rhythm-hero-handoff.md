# Handoff — rep counting & the hero tracker

**Status:** working notes / handoff. Scratch experiments validated at L1 across RepCount-A N=8. Core changes
committed: the amplitude gate only (`a9c5aa0`). The hero-tracker work below is **scratch, not committed.**

---

## Problem

Count repetitions from video (the `rhythm` sense): the eye measures decomposed pose channels, the brain counts
cycles. On RepCount-A (N=8) the shipped differential counter scored **OBO 0.50 / MAE 0.89** — and the residual
error was not where we first looked.

## Observation

- **The counter is fine; the hero is the problem.** When the subject is correctly tracked, counting is good —
  bench counts **16 exactly** once the *lifter* is held. When the tracker drifts, the count is whatever the
  wrong body did.
- The geometric trackers **drift / fragment on real video**: nearest-hip and the entity-graph re-id both left
  the lifter for the standing spotter (one debug run spawned **19 entity-ids** for one clip; the "count" was the
  spotter's arm gestures).
- **Pommel is two compounding problems**: the apparatus occludes the legs so pose fragments, *and* the rep DOF
  (legs circling) isn't captured by any pose channel (`ankle-spread` is the wrong measure — the legs swing
  together, they don't spread).
- **The GT-0 clip** (train143) is a fidget — small repetitive motion that isn't the target action; it
  hallucinates a count unless explicitly rejected.

## Approaches tried

| # | approach | result | verdict |
|---|---|---|---|
| 1 | differential counter (shipped) | OBO 0.50 / MAE 0.89 | counting is fine *given* the right hero/channel |
| 2 | **amplitude gate** (`expected_rom`, committed `a9c5aa0`) | train143 5→0; bench 18→16 | ships; rejects the fidget |
| 3 | entity-graph re-id (continuity + HSV histogram) | fragmented (19 ids; bench→spotter) | **rolled back** — too weak on real, jittery detections; similar clothes defeat color |
| 4 | single-track Viterbi (pose+motion candidates) | one clean track, but picks the **smoothest** (spotter), not the performer | association ✓, selection ✗ |
| 5 | silhouette-PCA (the POC's pommel counter) | pommel **33** ✓; bench **39**, jack **27** | **regime-overfit** — only the whole-body-motion case; great *fallback*, not a general counter |
| 6 | red-pants tracker / generic motion+Viterbi (POCs) | pommel 35 / 28 | clip/regime-specific; confirmed Viterbi + motion-candidates is the right shape |
| 7 | **performer-aware Viterbi + pose-coverage gate** (current) | see scorecard | the production shape |

## Learning

- **Hero tracking is the primary lever, not counting.** This is the load-bearing insight: fix the subject and
  the existing counter works.
- **"Smoothest track" ≠ "the performer."** Global Viterbi linking solves *fragmentation* but still locks the
  most stable body (a standing spotter). You must link *multiple* tracklets and **select the performer = the one
  with the most rhythmic motion on the action channel.**
- **Appearance re-id by colour histogram is too weak on real video** (jitter + similar clothing). The dormant
  `AppearanceIdentity` seam is real but needs a stronger embedder than a histogram.
- **Silhouette-PCA is a regime-specific fallback** (whole-body periodic motion under occlusion), not a universal
  counter — using it everywhere overfits (bench 39, jack 27). Its own periodicity confidence tells you when it's
  valid (pommel 0.70 vs bench 0.52).
- **The phase-source gate must be a *measured* signal, never the action name.** `pose-coverage` (the action
  channel's keypoint confidence) cleanly detects pommel's occluded legs (`kp=0.00`) and routes to the fallback.
- **Don't trust `dur/period` or gap-extrema counters** — autocorr locks onto harmonics (counts ~2×). The shipped
  differential counter is harmonic-immune; reuse it.
- **No overfitting discipline held:** every knob (`POSE_CONF_MIN`, `KP_CONF_MIN`, silhouette gap = 0.6×period,
  PC-by-periodicity) is one principled constant validated across ≥3 clips — not tuned per clip.

## Our current approach

```
candidate sources      pose boxes/keypoints (RTMO)  +  motion-foreground blobs (bg-subtraction)
   → multi-tracklet link   (continuity; detector IDs are NOT identity truth)
   → PERFORMER selection    pick the tracklet with the most rhythmic motion on the action channel
   → phase-source gate      pose channel       if its keypoints are visible (pose-coverage ≥ τ)
                            silhouette-PCA     fallback when the channel is occluded
   → count                  shipped differential counter (pose) · period-gap extrema (silhouette)
```

Scratch impl: `scratch/{performer_select,hero_viterbi,silhouette_pca,generic_motion_hero_tracker}.py`; video at
`scratch/hero_final_viz.html`. Principled: gates by measured pose-coverage + periodicity, one knob set for all 8.

## Scorecard (N=8, performer-aware, scratch)

```
clip      action       gt | pose c/kp/torso          | silh | SOURCE       cnt  err
stu10_5   bench         16 | 16  0.81/0.82/0.05 lying |  39  | pose          16   0   ✓ holds the LIFTER
train143  bench          0 |  4  0.41/0.99/0.08 lying |  16  | pose           4   4   (no amplitude gate here)
stu8_23   front_raise    3 |  3  0.45/0.99/0.37 stand |   3  | pose           3   0   ✓
test538   front_raise    5 |  6  0.82/0.97/0.38 stand |   6  | pose           6   1   ✓
stu8_34   jump_jack      5 |  7  0.43/1.00/0.17 stand |  27  | pose           7   2   counter over
train1665 jump_jack      3 |  7  0.40/0.99/0.20 stand |   4  | silhouette     4   1   ✓
stu4_28   others        10 |  3  0.55/0.99/1.29 stand |  17  | pose           3   7   ✗ bad tracklet (torso 1.29)
stu13_33  pommelhorse   33 |  7  0.71/0.00/0.08 lying |  33  | silhouette    33   0   ✓ occluded → fallback

N=8   OBO 0.62   MAE 0.70   (MAE 0.23 excluding the GT-0 clip)   source mix: pose 6 / silhouette 2
```

**Baselines for comparison:** shipped differential **0.50 / 0.89**; +amplitude-gate (committed) **0.625 / 0.25**.

**Read (blunt): on N=8 this is WORSE than the committed baseline (MAE 0.70 vs 0.25).** Per-clip vs the gated
baseline it **traded two losses for one win**: won **pommel (12→33)** (the occluded case, via the silhouette
fallback); lost **others (10→3)** (a degenerate tracklet, torso-vert 1.29) and **train143 (0→4)** (this path
doesn't run the amplitude gate). Bench is the same number but now on the *correct* body. Folding the committed
amplitude gate in would recover train143; the "others" regression is real. Excluding the GT-0 clip, MAE is 0.23
(~ the gated baseline) — but the path is far heavier (Viterbi + background model + silhouette PCA).

**Verdict:** **do not adopt on N=8 evidence** — it scores worse for more complexity. Its only real gain is
*structural* (right subject under crowd/occlusion), which **N=8 cannot measure** (just 2 hero-relevant clips).
Keep the shipped differential + amplitude gate as the N=8 best. Pursue the hero tracker **only with a
multi-person / occlusion eval set**; if it doesn't beat the baseline there, the weight isn't justified.

### ADDITIVE test (the decisive one) — `scratch/additive_hero.py`

Keep the committed pipeline EXACTLY (differential counter + amplitude gate); change ONLY the hero feeding it
(drifty `hero_follow` → performer-aware tracklet). This isolates the hero's value as a drop-in.

```
                       OBO    MAE
base hero, no gate     0.50   0.92
perf hero, no gate     0.38   0.93   ← WORSE even ungated
base+gate (COMMITTED)  0.50   0.90
perf+gate (ADDITIVE)   0.25   0.96   ← WORSE
```

Per-clip: the performer hero helps **bench (18→16, holds the lifter)** but its selector **regresses others
(10→3, degenerate tracklet)** and is mildly worse on jack/test538/pommel. **The "most-rhythmic tracklet"
selector is not reliable enough to be additive** — it wins the one clean multi-person clip and hurts the rest.
**Conclusion: the hero tracker is not ready even as a drop-in swap; it must beat the baseline on a proper
multi-person eval before it earns its weight.** Fix the selector's degenerate-tracklet failure first.

## Recommended next steps

1. **Combine with the shipped amplitude gate.** The performer path doesn't run `expected_rom`; folding it in
   takes train143 → 0 and trims the jack over-count. (Both already exist; just wire them.)
2. **Debug "others" (3 vs 10).** `torso-vert 1.29` means the selected tracklet is a degenerate pose — the
   performer selection needs a sanity floor (reject implausible body geometry) or better tracklet quality.
3. **jack counter (7 vs 5).** Minor over-count on `ankle-spread`; check the differential prominence/debounce on
   fast reps, or whether amplitude-gating fixes it.
4. **Decide what to land.** The structural win is the **performer-aware hero association + pose-coverage gate**.
   It's L1-validated on N=8 but the aggregate claim needs a **multi-person eval set** (N=8 has few crowds) and a
   **rotational/occluded set** for the silhouette fallback.
5. **Productionization questions** (not yet answered): the Viterbi link + background model + silhouette PCA need
   cv2/numpy and are **offline/global** — decide the home (harness/edge) and the **online approximation**
   (windowed Viterbi / beam search). Replace the histogram embedder with a real re-id embedding if appearance
   re-acquire is wanted.
6. **Stronger appearance** for the dormant `AppearanceIdentity` seam (a learned embedding, not a histogram) if
   we want re-acquire across long gaps in crowds.

## UPDATE — the FULLY-CONNECTED run (committed code, not scratch)

We built the connected version into tracked code and A/B'd it on the same N=8 (all PRIMED, fps=6). The
producer that was missing (commit `1d020f8` shipped the brain half with no eye) is now wired:

- **eye** `PoseChannels.measure_entities` → one `PersonMeasure` per person (box + channels + per-channel
  keypoint coverage) — `percept-harness/detectors.py`.
- **brain** `RhythmHero` (`percept/rhythm_hero.py`) → `EntityGraph`/`AppearanceIdentity` association →
  entity-scoped `perceive_rhythm` (the brain selects the performer). The amplitude gate runs per-entity for
  free (the handoff's "performer path doesn't run the gate" gap is closed).
- **measured phase-source gate** → low prior-channel coverage routes to a `silhouette` fallback
  (`eval/bench/silhouette.py`, lifted out of scratch; offline/static-camera, marked).
- **two-tier `escalate_count`** → low periodicity confidence escalates to a LIVE Gemini recount.
- Replays a cached RTMO detection store (`eval/bench/pose_cache.py`). Runner: `eval/bench/hero_ab.py`.

```
clip       gt |  A   B   C | src         conf  cov  htc ents  vlm      A=committed single-subject
stu10_5    16 | 15  17   7 | pose        0.28 0.82 0.94  84   →7       B=connected edge-only
train143    0 |  4   4   4 | pose        0.41 0.99 1.00   1    -        C=connected + LIVE VLM
stu8_23     3 |  4   4   2 | pose        0.01 0.99 0.76   5   →2
test538     5 |  6   6   6 | pose        0.82 0.97 1.00   4    -
stu8_34     5 |  7   2   1 | pose       -0.11 1.00 0.57   3   →1
train1665   3 |  6   6   3 | pose        0.37 0.99 0.95   2   →3 ✓
stu4_28    10 |  9   6   1 | pose        0.14 1.00 0.44   6   →1
stu13_33   33 |  9  33  33 | silhouette  0.70 0.00 0.98  10    -

A committed            OBO 0.500  MAE 0.853  (noGT0 0.571 / 0.403)
B connected (edge)     OBO 0.500  MAE 0.824  (noGT0 0.571 / 0.371)
C connected + LIVE VLM OBO 0.500  MAE 0.849  (noGT0 0.571 / 0.399)
```

**Findings (the decisive ones):**
- **The two structural mechanisms are real and now reproducible.** Performer selection holds the lifter on
  the crowd clip (stu10_5, avg 3.4 ppl → 17 vs GT 16, on the *right* body); the **measured** coverage gate
  catches pommel's occluded legs (`cov 0.00`) and routes to silhouette → **9→33 exact**. No action-name
  hardcoding anywhere.
- **Edge-only B ≈ committed A on N=8** (MAE 0.824 vs 0.853): the pommel win is cancelled by hero
  **fragmentation on jack (7→2) and others (9→6)** — low `htc` (0.57, 0.44). Smoking gun: **84 entities on a
  ~3–9-person clip** → the HSV-histogram embedder is the limiter, exactly as predicted. The wiring is sound.
- **The live VLM backstop is a NET NEGATIVE (C 0.849 > B 0.824).** The windowed Gemini recount systematically
  **undercounts** (stu10_5 17→7, stu4_28 6→1, stu8_34 2→1); it only helped train1665 (6→3). And the noisy
  autocorr confidence escalated a clip the edge already had right (stu10_5, conf 0.28) and broke it. The
  fidget (train143) is **un-escalatable by periodicity** (it's periodic, conf 0.41) — O8 confirmed live.

**Verdict:** do not ship the hero or VLM-escalation as configured — N=8 can't convert the structural wins to
an aggregate win, and the VLM recount + autocorr gate hurt. Three *separable* fixes the data points at: (1)
the **embedder** (HSV → learned re-id; it's what fragments jack/others), (2) the **escalation confidence**
(autocorr is noisy → use `regularity` / diff-vs-autocorr agreement), (3) the **VLM counting protocol**
(windowed-direct undercounts → sliding-completion or a denser prompt). And the aggregate hero claim still
needs a real multi-person eval.

## UPDATE 2 — the continuity fix: predictive (motion) continuity in the EntityGraph

Probing *why* the hero fragments showed the root cause is **continuity, not identification**: the matcher
([identity.py](../../packages/percept-vision/percept/identity.py)) is **memoryless IoU** (compares each
detection to the entity's *stale last box*, threshold 0.5) — there was no velocity/prediction anywhere in
core. A subject that displaces faster than its own box-overlap breaks → new id → the weak HSV re-id can't
re-acquire → fragment. Read-only analysis: jack's breaks are *mild* (24% of frames dip below IoU 0.5, **0%
below 0.3**) → recoverable with light motion memory.

Isolated in scratch (`scratch/hero_lab.py`, ablating appearance-only / predict-only / predict+appearance):
prediction lifts single-person hold to **htc ~1.0** (jack 0.39→1.0) but pure prediction **breaks crowds**
(needs appearance), and the scratch reimpl diverged from core (couldn't reproduce bench) — so we integrated
into the real `EntityGraph` behind an opt-in `motion` flag and A/B'd on the real harness.

**Key design correction:** prediction must **AUGMENT, not replace**. The first integration pre-advanced every
box before matching → it disrupted the appearance match holding the lifter and **regressed bench (17→10, htc
0.94→0.46)**. The fix: keep normal continuity+appearance as the primary match; only when it *fails* try the
**predicted box** (constant velocity, coasting within `coast_s`). That recovers the fast-motion break while
leaving a held crowd subject untouched.

```
              motion OFF        motion ON (augment)
bench   16:   17 / htc 0.94  →  17 / htc 0.94   preserved (no crowd regression)
others  10:    6 / htc 0.44  →   8 / htc 0.67   closer
jack     5:    2             →   3              closer
stu8_23  3:    htc 0.76      →   htc 0.85

connected hero (Arm B): MAE 0.824 → 0.774   (noGT0 0.371 → 0.314)   OBO 0.500 unchanged, zero regressions
```

**Landed (opt-in, default off):** `Entity.vel`/`anchor_box`/`predicted_box` + `EntityGraph(motion=, vel_alpha=,
predict_iou=, coast_s=)` with a `_predict_match` recovery path
([entity.py](../../packages/percept-vision/percept/entity.py)); enabled in the bench via
`PERCEPT_HERO_MOTION=1`. Tests: `test_entity_motion.py` (a fast box memoryless loses but motion holds across a
dropped frame; coast bound stops it grabbing a passer-by). Full suite green (504 + 130). It's a **clean,
regression-free improvement** for *every* sense (continuity is shared), not just rhythm. **Residual:** jack
(3 vs 5) / others (8 vs 10) still undercount — closing them needs a stronger re-id embedding or optical-flow
coast (the B/C slots), which are diminishing returns on N=8.

*Scratch evidence is gitignored; this doc is the durable record. The runs above ARE committed (tracked eval +
core code); only the cache/clip data is gitignored. See [architecture.md](../architecture.md) and
[rhythm-sense-and-the-hero.md](rhythm-sense-and-the-hero.md).*
