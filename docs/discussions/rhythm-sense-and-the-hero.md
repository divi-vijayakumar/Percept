# Design note — the rhythm sense & the hero

**Status:** proposal, for review. Describes shipped state, what we learned probing it, and the proposed
next changes. *Not* an adopter doc — see [architecture.md](../architecture.md) for what ships today.

**Scope:** repetition counting (the `rhythm` sense) and the subject-tracking ("hero") it sits on.

---

## 1. Where this fits in the architecture

Rep counting runs on the **edge-only, VLM-free path** (see [architecture.md](../architecture.md) §"two firing
paths"): the eye *measures* decomposed pose channels, the brain *counts* causally, no model call. Concretely:

```
feed → pose detector (eye) → channels → perceive_rhythm → RhythmTracker → gate → incident
```

The whole proposal touches only the **measures layer** (which channels, which subject) and the **prior** (what
a real rep looks like) — never the counter or the gate. The counting algorithm is settled (§3, O1).

---

## 2. The problem

A rhythmic action produces a roughly periodic pose signal; we want the **rep count**, and we want to **not
count motion that isn't a rep**. Shipped today (`percept/rhythm.py`, committed): a causal differential counter
(trough→peak→trough, time-debounced, prominence-guarded). On RepCount-A (N=8): **OBO 0.50 / MAE ≈ 0.89**.

The open question is the residual error — and, more importantly, *which* interventions are real vs artifacts.

---

## 3. Observations (what the probing actually showed)

Each is grounded in a scratch experiment; evidence in parentheses.

- **O1 — the differential counter is sufficient for any rhythm.** No phase-portrait / RepNet / continuous-motion
  machinery is needed; that whole branch is closed. (Pommel, the supposed "continuous" hard case, counts with
  the plain differential once the subject + channel are right.)
- **O2 — differential vs autocorrelation are complementary, and the differential is the better default.**
  Differential counts events (harmonic-immune, causal, emits spans) but undercounts amplitude-modulated motion;
  autocorr measures a period (amplitude-robust) but is fooled by harmonics and is batch. Head-to-head N=8: the
  differential wins 3/4; autocorr's one win is pommel. **Their disagreement is a usable confidence signal.**
- **O3 — the "perception wall" was a tracking artifact, not a perception limit.** Pommel's failure was the
  tracker drifting to the audience, not occlusion or motion difficulty. (With the gymnast held, the legs are
  visible 81%, not 30%.)
- **O4 — the residual decomposes into four independent failure modes / levers**, each a separate fix:
  | lever | failure mode (dimension) | example |
  |---|---|---|
  | **A — hero** | wrong subject (*who*) | bench tracks a bystander |
  | **B — horizontal channels** | wrong DOF for rotation (*which channel*) | pommel legs circle, don't spread |
  | **C — amplitude gate** | counting non-reps (*is it a rep*) | a fidget counts as reps |
  | D — joint confidence | noisy joints (*which joints*) | deprioritized (not a clear win) |
- **O5 — geometric hero-tracking drifts; it is a HOLD failure, not a detection or seed failure.** The subject
  *is* detected (bench: lying lifter present in 88% of frames) and the seed picks them, but nearest-hip
  continuity teleports to bystanders (13 identity-jumps on bench) and never returns, so the average hero is a
  standing spotter. The "limb-vertical" signal it then measures is the spotter's arm gestures. (Same drift on
  pommel: signal flatlines mid-routine.)
- **O6 — the amplitude gate (C) is the only lever validated on N=8, but its headline is narrow.** Full ablation:
  every config *with* C → MAE ≈ 0.25; every config *without* → MAE ≈ 0.91–0.94. **But ~70% of that MAE win is a
  single clip** (train143, GT-0, where one false count dominates the relative metric). Stripped of that clip, C
  is roughly a wash: it trims real spurious overcounts (bench 25→20, jack 6→5) **and** over-trims a
  variable-amplitude action (others 10→7). **There is no universal amplitude threshold** — the fidget (1.04
  torso-normalized) is *larger* than legitimate jack (0.35) and pommel (0.26) reps; the expected amplitude must
  be action-specific.
- **O7 — A and B are not disproven, they are *unprovable on N=8*.** A only matters in crowds (one clean
  multi-person clip), B only matters for rotation (one clip), and B is *gated on a strong hero that A doesn't
  yet deliver*. On N=8 their aggregate effect is within noise; and where A appeared to help (bench 25→18) it was
  tracking the wrong person (O5). **We have zero validated evidence that hero-following works.**
- **O8 — the negative case is separable only by absolute amplitude.** train143 (GT-0) is signal-identical to a
  real clip on magnitude-normalized, confidence, and periodicity metrics. In *absolute* frame terms the fidget
  wrist travels 0.085 vs a real bench's 0.275 (3.2×). Torso-normalization (added for cross-person comparison)
  inflates the close-up fidget to 1.04 and hides this. So amplitude *can* separate it — but only against an
  action-specific expectation.

---

## 4. Hypotheses

- **H1 (hero/re-ID).** Replacing nearest-hip continuity with **appearance-gated re-ID** (hold the subject by an
  appearance descriptor, refuse to switch to a non-matching candidate) will *hold* the subject where geometry
  drifts — and is the prerequisite the other levers wait on. *This is the foundation; A threads across the
  measures layer and unblocks B.*
- **H2 (horizontal channels).** With a *held* hero + an **absolute horizontal channel** (`ankle.x`), pommel-style
  rotational motion becomes countable (~28 vs GT 33) — far better than the spread channel (~13). *B is real but
  only realizable on top of H1.*
- **H3 (amplitude gate).** An **action-specific expected-ROM prior** + a gate that rejects cycles below a
  fraction of it will suppress the fidget false-positive (and trim small spurious overcounts) **without clipping
  genuinely small reps** — *provided* the expected value is action-specific. The residual (big, periodic,
  wrong-action movement) is rejected only by the VLM backstop. Two-tier: amplitude gate → semantic confirm.
- **H4 (confidence).** The differential-vs-autocorr disagreement (O2) yields a calibrated per-count confidence;
  low-confidence counts escalate to the VLM rather than being reported as clean.

### Open question for review — where does the amplitude prior come from?

Three constraints make this non-trivial (O6/O8): the VLM can't give a number in our channel units, a global
floor is impossible, and the expectation is action-specific. **Proposed:** the goal-setting VLM emits a *coarse
category* — best phrased as **"fraction of full range of motion"** (a bench press = full extension; a fidget = a
sliver) — and a **calibration table** maps {channel, category} → a numeric expected amplitude. This is a
"pre-trained notion," but *calibrated from data, not guessed* — which is the distinction that defuses the
brittleness concern. **Reviewers: is the VLM-category + calibration split acceptable, or do we want the prior to
stay relative (no absolute number) and accept that the all-fidget negative needs the VLM?**

---

## 5. Approach (proposed changes, ordered by dependency)

1. **A — hero in the watchspec + appearance-gated tracker.** Add a `subject`/hero field to the watchspec; the
   measures layer holds it via an appearance descriptor (colour histogram / embedding of the person crop) +
   motion, associating detections by *both* and refusing low-match switches; re-acquire by appearance after
   gaps. Seed from a brain/VLM subject prior, not central+confident geometry. *This is the standard MOT/DeepSORT
   shape; today's tracker is single-track greedy nearest-neighbour with no appearance.* Threads across layers
   (re-ID is reusable beyond rhythm).
2. **B — horizontal channels.** Add absolute `ankle.x` / `wrist.x` channels to the `pose_channels` basis (hero
   hold makes absolute position usable — do *not* subtract the hip, which cancels the rotational swing). Expand
   the channel-prior vocabulary so the VLM can select them.
3. **C — expected-ROM prior + amplitude gate.** Add `expected_rom` (or a `rom_category`) to the rhythm prior;
   gate `_ChannelState` so a cycle counts only if `rep.amplitude ≥ frac × expected`. `rep.amplitude` already
   exists — this is small.
4. **H4 — confidence cross-check.** Run autocorr alongside the differential on the selected channel; emit a
   confidence from their agreement; route low-confidence counts to the VLM backstop.

Each lands in a clear place: A/B in `percept-harness` (measures), C/H4 in `percept-vision` (cognition/prior).
None touch the gate.

---

## 6. What we are testing (and the honest gates)

The N=8 ablation already ran; its limitation is the headline (O7) — it cannot test A or B. So the test plan is
as much about **building the right eval sets** as running them.

| hypothesis | test | pass criterion | data we have / need |
|---|---|---|---|
| **H1 hero** | appearance-gated tracker on bench + pommel; measure whether it *holds* the subject | bench hero stays "lying" (torso-vert ≈ 0.02) and identity-jumps → ~0; pommel signal does not flatline | **have** (frames) for the unit test; **need** a multi-person eval set for the aggregate claim |
| **H2 channels** | A+B on pommel-style clips, hero held | pommel ≈ GT without per-clip tuning; no regression on single-person clips | **need** several rotational clips (cycling/rowing/pommel) with GT — one clip = overfit risk |
| **H3 amplitude** | C with a calibrated prior across N=8 **+ a negatives set** | train143 → 0; jack/pommel/others reps preserved (the false-negative trap) | **need** a *negatives* set (fidgets, warmups, wrong-action) with GT 0 |
| **H4 confidence** | correlate the diff-vs-autocorr gap with actual error | high-gap counts are the wrong ones (calibration) | **have** (N=8) for a first read |

**Acceptance is structural, not just a number.** Because the N=8 MAE is dominated by one GT-0 clip (O6), we
report metrics **with and without** train143, and **per-clip**, never a single aggregate. A lever "passes" only
if it improves the *mechanism* visible in the per-clip video view (`scratch/ablation_viz.py`: the hero overlay,
the channel signal, the kept/gated rep dots), not just the score.

---

## 7. Risks & constraints

- **Overfitting (the standing constraint).** Every lever must pass *parameter-free or with a single principled
  constant*, judged across multiple clips — never tuned to one clip (pommel's "28" required a hand-tuned
  central-lock and is treated as overfit, not a result). The amplitude `frac` and the calibration table are the
  two knobs to watch.
- **Metric artifact.** The GT-0 clip dominates MAE; do not let a lever "win" by gaming it. Hence with/without
  reporting.
- **Pommel may stay hard.** The generalizable tools (held hero + horizontal channel) reach ~12–28 depending on
  hero quality; closing it fully needs appearance re-ID, not channel tuning. Acceptable to mark it a known-hard
  case rather than overfit.
- **Prior sourcing (§4 open question).** The amplitude lever's value depends on the calibration story being
  acceptable; if not, C degrades to "trim spurious overcounts" + VLM-for-the-rest.

---

## 8. Status summary

| piece | status |
|---|---|
| differential counter + concern/episode tie-in | **shipped** (core, tested) |
| O1–O8 observations | **validated** (scratch) |
| C — amplitude gate | **validated narrow** (N=8, dominated by one clip) → propose to land with negatives-set test |
| A — hero/re-ID | **unvalidated** (geometric version drifts; appearance version not built) → the foundation, build next |
| B — horizontal channels | **blocked on A**; channel definition known (absolute, not hip-relative) |
| H4 — confidence | **scratch idea**, first read available |

**Recommended order:** build **A (appearance re-ID)** first — it's the foundation, it unblocks B, and it's the
one thing we have *no* validated evidence for — then re-run the ablation with a held hero, then land C with a
proper negatives set.

---

*Scratch evidence: `scratch/{rhythm_capture,e2e_layers,hero_mask,ablation,ablation_viz,video_viz}.py` and the
`*.html` viewers. Confirmed architecture: `docs/architecture.md`.*
