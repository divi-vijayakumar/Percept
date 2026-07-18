# Percept MVP — implementation plan

> H0 is viable. This plan productizes the proven runtime and, critically, makes the H0 optimization loop a
> *permanent* part of the product so learnings flow in continuously — not a one-time handoff.

## 1. Where H0 leaves us

H0 proved the runtime end to end on real video (charades-e2e, N=20): the fovea (tier-0 salience gate) →
judge (VLM) → rising-edge gate path detects persistent watch conditions, and the recall bar is **clearable**
at the operating point.

Ablation trajectory (operating point, ≤3 distinct goals, qwen3-vl-32b judge):

| stack | recall | precision | edge recall |
|---|---|---|---|
| baseline (mean · single frame) | 0.39 | 0.42 | 0.68 |
| + block_max (localized eye) | 0.68 | 0.40 | 1.00 |
| + ordered window (temporal input) | 0.73 | 0.45 | 1.00 |
| + gate-relax (**full stack**) | **0.90** | 0.42 | 1.00 |

**What this establishes:**
- **The eye is the biggest lever** — localized salience (`block_max`), not the model. The judge model is a
  cost/latency choice (qwen ≈ gemini on quality), not the recall lever.
- **The temporal input is an ordered window, not a montage** — sprites regressed at every density.
- **Recall clears 0.90** with the full stack; **precision (~0.42) is the open axis** — partly re-fires and
  over-detection, partly Charades' sparse labels penalizing correct-but-unlabeled detections.
- Every remaining gap has a **named, addressable cause**. Nothing is mysterious.

## 2. The thesis: quality is Q(p | environment, goal, state)

Precision and recall are **not one fixed number**. They are a function of the deployment context —
`Q(p | environment, goal, state)`:
- **environment** — camera mount (fixed vs handheld), scene motion, lighting → sets the *eye* (salience mode,
  threshold, ego-compensation).
- **goal** — what's being watched, its temporal character (stative vs dynamic), its authoring quality → sets
  the *judge input* (single frame vs window vs episode memory) and the goal density.
- **state** — already-true-at-start vs transition, refractory history → sets the *gate*.

H0's knobs — `salience_stat`, threshold, `plan_mode`, cadence, gate config, goal density — are precisely the
free variables of Q. **The MVP's core abstraction is a `WatchProfile`: a named binding of those knobs to an
(environment, goal-type, state) context.** Productizing the MVP = making these knobs first-class,
profile-selected, and continuously re-tunable — never hardcoded in runtime logic.

## 3. Bringing H0 learnings into the MVP — continuously

**Yes — and by construction, not by copying.** H0 and the MVP already share the *same config surface*:
`salience_stat`, `stat_thresholds`, `plan_mode`, `SensePolicy` cadence, `require_false_first`, goal density
are the same objects in `percept/config.py`, `compile_watchspec`, and the judge path. H0 *tunes* those knobs
on fixtures; the MVP *consumes* them via `WatchProfile`s.

So a new H0 finding is a **versioned profile/config delta**, loaded by the runtime with no code change:

```
  real stream telemetry ─┐                         ┌─► WatchProfile vN (versioned config artifact)
   (edge/verdict/gate     │   H0 eval harness       │      salience_stat, threshold, plan_mode,
    traces → fixtures)    ├─►  (ablation ladder,  ──┤      cadence, gate, goal-density guidance
                          │    per-environment)     │
  new environment fixtures┘                         └─► MVP runtime loads it per (env, goal, state)
```

The commitment that makes this work (a hard architectural rule): **the runtime reads its behavior from a
profile; the eval harness writes profiles. No knob is hardcoded in firing logic.** A learning becomes a new
default or a new profile, shipped like any config — same day it's found.

Concrete loop: onboard a new environment → capture/label a small fixture set → run the shared ablation job →
it emits a tuned `WatchProfile` → the runtime picks it up. The precision axis is optimized the same way,
per-environment, on the same rails.

## 4. Workstreams

### E1 — Runtime GA (the firing path to production)
- Freeze the public contract: `watch(stream, goals, profile?) → stream<Event{fire|ask|silent, …}>`.
- Bake H0 defaults behind profiles (block_max for fixed cameras, window for dynamic goals, calibrated
  `stat_thresholds`, gate config). No behavior hardcoded — all via the default profile.
- Resilience to GA: per-attempt timeout (done), circuit breaker (done), backpressure, 24h soak, degrade-loud
  health events.
- SLOs from H0: reflex latency p95, `$/stream-hour`, call-reduction floor — asserted in CI.

### E2 — WatchProfiles: the Q(p | env, goal, state) surface  ← the core MVP abstraction
- `WatchProfile` schema + registry: `{salience_stat, stat_thresholds, plan_mode, cadence{base,max,escalate},
  gate{require_false_first, refractory_s}, goal_density_guidance}` keyed by (environment, goal-type, state).
- Resolution rules: goal temporality → plan_mode (stative→frame, dynamic→window); camera type → salience;
  start-anchored state → gate. Wire the authoring guide's classifications in.
- Ship the H0-derived defaults as the first profiles (`fixed-cam-localized`, `dynamic-action`, `handheld`).
- Versioning: profiles are content-addressed + provenance-stamped (which eval run produced them).

### E3 — Judge / model layer
- Swappable judge behind one seam (qwen3-vl-32b default = cheap/hosted/parallel; gemini/claude optional);
  token/cost metering; typed fallback.
- Temporal judge inputs: **ordered window** (shipped H0 win); **deprecate sprite**; build the
  **episode/entity-memory** judge (process-QA running-answer) — the deferred H0 arm, the next precision/recall
  lever.

### E4 — Continuous-learning bridge (H0 → MVP)  ← answers "learnings as they come"
- Promote the ablation ladder into a reusable **profile-derivation job**: (environment fixtures, bars) → tuned
  `WatchProfile` + a scorecard. Same harness H0 used.
- Telemetry → fixtures: production traces (edge/verdict/gate/reason) captured; real misses & false-fires become
  new eval cases (closed loop).
- **Precision track** (the open axis): time-refractory for re-fires, over-detection tuning, label-completeness
  accounting — run as its own pre-registered optimization on the shared harness.
- Keep pre-registration / no-tune-to-green discipline as the profile-acceptance gate.

### E5 — Observability & guardrails
- The H0 forensic trace (per-frame edge decision, verdict, gate action + reason) becomes **production
  telemetry** and the debugging surface.
- Consequence gate (FIRE/ASK/SILENT); health events; the eval CI gate blocking regressions.

### E6 — Product surface / DX
- The SDK front door: `watch()` + goal authoring (the how-to guide) as the primary DX.
- Notification delivery, quickstart, profile selection UX.

## 5. Sequencing

E1 (runtime GA) and E2 (profiles) are the foundation and run first — E2 depends on E1's contract. E4 (the
bridge) starts as soon as E2's profile schema exists, because it *writes* profiles. E3's episode-memory and
E4's precision track are the first learnings to flow through the bridge (proving the loop). E5 threads
throughout; E6 gates on E1+E2 being stable.

## 6. Risks
- **Profile sprawl** — too many hand-tuned profiles. Mitigation: derive, don't hand-write; version + provenance.
- **Precision on sparse-label evals** — don't over-fit to Charades' label gaps. Mitigation: real-environment
  fixtures + label-completeness accounting in the precision track.
- **Hardcoding creep** — a knob sneaks into firing logic and breaks the continuous loop. Mitigation: the
  runtime-reads / harness-writes rule is a review checkpoint.
