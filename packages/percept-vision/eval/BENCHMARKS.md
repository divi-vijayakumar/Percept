# Percept — Benchmark Findings

**Version 2 · 2026-06-21** · maintained alongside the `eval/exp_*.py` runners. This is a living
document: **§1 is the standing Eval Protocol — follow it for every new run**; append a dated entry to the
Changelog on every material result, and update *What's not working* as ceilings move. The goal is that
anyone (incl. future us) can see, at a glance, **what the cognition layer is worth, where it isn't, what's
unmeasured, and what to fix next.**

---

## 1. Eval Protocol — the standing plan (READ FIRST; every run follows this)

This encodes what we learned about apples-to-apples comparison, the two different "latencies," and which
architectural layer is actually under test. **Any new test / eval / benchmark must follow 1a–1g.**

### 1a. Two comparison types — never conflate
- **Controlled A/B (the trustworthy claim):** Percept vs a raw-VLM-loop with **perception held constant**
  (one verdict/frame, captured once and *shared*). Only the cognition layer varies → the delta is
  attributable to it. Deltas (positive, neutral, **and zero**) are real. **Lead with this.**
- **Reference bar (context only):** vs an external leaderboard (TML, OmniPro models). **NOT** equal-conditions —
  favorable subsets, our perception model, our metric impl, judge mismatch, closed competitor models. Never
  state as a leaderboard "win." (e.g. TML's model + harness are closed → not reproducible; OmniPro we score
  on the visual slice with their MIT scorer but Gemini-Flash ≠ Gemini-3-Flash perception.)

### 1b. Metric taxonomy — report **Core** and **User-facing** in SEPARATE columns
| Class | Metrics | Unit | On leaderboards? | Status |
|---|---|---|---|---|
| **Core** (accuracy, video-time) | time_F1, PAUC, mIoU, precision/recall, content/joint, off-by-one | video-sec / fraction | yes | **measured** |
| **User-facing** (system, wall-clock) | event→alert latency, VLM call-rate, $/hr, interruption rate | wall-clock ms / $ | no | **NOT YET measured** |

A User-facing number requires a real-time run; if you don't have one, write **"not measured"** — never imply it.

### 1c. The two "latencies" — say WHICH one, every time
- **Temporal accuracy (video-time):** *where in the stream* you respond, scored ±tolerance (OmniPro time_F1
  ±3s; our PAUC). **Compute-agnostic** — leaderboards run **offline, frame-by-frame**; a slow model isn't
  penalized (the emit carries a video timestamp). We measure this; it's apples-to-apples.
- **Wall-clock / voice-to-voice:** *event → response in real ms* (TML's "200ms"). **Not on any leaderboard**;
  each lab self-reports. Percept's = edge gate (ms) + **VLM round-trip (dominant, ~0.5–2s)** + core gate (ms)
  → a **~1–2s event-to-alert watcher** with a frontier VLM, **not** a 200ms native model. Different operating
  point — state it.
- **RULE: never compare a video-time number to a wall-clock number** (our time_F1 vs TML's 200ms = invalid).

### 1d. Layer-coverage declaration — every run states what it exercised
Declare ✓/✗ for **Edge** (Tier0 motion/salience gate) · **Harness** (real-time loop, edge↔core) · **Core**
(gate/dedup/episodes) · **Perception** (VLM) · **Latency** (wall-clock timed?).
**All runs to date: Core ✓ · Perception ✓ · Edge ✗ · Harness ✗ · Latency ✗** (offline replay, uniform frame
sampling — the edge gate was bypassed; nothing timed). Accuracy numbers therefore assume *every sampled frame
gets a VLM call* (an upper bound) and say nothing about edge-gated latency/cost. **A real-time edge→harness→core
run is required to fill the User-facing column.**

### 1e. ASK is a *fire*, not a third state — eval it on the right shape
The gate is **binary: silent vs engage.** FIRE and ASK are both *engagements*; **ASK = fire-as-a-question**
(probing the user to resolve uncertainty). So the gate's firing discipline (timing/refractory) applies to ASK
identically — ASK-vs-FIRE is a downstream *content* decision. ASK's value (ambiguous conditions, asymmetric
error cost) is **invisible to clean-label benchmarks** (an ASK = time-hit, content-miss). **Do NOT eval ASK on
Charades/OmniPro/RepCount/PAUC.** Eval it on **ambiguity-aware** sets (multi-annotator disagreement —
RethinkingVAD / SmartHome-Bench) or as **selective prediction** (risk–coverage), as a **product/trust** metric.

### 1f. Claims discipline
1. Label which latency (video-time vs wall-clock); qualify "real-time" (**causal-streaming** ✓ vs **low-wall-clock** ✗).
2. Never compare across axes (Core vs User-facing; video-time vs wall-clock).
3. Lead with the controlled A/B; mark external comparisons as reference bars.
4. State the operating point (cognition-on-VLM ~1–2s vs native ~200ms).
5. Flag every favorable subset, perception-model substitution, and judge mismatch.

### 1g. Standing checklist for any new run
☐ Controlled A/B (perception constant) + reference bar if external · ☐ Layer-coverage declaration (1d) ·
☐ Core metrics (separate column) · ☐ User-facing metrics IF real-time, else "not measured" · ☐ Caveats
(subset / perception model / metric impl / judge) · ☐ Tag each number's axis — never cross.

Perception model for the current suite: **Gemini 2.5 Flash** (held constant). ~$0.30/$2.50 per MTok; whole
suite to date ~a few dollars. Re-scoring is free — every runner caches per-frame verdicts to a gitignored
JSON, so sweeping policy/threshold/ω costs $0 after capture.

---

## 2. Results — Gemini Flash perception, Percept vs Gemini-alone vs TML

| Benchmark (TML metric) | Gemini alone | **Percept + Gemini** | Δ (controlled A/B) | TML (ref) | runner |
|---|---|---|---|---|---|
| **Charades** — mIoU ×100 | 36.1 | **60.3** | **+24.3** ✅ | 32.4 | `exp_charades_miou.py` |
| **ProactiveVideoQA** — PAUC@0.5 | 44.6 *(321 interrupts)* | 33.0 *(91)* | parity, **4× less spam** ⚠️ | 33.5 | `exp_proactiveqa.py` |
| **RepCount** — off-by-one ×100 | 15.0 | 15.0 | **0** ❌ | 35.4 | `exp_repcount.py` |

Notes: GPT-realtime-2.0 (TML's baseline) = 0 on Charades, 25.0 no-response floor on PAUC (our port
reproduces 25.0 exactly), 1.3 competitors on RepCount. ProactiveVideoQA also: PAUC@0 loop 39.2/percept
30.9; PAUC@1 loop 50.0/percept 35.0.

### 2a. Earlier controlled e2e (interval scoring, replay-and-align)
- **Charades (Haiku perception), interval precision:** baseline 325 fires / prec 0.08 / recall 0.81 →
  percept-event 44 fires / prec **0.45** / recall 0.65 → **5.9× precision**, count-error 302→21 (14×).
- **Perception Test (Gemini), interval precision:** baseline 269 fires / prec 0.09 → percept-event prec
  **0.21** (2.3×), count-error 242→77. Ceiling = state-vs-event mismatch ("is pouring" true the whole
  action; GT is the onset).

---

## 3. The unifying principle (the real deliverable)

Across three controlled A/Bs, the cognition layer's value has a sharp, **predictable boundary**:

> **The gate / event-dedup helps when the raw per-frame signal OVER-FIRES** — fragmented detections
> (Charades → +24 mIoU, 5.9× interval precision) or answer-spam (ProactiveVideoQA → 4× fewer
> interruptions). **It does NOT help when the raw signal has the WRONG COUNT or CONTENT** (RepCount →
> Δ0) — that is a *perception ceiling*, the same wall as the wake-up / state-vs-event findings.

Cognition removes *redundancy*; it cannot manufacture a signal the eyes got wrong.

---

## 4. The trust investigation (why the gate's ASK valve was dead, and the two failure modes)

- **Scorecard (Haiku @0.7, 25 cases):** false-fire 3/10, fire-recall 9/10, **ASK 0/5** (valve never
  engaged), 64% agreement.
- **Calibration sweep — NEGATIVE/important:** zeroing false-fire needs threshold 0.96, which collapses
  recall to ~0. Model numeric confidence is **not discriminative** (wrong-confident ≈ right-confident);
  across 405 frames only 1 fell below 0.70. Threshold is the **wrong lever**. (Confirms spec §6.1.)
- **Two failure modes, two fixes:**
  1. *Confident-wrong on AMBIGUOUS conditions* ("is this suspicious?") → a **categorical {yes/no/unsure}**
     judge revives ASK (0/5 → **3/5**); false-fire 3→1. Categorical ambiguity is a usable uncertainty
     signal where numeric confidence is not.
  2. *Confident-wrong on CLEAR conditions* ("is someone pouring" when not) → categorical does **not** help
     (the model isn't uncertain, just wrong) → needs **grounding (ROI+exemplar)** or a **stronger model**.
- **Model A/B (Haiku/Opus/Gemini, 53 PT specific cases):** haiku ff 8/18 @conf 0.87 (confidently wrong);
  opus ff 0/18 @conf 0.83 (calibrated, asks 7); gemini-flash ff 0/18 @conf 0.96 (confidently right).
  Overconfidence is largely a **Haiku-capacity** issue, not universal. Both Opus and Gemini-flash
  eliminate clear-condition false-fire; Gemini at flash cost → candidate vision backend.

---

## 5. What's NOT working (open problems — fix these)

| Problem | Evidence | Lever to try |
|---|---|---|
| **Perception ceiling on clear conditions** | RepCount wrong-count (Δ0); PT clear-condition false-fires | stronger model (Opus / Gemini-3); grounded VQA (ROI + deictic exemplar) — both UNTESTED e2e |
| **Counting = undersampling + unreliable phase** | RepCount short clips ~2 frames/rep (sub-Nyquist); phase doesn't toggle once/rep | denser sampling; hysteresis/peak-detect counter; ask-VLM-to-count baseline |
| **Real-time latency UNMEASURED — edge + harness never run** | all runs are offline replay w/ uniform sampling; nothing timed; User-facing column empty (§1d) | wire edge→harness→core, log event→alert wall-clock + VLM call-rate on a few clips (the missing User-facing axis) |
| **ASK mis-scoped, not "broken"** — it's a *fire*-shaped product/trust lever (§1e) | 1/405 frames<0.70 only on *clean-label* cases where ASK shouldn't engage; categorical revives it 0/5→3/5 on the *ambiguous* cases | eval ASK on ambiguity-aware / selective-prediction sets, NOT clean-label benchmarks; productionize categorical mode for the product |
| **State transitions invisible to per-frame judge** | wake-up case; `detect_events` is a stub returning `[]` | wire a real event detector for the Phase-5 EVENT track |
| **Data: subsets + favorable selection** | Charades clean-verb pick; 31 intervals / 15 q / 20 videos | scale to full sets; stratified (not cherry-picked) sampling |
| **Metric caveats** | our Charades mIoU is lenient (no spurious-span penalty → flatters raw loop) | precision-aware mIoU (would *widen* the +24) |
| **vs-TML not equal-conditions** | Haiku≠GPT-4.1 judge; subsets; closed TML model | full sets + official metric code + GPT-4.1 judge for a true head-to-head |
| **Golden fixture set is YouTube/unreviewed** | 48 scenes UNCLEARED YouTube; 180/205 gate cases unreviewed | re-source off CC-licensed sets; self-record the egocentric trust set |

---

## 6. Reproduce

All gated behind `PERCEPT_LIVE=1` + a key. Use framework Python 3.12 (bare `python3` is Xcode 3.9 and
breaks on `X|None`): `/Library/Frameworks/Python.framework/Versions/3.12/bin/python3`.

```
PERCEPT_LIVE=1 PYTHONPATH=packages/percept-vision <py3.12> -m eval.exp_charades_miou   # Charades mIoU
PERCEPT_LIVE=1 PYTHONPATH=packages/percept-vision <py3.12> -m eval.exp_proactiveqa      # PAUC
PERCEPT_LIVE=1 PYTHONPATH=packages/percept-vision <py3.12> -m eval.exp_repcount         # RepCount OBO
```

Data sources (all gitignored): Charades (remotezip subset, non-commercial), ProactiveVideoQA EGO
(remotezip from HF `wangyueqian/ProactiveVideoQA`), RepCount (ungated HF `lmms-lab-eval/repcounta-lance`,
`video_blob` read via `pylance` `ds.take_blobs(...)`). Set `PERCEPT_RECAPTURE=1` to re-run perception;
omit it to re-score the cached capture for free.

---

## 8. External benchmarks worth running (real leaderboards, not reference bars)

- **OmniPro** — *A Comprehensive Benchmark for Omni-Proactive Streaming Video Understanding*
  (arXiv 2605.18577). Task: "autonomously decide **when** to speak and **what** to say from continuous
  audio-visual streams" (9 sub-tasks / 3 cognitive levels). **Probe Mode** = accuracy 0-100 (easy);
  **Online Mode = F1 @ ±3s tolerance** (the real streaming test). 11-model field. **The whole field is
  weak:** Gemini-3-Flash 40.4% probe (top); best Online F1 ~20.9% (MiniCPM-o 4.5), most single-digit
  (LiveStar 3.6%). Qwen3-Omni 22.6% (top open). **Two collapse axes:** (a) **long-term triggers** —
  ~37% retention vs short-term; Online Mode craters 29.1%→0.3% at 180s+; (b) **visual+non-speech-sound**
  (15-22%, shared bottleneck). Pure-visual best 31.1% (Qwen3-Omni).
  - **Why it's the most aligned external benchmark for Percept:** the #1 failure axis (long-term
    collapse) **is Percept's core thesis** — persistent state/episodes vs a forgetful raw-VLM loop;
    OmniPro *measures the gap we claim to close*. Online-Mode F1@±3s = the gate's firing-timing job
    (the over-firing our layer fixed on Charades/ProactiveVideoQA). PUBLISHED (unlike closed TML) → a
    REAL leaderboard, not a reference bar.
  - **Scope honesty:** OmniPro is omni-MODAL; the audio axes are out of scope for the vision cognition
    layer (edge has VAD/Deepgram but no wired audio perception). **Target the VISUAL + LONG-TERM
    sub-axes only.** Data access UNCHECKED — first step before any run.

## 9. OmniPro — RAN on the visual slice with OmniPro's OWN scorer (the first real-leaderboard result)

`exp_omnipro.py` imports OmniPro's official `metrics.online.scorer` (MIT) — greedy 1-to-1 emit↔trigger
matching @±3s → `time_F1` + `joint_F1`. **Scope (honest):** OmniPro is ~84% audio-dependent; Percept is
vision-only, so we ran ONLY the `audio_dependency=none` slice (20 samples / 72 triggers / 8 tasks = the
easy ~16%). Gemini 2.5 Flash perception held constant; Percept (fire-once) vs Gemini-loop (fire-every-frame).

| system | emits (TP/FP) | time_F1 | joint_F1 |
|---|---|---|---|
| gemini-loop | 50/277 | 0.251 | 0.030 |
| **percept** | 39/64 | **0.446** | 0.057 |

- **TIMING (the gate's competency): cognition nearly 2× time_F1 (25.1→44.6, +19.5), cuts FP 277→64**
  (precision 0.15→0.38), modest recall cost (0.69→0.54). Per-task: big wins where the loop over-fires
  (dedup 15→50, step-inst 18→48, event-alert 20→43); slight cost on closely-spaced triggers (state
  34→33, cond-alert 40→36, where the refractory can merge distinct triggers).
- **CEILING (no overclaim): joint_F1 stays LOW for both (5.7 vs 3.0)** — emits land at the right TIME but
  CONTENT (counts/states) is wrong = perception/format ceiling, NOT cognition. **Do not claim a leaderboard
  win:** our 44.6 is `time_F1` on the easy visual slice; the published "Online F1" band (MiniCPM-o 20.9 /
  MMDuet2 11.3 / LiveStar 3.6) is ~`joint_F1` on the FULL omni set — not comparable. Like-for-like (joint),
  Percept ~5.7 ≈ LiveStar, below MiniCPM-o. **Timing strong & beats the raw loop on OmniPro's own metric;
  content is the unsolved perception problem** = the audio-modality + grounding roadmap target.
- Confirms the §3 unifying principle on the most comprehensive proactive benchmark, scored by ITS OWN code.

## 7. Changelog

- **v2 · 2026-06-21** — Rewrote §1 into the standing **Eval Protocol** (follow for every run): Core vs
  User-facing metric taxonomy; the two "latencies" (video-time temporal accuracy — what leaderboards measure,
  compute-agnostic/offline — vs wall-clock voice-to-voice — TML's 200ms, on no leaderboard, unmeasured);
  layer-coverage declaration (all runs so far = Core+Perception only, offline, no edge/harness/latency); ASK
  reframed as a fire-shaped product lever (eval on ambiguity/selective-prediction, not clean-label); claims
  discipline + standing run checklist. §5: added "real-time latency unmeasured" + reclassified ASK.
- **v1.2 · 2026-06-21** — Added §9: RAN OmniPro visual slice with its OWN scorer. Percept time_F1 25.1→44.6
  (+19.5, FP 277→64) over the raw loop; joint_F1 low (5.7) = content/perception ceiling. First result
  scored by an external benchmark's own code. Timing wins; content unsolved.
- **v1.1 · 2026-06-20** — Added §8: OmniPro as the most-aligned candidate *real-leaderboard* axis
  (proactive streaming; long-term-collapse axis = Percept's thesis; Online F1@±3s = the gate's job).
- **v1 · 2026-06-20** — Initial. Three TML axes run with Gemini Flash held constant (Charades mIoU +24,
  ProactiveVideoQA PAUC parity/4×-less-spam, RepCount OBO Δ0 honest negative) + the trust investigation
  (dead ASK valve, two failure modes, model A/B). Established the unifying principle and the controlled-A/B
  vs reference-bar methodology.
