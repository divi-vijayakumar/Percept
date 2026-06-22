# Percept Benchmark v1 — review, data plan, and implementation plan

Companion to `percept-benchmark-v1-spec.md` (spec v1.0). This doc (a) reviews the spec, (b) reshapes the
**data plan** around the DeepMind **Perception Test**, and (c) gives a **phased implementation plan** mapped
to the harness we already have.

---

## 1. Verdict on the spec — adopt, with 6 amendments

The spec is strong and **publishable in shape**. Its core decisions are right and match what we *measured*
this cycle, so adopt them wholesale:

- **Control-layer thesis** (hold the backbone fixed, measure the orchestration delta). This is exactly the
  perception-ceiling finding turned into an eval: the model is the constant, the architecture is the variable.
- **3-config ladder raw→core→e2e**, never collapsed — separates the gate (free, portable, same calls) from
  the edge (cost). Correct.
- **Native ceiling as a reference line, not a rung** — honest; we recover a fraction of proactivity at a
  fraction of cost on hardware it can't run on.
- **Absence as a track** — the one primitive a per-frame VLM structurally lacks. The single best differentiator.
- **No composite score**; vector of headlines — resists gaming and the self-defined-metric critique.
- **Measured-vs-modeled rule + commit/manifest-stamped Snapshot Cards** — the discipline that makes a number
  citable.
- **Gate ledger (G3 no world model · R3 confidence-doesn't-discriminate)** — R3 is our *measured* result (the
  flat calibration sweep); gating consequence-claims behind it is exactly right.

### The 6 amendments (where I'd push)

1. **Add a v0.5 "runnable-now" slice.** The spec is publication-grade and therefore data- and harness-heavy;
   the risk is it never *runs*. Define an explicit minimal slice (relational + counting + acoustic, sampled,
   modeled-e2e, **DRAFT** card) that we can emit this week with what we already have, then grow to v1-full.
   (§4 phasing.)

2. **Make AUROC-of-confidence a measured headline, not just a gate footnote.** R3 lives in the card as
   `AUROC <value>`. Promote it: "does the backbone's confidence separate right from wrong?" is a clean,
   citable number (we already saw it's ≈flat). It *justifies* the ASK discipline and the consequence-model
   gate — it's the benchmark's most defensible single finding. Report it per track.

3. **Validate the e2e-relational modeling assumption — don't just assert "events preserved."** Modeling
   e2e accuracy as `core × edge-call-reduction` assumes the motion-gate escalates on *every* true event.
   That's unproven and plausibly false for *low-motion* events (a still "child in danger"). **Measure the
   edge's event-recall** (fraction of labelled onsets where motion clears threshold) as a first-class number;
   if it's < ~1.0, e2e accuracy is `core × edge-event-recall`, not ≈core. This is a real hole in the modeling.

4. **The relational track's honest story is *discipline + cost*, not accuracy.** The spec's own hypothesis is
   "P/R ~flat across configs." State plainly that the **accuracy** lift lives in *counting* + *acoustic*; the
   relational track sells the **ASK-rate / $-per-correct-event / call-reduction**. Don't let a reader infer a
   relational accuracy win that isn't there.

5. **Downgrade "gap recovered %" from headline to directional until the full run.** It depends on the native
   ceiling, which is the most expensive and least-controlled measurement (different sampling regime, §2.0).
   Lead the sampled card with **call-reduction + count-OBO + acoustic recall-flip + AUROC**; treat
   gap-recovered as a v1-full headline.

6. **Absence has no public dataset — derive it (transparently) from Perception Test TAL.** Pick a recurring
   labelled action, set a deadline `D`; episodes where it's absent within `D` are the positives. Flag it as a
   *constructed* task (weaker external validity than the localisation tracks); keep the self-recorded
   Pumpkin-Pi case as the honest demo.

---

## 2. Perception Test — the data unlock (and its limits)

`github.com/google-deepmind/perception_test` · **CC-BY-4.0** data + Apache-2.0 code · **GCS direct zips, no
registration** · 11,609 videos. Tracks (with µs start/end timestamps + frame IDs):

| PT track | size | maps to OUR track | why it fits |
|---|---|---|---|
| **Temporal Sound Localisation** | 137,128 sound segments + **WAV audio** | **acoustic** | onset-timed, audio-determinable events — the data our acoustic lever was missing (we had *one* hairdryer clip) |
| **Temporal Action Localisation** | 73,503 action segments | **timeliness** (P-PAUC + reaction-λ) + derive **absence** | precise onsets = the "turns" P-PAUC needs; recurring actions + a deadline = absence episodes |
| Single Object / Point Tracking | 189,940 / point tracks | counting/tracking (**v2**, G3-gated) | tracking continuity; relations are G3-gated |
| Multiple-Choice / Grounded VQA | 38,060 / 6,086 | not used (offline QA shape ≠ fire/ask/silent) | wrong output shape; we borrow data+vocab, not scoring |

**What PT gives us:** redistributable, public-comparable, **onset-timed, audio-bearing** data for the two
tracks we couldn't source — fixing both the data gap *and* the golden corpus's uncleared-YouTube licensing
problem for those tracks.

**What PT does NOT give us:** per-annotator / ambiguity labels. So the **relational / ASK-discipline** track
**stays on the golden corpus** (the ambiguity set with multi-labeler disagreement) — PT can't replace it.

### Per-track data sources (final)

| track | source | license | run-now? |
|---|---|---|---|
| relational / ASK | golden gate-cases (private) | uncleared — dev/eval only | yes (have it) |
| counting | user clips (alphabets, jumping jacks) + RepCount-A (lance) | clips ad-hoc; RepCount research | partial (clips yes; RepCount needs refetch) |
| acoustic | **PT Temporal Sound Localisation** (+ hairdryer demo) | CC-BY-4.0 | **yes (GCS fetch)** |
| timeliness | **PT Temporal Action Localisation** | CC-BY-4.0 | **yes (GCS fetch)** |
| absence | derived from PT-TAL (+ self-recorded) | CC-BY-4.0 | yes (construct) |
| native ceiling | a streaming model on shared tracks | — | v1-full |

---

## 3. Harness gap — what `bench_matrix` must become

Today `bench_matrix` is a **cache re-scorer** (re-reads cached run JSONs → a fixed table). The spec needs it
to become the **Snapshot-Card emitter**: run tracks × configs, compute the metric vector (§4 of the spec),
and emit a stamped card with a provenance block (measured vs modeled), commit SHA, and data-manifest hash.
That is the central build. The per-track scorers already exist and slot in as entry points:

| track | existing tool → role |
|---|---|
| relational | `scorecard.py` (per-frame `judge` → `_predict`) → raw vs core firing |
| counting | `count_video.py` / `pose_count.py` → raw-VLM-count vs edge/pose vs gt |
| acoustic | `fused_judge.py` → vision-only vs fused (+ `AcousticOnset` for onsets) |
| absence | agent-loop replay (deadline rearm) → absence recall/precision |
| matrix | `bench_matrix.py` → **rewrite as card emitter** |

New scorers to write: **P-PAUC + reaction-λ** (timeliness, from PT-TAL onsets), **AUROC-of-confidence**
(per track, from the judged verdicts vs labels), **cost block** (VCPM, $/correct-event from a pinned price
sheet), **edge event-recall** (amendment #3).

---

## 4. Phased implementation plan

### Phase A — v0.5 "runnable-now" DRAFT card (this week; no new data)
Goal: emit a real, stamped DRAFT Snapshot Card on the data + tools we have, to lock the card format and the
measured-vs-modeled plumbing.
1. **Card emitter skeleton** — rewrite `bench_matrix` to emit the §7 card (headline vector + per-track×config
   table + provenance), commit SHA + a data-manifest hash, `stamp=DRAFT`.
2. **Relational** (raw vs core): wire `scorecard` for ~20 golden dev cases → P/R/F1 + ASK-rate + **AUROC**.
3. **Counting** (raw vs e2e): `count_video` vs `pose_count` on the user clips → OBO/MAE (pose=12, alphabets=3).
4. **Acoustic** (e2e): `fused_judge` on the hairdryer (+ any clips) → recall-flip + fusion-FP.
5. **Cost**: VCPM + $/correct-event from a pinned Gemini price sheet; e2e call-reduction from the edge gate.
6. **Modeled cells** flagged with `~` (e2e relational accuracy); card stays `DRAFT`.

### Phase B — Perception Test integration (the data unlock)
1. **Fetcher + manifest**: a `perception_test` loader (GCS zips → frames + WAV + onset annotations), pinned
   by a manifest hash; gitignore the raw data.
2. **Acoustic track at scale**: PT Temporal Sound Localisation → vision-only vs fused on N sampled
   audio-cued events → recall-flip with CIs. (Replaces the single-clip demo.)
3. **Timeliness track**: PT Temporal Action Localisation onsets → **P-PAUC (dual: PVQA-native + Percept-aware)
   + reaction-λ p50/p90**. New scorer.
4. **Edge event-recall** (amendment #3): on PT onsets, does the motion-gate escalate? → validates/denies the
   e2e modeling.

### Phase C — v1-full RELEASE card
1. **Absence track**: construct from PT-TAL (recurring action + deadline) + the self-recorded case →
   absence recall/precision; raw forced-silent (0 by construction).
2. **Agent-loop replay**: measure e2e accuracy + reaction-λ end-to-end (remove the modeled cells) on a target
   device; ≥3 seeds + bootstrap CIs.
3. **Native ceiling**: a streaming model on the shared (PT-derived + PVQA) tracks under PVQA-native →
   gap-recovered + cost ratio.
4. Empty `modeled` block + pinned manifest + ≥3 seeds → **stamp=RELEASE**.

### v2 (gated — do not build until the gate clears)
- **Consequence-weighted firing** — only when **R3** clears (AUROC > 0.5 on live capture). Until then the
  two-axis cost model stays inert and unclaimed.
- **Relations / aggregates / multi-stream** — only when **G3** lands (a shared world model).

---

## 5. Critical-path risks (call them before running)

- **Runnability over ambition.** Ship the Phase-A DRAFT card first; a perfect spec that never emits a card is
  worth less than a DRAFT card that does. (Amendment #1.)
- **The modeling hole (amendment #3).** If the motion-gate misses low-motion events, "e2e ≈ core" is false.
  Measure edge event-recall in Phase B *before* citing any modeled e2e relational number.
- **Ceiling is apples-to-oranges** (sampling regime, §2.0). Keep it directional until Phase C; don't lead with
  gap-recovered.
- **Accuracy story is narrow** — counting + acoustic, not relational. Sell relational as discipline + cost.
- **PVQA comparability** needs reproducing published numbers on ProactiveVideoQA; if we can't, the dual-report
  comparability weakens to "directional."

---

## 6. References

Every Snapshot Card carries a `references:` block citing the works the run builds on. Cite these (the
verified three in full; confirm author lists for the rest before a `RELEASE` card):

- **Interaction Models** — the *native interaction model* concept underlying the §2.1 native-ceiling
  reference line (the duplex omni-model we measure the gap against, never claim to beat):

  > Thinking Machines Lab, "Interaction Models: A Scalable Approach to Human-AI Collaboration",
  > *Thinking Machines Lab: Connectionism*, May 2026.

  ```bibtex
  @article{thinkingmachines2026interactionmodels,
    author  = {Thinking Machines Lab},
    title   = {Interaction Models: A Scalable Approach to Human-AI Collaboration},
    journal = {Thinking Machines Lab: Connectionism},
    year    = {2026},
    month   = {May},
    note    = {https://thinkingmachines.ai/blog/interaction-models/},
    doi     = {10.64434/tml.20260511},
  }
  ```

- **Perception Test** — CC-BY-4.0 audio + onset data for the acoustic & timeliness tracks (§2):

  > V. Pătrăucean, L. Smaira, A. Gupta, A. Recasens, et al., "Perception Test: A Diagnostic Benchmark for
  > Multimodal Video Models", *NeurIPS 2023* (Datasets and Benchmarks). arXiv:2305.13786.

  ```bibtex
  @inproceedings{patraucean2023perception,
    title     = {Perception Test: A Diagnostic Benchmark for Multimodal Video Models},
    author    = {P{\u{a}}tr{\u{a}}ucean, Viorica and Smaira, Lucas and Gupta, Ankush and Recasens, Adri{\`a} and others},
    booktitle = {Advances in Neural Information Processing Systems (NeurIPS), Datasets and Benchmarks Track},
    year      = {2023},
    note      = {arXiv:2305.13786; data CC-BY-4.0},
  }
  ```

- **ProactiveVideoQA / PAUC** — the proactive timeliness metric we adapt into P-PAUC (§4.2 of the spec):

  > "ProactiveVideoQA: A Comprehensive Benchmark Evaluating Proactive Interactions in Video Large Language
  > Models", arXiv:2507.09313, 2025. (PAUC = Proactive Area Under Curve.)

  ```bibtex
  @article{proactivevideoqa2025,
    title   = {ProactiveVideoQA: A Comprehensive Benchmark Evaluating Proactive Interactions in Video Large Language Models},
    journal = {arXiv preprint arXiv:2507.09313},
    year    = {2025},
  }
  ```

- **RepCount-A** (counting track) — Hu et al., "TransRAC: Encoding Multi-scale Temporal Correlation with
  Transformers for Repetitive Action Counting", *CVPR 2022*. *(confirm full author list before RELEASE)*
- **Streaming perception** (latency-coupled scoring philosophy, §1.4) — Li, Wang, Ramanan, "Towards Streaming
  Perception", *ECCV 2020*. *(confirm before RELEASE)*
- **Charades** (if used) — Sigurdsson et al., "Hollywood in Homes: Crowdsourcing Data Collection for Activity
  Understanding", *ECCV 2016*. *(confirm before RELEASE)*

Where each is cited: the **interaction model** anchors the native-ceiling rung; **Perception Test** is the
acoustic/timeliness/(v2 tracking) data; **ProactiveVideoQA** is the PAUC source we adapt + the PVQA-native
comparability baseline; **RepCount-A** is the counting data; **streaming perception** is the timeliness
scoring philosophy.
