# Outcome-Labeling / Shadow-Audit Pipeline (design spike S3 · R13)

**Status:** design only — no production code change. **Owner of the dependency:** calibration
(`eval/calibrate.py`), and therefore the spec's "consequence sets the firing threshold" (§6.1, §10.3).

## Why this exists (the hidden upstream dependency)

The spec assumes the gate threshold is a meaningful lever: higher-consequence goals demand higher
confidence before firing. Measurement says that lever is currently **inert**:

- The vision model's numeric `confidence` is **non-discriminative** — across 405 captured frames only
  **1** fell below 0.70, values cluster at ~0.85–0.95, and wrong-confident ≈ right-confident
  (`eval/BENCHMARKS.md` lines 115–117; confirmed by the `eval/calibrate.py` sweep, which needs
  threshold ≈ 0.96 to zero false-fire, collapsing recall to ~0).
- "Consequence-as-threshold" cannot work on a confidence axis that doesn't separate right from wrong.

To make confidence usable you must **calibrate** it, and calibration requires **labeled outcomes**:
for each verdict the model emitted, *was it actually correct?* No production pipeline captures
verdicts-in-the-wild and later attaches a ground-truth label. This doc designs the minimal one.

The crucial observation that makes a first step cheap: **the offline golden corpus already has the
labels.** `eval/calibrate.py` captures per-frame `(satisfied, confidence)` into
`calibration_capture.json`, and `gate_cases.jsonl` carries the human `expected_verdict` per scene. We
can derive `(confidence, was_correct)` pairs and produce a reliability number **today**, before any
production capture exists. The production capture/label loop (parts 1–2) is what extends this from the
golden set to the wild.

---

## What already exists (ground truth for the design)

| Piece | Path | Shape / role |
|---|---|---|
| Verdict DTO | `percept/adapters.py` | `Verdict{satisfied: bool, confidence: float, evidence: str}` — frozen membrane |
| Verdict production | `percept/agent.py` `Percept._judge` | calls `vision.judge(...)`, stores in cache, returns `Verdict`. **The capture hook point.** |
| Gate | `percept/gate.py` `classify` / `Gate.evaluate` | maps `(verdict, freshness, threshold)` → `fire`/`ask`/`silent` |
| Fire audit log | `percept/consent.py` `ActionLog` / `ActionRecord` | append-only `{kind, goal_id, detail, t, allowed}`; written by `agent.py:_log_fires`. Audits *fires*, **not verdicts** — see gap below |
| Golden loader + split discipline | `percept/eval/__init__.py` | `load_golden`, `find_golden_root`; enforces disjoint train/dev/test, `test` held out |
| Gate cases (human labels) | `eval/fixtures-golden/gate_cases.jsonl` | per scene: `expected_verdict`, `proposer_verdict`, `proposer_confidence`, `review_status`, `difficulty`, `split` |
| Review audit trail | `eval/fixtures-golden/reviews.jsonl` | per scene: `{scene_id, reviewer, decision, reviewed_at, notes, proposal{...}}` — **the existing label-provenance pattern to reuse** |
| Calibration capture | `eval/fixtures-golden/calibration_capture.json` | list of `{scene_id, split, expected, verdicts:[[satisfied, confidence], ...]}` |
| Calibration sweep | `eval/calibrate.py` | captures once live, sweeps threshold offline, writes `calibration.json` |
| Trust scorecard | `eval/scorecard.py` | replays golden gate cases, reports false-fire / agreement |
| Frame decode | `eval/frames.py` | `frames_for_scene` → `(ts, (media_type, bytes))` |

**The gap:** `ActionLog` records *what fired* but never the **verdict** (`confidence`, `satisfied`)
nor the *silent/ask* verdicts that didn't fire — and a verdict that was suppressed is exactly the
data calibration needs. There is no sink that captures **every** judged verdict with a frame-ref so a
label can be attached later. That sink is part 1.

---

## Design overview (the loop)

```
                     production / replay
  frame ─► _judge ─► Verdict ──┐
                               ├─► (1) CAPTURE  ─► audits/verdicts-*.jsonl   (shadow-audit sink)
                               │       (confidence, satisfied, condition, frame-ref, t, source)
                               │
                               ▼
                        (2) LABEL  ─► audits/labels-*.jsonl
                               human review (reviews.jsonl pattern) and/or stronger-model adjudicator
                               row: (verdict_id, confidence, was_correct, label_source)
                               ▼
                        (3) CALIBRATE  ─► reliability diagram + ECE, then a calibration map
                               extends eval/calibrate.py; TESTS whether confidence discriminates
```

Three append-only JSONL files under a single `eval/fixtures-golden/audits/` dir, joined by a
stable `verdict_id`. Everything offline after capture; nothing on the hot path beyond one append.

---

## Part 1 — Capture (the shadow-audit sink)

**Goal:** log every judged verdict with enough context to label it later, cheaply and boundedly.

**Where it hooks in:** `percept/agent.py` `Percept._judge` — the single chokepoint where *every*
watch verdict is produced (the transition path `_perceive_transition` calls `vision.judge` directly
and would need the same one-line emit; capturing only `_judge` is acceptable for v1). It must capture
**before** the gate, so suppressed (silent/ask) verdicts are recorded too — those are the negatives
calibration is starving for.

Hook shape (a side-effect sink, injected like the existing `audit` callable on `Percept` so
production stays opt-in and the membrane is untouched):

```python
# conceptual — NOT applied here (design only)
verdict = await self.vision.judge(goal.condition, frame, roi=roi, corrections=corrections or None)
if self.verdict_sink is not None:           # default None -> zero overhead, no behaviour change
    self.verdict_sink.capture(goal, verdict, frame, roi, entity_id, source, t=now)
self.cache.store(cache_key, verdict, verdict.confidence)
return verdict
```

**What it writes** — `eval/fixtures-golden/audits/verdicts-<date>.jsonl`, one row per judged verdict:

```json
{
  "verdict_id": "v_2026-06-21T08:14:02.511Z_3f9c1a",
  "t": 4.637,
  "source": "camera",
  "goal_id": "g_chase",
  "condition": "we reach Chase",
  "confidence": 0.85,
  "satisfied": false,
  "evidence": "city sidewalk, no Chase sign visible",
  "frame_ref": {"kind": "dhash", "value": "e3b0c44298fc1c14", "roi": [0.1,0.2,0.5,0.6]},
  "gate_action": "silent",
  "backend": "claude",
  "split": null
}
```

Field notes:
- `verdict_id` — stable join key for label/calibrate (timestamp + short hash of `goal_id|condition|frame_ref`).
- `frame_ref` — **a reference, never pixels.** Reuse the cache's `FrameHashLayer` dHash (already
  computed for the ROI in `CacheKey`), so capture costs nothing extra and stays tiny. In replay/golden
  mode `frame_ref` can instead be `{"kind":"scene","scene_id":...,"ts":...}` so the labeler can pull
  the exact frame from `scenes/<id>/clip.json` via `eval/frames.py`.
- `confidence, satisfied, evidence` — straight off the `Verdict` (`adapters.py`).
- `gate_action` — what the gate *did* (`fire`/`ask`/`silent`), recorded after the gate so we can later
  ask "of the things we suppressed, how many were wrong?" (the false-negative side calibration needs).
- `split` — `null` in production; set when capturing over the golden set so calibration can honour the
  held-out `test` split.

**Cheap and bounded:**
- One serialized line appended per judge call; no pixels, no network, no model call.
- **Sampling:** reuse the existing `DeterministicAuditor` counter from `agent.py` (stratified 1-in-N,
  no RNG on the L1 path) so production captures e.g. 1-in-20 verdicts, not all of them.
- **Bounded:** size-capped rotating files (`verdicts-<date>.jsonl`, rolled at N MB); the sink is a sink,
  not a store. The label step drains it into the durable `labels-*.jsonl`.
- **Opt-in:** `verdict_sink=None` by default → the membrane and the hot path are byte-for-byte unchanged
  when capture is off. This is a *shadow* audit: it never affects a fire/ask/silent decision.

---

## Part 2 — Label (attach a ground-truth outcome)

**Goal:** turn a captured verdict into a labeled row `(confidence, was_correct)`. `was_correct` means
the model's `satisfied` bit matched reality for that frame+condition — that, not the scene-level
`fire/silent/ask`, is the binary calibration scores against.

Two label sources, mirroring the corpus's existing two-track review:

**(a) Deferred human review — reuse the `reviews.jsonl` pattern.** The corpus already has a
proposer→reviewer audit trail (`reviews.jsonl`: `{scene_id, reviewer, decision, reviewed_at, notes,
proposal}`) and a proposer-vs-expected record on every gate case (`gate_cases.jsonl`:
`proposer_verdict` / `expected_verdict` / `review_status`). The labeler writes the **same shape** for
verdicts: a reviewer is shown `frame_ref` + `condition` + the model's `(satisfied, confidence,
evidence)` and records the true `satisfied`. Immutability and split discipline carry over unchanged —
labels are versioned, PR-reviewed, and a verdict whose `frame_ref` resolves to a `test`-split scene
stays in `test`.

**(b) Stronger-model adjudicator** (cheaper, scales the negatives). A higher-capacity judge
(Opus / Gemini-3) re-judges the **same `condition` + frame** and its verdict is taken as the
ground-truth `satisfied`. This is the same idea `eval/exp_model_ab.py` already exercises (the
BENCHMARKS note that Opus/Gemini-flash reach false-fire 0 where Haiku doesn't — overconfidence is a
*capacity* issue), repurposed as a labeler rather than as the production judge. Adjudicator labels are
marked `label_source:"opus"` so a later human pass can spot-audit them; on disagreement, human wins.

> Provider note: the adjudicator is an Anthropic (Opus) or Gemini call. Before wiring it, read the
> `claude-api` skill for current model IDs/pricing — do not hardcode from memory.

**The labeled row** — `eval/fixtures-golden/audits/labels-<date>.jsonl`:

```json
{
  "verdict_id": "v_2026-06-21T08:14:02.511Z_3f9c1a",
  "confidence": 0.85,
  "model_satisfied": false,
  "true_satisfied": false,
  "was_correct": true,
  "label_source": "human",          // "human" | "opus" | "gemini-3"
  "reviewer": "reviewer",
  "reviewed_at": "2026-06-21T09:02:00Z",
  "notes": "no Chase visible in this frame",
  "split": "dev"
}
```

`was_correct = (model_satisfied == true_satisfied)`. The minimal pair calibration consumes is just
`(confidence, was_correct)`; the rest is provenance so the label set stays auditable like the golden
corpus.

---

## Part 3 — Calibrate (reliability diagram + ECE, then a map)

**Goal:** from `(confidence, was_correct)` pairs, first **test whether confidence discriminates**, then
(only if it does) fit a calibration map that the gate/consequence logic can use.

**Reliability diagram + ECE.** Bin verdicts by confidence (e.g. 10 bins of width 0.1). In each bin
compare mean confidence to observed accuracy (`mean(was_correct)`):

```
ECE = Σ_b (n_b / N) · | acc(b) − conf(b) |
```

A well-calibrated model lies on the diagonal (acc ≈ conf in every bin); ECE → 0. The expected — and
diagnostically important — result here is the opposite: nearly all mass sits in the 0.85–0.95 bins, so
the diagram collapses to two or three points and ECE just measures the **overconfidence gap** (high
confidence, mediocre accuracy). That is the *number that proves* the lever is broken, replacing the
current anecdote ("1/405 frames < 0.70") with a single reproducible scalar tracked over time.

The richer test is **discrimination, not just calibration**: split `was_correct` by confidence and ask
whether the distributions differ at all (AUROC of confidence vs. `was_correct`, or simply
mean-confidence-when-right vs. mean-confidence-when-wrong). AUROC ≈ 0.5 / equal means says *no
threshold on this axis can separate right from wrong* — the spec's threshold lever is dead and the fix
must come from a **different signal** (categorical/selective-prediction ASK per BENCHMARKS line 138, or
a higher-capacity judge), not from tuning `PERCEPT_GATE_THRESHOLD`.

**Fit a map (later).** If/when a signal *does* discriminate (e.g. a different prompt that elicits
spread confidence, or the adjudicator's confidence), fit a monotone calibration map — temperature
scaling or isotonic regression — `confidence → P(correct)`, validated on `dev`, frozen on `test`,
written next to the threshold in the calibration artifact. Only *then* does consequence-as-threshold
become meaningful: it would threshold on **calibrated** `P(correct)`, not raw model confidence.

**How this extends `eval/calibrate.py`.** `calibrate.py` already: (1) captures per-frame
`(satisfied, confidence)` once, (2) sweeps the gate threshold on `train`, (3) validates on `dev`/`test`
without tuning on `test`, (4) writes an auditable `calibration.json`. It is a **threshold** sweep over
scene-level verdicts. Calibration adds a **per-verdict reliability/ECE** computation that reads the same
splits and writes alongside the existing artifact:

```json
{
  "default_threshold": 0.7,
  "chosen_threshold": 0.96,
  "train_sweep": [ ... ],                     // existing
  "reliability": {                            // NEW
    "n": 405,
    "ece": 0.18,
    "auroc_conf_vs_correct": 0.51,
    "bins": [{"lo":0.8,"hi":0.9,"n":120,"conf":0.86,"acc":0.71}, ...],
    "mean_conf_right": 0.91, "mean_conf_wrong": 0.89,
    "discriminative": false,
    "verdict": "confidence does NOT discriminate; threshold is the wrong lever"
  }
}
```

---

## Part 4 — Minimal first step (ships now, before any production deploy)

**The smallest viable thing: one offline calibration script over the golden set's existing labels.**

No capture hook, no production deploy, no new labeling needed for the *first reliability number* —
because the labels are already on disk:

- `calibration_capture.json` has per-frame `(satisfied, confidence)` per scene.
- `gate_cases.jsonl` has the human `expected_verdict` per scene.

Derive `was_correct` per frame from the human label and compute the reliability diagram + ECE today.

**Concretely:** add `eval/reliability.py` (a sibling of `calibrate.py`, same `load_golden` /
`find_golden_root` / split discipline, same dotenv gating), that:

1. loads `calibration_capture.json` (offline; no `PERCEPT_LIVE` needed — the capture already exists);
2. for each scene's frames, derives per-frame `was_correct` from the scene's human label. The honest
   reduction for v1: on a `fire`/`expected-true` scene the correct per-frame `satisfied` is `true` for
   frames at/after the labeled rising edge and `false` before it; on a `silent` scene every frame's
   correct `satisfied` is `false`; `ask` scenes are excluded from the reliability set (no crisp
   per-frame truth). Where per-frame truth is ambiguous, count at the scene level (did *any* confident-
   satisfied verdict occur ⟺ should it have) rather than over-claiming frame labels — and emit a clear
   caveat in the artifact;
3. bins by confidence, computes `acc(b)`, `conf(b)`, **ECE**, and mean-conf-right vs mean-conf-wrong;
4. prints a text reliability diagram (same house style as `scorecard.py` / `calibrate.py`) and writes
   `reliability.json`.

**What it buys immediately:** a single reproducible **ECE / discrimination** number replacing the
"1/405" anecdote, computed from data already in the repo, that *quantitatively confirms the lever is
inert* — the green light (or the kill) for any further work on consequence-as-threshold. It also
becomes the regression target: when a prompt/model change is tried (categorical ASK, Opus judge), rerun
`reliability.py` and watch ECE/AUROC move.

**Then, in order:** (i) generalize the per-frame `was_correct` derivation behind a real label file by
landing the `labels-*.jsonl` shape (part 2) over the adjudicator so the reliability set isn't limited
to clean-label golden scenes; (ii) add the `verdict_sink` capture hook (part 1) behind a default-`None`
injection so production can start filling `verdicts-*.jsonl`; (iii) only after a signal that *does*
discriminate appears, fit and freeze the calibration map (part 3).
