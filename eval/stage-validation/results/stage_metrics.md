# Per-stage recall, precision, economy and drop rate

Runtime under test: `percept` wheel at `083b972c75b012df32d811f93d39b6bc25406999` (`/private/tmp/percept-launch/oss-final-venv-083b972/bin/python`).
Corpus: 6 of 6 video cases; 156 s of ground-truth-positive time over 164 s of media. Generated 2026-07-27T03:50:51Z.

No provider or network call was made. Judges are scripted and offline: Stage 1 runs against the real decoded media through the shipped gate, and Stages 3-4 are driven by a ground-truth oracle so they measure PLUMBING, not judge quality (explicitly out of scope).

## Headline - where the pipeline loses

Recall is reported twice at every stage. **stage-local** is versus what that stage actually received; **cumulative** is versus the original source ground truth. A stage whose stage-local recall is 1.000 lost nothing itself, however low its cumulative number is — that pair is the whole point, and it is what an end-to-end number cannot give you.

| Stage | Recall (stage-local) | Recall (cumulative) | Precision | Economy | Drop rate | Units in -> out | n |
|---|---|---|---|---|---|---|---|
| Stage 0 - Ingest + normalize | 1.000 | 1.000 | 0.947 | 0.542 | 0.000 | 2,285 -> 1,238 | 6 |
| Stage 1 - Admission (VID tier-0 gate) | 0.186 | 0.186 | 0.940 | 0.179 | 0.000 | 1,238 -> 222 | 6 |
| Stage 2 - Local perception | 1.000 | 0.186 | 0.940 | 1.000 | 0.000 | 222 -> 222 | 6 |
| Stage 3 - Judge dispatch | 1.000 | 0.186 | 0.940 | 1.000 | 0.000 | 222 -> 222 | 6 |
| Stage 4 - Cognition + disposition | 0.999 | 0.186 | 0.998 | 0.937 | 0.000 | 222 -> 208 | 6 |

Recall above is the spec's literal definition: the fraction of ground-truth-positive SOURCE TIME the stage passed through. On a sampled pipeline that quantity is bounded above by the stage's own economy — a gate admitting 18% of units cannot cover more than 18% of the clock — so at Stage 1 it largely restates economy and understates whether the event actually got through. The table below is the same two-population pair computed on OCCURRENCES: did at least one unit inside each ground-truth event survive this stage? This is the definition the pre-registered tier-0 numbers in `percept/config.py` use, and it is the one that localises a regression.

| Stage | Event recall (stage-local) | Event recall (cumulative) | GT events in -> out |
|---|---|---|---|
| Stage 0 - Ingest + normalize | 1.000 | 1.000 | 6 -> 6 |
| Stage 1 - Admission (VID tier-0 gate) | 0.667 | 0.667 | 6 -> 4 |
| Stage 2 - Local perception | 1.000 | 0.667 | 4 -> 4 |
| Stage 3 - Judge dispatch | 1.000 | 0.667 | 4 -> 4 |
| Stage 4 - Cognition + disposition | 1.000 | 0.667 | 4 -> 4 |

**The stage that loses the most is Stage 1 - Admission (VID tier-0 gate)**: stage-local time recall 0.186 and stage-local event recall 0.667 — it discards 127 s of the 156 s of ground-truth-positive time it was handed, and loses 2 ground-truth events entirely. Lossless on positive time at this boundary: Stage 0 - Ingest + normalize; Stage 2 - Local perception; Stage 3 - Judge dispatch.

## The funnel, source units to Incidents

| Boundary | Units | Fraction of source frames |
|---|---|---|
| Native frames decoded from media | 2,285 | 1.0000 |
| Selected by deliberate decimation | 1,238 | 0.5418 |
| Envelopes emitted by ingest | 1,238 | 0.5418 |
| Admitted by tier-0 (Stage 1) | 222 | 0.0972 |
| Dispatched to local perception (Stage 2) | 222 | 0.0972 |
| Judge calls started (Stage 3, logical) | 222 | 0.0972 |
| Judge calls completed (Stage 3) | 222 | 0.0972 |
| Gate classifications (Stage 4) | 222 | 0.0972 |
| Gate concluded condition true | 208 | 0.0910 |
| Incidents emitted | 4 | 0.0018 |

Ground-truth occurrences across the corpus: 6; occurrences matched by at least one Incident: 4 (0.667).

## Stage 1 - the statistic change: block_max@0.20 vs mean

The default salience statistic changed to `block_max` @ 0.20 in this wheel (`percept/config.py`), so every prior Stage-1 number computed under `mean` @ 0.10 is stale. Both are regenerated here against the same corpus and the same reproduced gate frames, scored with the wheel's own `percept.harness.tier0.salience_stats`.

Recall here is EVENT recall (fraction of ground-truth occurrences given at least one admitted frame) — the same definition as the pre-registered numbers this replaces. Time recall is carried alongside.

| Configuration | Event recall | Time recall | Precision | Economy | Reduction | Cases admitting zero frames | Cases missing every positive |
|---|---|---|---|---|---|---|---|
| `block_max @ 0.2 (SHIPPED DEFAULT)` | 0.667 | 0.186 | 0.940 | 0.179 | 0.821 | 2 | 2 |
| `mean @ 0.10 (historical default)` | 0.500 | 0.039 | 0.990 | 0.040 | 0.960 | 3 | 3 |
| `mean @ 0.0546 (MATCHED ECONOMY)` | 0.667 | 0.182 | 0.944 | 0.179 | 0.821 | 2 | 2 |

**At matched economy** (0.179), which is the only fair comparison — comparing each statistic at its own threshold changes the statistic and the budget at once, and the conclusion can invert — `block_max` event recall is 0.667 against `mean`'s 0.667, a difference of **0.000** at identical admission spend.

At equal admission spend the two statistics are within noise on this corpus; the shipped default's advantage at its own threshold is a budget effect, not a statistic effect.

Per-`goal_kind` event recall at each configuration:

| Configuration | `duration` |
|---|---|
| `block_max @ 0.2 (SHIPPED DEFAULT)` | 0.667 |
| `mean @ 0.10 (historical default)` | 0.500 |
| `mean @ 0.0546 (MATCHED ECONOMY)` | 0.667 |

## Per-stage breakdown by goal_kind

`state` has n=4 and cannot support a claim: it is reported, never gated.

### Stage 0 - Ingest + normalize

| goal_kind | Recall (stage-local) | Recall (cumulative) | Event recall (stage-local) | Event recall (cumulative) | Precision | Economy | Drop rate | n cases | n GT events |
|---|---|---|---|---|---|---|---|---|---|
| duration | 1.000 | 1.000 | 1.000 | 1.000 | 0.947 | 0.542 | 0.000 | 6 | 6 |

### Stage 1 - Admission (VID tier-0 gate)

| goal_kind | Recall (stage-local) | Recall (cumulative) | Event recall (stage-local) | Event recall (cumulative) | Precision | Economy | Drop rate | n cases | n GT events |
|---|---|---|---|---|---|---|---|---|---|
| duration | 0.186 | 0.186 | 0.667 | 0.667 | 0.940 | 0.179 | 0.000 | 6 | 6 |

### Stage 2 - Local perception

| goal_kind | Recall (stage-local) | Recall (cumulative) | Event recall (stage-local) | Event recall (cumulative) | Precision | Economy | Drop rate | n cases | n GT events |
|---|---|---|---|---|---|---|---|---|---|
| duration | 1.000 | 0.186 | 1.000 | 0.667 | 0.940 | 1.000 | 0.000 | 6 | 6 |

### Stage 3 - Judge dispatch

| goal_kind | Recall (stage-local) | Recall (cumulative) | Event recall (stage-local) | Event recall (cumulative) | Precision | Economy | Drop rate | n cases | n GT events |
|---|---|---|---|---|---|---|---|---|---|
| duration | 1.000 | 0.186 | 1.000 | 0.667 | 0.940 | 1.000 | 0.000 | 6 | 6 |

### Stage 4 - Cognition + disposition

| goal_kind | Recall (stage-local) | Recall (cumulative) | Event recall (stage-local) | Event recall (cumulative) | Precision | Economy | Drop rate | n cases | n GT events |
|---|---|---|---|---|---|---|---|---|---|
| duration | 0.999 | 0.186 | 1.000 | 0.667 | 0.998 | 0.937 | 0.000 | 6 | 6 |

## Per-stage breakdown by dataset

### Stage 0 - Ingest + normalize

| dataset | Recall (stage-local) | Recall (cumulative) | Event recall (stage-local) | Event recall (cumulative) | Precision | Economy | Drop rate | n cases | n GT events |
|---|---|---|---|---|---|---|---|---|---|
| charades | 1.000 | 1.000 | 1.000 | 1.000 | 0.947 | 0.542 | 0.000 | 6 | 6 |

### Stage 1 - Admission (VID tier-0 gate)

| dataset | Recall (stage-local) | Recall (cumulative) | Event recall (stage-local) | Event recall (cumulative) | Precision | Economy | Drop rate | n cases | n GT events |
|---|---|---|---|---|---|---|---|---|---|
| charades | 0.186 | 0.186 | 0.667 | 0.667 | 0.940 | 0.179 | 0.000 | 6 | 6 |

### Stage 2 - Local perception

| dataset | Recall (stage-local) | Recall (cumulative) | Event recall (stage-local) | Event recall (cumulative) | Precision | Economy | Drop rate | n cases | n GT events |
|---|---|---|---|---|---|---|---|---|---|
| charades | 1.000 | 0.186 | 1.000 | 0.667 | 0.940 | 1.000 | 0.000 | 6 | 6 |

### Stage 3 - Judge dispatch

| dataset | Recall (stage-local) | Recall (cumulative) | Event recall (stage-local) | Event recall (cumulative) | Precision | Economy | Drop rate | n cases | n GT events |
|---|---|---|---|---|---|---|---|---|---|
| charades | 1.000 | 0.186 | 1.000 | 0.667 | 0.940 | 1.000 | 0.000 | 6 | 6 |

### Stage 4 - Cognition + disposition

| dataset | Recall (stage-local) | Recall (cumulative) | Event recall (stage-local) | Event recall (cumulative) | Precision | Economy | Drop rate | n cases | n GT events |
|---|---|---|---|---|---|---|---|---|---|
| charades | 0.999 | 0.186 | 1.000 | 0.667 | 0.998 | 0.937 | 0.000 | 6 | 6 |

## Long-horizon sources - multi-occurrence and re-arm

| Source | Media s | GT occurrences | Admission event recall | Admission time recall | Incidents | Occurrences matched | Duplicate fires | Re-armed on every occurrence |
|---|---|---|---|---|---|---|---|---|

`Duplicate fires` counts Incidents beyond the first inside one ground-truth occurrence — a gate that re-fired without the condition having gone false. `Re-armed on every occurrence` is the multi-occurrence question these sources exist to answer: after firing and closing, does the gate come back for the next one?

## Audio channel (VAD / acoustic admission)

Audio ground-truth corpus: 53 cases — 40 `audio_annotated`, 3 `audio_annotated_point`, 10 `proxy`. Only `audio_annotated` (a human annotated a SOUND event with a start and an end) is used as a bar. `proxy` cases carry a Charades label that is usually audible but whose annotated interval is VISUAL, and are excluded from every audio number below.

**The audio-only entrypoint is not reachable with an ordinary `Goal`.** `run(source=None, audio_source=...)` raises `ValueError: audio-only execution requires a compiled Watch; 'g' is unbound`. Audio therefore had to be measured on the FUSED path (`source=` and `audio_source=` both set).

| Audio boundary | Units received | Units passed | Economy | Drop rate | Recall | Precision |
|---|---|---|---|---|---|---|
| Stage 0 — audio ingest | 205 | 205 | 1.000 | 0.000 | 1.000 | n/a |
| Stage 1 — acoustic admission | 205 | 0 | 0.000 | 0.000 | **not instrumentable** | **not instrumentable** |

The runtime emits NO per-audio-chunk admission event. `edge.decision` is the video gate only, and an acoustic wake surfaces solely as the aggregate counters `RuntimeStats.audio_wakes` / `admission_by_cause['audio']`. Without a time-resolved admission series there is nothing to intersect with audio ground truth, so audio Stage-1 recall and precision are NOT instrumentable through the public API.

For a number that does exist, the shipped `AcousticOnset` gate was driven directly at the shipped capture cadence over the 17 annotated sound events in the 40 `audio_annotated` cases: **event recall 0.000**, economy 0.000 (0 wakes over 205 capture chunks), at a declared 0.250 s tolerance (one capture chunk — an onset is reported at the chunk's END). This is a COMPONENT measurement, not the public path, and must not be quoted as an end-to-end audio number.

## Statistical power

Per-cell ground-truth event counts at Stage 1: `duration` n=6. A 0.95 recall bar is NOT statistically ratifiable on this corpus at any configuration: an exact one-sided Clopper-Pearson lower bound clearing 0.95 with ZERO misses needs n=59 positives — even a flawless 6-event `duration` cell bounds only at 0.607. Measured exact lower bounds on Stage-1 event recall: `duration` 0.667 (95% lower 0.271, n=6). `state` (n=0 events over 0 cases) is reported and NEVER gated — it cannot support a claim at any confidence.

## Boundaries that could NOT be instrumented

An honest 'not instrumentable at this boundary' is a finding. These are the boundaries this harness could not measure through the public API, and why.

- **Audio / VAD admission recall and precision (Stage 1, acoustic)** — The runtime emits NO per-audio-chunk admission event. `edge.decision` is the VIDEO gate only; an acoustic wake surfaces solely as the aggregate counters `RuntimeStats.audio_wakes` and `admission_by_cause['audio']`. With no time-resolved admission series there is nothing to intersect with audio ground truth, so audio Stage-1 recall/precision cannot be computed from the public API. Separately, the audio-ONLY entrypoint (`run(source=None, audio_source=...)`) refuses an ordinary `Goal` — it requires a compiled Watch — so audio had to be measured on the fused path. A COMPONENT-level `AcousticOnset` recall is reported instead, clearly labelled as not-the-public-path.
- **Stage 2 as a measuring point (local perception)** — With no `EvidenceExtractor` armed, Stage 2 is an identity pass-through: it emits one `source.frame` per admitted unit and `perceive_dispatches == escalations` exactly. Its economy of 1.000 therefore cannot distinguish 'lossless' from 'absent'. The senses (`rhythm`, `identity`, `entity`, `sequencing`) are independent modules with no shared metrics contract, so there is no per-sense precision/recall to read.
- **Physical provider attempts and typed terminal receipts (Stage 3)** — `run()` exposes no route/receipt surface. `JudgeRouteReceipt` outcomes (`deadline`, `capacity`, `no_route`, `cancelled`) exist only on a `JudgePool` the caller constructs and holds. Physical attempts here are counted at the ADAPTER boundary (the scripted judge's own call counter), which equals logical calls because a direct adapter does not hedge. Route-level hedging is unmeasured by this harness.
- **`metrics_collector=` on a video run** — Accepted by `run()` and SILENTLY IGNORED on the video path: it is forwarded only to `_run_audio_only`, so the video `_JudgeScheduler` never receives it. Queue depth and oldest-pending-age for video are therefore unobservable through the documented mechanism. This harness reconciles against `RuntimeStats` and the `on_health` `runtime_ended.source_coverage` ledger instead.
- **Units offered UPSTREAM of the decoder** — Stage 0's denominator here is frames the decoder actually yielded. Frames lost inside the container/codec before `cap.read()` returns them are invisible to any counter in the runtime, so ingest recall is measured against what the decoder delivered, not against what the file nominally contains.
- **Judge logical-vs-physical split under batching** — `RuntimeStats.judge_calls` counts scheduler BATCH launches while `perceive_dispatches` counts per-job launches; they are different units and must not be divided by each other. Both are reported separately rather than combined into one economy figure.

## Reconciliation against the runtime's own ledgers

Each per-stage number this report derives from the trace stream is cross-checked against an INDEPENDENT counter the runtime keeps for itself, over 6 cases.

| Check | Cases agreeing | Verdict |
|---|---|---|
| funnel.seen == envelopes emitted (Stage 0 out) | 6/6 | RECONCILES |
| funnel.gated == units admitted (Stage 1 out) | 4/6 | DISAGREES |
| RuntimeStats.escalations == admitted units | 6/6 | RECONCILES |
| funnel.judged == units admitted | 6/6 | RECONCILES |
| funnel.judged == fired + refused + silent | 6/6 | RECONCILES |

## Method and self-checks

- Runtime under test verified at import: `build_info().git_commit == 083b972c75b012df32d811f93d39b6bc25406999`.
- Zero provider or network calls: the only judge is a scripted in-process object keyed on a payload hash.
- Stage-1 substitution VERIFIED per case, not assumed: the FILE arm and the ORACLE arm must admit the identical set of timestamps. Held on 6 of 6 cases where it was checkable.
- Salience reproduction VERIFIED per case against the wheel's own `salience_stats`: `trace.motion == mean` and `escalate == (block_max >= 0.2)` held exactly on 6 of 6 cases; only those contribute to the threshold sweep.
- Aggregates pool numerators and denominators over time and units. Per-case means are reported alongside so a pooled figure driven by one long clip is visible.
- Suppression is never counted as a drop: a tier-0 rejection emits its own `edge.decision` record naming the signal and the threshold it failed, which is the receipt.
- Economy floor cross-checked against the value pinned in the runtime (`percept/slo.py` `FOVEA_CALL_REDUCTION_FLOOR = 0.5`).
- omnipro ground truth is zero-width instants; widened by a declared 1.0 s tolerance and tagged `gt_kind=instant` so it never enters an interval bar unannounced.
