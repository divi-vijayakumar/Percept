# Stage-2 LOCAL PERCEPTION validation — SV-2.5 / SV-2.6 / SV-2.7 + counting harness

- generated: `2026-07-26T15:42:43-0700` (wall 4.7s)
- runtime under test: `percept 1.0.0` @ `/private/tmp/percept-launch/oss-final-venv-bba572a/lib/python3.12/site-packages/percept/__init__.py`
- runtime is the repo tree (MUST be false): `False`
- `tools.senseprobe` available: **True**
- provider/network calls: **none** (all judges scripted, all signals synthetic or corpus-ground-truth derived; media is decoded locally with cv2 only)

## Verdicts

| test | verdict | checks | title |
|---|---|---|---|
| SV-2.5 | **PASS** | 13/13 | Reducer determinism (exact) |
| SV-2.6 | **PASS** | 11/11 | Temporal object continuity (occlusion / revise-not-duplicate) |
| SV-2.7 | **PASS** | 10/10 | Stage-2 economy (units in from Stage 1 / units out to Stage 3) |
| SV-2.1-groundwork | **PASS** | 8/8 | Counting / rhythm measurement harness (20 repcount cases) |

## SV-2.5 — Reducer determinism (exact)

|  | check | detail |
|---|---|---|
| PASS | repeated in-process runs are byte-identical |  |
| PASS | cross-process trace is byte-identical (3 subprocesses, 3 hash seeds) |  |
| PASS | latest_only: newest admitted seq applies, every earlier one drops as 'superseded_by_newer_admission' |  |
| PASS | a DROPPED decision is never FIRE/ASK (action pinned to SILENT) |  |
| PASS | a DROPPED decision never advances last_applied (gate untouched) |  |
| PASS | episode_ordered: a straggler from an EARLIER episode is NOT penalised by a newer episode's higher admission |  |
| PASS | episode_ordered: supersession WITHIN an episode still drops ('superseded_within_episode') |  |
| PASS | episode_ordered with episode_id=None falls back to key-wide latest_only |  |
| PASS | stale_after_ms drops an aged result independently of supersession ('stale_timeout') |  |
| PASS | apply_after_supersession=True applies the would-be-drop and FLAGS it |  |
| PASS | concurrent same-key submits serialize: exactly the newest admission applies |  |
| PASS | forget() resets the per-key counter to 1 and re-admits cleanly |  |
| PASS | EvidenceSequencer admit/tracked_keys is deterministic and insertion-ordered |  |

### Declared out-of-order policy (as read from the runtime, not assumed)

`percept/sequencing.py:63-80` (`ResultPolicy`) and `percept/sequencing.py:242-260` (`_stale_reason`) declare:

- `ordering="latest_only"` (default) — a result is stale the instant a NEWER job has been **admitted** for the key (`evidence_seq < latest_admitted_sequence`), not merely applied. Drop reason `superseded_by_newer_admission`.
- `ordering="episode_ordered"` — supersession is scoped to `episode_id`; drop reason `superseded_within_episode`. With `episode_id=None` it falls back to the key-wide `latest_only` comparison.
- `stale_after_ms` — an independent age drop (`stale_timeout`), checked FIRST.
- `apply_after_supersession=True` — the would-be drop is applied anyway and flagged `applied_after_supersession=True` (with `drop_reason` retained).
- A dropped decision is always `action=SILENT`, `dropped=True`, and never touches the gate.

- determinism digest (sha256 of the canonical decision trace): `992d74bbd670b67439ad4f678e8e3690d0271e10ba6c5bc56ea2679fe07b20b9`
- in-process runs: 5, distinct digests: 1
- subprocess runs (PYTHONHASHSEED 0/1/12345): 3, distinct digests: 1

## SV-2.6 — Temporal object continuity (occlusion / revise-not-duplicate)

|  | check | detail |
|---|---|---|
| PASS | occlusion of 4.999999s (< continuity_gap_s=5.0) CONTINUES the same episode |  |
| PASS | the continuation COALESCES (no new revision row) and leaves exactly one episode |  |
| PASS | occlusion of EXACTLY continuity_gap_s (5.0s) SPLITS (inclusive boundary, percept/episodes.py:762) |  |
| PASS | occlusion just beyond the gap SPLITS and the split is TRACED (subject_matched + prior_episode_id + gap_s + denial reason) |  |
| PASS | a value change REVISES the single episode (one episode, kinds == [opened, updated]) |  |
| PASS | 500 in-policy confirmations produce exactly ONE revision row and ONE episode |  |
| PASS | PASS BAR: zero SILENT duplicates over 400 observations with mixed occlusions (every re-open of an already-seen subject carries subject_matched + prior_episode_id + gap_s + a DENIAL_REASONS code) |  |
| PASS | two distinct subjects never merge, and the first subject still continues its own episode |  |
| PASS | a terminal episode splits (TRACED 'episode_terminal') unless allow_reopen, in which case it REOPENS the same episode and clears t_end |  |
| PASS | the default EpisodePolicy (continuity_gap_s=None) is unbounded — a day-long occlusion still continues (documented default, percept/episodes.py:287) |  |
| PASS | boundary is exact to 1e-9 below continuity_gap_s (still continues) |  |

### Occlusion boundary

| case | observation t (s) | is_new | reason | subject_matched | prior_episode_id | gap_s |
|---|---|---|---|---|---|---|
| just within (gap-1e-6) | 4.999999 | False | continued | n/a | n/a | n/a |
| exactly_gap | 5.000000 | True | gap_exceeded | True | g-exactly_gap:ep1 | 5.0 |
| gap_plus_eps | 5.000001 | True | gap_exceeded | True | g-gap_plus_eps:ep1 | 5.000001 |

### Silent-duplicate bar

- observations folded: **400**
- episodes opened: **134**  (continuations: 266; first open reason: `new_subject`)
- duplicate opens (every open after the subject's first): **133**
- traced splits: **133**
- **SILENT duplicates: 0** (bar: 0)
- split reasons observed: `['gap_exceeded']`

### Revise-not-duplicate

- `{"episode_count": 1, "episode_ids": ["g-revise:ep1"], "revision_kinds": ["opened", "updated"], "observations": 9, "value": "half-full", "updated_row_is_Revision": true}`
- heartbeat boundedness: `{"confirmations": 500, "revision_rows": 1, "kinds": ["opened"], "episode_count": 1, "observations": 501}`

## SV-2.7 — Stage-2 economy (units in from Stage 1 / units out to Stage 3)

|  | check | detail |
|---|---|---|
| PASS | admit() alone applies NOTHING — last_applied stays 0 until a submit() is awaited (percept/sequencing.py:183-191) |  |
| PASS | AppearanceIdentity re-id is genuinely on the APPEARANCE path (matches across a zero-IoU gap with signature=, returns None without it; continuity control also None) |  |
| PASS | SpanAssociator type-safety is characterised (late-failure behaviour recorded, not asserted away) |  |
| PASS | every sense was measured through the real SenseProbe adapter |  |
| PASS | all 8 documented REGISTRY keys are present and were exercised |  |
| PASS | every sense compresses (units_out <= units_in) |  |
| PASS | no probe recorded an error |  |
| PASS | span association is idempotent under denser observation (3x candidates -> 1x events) |  |
| PASS | the Stage-2 funnel strictly narrows (frames >= reps >= spans >= episodes >= delivered) |  |
| PASS | Stage-2 delivers strictly fewer units to Stage 3 than it received from Stage 1 |  |

## SV-2.1-groundwork — Counting / rhythm measurement harness (20 repcount cases)

|  | check | detail |
|---|---|---|
| PASS | counting rule counts a SINGLE rep from cold start (no warm-up) |  |
| PASS | counting rule is rate-agnostic on a clean signal (6/12/30/60 Hz all give 8) |  |
| PASS | dominant_period recovers the true period in SECONDS (within one grid step) |  |
| PASS | periodicity_score is high and the period survives decimation (stability >= 0.9) |  |
| PASS | all 20 repcount cases resolved their media and their own fps |  |
| PASS | fps is read per file, not assumed constant (>1 distinct fps in the corpus) |  |
| PASS | ground truth is self-consistent: len(cycle_bounds_frames)/2 == expected_count for all 20 |  |
| PASS | every case produced a numeric prediction from both counting rules at every rate |  |

## SV-2.7 — per-sense economy

`units_in` = units received from Stage 1; `units_out` = units passed on to Stage 3.

| sense | probe | units_in | units_out | out/in | probe.units_fed | probe.reads | wall_s | errors | meaning |
|---|---|---|---|---|---|---|---|---|---|
| rhythm | senseprobe | 600 | 30 | 0.0500 | 600 | 1 | 0.0014 | 0 | eye frames observed -> confirmed reps on the selected channel |
| entity | senseprobe | 1200 | 2 | 0.0017 | 600 | 1 | 0.0021 | 0 | detections ingested -> distinct tracked entities handed up |
| span | senseprobe | 120 | 20 | 0.1667 | 120 | 1 | 0.0002 | 0 | raw event candidates -> deduped event spans (denser observation adds no events) |
| sequencer | senseprobe | 150 | 8 | 0.0533 | 150 | 1 | 0.0000 | 0 | admissions -> distinct per-key counters carried (latest_admitted for the last key fed = 19) |
| reducer | senseprobe | 150 | 81 | 0.5400 | 150 | 0 | 0.0001 | 0 | judge results submitted (via .sense.submit) -> decisions DELIVERED to Stage 3 (FIRE / ASK) |
| continuity_id | senseprobe | 200 | 1 | 0.0050 | 200 | 0 | 0.0001 | 0 | match() calls -> distinct ids minted (199 re-identified, 1 new) |
| appearance_id | senseprobe | 200 | 1 | 0.0050 | 200 | 0 | 0.0001 | 0 | match() calls -> distinct ids minted (199 re-identified, 1 new) |
| extractor | senseprobe | 600 | 60 | 0.1000 | 600 | 0 | 0.0004 | 0 | frames offered -> non-empty Enrichments forwarded as judge context |

### End-to-end Stage-2 funnel (one pipeline, 600 synthetic eye frames @ 10 Hz)

| stage | units |
|---|---|
| Stage-1 frames in | 600 |
| rhythm reps | 30 |
| span events (deduped) | 30 |
| episodes opened | 1 |
| episode continuations (folded into an open episode, not re-delivered) | 29 |
| Stage-3 delivered (FIRE / ASK) | 1 |

- overall Stage-2 economy ratio (Stage-3 out / Stage-1 in): **0.0016666666666666668**

The tail is narrow by design, not by accident: all 30 reps belong to ONE subject inside one `continuity_gap_s`, so 29 of them CONTINUE the open episode rather than minting rivals, and the gate's rising-edge rule delivers the episode once. That is the Stage-2 job — 600 frames of Stage-1 evidence become 1 unit of Stage-3 attention.

## SV-2.1 groundwork — rep-count measurement (20 real `repcount` cases)

Ground truth is the corpus' own `provenance.expected_count`; `provenance.cycle_bounds_frames` are FRAME INDICES converted to seconds with each media file's OWN fps (read via cv2 — the corpus spans 23.976 / 25 / 29.97 / 30 fps, so a constant would have been wrong). **No pass bar is asserted here** — these are raw measurements for the pre-registration agent.

Two counting rules are driven:

- **`count_edges`** — the candidate-source refractory rule (`packages/percept-vision/eval/exp_repcount.py:55`), reproduced in this file (`sv2_senses.py:count_edges`); refractory = 0.5 x the case's own mean cycle length.
- **`RhythmTracker`** — the shipped sense (`percept/rhythm.py`), run at library defaults.

Signals:

- *oracle signal* — a scripted trough->peak->trough arch (and its binary up/down phase) laid on the REAL annotated cycle bounds. This isolates the COUNTING RULE from the eye.
- *+noise* — the same, plus deterministic Gaussian noise (seeded).
- *crude pixel eye* — a local, model-free motion-energy/centroid front end over the decoded pixels. Included only as an honest end-to-end datapoint; it is **not** the shipped perception path.

### Error distribution (|predicted - expected_count|)

| configuration | n | MAE | exact (err=0) | err&le;1 | median | p90 | max | histogram of abs err |
|---|---|---|---|---|---|---|---|---|
| count_edges@3Hz | 20 | 0.00 | 20/20 | 20/20 (100.0%) | 0.0 | 0 | 0 | {"0": 20} |
| RhythmTracker@3Hz | 20 | 1.40 | 14/20 | 17/20 (85.0%) | 0.0 | 7 | 9 | {"0": 14, "1": 3, "2": 0, "3": 0, "4": 0, "5": 0, "6": 0, "7": 1, "8": 0, "9": 2} |
| count_edges@6Hz | 20 | 0.00 | 20/20 | 20/20 (100.0%) | 0.0 | 0 | 0 | {"0": 20} |
| RhythmTracker@6Hz | 20 | 0.05 | 19/20 | 20/20 (100.0%) | 0.0 | 0 | 1 | {"0": 19, "1": 1} |
| count_edges@30Hz | 20 | 0.05 | 19/20 | 20/20 (100.0%) | 0.0 | 0 | 1 | {"0": 19, "1": 1} |
| RhythmTracker@30Hz | 20 | 0.05 | 19/20 | 20/20 (100.0%) | 0.0 | 0 | 1 | {"0": 19, "1": 1} |
| RhythmTracker@6Hz+noise0.05 | 20 | 1.70 | 7/20 | 12/20 (60.0%) | 1.0 | 4 | 9 | {"0": 7, "1": 5, "2": 5, "3": 0, "4": 1, "5": 0, "6": 1, "7": 0, "8": 0, "9": 1} |
| RhythmTracker@6Hz+noise0.15 | 20 | 3.45 | 5/20 | 10/20 (50.0%) | 1.5 | 11 | 14 | {"0": 5, "1": 5, "2": 3, "3": 1, "4": 2, "5": 0, "6": 0, "7": 0, "8": 0, "9": 1, "10": 0, "11": 1, "12": 0, "13": 1, "14": 1} |
| RhythmTracker@10Hz+crude-pixel-eye | 20 | 4.05 | 1/20 | 4/20 (20.0%) | 3.5 | 6 | 12 | {"0": 1, "1": 3, "2": 3, "3": 3, "4": 3, "5": 2, "6": 3, "7": 0, "8": 0, "9": 0, "10": 0, "11": 1, "12": 1} |

### Per case

| case_id | action | fps | dur_s | gt | mean_cycle_s | edges@6Hz | err | rhythm@6Hz | err | rhythm@30Hz | err | rhythm+noise.15@6Hz | err | pixel_eye | err |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| repcount::stu6_15::0 | jump_jack | 23.976 | 15.0 | 12 | 1.00 | 12 | 0 | 12 | 0 | 12 | 0 | 14 | 2 | 1 | 11 |
| repcount::stu9_34::0 | jump_jack | 25.000 | 12.0 | 8 | 1.26 | 8 | 0 | 8 | 0 | 8 | 0 | 9 | 1 | 5 | 3 |
| repcount::stu4_63::0 | squat | 30.000 | 35.0 | 14 | 1.43 | 14 | 0 | 14 | 0 | 14 | 0 | 18 | 4 | 8 | 6 |
| repcount::stu4_12::0 | front_raise | 29.970 | 16.0 | 4 | 3.59 | 4 | 0 | 4 | 0 | 4 | 0 | 5 | 1 | 6 | 2 |
| repcount::test2025::0 | situp | 30.000 | 10.0 | 5 | 1.77 | 5 | 0 | 5 | 0 | 5 | 0 | 5 | 0 | 5 | 0 |
| repcount::stu8_49::0 | situp | 30.000 | 60.0 | 9 | 5.71 | 9 | 0 | 9 | 0 | 9 | 0 | 22 | 13 | 4 | 5 |
| repcount::stu6_13::0 | front_raise | 25.000 | 25.0 | 9 | 2.65 | 9 | 0 | 9 | 0 | 9 | 0 | 10 | 1 | 7 | 2 |
| repcount::stu7_13::0 | bench_pressing | 29.970 | 60.0 | 11 | 4.71 | 11 | 0 | 11 | 0 | 11 | 0 | 25 | 14 | 17 | 6 |
| repcount::stu7_17::0 | front_raise | 29.970 | 16.0 | 5 | 3.00 | 5 | 0 | 5 | 0 | 5 | 0 | 6 | 1 | 1 | 4 |
| repcount::stu9_45::0 | pull_up | 30.000 | 33.0 | 12 | 2.10 | 12 | 0 | 12 | 0 | 12 | 0 | 16 | 4 | 15 | 3 |
| repcount::stu2_6::0 | bench_pressing | 25.000 | 33.0 | 4 | 5.54 | 4 | 0 | 3 | 1 | 3 | 1 | 13 | 9 | 5 | 1 |
| repcount::stu3_57::0 | situp | 29.970 | 26.0 | 19 | 1.21 | 19 | 0 | 19 | 0 | 19 | 0 | 21 | 2 | 18 | 1 |
| repcount::stu10_48::0 | push_up | 29.970 | 30.0 | 5 | 1.96 | 5 | 0 | 5 | 0 | 5 | 0 | 16 | 11 | 7 | 2 |
| repcount::test538::0 | frontraise | 30.000 | 10.0 | 5 | 1.73 | 5 | 0 | 5 | 0 | 5 | 0 | 5 | 0 | 11 | 6 |
| repcount::stu2_62::0 | situp | 30.000 | 61.0 | 19 | 3.03 | 19 | 0 | 19 | 0 | 19 | 0 | 21 | 2 | 20 | 1 |
| repcount::stu2_34::0 | pull_up | 30.000 | 29.0 | 17 | 1.56 | 17 | 0 | 17 | 0 | 17 | 0 | 17 | 0 | 13 | 4 |
| repcount::stu5_27::0 | jump_jack | 30.000 | 21.0 | 14 | 1.02 | 14 | 0 | 14 | 0 | 14 | 0 | 17 | 3 | 2 | 12 |
| repcount::stu2_14::0 | front_raise | 23.976 | 30.0 | 7 | 3.52 | 7 | 0 | 7 | 0 | 7 | 0 | 8 | 1 | 12 | 5 |
| repcount::stu6_63::0 | squat | 29.970 | 37.0 | 17 | 1.75 | 17 | 0 | 17 | 0 | 17 | 0 | 17 | 0 | 14 | 3 |
| repcount::stu8_57::0 | situp | 29.970 | 13.0 | 7 | 1.83 | 7 | 0 | 7 | 0 | 7 | 0 | 7 | 0 | 3 | 4 |

### Measurement-layer invariants (`percept/measurement.py`)

- `{"dominant_period_s": 1.2333333333333334, "true_period_s": 1.25, "periodicity_score": 0.9359, "robust_amplitude": 0.9802, "subsample_stability": 1.0, "channel_score": 14.8108}`

### Counting-rule sanity on purely synthetic signals

| reps | period_s | obs_hz | count_edges | err | RhythmTracker | err |
|---|---|---|---|---|---|---|
| 1 | 1.5 | 30.0 | 1 | 0 | 1 | 0 |
| 5 | 1.0 | 30.0 | 5 | 0 | 5 | 0 |
| 12 | 0.8 | 30.0 | 12 | 0 | 12 | 0 |
| 8 | 2.5 | 6.0 | 8 | 0 | 8 | 0 |
| 8 | 2.5 | 12.0 | 8 | 0 | 8 | 0 |
| 8 | 2.5 | 30.0 | 8 | 0 | 8 | 0 |
| 8 | 2.5 | 60.0 | 8 | 0 | 8 | 0 |

## Observations (measurement, not verdicts)

These are raw readings from the harness, offered to whoever pre-registers the SV-2.1 bars. None of them is asserted here.

1. **Observation rate dominates `RhythmTracker`, not the cadence.** On the identical oracle signal it goes from MAE 1.4 (max |e| 9) at 3 Hz to MAE 0.05 at 6 Hz. The binary-edge rule is unaffected (`count_edges` MAE 0.0 at 3 Hz) because a rising edge survives coarse sampling while the trough->peak->trough *differential* the tracker needs does not. Corpus cadences run 0.79 s - 12.68 s per cycle, so 3 Hz is under two samples per half-cycle for the fastest cases. Any SV-2.1 bar must fix the observation rate.

2. **`RhythmTracker` at library defaults is noise-fragile.** Adding seeded Gaussian noise to the same oracle signal at 6 Hz takes MAE from 0.05 to 1.7 at sigma=0.05 and 3.45 at sigma=0.15 (5% and 15% of amplitude). `RhythmTracker.smooth_s` defaults to `0.0` — the causal low-pass that exists precisely for this is OPT-IN (`percept/rhythm.py:278-285` documents the trade-off and why it was left off). Numbers above are the DEFAULT config; the harness was not tuned to improve them.

3. **The eye is the dominant error source, not the counting rule.** With a perfect (oracle) signal the rule scores MAE 0.05 / 19/20 exact. With the crude local motion-centroid eye on the same 20 clips and the same tracker, MAE is 4.05 / 1/20 exact. That gap is the front end, not `rhythm.py`. (The crude eye is a floor, not the shipped path — it is here only to bound the question.)

4. **Ground truth is internally consistent.** For all 20 cases `len(provenance.cycle_bounds_frames) / 2 == provenance.expected_count`, so the bounds can be used as a per-cycle oracle without re-deriving the count.

5. **fps really does vary** across the corpus (23.976 / 25 / 29.97 / 30). Converting `cycle_bounds_frames` with a constant 30 fps would shift every 23.976 fps case's cycle timing by ~25%.

## Reproduce

```sh
/private/tmp/percept-launch/oss-final-venv-bba572a/bin/python \
    eval/stage-validation/sv2_senses.py            # full run (~4 s)
    #  --quick   skip the crude pixel-eye media pass
```

## Defects

- ROBUSTNESS (non-blocking): `SpanAssociator.add` validates nothing (`percept/individuation.py:133-147`). A malformed candidate is ACCEPTED on the first add (`count()` becomes 1) and only blows up on the SECOND add, when `_same_event` (`percept/individuation.py:104-116`) dereferences `cand.kind` — `AttributeError: 'str' object has no attribute 'kind'`. The error therefore surfaces one unit LATE and names the wrong call. No Stage-2 correctness property in SV-2.5/2.6/2.7 depends on it, so it is reported, not failed.

No CONTRACT defect was found. On every point SV-2.5 / SV-2.6 / SV-2.7 tested, the implementation matched the behaviour DECLARED in the runtime source — `percept/sequencing.py`, `percept/episodes.py`, `percept/rhythm.py`, `percept/measurement.py`, `percept/individuation.py`, `percept/identity.py`. The item above is a robustness/ergonomics gap, not a broken guarantee.

### Validation traps encountered (recorded so a later harness does not fall into them)

- `SerializedDecisionReducer.admit()` is synchronous and applies NOTHING; only the awaited coroutine `submit()` moves gate state (`percept/sequencing.py:183-191` vs `193-240`). A determinism test built on `admit()` alone would measure nothing and pass. SV-2.5 drives `submit()`; SV-2.7 asserts `last_applied == 0` after admits (see the `admit_applies_nothing` row in the JSON).
- `AppearanceIdentity.match()` silently degrades to pure continuity when `signature=None` (`percept/identity.py:72-87`), so a re-id number measured without a signature is a continuity number wearing a re-id label. SV-2.7 proves the appearance path is live before reporting: same observation across a zero-IoU gap returns `e-far` WITH a signature and `None` without, and the `ContinuityIdentity` control also returns `None`.
