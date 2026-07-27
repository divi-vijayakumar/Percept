# SV-AUDIO — audio stage validation

Generated: 2026-07-27T02:56:31Z
Runtime under test: `percept 1.0.0` at `/private/tmp/claude-501/-Users-dhivyavijayakumar-git-Percept-sdk/3d8c8120-cb64-4426-8af7-8eb60fc4a085/scratchpad/audiovenv/lib/python3.12/site-packages/percept/__init__.py`

**PASS 4 · FAIL 3 · ERROR 0**

| Test | Status | Checks | Title |
|---|---|---|---|
| `SV-A1.6R` | **FAIL** | 6/9 | VAD / acoustic admission recall against real audio ground truth (retry of the refused SV-1.6) |
| `SV-A2.1` | **PASS** | 11/11 | Audio senses — determinism and the one-energy-quantity contract |
| `SV-A2.2` | **PASS** | 6/6 | Audio capture on real media — byte determinism and economy |
| `SV-A4.1` | **PASS** | 6/6 | Audio cognition — rising edge, refractory, distinct Incidents |
| `SV-A4.7` | **PASS** | 8/8 | Audio evidence integrity — an Incident resolves to the exact PCM fed |
| `SV-A4.8` | **FAIL** | 4/6 | Audio drop reconciliation — units in == admitted + suppressed + receipted |
| `SV-A4.9` | **FAIL** | 4/5 | Audio judge boundedness — what bounds the compiled audio dispatch |

## The corpus

**53 audio-bearing cases**, seeded selection (`seed=20260726`), every media file ffprobe-verified to carry an audio stream.

| Source | Built | Ground-truth strength | What the positives are |
|---|---|---|---|
| `perception_test_sound` | 40/40 | `audio_annotated` | human-annotated SOUND intervals (`sound_localisation`) |
| `omnipro_audio` | 3/10 | `audio_annotated_point` | instants a speech/sound-carried event became answerable |
| `charades_audible` | 10/10 | `proxy` | VISUAL action intervals whose label names an audible event |

Refused sources:

- `proactiveqa` — no audio stream — cannot carry audio ground truth (0/8 probed files carry an audio stream)
- `repcount` — no audio stream — cannot carry audio ground truth (0/8 probed files carry an audio stream)

Ground-truth defence (the three defect classes already found in the video corpus):

- **units** — Perception Test `sound_localisation` timestamps are MICROSECONDS. Verified: max `|timestamps/1e6 - frame_ids/frame_rate|` = **1e-06 s** over 10184 interval bounds.
- **containment** — 0 intervals past the end of the media (dropped, not clamped silently); every retained interval was re-checked against the ffprobe audio duration.
- **seeding** — selection is `random.Random(20260726).shuffle` over a **sorted** candidate list, never filesystem order.

## SV-A1.6R — VAD / acoustic admission recall against real audio ground truth (retry of the refused SV-1.6)

Status: **FAIL**

| Check | Result | Measured |
|---|---|---|
| ground truth exists: >= 20 interval-annotated audio cases | PASS | 40 cases with human-annotated SOUND intervals (+3 point-annotated) |
| timestamp UNITS verified, not assumed (microseconds vs frame_ids/frame_rate) | PASS | max \|timestamps/1e6 - frame_ids/frame_rate\| = 1e-06 s over 10184 interval bounds |
| every ground-truth interval falls inside its media's audio duration | PASS | all 166 intervals inside the media |
| every selected media file HAS an audio stream (ffprobe-probed, never assumed) | PASS | 43 cases probed, 0 without audio |
| the golden `perception-test-acoustic` track's case anchors are SOUND events, not silence | **FAIL** | golden_sound_events() returned 2 anchor(s); 2 carry a non-sound label: [('video_8241', 'Muted'), ('video_9253', 'Muted')] |
| the SHIPPED audio gate admits at least one annotated sound event | **FAIL** | AcousticOnset(high=0.09, low=0.0225) woke 0 time(s) across 3973 capture chunks and matched 0/204 annotated sound events (recall 0.0); corpus energy p99=0.00974, max=0.0277 |
| a threshold exists where the gate has non-zero recall AND spends < 25% of chunks | PASS | 11/14 swept thresholds are on a usable frontier; best recall 0.461 at high=5e-05 (economy 0.0468) |
| utterance_threshold is an AMPLITUDE knob squared into mean-square energy | PASS | utterance_threshold=0.01 reaches EnergyVad as energy 0.0001 — two orders of magnitude below the number a reader sees |
| the shipped utterance endpointer endpoints >= 1 utterance per minute of speech material | **FAIL** | utterance_threshold=0.01 -> speech_fraction 0.833, 3 endpoint(s) over 400s (0.45/min) on the speech-bearing slice |

> Recall is scored against the ANY-SOUND ground truth (every annotated sound segment in the clip, merged): an energy-threshold gate cannot distinguish one sound label from another, so scoring it against a single label would measure the wrong thing. Label-specific recall at the same operating point is reported beside it.

> An admission counts as a hit when a wake instant lands within 0.25s of the annotated interval — one capture chunk, because an onset is reported at the chunk's END.

> No default is proposed. The frontier is reported; picking an operating point needs a consequence model (miss cost vs. spend) this suite does not have.

> The speech slice is SMALL — 3 case(s), 400s. The local OmniPro media is the benchmark's VISUAL subset (19 of 245 videos), so only a handful of rows with an audio-carried trigger have media on disk. Read the endpoint counts as an order of magnitude, not a rate; the speech-fraction column is the load-bearing measurement and it is stable across the slice.

### VAD / onset threshold frontier

Corpus: 987.3 s of annotated audio, 204 annotated sound events. Admission tolerance 0.25 s (one capture chunk at `chunk_s=0.25`).

| `audio_wake.high` (energy) | `low` | recall | hits/positives | wakes | economy (wakes/chunk) |
|---|---|---|---|---|---|
| 0.09 **(shipped default)** | 0.0225 | 0.000 | 0/204 | 0 | 0.0000 |
| 0.05 | 0.0125 | 0.000 | 0/204 | 0 | 0.0000 |
| 0.03 | 0.0075 | 0.000 | 0/204 | 0 | 0.0000 |
| 0.01 | 0.0025 | 0.069 | 14/204 | 14 | 0.0035 |
| 0.005 | 0.00125 | 0.108 | 22/204 | 28 | 0.0070 |
| 0.003 | 0.00075 | 0.191 | 39/204 | 43 | 0.0108 |
| 0.001 | 0.00025 | 0.294 | 60/204 | 72 | 0.0181 |
| 0.0003 | 7.5e-05 | 0.368 | 75/204 | 112 | 0.0282 |
| 0.0001 | 2.5e-05 | 0.422 | 86/204 | 156 | 0.0393 |
| 5e-05 | 1.25e-05 | 0.461 | 94/204 | 186 | 0.0468 |
| 3e-05 | 7.5e-06 | 0.446 | 91/204 | 185 | 0.0466 |
| 1e-05 | 2.5e-06 | 0.392 | 80/204 | 166 | 0.0418 |
| 3e-06 | 7.5e-07 | 0.299 | 61/204 | 137 | 0.0345 |
| 1e-06 | 2.5e-07 | 0.245 | 50/204 | 112 | 0.0282 |

Corpus chunk-energy distribution: p10 `1.46e-07`, p50 `3.98e-06`, p90 `0.000112`, p99 `0.00974`, max `0.0277`.

### `utterance_threshold` sweep (the saturation)

`utterance_threshold` is an **amplitude** knob; `_amplitude_threshold_to_energy` squares it before it reaches `EnergyVad` (`0.01` → energy `0.0001`). `hangover_frames=8` at `chunk_s=0.25` means an endpoint needs **2.00 s of continuous silence**.

**speech_slice**

| `utterance_threshold` | energy | speech fraction | endpoints | endpoints/min |
|---|---|---|---|---|
| 0.3 | 0.09 | 0.165 | 3 | 0.45 |
| 0.2 | 0.04 | 0.398 | 0 | 0.00 |
| 0.1 | 0.01 | 0.668 | 1 | 0.15 |
| 0.05 | 0.0025 | 0.721 | 4 | 0.60 |
| 0.03 | 0.0009 | 0.748 | 3 | 0.45 |
| 0.02 | 0.0004 | 0.761 | 2 | 0.30 |
| 0.01 **(shipped)** | 0.0001 | 0.833 | 3 | 0.45 |
| 0.005 | 2.5e-05 | 0.930 | 2 | 0.30 |
| 0.003 | 9e-06 | 0.959 | 2 | 0.30 |
| 0.001 | 1e-06 | 0.990 | 0 | 0.00 |

**sound_slice**

| `utterance_threshold` | energy | speech fraction | endpoints | endpoints/min |
|---|---|---|---|---|
| 0.3 | 0.09 | 0.000 | 0 | 0.00 |
| 0.2 | 0.04 | 0.000 | 0 | 0.00 |
| 0.1 | 0.01 | 0.010 | 1 | 0.06 |
| 0.05 | 0.0025 | 0.025 | 13 | 0.79 |
| 0.03 | 0.0009 | 0.044 | 17 | 1.03 |
| 0.02 | 0.0004 | 0.066 | 21 | 1.28 |
| 0.01 **(shipped)** | 0.0001 | 0.104 | 27 | 1.64 |
| 0.005 | 2.5e-05 | 0.210 | 37 | 2.25 |
| 0.003 | 9e-06 | 0.341 | 30 | 1.82 |
| 0.001 | 1e-06 | 0.720 | 17 | 1.03 |

## SV-A2.1 — Audio senses — determinism and the one-energy-quantity contract

Status: **PASS**

| Check | Result | Measured |
|---|---|---|
| pcm_energy is bit-exactly repeatable on real PCM | PASS | 24 real capture chunks x3 passes, identical float results |
| pcm_energy == runtime.frame_energy on the same signal (the documented single quantity) | PASS | max \|pcm_energy - frame_energy\| = 0 over 24 real chunks |
| pcm_energy is mean-square, NOT RMS (full-scale square wave -> 1.0) | PASS | full-negative-scale energy = 1.0, full-positive-scale = 0.9999389657750726 |
| pcm_energy rejects a misaligned s16le payload instead of guessing | PASS | odd-length s16le PCM raises ValueError |
| normalize_audio_unit derives the interval from the PAYLOAD's true duration | PASS | [9.75, 10.0] for a 0.25s payload anchored at t=10.0 |
| normalize_audio_unit is deterministic and computes the missing energy from the PCM | PASS | energy 2.64233467169106e-05 == pcm_energy(payload) 2.64233467169106e-05 |
| canonicalize_audio_unit COPIES the payload (a later mutation cannot rewrite evidence) | PASS | the caller's buffer was zeroed after canonicalization; the canonical PCM is unchanged |
| canonicalization is deterministic (same bytes in -> same judge token out) | PASS | the frozen provider token is a pure function of the canonical PCM |
| EnergyVad: same energy sequence -> identical endpoints, and reset() re-arms exactly | PASS | 0 endpoints on 66 real chunks; streaming == reset-replay == batch |
| EnergyVad refuses an out-of-range energy instead of clamping silently | PASS | energy outside [0, 1] and non-finite energy both raise ValueError |
| EnergyVad rejects a non-positive hangover/min-speech configuration | PASS | the three constructor bounds all fail closed |

## SV-A2.2 — Audio capture on real media — byte determinism and economy

Status: **PASS**

| Check | Result | Measured |
|---|---|---|
| re-capturing the same media yields BYTE-IDENTICAL PCM and identical energies | PASS | 5 files captured twice; 0 differed |
| captured units are contiguous and sequentially numbered (no silent gap) | PASS | 483 chunks across 5 files, 0 file(s) with a gap or sequence jump |
| every full capture chunk spans exactly the shipped chunk_s (tail chunk may be shorter) | PASS | interior chunk durations [0.25] against chunk_s=0.25; 0 file(s) with an over-long tail chunk |
| one second of listening costs exactly one second of 16 kHz mono s16 PCM | PASS | observed [32000] B/s against 32000 B/s (16000 Hz x 1 ch x 2 B) |
| capture keeps up with real time by a wide margin (>= 20x) | PASS | slowest observed 468x realtime across 5 files |
| the retention window is bounded by seconds AND by unit/byte caps | PASS | AudioWindow(4.0) -> max_units=4096, max_bytes=16777216; a 4 s window at 32000 B/s holds 128000 B, 0.76% of the byte cap |

## SV-A4.1 — Audio cognition — rising edge, refractory, distinct Incidents

Status: **PASS**

| Check | Result | Measured |
|---|---|---|
| every acoustic wake reaches the judge exactly once | PASS | 4 wakes -> escalations=4, judge_calls=4, perceive_dispatches=4 |
| the audio gate fires on the RISING EDGE and only there | PASS | fired at [0.75, 2.75]; gate reasons ['rising_edge', 'not_satisfied', 'rising_edge', 'not_satisfied'] |
| a held-true audio occurrence fires exactly ONCE | PASS | 1 incident(s); gate reasons ['rising_edge', 'already_fired', 'already_fired', 'already_fired'] |
| a repeated audio occurrence yields a DISTINCT Incident | PASS | 2 incidents; 2 distinct ids, 2 distinct judgment_event_ids, 2 distinct evidence_seqs, 2 distinct evidence hashes |
| refractory suppresses an audio re-fire inside the window, with a typed receipt | PASS | refractory_s=2.5: fired at [0.75], suppressed [('refractory', 2.75)] |
| the second occurrence WOULD have fired without refractory (the suppression is the cause) | PASS | same wakes and verdicts: 2 fires at refractory_s=0.0, 1 at refractory_s=2.5 |

## SV-A4.7 — Audio evidence integrity — an Incident resolves to the exact PCM fed

Status: **PASS**

| Check | Result | Measured |
|---|---|---|
| the media under test is the exact file the corpus pinned | PASS | eval/bench-staging/perception_test/raw/videos/video_10015.mp4 sha256 c3c55abc8dce0119... matches the corpus pin |
| the real-audio Watch produced at least one Incident to audit | PASS | 1 incident(s) from 12 real capture chunks of perception_test_sound::video_10015::0 |
| the Incident carries an AUDIO snippet, not a substituted modality | PASS | snippet.kind=audio |
| every evidence frame's PCM hashes to the EXACT bytes that were fed | PASS | 9 evidence frames; hashes match; seqs [0, 1, 2, 3, 4, 5, 6, 7, 8] |
| the concatenated evidence is byte-identical to the concatenated capture | PASS | no re-encode, no resample, no substitution between capture and Incident |
| the evidence interval equals the real media interval it was taken from | PASS | snippet [0.0, 2.25] vs capture [0.0, 2.25] |
| what the judge was shown and what the Incident carries are the SAME bytes | PASS | provider payload is audio/wav (72044 B); its PCM data chunk matches the retained evidence (72000 B) at 16000 Hz x 1 ch |
| evidence energy is the real signal, not a placeholder | PASS | evidence energies [0.0, 2e-08, 4.8e-07, 4e-07, 1.58e-06, 3.9e-06]... |

## SV-A4.8 — Audio drop reconciliation — units in == admitted + suppressed + receipted

Status: **FAIL**

| Check | Result | Measured |
|---|---|---|
| every malformed carrier is dropped WITH a typed receipt | PASS | 2 malformed carriers -> 2 audio_unit_rejected event(s) |
| accepted units are counted exactly once | PASS | 12 valid units, 5 suppressed -> audio_chunks=7 |
| admissions reconcile with judge dispatches and goal admissions | PASS | escalations=2, judge_calls=2, perceive_dispatches=2, goal_admissions=2 |
| the end-of-run coverage report accounts for every unit offered to the runtime | **FAIL** | 14 units offered; audio_chunks=7 + receipts=2 = 9; 5 unit(s) vanished with no counter and no health event. runtime_ended reports {'observed': 7, 'excluded': 0, 'discontinuities': 0, 'missing_sequences': 0} |
| mute/pause suppression emits a receipt naming how many units it swallowed | **FAIL** | health kinds observed: ['audio_unit_rejected', 'judge_dispatch', 'runtime_ended', 'runtime_started'] |
| evidence trimmed by the provider byte cap is dropped WITH a receipt | PASS | 1 audio_evidence_capped event(s): [1] |

## SV-A4.9 — Audio judge boundedness — what bounds the compiled audio dispatch

Status: **FAIL**

| Check | Result | Measured |
|---|---|---|
| a JudgePool that DECLARES the 'audio' evidence shape can back a compiled audio Watch | **FAIL** | JudgePool declares evidence_shapes=['audio', 'fused', 'single_image'] yet prepare_audio_judge_binding raised [ValueError: compiled audio requires an exact SDK-owned FakeVision or GeminiVision adapter] and apply_watch_plan returned 'rejected'; the compiled audio path accepts only ['FakeVision', 'GeminiVision'] |
| the audio path is bounded by explicit byte caps on input and provider payload | PASS | AudioInputCapability(max_input_bytes=16777216, max_provider_bytes=16777216) — 524s of 16 kHz mono s16 audio |
| the retained audio timeline is bounded by seconds, unit count AND bytes | PASS | AudioWindow(seconds=4.0, max_units=4096, max_bytes=16777216) |
| compiled audio at the DEFAULT judge_dispatch='async' gets a single-flight scheduler | PASS | in-flight and pending are single scalar slots (J<=1, P<=1 by construction), with stall/backpressure/deadline defaults {'stall_after_s': 20.0, 'backpressure_after': 3, 'decision_deadline_s': None} |
| judge_dispatch='inline' makes the audio capture loop WAIT on cognition (documented, not a bug) | PASS | 2 admissions x 0.25s judge latency stalled the pump for 0.50s — the inline path has no deadline; the bound comes only from judge_dispatch='async' |

## Defects

### `GOLDEN-ACOUSTIC-MUTED-ANCHOR` — high

**Where:** `packages/percept-vision/eval/bench/perception_test.py:197`

golden_sound_events() takes segments[0] (_event_from_row, perception_test.py:172-190) as each acoustic row's ground-truth event. For both rows in the golden `perception-test-acoustic` track (video_8241, video_9253) segment 0 is labelled 'Muted' — the annotators' marker for a stretch with NO audible source. label_phrase then renders it as 'right now there is an audible muted sound or event', so the track's recall / fusion_fp / recall_flip metrics would be scored against SILENCE. The valid-split annotations carry no 'Muted' label, so only the checked-in train-split sample.json rows are affected — which is exactly what the golden manifest pins. This corpus filters NON_SOUND_LABELS instead.

**Repro:** `python -c "import importlib.util,sys;s=importlib.util.spec_from_file_location('pt','packages/percept-vision/eval/bench/perception_test.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m);print([(e['video_id'],e['label']) for e in m.golden_sound_events()])"`

### `AUDIO-GATE-DEAF` — blocker

**Where:** `percept/harness/detectors.py:233`

AcousticOnset's shipped thresholds (high=0.09, low=0.0225) are mean-square ENERGY, and no capture chunk in 40 annotated clips (3973 chunks, 987s) reaches 0.09: corpus max energy is 0.0277. run(audio_gate='acoustic-onset') with default audio_wake therefore NEVER admits on this material — an audio Watch is silently deaf, with no health event to say so.

**Repro:** `eval/stage-validation/sv_audio.py --only SV-A1.6R (check: 'the SHIPPED audio gate admits at least one annotated sound event')`

### `AUDIO-VAD-HANGOVER-UNITS` — high

**Where:** `percept/runtime/_vad.py:54`

EnergyVad's defaults are dimensionless FRAME counts (hangover_frames=8, min_speech_frames=3) ported from percept-edge/src/vad.js, but the Python runtime feeds it capture chunks of AudioSourceConfig.chunk_s=0.25s (percept/runtime/audio.py:56). The effective hangover is therefore 2.00s of CONTINUOUS silence and the minimum utterance 0.75s. Combined with utterance_threshold=0.01 arriving as energy 1e-4, continuous narration reads as one unbroken speech run and endpoints essentially never — the saturation observed on omnipro, reproduced here.

**Repro:** `eval/stage-validation/sv_audio.py --only SV-A1.6R (measurements.utterance_frontier)`

### `AUDIO-SUPPRESSED-NO-RECEIPT` — high

**Where:** `percept/runtime/worker.py:2760`

In the compiled audio-only path, a unit arriving while muted()/paused() is `continue`d with no counter increment and no RuntimeHealthEvent. 5 of 14 units offered in this run are therefore invisible: RuntimeStats has no audio-suppressed field (only frames_dropped, which is the video path), SourceTracker never sees them (runtime_ended source_coverage reports excluded=0), and no health event names them. Units in != admitted + suppressed + dropped-with-receipt: the suppressed term has no term.

**Repro:** `eval/stage-validation/sv_audio.py --only SV-A4.8 (check: 'the end-of-run coverage report accounts for every unit offered')`

### `AUDIO-JUDGEPOOL-REJECTED` — medium

**Where:** `percept/audio_input.py:1036`

prepare_audio_judge_binding accepts only an exact percept.backends.fake.FakeVision or percept.backends.gemini.GeminiVision instance, so a JudgePool — including one whose JudgeCapability explicitly declares the 'audio' and 'fused' evidence shapes (percept/backends/judge_pool.py:35) — is rejected and apply_watch_plan returns 'rejected'. The compiled audio path therefore never inherits the pool's per-route capacity (max_inflight), latency SLO/deadline, hedging or JudgeRouteReceipts. It is NOT unbounded: the byte caps, the AudioWindow caps and the default judge_dispatch='async' single-flight scheduler bound it — but the bound is the runtime's, not the pool's, and there is no receipt trail per route.

**Repro:** `eval/stage-validation/sv_audio.py --only SV-A4.9`
