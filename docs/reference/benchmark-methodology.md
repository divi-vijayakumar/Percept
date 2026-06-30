# Percept Benchmark Methodology Spec

Status: draft

> **No Percept accuracy claims yet.** Every number in this document is one of: an *external reference bar*
> (e.g. a TML-reported figure), an *illustrative scorecard-format example*, or a *proxy* with a stated
> release blocker. Percept's own measured results are **not published until the model-sweep phase** runs a
> reproducible, full-mode scorecard. Treat the figures here as methodology, not as benchmarks.

This spec replaces the current "one benchmark answers everything" framing. Percept's benchmark must run
the product path first: edge/conductor wakes candidate moments, the backend judges only candidates, and
the core emits `fire`/`ask`/`silent`.

Dense raw VLM loops are not an executable benchmark leg. They are too expensive, and for several tracks
they do not measure the actual task: RepCount-A is temporal-cycle counting, Charades is start/stop span
behavior, Perception Test is goal-conditioned notification, and ProactiveVideoQA is evidence-timed
answering. When a dense baseline is useful, estimate its call count from the sampling schedule or use an
empty/no-op probe to validate request overhead. Do not send video frames through a raw loop.

## Hypothesis

Percept is a cognition and control layer for ambient perception agents. With the edge/harness enabled,
it should reduce heavy-model calls by escalating only candidate moments while preserving enough event
recall and answer quality to remain useful.

The benchmark should support or reject these claims:

- H1, edge recall: cheap signals wake on enough true event/answer windows to support useful E2E behavior.
- H2, cost: Percept-E2E uses materially fewer heavy-model calls than a modeled dense sampling schedule.
- H3, product: Percept-E2E's final user-facing precision/recall is competitive enough for ambient
  assistant use cases at materially lower call volume.

The benchmark must not claim:

- Percept improves the raw visual acuity of Gemini/OpenAI/local VLMs.
- A backend reference number is a Percept product number.
- A cache with incomplete capture is a valid full-sample result.
- A live raw VLM loop is required for a publishable result.

## Configurations

Every track must say which configuration it measures.

| Config | What runs | What it measures | VLM calls |
| --- | --- | --- | --- |
| `dense-modeled` | No video frames sent; compute dense calls from fps/window schedule | Cost/call reference only | modeled |
| `edge` | Cheap edge/conductor only, no final VLM judgment | Candidate wake recall, calls/min, latency proxy | modeled/would-call |
| `e2e` | Edge selects candidates, backend judges candidates, Percept emits events | Product behavior and cost together | measured candidate calls |
| `core-replay` | Optional offline replay over already-captured E2E observations | Dedupe/ASK/no-nag regression | no new calls |

`edge` must not be presented as final answer quality because it has not run the VLM. `e2e` is the only
product-level lane. `core-replay` is allowed only when it reuses observations already captured by E2E;
it must not trigger additional backend calls.

## Validity Gates

A result is invalid if any of these are true:

- Capture coverage is below the track's threshold.
- Failed calls are cached as successful empty observations.
- A scorer silently swallows API failures without recording failure status.
- The manifest does not identify the exact sampled rows and media hashes.
- The result mixes `core-replay` cognition diagnostics with `edge/e2e` cost metrics as if they were the
  same run.
- The run executed dense raw video-frame VLM calls.
- The scorecard lacks commit, manifest hash, backend/model, sample size, and coverage.

Minimum capture metadata per row:

```json
{
  "row_id": "charades-LR9KL-Holding_a_vacuum",
  "status": "complete | incomplete | failed",
  "expected_observations": 31,
  "captured_observations": 31,
  "failed_observations": 0,
  "error_summary": null
}
```

Scorers must refuse a headline result when `captured_observations / expected_observations` is below the
declared threshold. For E2E candidate-call tracks, the default threshold is `>= 0.95` over expected
candidate judgments.

## Dataset Map

| Dataset | Shape | Primary use | License note | Percept processing |
| --- | --- | --- | --- | --- |
| Perception Test action | videos with action onset/end annotations | goal-conditioned action notification | CC-BY-4.0 | Label -> Percept goal -> WatchSpec -> edge wake -> VLM confirm -> fire/ask/silent |
| Perception Test sound | videos with audio event annotations | goal-conditioned sound notification | CC-BY-4.0 | Sound label -> Percept goal -> audio/edge wake -> optional confirm -> fire/ask/silent |
| Charades | home activity videos with many action intervals per video | multi-goal temporal notifications | non-commercial reference | Each action label becomes a Percept goal; one WatchSpec per video; score notification/spans per goal |
| RepCount-A | videos with repetition counts | unbiased counting | research dataset | Full action-set Percept E2E counting; no raw per-frame VLM baseline |
| ProactiveVideoQA | videos, questions, answer timing | proactive answer timing | dataset-specific | Question -> Percept concern; edge wakes on answer evidence; VLM answers candidates; core emits once/revises |
| CueSpeak/TimeSpeak prompts | authored interaction prompts with cue windows and expected utterances | proactive speech quality/timing | authored | Sim regression first; publishable track requires held-out prompts, judge, and timing windows |

## Track Specs

### Goal Writing Modes

Perception Test and Charades labels are not always natural user goals. Every goal-conditioned track must
declare how the label became a Percept goal.

| Mode | Input | Goal text | What it tests |
| --- | --- | --- | --- |
| `literal` | Dataset label | Direct label-to-goal wording | Can Percept execute the benchmark label as written? |
| `operationalized` | Dataset label + Percept goal writer | Observable, less ambiguous watch goal | Can the brain compile vague user intent into a better WatchSpec? |

Examples:

```text
specific label: "Opening a door"
literal goal:   "Notify me when someone is opening a door."

vague label:    "Touching something"
literal goal:   "Notify me when someone is touching something."
operationalized:
  "Notify me when a person's hand begins making deliberate contact with a visible object."

vague label:    "Moving something"
literal goal:   "Notify me when someone is moving something."
operationalized:
  "Notify me when a visible object starts changing position because a person is manipulating it."
```

Goal-writing is itself measured. For vague labels, the benchmark should report literal and
operationalized goal results separately. If the label is under-specified enough that a real assistant
would ask for clarification, Percept should prefer `ask` or an explicit "needs clarification" outcome
over silent overfitting.

Goal-writing metadata per row:

```json
{
  "dataset_label": "Moving something",
  "ambiguity": "low | medium | high",
  "goal_mode": "literal | operationalized",
  "goal_text": "Notify me when a visible object starts changing position because a person is manipulating it.",
  "detector_hints": ["motion", "change", "object_presence", "semantic_relevance"],
  "needs_clarification": false
}
```

### 1. Dense Call Model

Question: how many heavy-model calls would a dense strategy have made?

This lane does not send media to a VLM. It computes the call denominator from the declared sampling
policy.

```text
dense_calls = sum(sampled_frames_or_windows_per_goal)
e2e_calls = count(candidate backend judgments actually made)
call_reduction = 1 - e2e_calls / dense_calls
```

Metrics:

- dense modeled calls.
- E2E measured candidate calls.
- calls per video/minute.
- estimated cost from measured candidate calls plus modeled dense reference.
- optional empty/no-op API probe for request overhead only, with no media payload.

Expected outcome:

- Establish the cost denominator without burning VLM credits.
- Make call-reduction claims reproducible from the manifest, fps, windows, and goal count.

Actual outcome field:

```text
dense_modeled_calls=<N>, e2e_measured_calls=<N>, call_reduction=<%>,
fps=<value>, window_policy=<value>, empty_probe_ms=<optional>
```

This lane is diagnostic only. It is not answer quality.

### 2. Core Replay Lane

Question: given observations already captured by E2E, does Percept's core reduce duplicate/spurious
outputs?

How Percept is called:

```text
for each E2E-captured Verdict at candidate time t:
  Percept-Core: Gate.evaluate(goal_id, verdict, freshness)
  Percept-Event: merge/dedupe rising-edge episodes
```

Metrics:

- duplicate fires / fires per episode.
- interval precision/recall.
- mIoU/time-F1 from spans.
- ASK rate/refusal behavior.
- no new backend calls.

Expected outcome:

- Higher precision or fewer duplicate/spam outputs than naive emission over the same captured candidate
  observations.
- Cleaner spans/events.

Actual outcome field:

```text
e2e_calls=<N>, naive_candidate_fires=<N>, core_fires=<N>,
candidate_metric=<value>, core_metric=<value>, delta=<value>
```

### 3. Edge/Conductor Lane

Question: can cheap signals decide when it is worth asking the VLM?

How Percept is called:

```text
core compiles goals -> WatchSpec
MotionGate / WakePolicy evaluates each frame
record candidate wakes only; do not call the VLM
```

Metrics:

- event wake recall: fraction of labelled events with at least one wake in the event window.
- precision proxy: fraction of wakes inside labelled windows.
- calls/min: candidate wakes per minute.
- median wake latency from event onset.
- cost reduction versus the dense modeled call schedule.

Expected outcome:

- Motion-only may have low recall on subtle events.
- Weighted/fallback policy should improve recall while preserving meaningful call reduction.
- Precision proxy may be low before VLM filtering; this is acceptable only if e2e filtering later fixes
  final precision.

Actual outcome field:

```text
policy=<motion|weighted|fallback>, event_recall=<value>, calls_per_min=<value>,
cost_reduction=<value>, median_latency_s=<value>, precision_proxy=<value>
```

### 4. Product E2E Lane

Question: does the actual Percept architecture deliver useful final behavior at lower call volume?

How Percept is called:

```text
user goal(s)
  -> Percept core compiles WatchSpec
  -> edge/harness scans the stream
  -> VLM judges only escalated candidate frames/windows
  -> Percept gate emits fire/ask/silent
  -> scorer compares final events/spans to labels
```

Metrics:

- final precision/recall or mIoU/time-F1.
- VLM calls, calls/min, and cost estimate.
- median and p90 reaction latency.
- ASK rate and unresolved/uncertain rate.
- incident/snippet availability where relevant.

Expected outcome:

- VLM calls materially lower than the dense modeled call schedule.
- Recall and precision acceptable for the use case.
- Precision/output quality improved by VLM confirmation and core dedupe.
- Latency acceptable for ambient assistant workflows.

Actual outcome field:

```text
dense_modeled_calls=<N>, e2e_calls=<N>, call_reduction=<%>,
e2e_metric=<value>, precision=<value>, recall=<value>,
latency_p50=<s>, latency_p90=<s>, ask_rate=<value>
```

## Perception Test Action/Sound E2E

Perception Test action and sound should be the first goal-conditioned E2E tracks. The dataset label becomes
the user's watch goal, then Percept is evaluated on whether it finds and notifies the event at the right
time.

### Action Shape

Input row:

```json
{
  "video_id": "pt_video_123",
  "label": "Opening something",
  "onset_s": 8.2,
  "end_s": 10.5
}
```

Percept invocation:

```text
label -> goal writer
goal: "Notify me when someone is opening something."

Percept.create(...)
agent.add_goal(Goal(id="pt_video_123:opening", condition="someone is opening something", ...))
GatedWatchLoop(agent, goals=[goal], ...)
  -> core compiles WatchSpec
  -> edge/harness evaluates sampled frames
  -> VLM judges only escalated candidate frames/windows
  -> core emits fire/ask/silent
```

Success example:

```text
ground truth: opening something [8.2, 10.5]
edge wakes:   7.9, 8.6, 9.1, 14.0
VLM calls:    4
fire:         8.8

edge recall:              hit
edge precision proxy:      3/4 wakes in/near window
notification recall:       hit
notification precision:    fire inside labelled window
notification latency:      0.6s
dense modeled calls:       sampled frames, e.g. 31
e2e calls:                 4
call reduction:            87.1%
```

Failure examples to classify:

| Failure | Meaning | Owner |
| --- | --- | --- |
| No edge wake near interval | conductor missed the event | edge/goal compilation |
| Edge wakes but VLM says no | backend perception miss or bad prompt | backend/goal wording |
| VLM says yes but no fire | cognition/gate bug | core |
| Fire outside interval | final precision failure | edge/backend/core, inspect evidence |
| ASK instead of FIRE | uncertainty surfaced; not a false fire | product policy |

Metrics:

- edge event recall: labelled events with at least one wake in or near `[onset_s, end_s]`.
- edge precision proxy: wakes inside labelled windows divided by all wakes.
- final notification recall: labelled events with a `fire` inside the window.
- final notification precision: fires inside labelled windows divided by all fires.
- notification latency: first valid `fire_time - onset_s`.
- VLM calls and call reduction against the dense modeled sampling baseline.
- ASK rate and unresolved rate.

### Sound Shape

Input row:

```json
{
  "video_id": "pt_video_456",
  "label": "Interaction:knocking",
  "onset_s": 12.0,
  "end_s": 12.8,
  "is_visible": 0
}
```

Percept invocation:

```text
label -> goal writer
goal: "Notify me when there is a knocking sound."

WatchSpec includes audio/onset detector hints.
Edge uses audio energy/onset first; VLM or audio classifier confirms only when needed.
Core emits fire/ask/silent.
```

Sound is especially valuable because a vision-only loop may miss non-visible events. Report visible and
non-visible sound sources separately.

Metrics:

- audio edge recall and latency.
- final notification recall/precision.
- VLM/audio calls and call reduction.
- visible vs non-visible source breakdown.
- ASK rate for ambiguous audio.

### Literal vs Operationalized Goals

For specific labels, literal goals are the default. For vague labels, run both:

```text
label: "Moving something"

literal:
  "Notify me when someone is moving something."

operationalized:
  "Notify me when a visible object starts changing position because a person is manipulating it."
```

Report:

```text
label_group=specific|vague
goal_mode=literal|operationalized
edge_recall=<value>
final_recall=<value>
final_precision=<value>
latency_p50=<value>
calls=<value>
ask_rate=<value>
```

Expected outcome:

- Specific labels should work in literal mode.
- Vague labels may improve with operationalized goals.
- If a vague label cannot be made observable, Percept should ask/decline rather than fire loosely.

## RepCount-A E2E

RepCount-A is not a raw VLM benchmark for Percept. A per-frame VLM loop cannot reliably measure
repetition count because counting is a temporal-cycle problem, not a single-frame semantic judgment. Do
not spend benchmark budget asking a VLM "is this the peak of a rep?" on every frame and do not publish
that as a baseline. It mostly measures undersampling, phase ambiguity, and backend prompt instability.

The Percept question is:

```text
Given a video of a repetitive action, can Percept count distinct completed repetitions with cheap local
signals, escalate only when the signal is ambiguous, and produce a final count comparable to published
RepCount-A references?
```

### Required Unbiased Scope

The publishable RepCount-A track must use the full selected RepCount-A split, not only actions that are
friendly to a specific detector. A detector-specific subset is allowed only as a diagnostic card.

Allowed result labels:

| Label | Meaning | Publish use |
| --- | --- | --- |
| `repcount-a/full-e2e` | all sampled/full RepCount-A actions; Percept may choose pose, motion, windowed VLM, or ASK fallback | publishable |
| `repcount-a/pose-diagnostic` | only actions where pose landmarks are expected to be visible and stable | diagnostic only |
| `repcount-a/motion-diagnostic` | motion-periodicity only, no semantic confirmation | diagnostic only |

Not allowed:

- "Full RepCount-A" when the sampler excluded hard action types such as apparatus/occlusion cases.
- "Percept beats raw VLM" as the headline. Raw VLM is not the right measuring instrument here.
- Treating pose failure as dataset filtering. Pose failure must be counted as a Percept fallback case.

### Product Protocol

Input row:

```json
{
  "video_id": "repcount_001",
  "action_type": "push_up",
  "count": 12
}
```

Percept goal:

```text
"Count the completed push-up repetitions in this routine."
```

For a hard case such as pommel horse, the goal stays in the benchmark:

```text
"Count the completed pommel horse repetitions in this routine."
```

This uses the existing Percept architecture: the core/brain owns goal semantics and event identity
before any detector runs. For counting, that means the benchmark records the count contract the core is
already responsible for:

```text
action: pommel horse
routine_start: first moment the athlete begins the routine / first visible cycle evidence
routine_stop: routine end, dismount, or no further repetition evidence
unit_to_count: one completed action-specific cycle
partial_rep_rule: do not count a cycle that starts before the visible routine window or fails to complete
fallback_policy: if the cheap signal is unreliable, escalate or return unresolved/ASK
```

Percept E2E processing:

```text
goal -> existing core/compiler semantics:
  routine start/stop policy
  action-specific unit-to-count
  detector candidates and fallback plan

count WatchSpec -> edge/harness attempts measurement
edge/harness extracts cheap signals:
  motion periodicity
  pose landmarks when available
  optional object/apparatus/occlusion diagnostics

core counts candidate repetition boundaries against the compiled count contract

if confidence is low:
  escalate to a short temporal window / strip model call:
    "How many full {action_type} repetitions completed in this window?"

final answer:
  predicted_count
  confidence
  fallback_reason, if any
```

This preserves the architecture thesis:

```text
cheap local signal proposes boundaries
brain counts distinct completed cycles
heavy model is used only for semantic confirmation or ambiguous windows
```

If pommel horse fails because pose landmarks are unstable or the apparatus occludes the body, that is a
valid measured failure. The benchmark should report it as `unresolved`, `pose_unavailable`,
`window_count_wrong`, or another failure class, not remove it from the sample.

### Metrics

- OBO: off-by-one accuracy, `abs(predicted_count - gt_count) <= 1`.
- normalized MAE: `abs(predicted_count - gt_count) / max(gt_count, 1)`.
- absolute MAE: `abs(predicted_count - gt_count)`.
- fallback rate by action type.
- pose success/failure rate, but not as an exclusion criterion.
- heavy-model calls per video and calls per minute.
- cost estimate.
- unresolved/ASK rate.

Scorecard rows:

```text
repcount-a/full-e2e:
  N=<all sampled rows>
  OBO=<value>
  normalized_MAE=<value>
  absolute_MAE=<value>
  model_calls_per_video=<value>
  fallback_rate=<value>
  unresolved_rate=<value>

reference:
  TransRAC / TML RepCount-A references, cited as external bars
```

### Failure Classification

| Failure | Meaning | Fix target |
| --- | --- | --- |
| Pose unavailable | body landmarks missing/unstable due occlusion, apparatus, camera angle | fallback to motion/windowed VLM |
| Motion aliasing | cheap motion sees two peaks per rep or misses fast reps | better periodicity/hysteresis |
| Window count wrong | VLM window over/under-counts completed cycles | prompt/window length/model |
| Start/end ambiguity | routine begins/ends mid-rep | boundary policy / partial-rep rule |
| Semantic mismatch | wrong action type confirmed | goal/action classifier |
| Low confidence | system cannot count reliably | ASK/unresolved, not a false precise count |

### Expected Outcome

The first publishable RepCount-A target is not "beat every specialized trained counter." It is:

- unbiased full-split reporting.
- materially fewer heavy-model calls than any dense cloud-video strategy.
- reasonable OBO/MAE on common visible reps.
- honest fallback/ASK behavior on occluded or apparatus-heavy actions.
- clear evidence about which detector or fallback needs investment next.

## ProactiveVideoQA E2E

ProactiveVideoQA is the benchmark closest to Percept's "ambient assistant" product shape. A user asks a
question at the start of a video, then the system should answer only when the answer becomes visible.
The key product behavior is not constant narration. It is timely, sparse, useful intervention.

The Percept question is:

```text
Given a standing user question, can Percept watch the stream, wake when answer evidence appears, answer
at the right moment, avoid repeated interruptions, and spend fewer heavy-model calls than dense
streaming schedule?
```

### Product Protocol

Input row:

```json
{
  "video": "ego_001.mp4",
  "question_id": "q_001",
  "question": "What ingredient should I add next?",
  "answer": [
    {
      "content": "Add the chopped onions.",
      "reply_timespan": [42.0, 50.0]
    }
  ]
}
```

Percept goal/concern:

```text
"Answer when there is enough visual evidence to answer: What ingredient should I add next?"
```

This uses the existing Percept architecture:

```text
question -> core creates a standing concern
core derives answer-evidence WatchSpec:
  semantic relevance to the question
  object/action/text/audio hints where applicable
  novelty/change policy
  answer confidence threshold

edge/harness scans the stream cheaply:
  motion/change/OCR/audio/object cues where available
  candidate answer-evidence windows

on candidate wake:
  VLM sees a frame/strip/window and the question
  VLM returns answer evidence or "not enough evidence"

core:
  emits one answer when evidence is sufficient
  suppresses repeated equivalent answers
  may update/revise if genuinely new answer evidence appears
  stays silent or asks when evidence is insufficient
```

### TML / Official Comparability

The official ProactiveVideoQA-style metric is PAUC: answer quality integrated over the allowed response
window, usually with `omega=0.5`. TML reports ProactiveVideoQA PAUC@0.5 as a reference number.

Percept can be compared on this axis only when:

- the same question rows/split are used.
- the same answer windows are used.
- the same PAUC math and silent/no-response convention are declared.
- the answer judge is specified and independent from the answer generator.
- coverage and failed-call rows are reported.

Percept should report additional product metrics that PAUC underprices:

- interruptions: total user-facing answers emitted.
- duplicate-answer rate.
- first-answer latency relative to `reply_timespan[0]`.
- final answer precision/quality.
- heavy-model calls per video/minute.
- call reduction versus a declared dense modeled strategy.
- ASK/unresolved rate.

### Configurations

| Config | Meaning | Claim |
| --- | --- | --- |
| `dense-modeled` | no media calls; compute sampled frame/window call denominator | upper-cost diagnostic |
| `core-replay` | same cached backend observations through Percept dedupe/rising-edge answer policy | interruption/spam reduction, no call reduction |
| `edge-e2e` | edge wakes candidate answer moments; backend answers candidates; core emits sparse answers | product quality plus cost |

The current `exp_proactiveqa.py` style of "Gemini narrates every sampled frame, then Percept dedupes"
belongs to `core-replay`. It can support an interruption/spam claim, but not an E2E call-reduction
claim, and should not be run live for publishable benchmarking.

### Metrics

Scorecard rows:

```text
proactiveqa/edge-e2e:
  N_questions=<N>
  N_answer_turns=<N>
  PAUC@0=<value>
  PAUC@0.5=<value>
  PAUC@1=<value>
  interruptions=<count>
  duplicate_rate=<value>
  first_answer_latency_p50=<seconds>
  first_answer_latency_p90=<seconds>
  heavy_model_calls=<count>
  calls_per_min=<value>
  unresolved_rate=<value>

reference:
  TML / official ProactiveVideoQA references, cited as external bars
```

### Worked Example

Dataset row:

```json
{
  "video": "EGO_bus_stop_017.mp4",
  "question_id": "bus-route-visible",
  "question": "Which bus should I take?",
  "answer": [
    {
      "content": "Take the 38 bus.",
      "reply_timespan": [42.0, 50.0]
    }
  ],
  "duration": 72.0
}
```

User-facing Percept concern:

```text
"Tell me when there is enough visual evidence to answer: Which bus should I take?"
```

The existing core semantics compile that into a standing answer concern:

```text
answer_evidence:
  question: "Which bus should I take?"
  target: readable bus route / destination evidence
  wake hints: bus enters scene, route-number OCR, destination sign text, vehicle stop/approach
  fire rule: answer once when evidence is sufficient
  re-arm rule: suppress equivalent repeated answers; allow a new answer only if a different bus/route appears
```

Example E2E trace:

```text
ground truth answer window: [42.0, 50.0]
ground truth answer:        "Take the 38 bus."

edge wakes:
  12.0  car passes / street motion                    -> VLM says not enough evidence
  39.5  bus enters scene, sign still unreadable       -> VLM says not enough evidence
  43.0  route number and destination are readable     -> VLM says "Take the 38 bus."
  44.5  same bus sign still visible                   -> core suppresses duplicate
  58.0  different vehicle passes                      -> VLM says not enough evidence for this question

Percept user-facing outputs:
  fire @43.0: "Take the 38 bus."

heavy-model calls:
  dense modeled 5s schedule over 72s video: 15 calls
  edge-e2e candidate calls:     4 calls
  call reduction:               73.3%

latency:
  first_answer_latency = 43.0 - 42.0 = 1.0s

interruptions:
  dense schedule might have produced route updates on every readable frame
  core emits 1 user-facing answer
```

Scored output row:

```json
{
  "question_id": "bus-route-visible",
  "config": "edge-e2e",
  "answer_window": [42.0, 50.0],
  "predicted_answers": [
    {
      "t": 43.0,
      "text": "Take the 38 bus.",
      "judge_score": 2
    }
  ],
  "pauc@0.5": 0.88,
  "first_answer_latency_s": 1.0,
  "interruptions": 1,
  "duplicate_answers": 0,
  "heavy_model_calls": 4,
  "dense_baseline_calls": 15,
  "unresolved": false
}
```

The same row can expose different failure modes:

```text
no wake:
  edge wakes only at 12.0 and 58.0, misses [42.0, 50.0]
  -> edge/goal compilation failure

wake but no answer:
  edge wakes at 43.0, VLM says "not enough evidence"
  -> backend/window-selection failure

too early:
  system answers "Take the 38" at 39.5 before the route sign is readable
  -> core evidence threshold too loose

spam:
  system answers at 43.0, 44.5, 45.0, 46.0 with equivalent text
  -> core answer-memory/dedupe failure

ask/unresolved:
  the route number is blurry, system asks "I see a bus, but cannot read the route yet."
  -> acceptable if counted as unresolved, not as a confident answer
```

Failure classification:

| Failure | Meaning | Fix target |
| --- | --- | --- |
| No wake in answer window | edge/goal compilation missed answer evidence | wake policy, relevance cues |
| Wake but no answer | backend did not see enough evidence | frame/window selection, model |
| Correct answer too early | answer emitted before evidence window | core evidence threshold |
| Correct answer too late | answer after useful window | latency/cadence/wake policy |
| Repeated same answer | dedupe/rising-edge failure | core answer memory |
| Wrong answer | perception/reasoning failure | backend prompt/window/context |
| Unresolved/ASK | evidence insufficient or ambiguous | acceptable if surfaced honestly |

### Expected Outcome

The first publishable ProactiveVideoQA target is:

- comparable PAUC reporting on a declared split/judge.
- materially fewer interruptions than dense narration.
- materially fewer heavy-model calls in `edge-e2e`.
- clear latency distribution for first useful answer.
- no hidden cache failures or zero-response pollution.

### Current Smoke Outcome

The first `proactiveqa-e2e` N=1 live smoke is an honest negative:

```text
video: f6a6c5d5-3fe0-42af-9ddb-fa1a63521c0d-2160-2520.mp4
duration: about 360s
answer turns: 11
dense_modeled_calls: 65
e2e_candidate_calls: 55
call_reduction: 15.4%
edge_candidate_recall: 0.82
edge_candidate_precision_proxy: 0.55
interruptions: 0
unresolved_rate: 1.00
PAUC@0.5 proxy: 25.0
```

Interpretation: the product adapter is wired and the scorecard is reproducible, but this track carries
a release blocker until an independent answer judge replaces the token-overlap PAUC proxy. The default
motion edge wakes too often on first-person cooking video and the candidate-answer prompt is too
conservative, producing no answers. The next PVQA work is goal-aware/semantic wake filtering plus a
better answer-evidence prompt and an independent answer judge.

## CueSpeak / TimeSpeak

CueSpeak and TimeSpeak should be treated as interaction-protocol benchmarks for Percept's speech
surface.

- CueSpeak asks: when a cue happens, does the agent say the right thing once?
- TimeSpeak asks: when a cue becomes answerable/actionable, does the agent speak at the right time?

For the publishable benchmark, CueSpeak and TimeSpeak are derived scorers over existing E2E runs, not
separate video runs. The same Perception Test and ProactiveVideoQA executions should emit
`fire`/`ask`/`silent` events plus answer text; CueSpeak/TimeSpeak score those outputs through an
interaction lens.

The current repo tracks are deterministic simulations. They remain fast regression tests because they
exercise the real Percept gate/executor with scripted perception, but they are not TML-comparable
release results.

### Derived Sources

| Source run | Derived CueSpeak | Derived TimeSpeak | Extra video pass? |
| --- | --- | --- | --- |
| Perception Test action/sound E2E | Did Percept say the right cue message for the labelled event? | Did it speak near event onset? | no |
| ProactiveVideoQA edge-E2E | Did Percept say the right answer content? | Did it answer inside the answer window, not before? | no |
| Charades E2E | Optional broad action-cue view; softer intervals | Optional action-start timing view | no, optional |
| authored prompt corpus | Optional supplemental speech UX coverage | Optional supplemental timing UX coverage | yes, only if created |

Release priority should be:

1. derive from Perception Test action/sound.
2. derive from ProactiveVideoQA edge-E2E.
3. use authored prompts only for gaps not covered by those datasets.

### Perception Test Derived Scoring

Inputs come from the same PT E2E run:

```json
{
  "video_id": "pt_video_123",
  "label": "Opening a door",
  "onset_s": 8.2,
  "end_s": 10.5,
  "percept_outputs": [
    {"t": 8.8, "action": "fire", "text": "Someone is opening the door."}
  ],
  "heavy_model_calls": 4
}
```

Derived prompts:

```text
CueSpeak view:
  goal: "Tell me if someone opens the door."
  expected semantics: "someone is opening/opened the door"
  valid speech window: [onset_s - pre_tolerance, end_s + post_tolerance]

TimeSpeak view:
  goal: "Tell me when someone starts opening the door."
  speak window: [onset_s, onset_s + omega]
  early window: t < onset_s
```

Computation:

```text
fires = outputs where action == "fire"
asks = outputs where action == "ask"

valid_fire(f):
  CueSpeak: f.t in [onset_s - pre_tolerance, end_s + post_tolerance]
            and semantic_judge(f.text, expected_semantics) >= threshold
  TimeSpeak: f.t in [onset_s, onset_s + omega]
             and semantic_judge(f.text, expected_semantics) >= threshold

CueSpeak per row:
  cue_recall = 1 if any valid CueSpeak fire else 0
  cue_precision = valid CueSpeak fires / max(all fires for this goal, 1)
  fires_per_episode = all fires for this goal
  duplicate_count = max(0, equivalent valid fires - 1)
  unresolved = 1 if no valid fire and asks > 0 else 0

TimeSpeak per row:
  first_valid_fire = earliest valid TimeSpeak fire
  latency_s = first_valid_fire.t - onset_s, if present
  timing_hit = 1 if first_valid_fire exists else 0
  early_fire = 1 if any semantic-matching fire before onset_s else 0
  late_fire = 1 if no timing_hit and any semantic-matching fire after onset_s + omega else 0
  no_response = 1 if no fire and no ask else 0
```

PT-derived aggregate metrics:

```text
cue_recall = mean(cue_recall rows)
cue_precision = sum(valid CueSpeak fires) / max(sum(all fires), 1)
fires_per_episode = sum(all fires) / N_events
duplicate_rate = sum(duplicate_count) / max(sum(all fires), 1)
unresolved_rate = mean(unresolved rows)

timing_accuracy = mean(timing_hit rows)
p_pauc = mean(max(0, 1 - latency_s / omega) for timing hits, 0 for misses) * 100
latency_p50/p90 = percentiles(latency_s over hits)
early_fire_rate = mean(early_fire rows)
late_fire_rate = mean(late_fire rows)
no_response_rate = mean(no_response rows)
```

Example:

```text
PT label: "Opening a door"
ground truth: [8.2, 10.5]
Percept output: fire @8.8 "Someone is opening the door."

CueSpeak:
  valid semantic fire: yes
  cue_recall: 1
  cue_precision: 1/1
  fires_per_episode: 1.0

TimeSpeak with omega=1.0:
  latency: 0.6s
  timing_hit: 1
  p_pauc contribution: (1 - 0.6 / 1.0) * 100 = 40
```

### ProactiveVideoQA Derived Scoring

Inputs come from the same PVQA edge-E2E run:

```json
{
  "question_id": "bus-route-visible",
  "question": "Which bus should I take?",
  "answer_window": [42.0, 50.0],
  "expected_answer": "Take the 38 bus.",
  "percept_outputs": [
    {"t": 43.0, "action": "fire", "text": "Take the 38 bus."}
  ],
  "heavy_model_calls": 4
}
```

Computation:

```text
answer_score(f):
  independent judge scores f.text against expected_answer.
  normalized to [0, 1] or official PAUC score scale.

valid_answer(f):
  f.t in answer_window and answer_score(f) >= threshold

CueSpeak from PVQA:
  answer_precision = valid answers / max(all answer fires, 1)
  answer_recall = 1 if any valid answer in answer_window else 0
  semantic_score = best answer_score in answer_window
  duplicate_count = equivalent repeated valid answers after first
  unresolved = 1 if no valid answer and asks > 0 else 0

TimeSpeak from PVQA:
  first_valid_answer = earliest valid answer
  latency_s = first_valid_answer.t - answer_window[0], if present
  timing_hit = 1 if first_valid_answer exists else 0
  early_answer = 1 if any semantic-matching answer before answer_window[0] else 0
  late_answer = 1 if no timing_hit and any semantic-matching answer after answer_window[1] else 0
  PAUC = official PVQA PAUC computed over answer scores and answer_window
```

PVQA-derived aggregate metrics:

```text
answer_recall = mean(answer_recall rows)
answer_precision = sum(valid answers) / max(sum(all answer fires), 1)
semantic_score = mean(best answer_score rows)
duplicate_rate = sum(duplicate_count) / max(sum(all fires), 1)
unresolved_rate = mean(unresolved rows)

timing_accuracy = mean(timing_hit rows)
pauc@0.5 = official PVQA PAUC over all answer turns
latency_p50/p90 = percentiles(latency_s over valid answers)
early_answer_rate = mean(early_answer rows)
late_answer_rate = mean(late_answer rows)
```

Example:

```text
PVQA question: "Which bus should I take?"
answer window: [42.0, 50.0]
expected answer: "Take the 38 bus."
Percept output: fire @43.0 "Take the 38 bus."

CueSpeak:
  answer judge: correct
  answer_recall: 1
  answer_precision: 1/1
  semantic_score: 1.0

TimeSpeak:
  latency: 1.0s
  timing_hit: 1
  early_answer: 0
  PAUC: computed by official PVQA PAUC using the answer score at t=43.0
```

### Configurations

| Config | Meaning | Claim |
| --- | --- | --- |
| `sim-core` | scripted verdict stream through real Percept core | cognition regression only |
| `dense-modeled` | no media calls; compute sampled frame/state call denominator | upper-cost diagnostic |
| `edge-e2e` | edge wakes candidate cue moments; backend confirms; core speaks once | product quality plus cost |

The existing `track_cuespeak.py` and `track_timespeak.py` are `sim-core` tracks. They should stay as
fast regression tests when no E2E artifacts exist. When local PT/Charades/PVQA E2E artifacts are
present, they now derive draft CueSpeak/TimeSpeak rows directly from those past runs, with no new video
pass and no model calls.

Current derived N=1 artifact snapshot:

```text
sources:
  pt-action-e2e: 1 event
  charades-e2e: 7 action windows
  proactiveqa-e2e: 11 answer turns

cuespeak:
  precision: 0.80
  recall: 0.21
  fires_per_episode: 0.26
  duplicate_rate: 0.00

timespeak:
  timing_accuracy: 0.00
  p_pauc: 0
  latency_p50: 10.1s
  early_fire_rate: 0.05
```

Interpretation: the E2E runs are not spammy, and when Percept speaks the cue is usually in a valid
window, but it misses most opportunities and its strict timing quality is currently poor. This derived
view is useful for diagnosis, not a release claim, because the source runs are local sampled artifacts
and PVQA still uses a proxy answer scorer.

### Release Scorecard

```text
cuespeak/edge-e2e:
  source=<perception-test|proactivevideoqa|combined>
  N_events_or_turns=<N>
  cue_recall=<value>
  utterance_precision=<value>
  semantic_score=<value>
  fires_per_episode=<value>
  duplicate_rate=<value>
  unresolved_rate=<value>
  heavy_model_calls=<count>

timespeak/edge-e2e:
  source=<perception-test|proactivevideoqa|combined>
  N_events_or_turns=<N>
  timing_accuracy=<value>
  p_pauc=<value>
  latency_p50=<seconds>
  latency_p90=<seconds>
  early_fire_rate=<value>
  guess_rate=<value>
  unresolved_rate=<value>
  heavy_model_calls=<count>

reference:
  TML CueSpeak / TimeSpeak references, cited as external bars
```

### Failure Classification

| Failure | Meaning | Fix target |
| --- | --- | --- |
| Missed cue | no fire in cue/speak window | wake policy / cadence |
| Wrong utterance | speech does not match expected semantics | answer generator / context |
| Too early | speaks before evidence is sufficient | core threshold / ASK discipline |
| Too late | speaks after useful window | cadence / wake latency |
| Repeated utterance | multiple equivalent fires for one cue | rising-edge/dedupe |
| Over-refusal | asks or stays silent despite clear cue | threshold / backend prompt |
| Unsafe guess | confident speech under ambiguous evidence | gate policy |

### Expected Outcome

The first publishable CueSpeak/TimeSpeak target is:

- derived scoring from existing Perception Test and ProactiveVideoQA E2E runs.
- judge-backed semantic scoring for CueSpeak.
- strict timing and early-guess accounting for TimeSpeak.
- interruption/no-nag metrics, not only macro accuracy.
- clear separation between deterministic `sim-core` regression and real `edge-e2e` product results.

## Charades Worked Example

Charades should be framed exactly like a real user asking Percept to watch a home video for multiple
ambient conditions at once. Each annotated action label becomes a watch goal. All goals for a video are
active together so the benchmark measures the actual product shape: one stream, many user goals, cheap
edge/conductor wakeups, selective VLM confirmation, and rising-edge notifications.

This is also the shape that can eventually make a TML-style comparison meaningful. TML's Interaction
Models Charades task asks the model to say when an action starts and when it stops, then scores temporal
IoU. Percept's current public watch contract emits rising-edge notifications. The current publishable
Charades adapter therefore reports notification recall/precision/latency and call reduction. A
candidate-only sparse span mIoU may be shown as diagnostic, but direct TML mIoU comparability requires a
first-class start/stop span contract.

The comparison must stay precise:

- TML reference: interaction quality, reported as Charades mIoU.
- Percept E2E today: goal-conditioned notification quality, reported as interval hit recall/precision,
  notification latency, and heavy-model call reduction.
- Percept E2E future: start/stop span quality, reported as Charades mIoU/time-F1 on the same
  label-to-interval task.
- Percept additional axis: heavy-model calls, calls/min, cost estimate, and latency.

Do not claim direct TML reproduction unless the prompt protocol, scorer, dataset split, and sampling
match. The default public wording today should be: "On a Charades-style goal-conditioned temporal
notification benchmark, Percept E2E achieves X notification recall / Y precision at Z fewer heavy-model
calls. TML reports 32.4 mIoU on its separate Interaction Models Charades setup."

Charades row shape:

```json
{
  "id": "LR9KL",
  "length": 30.42,
  "actions": [
    ["Holding a vacuum", 0.0, 20.9],
    ["Someone is running somewhere", 13.7, 31.0],
    ["Tidying something on the floor", 0.0, 19.7]
  ]
}
```

Goalized watch spec for that row:

```text
g1: "Notify me when someone is holding a vacuum and stop vacuuming."
    label: "Holding a vacuum"
    interval: [0.0, 20.9]

g2: "Notify me when someone is tidying something on the floor."
    label: "Tidying something on the floor"
    interval: [0.0, 19.7]

g3: "Notify me when someone is holding a blanket."
    label: "Holding a blanket"
    interval: [9.3, 19.7]

g4: "Notify me when someone is running somewhere."
    label: "Someone is running somewhere"
    interval: [13.7, 31.0]
```

This catches a common benchmark mistake: the question is not "can Gemini answer one isolated label at
one timestamp?" The Percept product question is "given these watch goals, did the agent notify during
the right temporal interval, avoid spam, and spend fewer heavy calls than a dense modeled schedule?"

Per-row expected output:

```json
{
  "video_id": "LR9KL",
  "goals": [
    {
      "goal_id": "g1",
      "goal_text": "Notify me when someone is holding a vacuum and stop vacuuming.",
      "label": "Holding a vacuum",
      "gt_interval": [0.0, 20.9],
      "edge_wakes": [0.4, 1.2, 7.6, 20.1],
      "vlm_calls": 4,
      "fires": [{"t": 1.2, "action": "fire"}],
      "asks": [],
      "latency_s": 1.2
    }
  ]
}
```

### Dense/Replay Charades

Dense Charades is a call model, not a live run. It counts how many calls a dense `(goal, timestamp)`
strategy would have made. Core replay may reuse E2E-captured candidate observations to test dedupe, but
it must not trigger additional Gemini calls.

Processing:

```text
goal: "Notify me when someone is holding a vacuum and stop vacuuming."
timestamps: 0s, 1s, ..., 30s
dense_modeled_calls: 31 for this goal
E2E candidate verdicts: only timestamps where edge woke
core-replay spans: rising-edge start -> confident-no / timeout close
score: temporal IoU against [0.0, 20.9]
```

Expected:

- Dense calls are modeled from the sampling schedule and goal count.
- Core replay uses only observations already captured during E2E.
- No live dense VLM calls.

Miss catch:

If an expected E2E candidate judgment fails, the row is `failed` or `incomplete`, not a zero-score
complete row. It must not be included in the headline.

### E2E Charades

E2E is now available as `charades-e2e` in `tools/percept_benchmark.py`. It is not part of
`safe-live` because a single video can contain many active goals, so it should be run deliberately.

Protocol:

```text
Current Percept E2E:
  "Notify me when someone is {action}."
  start = first valid fire/rising edge

Future TML-style Percept E2E:
  "Say start when the person starts doing {action}; say stop when they stop."
  start = first valid fire/rising edge
  stop = predicted span close after confident no / timeout / end-of-evidence policy
```

Processing:

```text
video LR9KL
goals:
  g1: Notify me when someone is holding a vacuum and stop vacuuming.
  g2: Notify me when someone is tidying something on the floor.
  g3: Notify me when someone is holding a blanket.
  g4: Notify me when someone is running somewhere.

core -> WatchSpec([g1, g2, g3, ...])
edge scans frames at 4fps or configured fps
on candidate wake:
  VLM judges candidate frame/window for active goals
core turns verdicts into fire/ask/silent events
score fires per goal against Charades intervals
derive sparse spans from candidate verdicts only as a diagnostic
```

Expected:

- Fewer VLM calls than the dense modeled schedule.
- Some recall loss is possible if edge misses low-motion events.
- Final precision should improve after VLM confirmation and cognition dedupe.
- The scorer must report goal-level misses, not only an average, so we can see whether failures are
  systematic by verb family: holding, cleaning, locomotion, object interaction, etc.

Actual outcome must report both quality and calls:

```text
dense_modeled_calls=217
e2e_calls=35
call_reduction=83.9%
notification_recall=0.57
notification_precision=0.80
notification_latency_p50=12.10s
sparse_span_mIoU_diagnostic=0.12
coverage=pass
```

Failure classification:

| Failure | Example | Meaning |
| --- | --- | --- |
| Goal writer miss | "Holding a vacuum" rewritten as "vacuuming" only | bad label-to-goal transform |
| Edge miss | no wake during blanket-holding interval | conductor did not surface low-motion semantic state |
| Backend miss | VLM says no on visible vacuum frames | perception/backend ceiling |
| Core miss | VLM yes but no fire/span | cognition/gate bug |
| Boundary miss | fires late or span ends too early | temporal policy needs tuning |
| Duplicate spam | many fires for one interval | rising-edge/dedupe failure |

## Known Current Outcomes

These are current findings, not final benchmark claims.

| Finding | Meaning | Status |
| --- | --- | --- |
| Charades `60.3` mIoU | Came from a small/easy cached subset | invalid as public claim |
| Charades `5.6` mIoU | Dominated by zero-verdict cache entries after Gemini quota exhaustion | invalid as full-sample score |
| Charades current scripts | Legacy dense/core replay over shared Gemini verdicts | do not run live; useful only for offline historical analysis |
| Escalation track | Measures edge candidate recall/call tradeoff without final VLM judgment | valid as conductor diagnostic |
| Charades E2E notification N=1 | `LR9KL`: 217 dense-modeled calls -> 35 candidate Gemini calls; recall 0.57, precision 0.80 | valid smoke, sampled DRAFT only |
| Full Charades start/stop E2E | Product architecture needs a span contract and scorer before direct TML mIoU comparison | future implementation target |

## Benchmark Implementation Order

1. Add run guards so legacy dense/raw runners cannot execute live by default:
   - no media-bearing dense VLM calls.
   - dense baselines are call-count models only.
   - optional empty/no-op API probe may measure overhead, not answer quality.
2. Implement Perception Test goal-conditioned E2E for action and sound:
   - action/sound label -> literal Percept goal.
   - vague action labels -> literal and operationalized goal variants.
   - core compiles WatchSpec before edge scoring.
   - report edge recall/precision, final notification recall/precision, latency, calls, ASK rate.
3. Build Charades E2E notification adapter:
   - one WatchSpec per video with all sampled action goals.
   - edge selects candidate frames/windows.
   - VLM judges only candidates.
   - score interval notification recall/precision plus calls/cost/latency.
4. Add derived CueSpeak/TimeSpeak scorers over PT/PVQA E2E outputs.
5. Add start/stop span contract if direct TML Charades mIoU is required.
6. Only then create a product-level scorecard.

Legacy cleanup:
   - row status: `complete`, `incomplete`, `failed`.
   - no empty verdict caches.
   - explicit quota/rate-limit stop.
   - coverage threshold before scoring.

## Publish Rules

Allowed:

- "On Perception Test action/sound goals, Percept-E2E finds labelled events with X recall, Y latency,
  and Z fewer calls."
- "The edge policy reduces candidate VLM calls by Y while retaining Z event wake recall" only when clearly
  labelled as an edge/conductor diagnostic.
- "In E2E mode, Percept achieves X quality at Y fewer calls" only after the E2E lane exists.

Not allowed:

- "Percept beats TML on Charades" unless a valid, comparable, fully covered run supports it.
- "Gemini score" as a Percept product score.
- "Call suppression" from any core replay that used shared verdict caches.
- Any live dense/raw media-bearing VLM loop.
- Any result with failed/empty cache rows included as ordinary misses.
- Treating a vague dataset label as a well-specified user goal without reporting goal-writing mode.
