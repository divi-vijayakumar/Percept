# Percept — Implementation Plan (build from scratch)

> **Audience:** a coding agent (e.g. Claude Code) building `percept` greenfield. This plan assumes
> **no existing code**. Companion to `percept-sdk-spec.md` (the *what/why*). This is the *how*:
> repo layout, the exact contracts, phase-by-phase build steps, the tests for each, and the
> definition-of-done. Build in order; each phase ships green on its own CI lanes before the next
> starts.

---

## 0. How to build this (read first)

**Cardinal rule — the firing contract.** The vision seam judges a goal's **plain-language
condition** ("is the user drinking coffee?") and returns `{satisfied, confidence, evidence}`. It does
**not** compile the goal to a rigid `attr_key=value` and exact-match. The historical failure: an
attribute-match path passes fake tests (a fake hardcoded to emit the matching value) while dying
silently on real video (the model says "coffee cup", never `{contains: coffee}`). Build the semantic
path first; never reintroduce an exact-match firing fork.

**Contract-first.** Phase 0 freezes the Protocols + DTOs (the "membrane"). Every later phase
implements behind a frozen interface; if a phase needs to change a frozen contract, stop and flag it
rather than editing silently.

**Fakes are load-bearing — but never test intelligence with them.** Three CI lanes:

| Lane | Tests | Backends | Runs | Command |
|---|---|---|---|---|
| **L1 unit** | OUR deterministic logic (gate state machine, cascade composition, refractory, registry, episode dedup) | fakes / stubs | every push, <2s, no keys | `make test` |
| **L2 live eval** | backend INTELLIGENCE + integration (real vision judging, real routing) | real, gated | pre-merge / nightly | `PERCEPT_LIVE=1 make eval-live` |
| **L3 browser e2e** | the real system a user drives (mic/cam -> turn -> reply, goal fire) | real, headless | pre-release | `make e2e` |

L1 stays fake-only and fast; anything needing a key lives in L2/L3 and **skips** without it. A fake
proves the *seam wiring* and *our logic*; only L2 proves the model behaves.

**DoD template (every phase):** *scope · the frozen contracts it may not edit · L1 tests · L2/L3
gate · no-drift check (lint+types+full suite green) · one demo line that runs.*

**Hard CI gates (fail the build):** gate false-fire **== 0** on the golden test split · refractory
**== 100%** · cache false-serve **<= bound** on the test split · routing **>= 95%**.

---

## 1. Repo & package layout

```
percept/                                   # monorepo
  pyproject.toml                           # workspace / or per-package
  Makefile                                 # test · eval-live · e2e · bench · check · reproduce
  .github/workflows/ci.yml                 # L1 always; L2/L3 gated/nightly
  README.md                                # thesis + envelope/refusals + quickstart GIF

  packages/
    percept-vision/                        # === THE SDK (pip install percept-vision; import percept)
      pyproject.toml                        # core = pure stdlib; extras: [claude],[deepgram],[ui],[audio],[all]
      percept/
        __init__.py                         # public surface + __version__
        adapters.py                         # Protocols + DTOs (the FROZEN membrane)   [Phase 0]
        registry.py                         # name -> backend factory + entry-point discovery [Phase 0]
        config.py                           # Settings tree, env overrides, no literals in logic [Phase 0]
        agent.py                            # Percept orchestrator (reactive turn + watch loop)  [Phase 1/3]
        entity.py                           # EntityGraph (continuity + attributes/provenance)   [Phase 1]
        gate.py                             # three-state gate (the trust layer)                 [Phase 1]
        goals.py                            # Goal / GoalOp / GoalRegistry                       [Phase 1]
        identity.py                         # Identity Protocol + continuity + appearance ref    [Phase 4]
        episodes.py                         # episode track (counting)                           [Phase 4]
        events.py                           # event log + event gate + frame window              [Phase 5]
        scheduler.py                        # monotonic-clock timers -> proactive fire           [Phase 5]
        cache/
          __init__.py
          cascade.py                        # CacheCascade + CacheLayer + CacheResult            [Phase 0/2]
          framehash.py                      # FrameHashLayer (perception cache)                  [Phase 2]
        routing/
          engine.py                         # generic evaluator (walk rules, first match)        [Phase 3]
          rules.py                          # ordered DEFAULT_RULES (defer-on-miss structural)   [Phase 3]
          learned.py                        # LearnedRouteLayer (embedding classifier, ONNX)     [Phase 3.1]
        persistence.py                      # Persistence Protocol + in-memory + sqlite ref      [Phase 7]
        consent.py                          # profile flags + action log                         [Phase 7]
        observability.py                    # JSON logs + metrics + /health                      [Phase 0]
        backends/
          __init__.py                       # make_vision/specialist/conductor/stt/tts factories
          fake.py                           # deterministic offline backends                     [Phase 0]
          anthropic.py                      # Claude vision/specialist/conductor                 [Phase 1/3]
          deepgram.py                       # Deepgram STT/TTS                                    [Phase 3]
          _resilience.py                    # retry + backoff + circuit breaker                  [Phase 0]
      tests/                                # L1 unit (no keys)
      eval/
        golden/                             # scenes/ labels.jsonl gate_cases.jsonl splits.json  [Phase 0]
        replay.py                           # replay-and-align scorer                            [Phase 2]
        run.py                              # orchestrate + CI gates -> results.json             [Phase 2]
      e2e/                                  # L3 real-socket / Playwright

    percept-harness/                        # === LiveKit Agents worker (pip install percept-harness)
      percept_harness/
        session.py                          # worker entrypoint; wires STT/TTS around the brain
        watch.py                            # proactive watch loop -> agent.perceive -> say
        tier0.py                            # server-side motion gate (Tier0Signal reference)    [Phase 6]

    percept-edge/                           # === on-device (npm i @percept/edge), JS/WASM         [Phase 6]
      src/                                  # Silero VAD + motion/CLIP/pose VID bank + LiveKit publish

  apps/
    percept-assistant/                      # === PROJECT 2 — out of scope for this plan
      ...                                    #     (UI + voice persona + product demos; built ON the SDK)

  percept-bench/                            # === public benchmark repo (or submodule)            [Phase 2+]
    Makefile (reproduce) · sampled corpus · harness · scorecard
```

> **This plan builds the SDK (project 1) only.** `percept-assistant` is project 2 — a separate build
> on top of the SDK (no UI, no voice persona, no product goals belong in `percept`). The SDK's job
> toward it is clean seams; the caffeine/boil scenarios here are **test/bench fixtures**, not the
> product demo.

---

## 2. Phase 0 — Contracts, skeleton, fakes, CI

**Goal:** freeze the membrane and stand up the three CI lanes so every later phase has immutable
interfaces and an objective gate. After Phase 0 the system *runs* (offline, with fakes) but *does
nothing intelligent* — an empty cache cascade always DEFERs; the gate has no goals.

### 2.1 Build — `adapters.py` (FROZEN DTOs + Protocols)

```python
# ---- DTOs (vendor-neutral nouns) ----
@dataclass(frozen=True)
class Box:                      # normalized 0..1
    x: float; y: float; w: float; h: float
    def iou(self, o: "Box") -> float: ...

@dataclass
class Observation:
    label: str; box: Box; confidence: float; attrs: dict[str, str]

@dataclass
class Verdict:                  # the firing-contract return
    satisfied: bool; confidence: float; evidence: str = ""

Provenance = Literal["observed", "asserted", "cached", "llm"]

@dataclass
class Attribute:
    value: str; source: Provenance; confidence: float; last_confirmed: float

@dataclass
class Goal:
    id: str; condition: str; say: str
    source: str = "camera"
    modality: Literal["speak", "act"] = "speak"
    state: dict = field(default_factory=lambda: {"type": "watch", "count": 0})
    expiry: str = "until_cancelled"

@dataclass
class Decision:                 # conductor / router output
    route: Literal["respond_direct","respond_with_vision","task_with_tools","act","proactive_speak","ignore"]
    look: Literal["none","request_frame"] = "none"
    filler: str | None = None
    say: str | None = None
    goal_ops: list = field(default_factory=list)
    endpoint: Literal["complete","incomplete","barge_in","n/a"] = "n/a"

@dataclass
class STTEvent: text: str; is_final: bool; ts: float
@dataclass
class TurnResult: route: str; source: str; reply: str

# edge wire-contract
@dataclass
class WatchCondition: goal_id: str; anchor_text: str; threshold: float; region_hint: Box | None
@dataclass
class WatchSpec: conditions: list[WatchCondition]; version: int = 1
@dataclass
class Tier0Signal: motion: float; structural: float; semantic: float; ts: float

# ---- Protocols (implement to swap) ----
class Vision(Protocol):
    async def judge(self, condition: str, frame, *, roi: Box | None = None,
                    exemplar=None, corrections: dict | None = None) -> Verdict: ...
    async def detect(self, frame, goals) -> list[Observation]: ...                 # entity-attr substrate
    async def detect_events(self, frames, goals) -> list: ...                      # Phase 5; default []

class Specialist(Protocol):
    async def stream(self, prompt, *, frame=None, fast: bool=False): ...           # yields tokens
class Conductor(Protocol):
    async def run(self, event: str, utterance: str, context: dict) -> Decision: ...
class STT(Protocol):
    async def stream(self, audio): ...                                             # yields STTEvent
class TTS(Protocol):
    def push(self, text: str) -> None: ...
    def flush(self) -> None: ...
class Identity(Protocol):                                                          # Phase 4
    def match(self, obs: Observation, known: list) -> str | None: ...
class Persistence(Protocol):                                                       # Phase 7
    def save(self, key: str, value) -> None: ...
    def load(self, key: str): ...
```

### 2.2 Build — the rest of the skeleton

- `cache/cascade.py`: `CacheResult{status: HIT|DEFER, verdict, confidence, provenance, source_layer}`;
  `CacheLayer` Protocol (`lookup`, `store`); `CacheCascade{layers, threshold}` — `resolve()` returns
  the first HIT above threshold else DEFER. **Empty cascade always DEFERs** (today's behavior).
- `backends/fake.py`: `FakeVision/FakeSpecialist/FakeConductor/FakeSTT/FakeTTS` — deterministic,
  scripted, no RNG, no keys. `FakeVision.judge` returns a scripted `Verdict` from the frame token
  (e.g. frame `"sip-coffee"` -> satisfied=True conf=0.9; `"sip-water"` -> satisfied=False).
- `registry.py`: `make_vision("fake"|"claude")`, etc.; resolves a name -> adapter; discovers
  entry-point plugins (the open-core hook). Unknown name -> clear error.
- `config.py`: `Settings` tree, every tunable env-overridable (`PERCEPT_GATE_THRESHOLD`,
  `PERCEPT_IOU_MATCH`, `PERCEPT_MAX_ENTITIES`, model ids). No literals in logic.
- `observability.py`: JSON logging + a metrics registry + `/health`.
- `eval/golden/` scaffold + loader + `splits.json` (train/dev/test). Seed ~10 hand-labeled scenes;
  a capture script to grow it.
- `Makefile` + `ci.yml` wiring L1 (always) / L2 / L3 / `bench`.

### 2.3 Tests (L1) · DoD · Demo
- **L1:** Box.iou math; cascade HIT>=threshold short-circuits, below-threshold + empty -> DEFER;
  registry resolves fake names; golden loader + split determinism; CacheResult/Decision shapes.
- **L2/L3:** none yet (scaffolding) — lanes execute and skip cleanly without keys.
- **DoD:** contracts frozen; `make test` green offline with no keys; all three lanes execute.
- **Demo:** `python -c "import percept; a=percept.Percept.create(vision='fake'); print('ok')"`.

---

## 3. Phase 1 — Cognition core + the firing contract  *(V0.0)*

**Goal:** the caffeine *restraint* loop works — offline on fakes and (L2) on real Claude. This is the
heart; build it carefully.

### 3.1 Build
- `entity.py` — `EntityGraph.ingest(observations, frame, source)`: match obs->entities by IoU+label,
  mint/update ids, record attributes with provenance, **never overwrite an `asserted` attribute**,
  evict to `max_entities`. `in_focus()` (largest entity this frame). `assert_attribute(id, key, value)`
  (asserted beats observed). `note_mention(text)` (voice->visible-entity fusion).
- `gate.py` — the three-state gate:
  - `judge(verdict: Verdict, freshness: float, threshold: float) -> FIRE|ASK|SILENT`:
    confident-yes->FIRE, confident-no->SILENT, low-conf **or stale** (freshness past validity)->ASK.
  - rising-edge: track previous truth per goal; FIRE only on false->true.
  - refractory: never double-fire the same (goal, entity/episode); re-arm on KNOWN-FALSE.
- `goals.py` — `Goal`, `GoalOp(create|update|complete|cancel)`, `GoalRegistry.apply(ops)`.
- `agent.py` (watch path) — `async perceive_judged(frame, source)`: for each active goal, resolve the
  ROI, call `vision.judge(condition, frame, roi, exemplar, corrections)` (or cache), run through the
  gate, fire on rising edge, emit `(goal_id, action, text)`. `assert_attribute(...)` binds a user
  correction to the in-focus entity, clears the goal's edge state, invalidates the cache.

### 3.2 Tests
- **L1 (gate matrix, deterministic):** fire on rising edge; silent on repeat / asserted-override;
  ask on mid-confidence; never fire on low-conf; **stale asserted -> ask** (the t3 case); refractory
  never double-fires; assertion beats observation; new-entity gets a fresh edge.
- **L2 (golden firing eval, real Claude):** real `vision.judge` on labeled frames -> fire/no-fire ==
  expected; **false-fire == 0** on the test split (the trust gate).
- **L3:** caffeine on the bridge — "watch my coffee" -> coffee fires once -> water stays silent ->
  "this is tea" -> silent.
- **DoD:** one firing path; gate false-fire == 0 on golden test; the t0->t3 caffeine restraint trace
  passes; refractory 100%.
- **Demo:** the §3.1 trace from the spec runs end-to-end on fakes, and on real Claude under L2.

---

## 4. Phase 2 — Perception cache + narrow grounded VQA  *(V0.0)*

**Goal:** cut VLM cost without false-serving; switch perception to narrow grounded VQA on a cropped
ROI; stand up the replay-and-align scorer.

### 4.1 Build
- `backends/anthropic.py` `judge`: **narrow grounded question on a cropped ROI** + the exemplar image
  when present ("is the user drinking from THIS mug <exemplar>?") — not whole-frame description.
- `cache/framehash.py` — `FrameHashLayer`: per-goal ROI dHash; Hamming near-match
  (`<= tau_bits`); `confidence = min(verdict.conf, sim*age_decay)`; invalidation order goal-change ->
  assertion -> TTL -> L0 frame-diff. Wire into the cascade in front of `judge`.
- `agent._audit_or_use`: on cache HIT, re-perceive a random `cache_audit_rate` fraction; a divergence
  is a **false-serve**, the fresh read wins.
- `eval/replay.py` + `eval/run.py` — replay a labeled clip -> record fires -> align to ground-truth
  rising edges within omega -> precision/recall/latency(PAUC)/count/cost/context. CI gates wired.

### 4.2 Tests
- **L1:** near-match (identical->HIT, structural change->DEFER); confidence math (stale->defer,
  low-conf verdict never served confidently); invalidation (goal-change drops, assertion drops entity,
  TTL backstop); deterministic synthetic frames/hashes.
- **L2:** false-serve eval — re-perceive epsilon% of HITs through real vision and compare;
  **false-serve <= bound** (hard fail); report hit-rate (>=70%) + VLM-calls/min drop.
- **DoD:** >=70% hit-rate and false-serve <= bound on golden; the replay scorer emits a scorecard;
  CI gates on false-fire and false-serve are live.
- **Demo:** a still scene spends <=1 vision call over N seconds; motion re-perceives.

---

## 5. Phase 3 — Reactive voice loop (+ learned routing)  *(V0.0 core; routing -> V0.1)*

**Goal:** the spoken turn — router -> conductor -> specialist -> TTS — plus the proactive conductor
path. Then (V0.1) replace brittle regex routing with a learned tier.

### 5.1 Build (V0.0)
- `routing/engine.py` + `routing/rules.py`: ordered `DEFAULT_RULES`, generic evaluator, first match
  wins, **returns `None` to DEFER** to the conductor on anything goal-authoring / conditional /
  ambiguous / a fragment. Defer-on-miss is structural (a confidence below threshold defers), not a
  regex. (Regex lexicons acceptable for V0.0; replaced in V0.1.)
- `backends/anthropic.py` conductor (Haiku, prompt-cached): routes + authors goals (the only tier that
  writes standing state); specialist (streams the reply, optionally with a frame).
- `agent.on_speech_endpoint(utterance)`: note mentions (fusion) -> `local_route` -> if handled
  `_execute`; else defer to conductor -> author goals / route / proactive-speak.
- `backends/deepgram.py`: streaming STT + TTS behind the seams.
- **Proactive conductor path:** scene deltas (person entered / something appeared) reach the conductor
  so it can speak unprompted, not only fire pre-armed goals.

### 5.2 Build (V0.1 — learned routing)
- Data: conductor-as-teacher — log `(utterance -> route)` from real traffic + the labeled set.
- `routing/learned.py` `LearnedRouteLayer`: small embedding + linear/MLP intent classifier (ONNX,
  app-tier, lazy import). High-conf -> route locally (<10ms); low-conf -> DEFER. Language-agnostic.

### 5.3 Tests
- **L1:** router rule-table (each rule's `examples/expect` auto-generate cases); `LearnedRouteLayer`
  GATING logic with a **stub classifier** (high-conf->HIT, low-conf->DEFER) — do NOT unit-test model
  accuracy.
- **L2:** real-classifier accuracy >= Haiku-parity on the labeled set, AND **defers (not mis-routes)**
  on an adversarial set (zero confident mis-routes); end-to-end latency drop on common turns.
- **L3:** "where is he?" routes to vision locally; a full spoken turn streams reply + barge-in stops
  cleanly.
- **DoD (V0.0):** reactive turn works end-to-end; routing >=95%. **(V0.1):** learned tier at
  Haiku-parity, zero confident mis-routes, measured latency drop.
- **Demo:** spoken question -> streamed answer < budget; "watch X" authors a goal.

> **V0.0 first-launch boundary:** Phases 0–3 (with regex routing) + the Phase-2 cache + the
> measurement scaffold + the two demos (caffeine restraint, boil) + the README envelope = the first
> public release. Everything below is an OSS fast-follow.

---

## 6. Phase 4 — Identity (appearance re-id) + episode counting  *(V0.2)*

**Goal:** recognize a thing across a gap, distinguish a new one, and count *episodes* (the cup
problem).

### 6.1 Build
- `identity.py` — `Identity` Protocol; ship two impls behind it: `ContinuityIdentity` (IoU+label,
  current behavior) and `AppearanceIdentity` (an embedding/signature on each entity; `match(obs,
  known)` = nearest within a threshold over the **recent set**, brute-force). EntityGraph uses the
  configured matcher to assign ids. Retain the appearance signature on every entity even before
  re-id is needed (forward-compat).
- `episodes.py` — per counting-goal `EpisodeLog` (bounded by the counting window). On a candidate
  drinking-event: ask the LLM *new-episode-vs-continuation* against the log summaries + current
  exemplar; append or update; `count = len(distinct episodes)`. Wire into the gate's count goals.

### 6.2 Tests
- **L1:** appearance matcher gating (same-signature->same id, distinct->new id, threshold honored)
  with a **stub embedder**; episode dedup logic with a stub judge (continuation->no new episode,
  new->increment); window bound respected.
- **L2:** re-id accuracy on golden clips with gaps (object leaves + re-enters -> same id; a genuinely
  new object -> new id); episode-count off-by-one on the cup clips (sip-sip-sip->1; fresh cup->2).
- **DoD:** appearance re-id beats pure-IoU on the gap clips; episode count off-by-one == target.
- **Demo:** drink one cup over a gap -> counted once; pick up a new cup -> counted twice / fires.

---

## 7. Phase 5 — Events + scheduler  *(V0.3)*

**Goal:** count/​narrate events (passes, reps) and fire on a clock (TimeSpeak).

### 7.1 Build
- Frame window: replace keep-latest with a ring buffer (~8–16 frames over 4–8s per source).
- `events.py` — `Event{type, t, participants, confidence}` log; `Vision.detect_events(frames, goals)`;
  an **event gate** reusing three-state semantics; **event identity/dedup** (the temporal analog of
  IoU continuity) so a multi-frame event counts once. Counting falls out (count events of a type).
- Conductor: recognize event goals ("tell me when X happens", "count the passes") -> compile to
  `event_type` instead of an attribute.
- `scheduler.py` — monotonic-clock timers; on tick, re-invoke the conductor with a `scheduler_tick`
  event -> `proactive_speak` or cancel. Timing lives in the clock; content judgment in the model.

### 7.2 Tests
- **L1:** event dedup ([hit,hit,hit]->1 event; [hit,miss,hit]->2); scheduler fires once per interval;
  event gate three-state matrix with stub detector.
- **L2:** event-count off-by-one on labeled event clips (a soccer/reps clip); scheduled fire timing.
- **DoD:** event count off-by-one == target on golden; scheduler fires on time with correct content.
- **Demo:** "count the reps" over a clip; "remind me every 20 min."

---

## 8. Phase 6 — On-device edge (`@percept/edge`)  *(V0.4)*

**Goal:** the cost/latency/privacy unlock — VAD + VID gating on-device, the WatchSpec↓/Tier0Signal↑
loop wired end-to-end.

### 8.1 Build
- `percept-harness/tier0.py` (Python, server-side reference): `motion_score`, `compile_watchspec`
  (goals -> WatchSpec), gate `perceive_judged` on motion (`PERCEPT_MOTION_GATE`). **Make
  `compile_watchspec`'s output actually consumed and reported back as a `Tier0Signal`** — the single
  wired loop is the proof point.
- `percept-edge` (JS/WASM): Silero VAD (ONNX) endpointing; motion (pixel-diff) -> CLIP gate
  (transformers.js/ONNX) -> pose/hands (MediaPipe) salience; publish gated media + `Tier0Signal` to
  LiveKit; receive `WatchSpec` down. Detectors pretrained, threshold a shipped default (auditable
  sweep), optional per-goal bias.

### 8.2 Tests
- **L1 (JS, vitest):** VAD endpoint on fixture audio; motion gate on fixture frames; WatchSpec compile.
- **L3:** Playwright with synthetic mic/cam — upstream gated (no frames on silence/still); measured
  latency drop vs server-only.
- **DoD:** on-device VAD endpoints locally; upstream gated; the down/up loop runs end-to-end
  (even motion-only); measured vision-call + latency reduction.
- **Demo:** still scene -> no cloud calls; motion -> escalation; the cost graph vs ungated.

---

## 9. Phase 7 — Persistence + consent  *(V0.5)*

**Goal:** survive disconnect (endurance becomes real); safe `act` modality.

### 9.1 Build
- `persistence.py` — `Persistence` Protocol; `InMemory` default + a `Sqlite` reference. Durable
  goals / entities / fire-log; rehydrate on reconnect. This is what turns the *architectural*
  endurance property into a deployable claim.
- `consent.py` — **SDK primitives only:** profile flags (`allow_computer_use`); `act` routes only when
  the flag is set and the per-goal `modality == "act"`, scoped to a declared action set; the SDK
  **enforces** the gate, **logs** every action, and **emits** an event per action. (The consent
  *strip* and the "what I can see" affordance are **project 2 / the assistant** — it renders the
  events the SDK emits; do not build UI here.) The action log doubles as the V1 feedback corpus.

### 9.2 Tests
- **L1:** save/load round-trip; rehydrate reproduces goals/entities/fire-log; `act` blocked without the
  flag; action-log completeness.
- **L3:** kill + restart the session -> goals/timeline intact; an `act` goal **emits + logs an action
  event** per action (the assistant renders the strip from these — out of scope here).
- **DoD:** state survives a restart; `act` is flag-gated, scoped, logged, visible.
- **Demo:** leave it running, disconnect, reconnect -> the timeline + goals are intact (the endurance
  story).

---

## 10. Measurement, golden set & benchmark (cross-cutting)

Stand the scaffold up in Phase 0; mature it through every phase; the full benchmark lands with V0.0.

### 10.1 Golden set (`eval/golden/`, versioned, PR-reviewed)
```
scenes/        captured clips/frames + metadata (cam type, motion, lighting, #objects)
labels.jsonl   per-scene ground truth: entities(id,label,box-track,attrs[]), events[], rising-edge ts
gate_cases.jsonl  (scene_id, goal, expected_verdict) tagged: clear-fire · corrected-false ·
                  ambiguous-ask · occlusion · similar-objects · low-light · multi-object
utterances.jsonl  utterance + ASR-messy variant + gt-intent + gt-surface
splits.json    train 40% (calibrate) · dev 20% · test 40% (held-out, NEVER tuned on)
```
**Two data sources (minimal collection — the set measures two different things):**

1. **Benchmark axes (proactiveness / events / counting / latency) -> existing public datasets, no
   collection.** Adopt **Ego4D** + **Epic-Kitchens** (egocentric; action start/end = rising/falling
   edges for free), **Charades / Charades-Ego** (activity intervals), **RepCount** (rep counts), plus
   any TML-released sets. Human-labeled, licensed for research, and they make `percept-bench`
   reproducible (others rerun on the same public data). Confirm each license before adopting.
2. **Trust / gate scenarios -> a small hand-built set (~30–50 clips you record).** No public dataset
   has *"this is tea,"* *"ambiguous -> ask,"* the second-cup boundary, or the deictic exemplar. Record
   webcam/screen clips; label the **rising-edge timestamp** + the **expected verdict** per clip. This
   small set is the irreplaceable part — it populates `gate_cases.jsonl`.

**Labeling = VLM-proposes + human-reviews**, on *your own* clips. Cautions: VLM labels are the target
for the close-to-frontier agreement metric but **the trust metric needs human ground truth** (a VLM
can't ground-truth its own blind spots); **avoid YouTube** (wrong domain — produced/third-person vs
first-person webcam — and not redistributable). Keep the test split immutable.

### 10.2 Replay-and-align (the scoring core)
Replay a labeled clip through the system at production cadence; record every fire `(goal, ts, conf,
cost, tokens)`; align each ground-truth rising edge to a fire within `[edge, edge+omega]` ->
TP/FN/FP. Metrics: reaction latency **PAUC@omega**, precision (=1-false-fire), recall, count
off-by-one, id-stability, cost (¢/session, ¢/fire), context tokens. *A right answer at the wrong time
scores wrong.*

### 10.3 Calibration loop (the gate stops being magic)
Run the vision adapter over `scenes/` -> `(confidence, gt)` per attribute-class on the **train**
split; sweep threshold -> pick the value that zeroes false-fire while maximizing recall; write to
`config.gate.thresholds`; commit the sweep artifact. CI fails if test-split false-fire > 0.

### 10.4 Benchmark (`percept-bench`, public)
- benchmark-small (sampled, real models, CI) + benchmark-full (whole corpus + a **raw-VLM-in-a-loop
  baseline**, the publishable scorecard).
- **Tier 1 (headline):** proactiveness composite · ProactiveVideoQA(latency) · Charades(mIoU) ·
  RepCount(episode off-by-one) · false-fire. *Killer result:* Claude-in-Percept moves from ~0 toward
  native-model range without a native model.
- **Tier 2 (legibility, scope-flagged):** CueSpeak-shape (cue analog) · TimeSpeak-shape (table-stakes).
- **V0-specific:** the **oracle-cost baseline** (`$/perception-minute`, the control V1 must beat) and
  **per-stage latency** (p50/p95/p99 per demo path). `make reproduce` per row.

---

## 11. Dependency graph & sequencing for parallel agents

```
Phase 0 (contracts)  ── serial, first; everything depends on the frozen membrane
   │
   ├─► Phase 1 (cognition core + firing)         ← the heart; build before anything that fires
   │       │
   │       ├─► Phase 2 (cache + grounded VQA + replay scorer)
   │       ├─► Phase 3 (voice loop)  ── 3.1 learned routing rides the cache contract
   │       ├─► Phase 4 (identity + episodes)      ← needs entity graph
   │       └─► Phase 5 (events + scheduler)       ← needs the gate + frame source
   │
   ├─► Phase 6 (edge)        ← needs the WatchSpec/Tier0Signal contract (Phase 0) + a watch loop (3)
   └─► Phase 7 (persistence + consent)            ← needs the memory structures (1) + act route (3)

Measurement: scaffold in 0, scorer in 2, full benchmark + calibration across 2–5.
```

**Parallelizable after Phase 1:** {2 cache} · {3 voice} · {4 identity} · {5 events} are independent
seams once the core + contracts are frozen — partition one agent per seam. **Serial spine:** 0 -> 1
-> (2 | 3) before the first launch. **Verify-not-rewrite:** once Phase 1's core is green, later phases
add *beside* it (events beside attributes, episodes beside entities) — they must not edit the frozen
gate/graph contracts; if they need to, stop and flag.

**Critical paths to optimize around:** (a) *fastest-to-first-launch* = 0->1->2->3 + demos + README;
(b) *fastest-to-scorecard* = 0->1->2 + golden set + benchmark-small. Hit both early.

---

## 12. Per-phase definition-of-done (summary)

| Phase | DoD (all on top of: lint+types+full suite green, contracts unedited) | Demo line |
|---|---|---|
| 0 contracts | membrane frozen; `make test` green offline no keys; 3 lanes execute | `import percept` runs |
| 1 core+firing | one firing path; gate false-fire 0 on golden; t0–t3 caffeine restraint passes; refractory 100% | caffeine restraint trace |
| 2 cache+VQA | >=70% hit-rate, false-serve <= bound; replay scorer emits scorecard; grounded-VQA on ROI | still scene = <=1 call |
| 3 voice (+3.1 routing) | reactive turn end-to-end; routing >=95%; (3.1) learned tier Haiku-parity, 0 confident mis-routes | spoken Q -> streamed A |
| 4 identity+episodes | appearance re-id beats IoU on gap clips; episode off-by-one == target | new cup counted twice |
| 5 events+scheduler | event off-by-one == target; scheduler fires on time | "count the reps" |
| 6 edge | VAD endpoints on-device; down/up loop wired; measured call+latency drop | still scene = no cloud calls |
| 7 persist+consent | state survives restart; act flag-gated, scoped, logged, visible | reconnect -> timeline intact |

---

## 13. Standing rules for the build agent

1. **Never reintroduce an exact-attribute firing fork.** Semantic judgment -> three-state gate. One path.
2. **Never assert backend intelligence with a fake.** Fakes test our logic (L1); real models are L2/L3.
3. **The frozen contracts in `adapters.py` are immutable** within a phase — if a change is needed,
   stop and flag, don't edit silently.
4. **A fix ships with its regression case** — golden or unit. Nothing returns to hand-verified.
5. **Every tunable lives in `config.py`** with an env override — no literals in logic.
6. **The core never imports a vendor SDK** — lazy adapters behind Protocols + the name registry only.
7. **Stay in the envelope** — no surveillance / safety-critical / third-party-target demos or tests.
8. **Integrate frequently; keep the first release small** — Phases 0–3 ship, the rest are fast-follows.
