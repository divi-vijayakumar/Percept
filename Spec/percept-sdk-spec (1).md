# Percept — SDK Specification

> **What this is.** The architecture spec for `percept`, the open-source cognition layer for
> goal-driven, proactive vision agents. This is the *what and why*. The companion
> `percept-implementation-plan.md` is the *how* — a phase-by-phase build a coding agent can
> execute from scratch.
>
> **Status:** authoritative · supersedes prior SDK specs · folds in the firing-contract fix, the
> layer/memory model, the redrawn open-core line, the honest benchmark, and the right-sized consent.

---

## 1. What Percept is

You state a goal in plain language — *"nudge me when I drink a coffee"*, *"tell me when the kettle
boils"*, *"warn me when the build breaks."* Percept turns a live audio-visual stream (camera /
screen / mic) into an agent that:

- **reasons over entities across time** — a tracked thing with a stable id, carrying attributes with
  provenance (*observed* vs *you-told-me*) and freshness;
- **fires proactively** but only on the **rising edge** of a condition becoming true;
- **refuses to guess** — a **three-state gate** maps *known → act · not → silent · unknown → ask*, so
  it never delivers a confidently-wrong nag.

The differentiator is the **cognition layer** (entity memory + three-state gate + temporal
reasoning), which a raw VLM-in-a-loop and a per-frame pipeline both lack. Percept builds on frontier
models for perception/reasoning behind vendor-neutral seams.

### 1.1 The envelope (a design refusal, stated in the README)

- **Assistant-class, never the safety mechanism.** Percept informs a human who stays responsible. It
  does not offer bounded/certifiable latency and is never the thing standing between a person and harm
  on a clock it can't guarantee. *Out:* driver-drowsiness intervention, turn detection. *In:* a
  post-hoc driving debrief, a parked-car walkaround.
- **Watches the user's own world, with the user as beneficiary** — never a non-consenting third party
  to the wearer's advantage. *Out:* covert microexpression analysis of someone you're negotiating
  with; card-counting against a live casino. *In:* a play-money practice trainer.

The envelope is a feature: it signals the project has thought about misuse, which is itself a
differentiator in this category. **Do not lead with surveillance demos.**

### 1.2 Positioning

- **Not a media-orchestration framework** (Pipecat, Stream Vision Agents own that). Percept is the
  opinionated cognition layer *above* the glue — and can run *as* a processor inside them.
- **Not a surveillance product** (Ambient.ai owns that). Percept is personal, developer-facing,
  open-goal, on-device-first.
- **The wedge is temporal cognition.** Raw VLMs are reactive, memoryless, per-frame. Percept
  remembers entities, binds facts to them, reasons over events across time, and fires proactively.
  *Honest framing:* the cognition layer is a head-start, not a fortress — the kill-gate is real pull,
  not a clever architecture.

---

## 2. The open-core line (drawn on defensibility, not effort)

The rule: **if a competent team will build it anyway, it goes in OSS.** Gating commoditizing work
behind paid protects nothing and only slows adoption — and adoption is the entire job of the OSS
layer. The two genuinely defensible things are reserved for paid.

| Layer | What | Why here |
|---|---|---|
| **OSS — the complete reactive framework** | goals · three-state gate · entity graph + attribute provenance/validity · **re-id by appearance (reference impl)** · **episode counting** · **events** · **scheduler** · perception cache · **on-device reactive edge (VAD + VID bank)** · the benchmark + oracle-cost baseline | none of it is deep tech; it commoditizes. Own the category by giving it away — offense, not charity. |
| **Paid V1 — anticipation** | **predictive VID** — perceive *before* it fully happens, closing the gap to native-VLM latency *without* a native model | genuinely hard; flywheel-dependent (predict what-precedes-what from accumulated patterns); compounds with adoption, can't be cloned from a weekend read. The technical moat. |
| **Paid V1 — hosted enterprise certainties** | SLAs · governance · multi-tenancy · audit · DLP · the **scale-calibrated** re-id and gate (telemetry flywheel) | ops + trust + scale. Where the revenue is. |

OSS feeds adoption → adoption feeds the flywheel → the flywheel powers anticipation → anticipation +
hosting is the paid tier. Each layer strengthens the next.

> **Caution carried into the plan:** "OSS vs paid" is *not* "first release vs everything-at-once."
> The OSS *scope* grew; the first *release* still ships small (§10). And the give-everything-away bet
> rests on anticipation + hosting being defensible **and** monetizable — validate pull before
> committing.

---

## 3. The layer model — and where memory lives

The single clarifying idea: **the perception layers hold almost no memory — they are reflexes. All
the memory that matters lives in the cognition core.**

| Layer | Where | Job | Memory it holds |
|---|---|---|---|
| **Edge** | on-device (WASM) | "Is something *goal-relevant* happening? Worth a cloud call?" | almost none — the current `WatchSpec` + a 1–2 frame motion buffer. No entities, no history. |
| **Cache** | server | skip the VLM if the ROI hasn't changed | short-lived `(roi_hash, goal_id) -> verdict`, TTL'd in seconds. Forgettable. Pure cost. |
| **VLM (eyes)** | cloud | judge one frame: "is this coffee? a new cup?" | none — stateless per call. Answers the question the core poses. |
| **Core** | server (the SDK) | remember + decide | **everything that matters** — three structures below. |

Inside the **core**:

- **Goal registry** — remembers the *goal*. `{id, condition (free text), say, source, modality, state}`.
- **Entity graph** — remembers the *things*. `Entity{id, label, geometry, appearance_signature,
  first/last_seen, attributes}`. Identity = "is this the same thing I saw before?"
- **Attribute records** (inside entities) — remember each *fact*, with **who said it**
  (`observed | asserted`) and **how fresh** (`confidence`, `last_confirmed` / validity).
- **(temporal) Episode track** — per counting goal, a bounded log of episodes for "is this a new
  cup or the one I've been drinking?"

The **gate** is not memory — it's the decision function over the above.

### 3.0 Deployment topology (conceptual roles vs units of code)

The four rows above are *roles in the pipeline*, not four things you deploy. As **deployment units there
are only two of your code**, and for the first launch only one:

- **Core (server-side)** — the brain (cognition, gate, goals, entities, episodes) **plus the
  perception cache as a module inside it**. A library you run in a server process.
- **Edge (on-device)** — the reflex gate (VAD + VID + a coarse motion gate). A WASM package, added in
  V0.4 for always-on economics. *Optional for V0.0.*

Everything else is not a unit you deploy:

- **VLM / STT / TTS** are **third-party APIs the core calls out to** — "lives in cloud" just means an
  outbound API call, not a layer you run.
- **The perception cache is server-side, in the core** — it's keyed on `(goal, roi_hash) -> last
  verdict`, state only the core holds. The edge has a *different, coarser* gate (raw-pixel motion,
  "don't even send the frame"); the two compose (edge gate before the network, core cache before the
  cloud call).
- **The harness is the server-side transport shell around the core**, not a second brain. The
  alternative shell is the SDK's built-in WS bridge (also server-side, zero-infra, for demos).
  `harness` vs `bridge` is just *which transport dresses the core*.

```
ON DEVICE            YOUR SERVER                          EXTERNAL (3rd-party APIs)
percept-edge   --->  core (brain + perception cache)  --> VLM (Claude/Gemini)
(VAD+VID,            wrapped by:  built-in bridge (demo)   STT (Deepgram/SenseVoice)
 coarse gate)                  or percept-harness (prod)   TTS (Deepgram/Kokoro)
```

**Minimal V0.0 = one unit:** the server core + its built-in bridge, calling cloud APIs. No edge, no
harness required. Edge and harness are upgrades you add for always-on / production.

### 3.1 Worked example (the model in one trace)

Goal: *"keep me off caffeine — nudge me when I drink coffee."*

| t | event | layer activity | gate |
|---|---|---|---|
| t0 | goal set | Core/Goals stores `g_1`; Edge gets a `WatchSpec` (watch cups/drinking) | — |
| t1 | cup1 up | Edge escalates -> VLM "maybe coffee, unsure" -> Core mints **cup1**, `contains=observed-uncertain` | **UNKNOWN -> ask** *"is that coffee?"* -> user: "non-caffeinated tea" -> `cup1.contains=(tea, asserted, @t1)` |
| t2 | cup1 up | Edge escalates -> Core **re-ids cup1**, `contains=tea` fresh | **KNOWN-FALSE -> silent** |
| t3 | long gap, cup1 up | Cache TTL expired -> re-perceive; Core re-ids cup1 but `tea` is **stale** | **stale -> UNKNOWN -> ask** *"still tea?"* -> "yes" -> freshness refreshed |
| t4 | cup2 up | Edge escalates -> Core re-id: **no match -> new entity cup2**, `contains=unknown` | **fire** *"that looks like a new cup — stay off caffeine"* |

The edge **gates**; the core **remembers and judges**; the VLM **answers**. Recognizing cup1 across a
gap and distinguishing cup2 is the **re-id** job — IoU continuity in the simplest case, appearance
matching when the object left tracking.

---

## 4. Packages

| Package | Install | Is | OSS? |
|---|---|---|---|
| `percept-vision` (`import percept`) | `pip install percept-vision` | the cognition core SDK | yes |
| `percept-harness` | `pip install percept-harness` | LiveKit Agents worker — production WebRTC transport + STT/TTS wiring + server-side tier-0 gate | yes |
| `percept-edge` | `npm i @percept/edge` | **on-device** VAD + VID bank (JS/WASM) — *not pip* | yes |
| `percept-assistant` | (app, not a lib) | the reference consumer app (UI + voice + goals) | yes |

One-directional dependency: assistant -> (harness | bridge) -> SDK; edge speaks the SDK's `WatchSpec`
/ `Tier0Signal` contract over the wire. The core never imports a vendor SDK directly.

> **Scope line — this spec is the SDK only (project 1).** `percept-assistant` is **project 2**, a
> separate build that lives *on top of* the SDK. The SDK is a library with **no UI, no voice persona,
> no product goals** — it owns the mechanism and the seams; the assistant owns the orb UI, the voice,
> the consent strip, the "what I can see" affordance, and the polished demo goals. The SDK's only
> obligation toward project 2 is to expose those seams cleanly so the assistant is built *on* it, never
> a fork of it.

---

## 5. The cognition core

### 5.1 The firing contract (the load-bearing lesson)

**Do not compile the goal to a rigid attribute and exact-match it.** That path silently dies on real
video (the model says "coffee cup", never `{contains: coffee}`) and is invisible because fakes can be
hardcoded to satisfy it. Instead:

```
vision.judge(condition: str, frame, roi?, exemplar?, corrections?) -> Verdict{satisfied, confidence, evidence}
gate.judge(verdict, freshness) -> FIRE | ASK | SILENT
```

The judge receives the goal's **plain-language condition** and judges *"is this true right now?"* The
**three-state gate** maps the judgment to an action, with **rising-edge** (fire on false->true only)
and **refractory** (no double-fire) on top. One path, one gate — the attribute path is a degenerate
case, never a fork.

### 5.2 The three states

| Verdict | Action | Rule |
|---|---|---|
| confident yes | **FIRE** (rising edge only) | the condition just became true |
| confident no | **SILENT** (re-arms the edge) | clearly false |
| low-confidence / stale / missing | **ASK** | never a confident-wrong nag — the trust property |

### 5.3 Memory structures (DTOs)

```
Goal      { id, condition: str, say: str, source, modality: speak|act,
            state: {type: watch|count|remind, count: int}, expiry }
Entity    { id, label, box: Box, appearance: Signature, first_seen, last_seen,
            attributes: {key -> Attribute}, sources: set }
Attribute { value, source: observed|asserted, confidence, last_confirmed }   # validity = freshness
Episode   { id, goal_id, started_at, last_seen, exemplar, representative }     # per counting goal
Verdict   { satisfied: bool, confidence: float, evidence: str }
```

**Invariants:** *assertion beats observation* (a user "this is water" wins over a detection);
*asserted facts go stale* (re-confirmed after a validity window, the t3 re-ask); *identity is
established below the LLM and handed up as a symbol* (never re-derived from pixels turn to turn).

### 5.4 Re-identification (the "same thing?" question)

- **V0 reference:** IoU + label continuity (frame-to-frame) plus **appearance matching** (an embedding
  / signature) so an object is recognized after it leaves and re-enters tracking, and a new object is
  distinguished. Ship a brute-force-over-the-recent-set matcher behind an `Identity` Protocol.
- **Bounded-claim discipline:** state plainly that continuity is correct for *single dominant object,
  near-static camera*; the demo must not imply generality.
- **Paid V1:** the fleet-calibrated, high-accuracy re-id *service* (benefits from aggregate data).

### 5.5 Episode counting (the cup problem)

Counting "cups of coffee" is **episode counting**, not sip-counting (over-counts) and not
IoU-track-counting (mis-counts across gaps). Mechanism:

- a per-goal **episode log**, bounded by the counting window ("today's cups" — a handful);
- on each candidate drinking-event, the **LLM judges new-episode-vs-continuation** against the log
  (hand it the episode summaries + current exemplar);
- count = distinct episodes.

This stays inside the endurance story (the set is bounded by the window, not all history). A vector
index (e.g. moss.dev) is a *scale pre-filter on the match*, never the mechanism, and never on the hot
path.

### 5.6 Events (temporal perception)

A **pass / a rep / a discrete action** is not an entity-with-an-attribute — it's an event over a span.
Add a parallel **event track** beside the object/attribute track (additive, not a rewrite):

- a rolling **frame window** (replaces keep-latest);
- `vision.detect_events(frames, goals) -> [Event]`;
- an **event gate** reusing the three-state semantics;
- **event identity / dedup** (the temporal analog of IoU continuity) so a multi-frame event counts once.

---

## 6. The perception layer

### 6.1 Narrow grounded VQA (the single biggest quality/cost lever)

Stop calling the cloud VLM as *"describe the scene."* Call it as a **narrow grounded question about a
cropped ROI**: *"is the user drinking from THIS mug <exemplar>?"* This is a pure adapter change and a
large quality + cost + latency win. Attach the deictic full-res frame as an **exemplar** when the user
demonstrates ("this mug") — the fix for cup<->headphones / coffee-vs-water confusions (weak grounding,
never a threshold problem).

### 6.2 The perception cache (cost/latency only — not the moat)

`FrameHashLayer`: per-goal **ROI-cropped dHash**, Hamming near-match, `confidence =
min(verdict.conf, sim*age_decay)`. Skips the VLM when the goal's ROI hasn't changed (the 70–90% skip).
Invalidation order: goal-change -> assertion-override -> TTL -> L0 frame-diff. **False-serve is the
cache's analog of false-fire** — re-perceive a random epsilon% of HITs and compare; CI fails if
test-split false-serve exceeds the bound. Ships a *contract + cascade*; an empty cascade always DEFERs
(safe). TTL is a hard ceiling that bounds the slow-drift the hash can't see.

### 6.3 The edge (salience gating)

The perception analog of voice VAD: an on-device pass that decides *when* to spend a cloud call.

- **down:** the brain compiles active goals to a `WatchSpec` (anchors + thresholds);
- **up:** the edge reports a `Tier0Signal` (motion / structural / semantic salience);
- the expensive vision call runs only when salience clears threshold.

VID detectors are **pretrained, zero user data** (frame-diff is arithmetic; pose is MediaPipe; CLIP is
zero-shot); the threshold is a **shipped default** from an auditable sweep, with optional per-goal
bias. A fresh install works with no accumulated usage. **Wire the loop end-to-end (even motion-only):
`compile_watchspec`'s output actually consumed by an on-device gate that reports `Tier0Signal` back
up** — that single wired loop is the proof point.

---

## 7. The seams (the membrane — FROZEN public surface)

Everything vendor-specific sits behind a `Protocol`, resolved by name through a backend registry.

| Seam | Method(s) | `fake` | real |
|---|---|---|---|
| `Vision` | `judge(condition, frame, roi?, exemplar?, corrections?)` · `detect_events(frames, goals)` | `FakeVision` | Claude / Gemini |
| `Specialist` | `stream(...) -> tokens` | `FakeSpecialist` | Claude / Gemini |
| `Conductor` | `run(event, utterance) -> Decision` | `FakeConductor` | Claude (Haiku) |
| `STT` | `stream(audio) -> events` | `FakeSTT` | Deepgram / SenseVoice |
| `TTS` | `push(text)` · `flush()` | `FakeTTS` | Deepgram / Kokoro |
| `Identity` | `match(obs, known) -> id \| None` | continuity ref | appearance / V1 ReID |
| `CacheLayer` | `lookup` · `store` | exact-match ref | `FrameHashLayer` |
| `Edge` (wire contract) | `WatchSpec` down · `Tier0Signal` up | server motion gate | `@percept/edge` |
| `Persistence` | `save` · `load` (goals/entities/fire-log) | in-memory | sqlite / kv |

**Fakes are load-bearing** (deterministic, free, offline) — they enable the CI gate, isolate "our
logic" from "the vendor", power reproducible demos, and let `pip install` -> run with no keys. **But
never assert backend *intelligence* with a fake** — that's exactly how the firing-contract break hid.

---

## 8. Consent (V0 — right-sized)

The always-on sensor's trust posture, kept minimal for V0. Split by project — the **SDK provides the
primitives**, the **assistant (project 2) provides the UI**:

- **Profile flag** *(SDK)* gates the higher-trust modality: `act` (computer-use) is opt-in via
  `allow_computer_use`, scoped to a declared action set; the SDK enforces it.
- **Action log** *(SDK)* — every fire and every act recorded (the audit seed + the V1 feedback corpus).
- **Consent strip** *(assistant)* — states what the agent may do and shows each action as it fires.
  Nothing acts silently. The SDK exposes the events; the assistant renders the strip.
- **"What I can see"** *(assistant)* — thumbnails of the last camera/screen frame actually looked at,
  "looked 3s ago." The SDK emits the looked-at events; the assistant renders the affordance.

The full capability-mandate / MCP-host / AP2 apparatus is the **V1 design of record**, not V0. Same
trust guarantee from a profile flag + a visible strip; a fraction of the surface.

---

## 9. Measurement — the trust thesis is measured, not asserted

Three layers, three questions, one shared golden corpus.

- **Tests (L1)** — "does the code work?" Unit + integration + e2e on **fakes**; binary; every commit;
  no keys. CI hard gates: gate false-fire **0**, refractory **100%**, routing **>=95%**.
- **Eval (L2)** — "how good is the cognition, improving?" Metrics on the golden corpus via
  **replay-and-align** (replay clip -> record fires -> align to ground-truth rising edges within omega
  -> TP/FN/FP). Metrics: reaction latency (PAUC@omega), precision, recall, count off-by-one,
  id-stability, cost (¢/session, ¢/fire), context tokens. *A right answer at the wrong time scores
  wrong.* Tracked vs own history on a frozen train/test split.
- **Benchmark (L3)** — "better than the alternative?" Same machinery vs a **raw-VLM-in-a-loop
  baseline** on the identical corpus.

**Benchmark scorecard — two tiers (honest about lane):**

*Tier 1 — moat axes (headline; a VLM-loop scores ~0):* proactiveness composite (recall x promptness x
precision) · ProactiveVideoQA -> reaction latency · Charades start/stop -> rising/falling edge (mIoU) ·
RepCount -> **episode** counting (off-by-one; mark object-identity vs episode honestly) · false-fire
(precision). *Killer result:* Claude-in-Percept moves from vanilla-model ~0 toward native-model range
**without a native model**.

*Tier 2 — legibility axes (secondary; TML vocabulary, scope-flagged):* CueSpeak-shape *(cue analog —
perceptual cue->timed fire only; literal CueSpeak is voice-native, out of lane)* · TimeSpeak-shape
*(table-stakes — a scheduler nails it; included for completeness)*.

**Two V0-specific measurements that make V1 credible:**

- **Oracle-cost baseline** — V0 is cloud/Haiku-everything; publish `$/perception-minute` per demo as
  the *control* the V1 VID must beat. Without it the "VID cut COGS N×" claim is non-credible.
- **Per-stage latency** — instrument every boundary (edge -> conductor -> specialist -> TTS); publish
  p50/p95/p99 per demo path.

Publish reproducibly via a separate public `percept-bench` repo (harness + sampled corpus +
`make reproduce`).

### 9.1 Where the golden data comes from (two buckets — minimal collection)

The golden set measures two different things; each takes a different source:

- **Benchmark axes (proactiveness / events / counting / latency) -> existing public datasets. No
  collection.** Reuse **Ego4D** and **Epic-Kitchens** (egocentric — the right domain — with action
  *start/end* timestamps = rising/falling edges for free), **Charades / Charades-Ego** (activity
  intervals -> the Charades axis), **RepCount** (-> the RepCount axis), plus whatever TML released.
  These are human-labeled, properly licensed for research, and make `percept-bench` *reproducible*
  (anyone reruns on the same public data). Confirm each dataset's current license before adopting.
- **Trust / gate scenarios (the Percept-specific UX) -> a small hand-built set (~30–50 clips).** No
  public dataset contains *"this is tea, not coffee,"* *"ambiguous -> ask,"* the second-cup boundary,
  or the deictic exemplar. Record these yourself (webcam / screen), label two things per clip: the
  **rising-edge timestamp** and the **expected verdict** (fire/silent/ask). This small set is the
  irreplaceable part.

**Labeling method:** VLM-**proposes**, human-**reviews** — on *your own* clips, not scraped video.
Two cautions: (1) **circularity** — VLM labels are fine as the target for the "close-to-frontier"
agreement metric, but the **trust metric needs human ground truth** (a VLM labeler can't ground-truth
its own blind spots — exactly where confident-wrong hides). (2) **Avoid YouTube** for the published
benchmark: wrong domain (produced/third-person vs first-person webcam) and not redistributable; use
the licensed research datasets instead.

---

## 10. Phases (the release sequence)

OSS scope is large; the **first release is small**. Ship the clean reactive core to plant the flag,
then release the rest as OSS fast-follows — each a "now it does more" moment.

| Phase | Ships | The "now it does more" |
|---|---|---|
| **V0.0 — reactive core** *(first launch)* | contracts + fakes · entity graph + attributes/provenance/validity · three-state gate · **semantic firing** · goal registry · narrow grounded VQA · perception cache · reactive voice loop · the golden set + the benchmark + oracle-cost baseline · the envelope README · the caffeine *restraint* + boil scenarios **as test/bench fixtures** (the *product* versions of these are project 2) | a proactive watcher that **refuses to guess**, with a published scorecard |
| **V0.1 — learned routing** | data-driven/embedding router replacing brittle regex; defer-on-miss structural | language-agnostic, non-brittle routing |
| **V0.2 — identity + counting** | appearance re-id (the cup problem) + the episode track | recognizes things across gaps; counts cups correctly |
| **V0.3 — events + scheduler** | rolling window + event gate + event-typed goals; the scheduler (TimeSpeak) | counts reps/passes; time-triggered reminders |
| **V0.4 — on-device edge** | `@percept/edge` VAD + VID bank; the WatchSpec-down/Tier0Signal-up loop wired end-to-end | always-on cost/latency/privacy unlock |
| **V0.5 — persistence + consent** | persistence port (durable goals/entities/fire-log); the consent **primitives** (act profile-flag + action log; the strip UI is project 2) | survives disconnect; **endurance becomes real**; safe `act` modality |
| **V1 — anticipation + hosted** | predictive VID; hosted enterprise (SLA/governance/scale-calibration) | the latency endgame + the business |

**Endurance note:** V0 demonstrates the *architectural* endurance property (bounded structured state ->
flat token curve over a long supervised session). "Leave it on for weeks, on-device, unattended" needs
persistence (V0.5) + edge (V0.4) — that's when endurance becomes a deployable product claim.

---

## 11. Key invariants

1. **Never confidently wrong.** Ambiguity -> `ask`, never a wrong `fire`. The gate is the trust surface
   (eval-gated at 100%).
2. **When in doubt, defer.** The router fast-paths only confident cases; everything else falls to the
   conductor. A false defer costs latency; a false fast-path could skip goal authoring.
3. **One firing path, one gate.** Semantic judgment + rising-edge + refractory — no env-flag fork.
4. **Identity below the LLM.** Stable ids from continuity/appearance, handed up as symbols.
5. **Provenance + freshness everywhere.** Asserted beats observed; asserted goes stale and is
   re-confirmed.
6. **Caching ships a contract, not a policy.** Empty cascade defers; the cache is trust-but-verify
   (false-serve audited).
7. **Vendor-neutral core.** The core never imports a vendor SDK; backends are lazy adapters behind
   Protocols + a name registry.
8. **Memory lives in the core.** Edge and VLM are stateless reflexes; the retrieval/index question is
   a V1 *cold-path* concern, never the hot loop.
9. **Stay in the envelope.** Assistant-class, user-as-beneficiary; refusals stated proudly in the
   README.

---

## 12. V1 — what's deferred and why

- **Predictive VID (anticipation)** — perceive before it fully happens; the latency endgame, the
  technical moat. Flywheel-dependent (the V0 feedback log is its training seed).
- **Hosted enterprise** — SLA, governance, multi-tenancy, audit, DLP, and the *scale-calibrated* re-id
  / gate. The revenue.
- **Recall at scale** — cross-session semantic recall behind the `Identity`/recall seam; an in-memory
  brute-force default ships in OSS, a vector runtime (moss.dev / FAISS) is a swappable adapter, fleet
  calibration is paid.

The forward-compat contract: V0 exposes the seams, the schema, and the event vocabulary so V1 is
`percept` + modules/services, **never a fork**.
