# Discussion — Edge capabilities, deployment topologies & dynamic skill provisioning

> **Status: OPEN / in-progress design discussion.** Not a spec, not a decision record yet.
> Captures a working conversation (2026-06-23) to be continued. The "Settled" parts are agreed
> framings; the "Open threads" need more discussion before any build.

---

## Why this doc exists
Working through *how capable the on-device edge (`percept-edge`) should be*, and whether the server
(brain) should provision skills to the edge on demand. The deployment architecture below is the
settled foundation; the edge-capability + skill-provisioning questions are the open frontier.

---

## Settled — the deployment architecture

### One cut-point, not free placement
There is exactly **one** network-spannable seam in the system: the **`WatchSpec` ↓ / `Tier0Signal` ↑
JSON contract** between the edge and the core. Everything else talks via in-process method calls and
is welded together.

- `vision` (`percept-vision`) — the brain. Defines the contract dataclasses (source of truth).
- `harness` (`percept-harness`) — wraps the core **in-process** (`GatedWatchLoop(core)` calls
  `core.perceive_judged(...)` directly — no JSON between them). It IS "gate + brain" in one deployable.
  It also hosts the **detector skill registry**.
- `percept-edge` — the **only relocatable half**: its entire conversation with the system is the JSON
  contract, so it can run wherever the camera is (browser / phone / Pi). JS precisely so it can live
  off-server, next to the camera.

**Rule:** the gate runs **exactly once**, on whichever side holds the edge. Never zero, never twice.

### What lives where
| capability | `percept-edge` (device) | `percept-harness` (server edge) | `percept-vision` (brain) |
|---|:--:|:--:|:--:|
| wire contract | mirrors (JS) | uses + `compile_watchspec` | **defines** (source of truth) |
| motion gate · VAD | ✅ | ✅ | — |
| detector **skills** (periodicity / acoustic-onset / pose) | ❌ adapter slots | ✅ (`pose` opt-in) | — |
| **which** skills run per goal | — | compiles | **authors** (conductor + presets) |
| VLM judge (perception seam) | — | — | ✅ |
| `PerceptionSpec` (how the VLM looks) | — | — | ✅ |
| three-state gate (fire/ask/silent) + rising-edge | — | — | ✅ |
| executor · entity graph · memory · the count | — | emits boundaries | ✅ consumes |

> **Correction captured:** the skills are **NOT on both edge + harness today.** The registry lives in
> `percept-harness` (Python). `percept-edge` has only the motion + VAD *primitives*; pose/periodicity
> are unported adapter slots. The `detectors` field now rides the wire (so the edge *receives* the
> selection) but the JS edge can't *execute* the skills yet. That port is the on-device parity gap.

### Topologies
- **A — all-server** (e.g. 20 cloud cameras / security): frames already centralized → run
  `percept-harness` in the cloud; its `MotionGate` is the edge, and it wraps the brain. **No
  `percept-edge`.** Full skills. ONE deployable.
- **B — edge on-device** (e.g. fish-tank / glasses): `percept-edge` gates locally, escalates only when
  something fires; server runs `percept-vision` (**brain only**) + a thin receiver shim. **Not** a
  second `GatedWatchLoop` (that would gate twice). TWO deployables. Skills = motion/VAD today.
- **A′** — a bare IP camera (no JS runtime at the source) falls back to A.

### Latency / cost / quality (per topology)
- **Meaning-quality is identical** in A and B — same brain, same model. Topology sets cost / latency /
  privacy, **not** the quality of the judgement.
- **B wins** bandwidth (only escalations cross the wire), time-to-detect-the-moment (local), and
  privacy (frames stay on-device). **A wins** simplicity (one deployable) and, *today*, moment-detection
  quality (full skill registry vs motion/VAD).
- The **VLM call is the only expensive unit**; the gate — wherever it runs — is the lever (measured
  ~83% call cut). Edge calibration matters: measured **edge event-recall 0.00** on subtle actions — a
  mis-tuned gate silently starves the brain of moments on either side.

---

## Settled — *which axis* the edge should grow on
Not "should the edge equal the harness," but **which axis**:
- **propose axis** — *"is this kind of moment happening?"* (cheap, local, high-recall) → **grow here.**
- **judge axis** — *"does it mean what the goal cares about, given who/what/history?"* (expensive,
  central, high-precision) → **never** on the edge; that's the brain, always.

**Test for porting a skill to `percept-edge` — needs all three:**
1. **continuous** — can't afford a cloud round-trip per frame;
2. **constrained** — battery / bandwidth / privacy at the source;
3. **locally cheap to detect** — DSP or a small model, not a frontier VLM.

### Use cases that bound it
| use case | edge needs | why |
|---|---|---|
| 20 cloud cameras (security) | nothing new — gate in cloud (harness) | hits none of the 3 → keep skills in harness |
| fish-tank distress | on-device gate; meaning stays server-side | continuous + privacy; "distress" is semantic = brain |
| **smart glasses — "tell me when my baby needs me; focus mode"** | **acoustic-onset + small cry-spotter, on-device** | signal is **audio** (cam not on the baby); hits all 3 → strongest case for a richer edge |
| **PDF → bullets** | ~nothing — structural gate + debounce | work is all brain (read + summarize + accumulate) |

- **Focus mode = a brain policy, not an edge one:** the brain raises the firing bar (consequence×novelty),
  keeps low-stakes silent/queued; the edge just keeps **recall** up (don't miss the cry). Edge = recall,
  brain = precision.
- Even the workout splits by grain: counting "missing reps" = `motion-periodicity` (light DSP, bundle);
  judging *form* = `pose` (heavier, lazy).

### Proposed edge roadmap (priority)
1. `acoustic-onset` + small sound-event/cry adapter — audio is the signal the camera can't see.
2. `motion-periodicity` — pure DSP, cheap, port-friendly.
3. `pose-openness` — heaviest; phone-class only, lazy adapter slot.
All: brain-selected (the `detectors` field), lazy by device budget, deliberate register-and-calibrate.

### Progress (2026-06-23) — `percept-edge` 0.5.0
- ✅ **JS skill registry shipped** (`src/skills.js`, commit `4243798`, 54 vitest green): the twin of
  `percept_harness/detectors.py` — registry (`registerSkill`/`makeDetector`/`manifestOf`/`listManifests`/
  `registered`/`signalOf`), `Boundary` + manifest matching the Python field names, **`MotionPeriodicity`
  + `AcousticOnset` ported** (pure DSP), and `SkillBank` (routes motion/energy/frame).
- ✅ **#2 lazy local load** mechanism: `SkillBank.load()` inits a model-backed skill once on activation.
- ✅ **`registerSkill(...)` extension point** live — the per-device customization surface.
- ✅ contract `detectors` field verified by execution (was inspection-only).
- ✅ **(a) `SkillBank` wired into the `Edge` loop** (commit `e7eda43`): `applyWatchSpec` builds one bank
  per goal from REGISTERED detectors (unknowns degrade → `unsupportedDetectors`); `processFrame` runs
  them every frame; `Edge.capabilities()` advertises the set; `Edge.loadSkills()` lazy-inits models.
- ✅ **(b) Boundary up-channel — Option 2 chosen + built**: `Boundary` is a frozen wire shape in
  `contract.js` (`makeBoundary`/`parseBoundary`, `kind ∈ {peak,onset,threshold}`); the edge emits a
  per-tick batch of `{goal_id, boundary}` on a **separate** `onBoundary` channel, mirroring the harness's
  per-event `perceive_boundary` dispatch. `Tier0Signal` stays frozen. 61 vitest green.
- ✅ **(c) capability handshake — brain side** (commit `84204ac`): `partition_by_capability(names, caps)`,
  `compile_watchspec(..., edge_capabilities=)` filters each goal's detectors to the advertised set, and
  `watchspec_unsupported(goals, caps)` reports the `(goal_id, dropped)` pairs for fallback.
  `edge_capabilities=None` = in-process topology A (whole registry). 55 harness L1 green. The edge half
  (`Edge.capabilities()`) was already done.
- ✅ **(e) topology-B wire transport** (commit `9446a4f`): JS `EdgeCapabilities` (advertise) +
  `BoundaryReport` (the framed `onBoundary` batch) in `contract.js` with make/parse; `Edge.advertise()`;
  Python **`RemoteEdgeReceiver`** — `on_capabilities` → `watchspec()` (capability-filtered down),
  `on_boundaries(report)` → `core.perceive_boundary` 1:1 (no gate, no re-detect). 65 JS + 60 harness green.
  **The full topology-B loop now closes: advertise → filter down → run skills → BoundaryReport up →
  server → perceive_boundary.**
- ✅ **(d) `pose-openness`** (commit `fdd8826`): the model-backed lazy JS skill — `openness =
  ankle-spread/shoulder-width + wrist-elevation` → inner `MotionPeriodicity` → peaks. Loads on
  `load()`, never at construct; the pose model is a **pluggable landmarker adapter** the product supplies
  via `registerPoseSkill(landmarker)` (mirrors `percept-harness[pose]`). 70 vitest green.
- ✅ **EDGE ↔ HARNESS PARITY proven** (`parity/compare.py`): identical inputs through the JS edge and the
  Python harness → **12/12 byte-for-byte MATCH** (motion-periodicity, acoustic-onset, motion-gate
  scores). So the edge route ≡ the harness route by construction → it inherits the harness route's
  precision/recall/proactivity (the routes differ only in WHERE the gate runs). **Edge latency ~5 µs/frame
  (~193k fps)** — negligible vs the ~1 s VLM call.

**The edge is feature-complete:** registry + all 3 skills (2 DSP + pose) + lazy-load + extension point +
handshake (both ends) + the full topology-A/B wire + parity-validated against the harness. Remaining is
optional: a fresh *live* P/R/proactivity re-run (needs the Perception-Test data re-fetched + a valid key).

---

## OPEN THREAD — dynamic skill provisioning (continue here)

**"Load skills on demand" is three different things** with very different costs:

| | what moves | cost / risk | status |
|---|---|---|---|
| **1. Activation** | nothing — config (`detectors:["pose"]` in the WatchSpec) | ~zero | **built** |
| **2. Lazy local load** | nothing over the wire — model on-device, loaded into RAM only when needed | battery/RAM only when active | **right next step** |
| **3. Runtime push** | skill **code/model travels server→edge** | security (RCE), latency, connectivity, calibration | **deferred v2** |

The workout example (*"watch my workout, nudge me on a miss"*) runs **today in #1+#2** — brain selects
pose, edge lazy-loads it, proposes rep boundaries, brain counts/judges/nudges, edge unloads after. No
push required.

**DECIDED (2026-06-23):** pursue **#1 + #2** (activation + lazy local load). **#3 runtime push = explore
later** (security/latency/calibration — see below).

### Per-device customization — the registry IS the product extension point
The reason #1+#2 matter: they let **each assistant built on Percept register the skill set *it* needs on
its edge.** Same `percept-edge` runtime, different registered skills:

| product | edge skills it registers | signal |
|---|---|---|
| **AI fitness coach** | `motion-periodicity` (reps) + `pose-openness` (form) | vision/pose |
| **AR glasses** (baby/focus) | `acoustic-onset` + cry-spotter | audio |
| **drone** | optical-flow-obstacle + altitude-hold | vision/IMU |

A product bundles **only** the skills it needs (small app, right capabilities), the edge **advertises**
that set up the wire, and the brain composes goals **against what's actually registered**. This is the
"skills library" through-line extended to the device: a curated, self-describing registry, brain-composed,
register-and-calibrate — never runtime-synthesized. The registry is the extension point third parties
build on; their custom skill is a `defineSkill({manifest, load})` + `registerSkill(...)`, nothing more.

**Why push (#3) is deliberately last** (aligns with build-plan locked decision #4 — "no runtime code
download, a deferred security surface"):
- **Security** — pushing executable code is an RCE channel (needs signing/verification/sandbox). Pushing
  *params* = low risk; *model weights* = medium; *code* = the Posture-C anti-goal (never).
- **Latency** — workouts tolerate a few-second cold fetch; "warn me of danger now" doesn't.
- **Connectivity** — push assumes online-at-activation; bundled-lazy works offline.
- **Calibration** — a freshly-arrived skill is uncalibrated for *this* device; skills are
  register-and-calibrate events.
- So push only earns its place as an **app-size optimization** (tiny-storage glasses), and even then:
  push **signed weights**, never code.

### The missing piece = the real next step: a **capability handshake**
Today the brain *commands* `detectors:["pose"]` blindly — nothing lets the edge say *"I have motion+VAD,
can lazy-load pose, but no cry-spotter."* Silent-failure risk (brain selects pose; glasses can't run it;
moment missed).

```
on connect:  edge → brain   "my skills: [motion, vad, motion-periodicity, pose(lazy)]"   (advertise)
per goal:    brain selects ONLY from what the edge advertised                              (negotiate)
mismatch:    degrade gracefully — coarser edge skill, OR run it server-side on escalated
             frames, OR (v2) fetch signed weights
```

Turns `detectors` from a *command* into a *negotiation* (brain proposes, edge confirms). Same manifest
contract the registry already uses, reported **up** the wire instead of only **down**. Cheap, safe, and
the prerequisite for any future push.

### Questions to resume on
- Where does the capability advertisement live — extend `Tier0Signal`, or a separate connect-time
  `EdgeCapabilities` message? (Keep the frozen `Tier0Signal` clean?)
- Brain-side "select within advertised" logic — where in the conductor / `compile_watchspec`?
- Graceful-degrade policy: when the edge lacks a selected skill, fall back to (a) coarser edge skill,
  (b) server-side detection on escalated frames, or (c) skip — per skill? per consequence level?
- Device-tier bundles: glasses vs phone vs Pi vs server — what ships in each JS build?
- First skill to actually port to JS: `acoustic-onset` (driven by the glasses/audio class).

---

## Decisions parked (do not re-litigate without revisiting)
- Edge is a **curated registry, brain-selected** — never runtime-synthesized detectors (Posture C).
- Gate runs **once**, on whichever side holds the edge.
- The brain (`vision`) **never** sees pixels — only `Verdict`/`Tier0Signal` streams.
- `gate.py` semantics unchanged by any of this — levers change how a `Verdict` is *produced*, not gated.
