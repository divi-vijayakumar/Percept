# Concepts

Percept splits cleanly into two layers. Understanding the split is the key to using it well.

```
┌──────────────── COGNITION (the "brain") ─────────────────┐
│  the three-state gate · rising-edge + refractory         │
│  the BehaviorExecutor · the concern/consequence framework│
│  the entity graph · the edge-detector registry           │
│  general · stateful (has memory) · deterministic · cheap │
└──────────────────────────────────────────────────────────┘
                     ▲  feeds on Verdict{satisfied, confidence, evidence}
┌──────────────── PERCEPTION (the "eyes") ─────────────────┐
│  vision.judge(condition, frame) -> Verdict   ★ ONE seam  │
│  stateless · per-call VLM (Gemini / Claude / fake)       │
└──────────────────────────────────────────────────────────┘
```

- **Perception** is a stateless, per-call VLM. Its entire contract is one method:
  `vision.judge(condition, frame) -> Verdict{satisfied, confidence, evidence}`. It answers "is this
  plain-language condition true of this frame, right now?" and nothing more.
- **Cognition** is everything else. **The brain never sees pixels** — it reasons only over the
  `Verdict` stream. That's what makes it cheap, deterministic, and backend-agnostic.

Everything below is in the cognition layer.

For the cheap "should we spend a VLM call?" layer that runs before perception, see
[Edge Wake Policy](edge-policy.md).

---

## The three-state gate

The gate is the **trust surface**. It maps a `Verdict` (plus a freshness value) to exactly one of
three actions:

| condition                                       | action     |
|-------------------------------------------------|------------|
| confident *yes*, fresh, on a **rising** edge    | `FIRE`     |
| confident *yes*, but already fired / not rising | `SILENT`   |
| confident *no*                                  | `SILENT` (re-arms the edge) |
| low-confidence **or** stale fact                | `ASK`      |

The third state — **ASK** — is the whole point. When confidence is below threshold (default `0.7`)
or a supporting fact has gone stale, the gate **refuses to guess**. It asks instead of delivering a
confidently-wrong nag. This is the "refuse-to-guess" property: ambiguity never becomes a confident
fire.

```python
from percept.gate import Gate, FIRE, ASK, SILENT
from percept.adapters import Verdict

gate = Gate(threshold=0.7)
k = ("goal-id", "entity-id")
gate.evaluate(k, Verdict(satisfied=True,  confidence=0.9))   # FIRE   (rising edge)
gate.evaluate(k, Verdict(satisfied=True,  confidence=0.9))   # SILENT (already fired)
gate.evaluate(k, Verdict(satisfied=True,  confidence=0.4))   # ASK    (too unsure to fire)
```

You normally never call the gate directly — `agent.perceive_judged(...)` runs every active goal
through it for you and returns the resulting `FireEvent`s. `FireEvent.action` is `"fire"` or
`"ask"`; `SILENT` produces no event.

---

## Rising-edge firing + refractory

Percept fires on the **rising edge** — the moment a condition flips **false → true** — not on every
frame where it's true. After firing, a per-key **refractory** holds it silent until the condition
goes confidently false again (which **re-arms** the edge). So:

- A condition that's true for 100 frames fires **once**, not 100 times.
- It can fire **again** only after it has been confidently false in between.

The gate key is `(goal_id, entity_or_episode_id)`. Because the *entity* is part of the key, a
**brand-new entity gets a fresh edge**: if you're watching "the user is drinking coffee" and a new
cup appears, it fires for the new cup even though the old cup already fired. The same cup never
double-fires.

A user correction (`agent.assert_attribute(...)`) clears the edge so the corrected fact is re-judged
rather than suppressed by a stale edge.

---

## BehaviorProgram vs PerceptionSpec

Every concern has two orthogonal halves. Keep them straight:

- **BehaviorProgram — WHAT fires.** The cognition-side program: `sense · key · accumulate`, plus the
  re-arm rule. It decides *when an event counts as a fire*.
- **PerceptionSpec — HOW to look.** The vision-side twin: `modality · temporal · roi`. It decides
  *what input the eye is given* for the judge call.

Both default to today's behavior, so an un-authored goal is a plain single-frame watch.

### BehaviorProgram (`percept.behavior.BehaviorProgram`)

- **sense** — `judge` (default), `transition` (fire on A→B, but only after A was seen),
  `episode` (count distinct occurrences), `absence` (fire when something is overdue).
- **key** — how the re-arm boundary folds into the gate key. This is the *one general re-arm
  mechanism*: `on_false` (default), `never` (a confident-false never reaches the gate), `each_event`,
  `interval` (a time bucket), `deadline`.
- **accumulate** — an optional fold over the event stream. `fire_when` gates the user-facing fire on
  a count, e.g. `"n == 3"` for "alert me the 3rd distinct time". (Authored `fire_when` expressions
  are parsed with an AST whitelist over `{n, t}` — never `eval()` — so an LLM-authored expression
  can't be an injection vector.)

### PerceptionSpec (`percept.behavior.PerceptionSpec`)

- **modality** — `vision` (default), `audio`, `fused`.
- **temporal** — `single` (one frame, default), `strip` (a few frames for context), `window`.
- **roi** — `focus` (the in-focus entity's box, default), `hint` (an authored region), `full`.

Defaults reproduce today's behavior byte-for-byte: a single, vision-only, focus-cropped frame.

---

## The edge-detector registry (cheap on-device "skills")

Some questions shouldn't cost a cloud VLM call at all. **Counting reps** is the canonical example:
"how many jumping jacks?" is an oscillation-counting job, not a per-frame semantic-vision job.

The edge-detector registry (installed through `percept-vision[edge]`, implemented internally in
`percept-harness`) is a curated, registry-extensible library of
**cheap, high-fps signal detectors** the brain selects from per goal — the *skills* model. Each
detector consumes a scalar signal stream and emits **timed boundaries** (a rep peak, a sound onset).
The pattern is **edge-proposes · brain-counts · VLM-confirms**: the edge says *where* something
happened cheaply, the brain counts the boundaries, and the VLM confirms the semantics only sparingly.

Built-in detectors:

| detector              | senses                          | use it for                                          |
|-----------------------|---------------------------------|-----------------------------------------------------|
| `motion-periodicity`  | motion magnitude (frame-diff)   | counting repetitive motion — reps, strokes, paces   |
| `acoustic-onset`      | audio energy (the mic stream)   | audio-cued events — a boil hiss, a timer beep, a knock |
| `pose-openness`       | BlazePose keypoints (opt-in)    | fast whole-body reps — jumping jacks, squats, push-ups |

Detectors are **opt-in and brain-selected**, never synthesized at runtime: adding one is a deliberate
`register(impl, manifest)` + calibrate event. The brain *composes* registered detectors by name (via
the manifest's `when_to_use`); it never *writes* them. `motion-periodicity` and `acoustic-onset` are
pure stdlib DSP (no numpy, no model); `pose-openness` lazily loads MediaPipe only on first use.

---

## The consequence × novelty escalation ladder

A concern that keeps reading **ambiguous** (a run of low-confidence / `unknown` verdicts) shouldn't
just re-ask the human every tick. Instead Percept **escalates**, climbing a ladder from cheap to
expensive:

```
single frame  →  + temporal strip  →  + fused audio  →  re-invoke the LLM (refine the program / ask)
└──────────── cheaper: "look harder" ───────────┘   └──── more expensive: "think harder" ────┘
```

Two dials govern the climb:

- **Consequence** sets *how soon* to escalate. A `severe`-miss-cost concern escalates after fewer
  ambiguous reads (threshold 2); a `mild` one waits longer (threshold 4); a `trivial` one never
  spends an LLM call.
- **Novelty / recurrent ambiguity** is the *trigger*. A calm, confident concern costs nothing —
  neither rung fires. A per-concern cooldown acts as a deliberative refractory so the LLM rung can't
  thrash.

The cheap perception rungs (strip → audio) always climb first; only once perception is maxed and the
concern is *still* ambiguous does Percept re-invoke the conductor (the LLM) to refine the program or
decide to ask. When a confident read finally resolves the ambiguity, the ladder **decays** back to
the cheap single-frame default.

You set the dial via a goal's `state["miss_cost"]` (`"trivial"` / `"mild"` / `"severe"`).

---

## How the pieces connect

```
add_goal(Goal(condition, say))            # state a goal in plain language
        │
        ▼
perceive_judged(frame)                    # per frame, per active goal:
  ├─ vision.judge(condition, frame) ────► Verdict{satisfied, confidence}   (PERCEPTION)
  ├─ classify(verdict, freshness) ──────► yes / no / unknown
  ├─ gate.evaluate(key, verdict) ───────► FIRE / ASK / SILENT              (the gate)
  └─ BehaviorExecutor ──────────────────► sense·key·accumulate, re-arm, escalation
        │
        ▼
list[FireEvent{goal_id, action, text}]    # what to speak / surface
```

The goal-authoring path can also be driven in natural language: `agent.on_speech_endpoint(utterance)`
routes a spoken turn to the **conductor** (the only tier that writes standing state), which authors
goals from plain language behind the same guarded firewall. See
[e2e-flow-milk-boilover.md](e2e-flow-milk-boilover.md) for a full trace.
