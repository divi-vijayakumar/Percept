# Core concepts: the cognition/perception split, the gate, rising-edge firing, memory

Percept splits cleanly into two layers. Understanding the split is the key to using it well.

```
┌──────────────── COGNITION — the "brain" (System 1) ──────────────┐
│  the three-state gate · rising-edge firing · entity memory       │
│  routing · cache · the behavior/concern framework                │
│  reasons over Verdict streams · deterministic · cheap · stdlib   │
└──────────────────────────────────────────────────────────────────┘
                 ▲  feeds on Verdict{satisfied, confidence, evidence}
┌──────────────── PERCEPTION — the "eyes" (System 2) ──────────────┐
│  vision.judge(condition, frame) -> Verdict     ★ ONE seam        │
│  a stateless, per-call VLM (Claude / Gemini / a keyless fake)    │
└──────────────────────────────────────────────────────────────────┘
```

## The perception seam

Perception is a stateless, per-call vision-language model. Its entire contract is one method on the
`Vision` protocol: `vision.judge(condition, frame) -> Verdict{satisfied, confidence, evidence}`. It
answers *"is this plain-language condition true of this frame, right now?"* and nothing more.

**The cognition layer never sees pixels.** It reasons only over the `Verdict` stream. That is what makes
it cheap, deterministic, and backend-agnostic: swap Claude for Gemini for a keyless offline fake and the
brain is unchanged. Run the whole thing without keys via `Percept.create(mode="offline")`, which wires
deterministic fake backends that prove wiring, not model intelligence. See
[Backends](../reference/backends.md).

## The three-state consequence gate

The gate is the **trust surface**. It maps a `Verdict` (plus a freshness value) to exactly one of three
actions — it never guesses:

| the fact says… | action | why |
|---|---|---|
| confident **yes**, fresh, on a **rising** edge | `FIRE` | the condition just became true — act once |
| confident **yes**, already fired / not rising | `SILENT` | duplicate — already reported |
| confident **no** | `SILENT` (re-arms the edge) | not true — stay quiet |
| low-confidence **or** stale | `ASK` | too unsure to fire — ask instead of guess |

`FIRE` and `ASK` are the only actions a subscriber sees; `SILENT` is internal and produces no event. The
precise contract — the reason vocabulary, freshness, suppression guards — is in
[../reference/consequence-gate.md](../reference/consequence-gate.md).

## ASK — the trust state

The third state is the whole point. When a verdict's confidence is below the gate threshold (default
**0.7**) or a supporting fact has gone stale, the gate **refuses to guess**. Instead of delivering a
confidently-wrong nag, it asks the human to confirm. Ambiguity never becomes a confident fire — it
becomes a question.

This is why a raw classifier that only emits yes/no is not enough: it has no principled way to abstain.
Percept's gate treats "I am not sure" as a first-class outcome.

## Rising-edge firing

Percept fires on the **rising edge** — the moment a condition flips false → true — not on every frame
where it stays true.

- A condition that is true for a hundred frames fires **once**, not a hundred times.
- After a fire, the edge is held until the condition goes confidently **false** again, which **re-arms**
  it. Only then can it fire a second time.
- An optional **refractory** window suppresses a re-fire that lands within N seconds of the same key's
  last real fire, defeating mid-episode model jitter. It is off by default.

The gate key is `(goal_id, entity_or_episode_id)`. Because the *entity* is part of the key, a brand-new
entity gets a fresh edge: watching "the user is drinking coffee", a new cup fires even though the old cup
already fired — and the same cup never double-fires until it goes confidently false in between.

## Entity memory and the "cup problem"

The brain keeps a lightweight, session-scoped `EntityGraph` — the tracked *things* below the model.
Identity is established from geometry and handed up as a stable symbol; it is never re-derived from
pixels turn to turn. Two questions matter, and they are different:

- **Continuity** — *"is this the same track right now?"* Answered from geometry: same label, overlapping
  box frame to frame. This is the cheap default and is correct for a single dominant object under a
  near-static camera.
- **Appearance** — *"is this the same thing I saw before the gap?"* When tracking breaks — the object
  leaves the frame and returns — continuity alone mints a *new* id. An optional appearance signature lets
  the graph recognize the returning object as the same one.

This is the **cup problem**: a cup set down and picked back up is the same cup (appearance identity), but
a *second, different* cup is genuinely new (a fresh edge). Continuity answers the first half; appearance
answers the second. Getting it wrong means either double-firing on a re-entry or missing a genuinely new
subject. Entity memory is session-only — it is not persisted across process restarts.

## How the pieces connect

```
add_goal(Goal(condition, say))                # state a goal in plain language
        │
        ▼
perceive_judged(frame)                        # per frame, per active goal:
  ├─ vision.judge(condition, frame) ──► Verdict{satisfied, confidence}   (PERCEPTION)
  ├─ classify(verdict, freshness) ────► yes / no / unknown
  └─ gate.decide(key, verdict) ───────► FIRE / ASK / SILENT              (the gate)
        │
        ▼
list[FireEvent{goal_id, action, text}]        # what to surface or speak
```

You normally never call the gate directly — `agent.perceive_judged(...)` runs every active goal through
it and returns the resulting fire events. The plain-language goal you author, and the frozen `watch()`
seam that compiles it, are specified in [../reference/watch-contract.md](../reference/watch-contract.md);
for how to author goals and watch plans see
[../how-to/authoring-goals-and-watch-plans.md](../how-to/authoring-goals-and-watch-plans.md).

## See also

- [Architecture](architecture.md) — the four tiers and the two-layer brain.
- [End to end: one judged request](e2e-flow-milk-boilover.md) — a full trace of a single goal.
- [../reference/consequence-gate.md](../reference/consequence-gate.md) ·
  [../reference/watch-contract.md](../reference/watch-contract.md)
- CodeWiki deep dives: [Core & Agent](../../codewiki/Core-and-Agent.md) ·
  [Sensing & Signals](../../codewiki/Sensing-and-Signals.md)
