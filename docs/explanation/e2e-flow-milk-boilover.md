---
title: End to end: one judged request (milk boiling over)
updated: 2026-07-23
applies_to: percept-vision 1.0.0
status: stable
---
# End to end: one judged request (milk boiling over)

This page traces a single goal — **"the milk is boiling over"** — through the whole pipeline, for a fixed
kitchen camera pointed at a pot on the stove. It names the real component at each step so you can follow
the same path in code. If you have not read [Core concepts](concepts.md), start there.

The scenario is deliberately simple: one static camera, one standing goal, one thing that eventually
happens. The point is to see how the cheaper tiers absorb the quiet frames so that a paid model call
happens exactly once, on the real event.

## Step 0 — authoring the goal

The user says *"tell me when the milk is boiling over."* The spoken turn hits the local fast-path first:
the [`RouteEngine`](../../packages/percept-vision/percept/routing) recognizes this as a **goal-authoring**
utterance ("tell me when …" is a defer marker) and **defers** — it never tries to answer it locally. The
turn reaches the conductor, the only tier allowed to write standing state, which authors a watch goal:

```
condition = "the milk is boiling over"
say       = "Heads up — the milk is boiling over!"
```

A standing goal now lives in the brain. Nothing fires yet. From here on, every frame the fixed camera
streams runs the per-decision lifecycle.

## Step 1 — a frame arrives

The runtime front door ([`percept.run` / `RuntimeSession`](../../codewiki/Runtime.md)) opens the camera
source and drives the cognition core frame by frame. Each captured frame is stamped with the judgment
clock and pushed onto a rolling `FrameWindow` inside
[`Percept.perceive_judged`](../../packages/percept-vision/percept/agent.py). Cheap world-state ingest —
`vision.detect` plus `EntityGraph.ingest` — always proceeds; it is goal-independent and never costs a
model call.

## Step 2 — the tier-0 motion gate

Before any expensive perception, the tier-0 gate
([`percept.harness`](../../packages/percept-vision/percept/harness), see
[Motion score and the temporal strip](motion-and-strip.md)) asks one cheap question: *did anything worth
a closer look change?*

- **Still pot, calm milk.** The frame-diff against the previous frame is near zero. The motion score sits
  below threshold, so the fovea sampler
  ([`sampling.py`](../../packages/percept-vision/percept/sampling.py)) coasts and the goal spends
  **nothing** — no cache lookup, no judge call. This is the common case, and it is free.
- **Milk foams and rises.** Now the pixels in the pot region change frame to frame. The motion score
  clears threshold and the sampler **escalates** toward its cost ceiling — this frame is worth a closer
  look.

## Step 3 — the cache

An escalated frame reaches the frame-hash cascade
([`cache/`](../../packages/percept-vision/percept/cache)). It hashes the goal's region of interest and
compares it against the last stored verdict for this `(goal, entity, condition)` slot.

Because the foam is actively rising, the ROI has visibly changed since the last stored hash, so the
lookup is a **miss** (`DEFER`) — the cache refuses to serve a stale "calm" verdict against a changed
scene. (Had the frame been near-identical to a recent one, the cache could have served the prior verdict
at a decayed confidence and skipped the model entirely.) A miss escalates to the judge.

## Step 4 — the judge

A judge is never taken on trust. Before this goal ever armed, the runtime ran the fail-closed chain from
[Judges](../../codewiki/Judges.md): it **authorized** the goal's `WatchPlan.judge` request against the
deployment-owned catalog, **bound** any static context (a fixed rubric verified byte-for-byte against a
pinned SHA-256), and constructed the adapter to dispatch. If any step had failed — an unknown ref, a hash
mismatch, an uninstalled factory — the goal would have refused to arm rather than silently fall back to a
generalist model.

Now the runtime dispatches **one judge call** through the bound vision adapter:

```
vision.judge("the milk is boiling over", frame) -> Verdict{satisfied, confidence, evidence}
```

The call is timed and its cost/usage recorded, all inside a fail-closed boundary: a timeout, a provider
error, or a malformed response degrades *this tick* to ASK or SILENT — never a crash, never a spurious
FIRE. On a clean return, the fresh verdict is written back into the cache for the next frame.

## Step 5 — classify and gate

The verdict flows through the same unmodified gate every fire passes through.
[`classify(verdict, freshness, threshold)`](../../packages/percept-vision/percept/gate.py) maps it to
`yes` / `no` / `unknown`, and [`Gate.decide(key, verdict)`](../../packages/percept-vision/percept/gate.py)
applies the edge logic:

- The pot was calm on earlier frames, so this is the **rising edge** — the condition just became true.
  The gate returns **FIRE** on key `(goal_id, entity_id)`.
- Had confidence sat below the **0.7** threshold, the gate would have returned **ASK** — surface a
  question instead of a confident guess.
- On the *next* boiling frame the same key is already fired, so the gate returns **SILENT**: one alert
  per episode, not one per frame. It re-arms only after a confident-false (the pot goes calm again).

## Step 6 — deliver

The FIRE becomes a `FireEvent`, which the runtime adapts into an `Incident` delivered to the configured
sink (idempotent, supersession-aware, dead-lettered). Alongside delivery:

- **Evidence** — the frames that grounded the verdict are retained behind an **opaque handle**, not
  inlined; dereference is fail-safe. See [Episodes & Evidence](../../codewiki/Episodes-and-Evidence.md).
- **Episode** — the episode ledger records that this occurrence was reported, so a continuation of the
  same boil-over is not re-reported as a new one.
- **Trace** — the funnel counters (observed → admitted → judged → fired) and trace stages are recorded
  for offline replay and the visualizer. See [Observability](../../codewiki/Observability-and-Diagnostics.md).

The user hears: *"Heads up — the milk is boiling over!"* — once.

## What the cheaper tiers absorbed

The single paid judge call in this trace is the exception, not the rule. Everything before it exists to
avoid it:

| tier | what it absorbed here | cost |
|---|---|---|
| **sampling / tier-0** | every still frame while the milk was calm | free |
| **routing** | the goal-authoring turn (deferred, never answered locally) | free |
| **cache** | a near-identical frame would have reused the last verdict | free on a hit |
| **gate** | every duplicate "still boiling" frame after the first (SILENT) | free |

A raw VLM-in-a-loop would have paid for a model call on *every* one of those frames and re-alerted on
each positive one. Here, one judge call fires one incident on the rising edge.

## See also

- [Architecture](architecture.md) · [Core concepts](concepts.md) ·
  [Motion score and the temporal strip](motion-and-strip.md)
- [../reference/consequence-gate.md](../reference/consequence-gate.md) ·
  [../how-to/authoring-goals-and-watch-plans.md](../how-to/authoring-goals-and-watch-plans.md)
