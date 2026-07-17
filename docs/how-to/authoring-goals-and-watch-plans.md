# Authoring goals and watch plans

Two authoring decisions govern detection quality *before* the model ever runs: **how you phrase a goal**
and **how the watch plan looks at it**. A weak goal or a mismatched plan degrades recall and precision no
matter how strong the judge is — the failures then look like model errors but are not. This guide is the
best-practice checklist for both.

> These are quality levers observed directly in evaluation: most "the judge missed it" cases trace back to
> a competing goal in the same scene or a plan that showed the judge a single still of an inherently
> temporal action. Fix the goal and the plan first.

---

## Part 1 — Goal authoring

A goal is a `Goal(id, condition, say, state=...)`. The `condition` is the plain-language anchor the judge
confirms and the edge salience-matches. Author it so it is **falsifiable, specific, and distinct**.

### 1. One concern per goal — do not stack competing goals on one scene
When many goals are active on the same frames, a single-frame judge can lock onto whichever action is most
salient and answer *about the wrong goal* — it correctly reports "someone is holding a vacuum, not a
blanket" while judging the blanket goal. The action was happening; the judge just confirmed a **neighbor**.

- Keep the active goal set **small and mutually distinct** (a handful, not a dozen). This is the single
  highest-leverage authoring choice.
- If two goals share an object or actor ("holding a blanket" / "holding some clothes"), expect cross-talk.
  Merge them, or disambiguate (below).

### 2. Make the condition specific and falsifiable
Terse gerunds ("holding a blanket") force the judge to disambiguate against everything else in frame. Add
the discriminators that make the target unambiguous:

- **name the object and a distinguishing attribute** — "holding a *folded blue* blanket" > "holding a blanket".
- **anchor the actor or location** when several people are present — "the person *on the couch* is …".
- **state the observable, not the intent** — "a pot is boiling over" (visible) > "cooking is going wrong".

### 3. Decide the edge semantics for already-true states
If the watched state may **already be true when the watch starts** (someone is already holding the blanket),
choose the behavior deliberately:

- want a notification for the standing state → allow the first fire.
- want only *transitions* into the state → set `require_false_first` (env
  `PERCEPT_GATE_REQUIRE_FALSE_FIRST`, or `GateSettings.require_false_first`) so the gate waits for a
  false→true edge and does not fire on a state that was true from frame one.

See [Wake policy](../reference/edge-policy.md) for how the gate arms and fires.

### 4. Give the plan a prior when you know the temporal shape
The goal's `state` carries hints the plan reads (see Part 2): whether the event is **brief**
(`state['sense']['brief_window_s']`), a **region of interest** (`region_hint`), or needs a
**localized salience mode** (`salience_stat`). Authoring these on the goal is how a single condition string
still gets the right plan.

---

## Part 2 — Watch plan creation

The watch plan is *how to look*: which sense primitive, what cadence (fovea), which salience statistic. The
rule is simple — **match the plan to the action's temporal character.** A plan that shows the judge one
still frame is correct for a *state* and wrong for a *motion*.

### 1. Match the sense primitive to the action
Primitives live in `percept.sensorplan`. Perception primitives choose *what the judge sees*; sense
primitives choose *the temporal logic*.

| The action is… | Use | Why |
|---|---|---|
| a **persistent state** (holding, wearing, a pot on the stove) | single frame / `strip` | one frame already shows it |
| **inherently temporal** (laughing, running, taking X from somewhere) | `window` or `sprite` | a single still cannot show motion; fold several frames so the judge sees the movement |
| **ordered small detail** (a hand-off, a label read) | `strip` — N ordered frames | ordering and detail matter |
| a **brief, sub-second** event ("tell me when X *briefly* happens") | `judge` with Nyquist cadence | the window can open and close between two coarse frames — sample above its Nyquist so a quiet-but-in-window frame is still judged |
| a **change of state** A→B | `transition` | fire on the edge, not on either steady state |
| **counting distinct occurrences** | `episode` | de-duplicate one long event from N separate ones |
| **something expected that did *not* happen** | `absence` | fire on a deadline, not on an observation |

The most common plan error is judging an inherently temporal action on single frames. If a goal's verb is a
motion, give it `window`/`sprite`, not a still.

### 2. Choose the salience statistic for the camera and action
The edge gate decides *when to spend a judge call* from cheap motion. The statistic it uses
(`EdgeSettings.salience_stat`, env `PERCEPT_SALIENCE_STAT`, or per-goal `state['salience_stat']`) must
match the scene:

| Scene / action | `salience_stat` | Effect |
|---|---|---|
| whole-scene motion, action fills the frame | `mean` (default) | frame-averaged motion |
| **static camera, small/localized action** (a hand, a cup, standing still) | `block_max` | one active block in an 8×6 grid clears threshold even when the frame-mean is ~0 — catches localized motion `mean` washes out |
| localized but noisy | `p95` | noise-robust cousin of `block_max` |
| known region of interest | `region_hint` on the goal | score motion only inside the box |
| a **moving/handheld camera** | `ego_compensate` | subtract global camera shift before scoring |

Rule of thumb: **fixed mount + localized events → `block_max`.** Leaving it on `mean` is the classic reason
low-motion actions (holding, watching, standing) never escalate and are silently missed.

### 3. Tune the fovea cadence to the concern's urgency
The fovea (`percept.sampling.SensePolicy`) coasts at `base_fps` and saccades toward `max_fps` on motion:

- **`base_fps`** — the quiet floor. Raise it for fast events that must not be missed between saccades;
  keep it low for slow "watch for my dog" concerns to stay cheap.
- **`max_fps`** — the cost ceiling; processing is never denser than this.
- **`escalate_threshold`** — the motion level that triggers a saccade. **Lower it** for subtle in-window
  motion the default would suppress; raise it to ignore ambient jitter.
- **`hold_s`** — keeps the escalation alive for a beat after a spike so the saccade settles.

See [Motion and strip](../explanation/motion-and-strip.md) for the motion score and
[Wake policy](../reference/edge-policy.md) for how salience becomes a judge call.

---

## Checklist

Goal:
- [ ] one concern; the active set is small and mutually distinct (no cross-talk neighbors)
- [ ] condition is specific and falsifiable (object + attribute; actor/location when ambiguous)
- [ ] edge semantics chosen for already-true states (`require_false_first` if transitions-only)

Plan:
- [ ] sense primitive matches temporality (state→frame/`strip`; motion→`window`/`sprite`; brief→`judge`+Nyquist)
- [ ] `salience_stat` matches the camera/action (fixed mount + localized → `block_max`, not `mean`)
- [ ] fovea `base_fps`/`escalate_threshold` tuned to the concern's urgency
