---
title: The consequence gate — the stable decision surface
updated: 2026-07-23
applies_to: percept-vision 1.0.0
status: stable
---
# The consequence gate — the stable decision surface

`percept.gate.Gate` is the **one firing path**. Every proactive output the SDK produces comes out of
it, and nothing else in the codebase may decide to notify a human. This page is the normative
reference for that decision model: the actions, the reason codes, the rising-edge and refractory
rules, and what an integrator is and is not allowed to depend on.

Everything on this page is **stable surface** (card O2). Reason codes are never renamed or
repurposed; a new decision path adds a new code.

## The three actions

```python
from percept import FIRE, ASK, SILENT, GATE_ACTIONS, DELIVERED_ACTIONS
```

| action   | meaning | delivered? |
|----------|---------|------------|
| `"fire"` | Confident yes, fresh, on a **rising edge** — the proactive notification. | yes |
| `"ask"`  | Low confidence **or** stale support — the refusal to guess. Never a confident-wrong nag. | yes |
| `"silent"` | Nothing to say: not satisfied, already fired this episode, or suppressed. | **no** |

`DELIVERED_ACTIONS == (FIRE, ASK)`. **`SILENT` is internal-only.** It is a value of the decision
model, not an event you will ever receive: the firing path drops a silent decision rather than
surfacing it (`BehaviorExecutor.fire_judged` returns `None`), so no `watch()` event and no
`on_incident` callback ever carries `action == "silent"`. Code that switches on `event.action` needs
exactly two branches.

## The rising edge

A **key** is `(goal_id, entity_or_episode_id)`. Each key carries one bit — armed or spent:

- a confident **yes** on an armed key fires, and spends the edge;
- a confident **no** re-arms it;
- a confident **yes** on a spent edge is `SILENT` (`already_fired`).

So a brand-new entity fires even though the previous one already did (a *new* cup is a new key),
while the same entity never double-fires until it genuinely goes away and comes back.

Two opt-in guards sit on top, both **off by default**:

- **`require_false_first`** — a key whose *first* observation is already confidently true does not
  fire; only a witnessed false→true edge does. ("Already present when the runtime started" is not
  something you asked to be told about.)
- **`refractory_s`** — a fire landing within N seconds of the *same key's* last real fire is
  withheld, even if a spurious confident-false blip re-armed the edge in between. This is the fix for
  mid-episode judge jitter, where a meaningful share of false fires are re-fires of one true episode
  after a momentary flip. A genuine second episode still fires once the window has elapsed.

Both are `GateSettings` fields (`PERCEPT_GATE_REQUIRE_FALSE_FIRST`, `PERCEPT_GATE_REFRACTORY_S`) and
both are `WatchProfile.gate` knobs, so a profile can turn them on per deployment context without any
code change.

An orthogonal, per-goal knob: **`hold_s` (dwell)** requires the condition to stay continuously true
for N seconds before firing, so a transient blip never notifies.

## Reason codes

`Gate.decide(...)` returns a typed `GateOutcome` — the action **plus** the reason. `Gate.evaluate(...)`
is unchanged and still returns the bare action; `evaluate` is exactly `decide(...).action`.

```python
from percept import GATE_REASONS, REASON_ACTION, SUPPRESSION_REASONS

outcome = gate.decide(("goal-1", "cup-3"), verdict, now=t)
outcome.action     # "fire" | "ask" | "silent"
outcome.reason     # one of GATE_REASONS
outcome.suppressed # True iff a fire was withheld by a precision guard
outcome.delivered  # True iff action is FIRE or ASK
```

| reason | action | meaning |
|--------|--------|---------|
| `rising_edge` | `fire` | Confident yes on an armed edge — the canonical fire. |
| `dwell_satisfied` | `fire` | ...and it held continuously for the goal's `hold_s` first. |
| `low_confidence` | `ask` | `verdict.confidence` is below the gate threshold. |
| `stale` | `ask` | The supporting perception/fact is older than `fresh_min` allows. |
| `not_satisfied` | `silent` | Confident false — re-arms the edge. |
| `already_fired` | `silent` | Confident true, but this episode's edge is already spent. |
| `start_anchored` | `silent` | **Suppression.** Already true on the first observation (`require_false_first`). |
| `refractory` | `silent` | **Suppression.** A fire was due but lands inside this key's refractory window. |
| `dwell_pending` | `silent` | True, but has not held for `hold_s` yet — a deferral, not a suppression. |

Each reason implies **exactly one** action (`REASON_ACTION`), so keying off either can never
disagree. A `GateOutcome` whose action contradicts its reason cannot be constructed.

## Suppression is observable

`SUPPRESSION_REASONS == ("start_anchored", "refractory")` — the cases where a fire the *baseline*
rising-edge gate would have delivered was deliberately withheld by an opt-in precision guard. That is
the one class of decision an operator must be able to see: without it, turning on a guard makes the
runtime quieter with no signal saying why.

Every suppression emits a structured `gate.suppressed` log line, and — when running under
`percept.runtime.worker.run` — a health event:

```python
async for incident in run(source, core, on_health=my_sink):
    ...

# my_sink receives:
RuntimeHealthEvent(
    kind="gate_suppressed",
    ts=<draining tick>,
    data={"key": "('goal-1', 'cup-3')", "reason": "refractory", "at": 12.5},
)
```

`already_fired`, `dwell_pending` and `not_satisfied` are deliberately **not** suppressions. They are
the base contract — every tick of any continuing true state produces `already_fired` — and reporting
them would be per-frame noise, not signal.

## What does not change

Both precision guards are off by default, so a default-configured deployment:

- suppresses nothing,
- emits no `gate_suppressed` health event,
- and produces exactly the actions it produced before this surface existed.

`Gate.evaluate`'s signature and semantics are unchanged, as is `watch()`'s (`WATCH_VERSION` is
frozen). `decide()`, the reason vocabulary and the suppression hook are purely additive.

## Related

- [`watch-contract.md`](watch-contract.md) — the frozen public entrypoint the gate feeds.
- [`edge-policy.md`](../admission/wake-policy.md) — the *tier-0* gate (escalate-or-not), which runs strictly
  before this one. Note that `percept.harness.tier0.GateDecision` and
  `percept.sequencing.GateDecision` are different types from `percept.GateOutcome`; only the latter
  is the consequence gate's own outcome, and only it is exported from the top-level namespace.
