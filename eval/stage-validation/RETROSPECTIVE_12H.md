# 12-hour retrospective — observation, hypothesis, plan

## OBSERVATION

Two things happened in parallel over the last 12 hours, and only one of them was the plan.

**The planned work:** 8 commits landed on the candidate, the suite grew to 4,907 passing tests, and six
"wins" were claimed — reflex frame bound, `block_max` salience, 0.7 s incident merge,
`media_resolution=LOW`, the Haiku conductor, and the rhythm route.

**The unplanned work:** almost everything got weaker under scrutiny.

| claim | fate |
|---|---|
| resolution buys latency | **refuted** — Gemini bills a flat ~1,100 tokens/frame regardless of pixels |
| `media_resolution=LOW` lifts recall +0.115 | **retracted** — +0.017 at n=40, inside the noise band |
| "episodes never open" (a release blocker) | **retracted** — correct behaviour on a camera source |
| baseline precision 0.291 | **corrected** to 0.340 |
| `pose-channels` lacks `measure` | **my own probe bug** — `_REGISTRY[k]` is a tuple |
| count needs a detector-selection policy | **null** — nothing beat answering zero, and the key was unreachable anyway |
| rhythm route is drivable | **broken end-to-end** — never decoded a real frame |

Five retractions. Three were mine, from repeating an agent's claim without checking it.

**The pattern underneath is sharper than any individual error.** Look at what the failures have in
common:

- The **rhythm route** had passing tests and could not decode a frame.
- **Count goals** armed successfully and never fired, silently, forever.
- `_TYPE_DETECTOR_PRESET["count"]` would have been **inert even with a winner** — no `DetectorBank` is
  ever built for that route.
- **Episodes** were measured on a source class whose identity strategy makes them correctly impossible.
- The consolidated arm B would have scored **zero everywhere** on an invalid enum value I supplied.

Every one is a failure at a **boundary between components that each have passing unit tests**. Not one
was a logic error inside a function.

And note which activities actually found bugs. The 4,907-test suite found none of them. They were
found by: an 82-minute run on real media, a 40-case run with replicates, an ablation on real clips, and
a consolidated run on two real builds. **Integration reality-checks found every real defect; the unit
suite found none.**

## HYPOTHESIS

> **The green suite is not evidence of function.** It validates component shapes, not wiring. Concretely:
> of the 8 declared `BehaviorRoute` values, **fewer than half will produce an incident end-to-end on a
> real source today** — and the failures will sit at component boundaries (payload shape, missing
> driver, unreachable config key), not inside components.

Falsifiable: if all 8 routes deliver an incident on a real payload, the hypothesis is wrong and the
suite is doing its job.

Corollary worth testing separately: **of the six claimed wins, only one (the reflex frame bound, Δ0.49)
has evidence exceeding the measured 0.33 recall noise band.** The rest are either deterministic
(cost/token counts — solid) or unproven.

## PLAN — 1 hour, 12:44 → 13:44

Hard constraint: **a paid consolidated run is live and latency-sensitive.** Everything below is
CPU-light and provider-free. No wheel rebuilds, no heavy extraction, nothing touching `venvs/`.

| when | what |
|---|---|
| 12:45 | Dispatch A — **Route Reality Matrix**: all 8 routes, real payloads, `FakeVision`, does each arm / receive / fire? |
| 12:45 | Dispatch B — **Evidence audit**: per claimed win, what is actually proven vs assumed |
| 13:25 | Synthesise both into one page |
| 13:40 | Report: what demonstrably runs, what is merely tested, what to do before launch |

**Deliverable:** a one-page matrix stating, per route and per claimed win, *what demonstrably runs* and
*what is only tested* — so the launch claim set is bounded by evidence rather than by commit count.

## Why this is the right hour

The consolidated run answers "did the stack get better." It cannot answer "does the thing work at
all," and today produced four separate instances of the second question having an uncomfortable
answer. Shipping a 1.0.0 whose counting path cannot decode a frame would be worse than shipping late.
