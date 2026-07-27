# One-hour result — 12:44 → 13:44

## The hypothesis was WRONG

> Predicted: *"fewer than half of the 8 declared routes will produce an incident end-to-end on a real
> payload today."*

**Measured: 7 of 8 routes fire on a real payload. The 8th (`UNSUPPORTED`) correctly refuses to arm.**

Driven with the genuine published payload — `(t, Frame(48×36 gray), ("image/jpeg", bytes))`, built the
way `runtime/source.py:_capture` builds it — through a judge and a pose provider that **raise** on
anything else. Mutation-tested both ways: swapping a token string in makes `FRAME_WATCH` fail; restoring
the pre-fix `PoseChannels.measure` makes rhythm die on raw JPEG bytes. The assertions are not vacuous.

| route | fires? | how |
|---|---|---|
| FRAME_WATCH | ✅ `Incident` | straight out of `percept.run` |
| FRAME_TRANSITION | ✅ `Incident` | straight out of `percept.run` |
| FRAME_PROCESS | ✅ `Incident` | needs quiet > `episode_inactivity_s` (**defaults to 90 s**) |
| EXTERNAL_RHYTHM | ✅ `Incident` | fixed today (`ba68bf6`); 72 JPEGs → 5 reps → milestone |
| EXTERNAL_EVENT_WINDOW | ✅ `FireEvent` | host calls `push_frame` + `perceive_events` |
| EXTERNAL_BOUNDARY | ✅ `FireEvent` | host calls `perceive_boundary` |
| EXTERNAL_SCHEDULER | ✅ `FireEvent` | host calls `schedule_*` + `tick`; only when the conductor answers `proactive_speak` |
| UNSUPPORTED | ⛔ raises | by design, batch-atomic |

So the wiring is in far better shape than I claimed. **I over-read four bad findings into a systemic
verdict.** Recorded as refuted.

## But the underlying worry was right in a different place

The failures are not *inside* routes. They are in **composition** and in **claims**.

### 🔴 New defect — a count goal's authored `fire_when` is silently dropped

`behavior.py:1650` builds count as `Accumulator(kind="count_distinct")` with **no** `fire_when`; the
rhythm branch four lines below passes it. So `"fire_when": "n == 10"` survives arming, never reaches
the program, and **"tell me when I hit 10" fires twelve times, starting at rep 1.**

This makes **the remedy text we shipped this morning a lie** — `7e17378` tells a count goal's author to
drive it with `perceive_boundary`, and `perceive_boundary`'s own docstring promises `fire_when`
milestone semantics that do not exist. Reachable straight from the conductor, which authors
`fire_when` into state for count goals.

### 🔴 New defect — one host-driven goal aborts the whole frame run

`require_worker_routes` (`worker.py:3796`, `:3961`) checks **all** active goals on `gate_source`. Arming
a `count` goal beside a `watch` goal makes `percept.run` raise before frame 1 — **and the perfectly
good watch goal dies with it.** Binding the host-driven goal to a different `source` key is the working
topology, and it is documented nowhere.

A client following our own advice gets a dead run with no obvious cause.

### Also
- `apply_watch_plan` **cannot arm the three host-driven routes at all** — `add_goal`/`apply_goal_ops` is
  their only path.
- `FRAME_PROCESS` emits nothing on any stream shorter than `episode_inactivity_s` (default 90 s).

## Reconciling two agents that appear to contradict

The evidence audit found `transition` producing **0 incidents from 19 decisive verdicts** on a real
corpus case. The route matrix found `FRAME_TRANSITION` **firing** on a synthetic stream. Both are right,
and the difference is the finding: **the transition route is wired correctly and does not fire on real
content.** That is a judgment-quality problem, not a wiring problem — and it would have been misfiled as
"broken route" without both measurements.

## The real lesson, restated

Every defect found today survived because **a test substituted a permissive fake exactly where the real
contract lived**:

- rhythm drove the real eye with `measure(None)` — hid a payload-contract break
- the conductor was scored against `goal_kind` labels that are **per-dataset constants** — a classifier
  knowing only the dataset scores 100/100
- the count preset was ablated on a route that **builds no `DetectorBank`**
- episodes were measured on a source class where they **correctly cannot open**

The fix is not more unit tests. It is that **every contract boundary needs one test whose double is
hostile** — raises unless handed exactly what the contract promises. `test_behavior_route_matrix.py`
(16 tests, 0.4 s, no provider, no model) is that pattern, and it is the durable artifact of this hour.

## Launch position

**Safe to claim:** input tokens −69% at `media_resolution=LOW` (reproduced ×3). 7 of 8 routes deliver
on a real payload.

**Must soften:** every recall/precision number — all are single cross-run comparisons against a measured
0.33 noise band, and four of six wins were never in the binary that produced their headline.

**Must drop:** the conductor as a default (it routes ~56% of goals to primitives that deliver nothing
today, and turns a working 0.75-recall case into a crash); "merge deletes no real occurrences"; the
`block_max` recall figure.
