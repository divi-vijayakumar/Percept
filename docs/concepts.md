# Concepts

Percept sits between a video stream and your agent. You give it plain-language conditions; it
tells your agent, once per real occurrence, when one becomes true. It doesn't replace your
cameras, your models or your agent.

## Trigger → Judge → Decide

Every stream runs the same three steps.

1. **Trigger: when is it worth looking?** Triggers are cheap, pure functions of the frames. They
   decide when a condition is worth asking a model about.
   - `on_change()` (the default) fires when something changes inside the condition's zone against
     a slow moving average of the scene.
   - `every(seconds)` is a heartbeat: one look per period, and one at the start.
   - A state with no trigger set uses both: `on_change()` and `every(300)`. An event uses
     `on_change()`.
   - `Monitor.poke(stream_id, condition_ids, ...)` is the same path from your own code: a
     detector, a tracker, a door sensor, another system's events.

   Before any trigger runs, a quality gate drops corrupt or blurry frames and a scene check
   notices when most of the picture changes at once (a bumped camera, an infrared day/night
   switch).
2. **Judge: what is true?** A judge (a vision-language model behind one interface) answers each
   condition with `yes`, `no` or `unknown`, a confidence and a short reason. It sees the frame (or,
   for events, a short clip), cropped to the condition's zones. Requests are limited by a hard
   per-stream budget and an optional cost cap. See [judges.md](judges.md).
3. **Decide: does the agent hear about it?** A pure reducer turns answers into decisions: it
   confirms, verifies, holds, clears, rearms and suppresses. A timeout or outage is never taken as
   an answer. See [decision-semantics.md](decision-semantics.md).

## Known limitation: small, distant or brief activity

The change trigger compares blocks of a heavily downscaled frame (48×36 grey pixels in 8×6
blocks) with a moving average. In a busy surveillance scene, an activity that is small, far from
the camera or over in a moment often changes too little of any block to fire it, so the judge is
never asked and the occurrence is missed. Nothing reports it as a "no"; coverage shows a
condition that was rarely or never looked at. For such conditions:

- give the condition a heartbeat, `trigger=percept.every(seconds)` (a state's default already
  includes `every(300)`; a shorter period looks more often, within the request budget);
- draw a tighter zone around where the activity happens, so it is a larger share of each block;
- lower the change threshold, `trigger=percept.on_change(threshold=...)` (more looks, more cost);
- or poke the condition from your own detector with `Monitor.poke`;

and check `Monitor.coverage()` to see how often the condition was actually looked at.

## States and events

A condition has a `kind`.

| | `state` (the default) | `event` |
|---|---|---|
| Example | "the garage door is open" | "a person falls to the floor" |
| Is true | for a while | at a moment |
| Checked again | on a fresh frame, every `recheck_every_s`, while held | once, on the **same frames** that triggered it, before it is reported |
| Ends | `clear_after_no` confident "no" answers, or the application resolves it | by itself: it rearms after `rearm_s` |
| Decisions | `met`, then optionally `expired` and `cleared` | `met`, or `uncertain` when the check can't confirm it |

A later frame shows only the aftermath of an event (a person lying on the floor, not falling), so
events are verified on their original evidence, never on a new frame.

## Occurrences and decisions

An **occurrence** is one real-world instance of a condition becoming true. Every decision about
it carries its `occurrence_id`. A **decision** is what your agent receives:

| `type` | Meaning |
|---|---|
| `met` | the condition became true (once per occurrence) |
| `uncertain` | something was seen, but the check could not confirm it (see `cause`) |
| `cleared` | a state stopped being true (only with `emit_cleared=True`) |
| `expired` | a state is still true after `max_open_s`; it stays held |

Each decision has a stable `decision_id`, a `cause` your code can branch on, a `reason` for
people, a `confidence`, `t_observed` (when the evidence shows it) and `t_decided`, its `cost`, and
a keyframe. The full contract is in [delivery.md](delivery.md#the-decision-contract).

A **subject** separates occurrences of one condition on one stream. It is `"scene"` unless you
poke with `subject=...` (for example a tracked person's id), in which case each subject has its
own occurrences.

## Coverage and blind intervals

Percept reports what it did not see as plainly as what it saw.

`Monitor.coverage(stream_id, since_wall=0.0)` returns, per condition, the **fate** of every
look, counted per minute:

- `judged`: a confident answer;
- `judged_weak`: below `min_confidence`, or "unknown";
- `failed`: a timeout, outage, refusal or malformed reply;
- `coalesced`: merged into a look already waiting;
- `dropped_budget`: waited past its deadline (3 s for an event detection, 10 s for a state), or the stream ended first;
- `dropped_for_verification`: a detection that waited past its deadline because the attempts it needed were kept for a waiting verification;
- `dropped_quality`: the frame was vetoed as corrupt or blurry;
- `stale`: the answer arrived for an earlier epoch, condition version or occurrence;
- `dropped_capacity`: refused because the stream's 64 decision slots were all in use.

It also counts `frames_capped` (frames skipped by the processing rate), `evidence_unavailable`
(looks that waited for a current frame) and `evidence_short` per condition (events judged on less
of their evidence window than asked, because the stream had just started or reconnected).
`triggers` splits the looks per condition by what made them due (a trigger's label such as
`on_change(0.2)` or `every(5)`, or `poke`, `settled`, `recheck`, `verify`), counting `due`,
`admitted`, `coalesced`, `dropped_budget` and `dropped_for_verification`.

**Blind intervals** are spans of wall-clock time when Percept could not see or could not judge:

| Kind | When |
|---|---|
| `link_down` | the camera or push feed is down |
| `degraded_video` | too many recent frames were corrupt or blurry |
| `scene_reset` | most of the picture changed at once; triggers pause while it settles, then every condition gets one look |
| `budget_exhausted` | looks were due but the request or cost budget refused them |
| `judge_down` | 3 judge requests in a row failed completely |
| `paused` | the application paused a condition |

The same events appear on the health channel (`Monitor.health()`); see
[delivery.md](delivery.md#health-stats-and-coverage).

## What this release assumes

- **Static cameras.** Fixed-viewpoint cameras only. A moving camera looks like a constant scene
  change. See [sources.md](sources.md#the-static-camera-assumption).
- **Replaceable models.** Every judge returns the same observation shape; the core imports no
  vendor SDK.
- **A hard request budget, a configurable cost policy.** See [judges.md](judges.md#budgets).
- **Two kinds of reliability.** Decision reliability (one event per occurrence, no lost decisions,
  an outage never reported as an answer) is Percept's job. Semantic accuracy is the model's;
  Percept reports it per condition but can't guarantee it.
