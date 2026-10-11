# Conditions

A `Condition` is a plain-language statement to watch for, and how to decide about it. It is a
frozen value, validated when it is built: an invalid field raises `ValueError` at construction.

```python
percept.Condition(
    "stove_unattended",                                  # id
    "the stove is on and nobody is in the kitchen",      # text
    zones=(((0.1, 0.2), (0.9, 0.2), (0.9, 0.95), (0.1, 0.95)),),
    trigger=percept.every(30),
    emit_cleared=True,
    max_open_s=900,
)
```

The defaults are starting points, not calibrated values. Use [coverage](concepts.md#coverage-and-blind-intervals)
and the [trace](delivery.md#trace-and-replay) to tune them on your own footage.

## What to watch

| Field | Default | Meaning and advice |
|---|---|---|
| `id` | required | A token your code uses: no surrounding spaces, no `/`. Unique within a monitor. |
| `text` | required | The condition, as the judge reads it. One observable fact per condition. Say what is visible ("a person is lying on the floor"), not an inference ("someone needs help"). |
| `kind` | `"state"` | `"state"` for something that stays true (held, rechecked, cleared on evidence); `"event"` for a moment (verified on its own frames, reported once, then rearmed). See [concepts.md](concepts.md#states-and-events). |
| `not_when` | `()` | Plain-language cases that are **not** the condition, sent to the judge as "answer no" cases: `not_when=("a person lying on the sofa",)`. The cheapest fix for a recurring false alarm. |
| `zones` | `()` | Polygons in normalised image coordinates (`(x, y)`, both in `[0, 1]`, `x` right, `y` down), at least 3 points each. The judge sees the frame cropped to the zones' bounding box (plus a 5% margin), and change triggers only look inside the zones. No zones: the whole frame. |
| `exclude_zones` | `()` | Polygons subtracted from what change triggers watch (a clock, a TV, a road). The judge's crop is a rectangle, so it may still show them. |
| `trigger` | `None` | When to look. `None`: `on_change()` for events; `on_change()` plus `every(300)` for states. See [Triggers](#triggers). |
| `judge` | `None` | The name of a member judge inside a `cascade` to ask for this condition only (a route). An unknown name raises when the condition is added. See [judges.md](judges.md#cascades-and-routes). |

## How sure to be

| Field | Default | Meaning and advice |
|---|---|---|
| `min_confidence` | `0.6` | Answers below it are **weak** and change nothing, for "yes" and "no" alike. Raise it to cut false alarms; lower it if a judge is consistently unsure about true cases. A weak "no" never clears a state. |
| `confirm` | `1` | Confident "yes" answers needed before the occurrence counts; the first one counts. Raise to 2 or 3 for a noisy condition, at the cost of latency (each extra yes is another look). |
| `confirm_window_s` | `20.0` | How close together, in **evidence time**, the confirming answers must be. |
| `verify` | `"auto"` | A second, stricter look before reporting. `"auto"` means `"original"` for events (the same frames) and `"off"` for states. States may use `"fresh"` (a new frame). Events can't use `"fresh"`, states can't use `"original"`. |
| `fast_path_confidence` | `None` | A confirming "yes" at or above this skips verification. Must be at least `min_confidence`. Use it for a judge you trust at high confidence, to save a request and latency. |
| `candidate_deadline_s` | `30.0` | How long verification may take after confirmation, and how long an unfinished confirm window waits for answers. |
| `on_unverified` | `"uncertain"` | When verification can't confirm a candidate (it said "unknown", or ran out of time): `"uncertain"` delivers an `uncertain` decision; `"health"` reports health `unverified` instead, for agents that should only hear confirmed events. |

## How long it lasts

| Field | Default | Applies to | Meaning and advice |
|---|---|---|---|
| `rearm_s` | `120.0` | events | After an event is reported, repeats are absorbed for this long (and do not extend it). Shorter for events that can genuinely repeat quickly; longer to suppress a burst. |
| `recheck_every_s` | `60.0` | states | While a state is held, look again this often on a fresh frame. Shorter means faster `cleared`, at more requests. |
| `clear_after_no` | `3` | states | Confident "no" answers in a row (a confident "yes" resets the count; weak, unknown and failed answers neither count nor reset) that clear a held state. Raise it if a state flickers off on a bad frame. See [How fast a state clears](#how-fast-a-state-clears). |
| `max_open_s` | `None` | states | If set, a state still held this long after `met` emits `expired` once, and stays held. Use it for "still open after 15 minutes" alerts. |
| `cooldown_s` | `0.0` | both | After a `met`, another `met` for the same condition and subject is suppressed for this long, across occurrences. A suppressed state `met` is delivered later if the state is still held ([deferred](decision-semantics.md#13-emission-rules)); a suppressed event `met` is dropped. |
| `emit_cleared` | `False` | states | Deliver `cleared` when a held state ends. Off by default: many agents only act on `met`. |
| `version` | `1` | both | Raised automatically when `Monitor.add()` replaces a condition with the same id. Changing a condition closes its open occurrences (a held state emits `cleared`, cause `condition_changed`). |

### How fast a state clears

A held state clears on its `clear_after_no`-th confident "no", so the delay after the real end is
set by how often it is looked at: a look when the end changes the zone (`on_change`), rechecks
every `recheck_every_s` (60 s), and heartbeats. With the defaults (3 answers, 60 s rechecks), a
door closed in a quiet scene is `cleared` 1 to 2 minutes after it closed: one look on the change,
then two rechecks. A failed or unsure look adds a recheck period; looks that wait for the request
budget add their wait. To clear faster, lower `recheck_every_s` (more requests), use a faster
heartbeat trigger such as `every(10)` (a `trigger` replaces both defaults), or lower `clear_after_no` (more risk of clearing on one bad
frame). `cleared` is delivered only with `emit_cleared=True`.

## Triggers

A trigger decides when a condition is worth a look. Triggers are values: two equal triggers share
one computation per frame.

- `percept.on_change(threshold=0.20, exit_threshold=None, baseline_s=5.0)` fires when a block of
  the zone differs from a moving average of the scene. The frame is reduced to 48×36 grey pixels
  and cut into 8×6 blocks; `threshold` is a block's mean absolute grey-level change (0 to 1),
  scaled by the zone's contrast. `exit_threshold` (default: half of `threshold`) is where a block
  rearms; `baseline_s` is how fast lasting changes and lighting drift are absorbed. A lasting
  change fires once. **A change that ramps in over roughly 10 s or more may never fire**: pair
  slow conditions with a heartbeat.
- `percept.every(seconds)` is a heartbeat: one look per period on the stream clock, and one at
  the start. A gap never causes a burst of catch-up looks.
- `Monitor.poke(stream_id, condition_ids, subject=None, observed_at=None, reason=..., evidence=None)`
  asks for a look now, from your own code. `observed_at` (wall time) anchors an event's clip (up
  to 4 frames from the 4 s before it); `evidence` (one or more `uint8` images) is judged instead of
  the stream's frames.
- A custom trigger is any object with
  `step(frame, conditions, state, now) -> (due, state)`, where `due` maps the ids of conditions to
  look at to a reason. It must be pure; `examples/frigate_semantic.py` uses one that never fires,
  so only pokes cause looks.

Looks are limited by the stream's request budget: a trigger that fires on every frame (a busy or
moving scene) queues looks that are then dropped as `dropped_budget`. Coverage shows it.

## Changing conditions at runtime

`Monitor.add(condition)`, `remove(condition_id)`, `pause(condition_id)`, `resume(condition_id)`
and `snooze(condition_id, until_wall, subject=None)` take effect at once on every stream. See
[decision-semantics.md](decision-semantics.md#7-condition-changes).
