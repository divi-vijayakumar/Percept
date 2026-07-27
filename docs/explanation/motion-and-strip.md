# Motion score and the temporal strip

The tier-0 wake layer answers one cheap question per frame — *is anything worth a closer look?* — so the
brain spends a paid model call only when something plausibly changed. Three decisions are involved, and
keeping them distinct is the whole game: **how** motion is measured, **whether** the camera itself is
moving, and **how much history** the model sees once it does look. This page explains each conceptually;
for the exact env vars and thresholds see [Configuration](../configuration.md) and
[Edge policy](../reference/edge-policy.md).

## The motion score

The tier-0 gate diffs each frame against the previous one and summarizes the difference into a single
`0..1` motion score. It is pure, deterministic, and model-free. The score is compared against a cutoff:
below it the goal coasts and spends nothing; above it the frame escalates toward perception.

*How* the frame-diff is summarized is the key choice — the `salience_stat` field, resolved over an 8×6
block grid:

| statistic | what it summarizes | catches |
|---|---|---|
| `mean` | the average difference across the whole frame | large, whole-frame change — someone walks in, the scene shifts |
| `block_max` (default) | the single most-changed block | a *localized* action — a coffee sip, a shelf grab, one error line on a large screen |
| `p95` | the 95th-percentile block | localized change, but more robust to a lone noisy block than `block_max` |

The whole-frame **mean averages a small action away**: a hand moving in one corner barely moves the
average of a mostly-still frame, so the mean-based gate skips it. A **localized** statistic keeps the
signal where it is strong instead of diluting it across the frame, so a small or low-motion event can
still clear the bar. That is why `salience_stat` defaults to `"block_max"` (env
`PERCEPT_SALIENCE_STAT`): measured over a 100-case corpus, the whole-frame `mean` at its `0.10` cutoff
admitted **zero frames on 24 cases**, while `block_max@0.20` did so on 2. Choose `p95` instead when a
lone noisy block is a concern, or `mean` only when the action genuinely fills the frame.

## Thresholds are per-statistic — not interchangeable

`block_max` and `p95` take the *max* / *95th-percentile* block, so their values sit on a **higher scale**
than the whole-frame mean. The mean-tuned cutoff is far too low for them — it would over-fire constantly.
Each statistic therefore has its own base cutoff, and the compiler picks the right one for a goal's
resolved `salience_stat`:

| statistic | env var | default cutoff |
|---|---|---|
| `mean` | `PERCEPT_MOTION_GATE` | `0.10` |
| `block_max` | `PERCEPT_MOTION_GATE_BLOCK_MAX` | `0.20` |
| `p95` | `PERCEPT_MOTION_GATE_P95` | `0.15` |

The rule: **never carry a threshold across statistics.** A `0.10` cutoff that is right for the mean is
wrong for `block_max`, which lives on a higher scale. The localized statistic is used only for the
*escalate decision*; the reported `Tier0Signal.motion` gauge stays the whole-frame mean so it remains
comparable across configurations.

## Ego-motion compensation for a moving camera

On a wearable, drone, robot, or handheld camera, the *camera* moves, so a raw frame-diff lights up on
every frame — the whole scene shifts even when nothing in it happened. `ego_compensate` (default
`False`, env `PERCEPT_EGO_COMPENSATE`) removes the global camera/head translation via a coarse
integer-pixel shift search and gates on the **residual** motion — the part that is not explained by the
camera moving. A pan or a head turn no longer escalates; a hand entering the frame still does.

Set `ego_compensate` once for a moving-camera build. Leave it off for a fixed mount, where there is no
ego-motion to subtract.

## Selection: deployment default, per-goal override

Both `salience_stat` and `ego_compensate` follow the same pattern as other tier-0 knobs: the **deployment
declares a default** (via the settings above) and a **goal may override** it through
`goal.state["salience_stat"]` / `goal.state["ego_compensate"]`. The compiler resolves the effective value
onto each watch condition so it rides the wire and the edge honors it without reading server config; a
malformed override falls back to the deployment default. One static camera can thus run a `block_max`
shelf watch beside a whole-frame `mean` crowd-density watch.

## The temporal strip

Once the gate escalates, the model need not see only the single triggering frame. Every frame is cheaply
buffered into a rolling `FrameWindow`, so a goal can ask the model to judge a **strip** — a short,
ordered sequence of recent frames — instead of one snapshot. The strip is what lets the model see a
*transition* rather than a still: the descent of a fall, not just "on the floor"; the moment a value
crosses a line, not just the after-state.

The two decisions stay separate: the **motion score decides whether to look**, and the **strip decides
how much history the model sees when it does.** You select it per goal via the perception spec
(`temporal="strip"`, `n_frames=…`); the default is a single frame.

## When there is no pixel motion: fixed cadence

Some content changes *meaning* without changing *pixels* enough to move the motion score — a dashboard, a
document, a screenshare that scrolls or updates a number. The motion gate is structurally blind to these:
frame-to-frame difference stays near zero, so it never escalates, and the strip is never even assembled.

The fix is not a better motion statistic — it is to stop gating on motion at all. For this content a goal
uses a **fixed cadence**: `gate_on="fixed"` makes the gate escalate on a timer every
`min_judge_interval` seconds regardless of motion, giving the model a steady look at content that a
frame-diff cannot see. This is the pattern behind
[watching a screenshare](../how-to/screenshare.md) — use `gate_on="fixed"` for screens, documents, and
reminders, and the motion gate for physical events.

## See also

- [../how-to/screenshare.md](../how-to/screenshare.md) — the fixed-cadence screen-watching template.
- [../how-to/authoring-goals-and-watch-plans.md](../how-to/authoring-goals-and-watch-plans.md) — where
  `salience_stat`, `ego_compensate`, and `gate_on` are set on a goal.
- [../configuration.md](../configuration.md) · [../reference/edge-policy.md](../reference/edge-policy.md)
  — the exact env vars, thresholds, and wake-policy channels.
- [Architecture](architecture.md) · [End to end: one judged request](e2e-flow-milk-boilover.md)
