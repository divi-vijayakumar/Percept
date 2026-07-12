# Edge Wake Policy

Percept should spend the heavy perception call only when cheap evidence says a goal is plausible.
The shared edge/harness wake policy treats each detector output as a channel and fuses them into a
single cheap score:

```text
P(goal) = noisy_or(weight[channel] * signal[channel])
```

With no policy weights, behavior stays the legacy path: `max(motion, structural, semantic)`. A policy
lets several weak cues combine, such as `presence + audio_event + change`, without making one detector
authoritative.

Core channels:

| Channel | Meaning | Example detector/source |
| --- | --- | --- |
| `motion` / `change` | frame-diff magnitude or rate of change | smoke plume, door opening, hand motion |
| `presence` | person/animal/object present | person near car, bottle near user |
| `pose` | body/keypoint action shape | fall, posture, jumping-jack openness |
| `audio_event` | scream, help phrase, thud, glass, alarm, water leak | mic/VAD/STT/audio classifier |
| `semantic_relevance` | CLIP/image similarity to the goal | smoke-like, water-like, rambutan-like |
| `structural` | local geometry/layout salience | hand pointing, covered camera, screen layout |
| `semantic` | semantic adapter salience | goal-specific image relevance |

Use `gate_on="fixed"` for low-motion content where frame-diff will miss the event: reading, menus,
screenshares, camera-covered checks, and reminders. Use `gate_on="motion"` for physical changes, then
raise recall with the weighted policy and detectors.

## Salience modes (how the `motion` channel is measured)

The `motion` channel has two opt-in modes that fix the whole-frame mean's blind spots. Both default
off (the deployment-wide whole-frame mean is byte-for-byte unchanged) and are picked per **camera
mount**, not per goal:

| Field | Values | What it does | Mount it fits |
| --- | --- | --- | --- |
| `salience_stat` | `mean` (default) · `block_max` · `p95` | summarizes the frame-diff over an 8×6 block grid instead of averaging the whole frame, so a *localized* event (a coffee sip, a shelf grab, a terminal error on a 4K screen) clears threshold even when its whole-frame mean is ~0 | static camera — desk, security, screenshare |
| `ego_compensate` | `false` (default) · `true` | removes whole-frame translation (camera pan / head turn) via a coarse integer-pixel shift search and gates on the *residual*, so a moving camera doesn't escalate every frame | wearable / drone / robot / handheld |

Selection is **deployment default ⊕ per-goal override**, mirroring `threshold`/`region_hint`:

- Deployment default: `settings.edge.salience_stat` / `settings.edge.ego_compensate` (env
  `PERCEPT_SALIENCE_STAT` / `PERCEPT_EGO_COMPENSATE`). A wearable/drone build sets `ego_compensate`
  once; a static security cam sets `salience_stat="block_max"` once.
- Per-goal override: `goal.state["salience_stat"]` / `goal.state["ego_compensate"]`, so one static
  camera can run a `block_max` shelf-theft watch beside a whole-frame crowd-density watch.
  `compile_watchspec` resolves the effective value onto each `WatchCondition`, so the value rides the
  wire and the edge honors it without reading any server config. A malformed override falls back to the
  deployment default (the same fail-safe firewall as `region_hint`/`schedule`).

The `block_max`/ego statistic is used only for the **escalate decision**; the reported
`Tier0Signal.motion` gauge stays the full-frame mean so it is comparable across configs. `ego_compensate`
currently uses the whole-frame residual (block/ROI compose is deferred), and an IMU fast-path (feeding
device angular velocity instead of the vision search) is not yet wired.

### Per-stat thresholds (they live on different value scales)

`block_max`/`p95` take the max / 95th-percentile block, so their values sit **above** the whole-frame
mean — the mean-tuned `0.10` cutoff over-fires them. Each stat therefore has its own base cutoff, and
`compile_watchspec` picks the right one for a goal's resolved `salience_stat`:

| stat | setting | default | calibrated recall (PT action, N=80) |
| --- | --- | --- | --- |
| `mean` | `PERCEPT_MOTION_GATE` | 0.10 | 0.06 (the whole-frame mean is blind to localized events) |
| `block_max` | `PERCEPT_MOTION_GATE_BLOCK_MAX` | 0.20 | 0.85 |
| `p95` | `PERCEPT_MOTION_GATE_P95` | 0.15 | 0.79 |

The defaults come from a sweep on Perception Test action events (localized hand/object events on a
static camera): `block_max@0.20` recovers **0.85** recall where the whole-frame mean gets **0.06**, at
roughly half the calls of the naive `0.10`. Raising the cutoff trades recall for cost
(`block_max`: 0.94 @ 0.15 · 0.85 @ 0.20 · 0.68 @ 0.25) — tune per deployment.

### Audio fusion (motion-invariant recall)

Audio is motion-invariant, so it is the most reliable wake when the visual channel is weak. Setting a
`wake_policy` that weights `audio_event` makes a loud event escalate even at zero motion (the gate fuses
`energy` into the cheap score, and `Tier0Signal.energy` reports it upward). On Perception Test **sound**
events whose source is *not on screen* (`is_visible=0`), motion-only recall is **0.28**; adding the audio
channel lifts it to **0.70** — a 2.5× recovery on events the visual gate is structurally blind to.

## Pressure Test

| Use case | Cheap wake before VLM | Wake trigger | VLM confirms | Ask/refuse behavior |
| --- | --- | --- | --- | --- |
| Read a book with me | fixed cadence + OCR/text change + semantic relevance | stable readable page or new passage | notes, topics, summary | ask on unreadable text or uncertain section |
| Smoke or sparks | change + semantic relevance + optional audio event | plume, flicker, glow, alarm sound | smoke/sparks visible and severity | ask on haze, steam, glare |
| Flooding | fixed cadence + change + water/object presence | water-like region appears/grows | flooding/leak and visible source | ask on reflections or shadows |
| Breaking into car | person presence + change + audio event | person near car plus impact/open-door/glass | forced-entry indicators | ask unless clear; avoid identity claims |
| Speeding car ticketing | vehicle presence + tracking/change | vehicle crosses zone quickly | "vehicle passed quickly" only | refuse ticketing/enforcement claims |
| Elder fall | motion + pose + presence + audio event | rapid descent, horizontal/still person, thud/help | person appears fallen/on floor | ask on occlusion or slow slump; not sole safety system |
| Somebody asking for help | audio event/STT + presence | distress phrase or scream | visible person/context | ask on ambiguous speech |
| Camera covered | fixed cadence + dark/blur/low-texture change | persistent occlusion-like frame | lens covered vs lights off | ask on natural darkness |
| Correct my posture | pose + fixed cadence + presence | poor posture sustained | posture relative to goal | ask when body is partly visible |
| Count jumping jacks | pose-openness + motion-periodicity | rep boundary peaks | activity type periodically | count boundaries, not per-frame VLM |
| Translate my menu | fixed cadence + OCR/text | readable menu region | translation/layout context | ask on glare, low OCR confidence |
| Tell me about pointed menu item | hand/pose + OCR + semantic relevance | pointer near an item | referenced item, e.g. mapo tofu | ask if multiple nearby items |
| How do I peel this fruit | object presence + semantic relevance + hand motion | rambutan-like object in hand | fruit/action state | ask if fruit is uncertain |
| Drink-water reminder | schedule/fixed cadence + presence/object/action | reminder time plus user context | whether drinking happened if needed | use rising edge; avoid repeated nagging |

This keeps the product thesis intact: cheap local channels maximize recall and cost control; the VLM
does semantic confirmation and explanation; the cognition gate still fires only on the rising edge and
asks when the evidence is not trustworthy.
