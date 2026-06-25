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
