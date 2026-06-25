# The runtime worker — `run(source) → incidents`

The runtime owns the **camera → incidents** lifecycle: capture, the gate loop, judging off the display
thread, the live feed, and rising-edge **incident** emission — so your app stops writing that glue.

```python
from percept import Percept, Goal
from percept.runtime import FileDeadLetterQueue, RuntimeStats, run, webhook

agent = Percept.create(vision="gemini")            # pip install "percept-vision[gemini]" + GEMINI_API_KEY
agent.add_goal(Goal(id="zone", condition="a person is in the restricted zone", say="Alert: person in zone"))

notify = webhook("https://example.com/percept/incidents")
stats = RuntimeStats()
health = []

async for incident in run(
    agent,
    source="rtsp://door-cam/stream",
    on_incident=notify,
    on_health=health.append,
    snippet_kind="clip",
    stats=stats,
):
    ...                                            # run() owns capture + gate + judge + incident emission
```

`pip install "percept-vision[runtime]"` adds the camera/video/RTSP capture path (opencv) plus the
Python edge gate. A pre-decoded **iterable** source (`(t, gate_frame, judge_frame)` tuples) needs no
extra — handy for tests and offline replay.

## What `run()` does, per frame
1. push the frame into a rolling pre-roll buffer (for the incident clip);
2. run the cheap **tier-0 gate** (motion + the per-goal cadence) — most frames skip here, no cloud call;
3. on the rising edge, spend one **`vision.judge`** call (the brain — unchanged);
4. each `fire` → an **`Incident`** (with a pre-roll snippet) → `on_incident(...)` → yielded;
5. each `ask` → `on_ask(...)` if provided, but not an Incident.

`run(...)` parameters: `source` (a URI or iterable), `on_incident` (the notify sink), `on_ask` (the
refusal sink), `on_frame(t, frame)` (the live-feed hook — wire an MJPEG streamer here), `fps`,
`source_queue_size`, `source_reconnects`, `source_reconnect_backoff_s`, `source_rtsp_tcp`,
`snippet_pre_s` (seconds of pre-roll per clip), `snippet_kind` (`"frame"` or `"clip"`), and `sink_errors`
(`"raise"` or `"ignore"`). Pass `stats=RuntimeStats()` to collect frame, escalation, judge-call, fire,
ask, incident, and sink-error counters.

`on_health(event)` receives structured lifecycle events such as `runtime_started`, `escalation`, `ask`,
`incident`, `incident_sink_error`, `runtime_error`, and `runtime_ended`.

## The `Incident`
```
Incident { id, concern_id, verdict{satisfied, confidence, evidence}, ts, label, snippet, message }
```
`message` is the goal's `say` (the human alert line); `verdict.evidence` is the VLM's reasoning;
`snippet` is the pre-roll (`kind="frame"` stores decoded frames; `kind="clip"` stores a portable encoded
frame bundle suitable for webhook/storage sinks);
`label` is `None` until a human / the calibration loop marks it `"real"` / `"false_alarm"`.

## Tuning a goal — three independent knobs
A goal's `state` carries three orthogonal levers (the conductor can author them from plain language):

| knob | `state` key | what it controls | example |
|---|---|---|---|
| **what fires** | (the `condition` + behavior) | the plain-language trigger + rising-edge logic | "a person has fallen" |
| **how deep a look** | `perception` (PerceptionSpec) | single frame vs a **strip** of N (see the transition), ROI, fused audio | `{"temporal": "strip", "n_frames": 4}` |
| **how often to look** | `cadence` | motion-gated (default) vs a **fixed** cadence for low-motion content | `{"min_judge_interval": 0.5, "gate_on": "fixed"}` |
| **how much it matters** | `miss_cost` / `false_fire_cost` | biases the gate (a severe miss → asks instead of staying silent) | `"miss_cost": "severe"` |

### Cadence — `gate_on`
`gate_on` picks the signal that arms a goal (`how-often`, distinct from `perception` = `how-deep`):
- **`motion`** (default) — pixel-motion gate. Right for **physical** events (a person, a package, a fall).
- **`fixed`** — escalate every `min_judge_interval` seconds **regardless of motion**. Required for
  **content** events (a screenshare, a dashboard) where pixel motion is ~0 and the motion gate would
  silently under-sample. `cadence_s` also rate-limits a motion goal (escalate at most every N seconds).
- **`structural` / `semantic`** — the richer Tier-0 channels (CLIP/pose); filled by `percept-edge`.

## Worked scenarios
- [`examples/fall_notification.py`](../examples/fall_notification.py) — a **physical** event: motion-gated,
  `miss_cost="severe"`, `temporal="strip"`. See [docs/scenarios/fall.md](scenarios/fall.md).
- [`examples/screenshare_assistant.py`](../examples/screenshare_assistant.py) — a **content** event:
  `gate_on="fixed"` on a zero-motion screen. See [docs/scenarios/screenshare.md](scenarios/screenshare.md).

> **Assistant-class, not a guarantee.** The runtime *alerts a human who stays responsible*. It is not a
> certified safety/detection system on a clock it can't promise. See the fall scenario for the disclaimer.

## Webhook sink

```python
from percept.runtime import webhook

dead_letter = FileDeadLetterQueue("var/percept-dead-letter.jsonl")
sink = webhook(
    "https://example.com/percept/incidents",
    headers={"Authorization": "Bearer ..."},
    signing_secret="shared-secret",
    retries=2,
    dead_letter=dead_letter,
)
async for incident in run(agent, source="webcam://0", on_incident=sink, snippet_kind="clip"):
    ...
```

Webhook payloads redact media by default and keep incident metadata, message, evidence, timestamps, and
snippet bounds. Pass `include_media=True` only for trusted local/dev sinks. If `signing_secret` is set,
the sink adds `X-Percept-Signature: sha256=<hmac>` over the JSON body. Failed deliveries retry, then append
to `dead_letter` if supplied before re-raising to the runtime.
