# The runtime worker — `run(source) → incidents`

The runtime owns the **camera → incidents** lifecycle: capture, the gate loop, judging off the display
thread, the live feed, and rising-edge **incident** emission — so your app stops writing that glue.

```python
from percept import Percept, Goal
from percept.runtime import FileDeadLetterQueue, RuntimeStats, run, webhook

agent = Percept.create(vision="gemini")            # [gemini] extra + GEMINI_API_KEY; code-armed watch
                                                   # (goals via add_goal) — the default conductor stays
                                                   # SCRIPTED (loud log, no crash). Keyless? mode="offline".
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

### Build the core in the call — the convenience form

You can hand `run()` a pre-built `agent`/core (as above) **or** let it build one for you: pass `goals=`
(plain-language conditions) and `vision=` and skip `Percept.create` entirely.

```python
async for incident in percept.run(
    source="webcam:0",
    goals=["the milk is boiling over"],   # each string becomes a Goal
    vision="gemini",                      # backend by name (default: fake)
):
    print(incident.message)
```

### `run(...)` parameters

**Source & core**
- `source` — a URI (`"webcam:0"`, `"rtsp://…"`, a file path) or a pre-decoded `(t, gate_frame, judge_frame)` iterable.
- `goals` — plain-language conditions (convenience form); each becomes a `Goal`. **Only for the coreless form** — passing `goals=` (or `vision=`) *together with* a built core raises `ValueError` rather than silently discarding them.
- `vision` — the vision backend by name (`"gemini"` / `"claude"` / `"fake"`) for the convenience form.
- `fps` — frame rate to pull the source at. The runtime may open at a **higher** rate (a brief-event plan's cadence, or the 8 fps headroom cap a URI source must take at open time). That is never silent: `stats.effective_config.fps` records `requested_fps`, `effective_fps` and a `reason` (`brief-window-coverage` / `source-limit` / `profile-default`), also emitted once as an `effective_config` health event and mirrored on `core.effective_config`.
- `timebase` — which clock frame `t` (and so `incident.ts`) is stamped on. `"auto"` (default) = **media** time for recorded sources (a file — so a timestamp means something against the media) and **wall**/monotonic for live devices/streams (`webcam:0`, `rtsp://…` — those share the agent's deadline clock). `"media"`/`"wall"` force it. Both clocks are recorded either way: `stats.last_media_t` / `stats.last_wall_t`.
- `source_queue_size`, `source_reconnects`, `source_reconnect_backoff_s`, `source_rtsp_tcp` — capture/reconnect tuning for live URIs.

**Sinks & hooks**
- `on_incident` — the notify sink (webhook/SMS/log) for each rising-edge fire.
- `on_ask` — the low-confidence **refusal** sink (an `ask`, not an Incident).
- `on_frame(t, frame)` — the live-feed hook; wire an MJPEG streamer here.
- `on_health(event)` — structured lifecycle events: `runtime_started`, `escalation`, `ask`, `incident`, `incident_sink_error`, `runtime_error`, `runtime_ended`.
- `on_trace(event)` — the per-frame **decision trace** (why a frame stayed silent — see below).
- `sink_errors` — `"raise"` or `"ignore"` when a sink throws.
- `stats` — pass `RuntimeStats()` to collect frame, escalation, judge-call, fire, ask, incident, and sink-error counters.

**Snippet (the incident clip)**
- `snippet_pre_s` — seconds of pre-roll per clip.
- `snippet_size` — max frames/size of the buffer.
- `snippet_kind` — `"frame"` or `"clip"`.

**Gate & episode**
- `gate_source` — an alternate cheap-gate frame source (decoupled from the judge frame).
- `episode_inactivity_s` — idle gap that closes an episode.
- `coverage_staleness_s` — how long without an observation before a coverage-loss ("I can't see the stove") fires.
- `regate_on_goal_change` — re-run the tier-0 gate when goals are armed/disarmed at runtime (default off).

**Audio (fused listen)**
- `audio_source` — a mic/audio URI (`"mic:0"`) that fills the audio window for `modality="fused"` goals. Extra: `[mic]`.
- `audio_gate` — arm the tier-0 audio channel (onset/energy) so a sound can escalate a frame the motion gate would skip.
- `audio_wake` — treat a wake phrase on the audio channel as an escalation.
- `on_utterance` — sink for finalized speech utterances.
- `vad`, `utterance_threshold`, `utterance_hangover_frames`, `utterance_min_speech_frames` — voice-activity / endpointing tuning.
- `wake_policy` — the weighted edge wake policy (fuse `audio_event`, `presence`, `change`, … into one cheap score). See [reference/edge-policy.md](../reference/edge-policy.md).

**Runtime control & durability**
- `paused` / `muted` — start paused (no judging) or muted (judge, but suppress alerts).
- `clock` — inject a clock (tests / deterministic replay).
- `persistence` / `restore` / `autosave_every_s` — persist cognition state to a store, restore it on start, and autosave every N seconds so a restart doesn't lose the gate/entity state.

> **Which gate for your camera? (the decision that sinks realtime demos)**
>
> The default tier-0 gate is a **whole-frame motion** gate. On a **static or low-motion** subject it
> will **never escalate** — the whole-frame mean stays ~0 and no judge call is ever spent, so the agent
> looks broken. Pick one:
> - **A still / content scene** (a screenshare, a dashboard, a stove that isn't moving) → `gate_on="fixed"`
>   + `min_judge_interval` so it escalates on a **cadence** regardless of motion.
> - **A localized event on an otherwise-still frame** (a coffee sip, a shelf grab, a terminal error on a
>   4K screen) → `salience_stat="block_max"` — measure the frame-diff over a block grid, not the whole-frame
>   mean, so the local event clears threshold.
> - **A moving / wearable camera** (head-cam, drone, handheld) → `ego_compensate=True` so whole-frame
>   pan/head-turn is subtracted and the gate fires on the **residual**, not on the camera's own motion.
>
> `salience_stat`, `ego_compensate`, and their per-stat thresholds live in
> [reference/edge-policy.md](../reference/edge-policy.md).

### `on_trace` — why a frame stayed silent

`on_trace(event)` receives the **decision trace**: the per-frame stages the runtime walks, so you can
answer "why didn't that fire?" without guessing. The stages:

- `source.frame` — a frame arrived from the source.
- `judge.started` — the tier-0 gate escalated; a `vision.judge` call is being spent.
- `judge.finished` — the judge returned a `Verdict` (satisfied / confidence / evidence).
- `gate.classified` — the cognition gate mapped the verdict to `fire` / `ask` / `silent`.
- `incident.created` — a rising-edge `fire` became an `Incident`.

A silent frame that has a `source.frame` but no `judge.started` was **gated out cheaply** (motion below
threshold, or off-cadence) — the whole-frame-motion trap above. A `judge.finished` with no
`incident.created` means the VLM looked and the gate held (not satisfied, still refractory, or an `ask`).

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
| **what fires** | (the `condition` + behavior) | the plain-language trigger + rising-edge logic | "a package is on the porch" |
| **how deep a look** | `perception` (PerceptionSpec) | single frame vs a **strip** of N (see the transition), ROI, fused audio | `{"temporal": "strip", "n_frames": 4}` |
| **how often to look** | `cadence` | motion-gated (default) vs a **fixed** cadence for low-motion content | `{"min_judge_interval": 0.5, "gate_on": "fixed"}` |
| **how much it matters** | `miss_cost` / `false_fire_cost` | biases the gate (a severe miss → asks instead of staying silent) | `"miss_cost": "severe"` |

### Cadence — `gate_on`
`gate_on` picks the signal that arms a goal (`how-often`, distinct from `perception` = `how-deep`):
- **`motion`** (default) — pixel-motion gate. Right for **physical** events (a person raising a hand, a
  package dropped off).
- **`fixed`** — escalate every `min_judge_interval` seconds **regardless of motion**. Required for
  **content** events (a screenshare, a dashboard) where pixel motion is ~0 and the motion gate would
  silently under-sample. `cadence_s` also rate-limits a motion goal (escalate at most every N seconds).
- **`structural` / `semantic`** — the richer Tier-0 channels (CLIP/pose); filled by `percept-edge`.

## Worked scenarios
- **Physical event (porch package)**: motion-gated, `temporal="strip"`. See [porch-package.md](porch-package.md).
- **Content event (screenshare)**: `gate_on="fixed"` on a zero-motion screen. See [screenshare.md](screenshare.md).

> **Assistant-class, not a guarantee.** The runtime *alerts a human who stays responsible*. It is not a
> certified safety/detection system standing between a person and harm — percept is never the sole
> safety mechanism for a high-consequence, life-safety condition. See the porch-package scenario for
> the disclaimer pattern.

## Beyond fires — what else `run()` surfaces

`run()` isn't only rising-edge fires. These ride the same loop, opt-in per goal (default off → unchanged):

- **Listen too (fused audio).** `run(agent, source="webcam:0", audio_source="mic:0")` fills the audio
  window so a `modality="fused"` goal judges sound + vision together, and an off-camera sound onset can
  escalate a frame the motion gate would skip. Extra: `[mic]`. See [audio-modality.md](../reference/audio-modality.md).
- **Absence-native fires.** A goal with `state["miss_cost"]="severe"` fires a **coverage-loss** incident
  when it stops being observed ("I can't see the stove") — perception-robust (fires on the *lack* of
  evidence). A goal with `state["rearm"]="deadline", state["deadline_s"]=…` fires **overdue** when an
  expected event doesn't happen in time. Both flow through `on_incident`.
- **Cost / escalation metering.** Pass `stats=RuntimeStats()` for frame/escalation/judge/fire counters;
  `agent.metrics` carries the `seen→gated→judged→refused→fired` funnel plus per-judge **token counts**
  (each `Verdict.usage`, mirrored to OpenTelemetry via the `[otel]` extra).
- **Deictic grounding (`exemplar`).** Give a goal a reference image via `state["exemplar"]=(mime, bytes)`
  and the judge answers "is this **the** specific X?" against it — individuation, not just category.
- **Drive it live.** Wrap the watch in a `RuntimeSession` to arm goals, ask frame-grounded questions,
  and pause/mute at runtime — the command seam: [command-seam.md](../reference/command-seam.md).

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
to `dead_letter` if supplied before re-raising to the runtime. Dead-letter records are redacted (the URL is
reduced to `scheme://host[:port]`; the body is stored only as a SHA-256; no headers/signature).

### URL safety (SSRF)

By default a webhook URL must be **https to a publicly routable host** — `UrlPolicy` blocks `http://`,
loopback, private, link-local (the cloud metadata IP), CGNAT, and reserved targets, so a delivery can't be
aimed at an internal service. For a local/dev sink, opt out explicitly:

```python
from percept.runtime.sinks import UrlPolicy
sink = webhook("http://localhost:8080/hook", url_policy=UrlPolicy.dev())
```

**If the webhook URL comes from untrusted input** (e.g. a multi-tenant app where users choose where alerts
go), pin the hosts you actually deliver to — this is the recommended hardening and closes the vector:

```python
sink = webhook(user_url, url_policy=UrlPolicy(
    allowed_hosts=frozenset({"hooks.slack.com", "discord.com"}),
))
```

> **Known limitation.** The private-IP check re-resolves the host just before sending, but `urllib` resolves
> again when it connects, so a DNS-rebinding attacker with a very short TTL has a narrow residual window
> (time-of-check/time-of-use). The default policy plus `allowed_hosts` closes the practical vector; for an
> untrusted-input deployment you should **also restrict egress at the network layer** (no route to
> `169.254.169.254` / internal ranges) — the real backstop for SSRF. Closing the residual in-library
> (pinning the resolved IP through the TLS connection) is planned but not yet shipped.
