---
title: Two apps on one command seam — a silent watchdog and a voice assistant
updated: 2026-07-16
applies_to: percept-vision 1.0.0
status: stable
---
# Two apps on one command seam — a silent watchdog and a voice assistant

`percept.run` is the watch; `RuntimeSession` is the [command seam](../watches/driving-a-live-watch.md). The
same SDK underneath, two very different application **modalities** on top:

- **Watchdog** — silent, headless, programmatic. Fully runnable below.
- **Voice** — a voice-to-voice adapter. Needs a mic/camera/key, so it's shown as an app shape, not an
  offline snippet.

The SDK ships neither modality — it ships the seam. This is the boundary: *how* you command the agent
(a webhook config, a voice loop, a Slack bot) is your application's code.

---

## A. Silent watchdog (headless, programmatic) — fully runnable

Commands arrive **in code** (goals armed programmatically); output goes to a **notification sink**. No
mic, no speaker, no UI. This is the construction-inspection / security / compliance shape. It dogfoods
the SDK's own `WebhookSink` (SSRF policy, retries, signing) — you don't hand-roll the POST.

```bash
pip install "percept-vision[gemini,runtime]"     # + a camera / RTSP / video file
export GEMINI_API_KEY=...
```

```python
import asyncio
import percept
from percept.adapters import Goal
from percept.runtime import WebhookSink   # ships in the SDK

async def main():
    core = percept.Percept.create(vision="gemini", discover_plugins=False)
    # Programmatic command modality: goals armed in code (a config file / API would do the same).
    core.add_goal(Goal(id="no-hardhat", condition="a person on site without a hard hat",
                       say="PPE violation: a worker without a hard hat."))

    notify = WebhookSink("https://example.com/alerts", retries=1)   # POSTs each incident as JSON

    # The whole app: the SDK drives capture → gate → judge → incident; you consume incidents.
    async for incident in percept.run(core, source="webcam:0", on_incident=notify):
        print(f"[WATCHDOG] {incident.concern_id}: {incident.message}")

asyncio.run(main())
```

That's it — no capture loop, no frame encoding, no HTTP plumbing. For a purely local sink, log the
`incident_payload(incident)` dict to a file instead of (or alongside) the webhook.

---

## B. Voice-to-voice assistant — the adapter shape (needs hardware)

Commands arrive **by voice**; fires are **spoken**. The voice engine (mic, STT, TTS, barge-in) is the
*application's* adapter — the SDK only supplies the frame-grounded seam. Sketch:

```python
import percept
from percept.runtime import start          # RuntimeSession over a running watch

async def main():
    core = percept.Percept.create(vision="gemini", discover_plugins=False)
    session = await start(core, source="webcam:0",
                          on_incident=speak,           # proactive nudge → your TTS
                          on_frame=stash_for_ui)       # multi-consumer: UI + voice both read frames

    # Your voice loop (Gemini Live / a realtime engine — APP code, not the SDK):
    #   • a wake word opens a conversation session (ambient entry, not a talk button)
    #   • per spoken turn: send [ session.latest_frame() + the transcript ] as ONE ordered
    #     client_content turn so Live actually SEES the frame ("what do you see?" works)
    #   • an arm_goal tool-call → session.arm(Goal(...))  (the watch re-arms)
    #   • the assistant's own reply is muted from the mic (session-boundary / mute-while-speaking)
```

Key facts this shape depends on (verified against the Gemini Live API):

- **Reliable vision needs the ordered `client_content([image, text])` turn.** Streaming a single frame
  via realtime `video=`/`media=` races the question and hallucinates; `client_content` rejects audio
  parts — so a spoken question is transcribed to text, then sent *with* the frame in one ordered turn.
- **Engagement is app policy, not SDK policy.** Wake word / follow-up window / push-to-talk all live in
  your adapter; the SDK stays policy-free.
- **The watch stays on Percept (`percept.run`, stateless), the Live session is ephemeral** (opened per
  conversation) — the 24/7 promise is the watch's, not a held socket's.

The tracked `templates/realtime_voice_watch_agent/` application demonstrates the camera, microphone,
and voice-engine composition; a live run requires the corresponding provider key.

---

See also: [runtime.md](../watches/creating-a-watch.md) (the watch in full), [command-seam.md](../watches/driving-a-live-watch.md)
(`RuntimeSession`), [audio-modality.md](audio.md) (audio as a *sensor* vs. audio as
a *command*).
