# The command seam — `RuntimeSession`

`percept.run` is the **watch** — capture → gate → judge → incidents, running headless. `RuntimeSession`
is the **command seam**: the small, stable interface for *driving* a running watch — arm goals, ask a
question grounded on the current frame, read frames, pause/mute — while it keeps watching.

The boundary this draws: **the SDK ships the command *interface*; your application ships the command
*modality*.** How commands arrive — a voice loop, a programmatic call, a text box, Slack — is app code.
The SDK never learns what "low latency" or "wake word" means; that's the app's concern.

- A **voice** assistant builds a voice-to-voice adapter on this seam (mic → STT/engine → `arm`/`ask`;
  incidents → speech).
- A **construction-inspection** agent uses it **programmatically** (arm goals in code; `on_incident`
  → email/report; no voice).

Same seam underneath both.

## Start a session

```python
import percept
from percept.runtime import start   # or: percept.start

session = await start(
    core,                       # percept.Percept.create(vision="gemini")
    source="webcam:0",
    on_incident=notify,         # proactive fires (the watch keeps running underneath)
    on_frame=ui_panel,          # optional; see subscribe() for multiple consumers
)
```

`start` wraps `percept.run(...)` in a background task and returns the session. The headless
`async for incident in percept.run(...)` remains the front door; this is the interactive layer on top.

## The interface

| Method | What it does |
|---|---|
| `arm(goal)` | Arm a watch goal at runtime (e.g. from a voice command). Bumps the goal registry so the gate re-arms for it. |
| `disarm(goal_id)` | Stop watching for a goal. |
| `ask(utterance, *, t=None)` | Answer a spoken/typed turn **through the conductor, grounded on `latest_frame()`** — this is how "what do you see?" is answered *with vision*. Runs as a cancellable task. |
| `interrupt()` | Cancel an in-flight `ask` (the barge-in boundary — *detecting* barge-in is the app's job). |
| `latest_frame()` | The watch's most recent frame — what `ask` scopes a vision answer to, and what a voice engine streams in. |
| `subscribe(cb)` | Register a frame consumer `cb(t, frame)` (sync or async). **Multi-consumer**: a UI panel *and* a voice adapter can both read frames. Returns `unsubscribe()`. |
| `pause()` / `resume()` | **SLEEP**: stop spending gate/judge on frames (cost-free) while the session stays alive and frames keep streaming. Not `stop` (which is terminal). |
| `detach_audio_source()` / `attach_audio_source()` | **MUTED**: stop feeding the mic to perception (fused judge goes video-only, energy/onset gate freezes) without ending the session; re-enable to resume. |
| `stop()` | Terminal — end the background watch. |

> A shared store (`EvidenceMemory` / `WatchPlan`) can be injected at construction (`start(..., store=…)`);
> the session *uses* it, does not *own* it. Semantics are part of the control-plane design (currently a
> stub).

## Example — arm by command, answer with vision

```python
session = await start(core, source="webcam:0", on_incident=notify)

# a command modality (voice STT, an HTTP handler, a config loader…) calls the seam:
await session.arm(Goal(id="coke", condition="the user is drinking a soda",
                       say="That's caffeine + sugar — want water instead?"))

result = await session.ask("what do you see right now?")   # conductor + latest_frame → grounded answer
print(result.reply)

session.pause()      # SLEEP: stop judging, keep the panel live
session.resume()
```

The **fires** (proactive nudges) flow out `on_incident`; the **commands** flow in through the seam. What
turns a microphone, a webhook, or a text box into those calls is the application adapter — not the SDK.

See also: [how-to/runtime.md](../how-to/runtime.md) (the headless watch), [reference/audio-modality.md](./audio-modality.md)
(audio as a sensor vs. audio as a command), [quickstart.md](../tutorials/quickstart.md).
