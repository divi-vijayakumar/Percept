---
name: percept-agent
description: Build a proactive vision/perception agent with Percept — watch a stream for a plain-language
  condition and fire/ask only on the rising edge. Use when the user wants a "watch X / tell me when Y" agent.
---

# Build a Percept watch agent

Percept is the OSS **cognition layer** for goal-driven, proactive vision agents. You state a goal in
plain language ("tell me when the kettle boils") and Percept turns a stream of per-frame perception
verdicts into an agent that **fires once on the rising edge** and **refuses to guess** when unsure.

Use this skill when the user wants a "watch X / notify me when Y" perception agent. Scaffold from the
template in this directory, then adapt.

## 1. Install

```bash
pip install percept-vision        # core: PURE STDLIB, offline with the FAKE backend, no keys
```

For a real (perceiving) agent, add one extra + its key:

```bash
pip install "percept-vision[gemini]"   # then: export GEMINI_API_KEY=...   (or GOOGLE_API_KEY)
pip install "percept-vision[claude]"   # then: export ANTHROPIC_API_KEY=...
```

The package is `percept-vision`; the import is `import percept`. Python >= 3.10.

## 2. The real watch agent — `percept.run` owns the capture loop

**Default path.** `percept.run` opens the source, runs the tier-0 motion gate (a still scene spends no
cloud calls), judges, and **yields an `Incident` on each rising-edge fire**. You do NOT write a capture
loop. Needs `[runtime]` (opencv) + a backend + a key.

```python
import asyncio
import percept
from percept.adapters import Goal

async def main():
    agent = percept.Percept.create(vision="gemini", discover_plugins=False)
    agent.add_goal(Goal(
        id="kettle",
        condition="the kettle has finished boiling",   # plain language — the VLM judges this
        say="The kettle's done!",                       # spoken/surfaced when it fires
    ))
    # source: "webcam:0" | an "rtsp://…" URL | a video file path. Add audio_source="mic:0" to also LISTEN.
    async for incident in percept.run(agent, source="webcam:0", on_incident=notify):
        print(incident.concern_id, incident.message)

asyncio.run(main())
```

`run` also takes `on_ask=` (low-confidence refusals), `on_frame=` (a live-feed hook), `stats=RuntimeStats()`
(cost/escalation counters), and `audio_source="mic:0"` (fused audio+video). To **drive** a running watch —
arm goals by voice, ask "what do you see?", pause/mute — wrap it in a `RuntimeSession` (the command seam):
`docs/reference/command-seam.md`.

## 2b. Offline / test path (no keys) — `perceive_judged` + FakeVision

For deterministic tests you control the *perception* by hand: `FakeVision` maps a string token to a
scripted `Verdict`, and you drive the cognition layer frame-by-frame. This is the "bring-your-own-frames"
low-level path — use it for tests, not for a real capture loop.

```python
agent = percept.Percept.create(vision="fake", discover_plugins=False)
agent.add_goal(Goal(id="kettle", condition="the kettle has finished boiling", say="The kettle's done!"))
for token, t in [("calm", 1.0), ("boiling", 2.0)]:      # scripted tokens, not a live camera
    for ev in await agent.perceive_judged(token, t=t):
        if ev.action == "fire": handle(ev.text)
        elif ev.action == "ask": clarify(ev.text)        # low-confidence: ask, don't guess
```

Confirmed signatures (from `packages/percept-vision/percept/agent.py` and `adapters.py`):

- `percept.run(core, *, source, on_incident=None, on_ask=None, on_frame=None, audio_source=None,
  stats=None, ...) -> AsyncIterator[Incident]` — the batteries-included watch (async generator).

- `Percept.create(*, vision=None, discover_plugins=True, ...)` — `vision` is a backend **name**
  (`"fake"`/`"gemini"`/`"claude"`) or an adapter instance. Pass `discover_plugins=False` for the
  offline/no-plugin path.
- `agent.add_goal(Goal(id, condition, say))` — `Goal` also has optional `source="camera"`,
  `modality="speak"`, `state`, `expiry`.
- `await agent.perceive_judged(frame, source="camera", *, t=None) -> list[FireEvent]` — async.
- `FireEvent(goal_id, action, text, entity_id=None, verdict=None)` — `action` is `"fire"` or `"ask"`
  (a `SILENT` decision yields **no** event).

## 3. The three-state gate + rising-edge firing

This is the whole point — the agent never nags and never confidently-wrong fires:

| verdict for this frame                          | event                          |
|-------------------------------------------------|--------------------------------|
| confident **yes**, fresh, on a **rising** edge  | `fire` (once)                  |
| confident **yes**, already fired / still true   | *nothing* (refractory — no nag)|
| confident **no**                                | *nothing* (re-arms the edge)   |
| **low confidence** or stale fact                | `ask` (refuses to guess)       |

- **Rising edge:** a condition true for 100 frames fires **once**, not 100 times. It can fire again
  only after going confidently false in between (which re-arms it).
- **ASK** is the trust state: below the confidence threshold (default `0.7`), the gate asks instead of
  delivering a wrong alert.

## 4. The frame contract

A "frame" is whatever the chosen backend accepts:

- **FAKE backend** — a frame is a **string token** you script to a `Verdict`. This is for deterministic
  offline tests: you control perception by hand while the cognition layer does real work. Build it with
  `FakeVision(script={"token": Verdict(satisfied=..., confidence=..., evidence=...)})` and pass the
  instance as `vision=`. (Bare `vision="fake"` uses a few built-in demo tokens like `"sip-coffee"`.)
- **gemini / claude** — a frame is **real image input**: image bytes, or a `(media_type, bytes)` pair,
  e.g. from a webcam via OpenCV: `cv2.imencode(".jpg", bgr_frame)[1].tobytes()`. Everything above the
  perception seam (gate, rising edge, memory) is identical — only the eyes change.

The cognition layer never sees pixels; it reasons only over the `Verdict{satisfied, confidence,
evidence}` stream.

## 5. Decision guide — offline prototype vs real

| you want…                                  | use                                                              |
|--------------------------------------------|-----------------------------------------------------------------|
| prove wiring + cognition logic, no keys    | `vision=FakeVision(script={...})`, scripted **string** frames   |
| a real agent that perceives the world      | `vision="gemini"` (+ `GEMINI_API_KEY`), frames = camera **bytes** |

Start offline with the FAKE backend to nail the goal wording and the fire/ask behavior
deterministically, then swap `vision=` and the frame source for the real thing — no other code changes.

## Scaffold from the template

`template_watch_agent.py` (in this directory) is a SHORT, runnable offline watch agent: it scripts a
few frames and prints an ASK (on a low-confidence frame) then a FIRE (on the rising edge). Copy it,
rename the goal, and either keep the FAKE backend for tests or swap to `vision="gemini"` + a real frame
source (a `cv2.VideoCapture` loop yielding JPEG bytes).

Verified output of the template (offline, no keys):

```
[ASK ] kettle: About "the kettle has finished boiling" — can you confirm?
[FIRE] kettle: The kettle's done — go make your tea.
```

## Notes / gotchas

- `perceive_judged` is **async** — `await` it inside an `asyncio` event loop.
- Import goal/verdict DTOs from `percept.adapters` (`Goal`, `Verdict`, `FireEvent`); `Percept`, `Goal`,
  `Verdict`, `FireEvent` are also re-exported at the top level (`from percept import ...`).
- Pass a monotonic `t=` (seconds) per frame so freshness/refractory logic is deterministic in tests;
  omit it to use the real clock.
- Don't lead with surveillance demos — Percept is assistant-class (informs a consenting user who stays
  responsible), never a safety mechanism on a clock it can't guarantee. See the repo README "envelope".
- More: `docs/quickstart.md`, `docs/concepts.md` (gate, rising edge, escalation ladder),
  `docs/backends.md` (extras, keys, env-var selection).
