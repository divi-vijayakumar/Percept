---
name: percept-agent
description: Build a proactive vision/perception agent with Percept — watch a stream for a plain-language
  condition and fire/ask only on the rising edge. Use when the user wants a "watch X / tell me when Y" agent.
---

<!-- KEEP IN SYNC: this file is duplicated at .claude/skills/percept-agent/SKILL.md and
     .agents/skills/percept-agent/SKILL.md. Edit BOTH together (no symlink). Same for the
     template_watch_agent.py / template_watch_agent_live.py pairs in each dir. -->

# Build a Percept watch agent

Percept is the OSS **cognition layer** for goal-driven, proactive vision agents. You state a goal in
plain language ("tell me when the kettle boils") and Percept turns a stream of per-frame perception
verdicts into an agent that **fires once on the rising edge** and **refuses to guess** when unsure.

Percept is **v1.0.0** — `pip install percept-vision`, `import percept`. Home: <https://percept.percia.ai>.

Use this skill when the user wants a "watch X / notify me when Y" perception agent. Scaffold from the
template in this directory, then adapt.

## 0. Fastest path — the `percept-watch` CLI (no code)

```bash
pip install "percept-vision[runtime,openrouter]"     # capture (OpenCV) + the Qwen judge
export OPENROUTER_API_KEY=sk-or-...                   # cheap, fast, hosted
percept-watch --goal "someone raises their hand" --goal "a package is on the porch"
#   --source webcam:0 | clip.mp4 | rtsp://…   --egomotion (moving camera)   --fps 1  --refractory 15
```
Prints a `🔔` once per event. This is the whole demo — reach for the API below when embedding in an app.

## 1. Install

```bash
pip install "percept-vision[runtime,openrouter]"   # RECOMMENDED: capture + Qwen (cheap/fast). key: OPENROUTER_API_KEY
```
Other judges (swap `vision=`): `pip install "percept-vision[gemini]"` (`GEMINI_API_KEY`) ·
`"percept-vision[claude]"` (`ANTHROPIC_API_KEY`). Core install is pure-stdlib and runs offline on the FAKE
backend (no keys) for tests. The package is `percept-vision`; the import is `import percept`. Python >= 3.10.

> **Recommended defaults for a real watch:**
> ```bash
> export PERCEPT_SALIENCE_STAT=block_max     # localized motion — catches low-motion actions (holding, sitting)
> export PERCEPT_GATE_REFRACTORY_S=15        # one notification per episode (suppress duplicate fires)
> export PERCEPT_EGO_COMPENSATE=true         # MOVING camera only (glasses/body-cam): subtract head motion
> ```
> On a **fixed** camera use `block_max` + refractory. On **glasses/body-cam**, also set `ego_compensate`:
> a moving camera otherwise escalates on nearly every frame, and ego-compensation subtracts the head
> motion so only real scene changes reach the judge. Full guide: `docs/watches/creating-a-watch.md`. Every tunable env
> var is listed in `docs/configuration/overview.md`.

## 2. The real watch agent — `percept.run` owns the capture loop

**Default path.** `percept.run` opens the source, runs the tier-0 motion gate (a still scene spends no
cloud calls), judges, and **yields an `Incident` on each rising-edge fire**. You do NOT write a capture
loop. Needs `[runtime]` (opencv) + a backend + a key.

```python
import asyncio
import percept
from percept.adapters import Goal

async def main():
    # vision="gemini" needs GEMINI_API_KEY. Code-armed watch (goal armed below, no voice): the default
    # conductor stays SCRIPTED, so this constructs with a loud "language goal-authoring is off" log — not
    # a crash. For a keyless smoke test use mode="offline"; to author goals from speech, conductor="gemini".
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
`docs/watches/driving-a-live-watch.md`.

## 2b. Offline / test path (no keys) — `perceive_judged` + FakeVision

For deterministic tests you control the *perception* by hand: `FakeVision` maps a string token to a
scripted `Verdict`, and you drive the cognition layer frame-by-frame. This is the "bring-your-own-frames"
low-level path — use it for tests, not for a real capture loop.

```python
agent = percept.Percept.create(mode="offline", vision="scripted", discover_plugins=False)  # "scripted" is canonical; "fake" still works
agent.add_goal(Goal(id="kettle", condition="the kettle has finished boiling", say="The kettle's done!"))
for token, t in [("calm", 1.0), ("boiling", 2.0)]:      # scripted tokens, not a live camera
    for ev in await agent.perceive_judged(token, t=t):
        if ev.action == "fire": handle(ev.text)
        elif ev.action == "ask": clarify(ev.text)        # low-confidence: ask, don't guess
```

Confirmed signatures (from `packages/percept-vision/percept/agent.py` and `adapters.py`):

- `percept.run(core, *, source, on_incident=None, on_ask=None, on_frame=None, audio_source=None,
  stats=None, ...) -> AsyncIterator[Incident]` — the batteries-included watch (async generator).

- `Percept.create(*, vision=None, conductor=None, mode=None, allow_scripted_brain=False,
  discover_plugins=True, ...)` — `vision`/`conductor` are a backend **name** (`"scripted"` (canonical;
  `"fake"` still works) / `"gemini"` / `"claude"`) or an adapter instance. **The default is NOT
  scripted:** a bare `create()` in live mode raises `ModeError` rather than silently faking — pass
  `mode="offline"` for keyless scripted backends, or select a real backend + its API key. A scripted
  conductor behind a live voice path (`on_utterance`, or an app-side voice engine) is a `ModeError`
  ("scripted brain behind a live voice"); `allow_scripted_brain=True` is the ugly-on-purpose override.
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

Two templates ship in this directory — start from whichever matches what you're building:

- **`template_watch_agent_live.py` — the real watch agent (start here).** `create(vision="gemini")`
  → `add_goal(...)` → `async for incident in percept.run(agent, source="webcam:0", on_incident=...)`.
  `run` owns the capture loop; to point it elsewhere just change `source=` to an `"rtsp://…"` URL or a
  video-file path. Do **not** hand-roll a `cv2.VideoCapture` loop — that's what `run` is for.
- **`template_watch_agent.py` — the offline / BYO-frames path (tests).** A SHORT, runnable, keyless agent
  that scripts a few frames and prints an ASK (low-confidence frame) then a FIRE (rising edge). Copy it,
  rename the goal, keep the FAKE backend for deterministic tests.

Verified output of the offline template (no keys):

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
- More docs:
  - `docs/get-started/quickstart.md` — pip-install to a firing event, offline then real backend.
  - `docs/resources/how-it-works.md` — gate, rising edge, escalation ladder.
  - `docs/judges/backends.md` — extras, keys, env-var backend selection.
  - `docs/configuration/overview.md` — every `PERCEPT_*` env var / threshold; `docs/configuration/profiles.md` — author a profile.
  - `docs/judges/custom-specialist-judge.md` — bring your own vision backend or a specialist judge (with the conformance suite).
  - `docs/admission/tuning-admission.md` — tune the eye for your content; `docs/configuration/optimize.md` — search safe tunables.
  - `docs/evaluation/evaluate-your-data.md` · `docs/evaluation/running-benchmarks.md` — evaluate + benchmark on your data.
  - `docs/resources/cli.md` — all nine `percept-*` commands; `docs/watches/creating-a-watch.md` — the full runtime integration guide.
