# Quickstart

Percept is the **cognition layer** for goal-driven, proactive vision agents. You state a goal in
plain language; Percept turns a stream of per-frame perception verdicts into an agent that **fires
once, on the rising edge** of a condition becoming true — and **refuses to guess** when it isn't sure.

This page gets you from `pip install` to a firing event in about 60 seconds — first fully offline
(no keys), then against a real vision backend.

---

## Install

```bash
pip install percept-vision        # core is pure stdlib — no heavy deps, no keys
```

The import is `percept`:

```python
import percept
```

The **core runs offline** with deterministic `fake` backends, so you can build and test the whole
cognition path — goals, the gate, entity memory, escalation — with no API keys. Real model backends
are optional extras (see [backends.md](../reference/backends.md)).

---

## 60-second offline example (no keys)

The `fake` vision backend is **scripted, not intelligent**: it maps a string "frame" token to a
pre-written `Verdict`. That's exactly what you want for a deterministic offline demo — it lets the
*cognition* layer (the gate, the rising-edge logic) do real work while you control the perceptual
input by hand.

```python
import asyncio
import percept
from percept.adapters import Goal, Verdict
from percept.backends.fake import FakeVision


async def main():
    # FakeVision maps a string "frame" token to a scripted Verdict. No model, no keys.
    vision = FakeVision(script={
        "calm":     Verdict(satisfied=False, confidence=0.95, evidence="no foam"),
        "boiling":  Verdict(satisfied=True,  confidence=0.95, evidence="foam at the rim"),
    })

    agent = percept.Percept.create(vision=vision, discover_plugins=False)
    agent.add_goal(Goal(
        id="milk",
        condition="the milk is boiling over",
        say="Heads up — the milk is boiling over!",
    ))

    # Feed scripted frames. The gate fires ONCE on the false->true rising edge.
    for token, t in [("calm", 1.0), ("calm", 2.0), ("boiling", 3.0), ("boiling", 4.0)]:
        events = await agent.perceive_judged(token, t=t)
        for ev in events:
            print(f"t={t}  {ev.action.upper():6}  {ev.text}")


asyncio.run(main())
```

Run it:

```bash
python3 quickstart.py
```

Output (verified):

```
t=3.0  FIRE    Heads up — the milk is boiling over!
```

**What just happened:**

- `calm` frames (`satisfied=False`) keep the gate **silent** — nothing fires.
- The first `boiling` frame is the **rising edge** (false → true) → exactly one `FIRE`.
- The second `boiling` frame is *still true*, so the gate's **refractory** keeps it silent — no
  double-fire. (Feed a `calm` frame in between and it would re-arm and fire again.)

`perceive_judged` returns a `list[FireEvent]`; each has `.goal_id`, `.action`
(`"fire"` / `"ask"`), and `.text`.

> The `fake` backend proves *your wiring and the cognition logic* — never model intelligence. To
> test that Percept actually perceives the world, point it at a real backend below.

---

## Real-backend example (Gemini)

Swap the `fake` vision backend for a real VLM by name. Everything above the perception seam — the
goal, the gate, the firing path — is **identical**; only the eyes change.

```bash
pip install "percept-vision[gemini]"
export GEMINI_API_KEY=...          # or GOOGLE_API_KEY
```

```python
import asyncio
import percept
from percept.adapters import Goal


async def main():
    # Resolve the real backend by NAME via the registry. Needs the [gemini] extra + a key.
    agent = percept.Percept.create(vision="gemini", discover_plugins=False)
    agent.add_goal(Goal(
        id="milk",
        condition="the milk is boiling over",
        say="Heads up — the milk is boiling over!",
    ))

    # `frame` is now a real image (e.g. bytes / a PIL image your capture loop produces),
    # not a scripted token. The backend calls the VLM: judge(condition, frame) -> Verdict.
    frame = capture_one_frame()          # your camera/video source
    events = await agent.perceive_judged(frame, source="camera")
    for ev in events:
        print(ev.action, ev.text)


asyncio.run(main())
```

That's the whole substitution: `vision="fake"` → `vision="gemini"` (or `"claude"`). The gate,
refractory, entity memory, and escalation are unchanged — they only ever see the `Verdict` stream,
never pixels.

You can also select backends per-seam via env vars without touching code
(`PERCEPT_VISION_BACKEND=gemini`, etc.) — see [backends.md](../reference/backends.md).

---

## Watch a live source — the batteries-included path

The example above calls `perceive_judged` frame-by-frame so you can see the gate work. For a **real**
watch agent you don't write a capture loop — **`percept.run` owns it**: it opens the source, runs the
tier-0 motion gate (so a still scene spends no cloud calls), judges, and **yields an `Incident` on each
rising-edge fire**.

```bash
pip install "percept-vision[gemini,runtime]"   # + a camera; runtime = opencv capture
export GEMINI_API_KEY=...
```

```python
import asyncio
import percept
from percept.adapters import Goal


async def main():
    agent = percept.Percept.create(vision="gemini", discover_plugins=False)
    agent.add_goal(Goal(id="milk", condition="the milk is boiling over",
                        say="Heads up — the milk is boiling over!"))

    # source: "webcam:0" | an "rtsp://…" URL | a video file path. No manual cv2 loop, no frame encoding.
    async for incident in percept.run(agent, source="webcam:0"):
        print(f"[FIRE] {incident.concern_id}: {incident.message}")

asyncio.run(main())
```

That's the whole agent. `run` also takes `on_incident=` (a notification sink — webhook/SMS/log),
`on_ask=` (low-confidence refusals), `audio_source="mic:0"` (fused audio+video — hear a kettle whistle,
not just see steam), and returns `RuntimeStats` for cost/escalation counters. Full options:
[how-to/runtime.md](../how-to/runtime.md).

To **drive** a running watch — arm goals by voice, ask "what do you see?", pause/mute it — wrap it in a
`RuntimeSession` (the command seam): [reference/command-seam.md](../reference/command-seam.md).

---

## Where to go next

- **[how-to/runtime.md](../how-to/runtime.md)** — `percept.run` in full: sources, sinks, audio, tuning.
- **[reference/command-seam.md](../reference/command-seam.md)** — `RuntimeSession`: arm/ask/pause/mute a
  live watch; the interface voice / programmatic / text apps build on.
- **[concepts.md](../explanation/concepts.md)** — the three-state gate, rising-edge firing, BehaviorProgram vs
  PerceptionSpec, edge detectors, and the escalation ladder.
- **[backends.md](../reference/backends.md)** — install extras, keys, env-var selection, and swapping backends.
- **[e2e-flow-milk-boilover.md](../explanation/e2e-flow-milk-boilover.md)** — a full trace of one request through
  the whole system, and where the perception bottleneck lives.
