# AGENTS.md — building with Percept

Guidance for AI coding assistants (Codex, Cursor, Copilot, Claude) integrating the **Percept** SDK
(`percept-vision 1.0.0`). Percept is a **cognition layer** for goal-driven, proactive vision agents: you
state a goal in plain language and it turns an audio-visual stream into an agent that fires **only on
the rising edge** of a condition becoming true, and **refuses to guess** when it's unsure. Docs online:
**[percept.percia.ai](https://percept.percia.ai)**.

## Mental model — two layers

```
  PERCEPTION (the eyes)                COGNITION (the brain)
  vision.judge(condition, frame)  ->   three-state gate:  fire · ask · silent
    -> Verdict{satisfied,                rising-edge firing (fires ONCE on false->true)
       confidence, evidence}             + executor + entity graph + edge detector registry
  stateless, per-call VLM              the brain never sees pixels — only Verdict streams
```

The seam is `vision.judge`. The brain consumes `Verdict`s; it cannot recover information a verdict
doesn't contain. Keep this split in mind when writing code: perception = "what's true in this frame",
cognition = "should I act, given the history".

## Install

```bash
pip install percept-vision                # CORE — pure stdlib, runs OFFLINE with the fake backend, no keys
pip install "percept-vision[gemini]"      # + Gemini backend     -> export GEMINI_API_KEY=...  (or GOOGLE_API_KEY)
pip install "percept-vision[claude]"      # + Claude backend     -> export ANTHROPIC_API_KEY=...
pip install "percept-vision[openrouter]"  # + OpenRouter (Qwen…) -> export OPENROUTER_API_KEY=...
pip install "percept-vision[edge]"        # + Python tier-0 gate + detector registry
pip install "percept-vision[runtime]"     # + camera/video/RTSP runtime worker
```

Python **>=3.10**. Import is `import percept`. The server-side edge (tier-0 salience gate + the cheap
on-device detector "skills") is imported as `percept.edge` and installed through the `[edge]` or
`[runtime]` extras. The harness ships bundled inside `percept-vision` as the `percept.harness` subpackage.

## The minimal agent (verified API)

```python
import asyncio
from percept import Percept, Goal
from percept.adapters import Verdict
from percept.backends.fake import FakeVision

# Offline: FakeVision maps a frame TOKEN (a string) to a scripted Verdict, so the cognition
# layer does the real work while you control perception by hand. Swap vision="gemini" for real.
vision = FakeVision(script={
    "boiling":  Verdict(satisfied=False, confidence=0.95, evidence="still boiling"),
    "stopped":  Verdict(satisfied=True,  confidence=0.95, evidence="steam stopped"),
})
agent = Percept.create(vision=vision, discover_plugins=False)
agent.add_goal(Goal(id="kettle", condition="the kettle has finished boiling",
                    say="The kettle's done — go make your tea."))

async def main():
    for t, frame in enumerate(["boiling", "boiling", "stopped", "stopped"]):
        for ev in await agent.perceive_judged(frame, t=float(t)):
            print(ev.action.upper(), ev.text)   # FIRE once on the rising edge; refractory suppresses the repeat

asyncio.run(main())
```

### API surface (confirm signatures in `packages/percept-vision/percept/agent.py`)
- `Percept.create(*, vision="scripted"|"gemini"|"claude"|"openrouter"|<instance>, discover_plugins=False, ...)` — the agent. (`"scripted"` is the canonical fake; `"fake"` still works.)
- `agent.add_goal(Goal(id=<str>, condition=<plain language>, say=<what to say on fire>))`.
- `await agent.perceive_judged(frame, source="camera", *, t=None) -> list[FireEvent]` — judges every active
  goal, runs each verdict through the gate, returns the rising-edge events for this frame (`[]` = silent).
- `FireEvent`: `.goal_id`, `.action` ∈ `{"fire", "ask"}`, `.text`. (`ask` = the gate refused to guess on a
  low-confidence read — surface it as a question, don't treat it as a fire.)

## Frame formats (per backend)
- **fake** — a `str` token you script in `FakeVision(script=...)`. Deterministic; ideal for tests.
- **gemini / claude** — image bytes, or a `(media_type, bytes)` pair, e.g.
  `("image/jpeg", cv2.imencode(".jpg", bgr_frame)[1].tobytes())`. This is the frame type `percept.run`
  hands the judge automatically from a `source=`; you only construct it yourself on the manual
  `perceive_judged` path.

## Gotchas / conventions
- **`percept.run` owns the capture loop.** Point it at `source="webcam:0"` / an RTSP URL / a video file
  and it drives capture → gate → judge → incident for you (`async for incident in percept.run(core,
  source=…)`; add `audio_source="mic:0"` for sound). See `docs/sources/voice-and-watchdog.md` for worked
  application shapes. The manual `perceive_judged`-per-frame loop is the advanced "bring-your-own-capture"
  path, not the default.
- **Rising-edge, not level.** A goal fires once when its condition goes false→true, then stays quiet
  (refractory) — so the agent never nags. Don't poll for "is it true now"; react to the `FireEvent`.
- **`ask` is a feature.** Low confidence yields `ask`, not a guess. This is the "refuse to deliver a
  confidently-wrong nag" contract — keep it.
- **Offline-first.** Write tests against `FakeVision` (no keys, deterministic). Only reach for a real
  backend for live perception.
- **Don't put pixels in the brain.** Everything below `vision.judge` works on `Verdict`s — keep
  perception logic on the perception side of the seam.

## Where to look
- `docs/README.md` (index — generated, every page dated) · `docs/get-started/quickstart.md` · `docs/resources/how-it-works.md` — adopter docs.
- `docs/get-started/coding-agents.md` — agent onboarding: install the skill, the three things agents get wrong.
- `docs/configuration/tunables.md` — **all 157 tunables, 31 safe to optimize.** Generated from `percept.tunables.REGISTRY`.
- `docs/judges/registries.md` — **backend seams, detectors, profile registries**, and how to register your own. Generated.
- `docs/configuration/overview.md` — the `Settings` tree + resolver precedence; `docs/configuration/profiles.md` — author a profile.
- `docs/resources/cli.md` — all nine console scripts and their flags.
- `docs/watches/creating-a-watch.md` — the `run()` API and event shape.
- `docs/judges/custom-specialist-judge.md` — bring your own vision backend or a specialist judge (with a conformance suite).
- `docs/evaluation/evaluate-your-data.md` · `docs/configuration/optimize.md` · `docs/evaluation/benchmarks.md` — the offline quality rail.
- `docs/demo/README.md` — the browser harness; simulation mode vs BYOK.
- `.claude/skills/percept-agent/` — a Claude Code skill that scaffolds an agent from a plain-language goal (kept in sync with `.agents/skills/percept-agent/`).
- `percept.edge` / `percept.harness` — the server-side edge (tier-0 gate + detector registry)
  for Python deployments.

## Where things live (so you put new files in the right place)
- `docs/` — **public only.** Adoption material that feeds the website. No ADRs, no internal notes.
- `.adr/` — architecture decision records. Tracked (six drift-guard tests read them), but not adoption docs.
- `scratch/`, `execution-docs/`, `codewiki/`, `.dispatch-board/` — gitignored local working areas. All
  strategy, specs, plans, and probe results live here. Never shipped.
- Regenerate generated docs with `make docs`; gate them with `make docs-check docs-links`.
