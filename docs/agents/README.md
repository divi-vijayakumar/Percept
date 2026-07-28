---
title: Coding agents
updated: 2026-07-28
applies_to: percept-vision 1.0.0
status: stable
---
# Building with Percept from a coding agent

Percept is designed to be adopted by an agent as easily as by a person. Everything an agent needs to
write correct Percept code — the verified API surface, the naming traps, the frame formats — is in
the repo as machine-readable text. Point your agent at it and it will not have to guess.

## The three files an agent should read

| file | what it gives you |
|---|---|
| [`llms.txt`](../../llms.txt) | the whole-repo map, in the `llms.txt` convention — start here |
| [`AGENTS.md`](../../AGENTS.md) | the verified integration guide: mental model, minimal agent, API signatures, gotchas |
| [`docs/README.md`](../README.md) | the full documentation index |

Most agent harnesses (Codex, Cursor, Copilot, Claude Code) pick up `AGENTS.md` automatically when it
sits at the repo root. It is there.

## Install the `percept-agent` skill

The repo ships a skill that teaches an agent to scaffold a watch agent end to end — CLI path, the
`run()` API, a live-source template, and the refusals to respect.

```
.claude/skills/percept-agent/     # Claude Code
.agents/skills/percept-agent/     # Codex, and other agents reading .agents/
```

Both directories carry the same three files: `SKILL.md`, `template_watch_agent.py`, and
`template_watch_agent_live.py`.

**Claude Code** — copy the skill into your own project (or install it globally):

```bash
mkdir -p ~/.claude/skills
cp -R .claude/skills/percept-agent ~/.claude/skills/
```

Then ask for a watch agent in plain language — *"watch my webcam and tell me when someone's at the
door"* — and the skill activates on its own.

**Other agents** — copy `.agents/skills/percept-agent/` into whatever directory your harness reads,
or just paste `SKILL.md` into context. It is self-contained and provider-neutral.

> These two directories are **duplicated, not symlinked**. If you change one, change both — the
> header comment in each `SKILL.md` says so, and it is the kind of drift a reviewer misses.

## The three things agents get wrong

Give these to your agent up front; they cost more debugging time than anything else in the SDK.

**1. The package name is not the import name.**

```python
# pip install percept-vision   →   import percept
import percept                 # ✅
import percept_vision          # ❌ does not exist
```

**2. `ask` is not `fire`.** `FireEvent.action` is `"fire"` or `"ask"`. An `ask` is the gate
*refusing to guess* because confidence fell below threshold. Treat it as a question to a human, never
as a positive detection. Collapsing the two silently destroys the property the gate exists to provide.

**3. Keyless runs must be asked for out loud.** `mode="offline"` selects scripted backends —
deterministic, no API key, and they prove **wiring, not intelligence**. You never fall onto a
scripted double by omission; a live backend with a missing key prints one actionable line instead.

```python
agent = percept.Percept.create(mode="offline", discover_plugins=False)
```

## The verified minimal agent

This runs as-is, with no keys:

```python
import asyncio
from percept import Percept, Goal

async def main():
    agent = Percept.create(mode="offline", discover_plugins=False)
    agent.add_goal(Goal(
        id="kettle",
        condition="the kettle has finished boiling",
        say="The kettle is done.",
    ))
    for ev in await agent.perceive_judged("stopped"):
        print(ev.action, ev.goal_id, ev.text)

asyncio.run(main())
```

Swap in a real judge with one keyword — the cognition layer is unchanged:

```python
agent = Percept.create(vision="gemini")     # percept-vision[gemini] + GEMINI_API_KEY
```

## What an agent should not do

Percept ships refusals on purpose, and an agent building on it inherits them:

- **Assistant-class, never the safety mechanism.** Percept informs a human who stays responsible. It
  is never the thing standing between a person and harm on a clock it cannot guarantee.
- **Watches the user's own world, with the user as the beneficiary** — never a non-consenting third
  party to the operator's advantage.

Do not generate surveillance demos. The full envelope is in the
[README](../../README.md#the-envelope-refusals-stated-proudly).

## Related

- [Quickstart](../tutorials/quickstart.md) — the 60-second offline first run.
- [Runtime worker](../how-to/runtime.md) — the `run()` API and event shape.
- [Registries and adapters](../reference/registries.md) — bring your own backend or detector.
- [`watch()` contract](../reference/watch-contract.md) — the frozen entrypoint.
