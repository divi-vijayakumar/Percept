# percept

**The open-source cognition layer for goal-driven, proactive vision agents.**

Repository development version: `percept-vision 1.0.0` (**unreleased**) · latest public release:
`0.7.0` · Apache-2.0 · Python ≥ 3.10 · docs & demos:
**[percept.percia.ai](https://percept.percia.ai)**

---

## The problem

Point a vision-language model at a live camera and you get one of two bad outcomes.

**Send every frame** and you pay a frontier model to look at an empty kitchen 3,600 times an hour.
The cost scales with wall-clock time, not with anything happening.

**Send frames on a timer** and the model has no memory between calls. It cannot tell you *"the
kettle just finished"* — only *"there is a kettle."* Every frame is judged from scratch, so a
condition that stays true fires over and over, and one event arrives as forty notifications.

Both fail the same way. A VLM answers *"what is in this image?"* But a proactive agent has to answer
**"has the thing I'm waiting for just become true?"** That is a question about *change over time*,
about *which* object this is compared to the one a minute ago, and about *when to stay silent* —
none of which a stateless per-frame call can express.

## The solution

Percept keeps a cheap, stateful **System 1** in the loop the whole time — senses, a three-state
gate, entity memory — and summons a frontier VLM as **System 2** only when the cheaper tiers cannot
resolve the goal. A still scene costs nothing. The model is called only when it must be.

You state a goal in plain language. Percept turns the stream into an agent that:

- **fires once, on the rising edge** — when the condition *becomes* true, not every frame it stays true;
- **reasons over entities across time** — *this* mug, still the same one, thirty seconds later;
- **refuses to guess** — a three-state gate maps *known → act · not → silent · **unknown → ask***.

That last one is the point. An `ask` is the gate declining to fabricate confidence it does not have.
Most systems collapse it into a yes or a no; Percept surfaces it as a question to a human.

Frontier models sit behind vendor-neutral seams, and the core is **pure standard library** — it
installs and runs with no keys and no network.

## Getting started

### Watch something in 30 seconds

```bash
pip install "percept-vision[runtime,openrouter]"   # capture (OpenCV) + a cheap, fast hosted judge
export OPENROUTER_API_KEY=sk-or-...
percept-watch --goal "someone raises their hand" --goal "a package is on the porch"
#   --source webcam:0 | clip.mp4 | rtsp://…   --egomotion (glasses/body-cam)   --fps 1  --refractory 15
```

Point it at a webcam, a video file, or an RTSP camera. Get a `🔔` once per event.

### Install

```bash
pip install percept-vision                          # core: PURE STDLIB — offline, no keys, no network
pip install "percept-vision[runtime,openrouter]"    # + capture + a hosted judge, for a live watch
```

**The package is `percept-vision`; the import is `import percept`.** Say it out loud once and you
will not lose an hour to it.

Those two lines cover almost everyone. Every optional extra — `[gemini]`, `[claude]`, `[deepgram]`,
`[mic]`, `[edge]`, `[bench]`, `[pose]`, `[otel]`, `[all]` — with what it pulls in and
which key it needs, is in **[docs/judges/backends.md](docs/judges/backends.md)**.

### 60-second quickstart — no keys, fully offline

This script runs **as-is**:

```python
import asyncio
from percept import Percept, Goal

async def main():
    # mode="offline" selects scripted backends — keyless, deterministic. It proves WIRING,
    # not intelligence. You ask for it OUT LOUD; you never fall onto it by omission.
    agent = Percept.create(mode="offline", discover_plugins=False)

    agent.add_goal(Goal(
        id="caffeine",
        condition="the user is drinking coffee",
        say="Heads up — stepping back from caffeine?",
    ))

    # Each frame is judged; the gate fires ONCE on the rising edge.
    for ev in await agent.perceive_judged("sip-coffee"):
        print(ev.action, ev.goal_id, ev.text)   # -> fire caffeine Heads up — …

    # The same frame again does NOT re-fire. Rising-edge, not level-triggered.
    print(await agent.perceive_judged("sip-coffee"))   # -> []

asyncio.run(main())
```

Swap the scripted backend for real eyes — the cognition layer is unchanged:

```python
agent = Percept.create(vision="gemini")     # percept-vision[gemini] + GEMINI_API_KEY
```

### Watch a live source

For a real watch agent you don't write a capture loop — **`percept.run` owns it**: it opens the
source, runs the cheap tier-0 gate so a still scene spends no cloud calls, judges, and yields an
`Incident` on each rising-edge fire. One `async for` is the whole integration surface.

- **[Quickstart](docs/get-started/quickstart.md)** — the guided path: offline → a real backend → a live
  source, with what each step is actually doing.
- **[Runtime worker](docs/watches/creating-a-watch.md)** — everything for embedding it: sources, sinks
  (`on_incident` → Slack/webhook), refusals (`on_ask`), fused audio, the decision trace, moving
  cameras, cost, and a troubleshooting table.

### See it run in a browser

```bash
python docs/demo/serve.py        # http://localhost:8788
```

**Demo 04 — live stream + replay** watches a live feed being judged as it happens, then replays
every decision on the media timeline. It drives the **real** runtime: every gate decision and
`.ptrace` is genuine output, not a mock. With no API key it runs a bundled fixture (scripted
verdicts, real machinery); set a key to judge your own media. Details in
**[docs/demo/README.md](docs/demo/README.md)**.

## Build with a coding agent

Percept is built to be adopted by an agent as easily as by a person.

```bash
cp -R .claude/skills/percept-agent ~/.claude/skills/     # Claude Code
```

Then just ask: *"watch my webcam and tell me when someone's at the door."* The skill carries the
verified API, the templates, and the traps. Codex and other harnesses read
`.agents/skills/percept-agent/`, and `AGENTS.md` + [`llms.txt`](llms.txt) are picked up
automatically at the repo root.

Full guide: **[docs/get-started/coding-agents.md](docs/get-started/coding-agents.md)**.

## Architecture

The diagram is **generated from the code** by
[`tools/build_architecture.py`](tools/build_architecture.py) — a stale code-pointer or a false
integration marker fails the build, so the picture cannot drift from what ships.

![Percept architecture — four tiers, generated from code](docs/architecture.svg)

Four tiers, signal flowing up: **Feeds** (`percept-edge` devices) → **Measures** (`percept.harness`
detectors/pose) → **Cognition / System 1** (`percept` — senses · the three-state gate · routing ·
cache; cheap, deterministic, *always in the loop*) → **VLM / System 2** (`percept.backends` —
summoned behind vendor-neutral registry seams, paid per call).

*Prose map: [docs/resources/architecture.md](docs/resources/architecture.md) · interactive:
[architecture.html](docs/architecture.html).*

## What you can configure

Sensible defaults ship. Percept declares **157 tunables**, of which **31 are safe to optimize** —
the only ones `percept-optimize` may search. A few worth knowing:

| what | env var | default | note |
|---|---|---|---|
| confidence to fire | `PERCEPT_GATE_THRESHOLD` | `0.7` | below it, the gate **asks** instead of guessing |
| suppress repeat fires | `PERCEPT_GATE_REFRACTORY_S` | `0.0` | set `15` for one notification per episode |
| tier-0 salience statistic | `PERCEPT_SALIENCE_STAT` | `mean` | set `block_max` for small/localized actions on a fixed camera |
| moving-camera compensation | `PERCEPT_EGO_COMPENSATE` | `false` | set `true` for glasses/body-cam |

- **[Every tunable](docs/configuration/tunables.md)** — all 157, typed, owned, with bounds. Generated from the registry.
- **[Registries and adapters](docs/judges/registries.md)** — the 7 backend seams, the detectors, and how to add your own.
- **[Configuration reference](docs/configuration/overview.md)** — the `Settings` tree and resolver precedence.
- **[Author a profile](docs/configuration/profiles.md)** — a `SourceProfile` / `WatchProfile` for your use case.

## Judges (and bringing your own)

An admitted frame falls through three tiers of rising cost: a local regex **route**, a near-frame
**cache**, and only on a miss the **judge** — a vision model deciding *"is `<goal>` true?"* The
judge is **fail-closed**: a request for a specialist judge is authorized against a deployment
catalog, bound to an immutable SHA-256-verified static context, and only then dispatched. Any
mismatch refuses the arm rather than silently falling back.
→ **[docs/judges/custom-specialist-judge.md](docs/judges/custom-specialist-judge.md)**

## Tune, evaluate, benchmark

An offline quality rail — deterministic, model-free where possible, statistically disciplined:

```bash
percept-optimize --fixtures ./fixtures --space space.json --bars bars.json --out-lock lock.json
percept-eval     --baseline base.json --candidate cand.json --fixtures ./fixtures --bars bars.json
percept-bench    eval/benchmark_ci_gate/manifest.json --out benchmark.json     # per-concern P/R
```

**No Percept accuracy numbers are published yet.** The methodology and runbook exist; the
reproducible model-sweep run is a separate phase. See **[docs/evaluation/](docs/evaluation/benchmarks.md)**
for the status and the gate a published claim must pass. Every command has a full flag reference in
[docs/resources/cli.md](docs/resources/cli.md).

## The envelope (refusals, stated proudly)

- **Assistant-class, never the safety mechanism.** percept informs a human who stays responsible. It is
  never the thing standing between a person and harm on a clock it can't guarantee. *Out:*
  driver-drowsiness intervention. *In:* a post-hoc driving debrief.
- **Watches the user's own world, with the user as beneficiary** — never a non-consenting third party to
  the wearer's advantage. *Out:* covert analysis of someone you're negotiating with; real-time
  assistance during live casino play. *In:* a play-money practice trainer.

These refusals are a feature. **Do not lead with surveillance demos.**

## Packages

| package | what | install |
|---|---|---|
| **percept-vision** | the Python SDK: cognition core, scripted doubles, runtime worker, Python edge/harness, benchmark tooling | `pip install percept-vision` |
| **percept-edge** | on-device reactive edge in JS/WASM: VAD + motion gate over the same wire contract | `npm install percept-edge` |

> **Versioning.** This repository currently carries an **unreleased** `percept-vision 1.0.0`
> development version. The latest public release is `0.7.0`; do not describe `1.0.0` as shipped or
> stable. The **wire contract** versions separately — `CONTRACT_VERSION = "1.10.0"`
> (additive-only).

### What you can rely on

**Frozen** — locked; a breaking change means a new surface. **Stable** — may grow, won't break within
`1.x`. **Experimental** — usable, may reshape within `1.x`. **Internal** — don't depend on it.

| surface | status |
|---|---|
| `watch()` + `WATCH_VERSION` (`percept/watch.py`) | **Frozen** |
| wire contract `CONTRACT_VERSION` (`percept/contracts`) | **Stable** (additive-only) |
| `percept.run()` / `RuntimeSession` / `Incident` | **Stable** |
| `percept-watch` CLI + flags | **Stable** |
| `Percept.create()` / `Percept.configure()`, backend registry, conformance suite | **Stable** |
| `WatchProfile` / `SourceProfile`, profile resolution, episode memory, trace telemetry | **Experimental** |

When in doubt: pin a version, and treat anything not listed as **Internal**.

## Documentation

Full index: **[docs/README.md](docs/README.md)** — generated, every page dated. Online at
**[percept.percia.ai](https://percept.percia.ai)**.

| I want to… | go to |
|---|---|
| run something that works | [Quickstart](docs/get-started/quickstart.md) |
| embed it in my app | [Runtime worker](docs/watches/creating-a-watch.md) |
| make it fire on *my* content | [Tune admission](docs/admission/tuning-admission.md) |
| bring my own model | [Custom judge & backend](docs/judges/custom-specialist-judge.md) |
| see every knob | [Tunables](docs/configuration/tunables.md) |
| understand the design | [Architecture](docs/resources/architecture.md) · [Concepts](docs/resources/how-it-works.md) |
| measure it | [Benchmarks](docs/evaluation/benchmarks.md) |
| build with an agent | [Coding agents](docs/get-started/coding-agents.md) |

## Community & support

- **Hosted Percept** — managed runtime, no infrastructure to run: **[percept.percia.ai](https://percept.percia.ai)**
- **Discord** — questions, showcases, help: **[discord.gg/E2jC4bbzp](https://discord.gg/E2jC4bbzp)**
- **Issues & contributions** — [CONTRIBUTING.md](CONTRIBUTING.md) · [SECURITY.md](SECURITY.md)

## Layout

```
packages/percept-vision/    the SDK (pip install percept-vision; import percept)
                            ├─ percept.harness    bundled Python tier-0 edge + detector skills
                            └─ percept.contracts  bundled frozen wire-contract (DTOs + Protocols)
packages/percept-edge/      on-device JS/WASM edge (npm)
docs/                       documentation + the generated architecture.svg/html + the demo harness
examples/                   short, runnable snippets
eval/                       golden corpus + regression gate CI template
tools/                      architecture, tunables, registries, and docs-index generators
.adr/                       architecture decision records (internal reasoning, not adoption docs)
```

## Status

**v1.0.0 — unreleased development candidate.** The cognition core, runtime, configuration,
fail-closed judge, and offline test rail exist, but the end-to-end qualification is currently a
no-go and no supporting benchmark has been published. The latest public package remains `0.7.0`.

Sustained-duration Watches are supported by the in-process Python runtime only. They are
session-scoped, remote `percept-edge` compilation fails closed, and
`facility_observation@1/door_left_open` carries no quality claim until a matched duration evaluation
ships. Read the [release contract and migration note](docs/watches/sustained-duration-watches.md).

## License

Apache-2.0.
