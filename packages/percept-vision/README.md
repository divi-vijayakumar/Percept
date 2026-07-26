# percept-vision

The open-source **cognition layer** for goal-driven, proactive vision agents.
**Package: `percept-vision` · import: `import percept`** (name ≠ import).

**percept-vision 1.0.0** · docs & demos: **[percept.percia.ai](https://percept.percia.ai)** · Apache-2.0

You state a goal in plain language — *"nudge me when I drink a coffee"*, *"tell me when the kettle
boils"* — and percept turns a live audio-visual stream into an agent that **reasons over entities
across time**, **fires only on the rising edge** of a condition becoming true, and **refuses to
guess** — a three-state gate maps *known → act · not → silent · unknown → ask*.

percept keeps a cheap, stateful **System 1** (senses, the gate, entity memory) in the loop the whole
time and summons a frontier VLM as **System 2** only when the cheaper tiers can't resolve the goal —
so a still scene costs nothing. Frontier models sit behind vendor-neutral seams; the core is pure
standard library.

## Install

```bash
pip install percept-vision        # core: PURE STDLIB — runs offline with fake backends, no keys
```

The core has **zero dependencies** and runs with deterministic fake backends, so `pip install` then run
works with no API keys. Python **≥ 3.10**. Capture and frontier backends are opt-in extras:

```bash
pip install "percept-vision[runtime]"    # camera/video/RTSP capture worker (OpenCV)
pip install "percept-vision[gemini]"     # GeminiVision             (GEMINI_API_KEY)
pip install "percept-vision[claude]"     # AnthropicVision / Claude  (ANTHROPIC_API_KEY)
pip install "percept-vision[openrouter]" # OpenRouter (e.g. Qwen)   (OPENROUTER_API_KEY)
pip install "percept-vision[deepgram]"   # Deepgram STT + TTS        (DEEPGRAM_API_KEY)
pip install "percept-vision[bench]"      # benchmark + scorecard tooling (from a repo checkout)
pip install "percept-vision[edge]"       # Python tier-0 gate + detector registry
pip install "percept-vision[all]"        # everything
```

## 60-second quickstart (no keys)

Fully offline — fake backends, deterministic, nothing to configure. This script runs **as-is**:

```python
import asyncio
from percept import Percept, Goal

async def main():
    # mode="offline" selects scripted backends — keyless, deterministic. (A bare live create()
    # raises ModeError rather than silently faking; you ask for scripted OUT LOUD.)
    agent = Percept.create(mode="offline", discover_plugins=False)

    agent.add_goal(Goal(
        id="caffeine",
        condition="the user is drinking coffee",
        say="Heads up — stepping back from caffeine?",
    ))

    # Each frame is judged; the gate fires ONCE on the rising edge.
    # ("sip-coffee" is a token the fake vision backend scripts as a confident YES.)
    fires = await agent.perceive_judged("sip-coffee")
    for ev in fires:                       # ev is a FireEvent
        print(ev.action, ev.goal_id, ev.text)   # -> fire caffeine Heads up — stepping back from caffeine?

    # The same frame again does NOT re-fire — rising-edge, not level-triggered.
    print(await agent.perceive_judged("sip-coffee"))   # -> []

asyncio.run(main())
```

`perceive_judged(frame)` returns a list of `FireEvent(goal_id, action, text, entity_id, verdict)`,
where `action` is `"fire"` or `"ask"`. Swap the fake for real eyes — the cognition layer is unchanged:

```python
agent = Percept.create(vision="gemini")          # needs percept-vision[gemini] + GEMINI_API_KEY
```

Backends are selected by name (`"scripted"` (canonical) · `"gemini"` · `"claude"` · `"openrouter"`), by
adapter instance, or by env (`PERCEPT_VISION_BACKEND`).

## Official free packs

`physical_scene_starter@1` and `ego_motion_starter@1` ship as mechanics-only DomainPacks. They render
portable `WatchPlan` values and never auto-arm a runtime. Both deliberately declare empty
qualification fixtures and quality bars: availability is not a claim about real-media accuracy,
provider latency, or cost. See the
[DomainPack reference](https://github.com/divi-vijayakumar/Percept/blob/main/docs/reference/domain-packs.md)
for the template names and explicit identity/tracking/reacquisition boundaries.

## Watch a live source

For a real watch agent you don't write a capture loop — **`percept.run` owns it**: it opens the source,
runs the cheap tier-0 gate (a still scene spends no cloud calls), judges, and **yields an `Incident` on
each rising-edge fire**. Needs `percept-vision[runtime]` + a backend + a key:

```python
import asyncio, percept

async def main():
    async for incident in percept.run(
        source="webcam:0",                       # "rtsp://…" URL | a video file path, too
        goals=["the milk is boiling over"],      # each string becomes a Goal
        vision="gemini",
    ):
        print(f"[FIRE] {incident.message}")

asyncio.run(main())
```

## Architecture — two layers, four tiers

percept cleanly splits the **eyes** from the **brain**: a cheap, stateful **System-1 cognition** that
reasons over a stream of small structured facts (the three-state gate, the entity graph, the concern
primitive — always in the loop), and an expensive, stateless **System-2 perception** it summons only
when it must (`vision.judge(condition, frame) → Verdict`, behind vendor-neutral registry seams). A raw
VLM-in-a-loop collapses the two; percept keeps them apart so the common case costs nothing.

Full architecture — the four tiers (Feeds → Measures → Cognition → VLM), every claim anchored to code,
plus the **generated** architecture diagram — is in the docs:
[Architecture](https://github.com/divi-vijayakumar/Percept/blob/main/docs/explanation/architecture.md)
· [Concepts](https://github.com/divi-vijayakumar/Percept/blob/main/docs/explanation/concepts.md)
· [docs index](https://github.com/divi-vijayakumar/Percept/blob/main/docs/README.md)
· [percept.percia.ai](https://percept.percia.ai).

## The envelope (refusals, stated proudly)

- **Assistant-class, never the safety mechanism.** percept informs a human who stays responsible. It is
  never the thing standing between a person and harm on a clock it can't guarantee.
- **Watches the user's own world, with the user as beneficiary** — never a non-consenting third party.

These refusals are a feature. **Do not lead with surveillance demos.**

## Packages & versioning

- **percept-vision** (this package) — the Python SDK: cognition core, fakes, runtime worker, the
  Python tier-0 edge/harness, and benchmark tooling (extras). The tier-0 harness and the frozen
  wire-contract are bundled in as the `percept.harness` and `percept.contracts` subpackages.
- **percept-edge** (npm) — the on-device reactive edge in JS/WASM: VAD + motion gate that share the
  Tier0Signal/detector wire-contract. The edge's detector + gate **outputs** are byte-for-byte
  identical to the Python harness (verified); the WatchSpec it consumes is a documented subset.

> **Versioning.** Package version `percept-vision 1.0.0`. The **wire-contract** version is separate —
> `CONTRACT_VERSION = "1.10.0"` (additive-only); the `watch()` entrypoint is `WATCH_VERSION = "1.1.0"`.
> A `1.0.0` install exporting contract `1.10.0` is expected, not a bug.

## Benchmark status

**No Percept accuracy numbers are published yet.** The benchmark is designed but the reproducible
model-sweep run is a separate, later phase — see the
[methodology](https://github.com/divi-vijayakumar/Percept/blob/main/docs/reference/benchmark-methodology.md)
and the [benchmarks how-to](https://github.com/divi-vijayakumar/Percept/blob/main/docs/how-to/benchmarks.md).
The scorecard tooling operates on a manifest you provide; the bundled golden set ships with the **source
repo, not the wheel**, so the default-manifest commands run from a repo checkout:

```bash
# from a repo checkout (the golden set is repo data, not packaged):
percept-bench scorecard --manifest eval/golden-v1/MANIFEST.json --out out/scorecard.txt
# or point --manifest at your own labelled-clip manifest, anywhere.
```

## Status

**v1.0.0 — first stable release.** The cognition core, the batteries-included runtime, layered
configuration, the fail-closed judge, and the offline quality rail run and are tested; the frontier
backends and the edge packages are wired behind their seams. No benchmark numbers are published yet.
Issues and contributions welcome:
[github.com/divi-vijayakumar/Percept](https://github.com/divi-vijayakumar/Percept).

Sustained-duration Watches are supported by the in-process Python runtime only. They are
session-scoped, remote `percept-edge` compilation fails closed, and
`facility_observation@1/door_left_open` carries no quality claim until a matched duration evaluation
ships. See the public
[release contract and migration note](https://github.com/divi-vijayakumar/Percept/blob/main/docs/reference/sustained-duration-watches.md).

## License

Apache-2.0.
