# percept

The open-source **cognition layer** for goal-driven, proactive vision agents.

**percept-vision 1.0.0** · docs & demos: **[percept.percia.ai](https://percept.percia.ai)** · Apache-2.0

You state a goal in plain language — *"nudge me when I drink a coffee"*, *"tell me when the kettle
boils"* — and percept turns a live audio-visual stream into an agent that **reasons over entities
across time**, **fires only on the rising edge** of a condition becoming true, and **refuses to
guess**: a three-state gate maps *known → act · not → silent · unknown → ask*.

percept keeps a cheap, stateful **System 1** (senses, the gate, entity memory) in the loop the whole
time and summons a frontier VLM as **System 2** only when the cheaper tiers can't resolve the goal —
so a still scene costs nothing, and the model is called only when it must be. Frontier models sit
behind vendor-neutral seams; the core is pure standard library.

## Watch something in 30 seconds (the live runtime)

```bash
pip install "percept-vision[runtime,openrouter]"     # capture (OpenCV) + a cheap, fast hosted judge (Qwen)
export OPENROUTER_API_KEY=sk-or-...
percept-watch --goal "someone raises their hand" --goal "a package is on the porch"
#   --source webcam:0 | clip.mp4 | rtsp://…    --egomotion (glasses/body-cam)    --fps 1  --refractory 15
```

Point it at a webcam, a video, or an RTSP camera; give it goals in plain English; get a `🔔` once per
event. Embedding it in your own app? → **[docs/integration.md](docs/integration.md)** (the `run()` API,
the event shape, recipes). Moving camera (smart glasses)? add `--egomotion`.

## Install

```bash
pip install percept-vision        # core: PURE STDLIB — runs offline with fake backends, no keys
```

**The package is `percept-vision`; the import is `import percept`.** (Name ≠ import — say it out loud.)
The core has **zero dependencies** and runs with deterministic fake backends, so `pip install` then run
works with no API keys. Python **≥ 3.10**. Frontier backends and capture are opt-in extras:

```bash
pip install "percept-vision[runtime]"    # camera/video/RTSP capture worker (OpenCV)
pip install "percept-vision[gemini]"     # GeminiVision            (GEMINI_API_KEY)
pip install "percept-vision[claude]"     # AnthropicVision / Claude (ANTHROPIC_API_KEY)
pip install "percept-vision[openrouter]" # OpenRouter (e.g. Qwen)  (OPENROUTER_API_KEY)
pip install "percept-vision[deepgram]"   # Deepgram STT + TTS      (DEEPGRAM_API_KEY)
pip install "percept-vision[screen]"     # screen-capture source (mss)
pip install "percept-vision[mic]"        # microphone capture for fused audio+video
pip install "percept-vision[bench]"      # benchmark + scorecard tooling (from a repo checkout)
pip install "percept-vision[edge]"       # Python tier-0 gate (parity with the percept-edge JS package)
pip install "percept-vision[all]"        # everything
```

## 60-second quickstart (no keys)

Fully offline — fake backends, deterministic, nothing to configure. This script runs **as-is**:

```python
import asyncio
from percept import Percept, Goal

async def main():
    # mode="offline" selects scripted backends — keyless, deterministic; proves WIRING, not
    # intelligence. (You ask for scripted OUT LOUD; you never fall onto it by omission.)
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

Backends are selected by name (`"scripted"` (canonical) · `"gemini"` · `"claude"` · `"openrouter"`),
by adapter instance, or by env (`PERCEPT_VISION_BACKEND`). Keyless runs are **`mode="offline"`**
(scripted backends, announced out loud) — you never fall onto scripted doubles by omission; a live
backend with no key prints a one-line instruction, never a stack trace. See
[`templates/`](templates/) for copyable reference applications.

## Watch a live source

For a real watch agent you don't write a capture loop — **`percept.run` owns it**: it opens the source,
runs the cheap tier-0 motion gate (a still scene spends no cloud calls), judges, and **yields an
`Incident` on each rising-edge fire**. `pip install "percept-vision[gemini,runtime]"` + a `GEMINI_API_KEY`:

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

Full options — sinks (`on_incident`), refusals (`on_ask`), fused audio (`audio_source="mic:0"`), the
decision trace (`on_trace`), and gate tuning: [`docs/how-to/runtime.md`](docs/how-to/runtime.md).

## Set up a custom run

Build the core yourself when you want to choose backends, apply a profile, or drive delivery: construct
with `Percept.create(...)` (or the resolver door `Percept.configure(...)`), hand it to `run(core,
source=...)`, and wire your own sinks (`on_incident` → Slack/webhook/notification). To *drive* a running
watch — arm goals by voice, ask "what do you see?", pause/mute — wrap it in a `RuntimeSession` (the
command seam). See [`docs/how-to/runtime.md`](docs/how-to/runtime.md) and
[`docs/reference/command-seam.md`](docs/reference/command-seam.md).

## Architecture

The diagram below is **generated from the code** by
[`tools/build_architecture.py`](tools/build_architecture.py) — a stale code-pointer or a false
integration marker fails the build, so the picture can never drift from what ships.

![Percept architecture — four tiers, generated from code](docs/architecture.svg)

*Interactive version: [`architecture.html`](architecture.html) · the prose map:
[`docs/explanation/architecture.md`](docs/explanation/architecture.md).*

Four tiers, signal flowing up: **Feeds** (`percept-edge` devices) → **Measures** (`percept.harness`
detectors/pose) → **Cognition / System 1** (`percept` — senses · the three-state gate · routing · cache;
cheap, deterministic, *always in the loop*) → **VLM / System 2** (`percept.backends` — summoned behind
vendor-neutral registry seams, paid per call). A raw VLM-in-a-loop collapses System 1 and System 2 into
one expensive call every frame; percept keeps them apart so the common case costs nothing. Deep
per-module reference: the [CodeWiki](codewiki/) (generated from source).

## What you can configure

Sensible defaults ship; the knobs that matter are env vars you can set before `import percept`, or pass
through a profile. A few worth knowing:

| what | env var | default | note |
|---|---|---|---|
| confidence to fire | `PERCEPT_GATE_THRESHOLD` | `0.7` | below it, the gate **asks** instead of guessing |
| suppress repeat fires | `PERCEPT_GATE_REFRACTORY_S` | `0.0` | set `15` for one notification per episode |
| tier-0 salience statistic | `PERCEPT_SALIENCE_STAT` | `mean` | set `block_max` for small/localized actions on a fixed camera |
| moving-camera compensation | `PERCEPT_EGO_COMPENSATE` | `false` | set `true` for glasses/body-cam |
| judge model id | `PERCEPT_VISION_MODEL` | `claude-opus-4-8` | per-provider model id |

The full `Settings` tree, every env var, and the `SourceProfile` / `WatchProfile` system are in
**[docs/configuration.md](docs/configuration.md)**. Author your own profile for a use case:
**[docs/how-to/custom-configuration.md](docs/how-to/custom-configuration.md)**.

## Judges (and bringing your own)

An admitted frame falls through three tiers of rising cost: a local regex **route**, a near-frame
**cache**, and only on a miss the **judge** — a vision model invoked to decide *"is `<goal>` true?"*.
The judge is **fail-closed**: a request for a specialist judge is authorized against a deployment
catalog, bound to an immutable **SHA-256-verified** static context, and only then dispatched; any
mismatch refuses the arm rather than silently falling back. Bring your own vision backend or author a
custom specialist judge (with a runnable conformance suite):
**[docs/how-to/custom-judge.md](docs/how-to/custom-judge.md)**.

## Tune, evaluate, benchmark

An offline quality rail — deterministic, model-free where possible, statistically disciplined:

```bash
percept-optimize --fixtures ./fixtures --space space.json --bars bars.json --out-lock lock.json
percept-eval     --baseline base.json --candidate cand.json --fixtures ./fixtures --bars bars.json
percept-bench    eval/benchmark_ci_gate/manifest.json --out benchmark.json     # per-concern P/R
```

- **Optimize** — search safe tunables → a validated `OptimizationLock`: [`docs/how-to/optimize.md`](docs/how-to/optimize.md).
- **Eval** — per-concern precision/recall on your own labelled manifest, and paired A/B differentials:
  [`docs/how-to/eval.md`](docs/how-to/eval.md).
- **Benchmarks** — the scorecard, the audit gate, and the reproducible model-sweep runbook:
  [`docs/how-to/benchmarks.md`](docs/how-to/benchmarks.md).

Every command has a full flag reference in [`docs/reference/cli.md`](docs/reference/cli.md).

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
| **percept-vision** | the Python SDK: cognition core, fakes, runtime worker, Python edge/harness, benchmark tooling (extras). | `pip install percept-vision` |
| **percept-edge** | on-device reactive edge in JS/WASM: VAD + motion gate over the same WatchSpec/Tier0Signal wire-contract. | `npm install percept-edge` |

The tier-0 edge harness and the frozen wire-contract are bundled inside `percept-vision` as the
`percept.harness` and `percept.contracts` subpackages — so OSS users install **one** Python package
(heavy deps are opt-in extras) plus the optional npm edge package.

> **Versioning.** The package version is `percept-vision 1.0.0`. The **wire-contract** version is
> separate — `CONTRACT_VERSION = "1.10.0"` (additive-only). A `1.0.0` install exporting contract
> `1.10.0` is expected, not a bug.

### What can you rely on?

**Frozen** — the contract is locked; a breaking change means a new surface, not a change to this one.
**Stable** — may grow (additive) but won't break existing callers within `1.x`.
**Experimental** — usable today, but may change shape or be renamed within `1.x` without a deprecation cycle.
**Internal** — an implementation detail; not part of the public surface, don't depend on it directly.

| surface | status | notes |
|---|---|---|
| `watch()` + `WATCH_VERSION` (`percept/watch.py`) | **Frozen** | the one seam every downstream capability is built to; `WATCH_VERSION = "1.1.0"`. |
| wire contract `CONTRACT_VERSION` (`percept/contracts`) | **Stable** | `1.10.0`, additive-only — new fields, never breaking ones. |
| `percept.run()` / `percept.RuntimeSession` / `Incident` | **Stable** | the batteries-included runtime front door + command seam. |
| `percept-watch` CLI + flags (`--source`, `--goal`, `--say`, `--vision`, `--model`, `--fps`, `--refractory`, `--egomotion`, `--offline`) | **Stable** | the pip-installable quickstart path above. |
| `Percept.create()` / `Percept.configure()`, backends registry, conformance suite | **Stable** | the extension seams (bring your own backend / judge / profile). |
| `WatchProfile` / `SourceProfile` + registries, profile resolution, episode memory, trace telemetry | **Experimental** | newer cognition-layer surfaces; useful now, most likely to reshape within `1.x`. |

When in doubt: pin a version, and treat anything not listed here as **Internal**.

## Benchmark status

**No Percept accuracy numbers are published yet.** The benchmark is designed but the reproducible
model-sweep run is a separate, later phase. What exists today is the *methodology* and the *runbook*:
[`docs/reference/benchmark-methodology.md`](docs/reference/benchmark-methodology.md) and
[`docs/reference/publishable-benchmark.md`](docs/reference/publishable-benchmark.md). Figures in those
docs are external reference bars or illustrative examples — not Percept claims. The benchmark harness
and the golden set are **repo/dev tooling, not packaged in the wheel** (the wheel is just `import
percept`), so benchmark commands run from a source checkout:

```bash
# from a repo checkout (the [bench] extra installs numpy/opencv; --manifest can be your own clips):
pip install -e "packages/percept-vision[bench]"
PYTHONPATH=packages/percept-vision percept-bench scorecard \
    --manifest eval/golden-v1/MANIFEST.json --out out/scorecard.txt
```

## Docs

Full index: **[`docs/README.md`](docs/README.md)** (organized by Diátaxis) · online at
**[percept.percia.ai](https://percept.percia.ai)**. Highlights:

- **New here?** [`docs/tutorials/choose-your-path.md`](docs/tutorials/choose-your-path.md) — the narrow
  first path (pick one row, ignore the rest).
- **Quickstart** — [`docs/tutorials/quickstart.md`](docs/tutorials/quickstart.md) (the snippet above).
- **Architecture** — [`docs/explanation/architecture.md`](docs/explanation/architecture.md): the four
  tiers, the two-layer brain, the firing paths. Then
  [`docs/explanation/concepts.md`](docs/explanation/concepts.md) (the cognition/perception split) and
  [`docs/explanation/e2e-flow-milk-boilover.md`](docs/explanation/e2e-flow-milk-boilover.md) (a worked
  judged flow).
- **Runtime** — [`docs/how-to/runtime.md`](docs/how-to/runtime.md): `run(source) → incidents`, and custom runs.
- **Configuration** — [`docs/configuration.md`](docs/configuration.md) (everything tunable) and
  [`docs/how-to/custom-configuration.md`](docs/how-to/custom-configuration.md) (author a profile).
- **Backends & judges** — [`docs/reference/backends.md`](docs/reference/backends.md) (the seams) and
  [`docs/how-to/custom-judge.md`](docs/how-to/custom-judge.md) (bring your own).
- **Evaluate / optimize / benchmark** — [`docs/how-to/eval.md`](docs/how-to/eval.md),
  [`docs/how-to/optimize.md`](docs/how-to/optimize.md), [`docs/how-to/benchmarks.md`](docs/how-to/benchmarks.md).
- **CLI reference** — [`docs/reference/cli.md`](docs/reference/cli.md) (all nine commands).
- **AI-agent map** — [`llms.txt`](llms.txt) and [`AGENTS.md`](AGENTS.md).

## Layout

```
packages/percept-vision/    the SDK (pip install percept-vision; import percept)
                            ├─ percept.harness    bundled Python tier-0 edge + detector skills
                            └─ percept.contracts  bundled frozen wire-contract (DTOs + Protocols)
packages/percept-edge/      percept-edge — on-device JS/WASM edge (npm)
docs/                       Diátaxis docs + the generated architecture.svg
codewiki/                   generated per-module reference (source of truth after the code)
templates/                  copyable reference applications (e.g., realtime voice watch agent)
eval/                       golden corpus + regression gate CI template
tools/build_architecture.py the architecture-diagram generator
Makefile                    test · parity · architecture · check · eval-live · e2e
```

## Status

**v1.0.0 — first stable release.** The cognition core, the batteries-included runtime, layered
configuration, the fail-closed judge, and the offline quality rail run and are tested; frontier
backends (Gemini, Claude, OpenRouter, Deepgram) and the edge packages are wired behind their seams. No
benchmark numbers are published yet (see [Benchmark status](#benchmark-status)). Issues and
contributions welcome ([CONTRIBUTING.md](CONTRIBUTING.md)).

Sustained-duration Watches are supported by the in-process Python runtime only. They are
session-scoped, remote `percept-edge` compilation fails closed, and
`facility_observation@1/door_left_open` carries no quality claim until a matched duration evaluation
ships. Read the [release contract and migration note](docs/reference/sustained-duration-watches.md).

## License

Apache-2.0.
