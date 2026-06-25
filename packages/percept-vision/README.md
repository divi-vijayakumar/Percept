# percept

The open-source **cognition layer** for goal-driven, proactive vision agents.

You state a goal in plain language — *"nudge me when I drink a coffee"*, *"tell me when the kettle
boils"* — and percept turns a live audio-visual stream into an agent that:

- **reasons over entities across time** — tracked things with stable ids, attributes carrying
  provenance (*observed* vs *you-told-me*) and freshness;
- **fires proactively**, but only on the **rising edge** of a condition becoming true;
- **refuses to guess** — a three-state gate maps *known → act · not → silent · unknown → ask*, so it
  never delivers a confidently-wrong nag.

The wedge is **temporal cognition** (entity memory + the three-state gate + reasoning over events),
which a raw VLM-in-a-loop and a per-frame pipeline both lack. percept builds on frontier models for
perception behind vendor-neutral seams.

> **⚠️ Research preview — v0.1.0.** Published for real-life testing and feedback, **not** for
> production. The cognition core (gate, entity graph, executor, events, scheduler, consent) runs and is
> tested; the benchmark card below is a **v0.5 DRAFT**. APIs may change between `0.1.x` releases —
> pin a version. Issues and feedback welcome. ([Status](#status).)

## The envelope (refusals, stated proudly)

- **Assistant-class, never the safety mechanism.** percept informs a human who stays responsible. It
  is never the thing standing between a person and harm on a clock it can't guarantee. *Out:*
  driver-drowsiness intervention, turn detection. *In:* a post-hoc driving debrief.
- **Watches the user's own world, with the user as beneficiary** — never a non-consenting third party
  to the wearer's advantage. *Out:* covert analysis of someone you're negotiating with; card-counting
  against a live casino. *In:* a play-money practice trainer.

These refusals are a feature. **Do not lead with surveillance demos.**

## Install

```bash
pip install percept-vision        # core: PURE STDLIB — runs offline with fake backends, no keys
```

The **core has zero dependencies** and runs with deterministic fake backends, so `pip install` then
run works with no API keys. Frontier backends are opt-in extras:

```bash
pip install "percept-vision[gemini]"     # GeminiVision
pip install "percept-vision[claude]"     # AnthropicVision (Claude)
pip install "percept-vision[deepgram]"   # Deepgram STT + TTS (voice)
pip install "percept-vision[audio]"      # numpy fast-path for acoustic/edge
pip install "percept-vision[runtime]"    # camera/video/RTSP runtime worker
pip install "percept-vision[bench]"      # benchmark + scorecard tooling
pip install "percept-vision[edge]"       # Python tier-0 gate + detector registry
pip install "percept-vision[all]"        # everything
```

Python **>= 3.10**. The package name is `percept-vision`; the import is `import percept`.

## 60-second quickstart (no keys)

Fully offline — fake backends, deterministic, nothing to configure. This script runs as-is:

```python
import asyncio
from percept import Percept, Goal

async def main():
    # Fake backends by default — offline, no keys. discover_plugins=False skips plugin lookup.
    agent = Percept.create(discover_plugins=False)

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
where `action` is `"fire"` or `"ask"`. With real eyes, swap the fake for a frontier vision backend —
the cognition layer above is unchanged:

```python
agent = Percept.create(vision="gemini")          # needs percept-vision[gemini] + GEMINI_API_KEY
# frame is now real image bytes; everything downstream (gate, graph, executor) is identical.
```

Backends are selected by name (`"fake"` · `"gemini"` · `"claude"`) or by passing an adapter instance,
and can also be set via env (`PERCEPT_VISION_BACKEND`, etc.).

## Architecture — two layers

percept cleanly splits the **eyes** from the **brain**. Perception is one stateless seam;
everything stateful and proactive is cognition.

```
┌───────────────────────── COGNITION — the "brain" ──────────────────────────┐
│  three-state GATE       fire (known-yes) · ask (unknown) · silent (known-no) │
│      ▲ rising edge: fires only on false→true, with a refractory (no nag)      │
│  EXECUTOR               one firing path (sense · key · accumulate);           │
│                         transitions A→B, counting, deadlines/absence, verify  │
│  ENTITY GRAPH           stable ids; attributes with provenance + freshness    │
│                         (observed vs you-told-me)                             │
│  EDGE DETECTOR REGISTRY opt-in cheap signals propose; the brain counts/gates  │
│  general · deterministic · cheap · STATEFUL (has memory)                      │
└──────────────────────────────────────────────────────────────────────────────┘
                          ▲  feeds on  Verdict{satisfied, confidence}
┌───────────────────────── PERCEPTION — the "eyes" ──────────────────────────┐
│  vision.judge(condition, frame) -> Verdict        ★ ONE seam                  │
│  GeminiVision / AnthropicVision · stateless · one frame · ~1s/call cloud      │
└──────────────────────────────────────────────────────────────────────────────┘
```

The **gate** turns a noisy verdict stream into at most one alert on the rising edge, and falls back
to **ASK** rather than guess when confidence is unreliable. The **executor** is the single firing
path for every concern shape (watch, transition, count, deadline/absence, verification). The
**entity graph** carries memory: stable ids and attributes that know whether they were *observed* or
*asserted by the user*, and whether they're still fresh.

The **edge detector registry** is the *edge-proposes · brain-counts · VLM-confirms* split: cheap,
opt-in detectors emit timed boundaries that the brain accumulates — counting reps without spending a
vision token per frame. Three reference skills ship as registered detectors:

- **`motion-periodicity`** — frame-diff motion peaks (rep boundaries);
- **`acoustic-onset`** — energy spikes on the existing mic stream (no new capture path);
- **`pose-openness`** — BlazePose-based rep peaks (opt-in: `percept-vision[pose]`, MediaPipe).

A full request trace — *"tell me when the milk is about to boil over"* — is in
[`docs/e2e-flow-milk-boilover.md`](docs/e2e-flow-milk-boilover.md), including an honest map of where
the perception ceiling bites.

## Packages

| package | what | install |
|---|---|---|
| **percept-vision** | the Python SDK: cognition core, fakes, runtime worker, Python edge/harness, and benchmark tooling via extras. | `pip install percept-vision` |
| **percept-edge** | on-device reactive edge in JS/WASM: VAD + motion gate over the same WatchSpec/Tier0Signal wire-contract. | `npm install percept-edge` |

The repo stays internally modular (`packages/percept-harness/`, `packages/percept-contracts/`), but
OSS users should start from one Python package plus the optional npm edge package.

Release evidence:

- Edge parity: [`../../docs/edge-parity.md`](../../docs/edge-parity.md)
- Runtime worker: [`../../docs/runtime.md`](../../docs/runtime.md)
- Benchmark/golden set: [`../../docs/eval.md`](../../docs/eval.md)
- Enterprise architecture: [`../../docs/enterprise-architecture.md`](../../docs/enterprise-architecture.md)
- Static OSS demo: [`../../examples/web-demo/index.html`](../../examples/web-demo/index.html)

## Benchmark — Percept Benchmark v1 (v0.5 DRAFT)

The benchmark holds the **backbone fixed and measures the orchestration delta** across a `raw → core
→ e2e` config ladder (no composite score — a vector of headlines). The current card is a **v0.5
DRAFT** (sampled, N≈12/track, gemini-2.5-flash) on the DeepMind **Perception Test** (CC-BY-4.0), the
private **golden** ambiguity corpus, and **RepCount-A**; the e2e-relational accuracy cell is modeled
(flagged `~`), so the card is stamped `DRAFT`.

| track | measured (DRAFT, N≈12) | reading |
|---|---|---|
| **counting** (RepCount-A) | pose OBO **0.33** vs **0.29** (TransRAC, CVPR'22); nMAE 0.80 vs 0.44 | training-free pose ties a *trained* baseline on OBO, but its failures on low-amplitude actions cost it on MAE |
| **acoustic** (PT Sound Loc., unseen source) | vision-only recall **0.00 → 0.25–0.33** fused; fusion-FP 0–0.17 | the **recall-flip**: fusion rescues ~a third of sound events vision-only never hears (honest at scale, not the single-clip 1.0) |
| **relational** (golden) | confidence **AUROC ≈ 0.51** (≈ chance); ASK-rate ≈ 0.4 | the VLM's confidence does **not** separate right from wrong → justifies the **ASK** discipline over threshold-tuning |
| **timeliness** (PT action onsets) | **P-PAUC ≈ 60**; reaction λ p50 ≈ 1s | first real proactive-timeliness numbers (adapted PAUC) |
| **edge event-recall** (locally staged PT sample onsets) | **0.12** @ thr 0.10 (N=8) | ⚠️ the motion-gate only catches **1/8** seeded action onsets → e2e ≠ core there; a real **calibration** finding |

A v0.5 DRAFT, not a RELEASE claim. Two findings the benchmark surfaced that a self-congratulatory
eval would hide: the **flat confidence AUROC** (*why* the gate refuses to guess) and the low **edge
event-recall** on subtle actions (the motion-gate is mis-calibrated for them). Full plan, data,
and references:
[`packages/percept-vision/eval/BENCHMARK_PLAN.md`](packages/percept-vision/eval/BENCHMARK_PLAN.md).

Run the public scorecard:

```bash
percept-bench scorecard --manifest ../../eval/golden-v1/MANIFEST.json --out scorecard.txt
```

## Layout

```
packages/percept-vision/    the SDK (pip install percept-vision; import percept)
packages/percept-harness/   internal Python tier-0 edge reference + detector skills
packages/percept-edge/      percept-edge — on-device JS/WASM edge
docs/                       architecture & flow traces
eval/                       golden corpus (benchmark plan: packages/percept-vision/eval/)
Spec/                       the spec + implementation plan
Makefile                    test · eval-live · e2e · bench · check · reproduce
```

## Docs

- **Quickstart** — the 60-second offline snippet above; `make test` runs the fake-only unit lane.
- **Architecture / concepts** — [`docs/e2e-flow-milk-boilover.md`](docs/e2e-flow-milk-boilover.md)
  (the two layers, the firing path, the perception ceiling) and [`Spec/`](Spec/) (the full spec +
  phase plan).
- **Backends** — select by name (`fake` · `gemini` · `claude` · `deepgram`) or env
  (`PERCEPT_*_BACKEND`); add via the `percept.backends` entry-point group. Extras:
  `[gemini]` · `[claude]` · `[deepgram]` · `[audio]`.
- **Benchmark** — [`eval/BENCHMARK_PLAN.md`](packages/percept-vision/eval/BENCHMARK_PLAN.md).

## Status

**v0.1.0 — early / first public release.** The cognition core runs and is tested offline with fakes
(the L1 lane, no keys); the frontier backends (Gemini, Claude, Deepgram) and the edge packages are
wired behind their seams. The benchmark is a **v0.5 DRAFT** card. We are now in **real-life
testing** — APIs and numbers may change. Issues and contributions welcome.

## License

Apache-2.0.
