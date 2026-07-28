---
title: Evaluating Percept on your own benchmark
updated: 2026-07-23
applies_to: percept-vision 1.0.0
status: stable
---
# Evaluating Percept on your own benchmark

Percept ships the eval harness it uses on itself, so you can measure the only number that matters
for a proactive agent on **your** concerns: does it fire when it should, stay silent when it
shouldn't, and **refuse to guess** when it isn't sure?

## Which tool?

Three distinct CLIs cover three distinct questions. Pick by what you are trying to learn:

| tool | question it answers | input | output / verdict |
|---|---|---|---|
| `percept-bench` | per-concern **precision / recall** for one configuration | a labelled-clip **manifest** (one JSON file) | a P/R report; offline, deterministic `fake` backend. **The rest of this page.** |
| `percept-eval` | did the **candidate** beat the **baseline**? (paired A/B) | two **plans** + a **fixtures directory** + pre-registered **bars** | exit `0` pass / `1` a bar failed / `2` refused. See [Differential A/B](#differential-ab-percept-eval). |
| `percept-optimize` | which **safe tunables** to pick | a search space over declared-safe tunables | an `OptimizationLock`. See [optimize.md](./optimize.md). |

`percept-bench` scores **one** config against ground truth. `percept-eval` scores **two** configs
against each other on the same fixtures, and only pronounces pass/fail against bars you declared
*before* looking. `percept-optimize` searches the tunable space and hands you a lock, not a metric.

This page covers the **consumer-facing** path — a per-concern precision / recall report from a
manifest of labelled clips, runnable **fully offline** (the deterministic `fake` backend, no keys).
The SDK's own internal gate (`eval.run`, asserting `false_fire == 0` on a held-out golden split) is
unchanged and documented separately in [`packages/percept-vision/eval/`](../../packages/percept-vision/eval/).

The benchmark harness **runs from a source checkout** — it is repo/dev tooling and is **not** shipped
in the `percept-vision` wheel (only `import percept` is). The `percept-bench` console script exists on
an install, but from a bare wheel it prints a "run from a checkout" note; to actually run it, clone the
repo and put the package on `PYTHONPATH` (the `[bench]` extra provides the deps — numpy, opencv):

```bash
git clone https://github.com/divi-vijayakumar/Percept && cd Percept
pip install -e "packages/percept-vision[bench]"
PYTHONPATH=packages/percept-vision python3 -m percept.bench eval/benchmark_ci_gate/manifest.json
# or, equivalently, the Makefile targets: `make eval-benchmark` / `make scorecard` / `make audit-benchmark`
```

For the multi-track public snapshot card:

```bash
percept-bench audit --manifest eval/golden-v1/MANIFEST.json \
    --baselines packages/percept-vision/eval/baselines.json
percept-bench scorecard --manifest eval/golden-v1/MANIFEST.json --out scorecard.txt
```

For the public-grade live benchmark plan, including sample-size gates, cost estimates, and the
decision to use OpenAI/TML as reference rows rather than paid reproduced legs, see
[the benchmarks hub](../benchmarks/README.md).

---

## What you get: per-concern P/R

For every concern under test the report gives:

| metric | meaning |
|---|---|
| **precision** | of the fires this concern emitted, the fraction that were right (`TP / (TP+FP)`). `1 − false-fire-rate`. |
| **recall** | of the moments it *should* have fired, the fraction it caught (`TP / (TP+FN)`). |
| **false_fire** | absolute count of fires that aligned to no labelled edge — the **trust gate**. |
| **false_fire_rate** | `FP / fires` — the fraction of emitted fires that were wrong. |
| **refuse_rate** | `asks / (fires + asks)` — how often the gate declined to guess (ASK) rather than fire. |

A fire only counts as a true positive if it lands inside the **reaction window** `[edge, edge+omega]`
— *a right answer at the wrong time scores wrong*. `omega` defaults to 2.0s and is set per manifest.

The `overall` row micro-averages (sums TP/FP/FN across concerns, then divides), and reports
`false_fire` as the absolute total so a CI gate can assert it is exactly `0`.

---

## The benchmark input format (manifest)

A benchmark is one JSON file: the **concerns** under test plus the **labelled clips**.

```json
{
  "omega": 2.0,
  "concerns": [
    {
      "id": "coffee",
      "condition": "the user is drinking coffee",
      "say": "Heads up — stepping back from caffeine?"
    }
  ],
  "clips": [
    {
      "clip_id": "morning-coffee",
      "frames": [[0.0, "empty"], [1.0, "sip-coffee"], [2.0, "sip-coffee"]],
      "expected": { "coffee": [1.0] }
    },
    {
      "clip_id": "water-only",
      "frames": [[0.0, "empty"], [1.0, "sip-water"]],
      "expected": { "coffee": [] }
    }
  ]
}
```

### Fields

- **`omega`** *(float, default 2.0)* — the reaction window in seconds. A fire satisfies a labelled
  edge only within `[edge, edge + omega]`.
- **`concerns`** *(required)* — the watch conditions. Each needs:
  - `id` — a stable string id, used in the report and to key `expected`.
  - `condition` — the plain-language watch condition (what the backend judges each frame against).
  - `say` *(optional)* — the message the agent would speak on fire (cosmetic for scoring).
- **`clips`** *(required)* — the labelled clips. Each has:
  - `clip_id` — a stable id for the clip.
  - `frames` — `[[t, frame], ...]`, replayed in order. `frame` is whatever your backend consumes:
    a **scripted string token** for the `fake` backend (offline), or decoded image bytes /
    `(media_type, bytes)` for a real backend. The format is identical either way.
  - `expected` — `{concern_id: [t, ...]}`, the **rising-edge timestamps** each concern should fire
    at on this clip. A concern absent from `expected` (or mapped to `[]`) is a **no-fire** clip for
    that concern — any fire there is a false-fire. Every id in `expected` must be a declared concern
    (a typo fails loudly).

> **Offline-fake note.** The `fake` backend scripts verdicts by **frame token, not by condition**,
> so on a shared frame every concern sees the same verdict. For faithful offline examples, script
> one concern per clip (as the example manifest does). A real backend judges each concern's
> condition independently against the same frame, so multi-concern clips are exact.

### Where labels come from

`expected` is *your* ground truth — the moment a human (or a stronger adjudicator model) says the
condition became true. The SDK's own labelling/review provenance pattern is documented in
[`eval/LABELING_PIPELINE.md`](../../packages/percept-vision/eval/LABELING_PIPELINE.md).

---

## Running it

### CLI

```bash
percept-bench eval/benchmark_ci_gate/manifest.json \
    --out benchmark.json \
    --max-false-fire 0 \
    --min-recall 0.9
```

```bash
percept-bench scorecard --manifest eval/golden-v1/MANIFEST.json --out scorecard.txt
```

Writes the JSON report to `--out` and prints a table. Exits non-zero if the gate fails — so the CLI
*is* a usable CI gate on its own. There's also a Make target:

```bash
make eval-benchmark MANIFEST=eval/benchmark_ci_gate/manifest.json
```

### From Python

```python
from percept.bench import Manifest, run_manifest, format_report

report = run_manifest(Manifest.load("eval/benchmark_ci_gate/manifest.json"))
print(format_report(report))
print(report["overall"]["false_fire"])   # 0
print(report["concerns"]["coffee"]["recall"])
```

`run_manifest` replays each clip through a real `Percept` on the `fake` backend by default (offline,
no keys) — the gate, refractory, and rising-edge logic all run for real. Pass `vision=` / `settings=`
to drive a different backend or gate configuration.

### Already have fires from a live run?

If you replayed your clips against a real backend and captured the fires, skip the replay and score
the recorded observations directly:

```python
from percept.bench import ConcernObservation, per_concern_report
from eval.replay import Edge, Fire

report = per_concern_report(
    [
        ConcernObservation(
            concern_id="coffee",
            expected_edges=[Edge("clip1::coffee", 10.0)],
            fires=[Fire("clip1::coffee", 10.4)],
            n_asks=1,
        )
    ],
    omega=2.0,
)
```

---

## Wiring a regression gate into CI

The harness mirrors the SDK's own posture: **never confidently-wrong fire** (`false_fire == 0`),
plus any per-concern precision/recall floor your product needs. `eval/benchmark_ci_gate/ci_gate.py` is a
copy-pasteable template:

```python
from percept.bench import BenchmarkGate, Manifest, check_benchmark_gate, run_manifest

GATE = BenchmarkGate(max_false_fire=0, min_precision=1.0, min_recall=0.9)

report = run_manifest(Manifest.load("eval/benchmark_ci_gate/manifest.json"))
passed, reasons = check_benchmark_gate(report, GATE)
raise SystemExit(0 if passed else 1)   # non-zero fails the build, with `reasons` printed
```

In GitHub Actions:

```yaml
- name: Percept per-concern regression gate
  run: PYTHONPATH=packages/percept-vision python3 eval/benchmark_ci_gate/ci_gate.py
```

`false_fire` is checked on the **overall total** (the absolute trust gate). `min_precision` /
`min_recall` are checked **per concern**, so one weak concern fails the build with a precise reason
rather than hiding behind a healthy average.

> This consumer gate is **additive** — it does not touch the SDK's own `eval.run` gate
> (`false_fire == 0` on the held-out golden test split), which keeps running unchanged.

## Golden Set + Baselines

- Golden v1 draft manifest: [`../eval/golden-v1/MANIFEST.json`](../../eval/golden-v1/MANIFEST.json)
- Baseline registry: [`../packages/percept-vision/eval/baselines.json`](../../packages/percept-vision/eval/baselines.json)

These are intentionally separate from private fixtures. A release scorecard must use license-cleared,
content-hashed test rows and cite the manifest hash.

The current golden v1 draft seeds its `test` split with public
`google-deepmind/perception_test` sample videos by reference. Each row records the local staging path,
video id, task, license, media SHA-256, and annotation SHA-256; the media stays in
`eval/bench-staging/perception_test/raw/` and is not packaged.

From Python:

```python
from percept.bench import validate_baselines, validate_golden_manifest

validate_baselines("packages/percept-vision/eval/baselines.json")
validate_golden_manifest("eval/golden-v1/MANIFEST.json")                 # draft-shape check
validate_golden_manifest("eval/golden-v1/MANIFEST.json", release=True)   # must fail until release-ready
```

From the CLI:

```bash
percept-bench audit \
    --manifest eval/golden-v1/MANIFEST.json \
    --baselines packages/percept-vision/eval/baselines.json

percept-bench audit --release \
    --manifest eval/golden-v1/MANIFEST.json \
    --baselines packages/percept-vision/eval/baselines.json
```

The first command is the draft artifact check used by OSS contributors. The `--release` command is
expected to fail until the golden manifest is marked `release` and its test split contains
license-cleared, content-hashed media rows. That is the guardrail between “useful draft scorecard”
and “publishable benchmark claim.”

Use the scorecard gate in release automation:

```bash
percept-bench scorecard --mode full --require-release \
    --manifest eval/golden-v1/MANIFEST.json \
    --out scorecard.txt
```

`--require-release` writes the card, then exits non-zero unless the rendered stamp is `RELEASE`
(`mode=full`, pinned manifest, and no modeled cells).

---

## Differential A/B (`percept-eval`)

`percept-bench` measures **one** configuration. `percept-eval` answers a different question: **did a
change help?** It runs two **plans** — a `baseline` and a `candidate` — *paired* over the same fixtures,
so the difference is attributable to the one thing you changed, and pronounces PASS/FAIL only against
**bars you declared before scoring**.

```bash
percept-eval --baseline base.yaml \
             --candidate cand.yaml \
             --fixtures ./fixtures \
             --bars bars.yaml \
             --split test \
             --json report.json
```

- **`--baseline` / `--candidate`** *(required)* — the two plan files (`.yaml`/`.json`).
- **`--fixtures`** *(required)* — the fixtures root (layout below).
- **`--bars`** — the **client-supplied** bars file. Omit it and `percept-eval` reports the numbers
  *without* pronouncing a verdict — Percept never sets a threshold on your behalf. A bar you set after
  seeing the result is not a bar.
- **`--split`** — which split to score; `test` (default) is the held-out one. Tune against it and you
  have measured nothing.
- **`--json`** — write the full paired report as JSON.
- **`--frontier`** — also emit the **operating-frontier report**: exact one-sided 95% bounds on FIRE
  recall and false FIREs per scheduled watch-hour, ASK rate, coverage, and duplicate FIREs per episode.
  See [operating-frontier.md](../reference/operating-frontier.md).

Exit codes: **`0`** pass (or reported, when no bars) / **`1`** a bar failed / **`2`** refused (bad input,
a missing/broken split, or a configuration that drifted after the bars were declared).

### The fixtures directory

`percept-eval`, `percept-derive-profile`, and `percept-optimize` all read the same fixtures layout —
predictions and ground truth on disk, not a live backend:

```
fixtures/
  splits.json                 # optional; "test" is the held-out split
  <fixture_id>/
    ground_truth.json         # goal_shape + labeled episodes
    <arm>/events.json         # the arm's predicted JudgmentEvents
    <arm>/trace.json          # the arm's TraceSummary artifact (optional)
```

`<arm>` is `baseline` / `candidate` for a differential A/B; for a `percept-derive-profile` ladder it is
each rung/template name instead (see [derive-a-profile.md](./derive-a-profile.md)). Everything but
`ground_truth.json` is optional — an arm with no `trace.json` is scored on events alone and simply
excluded from the funnel comparison.

Scope honesty carries over: a fixture declares a `goal_shape`, and shapes the one-to-one event matcher
cannot honestly validate are **reported but never scored** into a delta or a bar.

> Still no published Percept accuracy numbers. `percept-eval` is the tool you point at **your** fixtures
> with **your** bars; the SDK ships the harness, not a benchmarked claim.
