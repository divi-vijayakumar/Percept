# Evaluating Percept on your own benchmark

Percept ships the eval harness it uses on itself, so you can measure the only number that matters
for a proactive agent on **your** concerns: does it fire when it should, stay silent when it
shouldn't, and **refuse to guess** when it isn't sure?

This page covers the **consumer-facing** path — a per-concern precision / recall report from a
manifest of labelled clips, runnable **fully offline** (the deterministic `fake` backend, no keys).
The SDK's own internal gate (`eval.run`, asserting `false_fire == 0` on a held-out golden split) is
unchanged and documented separately in [`packages/percept-vision/eval/`](../packages/percept-vision/eval/).

The harness lives in the `eval` package under `packages/percept-vision`. Point `PYTHONPATH` at it:

```bash
PYTHONPATH=packages/percept-vision python3 -m eval.benchmark examples/benchmark/manifest.json
```

---

## What you get: per-concern P/R

For every concern under test the report gives:

| metric | meaning |
|---|---|
| **precision** | of the fires this concern emitted, the fraction that were right (`TP / (TP+FP)`). `1 − false-fire-rate`. |
| **recall** | of the moments it *should* have fired, the fraction it caught (`TP / (TP+FN)`). |
| **false_fire** | absolute count of fires that aligned to no labelled edge — the **trust gate** (spec §0). |
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
[`eval/LABELING_PIPELINE.md`](../packages/percept-vision/eval/LABELING_PIPELINE.md).

---

## Running it

### CLI

```bash
PYTHONPATH=packages/percept-vision \
  python3 -m eval.benchmark examples/benchmark/manifest.json \
    --out benchmark.json \
    --max-false-fire 0 \
    --min-recall 0.9
```

Writes the JSON report to `--out` and prints a table. Exits non-zero if the gate fails — so the CLI
*is* a usable CI gate on its own. There's also a Make target:

```bash
make eval-benchmark MANIFEST=examples/benchmark/manifest.json
```

### From Python

```python
from eval.benchmark import Manifest, run_manifest, format_report

report = run_manifest(Manifest.load("examples/benchmark/manifest.json"))
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
from eval.benchmark import ConcernObservation, per_concern_report
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
plus any per-concern precision/recall floor your product needs. `examples/benchmark/ci_gate.py` is a
copy-pasteable template:

```python
from eval.benchmark import BenchmarkGate, Manifest, check_benchmark_gate, run_manifest

GATE = BenchmarkGate(max_false_fire=0, min_precision=1.0, min_recall=0.9)

report = run_manifest(Manifest.load("examples/benchmark/manifest.json"))
passed, reasons = check_benchmark_gate(report, GATE)
raise SystemExit(0 if passed else 1)   # non-zero fails the build, with `reasons` printed
```

In GitHub Actions:

```yaml
- name: Percept per-concern regression gate
  run: PYTHONPATH=packages/percept-vision python3 examples/benchmark/ci_gate.py
```

`false_fire` is checked on the **overall total** (the absolute trust gate). `min_precision` /
`min_recall` are checked **per concern**, so one weak concern fails the build with a precise reason
rather than hiding behind a healthy average.

> This consumer gate is **additive** — it does not touch the SDK's own `eval.run` gate
> (`false_fire == 0` on the held-out golden test split), which keeps running unchanged.
