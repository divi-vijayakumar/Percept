# Evaluating a configuration

> **Experimental.** `percept.plans.load_labels`, `split`, `evaluate` and `percept.plans.run` may
> change in a minor release.

Percept's evaluation toolkit measures what a watch plan does on your own labelled clips: recall,
precision, false alerts, onset-to-decision latency, cost, and where each missed occurrence was
lost. It reports; it never ranks or chooses between configurations. A runnable, offline walk
through the whole flow is [examples/plan_evaluation.py](../examples/plan_evaluation.py).

## 1. Labels

```python
from percept import plans

labels = plans.load_labels({
    "conditions": [{"label": "person_enters", "text": "A person walks into view through the doorway."}],
    "clips": [
        {"file": "dock-2026-10-01.mp4", "group": "dock-2026-10-01",
         "occurrences": [{"label": "person_enters", "start": 12.5, "end": 15.0}]},
    ],
})
```

Each clip names its file, an optional recording `group` (a camera and day, say), and its labelled
occurrences in seconds. Overlapping occurrences of one label merge. A malformed document raises
`LabelError` with the path of the field. `labels.labels_hash` is the same on every machine.

## 2. A frozen split

```python
manifest = plans.split(labels)
manifest.clips["dev"], manifest.clips["validation"], manifest.clips["test"]
```

Clips are assigned to dev, validation or test by a hash of their group, so clips from one
recording never straddle splits, and adding clips never moves existing ones. `manifest.manifest_hash`
identifies the split.

## 3. The trigger stage alone, at no cost

```python
from percept.plans import run as evaluation

runs = evaluation.attention_only(compiled, labels, manifest.clips["dev"], video_root="clips/")
score = plans.evaluate(labels, runs)
print(score.funnel["candidate"], "of", score.occurrences, "occurrences got a look")
```

`attention_only` runs the attention step the Monitor runs, with no judge calls. It answers the
first question cheaply: would any look have seen each occurrence? An occurrence no trigger reaches
can't be found by any judge.

## 4. Configurations you declare, side by side

```python
scores = evaluation.sweep({"on_change": plan_a, "every 5 s": plan_b}, labels, manifest,
                          split="validation", video_root="clips/", judge_cap_usd=1.00)
```

`sweep` runs each compiled plan over the split, with the full runtime and a real judge, behind a
hard dollar cap. It returns the scores in your order and does not pick one. `run` does the same
for a single plan. A reply cache replays a repeated request with its original usage and latency, so
an identical re-run decides the same and pays nothing again. `cost_usd` stays the live-equivalent
cost; `paid_usd` is what was paid.

Runs are paced in real time by default, because triggers, budgets and latency only mean something
at the speed of the stream. A faster `speed` is for smoke tests.

## 5. The test split, once

```python
registration = evaluation.register(chosen, manifest, labels, split="test", created="2026-10-15T09:00:00Z")
test = evaluation.score_split(chosen, labels, manifest, split="test", video_root="clips/",
                              judge_cap_usd=2.00, registration=registration, ledger="ledger.jsonl")
```

`register` freezes the chosen plan with its plan, manifest and labels hashes. `score_split`
refuses the test split for a plan without a matching registration. With a `ledger` file, it also
scores a registration only once, unless you pass `rescore=True`.

## What a score reports

`plans.evaluate` returns a `Score`:

| Field | Meaning |
|---|---|
| `recall`, `precision` | matched occurrences over labelled ones, and over predictions (duplicates count against precision) |
| `onset_recall`, `recall_ceiling` | occurrences caught near their start; the best recall the decision cooldown allows |
| `false_per_hour`, `false_per_100_occurrences` | false alerts, by time and by labelled occurrence |
| `latency_p50_s`, `latency_p95_s` | onset to decision, for matched occurrences |
| `cost_usd`, `cost_per_true_event`, `compute_s` | judge cost as reported by the judge, per correct detection, and local compute |
| `funnel` | how many occurrences survived each stage: candidate, admitted, evaluated, positive, verified, delivered |
| `misses_by_stage` | why each miss happened: the trigger never fired, the budget dropped it, the judge failed or said no, verification rejected it, the cooldown absorbed it, or decide suppressed it |
| `capped` | a spend cap refused requests, so the run is incomplete |

Intervals come from a seeded bootstrap over clips. A match needs a decision whose evidence covers
the occurrence within a ±2 s tolerance.

## Reading the results honestly

- Small, distant or brief activity is often missed: check `misses_by_stage` before blaming the
  judge. A miss at `trigger_never_fired` needs a different attention setting, not a different model.
- Confidence is the judge's and is advisory. Labels are yours: if they omit real occurrences,
  precision will read lower than it is.
- Validation is for comparing configurations. Score the test split once, after you've chosen.
