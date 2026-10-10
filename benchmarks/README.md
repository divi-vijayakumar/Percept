# Benchmarks

A reproducible harness that runs Percept and the pipelines a developer would otherwise write over
the same public footage, with the same condition text and the same judge, and reports every arm,
including the ones Percept loses. It needs the video extra:
`pip install -e "packages/percept-vision[dev,video]"`.

```bash
# Offline dry run: generated clips, an oracle judge, no key, no network (about 15 s)
python -m benchmarks.run --dataset synthetic --judge fake --out benchmarks/results/dry-run

# E1 on MEVA with a real judge and a hard spend cap
python -m benchmarks.fetch_meva --hours 5                 # about 8.3 GB, into eval/bench-staging/meva
python -m benchmarks.run --config benchmarks/configs/e1.json --judge gemini:<model> \
    --price <usd_per_mtok_in>,<usd_per_mtok_out>,<tokens_per_image> --max-cost-usd 25 \
    --out benchmarks/results/e1-meva
```

`benchmarks/results/` and `results/` are ignored by git. Footage, private labels, keys and
results are never committed.

## Datasets

| `--dataset` | footage | notes |
|---|---|---|
| `meva` | MEVA (CC BY 4.0): fixed indoor and outdoor cameras, five-minute clips | the primary real continuous set; fetch with `fetch_meva.py` |
| `perception-test` | Perception Test (CC-BY 4.0) action-localisation validation clips | short clips; `--root` points at the download |
| `perception-test-sparse` | Perception Test clips joined with idle and distractor gaps at several occupancy levels | **synthetic**, for the cost-versus-occupancy curve only |
| `local:<dir>` | your MP4s plus `labels.json` or `labels.yaml` | private clips, evaluated locally (format in `datasets.py`) |
| `synthetic` | generated clips | for CI |

Condition text is mechanical ("`<label> begins`"). Every case also watches `--negatives` labels
that never occur in it, so precision and false events per hour are defined.

## Arms

| arm | what it does |
|---|---|
| `percept` | a `Monitor` with its defaults, fed in real time (`--speed 1`, the only publishable setting) |
| `chunk5`, `chunk10`, `chunk30` | judge the last T seconds every T seconds, all conditions batched, then a cooldown |
| `detector_debounce` | an OpenCV people (HOG) or motion (MOG2) detector with a cooldown; only conditions a detector can express |
| `detector_then_vlm` | the same detectors gate judge calls; other conditions are gated by motion |

All arms decode the same frames (5 per second, longest side 640 px; checked by digest) and judge
the same evidence format (up to 4 frames). OpenCV 5 has no built-in people detector; with it, no
condition is person-expressible and the report says so.

## What the report states

- recall, precision (duplicates count against it), false events per source-hour (absent-condition
  rows included), duplicates per occurrence, onset-to-decision latency p50/p95 with misses beside it;
- cost per source-hour and per true event, from the usage each judge reports;
- every miss attributed to a stage: `trigger_never_fired`, `budget_dropped`, `judge_failed`,
  `judge_said_no`, `decide_suppressed` (Percept's from its trace);
- CPU, peak traced memory and pre-roll frames;
- coverage: cases, conditions an arm can't express, and whether the run was aborted.

Matching is one-to-one: a decision matches an occurrence of its condition when its evidence
interval overlaps the occurrence widened by `--tolerance` seconds. Details are in `metrics.py`.

`--max-cost-usd` covers the whole run. Each request reserves its worst case
(`percept.judges.base.worst_case_cost`, every attempt) before it is sent; a request that would
not fit stops the run cleanly, the report is written and marked aborted, and the exit code is 3.

The `fake` judge is an oracle that answers from the ground truth with injected noise
(`--oracle-fn`, `--oracle-fp`, `--oracle-latency`). Its numbers test the harness, not a model.

E2 (best practical pipelines, cascades allowed) is scaffolded in `configs/e2.json` and not wired
into the runner yet.
