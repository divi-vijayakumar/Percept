# Benchmarks

A reproducible harness that runs Percept and the pipelines a developer would otherwise write over
the same public footage, with the same condition text and the same judge, and reports every arm,
including the ones Percept loses.

```bash
pip install -e "packages/percept-vision[dev]" -r benchmarks/requirements.txt   # OpenCV 4, see Arms

# Offline dry run: generated clips, an oracle judge, no key, no network (about 15 s)
python -m benchmarks.run --dataset synthetic --judge fake --out benchmarks/results/dry-run

# E1 on MEVA with a real judge and a hard spend cap
python -m benchmarks.fetch_meva --hours 5                 # about 8.3 GB, into eval/bench-staging/meva
python -m benchmarks.run --config benchmarks/configs/e1.json --judge gemini:<model> \
    --price <usd_per_mtok_in>,<usd_per_mtok_out>,<tokens_per_image> --max-cost-usd 10 \
    --out benchmarks/results/e1-meva
```

`benchmarks/results/` and `results/` are ignored by git. Footage, private labels, keys and
results are never committed.

## Datasets

| `--dataset` | footage | notes |
|---|---|---|
| `meva` | MEVA (CC BY 4.0): fixed indoor and outdoor cameras, five-minute clips | the real continuous set, and the only one E1 publishes; fetch with `fetch_meva.py` |
| `perception-test` | Perception Test (CC-BY 4.0) action-localisation validation clips | short, dense clips: diagnostics only, never a published comparison |
| `perception-test-sparse` | Perception Test clips joined with idle and distractor gaps at several occupancy levels | **synthetic**, for the cost-versus-occupancy curve only |
| `local:<dir>` | your MP4s plus `labels.json` or `labels.yaml` | private clips, evaluated locally (format in `datasets.py`) |
| `synthetic` | generated clips | for CI |

Condition text is mechanical ("`<label> begins`"). Every case also watches `--negatives` labels
that never occur in it (nor in a long-sparse case's distractor clips), so precision and false
events per hour are defined.

`fetch_meva.py` builds annotation and video URLs from the layout of the public MEVA repository
and bucket (`kitware/<date>/<hour>/` and `drops-123-r13/<date>/<hour>/`, the hour being the
clip's end or start hour). That layout was checked by listing, not yet by a completed fetch:
treat it as unverified until a real fetch succeeds.

## Arms

| arm | what it does |
|---|---|
| `percept` | a `Monitor` with the product defaults (6 attempts a minute per stream, 2 reserved per request) **except the rearm**, which E1 standardises to 30 s (product default 120 s); fed in real time. Reports name it "percept (product defaults; rearm standardised to 30 s)" |
| `percept-matched` | the same with `max_requests_per_min` set by `--percept-budget` (default 12: `chunk5`'s request rate) |
| `chunk5`, `chunk10`, `chunk30` | judge the last T seconds every T seconds, all conditions batched, then a cooldown |
| `detector_debounce` | a people (HOG) or motion (MOG2) detector with a cooldown; only conditions a detector can express |
| `detector_then_vlm` | the same detectors gate judge calls; other conditions are gated by motion |

All arms decode the same frames (5 per second, longest side 640 px; checked by digest) and judge
the same evidence format (up to 4 frames). Percept runs only at `--speed 1` for published
numbers: its budget and timers use the real clock. E1 uses one suppression window for every
arm: the baseline cooldown and Percept's `rearm_s` are both 30 s (Percept's product default is
120 s). So the `percept` arm is the product defaults with one change, not the product
configuration unchanged: every Percept arm's label in the report, and its
`changed_from_product_defaults` in the manifest, name each setting that differs.

The people detector is OpenCV 4's built-in HOG detector run on a 320-pixel frame: a weak
detector, there to stand in for "a detector a developer has to hand", not a strong one. OpenCV 5
removed it; with OpenCV 5 installed, no condition is person-expressible and the report says so.

## What the report states

- whether the run is **publishable**, and if not, why: an oracle judge, a dataset other than
  MEVA (every such report carries a "diagnostic, not publishable" banner), `--speed` other
  than 1, a spend-cap abort, skipped cases, unintended frame loss, or a Percept feed that fell
  more than a second behind real time;
- frames, in two separate counts. **Intentional sampling**: frames the configured policies
  skip (the harness's `--fps` sampling of the source, the Monitor's processing fps cap if set);
  expected, never disqualifying. **Unintended loss**: frames lost because decoding, a queue or
  processing couldn't keep up (decode gaps, frames replaced in the push handoff, frames pushed
  more than a second late); any of it makes the run not publishable;
- the configuration of every Percept arm, and every setting changed from the product defaults;
- recall and the recall ceiling a perfect judge could reach under the arm's cooldown, precision
  (duplicates count against it), false events per source-hour (absent-condition rows included),
  duplicates per occurrence, onset-to-decision latency p50/p95 with misses beside it;
- cost per monitored hour and per true event, from the usage each judge reports, and the
  actual spend of the run (kept, with the partial results, when the spend cap stops it);
- per arm: budget drops, `trigger_never_fired` and `verify_rejected` misses, unverified and
  uncertain decisions, and the seconds of footage actually watched;
- every miss attributed to a stage: `trigger_never_fired`, `budget_dropped`, `judge_failed`,
  `judge_said_no`, `cooldown_absorbed`, `verify_rejected`, `decide_suppressed` (Percept's from its
  trace; its `budget_dropped` is approximate, because the trace counts drops per minute);
- CPU per source-hour, the process's peak resident memory, pre-roll frames, frames capped and
  real-time lag;
- coverage: cases watched, conditions an arm can't express, and cases skipped with the reason.

Matching is one-to-one: a decision matches an occurrence of its condition when its evidence
interval overlaps the occurrence widened by `--tolerance` seconds and ends no earlier than half a
second before the onset. Only footage an arm actually watched counts. Details are in `metrics.py`.

`--max-cost-usd` covers the whole run. Each request reserves its worst case
(`percept.judges.base.worst_case_cost`, every attempt) before it is sent; a request that would
not fit stops the run cleanly, the report is written and marked aborted, and the exit code is 3.

The `fake` judge is an oracle that answers from the ground truth with injected noise
(`--oracle-fn`, `--oracle-fp`, `--oracle-latency`). It says yes when one of the sampled frames it
is shown falls within half a second before to two seconds after an onset, and a second look at
the same frames gets the same answer. Its numbers test the harness, not a model.

## Known limitation: dense, continuously changing scenes

The benchmark exists to find where Percept wins and where it loses, and both are reported. A
loss on MEVA will be published too.

An offline diagnostic on 16 Perception Test clips (short, handheld, dense) found Percept behind
simple periodic sampling. **This is a non-publishable diagnostic with a simulated (oracle)
judge, not validated model accuracy.** Every arm used a 30 s cooldown; the observable recall
ceiling under that cooldown was 46.4%.

| arm | recall | relative to the 46.4% ceiling |
|---|---|---|
| Percept defaults (cooldown standardised to 30 s) | 17.9% | ≈39% |
| Percept matched budget (12 a minute) | 26.2% | ≈57% |
| model every 5 s | 40.5% | ≈89% |
| model every 30 s | 21.4% | – |

The cause is the request budget: on handheld footage the change trigger fires on nearly every
frame, so looks queue up and the budget drops them (54 of the 69 misses of the Percept-defaults
arm were budget drops). Dense handheld scenes are outside Percept 1.0's envelope, which is
static cameras; that is why MEVA, with fixed cameras, is the primary public benchmark.

E2 (best practical pipelines, cascades allowed) is scaffolded in `configs/e2.json` and not wired
into the runner yet.
