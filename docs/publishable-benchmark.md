# Publishable Benchmark Runbook

This runbook is for a public-grade Percept benchmark. It deliberately does **not** run an OpenAI
Realtime leg. OpenAI Realtime and TML numbers are cited as reference rows; Percept's controlled run
uses Gemini live as the perception backbone and compares:

- raw Gemini loop: judge every sampled frame / cue.
- Percept core: same Gemini verdicts through rising-edge cognition and ASK discipline.
- Percept edge: cheap wake policy before Gemini calls.

That keeps the benchmark affordable and isolates the claim: Percept improves proactivity, trust, and
cost around the same perception model.

## Reference Numbers

Use these as external reference bars, not as rows we reproduce in CI:

| Source | Metric | Reference |
| --- | --- | --- |
| Thinking Machines Lab Interaction Models | TimeSpeak macro accuracy | `64.7` |
| Thinking Machines Lab Interaction Models | CueSpeak macro accuracy | `81.7` |
| Thinking Machines Lab Interaction Models | RepCount-A off-by-one | `35.4` |
| Thinking Machines Lab Interaction Models | ProactiveVideoQA PAUC@omega=0.5 | `33.5` |
| Thinking Machines Lab Interaction Models | Charades mIoU | `32.4` |
| Thinking Machines Lab Interaction Models | GPT-realtime-2.0 Charades mIoU | `0` |

## Current Local Inventory

As of the draft golden set:

| Track | Local sample count | Publishable? |
| --- | ---: | --- |
| Perception Test action/timeliness | `8` events | no |
| Perception Test sound/acoustic | `2` events | no |
| Charades selection | `10` videos | no |
| OmniPro visual slice | `20` samples | no |
| CueSpeak / TimeSpeak | deterministic simulation only | no |

These are useful for smoke tests and debugging. They are not enough for a public benchmark claim.

## Release Sample Targets

Minimum targets before stamping a scorecard as publishable:

| Track | Minimum test size | Notes |
| --- | ---: | --- |
| Perception Test action/timeliness | `100-200` events | public, license-cleared, content-hashed |
| Perception Test sound/acoustic | `50-100` events | include visible and non-visible sources |
| Charades mIoU | `100+` video/action goals | compare raw Gemini loop, Percept gate, Percept event dedup |
| OmniPro visual-only | `100+` prompts | use official OmniPro scorer where possible |
| CueSpeak | `100+` authored prompts | held-out timing window + expected semantic response |
| TimeSpeak | `100+` authored prompts | held-out timing window + expected semantic response |

## Cost Model

Do not run OpenAI Realtime for our benchmark. Use public reference numbers.

Percept live cost is mostly Gemini input tokens from sampled frames plus small text outputs. Use a
two-stage budget:

| Run | Purpose | Expected cost |
| --- | --- | ---: |
| Smoke live | 8-20 samples, verify keys/cache/scorers | `$5-$25` |
| Pilot live | 50-100 total rows | `$25-$100` |
| Publishable live | full release sample targets, one seed | `$75-$250` |
| Publishable rerun | extra seeds / model comparison | `$250-$750` |

Prefer Gemini Flash/Lite-class models for the first pass. Keep all captured verdicts in cache so
scorecard changes can re-score without paying again.

## Run Plan

0. Prepare the local benchmark workspace.

```bash
# These live in ignored .env for the local machine:
# PERCEPT_BENCH_DATA_DIR=eval/bench-staging
# PERCEPT_BENCH_RUNS_DIR=eval/local-benchmark/runs
# PERCEPT_BENCH_CACHE_DIR=eval/local-benchmark/cache
make benchmark-prepare
# Optional network/data materialization step:
make benchmark-fetch VIDEOS=12 REPCOUNT=12
```

The tracked local control-room docs live in `eval/local-benchmark/`; datasets,
logs, caches, and generated cards remain ignored.

1. Expand and freeze the public manifest.
   - Add rows to `eval/golden-v1/MANIFEST.json`.
   - Require media SHA-256 and annotation SHA-256 for every row.
   - Keep `dev` and `test` disjoint.
   - Move manifest status from `draft` to `release` only after review.

2. Run the artifact audit.

```bash
make audit-benchmark
percept-bench audit --release \
  --manifest eval/golden-v1/MANIFEST.json \
  --baselines packages/percept-vision/eval/baselines.json
```

3. Run live Percept tracks only.

```bash
export PERCEPT_LIVE=1
export GEMINI_API_KEY=...
export PERCEPT_GEMINI_MODEL=gemini-3.5-flash

make benchmark-run LIVE=1 TRACKS=stable N=20 WORKERS=2 ATTEMPTS=2

# Direct lower-level experiment lanes remain available:
PERCEPT_LIVE=1 make e2e-bench
PERCEPT_LIVE=1 make charades-bench
PERCEPT_LIVE=1 PYTHONPATH=packages/percept-vision python3 -m eval.exp_charades_miou
PERCEPT_LIVE=1 PYTHONPATH=packages/percept-vision python3 -m eval.exp_omnipro
```

4. Render the public card.

```bash
make benchmark-collect RUN_DIR=eval/local-benchmark/runs/<run-id>

# Release gate still uses the official scorer/card command:
percept-bench scorecard --mode full --require-release \
  --manifest eval/golden-v1/MANIFEST.json \
  --out scorecard.txt
```

## Claims To Make

Allowed:

- controlled raw Gemini loop vs Percept, same model, same frames/verdicts where possible.
- call reduction, fire deduplication, no-nag behavior, ASK/refuse discipline.
- edge wake recall/precision tradeoff with explicit policy.
- comparison to published reference bars with citations.

Avoid:

- "beats OpenAI Realtime" unless we pay to run a matched leg.
- "publishable" while manifest status is `draft`, dev split is empty, or modeled cells remain.
- treating CueSpeak/TimeSpeak simulations as TML-comparable results.

## Likely Improvement Targets

The current draft already exposes the next work:

- Edge policy recall improves over motion-only, but precision is low; add hand/pose/object-state and
  audio-event detectors before further threshold lowering.
- Low-motion hand/object events need specialist edge skills.
- CueSpeak/TimeSpeak need real prompt sets and judge-backed timing windows.
- Charades and OmniPro runs should cache verdicts so cognition changes can re-score cheaply.
