---
title: Run benchmarks
updated: 2026-07-28
applies_to: percept-vision 1.0.0
status: stable
---
# Run benchmarks

Percept ships a benchmark harness with three parts: a snapshot **scorecard**, an artifact **audit**,
and a reproducible local **model-sweep** runbook. This guide is how to run them.

**No Percept accuracy numbers are published yet.** The harness is the reproducible machinery for
producing them; the numbers themselves are a separate, later phase. Everything below is runnable
today, but the results are yours to generate on your own data — do not expect a published figure.

## Prerequisite: run from a checkout

The benchmark harness is repo/dev tooling — it lives in the top-level `eval` package and does **not**
ship in the installed wheel (`percept.bench` imports it lazily and degrades to a "run from a
checkout" note on a bare install). So clone and install with the `bench` extra:

```bash
git clone https://github.com/divi-vijayakumar/Percept && cd Percept
pip install -e "packages/percept-vision[bench]"
export PYTHONPATH=packages/percept-vision
```

## Snapshot scorecard

The scorecard renders a multi-track card. Offline it fills cells with deterministic stubs; real
numbers need `--live` and provider keys.

```bash
PYTHONPATH=packages/percept-vision python3 -m percept.bench scorecard \
    --manifest eval/golden-v1/MANIFEST.json --out scorecard.txt
```

This is exactly what `make scorecard` runs. Add `--live` (plus keys) to score real cells.

The card carries a **stamp**. It is `DRAFT` unless the run is `mode=full`, points at a pinned release
manifest, has no modeled cells, and passes coverage — only then is it `RELEASE`. Pass
`--require-release` to make CI exit non-zero on a card that renders `DRAFT`.

## Artifact audit

The audit validates the golden manifest and the baseline registry artifacts:

```bash
make audit-benchmark
# equivalently:
percept-bench audit --manifest eval/golden-v1/MANIFEST.json \
    --baselines packages/percept-vision/eval/baselines.json
```

Add `--release` to require release-ready artifacts. This is **expected to fail** today: the golden
manifest is still status `draft`, and `--release` demands it flip to `release` with a
license-cleared, content-hashed test split. The failure is the gate working, not a bug.

## Reproducible local model-sweep runbook

The model sweep is a reproducible local run against real providers. All fetched media and run
outputs are gitignored — nothing here is committed.

```bash
make benchmark-prepare                          # create ignored local folders + inventory
make benchmark-fetch VIDEOS=12 REPCOUNT=12      # pre-cache benchmark media
make benchmark-manifest                         # write a content-hashed candidate manifest

export PERCEPT_LIVE=1 GEMINI_API_KEY=...         # real provider calls
make benchmark-run LIVE=1 TRACKS=stable N=20 WORKERS=2

make benchmark-collect RUN_DIR=eval/local-benchmark/runs/<run-id>   # re-render a scorecard
```

`benchmark-run` writes a run directory under `eval/local-benchmark/runs/`; pass it to
`benchmark-collect` to (re-)render the card from the collected results.

## Results (pending final run)

The reproducible model-sweep above is a **separate, later phase**, and **no Percept accuracy numbers
are published yet**. This section is a deliberate placeholder — there is no Percept figure to paste
here.

External reference bars (for example TML, TransRAC) appear in
[methodology.md](methodology.md) and
[publishing-runbook.md](publishing-runbook.md) **only** as external
comparison points. They are never presented as Percept claims.

To measure precision/recall on your own labeled manifest today, see [./eval.md](../how-to/eval.md). For the
honest form of a quality claim (report the bound, not the point estimate) and what a release-grade
number requires, see [methodology.md](methodology.md)
and [publishing-runbook.md](publishing-runbook.md).
