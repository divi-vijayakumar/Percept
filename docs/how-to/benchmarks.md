# Run benchmarks

Percept ships a benchmark harness with three parts: a snapshot **scorecard**, an artifact **audit**,
and a reproducible local **model-sweep** runbook. This guide is how to run them.

**No Percept accuracy numbers are published yet.** The harness is the reproducible machinery for
producing them; the numbers themselves are a separate, later phase. Everything below is runnable
today, but the results are yours to generate on your own data — do not expect a published figure.

## Prerequisite: checkout plus an installed candidate wheel

The benchmark harness is repo/dev tooling — it lives in the top-level `eval` package and does **not**
ship in the installed wheel. Qualification deliberately combines source-only benchmark tooling with
the product runtime from one exact wheel; importing `percept` from the checkout is rejected.

```bash
git clone https://github.com/divi-vijayakumar/Percept && cd Percept
python3 -m build --sdist --wheel --outdir dist packages/percept-vision
python3 -m venv .benchmark-venv
.benchmark-venv/bin/python -m pip install \
  "dist/percept_vision-1.0.0-py3-none-any.whl[bench,runtime,openrouter]"
```

Do not set `PYTHONPATH=packages/percept-vision` for a qualification run. The runner verifies the
wheel SHA-256, build stamp, metadata, import origin, and every hashed installed RECORD file in the
parent and each track worker.

Before spending provider budget, exercise the installed product paths and audit both artifacts:

```bash
make wheel-smoke \
  BENCH_PY=.benchmark-venv/bin/python \
  SDK_WHEEL=dist/percept_vision-1.0.0-py3-none-any.whl \
  SDK_SDIST=dist/percept_vision-1.0.0.tar.gz
```

This deterministic smoke covers compiled video, audio, and fused Watches, local Incident delivery,
`ask`, bounded judge timeout cleanup, and the Physical Scene/Ego Motion starter-pack bindings. It
proves SDK mechanics, not provider perception quality.

## Snapshot scorecard

The public `percept-bench scorecard` command renders an offline multi-track diagnostic with
deterministic stubs. It does not make provider claims.

```bash
PYTHONPATH=packages/percept-vision python3 -m percept.bench scorecard \
    --manifest eval/golden-v1/MANIFEST.json --out scorecard.txt
```

This is exactly what `make scorecard` runs. The old direct `--live` path is disabled because it did
not pin or execute an exact installed wheel; use `make benchmark-run` for provider evidence.

The benchmark card carries a **stamp**. It is `DRAFT` unless the run is `mode=full`, points at a pinned release
manifest, has no modeled cells, and passes coverage — only then is it `RELEASE`. Pass
`--require-release` to make CI exit non-zero on a benchmark card that renders `DRAFT`.

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

export PERCEPT_LIVE=1
export PERCEPT_VISION_BACKEND=openrouter
export PERCEPT_VISION_MODEL=qwen/qwen3-vl-32b-instruct
export OPENROUTER_API_KEY=...

# First validate one pinned case. OpenRouter qualification is intentionally
# single-worker and single-attempt so one budget ledger accounts for every call.
make benchmark-run LIVE=1 N=1 WORKERS=1 ATTEMPTS=1 \
  BENCH_PY=.benchmark-venv/bin/python \
  SDK_WHEEL=dist/percept_vision-1.0.0-py3-none-any.whl

# Run N=20 only after the N=1 artifact and accounting gates pass.
make benchmark-run LIVE=1 N=20 WORKERS=1 ATTEMPTS=1 \
  BENCH_PY=.benchmark-venv/bin/python \
  SDK_WHEEL=dist/percept_vision-1.0.0-py3-none-any.whl

make benchmark-collect RUN_DIR=eval/local-benchmark/runs/<run-id>   # re-render a scorecard
```

`benchmark-run` writes a run directory under `eval/local-benchmark/runs/`; pass it to
`benchmark-collect` to (re-)render the benchmark card from the collected results.

The OSS scorecard is deliberately capability-scoped. The PT action/runtime-v2 lane is the only
provider-quality candidate. Physical Scene and Ego Motion starter packs are available with
mechanics/conformance coverage but no pack-specific accuracy, latency, or cost claim. Audio and
fused Watches are preview APIs with bounded deterministic runtime coverage; the legacy acoustic
direct-judge experiment is diagnostic and cannot qualify them.

## Results (pending final run)

The reproducible model-sweep above is a **separate, later phase**, and **no Percept accuracy numbers
are published yet**. This section is a deliberate placeholder — there is no Percept figure to paste
here.

External reference bars (for example TML, TransRAC) appear in
[../reference/benchmark-methodology.md](../reference/benchmark-methodology.md) and
[../reference/publishable-benchmark.md](../reference/publishable-benchmark.md) **only** as external
comparison points. They are never presented as Percept claims.

To measure precision/recall on your own labeled manifest today, see [./eval.md](./eval.md). For the
honest form of a quality claim (report the bound, not the point estimate) and what a release-grade
number requires, see [../reference/benchmark-methodology.md](../reference/benchmark-methodology.md)
and [../reference/publishable-benchmark.md](../reference/publishable-benchmark.md).
