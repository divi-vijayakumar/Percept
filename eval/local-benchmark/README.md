# Local benchmark workspace

This folder is the local control room for reproducible Percept benchmark runs.

Tracked files in this folder explain how to organize the run. Heavy datasets,
model caches, logs, and generated scorecards must stay local-only.

## Local paths

Set these in the ignored repo `.env`:

```bash
PERCEPT_BENCH_DATA_DIR=eval/bench-staging
PERCEPT_BENCH_RUNS_DIR=eval/local-benchmark/runs
PERCEPT_BENCH_CACHE_DIR=eval/local-benchmark/cache
```

`eval/bench-staging/` is the canonical local data cache for public benchmark
datasets. It is ignored because it can contain large media and copied benchmark
repos. `eval/local-benchmark/runs/` stores one timestamped run folder per
benchmark invocation.

## Runner

```bash
python3 tools/percept_benchmark.py prepare
python3 tools/percept_benchmark.py fetch --videos 12 --repcount 12
python3 tools/percept_benchmark.py fetch-charades --n 100
python3 tools/percept_benchmark.py fetch-proactiveqa --n 100
python3 tools/percept_benchmark.py manifest-candidate --pt-action 200 --pt-sound 100
python3 tools/percept_benchmark.py run --live --n 8 --workers 2 --start-delay-s 2 --exclude counting
python3 tools/percept_benchmark.py run --live --n 8 --workers 2 --start-delay-s 2 --exclude counting --background
python3 tools/percept_benchmark.py collect --run-dir eval/local-benchmark/runs/<run-id>
```

The runner invokes the SDK benchmark tracks in isolated subprocesses, retries
failed tracks, logs stdout/stderr per track, and emits a scorecard from whatever
track results are available. Runs resume by default: if a valid
`results/<track>.json` already exists, the track is skipped instead of rerun.
Use `--no-resume` only for deliberate overwrites.

For API-backed runs, keep `--workers` at the concurrency your quota allows and
use `--start-delay-s` to stagger track start times. Background runs write
`background.json`, `benchmark.log`, per-track logs, checkpoint events, and
atomic result JSON files into the run folder, so interrupted runs can be resumed
or collected without losing completed tracks.
