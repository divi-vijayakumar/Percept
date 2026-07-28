# Model comparison — directional cross-model sweep

A **directional** comparison of vision backends on the Percept E2E path: the same edge gate, cadence,
and firing contract on every arm — **only the model swaps**. Backends compared: `gemini-3.5-flash`
(REST), `qwen3-vl-32b` (OpenRouter), `gemini-3.1-flash-live` (realtime).

> ⚠️ **Directional, small sample (N=8 cross-task; N=10 pt-action). NOT a release claim.** These are
> exploratory numbers to guide model selection, not a published benchmark. The reproducible,
> release-grade scorecard is a separate effort (see `docs/benchmarks/methodology.md`). Cost is
> measured `candidate_calls`; token figures use a documented proxy (see `build.py`).

## Files
- `build.py` — regenerates `scorecard.md` + the two PNG curves from `_data.json`.
- `_data.json` — the committed data snapshot (per-model, per-track `cells` + coverage). Makes this
  directory reproducible from tracked files alone.
- `scorecard.md` — the full per-track metric tables (generated).
- `cost-quality.png`, `latency-quality.png` — quality-vs-cost / quality-vs-latency curves (generated).

## Reproduce
```bash
python eval/model-comparison/build.py          # rebuild scorecard.md + PNGs from _data.json
```
Charts need `matplotlib` (`pip install matplotlib`); without it the script still writes `scorecard.md`.

## Refresh the snapshot from new runs
`_data.json` is a snapshot of `eval/local-benchmark/runs/` (git-ignored, local-only). After a fresh
sweep, rebuild the snapshot and outputs:
```bash
python eval/model-comparison/build.py --refresh
```

## Reading it
Compare **per completed clip**, not totals — a model that completes fewer clips does less total work,
which is coverage, not efficiency. `scorecard.md` calls out where that artifact bites.
