---
title: Run an optimization
updated: 2026-07-23
applies_to: percept-vision 1.0.0
status: stable
---
# Run an optimization

`percept-optimize` searches a small set of **safe numeric tunables** and, when a candidate clears
your bars, emits a hashed `OptimizationLock` the resolver enforces. It is deterministic and offline:
no network, no model calls, no wall-clock reads. It sits on top of `percept-derive-profile` and
reuses every pre-registration, held-out-split, and bars discipline verbatim.

## optimize vs derive-profile

They answer different questions, so reach for the right one:

| Command | Question | You supply | It emits |
| --- | --- | --- | --- |
| [`percept-derive-profile`](../sources/source-profiles.md) | which **hand-authored profile** wins? | a ladder of candidate profiles | a `VersionedProfile` |
| `percept-optimize` | within one fixed **shape**, which value of a few safe numeric knobs scores best? | a template + a search space | an `OptimizationLock` |

`percept-optimize` never invents the shape — that stays an operator-declared template. It expands
`(template, search-point)` candidates into a ladder and hands them to `derive_profile` unchanged, so
the "no bars met, no artifact" refusal is inherited, not re-decided.

## Command

```bash
percept-optimize --fixtures ./fixtures --space optimizer.json --bars bars.json \
                 --plan watchplan.json --split test \
                 --strategy grid|random --n-random 8 --seed 0 \
                 --out-lock lock.json --json report.json
```

- `--space` is the optimizer document (`.json`/`.yaml`): the templates plus the search space.
- `--bars` is your bars file; omit it to report every candidate and declare no winner.
- `--plan` folds a `WatchPlan`'s content hash in as `base_plan_hash`.
- `--strategy` overrides the document's own strategy; `--n-random` / `--seed` size a random search.
- `--out-lock` writes the `OptimizationLock` only when a winner was declared.

**Exit codes:** `0` lock written / reported, `1` no candidate met the bars, `2` refused (malformed
input, a guardrail-forbidden or unsafe path, or config drift).

## Guardrails: five things a scored number must never move

The guardrail check runs **before** `percept.tunables.REGISTRY` is even consulted — a hard-coded
deny-list, so a future flag flip cannot silently hand the optimizer a forbidden path. It never
optimizes:

- **gate vocabulary** — `Settings.gate.*`, `WatchProfile.gate.*` (the Gate is the sole firing
  authority);
- **goal meaning** — the entire `Goal.state.*` namespace (conditions, `fire_when`, cost, re-arm);
- **consent / envelope** — `Settings.consent.*` and the scripted-brain switch;
- **retention / egress** — `Settings.episodes.window_s`, `Settings.events.preroll_max_bytes`, and
  coverage floors.

After the deny-list, a path must additionally be **registry-known and `safe_to_optimize=True`** to
be searchable. The config **shape** is fixed throughout — the optimizer moves values, never structure.

## Determinism

The bootstrap seed derives from the registration hash, never wall time, so byte-identical inputs
produce byte-identical intervals and the same winning search-point on every rerun. The lock stamps
`locked_hash = content_hash(values)`, and the resolver treats a later deployment/env change touching
a locked field as a loud diagnostic rather than a silent outranking.

## Both commands refuse if the config moved

If the configuration changed after your bars were declared, scoring it is tuning, not evidence — so
both `percept-optimize` and `percept-derive-profile` refuse (exit `2`). Pin the expected state with
`--expect-registration-hash <hash>`; a mismatch raises before anything is scored.

## Fixtures

Both jobs read the same fixtures directory layout — one directory per fixture with
`ground_truth.json`, per-arm `events.json` / `trace.json`, and an optional `splits.json` whose
`test` split is held out. See [./eval.md](../evaluation/evaluate-your-data.md) for the full layout, and
[./derive-a-profile.md](../sources/source-profiles.md) for the ladder/bars document shapes `--space` and
`--bars` build on.
