# Derive a profile for a new environment

**The claim this command exists to make good on: a new environment should yield a _profile_, not a
code change.**

`percept-derive-profile` takes a labeled fixture directory and a **ladder** of candidate
configurations you declared, scores every rung with the same matcher `percept eval` and
`percept doctor` use, and — only if you supplied bars and a rung met them — emits a
provenance-stamped, content-addressed `WatchProfile` plus a scorecard.

```
percept-derive-profile --fixtures ./fixtures --ladder ladder.json --bars bars.json \
                       --out-profile derived.json --json scorecard.json
```

Exit codes: `0` derived or reported, `1` no rung met your bars, `2` refused (bad input, or the
configuration drifted after the bars were declared).

## What percept will not decide for you

The bars, the held-out split, the ladder and **the ladder's order** are all inputs.

- **No `--bars`** → the whole ladder is reported and *no winner is declared*. There is no default
  threshold in this command and no code path that picks a "best" rung on a metric you did not
  declare a bar on.
- **No `splits.json`** → bars are not applied. Scoring a bar on a split chosen after seeing the
  ladder is tune-to-green.
- **No rung meets the bars** → `FAIL`, and still no profile. The next move is a better candidate,
  not a lower bar.

When more than one rung meets every bar, the winner is the **earliest rung in your declared order**.
That rule is hashed into the pre-registration *before* scoring, so it cannot be chosen after the
numbers are in. Disagree with the outcome? Re-order the ladder and re-derive — the reordering
changes the `registration_hash`, which is the point.

## Two axes, because one is not enough (ADR-005)

A rung is a pair:

| axis | answers | example |
| --- | --- | --- |
| `SourceProfile` | what **kind of world** is on the other end of the stream | `physical_scene`, `screen_document`, `conversation` |
| `WatchProfile` | how to tune the **eye within that world** | `salience_stat`, `plan_mode`, cadence, gate |

`WatchProfile` is camera-shaped: its `environment` axis is a camera *mount*. Putting a source class
there (`environment: "screen_document"`) is rejected before anything is scored — it would silently
break the registry's wildcard fallback chain. Name the source on the rung's `source_profile` field.
Every rung of one ladder must name the same source profile: a ladder tunes knobs within one world,
it does not compare worlds.

## Inputs

`ladder.json` — each rung's `profile` body is a serialized `WatchProfile`
(`percept.profile_versioning.profile_to_dict`). `rungs[0]` is the baseline every delta is measured
against. Each rung reads its predictions from `<fixtures>/<fixture_id>/<fixture_dir>/`
(`fixture_dir` defaults to the rung name) — the same arm layout `percept eval` uses.

```json
{
  "source_profile": "physical_scene",
  "declared_at": "2026-07-20",
  "rungs": [
    {"name": "baseline", "description": "shipped mean-stat defaults", "profile": { }},
    {"name": "blockmax", "description": "+ localized salience",       "profile": { }},
    {"name": "window",   "description": "+ window judging + gate relax", "profile": { }}
  ]
}
```

`bars.json` — identical to `percept eval`'s bars file (`percept.eval.differential.load_bars`).
Delta metrics are measured against `rungs[0]`.

```json
{"declared_by": "ops", "declared_at": "2026-07-20",
 "bars": [{"metric": "candidate_recall", "direction": "min", "value": 0.85}]}
```

`fixtures/` — one directory per fixture with `ground_truth.json` (a `goal_shape` plus labeled
episodes) and one sub-directory per rung containing `events.json` and, optionally, `trace.json`.
`splits.json` declares `train`/`dev`/`test`; `test` is held out.

## Proving the bars did not move

Four hashes are stamped into both the scorecard and the derived profile:

- `registration_hash` — covers every rung's profile body, the ladder order, the split, the bars and
  the fixture labels. Pass it back with `--expect-registration-hash` and a changed configuration
  **refuses to score**.
- `fixture_set_hash` — the labels themselves (ids, shapes, every episode window), not paths or
  mtimes. Relabel one episode and it changes.
- `bars_hash` — the declared bars.
- `eval_run_id` — `derive-<registration_hash[:12]>`; the same inputs always name the same run.

The bootstrap seed for the confidence intervals is derived from `registration_hash`, never from
wall time, so two runs over the same inputs produce byte-identical intervals.

## Reading the output

The scorecard prints **every rung, including the losers** — a void is not data, a scored loser is.
Per rung: mean recall/precision/false-fires/cost on the held-out split, the paired delta vs the
baseline rung with its 95% bootstrap CI, and each bar's observed value and PASS/FAIL.

`derived.json` is a `VersionedProfile` (card P4) next to its `SourceProfile`. It verifies its own
content hash on load, so a hand-edited knob cannot masquerade under the old address, and it diffs
against the shipped registry:

```python
from percept.profile_versioning import diff_registries
diff = diff_registries(before, after)
print(diff.summary())
diff.inverse()   # the diff that reverts it
```

## Scope honesty

Fixtures declare a `goal_shape`, and shapes the one-to-one event matcher cannot honestly validate
(supersession, correction, settled interval, speaker-attributed claim, document identity change —
ADR-006) are **reported but never scored**, and never folded into a delta or a bar. If your
held-out split contains only unsupported shapes, the job says so and declares nothing.

## See also

`percept-derive-profile` and [`percept-optimize`](./optimize.md) are **siblings**: this command picks
the winning rung of a **ladder you authored** and emits a content-addressed `WatchProfile`;
`percept-optimize` *searches* a typed space of declared-safe tunables and emits an `OptimizationLock`.
Both refuse to declare a winner without pre-registered bars and a held-out split.

- [`optimize.md`](./optimize.md) — tune safe tunables → `OptimizationLock`.
- [`eval.md`](./eval.md) — the paired A/B (`percept-eval`) that shares this fixtures-directory layout,
  and the wider benchmark harness.
