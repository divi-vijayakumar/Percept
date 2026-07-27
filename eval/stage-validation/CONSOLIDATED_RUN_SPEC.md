# Consolidated N=100 run — arm specification

The question: **stacked, do the launch fixes beat where we started, on the whole corpus?**

Every number quoted so far was measured on a slice — the reflex fix on one dataset, the resolution
arm on 15 long-media cases, the merge window in offline simulation. This run is the one place all of
them meet the same 100 cases at once, so it is also the place a regression hiding in a slice would
finally show.

## The wins being stacked

| # | win | landed as | how the BASELINE arm turns it off | verified by |
|---|---|---|---|---|
| 1 | reflex evidence bounded to 3 frames | `_REFLEX_MAX_FRAMES = 3`, `backends/anthropic.py:178` | **hardcoded — no flag.** Harness must monkeypatch the module constant | assert the judge request carries >3 frame parts in baseline, ==3 in stacked |
| 2 | `salience_stat = block_max` | `config.py:198`, env `PERCEPT_SALIENCE_STAT` | `PERCEPT_SALIENCE_STAT=mean` | assert `Settings.gate.salience_stat` reads back the arm's value |
| 3 | incident merge window 0.7 s | `INCIDENT_MERGE_WINDOW_S`, `behavior.py:796` | **hardcoded — no flag.** Monkeypatch to `0.0` (= no merging, the pre-fix behaviour) | assert two incidents 0.3 s apart both deliver in baseline, one in stacked |
| 4 | `media_resolution = LOW` | **not landed** — harness env only | `PERCEPT_SV_MEDIA_RESOLUTION` unset (= HIGH) | assert input tokens/call ≈3560 baseline vs ≈1100 stacked |
| 5 | Haiku conductor authoring | `percept/conductor.py`, opt-in | do not call it; goals keep their corpus-declared type | assert the authored primitive differs from the declared type on ≥1 case |
| 6 | rhythm route drivable | `RhythmDrive`, `worker_can_drive()` | only reachable for `type="rhythm"` goals, which only the conductor selects — so #6 is *entailed by* #5 | assert `judge.calls == 0` on a rhythm-routed case in stacked |

**#1 and #3 are the dangerous ones.** They are module constants with no env seam, so the baseline arm
is produced by patching, and a patch that silently fails to apply would make the baseline look
identical to the stacked arm and manufacture a null result. Every arm must **assert its own
configuration took effect before the first case runs**, and record the observed value in the results
JSON. An arm that cannot prove it is configured correctly must abort, not run.

## Arms

- **A — baseline**: all six off. This is the pre-launch-work SDK.
- **B — stacked**: all six on.

Both arms: same 100 cases, same mode, same backend, same `quality()` scorer, matched per case.

## What gets reported

Pooled and **per dataset** (charades / proactiveqa / perception_test / omnipro / repcount):
recall, precision, timely recall @2 s, ASK %, FIRE %, incidents delivered, input tokens/call, cost,
provider p50/p95, deadline expiries. Plus cases better / same / worse, and **every case that got
worse, named** — a stacked win that quietly regresses one dataset is the single most important thing
this run can find, and pooling would hide it.

## Confounds to state, not hide

- **#5 changes which primitive a goal uses**, so arm B is not "the same pipeline, tuned" — it can
  route a goal to a different behaviour entirely. Per-dataset reporting is what keeps that legible.
- **repcount is the sharpest test**: it is where the merge window (#3) was measured to be dangerous,
  and where the conductor sends all 20 goals to `rhythm` (#5/#6). If the stack is going to hurt
  anywhere, expect it here.
- `count`-routed goals may not arm at all pending the detector ablation. Any case affected must be
  **excluded explicitly and named**, not silently scored as a miss.

## Budget

Cap **$12**. Arm A runs at HIGH resolution and is the expensive one. Checkpoint every case so a crash
never re-spends. Report actual provider usage, not an estimate.

## Rules

Measurement only. No threshold, window, or parameter may be adjusted to improve either arm's numbers.
If arm B loses somewhere, that is the finding.
