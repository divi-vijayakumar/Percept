# Stage-1 ADMISSION validation — SV-1.4 / SV-1.5 / SV-1.7

**Runtime under test:** `percept` 1.0.0 (launch wheel,
`/private/tmp/percept-launch/oss-final-venv-bba572a/lib/python3.12/site-packages/percept`) ·
Python 3.12.8
**Harness:** `eval/stage-validation/sv1_admission.py`
**Machine-readable:** `eval/stage-validation/results/sv1_admission.json`
**Mode:** fully offline (scripted `FakeVision`, in-memory `SourceEnvelope` lists), zero provider
calls, zero network. Verified by running the suite with `socket.connect` monkey-patched to raise —
no attempt was made. Byte-identical JSON across repeat runs (deterministic).

| Test | Verdict | Checks | What it establishes |
|---|---|---|---|
| **SV-1.4** Dwell correctness | **PASS** | 53/53 | Hysteresis, both debounce counters, and the expiry split/merge boundary are exact. |
| **SV-1.5** Monotonous dwell | **FAIL** | 13/20 | Holds perfectly at the engine layer and on a constant-motion live positive; **flaps badly on the live path whenever the admission *cause* changes**. |
| **SV-1.7** Per-Watch attribution | **PASS** | 11/11 | Exact attribution across 4 Watches on one source; distractor dispatches nothing. |
| **CHAR** State-transition table | **PASS** | 29/29 | Tier-0 admission is **not** a binary — it is a 6-state machine. |

Overall: **7 failed checks, all in SV-1.5, all the same root cause.**

---

## Headline: tier-0 admission is not a binary

The direct answer to "is tier-0 in/out, or is there a dwell state?" — empirically derived by
feeding a scripted score sequence to a real `InterestIntervalEngine`
(`enter_threshold=0.500`, `exit_threshold=0.250`, `enter_debounce=2`, `exit_debounce=2`) and
recording what came out. Every row was also cross-checked against an independent reference model
written from the docstring alone (`ReferenceDwell` in the harness); the engine matched it exactly
on all 13 rows.

| # | score | input | state | events emitted |
|---:|---:|---|---|---|
| 0 | 0.125 | below `exit_threshold`, closed | `suppress` | — |
| 1 | 0.375 | inside the hysteresis band, closed | `suppress` | — |
| 2 | 0.500 | at `enter_threshold` (1 of 2) | `accumulate` | — |
| 3 | 0.375 | band sample resets the accumulator | `suppress` | — |
| 4 | 0.500 | at `enter_threshold` (1 of 2, again) | `accumulate` | — |
| 5 | 0.500 | at `enter_threshold` (2 of 2) | `admit` | `interest.entered` |
| 6 | 0.500 | still above `enter_threshold` | `dwell` | `interest.updated` |
| 7 | 0.375 | inside the hysteresis band, **open** | `dwell` | `interest.updated` |
| 8 | 0.250 | at `exit_threshold` (1 of 2) | `exiting` | — |
| 9 | 0.375 | band sample resets the exit accumulator | `dwell` | `interest.updated` |
| 10 | 0.250 | at `exit_threshold` (1 of 2, again) | `exiting` | — |
| 11 | 0.250 | at `exit_threshold` (2 of 2) | `exit` | `interest.exited` |
| 12 | 0.125 | below `exit_threshold`, closed again | `suppress` | — |

**8 of 13 samples emit no event at all** while still changing internal state. Six distinct states
are observable through the public API: `suppress`, `accumulate`, `admit`, `dwell`, `exiting`,
`exit`. Rows 2→3→4 prove `accumulate` is a real buffered state that a single sub-threshold sample
discards; rows 7 and 9 prove the hysteresis band *holds* an open interval that the same score would
never have opened.

---

## SV-1.4 — Dwell correctness — **PASS** (53/53)

All assertions are exact equalities. Boundary probes use `math.nextafter`, so they are as tight as
IEEE-754 allows; all fixed constants (`0.5`, `0.25`, `0.375`, `0.125`, `2.0`) are binary-exact, so
no float drift can flatter the result.

**A. Hysteresis is enforced at construction.** `exit_threshold == enter_threshold` and
`exit_threshold > enter_threshold` both raise `ValueError`
(`percept/interest.py:129-130`).

**B/C. Threshold boundaries are inclusive on both sides, exactly.**

| probe | result |
|---|---|
| `score == enter_threshold` from closed | admits |
| `score == nextafter(enter_threshold, 0)` | does not admit |
| 4× `score` inside the band from closed | never admits |
| open + `score == exit_threshold` | exits |
| open + `score == nextafter(exit_threshold, 1)` | stays open, updates |

**D. Zero flap.** 41 samples oscillating `0.500 → 0.375 → 0.500 …` across the *enter* threshold
while an interval is open: **1 enter, 40 updates, 0 exits, 1 interval id.** A single-threshold
comparator at 0.500 would have produced **40 transitions** over the identical input. The
counterfactual is recorded in the JSON (`measurements.zero_flap`). The mirror case — oscillating
between `0.375` and `nextafter(0.5, 0)` from closed — emits **zero events**.

**E. `enter_debounce` is exact**, verified at n = 2, 3, 5:
- `n-1` consecutive above-threshold samples emit nothing;
- the `n`-th emits exactly one `interest.entered`;
- a single hysteresis-band sample **resets** the accumulator (`n-1` + band + `n-1` still emits
  nothing), and so does a below-exit sample.

**F. `exit_debounce` is exact**, same three values:
- `n-1` consecutive at-or-below-exit samples emit no exit and `engine.active` is still `True`;
- the `n`-th emits exactly one `interest.exited`;
- a band sample resets the exit accumulator and re-emits an `interest.updated`.

**G. `expiry_s` splits vs merges exactly at the boundary.** With `expiry_s = 2.0` and the last
sample at observed `4.0`, the deadline is exactly `6.0`:

| gap | outcome | interval ids |
|---|---|---|
| `nextafter(6.0, 0)` | **merge** — one `interest.updated` | 1 |
| `6.0` (exactly) | **split** — `interest.exited(expired)` + `interest.entered`, same `observe()` call | 2 |
| `nextafter(6.0, +∞)` | **split** | 2 |

`advance(observed_at=...)` (observer clock only) fires at, and not one ULP before, the same
deadline. A partial enter-debounce is correctly discarded across an expiry gap rather than bridging
it (`percept/interest.py:478-480`).

**H. Differential against an independent oracle.** A seeded integer-LCG score walk of 2 000 samples
(65 binary-exact levels spanning both thresholds) driven through 4 debounce configurations —
`(1,1)`, `(1,3)`, `(3,1)`, `(4,4)`. The engine's complete event-kind stream equalled the reference
model's, sample for sample, in all four: **5 195 events compared, zero divergences.**

---

## SV-1.5 — Dwell under a monotonous signal — **FAIL** (13/20)

**Pass bar:** the interval remains open for the full positive; **zero** spurious re-enters.

### Engine layer (`InterestIntervalEngine`) — passes cleanly

| case | samples | enters | updates | exits | interval ids |
|---|---:|---:|---:|---:|---:|
| M1 constant score exactly at `enter_threshold` | 300 | 1 | 299 | **0** | 1 |
| M2 onset then a 299-sample plateau *inside the hysteresis band* | 300 | 1 | 299 | **0** | 1 |
| M3 constant positive with `expiry_s=2.0`, sampled every 1.0 s | 300 | 1 | 299 | **0** | 1 |

M2 is the important one: a signal that decays to a plateau *below* the enter threshold but above
the exit threshold holds indefinitely. The hysteresis band is doing exactly its job.

### Live layer (`percept.run`) — this is where it breaks

Measured on the real worker with an `InterestSubscription`, 8×8 gate frames, one Watch:

| case | units | enters | updates | exits | **re-enters** | exit reasons |
|---|---:|---:|---:|---:|---:|---|
| **M4** constant above-gate motion (0.235/unit), ordinary Watch | 32 | 1 | 30 | 1 | **0** | `end_of_stream` ×1 |
| **M5** onset then a genuinely static positive, ordinary Watch | 32 | 1 | **0** | 1 | 0 | `signal_lost` ×1 |
| **M6** same static positive, Watch with the coast route armed | 32 | 3 | 29 | 3 | **2** | `admission_changed` ×2, `end_of_stream` ×1 |
| **M7** oscillating monotonous positive, coasting Watch | 60 | **40** | 20 | **40** | **39** | `admission_changed` ×39, `end_of_stream` ×1 |

**M4 passes.** A monotonous positive whose *pixels keep changing by a constant amount* — a fan, a
traffic lane, someone pacing — holds one interval for the entire 32-unit run with zero re-enters.
This is the case the pass bar was written for, and it is clean.

**M5 fails, and it is the operationally common case.** A positive where the scene *stops changing* —
someone standing still, a paused screen share, a person asleep in frame — collapses after a single
unit. The interval opens on the onset and closes with `signal_lost` on the very next unit. Dwell
coverage: **1 of 31 post-onset units (3.2%)**. There is no coast for an ordinary Watch: `_capture_plan`
returns `(base_fps, None, 0.0)` unless a goal carries a `sense.brief_window_s` prior, so the worker's
`coasts` dict is empty and `coast_admit` is permanently `False`
(`percept/runtime/worker.py:2425`, consumed at `percept/runtime/worker.py:6035-6040`).

**M6 and M7 fail on the same defect** — see below. M6's *coverage* is actually complete (all 32
units are admitted, 33 of them via `coast`), but it is delivered as **three separate intervals with
three different `interval_id`s**, which is not the same thing as one dwell. M7 is the severe case:
**40 enter/exit pairs over a single continuous 60-unit positive.**

### Defect: a cause flip tears down and re-mints the interval

`percept/interest.py:1165-1180` — `LiveInterestTracker.observe`:

```python
if opened is not None and (
    opened.admission.cause != admission.cause
    or opened.admission.admission_id != admission.admission_id
    or opened.admission.detector_id != admission.detector_id
):
    events.append(self._close(key, reason="admission_changed", ...))
    opened = None
```

Any change in *why* a Watch was admitted closes the open interval and opens a new one with a fresh
`interval_id`. On a coasting Watch the cause alternates between `motion` (the gate cleared
threshold) and `coast` (the fovea cadence floor fired on a quiet unit) as the signal crosses the
gate — which is what *every* real monotonous scene does. The `admission_id` and `detector_id` flip
with it (`tier0.signal.motion` / `motion_score.mean` ↔ `runtime.admission.coast` / `runtime.coast`,
built at `percept/runtime/worker.py:6132` and `:6137`), so all three equality tests fire together.

Contributing site: `percept/runtime/worker.py:6035-6054`, where the coast route appends `"coast"` as
the cause for any goal the `SamplingController` admits on this unit, with no memory of the cause the
Watch's currently-open interval was entered under.

**Severity.** `admission_changed` is a declared member of `LiveInterestExitReason`
(`percept/interest.py:45`), so this is *specified* behaviour, not a crash. But a consumer asking the
obvious question — "was this Watch continuously interested?" — cannot answer it from the lifecycle
stream: on a coasting Watch a single continuous positive is delivered as 40 intervals. The typed
exit reason makes the flap *diagnosable*, not absent. Any consumer that treats
`interest.entered` as a rising edge will fire 40 times on one event.

**Also noted (not scored):** in M6 the coasting Watch emits `interest.entered` at source unit **0**
with `score=0.0` and `admission_cause="coast"` — before the gate has any predecessor frame to diff
and therefore before any evidence exists at all. The coast floor admits on a cold start.

### Verdict

SV-1.5 **FAIL**. The dwell state machine itself is sound (M1–M3, M4). The failure is entirely in
the live admission layer above it, and it splits in two:

1. **No dwell for a static positive on an ordinary Watch** (M5) — nothing keeps a Watch admitted
   once the pixels stop moving.
2. **Cause-flip teardown** (M6, M7) — the mechanism that *does* provide dwell (`coast`) destroys
   interval identity every time it hands off to or from `motion`.

---

## SV-1.7 — Per-Watch attribution — **PASS** (11/11)

Four Watches on one 8-unit source (8×8 frames, left/right halves driven independently), all
admission expectations recomputed independently inside the harness from `motion_score` over the raw
frames — never read back from the runtime.

| Watch | admission route | threshold | admits on units |
|---|---|---:|---|
| `w-left` | `region_hint` = left half | 0.100 | 1, 4, 6 |
| `w-right` | `region_hint` = right half | 0.100 | 3, 6 |
| `w-any` | full-frame mean | 0.100 | 1, 3, 4, 6 |
| `w-distractor` | `motion_bias=0.95` → threshold pinned to 1.0 | **1.000** | **none** |

The fixture is self-checked: its maximum full-frame score is 0.784, so the distractor's 1.0
threshold is genuinely unreachable rather than merely untested.

- **Exact stream match.** All 17 emitted lifecycle events matched the independently derived
  expectation on `(watch_id, kind, source unit, reason, score)` — including the contract that an
  `interest.exited` reports the **last admitted** unit's score, not the closing unit's
  (`percept/interest.py:1281-1300`).
- **Correct threshold per Watch.** Every event carried its own Watch's compiled threshold; zero
  mismatches.
- **Zero cross-dispatch.** No `interval_id` was ever shared between two Watches.
- **Distractor is silent.** Zero lifecycle events, zero incidents.
- **Incident attribution.** With a `satisfied=True` scripted verdict, incidents fired only for
  `w-any`, `w-left`, `w-right` — exactly the Watches that admitted. `w-distractor` produced none.
- Interleaving is correct under simultaneous admission: at unit 6 both halves move and all three
  real Watches enter on the same unit, each with its own score.

### API-contract note

The brief refers to `percept.run(on_interest=...)`. **That parameter exists but always rejects.**
`run(..., on_interest=<callable>)` raises `InterestCallbackUnsupportedError`
(`percept/runtime/worker.py:6723` and `:6744`) — callbacks are deliberately not a live
lifecycle transport. The supported path, and the one this harness drives, is
`run(..., interest_subscription=InterestSubscription(), interest_session_id=...)` consumed by async
iteration in a separate task. This is asserted as a check in SV-1.7 (case `D.api-contract`) so the
rejection stays pinned.

---

## Defects

| # | Location | Description | Severity |
|---|---|---|---|
| 1 | `percept/interest.py:1165-1180` (with `percept/runtime/worker.py:6035-6054`, `:6132`, `:6137`) | A change in admission *cause* (`motion` ↔ `coast`) closes the open interval as `admission_changed` and mints a new `interval_id`. On a coasting Watch over a monotonous positive this produced **40 enter/exit pairs across one continuous 60-unit positive** (39 `admission_changed` exits). Interval identity does not survive the handoff between the two routes that jointly provide dwell. | **High** — specified but breaks "is this Watch continuously interested?" for any consumer. |
| 2 | `percept/runtime/worker.py:2425` (consumed at `:6035-6040`) | Ordinary Watches get no coast controller at all (`_capture_plan` returns `None` without a `sense.brief_window_s` prior), so a positive whose scene goes static closes after one unit. Measured dwell coverage on a 32-unit static positive: **1 of 31 post-onset units.** | **Medium** — by design, but the design leaves the common static-positive case with no dwell. |
| 3 | `percept/runtime/worker.py:6035-6054` | A coasting Watch emits `interest.entered` on source unit 0 with `score=0.0`, before the gate has a predecessor frame and therefore before any evidence exists. | **Low** — observational. |

No defect was found in `InterestIntervalEngine` itself. Across 53 exactness checks, a 13-row state
table, and a 4-configuration × 2 000-sample differential against an independent oracle, it matched
its documented contract without a single divergence.

## Reproduce

```
/private/tmp/percept-launch/oss-final-venv-bba572a/bin/python eval/stage-validation/sv1_admission.py
```

Exit code 0 when every check passes, 1 otherwise (currently 1, on SV-1.5). The harness is also
importable — `run_all()` returns the structure written to `sv1_admission.json`.
