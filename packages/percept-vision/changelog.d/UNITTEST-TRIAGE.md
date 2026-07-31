### Fixed — 64 defects from the unit-test triage, and the gates that were meant to catch them

A 65-bug audit (64 distinct; one duplicate) landed as thirteen changes. Zero were rejected outright;
two *halves* were rejected as documented intent. The suite moves from **25 failed / 6146 passed /
48 xfailed** to **0 failed / 6357 passed / 3 xfailed**, and **no `pytest.mark.xfail` remains** — the
45 strict pins are all either passing or rewritten.

**Three of the four instruments were blind, and were fixed first.**

- `make parity` is CI-wired and three shipped docstrings call it release-gated for *bit-identity*.
  Mutation testing showed **6 of 8 salience forks and 1 of 5 ego forks passed all six frozen vectors
  with `max|delta| = 0.000e+0`**, at a `1e-9` tolerance that is ~6.4e6 ULP at the magnitudes involved.
  The vectors were structurally blind: 48x36 divides exactly by the [8,6] grid, 13x11 carries
  near-uniform diff, the one ROI vector's bounds were integral, and no vector had an active area
  smaller than the grid. Now **81 checks at exact equality**, with families covering block boundaries,
  percentile rank, region rounding and origin, empty blocks, ego shift-search extent, and the wake
  fusion. Harnesses report 18/19 motion+ego forks and 9/9 wake forks caught. This is what surfaced the
  `percept-edge` acoustic-threshold divergence.
- `hero_track_coverage` was pinned at `1.0` because `RhythmHero.observe` incremented its denominator
  *after* an early return, so the blind-count VLM escalation keyed on `coverage < _TAU_KP` could never
  fire. **`_TAU_KP` was therefore never calibrated** — its input was constant.
- `capture_plan_unmet` was suppressed by the very NaN it exists to report, because `max(0.0, nan)`
  is `0.0` and `required_fps` collapsed.

**Six shared authorities replace N drifted copies.**
`contracts/verdict_validation.py`, `_snapshot.py`, `_numeric.py`, `_streamtime.py`, `_time_compare.py`,
`runtime/_bounds.py`. Each closes a rule that had been reimplemented and had diverged: verdict
well-formedness (five copies, only `conformance.assert_verdict` complete), snapshot restore (five, only
`GoalRegistry.restore` validate-then-swap), the `clamp01` rule (three, two of which let `NaN` through
while the JS twin did not — so the runtimes forked on admission), the stream-time firewall, the
float-time boundary, and the bounded-recency map (four hand-rolled LRUs, only one traced its drops).

**The firing path.** `agent._is_valid_verdict` checked three terms where its `judge_pool` twin checked
six, so `gate.classify` returned `"yes"` for `nan`, `inf`, `7.5` and `True` — `nan < threshold` is
`False`. Reachable from a live provider: every backend builds `Verdict(confidence=float(...))` with no
clamp and `json.loads` accepts bare `NaN`. ADR-001 §10.4 requires normalization at *one* place, which
`percept/contracts/verdict_validation.py` now is; `gate.classify` carries one non-finite term as
defence in depth, provably behaviour-preserving on every finite input. The inline direct-judge path,
which reached `classify` having passed **no** validator at all, is now validated and degrades.

**Gate state.** Six parallel per-key dicts became one `_KeyState` record, so `reset_entity` can no
longer orphan dwell state, the LRU counts the real keyspace, and a lost nesting level in a snapshot is
unrepresentable rather than merely detectable. `snapshot()` is v2 self-describing records plus a v1
mirror for older readers; `restore()` validates then swaps and never raises on a restart path.

**Also fixed:** the cadence envelope (validated in four places, and `SensePolicy` — the type every
consumer runs on — was the unvalidated one, so `base_fps=0` meant permanent silent blindness);
`GatedWatchLoop` compiling `mean@0.10` where the deployment configured `block_max@0.20`, so every
number measured through that loop described a configuration nobody deployed; env overrides discarded
with no diagnostic while `tunables.py` already declared every bound and vocabulary; an appearance
descriptor that discarded the value channel, so black and white subjects re-identified as one entity at
cosine 1.0; a rhythm amplitude EMA seeded from one sample that made seven reps read as one and made the
count depend on rep *order*; an absolute anti-fidget gate expressed in units a `body_scale` clamp
silently redefined, turning it into a distance gate (near 8 reps, far 0); one `core.rhythm` slot with
two owners that doubled counts and discarded them mid-clip; and interest intervals scored from a gauge
that never drove the admission, so replay and live disagreed about the same admitted frame.

### Changed — behaviour worth a release note

- **A non-finite timestamp now raises** out of `MotionGate.evaluate` and
  `SamplingController.should_process`. It previously bypassed the cadence brake — every comparison
  against `NaN` is `False` — and was then stored, poisoning that goal for the rest of the run. A
  *backwards* timestamp does not raise: it is a recoverable source discontinuity (reconnect, seek), so
  the affected timing state is dropped, `StreamTimeRewindWarning` is emitted, and the run continues.
- **A non-retryable provider error no longer moves the circuit breaker.** The wrapper recorded a
  breaker failure *before* classifying the error, so a locally-raised budget refusal — raised before any
  socket was touched — opened the breaker against a healthy provider and destroyed the real reason.
  Consequence: a provider returning a permanent error on every call (a 401) now has **no backoff**;
  each call costs one attempt and the real exception reaches the caller.
- **`gate_on="fused"`** is a new additive member. A deployment relying on a `wake_policy` overriding
  `gate_on` must now set it explicitly; a deployment with no `wake_policy` is byte-for-byte unaffected.
- **A malformed `wake_policy` raises** instead of degrading to legacy `max(signals)`.

### Two re-baselines required before any benchmark number is comparable

- **`_TAU_KP`** (`eval/bench/repcount.py`) — see above; it is being tuned for the first time.
- **`identity.appearance_threshold`** — the old descriptor's near-perfect same-subject scores were an
  artifact of ignoring brightness (same subject at +10% lighting: 0.9994 before, **0.4179** now). At
  0.8 the fix trades the merge bug for missed re-acquisitions. Calibration needs real person tracks the
  repo does not have, and the `vbins` sweep is non-monotonic — hard bin edges, not a tuning plateau.
