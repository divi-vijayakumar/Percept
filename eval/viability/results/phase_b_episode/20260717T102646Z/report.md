# Phase-B Episode-Level Quality — For-Record (frozen protocol)

**Model:** `gemini-3.1-flash-lite` · protocol `QUALITY_GATE_PROTOCOL.md` (pre-registered) · run `20260717T102646Z`
**n_episodes = 6** (approved+pinned §5a) · predictions 3 · calls 42

## Event-level result (as-is, no tuning)
- **recall 0.167** (min 0.9) · **precision 0.333** (min 0.9)
- matches 1 · misses 5 · false-fires 2
- **VERDICT: REVIEW/NO-GO**

## Stopping rule (frozen, applied)
G2 quality REVIEW/NO-GO. Per protocol: NO tune-on-eval-set. Founder decision: accept NO-GO / ONE pre-registered held-out attempt / committed bar change.

## Power caveat
n_episodes=6 (§5a single-reviewer) — wide CIs; not a rigorous gate alone.

## Per-episode
| scene | fired_at | eligible window | in-window |
|---|---|---|---|
| eval1_s02 | 0.0 | [4.64, 11.0] | False |
| eval2_s01 | None | [130.0, 131.0] | False |
| eval2_s02 | 7.0 | [90.0, 100.0] | False |
| eval2_s03 | None | [125.73, 128.0] | False |
| eval4_s01 | None | [6.81, 9.5] | False |
| eval4_s02 | 3.0 | [82.93, 134.5] | False |
