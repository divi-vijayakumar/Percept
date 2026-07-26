# The Operating-Frontier Report (§3.2)

Status: reference — the report a 1.0.0 release publishes when it makes a quality claim.

> A point estimate from six hours of footage is not a quality claim. This report exists so that a
> claim about percept's operating point is **legible and refutable**: every headline number is an
> exact one-sided 95% **bound** with its evidence printed beside it, and every metric the evidence
> cannot support says so — and says what it would take.

Run it through the entry point that already exists. There is no second CLI:

```
percept-eval --baseline base.json --candidate cand.json --fixtures ./fixtures \
             --frontier --frontier-bars frontier-bars.json [--frontier-json out.json]
```

`--frontier` appends the §3.2 report to the paired-differential report from the implementation.
`--frontier-arm` (default `candidate`) picks which arm the frontier describes. Exit code is `1`
when a declared operating bar is not supported by the evidence, `2` on a refusal.

## 1. What the report contains

| Section | Contents |
|---|---|
| Bars | The client-supplied `false_fire_max` (per **scheduled watch-hour**) and `recall_min`. Never invented. |
| Evaluated hours | Scheduled watch-hours, of which healthy / degraded; source-hours and mean active watches per source. |
| Pooled frontier | FIRE-recall bound, assisted-recall bound, false-FIRE-rate bound, false FIREs per source-hour, coverage, ASK rate (per decision and per scheduled watch-hour), duplicate FIREs per episode. |
| Abstention ledger | fires / asks / silents / suppressed / unresolved judge work. Nothing here is netted out of the numbers above. |
| Per-archetype | One row per core gating archetype — `state`, `transition`, `brief_action`, `persistence`, `absence`, `repeated_episode` — including archetypes with **no** evidence, printed as `NO EVIDENCE` rather than omitted. |
| What the evidence supports | Per archetype, per metric: the verdict (`SUFFICIENT` / `INSUFFICIENT` / `NO_EVIDENCE` / `NOT_ASSESSED`), the observed counts, and the hours/episodes that would be required. |

## 2. The math

`percept.eval.exact_ci`, consistent with `eval-hours-costing.md` and asserted cell-by-cell against
its tables in `tests/test_eval_frontier.py`:

- **False FIREs per scheduled watch-hour** — Garwood/Poisson exact:
  `λ_upper = χ²₀.₉₅,₂₍ₖ₊₁₎ / 2T = factor(k) / T`, `factor(0)=2.996`, `factor(1)=4.744`,
  `factor(2)=6.296`. Required hours: `T = factor(k) / bar`.
- **FIRE recall / assisted recall** — Clopper–Pearson exact:
  `recall_lower = Beta⁻¹(0.05; x, n−x+1)`; for the all-hits case, `0.05^(1/n)`. Required episodes:
  the smallest `n` whose bound clears the bar at the observed miss count.

No normal approximation, anywhere. At these counts a Wald interval on a 29-for-29 run returns a
zero-width interval at 1.0; the exact bound returns 0.9019, which is the number that can be
defended.

## 3. Definitions that decide whether a number is honest

**Scheduled watch-hours** is the bar-eligible denominator. Concurrency is already inside it (three
watches armed for an hour on one camera is three scheduled watch-hours), and degraded time stays
inside it — a watch scheduled for an hour that was blind for twenty minutes still owes those twenty
minutes. Degraded time is subtracted from **coverage**, never from a rate denominator.

**Per source-hour at a stated active-watch count** is the second, deployment-economics denominator.
It is published only when the active-watch count was actually stated; a bare per-stream-hour figure
is underspecified (costing doc §4), and this report omits it rather than printing it with an
implied concurrency of 1.

**Assisted recall** is the same matcher with `predicted_actions={"fire","ask"}` — "the episode was
surfaced to the user at all", scored by the identical window rule as FIRE recall. It is never a
substitute for FIRE recall; both are printed.

**Abstentions** — ASKs, SILENTs, typed suppressions, and judge work that entered the funnel and
never produced a decision (`judge_degraded + judge_dropped`) — are itemised in the ledger. The last
of these is the one a point estimate would quietly erase.

**Zero episodes is not recall 1.0.** With no labeled positive episodes the bound is `null`, the
verdict is `NO_EVIDENCE`, and the report says what a claim would cost. Same for zero scheduled
watch-hours and the false-FIRE rate.

## 4. Fixture layout

The frontier reuses the `percept eval` fixture tree verbatim. Each `ground_truth.json` adds two
keys:

```json
{
  "goal_shape": "episode",
  "archetype": "absence",
  "exposure": {
    "scheduled_watch_hours": 9.5,
    "source_hours": 3.2,
    "active_watches": 9.5,
    "degraded_hours": 0.4
  },
  "episodes": [ ... ]
}
```

`exposure.scheduled_watch_hours` is **required** — a recall or false-fire claim with an unstated
denominator is precisely what §3.2 exists to prevent, so the loader refuses rather than assuming.
`active_watches` is watch-hours accumulated across the `source_hours` (so
`active_watches / source_hours` is the mean concurrency). `degraded_hours` may be omitted when a
`trace.json` is present: it is then read off `TraceSummary` (`watch_duration_s − source_healthy_s`),
the same object `doctor`, `eval` and the visualizer read.

Only the held-out split contributes, for the same reason the differential holds one out.

## 5. The verdict

`PASS` only when every declared bar is cleared by its **bound** in **every** core archetype. A run
that skips archetypes is `NOT_ASSESSED`, not partially passed — a partial frontier report is worse
than none, because it invites a cherry-picked read.
