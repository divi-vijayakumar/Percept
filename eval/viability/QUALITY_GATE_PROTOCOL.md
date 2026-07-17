# G2 Quality Gate — Pre-Registered Protocol & Stopping Rule

> **Committed BEFORE the episode-level result exists** (see git history: this file lands in its own
> commit with no results). Purpose: make "resolve the G2 quality gate" a single honest measurement,
> not "tune until it's green." Same discipline as `bars_locked` — the rule is fixed before inspection.

## What is measured (frozen)
- **Model:** `gemini-3.1-flash-lite`, **stock judge prompt, default confidence threshold — NO tuning.** This is the as-is number.
- **Substrate:** the §5a golden clips (real frames), scored at the **episode** level (not frame level).
- **Predicted-event derivation (fixed rule):** for each scene×condition, walk the clip frames in time
  order; emit **one** predicted FIRE at the **first** frame where the judge returns `satisfied=True`
  (rising-edge, one per episode); `decision_emitted_ts` = that frame's time. Never satisfied → no
  prediction. This firing rule is **frozen** for the as-is measurement.
- **Ground truth:** each scene's `GroundTruthEpisode` (pinned rising-edge `decision_eligible_ts`,
  `episode_end_ts`, `allowed_lateness_s`) from `golden_scenes.load_real_clip_episodes`.
- **Scoring:** `golden_eval.score` — Viability §5a event matching (a prediction matches ≤1 episode,
  within `[decision_eligible_ts, episode_end_ts + allowed_lateness]`; extras = duplicate FP; unmatched
  prediction = FP; unmatched episode = miss). **Event-level recall & precision.**
- **Bar:** recall ≥ **0.90** AND precision ≥ **0.90** (the locked §2 bars).

## The measurement
**ONE** for-record run. Its recall/precision is recorded **as-is**, whatever it is. Committed with the
run-stamp (git SHA, model, seed, scene list).

## Stopping rule (frozen — this is the anti-p-hacking commitment)
1. **PASS** (recall ≥ 0.90 AND precision ≥ 0.90): G2 quality = GO. Emit `launch_profile`
   (`reflex`, since G3 latency is GREEN and economics PASS).
2. **FAIL**: the number stands as the honest as-is quality. We do **NOT** tune the prompt/threshold and
   re-run on this same set to chase 0.90. The gate resolves to **REVIEW/NO-GO**, and the next move is a
   **founder decision among these three — none of which is "keep tuning until green":**
   - **(a)** Accept NO-GO on quality for this bar/workload (narrow the workload or ship deliberative with a lower, justified quality claim).
   - **(b)** ONE pre-registered improvement (prompt/threshold), evaluated **once** on a **held-out** split
     that was never used to choose the change — reported once, against this same 0.90 bar. Requires enough
     labeled episodes to split; see the power caveat.
   - **(c)** Re-examine whether 0.90 event-level is the right bar — but only with **written justification
     + a new commit that changes the locked bar**, never a silent post-hoc move.
3. **No unlimited rounds.** At most one pre-registered, held-out re-measurement. The bar is not moved
   after seeing a result without an explicit committed bar change.

## Statistical-power caveat (stated up front, not after a bad result)
The strict §5a golden set is **~6–8 usable episodes** (single-reviewer). At that n, event-level
recall/precision has **very wide confidence intervals** — one miss swings recall ~12–16%. So this
measurement **cannot be a rigorous gate on its own**; a real quality gate needs more labeled episodes
(two-annotator, more scenes). If the honest conclusion is "insufficient labeled data to decide at 0.90,"
**that is the finding** — the resolution is better labels, not tuning on n≈7.

---

## Addendum (2026-07-17) — benchmark substrate at N=10 (founder-directed, pre-registered before the run)

The golden set's labels were found **invalid** (off-timescale rising edges, single-reviewer, n=6) — un-measurable, not merely below-bar. Founder direction: measure quality on the **properly-annotated benchmark** instead of hand-labeling. Fixed BEFORE the run, no track-shopping:

- **Track (fixed):** `pt-action-e2e` = Perception Test action localization, E2E (repo benchmark harness `tools/percept_benchmark.py`). Chosen for principled reasons: clean temporal action annotations, license-clean (CC-BY-4.0), and it is already the repo's action-localization end-to-end track. **Not** selected by peeking at results.
- **N = 10**, **live** judge = `gemini-3.1-flash-lite` (as wired).
- **Metric (as-is):** the harness headline `e2e:notification_recall` / `e2e:notification_precision` (plus edge recall/precision, call-reduction, median latency), reported against the locked 0.90 bar.
- **ONE run.** Coverage (rows observed vs failed/skipped) is reported honestly — a low-coverage run is a data-readiness finding, not a hidden denominator. Same stopping rule as above: PASS → GO; FAIL → REVIEW/NO-GO, founder decision, **no tuning-to-green**.
