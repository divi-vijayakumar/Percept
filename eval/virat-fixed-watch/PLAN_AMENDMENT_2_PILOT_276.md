# Percept benchmarking plan — Amendment 2: PILOT-276 engineering path

**Status:** Founder-approved on 2026-09-09

**Applies after:** Amendment 1, F2-Z1 zone-entry Watch

All publication and OSS launch bars remain unchanged. This amendment creates a strictly labeled
engineering path; it does not convert pilot evidence into qualification evidence.

## Split B1 disposition

B1 has two independent dispositions:

- **ENGINEERING ADMIT:** rights, media/build identities, storage, source-group partitions, frozen
  polygons, and provisional official-track ground truth are sufficient for zero-paid B2–B4 work.
- **PUBLICATION-QUALIFICATION FAIL:** `276` provisional qualification occurrences are below the exact
  `299` zero-failure minimum for a one-sided 95% lower confidence bound above `0.99`. The independent
  complete-zone audit and three terminal frame-boundary adjudications are also pending.

Even if PILOT-276 observes zero failures, its one-sided 95% lower bound is only
`0.05^(1/276) = 0.9892045907834554`. It cannot establish `>=0.99` population performance, publishable
qualification, or OSS launch GO.

## Execution order and isolation

1. Execute B2, B3, and B4 with calibration and validation sources only. Qualification media, label
   rows, event times, track identities, and adjudications remain sealed from candidate construction
   and tuning.
2. Independently test the GT transformer and one-to-one scorer, including eligibility, polygon
   boundaries, left-censoring, excluded intervals, dwell/hysteresis, timely windows, ASK, duplicates,
   misses, and reconciliation.
3. Run a production-path zero-paid rehearsal and author/tune the Watch on calibration and validation.
4. Before opening PILOT-276, freeze exact source commit, clean wheel SHA-256, goal, zone manifest,
   profile, direct judge, tunables, scorer, GT version, and comparator identities.
5. Run the sealed 276-event corpus once as **PILOT-276**. Report only measured TP/FP/FN, precision,
   recall, timely recall, ASK, duplicates, misses, stage and onset-to-receiver-ACK latency
   p50/p95/p99/max, provider and all-in economy, and exact episode/context-ledger/outbox/delivery
   reconciliation. Include confidence intervals and caveats.

If the complete-zone audit is unfinished, quality metrics are explicitly provisional and exhaustive
precision/recall must not be inferred. Any candidate-affecting change after PILOT-276 is opened
permanently exposes its source groups for future qualification and requires a new never-exposed
holdout for the publication rerun.

No provider call, paid benchmark, or 72-hour soak is authorized by this amendment. Each still
requires separate explicit budget authorization and an exact reservation.

## Milestone gates

- B1 engineering admission permits B2.
- A B2 or B3 technical failure stops subsequent work until corrected and retested.
- B4 may calibrate and validate only; it must not inspect qualification labels.
- PILOT-276 begins only after the complete candidate identity is frozen and its pre-open receipt is
  committed.
- Publication qualification remains failed until a future clean, independently audited rerun meets
  the original sample-size and quality gates.
