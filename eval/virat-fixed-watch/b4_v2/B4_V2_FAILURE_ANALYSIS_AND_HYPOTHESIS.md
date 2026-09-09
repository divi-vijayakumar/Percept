# B4 v2 pre-implementation freeze

**Frozen before implementation and before the complete v2 calibration result.**

## Prior-candidate failure analysis

The v1 best configuration produced 13 TP, 26 provisional FP, 3 FN, and 3 duplicates on 16
provisional occurrences. Replaying the exact committed candidate against the immutable detection
caches produced these mutually exclusive FP timing classes:

- 15 fires in sources with no provisional GT occurrence. Because the complete-zone audit is pending,
  these are not asserted to be exhaustive false entries; they are provisional unmatched fires.
- 6 unrelated transitions more than two seconds from any provisional occurrence, consistent with
  short-lived association/track fragmentation.
- 2 early boundary fires, at 0.1151 s and 0.8728 s before the nearest provisional onset, consistent
  with detector-box footpoint jitter at the polygon edge.
- 3 duplicate-window fires after an already matched occurrence, including same-second adjacent track
  identities, consistent with duplicate boxes or identity fragmentation.

All three unmatched official-track occurrences had no fire within plus or minus two seconds. Their
source/onset locations were `050201_00` at 59.626233 s, `050202_09` at 26.826800 s, and `050203_00`
at 53.453400 s. The exact 21-source distribution is frozen in
[`b4-v2-failure-distribution.json`](b4-v2-failure-distribution.json).

The diagnosis rules out a threshold-only revision: v1 already swept score, association distance,
gap, and dwell, and its precision/recall tradeoff never approached the joint bar.

## Revised hypothesis

Candidate family `virat-f2-z1-persistent-track-v2` keeps the same official Faster R-CNN detections,
source-specific `0502` polygon, 1 Hz cadence, and zero-second dwell. It changes one coherent layer:
the zone-entry state machine becomes a persistent motion-aware multi-object tracker.

The tracker will:

1. suppress duplicate person boxes using a single source-independent geometric rule;
2. globally associate detections to velocity-predicted tracks rather than greedily matching only the
   previous footpoint;
3. preserve tracks across bounded detector gaps;
4. require a configurable count of observable outside samples before arming;
5. treat a configurable band around the polygon boundary as ambiguous, neither outside nor inside;
6. fire once when an armed persistent track becomes observably inside;
7. disarm until the same track is observably outside again;
8. suppress spatially local repeated fires during a bounded scene refractory window;
9. emit ASK only when an armed, confirmed track enters the ambiguous band and expires without an
   observable resolution; left-censored or never-confirmed tracks fail closed silently.

This preserves the launch Watch's logical rising edge. It does not add per-clip rules, GT-derived
regions, source-specific thresholds, provider calls, or a new detector/model.

## Timing and observability rationale

Nonzero dwell at 1 Hz requires another sampled inside observation and cannot guarantee a two-second
onset-to-ACK deadline once sampling and measured inference are included. V2 therefore freezes dwell
at `0.0` and seeks reliability from pre-entry outside continuity, not post-entry delay. Receiver ACK
timeliness remains unmeasured at calibration and cannot be inferred from detector latency.

## Acceptance criteria

On all 21 admitted exposed calibration sources, at least one predeclared configuration must have:

- provisional precision and recall both at least 0.99 (with 16 occurrences this implies 16 TP,
  zero FN, and zero provisional FP);
- zero emitted duplicate fires;
- detector inference p99 at most 1.0 s and max at most 1.5 s;
- tracker-stage p99 at most 0.01 s and max at most 0.05 s;
- no invariant or reconciliation failure.

ASK is reported exactly but has no invented numeric launch bar. Precision/recall remain provisional
until complete-zone and occlusion audit. If no configuration passes, stop at B4 without validation.

