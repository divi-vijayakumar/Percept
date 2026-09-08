# Percept benchmarking plan — Amendment 1: zone-entry launch Watch

**Status:** Founder-approved on 2026-09-08
**Supersedes:** the F1 launch role in §§1, 5.1, 10, 11, and decision 12.1 of the plan whose SHA-256 is
`2cf6790f7bba949063c3024bc8b7389156a3e42abda70c3bf465b1474aabc040`.

All unmodified plan controls remain in force, including the metric bars, source-group isolation,
candidate identity rules, earliest-failing-gate stop rule, separate non-blocking scorecards, and
separate paid-budget authorization.

## Revised launch claim

The sole fixed-camera quality launch Watch is now F2-Z1:

> Fire when an eligible person enters the monitored area.

F2-Z1 plus its relevant production-path runtime-reliability gates blocks the narrow OSS 1.0 claim.
Original whole-frame F1 remains a non-blocking adversarial/stress scorecard and cannot be reported as
qualified without independently exhaustive whole-frame labels.

## Frozen F2-Z1 scoring contract

- A source-group-specific monitored polygon is authored from reference imagery without opening
  object/event labels, represented in normalized image coordinates, hashed, and frozen with the
  candidate.
- A person is eligible on a frame when the official class is `Person`, the bounding-box footpoint is
  inside the polygon, bounding-box height is at least `0.025 * frame_height`, and the annotation is
  not `mostly` occluded.
- A qualifying occurrence begins on the first eligible outside-to-inside transition. A positive state
  at source start is left-censored and is not an entry.
- Unusable frames and declared ignore regions are excluded, never negative. A boundary transition
  touching an excluded interval is excluded unless a clean outside observation precedes it and a
  clean inside observation follows it.
- Boundary hysteresis/dwell candidates are predeclared as `{0.0, 0.25, 0.5}` seconds and may be
  selected on validation only. The selected value is applied identically to truth and predictions
  and becomes part of the candidate identity.
- Official tracks seed occurrences. Before a source can qualify, a label-isolated reviewer must audit
  the complete monitored area for missed qualifying entries and unusable intervals. Any real eligible
  but officially untracked entry is added to truth; it is never scored as a false positive merely
  because the official track is absent.
- Qualification labels and adjudications remain hidden from candidate authors until code, goal,
  prompt, zone, profile, judge, tunables, GT transform, scorer, comparator, and wheel are frozen.

The unchanged event bars remain precision, recall, and timely recall `>= 0.99`, receiver ACK within
`2.0 s`, conservative all-in economy `<= $0.10/source-hour`, zero duplicate logical fires and
deliveries, and exact runtime reconciliation.

## Revised source partition

Physical scene remains the minimum source group. The currently exposed groups `0002`, `0400`, and
`0401` remain calibration-only. The official Release 2.0 legacy annotation package adds object-track
inventory for eleven physical groups; no qualification labels were opened during inventory.

- Calibration: exposed groups `0002`, `0400`, `0401`.
- Validation reservation: clean group `0100`.
- Qualification reservations: clean groups `0000` and `0500`, treated as distinct sites.
- Reserve/adversarial pool: `0001`, `0101`, `0102`, `0502`, `0503`.

Reservations become final only after selected media hashes/clocks, zone feasibility, annotation
coverage, and label-isolated event-rate counts pass B1. If any reserved group fails, replace it only
with a still-sealed reserve group and append the disposition to the exposure registry.

## Revised milestones

- **B0:** freeze F2-Z1 as the sole launch Watch; retain F1 as non-blocking stress evidence.
- **B1:** verify selected media and annotations; freeze source groups and exposure registry; confirm
  that event-only independent audit can make zone-entry occurrences exhaustive.
- **B2:** implement and independently verify person-track → eligibility → polygon-entry → rising-edge
  transformation, excluded intervals, left-censoring, hysteresis, and one-to-one scoring.
- **B3:** unchanged zero-paid production-path rehearsal.
- **B4/B5:** calibrate, validate, and qualify F2-Z1. F1 results remain separate and non-blocking.
- **B6–B9:** unchanged except every reference to the launch-blocking F1 candidate means F2-Z1.

## Immediate NO-GO interpretation

“Missing or non-exhaustive GT negatives” now applies within the frozen monitored area and eligibility
contract, not to every visible person anywhere in the frame. Missing qualifying zone-entry events,
unresolved unmatched fires, label leakage, or absent outside-state evidence still cause immediate
NO-GO.
