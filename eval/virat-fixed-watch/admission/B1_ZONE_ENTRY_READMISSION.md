# B1 zone-entry readmission report

Status: **FAIL — stop before B2**

Recorded: `2026-09-09T08:34:15Z`

The founder-approved F2-Z1 amendment resolves the original mismatch between whole-frame occupancy
and VIRAT's Background/Unannotated semantics. The effective launch question is now **“Fire when an
eligible person enters the monitored area.”** Original F1 is non-blocking stress evidence.

No provider call, paid benchmark, or soak was run.

## What passed

- The signed VIRAT agreement is admitted; protected media was not redistributed.
- Official Release 2.0 `annotations-and-docs.zip` was acquired at exactly `41,225,862` bytes with
  SHA-256 `c83f541a425459722da1a594eedf58f60234210b5ec974f7d5d1bb839a40ed00`.
- Its format README says object files contain all annotated objects and warns that tracks may be
  fragmented. Inventory spans 11 physical source groups, enabling disjoint site-level partitions.
- Monitored polygons were authored from official scene-reference imagery before the aggregate rate
  survey. No qualification event time, track identity, or label row was intentionally surfaced.
- `138` selected H.264 files (`12,310,043,457` bytes) passed official package-size checks and local
  SHA-256/ffprobe binding. Rates are `2397/100` for 104 files and `30000/1001` for 34 files.
- The current clean split is validation `0101`; qualification `0000`, `0001`, `0100`, `0500`, and
  `0503`; reserve `0102`.

## Contamination and integrity dispositions

The annotation manifest contains `315` proper object files; six contain a measured total of `12`
malformed rows. While validating them, one raw row from
qualification group `0502` appeared in an exception. Under the frozen exposure rule, that exposes the
entire physical group even though no benchmark result was opened. Group `0502` was immediately moved
from qualification to calibration-only and recorded append-only. Its media cannot return to a clean
holdout.

The malformed `VIRAT_S_000002` source was removed from selected qualification media without exposing
its label contents. Six `0502` files with no proper object file and all interrupted downloads are
quarantined outside qualification paths.

Three otherwise selected sources have an official maximum annotation frame equal to the decoded
frame count. With zero-based decoded frames, that terminal annotation is one position beyond the
addressable range. The manifest records these defects; it does not clamp, discard, or invent their
effect on occurrences.

## Earliest failing joint gate

The frozen polygons yield `276` official-track seed occurrences across the remaining clean
qualification groups. For zero observed failures, the exact minimum sample size for a one-sided 95%
lower confidence bound above `0.99` is `299`; B1 is short by `23` measured occurrences.

An independent complete-zone audit is still required because official tracks seed truth rather than
replace adjudication of missed eligible entries. That audit could add or exclude occurrences, but no
future result is inferred. Therefore the current clean corpus cannot yet support the frozen 99%
claim, and B2 is not admitted.

## Precise next action

A label-isolated reviewer must audit every selected monitored zone, resolve the three terminal
frame-boundary defects, and return only corrected aggregate counts plus an immutable label manifest.
If the clean qualification count reaches at least `299`, rerun B1. If it remains below `299`, add a
never-exposed, license-cleared fixed-camera physical site and author its polygon before opening its
labels.

Machine-readable authorities:

- [`b1-zone-entry-readmission.json`](b1-zone-entry-readmission.json)
- [`f2-z1-zones-and-partitions.v1.json`](f2-z1-zones-and-partitions.v1.json)
- [`virat-media-manifest.v1.json`](virat-media-manifest.v1.json)
- [`exposure-registry.jsonl`](exposure-registry.jsonl)
