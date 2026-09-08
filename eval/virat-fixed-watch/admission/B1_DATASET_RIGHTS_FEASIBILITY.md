# B1 dataset, rights, annotation, and source-rate feasibility

Status: **FAIL — stop before B2**
Recorded: `2026-09-08T19:23:34Z`

No provider call, paid benchmark, or soak was run. Capacity is sufficient for the identified package,
but the ground-truth contract fails independently of capacity and the protected-media acquisition is
not rights-admitted.

## Preflight receipt

The isolated writer was `codex/percept-benchmarking-b0-b1`. It moved cleanly from
`98bf9782c16ea529f61dc535fea966619b60e776` to founder base
`11d87a84a1dd44a021c54a21409505a31317b6db` before benchmark changes. The primary worktree was not
modified.

At `2026-09-08T19:23:34Z`, the host reported an Apple M3 Pro with 11 logical CPUs (5 performance,
6 efficiency), load averages `3.39 / 3.17 / 2.83`, 18 GiB physical memory and 61% free by
`memory_pressure`. The internal volume had `25,718,400 KiB` available. The evidence target
`/Volumes/Percept/Percept-Eternal/Percept-Benchmarks/virat-fixed-watch` was mounted, writable, on
exFAT, with `61,427,679,232` bytes available. No benchmark, provider-client, pytest, ffmpeg, or soak
process was active. Active and prunable worktrees were inspected.

The source declares `percept-vision 1.0.0`, but no wheel was present and source `_build_info.py`
fields were `unreleased`; therefore no wheel/build identity is qualified yet.

## Official corpus and exact acquisition projection

The [official VIRAT site](https://viratdata.org/) identifies Ground Release 2.0 as approximately
8.5 hours of stationary-camera HD video across 11 scenes, with source rates from 2 to 30 Hz. The
official Kitware collection (`56f56db28d777f753209ba9f`) reports `111,503,039,497` bytes total and
`37,625,884,426` bytes in the original-video folder.

The identified release package is:

| Object | Bytes |
|---|---:|
| `annotations-and-docs.zip` | 41,225,862 |
| `videos-00.zip` | 14,161,488,542 |
| `videos-01.zip` | 1,450,528,397 |
| `videos-04.zip` | 14,846,415,805 |
| `videos-05.zip` | 7,167,528,534 |
| Published checksum/sidecar files | 281,193 |
| Zip-package subtotal | **37,667,468,333** |
| DIVA annotations at `f1eff0d7db0dc0b25e1c99951ac0907521f3a216` | 192,021,654 |
| **Exact projected acquisition** | **37,859,489,987 (35.259398 GiB)** |

The public DIVA commit archive was placed in the preferred evidence volume as
`viratannotations-f1eff0d7db0dc0b25e1c99951ac0907521f3a216.tar.gz`; its SHA-256 is
`e1d440e4e6330c1921627fa21abd01982846991d283f468ff03ab0a13d334e04`. It expands to 476 annotation
files covering 119 stems: 64 train and 55 validation stems. Each stem has activity, geometry,
region, and type data; there is no scene/ignore-mask file. Two failed exFAT Git clone attempts were
retained on the evidence volume as forensic acquisition evidence and were not used as authorities.

Protected video packages were not downloaded.

## Rights and redistribution

The official [VIRAT protection agreement](https://viratdata.org/resources/VIRAT-Video-Data-Set-Protection-Agreement-1-4-11.pdf)
requires every person with access to sign, requires protection of personally identifying
information, prohibits attempts to identify people, and restricts redistribution to other
signatories (apart from limited scientific-publication or lecture excerpts). No operator-specific
signed acceptance receipt was found or verified, so protected acquisition is blocked.

## Annotation semantics and ignore policy

The official public DIVA annotation set describes full tracks on movers, but its Version 1.0
guideline also defines **Background** regions that may contain people or vehicles which need not be
annotated, and **Unannotated** regions whose low contrast, obstruction, or resolution makes them not
reasonably annotatable. The commit archive contains no scorer-ready mask that converts those regions
into ignored intervals.

Consequently, absence of a `Person` track is not proof that an entire frame contains no visible
person. The frozen F1 Watch requires whole-frame `no visible person -> at least one visible person`
episodes, so the available labels cannot supply exhaustive negative intervals. This independently
triggers the plan's immediate NO-GO for missing or non-exhaustive ground-truth negatives.

## Local media, hashes, and clocks

Eight official MP4s (`4,107,194,736` bytes) pre-existed in an author-accessible VIRAT tree. Their
MD5 values match the official checksum list, and their local SHA-256 values are frozen in the
append-only exposure registry. Probes found only `30/1` and `30000/1001` nominal rates:

| Source | Dimensions | Rate | Frames | Duration (s) |
|---|---:|---:|---:|---:|
| `VIRAT_S_000201_00_000018_000380` | 1280x720 | 30/1 | 10,834 | 361.133333 |
| `VIRAT_S_040000_03_000221_000372` | 1920x1080 | 30000/1001 | 4,522 | 150.884067 |
| `VIRAT_S_040001_01_000448_001101` | 1920x1080 | 30000/1001 | 19,567 | 652.885567 |
| `VIRAT_S_040005_06_000886_001016` | 1920x1080 | 30000/1001 | 3,891 | 129.829700 |
| `VIRAT_S_040005_07_001026_001223` | 1920x1080 | 30000/1001 | 5,895 | 196.696500 |
| `VIRAT_S_040103_00_000000_000120` | 1920x1080 | 30000/1001 | 3,594 | 120.019900 |
| `VIRAT_S_040104_01_000227_000457` | 1920x1080 | 30000/1001 | 6,890 | 229.896333 |
| `VIRAT_S_040104_05_000939_001116` | 1920x1080 | 30000/1001 | 5,304 | 176.976800 |

For the already-exposed `000201` sample, annotation frame `10833`/timestamp `361.1 s` aligns with
the media's final zero-based frame at 30 fps. This is a sample observation, not a full-corpus claim;
full-track clock/hash alignment remains unverified until rights-admitted acquisition.

## Exposure freeze and holdout feasibility

The first four records in [`exposure-registry.jsonl`](exposure-registry.jsonl) freeze the policy and
the current exposure state. Physical scene is the minimum group because adjacent clips share site
and appearance. Author-accessible media, annotations, `ground_truth.json`, and authored run output
expose complete groups `0002`, `0400`, and `0401`; they are calibration-only forever.

Only `0000` and `0500` remain clean in the public DIVA inventory. They remain sealed and unassigned:
allocating one to validation leaves only one qualification site, while allocating both to
qualification leaves no clean validation. Candidate authors must never inspect qualification labels;
the label custodian must be independent. Any code, goal, prompt, zone, profile, direct-judge,
tunable, ground-truth transform, or scorer change mints a candidate identity and exposes any holdout
on which it was scored.

A feasible partition needs the exposed groups for calibration, at least one never-exposed group for
validation, and at least two distinct never-exposed site/session groups for qualification. At least
one additional clean physical scene/site group is therefore required even if `0000` and `0500` are
ultimately usable.

## Candidate baseline decision

Candidate `0623784d5db5cd07b1abf13ac9ec510b9c7ba2fb` is retained only as historical
calibration/reference input. It is not an ancestor of the execution base; its merge base is
`2217d09d`, with 85 candidate-side commits and 322 changed paths relative to the founder checkpoint.
Its fixed-occupancy and holdout work may be inspected for independently tested components, but its
claims, build identity, and holdout cleanliness are not assumed. Execution remains based on
`11d87a84a1dd44a021c54a21409505a31317b6db`.

## Blockers and precise next action

1. Whole-frame negative occupancy labels are not exhaustive and no usable ignore masks exist.
2. Only two clean scene groups remain; validation plus two-site qualification cannot be disjoint.
3. Operator-specific signed protection-agreement acceptance has not been verified.
4. Full-corpus hashes and source clocks cannot yet be verified.

**Stop before B2.** Obtain operator-specific signed agreement receipts and a license-cleared,
exhaustive whole-frame person-occupancy annotation supplement with scorer-ready ignore regions, plus
at least one additional never-exposed physical scene/site group. Then freeze clean validation and
two-site qualification partitions under an independent label custodian, verify every media hash and
source clock, and rerun B1. No later milestone is admitted until B1 passes.

The machine-readable authority is [`b1-feasibility.json`](b1-feasibility.json).
