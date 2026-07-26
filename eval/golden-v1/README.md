# Golden set v1

This folder is the public manifest home for the v0.4 benchmark. It is intentionally a draft until every
test item is license-cleared and content-hashed.

Current seed:

- `MANIFEST.json` includes eight `google-deepmind/perception_test` sample videos plus fourteen
  deterministic hash-ranked valid-split videos in the frozen `test` split by reference, with
  SHA-256 hashes for every local MP4 and annotation artifact.
- The videos are not vendored into the package; fetch/cache them under
  `eval/bench-staging/perception_test/raw/` when reproducing the scorecard.
- Twenty disjoint rows are pinned to the `perception-test-action` track and map explicitly to the
  `pt-action-e2e` runner. The first non-catch-all label in each action row defines the Watch goal,
  and all occurrences of that label define the event-recall windows. Two separate rows are pinned
  to `perception-test-acoustic`.
  Although those acoustic videos also contain action annotations, a runner never borrows cases
  from another logical track.

Rules:

- No private YouTube fixtures.
- Every row needs source, license, split, immutable media hash, and task labels.
- `dev` is for calibration; `test` is frozen.
- A release scorecard must cite the manifest hash.
- Runner coverage must report the exact manifest case IDs; a count without IDs is not qualifying.
- Source split provenance remains explicit (`sample/train` versus `valid`) even though both source
  subsets are frozen into this benchmark's `test` split.
- Modeled cells make a scorecard `DRAFT`.

The current private fixtures remain useful for development, but they are not publishable benchmark data.
