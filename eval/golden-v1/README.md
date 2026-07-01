# Golden set v1

This folder is the public manifest home for the v0.4 benchmark. It is intentionally a draft until every
test item is license-cleared and content-hashed.

Current seed:

- `MANIFEST.json` includes eight `google-deepmind/perception_test` sample videos in the frozen `test`
  split by reference, with SHA-256 hashes for each local MP4 and the sample annotation JSON.
- The videos are not vendored into the package; fetch/cache them under
  `eval/bench-staging/perception_test/raw/` when reproducing the scorecard.
- Two rows carry sound-localisation annotations (`perception-test-acoustic`); all eight carry
  action-localisation annotations (`perception-test-action` / timeliness and edge-recall use).

Rules:

- No private YouTube fixtures.
- Every row needs source, license, split, immutable media hash, and task labels.
- `dev` is for calibration; `test` is frozen.
- A release scorecard must cite the manifest hash.
- Modeled cells make a scorecard `DRAFT`.

The current private fixtures remain useful for development, but they are not publishable benchmark data.
