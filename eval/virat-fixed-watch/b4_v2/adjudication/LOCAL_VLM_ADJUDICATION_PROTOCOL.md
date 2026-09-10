# Local-VLM review of zero-provisional-GT B4 v2 fires

Status: **FROZEN BEFORE MODEL EXECUTION — calibration-only diagnostic**

## Purpose and evidentiary limit

Review the 15 unmatched `virat-f2-z1-persistent-track-v2` fires that occurred in six exposed `0502`
calibration sources containing zero provisional GT entries. This is model-assisted triage of a known
GT-completeness risk. It does not create official ground truth, make provisional metrics exhaustive,
or reopen the failed B4 gate. Validation and PILOT-276 labels remain sealed.

Protected VIRAT pixels must remain local. No clip, frame, or derived image may be sent to a hosted
model or other party. The review uses a locally downloaded model and records zero provider calls and
zero provider cost.

## Frozen case construction

- Reconstruct the exact best-candidate fires from `b4-v2-calibration-result.json`, the immutable
  detector cache, official annotation archive, frozen F2-Z1 polygon, and v2 implementation.
- Select only unmatched fires whose source has zero provisional GT occurrences.
- Create one local review clip per fire spanning `[fire - 3 s, fire + 3 s]`, clipped to media bounds.
- Overlay the frozen zone polygon in yellow, the candidate footpoint in red at the fire instant, and
  an elapsed-time label. The overlay is explanatory only; no inputs or labels are changed.
- Store protected clips and model outputs only under the benchmark evidence volume.

## Frozen question and labels

Question: **Does an identifiable person's bounding-box footpoint make an observable outside-to-inside
crossing of the yellow zone boundary at the red-cursor time?**

Exactly one label is required:

- `true_eligible_entry`: an identifiable person is visibly outside, remains trackable, and then their
  footpoint crosses inside at the marked time.
- `not_entry_tracking_error`: no eligible crossing occurs; the apparent transition is detector-box
  jitter, identity fragmentation/switch, a non-person detection, or another tracking artifact.
- `left_censored_or_already_inside`: the person is first observable inside, or was already inside
  before the review window; an outside-to-inside transition is not established.
- `ambiguous`: resolution, occlusion, boundary proximity, or identity uncertainty prevents a reliable
  determination.
- `unusable`: media/overlay/model failure prevents review.

The model must describe the visible person, outside observation, crossing, and uncertainty without
identifying any individual. Confidence is a diagnostic number, not a correctness probability.

## Model identity and execution

- Family: `Qwen2.5-VL-3B-Instruct`, 4-bit MLX conversion.
- Local artifact: `mlx-community/Qwen2.5-VL-3B-Instruct-4bit`.
- Exact repository revision and local file hashes must be recorded after download and before the
  first case.
- Runtime: `mlx-vlm` in an isolated temporary virtual environment on the local Apple-silicon host.
- Decoding: temperature `0`; one independent structured response per case; no response may alter the
  prompt, cases, candidate, scorer, GT, or subsequent inputs.

## Interpretation

`true_eligible_entry` means the fire is plausibly a missing provisional GT occurrence rather than a
true false positive. `not_entry_tracking_error` supports retaining it as a provisional FP.
`left_censored_or_already_inside`, `ambiguous`, and `unusable` remain provisional FPs for current
scoring and require independent human complete-zone adjudication before any GT change. All 15 remain
provisional regardless of the model response until that audit is complete.

