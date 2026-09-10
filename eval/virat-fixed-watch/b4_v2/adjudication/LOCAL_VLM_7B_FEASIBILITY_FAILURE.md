# Local Qwen2.5-VL-7B feasibility result

Status: **FAIL — stop before cases 2–15**

The frozen 7B feasibility case returned valid JSON but did not pass the predeclared substantive gate:

```json
{
  "label": "ambiguous",
  "confidence": 0.0,
  "person_description": "person walking",
  "outside_observation": "person walking outside the yellow zone",
  "crossing_observation": "person's footpoint crossing inside the yellow zone",
  "uncertainty": "resolution, occlusion, boundary proximity, or identity uncertainty prevents a reliable decision"
}
```

The response recognizes an apparent crossing but expressly refuses a reliable classification. The
gate required a non-ambiguous label, confidence above zero, and specific visible temporal evidence.
Execution therefore stopped before cases 2–15. Provider calls and provider cost remained zero.

## Disposition

Local Qwen2.5-VL at both 3B and 7B is insufficient to adjudicate these distant, boundary-adjacent
events. The model evidence confirms neither that the 15 fires are true false positives nor that they
are missing GT entries. All 15 retain provisional-FP status in the existing calibration score.

A definitive disposition requires independent human frame-by-frame complete-zone adjudication, or a
materially more capable local model that fits an authorized execution environment. Uploading VIRAT
clips to a hosted model remains prohibited by the signed dataset's redistribution restriction unless
the recipient is independently verified as an agreement signatory. Validation and PILOT-276 labels
remain sealed.

