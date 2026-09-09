# BUG-B4-001 — offline cache reported unmeasured timely recall as zero

- Severity: medium; the first calibration sweep serialized `timely_recall: 0.0` even though an
  offline detection cache has no receiver or ACK path.
- Evidence: append-only evidence hash
  `fa6a7cb4907d71bb02b316e47a9d8aa0dbff424df9ccdf566e70a76972cec89e`.
- Owner: benchmark execution task branch.
- Fix: the cache evaluator now emits `timely_recall: null`, `unacknowledged: null`,
  `receiver_ack_metrics_measured: false`, and an explicit interpretation.
- Retest gate: corrected append-only sweep hash
  `8521b4814adf86730942b2a96ae25de20ddc22ed0d1e98ac901bfeed30b3854e`; the original evidence is
  retained and superseded, not overwritten.
