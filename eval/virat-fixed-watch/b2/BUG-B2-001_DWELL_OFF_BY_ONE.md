# BUG-B2-001 — dwell confirmation was one frame early

- Severity: high for benchmark validity
- Owner: fixed-camera benchmark implementation
- Found in: pre-commit synthetic B2 qualification
- Evidence: at 4 FPS, a crossing at frame 1 with `0.5 s` dwell was initially confirmed at frame 2,
  only `0.25 s` after onset.
- Cause: the implementation counted the onset frame as one elapsed dwell frame.
- Fix: confirm only when `current_frame - onset_frame >= ceil(dwell_seconds * fps)`.
- Retest gate: the hand-calculated fixture must onset at frame 1 and confirm at frame 3; the full B2
  suite and independent matching oracle must pass.
- Disposition: fixed before any validation or qualification result was produced.
