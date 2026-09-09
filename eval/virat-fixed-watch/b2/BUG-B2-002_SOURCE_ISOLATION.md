# BUG-B2-002 — scorer source isolation

- Severity: high; a multi-source run could match an incident to an occurrence at the same local
  timestamp in a different source.
- Found: during B4 calibration-cache integration, before any calibration aggregate or validation
  score was produced.
- Evidence: `test_matching_never_crosses_sources` reproduces the collision and now passes.
- Fix owner: benchmark execution task branch.
- Fix: incidents carry an optional source identity and multi-source producers must populate it;
  candidate matching and duplicate classification require identical source identities.
- Retest gate: the B2 fixture suite and independent oracle must pass before B4 scoring. The optional
  source identity preserves the single-source oracle fixture API.
