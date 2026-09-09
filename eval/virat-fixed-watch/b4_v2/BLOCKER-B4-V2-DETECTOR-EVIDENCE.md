# BLOCKER-B4-V2 — fixed detection evidence is insufficient

- Severity: launch-blocking.
- Evidence: complete frozen 972-configuration sweep SHA-256
  `3f94f2b791ea5e89b8533449e28635a1f7186749e0b6ca81185bdcc75f4f2f84`; maximum TP 12/16,
  minimum FN 4, no zero-FP configuration.
- Owner: future perception candidate owner; this execution task cannot fix it inside the approved
  tracking-only family.
- Reproduction: run `sweep_v2.py` with the paths and identities in
  `b4-v2-candidate-freeze.json`.
- Retest gate: mint and pre-freeze a materially different detector/model or evidence architecture,
  rerun all admitted exposed calibration, and require the unchanged 0.99/0.99/zero-duplicate gate
  before opening validation.
- Current disposition: stop. This is not authorization for provider calls, paid compute, validation,
  qualification-label access, or PILOT-276.
