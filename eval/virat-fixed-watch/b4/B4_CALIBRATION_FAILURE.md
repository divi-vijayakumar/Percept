# B4 authored Watch calibration

Status: **FAIL — earliest failing joint gate; execution stopped**

The complete exposed `0502` calibration partition was evaluated locally. Twenty-four media files
were cached at 1 Hz; the scorer admitted the same 21 files as B2 and excluded three malformed legacy
annotation files. No validation or qualification labels were opened, no provider was called, and no
paid activity occurred.

The frozen search contained 135 combinations of score threshold, association distance, track gap,
and the three approved dwell values. Zero combinations met joint provisional precision and recall
of 0.99. The best-ranked configuration measured:

- TP 13, provisional FP 26, provisional FN 3;
- provisional precision 0.3333333333333333 and recall 0.8125;
- 39 fires, 3 duplicates, 3 misses, ASK 0;
- 42.05248032732404 provisional false fires/source-hour over 0.6182750648148149 scored hours;
- detector inference p50 0.7456225210007688 s, p95 0.8037559086993497 s, p99
  0.8683322737920025 s, max 1.3134737079999468 s.

Timely recall and source-onset-to-receiver-ACK latency were **not measured** by this offline cache.
Provider cost was exactly $0.00; all-in economy is absent because no local resource rate was supplied.
The complete-zone audit and occlusion adjudication remain pending, so quality counts and ratios are
provisional engineering measurements and cannot be presented as exhaustive or publication-grade.

## Candidate disposition

- Retained candidate commit `0623784d5db5cd07b1abf13ac9ec510b9c7ba2fb` is not a suitable direct
  baseline: it implements RTMO occupancy rather than per-person zone entry, omits required model
  artifacts, and carries an unrelated 322-file delta. None of its claims were assumed.
- Official RTMO-S was rejected by the exposed calibration feasibility probe.
- Official TorchVision Faster R-CNN ResNet50 FPN v2 had viable CPU inference timing but failed the
  complete calibration quality gate.

The corrected full sweep and cache manifest are stored on the designated evidence volume and bound
by [`b4-calibration-result.json`](b4-calibration-result.json). The first sweep remains append-only but
is superseded because `BUG-B4-001` found that it rendered unmeasured timely recall as numeric zero.

Per the earliest-failing-gate rule, validation, candidate freeze, PILOT-276, provider runs, and the
72-hour soak were not started. The precise next action is founder approval of a new B4 candidate plan;
the current plan has no admitted autonomous step beyond this failure.

