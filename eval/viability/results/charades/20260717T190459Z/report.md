# Charades persistent-state G2 quality — pre-registered N=20 (frozen)

**Model:** `gemini-3.1-flash-lite` · gate `require_false_first=True` · tolerance `[onset, end+1.0s]` · 1.0 fps · run `20260717T190459Z`
**Calls:** 5521 (0 errored/timed-out) in 643.8s

## Result (as-is)
- **recall 0.67** (min 0.9) · **precision 0.375** (min 0.9)
- tp 122 · fp 203 · fn 60 · median latency-from-onset 2.59s
- **G2 quality: REVIEW/NO-GO ⚠️**

## Emitted launch_profile
```yaml
launch_profile:
  deadline_class: UNDETERMINED — G2 quality below bar
  provider: gemini    model: gemini-3.1-flash-lite
  g1_boundedness: PASS (architecture)   g3_latency: GREEN   economics: PASS   g2_quality: REVIEW
  supported_camera: static
```

_Direct-judge runner applying the pre-registered charades rule (harness hung on a live socket send). Real percept.gate.Gate(require_false_first=True); tolerance [onset, end+1.0s]._
