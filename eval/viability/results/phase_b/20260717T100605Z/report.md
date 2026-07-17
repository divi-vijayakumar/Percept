# Percept Viability — Phase-B For-Record Run (REAL Gemini)

**Model:** `gemini-3.1-flash-lite` (generateContent) · git `e0eb3df` (+local) · run `20260717T100605Z`

> Quality is measured on REAL golden-clip frames (§5a secondary set) scored against reviewer-approved
> rising edges. **The §5a set is single-reviewer and small — sample sizes are reported.** Real 24h soak
> was NOT run against the provider (architecture-validated in Phase-A).

## G2 — Quality (real recall/precision)
- **recall 0.6445783132530121** (min 0.9) · **precision 0.6239067055393586** (min 0.9) → **REVIEW**
- positives n=332 · negatives n=1097 · TP=214 FP=129 FN=118 TN=968
- caveat: frame-level labels from clip.json true-span `intervals` (per-condition); §5a single-reviewer set

## G3 — Latency (real single-judge floor)
- p50 **0.711s** · **p95 1.202s** · p99 1.508s · n=1429 → band **GREEN**
- real judge latency under light concurrency (sem=6); full under-load G3-A/G3-B need the worker pipeline. Clean single-call floor was 0.728s at probe.

## G2 — Economics (real tokens)
- avg 1233.3 in / 39.4 out tokens/call → **$0.000139/call**
- at 240.0 judgments/stream-hour → **$0.0334/stream-hour** (ceiling $10.0) → **PASS**
- $ scales with the assumed flash-lite price; token counts are exact from usage_metadata (assumed $0.1/1M in, $0.4/1M out)

## Emitted launch_profile (candidate)
```yaml
launch_profile:
  deadline_class: UNDETERMINED — G2 quality below the 0.90 bar (resolve before emitting a profile)
  provider: gemini    model: gemini-3.1-flash-lite
  g1_boundedness: PASS (Phase-A architecture)
  g2_quality: REVIEW    g3_latency: GREEN    economics: PASS
  supported_camera: static
```

## Not run (stated, not faked)
- real 24h soak against provider (architecture-validated in Phase-A; ops decision)
- full under-load G3-A/G3-B through the worker pipeline (single-judge floor measured here)
