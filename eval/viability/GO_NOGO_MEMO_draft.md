# Percept H0 Viability — Go/No-Go Memo (INTERIM DRAFT, Phase-A only)

> **Status: INTERIM.** This memo records the **Phase-A (architecture-isolation, fake-judge)** result
> only. It does **not** yet emit a `launch_profile` — that requires **Phase-B** (real Gemini) latency
> and economics. Do not treat this as the final go/no-go. Authority: `Percept-Viability-Probe-Spec.md`
> §9/§10. Locked config: `eval/viability/reference_config.yaml` (`bars_locked: true`, commit `e0eb3df`).

## Locked reference (the config every number keys off)
- **Machine:** Mac15,6 · Apple M3 Pro (11 cores) · 18 GB · macOS 26.5 · Python 3.12.8
- **Provider (Phase-B):** Gemini `generateContent`, model **`gemini-3.1-flash-lite`** — **VALIDATED 2026-07-17** by live probe (structured-output vision OK; single-call **0.728s**, vs 1.804s for `gemini-3.5-flash`; `gemini-3.1-flash-live-preview` = 404 on generateContent). Wired via `PERCEPT_VISION_MODEL`. Probe: `results/model_probe_20260717.json`.
- **Founder bars:** cost ≤ **$10**/stream-hour · recall/precision **0.90** · naïve-VLM baseline **1.0 fps** · soak ≤ **24h**
- **Load N/M/E/F (assistant-proposed, accepted):** 4 / 2 / 4 / 1
- **Run:** `20260717T093226Z` · seed `20260717` · git `e0eb3df` · Phase-A commit `24d8aaf` (local)

## Per-gate status

| Gate | Phase-A (fake judge) | Verdict |
|---|---|---|
| **G1 — Runtime boundedness** | queue depth bounded (80, non-growing) · 0 queue-age/memory slope · 0 latency drift within tolerances · clean recovery after injected failure hour (6 sim hours) | **GO (architecture)** ✅ |
| **G2 — Watch economics** | VID **240** vs naïve **3600** judgments/stream-hour = **15× fewer judge calls** at locked E/F | **STRUCTURE ✅ · $/stream-hour PENDING Phase-B** |
| **G3 — Reflex latency** | admitted→decision p95 **0.54s** on a *fake* uniform(0.10,0.50)s dist — architecture floor only | **PENDING Phase-B** (fake ≠ product) |

## Interpretation
- **G1 is the kill-gate a faster model cannot save you on, and it PASSED on the fake substrate** —
  the runtime is bounded, stale-completion-safe, and recovers cleanly. This is the strongest possible
  Phase-A signal: the *architecture* is viable.
- **G2/G3 are deliberately unresolved.** The 15× volume reduction is real and favorable for the $10
  ceiling, but the actual $/stream-hour and the real p95 latency need real Gemini calls. The fake
  0.54s floor only proves the queue+dispatch path adds little overhead — it is **not** the product
  latency and must not be quoted as one.

## Phase-B results — REAL `gemini-3.1-flash-lite` (run 20260717T100605Z, 1429 real calls)
Quality on real golden-clip frames labeled by clip.json true-span `intervals`; latency + cost from the real calls.

| Gate | Phase-B (real Gemini) | Verdict |
|---|---|---|
| **G3 — Latency** | admitted→decision **p95 1.20s** (p50 0.71s), n=1429, light concurrency | **GREEN** ✅ (well under 2s) |
| **G2 — Economics** | avg 1233 in / 39 out tokens → **$0.033/stream-hour** at 240 judgments/hr (ceiling $10) | **PASS** ✅ (~300× margin) |
| **G2 — Quality** | **recall 0.645 · precision 0.624** (min 0.90 each); TP 214 / FP 129 / FN 118 / TN 968; n_pos 332, n_neg 1097 | **REVIEW** ⚠️ (below bar) |

## `launch_profile`: **UNDETERMINED** — latency & economics pass, but **G2 quality gates it**
```yaml
launch_profile:
  deadline_class: UNDETERMINED   # G2 quality (recall/precision) below the 0.90 bar
  provider: gemini    model: gemini-3.1-flash-lite
  g1_boundedness: PASS (architecture)   g3_latency: GREEN   economics: PASS   g2_quality: REVIEW
  supported_camera: static
```
Per Viability §9, a `reflex`/`deliberative` GO requires G1 **and** G2 to pass; latency (GREEN) only
picks *which* profile among passing workloads. It does not.

## Honest interpretation of the quality miss
0.645/0.624 is **frame-level** quality on the **single-reviewer §5a golden set**, and it is below the
0.90 bar. Before reading this as "Percept fails," note the confounders — each is a real next step, not an excuse:
- **Frame-level ≠ runtime.** The real runtime judges at the **episode** level with dwell/refractory/rearm —
  materially more forgiving than scoring every isolated frame. Episode-level scoring (golden_eval §5a
  matching over predicted vs ground-truth episodes) is the fairer measure and was **not** run here.
- **Single-reviewer labels + coarse ~6-frame clips** (§5a gap) — some `intervals` span the whole clip;
  some conditions are genuinely ambiguous ("clearly visible ahead").
- **No prompt/threshold tuning** — this is the stock judge prompt at default confidence.
- **The 0.90 bar** may be too strict for raw frame-level golden scoring; it was set for the runtime's
  event-level recall/precision.

## Where this leaves the go/no-go
- **G1 GO (architecture)** — bounded, recovers; the kill-gate a faster model can't save. ✅
- **G3 latency GREEN + economics excellent** — `gemini-3.1-flash-lite` at ~1.2s p95 and ~$0.03/stream-hour
  is comfortably reflex-capable and ~300× under the cost ceiling. ✅
- **G2 quality REVIEW** — the one open gate. **No `launch_profile` emitted, no v0.1 build**, until quality
  is resolved: (a) run **episode-level** golden scoring, (b) tune prompt/threshold, (c) re-examine whether
  0.90 frame-level is the right bar. This is a *scoped* quality-track task, not an architecture failure.
