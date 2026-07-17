# Percept H0 Viability — Go/No-Go Memo (INTERIM DRAFT, Phase-A only)

> **Status: INTERIM.** This memo records the **Phase-A (architecture-isolation, fake-judge)** result
> only. It does **not** yet emit a `launch_profile` — that requires **Phase-B** (real Gemini) latency
> and economics. Do not treat this as the final go/no-go. Authority: `Percept-Viability-Probe-Spec.md`
> §9/§10. Locked config: `eval/viability/reference_config.yaml` (`bars_locked: true`, commit `e0eb3df`).

## Locked reference (the config every number keys off)
- **Machine:** Mac15,6 · Apple M3 Pro (11 cores) · 18 GB · macOS 26.5 · Python 3.12.8
- **Provider (Phase-B):** Gemini `generateContent`, model **`gemini-3.1-flash-lite`** *(⚠ validate it resolves + returns structured output before a real run; adapter default/tested is `gemini-3.5-flash`)*
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

## `launch_profile`: NOT YET EMITTABLE
The profile (`reflex` | `bounded` | `deliberative`) is chosen by **Phase-B G3** latency against the 2s
dual gate. Until Phase-B runs, no profile is emitted. Do not start the v0.1 PRD build.

## Phase-B prerequisites (blocking a final memo)
1. Wire the Gemini API key (`.env PERCEPT_VISION_MODEL` / `PERCEPT_GEMINI_MODEL` are unset).
2. **Validate `gemini-3.1-flash-lite`** resolves on `generateContent` and returns structured output;
   fall back to `gemini-3.5-flash` if not.
3. Run Phase-B: real recall/precision (fixtures + §5a golden clips), real G3-A/G3-B latency under the
   N/M/E/F load, real $/stream-hour vs the $10 ceiling, real-adapter P-JUDGE-SAFE-1.
4. Re-run this memo → emit the typed `launch_profile`.

## Bottom line
**Phase-A: GO on architecture (G1).** Economics and latency are pending Phase-B, which is gated only
on the Gemini key + a model-ID validation. No `launch_profile`, no v0.1 build, until then.
