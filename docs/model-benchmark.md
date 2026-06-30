# Model performance in Percept — directional sweep & learnings

> **Directional read on a SMALL sample (N=8–10, single seed). Not a release claim.** Differences of a few
> events fall inside the noise band. A **full benchmark (N≥20, all tracks, multi-seed)** is planned with the
> chosen model — that run is the publishable one. This doc records what the small-sample sweep does and
> doesn't tell us.

Same product path on every arm — same edge gate, cadence, and firing contract (`percept/gate.py`
byte-for-byte identical) — only the **model** swaps via `PERCEPT_VISION_BACKEND` / `PERCEPT_VISION_MODEL`.
So any difference is the model's perception/answer quality (or backend practicality), not the harness.

## Scorecard — N=8 cross-task

| track / metric | gemini-3.5-flash (REST) | qwen3-vl-32b (OpenRouter) | gemini-3.1-flash-live (realtime) |
|---|---|---|---|
| **pt-action** edge recall | 0.88 | 0.88 | 0.88 |
| **pt-action** E2E recall | **0.62** | **0.62** | 0.50 |
| **pt-action** precision | 0.40 | 0.37 | **0.47** |
| **pt-action** coverage | **8/8** | **8/8** | 6/8 (2 fail) |
| **charades** recall | 0.32 | 0.29 | **0.38** |
| **charades** precision | 0.68 | 0.58 | **0.71** |
| **charades** coverage | **8/8** | **8/8** | 7/8 (1 fail) |
| **pvqa** PAUC@0.5 (ref 33.5) | **42.9** (8/8, 43min) | ⏱ timeout† | ⏱ timeout‡ |
| **timespeak** latency p50 | 2.94s ✓ | *(contaminated)* | *(contaminated)* |

Fire latency (p50) was ~2.0–2.1s on **all three** arms (cadence-dominated — see learning 4).

† qwen pvqa timeout is a **config/tail-latency artifact, NOT rate-limiting or a capability gap** — see learning 2.
‡ gemini-live pvqa timeout is **intrinsic** (13.5s/call spoken WebSocket) — see learning 2.

## First read — N=10 pt-action (separate, earlier run)
| model | edge | E2E recall | precision | latency | coverage |
|---|---|---|---|---|---|
| gemini-3.5-flash | 0.70 | 0.40 | 0.33 | 2.20s | 10/10 |
| gemini-3.1-flash-live | 0.90 | 0.50 | 0.30 | 0.71s | 9/10 |
| qwen3-vl-32b | 0.80 | **0.60** | 0.32 | 2.07s | 10/10 |

## Learnings

1. **The harness is model-independent — confirmed.** Edge candidate-recall is **0.88 for all three arms**
   (N=8). The cadence/gate fires the same candidates regardless of model; only the *judge verdict* differs.
   Cross-model deltas are pure model quality, validating the sweep design.

2. **The pvqa timeouts were diagnosed — and they are NOT rate-limiting.** The accumulating-answer track
   (**pvqa / `process_qa`**) is ~1600 multi-frame judge calls per run (each sends the current frame + a strip
   of accumulated context). A live probe (single-frame *and* process_qa paths, both backends) showed **zero
   429s** and these per-call costs: gem35 **3.20s**, qwen **2.80s** (faster than the winner), gemini-live
   **13.47s**. So:
   - **pvqa is expensive for everyone** — gem35 itself needed **43 of the 60 minutes**; the cap was tight.
   - **gemini-live's timeout is intrinsic & real** — 13.5s/call (fresh WebSocket + spoken transcription per
     turn) × ~1600 calls ÷ 1 worker ≈ 180 min. It cannot run pvqa in practical time. *(Its strength is
     sub-second voice interaction, which this judge benchmark does not measure.)*
   - **qwen's timeout is a config/tail-latency ARTIFACT, not a capability gap** — its process_qa is *faster
     per-call than the winner*. Its ~37-min baseline + OpenRouter tail latency (the odd call stalling up to
     the 120s client timeout, then 2 SDK retries) tipped it over a cap gem35 barely cleared. It needs a fair
     re-run (higher cap + tighter per-request timeout so stalls fail-fast); **its pvqa number is still open.**

3. **On the cheap tracks the models are within noise — the N=10 ranking did NOT hold.** N=10 put qwen first
   on pt-action E2E recall (0.60); at N=8 **gem35 and qwen tie (0.62)** and qwen is **lowest on charades
   (0.29)**. gemini-live edges charades (0.38/0.71) but is the least reliable. Treat the N=10 "qwen wins" as a
   single-sample artifact.

4. **Latency is cadence-dominated, not a model differentiator.** The gemini-live "0.71s vs ~2s" edge at N=10
   **vanished at N=8** (~2.07s for all three). Fire latency is onset → next sampled+judged frame, set by the
   cadence; an eager model just fires a frame earlier on some samples. **No latency claim survives this data.**

5. **gemini-live: least reliable AND can't do pvqa.** It owns the only coverage failures (pt-action 6/8,
   charades 7/8 — flaky Live sessions) and timed out on pvqa. Its real value — sub-second *voice
   interaction* — isn't what this judge benchmark measures; for the **judge** role it's the weakest arm here.

## Directional winner → full benchmark: **`gemini-3.5-flash`** (safe pick), qwen contender pending
`gemini-3.5-flash` is the **safe, proven** choice: 8/8 coverage on every track and the only *completed* pvqa.
But the choice is closer than the raw scorecard suggests:
- **qwen is a live contender that never got a fair pvqa number** — it ties gem35 on pt-action (0.62), is
  competitive on charades, and its process_qa is *faster per-call* than gem35. Its timeout was a config
  artifact (OpenRouter tail latency vs a tight cap), not incapability. A re-run with a higher cap should
  give it a real pvqa number — and could make it the winner on cost (open-weight, cheap).
- **gemini-live is out for the judge role** — least reliable + intrinsically too slow on pvqa.

**Recommendation:** run the full benchmark on `gemini-3.5-flash` (de-risked), and in parallel give qwen a
fair pvqa re-run; if qwen lands a competitive PAUC, re-evaluate on cost.

## Coverage / not run
- **GPT-realtime-2.0/1.5** — blocked: OpenAI's Realtime API is a WebSocket service not proxied by OpenRouter,
  and no `OPENAI_API_KEY` is configured. (GPT *vision* via OpenRouter works but was dropped as too costly.)
- **OMNI-plus-realtime** — no reachable vision/realtime model id identified.

## Caveats (read before quoting any number)
- **N = 8–10, single seed.** Each cell is 8–10 events; one clip moves a number by 0.10–0.12.
- **The cheap-track rankings are not stable** between the N=10 and N=8 runs — that instability IS a finding.
- **pvqa timeouts were diagnosed, NOT assumed** — a live probe showed **zero 429s**; qwen's per-call cost
  (2.80s) is *below* the winner's (3.20s). qwen's timeout = OpenRouter tail latency against a tight cap
  (gem35 itself used 43 of 60 min), fixable with a higher cap / tighter per-request timeout. gemini-live's
  (13.5s/call) is intrinsic. Neither is rate-limiting; qwen's is not a capability gap.
- **Derived timespeak for qwen & gemini-live is contaminated** — their pvqa timed out, so the derivation
  silently fell back to a stale `proactiveqa-e2e-n1` artifact. Those cells are dropped; only gem35's
  timespeak (derived from its own pvqa) is trustworthy.
- **Only the model varies** — gate/cadence/edge are identical (edge recall 0.88 everywhere confirms it).
- **The full benchmark** (N≥20, all tracks, multi-seed, on `gemini-3.5-flash`) is the publishable run; this
  cheap directional pass says "the contenders are close on easy tracks, but only REST Gemini ships the hard one."
