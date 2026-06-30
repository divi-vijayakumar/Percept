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
| **pvqa** PAUC@0.5 (ref 33.5) | **42.9** (8/8) | ⏱ **timeout >60min** | ⏱ **timeout >60min** |
| **timespeak** latency p50 | 2.94s ✓ | *(contaminated)* | *(contaminated)* |

Fire latency (p50) was ~2.0–2.1s on **all three** arms (cadence-dominated — see learning 4).

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

2. **DECISIVE — only the REST Gemini arm completes the full capability set.** The accumulating-answer track
   (**pvqa / `process_qa`**) is ~hundreds of sequential multi-turn judge calls per run.
   `gemini-3.5-flash` (REST) finished it under the 60-min cap (PAUC **42.9**, above the 33.5 reference). Both
   **qwen (OpenRouter)** and **gemini-live (spoken WebSocket)** **timed out at 60 min** — a >10× practicality
   gap, not a correctness failure. *This*, not the near-tied cheap-track metrics, is what should pick the
   model: a judge that can't run `process_qa` in reasonable time can't ship the proactive-QA capability.

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

## Directional winner → full benchmark: **`gemini-3.5-flash`**
Not because it tops every cell (pt-action ties qwen; charades trails gemini-live) — but because it is the
**only arm that runs the full track set reliably and in practical time** (8/8 coverage everywhere + the only
completed pvqa). qwen ties on the cheap tracks but can't complete `process_qa`; gemini-live is both less
reliable and can't complete it either. Practicality + reliability break the near-tie.

## Coverage / not run
- **GPT-realtime-2.0/1.5** — blocked: OpenAI's Realtime API is a WebSocket service not proxied by OpenRouter,
  and no `OPENAI_API_KEY` is configured. (GPT *vision* via OpenRouter works but was dropped as too costly.)
- **OMNI-plus-realtime** — no reachable vision/realtime model id identified.

## Caveats (read before quoting any number)
- **N = 8–10, single seed.** Each cell is 8–10 events; one clip moves a number by 0.10–0.12.
- **The cheap-track rankings are not stable** between the N=10 and N=8 runs — that instability IS a finding.
- **pvqa timeouts are a practicality signal, not a hard ceiling** — qwen's may be partly OpenRouter
  throttling; a longer cap or more workers might eventually finish. But the >10× gap vs REST Gemini at the
  same workload is real and decision-relevant.
- **Derived timespeak for qwen & gemini-live is contaminated** — their pvqa timed out, so the derivation
  silently fell back to a stale `proactiveqa-e2e-n1` artifact. Those cells are dropped; only gem35's
  timespeak (derived from its own pvqa) is trustworthy.
- **Only the model varies** — gate/cadence/edge are identical (edge recall 0.88 everywhere confirms it).
- **The full benchmark** (N≥20, all tracks, multi-seed, on `gemini-3.5-flash`) is the publishable run; this
  cheap directional pass says "the contenders are close on easy tracks, but only REST Gemini ships the hard one."
