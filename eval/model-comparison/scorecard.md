# Full model comparison — detailed scorecard (directional)

> **Directional, small sample (N=8 cross-task; N=10 pt-action). Not a release claim.** Same edge gate,
> cadence and firing contract on every arm — only the model swaps. Cost axis = measured `candidate_calls`;
> estimated tokens use a documented proxy (img~258 tok, prompt~250, frames/call {'pt-action-e2e': 5, 'charades-e2e': 5, 'proactiveqa-e2e': 13}, out 80/220).


## PT-action (N=8) — all metrics

| metric | gemini-3.5-flash | qwen3-vl-32b | gemini-3.1-flash-live |
|---|---|---|---|
| edge recall | 0.875 | 0.875 | 0.875 |
| edge precision (proxy) | 0.254 | 0.257 | 0.312 |
| E2E notification recall | 0.625 | 0.625 | 0.5 |
| E2E notification precision | 0.4 | 0.368 | 0.467 |
| median fire latency (s) | 2.068 | 2.068 | 2.068 |
| candidate calls | 248.0 | 249.0 | 205.0 |
| call reduction | 0.441 | 0.44 | 0.47 |
| **coverage (obs/exp, fail)** | 8/8, 0 | 8/8, 0 | 6/8, 2 |

## Charades (N=8) — all metrics

| metric | gemini-3.5-flash | qwen3-vl-32b | gemini-3.1-flash-live |
|---|---|---|---|
| edge recall | 0.515 | 0.515 | 0.485 |
| edge precision (proxy) | 0.917 | 0.917 | 0.885 |
| E2E notification recall | 0.318 | 0.288 | 0.379 |
| E2E notification precision | 0.677 | 0.576 | 0.714 |
| median fire latency (s) | 4.12 | 4.112 | 4.72 |
| candidate calls | 248.0 | 248.0 | 173.0 |
| call reduction | 0.86 | 0.86 | 0.898 |
| sparse-span mIoU (diag) | 0.123 | 0.131 | 0.105 |
| **coverage (obs/exp, fail)** | 8/8, 0 | 8/8, 0 | 7/8, 1 |

## ProactiveQA (N=8) — all metrics

| metric | gemini-3.5-flash | qwen3-vl-32b |
|---|---|---|
| edge recall | 0.872 | 0.872 |
| edge precision (proxy) | 0.315 | 0.315 |
| first-answer latency p50 (s) | 6.06 | 3.95 |
| candidate calls | 447.0 | 447.0 |
| calls/min | 10.198 | 10.159 |
| call reduction | 0.15 | 0.153 |
| duplicate rate | 0.037 | 0.066 |
| pvqa PAUC@0.5 | 42.885 | 52.859 |
| pvqa PAUC@0 | 37.694 | 46.103 |
| pvqa PAUC@1 | 48.077 | 59.615 |
| unresolved rate | 0.436 | 0.487 |
| interruptions | 160.0 | 167.0 |
| **coverage (obs/exp, fail)** | 8/8, 0 | 8/8, 0 |

## CueSpeak (N=8) — all metrics

| metric | gemini-3.5-flash | qwen3-vl-32b | gemini-3.1-flash-live |
|---|---|---|---|
| recall | 0.23 | 0.212 | 0.329 |
| precision | 0.045 | 0.04 | 0.66 |
| fires/episode | 6.336 | 6.637 | 0.588 |
| duplicate rate | 0.008 | 0.008 | 0.1 |
| unresolved rate | 0 | 0 | 0 |
| **coverage (obs/exp, fail)** | 113/113, 0 | 113/113, 0 | 85/85, 0 |

## TimeSpeak (N=8) — all metrics

| metric | gemini-3.5-flash | qwen3-vl-32b | gemini-3.1-flash-live |
|---|---|---|---|
| latency (s) | 2.94 | 3.788 | 4.116 |
| p-PAUC | 2.013 | 1.98 | 2.602 |
| early-fire rate | 0.068 | 0.081 | 0.081 |
| late-fire rate | 0 | 0.027 | 0.068 |
| timing accuracy | 0.054 | 0.041 | 0.041 |
| **coverage (obs/exp, fail)** | 74/113, 0 | 74/113, 0 | 74/85, 0 |

## pt-action — N=10 (earlier directional run)

| metric | gemini-3.5-flash | qwen3-vl-32b | gemini-3.1-flash-live |
|---|---|---|---|
| edge recall | 0.7 | 0.8 | 0.9 |
| E2E recall | 0.4 | 0.6 | 0.5 |
| E2E precision | 0.333 | 0.32 | 0.3 |
| median latency (s) | 2.202 | 2.068 | 0.709 |
| candidate calls | 308.0 | 311.0 | 282.0 |

## Estimated token cost per track (N=8) — total **and per completed clip**

> ⚠️ Compare **per-clip**, not total. g3.1-live's lower *totals* are a coverage artifact (it completed 6/8 pt-action, 7/8 charades — fewer clips = less total work, NOT efficiency). Per completed clip it is *more* expensive on pt-action; its lower charades cost reflects lower edge recall (under-firing), not savings.

| track | tokens/call | gemini-3.5-flash | qwen3-vl-32b | gemini-3.1-flash-live |
|---|---|---|---|---|
| PT-action | 1620 | 402k (50k/clip) | 403k (50k/clip) | 332k (55k/clip) |
| Charades | 1620 | 402k (50k/clip) | 402k (50k/clip) | 280k (40k/clip) |
| ProactiveQA | 3824 | 1709k (214k/clip) | 1709k (214k/clip) | — |
