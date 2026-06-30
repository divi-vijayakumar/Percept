# Model performance in Percept — directional sweep

How different vision/realtime models perform as the **perception judge** inside Percept (same edge gate,
same cadence, same firing contract — only the model swaps, via `PERCEPT_VISION_BACKEND` /
`PERCEPT_VISION_MODEL`). This is a **directional** read to pick a model for the full benchmark, not a
release claim.

## Setup
- Track: `pt-action-e2e` (Perception Test action notifications) — the cleanest single-model discriminator:
  edge recall (cadence coverage), E2E recall (did it fire on the action), precision, fire latency, call count.
- **N = 10**, sampled, local edge + RTMO pose path, fixed cadence/gate. One run per model arm.
- Backend-agnostic bench (`bench_vision`) — every arm runs the identical product path.

## Results (N=10, pt-action-e2e)

| model | backend | edge recall | **E2E recall** | precision | latency (p50) | calls | coverage |
|---|---|---|---|---|---|---|---|
| `gemini-3.5-flash` | gemini (REST) | 0.70 | 0.40 | 0.33 | 2.20s | 308 | 10/10 |
| `gemini-3.1-flash-live-preview` | gemini-live (realtime) | **0.90** | 0.50 | 0.30 | **0.71s** | 282 | 9/10 (1 fail) |
| `qwen/qwen3-vl-32b-instruct` | openrouter (open) | 0.80 | **0.60** | 0.32 | 2.07s | 311 | 10/10 |

## Directional read
- **Qwen3-VL leads the product metric** (E2E notification recall **0.60** > gemini-live 0.50 > gemini-3.5-flash
  0.40), at comparable precision (~0.32) and full reliability (10/10), cheap on OpenRouter. The open model is
  the surprise winner on "does it correctly fire on the action."
- **gemini-3.1-flash-live owns latency** (0.71s vs ~2.1s) and edge recall (0.90) — the realtime low-latency
  advantage is real — but it dropped one clip to a flaky Live session (9/10), and the spoken→transcript path
  costs a little precision. Best fit for *interactive/voice*, where latency dominates; weakest on reliability.
- **gemini-3.5-flash** is the steady baseline (10/10) but lowest E2E recall here.

**Winner (directional): `qwen/qwen3-vl-32b-instruct`** on the product metric, with `gemini-3.5-flash` the
reliable Gemini-native fallback and `gemini-live` the latency play.

## Coverage / not run
- **GPT-realtime-2.0 / 1.5** — blocked: OpenAI's Realtime API is a WebSocket service not proxied by
  OpenRouter, and no `OPENAI_API_KEY` is configured. (GPT *vision* via OpenRouter works but was dropped as
  too expensive for a directional pass.)
- **OMNI-plus-realtime** — not identified to a reachable model id (only `nvidia/nemotron-3-nano-omni` exists
  on OpenRouter; not a realtime/vision match).

## Caveats
- **N=10 is small** — the E2E-recall spread (0.40 / 0.50 / 0.60) is ~4–6 events. Directional, not definitive.
- **pt-action only.** charades (recognition) and proactiveqa (accumulating-answer) are not in this sweep;
  realtime pvqa in particular is ~10× slower (spoken `process_qa`). The full benchmark with the chosen winner
  covers all tracks and is the publishable run.
- `percept/gate.py` is byte-for-byte identical across all arms — only the model differs.
