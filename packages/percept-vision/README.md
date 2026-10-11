# percept-vision

**Percept watches a live video stream for plain-language conditions and tells your agent, once per
occurrence, when one becomes true.** This is its Python SDK.

**Status: 0.9.0 Developer Preview.** The API may change before 1.0. Releases 0.7.0 and earlier
are a research preview with a different API.

It is for developers building agents that must react to what fixed cameras see. It is not for
PTZ, body-worn, vehicle, drone or phone video, or screen content.

## Install

```bash
pip install "percept-vision[video,gemini]"
```

Extras: `video` (OpenCV, for cameras and files), and one per judge: `gemini`, `claude`, `openai`
(OpenAI and OpenAI-compatible local servers).

## Example

```python
import asyncio

import percept


async def handle(decision: percept.Decision) -> None:
    # Your agent acts here. A decision can be delivered more than once: deduplicate on decision_id.
    print(decision.type, decision.condition_id, decision.reason, decision.decision_id)


async def main() -> None:
    monitor = percept.Monitor(
        judge=percept.judges.gemini("gemini-2.5-flash"),    # reads GEMINI_API_KEY
        conditions=[
            percept.Condition("door_open", "the garage door is open", emit_cleared=True),
            percept.Condition("fall", "a person falls to the floor", kind="event"),
        ],
    )
    async with monitor:
        source = percept.open_source("rtsp://camera.local/stream", fps=5)
        async for delivery in monitor.watch(source, ack="manual"):
            await handle(delivery.decision)
            await delivery.ack()        # unacknowledged decisions are delivered again after a restart


if __name__ == "__main__":
    asyncio.run(main())
```

Cheap triggers decide when a condition is worth a look, a judge (a vision-language model) decides
what is true, and a pure reducer decides whether your agent hears about it: at most one `met` per
real occurrence, stored before delivery and redelivered until acknowledged. The architecture
diagram is in the
[repository README](https://github.com/REPLACE-ME/percept#architecture).

## Limits

- Static cameras only.
- Thresholds and defaults are starting points, not calibrated values; confidence is advisory.
- A judge failure (timeout, outage, refusal, malformed reply) is never an answer: it is reported
  on `health()` and `coverage()`.
- Small, distant or brief activity in busy surveillance scenes is often missed, because the change
  trigger does not fire on it. Pair such conditions with a heartbeat (`every`), a tighter zone or a
  lower `on_change` threshold, and check `coverage()`.
- Each judge request has an 8 s deadline; sparse requests to a slow provider can time out and are
  reported as failures.
- No performance numbers are published for this release.

## Links

- Repository: https://github.com/REPLACE-ME/percept
- Documentation: https://github.com/REPLACE-ME/percept/tree/main/docs
- Support: support@velvee.ai

Percept is developed by Velvee, Inc. Licensed under Apache-2.0.
