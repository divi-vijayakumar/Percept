# percept-vision

**Percept is an open-source runtime for continuous vision applications.** Configure what to
inspect, how to escalate to vision models, how much inference to spend, and when to deliver an
actionable decision; then evaluate those choices with reproducible precision, recall, latency and
cost measurements. This is its Python SDK.

**Status: Percept 1.0, Developer Preview.** The runtime API is the 1.0 contract: it changes
incompatibly only in a new major version. Watch plans and the evaluation toolkit (`percept.plans`)
are experimental. Releases 0.7.0 and earlier are a research preview with a different API.

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
diagram is on the [project site](https://percept.percia.ai/).

## Integrating Percept

The recipe for wiring Percept into an app or agent, for people and coding agents alike.

**1. Install the judge's extra.** `pip install "percept-vision[video,gemini]"` (or `[video,claude]`,
`[video,openai]`). `video` is OpenCV, for `open_source`. Without the judge's extra, building it
raises `ImportError: this judge needs the 'google.genai' package: pip install
'percept-vision[gemini]'`. `percept.judges.gemini` reads `GEMINI_API_KEY`.

**2. Ack after delivery, and watch health in a background task.**

```python
import asyncio, json, logging, os, urllib.request
import percept

log = logging.getLogger("app")
WEBHOOK_URL = os.environ["WEBHOOK_URL"]


def post(url: str, body: bytes) -> None:
    """Blocking POST; urllib raises on a non-2xx answer or a network error."""
    request = urllib.request.Request(url, body, {"Content-Type": "application/json"}, method="POST")
    urllib.request.urlopen(request, timeout=10).close()


async def post_with_retry(decision: percept.Decision, tries: int = 5) -> bool:
    body = json.dumps(decision.to_dict()).encode()
    for attempt in range(tries):
        try:
            await asyncio.to_thread(post, WEBHOOK_URL, body)   # keep the event loop free
            return True
        except OSError as exc:
            log.warning("POST failed (%s); retrying", exc)
        await asyncio.sleep(min(30.0, 0.25 * 2 ** attempt))
    return False


async def log_health(monitor: percept.Monitor) -> None:
    async for h in monitor.health():        # an async iterator, not a list; ends at aclose()
        log.warning("percept health %s on %s: %s", h.kind, h.stream_id, h.detail)


async def main() -> None:
    monitor = percept.Monitor(
        percept.judges.gemini("gemini-2.5-flash"),
        [percept.Condition("person_fell", "a person falls down", kind="event"),
         percept.Condition("door_open", "the door is left open", emit_cleared=True)],
    )
    health = asyncio.create_task(log_health(monitor))
    async with monitor:
        source = percept.open_source("rtsp://camera.local/stream", stream_id="hallway")
        async for delivery in monitor.watch(source, ack="manual"):
            if await post_with_retry(delivery.decision):
                await delivery.ack()        # only after the webhook accepted it
    await health


asyncio.run(main())
```

- With `ack="manual"`, an unacknowledged decision is **not** redelivered within the same
  `watch()`, and there is no `nack`: retry the hand-off yourself. It stays in the store and is
  delivered again (same `decision_id`, `attempt` + 1) by the next `watch()` of the stream or after
  a restart. With `max_unacked` (100) unacknowledged, delivery on the stream pauses.
- Delivery is at least once: deduplicate on `decision_id`. `decision.to_dict()` is the versioned
  JSON form.
- `health()` buffers events from construction in a bounded ring (256 aggregated entries), so the
  task may start before or after `watch()`. Use one consumer.

**3. Test offline with `FakeJudge`.** No key or network; the same prompt, parsing and failure
handling as a real judge. The script is called once per judge request with the frames a real
judge would see (BGR `uint8` arrays in `request.images`), and answers per condition id:

```python
def script(request):
    lit = (request.images[-1][..., 1] > 200).any()
    return {"door_open": ("yes", 0.9, "open") if lit else ("no", 0.9, "shut")}

monitor = percept.Monitor(percept.judges.FakeJudge(script), conditions,
                          store=percept.MemoryStore(), max_requests_per_min=600)
camera = percept.PushSource("test-cam")    # camera.push(frame), then camera.close()
```

`MemoryStore` logs once that it is not crash-safe: expected in tests and demos.

**4. Defaults that decide behaviour** (starting points, not calibrated):

- `max_requests_per_min=6` attempts per stream: about 5 requests a minute when none retries
  (a built-in judge reserves 2 attempts a request). Due conditions share a request.
- A state is looked at on a scene change (`on_change()`) plus every 300 s (`every(300)`); an
  event on a scene change only, or when you `poke()`. A `trigger=` replaces both.
- `clear_after_no=3`: a held state clears after 3 confident "no" answers in a row. With 60 s
  rechecks (`recheck_every_s`), a door closed in a quiet scene is `cleared` 1 to 2 minutes later.
- The store is `SqliteStore("percept.db")` in the working directory: crash-safe, one process per
  file. `MemoryStore()` is opt-in.
- Each judge request has an 8 s deadline; with a retry left, an attempt may use at most 60% of
  the time remaining, so a hung attempt is abandoned in time for the retry. A failure changes no
  decision and is reported on `health()`.

A complete, runnable version (a webhook with retries, a health task and an offline mode) is
`examples/webhook_agent.py` in the source repository; the integration guide is on the
[project site](https://percept.percia.ai/).

## Limits

- Static cameras only.
- Thresholds and defaults are starting points, not calibrated values; confidence is advisory.
- A judge failure (timeout, outage, refusal, malformed reply) is never an answer: it is reported
  on `health()` and `coverage()`.
- Small, distant or brief activity in busy surveillance scenes is often missed, because the change
  trigger does not fire on it. Pair such conditions with a heartbeat (`every`), a tighter zone or a
  lower `on_change` threshold, and check `coverage()`.
- Each judge request has an 8 s deadline (with a retry left, an attempt may use at most 60% of the
  time remaining); sparse requests to a slow provider can time out and are reported as failures.
- No performance numbers are published for this release.

## Links

- Project site and documentation: https://percept.percia.ai/
- Support: support@velvee.ai

Percept is developed by Velvee, Inc. Licensed under Apache-2.0.
