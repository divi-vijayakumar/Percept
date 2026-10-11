# Integrating Percept into an app

The recipe for wiring Percept into an application or agent, with the defaults that decide how it
behaves. The complete, runnable version is
[examples/webhook_agent.py](../examples/webhook_agent.py) (`python examples/webhook_agent.py
--offline` runs it with no camera, key or network).

## 1. Install with the judge's extra

```bash
pip install "percept-vision[video,gemini]"     # or [video,claude], or [video,openai]
```

`video` is OpenCV, for cameras and files (`open_source`); a `PushSource` fed by your own capture
doesn't need it. Each judge needs its own extra. `import percept` works without any, but building
a judge whose SDK is missing raises:

```text
ImportError: this judge needs the 'google.genai' package: pip install 'percept-vision[gemini]'
```

(`'anthropic'` and `[claude]`, `'openai'` and `[openai]` likewise.) `percept.judges.gemini` reads
`GEMINI_API_KEY` (then `GOOGLE_API_KEY`) and raises `ValueError` when neither is set.

## 2. The production loop

```python
import asyncio, logging
import percept

log = logging.getLogger("app")


async def log_health(monitor: percept.Monitor) -> None:
    # health() is an async iterator, not a list. It ends when the monitor closes.
    async for h in monitor.health():
        log.warning("percept health %s on %s: %s (x%d)", h.kind, h.stream_id, h.detail, h.count)


async def main() -> None:
    monitor = percept.Monitor(
        percept.judges.gemini("gemini-2.5-flash"),          # needs the [gemini] extra
        [percept.Condition("person_fell", "a person falls down", kind="event"),
         percept.Condition("door_open", "the door is left open", emit_cleared=True)],
        store=percept.SqliteStore("percept.db"),           # the default, shown explicitly
    )
    health = asyncio.create_task(log_health(monitor))
    async with monitor:
        source = percept.open_source("rtsp://camera.local/stream", stream_id="hallway")
        async for delivery in monitor.watch(source, ack="manual"):
            if await send_to_my_app(delivery.decision):    # your webhook, queue or agent; retry inside
                await delivery.ack()                       # ack only once it was handled
    await health

asyncio.run(main())
```

- **Events and states.** `kind="event"` is a moment ("a person falls down"): verified a second
  time on the same frames, reported once as `met`, then rearmed after `rearm_s` (120 s). The
  default `kind="state"` is something that stays true ("the door is left open"): `met` when it
  becomes true, and `cleared` when it stops, if `emit_cleared=True`.
- **Ack only after delivery succeeded.** With `ack="manual"` a decision leaves the store only when
  you `await delivery.ack()`. Percept does **not** retry for you within a `watch()`: an
  unacknowledged decision is not delivered again by the same loop, and there is no `nack`. So
  retry a failed POST yourself, with backoff, and ack on a 2xx. A decision you never ack stays in
  the store and is delivered again, first and with the same `decision_id`, by the next `watch()`
  of the stream or after a restart. Once `max_unacked` (100) decisions are unacknowledged,
  delivery on that stream pauses (health `store_lag`) until you ack one. Details in
  [delivery.md](delivery.md#at-least-once-delivery).
- **Deduplicate on `decision_id`.** Delivery is at least once.
- **Health in a background task.** `monitor.health()` buffers events from the moment the monitor
  is built (a bounded ring of 256 aggregated entries), so starting the task just before or just
  after `watch()` loses nothing. Use one consumer.
- **The wire form.** `decision.to_dict()` is JSON-ready and versioned (`contract_version`); see
  [delivery.md](delivery.md#the-decision-contract).

## 3. Test offline with FakeJudge

`percept.judges.FakeJudge(script)` replaces the vendor judge in tests and demos: no key, no
network, and the same prompt, parsing, retry and failure handling as a real judge. Feed frames
with a `PushSource`:

```python
import numpy as np
import percept

def script(request):
    # Called once per judge request. request.images: the BGR uint8 frames a real judge would see
    # (cropped to zones); request.items: the conditions asked; request.purpose: "detect" or "verify".
    open_ = (request.images[-1][..., 1] > 200).any()
    return {"door_open": ("yes", 0.9, "open") if open_ else ("no", 0.9, "shut")}

async def test_door():
    async with percept.Monitor(percept.judges.FakeJudge(script),
                               [percept.Condition("door_open", "the door is open", emit_cleared=True,
                                                  trigger=percept.every(0.5))],
                               store=percept.MemoryStore(), max_requests_per_min=600) as monitor:
        camera = percept.PushSource("test-cam")
        ...  # camera.push(frame) from a task, then camera.close(): watch() ends once all is delivered
        types = [d.decision.type async for d in monitor.watch(camera)]
```

- A script answers `{condition_id: (answer, confidence, reason)}`, `answer` being `"yes"`,
  `"no"` or `"unknown"`. It may also return a raw string (a malformed reply) or an exception
  instance (`percept.judges.base.RetryableError()`, `RefusalError()`, `TimeoutError()`) to test
  failures. A script that looks at the images lets states clear, as a real judge would.
- `MemoryStore` logs once, `percept.MemoryStore keeps decisions in memory only: delivery is not
  crash-safe`. In a demo or a test that warning is expected; in production keep the default
  SQLite store.
- Raise `max_requests_per_min` and use a short `every()` trigger in tests, so a few seconds of
  frames get enough looks. Keep the defaults in production.

## 4. Defaults that decide behaviour

| Default | Value | Consequence |
|---|---|---|
| `Monitor(max_requests_per_min=...)` | `6` attempts per stream in any 60 s | A built-in judge reserves 2 attempts a request (one try, one retry), so about 5 requests a minute when none retries, 3 when every one does. Due conditions with the same evidence share one request. A detection that waits too long for the budget (3 s for an event, 10 s for a state) is dropped (`dropped_budget`, health `budget_exhausted`); heartbeats and rechecks wait for the next free slot. |
| state trigger | `on_change()` plus `every(300)` | A state is looked at when its zone changes, and at least every 5 minutes. |
| event trigger | `on_change()` | An event is looked at only when its zone changes (or when you `poke()`). |
| `Condition.recheck_every_s` | `60.0` (states) | While a state is held it is looked at again at least this often. |
| `Condition.clear_after_no` | `3` (states) | A held state clears only after 3 confident "no" answers in a row. Weak, unknown and failed answers neither count nor reset. With the default looks (one when the closing changes the zone, then a recheck every 60 s), a door closed in a quiet scene is `cleared` 1 to 2 minutes after it closed, later if a look fails or is unsure. Lower `clear_after_no` or `recheck_every_s` (more requests) to clear sooner. |
| `Monitor(store=...)` | `SqliteStore("percept.db")` in the working directory | Crash-safe; one process per file (a second one raises `StoreLocked`). `MemoryStore()` is opt-in and warns. |
| `watch(ack=...)` | `"auto"` | Acknowledges a decision when the loop asks for the next one. Use `"manual"` when the hand-off can fail. |
| `watch(max_unacked=...)` | `100` | Unacknowledged decisions before delivery pauses on the stream. |
| `watch(fps=...)` | `5` | Frames per second checked by triggers; capture is not slowed. |
| judge deadline | 8 s a request | A slower reply fails as `timeout` and changes no decision. |

None of these values is calibrated. Check `monitor.coverage(stream_id)` and the
[trace](delivery.md#trace-and-replay) on your own footage, and tune the
[conditions](conditions.md).
