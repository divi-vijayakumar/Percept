# Percept

**Percept watches a live video stream for plain-language conditions and tells your agent, once per
occurrence, when one becomes true.**

> **Status: 0.9.0 Developer Preview.** The API may change before 1.0.

## Who it is for

- **For** developers building agents that must react to what fixed cameras see: a door left open,
  a person who falls, a package on a loading dock.
- **Not for** moving or non-camera video: PTZ, body-worn, vehicle, drone or phone video, or screen
  content.

Percept decides *when* it is worth asking a model, *what* is true, and *whether* your agent should
hear about it. It doesn't replace your models, your cameras or your agent.

## Install

```bash
pip install "percept-vision[video,gemini]"
```

Extras: `video` (OpenCV, for cameras and files), and one per judge: `gemini`, `claude`, `openai`
(OpenAI and OpenAI-compatible local servers). `import percept` needs none of them.

From a checkout of this repository:

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e "packages/percept-vision[video,gemini]"
```

## A 60-second offline demo

No camera, key or network: a synthetic scene and a scripted judge.

```bash
python examples/quickstart.py
```

```text
percept.MemoryStore keeps decisions in memory only: delivery is not crash-safe
met        box_in_zone   a red box sits in the zone  (confidence 0.95)
cleared    box_in_zone   the zone is empty  (confidence 0.90)
uncertain  package_fell  the verifier could not tell from the original evidence  (first look's confidence 0.80)

coverage: what was looked at, and what came of each look
  box_in_zone   looked at 8 times; 0 of those looks couldn't tell
  package_fell  looked at 2 times; 1 of those looks couldn't tell
  time when nothing could be seen or judged: 0 intervals

replay: 3 decisions from 10 inputs reproduced exactly
```

The first line is expected: the demo keeps decisions in memory, and `MemoryStore` says once that
this is not crash-safe. A real monitor uses the default SQLite store.

## Watch a camera

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

Each decision has a `type` (`met`, `uncertain`, `cleared` or `expired`), a stable `decision_id`,
the `occurrence_id` it belongs to, a machine-readable `cause`, a `reason` for people, a
`confidence`, the time the evidence shows it (`t_observed`), its `cost` and a keyframe. See
[docs/delivery.md](docs/delivery.md#the-decision-contract).

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
- Each judge request has an 8 s deadline; a failure changes no decision and is reported on
  `health()`.

The complete, runnable recipe is [examples/webhook_agent.py](examples/webhook_agent.py): decisions
POSTed to a webhook with retries and acknowledged after a 2xx, plus a health task.
`python examples/webhook_agent.py --offline` runs it with no camera, key or network. Details:
[docs/integration.md](docs/integration.md). For coding agents: [llms.txt](llms.txt) indexes the
documentation.

## Architecture

```mermaid
flowchart LR
  SRC["Source<br/>open_source / PushSource"] --> QG["QualityGate"]
  QG --> SM["SceneMonitor<br/>scene reset"]
  SM --> TR["Triggers<br/>on_change / every / poke<br/>+ zones"]
  SRC -.-> PR["PreRoll clips"]
  TR --> BUD{{"Request budget<br/>max_requests_per_min<br/>max_cost_per_hour"}}
  PR --> BUD
  BUD --> J["Judge<br/>gemini / claude /<br/>openai_compatible / cascade"]
  J --> D["decide<br/>(pure reducer)"]
  D --> OB["Store + outbox<br/>SqliteStore / MemoryStore"]
  OB --> AG["Your agent<br/>monitor.watch() → delivery.ack()"]
  QG -.-> H["health() + coverage()"]
  J -.-> H
  BUD -.-> H
  D -.-> T["trace → replay"]
```

Every stream runs **Trigger → Judge → Decide**. Frames pass a quality gate (corrupt or blurry
frames are dropped) and a scene check (a bumped camera or a day/night switch is a scene reset,
never an event). Cheap triggers then decide when a condition is worth a look: a change inside
its zone, a heartbeat, or a poke from your own code. Looks go to the judge, a vision-language
model behind one interface, under a hard per-stream request budget and an optional cost cap; it
answers yes, no or unknown with a confidence. A pure reducer turns those answers into at most one
`met` per real occurrence (events are verified a second time on the same frames), and each
decision is stored before it is delivered, until your agent acknowledges it. Everything Percept
could not see or judge is reported on `health()` and `coverage()`, and a trace replays every
decision without the video or the judge. More in [docs/concepts.md](docs/concepts.md).

## Limits

- **Static cameras only.** A moving camera looks like a constant scene change.
- **Thresholds and defaults are starting points, not calibrated values.** Confidence is the
  judge's and is advisory.
- **A judge failure is never an answer.** A timeout, outage, refusal or malformed reply changes no
  decision; it is reported on `health()` and `coverage()`.
- **Known limitation: small, distant or brief activity in busy surveillance scenes is often
  missed**, because the change trigger does not fire on it and the judge is never asked. Pair
  such conditions with a heartbeat (`every`), a tighter zone, or a lower `on_change` threshold,
  and check `coverage()`. See [docs/concepts.md](docs/concepts.md#known-limitation-small-distant-or-brief-activity).
- **Slow providers can time out.** Each judge request has an 8 s deadline; sparse requests to a
  slow provider can exceed it and are reported as failures.
- **No performance numbers are published for this release.** Semantic accuracy depends on the
  judge and the condition wording, and Percept can't guarantee it.

What this release assumes, in full: [docs/concepts.md](docs/concepts.md#what-this-release-assumes).

## Documentation

| Page | |
|---|---|
| [docs/quickstart.md](docs/quickstart.md) | install, the offline demo, a first camera |
| [docs/integration.md](docs/integration.md) | the integration recipe: manual ack, health, `FakeJudge` tests, defaults |
| [docs/concepts.md](docs/concepts.md) | Trigger → Judge → Decide, states and events, occurrences, coverage |
| [docs/conditions.md](docs/conditions.md) | every `Condition` field, its default and how to tune it |
| [docs/judges.md](docs/judges.md) | Gemini, Claude, OpenAI-compatible, cascades, prices and budgets |
| [docs/sources.md](docs/sources.md) | `open_source`, push sources, `set_link`, the static-camera assumption |
| [docs/delivery.md](docs/delivery.md) | the decision contract, acknowledgement, stores, health, trace and replay |
| [docs/decision-semantics.md](docs/decision-semantics.md) | the exact decision rules |

- [examples/](examples/): the offline quickstart, Frigate person events, a kitchen state on a
  heartbeat (all run offline with no arguments), and the webhook integration recipe
  (`--offline`).
- [benchmarks/](benchmarks/README.md): the reproducible benchmark harness and its methodology.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) and [AGENTS.md](AGENTS.md). Security reports:
[SECURITY.md](SECURITY.md).

## Support

Questions and bug reports: open an issue at https://github.com/REPLACE-ME/percept/issues, or email
support@velvee.ai.

---

Percept is developed by Velvee, Inc. Licensed under Apache-2.0.
