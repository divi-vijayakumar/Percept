# Percept

**Percept watches a live video stream for plain-language conditions and tells your agent, once per
occurrence, when one becomes true.**

Video in, decisions out. Percept decides *when* it is worth asking a model, *what* is true, and
*whether* your agent should hear about it. It doesn't replace your models, your cameras or your
agent.

> **Status: 0.9.0 Developer Preview.** The API below may still change before 1.0. It is not on
> PyPI yet, so install it from a checkout. The last release on PyPI, `percept-vision 0.7.0`, is a
> research preview with a different API.

```python
import percept

monitor = percept.Monitor(
    judge=percept.judges.gemini("gemini-2.5-flash"),
    conditions=[
        percept.Condition("door_open", "the garage door is open", emit_cleared=True),
        percept.Condition("fall", "a person falls to the floor", kind="event"),
    ],
)

source = percept.open_source("rtsp://camera.local/stream", fps=5)   # capture: rate, size, reconnection
async for delivery in monitor.watch(source, ack="manual"):         # monitoring and delivery
    d = delivery.decision
    await agent.handle(d)     # d.type, d.condition_id, d.cause, d.reason, d.confidence, d.decision_id
    await delivery.ack()      # unacknowledged decisions are delivered again after a restart
```

Each decision has a `type` (`met`, `uncertain`, `cleared` or `expired`), a stable `decision_id`
to deduplicate on, the `occurrence_id` it belongs to, a machine-readable `cause`, a `reason` for
people, a `confidence`, the time the evidence shows it (`t_observed`), its `cost` and a keyframe.

Try it offline first, with no camera, key or network: `python examples/quickstart.py`.

## Install (from a checkout)

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e "packages/percept-vision[video,gemini]"
```

Extras: `video` (OpenCV), `gemini`, `claude`, `openai` (also OpenAI-compatible local servers).

## What this release assumes

1. **Static cameras.** Fixed-viewpoint cameras only. Not PTZ, body-worn, vehicle, drone or phone
   video, and not screen content. A bumped camera or a day/night IR switch is a scene reset, never
   an event.
2. **Replaceable models.** Every judge (a vision-language model, a detector, a rule) plugs in through
   one interface and returns the same observation shape. The core imports no vendor SDK.
3. **A hard request budget, a configurable cost policy.** Each stream has a hard limit on outbound
   model requests and an optional cost cap. Measured cost against simpler baselines will be
   published with the release; it is not assumed.
4. **Two kinds of reliability.** Decision reliability (one event per occurrence, no lost decisions,
   an outage never reported as an answer) is Percept's job and is tested exhaustively. Semantic
   accuracy (is that person falling or lying on a sofa?) is the model's job: Percept improves it
   and reports it per condition, but can't guarantee it.

## In brief

- **Delivery is at least once.** A decision is stored (SQLite, `percept.db`, by default) before
  it is delivered, and stays until acknowledged. `ack="auto"` acknowledges when the loop body
  finishes; `ack="manual"` when you call `delivery.ack()`.
- **App-owned capture.** Push frames with `percept.PushSource(...).push(image)` or
  `monitor.push(stream_id, image)`, and report your feed's outages with `set_link(False)` /
  `set_link(True)`, so a dead feed is a `link_down` blind interval, not silence.
- **Cost.** `monitor.stats()` reports settled `cost_usd`, the `cost_reserved_usd` still held
  against `max_cost_per_hour`, and per-condition `requests`, `attempts` and `cost_usd` under
  `stats()["streams"][stream_id]["conditions"]`.
- **Coverage.** `monitor.coverage(stream_id)` says what was looked at, what came of each look, and
  when Percept could not see or judge at all.

## Documentation

| Page | |
|---|---|
| [docs/quickstart.md](docs/quickstart.md) | install, the offline demo, a first camera |
| [docs/concepts.md](docs/concepts.md) | Trigger → Judge → Decide, states and events, occurrences, coverage |
| [docs/conditions.md](docs/conditions.md) | every `Condition` field, its default and how to tune it |
| [docs/judges.md](docs/judges.md) | Gemini, Claude, OpenAI-compatible, cascades, prices and budgets |
| [docs/sources.md](docs/sources.md) | `open_source`, push sources, `set_link`, the static-camera assumption |
| [docs/delivery.md](docs/delivery.md) | the decision contract, acknowledgement, stores, health, trace and replay |
| [docs/decision-semantics.md](docs/decision-semantics.md) | the exact decision rules |

Examples: [quickstart](examples/quickstart.py), [Frigate person events](examples/frigate_semantic.py),
[a kitchen state on a heartbeat](examples/kitchen_watch.py). Benchmarks and how to reproduce them:
[benchmarks/README.md](benchmarks/README.md).

## Repository layout

| Path | What |
|---|---|
| `packages/percept-vision/` | The Python SDK |
| `docs/`, `examples/` | Documentation and runnable examples |
| `benchmarks/` | The reproducible benchmark harness |
| `tools/` | Structural checks, validation and the pre-publication scan |

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Security reports: [SECURITY.md](SECURITY.md).

## License

Apache-2.0
