# Percept

**Percept watches a live video stream for plain-language conditions and tells your agent, once per
occurrence, when one becomes true.**

Video in, decisions out. Percept decides *when* it is worth asking a model, *what* is true, and
*whether* your agent should hear about it. It doesn't replace your models, your cameras or your
agent.

> **Status: 1.0 is under active development on `main`.** The API below is the 1.0 design and is not
> released yet. The last published release on PyPI, `percept-vision 0.7.0`, is a research preview
> with a different API.

```python
import percept

monitor = percept.Monitor(
    judge=percept.judges.gemini("gemini-2.5-flash"),
    conditions=[
        percept.Condition("intruder", "A person is inside the fence"),
        percept.Condition("fall", "A person falls to the floor", kind="event"),
    ],
)

async for delivery in monitor.watch("rtsp://camera", ack="manual"):
    await agent.handle(delivery.decision)
    await delivery.ack()
```

## What 1.0 assumes

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

## Repository layout

| Path | What |
|---|---|
| `packages/percept-vision/` | The Python SDK |
| `packages/percept-edge/` | Earlier JavaScript edge package, frozen and outside 1.0 |
| `tools/guardrails/` | Structural checks run in CI |

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Security reports: [SECURITY.md](SECURITY.md).

## License

Apache-2.0
