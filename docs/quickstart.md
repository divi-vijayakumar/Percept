# Quickstart

Percept 1.0 is in pre-release. Install it from a checkout of this repository:

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e "packages/percept-vision[video,gemini]"   # OpenCV for cameras, the Gemini SDK
```

Extras: `video` (OpenCV, for cameras and files), `gemini`, `claude`, `openai` (one per judge).
`import percept` needs none of them.

## Run the offline demo first

```bash
python examples/quickstart.py
```

It needs no camera, key or network. It draws a tiny synthetic scene, watches it with a scripted
judge (`FakeJudge`) and prints:

- a `met` and a `cleared` for a state;
- an `uncertain` for an event the second look could not confirm;
- coverage (what was looked at, and what came of each look);
- a replay of the run's trace that reproduces every decision exactly.

`examples/frigate_semantic.py` and `examples/kitchen_watch.py` also run offline with no
arguments.

## Watch a camera

```python
import asyncio
import percept

async def main():
    monitor = percept.Monitor(
        judge=percept.judges.gemini("gemini-2.5-flash"),   # reads GEMINI_API_KEY
        conditions=[
            percept.Condition("door_open", "the garage door is open", emit_cleared=True),
            percept.Condition("fall", "a person falls to the floor", kind="event"),
        ],
    )
    async with monitor:
        source = percept.open_source("rtsp://camera.local/stream", fps=5)
        async for delivery in monitor.watch(source, ack="manual"):
            d = delivery.decision
            print(d.type, d.condition_id, d.reason, d.decision_id)
            await delivery.ack()        # after your agent has acted on it

asyncio.run(main())
```

What happens:

1. `open_source` opens the camera on a background thread and reconnects forever if it drops.
   `watch(source)` starts monitoring it.
2. Cheap triggers decide when a condition is worth a look: by default a change in the frame, plus,
   for states, a look every 5 minutes.
3. The judge answers yes, no or unknown, with a confidence, under a hard request budget
   (6 attempts a minute per stream by default).
4. The decision rules turn answers into at most one `met` per real occurrence. Events are checked
   a second time, on the same frames, before they are reported.
5. Each decision is written to a SQLite file (`percept.db` in the working directory) before it is
   delivered, and is delivered again after a crash until you acknowledge it.

## Next

- [concepts.md](concepts.md): triggers, judges, decisions, states and events, coverage.
- [conditions.md](conditions.md): every `Condition` field and how to tune it.
- [judges.md](judges.md): vendors, cascades, prices and budgets.
- [sources.md](sources.md): cameras, files, push sources and link health.
- [delivery.md](delivery.md): the decision contract, acknowledgement, stores, health, traces.
- [decision-semantics.md](decision-semantics.md): the exact rules.
