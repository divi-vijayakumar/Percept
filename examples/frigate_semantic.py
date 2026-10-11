"""Frigate finds the person; Percept decides what they are doing, once per occurrence.

Each Frigate person event (MQTT topic ``frigate/events``) pokes Percept to look at that moment, for
that person: "a person is leaving a package at the door". Percept judges the clip around the moment,
verifies it on the same frames, and tells the agent once. Live: ``python examples/frigate_semantic.py
--mqtt <broker> --rtsp rtsp://<frigate>:8554/front_door`` with ``pip install paho-mqtt
"percept-vision[video,gemini]"`` and ``GEMINI_API_KEY`` set. Offline (no camera, broker or key): run it
with no arguments, or ``--replay events.jsonl`` (one Frigate payload per line, with ``after.frame_time``).
"""

import argparse, asyncio, json, os, time
import numpy as np
import percept
from percept import Condition, MemoryStore, Monitor, PushSource
from percept.judges import FakeJudge

SAMPLE = [{"type": "new", "after": {"id": "a1", "camera": "front_door", "label": "person", "frame_time": 0.8}},
          {"type": "new", "after": {"id": "c7", "camera": "front_door", "label": "car", "frame_time": 1.0}},
          {"type": "new", "after": {"id": "b2", "camera": "front_door", "label": "person", "frame_time": 2.3}},
          {"type": "update", "after": {"id": "b2", "camera": "front_door", "label": "person", "frame_time": 2.7}}]


class WhenPoked:
    """A trigger that never fires by itself: only Frigate's events make Percept look."""
    def step(self, frame, conditions, state, now):
        return {}, state


def doorstep(t: float) -> np.ndarray:
    """A drawn doorstep: a person stands there from 0.6 s to 3.0 s; a package appears at 2.0 s."""
    y, x = np.mgrid[0:120, 0:160]
    frame = np.dstack([((x * 5 + y * 11) % 180 + 30).astype(np.uint8)] * 3)
    if 0.6 <= t < 3.0:
        frame[20:100, 60:85] = (60, 40, 30)                       # the person (BGR)
    if t >= 2.0:
        frame[85:105, 95:125] = (40, 80, 150)                     # a brown package
    return frame


def judge_script(request):
    """Stands in for a vision model: yes when a package appears during the clip it is shown."""
    package = [bool(((im[..., 2] > 120) & (im[..., 0] < 60)).mean() > 0.01) for im in request.images]
    return {i.condition.id: ("yes", 0.9, f"{i.subject} sets a package down") if package[-1] and not package[0]
            else ("no", 0.85, "nobody puts anything down") for i in request.items}


async def main(args) -> None:
    live = bool(args.rtsp)
    # A real judge needs a real camera and a key (see judges.md for prices and budgets);
    # otherwise FakeJudge stands in, offline.
    judge = percept.judges.gemini("gemini-2.5-flash") if live and os.environ.get("GEMINI_API_KEY") \
        else FakeJudge(judge_script)
    door = Condition("package_left", "a person is leaving a package at the door", kind="event",
                     trigger=WhenPoked())
    # store=None is the default durable SQLite store (percept.db); MemoryStore suits a demo.
    async with Monitor(judge, [door], store=None if live else MemoryStore(), max_requests_per_min=30) as monitor:
        source = percept.open_source(args.rtsp, stream_id=args.camera) if live else PushSource(args.camera)
        decisions = monitor.watch(source, ack="manual")

        def on_event(payload: dict) -> None:            # runs on the monitor's event loop
            after = payload.get("after") or {}
            if payload.get("type") in ("new", "update") and after.get("label") == "person" \
                    and after.get("camera") == args.camera:
                monitor.poke(args.camera, "package_left", subject=f"person {after['id']}",
                             observed_at=after["frame_time"] if live else time.time(),
                             reason=f"frigate {payload['type']}: person {after['id']}")

        async def offline() -> None:                    # replay events against the drawn doorstep
            events = [json.loads(line) for line in open(args.replay) if line.strip()] if args.replay else SAMPLE
            t0, start = min(e["after"]["frame_time"] for e in events) - 0.8, time.time()
            while (t := time.time() - start) < 4.0:
                source.push(doorstep(t))
                while events and events[0]["after"]["frame_time"] - t0 <= t:
                    on_event(events.pop(0))
                await asyncio.sleep(0.1)
            source.close()

        if args.mqtt:                                   # Frigate's events, live, from its broker
            import paho.mqtt.client as mqtt
            loop, client = asyncio.get_running_loop(), mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
            client.on_message = lambda _c, _u, msg: loop.call_soon_threadsafe(on_event, json.loads(msg.payload))
            client.connect(args.mqtt)
            client.subscribe("frigate/events")
            client.loop_start()
        player = None if live else asyncio.create_task(offline())
        async for delivery in decisions:
            d = delivery.decision
            print(f"{d.type:<9} {d.condition_id} ({d.subject}): {d.reason}  [{d.confidence:.2f}]")
            await delivery.ack()                        # after your agent has acted on it
        if player:
            await player


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mqtt", help="Frigate's MQTT broker host (needs paho-mqtt)")
    parser.add_argument("--rtsp", help="the camera's RTSP stream, e.g. Frigate's restream")
    parser.add_argument("--camera", default="front_door", help="the Frigate camera name")
    parser.add_argument("--replay", help="a JSON-lines file of Frigate event payloads, replayed offline")
    asyncio.run(main(parser.parse_args()))
