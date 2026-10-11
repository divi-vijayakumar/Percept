"""A state, watched on a heartbeat: "the stove is on and nobody is in the kitchen".

A state is held while it stays true and reported three ways: ``met`` when it becomes true,
``expired`` when it is still true after ``max_open_s`` (it stays held), and ``cleared`` when the
judge says no ``clear_after_no`` times in a row. A heartbeat trigger (``every``) looks at the
kitchen zone on a fixed period, so a stove left on is noticed even when nothing moves.

Live: ``python examples/kitchen_watch.py --camera rtsp://<camera>`` with
``pip install "percept-vision[video,gemini]"`` and ``GEMINI_API_KEY`` set. Offline (no camera or
key): run it with no arguments; a drawn kitchen is pushed frame by frame and ``FakeJudge`` answers.
"""

import argparse, asyncio, os, time
import numpy as np
import percept
from percept import Condition, MemoryStore, Monitor, PushSource, every
from percept.judges import FakeJudge

KITCHEN = ((0.1, 0.2), (0.9, 0.2), (0.9, 0.95), (0.1, 0.95))   # normalised (x, y) corners


def kitchen(t: float) -> np.ndarray:
    """A drawn kitchen: the cook leaves at 1 s with the stove on and comes back at 5 s."""
    y, x = np.mgrid[0:120, 0:160]
    frame = np.dstack([((x * 3 + y * 7) % 160 + 40).astype(np.uint8)] * 3)
    frame[60:80, 100:130] = (20, 40, 230)                          # the burner glows red (BGR)
    if t < 1.0 or t >= 5.0:
        frame[30:110, 40:65] = (200, 120, 40)                      # the cook, in blue
    return frame


def judge_script(request):
    """Stands in for a vision model. It sees the zone-cropped frame a real judge would."""
    image = request.images[-1]
    stove = ((image[..., 2] > 200) & (image[..., 0] < 60)).any()
    person = ((image[..., 0] > 180) & (image[..., 2] < 80)).any()
    if stove and not person:
        return {"stove_unattended": ("yes", 0.92, "the burner is lit and the kitchen is empty")}
    return {"stove_unattended": ("no", 0.9, "someone is at the stove" if person else "the stove is off")}


async def main(camera) -> None:
    unattended = Condition(
        "stove_unattended", "the stove is on and nobody is in the kitchen",
        zones=(KITCHEN,),            # judge only the kitchen, cropped from the frame
        trigger=every(0.5),          # a heartbeat: look every 0.5 s (a real camera: every(30))
        clear_after_no=2,            # two confident "no" answers in a row clear it
        emit_cleared=True,           # tell the agent when it stops being true
        max_open_s=2.0,              # still true 2 s after met: emit `expired` (a real kitchen: 900)
        recheck_every_s=1.0)
    # A real judge needs a real camera and a key; otherwise FakeJudge stands in, offline.
    judge = percept.judges.gemini("gemini-2.5-flash") if camera and os.environ.get("GEMINI_API_KEY") \
        else FakeJudge(judge_script)
    # max_requests_per_min=600 lets this demo look twice a second; the default (6) protects a budget.
    async with Monitor(judge, [unattended], store=MemoryStore(), max_requests_per_min=600) as monitor:
        if camera:   # open_source owns capture (rate, size, reconnection); watch owns decisions
            source = percept.open_source(camera, stream_id="kitchen", fps=2, max_side=640)
        else:
            source = PushSource("kitchen")
        decisions = monitor.watch(source, ack="manual")

        async def play() -> None:                     # the offline kitchen, 6.5 s at 10 fps
            start = time.time()
            while (t := time.time() - start) < 6.5:
                source.push(kitchen(t))
                await asyncio.sleep(0.1)
            source.close()

        player = None if camera else asyncio.create_task(play())
        async for delivery in decisions:
            d = delivery.decision
            print(f"{d.type:<8} {d.condition_id}  cause={d.cause}  {d.reason}")
            await delivery.ack()                      # acknowledge once the agent has acted
        if player:
            await player
        stats = monitor.stats()["streams"]["kitchen"]
        print(f"looked {stats['requests']} times; delivered {stats['delivered']}, acked {stats['acked']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--camera", help="a camera URI such as rtsp://...; omit to run offline")
    asyncio.run(main(parser.parse_args().camera))
