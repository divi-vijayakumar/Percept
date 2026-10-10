"""Percept in one minute, offline: no camera, no API key, no network.

We make a tiny synthetic video with numpy: a static scene where a red box enters a loading zone,
stays a while, and leaves. We watch it for two conditions:

* a *state*, "a red box is in the loading zone": reported when it becomes true (``met``) and when
  it stops being true (``cleared``);
* an *event*, "a package fell off the shelf": events are checked twice. Here our own detector
  pokes Percept, the judge says yes, but on a second look at the same clip it can't tell, so
  Percept reports ``uncertain`` instead of a confident alarm.

The judge is ``FakeJudge``, scripted below; swap in ``percept.judges.gemini()`` (or another real
judge) to run the same code on a real camera. Run it with ``python examples/quickstart.py``.
"""

import asyncio
import tempfile
from pathlib import Path

import numpy as np

import percept
from percept import Condition, Monitor, MemoryStore, PushSource, every
from percept.judges import FakeJudge

LOADING_ZONE = ((0.5, 0.4), (0.9, 0.4), (0.9, 0.9), (0.5, 0.9))   # normalised (x, y) corners
SHELF = ((0.05, 0.05), (0.4, 0.05), (0.4, 0.35), (0.05, 0.35))


def scene(box_in_zone: bool) -> np.ndarray:
    """One 160x120 frame: a textured wall, plus a red box (BGR colour order) when asked."""
    y, x = np.mgrid[0:120, 0:160]
    gray = ((x * 7 + y * 13) % 200 + 20).astype(np.uint8)
    frame = np.dstack([gray, gray, gray])
    if box_in_zone:
        frame[70:95, 100:130] = (30, 30, 220)
    return frame


def judge_script(request):
    """Stands in for a vision model. It sees the same (zone-cropped) images a real judge would."""
    answers = {}
    for item in request.items:
        if item.condition.id == "box_in_zone":
            red = (request.images[-1][..., 2] > 180) & (request.images[-1][..., 0] < 80)
            answers["box_in_zone"] = ("yes", 0.95, "a red box sits in the zone") if red.any() \
                else ("no", 0.9, "the zone is empty")
        elif request.purpose == "detect":
            answers["package_fell"] = ("yes", 0.8, "something seems to drop")
        else:                       # the second look, at the same clip
            answers["package_fell"] = ("unknown", 0.5, "can't tell from this clip")
    return answers


async def main(trace_path: Path) -> None:
    conditions = [
        Condition("box_in_zone", "a red box is in the loading zone", zones=(LOADING_ZONE,),
                  trigger=every(0.3), clear_after_no=1, emit_cleared=True),
        Condition("package_fell", "a package fell off the shelf", kind="event", zones=(SHELF,)),
    ]
    # MemoryStore keeps decisions in memory (it warns that this isn't crash-safe); the default
    # store is a SQLite file. `trace=` records every step so the run can be replayed below.
    # max_requests_per_min=600 lets this demo look every 0.3 s; the default (6) protects a real budget.
    async with Monitor(FakeJudge(judge_script), conditions, store=MemoryStore(),
                       max_requests_per_min=600, trace=trace_path) as monitor:
        camera = PushSource("dock-cam")
        # ack="auto" acknowledges each decision once the loop body finishes: fine for a demo.
        # A real agent uses ack="manual" and calls `await delivery.ack()` after acting.
        decisions = monitor.watch(camera, ack="auto")

        async def play():
            for i in range(60):                       # 3 seconds at 20 frames per second
                camera.push(scene(box_in_zone=15 <= i < 30))
                if i == 45:                           # your own detector says "look now"
                    monitor.poke("dock-cam", "package_fell", reason="shelf sensor tripped")
                await asyncio.sleep(0.05)
            camera.close()

        player = asyncio.create_task(play())
        async for delivery in decisions:
            d = delivery.decision
            # An uncertain decision carries the first look's confidence, not a verified one.
            label = "first look's confidence" if d.type == "uncertain" else "confidence"
            print(f"{d.type:<10} {d.condition_id:<13} {d.reason}  ({label} {d.confidence:.2f})")
        await player

        coverage = monitor.coverage("dock-cam")
        print("\ncoverage: what was looked at, and what came of each look")
        for condition_id, fates in coverage["fates"].items():
            unsure = fates.get("judged_weak", 0) + fates.get("failed", 0)   # no usable answer
            looks = fates.get("judged", 0) + unsure
            print(f"  {condition_id:<13} looked at {looks} times; {unsure} of those looks couldn't tell")
        print(f"  time when nothing could be seen or judged: {len(coverage['blind_intervals'])} intervals")

    # Every decision can be reproduced from the trace, without the video or the judge.
    result = percept.trace.replay(trace_path)
    verdict = "reproduced exactly" if result.matches else f"DIFFER: {result.differences}"
    print(f"\nreplay: {result.recorded} decisions from {result.inputs} inputs {verdict}")


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmp:
        asyncio.run(main(Path(tmp) / "trace.jsonl"))
