"""A watch plan, offline: one document describes the whole watch, and Percept compiles it.

The plan below (a plain dict; load it from JSON or YAML yourself) has one source and two concerns:
a lamp checked every 0.2 s, and a package falling, judged only when our own shelf sensor fires.
``percept.plans.compile`` checks it, says where every setting came from (``explain``), checks the
periodic demand against the request budget (``feasibility``) and returns the Monitor's arguments.
Watch plans are experimental: the schema may change before it is stable.

The judge is ``FakeJudge``, scripted below. Run it with ``python examples/watch_plan.py``.
"""

import asyncio
import json

import numpy as np

import percept
from percept.judges import FakeJudge

PLAN = {
    "schema_version": "percept.watch/v1alpha1",
    "plan_id": "dock-watch",
    "revision": 1,
    "sources": [{"id": "dock-cam", "kind": "video", "max_side": 320}],
    # A demo budget, so the lamp can be checked every 0.2 s; the default (6) protects a real one.
    "budget": {"max_requests_per_min": 600},
    "concerns": [
        {"id": "lamp_on", "source": "dock-cam", "description": "the lamp is on",
         "attention": {"any": [{"every": 0.2}]},
         "judgment": {"route": "fast"},
         "decision": {"kind": "state", "clear_after_no": 1, "emit_cleared": True}},
        {"id": "package_fell", "source": "dock-cam", "description": "a package fell off the shelf",
         "attention": {"any": [{"external_event": "shelf_sensor"}]},
         "judgment": {"route": "fast"},
         "decision": {"kind": "event"}},
    ],
    "metadata": {"note": "free-form; not part of the plan hash"},
}


def scene(lit: bool) -> np.ndarray:
    y, x = np.mgrid[0:120, 0:160]
    gray = ((x * 7 + y * 13) % 200 + 20).astype(np.uint8)
    frame = np.dstack([gray, gray, gray])
    if lit:
        frame[10:30, 10:30] = (40, 40, 250)
    return frame


def judge_script(request):
    """Stands in for a vision model: the lamp is on when a bright red patch is visible."""
    lit = bool(((request.images[-1][..., 2] > 240) & (request.images[-1][..., 0] < 80)).any())
    out = {}
    for item in request.items:
        if item.condition.id == "lamp_on":
            out["lamp_on"] = ("yes", 0.95, "the lamp is lit") if lit else ("no", 0.9, "the lamp is dark")
        else:
            out["package_fell"] = ("no", 0.9, "nothing falls")
    return out


async def main() -> None:
    compiled = percept.plans.compile(PLAN, {"fast": FakeJudge(judge_script)})
    explained = compiled.explain()
    print(f"plan {explained['plan_id']} r{explained['revision']} hash {compiled.plan_hash[:12]}")
    for concern in explained["concerns"]:
        from_plan = sorted(k for k, v in concern["fields"].items() if v["origin"] == "plan")
        print(f"  {concern['id']}: trigger {concern['trigger']['value']}, set by the plan: {from_plan}")
    print("feasibility:", json.dumps(compiled.feasibility()["sources"]["dock-cam"]["verdict"]))
    print("poke on events:", compiled.events)

    async with percept.Monitor(**compiled.monitor_kwargs, store=percept.MemoryStore()) as monitor:
        camera = percept.PushSource("dock-cam", max_side=compiled.sources[0]["max_side"])
        decisions = monitor.watch(camera)

        async def play():
            for i in range(30):
                camera.push(scene(lit=5 <= i < 18))
                await asyncio.sleep(0.05)
            camera.close()

        player = asyncio.create_task(play())
        async for delivery in decisions:
            d = delivery.decision
            print(f"{d.type:<8} {d.condition_id}: {d.reason}")
        await player


if __name__ == "__main__":
    asyncio.run(main())
