"""The evaluation driver: attention-only parity with the Monitor, the caching spend cap, frozen
registration of the test split, and side-by-side sweeps."""

from __future__ import annotations

import asyncio

import numpy as np
import pytest

import percept
from percept import PriceModel, plans
from percept import monitor as rt
from percept.judges import FakeJudge
from percept.plans import run as ev

cv2 = pytest.importorskip("cv2")

FPS, SECONDS = 10, 8.0
PRICE = PriceModel(0.10, 0.40, 258)


def write_clip(path, square=(2.0, 5.0)) -> str:
    y, x = np.mgrid[0:120, 0:160]
    wall = ((x * 7 + y * 13) % 200 + 20).astype(np.uint8)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), FPS, (160, 120))
    for n in range(int(SECONDS * FPS)):
        frame = np.dstack([wall] * 3).copy()
        if square[0] <= n / FPS < square[1]:
            frame[20:60, 20:60] = (40, 40, 220)
        writer.write(frame)
    writer.release()
    return path.name


def red(request) -> bool:
    im = request.images[-1]
    return bool(((im[..., 2] > 180) & (im[..., 0] < 90)).any())


def script(request):
    return {i.condition.id: ("yes", 0.95, "red square") if red(request) else ("no", 0.9, "none") for i in request.items}


def plan(rpm: int = 600, trigger=None, revision: int = 1) -> dict:
    concern = {"id": "red", "source": "cam", "description": "a red square appears",
               "judgment": {"route": "fast"}, "decision": {"kind": "event"}}
    if trigger is not None:
        concern["attention"] = {"any": [trigger]}
    return {"schema_version": plans.SCHEMA_VERSION, "plan_id": "eval-test", "revision": revision,
            "sources": [{"id": "cam", "kind": "video", "max_side": 160}],
            "budget": {"max_requests_per_min": rpm}, "concerns": [concern]}


def labels_for(files, group_of=lambda f: f):
    return plans.load_labels({
        "conditions": [{"label": "red", "text": "a red square appears"}],
        "clips": [{"file": f, "group": group_of(f), "occurrences": [{"label": "red", "start": 2.0, "end": 5.0}]}
                  for f in files]})


@pytest.fixture(scope="module")
def clip(tmp_path_factory):
    root = tmp_path_factory.mktemp("clips")
    return root, write_clip(root / "a.avi")


def compiled(judge=None, **kw):
    return plans.compile(plan(**kw), {"fast": judge or FakeJudge(script, price=PRICE)})


# --------------------------------------------------------------------------- attention only


def live_looks(monkeypatch, c, path, rate):
    """The looks a live Monitor makes due on the file, as (condition id, clip time)."""
    looks, seen = [], []
    on_frame, enqueue = rt._Stream.on_frame, rt._Stream.enqueue

    def spy_on_frame(self, frame):
        seen.append(frame)
        self._spying = True
        try:
            on_frame(self, frame)
        finally:
            self._spying = False

    def spy_enqueue(self, cid, subject, prio, **kw):
        if getattr(self, "_spying", False):
            looks.append((cid, seen[-1].t_wall - seen[0].t_wall))
        return enqueue(self, cid, subject, prio, **kw)

    monkeypatch.setattr(rt._Stream, "on_frame", spy_on_frame)
    monkeypatch.setattr(rt._Stream, "enqueue", spy_enqueue)

    async def main():
        async with percept.Monitor(**c.monitor_kwargs, store=percept.MemoryStore()) as m:
            source = percept.open_source(str(path), fps=rate, max_side=c.sources[0]["max_side"])
            async for _ in m.watch(source, fps=rate):
                pass

    asyncio.run(asyncio.wait_for(main(), 60))
    return looks


@pytest.mark.parametrize("trigger", [None, {"every": 0.5}])
def test_attention_only_fires_are_the_looks_a_live_monitor_makes(monkeypatch, clip, trigger):
    root, name = clip
    c = compiled(trigger=trigger)
    runs = ev.attention_only(c, labels_for([name]), [name], video_root=root)
    want = live_looks(monkeypatch, c, root / name, rt.DEFAULT_FPS)
    assert want and [(f.label, round(f.t, 6)) for f in runs[0].fires] == [(cid, round(t, 6)) for cid, t in want]
    assert runs[0].file == name and runs[0].cost_usd == 0 and abs(runs[0].seconds - SECONDS) < 0.5
    score = plans.evaluate(labels_for([name]), runs)
    assert score.funnel["candidate"] == 1 and score.cost_usd == 0
