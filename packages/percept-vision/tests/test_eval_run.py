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
SPEED = 4.0       # 4x real time keeps the tests short; the demo budget leaves room for every look


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


# --------------------------------------------------------------------------- the full runtime and the cache


def counting(sent):
    def answer(request):
        sent.append(request.request_id)
        return script(request)
    return FakeJudge(answer, price=PRICE)


def test_run_finds_the_square_and_a_second_identical_run_costs_nothing(clip):
    root, name = clip
    sent, cache, labels = [], {}, labels_for([name])
    c = compiled(counting(sent))
    first = ev.run(c, labels, [name], video_root=root, judge_cap_usd=1.0, cache=cache, speed=SPEED)
    score = plans.evaluate(labels, first)
    assert score.tp == 1 and score.fp == 0 and score.cost_usd > 0 and first[0].looks and sent
    n = len(sent)
    second = ev.run(c, labels, [name], video_root=root, judge_cap_usd=1.0, cache=cache, speed=SPEED)
    assert len(sent) == n and sum(r.cost_usd for r in second) == 0 and plans.evaluate(labels, second).tp == 1


def request(i: int, image=None) -> percept.JudgeRequest:
    cond = percept.Condition("red", "a red square appears", kind="event")
    img = np.full((8, 8, 3), i, np.uint8) if image is None else image
    return percept.JudgeRequest(f"r{i}", "cam", 0, (percept.JudgeItem(cond),), (img,), (0.0, 1.0))


def test_caching_judge_sends_nothing_for_an_identical_request():
    sent = []
    guard = ev.CachingJudge(counting(sent), cap_usd=1.0)
    a = asyncio.run(guard.judge(request(1)))
    b = asyncio.run(guard.judge(request(2, image=np.full((8, 8, 3), 1, np.uint8))))
    assert sent == ["r1"] and guard.requests == 1 and guard.hits == 1 and b.usage == ()
    assert b.observations[0].request_id == "r2" and b.observations[0].answer == a.observations[0].answer


def test_the_cap_is_never_passed_with_requests_in_flight_and_refusals_fail_with_budget():
    async def main():
        gate, inner = asyncio.Event(), FakeJudge(lambda r: {"red": ("yes", 0.9, "seen")}, price=PRICE)
        one = ev.CachingJudge(inner, cap_usd=None).reservation(request(0))
        guard = ev.CachingJudge(inner, cap_usd=2.5 * one, cache=None)
        real = inner.judge

        async def slow(r):
            await gate.wait()
            return await real(r)

        inner.judge = slow
        tasks = [asyncio.create_task(guard.judge(request(i))) for i in range(5)]
        await asyncio.sleep(0)
        held = guard.held
        gate.set()
        return guard, held, await asyncio.gather(*tasks)

    guard, held, results = asyncio.run(main())
    refused = [r for r in results if r.observations[0].failed]
    assert held <= guard.cap and len(refused) == 3 and guard.refused == 3 and guard.tripped.is_set()
    assert all(r.observations[0].failure == "budget" and r.observations[0].answer != "no" for r in refused)
    assert guard.spent <= guard.cap and guard.held == 0
