"""The live Monitor and the pure attention step make the same looks due on the same frames."""

from __future__ import annotations

import asyncio
import logging
import time
import types

import numpy as np
import pytest

import percept
from percept import attention
from percept import monitor as rt
from percept import sources
from percept.contracts import Condition
from percept.delivery import MemoryStore
from percept.judges import FakeJudge
from percept.triggers import every, on_change

TOP_LEFT = (((0.0, 0.0), (0.5, 0.0), (0.5, 0.5), (0.0, 0.5)),)
BOTTOM_RIGHT = (((0.5, 0.5), (1.0, 0.5), (1.0, 1.0), (0.5, 1.0)),)
DT = 0.1    # stream seconds between frames


def image(h: int = 96, w: int = 128) -> np.ndarray:
    y, x = np.mgrid[0:h, 0:w]
    gray = ((x * 7 + y * 13) % 200 + 20).astype(np.uint8)
    return np.dstack([gray, gray, gray])


def patched(img: np.ndarray, top: bool, delta: int = -90) -> np.ndarray:
    out = img.astype(np.int16)
    h, w = img.shape[:2]
    ys, xs = (slice(4, h // 2 - 4), slice(4, w // 2 - 4)) if top else (slice(h // 2 + 4, h - 4), slice(w // 2 + 4, w - 4))
    out[ys, xs] += delta
    return np.clip(out, 0, 255).astype(np.uint8)


def smeared(img: np.ndarray) -> np.ndarray:
    out = img.copy()
    for top in range(0, out.shape[0], 8):
        out[top:top + 8] = out[top]     # every row of a block copies its first row: a corrupt frame
    return out


def frames() -> list[np.ndarray]:
    base = image()
    flipped = 255 - base
    return ([base] * 12 + [patched(base, top=True)] * 6 + [base] * 6 + [smeared(base)] * 2 + [base] * 6
            + [patched(base, top=False)] * 6 + [base] * 4 + [flipped] * 30 + [patched(flipped, top=True)] * 6)


class Flaky:
    """A custom trigger: due every 5th frame, raises on every 7th."""

    def step(self, frame, conditions, state, now):
        if frame.seq % 7 == 3:
            raise RuntimeError("flaky trigger")
        return ({c.id: "flaky" for c in conditions} if frame.seq % 5 == 0 else {}), None


def conditions() -> list[Condition]:
    return [
        Condition("door", "the door is open"),                                         # on_change + every(300)
        Condition("fall", "a person falls", kind="event", zones=TOP_LEFT),             # on_change
        Condition("tick", "a light is on", trigger=every(1.0)),
        Condition("combo", "a box is on the shelf", trigger=(on_change(0.3), every(2.0)), zones=BOTTOM_RIGHT),
        Condition("flaky", "a cart is in the aisle", trigger=Flaky()),
    ]


def live(monkeypatch, fps):
    """Run the Monitor on frames() and record every look on_frame makes due, and the frames it saw."""
    clock = [1000.0]
    fake_time = types.SimpleNamespace(monotonic=lambda: clock[0], time=time.time)
    monkeypatch.setattr(sources, "time", fake_time)
    looks, seen = [], []
    on_frame, enqueue = rt._Stream.on_frame, rt._Stream.enqueue

    def spy_on_frame(self, frame):
        seen.append(frame)
        self._spying = True
        try:
            on_frame(self, frame)
        finally:
            self._spying = False

    def spy_enqueue(self, cid, subject, prio, *, t_trigger=None, frame=None, images=None, trigger=None):
        if getattr(self, "_spying", False):
            looks.append((len(seen) - 1, cid, prio, trigger, t_trigger))
        return enqueue(self, cid, subject, prio, t_trigger=t_trigger, frame=frame, images=images, trigger=trigger)

    monkeypatch.setattr(rt._Stream, "on_frame", spy_on_frame)
    monkeypatch.setattr(rt._Stream, "enqueue", spy_enqueue)

    async def main():
        judge = FakeJudge(lambda r: {item.condition.id: ("no", 0.9, "not seen") for item in r.items})
        async with percept.Monitor(judge, conditions(), store=MemoryStore(), max_requests_per_min=100_000) as m:
            src = percept.PushSource("cam")
            m.watch(src, fps=fps)
            for i, img in enumerate(frames()):
                src.push(img, t_wall=5000.0 + i * DT)
                clock[0] += DT
                for _ in range(100_000):      # wait for the frame to be processed, not for time
                    if len(seen) > i:
                        break
                    await asyncio.sleep(0)
            src.close()
            return m._groups, m._conds, m._labels, list(m._active())

    groups, conds, labels, active = asyncio.run(asyncio.wait_for(main(), 60))
    return looks, seen, groups, conds, labels, active


def offline(seen, groups, conds, labels, active, period):
    state, looks, steps = attention.State(), [], []
    quality, scene = percept.triggers.QualityGate(), percept.triggers.SceneMonitor()
    for i, frame in enumerate(seen):
        s, state = attention.step(frame, state, groups=groups, conditions=conds, labels=labels,
                                  quality=quality, scene=scene, period=period)
        steps.append(s)
        looks += [(i, k.condition_id, rt._LOOK_PRIORITY[k.kind], k.trigger, k.t_trigger) for k in s.looks]
        if s.settled:
            looks += [(i, cid, rt.STATE, "settled", None) for cid in active]
    return looks, steps


@pytest.mark.parametrize("fps", [None, rt.DEFAULT_FPS])
def test_live_monitor_and_offline_attention_make_the_same_looks(monkeypatch, caplog, fps):
    caplog.set_level(logging.ERROR, logger="percept")
    got, seen, groups, conds, labels, active = live(monkeypatch, fps)
    assert len(seen) == len(frames())
    want, steps = offline(seen, groups, conds, labels, active, 1.0 / fps if fps else 0.0)
    assert got == want
    # the sequence exercises every path
    assert any(s.capped for s in steps) == bool(fps)
    assert any(s.vetoed for s in steps)
    assert sum(s.scene_reset for s in steps) == 1 and sum(s.settled for s in steps) == 1
    assert any(s.suppressed for s in steps)
    assert {k for _, _, k, _, _ in want} == {rt.EVENT, rt.STATE, rt.HEARTBEAT}
    assert {"settled", "every(1)", "every(2)", "Flaky"} <= {lbl for _, _, _, lbl, _ in want}
    assert any(lbl.startswith("on_change") for _, _, _, lbl, _ in want)
    failures = sum(len(s.failed_triggers) for s in steps)
    assert failures and sum("trigger" in r.getMessage() and "failed" in r.getMessage() for r in caplog.records) == failures


def test_a_capped_vetoed_or_suppressed_frame_keeps_the_trigger_state():
    cond = Condition("c", "x", trigger=every(1.0))
    groups = [(cond.trigger, cond.trigger, (cond,))]
    kw = dict(groups=groups, conditions={"c": cond}, labels={cond.trigger: "every(1)"},
              quality=percept.triggers.QualityGate(), scene=percept.triggers.SceneMonitor())
    f0 = percept.contracts.Frame("cam", 0, 0, 0.0, 0.0, image())
    s, state = attention.step(f0, attention.State(), period=0.5, **kw)
    assert [k.condition_id for k in s.looks] == ["c"] and state.due == 0.5
    f1 = percept.contracts.Frame("cam", 0, 1, 0.1, 0.1, image())
    s, after = attention.step(f1, state, period=0.5, **kw)
    assert s == attention.Step(capped=True) and after is state
    f2 = percept.contracts.Frame("cam", 0, 2, 0.5, 0.5, smeared(image()))
    s, after = attention.step(f2, state, period=0.5, **kw)
    assert s.vetoed and not s.looks and after.triggers == state.triggers and after.scene is state.scene
