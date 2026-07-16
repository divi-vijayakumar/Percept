"""Command-seam: RuntimeSession.pause()/resume() = SLEEP (cost-free pause, keep the frame stream).

While paused the watch keeps capturing and streaming frames (the camera panel stays live) but spends
NO gate/judge/incident — a cost-free pause that resumes instantly. `paused=None` (default) is
byte-for-byte the pre-PR path.
"""
from __future__ import annotations

import asyncio
import unittest

import percept
from percept.adapters import Goal, Verdict
from percept.edge.tier0 import gray_frame
from percept.runtime.session import start

STILL = gray_frame(8, 8, 0)
MOVED = gray_frame(8, 8, 255)


class _AlwaysFire:
    async def judge(self, c, f, **k):
        return Verdict(satisfied=True, confidence=0.95, evidence="x")

    async def detect(self, f, g):
        return []

    async def detect_events(self, f, g):
        return []


def _core():
    core = percept.Percept.create(vision=_AlwaysFire(), discover_plugins=False)
    core.add_goal(Goal(id="z", condition="x", say="hi"))
    return core


async def _paced_source(n=60, dt=0.005):
    for i in range(n):
        yield (float(i), STILL if i % 2 == 0 else MOVED, "tok")  # alternating motion -> escalates
        await asyncio.sleep(dt)


class PauseResumeTests(unittest.TestCase):
    def test_pause_is_cost_free_sleep_resume_re_enables_judging(self):
        async def go():
            core = _core()
            frames: list = []
            session = await start(core, source=_paced_source(), on_frame=lambda t, f: frames.append(f))
            session.pause()
            await asyncio.sleep(0.10)  # ~20 frames go by, PAUSED
            judged_while_paused = core.metrics.snapshot().get("funnel.judged", 0)
            frames_while_paused = len(frames)
            session.resume()
            await asyncio.sleep(0.25)  # remaining frames judged
            judged_after_resume = core.metrics.snapshot().get("funnel.judged", 0)
            await session.stop()
            return judged_while_paused, frames_while_paused, judged_after_resume

        judged_paused, frames_paused, judged_resumed = asyncio.run(go())
        self.assertGreater(frames_paused, 0, "frames must keep streaming while paused")
        self.assertEqual(judged_paused, 0, "no judging while paused (SLEEP is cost-free)")
        self.assertGreater(judged_resumed, 0, "judging resumes after resume()")


if __name__ == "__main__":
    unittest.main()
