"""PR 2.1 — RuntimeSession: the input seam that fixes "what do you see" + arm-a-goal-at-runtime.

- ask(): a spoken vision question routes through the conductor (on_speech_endpoint) WITH the watch's
  latest frame. The v2v demo bug was piping audio to a voice model with NO frame -> "I can't see".
- arm(): a goal armed mid-run bumps the registry revision so run(regate_on_goal_change=True) re-arms
  the gate (broken today — the WatchSpec is compiled once at start).
- interrupt(): cancels an in-flight ask (the barge-in boundary).
"""
from __future__ import annotations

import asyncio
import unittest

import percept
from percept.adapters import Goal, Verdict
from percept.backends.fake import FakeVision
from percept.edge.tier0 import gray_frame
from percept.runtime import RuntimeStats, run
from percept.runtime.session import start

W, H = 8, 8
STILL = gray_frame(W, H, 0)
MOVED = gray_frame(W, H, 255)


def _core():
    vision = FakeVision(script={
        "seen": Verdict(satisfied=True, confidence=0.95, evidence="a person"),
        "x": Verdict(satisfied=False, confidence=0.9, evidence="nothing"),
    })
    core = percept.Percept.create(vision=vision, discover_plugins=False)
    core.add_goal(Goal(id="zone", condition="a person is here", say="alert"))
    return core


class AskCarriesFrameTests(unittest.TestCase):
    def test_ask_routes_through_conductor_with_latest_frame(self):
        async def go():
            core = _core()
            captured = {}

            async def spy(utterance, *, frame=None, t=None):
                captured["utterance"] = utterance
                captured["frame"] = frame
                return "ok"

            core.on_speech_endpoint = spy  # spy on the conductor entrypoint
            # a source that keeps yielding so the session stays live and stashes a frame
            src = [(float(i), STILL, "x") for i in range(3)]
            session = await start(core, source=src)
            await asyncio.sleep(0)  # let a frame get stashed
            await session.ask("what do you see?")
            await session.stop()
            return captured

        captured = asyncio.run(go())
        self.assertEqual(captured["utterance"], "what do you see?")
        # THE fix: the spoken vision question carried the watch's latest frame (not None)
        self.assertIsNotNone(captured["frame"])
        self.assertEqual(captured["frame"], "x")  # the judge_frame the source yielded, stashed by on_frame


class ArmAtRuntimeTests(unittest.TestCase):
    def test_goal_armed_midrun_rearms_the_gate(self):
        # An async source that arms a NEW goal after the first frame. Without regate the gate would
        # never see it (spec compiled once); with regate the worker recompiles and the new goal judges.
        async def go():
            core = _core()

            async def source():
                yield (0.0, STILL, "x")
                # arm a second goal mid-run (as a voice command would); revision bumps
                core.add_goal(Goal(id="late", condition="a late thing", say="late!", state={}))
                for i in range(1, 6):
                    frame = MOVED if i % 2 else STILL  # motion so the gate escalates -> judges "seen"
                    yield (float(i), frame, "seen")

            stats = RuntimeStats()
            incidents = [inc async for inc in run(core, source=source(), stats=stats,
                                                  regate_on_goal_change=True)]
            return incidents

        incidents = asyncio.run(go())
        # the late-armed goal produced at least one incident (it was judged after the regate)
        self.assertTrue(any(i.concern_id == "late" for i in incidents),
                        f"late goal never fired; concerns={[i.concern_id for i in incidents]}")


class InterruptTests(unittest.TestCase):
    def test_interrupt_cancels_inflight_ask(self):
        async def go():
            core = _core()

            async def slow(utterance, *, frame=None, t=None):
                await asyncio.sleep(10)  # a long "spoken reply" we will barge in on
                return "done"

            core.on_speech_endpoint = slow
            session = await start(core, source=[(0.0, STILL, "x")])
            task = asyncio.ensure_future(session.ask("tell me a long story"))
            await asyncio.sleep(0.05)
            barged = session.interrupt()
            cancelled = False
            try:
                await task
            except asyncio.CancelledError:
                cancelled = True
            await session.stop()
            return barged, cancelled

        barged, cancelled = asyncio.run(go())
        self.assertTrue(barged)
        self.assertTrue(cancelled)


if __name__ == "__main__":
    unittest.main()
