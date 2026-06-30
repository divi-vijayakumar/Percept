"""Per-frame decision trace — on_trace emits the right stages and is a no-op when unset."""

from __future__ import annotations

import unittest

from percept import Goal, Percept
from percept.adapters import Verdict
from percept.backends.fake import FakeVision
from percept.debug import collect_trace, format_trace
from percept.observability import TraceEvent


def _agent(on_trace=None):
    vision = FakeVision(script={"sip": Verdict(satisfied=True, confidence=0.92, evidence="a cup")})
    agent = Percept.create(vision=vision, discover_plugins=False, on_trace=on_trace)
    agent.add_goal(Goal(id="caffeine", condition="the user is drinking coffee", say="hi"))
    return agent


class TraceEventTests(unittest.IsolatedAsyncioTestCase):
    async def test_emits_decision_stages_with_correlated_id(self):
        events: list[TraceEvent] = []
        agent = _agent(on_trace=collect_trace(events))
        fires = await agent.perceive_judged("sip", source="cam", t=1.0)

        stages = [e.stage for e in events]
        for expected in ("source.frame", "judge.started", "judge.finished", "gate.classified"):
            self.assertIn(expected, stages)
        # every step of this frame shares one trace_id derived from source+timestamp
        self.assertTrue(all(e.trace_id == "cam@1.0" for e in events))

        jf = next(e for e in events if e.stage == "judge.finished")
        self.assertEqual(jf.goal_id, "caffeine")
        self.assertTrue(jf.data["satisfied"])
        self.assertEqual(jf.data["confidence"], 0.92)

        gc = next(e for e in events if e.stage == "gate.classified")
        self.assertEqual(gc.data["truth"], "yes")
        self.assertEqual(gc.data["action"], "fire")  # rising edge -> FIRE
        self.assertEqual([f.action for f in fires], ["fire"])

    async def test_no_tracer_is_a_noop(self):
        # identical firing behavior with no on_trace wired (the default hot path is unchanged)
        agent = _agent(on_trace=None)
        fires = await agent.perceive_judged("sip", source="cam", t=1.0)
        self.assertEqual([f.action for f in fires], ["fire"])

    async def test_raising_tracer_does_not_crash_perception(self):
        def boom(_ev):
            raise RuntimeError("bad tracer")
        agent = _agent(on_trace=boom)
        fires = await agent.perceive_judged("sip", source="cam", t=1.0)  # must not propagate
        self.assertEqual([f.action for f in fires], ["fire"])

    async def test_async_tracer_is_awaited(self):
        seen: list[str] = []
        async def atrace(ev):
            seen.append(ev.stage)
        agent = _agent(on_trace=atrace)
        await agent.perceive_judged("sip", source="cam", t=1.0)
        self.assertIn("gate.classified", seen)  # the coroutine was awaited, not dropped

    async def test_run_restores_core_on_trace(self):
        from percept.runtime.worker import run
        sentinel = lambda _ev: None  # noqa: E731
        agent = _agent(on_trace=sentinel)
        async for _ in run(agent, source=[], on_trace=lambda _ev: None):  # empty source
            pass
        self.assertIs(agent.on_trace, sentinel)  # this run's tracer was restored, not leaked

    def test_format_trace_is_readable(self):
        ev = TraceEvent(trace_id="cam@1.0", stage="gate.classified", goal_id="caffeine",
                        data={"truth": "yes", "action": "fire"})
        self.assertEqual(
            format_trace(ev),
            "[trace=cam@1.0 goal=caffeine] gate.classified truth=yes action=fire",
        )


if __name__ == "__main__":
    unittest.main()
