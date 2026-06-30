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

    def test_format_trace_is_readable(self):
        ev = TraceEvent(trace_id="cam@1.0", stage="gate.classified", goal_id="caffeine",
                        data={"truth": "yes", "action": "fire"})
        self.assertEqual(
            format_trace(ev),
            "[trace=cam@1.0 goal=caffeine] gate.classified truth=yes action=fire",
        )


if __name__ == "__main__":
    unittest.main()
