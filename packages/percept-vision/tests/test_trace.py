"""Per-frame decision trace — on_trace emits the right stages and is a no-op when unset."""

from __future__ import annotations

import unittest

from percept import Goal, Percept
from percept.adapters import Verdict
from percept.backends.fake import FakeVision
from percept.debug import collect_trace, format_trace
from percept.edge.tier0 import gray_frame
from percept.observability import TraceEvent
from percept.runtime import run


def _agent(on_trace=None):
    vision = FakeVision(script={"sip": Verdict(satisfied=True, confidence=0.92, evidence="a cup")})
    agent = Percept.create(vision=vision, discover_plugins=False, on_trace=on_trace)
    agent.add_goal(Goal(id="caffeine", condition="the user is drinking coffee", say="hi"))
    return agent


# O1 fixture (mirrors tests/test_runtime.py): (t, gate_frame, judge_frame) — a gray Frame drives the
# motion gate, a token string is what FakeVision judges. Still->moved escalates and FIREs on the
# rising edge; moved->still escalates and stays SILENT (confident-false re-arms).
_W, _H = 8, 8
_STILL = gray_frame(_W, _H, 0)
_MOVED = gray_frame(_W, _H, 255)
_RUN_SOURCE = [
    (0.0, _STILL, "clear"),     # prime — no motion, edge never escalates
    (1.0, _MOVED, "intruder"),  # still->moved: escalate -> FIRE (rising edge)
    (2.0, _STILL, "clear"),     # moved->still: escalate -> SILENT (confident-false re-arms)
]


def _zone_core() -> Percept:
    vision = FakeVision(
        script={
            "intruder": Verdict(satisfied=True, confidence=0.95, evidence="a person is in the zone"),
            "clear": Verdict(satisfied=False, confidence=0.95, evidence="the zone is empty"),
        }
    )
    core = Percept.create(vision=vision, discover_plugins=False)
    core.add_goal(Goal(id="zone", condition="a person is in the restricted zone", say="Alert"))
    return core


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

    async def test_run_restores_core_on_trace_on_error(self):
        # the restore must hold even when the run body raises mid-stream (the finally path), not only on
        # a clean/empty source — otherwise this run's tracer leaks into the next run(core).
        from percept.runtime.worker import run
        sentinel = lambda _ev: None  # noqa: E731
        agent = _agent(on_trace=sentinel)

        def boom_source():
            raise RuntimeError("capture blew up")
            yield  # pragma: no cover - makes this a generator

        with self.assertRaises(RuntimeError):
            async for _ in run(agent, source=boom_source(), on_trace=lambda _ev: None):
                pass
        self.assertIs(agent.on_trace, sentinel)  # restored despite the error

    # -- O1: production trace telemetry on the real runtime firing path (edge + judge + gate + reason) --

    async def test_run_trace_captures_edge_decision_every_frame(self):
        """The tier-0 ``edge.decision`` stage is emitted for EVERY frame the worker sees — escalated or
        not — so a silent run can be explained even when the edge never spent a cloud call."""
        events: list[TraceEvent] = []
        core = _zone_core()
        async for _ in run(core, source=_RUN_SOURCE, on_trace=collect_trace(events)):
            pass

        edge_events = [e for e in events if e.stage == "edge.decision"]
        self.assertEqual(len(edge_events), 3)  # one per source frame, gated or not
        self.assertEqual([e.t for e in edge_events], [0.0, 1.0, 2.0])
        for e in edge_events:
            for key in ("motion", "structural", "semantic", "threshold", "escalate"):
                self.assertIn(key, e.data)
        # t=0 is the priming frame — no prior frame to diff against -> no motion -> no escalation.
        self.assertFalse(edge_events[0].data["escalate"])
        # t=1 (still->moved) and t=2 (moved->still) both clear the motion gate.
        self.assertTrue(edge_events[1].data["escalate"])
        self.assertTrue(edge_events[2].data["escalate"])
        # edge.decision correlates with the same-frame judge/gate stages via trace_id (source@t).
        self.assertTrue(all(e.trace_id == f"camera@{e.t}" for e in edge_events))

    async def test_run_trace_captures_a_fire_and_a_silent_with_reason(self):
        """End-to-end through ``run()``: the trace captures the edge decision, the judge verdict, the
        gate action AND a human-readable reason for both a FIRE (rising edge) and a SILENT
        (confident-false) tick — the H0 forensic trace shape, on the production runtime path."""
        events: list[TraceEvent] = []
        core = _zone_core()
        incidents = [inc async for inc in run(core, source=_RUN_SOURCE, on_trace=collect_trace(events))]
        self.assertEqual(len(incidents), 1)  # one rising-edge fire (t=1.0)

        gate_events = {e.t: e for e in events if e.stage == "gate.classified"}
        self.assertEqual(set(gate_events), {1.0, 2.0})  # only escalated frames reach the gate

        fire = gate_events[1.0]
        self.assertEqual(fire.goal_id, "zone")
        self.assertEqual(fire.data["truth"], "yes")
        self.assertEqual(fire.data["action"], "fire")
        self.assertEqual(fire.data["reason"], "fire: rising_edge")

        silent = gate_events[2.0]
        self.assertEqual(silent.goal_id, "zone")
        self.assertEqual(silent.data["truth"], "no")
        self.assertEqual(silent.data["action"], "silent")
        self.assertEqual(silent.data["reason"], "silent: verdict_not_satisfied")

        # the matching judge.finished verdicts are also on the trace (edge/verdict/gate/reason, all present)
        jf = {e.t: e for e in events if e.stage == "judge.finished"}
        self.assertTrue(jf[1.0].data["satisfied"])
        self.assertFalse(jf[2.0].data["satisfied"])

    def test_format_trace_is_readable(self):
        ev = TraceEvent(trace_id="cam@1.0", stage="gate.classified", goal_id="caffeine",
                        data={"truth": "yes", "action": "fire"})
        self.assertEqual(
            format_trace(ev),
            "[trace=cam@1.0 goal=caffeine] gate.classified truth=yes action=fire",
        )


if __name__ == "__main__":
    unittest.main()
