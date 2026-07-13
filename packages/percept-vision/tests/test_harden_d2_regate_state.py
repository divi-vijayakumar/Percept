"""AGENT-D / D2 — a regate must not throw away per-goal throttle or detector-bank state.

``_build_gate()`` constructs a FRESH ``MotionGate`` (empty ``_last_escalate``, no ``prev``) and FRESH
``DetectorBank``s on every regate. So arming ONE goal by voice:
  * released the per-goal cadence brake on ALL other goals (they could re-escalate before ``cadence_s``);
  * reset every surviving goal's detector state (a rep in flight is dropped or double-counted).

These tests pin ``MotionGate.adopt_state`` (throttle + prev carried for SURVIVING ids only) and the
worker's bank carry-over across a mid-run regate.
"""
from __future__ import annotations

import asyncio
import unittest

from percept import Goal, Percept
from percept.adapters import Verdict
from percept.backends.fake import FakeVision
from percept.config import load_settings
from percept.edge.tier0 import MotionGate, compile_watchspec, gray_frame
from percept.runtime import RuntimeStats, run

W, H = 8, 8
STILL = gray_frame(W, H, 0)
MOVED = gray_frame(W, H, 255)


def _cadence_goal(i: int, interval: float = 5.0, gate_on: str = "motion") -> Goal:
    return Goal(id=f"g{i}", condition=f"c{i}", say="!",
                state={"cadence": {"min_judge_interval": interval, "gate_on": gate_on}})


class D2AdoptStateTests(unittest.TestCase):
    """MotionGate.adopt_state: carry throttle + prev for surviving goal_ids; forget cancelled ones."""

    def _gate(self, goals):
        return MotionGate(compile_watchspec(goals, threshold=0.1), settings=load_settings())

    def test_regate_preserves_throttle_for_surviving_goals_only(self):
        g1 = self._gate([_cadence_goal(1), _cadence_goal(2)])
        g1.evaluate(STILL, ts=0.0)                 # prime prev
        d = g1.evaluate(MOVED, ts=1.0)             # motion -> g1,g2 escalate, throttled until t=6
        self.assertTrue(d.escalate)
        self.assertEqual(g1._last_escalate, {"g1": 1.0, "g2": 1.0})  # noqa: SLF001

        # Arm a THIRD goal (regate). The new gate must adopt the old per-goal throttle + prev frame.
        g2 = self._gate([_cadence_goal(1), _cadence_goal(2), _cadence_goal(3)])
        g2.adopt_state(g1)
        self.assertEqual(g2._last_escalate, {"g1": 1.0, "g2": 1.0})  # noqa: SLF001 - throttle carried
        self.assertIsNotNone(g2._prev)                               # noqa: SLF001 - prev frame carried

        # 1s later (< 5s cadence): g1 & g2 must stay throttled; only the fresh g3 escalates. Feed STILL
        # so there IS motion vs the carried prev (MOVED) — proving the prev frame was adopted too.
        d2 = g2.evaluate(STILL, ts=2.0)
        self.assertTrue(d2.escalate)                      # g3 fired
        self.assertEqual(g2._last_escalate["g1"], 1.0)    # noqa: SLF001 - NOT re-escalated
        self.assertEqual(g2._last_escalate["g2"], 1.0)    # noqa: SLF001
        self.assertEqual(g2._last_escalate["g3"], 2.0)    # noqa: SLF001 - fresh escalate

    def test_cancelled_goal_throttle_is_forgotten(self):
        g1 = self._gate([_cadence_goal(1), _cadence_goal(2)])
        g1.evaluate(STILL, ts=0.0)
        g1.evaluate(MOVED, ts=1.0)
        g2 = self._gate([_cadence_goal(1)])   # g2 cancelled
        g2.adopt_state(g1)
        self.assertEqual(g2._last_escalate, {"g1": 1.0})  # noqa: SLF001 - g2's brake correctly dropped


class D2WorkerThrottleTests(unittest.TestCase):
    """Worker: arming a THIRD goal mid-run must not release goals 1&2's cadence brake."""

    def test_arming_a_third_goal_does_not_release_the_others_brake(self):
        # A single fixed-cadence goal (escalates every 5s, motion-independent). STILL frames at 1fps.
        # Its legit escalations are t=0, 5, 10. Arming a plain goal g3 at t=2 must NOT make g1 escalate
        # at t=2 (only 2s since its last fire) — the "regate released the brake" bug.
        vision = FakeVision(script={"x": Verdict(satisfied=False, confidence=0.9, evidence="n/a")})
        core = Percept.create(vision=vision, discover_plugins=False)
        core.add_goal(_cadence_goal(1, interval=5.0, gate_on="fixed"))

        async def source():
            for i in range(11):
                if i == 2:  # arm a third (plain) goal mid-run -> regate
                    core.add_goal(Goal(id="g3", condition="something else", say="!"))
                yield (float(i), STILL, "x")

        esc_times: list[float] = []

        async def on_health(ev):
            if ev.kind == "escalation":
                esc_times.append(ev.ts)

        asyncio.run(_drain(core, source(), on_health=on_health))
        self.assertNotIn(2.0, esc_times,
                         f"g1 must stay throttled across the regate; escalated at {esc_times}")
        self.assertEqual(esc_times, [0.0, 5.0, 10.0],
                         f"fixed-cadence schedule must survive the regate untouched; got {esc_times}")


class D2WorkerRepCountTests(unittest.TestCase):
    """Worker: a regate mid-rep must not drop or duplicate a rep (detector-bank carry-over)."""

    _PATTERN = [STILL, STILL, MOVED]

    def _osc(self, n=24):
        return [(float(i), self._PATTERN[i % 3], "x") for i in range(n)]

    def _core(self):
        vision = FakeVision(script={"x": Verdict(satisfied=False, confidence=0.9, evidence="n/a")})
        core = Percept.create(vision=vision, discover_plugins=False)
        core.add_goal(Goal(id="reps", condition="a rep happened", say="rep",
                           state={"detectors": ["motion-periodicity"]}))
        return core

    def test_regate_midrep_does_not_change_the_count(self):
        # Baseline: no mid-run regate.
        base_core = self._core()
        base_stats = RuntimeStats()
        asyncio.run(_drain(base_core, self._osc(), stats=base_stats))

        # With a mid-run regate: arm a plain goal partway (at frame 10, mid-oscillation).
        core = self._core()
        stats = RuntimeStats()

        async def source():
            for i, item in enumerate(self._osc()):
                if i == 10:
                    core.add_goal(Goal(id="other", condition="unrelated", say="!"))
                yield item

        asyncio.run(_drain(core, source(), stats=stats))
        self.assertGreater(base_stats.boundaries, 0)  # sanity: the baseline actually counts reps
        self.assertEqual(stats.boundaries, base_stats.boundaries,
                         "a regate mid-rep must be transparent to the rep count (no drop / double-count)")


async def _drain(core, source, **kw):
    return [inc async for inc in run(core, source=source, regate_on_goal_change=True, **kw)]


if __name__ == "__main__":
    unittest.main()
