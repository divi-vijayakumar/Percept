"""The novel task — "alert me the 3rd distinct time" — fires end-to-end with ZERO new agent code.

It is a plain watch goal + an authored ``fire_when``: each rising edge (off->on) is a distinct
occurrence (the gate already gives one fire per rising edge), and the executor's accumulate gates the
user-facing alert on ``fire_when(n)``. So it routes through the UNCHANGED dispatch (watch -> _fire_concern
-> executor.fire_judged). This is the payoff of the generalization: a task no code path was written for.
"""

import asyncio
import unittest

from percept.adapters import Box, Goal, Observation, Verdict
from percept.agent import Percept
from percept.backends.fake import FakeVision
from percept.behavior import BehaviorExecutor, program_of
from percept.config import GateSettings, Settings
from percept.gate import FIRE, Gate

YES = Verdict(True, 0.95, "yes")
NO = Verdict(False, 0.95, "no")
BOX = Box(0.4, 0.4, 0.2, 0.2)  # one stable entity across frames -> one gate key


class TestAccumulateUnit(unittest.TestCase):
    def test_fires_only_on_nth_rising_edge(self):
        ex = BehaviorExecutor(Gate())
        g = Goal(id="g", condition="c", say="3rd time")
        g.state["fire_when"] = "n == 3"
        out = []
        for v in [YES, NO, YES, NO, YES]:  # 3 rising edges
            f = ex.fire_judged(g, v, ("g", "e"), program_of(g))
            out.append(f.action if f is not None else None)
        self.assertEqual(out, [None, None, None, None, FIRE])  # alert only on the 3rd

    def test_no_fire_when_is_byte_for_byte(self):
        ex = BehaviorExecutor(Gate())
        g = Goal(id="g", condition="c", say="s")  # no fire_when
        out = []
        for v in [YES, NO, YES]:
            f = ex.fire_judged(g, v, ("g", "e"), program_of(g))
            out.append(f.action if f is not None else None)
        self.assertEqual(out, [FIRE, None, FIRE])  # every rising edge fires (unchanged)


class TestNthDistinctEndToEnd(unittest.TestCase):
    def _build(self):
        vision = FakeVision(
            script={"on": Verdict(True, 0.95, "on couch"), "off": Verdict(False, 0.95, "off couch")},
            detect_script={"on": [Observation("dog", BOX, 0.95)], "off": [Observation("dog", BOX, 0.95)]},
        )
        a = Percept.create(
            vision=vision, settings=Settings(gate=GateSettings(threshold=0.7)), discover_plugins=False
        )
        g = Goal(id="g-dog", condition="the dog is on the couch", say="that's the 3rd time!")
        g.state["fire_when"] = "n == 3"  # what the conductor authors from "alert me the 3rd distinct time"
        a.add_goal(g)
        return a

    def test_third_rising_edge_alerts_through_unchanged_dispatch(self):
        async def run():
            a = self._build()
            out = []
            for i, frame in enumerate(["on", "off", "on", "off", "on"]):  # rising edges at 0,2,4
                evs = await a.perceive_judged(frame, t=float(i))
                out.append([e.action for e in evs])
            return out

        out = asyncio.run(run())
        self.assertEqual(out, [[], [], [], [], [FIRE]])  # only the 3rd distinct time alerts
        # the firing event carries the user's say
        a = self._build()
        last = asyncio.run(self._final_fire(a))
        self.assertEqual(last.text, "that's the 3rd time!")

    async def _final_fire(self, a):
        ev = None
        for i, frame in enumerate(["on", "off", "on", "off", "on"]):
            evs = await a.perceive_judged(frame, t=float(i))
            if evs:
                ev = evs[0]
        return ev


if __name__ == "__main__":
    unittest.main()
