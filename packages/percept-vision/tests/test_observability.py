"""A2 — observability: losing perceptual coverage of a high-consequence concern fires (L1).

"Absence of evidence is not evidence of absence." A concern whose last confident observation has
gone stale has gone blind; for a high-miss-cost concern that loss INHERITS its consequence and fires
("I can't see the baby"). Perception-robust — it fires on the absence of observation, not a verdict.
"""

import unittest

from percept.adapters import Goal
from percept.agent import Percept
from percept.backends.fake import FakeVision
from percept.config import GateSettings, Settings
from percept.gate import FIRE


def _agent(validity_window_s=10.0):
    s = Settings(gate=GateSettings(threshold=0.7, validity_window_s=validity_window_s))
    return Percept.create(vision=FakeVision(script={}), settings=s, discover_plugins=False)


def _severe(a, gid="baby"):
    g = Goal(id=gid, condition="the baby is calm", say="check the baby")
    g.state["miss_cost"] = "severe"  # a high-consequence concern (conductor sets this; A3 wires it)
    a.add_goal(g)
    return g


class TestCoverageLoss(unittest.TestCase):
    def test_fires_once_when_coverage_goes_stale(self):
        a = _agent(validity_window_s=10.0)
        _severe(a)
        a.last_supported["baby"] = 0.0  # last confidently seen at t=0
        self.assertEqual(a.check_observability(t=5.0), [])  # within window -> still covered
        ev = a.check_observability(t=20.0)  # past the window -> coverage lost
        self.assertEqual([e.action for e in ev], [FIRE])
        self.assertIn("lost sight", ev[0].text)
        self.assertEqual(a.check_observability(t=30.0), [])  # still lost -> refractory, no double-fire

    def test_mild_concern_never_fires_coverage_loss(self):
        a = _agent()
        g = Goal(id="laundry", condition="the dryer is done", say="laundry")  # default miss_cost mild
        a.add_goal(g)
        a.last_supported["laundry"] = 0.0
        self.assertEqual(a.check_observability(t=1000.0), [])  # mild -> coverage-loss is silent

    def test_never_observed_is_acquisition_not_blindness(self):
        a = _agent()
        _severe(a)  # last_supported never set
        self.assertEqual(a.check_observability(t=1000.0), [])

    def test_recovery_rearms_then_fires_again(self):
        a = _agent(validity_window_s=10.0)
        _severe(a)
        a.last_supported["baby"] = 0.0
        self.assertEqual([e.action for e in a.check_observability(t=20.0)], [FIRE])  # lost -> fire
        a.last_supported["baby"] = 21.0  # coverage returns
        self.assertEqual(a.check_observability(t=22.0), [])  # covered -> re-arms the coverage edge
        self.assertEqual(
            [e.action for e in a.check_observability(t=40.0)], [FIRE]  # lost again -> fires again
        )


class TestCoverageRecordedOnConfidentRead(unittest.TestCase):
    def test_perceive_judged_records_coverage(self):
        # An integration touch: a confident judged verdict stamps last_supported (the coverage signal).
        import asyncio

        from percept.adapters import Box, Observation, Verdict

        vision = FakeVision(
            script={"f": Verdict(True, 0.95, "calm")},
            detect_script={"f": [Observation("baby", Box(0.4, 0.4, 0.2, 0.2), 0.95)]},
        )
        a = Percept.create(vision=vision, settings=Settings(gate=GateSettings(threshold=0.7)),
                           discover_plugins=False)
        _severe(a)
        asyncio.run(a.perceive_judged("f", t=3.0))
        self.assertEqual(a.last_supported.get("baby"), 3.0)  # confident read = coverage stamped


if __name__ == "__main__":
    unittest.main()
