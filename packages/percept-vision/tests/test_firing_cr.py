"""Unified (C, R) firing: per-concern re-arm via key synthesis and the never-contract (L1).

Drives ``Percept._fire_concern`` directly against the REAL gate (no perception machinery), proving:
  * on_false (default)  reproduces today — a confident-false re-arms, then a later YES re-fires;
  * never (one-shot)    fires exactly once and never re-arms, even across false->true->true (a
                        confident-false must not reach the gate);
  * interval            re-fires once per cadence time-bucket (key-synthesis folds floor(t/cadence)).
The full existing suite is the byte-for-byte tripwire for the watch/event re-route through the helper.
"""

import unittest

from percept.adapters import Goal, Verdict
from percept.agent import Percept
from percept.backends.fake import FakeVision
from percept.gate import ASK, FIRE

YES = Verdict(True, 0.95, "yes")
NO = Verdict(False, 0.95, "no")
UNSURE = Verdict(True, 0.4, "unsure")


def _agent():
    return Percept.create(vision=FakeVision(script={}), discover_plugins=False)


def _fire(a, goal, verdicts, base_key=("g", "e")):
    """Drive _fire_concern over a verdict sequence -> list of actions ('fire'/'ask'/None) at t=i."""
    out = []
    for i, v in enumerate(verdicts):
        ev = a._fire_concern(goal, v, base_key, freshness=1.0, t=float(i))
        out.append(ev.action if ev is not None else None)
    return out


class TestOnFalseDefault(unittest.TestCase):
    def test_rearm_on_confident_false(self):
        a = _agent()
        g = Goal(id="g", condition="c", say="s")  # default state -> rearm on_false
        # YES fires once; repeat YES is refractory; a confident NO re-arms; YES fires again.
        self.assertEqual(_fire(a, g, [YES, YES, NO, YES]), [FIRE, None, None, FIRE])

    def test_unsure_asks_without_touching_the_edge(self):
        a = _agent()
        g = Goal(id="g", condition="c", say="s")
        self.assertEqual(_fire(a, g, [UNSURE, YES]), [ASK, FIRE])

    def test_fire_event_carries_say_and_entity(self):
        a = _agent()
        g = Goal(id="g", condition="c", say="stay off caffeine")
        ev = a._fire_concern(g, YES, ("g", "cup2"), freshness=1.0, t=0.0)
        self.assertEqual((ev.action, ev.text, ev.entity_id), (FIRE, "stay off caffeine", "cup2"))


class TestNeverOneShot(unittest.TestCase):
    def test_fires_once_and_never_rearms(self):
        a = _agent()
        g = Goal(id="g", condition="c", say="s")
        g.state["rearm"] = "never"
        # Even with confident-false between, a one-shot never re-arms -> only the first FIRE.
        self.assertEqual(
            _fire(a, g, [YES, YES, NO, YES, NO, YES]),
            [FIRE, None, None, None, None, None],
        )

    def test_contract_vs_on_false(self):
        a = _agent()
        gn = Goal(id="gn", condition="c", say="s")
        gn.state["rearm"] = "never"
        self.assertEqual(_fire(a, gn, [YES, NO, YES], base_key=("gn", "e")), [FIRE, None, None])
        # contrast: the default on_false WOULD re-arm on the NO and re-fire
        go = Goal(id="go", condition="c", say="s")
        self.assertEqual(_fire(a, go, [YES, NO, YES], base_key=("go", "e")), [FIRE, None, FIRE])


class TestIntervalRearm(unittest.TestCase):
    def test_refire_once_per_time_bucket(self):
        a = _agent()
        g = Goal(id="g", condition="c", say="s")
        g.state["rearm"] = "interval"
        g.state["cadence_s"] = 10.0
        out = _fire(a, g, [YES] * 21)  # t = 0..20, cadence 10 -> buckets 0,1,2
        fire_times = [i for i, action in enumerate(out) if action == FIRE]
        self.assertEqual(fire_times, [0, 10, 20])  # one fire per bucket; silent within a bucket


if __name__ == "__main__":
    unittest.main()
