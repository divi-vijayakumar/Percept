"""A1 absence-native firing — an expected concern overdue past its deadline fires (L1).

"Remind me to add salt while cooking": the concern is "salt has been added" with a deadline. If the
expected event never happens by the deadline, the ABSENCE is the firing. Perception-robust by
construction — it fires on the LACK of a confident-yes, never on a guess.
"""

import unittest

from percept.adapters import Goal, Verdict
from percept.agent import Percept
from percept.backends.fake import FakeVision
from percept.gate import FIRE


def _agent():
    return Percept.create(vision=FakeVision(script={}), discover_plugins=False)


def _deadline_goal(a, gid="salt", deadline=10.0):
    g = Goal(id=gid, condition="salt has been added", say="don't forget the salt")
    g.state["rearm"] = "deadline"
    g.state["deadline_s"] = deadline
    a.add_goal(g)
    return g


class TestOverdue(unittest.TestCase):
    def test_fires_once_when_deadline_passes_unsatisfied(self):
        a = _agent()
        _deadline_goal(a, deadline=10.0)
        self.assertEqual(a.check_deadlines(t=5.0), [])  # before the deadline -> not overdue
        ev = a.check_deadlines(t=11.0)  # past the deadline, never satisfied -> overdue
        self.assertEqual([e.action for e in ev], [FIRE])
        self.assertIn("Still waiting", ev[0].text)
        self.assertEqual(a.check_deadlines(t=12.0), [])  # refractory -> no double-fire

    def test_satisfied_in_time_never_fires(self):
        a = _agent()
        _deadline_goal(a, deadline=10.0)
        a.last_satisfied["salt"] = 4.0  # the expected event happened before the deadline
        self.assertEqual(a.check_deadlines(t=11.0), [])  # not overdue -> silent

    def test_non_deadline_concern_is_ignored(self):
        a = _agent()
        a.add_goal(Goal(id="watch", condition="c", say="s"))  # default watch -> no deadline
        self.assertEqual(a.check_deadlines(t=1000.0), [])  # byte-for-byte: only deadline concerns


class TestSatisfactionStampedFromPerception(unittest.TestCase):
    def test_confident_yes_stamps_satisfaction_and_suppresses_overdue(self):
        import asyncio

        from percept.adapters import Box, Observation

        vision = FakeVision(
            script={"f": Verdict(True, 0.95, "salt added")},
            detect_script={"f": [Observation("pan", Box(0.4, 0.4, 0.2, 0.2), 0.95)]},
        )
        a = Percept.create(vision=vision, discover_plugins=False)
        _deadline_goal(a, deadline=10.0)
        asyncio.run(a.perceive_judged("f", t=4.0))  # salt goes in at t=4 (a confident-yes)
        self.assertEqual(a.last_satisfied.get("salt"), 4.0)  # the satisfaction is stamped
        self.assertEqual(a.check_deadlines(t=11.0), [])  # satisfied in time -> no overdue fires


if __name__ == "__main__":
    unittest.main()
