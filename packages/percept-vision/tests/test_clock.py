"""Clock primitive (L1, v3 §9) — the injectable clock drives run-time cognition deterministically.

Proves the replay/determinism premise without wall-clock waits: inject a FakeClock, advance it, and a
deadline/absence fire happens at exactly the modelled time. This is the seam the timeline-replay
harness drives."""
import unittest

from percept.adapters import Goal
from percept.agent import Percept
from percept.backends.fake import FakeVision
from percept.clock import Clock, FakeClock, WallClock
from percept.gate import FIRE


class FakeClockTests(unittest.TestCase):
    def test_is_a_drop_in_clock_callable(self):
        fake = FakeClock(2.0)
        self.assertEqual(fake(), 2.0)               # the Clock contract: seconds via __call__
        self.assertEqual(fake.advance(500), 2.5)    # advance is in milliseconds
        fake.set(10.0)
        self.assertEqual(fake(), 10.0)

    def test_drives_a_deadline_fire_with_no_explicit_t(self):
        # the whole point: cognition reads time through the injected clock (no t= passed), so advancing the
        # FakeClock past a deadline fires the absence deterministically.
        fake = FakeClock()
        a = Percept.create(vision=FakeVision(script={}), discover_plugins=False, clock=fake)
        g = Goal(id="salt", condition="salt has been added", say="don't forget the salt")
        g.state["rearm"] = "deadline"
        g.state["deadline_after_s"] = 10.0  # duration; add_goal resolves deadline_at = clock()+10 = 10
        a.add_goal(g)

        fake.advance(5_000)                          # t=5s — before the deadline
        self.assertEqual(a.check_deadlines(), [])
        fake.advance(6_000)                          # t=11s — past the 10s deadline, never satisfied
        self.assertEqual([e.action for e in a.check_deadlines()], [FIRE])
        fake.advance(1_000)                          # t=12s — refractory, no double-fire
        self.assertEqual(a.check_deadlines(), [])

    def test_wallclock_is_monotonic_seconds(self):
        self.assertIsInstance(WallClock(), float)
        self.assertGreaterEqual(WallClock(), WallClock() - 1.0)

    def test_clock_alias_is_the_callable_contract(self):
        c: Clock = FakeClock(1.0)
        self.assertEqual(c(), 1.0)


if __name__ == "__main__":
    unittest.main()
