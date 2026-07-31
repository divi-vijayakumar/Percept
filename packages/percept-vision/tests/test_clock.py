"""Clock primitive (L1, v3 §9) — the injectable clock drives run-time cognition deterministically.

Proves the replay/determinism premise without wall-clock waits: inject a FakeClock, advance it, and a
deadline/absence fire happens at exactly the modelled time. This is the seam the timeline-replay
harness drives."""
import unittest

from percept.adapters import Goal
from percept.agent import Percept
from percept.backends.fake import FakeVision
from percept._numeric import clamp01, finite
from percept.clock import Clock, FakeClock, StreamTimeGuard, WallClock
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


class StreamTimeGuardTests(unittest.TestCase):
    """The OTHER timeline — the one the caller supplies, which promises nothing.

    The clock above is ours and monotonic by construction. ``t`` comes from a file, a decoder, a
    socket or a host clock, and both of its failure modes silently break any ``t - last < window``
    operator: backwards mutes it forever, NaN bypasses it entirely and then poisons the state.
    """

    def test_the_first_sample_is_forward_at_any_value(self):
        guard = StreamTimeGuard()
        self.assertIsNone(guard.last)
        self.assertEqual(guard.check("ts", 1.0e9), "ok")
        self.assertEqual(guard.last, 1.0e9)
        self.assertEqual(guard.rewinds, 0)

    def test_it_separates_a_repeat_from_a_step_backwards(self):
        # Two readings at one instant are normal — a visual frame and an external wake share a ts —
        # and must not be confused with a timeline that actually went backwards.
        guard = StreamTimeGuard()
        self.assertEqual(
            [guard.check("ts", t) for t in (0.0, 1.0, 1.0, 0.5, 0.6)],
            ["ok", "ok", "duplicate", "rewind", "ok"],
        )
        self.assertEqual(guard.rewinds, 1)

    def test_a_rewind_adopts_the_new_timeline_rather_than_reporting_every_frame_after_it(self):
        # A restarted source is a NEW timeline, not one long anomaly. Measuring later frames against
        # the abandoned baseline would report every one of them as a rewind.
        guard = StreamTimeGuard()
        guard.check("ts", 100.0)
        self.assertEqual([guard.check("ts", t) for t in (0.0, 1.0, 2.0)], ["rewind", "ok", "ok"])
        self.assertEqual(guard.rewinds, 1)

    def test_a_non_finite_or_non_numeric_timestamp_is_refused_by_name(self):
        guard = StreamTimeGuard()
        for bad in (float("nan"), float("inf"), float("-inf"), None, "1.0", True, [1.0]):
            with self.subTest(t=repr(bad)):
                with self.assertRaises(ValueError) as caught:
                    guard.check("frame_ts", bad)
                self.assertIn("frame_ts", str(caught.exception))
        self.assertIsNone(guard.last)  # ...and a refused value never becomes the baseline

    def test_reset_forgets_the_baseline_but_keeps_the_tally(self):
        guard = StreamTimeGuard()
        guard.check("ts", 10.0)
        guard.check("ts", 5.0)
        guard.reset()
        self.assertIsNone(guard.last)
        self.assertEqual(guard.check("ts", 0.0), "ok")  # a declared restart is not a rewind
        self.assertEqual(guard.rewinds, 1)              # the run-level diagnostic survives


class NumericFirewallTests(unittest.TestCase):
    """One admission rule for an untrusted scalar, in one place — and it matches the JS edge."""

    def test_finite_refuses_what_float_would_have_saturated(self):
        self.assertEqual(finite("x", 3), 3.0)
        self.assertEqual(finite("x", -0.0), 0.0)  # normalized, so -0.0 never rides a wire
        for bad in (True, False, float("nan"), float("inf"), "1.0", None, [1]):
            with self.subTest(value=repr(bad)):
                with self.assertRaises(ValueError) as caught:
                    finite("bias", bad)
                self.assertIn("bias", str(caught.exception))

    def test_clamp01_matches_the_javascript_edges_clamp01(self):
        # packages/percept-edge/src/skills.js: non-number or non-finite -> 0, then pin to [0, 1].
        # The NaN row is the one the two runtimes used to disagree on.
        for value, expected in (
            (float("nan"), 0.0), (float("inf"), 0.0), (float("-inf"), 0.0),
            (None, 0.0), ("nonsense", 0.0), (True, 0.0), (False, 0.0),
            (-1.0, 0.0), (0.0, 0.0), (0.25, 0.25), (1.0, 1.0), (2.5, 1.0),
        ):
            with self.subTest(value=repr(value)):
                self.assertEqual(clamp01(value), expected)


if __name__ == "__main__":
    unittest.main()
