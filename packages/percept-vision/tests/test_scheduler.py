"""L1: scheduler — fires once per interval, skips backlog, one-shots, and agent.tick proactive path."""

import asyncio
import unittest

from percept.adapters import Decision, Goal
from percept.agent import Percept
from percept.backends.fake import FakeConductor
from percept.gate import FIRE
from percept.scheduler import Scheduler


class TestScheduler(unittest.TestCase):
    def test_repeater_fires_once_per_interval(self):
        s = Scheduler()
        s.every(20.0, "remind", now=0.0)
        self.assertEqual(s.due(now=10.0), [])  # not yet
        self.assertEqual(len(s.due(now=20.0)), 1)  # first slot
        self.assertEqual(s.due(now=25.0), [])  # mid-interval, quiet
        self.assertEqual(len(s.due(now=40.0)), 1)  # second slot

    def test_repeater_skips_backlog_after_a_stall(self):
        s = Scheduler()
        s.every(10.0, "g", now=0.0)
        fired = s.due(now=35.0)  # missed 10/20/30 while stalled
        self.assertEqual(len(fired), 1)  # fires ONCE, not a burst of 3
        self.assertEqual(s.due(now=39.0), [])  # next slot is 40, still future
        self.assertEqual(len(s.due(now=40.0)), 1)

    def test_one_shot_fires_then_is_gone(self):
        s = Scheduler()
        s.at(15.0, "once")
        self.assertEqual(s.due(now=14.0), [])
        self.assertEqual(len(s.due(now=15.0)), 1)
        self.assertEqual(s.due(now=20.0), [])  # removed after firing
        self.assertEqual(s.active(), [])

    def test_cancel(self):
        s = Scheduler()
        t = s.every(5.0, "g", now=0.0)
        self.assertTrue(s.cancel(t.id))
        self.assertEqual(s.due(now=100.0), [])


class TestAgentTick(unittest.TestCase):
    def test_tick_emits_proactive_fire(self):
        async def run():
            a = Percept.create(
                conductor=FakeConductor(Decision(route="proactive_speak", say="time to stretch")),
                discover_plugins=False,
            )
            a.add_goal(Goal(id="remind", condition="every 20 min", say="stretch"))
            a.schedule_every(20.0, "remind", now=0.0)
            self.assertEqual(await a.tick(t=10.0), [])  # not due
            fired = await a.tick(t=20.0)
            self.assertEqual([e.action for e in fired], [FIRE])
            self.assertEqual(fired[0].text, "time to stretch")
            self.assertEqual(await a.tick(t=21.0), [])  # already fired this interval

        asyncio.run(run())

    def test_tick_silent_when_conductor_ignores(self):
        async def run():
            a = Percept.create(conductor=FakeConductor(), discover_plugins=False)  # default: ignore
            a.add_goal(Goal(id="remind", condition="every 20 min", say="stretch"))
            a.schedule_every(20.0, "remind", now=0.0)
            self.assertEqual(await a.tick(t=20.0), [])  # due, but the model chose not to speak

        asyncio.run(run())


if __name__ == "__main__":
    unittest.main()
