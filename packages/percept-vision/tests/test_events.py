"""L1: events — temporal dedup, frame window, and the event gate via perceive_events."""

import asyncio
import unittest

from percept.adapters import Goal
from percept.agent import Percept
from percept.backends.fake import FakeVision
from percept.config import EventSettings, Settings
from percept.events import EventLog, FrameWindow
from percept.gate import ASK, FIRE


class TestEventLog(unittest.TestCase):
    def test_consecutive_hits_are_one_event(self):
        log = EventLog(merge_gap_s=1.5)
        for t in (0.0, 1.0, 2.0):  # hit, hit, hit at 1s spacing
            log.observe("rep", t)
        self.assertEqual(log.count("rep"), 1)

    def test_gap_opens_a_new_event(self):
        log = EventLog(merge_gap_s=1.5)
        log.observe("rep", 0.0)  # hit
        # (miss at t=1.0 — nothing observed)
        log.observe("rep", 2.0)  # hit after a >merge_gap gap
        self.assertEqual(log.count("rep"), 2)

    def test_first_is_new_then_continuation(self):
        log = EventLog(merge_gap_s=1.5)
        _, new1 = log.observe("rep", 0.0)
        _, new2 = log.observe("rep", 0.5)
        self.assertTrue(new1)
        self.assertFalse(new2)

    def test_counts_are_per_type(self):
        log = EventLog(merge_gap_s=1.5)
        log.observe("pass", 0.0)
        log.observe("rep", 0.0)
        self.assertEqual(log.count("pass"), 1)
        self.assertEqual(log.count(), 2)


class TestFrameWindow(unittest.TestCase):
    def test_ring_buffer_bounds(self):
        w = FrameWindow(size=3)
        for i in range(5):
            w.push(float(i), f"f{i}")
        self.assertEqual(len(w), 3)
        self.assertEqual(w.latest(), (4.0, "f4"))
        self.assertEqual([t for t, _ in w.frames()], [2.0, 3.0, 4.0])


def _event_agent(merge_gap_s=1.5):
    vision = FakeVision(
        events_script={
            "rep": [("rep", 0.9)],
            "rep-low": [("rep", 0.4)],  # ambiguous event -> ASK
            "still": [],
        }
    )
    settings = Settings(events=EventSettings(window=12, merge_gap_s=merge_gap_s))
    a = Percept.create(vision=vision, settings=settings, discover_plugins=False)
    a.add_goal(
        Goal(
            id="reps",
            condition="a rep happened",
            say="rep!",
            state={"type": "event", "event_type": "rep", "count": 0},
        )
    )
    return a


class TestEventGate(unittest.TestCase):
    def test_new_event_fires_once_continuations_silent(self):
        async def run():
            a = _event_agent()

            async def tick(frame, t):
                a.push_frame(frame, t=t)
                return await a.perceive_events()

            self.assertEqual([e.action for e in await tick("rep", 0.0)], [FIRE])  # rising edge
            self.assertEqual(await tick("rep", 1.0), [])  # continuation -> silent
            self.assertEqual(await tick("rep", 2.0), [])  # still the same rep
            self.assertEqual(a.goals.get("reps").state["count"], 1)
            # a real gap -> a new rep fires again
            self.assertEqual([e.action for e in await tick("rep", 5.0)], [FIRE])
            self.assertEqual(a.goals.get("reps").state["count"], 2)

        asyncio.run(run())

    def test_low_confidence_event_asks(self):
        async def run():
            a = _event_agent()
            a.push_frame("rep-low", t=0.0)
            events = await a.perceive_events()
            self.assertEqual([e.action for e in events], [ASK])

        asyncio.run(run())

    def test_miss_between_hits_counts_two(self):
        async def run():
            a = _event_agent()
            for frame, t in [("rep", 0.0), ("still", 1.0), ("rep", 2.0)]:
                a.push_frame(frame, t=t)
                await a.perceive_events()
            self.assertEqual(a.goals.get("reps").state["count"], 2)

        asyncio.run(run())


if __name__ == "__main__":
    unittest.main()
