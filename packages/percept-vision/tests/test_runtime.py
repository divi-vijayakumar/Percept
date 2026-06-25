"""Runtime worker (L1) — run(core, source) -> incidents, offline via an iterable source + FakeVision.

The source yields (t, gate_frame, judge_frame): a gray Frame drives the motion gate; a token string is
what FakeVision judges. No cv2, no camera — the iterable-source path makes the whole worker testable.
"""
from __future__ import annotations

import asyncio
import unittest

from percept import Goal, Percept
from percept.adapters import Verdict
from percept.backends.fake import FakeVision
from percept.runtime import run
from percept.edge.tier0 import gray_frame

W, H = 8, 8
STILL = gray_frame(W, H, 0)
MOVED = gray_frame(W, H, 255)

# (t, gate_frame, judge_frame): motion alternates to clear the gate; tokens drive FakeVision.
SOURCE = [
    (0.0, STILL, "clear"),     # prime — no motion
    (1.0, MOVED, "intruder"),  # still->moved: escalate -> "intruder" -> FIRE (rising edge) -> incident
    (2.0, STILL, "clear"),     # moved->still: escalate -> "clear" -> silent (re-arms the edge)
    (3.0, MOVED, "intruder"),  # still->moved: escalate -> "intruder" -> FIRE again -> incident
]


def _core() -> Percept:
    vision = FakeVision(
        script={
            "intruder": Verdict(satisfied=True, confidence=0.95, evidence="a person is in the zone"),
            "clear": Verdict(satisfied=False, confidence=0.95, evidence="the zone is empty"),
        }
    )
    core = Percept.create(vision=vision, discover_plugins=False)
    core.add_goal(Goal(id="zone", condition="a person is in the restricted zone", say="Alert: person in zone"))
    return core


async def _collect(core, source, **kw):
    return [inc async for inc in run(core, source=source, **kw)]


class RuntimeWorkerTests(unittest.TestCase):
    def test_emits_an_incident_per_fire(self):
        incidents = asyncio.run(_collect(_core(), SOURCE))
        self.assertEqual(len(incidents), 2)  # two rising edges
        self.assertTrue(all(i.concern_id == "zone" for i in incidents))
        self.assertTrue(all(i.message == "Alert: person in zone" for i in incidents))
        self.assertTrue(all(i.verdict.satisfied for i in incidents))
        self.assertTrue(all(i.snippet is not None and i.snippet.kind == "frame" for i in incidents))

    def test_on_incident_sink_is_called(self):
        seen = []
        asyncio.run(_collect(_core(), SOURCE, on_incident=seen.append))
        self.assertEqual(len(seen), 2)

    def test_on_frame_feed_hook_gets_every_frame(self):
        seen = []
        asyncio.run(_collect(_core(), SOURCE, on_frame=lambda t, f: seen.append(t)))
        self.assertEqual(seen, [0.0, 1.0, 2.0, 3.0])  # the live-feed hook sees every frame, gated or not

    def test_still_stream_emits_nothing(self):
        still = [(float(t), STILL, "clear") for t in range(5)]  # no motion -> no escalation -> no cloud call
        self.assertEqual(asyncio.run(_collect(_core(), still)), [])


if __name__ == "__main__":
    unittest.main()
