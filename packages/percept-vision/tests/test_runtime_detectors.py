"""PR 1.3 — edge detectors run in the product runtime, not just the bench loop.

Regression guard for a stranded capability: the DetectorBank (motion-periodicity, acoustic-onset,
pose) ran only inside GatedWatchLoop/RemoteEdgeReceiver, so counting/rhythm/onset were absent from
`percept.runtime.run`. These tests assert `run()` now builds banks from goals that selected
detectors and feeds their boundaries to `perceive_boundary` (edge proposes · brain counts) — while a
goal with NO detectors builds no banks (byte-for-byte the pre-PR path).
"""
from __future__ import annotations

import asyncio
import unittest

from percept import Goal, Percept
from percept.adapters import Verdict
from percept.backends.fake import FakeVision
from percept.edge.detectors import Boundary, register_detector, unregister
from percept.edge.tier0 import gray_frame
from percept.runtime import RuntimeStats, run

W, H = 8, 8
STILL = gray_frame(W, H, 0)
MOVED = gray_frame(W, H, 255)

# Period-3 (STILL, STILL, MOVED) → an OSCILLATING motion signal (0,0,1,1,0,1,1,...) that
# motion-periodicity turns into rep "peak" boundaries (a constant-high signal would NOT oscillate).
_PATTERN = [STILL, STILL, MOVED]
OSCILLATING = [(float(i), _PATTERN[i % 3], "x") for i in range(21)]


def _core(*, detectors) -> Percept:
    vision = FakeVision(script={"x": Verdict(satisfied=False, confidence=0.9, evidence="n/a")})
    core = Percept.create(vision=vision, discover_plugins=False)
    # A default (watch) goal that also selects an edge detector: it watches AND counts reps from the
    # cheap motion signal. (A type="count" goal runs its own track and never builds a gate bank.)
    state = {}
    if detectors:
        state["detectors"] = detectors
    core.add_goal(Goal(id="reps", condition="a rep happened", say="rep counted", state=state))
    return core


async def _collect(core, source, **kw):
    return [inc async for inc in run(core, source=source, **kw)]


class RuntimeDetectorTests(unittest.TestCase):
    def test_detectors_emit_boundaries_in_run(self):
        core = _core(detectors=["motion-periodicity"])
        stats = RuntimeStats()
        incidents = asyncio.run(_collect(core, OSCILLATING, stats=stats))
        self.assertGreater(stats.boundaries, 0)  # the bank ran at native fps and proposed peaks
        # perceive_boundary fires per boundary (no fire_when authored) -> counting incidents, no VLM call
        self.assertGreater(len(incidents), 0)
        self.assertTrue(all(i.concern_id == "reps" for i in incidents))

    def test_no_detectors_no_banks_byte_for_byte(self):
        core = _core(detectors=None)  # empty _TYPE_DETECTOR_PRESET -> no banks
        stats = RuntimeStats()
        asyncio.run(_collect(core, OSCILLATING, stats=stats))
        self.assertEqual(stats.boundaries, 0)

    def test_admission_detector_wakes_one_goal_judge_instead_of_counting_boundary(self):
        class OneCandidate:
            signal = "frame"
            name = "test-one-candidate"

            def __init__(self):
                self.seen = 0

            def reset(self):
                self.seen = 0

            def push(self, t, frame):
                self.seen += 1
                return (
                    [Boundary(self.name, "candidate", t, 1.0)]
                    if self.seen == 2 else []
                )

        register_detector(
            OneCandidate.name,
            OneCandidate,
            senses="test candidate",
            output="one candidate",
            route="admission",
        )
        try:
            core = _core(detectors=[OneCandidate.name])
            core.vision._script["x"] = Verdict(  # deterministic test backend fixture
                satisfied=True, confidence=0.95, evidence="verified"
            )
            stats = RuntimeStats()
            incidents = asyncio.run(
                _collect(core, [(float(i), STILL, "x") for i in range(4)], stats=stats)
            )
        finally:
            unregister(OneCandidate.name)

        self.assertEqual(stats.candidate_wakes, 1)
        self.assertEqual(stats.judge_calls, 1)
        self.assertEqual(stats.boundaries, 0)
        self.assertEqual(len(incidents), 1)


if __name__ == "__main__":
    unittest.main()
