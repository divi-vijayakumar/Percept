"""Edge boundary loop (L1) — the RepCount-Δ0 fix, end-to-end through GatedWatchLoop.

A watch goal that selects ``motion-periodicity`` gets its rep count from the EDGE: the detector counts
oscillation peaks of the cheap motion signal at native fps, DECOUPLED from the (sparse, motion-gated)
vision calls. A goal with no detectors emits no boundaries (byte-for-byte today).
"""

from __future__ import annotations

import asyncio
import unittest

from percept.adapters import Goal
from percept.agent import Percept

from percept_harness.detectors import Boundary, DetectorManifest, register
from percept_harness.tier0 import Frame, GatedWatchLoop, gray_frame

W, H = 8, 8


class _EveryOtherFrame:
    """Test-only frame-backed detector: a boundary on every 2nd frame (proves frame routing, not pose)."""

    signal = "frame"
    name = "test-every-other-frame"

    def __init__(self) -> None:
        self._n = 0

    def reset(self) -> None:
        self._n = 0

    def push(self, t, frame):
        self._n += 1
        return [Boundary(self.name, "peak", t, 1.0)] if self._n % 2 == 0 else []


register("test-every-other-frame", _EveryOtherFrame,
         DetectorManifest("test-every-other-frame", "frames", "wiring test", "a peak every 2nd frame"))


def still() -> Frame:
    return gray_frame(W, H, 0)


def moving() -> Frame:
    px = [0] * (W * H)
    for i in range(W * H // 2):  # half-bright -> a big, clean motion delta against `still`
        px[i] = 255
    return Frame(w=W, h=H, pixels=tuple(px))


def _run(goal: Goal, stream):
    core = Percept.create()  # offline fakes
    core.add_goal(goal)
    loop = GatedWatchLoop(core, [goal], threshold=0.05)
    return asyncio.run(loop.run(stream))


class TestEdgeBoundaryLoop(unittest.TestCase):
    def test_edge_counts_reps_decoupled_from_vision(self):
        g = Goal(id="reps", condition="a rep is performed", say="rep",
                 state={"type": "watch", "count": 0, "detectors": ["motion-periodicity"]})
        # still,still,move,move,... -> the motion signal rises on each change and falls between:
        # four scene changes -> the detector confirms a peak at each.
        stream = [still(), still(), moving(), moving(), still(), still(),
                  moving(), moving(), still(), still()]
        report = _run(g, stream)
        self.assertGreaterEqual(report.boundary_count("reps", kind="peak"), 3)  # counted from the edge
        self.assertTrue(all(gid == "reps" for gid, _ in report.boundaries))  # attributed to the goal
        self.assertLessEqual(report.vision_calls, report.frames)  # the count is NOT a per-frame VLM job

    def test_brain_fires_on_nth_edge_boundary_without_vision(self):
        from percept.gate import FIRE

        # "tell me when I hit 3 reps" — counted entirely from edge boundaries via the nth-distinct
        # accumulator (fire_when), no per-frame VLM call.
        g = Goal(id="reps", condition="a rep is performed", say="that's 3 reps!",
                 state={"type": "watch", "count": 0, "detectors": ["motion-periodicity"], "fire_when": "n == 3"})
        stream = [still(), still(), moving(), moving(), still(), still(), moving(), moving(),
                  still(), still(), moving(), moving(), still()]
        report = _run(g, stream)
        fires = [e for e in report.fire_events if e.action == FIRE and e.goal_id == "reps"]
        self.assertEqual(len(fires), 1)  # fired exactly once — at the 3rd rep
        self.assertEqual(fires[0].text, "that's 3 reps!")

    def test_frame_backed_detector_runs_in_loop(self):
        # the live loop feeds the raw FRAME to frame-backed detectors (pose et al.), not just motion
        g = Goal(id="fg", condition="x", say="s",
                 state={"type": "watch", "count": 0, "detectors": ["test-every-other-frame"]})
        report = _run(g, [still(), moving(), still(), moving(), still(), moving()])  # 6 frames -> 3 boundaries
        self.assertEqual(report.boundary_count("fg", kind="peak"), 3)
        self.assertTrue(all(gid == "fg" for gid, _ in report.boundaries))

    def test_acoustic_onset_fires_via_energy_channel(self):
        # the live loop feeds a per-frame audio-energy signal to energy-backed detectors (acoustic-onset)
        g = Goal(id="ag", condition="the kettle is boiling", say="!",
                 state={"type": "watch", "count": 0, "detectors": ["acoustic-onset"]})
        core = Percept.create()
        core.add_goal(g)
        loop = GatedWatchLoop(core, [g], threshold=0.05)
        # a quiet→loud→quiet energy track: the onset (>=0.3 having been <0.15) fires once
        report = asyncio.run(loop.run([still()] * 6, energies=[0.0, 0.0, 0.5, 0.5, 0.0, 0.0]))
        self.assertGreaterEqual(report.boundary_count("ag", kind="onset"), 1)
        self.assertTrue(all(b.detector == "acoustic-onset" for _, b in report.boundaries))

    def test_no_detectors_means_no_boundaries(self):
        g = Goal(id="w", condition="a cup is up", say="saw it")  # default: no detectors
        report = _run(g, [still(), moving(), still(), moving()])
        self.assertEqual(report.boundaries, [])  # byte-for-byte: the edge does nothing new


if __name__ == "__main__":
    unittest.main()
