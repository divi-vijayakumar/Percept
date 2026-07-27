"""E2E: tie the rhythm sense together — eye (pose-channels) → brain (perceive_rhythm) → concern lifecycle.

Exercises the WHOLE integrated path with a FAKE pose provider (no RTMO, deterministic): the real
``pose-channels`` detector MEASURES an oscillating pose, the brain COUNTS complete reps via
``perceive_rhythm``, and the count composes with the gate, ``fire_when``, the goal-set channel prior, the
discrete/continuous regime seam, and the windowed (endurance) rep count. Plus the legacy boundary fallback.
"""

from __future__ import annotations

import math
import unittest

from percept import Goal, Percept
from percept.behavior import coerce_authored_rhythm
from percept.harness.detectors import make_detector
from percept.harness.pose import PoseDetection, PoseKeypoint

try:
    import numpy as np
    _HAVE_NUMPY = True
except ImportError:  # pragma: no cover - numpy is an extra, absent in the L1 stdlib lane
    _HAVE_NUMPY = False


class FakeRepPose:
    """A pose provider that oscillates the wrists once per ``period`` frames → a clean rep on limb-vertical."""

    def __init__(self, period: int = 12, amp: float = 0.3):
        self.i, self.period, self.amp = 0, period, amp

    def detect(self, _frame):
        wy = 0.5 - self.amp * math.sin(2 * math.pi * self.i / self.period)  # wrists rise/fall once per cycle
        self.i += 1
        k = [PoseKeypoint(0.5, 0.5) for _ in range(17)]
        k[5], k[6] = PoseKeypoint(0.4, 0.5), PoseKeypoint(0.6, 0.5)      # shoulders
        k[9], k[10] = PoseKeypoint(0.4, wy), PoseKeypoint(0.6, wy)        # wrists (the moving DOF)
        k[11], k[12] = PoseKeypoint(0.45, 0.8), PoseKeypoint(0.55, 0.8)   # hips (torso length = 0.3)
        k[15], k[16] = PoseKeypoint(0.45, 0.95), PoseKeypoint(0.55, 0.95)  # ankles (still)
        return [PoseDetection(tuple(k), confidence=0.9)]


def _frame(w: int = 8, h: int = 8):
    """The eye's documented input: an RGB uint8 HxWx3 buffer.

    The pose model is faked here, the EYE is not: ``PoseChannels.measure`` decodes what the frame loop
    published before its provider sees it, so this must be a real pixel buffer and not ``None``."""
    return np.zeros((h, w, 3), dtype=np.uint8)


def _run(core, goal_id, *, n_cycles=8, period=12, fps=6.0):
    """Drive the real pose-channels eye → perceive_rhythm for n_cycles reps; return user fire events."""
    eye = make_detector("pose-channels")
    eye._provider = FakeRepPose(period=period)
    fires = []
    for i in range(n_cycles * period):
        fe = core.perceive_rhythm(goal_id, t=i / fps, channels=eye.measure(_frame()))
        if fe is not None:
            fires.append((i / fps, fe))
    return fires


@unittest.skipUnless(_HAVE_NUMPY, "needs numpy (a frame-backed eye reads pixels)")
class RhythmE2ETests(unittest.TestCase):
    def _core(self, **rhythm):
        core = Percept.create(discover_plugins=False)
        state = {"type": "rhythm", "count": 0}
        if rhythm:
            state["rhythm"] = rhythm
        core.add_goal(Goal(id="reps", condition="a completed push-up", say="rep", state=state))
        return core

    def test_full_chain_eye_to_brain_to_count(self):
        # eye MEASURES the oscillating pose, brain COUNTS complete reps, count lands on the concern.
        core = self._core(channel="limb-vertical")
        fires = _run(core, "reps", n_cycles=8)
        self.assertAlmostEqual(core.edge_counts["reps"], 8, delta=1)        # brain-owned count
        self.assertEqual(core.goals.get("reps").state["count"], core.edge_counts["reps"])
        self.assertEqual(len(fires), core.edge_counts["reps"])             # one fire per confirmed rep

    def test_reps_are_first_class_spans(self):
        core = self._core(channel="limb-vertical")
        _run(core, "reps", n_cycles=6)
        reps = core.rhythm["reps"].reps
        self.assertGreaterEqual(len(reps), 5)
        for r in reps:                                                     # start -> peak -> stop ordering
            self.assertLessEqual(r.start_t, r.peak_t)
            self.assertLessEqual(r.peak_t, r.stop_t)

    def test_windowed_count_is_the_endurance_story(self):
        core = self._core(channel="limb-vertical")
        _run(core, "reps", n_cycles=10, period=12, fps=6.0)               # ~2s of reps
        tr = core.rhythm["reps"]
        now = tr.last_rep.stop_t
        recent = tr.windowed_count(now, window_s=1.0)                     # reps in the last second
        self.assertLessEqual(recent, tr.count)                           # window <= total
        self.assertGreater(recent, 0)
        self.assertEqual(tr.windowed_count(now, window_s=1000.0), tr.count)  # huge window = all

    def test_fire_when_gates_user_alert_not_the_count(self):
        core = self._core(channel="limb-vertical", **{})  # add fire_when via state below
        core.goals.get("reps").state["fire_when"] = "n == 5"
        fires = _run(core, "reps", n_cycles=8)
        self.assertAlmostEqual(core.edge_counts["reps"], 8, delta=1)      # still counts every rep
        self.assertEqual(len(fires), 1)                                  # but alerts ONLY on the 5th

    def test_goal_set_prior_channel_is_honored(self):
        core = self._core(channel="limb-vertical")
        _run(core, "reps", n_cycles=6)
        self.assertEqual(core.rhythm["reps"].selected, "limb-vertical")  # the brain-supplied channel won

    def test_discrete_continuous_regime_seam(self):
        # the regime the brain declares for the eye is recorded + flows to the tracker (continuous = research
        # seam: still counted by the shipped discrete counter for now, but the mode is first-class).
        self.assertEqual(coerce_authored_rhythm({"mode": "continuous", "channel": "ankle-spread"})["rhythm"]["mode"], "continuous")
        self.assertEqual(coerce_authored_rhythm({"mode": "discrete"})["rhythm"]["mode"], "discrete")
        self.assertNotIn("mode", coerce_authored_rhythm({"mode": "bogus"}).get("rhythm", {}))  # firewall drops junk
        core = self._core(mode="continuous", channel="limb-vertical")
        _run(core, "reps", n_cycles=5)
        self.assertGreater(core.edge_counts["reps"], 0)                  # counts regardless of declared mode

    def test_expected_rom_gate_rejects_too_small_reps_e2e(self):
        # the fidget rejection end-to-end: a too-high expected ROM gates the (real-amplitude) reps to 0; a
        # matching one keeps them. The eye measures, the prior says how big a real rep must be on that channel.
        gated = self._core(channel="limb-vertical", expected_rom=20.0)  # min_amp 10 ≫ the rep prominence
        self.assertEqual(len(_run(gated, "reps", n_cycles=6)), 0)
        self.assertEqual(gated.edge_counts.get("reps", 0), 0)  # never written when every rep is gated out
        kept = self._core(channel="limb-vertical", expected_rom=1.0)   # min_amp 0.5 < the rep prominence
        _run(kept, "reps", n_cycles=6)
        self.assertGreater(kept.edge_counts["reps"], 0)

    def test_entity_scoped_count_follows_the_performer_not_a_bystander(self):
        # multi-person: two subjects through the SAME goal — A oscillates (real reps), B is still (a bystander).
        # The brain selects the performer (most reps on the prior channel); the goal's count is A's, not B's.
        core = self._core(channel="c")
        for i in range(96):
            tt = i / 6.0
            core.perceive_rhythm("reps", t=tt, channels={"c": 0.3 * math.sin(2 * math.pi * i / 12)}, entity_id="A")
            core.perceive_rhythm("reps", t=tt, channels={"c": 0.0}, entity_id="B")  # a still bystander
        self.assertGreaterEqual(core.edge_counts["reps"], 6)                  # the performer's reps are the count
        self.assertEqual(core.goals.get("reps").state["subject"], "A")        # the brain picked A, not the bystander
        self.assertGreaterEqual(core.rhythm["reps"].count, 6)                 # core.rhythm[gid] mirrors the performer

    def test_legacy_boundary_path_untouched(self):
        core = Percept.create(discover_plugins=False)
        core.add_goal(Goal(id="g", condition="rep", say="x", state={"type": "watch"}))
        for k in range(4):
            core.perceive_boundary("g", t=float(k))
        self.assertEqual(core.edge_counts["g"], 4)                       # blind +1 still works


if __name__ == "__main__":
    unittest.main()
