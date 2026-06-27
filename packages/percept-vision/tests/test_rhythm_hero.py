"""The rhythm HERO — connect the eye (per-person measures) to the entity graph so counting is per-performer.

Proves the producer that was missing: ``measure_entities`` stops collapsing to one body, ``RhythmHero``
associates detections into stable ids and the brain SELECTS the performer (most reps on the prior channel),
and the two-tier ``escalate_count`` defers a low-confidence count to an injected backstop. Deterministic, no
RTMO / no VLM: a fake multi-person pose provider + a synthetic performer-vs-bystander stream.
"""

from __future__ import annotations

import math
import unittest

from percept import Goal, Percept
from percept.rhythm_hero import RhythmHero, escalate_count
from percept_harness.detectors import PersonMeasure, make_detector, person_measures
from percept_harness.pose import PoseDetection, PoseKeypoint


def _person(wy: float, cx: float, conf: float = 0.9, leg_conf: float = 0.9) -> PoseDetection:
    """One COCO pose centred at ``cx`` with wrists at height ``wy`` (the moving DOF) and ankle confidence
    ``leg_conf`` (so a test can simulate occluded legs → low ankle-spread coverage)."""
    k = [PoseKeypoint(cx, 0.5) for _ in range(17)]
    k[5], k[6] = PoseKeypoint(cx - 0.05, 0.5), PoseKeypoint(cx + 0.05, 0.5)        # shoulders
    k[9], k[10] = PoseKeypoint(cx - 0.05, wy), PoseKeypoint(cx + 0.05, wy)          # wrists
    k[11], k[12] = PoseKeypoint(cx - 0.04, 0.8), PoseKeypoint(cx + 0.04, 0.8)       # hips
    k[15] = PoseKeypoint(cx - 0.04, 0.95, leg_conf)                                  # ankles (confidence varies)
    k[16] = PoseKeypoint(cx + 0.04, 0.95, leg_conf)
    return PoseDetection(tuple(k), confidence=conf, person_id=str(int(cx * 10)))


class FakeCrowdPose:
    """Two people: a performer whose wrists oscillate, and a still bystander elsewhere in the frame."""

    def __init__(self, period: int = 12, amp: float = 0.3):
        self.i, self.period, self.amp = 0, period, amp

    def detect(self, _frame):
        wy = 0.5 - self.amp * math.sin(2 * math.pi * self.i / self.period)
        self.i += 1
        return [_person(wy, cx=0.2), _person(0.5, cx=0.8)]  # performer (moving) + bystander (still)


class MeasureEntitiesTests(unittest.TestCase):
    def test_one_measure_per_person_with_channels_and_coverage(self):
        eye = make_detector("pose-channels")
        eye._provider = FakeCrowdPose()
        measures = eye.measure_entities(None)
        self.assertEqual(len(measures), 2)  # the eye no longer collapses to a single body
        for m in measures:
            self.assertIn("limb-vertical", m.channels)
            self.assertIn("ankle-spread", m.coverage)
            self.assertEqual(len(m.box), 4)

    def test_coverage_reflects_keypoint_confidence(self):
        # occluded legs (ankle confidence 0.0) → ankle-spread coverage ≈ 0 → the phase-source gate fires.
        measures = person_measures([_person(0.5, cx=0.5, leg_conf=0.0)])
        self.assertAlmostEqual(measures[0].coverage["ankle-spread"], 0.0, places=6)
        self.assertGreater(measures[0].coverage["limb-vertical"], 0.5)


class RhythmHeroTests(unittest.TestCase):
    def _core(self):
        from percept.identity import AppearanceIdentity
        try:
            from percept_harness.appearance import appearance_signature
        except Exception:  # pragma: no cover
            appearance_signature = None
        core = Percept.create(discover_plugins=False,
                              identity=AppearanceIdentity(), embedder=appearance_signature)
        core.add_goal(Goal(id="g", condition="a completed rep", say="rep",
                           state={"type": "rhythm", "count": 0, "rhythm": {"channel": "limb-vertical"}}))
        return core

    def _drive(self, core, *, n=72, period=12, fps=6.0):
        eye = make_detector("pose-channels")
        eye._provider = FakeCrowdPose(period=period)
        hero = RhythmHero(core, "g", prior_channel="limb-vertical")
        for i in range(n):
            hero.observe(i / fps, eye.measure_entities(None), frame=None)
        return hero

    def test_brain_selects_the_performer_not_the_bystander(self):
        core = self._core()
        hero = self._drive(core)
        res = hero.result()
        self.assertEqual(res.n_entities, 2)              # both people were tracked
        self.assertGreaterEqual(res.count, 4)            # the performer's reps are counted
        self.assertEqual(res.subject, core.goals.get("g").state.get("subject"))
        # the SELECTED subject is the moving one: its tracker carries the reps, the bystander's is ~0.
        bystander_reps = [tr.count for k, tr in core._rhythm_multi.items() if k[1] != res.subject]
        self.assertTrue(all(c <= 1 for c in bystander_reps), bystander_reps)

    def test_observability_is_populated(self):
        res = self._drive(self._core()).result()
        self.assertGreater(res.hero_track_coverage, 0.9)   # performer present nearly every frame
        self.assertGreater(res.channel_coverage, 0.5)      # limb keypoints visible
        self.assertEqual(len(res.signal), 72)              # the performer's prior-channel stream, for autocorr


class EscalateCountTests(unittest.TestCase):
    def test_low_confidence_defers_to_the_backstop(self):
        calls = []

        def recount():
            calls.append(1)
            return 33

        count, escalated = escalate_count(0.1, edge_count=7, recount=recount, tau=0.4)
        self.assertEqual((count, escalated), (33, True))   # blind count → VLM recount used
        self.assertEqual(len(calls), 1)

    def test_high_confidence_keeps_the_edge_count(self):
        def recount():  # pragma: no cover - must NOT be called
            raise AssertionError("recount should not run when confident")

        count, escalated = escalate_count(0.9, edge_count=7, recount=recount, tau=0.4)
        self.assertEqual((count, escalated), (7, False))   # confident → no VLM call


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
