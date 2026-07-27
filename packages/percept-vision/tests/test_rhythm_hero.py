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
from percept.harness.detectors import make_detector, person_measures
from percept.harness.pose import PoseDetection, PoseKeypoint

try:
    import numpy as np
    _HAVE_NUMPY = True
except ImportError:  # pragma: no cover - numpy is an extra, absent in the L1 stdlib lane
    _HAVE_NUMPY = False


def _frame(w: int = 8, h: int = 8):
    """The eye's documented input: an RGB uint8 HxWx3 buffer.

    The pose MODEL is faked below; the eye is real, and ``measure_entities`` decodes what the frame
    loop published before its provider sees it — so this must be pixels, not ``None``."""
    return np.zeros((h, w, 3), dtype=np.uint8)


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


@unittest.skipUnless(_HAVE_NUMPY, "needs numpy (a frame-backed eye reads pixels)")
class MeasureEntitiesTests(unittest.TestCase):
    def test_one_measure_per_person_with_channels_and_coverage(self):
        eye = make_detector("pose-channels")
        eye._provider = FakeCrowdPose()
        measures = eye.measure_entities(_frame())
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


@unittest.skipUnless(_HAVE_NUMPY, "needs numpy (a frame-backed eye reads pixels)")
class RhythmHeroTests(unittest.TestCase):
    def _core(self):
        from percept.identity import AppearanceIdentity
        try:
            from percept.harness.appearance import appearance_signature
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
            hero.observe(i / fps, eye.measure_entities(_frame()), frame=None)
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


class FakePommelCrowd:
    """A pommel-horse-like crowd: a PERFORMER whose ankles oscillate cleanly (the rep DOF on the
    apparatus) plus a near-static / irregularly-fidgeting BYSTANDER (a judge/spectator) at the edge of the
    frame. The two never swap ``cx`` so the EntityGraph keeps them as distinct ids; the performer reps once
    per ``period`` and the bystander only jitters slightly off-rhythm (no completed cycles)."""

    def __init__(self, period: int = 10, amp: float = 0.35):
        self.i, self.period, self.amp = 0, period, amp

    def detect(self, _frame):
        # performer: ankles spread/together once per period (a clean ankle-spread cycle)
        spread = self.amp * (1.0 + math.sin(2 * math.pi * self.i / self.period)) / 2.0
        performer = _pommel_performer(cx=0.25, spread=spread)
        # bystander: a near-static judge/spectator far from the performer — a tiny irregular, off-period,
        # sub-amplitude tremor that never completes a rep cycle (its ankle-spread reps ~0).
        jitter = 0.004 * math.sin(2 * math.pi * self.i / 3.7) + 0.004 * math.sin(2 * math.pi * self.i / 5.3)
        bystander = _pommel_performer(cx=0.85, spread=0.03 + jitter)
        self.i += 1
        return [performer, bystander]


def _pommel_performer(cx: float, spread: float) -> PoseDetection:
    """A COCO pose centred at ``cx`` whose ANKLES are separated by ``spread`` (the ankle-spread channel),
    with arms held still (so the only moving DOF is the legs — pommel-like)."""
    k = [PoseKeypoint(cx, 0.5) for _ in range(17)]
    k[5], k[6] = PoseKeypoint(cx - 0.06, 0.45), PoseKeypoint(cx + 0.06, 0.45)        # shoulders
    k[9], k[10] = PoseKeypoint(cx - 0.06, 0.45), PoseKeypoint(cx + 0.06, 0.45)        # wrists (still)
    k[11], k[12] = PoseKeypoint(cx - 0.05, 0.75), PoseKeypoint(cx + 0.05, 0.75)       # hips
    k[15] = PoseKeypoint(cx - spread / 2.0, 0.95)                                      # ankles (the rep DOF)
    k[16] = PoseKeypoint(cx + spread / 2.0, 0.95)
    return PoseDetection(tuple(k), confidence=0.9, person_id=str(int(cx * 100)))


@unittest.skipUnless(_HAVE_NUMPY, "needs numpy (a frame-backed eye reads pixels)")
class PommelHeroSelectionTests(unittest.TestCase):
    """The pommel-horse fix as an L1: on a multi-person stream the brain must SELECT the performer (not a
    bystander) and report the PERFORMER's count, regardless of the wrong-person geometry trap."""

    def _core(self):
        from percept.identity import AppearanceIdentity
        try:
            from percept.harness.appearance import appearance_signature
        except Exception:  # pragma: no cover
            appearance_signature = None
        core = Percept.create(discover_plugins=False,
                              identity=AppearanceIdentity(), embedder=appearance_signature)
        core.add_goal(Goal(id="g", condition="a completed pommel rep", say="rep",
                           state={"type": "rhythm", "count": 0, "rhythm": self._prior()}))
        return core

    @staticmethod
    def _prior():
        # the manifest prior the conductor reads for a pommel/jack action: ankle-spread channel + the
        # absolute anti-fidget gate (expected_rom in the channel's native units). Mirrors hero_ab.EXPECTED_ROM.
        return {"channel": "ankle-spread", "expected_rom": 0.35, "expected_rom_space": "channel_native"}

    def _single_performer_count(self, *, n, period, amp, fps):
        """Reference: the SAME performer's ankle-spread stream alone (single-subject) → the count the
        crowd hero must reproduce, proving the bystander does not perturb the performer's tally."""
        core = Percept.create(discover_plugins=False)
        core.add_goal(Goal(id="g", condition="a completed pommel rep", say="rep",
                           state={"type": "rhythm", "count": 0, "rhythm": self._prior()}))
        from percept.harness.pose import pose_channels
        for i in range(n):
            spread = amp * (1.0 + math.sin(2 * math.pi * i / period)) / 2.0
            ch = pose_channels(_pommel_performer(cx=0.25, spread=spread).keypoints)
            core.perceive_rhythm("g", t=i / fps, channels=ch)
        tr = core.rhythm.get("g")
        return tr.count if tr is not None else 0

    def test_brain_picks_performer_and_counts_performer_reps(self):
        n, period, amp, fps = 80, 10, 0.35, 6.0
        core = self._core()
        eye = make_detector("pose-channels")
        eye._provider = FakePommelCrowd(period=period, amp=amp)
        hero = RhythmHero(core, "g", prior_channel="ankle-spread")
        for i in range(n):
            hero.observe(i / fps, eye.measure_entities(_frame()), frame=None)
        res = hero.result()

        # 1) the brain selected the PERFORMER, not a bystander: the reported subject is the entity whose
        #    tracker carries the reps; every OTHER entity reps ~0 (the still/irregular bystander).
        self.assertEqual(res.n_entities, 2)
        self.assertEqual(res.subject, core.goals.get("g").state.get("subject"))
        other_reps = [tr.count for k, tr in core._rhythm_multi.items() if k[1] != res.subject]
        self.assertTrue(all(c <= 1 for c in other_reps), other_reps)
        self.assertGreater(res.count, max(other_reps or [0]))  # performer out-reps the bystander

        # 2) the reported count IS the performer's — identical to the same performer measured alone, so the
        #    bystander provably does not perturb the count.
        expected = self._single_performer_count(n=n, period=period, amp=amp, fps=fps)
        self.assertGreaterEqual(expected, 4)                 # the reference actually reps (sanity)
        self.assertEqual(res.count, expected)

        # 3) the performer was tracked nearly the whole clip → a high-coverage, trustworthy count.
        self.assertGreater(res.hero_track_coverage, 0.9)
        self.assertGreater(res.channel_coverage, 0.5)        # ankles visible (not the occluded real clip)


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
