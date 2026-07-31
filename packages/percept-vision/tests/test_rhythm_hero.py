"""The rhythm HERO — connect the eye (per-person measures) to the entity graph so counting is per-performer.

Proves the producer that was missing: ``measure_entities`` stops collapsing to one body, ``RhythmHero``
associates detections into stable ids and the brain SELECTS the performer (most reps on the prior channel),
and the two-tier ``escalate_count`` defers a low-confidence count to an injected backstop. Deterministic, no
RTMO / no VLM: a fake multi-person pose provider + a synthetic performer-vs-bystander stream.

Beyond the happy path this file measures the ARITHMETIC a count's honesty rests on, and the geometry
underneath it:

* **The coverage denominator.** ``hero_track_coverage`` is the only presence signal the multi-person
  path publishes — ``PoseChannels``' own per-frame health flags are written by ``measure()`` and never
  by ``measure_entities()`` — and the benchmark escalates a count on it. It divides by the frames that
  carried a detection, not by the frames observed, so a clip the eye mostly failed on reports 1.0.
* **The shared mirror slot.** ``result()`` reads one ``core.rhythm[goal_id]`` and never checks that the
  tracker there belongs to ``goal.state["subject"]``. Both rhythm producers write that slot.
* **The torso normalizer.** Every channel is divided by ``body_scale``, which is floored at
  ``_MIN_BODY_SCALE``. Below the floor a channel's amplitude is a function of the floor rather than of
  the body, which turns ``expected_rom``'s absolute anti-fidget gate into a distance gate. The clamp is
  now OBSERVABLE — ``body_scale_measured`` returns ``(scale, floored)`` and every ``PersonMeasure``
  carries ``scale_floored`` — so the units half of that contract is satisfiable; the brain-side half
  (not applying the absolute gate when the normalizer is not a real measurement) is still open, and
  ``TorsoNormalizerTests``' xfail is what stays red until it lands.
* **The single-subject lock.** The ``measure()`` path carries the last measurement forward through an
  occlusion of any length and never ages out its hip anchor.

Boundary values are hand-authored in binary-exact coordinates so the arithmetic is asserted exactly.
"""

from __future__ import annotations

import inspect
import math
import os
import unittest
from unittest import mock


from percept import Goal, Percept
from percept.rhythm_hero import RhythmHero, escalate_count
from percept.harness.detectors import PersonMeasure, make_detector, person_measures
from percept.harness.pose import (
    _MIN_BODY_SCALE,
    POSE_CHANNEL_NAMES,
    PoseDetection,
    PoseKeypoint,
    PoseSkeletonIncomplete,
    body_scale,
    body_scale_measured,
    pose_channels,
)

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


# ---------------------------------------------------------------------------
# The hero's coverage ARITHMETIC — what the denominator is, and what it is not
# ---------------------------------------------------------------------------
#
# ``hero_track_coverage`` is the only PRESENCE signal the multi-person path publishes: the eye's own
# per-frame health attributes are set by ``PoseChannels.measure`` and never by ``measure_entities``,
# so a consumer of the hero has nothing else to read about how often it saw anybody. (The one
# per-person health fact it does now get is ``PersonMeasure.scale_floored``, carried by
# ``person_measures`` itself — see ``ClampObservabilityTests``; it says nothing about presence.)
# The benchmark treats coverage as the blind-count trigger (``res.hero_track_coverage < 0.3`` ->
# escalate), which makes its denominator load-bearing.

_BLIND_TRACK_COVERAGE = 0.3  # the hero arm's escalate threshold (eval/bench/repcount.py, env-overridable)


def _rhythm_core(prior: dict | None = None, *, goal_id: str = "g"):
    """A core with an identity seam and one rhythm goal — the shape ``RhythmHero`` is built against."""
    from percept.identity import AppearanceIdentity
    try:
        from percept.harness.appearance import appearance_signature
    except Exception:  # pragma: no cover - the embedder is an extra
        appearance_signature = None
    core = Percept.create(discover_plugins=False,
                          identity=AppearanceIdentity(), embedder=appearance_signature)
    core.add_goal(Goal(id=goal_id, condition="a completed rep", say="rep",
                       state={"type": "rhythm", "count": 0,
                              "rhythm": prior if prior is not None else {"channel": "limb-vertical"}}))
    return core


def _wave(i: int, period: int = 12, amp: float = 0.3) -> float:
    """The performer's wrist height at step ``i`` — one full oscillation per ``period`` steps."""
    return 0.5 - amp * math.sin(2 * math.pi * i / period)


class HeroCoverageDenominatorTests(unittest.TestCase):
    """``hero_track_coverage`` divides by frames OBSERVED, not by frames that carried a detection.

    ``observe`` used to return early on an empty measure list *before* incrementing its frame tally, so
    a frame in which the eye found nobody never reached the denominator and numerator and denominator
    moved together — pinning the field at 1.0 for any clip whose only failure mode was empty detection.
    The published meaning of the field is "fraction of processed frames the performer was actually
    tracked", and every frame handed to ``observe`` was processed whether or not a body came back.
    """

    #: 100 frames handed to the hero, a detection on every fifth one — a performer visible 20% of the clip.
    N_OBSERVED, EVERY = 100, 5

    def _sparse_hero(self):
        core = _rhythm_core()
        hero = RhythmHero(core, "g", prior_channel="limb-vertical")
        for i in range(self.N_OBSERVED):
            seen = (i % self.EVERY == 0)
            measures = person_measures([_person(_wave(i), cx=0.2)]) if seen else []
            hero.observe(i / 6.0, measures, frame=None)
        return core, hero

    def test_the_denominator_counts_every_frame_the_hero_was_given(self):
        # Rewritten (BUG-A-06-01). The mechanism, pinned: 100 observe() calls, 20 of them with a body.
        _core, hero = self._sparse_hero()
        self.assertEqual(hero.measurement.observed, self.N_OBSERVED)
        self.assertEqual(hero.measurement.measured, self.N_OBSERVED // self.EVERY)
        self.assertEqual(sum(hero._present.values()), self.N_OBSERVED // self.EVERY)

    def test_coverage_is_the_fraction_of_observed_frames_the_performer_was_tracked(self):
        # Rewritten from `..._still_reports_full_coverage`, which pinned the defect (coverage 1.0 for a
        # performer visible in a fifth of the clip). The xfail beside it is removed: this IS the contract.
        _core, hero = self._sparse_hero()
        res = hero.result()
        self.assertIsNotNone(res.subject)
        honest = (self.N_OBSERVED // self.EVERY) / self.N_OBSERVED
        self.assertAlmostEqual(res.hero_track_coverage, honest, places=12)
        # ... and being honest is what makes the count escalate instead of being asserted clean.
        self.assertLess(res.hero_track_coverage, _BLIND_TRACK_COVERAGE)

    def test_a_fully_tracked_clip_and_a_mostly_empty_one_are_told_apart(self):
        """Rewritten (BUG-A-06-01): the two clips used to report the SAME coverage float."""
        _sparse_core, sparse = self._sparse_hero()
        dense_core = _rhythm_core()
        dense = RhythmHero(dense_core, "g", prior_channel="limb-vertical")
        for i in range(self.N_OBSERVED):
            dense.observe(i / 6.0, person_measures([_person(_wave(i), cx=0.2)]), frame=None)
        sparse_res, dense_res = sparse.result(), dense.result()
        self.assertLess(sparse_res.hero_track_coverage, dense_res.hero_track_coverage)
        self.assertEqual(dense_res.hero_track_coverage, 1.0)
        # ...over the SAME denominator, which the result now publishes so a ratio carries its base.
        self.assertEqual(sparse_res.frames_observed, dense_res.frames_observed)
        self.assertEqual(sparse_res.frames_measured, self.N_OBSERVED // self.EVERY)
        self.assertEqual(dense_res.frames_measured, self.N_OBSERVED)

    def test_the_published_evidence_base_separates_a_short_clip_from_a_mostly_blind_one(self):
        """``0.2 over 100 frames`` and ``0.2 over 5 frames`` are the same float and not the same fact."""
        long_core, long_hero = self._sparse_hero()
        short_core = _rhythm_core()
        short = RhythmHero(short_core, "g", prior_channel="limb-vertical")
        for i in range(5):
            measures = person_measures([_person(_wave(i), cx=0.2)]) if i == 0 else []
            short.observe(i / 6.0, measures, frame=None)
        self.assertAlmostEqual(long_hero.result().hero_track_coverage,
                               short.result().hero_track_coverage, places=12)
        self.assertEqual((long_hero.result().frames_observed, short.result().frames_observed), (100, 5))

    def test_a_bystander_present_throughout_does_lower_the_performers_coverage(self):
        """The denominator is not inert: a frame that carried SOMEBODY else counts against the
        performer, which is exactly why the empty-frame case reads as a bug rather than a choice."""
        core = _rhythm_core()
        hero = RhythmHero(core, "g", prior_channel="limb-vertical")
        for i in range(40):
            people = [_person(0.5, cx=0.8)]                       # the still bystander, every frame
            if i % 2 == 0:
                people.insert(0, _person(_wave(i), cx=0.2))        # the performer, every other frame
            hero.observe(i / 6.0, person_measures(people), frame=None)
        res = hero.result()
        self.assertEqual(hero._frames, 40)
        self.assertAlmostEqual(res.hero_track_coverage, 0.5, places=12)


class HeroResultBoundaryTests(unittest.TestCase):
    """``result()`` at its edges: no frames, no subject, no coverage samples, one entity, one rep."""

    def test_a_clip_where_nobody_was_ever_detected_reports_zeroes_not_a_divide_by_zero(self):
        core = _rhythm_core()
        hero = RhythmHero(core, "g", prior_channel="limb-vertical")
        for i in range(25):
            hero.observe(i / 6.0, [], frame=None)
        res = hero.result()
        self.assertEqual(hero.measurement.measured, 0)   # nobody, ever ...
        self.assertEqual(res.frames_observed, 25)        # ... over 25 frames that were still processed
        self.assertEqual(res.hero_track_coverage, 0.0)   # the guarded denominator
        self.assertEqual(res.channel_coverage, 0.0)
        self.assertEqual((res.count, res.subject, res.n_entities), (0, None, 0))
        self.assertEqual((res.signal, res.reps), ([], []))
        # No evidence is not confidence: an empty signal scores 0.0 and therefore escalates.
        self.assertEqual(res.periodicity, 0.0)
        self.assertEqual(res.regularity, 0.0)
        self.assertEqual(escalate_count(res.periodicity, res.count, lambda: 99), (99, True))

    def test_no_subject_leaves_coverage_at_zero_even_though_people_were_tracked(self):
        """A goal the core does not hold: the graph still assigns ids, but nothing selects a performer,
        so both coverage floats fall back to 0.0 — the same reading as a fully occluded clip."""
        core = _rhythm_core()
        hero = RhythmHero(core, "absent-goal", prior_channel="limb-vertical")
        for i in range(20):
            hero.observe(i / 6.0, person_measures([_person(_wave(i), cx=0.2)]), frame=None)
        res = hero.result()
        self.assertIsNone(res.subject)
        self.assertEqual(hero._frames, 20)                 # frames WERE processed
        self.assertEqual(res.n_entities, 1)                # a person WAS tracked
        self.assertEqual(res.hero_track_coverage, 0.0)     # ... and coverage still reads zero
        self.assertEqual(res.channel_coverage, 0.0)

    def test_a_prior_channel_the_eye_never_measures_reports_zero_coverage(self):
        """Empty ``cov_samples``: an unmeasured channel is reported with the same 0.0 as a fully
        occluded one, so ``channel_coverage`` alone cannot tell "not visible" from "not measured"."""
        core = _rhythm_core()
        hero = RhythmHero(core, "g", prior_channel="no-such-channel")
        for i in range(24):
            hero.observe(i / 6.0, person_measures([_person(_wave(i), cx=0.2)]), frame=None)
        res = hero.result()
        self.assertEqual(hero._cov, {})
        self.assertEqual(res.channel_coverage, 0.0)
        self.assertEqual(res.signal, [])          # nothing to score periodicity on either
        self.assertEqual(res.periodicity, 0.0)
        self.assertEqual(res.hero_track_coverage, 1.0)   # the performer WAS tracked the whole clip

        occluded = _rhythm_core()
        blind = RhythmHero(occluded, "g", prior_channel="ankle-spread")
        for i in range(24):
            blind.observe(i / 6.0, person_measures([_person(_wave(i), cx=0.2, leg_conf=0.0)]), frame=None)
        self.assertEqual(blind.result().channel_coverage, res.channel_coverage)

    def test_a_single_performer_is_reported_as_one_entity_at_full_coverage(self):
        core = _rhythm_core()
        hero = RhythmHero(core, "g", prior_channel="limb-vertical")
        for i in range(72):
            hero.observe(i / 6.0, person_measures([_person(_wave(i), cx=0.2)]), frame=None)
        res = hero.result()
        self.assertEqual(res.n_entities, 1)
        self.assertEqual(res.subject, core.goals.get("g").state.get("subject"))
        self.assertEqual(res.hero_track_coverage, 1.0)
        self.assertEqual(len(res.signal), 72)
        self.assertGreater(res.channel_coverage, 0.5)

    def test_regularity_is_zero_below_two_reps_and_positive_once_intervals_exist(self):
        """``regularity`` is 1/(1+CV) over inter-rep INTERVALS: one rep yields no interval, so the
        tracker publishes 0.0 rather than the ``_ChannelState`` default of 1.0."""
        def drive(n):
            core = _rhythm_core()
            hero = RhythmHero(core, "g", prior_channel="limb-vertical")
            for i in range(n):
                hero.observe(i / 6.0, person_measures([_person(_wave(i), cx=0.2)]), frame=None)
            return hero.result()

        one_rep = drive(14)
        self.assertEqual(one_rep.count, 1)
        self.assertEqual(one_rep.regularity, 0.0)
        two_reps = drive(20)
        self.assertEqual(two_reps.count, 2)
        self.assertGreater(two_reps.regularity, 0.0)


class HeroPerformerMirrorTests(unittest.TestCase):
    """One tracker store, keyed by identity — and ``core.rhythm[goal_id]`` is only a published mirror.

    ``_rhythm_multi[(goal_id, entity_id)]`` is the OWNED state, with ``entity_id=None`` keying the
    unattributed single-subject stream (that IS its identity). ``core.rhythm[goal_id]`` used to be a
    second, incompatible owner of the same fact: the legacy branch read it as storage and adopted
    whatever the per-entity branch had left there, while the per-entity branch overwrote it as a derived
    mirror. ``result()`` now reads the SUBJECT's tracker, so the count and the observability beside it
    describe one entity.
    """

    def _driven(self, n: int = 72):
        core = _rhythm_core()
        crowd = FakeCrowdPose()          # a moving performer + a still bystander, two entities
        hero = RhythmHero(core, "g", prior_channel="limb-vertical")
        for i in range(n):
            hero.observe(i / 6.0, person_measures(crowd.detect(None)), frame=None)
        return core, hero

    def test_the_reported_count_comes_from_the_subjects_own_tracker(self):
        # Rewritten (BUG-A-06-02): `result()` read `core.rhythm[goal_id]` and never checked the tracker
        # there belonged to `goal.state["subject"]`, so poking the shared slot made count and subject
        # describe different people with nothing in result() noticing.
        core, hero = self._driven()
        res = hero.result()
        performer = core._rhythm_multi[("g", res.subject)]
        self.assertIs(core.rhythm["g"], performer)         # the mirror agrees with the selection
        bystander_key = next(k for k in core._rhythm_multi if k[1] != res.subject)
        core.rhythm["g"] = core._rhythm_multi[bystander_key]      # point the mirror at a bystander
        still_honest = hero.result()
        self.assertEqual(still_honest.subject, res.subject)
        self.assertEqual(still_honest.count, performer.count)    # ... the SUBJECT's tally, regardless
        self.assertNotEqual(performer.count, core._rhythm_multi[bystander_key].count)

    def test_the_legacy_stream_gets_its_own_tracker_keyed_by_no_entity(self):
        """Rewritten (BUG-A-06-02): a later ``perceive_rhythm(entity_id=None)`` used to find the
        performer's tracker in the shared slot and fold unattributed samples straight into it. ``None``
        is now a key in the same store — a stream of its own, and never a candidate performer."""
        core, hero = self._driven()
        subject = hero.result().subject
        performer = core._rhythm_multi[("g", subject)]
        n_trackers = len(core._rhythm_multi)
        core.perceive_rhythm("g", t=999.0, channels=pose_channels(_person(0.9, cx=0.9).keypoints))
        self.assertEqual(len(core._rhythm_multi), n_trackers + 1)
        self.assertIn(("g", None), core._rhythm_multi)
        self.assertIsNot(core._rhythm_multi[("g", None)], performer)
        self.assertIs(core.rhythm["g"], core._rhythm_multi[("g", None)])   # the mirror follows the writer
        self.assertIs(core.rhythm_tracker("g", subject), performer)        # ...the store does not move

    def test_an_unattributed_legacy_stream_does_not_change_the_performers_count(self):
        # Rewritten from an xfail (BUG-A-06-02): a 72-frame unattributed stream DOUBLED the selected
        # performer's count (12 vs 6) by being folded into their tracker.
        core, hero = self._driven()
        subject = hero.result().subject
        performer = core._rhythm_multi[("g", subject)]
        before = performer.count
        self.assertGreaterEqual(before, 4)   # the performer really did rep (sanity)
        for i in range(72, 144):             # a second stream, belonging to nobody the hero tracked
            core.perceive_rhythm("g", t=i / 6.0,
                                 channels=pose_channels(_person(_wave(i), cx=0.9).keypoints))
        self.assertEqual(performer.count, before)
        self.assertEqual(hero.result().count, before)
        self.assertGreater(core._rhythm_multi[("g", None)].count, 0)   # it counted — on its own stream

    def test_an_unattributed_stream_is_never_elected_the_performer(self):
        """The selection ranges over ATTRIBUTED keys only, so a stream belonging to nobody the graph
        tracked cannot become the subject published beside the count."""
        core, hero = self._driven()
        subject = hero.result().subject
        for i in range(72, 240):             # a long, loud unattributed stream — more reps than anyone
            core.perceive_rhythm("g", t=i / 6.0,
                                 channels=pose_channels(_person(_wave(i), cx=0.9).keypoints))
        self.assertGreater(core._rhythm_multi[("g", None)].count,
                           core._rhythm_multi[("g", subject)].count)
        hero.observe(240 / 6.0, person_measures(FakeCrowdPose().detect(None)), frame=None)
        self.assertIsNotNone(core.goals.get("g").state["subject"])
        self.assertEqual(core.goals.get("g").state["subject"], subject)

    def test_the_first_per_entity_call_moves_the_mirror_but_not_the_legacy_stream(self):
        """The per-entity branch still assigns the mirror slot unconditionally — that is what a mirror
        is. What changed is that the legacy stream's tracker is no longer IN that slot to be discarded."""
        core = _rhythm_core()
        for i in range(48):
            core.perceive_rhythm("g", t=i / 6.0,
                                 channels=pose_channels(_person(_wave(i), cx=0.2).keypoints))
        legacy = core.rhythm["g"]
        self.assertGreaterEqual(legacy.count, 4)
        RhythmHero(core, "g", prior_channel="limb-vertical").observe(
            8.0, person_measures([_person(0.5, cx=0.9)]), frame=None)
        self.assertIsNot(core.rhythm["g"], legacy)
        self.assertIs(core.rhythm_tracker("g", None), legacy)   # still owned, still counted

    def test_switching_to_the_hero_mid_clip_keeps_the_legacy_streams_reps(self):
        """A producer switch mid-clip no longer DISCARDS anything — but it does change what the
        published tally MEANS, from "this stream's" to "the selected performer's", and those have
        different denominators. Refusing a producer switch outright is a separate ownership rule with a
        conductor-side owner; it is deliberately NOT applied here, so a client that legitimately
        alternates keeps working. What is pinned is that no state is lost either way."""
        core = _rhythm_core()
        for i in range(48):
            core.perceive_rhythm("g", t=i / 6.0,
                                 channels=pose_channels(_person(_wave(i), cx=0.2).keypoints))
        counted = core.goals.get("g").state["count"]
        self.assertGreaterEqual(counted, 4)
        # The runtime switches producer — same goal, same person, now measured per-entity.
        hero = RhythmHero(core, "g", prior_channel="limb-vertical")
        hero.observe(8.0, person_measures([_person(_wave(48), cx=0.2)]), frame=None)
        self.assertEqual(core.rhythm_tracker("g", None).count, counted)   # nothing was discarded
        # The published count is now the new performer's, who has just started — a different measurement,
        # not a lost one, and the legacy stream's evidence is still readable at its own key.
        self.assertEqual(core.goals.get("g").state["count"], hero.result().count)
        self.assertIsNot(core.rhythm_tracker("g", None), core.rhythm_tracker("g", hero.result().subject))


class EscalateCountBoundaryTests(unittest.TestCase):
    """The two-tier honesty policy at its threshold. ``tau`` is a bare literal on the signature."""

    def test_the_shipped_default_threshold_is_four_tenths(self):
        # Unregistered and unconfigurable from the outside: no tunable guards it, so this does.
        self.assertEqual(inspect.signature(escalate_count).parameters["tau"].default, 0.4)

    def test_confidence_exactly_at_tau_keeps_the_edge_count(self):
        # The comparison is `confidence < tau`, so the boundary itself is TRUSTED, not escalated.
        def recount():  # pragma: no cover - must NOT run at the boundary
            raise AssertionError("the boundary value must not escalate")

        self.assertEqual(escalate_count(0.4, edge_count=7, recount=recount, tau=0.4), (7, False))

    def test_a_hair_below_tau_escalates(self):
        self.assertEqual(escalate_count(0.4 - 1e-12, edge_count=7, recount=lambda: 33, tau=0.4), (33, True))

    def test_a_tau_of_zero_disables_escalation_even_for_a_zero_confidence_count(self):
        def recount():  # pragma: no cover - unreachable with tau=0
            raise AssertionError("tau=0 must never escalate")

        self.assertEqual(escalate_count(0.0, edge_count=3, recount=recount, tau=0.0), (3, False))

    def test_a_hero_with_no_signal_escalates_through_the_default_threshold(self):
        core = _rhythm_core()
        hero = RhythmHero(core, "g", prior_channel=None)   # nothing accumulates a signal
        for i in range(30):
            hero.observe(i / 6.0, person_measures([_person(_wave(i), cx=0.2)]), frame=None)
        res = hero.result()
        self.assertGreaterEqual(res.count, 1)              # a count WAS produced ...
        self.assertEqual(res.periodicity, 0.0)             # ... on no measured evidence
        self.assertEqual(escalate_count(res.periodicity, res.count, lambda: -1), (-1, True))


# ---------------------------------------------------------------------------
# The torso normalizer the whole channel basis is divided by
# ---------------------------------------------------------------------------

def _upright(torso: float, spread: float, *, cx: float = 0.5, hip_y: float = 0.5):
    """A vertical pose: shoulders exactly ``torso`` above the hips (same x, so the Euclidean
    shoulder->hip distance IS ``torso``), ankles separated by ``spread``."""
    k = [PoseKeypoint(cx, hip_y) for _ in range(17)]
    dx, sy = 0.0625, hip_y - torso
    k[5], k[6] = PoseKeypoint(cx - dx, sy), PoseKeypoint(cx + dx, sy)          # shoulders
    k[9], k[10] = PoseKeypoint(cx - dx, sy), PoseKeypoint(cx + dx, sy)          # wrists (still)
    k[11], k[12] = PoseKeypoint(cx - dx, hip_y), PoseKeypoint(cx + dx, hip_y)   # hips
    k[15] = PoseKeypoint(cx - spread / 2.0, hip_y + torso)                       # ankles (the rep DOF)
    k[16] = PoseKeypoint(cx + spread / 2.0, hip_y + torso)
    return tuple(k)


class TorsoNormalizerTests(unittest.TestCase):
    """``body_scale`` is the length unit every channel is divided by. Its floor is a clamp, not a
    repair: below it the amplitudes stop being a property of the body."""

    NEAR_TORSO, NEAR_ROM = 0.5, 0.25        # a near subject: rep range = 0.5 torso-lengths
    FAR_TORSO, FAR_ROM = 0.015625, 0.0078125  # the SAME 0.5 torso-lengths, three metres further away

    def test_the_shipped_floor_is_five_hundredths(self):
        self.assertEqual(_MIN_BODY_SCALE, 0.05)

    def test_a_torso_exactly_at_the_floor_returns_the_floor(self):
        self.assertEqual(body_scale(_upright(0.05, 0.1, hip_y=0.5)), 0.05)

    def test_a_torso_just_above_the_floor_is_the_true_measurement(self):
        self.assertEqual(body_scale(_upright(0.0625, 0.1, hip_y=0.5)), 0.0625)

    def test_a_pose_collapsed_to_a_single_point_returns_the_floor_not_zero(self):
        collapsed = tuple(PoseKeypoint(0.5, 0.5) for _ in range(17))
        self.assertEqual(body_scale(collapsed), _MIN_BODY_SCALE)

    def test_a_lying_pose_keeps_its_true_torso_length(self):
        """The case Euclidean was chosen for: shoulders and hips share a y, so a vertical-only
        normalizer would collapse to the floor and every channel would read ten times its amplitude."""
        lying = [PoseKeypoint(0.5, 0.5) for _ in range(17)]
        lying[5], lying[11] = PoseKeypoint(0.75, 0.5), PoseKeypoint(0.25, 0.5)
        lying[6], lying[12] = PoseKeypoint(0.75, 0.625), PoseKeypoint(0.25, 0.625)
        lying = tuple(lying)
        self.assertEqual(body_scale(lying), 0.5)
        vertical_only = 0.5 * (abs(lying[11].y - lying[5].y) + abs(lying[12].y - lying[6].y))
        self.assertEqual(vertical_only, 0.0)
        self.assertEqual(body_scale(lying) / max(vertical_only, _MIN_BODY_SCALE), 10.0)

    def test_an_asymmetric_torso_is_the_mean_of_the_two_sides(self):
        asym = [PoseKeypoint(0.5, 0.5) for _ in range(17)]
        asym[5], asym[11] = PoseKeypoint(0.25, 0.0), PoseKeypoint(0.25, 0.125)
        asym[6], asym[12] = PoseKeypoint(0.75, 0.0), PoseKeypoint(0.75, 0.375)
        self.assertEqual(body_scale(tuple(asym)), 0.25)

    def test_two_subjects_with_identical_body_relative_motion_report_different_amplitudes(self):
        # Same motion in body units ...
        self.assertEqual(self.NEAR_ROM / self.NEAR_TORSO, self.FAR_ROM / self.FAR_TORSO)
        self.assertEqual(body_scale(_upright(self.NEAR_TORSO, 0.0)), self.NEAR_TORSO)
        self.assertEqual(body_scale(_upright(self.FAR_TORSO, 0.0)), _MIN_BODY_SCALE)  # floored
        near = pose_channels(_upright(self.NEAR_TORSO, self.NEAR_ROM))["ankle-spread"]
        far = pose_channels(_upright(self.FAR_TORSO, self.FAR_ROM))["ankle-spread"]
        # ... but the far subject's amplitude is divided by the FLOOR, not by its own torso, so the
        # two channels differ by exactly floor / true-torso.
        self.assertAlmostEqual(near, 0.5, places=12)
        self.assertAlmostEqual(far, self.FAR_ROM / _MIN_BODY_SCALE, places=12)
        self.assertAlmostEqual(near / far, _MIN_BODY_SCALE / self.FAR_TORSO, places=9)

    def _count_reps(self, torso, rom, *, expected_rom, n=80, period=10, fps=6.0, report_clamp=True):
        """Drive one subject's ankle-spread stream through the brain and return the count.

        ``report_clamp`` is the contract's own hinge. The eye publishes whether the normalizer was the
        body or the clamp (``body_scale_measured``); a consumer that FORWARDS that fact gets the gate
        declined when it is meaningless, and a consumer that drops it keeps the old behaviour — which is
        exactly what "either the clamp is observable to the consumer, or the absolute gate is not applied"
        means. The two shipped producers (``RhythmDrive``, ``RhythmHero``) forward it automatically."""
        prior = {"channel": "ankle-spread"}
        if expected_rom is not None:
            prior.update(expected_rom=expected_rom, expected_rom_space="channel_native")
        core = _rhythm_core(prior)
        for i in range(n):
            # Opens AT the trough (1 - cos, not 1 + sin) so there is no half-amplitude opening
            # excursion: every excursion in the stream is a full rep, and the two distances are then
            # comparable rep-for-rep rather than differing by whether a partial one cleared the gate.
            spread = rom * (1.0 - math.cos(2 * math.pi * i / period)) / 2.0
            keypoints = _upright(torso, spread)
            floored = body_scale_measured(keypoints)[1] if report_clamp else False
            core.perceive_rhythm("g", t=i / fps, channels=pose_channels(keypoints),
                                 scale_floored=floored)
        tracker = core.rhythm.get("g")
        return tracker.count if tracker is not None else 0

    def test_the_absolute_rom_gate_becomes_a_distance_gate_for_a_caller_that_drops_the_clamp_flag(self):
        """The failure the floor buys, and it is still reachable: ``expected_rom`` is an ABSOLUTE
        prominence in channel-native units, and channel-native units are ``1 / body_scale``. For a
        subject under the floor those units shrink, so an identical rep silently falls under the
        anti-fidget gate and counts zero — unless the caller forwards the clamp the eye reported."""
        gate = 0.5  # expected ROM = the near subject's exact rep range; amp_gate_frac defaults to 0.5
        near = self._count_reps(self.NEAR_TORSO, self.NEAR_ROM, expected_rom=gate, report_clamp=False)
        far = self._count_reps(self.FAR_TORSO, self.FAR_ROM, expected_rom=gate, report_clamp=False)
        self.assertGreaterEqual(near, 4)   # the near subject reps
        self.assertEqual(far, 0)           # the identical motion, further away, reps nothing

    def test_the_distant_subject_counts_normally_once_the_absolute_gate_is_off(self):
        """Proof the zero above is the GATE meeting the floor, not a signal too small to count."""
        self.assertGreaterEqual(
            self._count_reps(self.FAR_TORSO, self.FAR_ROM, expected_rom=None, report_clamp=False), 4)

    def test_the_same_body_relative_motion_counts_the_same_at_any_distance(self):
        # The xfail is removed: the BRAIN half has landed. Same rep, same fraction of the body, two
        # distances. An anti-fidget gate expressed in torso-lengths must not become a distance gate,
        # so the tracker DECLINES it for any run whose normalizer was the clamp.
        self.assertEqual(self.NEAR_ROM / self.NEAR_TORSO, self.FAR_ROM / self.FAR_TORSO)
        near = self._count_reps(self.NEAR_TORSO, self.NEAR_ROM, expected_rom=0.5)
        far = self._count_reps(self.FAR_TORSO, self.FAR_ROM, expected_rom=0.5)
        self.assertGreaterEqual(near, 4)
        self.assertEqual(far, near)

    def test_the_decline_is_recorded_so_the_brain_can_route_around_it(self):
        """Declining silently would trade one invisible failure for another: a count that no longer
        carries the anti-fidget guarantee its prior asked for. The decline is published the same way an
        occluded channel's coverage is — a reason to route to the silhouette fallback, not to assert."""
        prior = {"channel": "ankle-spread", "expected_rom": 0.5, "expected_rom_space": "channel_native"}
        core = _rhythm_core(prior)
        for i in range(80):
            spread = self.FAR_ROM * (1.0 + math.sin(2 * math.pi * i / 10)) / 2.0
            keypoints = _upright(self.FAR_TORSO, spread)
            core.perceive_rhythm("g", t=i / 6.0, channels=pose_channels(keypoints),
                                 scale_floored=body_scale_measured(keypoints)[1])
        tracker = core.rhythm["g"]
        self.assertTrue(tracker.min_amp_declined)
        self.assertEqual(tracker.measurement.unnormalized, 80)

    def test_a_near_subject_keeps_the_absolute_gate_it_authored(self):
        """The decline is not a blanket disarm: a subject whose normalizer IS a real measurement is
        still held to the anti-fidget gate, which is the whole reason the prior exists."""
        fidget = self._count_reps(self.NEAR_TORSO, self.NEAR_ROM / 8.0, expected_rom=0.5)
        self.assertEqual(fidget, 0)
        core_reps = self._count_reps(self.NEAR_TORSO, self.NEAR_ROM, expected_rom=0.5)
        self.assertGreaterEqual(core_reps, 4)


class ClampObservabilityTests(unittest.TestCase):
    """The consumer can now TELL that a channel vector was normalized by the floor, not by the body.

    The contract: *either the clamp is observable to the consumer, or the absolute gate is not applied
    when the normalizer is not a real measurement.* This is the first half. ``body_scale`` alone
    discards which branch of the ``max()`` won, so a returned 0.05 was either a real 0.05 torso or a
    clamped 0.015625 one and nothing downstream could distinguish them — which is why the absolute
    ``expected_rom`` gate silently became a distance gate (7 reps near, 0 for the identical
    body-relative motion far away; amplitude ratio 0.05 / 0.015625 = 3.2).

    ``PersonMeasure.scale_floored`` is the load-bearing half. ``PoseChannels`` already recorded the same
    fact in ``_last_floored``, but ONLY on the single-subject ``measure()`` path — ``measure_entities``
    deliberately leaves those per-frame health flags alone (pinned elsewhere in this file), so the
    multi-person ``RhythmHero`` path had no backstop at all.
    """

    def _measure(self, torso: float, *, cx: float = 0.5, conf: float = 0.9):
        det = PoseDetection(_upright(torso, 0.1, cx=cx, hip_y=0.5), confidence=conf)
        return person_measures([det])[0]

    def test_a_near_subject_reports_a_real_normalizer(self):
        m = self._measure(0.5)
        self.assertFalse(m.scale_floored)
        self.assertEqual(body_scale_measured(_upright(0.5, 0.1)), (0.5, False))

    def test_a_distant_subject_reports_the_clamp(self):
        m = self._measure(0.015625)
        self.assertTrue(m.scale_floored)
        # ...and this is precisely the case whose channels are NOT in torso-lengths:
        self.assertAlmostEqual(m.channels["ankle-spread"], 0.1 / _MIN_BODY_SCALE, places=12)

    def test_the_flag_is_per_person_because_distance_is_not_a_frame_level_fact(self):
        # Two people, one frame, one near and one far: a single frame-level flag could not say this.
        near = PoseDetection(_upright(0.5, 0.1, cx=0.2), confidence=0.9)
        far = PoseDetection(_upright(0.015625, 0.0078125, cx=0.8), confidence=0.9)
        measures = person_measures([near, far])
        self.assertEqual([m.scale_floored for m in measures], [False, True])

    def test_the_flag_defaults_false_so_a_hand_built_measure_stays_constructible(self):
        # Additive field: existing callers that build a PersonMeasure directly are untouched.
        hand_built = PersonMeasure(box=(0.0, 0.0, 0.1, 0.4), channels={"ankle-spread": 1.0},
                                   pose_conf=0.9, coverage={"ankle-spread": 1.0})
        self.assertFalse(hand_built.scale_floored)

    @unittest.skipUnless(_HAVE_NUMPY, "needs numpy (a frame-backed eye reads pixels)")
    def test_the_single_subject_path_reports_the_same_fact_on_its_own_health_flag(self):
        # `_last_floored` now reads the clamp directly instead of inferring it from a 1% epsilon band
        # around the floor value — so a real torso of 0.0502, which the old `<= floor * 1.01` test
        # called clamped, is correctly reported as the measurement it is.
        eye = make_detector("pose-channels")
        eye._provider = _FixedPoses([PoseDetection(_upright(0.0502, 0.1), confidence=0.9)])
        eye.measure(_frame())
        self.assertLessEqual(0.0502, _MIN_BODY_SCALE * 1.01)  # inside the retired epsilon band
        self.assertFalse(eye._last_floored)
        eye._provider = _FixedPoses([PoseDetection(_upright(0.015625, 0.1), confidence=0.9)])
        eye.measure(_frame())
        self.assertTrue(eye._last_floored)


class ShortKeypointProviderTests(unittest.TestCase):
    """A provider that emits fewer than 17 keypoints: refused where the detection is BUILT.

    The ``keypoints[COCO_*]`` reads in ``pose_channels`` / ``channel_coverage`` and in the detectors
    are unguarded, so a short skeleton used to surface as a bare ``IndexError`` from whichever index
    reached past the end first — mid-frame, and out of ``DetectorBank.push`` into the frame loop. The
    arity check now lives once, in ``PoseDetection.__post_init__``, so the raise happens at the seam
    that produced the bad data instead of somewhere downstream of it, and it is typed. The reads stay
    unguarded on purpose: a guard would have to substitute a joint, which silently changes what the
    eye measures.
    """

    @staticmethod
    def _short_keypoints(count: int = 16) -> tuple[PoseKeypoint, ...]:
        """A COCO-shaped keypoint tuple truncated to ``count``, with a real torso in place."""
        k = [PoseKeypoint(0.5, 0.5) for _ in range(count)]
        k[5], k[6] = PoseKeypoint(0.5, 0.25), PoseKeypoint(0.5, 0.25)     # shoulders
        k[11], k[12] = PoseKeypoint(0.5, 0.75), PoseKeypoint(0.5, 0.75)   # hips
        return tuple(k)

    def test_a_short_skeleton_cannot_be_carried_by_a_detection_at_all(self):
        with self.assertRaises(PoseSkeletonIncomplete) as caught:
            PoseDetection(self._short_keypoints(), confidence=0.9)
        self.assertIsInstance(caught.exception, ValueError)
        self.assertEqual(caught.exception.got, 16)

    def test_body_scale_survives_because_it_only_reads_indices_below_thirteen(self):
        # Why the check belongs at the DTO and not at the index sites: the normalizer measures the
        # truncated tuple perfectly well, so there is no single read that could have caught this.
        short = self._short_keypoints()
        self.assertEqual(len(short), 16)
        self.assertEqual(body_scale(short), 0.5)

    def test_the_channel_basis_still_raises_on_a_raw_sixteen_keypoint_tuple(self):
        with self.assertRaises(IndexError):
            pose_channels(self._short_keypoints())

    def test_the_multi_person_producer_can_no_longer_be_handed_an_unreadable_person(self):
        # `person_measures` used to raise mid-frame, half way through building the list. Its input
        # is now arity-checked by construction, so every detection it sees is measurable.
        with self.assertRaises(PoseSkeletonIncomplete):
            person_measures([PoseDetection(self._short_keypoints(), confidence=0.9)])

    @unittest.skipUnless(_HAVE_NUMPY, "needs numpy (a frame-backed eye reads pixels)")
    def test_both_eye_entry_points_refuse_the_same_provider_with_the_typed_error(self):
        # A custom provider is the only way here — both shipped ones guarantee 17 (RTMO filters on
        # `len(kpts) >= 17`, BlazePose builds a fixed 17-element list). The raise now names the
        # arity instead of being an IndexError from an arbitrary joint index.
        short = self._short_keypoints()

        class _ShortSkeletonProvider:
            name = "short"

            def detect(self, pixels):
                return [PoseDetection(short, confidence=0.9)]

        for method in ("measure", "measure_entities"):
            with self.subTest(method=method):
                eye = make_detector("pose-channels")
                eye._provider = _ShortSkeletonProvider()
                with self.assertRaises(PoseSkeletonIncomplete):
                    getattr(eye, method)(_frame())

    def test_a_sub_confidence_pose_is_still_dropped_before_any_measurement(self):
        # The confidence filter runs first and is unaffected by the arity check upstream of it.
        low = PoseDetection(_person(0.5, cx=0.5).keypoints, confidence=0.1)
        self.assertEqual(person_measures([low]), [])


# ---------------------------------------------------------------------------
# The single-subject lock on the ``measure()`` path
# ---------------------------------------------------------------------------

class _FixedPoses:
    """A provider that returns whatever list it currently holds — the occlusion / newcomer script."""

    def __init__(self, poses=None):
        self.poses = list(poses or [])

    def detect(self, _frame):
        return list(self.poses)


class _Oscillator:
    """A single performer whose wrists oscillate; ``blank`` drops the detection entirely."""

    def __init__(self, period: int = 12):
        self.i, self.period, self.blank = 0, period, False

    def detect(self, _frame):
        if self.blank:
            return []
        wy = _wave(self.i, self.period)
        self.i += 1
        return [_person(wy, cx=0.2)]


@unittest.skipUnless(_HAVE_NUMPY, "needs numpy (a frame-backed eye reads pixels)")
class PoseChannelsSingleSubjectLockTests(unittest.TestCase):
    """``PoseChannels.measure`` tracks ONE person: a hip-centre anchor picks the nearest detection,
    and a dropped pose carries the previous measurement forward."""

    def _eye(self, provider):
        eye = make_detector("pose-channels")
        eye._provider = provider
        return eye

    def test_the_min_pose_confidence_is_three_tenths_and_the_boundary_is_kept(self):
        from percept.harness.detectors import PoseChannels

        self.assertEqual(PoseChannels._MIN_POSE_CONF, 0.3)   # class-private, unregistered, unconfigurable
        exact = self._eye(_FixedPoses([_person(0.5, cx=0.4, conf=0.3)]))
        self.assertNotEqual(exact.measure(_frame()), {})      # >= keeps the boundary detection
        below = self._eye(_FixedPoses([_person(0.5, cx=0.4, conf=0.29)]))
        self.assertEqual(below.measure(_frame()), {})

    def test_zero_detections_on_the_first_frame_is_an_empty_dict_not_a_carry_forward(self):
        eye = self._eye(_FixedPoses([]))
        self.assertEqual(eye.measure(_frame()), {})
        self.assertFalse(eye._last_present)
        self.assertIsNone(eye._anchor)

    def test_every_detection_below_min_conf_is_indistinguishable_from_an_empty_frame(self):
        eye = self._eye(_FixedPoses([_person(0.5, cx=0.2, conf=0.1), _person(0.5, cx=0.8, conf=0.2)]))
        self.assertEqual(eye.measure(_frame()), {})
        self.assertFalse(eye._last_present)

    def test_a_long_occlusion_carries_the_last_measurement_forward_unchanged(self):
        provider = _Oscillator()
        eye = self._eye(provider)
        for _ in range(12):
            live = eye.measure(_frame())
        provider.blank = True
        occluded = [eye.measure(_frame()) for _ in range(60)]   # ten seconds of nobody at 6 fps
        self.assertTrue(all(m == live for m in occluded))
        # Nothing in the RETURN VALUE distinguishes them: same keys, same floats, no gap marker.
        self.assertEqual(set(occluded[-1]), set(live))
        self.assertFalse(eye._last_present)   # the only signal, and it is a private attribute

    def test_the_carried_forward_signal_reads_as_a_still_subject_not_a_lost_one(self):
        """A ten-second gap produces no rep AND no gap marker — the count simply stops advancing,
        which is the same shape a subject who paused would produce."""
        provider = _Oscillator()
        eye = self._eye(provider)
        core = _rhythm_core()
        for i in range(48):
            core.perceive_rhythm("g", t=i / 6.0, channels=eye.measure(_frame()))
        during_reps = core.rhythm["g"].count
        self.assertGreaterEqual(during_reps, 3)

        provider.blank = True
        for i in range(48, 108):
            core.perceive_rhythm("g", t=i / 6.0, channels=eye.measure(_frame()))
        self.assertEqual(core.rhythm["g"].count, during_reps)          # no rep ...
        self.assertTrue(all(r.stop_t < 48 / 6.0 for r in core.rhythm["g"].reps))
        # ... and the remote-edge up-channel carries only "sample" boundaries, identical every frame.
        gap = [eye.push(t / 6.0, _frame()) for t in range(108, 112)]
        self.assertEqual({b.kind for batch in gap for b in batch}, {"sample"})
        self.assertEqual({b.value for b in gap[0]}, {b.value for b in gap[-1]})

    def test_two_equidistant_candidates_resolve_to_the_first_deterministically(self):
        # Exact binary coordinates so the tie is a real tie, not a float accident: hips at 0.25 and
        # 0.75 are equidistant from an anchor at 0.5.
        left = _hips_at(0.25, 0.5, conf=0.5)
        right = _hips_at(0.75, 0.5, conf=0.9)
        for _ in range(3):
            eye = self._eye(_FixedPoses([left, right]))
            eye._anchor = (0.5, 0.5)
            eye.measure(_frame())
            self.assertEqual(eye._anchor, (0.25, 0.5))   # min() keeps the first, not the most confident

    def test_without_an_anchor_the_most_confident_detection_opens_the_lock(self):
        eye = self._eye(_FixedPoses([_hips_at(0.25, 0.5, conf=0.5), _hips_at(0.75, 0.5, conf=0.9)]))
        self.assertIsNone(eye._anchor)
        eye.measure(_frame())
        self.assertEqual(eye._anchor, (0.75, 0.5))

    def test_the_anchor_never_ages_out_so_a_newcomer_at_the_same_place_inherits_the_lock(self):
        provider = _FixedPoses([_hips_at(0.25, 0.5, conf=0.9)])
        eye = self._eye(provider)
        eye.measure(_frame())
        self.assertEqual(eye._anchor, (0.25, 0.5))

        provider.poses = []
        for _ in range(500):                       # over a minute of nobody in frame
            eye.measure(_frame())
        self.assertEqual(eye._anchor, (0.25, 0.5))  # the lock outlives the subject

        # A DIFFERENT person walks into that spot, half as confident as someone across the frame.
        newcomer, elsewhere = _hips_at(0.26, 0.5, conf=0.4), _hips_at(0.9, 0.5, conf=0.99)
        provider.poses = [elsewhere, newcomer]
        eye.measure(_frame())
        self.assertEqual(eye._anchor, (0.26, 0.5))  # continuity picked the stranger, silently

    def test_reset_is_the_only_way_to_drop_the_lock(self):
        eye = self._eye(_FixedPoses([_hips_at(0.25, 0.5, conf=0.9)]))
        eye.measure(_frame())
        eye.reset()
        self.assertIsNone(eye._anchor)
        self.assertEqual(eye._last, {})
        eye._provider.poses = [_hips_at(0.25, 0.5, conf=0.5), _hips_at(0.75, 0.5, conf=0.9)]
        eye.measure(_frame())
        self.assertEqual(eye._anchor, (0.75, 0.5))   # back to most-confident, the cold-start rule

    def test_the_entity_path_neither_reads_the_anchor_nor_updates_the_health_signals(self):
        """``measure_entities`` is the multi-person producer: it keeps every person above the
        confidence floor and leaves the ``measure()`` lock and its per-frame health flags untouched."""
        eye = self._eye(_FixedPoses([_person(0.5, cx=0.2), _person(0.5, cx=0.8)]))
        eye.measure(_frame())
        anchor, present = eye._anchor, eye._last_present
        self.assertTrue(present)

        eye._provider.poses = []
        self.assertEqual(eye.measure_entities(_frame()), [])
        self.assertEqual(eye._anchor, anchor)          # untouched
        self.assertTrue(eye._last_present)             # still reports the previous measure() frame

        eye._provider.poses = [_person(0.5, cx=0.2), _person(0.5, cx=0.8), _person(0.5, cx=0.5, conf=0.1)]
        measures = eye.measure_entities(_frame())
        self.assertEqual(len(measures), 2)             # both keepers, the sub-confidence one dropped


@unittest.skipUnless(_HAVE_NUMPY, "needs numpy (a frame-backed eye reads pixels)")
class PoseChannelSelectionEnvTests(unittest.TestCase):
    """``PERCEPT_POSE_CHANNELS`` restricts the basis. It is read per call, inside ``pose_channels``,
    so it is scoped with a context manager and never left set for another module."""

    ENV = "PERCEPT_POSE_CHANNELS"

    def setUp(self):
        self.eye = make_detector("pose-channels")
        self.eye._provider = _FixedPoses([_person(0.5, cx=0.4)])
        self.addCleanup(os.environ.pop, self.ENV, None)   # belt and braces: never leak the var

    def test_the_default_basis_is_the_four_published_channels(self):
        self.assertEqual(tuple(self.eye.measure(_frame())), POSE_CHANNEL_NAMES)

    def test_a_named_subset_restricts_what_the_eye_emits(self):
        with mock.patch.dict(os.environ, {self.ENV: "ankle-spread, hip-vertical"}):
            channels = self.eye.measure(_frame())
        self.assertEqual(set(channels), {"ankle-spread", "hip-vertical"})
        # a filter, not a re-scaling: the surviving values are what the full basis carried
        self.assertEqual(channels["ankle-spread"], pose_channels(_person(0.5, cx=0.4).keypoints)["ankle-spread"])

    def test_an_all_unknown_selection_falls_back_to_the_whole_basis(self):
        # `or allch`: a typo silently restores every channel rather than starving the tracker.
        with mock.patch.dict(os.environ, {self.ENV: "torso-openness, wrist-angle"}):
            self.assertEqual(tuple(self.eye.measure(_frame())), POSE_CHANNEL_NAMES)

    def test_a_selection_of_only_separators_falls_back_to_the_whole_basis(self):
        with mock.patch.dict(os.environ, {self.ENV: " , , "}):
            self.assertEqual(tuple(self.eye.measure(_frame())), POSE_CHANNEL_NAMES)

    def test_the_variable_is_restored_when_the_scope_exits(self):
        with mock.patch.dict(os.environ, {self.ENV: "hip-vertical"}):
            self.assertEqual(set(self.eye.measure(_frame())), {"hip-vertical"})
        self.assertIsNone(os.environ.get(self.ENV))
        self.assertEqual(tuple(self.eye.measure(_frame())), POSE_CHANNEL_NAMES)

    def test_a_restricted_basis_narrows_what_the_multi_person_producer_publishes_too(self):
        with mock.patch.dict(os.environ, {self.ENV: "ankle-spread"}):
            measures = self.eye.measure_entities(_frame())
        self.assertEqual(set(measures[0].channels), {"ankle-spread"})
        # coverage is NOT filtered by the same switch — it still reports every channel's visibility.
        self.assertEqual(set(measures[0].coverage), set(POSE_CHANNEL_NAMES))


def _hips_at(x: float, y: float, *, conf: float) -> PoseDetection:
    """A pose whose hip centre is exactly ``(x, y)`` — the anchor geometry, in binary-exact values."""
    k = [PoseKeypoint(x, y) for _ in range(17)]
    k[11], k[12] = PoseKeypoint(x, y), PoseKeypoint(x, y)
    k[5], k[6] = PoseKeypoint(x, y - 0.25), PoseKeypoint(x, y - 0.25)
    return PoseDetection(tuple(k), confidence=conf)


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
