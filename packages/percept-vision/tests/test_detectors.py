"""Edge detector registry (L1) — cheap DSP that counts boundaries, the RepCount-Δ0 mechanism.

Deterministic stdlib tests (no model, no keys):
  * MotionPeriodicity   — a synthetic oscillation with N bumps -> N 'peak' boundaries; still -> 0.
  * AcousticOnset       — an energy stream with N spikes -> N 'onset' boundaries; sustained -> 1.
  * registry + manifest — the built-ins are registered + self-describing (the skills contract).
  * selection + firewall — authored detectors validated against the vocab; default empty (byte-for-byte).
  * DetectorBank        — routes each detector its signal channel (motion vs energy).
"""

from __future__ import annotations

import unittest

from percept.adapters import Goal

from percept.harness.detectors import (
    ACOUSTIC_ONSET_HIGH,
    ACOUSTIC_ONSET_LOW,
    CATEGORY_AUDIO,
    CATEGORY_AUDIO_EVENT,
    CATEGORY_CHANGE,
    CATEGORY_FRAME,
    CATEGORY_MOTION,
    CATEGORY_POSE,
    CATEGORY_PRESENCE,
    CATEGORY_SEMANTIC_RELEVANCE,
    _TYPE_DETECTOR_PRESET,
    AcousticOnset,
    DetectorBank,
    MotionPeriodicity,
    WakePolicy,
    coerce_authored_detectors,
    detectors_for,
    detectors_in_category,
    list_manifests,
    make_detector,
    manifest_of,
    partition_by_capability,
    registered,
    should_wake,
    wake_score,
)
from percept.harness.pose import PoseDetection, PoseKeypoint
from percept.harness.tier0 import compile_watchspec, watchspec_unsupported

try:
    import numpy as _np
    _HAVE_NUMPY = True
except ImportError:  # pragma: no cover - numpy is an extra, absent in the L1 stdlib lane
    _HAVE_NUMPY = False


def _rgb_frame(w: int = 8, h: int = 8):
    """The input a frame-backed eye documents: RGB uint8 HxWx3.

    Deliberately NOT a placeholder ``object()``. A frame-backed eye now DECODES what the frame loop
    published (``percept.harness.pose.frame_pixels``) before its provider sees it, and a placeholder
    is exactly the fiction that let a whole route ship measuring nothing on a real video source."""
    return _np.zeros((h, w, 3), dtype=_np.uint8)


def _drive(det, signal):
    out = []
    for t, v in enumerate(signal):
        out.extend(det.push(float(t), v))
    return out


class TestMotionPeriodicity(unittest.TestCase):
    def test_counts_oscillation_bumps(self):
        # three clean bumps peaking at 1.0 -> three peak boundaries
        signal = [0.0, 0.5, 1.0, 0.5, 0.0, 0.5, 1.0, 0.5, 0.0, 0.5, 1.0, 0.5, 0.0]
        peaks = _drive(MotionPeriodicity(), signal)
        self.assertEqual(len(peaks), 3)
        self.assertTrue(all(b.kind == "peak" and b.detector == "motion-periodicity" for b in peaks))
        self.assertEqual([b.t for b in peaks], [2.0, 6.0, 10.0])  # the peak times, not the descent times

    def test_still_scene_emits_nothing(self):
        self.assertEqual(_drive(MotionPeriodicity(), [0.0] * 10), [])

    def test_below_min_height_is_ignored_as_noise(self):
        # jitter under min_height never confirms a peak
        self.assertEqual(_drive(MotionPeriodicity(min_height=0.2), [0.0, 0.05, 0.1, 0.05, 0.0] * 3), [])


class TestAcousticOnset(unittest.TestCase):
    def test_counts_energy_spikes(self):
        energy = [0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.4, 0.0, 0.6, 0.0]
        onsets = _drive(AcousticOnset(), energy)
        self.assertEqual([b.t for b in onsets], [2.0, 6.0, 8.0])
        self.assertTrue(all(b.kind == "onset" for b in onsets))

    def test_sustained_high_is_one_onset(self):
        # no dip below `low` -> it never re-arms, so a held tone is a single onset
        self.assertEqual(len(_drive(AcousticOnset(), [0.0, 0.9, 0.9, 0.9, 0.9])), 1)

    # --- AUDIO-GATE-DEAF ------------------------------------------------------------------
    # The shipped bar used to be energy 0.09 == RMS amplitude 0.3. Measured over 987 s / 3,973
    # capture chunks of human-annotated sound, the LOUDEST chunk in the corpus reached 0.0277 — so
    # the shipped gate woke zero times on the only real annotated material there is. These pin that
    # the default is reachable by audio of that magnitude, which is the whole claim being made:
    # it is a fix for DEAFNESS, not for recall (a bare energy gate tops out near 0.46 recall on that
    # corpus at ANY threshold, so no default can be sold as a recall fix).

    #: Real measured chunk energies from the annotated corpus: p50, p90, p99, and the loudest chunk.
    CORPUS_ENERGIES = (3.98e-06, 1.12e-04, 9.74e-03, 2.77e-02)

    def test_shipped_default_wakes_on_real_corpus_energies(self):
        for energy in (9.74e-03, 2.77e-02):  # the corpus p99 and its loudest chunk
            with self.subTest(energy=energy):
                self.assertEqual(len(_drive(AcousticOnset(), [0.0, energy])), 1)

    def test_the_old_default_was_unreachable_by_the_whole_corpus(self):
        # The regression itself, stated as a fact rather than a memory: replay every measured
        # quantile through the PREVIOUS default and confirm it produces total silence.
        stream = [0.0]
        for energy in self.CORPUS_ENERGIES:
            stream.extend((energy, 0.0))
        self.assertEqual(_drive(AcousticOnset(high=0.09, low=0.0225), stream), [])
        self.assertGreater(len(_drive(AcousticOnset(), stream)), 0)

    def test_hysteresis_relationship_is_preserved(self):
        # A factor of 4 in energy is a factor of 2 in amplitude — the same relationship the original
        # 0.09/0.0225 pair encoded, and the ratio every row of the measured frontier was swept at.
        onset = AcousticOnset()
        self.assertEqual((onset.high, onset.low), (ACOUSTIC_ONSET_HIGH, ACOUSTIC_ONSET_LOW))
        self.assertAlmostEqual(onset.low, onset.high / 4.0)

    def test_default_still_ignores_the_corpus_noise_floor(self):
        # Economy is the other half of the claim: the retuned bar must not admit ordinary quiet.
        # p50 and p90 chunk energy stay below it, so a quiet stream still costs nothing.
        self.assertEqual(_drive(AcousticOnset(), [0.0, 3.98e-06, 1.12e-04, 0.0]), [])


class TestRegistryAndManifest(unittest.TestCase):
    def test_builtins_registered(self):
        self.assertIn("motion-periodicity", registered())
        self.assertIn("acoustic-onset", registered())

    def test_manifests_are_self_describing(self):
        for m in list_manifests():
            self.assertTrue(m.name and m.senses and m.when_to_use and m.output)
        self.assertEqual(manifest_of("motion-periodicity").name, "motion-periodicity")

    def test_make_detector_and_unknown_raises(self):
        self.assertIsInstance(make_detector("acoustic-onset"), AcousticOnset)
        with self.assertRaises(KeyError):
            make_detector("telepathy")


class TestSenseCategory(unittest.TestCase):
    """Acoustic-onset is first-class AUDIO; audio detectors are discoverable as a class."""

    def test_acoustic_onset_marked_audio(self):
        self.assertEqual(manifest_of("acoustic-onset").category, CATEGORY_AUDIO)

    def test_categories_of_other_builtins(self):
        self.assertEqual(manifest_of("motion-periodicity").category, CATEGORY_MOTION)
        self.assertEqual(manifest_of("pose-openness").category, CATEGORY_FRAME)

    def test_detectors_in_category_discovers_audio_class(self):
        # the modality-as-a-class query: list every audio Tier-0 sense the way motion ones group
        self.assertIn("acoustic-onset", detectors_in_category(CATEGORY_AUDIO))
        self.assertNotIn("motion-periodicity", detectors_in_category(CATEGORY_AUDIO))
        self.assertIn("motion-periodicity", detectors_in_category(CATEGORY_MOTION))

    def test_unknown_category_is_empty(self):
        self.assertEqual(detectors_in_category("smell"), [])

    def test_wake_policy_categories_are_public(self):
        self.assertEqual(CATEGORY_PRESENCE, "presence")
        self.assertEqual(CATEGORY_POSE, "pose")
        self.assertEqual(CATEGORY_AUDIO_EVENT, "audio_event")
        self.assertEqual(CATEGORY_CHANGE, "change")
        self.assertEqual(CATEGORY_SEMANTIC_RELEVANCE, "semantic_relevance")


class TestWakePolicy(unittest.TestCase):
    def test_no_policy_is_legacy_max_channel(self):
        self.assertAlmostEqual(wake_score({"motion": 0.2, "semantic": 0.7}), 0.7)

    def test_weighted_policy_combines_weak_cues(self):
        policy = WakePolicy(threshold=0.45, weights={"presence": 0.7, "audio_event": 0.7})
        score = wake_score({"presence": 0.5, "audio_event": 0.5}, policy)
        self.assertGreater(score, 0.45)
        d = should_wake({"presence": 0.5, "audio_event": 0.5}, policy)
        self.assertTrue(d.wake)
        self.assertEqual(d.top_channel, "presence")

    def test_policy_clamps_out_of_range_inputs(self):
        policy = WakePolicy(threshold=0.9, weights={"semantic_relevance": 2.0})
        self.assertEqual(wake_score({"semantic_relevance": 9.0}, policy), 1.0)


class TestSelectionFirewall(unittest.TestCase):
    def test_coerce_drops_unknown_and_dedups(self):
        self.assertEqual(
            coerce_authored_detectors(["motion-periodicity", "bogus", "motion-periodicity"]),
            ["motion-periodicity"],
        )
        self.assertEqual(coerce_authored_detectors("motion-periodicity"), [])  # not a list
        self.assertEqual(coerce_authored_detectors(None), [])

    def test_detectors_for_authored_override(self):
        g = Goal(id="g", condition="c", say="s", state={"type": "watch", "detectors": ["acoustic-onset"]})
        self.assertEqual(detectors_for(g), ["acoustic-onset"])

    def test_detectors_for_default_is_empty(self):
        # byte-for-byte: an un-authored goal selects no detectors (no edge behavior change)
        self.assertEqual(detectors_for(Goal(id="g", condition="c", say="s")), [])

    def test_detectors_for_drops_authored_garbage(self):
        g = Goal(id="g", condition="c", say="s", state={"type": "watch", "detectors": ["nope"]})
        self.assertEqual(detectors_for(g), [])


class TestCountPresetAblation(unittest.TestCase):
    """Pins the ablation's verdict for ``type="count"``: the preset stays EMPTY.

    Measured on RepCount-A (n=20 clips / 203 annotated reps, native frame rate, shipped detector
    defaults, shipped pose provider): count MAPE 100.0% for the no-detector control, 84.8% for
    ``motion-periodicity`` (better on 13/20 clips, worse on 5 — two-sided sign test p=0.096, exact on
    0/20, within +-1 on 2/20), 194.9% for ``pose-openness`` (p=0.648, within +-1 on 4/20). Nothing beat
    the control by a margin worth a default, so nothing was installed — the ``_TYPE_DETECTOR_PRESET``
    comment carries the full table and the two detector defects the run surfaced. This is the guard on
    that decision: it fails the moment a preset is installed without re-running the ablation.
    """

    def _count_goal(self, **state):
        return Goal(
            id="reps", condition="a completed rep", say="rep",
            state={"type": "count", "count": 0, **state},
        )

    def test_count_preset_is_empty_by_measurement(self):
        self.assertEqual(_TYPE_DETECTOR_PRESET.get("count", []), [])
        self.assertEqual(_TYPE_DETECTOR_PRESET, {})  # no goal type earned a default

    def test_any_preset_ever_installed_stays_inside_the_registry_vocab(self):
        # the invariant that survives a future re-run: a preset may only name calibrated skills, so a
        # typo can never activate a detector the firewall would silently drop.
        vocab = set(registered())
        for goal_type, names in _TYPE_DETECTOR_PRESET.items():
            with self.subTest(goal_type=goal_type):
                self.assertTrue(set(names) <= vocab, f"{goal_type} -> {names} outside {sorted(vocab)}")
                self.assertEqual(coerce_authored_detectors(names), list(names))

    def test_count_goal_selects_no_detector_and_emits_no_boundary(self):
        # the shipped consequence of the empty preset, end to end: an un-authored count goal runs an
        # EMPTY bank, so a whole oscillating stream produces zero boundaries for the brain to count.
        goal = self._count_goal()
        self.assertEqual(detectors_for(goal), [])
        bank = DetectorBank(detectors_for(goal))
        emitted = []
        for t, motion in enumerate([0.0, 0.5, 1.0, 0.5, 0.0] * 4):
            emitted.extend(bank.push(float(t), motion=motion, energy=motion))
        self.assertEqual(emitted, [])

    def test_authored_detectors_remain_the_way_to_drive_a_count_goal(self):
        # the null result leaves the authored path untouched: the same stream, counted by an
        # explicitly authored detector, still produces boundaries.
        goal = self._count_goal(detectors=["motion-periodicity"])
        self.assertEqual(detectors_for(goal), ["motion-periodicity"])
        bank = DetectorBank(detectors_for(goal))
        emitted = []
        for t, motion in enumerate([0.0, 0.5, 1.0, 0.5, 0.0] * 4):
            emitted.extend(bank.push(float(t), motion=motion))
        self.assertEqual(len(emitted), 4)
        self.assertTrue(all(b.detector == "motion-periodicity" for b in emitted))


class TestPoseSkillRegistration(unittest.TestCase):
    """The first MODEL-backed skill must register + carry a manifest WITHOUT importing pose runtimes."""

    def test_pose_registered_with_manifest(self):
        from percept.harness.detectors import manifest_of

        self.assertIn("pose-openness", registered())
        m = manifest_of("pose-openness")
        self.assertTrue(m.senses and m.when_to_use and m.output)

    def test_construct_is_lazy_no_pose_runtime_required(self):
        # constructing the detector must NOT import mmpose/mediapipe — only push() loads it (opt-in/lazy)
        d = make_detector("pose-openness")
        self.assertEqual(getattr(d, "signal", None), "frame")

    def test_bank_skips_frame_detector_without_a_frame(self):
        # a pose detector in a bank gets no boundaries (and never touches mediapipe) when only scalar
        # channels are pushed -> the L1 motion/energy lane is unaffected by a registered model skill
        bank = DetectorBank(["pose-openness", "motion-periodicity"])
        out = []
        for t, v in enumerate([0.0, 0.5, 1.0, 0.5, 0.0]):
            out.extend(bank.push(float(t), motion=v))  # no frame= -> pose is skipped, mediapipe never loads
        self.assertTrue(out and all(b.detector == "motion-periodicity" for b in out))

    @unittest.skipUnless(_HAVE_NUMPY, "needs numpy (a frame-backed eye reads pixels)")
    def test_pose_openness_counts_fake_coco_keypoint_peaks(self):
        class FakePose:
            def __init__(self):
                self.values = iter([0.0, 0.8, 1.4, 0.8, 0.0, 0.8, 1.4, 0.8, 0.0])

            def detect(self, frame):
                openness = next(self.values)
                k = [PoseKeypoint(0.0, 0.0, 0.0) for _ in range(17)]
                # shoulders define width; ankles widen + wrists rise as openness increases.
                k[5] = PoseKeypoint(0.4, 0.5)
                k[6] = PoseKeypoint(0.6, 0.5)
                k[9] = PoseKeypoint(0.4, 0.5 - openness * 0.1)
                k[10] = PoseKeypoint(0.6, 0.5 - openness * 0.1)
                k[15] = PoseKeypoint(0.5 - openness * 0.1, 0.9)
                k[16] = PoseKeypoint(0.5 + openness * 0.1, 0.9)
                return [PoseDetection(tuple(k), confidence=1.0, provider="fake")]

        d = make_detector("pose-openness")
        d._provider = FakePose()
        out = [b for t in range(9) for b in d.push(float(t), _rgb_frame())]
        self.assertEqual([b.t for b in out], [2.0, 6.0])

    @unittest.skipUnless(_HAVE_NUMPY, "needs numpy (a frame-backed eye reads pixels)")
    def test_pose_channels_measures_decomposed_basis_without_deciding(self):
        # The rhythm-sense eye MEASURES a channel basis and makes no rep decision.
        class FakePose:
            def detect(self, frame):
                k = [PoseKeypoint(0.5, 0.5) for _ in range(17)]
                k[5], k[6] = PoseKeypoint(0.4, 0.5), PoseKeypoint(0.6, 0.5)  # shoulders y=0.5
                k[9], k[10] = PoseKeypoint(0.4, 0.2), PoseKeypoint(0.6, 0.2)  # wrists raised (y=0.2)
                k[11], k[12] = PoseKeypoint(0.4, 0.8), PoseKeypoint(0.6, 0.8)  # hips below shoulders -> torso L=0.3
                k[15], k[16] = PoseKeypoint(0.2, 0.9), PoseKeypoint(0.8, 0.9)  # ankles spread
                return [PoseDetection(tuple(k), confidence=0.9, provider="fake")]

        self.assertEqual(manifest_of("pose-channels").category, CATEGORY_FRAME)
        d = make_detector("pose-channels")
        d._provider = FakePose()
        ch = d.measure(_rgb_frame())
        self.assertEqual(set(ch), {"limb-vertical", "ankle-spread", "hip-vertical", "torso-vertical"})
        # wrist 0.3 above shoulder, normalized by the EUCLIDEAN torso length L=0.3 (hips directly below
        # shoulders) -> 1.0. Euclidean keeps L stable for lying/horizontal poses where |Δy| would collapse.
        self.assertAlmostEqual(ch["limb-vertical"], 1.0, places=6)
        # push() carries each channel as a "sample" boundary (the remote-edge up-channel).
        bs = d.push(1.0, _rgb_frame())
        self.assertEqual({b.channel for b in bs}, set(ch))
        self.assertTrue(all(b.kind == "sample" for b in bs))

    @unittest.skipUnless(_HAVE_NUMPY, "needs numpy (a frame-backed eye reads pixels)")
    def test_pose_channels_carries_forward_on_dropped_pose(self):
        class FlakyPose:
            def __init__(self):
                self.n = 0

            def detect(self, frame):
                self.n += 1
                if self.n == 2:
                    return []  # dropped frame
                k = [PoseKeypoint(0.5, 0.5) for _ in range(17)]
                k[5], k[6] = PoseKeypoint(0.4, 0.5), PoseKeypoint(0.6, 0.5)
                return [PoseDetection(tuple(k), confidence=0.9, provider="fake")]

        d = make_detector("pose-channels")
        d._provider = FlakyPose()
        first = d.measure(_rgb_frame())
        second = d.measure(_rgb_frame())  # dropped -> carries forward, never an empty/zero spike
        self.assertEqual(first, second)


class TestDetectorBank(unittest.TestCase):
    def test_routes_motion_and_energy_channels(self):
        bank = DetectorBank(["motion-periodicity", "acoustic-onset"])
        # feeding only the motion channel -> only the motion detector reacts
        motion_out = []
        for t, v in enumerate([0.0, 0.5, 1.0, 0.5, 0.0]):
            motion_out.extend(bank.push(float(t), motion=v))
        self.assertTrue(motion_out and all(b.detector == "motion-periodicity" for b in motion_out))

        bank.reset()
        energy_out = []
        for t, v in enumerate([0.0, 0.9, 0.0]):
            energy_out.extend(bank.push(float(t), energy=v))
        self.assertTrue(energy_out and all(b.detector == "acoustic-onset" for b in energy_out))


class TestCapabilityHandshake(unittest.TestCase):
    """The brain-side handshake: select edge skills only within what the connected edge advertised."""

    def _goal(self, detectors):
        return Goal(
            id="reps", condition="count squats", say="done",
            state={"type": "watch", "detectors": detectors},
        )

    def test_partition_none_means_all_supported(self):
        supported, dropped = partition_by_capability(["motion-periodicity", "pose-openness"], None)
        self.assertEqual(supported, ["motion-periodicity", "pose-openness"])
        self.assertEqual(dropped, [])

    def test_partition_splits_by_capability(self):
        supported, dropped = partition_by_capability(
            ["motion-periodicity", "pose-openness"], {"motion-periodicity"}
        )
        self.assertEqual(supported, ["motion-periodicity"])
        self.assertEqual(dropped, ["pose-openness"])

    def test_compile_filters_to_advertised_capabilities(self):
        # a glasses-class edge that can do motion but not pose
        spec = compile_watchspec(
            [self._goal(["motion-periodicity", "pose-openness"])],
            threshold=0.1, edge_capabilities={"motion-periodicity"},
        )
        self.assertEqual(spec.conditions[0].detectors, ["motion-periodicity"])

    def test_compile_none_sends_full_selection(self):
        spec = compile_watchspec([self._goal(["motion-periodicity", "pose-openness"])], threshold=0.1)
        self.assertEqual(spec.conditions[0].detectors, ["motion-periodicity", "pose-openness"])

    def test_watchspec_unsupported_reports_drops(self):
        goals = [self._goal(["motion-periodicity", "pose-openness"])]
        self.assertEqual(
            watchspec_unsupported(goals, {"motion-periodicity"}), [("reps", ["pose-openness"])]
        )

    def test_watchspec_unsupported_empty_when_all_supported(self):
        goals = [self._goal(["motion-periodicity"])]
        self.assertEqual(watchspec_unsupported(goals, {"motion-periodicity", "pose-openness"}), [])


if __name__ == "__main__":
    unittest.main()
