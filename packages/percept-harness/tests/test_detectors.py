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

from percept_harness.detectors import (
    CATEGORY_AUDIO,
    CATEGORY_AUDIO_EVENT,
    CATEGORY_CHANGE,
    CATEGORY_FRAME,
    CATEGORY_MOTION,
    CATEGORY_POSE,
    CATEGORY_PRESENCE,
    CATEGORY_SEMANTIC_RELEVANCE,
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
from percept_harness.pose import PoseDetection, PoseKeypoint
from percept_harness.tier0 import compile_watchspec, watchspec_unsupported


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
    """B2 — acoustic-onset is the first-class AUDIO sense; audio detectors are discoverable as a class."""

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


class TestPoseSkillRegistration(unittest.TestCase):
    """The first MODEL-backed skill must register + carry a manifest WITHOUT importing pose runtimes."""

    def test_pose_registered_with_manifest(self):
        from percept_harness.detectors import manifest_of

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
        out = [b for t in range(9) for b in d.push(float(t), object())]
        self.assertEqual([b.t for b in out], [2.0, 6.0])

    def test_pose_channels_measures_decomposed_basis_without_deciding(self):
        # The rhythm-sense eye MEASURES a channel basis and makes no rep decision.
        class FakePose:
            def detect(self, frame):
                k = [PoseKeypoint(0.5, 0.5) for _ in range(17)]
                k[5], k[6] = PoseKeypoint(0.4, 0.5), PoseKeypoint(0.6, 0.5)  # shoulders y=0.5
                k[9], k[10] = PoseKeypoint(0.4, 0.2), PoseKeypoint(0.6, 0.2)  # wrists raised (y=0.2)
                k[11], k[12] = PoseKeypoint(0.45, 0.8), PoseKeypoint(0.55, 0.8)  # hips y=0.8 -> torso L=0.3
                k[15], k[16] = PoseKeypoint(0.2, 0.9), PoseKeypoint(0.8, 0.9)  # ankles spread
                return [PoseDetection(tuple(k), confidence=0.9, provider="fake")]

        self.assertEqual(manifest_of("pose-channels").category, CATEGORY_FRAME)
        d = make_detector("pose-channels")
        d._provider = FakePose()
        ch = d.measure(object())
        self.assertEqual(set(ch), {"limb-vertical", "ankle-spread", "hip-vertical", "torso-vertical"})
        # wrist 0.3 above shoulder, normalized by torso length L=0.3 -> 1.0 (channels are scale-comparable)
        self.assertAlmostEqual(ch["limb-vertical"], 1.0, places=6)
        # push() carries each channel as a "sample" boundary (the remote-edge up-channel).
        bs = d.push(1.0, object())
        self.assertEqual({b.channel for b in bs}, set(ch))
        self.assertTrue(all(b.kind == "sample" for b in bs))

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
        first = d.measure(object())
        second = d.measure(object())  # dropped -> carries forward, never an empty/zero spike
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
