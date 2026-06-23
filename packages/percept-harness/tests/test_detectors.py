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
    AcousticOnset,
    DetectorBank,
    MotionPeriodicity,
    coerce_authored_detectors,
    detectors_for,
    list_manifests,
    make_detector,
    manifest_of,
    partition_by_capability,
    registered,
)
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
    """The first MODEL-backed skill must register + carry a manifest WITHOUT importing mediapipe (L1-safe)."""

    def test_pose_registered_with_manifest(self):
        from percept_harness.detectors import manifest_of

        self.assertIn("pose-openness", registered())
        m = manifest_of("pose-openness")
        self.assertTrue(m.senses and m.when_to_use and m.output)

    def test_construct_is_lazy_no_mediapipe_required(self):
        # constructing the detector must NOT import mediapipe — only push() loads it (opt-in/lazy)
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
