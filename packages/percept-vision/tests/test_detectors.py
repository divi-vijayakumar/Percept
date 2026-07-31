"""Edge detector registry (L1) — cheap DSP that counts boundaries, the RepCount-Δ0 mechanism.

Deterministic stdlib tests (no model, no keys):
  * MotionPeriodicity   — a synthetic oscillation with N bumps -> N 'peak' boundaries; still -> 0.
  * AcousticOnset       — an energy stream with N spikes -> N 'onset' boundaries; sustained -> 1.
  * registry + manifest — the built-ins are registered + self-describing (the skills contract).
  * selection + firewall — authored detectors validated against the vocab; default empty (byte-for-byte).
  * DetectorBank        — routes each detector its signal channel (motion vs energy).

The ``…Boundaries`` / ``…Normalizer`` / ``…Routing`` suites below pin the KNOB VALUES and the edge
cases of the same five mechanisms: what each threshold does at exactly its own bar, which
mis-configurations degrade silently rather than loudly, and which of those degradations the shipped
code documents as intended versus which are defects. Everything here is pure stdlib arithmetic — no
model, no media, no clock.

The wake-fusion suite below carried five characterization tests that pinned BUG-A-01-05 / A-02-03 /
A-02-04 / A-02-08 as the contract, plus one strict xfail. All six were rewritten against the fixed
contract when the wake-fusion contract landed; each says what it used to assert.
"""

from __future__ import annotations

import unittest
import warnings

from percept.adapters import Goal

from percept.harness import detectors as _detectors_module
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
    PoseOpenness,
    WakePolicy,
    coerce_authored_detectors,
    detectors_for,
    detectors_in_category,
    list_manifests,
    make_detector,
    manifest_of,
    partition_by_capability,
    register_detector,
    registered,
    should_wake,
    wake_score,
)
from percept.harness.pose import (
    COCO_LEFT_ANKLE,
    COCO_LEFT_HIP,
    COCO_LEFT_SHOULDER,
    COCO_LEFT_WRIST,
    COCO_RIGHT_ANKLE,
    COCO_RIGHT_HIP,
    COCO_RIGHT_SHOULDER,
    COCO_RIGHT_WRIST,
    PoseDetection,
    PoseKeypoint,
    PoseSkeletonIncomplete,
)
from percept.harness.tier0 import (
    UNSUPPORTED_DETECTOR_REASON,
    UnsupportedEdgeCapabilityWarning,
    compile_watchspec,
    watchspec_unsupported,
)
from percept.runtime._vad import frame_energy

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


def _drive(det, signal, times=None):
    out = []
    for i, v in enumerate(signal):
        out.extend(det.push(float(i if times is None else times[i]), v))
    return out


class _DecodedPixels:
    """An already-decoded buffer, which ``pose.frame_pixels`` returns untouched.

    It only has to expose ``shape``/``dtype`` to take that passthrough branch, so the openness
    arithmetic below is testable in the stdlib lane with no numpy and no image decode. The scripted
    provider never reads the pixels — the point of these tests is what the detector does with the
    keypoints it is handed, not what the eye sees."""

    shape = (8, 8, 3)
    dtype = "uint8"


def _pose(*, spread, torso=0.25, shoulder_width=0.25, wrist_up=0.0, confidence=1.0, lying=False):
    """A 17-point COCO pose whose EUCLIDEAN shoulder→hip distance is exactly ``torso`` on both sides.

    ``torso`` is what ``PoseOpenness`` now divides by (via ``pose.body_scale``); ``shoulder_width`` is
    the two shoulders' separation in X, which is what it used to divide by and which is now inert.
    They are independent parameters on purpose — the whole point of the fix is that the second can be
    driven to zero without moving the first.

    ``lying=True`` rotates the body axis into the horizontal: shoulders and hips share a y and are
    separated only in x, so the two shoulders project onto x values ``shoulder_width`` apart while the
    torso keeps its full length. That is the real-camera shape (sit-ups, push-ups, bench press) that
    collapsed the retired normalizer.
    """
    k = [PoseKeypoint(0.5, 0.5) for _ in range(17)]
    if lying:
        ls, rs = (0.5 + torso / 2.0, 0.45), (0.5 + torso / 2.0 + shoulder_width, 0.55)
        lh, rh = (ls[0] - torso, ls[1]), (rs[0] - torso, rs[1])
    else:
        ls, rs = (0.5 - shoulder_width / 2.0, 0.5), (0.5 + shoulder_width / 2.0, 0.5)
        lh, rh = (ls[0], ls[1] + torso), (rs[0], rs[1] + torso)
    k[COCO_LEFT_SHOULDER], k[COCO_RIGHT_SHOULDER] = PoseKeypoint(*ls), PoseKeypoint(*rs)
    k[COCO_LEFT_HIP], k[COCO_RIGHT_HIP] = PoseKeypoint(*lh), PoseKeypoint(*rh)
    k[COCO_LEFT_WRIST] = PoseKeypoint(ls[0], ls[1] - wrist_up)
    k[COCO_RIGHT_WRIST] = PoseKeypoint(rs[0], rs[1] - wrist_up)
    k[COCO_LEFT_ANKLE] = PoseKeypoint(0.5 - spread / 2.0, 0.9)
    k[COCO_RIGHT_ANKLE] = PoseKeypoint(0.5 + spread / 2.0, 0.9)
    return PoseDetection(tuple(k), confidence=confidence, provider="fake")


class _ScriptedPose:
    """A pose provider that replays one detection list per frame (``[]`` = a dropped detection)."""

    name = "scripted"

    def __init__(self, frames):
        self.frames = list(frames)
        self.calls = 0

    def detect(self, pixels):
        self.calls += 1
        return self.frames.pop(0) if self.frames else []


def _drive_pose(detector, n):
    """Push ``n`` frames through a scripted ``PoseOpenness`` and return the boundaries it emitted."""
    return [b for i in range(n) for b in detector.push(float(i), _DecodedPixels())]


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
        policy = WakePolicy(threshold=0.45, weights={"presence": 0.7, "audio": 0.7})
        score = wake_score({"presence": 0.5, "audio": 0.5}, policy)
        self.assertGreater(score, 0.45)
        d = should_wake({"presence": 0.5, "audio": 0.5}, policy)
        self.assertTrue(d.wake)
        self.assertEqual(d.top_channel, "presence")

    def test_policy_clamps_out_of_range_inputs(self):
        policy = WakePolicy(threshold=0.9, weights={"semantic": 2.0})
        self.assertEqual(wake_score({"semantic": 9.0}, policy), 1.0)


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

    THE POSE ROW IS STALE and stays quoted only as history: ``pose-openness`` has since had its
    normalizer replaced (BUG-A-02-01), which moves the unit its knobs were ablated against. 194.9% is
    the OLD arithmetic's number. It is not evidence about the current detector in either direction, and
    the verdict it supported — install nothing — is unchanged, because the re-measurement that could
    change it has not been run.
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


class TestCapabilityHandshakeBoundaries(unittest.TestCase):
    """``None`` and ``[]`` are opposite instructions, and only one of them is loud.

    ``capabilities=None`` means "the in-process edge runs the whole registry"; a list means "only
    these". The polarity is not a spectrum: an edge that advertises an EMPTY list drops every
    detector the brain selected, and ``compile_watchspec`` throws the dropped half away, so the
    deployment ships a detector-less condition and nothing in the compile path says so.
    """

    def _goal(self, detectors):
        return Goal(
            id="reps", condition="count squats", say="done",
            state={"type": "watch", "detectors": list(detectors)},
        )

    def test_an_empty_capability_list_drops_every_detector_that_none_would_keep(self):
        names = ["motion-periodicity", "acoustic-onset"]
        self.assertEqual(partition_by_capability(names, None), (names, []))
        self.assertEqual(partition_by_capability(names, []), ([], names))

    def test_an_empty_capability_list_compiles_a_detector_less_condition_and_says_so(self):
        # Rewritten: this test used to assert ZERO warnings — the silence encoded as the contract.
        # The same goal that keeps both skills under `None` keeps none under `[]`, and the compile
        # used to throw the dropped half away, shipping a condition that admits frames but can never
        # emit a boundary, without a word. The drop is now reported on the compiling path itself.
        goal = self._goal(["motion-periodicity", "acoustic-onset"])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            empty = compile_watchspec([goal], threshold=0.1, edge_capabilities=[])
        full = compile_watchspec([goal], threshold=0.1, edge_capabilities=None)
        self.assertEqual(full.conditions[0].detectors, ["motion-periodicity", "acoustic-onset"])
        self.assertEqual(empty.conditions[0].detectors, [])
        self.assertEqual([type(w.message) for w in caught], [UnsupportedEdgeCapabilityWarning])
        rendered = str(caught[0].message)
        self.assertIn("reps", rendered)                 # WHICH goal
        self.assertIn("motion-periodicity", rendered)   # ...and WHICH skills
        self.assertIn(UNSUPPORTED_DETECTOR_REASON, rendered)  # ADR-002's reason code
        # ...and a caller that wired the structured channel gets the drop as data.
        reported = []
        compile_watchspec(
            [goal], threshold=0.1, edge_capabilities=[], on_unsupported=reported.append
        )
        self.assertEqual(
            reported, [[("reps", ["motion-periodicity", "acoustic-onset"])]]
        )
        # The reporting half still answers the same question, quietly, for a caller who ASKS
        # before arming anything.
        with warnings.catch_warnings(record=True) as asking:
            warnings.simplefilter("always")
            self.assertEqual(
                watchspec_unsupported([goal], []),
                [("reps", ["motion-periodicity", "acoustic-onset"])],
            )
        self.assertEqual(asking, [])

    def test_the_full_capability_registry_is_still_silent(self):
        # The report is scoped to an ACTUAL drop: `capabilities=None` (in-process, run everything)
        # and a capability list that covers the selection must both compile without a word.
        goal = self._goal(["motion-periodicity"])
        for capabilities in (None, ["motion-periodicity"], ["motion-periodicity", "pose-openness"]):
            with self.subTest(capabilities=capabilities):
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    compile_watchspec([goal], threshold=0.1, edge_capabilities=capabilities)
                self.assertEqual(caught, [])

    def test_the_two_halves_of_the_handshake_read_one_traversal(self):
        # They used to be two independent walks — needs_watch_condition + detectors_for +
        # partition_by_capability, recomputed — i.e. two implementations of one rule. A goal on a
        # route that is NOT a frame Watch must be invisible to BOTH halves, identically.
        watch = self._goal(["motion-periodicity", "pose-openness"])
        not_a_watch = Goal(
            id="reminder", condition="in ten minutes", say="ping",
            state={"type": "remind", "detectors": ["pose-openness"]},
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UnsupportedEdgeCapabilityWarning)
            spec = compile_watchspec(
                [watch, not_a_watch], threshold=0.1, edge_capabilities=["motion-periodicity"],
            )
        compiled_ids = [c.goal_id for c in spec.conditions]
        reported_ids = [
            gid for gid, _ in watchspec_unsupported([watch, not_a_watch], ["motion-periodicity"])
        ]
        self.assertEqual(compiled_ids, ["reps"])
        self.assertEqual(reported_ids, ["reps"])  # never "reminder", from either half

    def test_a_capability_the_brain_never_selected_changes_nothing(self):
        supported, dropped = partition_by_capability(
            ["motion-periodicity"], ["motion-periodicity", "pose-openness", "telepathy"]
        )
        self.assertEqual((supported, dropped), (["motion-periodicity"], []))

    def test_both_halves_keep_the_selection_order_not_the_capability_order(self):
        supported, dropped = partition_by_capability(
            ["pose-openness", "acoustic-onset", "motion-periodicity"],
            ["motion-periodicity", "pose-openness"],  # advertised in a different order
        )
        self.assertEqual(supported, ["pose-openness", "motion-periodicity"])
        self.assertEqual(dropped, ["acoustic-onset"])

    def test_duplicate_names_are_preserved_on_both_sides(self):
        # The partition is not a de-duplicator (the selection firewall upstream is). A caller that
        # hands it a doubled name gets a doubled name back — and a bank built from that runs the
        # detector twice, so one bump becomes two boundaries.
        supported, dropped = partition_by_capability(
            ["motion-periodicity", "acoustic-onset", "motion-periodicity"], ["motion-periodicity"]
        )
        self.assertEqual(supported, ["motion-periodicity", "motion-periodicity"])
        self.assertEqual(dropped, ["acoustic-onset"])
        emitted = []
        bank = DetectorBank(supported)
        for t, v in enumerate([0.0, 0.5, 1.0, 0.5, 0.0]):
            emitted.extend(bank.push(float(t), motion=v))
        self.assertEqual(len(emitted), 2)


class TestWakeFusionBoundaries(unittest.TestCase):
    """The noisy-OR fusion at its own bars, and the three ways a policy degrades silently."""

    def setUp(self):
        # `coerce_wake_policy` type-checks the policy it is handed, and `test_registration.py`
        # reloads `percept.harness.detectors` in a tearDown — which rebinds `WakePolicy` to a fresh
        # class object while this module still holds the import-time one. Resolving these four
        # names through the module at call time keeps the policies built here identical to the ones
        # the fusion checks against, whatever module ran before this one in the session.
        self.Policy = _detectors_module.WakePolicy
        self.score = _detectors_module.wake_score
        self.decide = _detectors_module.should_wake
        self.coerce = _detectors_module.coerce_wake_policy
        self.Refused = _detectors_module.UnsupportedWakePolicyError

    def test_empty_signals_score_zero_and_name_no_top_channel(self):
        self.assertEqual(self.score({}), 0.0)
        d = self.decide({})
        self.assertEqual((d.score, d.top_channel, d.wake), (0.0, None, False))

    def test_a_policy_with_no_weights_is_indistinguishable_from_no_policy_at_all(self):
        # The one silent degrade that SURVIVES, and deliberately: an empty `weights` is not a typo —
        # it is a policy that prices nothing, and a table of priors that is all-default is exactly
        # `max(signals)`. Stated here so the equivalence is a decision rather than an accident.
        signals = {"presence": 0.5, "audio": 0.5}
        self.assertEqual(self.score(signals, self.Policy(threshold=0.9)), self.score(signals))
        self.assertAlmostEqual(self.score(signals, self.Policy(threshold=0.9)), 0.5)
        self.assertAlmostEqual(
            self.score(signals, self.Policy(weights={"presence": 1.0, "audio": 1.0})), 0.75
        )

    def test_a_typo_in_the_weights_field_name_is_refused_by_name(self):
        # REWRITTEN (BUG-A-02-08). This used to pin the defect as the contract: config arrives as a
        # dict, `weights` was read by exact key, and a miss was not an error — so the calibrated
        # weighted OR silently became max() with the authored threshold intact, and nothing
        # distinguished "calibrated" from "running the legacy default".
        with self.assertRaises(self.Refused) as caught:
            self.coerce({"threshold": 0.9, "weigths": {"presence": 1.0}})
        self.assertIn("weigths", str(caught.exception))
        # the correctly-spelled key still builds the policy it names
        good = self.coerce({"threshold": 0.9, "weights": {"presence": 1.0}})
        self.assertEqual(good.weights, {"presence": 1.0})

    def test_the_fusion_runs_over_the_measured_channels_not_over_the_weight_table(self):
        # REWRITTEN (BUG-A-02-04). This used to pin that weighting ONE channel silenced every other
        # measured channel — the fused score fell BELOW the legacy max it replaced (0.8 against 0.9),
        # so naming a weight LOST recall, and a channel named only in the weights (a typo) scored
        # exactly 0.0 with every sensor reporting. Weights are per-channel PRIORS defaulting to 1.0,
        # which makes the weighted form a strict generalization of the unweighted one.
        signals = {"presence": 0.9, "motion": 0.8}
        self.assertAlmostEqual(self.score(signals), 0.9)
        weighted = self.score(signals, self.Policy(weights={"motion": 1.0}))
        self.assertAlmostEqual(weighted, 0.98)
        self.assertGreaterEqual(weighted, self.score(signals))  # never below the max it replaced
        # a typo in a CHANNEL name now costs that channel's prior, not every signal present
        self.assertAlmostEqual(self.score(signals, self.Policy(weights={"presenc": 1.0})), 0.98)
        # ...and a weight of 0.0 is the way to actually silence a channel, said explicitly
        self.assertAlmostEqual(
            self.score(signals, self.Policy(weights={"presence": 0.0})), 0.8
        )

    def test_a_channel_named_in_weights_but_absent_from_signals_casts_no_vote(self):
        # REWRITTEN (BUG-A-02-04, the other half). `clean.get(channel, 0.0)` made "this sensor is not
        # reporting" arithmetically identical to "this sensor sees nothing". Only the second is
        # evidence, and an unwired sensor must not move the score in either direction.
        policy = self.Policy(weights={"presence": 1.0, "motion": 1.0})
        self.assertAlmostEqual(self.score({"motion": 0.5}, policy), 0.5)
        self.assertAlmostEqual(self.score({"motion": 0.5, "presence": 0.0}, policy), 0.5)
        self.assertEqual(self.score({}, policy), 0.0)

    def test_a_weight_above_one_saturates_its_channel_rather_than_overflowing(self):
        policy = self.Policy(weights={"semantic_relevance": 2.0})
        self.assertEqual(self.score({"semantic_relevance": 0.6}, policy), 1.0)

    def test_a_floor_of_one_admits_every_frame_including_one_with_no_signals(self):
        policy = self.Policy(threshold=1.0, floor=1.0)
        self.assertEqual(self.score({}, policy), 1.0)
        self.assertEqual(self.score({"motion": 0.0}, policy), 1.0)
        self.assertTrue(self.decide({}, policy).wake)  # unconditional admission, no signal required

    def test_a_score_exactly_at_the_threshold_wakes(self):
        self.assertTrue(self.decide({"motion": 0.25}, self.Policy(threshold=0.25)).wake)
        self.assertFalse(self.decide({"motion": 0.25}, self.Policy(threshold=0.2500001)).wake)

    def test_coercion_refuses_a_non_numeric_weight_and_a_bool_alike(self):
        # REWRITTEN (BUG-A-02-08). The firewall used to be silent in both directions: a stringified
        # number vanished (degrading the policy toward max), while a bool passed as 1.0 because a
        # Python bool IS an int. Both landed in the `not p.weights` branch = legacy max(signals)
        # wearing a configured threshold.
        with self.assertRaises(self.Refused) as string_weight:
            self.coerce({"weights": {"presence": "0.7"}})
        self.assertIn("presence", str(string_weight.exception))
        with self.assertRaises(self.Refused) as bool_weight:
            self.coerce({"weights": {"pose": True}})
        self.assertIn("bool", str(bool_weight.exception))
        for field, bad in (
            ("threshold", "0.5"), ("threshold", True), ("threshold", float("nan")),
            ("floor", None), ("floor", float("inf")),
        ):
            with self.subTest(field=field, value=repr(bad)):
                with self.assertRaises(self.Refused):
                    self.coerce({field: bad})
        # a non-finite WEIGHT is refused too — clamp01 would have laundered inf into a prior of 1.0
        with self.assertRaises(self.Refused):
            self.coerce({"weights": {"motion": float("inf")}})
        self.assertEqual(self.coerce({"weights": {"motion": 0.5}}).weights, {"motion": 0.5})

    def test_coercion_passes_a_policy_and_none_through_and_refuses_a_non_mapping(self):
        # REWRITTEN (BUG-A-02-08). A list used to coerce to a default weightless policy — a shape
        # nobody authored, silently substituted for one they did.
        policy = self.Policy(threshold=0.9, weights={"motion": 0.5})
        self.assertIs(self.coerce(policy), policy)
        self.assertIsNone(self.coerce(None))
        with self.assertRaises(self.Refused):
            self.coerce(["motion"])
        with self.assertRaises(self.Refused):
            self.coerce({"weights": ["motion"]})
        with self.assertRaises(self.Refused):
            self.coerce({"weights": {1: 0.5}})

    def test_top_channel_is_the_argmax_of_the_contributions_and_ties_take_the_first(self):
        self.assertEqual(self.decide({"motion": 0.5, "presence": 0.5}).top_channel, "motion")
        self.assertEqual(self.decide({"motion": 0.4, "presence": 0.5}).top_channel, "presence")

    def test_top_channel_names_a_channel_that_contributed_to_the_score(self):
        # REWRITTEN from a strict xfail (BUG-A-02-03). `top_channel` was the argmax of the RAW
        # signals, computed with NO reference to `policy.weights`, then returned beside a score that
        # WAS computed from them — so a decision pointed an operator at a sensor that contributed
        # nothing to the number. It is now the argmax of the same contribution terms the score summed.
        d = self.decide(
            {"presence": 0.9, "motion": 0.4}, self.Policy(weights={"presence": 0.0, "motion": 1.0})
        )
        self.assertAlmostEqual(d.score, 0.4)
        self.assertEqual(d.top_channel, "motion")

    def test_a_policy_that_prices_none_of_the_measured_channels_names_no_channel(self):
        # It used to be non-None whenever ANY signal was present, even at a score of 0.0.
        d = self.decide({"presence": 0.9, "motion": 0.4}, self.Policy(weights={"presence": 0.0, "motion": 0.0}))
        self.assertEqual(d.score, 0.0)
        self.assertIsNone(d.top_channel)
        # ...and a floor lifts the score without inventing a channel that produced it
        floored = self.decide({}, self.Policy(threshold=0.5, floor=1.0))
        self.assertTrue(floored.wake)
        self.assertIsNone(floored.top_channel)


class TestMotionPeriodicityBoundaries(unittest.TestCase):
    """The three knobs at their own edges — including the two values that make the detector useless."""

    def test_a_relative_drop_of_zero_confirms_a_peak_on_any_non_increase(self):
        # rel_drop=0 asks for a descent to 100% of the peak, i.e. no descent at all: a flat top is
        # a completed rep, so a plateau counts as one and prominence stops meaning anything.
        peaks = _drive(MotionPeriodicity(rel_drop=0.0), [0.0, 0.5, 1.0, 1.0, 1.0])
        self.assertEqual([(b.t, b.value) for b in peaks], [(2.0, 1.0)])

    def test_a_relative_drop_of_one_needs_the_signal_back_at_zero(self):
        # rel_drop=1 asks for a descent to 0% of the peak. A bump that decays to 0.1 never confirms.
        self.assertEqual(_drive(MotionPeriodicity(rel_drop=1.0), [0.0, 0.5, 1.0, 0.5, 0.2, 0.1]), [])
        self.assertEqual(
            [b.t for b in _drive(MotionPeriodicity(rel_drop=1.0), [0.0, 0.5, 1.0, 0.1, 0.0])], [2.0]
        )

    def test_a_minimum_height_of_zero_counts_pure_jitter_as_reps(self):
        peaks = _drive(MotionPeriodicity(min_height=0.0), [0.0, 0.001, 0.0, 0.001, 0.0])
        self.assertEqual([b.value for b in peaks], [0.001, 0.001])

    def test_a_descent_exactly_to_the_prominence_bar_confirms_and_one_step_short_does_not(self):
        # `v <= peak_v * (1 - rel_drop)` — the bar itself is inside the accepting side.
        bar = MotionPeriodicity(min_height=0.1, rel_drop=0.5)
        self.assertEqual([b.t for b in _drive(bar, [0.0, 1.0, 0.5])], [1.0])
        self.assertEqual(_drive(MotionPeriodicity(min_height=0.1, rel_drop=0.5), [0.0, 1.0, 0.5000001]), [])

    def test_a_flat_a_single_sample_and_a_monotonic_ramp_all_emit_nothing(self):
        self.assertEqual(_drive(MotionPeriodicity(), [0.4] * 6), [])
        self.assertEqual(MotionPeriodicity().push(0.0, 1.0), [])
        self.assertEqual(_drive(MotionPeriodicity(), [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]), [])

    def test_a_negative_signal_never_arms_the_detector(self):
        # The running peak starts at 0.0 and only ever climbs, so a signal that lives below zero is
        # invisible; a negative TROUGH under a positive bump is fine and confirms normally.
        self.assertEqual(_drive(MotionPeriodicity(), [-1.0, -0.5, -2.0, -0.5]), [])
        self.assertEqual([b.t for b in _drive(MotionPeriodicity(), [0.0, 1.0, -1.0])], [1.0])

    def test_the_shipped_minimum_height_is_deaf_to_the_quietest_measured_clips(self):
        # The ablation recorded two clips whose ENTIRE motion range peaks at 0.013 and 0.017 —
        # both under the shipped 0.02 bar, so the shipped detector emits nothing on them. This is
        # the knob, not the shape: the same stream counts fine at a bar those clips can cross.
        stream = [0.0, 0.013, 0.0, 0.017, 0.0]
        self.assertEqual(_drive(MotionPeriodicity(), stream), [])
        self.assertEqual(len(_drive(MotionPeriodicity(min_height=0.01), stream)), 2)

    def test_the_first_peak_passes_any_refractory(self):
        # `_last_emit_t` starts at -inf, so the refractory can never suppress the opening rep.
        self.assertEqual(len(_drive(MotionPeriodicity(refractory_s=1e9), [0.0, 1.0, 0.0])), 1)

    def test_a_refractory_exactly_at_the_gap_admits_and_one_tick_longer_loses_the_peak(self):
        stream = [0.0, 0.5, 1.0, 0.5, 0.0, 0.5, 1.0, 0.5, 0.0]  # confirmations 4.0 s apart
        self.assertEqual([b.t for b in _drive(MotionPeriodicity(refractory_s=4.0), stream)], [2.0, 6.0])
        # Suppression is a DROP, not a deferral: the bump state resets, so the peak is gone.
        self.assertEqual([b.t for b in _drive(MotionPeriodicity(refractory_s=4.5), stream)], [2.0])

    def test_a_backwards_timestamp_mutes_the_refractory_for_the_rest_of_the_run(self):
        # One clock rewind makes `t - _last_emit_t` negative forever after, and `_last_emit_t` is
        # only advanced on an EMIT — so every later rep is silently discarded.
        times = [10, 11, 12, 3, 4, 5, 6]
        peaks = _drive(MotionPeriodicity(refractory_s=1.0), [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0], times)
        self.assertEqual([b.t for b in peaks], [11.0])


class TestAcousticOnsetBoundaries(unittest.TestCase):
    """Hysteresis and refractory at their own bars. ``high``/``low`` are mean-square ENERGY."""

    def test_energy_exactly_at_high_fires_and_exactly_at_low_does_not_rearm(self):
        # `>=` on the way up, strict `<` on the way down: a signal that decays to exactly `low`
        # leaves the detector disarmed, so the next spike of any size is silent.
        self.assertEqual(len(_drive(AcousticOnset(high=0.5, low=0.1), [0.0, 0.5])), 1)
        held_at_low = _drive(AcousticOnset(high=0.5, low=0.1), [0.0, 0.9, 0.1, 0.9, 0.05, 0.9])
        self.assertEqual([b.t for b in held_at_low], [1.0, 5.0])

    def test_zero_length_audio_is_zero_energy_and_never_wakes(self):
        self.assertEqual(frame_energy(()), 0.0)
        self.assertEqual(_drive(AcousticOnset(), [frame_energy(())] * 4), [])

    def test_high_is_mean_square_energy_so_an_amplitude_valued_knob_is_deaf(self):
        # A caller who reads `high` as RMS AMPLITUDE and passes 0.3 has actually asked for energy
        # 0.3 == amplitude 0.55. A stream whose amplitude really is 0.3 (energy 0.09) never crosses
        # it — which is exactly how the old 0.09 default came to be uncrossable.
        amplitude_third = [0.3, -0.3, 0.3, -0.3]
        self.assertAlmostEqual(frame_energy(amplitude_third), 0.09)
        self.assertEqual(_drive(AcousticOnset(high=0.3), [0.0, frame_energy(amplitude_third)]), [])
        self.assertEqual(len(_drive(AcousticOnset(), [0.0, frame_energy(amplitude_third)])), 1)

    def test_a_sustained_tone_is_one_onset_and_a_refractory_exactly_at_the_gap_admits(self):
        self.assertEqual(len(_drive(AcousticOnset(high=0.5, low=0.1), [0.9] * 5)), 1)
        gapped = [0.0, 0.9, 0.0, 0.0, 0.0, 0.9, 0.0]  # onsets 4.0 s apart
        self.assertEqual(
            [b.t for b in _drive(AcousticOnset(high=0.5, low=0.1, refractory_s=4.0), gapped)], [1.0, 5.0]
        )

    def test_an_onset_suppressed_by_the_refractory_is_lost_not_deferred(self):
        # The suppressed spike still consumes the arm, so it is not re-emitted when the refractory
        # expires: the boundary — and the louder value it carried — is gone from the record.
        stream = [0.0, 0.9, 0.0, 0.95, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.99]
        onsets = _drive(AcousticOnset(high=0.5, low=0.1, refractory_s=5.0), stream)
        self.assertEqual([(b.t, b.value) for b in onsets], [(1.0, 0.9), (10.0, 0.99)])

    def test_an_inverted_hysteresis_pair_is_refused_at_construction(self):
        # With `low` ABOVE `high` the two bars swap roles: every sample in [high, low) both fires
        # (v >= high) and re-arms (v < low), so a dead-steady level oscillated the state machine at
        # half the sample rate. Measured, before the guard: AcousticOnset(high=0.1, low=0.5) on a
        # constant [0.2] * 5 emitted 3 onsets at t = 0, 2, 4 — phantom starts in room tone.
        with self.assertRaises(ValueError) as caught:
            AcousticOnset(high=0.1, low=0.5)
        self.assertIn("low <= high", str(caught.exception))
        # It is refused at CONSTRUCTION, so the invalid state machine never exists to be pushed to.
        with self.assertRaises(ValueError):
            AcousticOnset(high=ACOUSTIC_ONSET_LOW, low=ACOUSTIC_ONSET_HIGH)

    def test_low_equal_to_high_stays_permitted_as_the_degenerate_no_band_case(self):
        # `low == high` is NOT the defect and stays legal: no sample can be both >= high and < low
        # when they are equal, so the oscillation is unreachable. It is a band of zero width — the
        # detector re-arms the instant energy drops below the fire bar, which is twitchy but honest.
        self.assertEqual([b.t for b in _drive(AcousticOnset(high=0.2, low=0.2), [0.2] * 5)], [0.0])
        self.assertEqual(
            [b.t for b in _drive(AcousticOnset(high=0.2, low=0.2), [0.2, 0.1, 0.2])], [0.0, 2.0]
        )

    def test_a_constant_energy_level_is_at_most_one_onset_for_any_constructible_pair_of_bars(self):
        # An onset is "a sound STARTING". A level that never changes contains exactly one start,
        # whatever the two bars are set to — and now every pair the constructor accepts satisfies
        # that, because the one pair that did not is no longer constructible.
        for high, low in ((0.5, 0.1), (0.2, 0.2), (ACOUSTIC_ONSET_HIGH, ACOUSTIC_ONSET_LOW)):
            with self.subTest(high=high, low=low):
                self.assertLessEqual(len(_drive(AcousticOnset(high=high, low=low), [0.2] * 5)), 1)
        with self.assertRaises(ValueError):
            AcousticOnset(high=0.1, low=0.5)


class TestPoseOpennessNormalizer(unittest.TestCase):
    """The openness normalizer: the floored EUCLIDEAN torso length, not shoulder width in x.

    ``openness = ankle_spread / body_scale + wrist_elevation``. The retired divisor was the shoulder
    separation measured in X ONLY, guarded against an EXACT zero (``or 0.1``) — two defects on one
    line. A non-upright pose projects both shoulders onto nearly the same x, and a falsiness test
    substitutes the guard for ``0.0`` while ``1e-9`` is truthy and divides straight through: the
    emitted openness reached 19999999.46, which clears ``min_height=0.6`` and confirms as a peak. That
    is where the shipped ablation's 101 boundaries for 9 true reps (194.9% count MAPE, roughly twice as
    wrong as answering zero) came from. ``pose.body_scale`` is orientation-invariant and floored, so no
    projection of the shoulders can invent a rep.

    SCOPE. This changes the openness UNIT, so ``min_height=0.6`` / ``rel_drop=0.4`` /
    ``refractory_s=0.4`` are no longer calibrated and nothing here re-tunes them. These tests pin the
    MECHANISM (which length the scalar is expressed in), never an accuracy claim.
    """

    def _detector(self, frames, **knobs):
        return PoseOpenness(provider=_ScriptedPose(frames), **knobs)

    def _openness(self, pose):
        d = self._detector([[pose]])
        d.push(0.0, _DecodedPixels())
        return d._last

    def test_the_normalizer_is_the_torso_length_and_the_shoulder_separation_is_inert(self):
        # Same body, same stance, three shoulder separations including an exact zero: one number.
        # The `or 0.1` guard used to make the zero case a whole separate code path (0.2 / 0.1 = 2.0).
        for shoulder_width in (0.3, 0.6, 0.0):
            with self.subTest(shoulder_width=shoulder_width):
                self.assertAlmostEqual(
                    self._openness(_pose(spread=0.2, torso=0.25, shoulder_width=shoulder_width)),
                    0.2 / 0.25,
                )

    def test_shoulders_projected_onto_one_x_no_longer_explode_the_openness_scalar(self):
        # The SAME body seen lying down: the two shoulders land 1e-9 apart in x, which the retired
        # falsiness guard read as a real measurement and divided straight through. The Euclidean
        # torso is unmoved by the rotation, so the openness is unmoved too.
        upright = _pose(spread=0.2, torso=0.25, shoulder_width=0.3)
        lying = _pose(spread=0.2, torso=0.25, shoulder_width=1e-9, lying=True)
        flat = _pose(spread=0.2, torso=0.25, shoulder_width=0.0, lying=True)
        self.assertAlmostEqual(self._openness(upright), 0.8)
        self.assertAlmostEqual(self._openness(lying), self._openness(upright))
        self.assertAlmostEqual(self._openness(flat), self._openness(upright))
        # What the retired arithmetic would have produced from those very keypoints, for the record.
        # The near-zero separation divided through to ~2e8 while the EXACT zero took the guard and
        # gave 2.0 — eight orders of magnitude apart, for the same body in the same stance.
        def _retired(pose):
            lm = pose.keypoints
            return 0.2 / (abs(lm[COCO_LEFT_SHOULDER].x - lm[COCO_RIGHT_SHOULDER].x) or 0.1)

        self.assertGreater(_retired(lying), 1e8)
        self.assertAlmostEqual(_retired(flat), 2.0)

    def test_a_closed_stance_is_never_a_rep_whatever_the_shoulders_project_to(self):
        # The detector's claim is one clean peak per rep of an open/close body shape. A stance that
        # never opens (ankle spread 0.02) has no rep in it, and no projection of the shoulders may
        # invent one. The degenerate middle frame used to mint a peak worth >1e6 off a single frame.
        for projection in (False, True):
            with self.subTest(lying=projection):
                d = self._detector([
                    [_pose(spread=0.0, torso=0.25, shoulder_width=0.3)],
                    [_pose(spread=0.02, torso=0.25, shoulder_width=1e-9, lying=projection)],
                    [_pose(spread=0.0, torso=0.25, shoulder_width=0.3)],
                ])
                self.assertEqual(_drive_pose(d, 3), [])

    def test_a_real_open_close_cycle_still_counts_at_the_shipped_knobs(self):
        # The other half of the claim: silencing the degenerate frame must not silence a true rep.
        # Spread 0.5 over a 0.25 torso is openness 2.0, well clear of min_height=0.6.
        d = self._detector([
            [_pose(spread=s, torso=0.25)] for s in (0.0, 0.5, 0.0, 0.5, 0.0)
        ])
        peaks = _drive_pose(d, 5)
        self.assertEqual([b.t for b in peaks], [1.0, 3.0])
        self.assertTrue(all(b.detector == "pose-openness" and b.kind == "peak" for b in peaks))

    def test_no_pose_on_the_first_frame_measures_zero_rather_than_skipping_the_frame(self):
        d = self._detector([[]])
        self.assertEqual(d.push(0.0, _DecodedPixels()), [])
        self.assertEqual(d._last, 0.0)

    def test_a_dropped_detection_freezes_the_last_openness_instead_of_dipping(self):
        # The eye going blind for a frame is indistinguishable downstream from the body holding
        # still: the previous scalar is re-pushed, so the rep counter reads a plateau, not a gap.
        d = self._detector([[_pose(spread=0.2, torso=0.4)], [], [_pose(spread=0.2, torso=0.4)]])
        d.push(0.0, _DecodedPixels())
        held = d._last
        d.push(1.0, _DecodedPixels())
        self.assertEqual(d._last, held)
        self.assertGreater(held, 0.0)

    def test_the_most_confident_pose_wins_and_an_exact_tie_takes_the_first(self):
        crowd = [_pose(spread=0.2, torso=0.4, confidence=0.5),
                 _pose(spread=0.6, torso=0.4, confidence=0.9)]
        d = self._detector([crowd])
        d.push(0.0, _DecodedPixels())
        self.assertAlmostEqual(d._last, 0.6 / 0.4)  # the spread in TORSO-lengths, not shoulder widths
        tied = [_pose(spread=0.2, torso=0.4, confidence=0.7),
                _pose(spread=0.6, torso=0.4, confidence=0.7)]
        d = self._detector([tied])
        d.push(0.0, _DecodedPixels())
        self.assertAlmostEqual(d._last, 0.2 / 0.4)  # max() keeps the first maximal element

    def test_a_skeleton_shorter_than_coco_seventeen_is_refused_at_the_dto_boundary(self):
        # The six COCO reads in this detector — and the three in `pose.py` the rhythm path uses —
        # were unguarded, so a third-party eye with a shorter skeleton raised a bare IndexError from
        # whichever index reached past the end first. The arity check now lives once, at the DTO.
        with self.assertRaises(PoseSkeletonIncomplete) as caught:
            PoseDetection(tuple(PoseKeypoint(0.5, 0.5) for _ in range(16)), confidence=1.0)
        self.assertIsInstance(caught.exception, ValueError)
        self.assertEqual(caught.exception.got, 16)

    def test_a_provider_that_emits_a_short_skeleton_drops_the_frame_instead_of_ending_the_run(self):
        # That raise used to escape `DetectorBank.push` into the frame loop, ending the run for
        # EVERY goal in the bank rather than only the one that asked for pose. It is typed now, and
        # caught here as exactly what it is: a detection the eye could not read, so the last
        # openness carries forward — this detector's documented dropped-detection behaviour.
        class _ShortSkeletonPose:
            name = "short"

            def detect(self, pixels):
                return [PoseDetection(tuple(PoseKeypoint(0.5, 0.5) for _ in range(16)))]

        d = self._detector([[_pose(spread=0.2, torso=0.25)]])
        d.push(0.0, _DecodedPixels())
        held = d._last
        d._provider = _ShortSkeletonPose()
        self.assertEqual(d.push(1.0, _DecodedPixels()), [])
        self.assertEqual(d._last, held)


class TestDetectorBankRouting(unittest.TestCase):
    """Channel routing, including the else-branch that hands an unknown channel the motion signal.

    Every test here mutates the process-global detector registry, so the whole registry is snapshot
    and restored per test — an unrestored registration changes the default detector set for every
    later test module in the session.
    """

    def setUp(self):
        self._registry = dict(_detectors_module._REGISTRY)

    def tearDown(self):
        _detectors_module._REGISTRY.clear()
        _detectors_module._REGISTRY.update(self._registry)

    def _spy(self, name, signal):
        """Register a detector that records exactly what channel value the bank routed to it."""
        recorded = []

        class _Spy:
            def reset(self):
                recorded.clear()

            def push(self, t, v):
                recorded.append((t, v))
                return []

        if signal is not None:
            _Spy.signal = signal
        register_detector(name, _Spy, senses="s", when_to_use="w", output="o")
        return recorded

    def test_a_missing_motion_channel_skips_a_motion_detector(self):
        seen = self._spy("spy-motion", "motion")
        bank = DetectorBank(["spy-motion"])
        bank.push(1.0, motion=None, energy=0.9)
        self.assertEqual(seen, [])
        bank.push(2.0, motion=0.0)  # 0.0 is a measurement, not an absence
        self.assertEqual(seen, [(2.0, 0.0)])

    def test_a_missing_energy_channel_skips_an_energy_detector(self):
        seen = self._spy("spy-energy", "energy")
        bank = DetectorBank(["spy-energy"])
        bank.push(1.0, motion=0.7, energy=None)
        self.assertEqual(seen, [])
        bank.push(2.0, energy=0.0)
        self.assertEqual(seen, [(2.0, 0.0)])

    def test_a_frame_detector_without_a_frame_is_skipped_and_never_loads_a_provider(self):
        provider = _ScriptedPose([[_pose(spread=0.2, torso=0.4)]])
        detector = PoseOpenness(provider=provider)
        register_detector("spy-frame", lambda: detector, senses="s", when_to_use="w", output="o")
        bank = DetectorBank(["spy-frame"])
        self.assertEqual(bank.push(1.0, motion=0.9, energy=0.9), [])
        self.assertEqual(provider.calls, 0)

    def test_an_empty_selection_is_a_silent_bank(self):
        bank = DetectorBank([])
        bank.reset()
        self.assertEqual(bank.push(1.0, motion=0.9, energy=0.9, frame=_DecodedPixels()), [])

    def test_a_detector_with_no_signal_attribute_defaults_to_the_motion_channel(self):
        seen = self._spy("spy-bare", None)
        DetectorBank(["spy-bare"]).push(1.0, motion=0.25, energy=0.9)
        self.assertEqual(seen, [(1.0, 0.25)])

    def test_an_unknown_signal_string_receives_no_input_at_all(self):
        # The channel vocabulary used to be two recognised names plus an else-branch, so a detector
        # declaring an unrecognised channel was fed frame-diff magnitude — and then emitted
        # boundaries the brain counted as coming from the sense it declared. A thermal detector
        # reading motion, with nothing raised and nothing logged. It now gets nothing, on every tick
        # and whatever else is present.
        seen = self._spy("spy-thermal", "thermal")
        bank = DetectorBank(["spy-thermal"])
        bank.push(1.0, motion=0.4, energy=0.9, frame=_DecodedPixels())
        bank.push(2.0, energy=0.9)
        self.assertEqual(seen, [])

    def test_a_typo_in_the_signal_string_is_a_silent_channel_not_a_silent_substitution(self):
        # The likelier form of the same defect: "enrgy" used to route to motion, so a mis-typed
        # audio detector kept emitting — off the wrong signal. Now it simply never fires, which is
        # a visible symptom instead of an invisible one.
        seen = self._spy("spy-enrgy", "enrgy")
        DetectorBank(["spy-enrgy"]).push(1.0, motion=0.4, energy=0.9)
        self.assertEqual(seen, [])

    def test_a_detector_whose_channel_is_not_provided_receives_no_input(self):
        # The bank's own contract: "a detector whose channel isn't provided this tick simply gets
        # no input". The bank has no thermal channel at all, so that must mean no input, ever.
        seen = self._spy("spy-thermal", "thermal")
        DetectorBank(["spy-thermal"]).push(1.0, motion=0.4, energy=0.9)
        self.assertEqual(seen, [])

    def test_the_bank_channel_vocabulary_is_narrower_than_the_manifest_category_vocabulary(self):
        # Worth stating because it is a real gap rather than a rounding error: `DetectorManifest.
        # category` is an explicit EIGHT-name sense class, while the bank carries three channels.
        # Four categories have no bank channel at all, so a detector can be discoverable as a class
        # the bank cannot feed. Reconciling the two vocabularies is a contract change; quietly
        # routing those detectors to motion is what this fix stopped doing.
        self.assertEqual(DetectorBank.CHANNELS, ("motion", "energy", "frame"))
        for category in (CATEGORY_PRESENCE, CATEGORY_CHANGE,
                         CATEGORY_SEMANTIC_RELEVANCE, CATEGORY_AUDIO_EVENT):
            with self.subTest(category=category):
                self.assertNotIn(category, DetectorBank.CHANNELS)
                seen = self._spy(f"spy-{category}", category)
                DetectorBank([f"spy-{category}"]).push(1.0, motion=0.4, energy=0.9)
                self.assertEqual(seen, [])

    def test_a_short_skeleton_from_a_frame_detector_no_longer_ends_the_run_for_the_whole_bank(self):
        # The arity raise used to escape `DetectorBank.push` into the frame loop. Pinned end to end:
        # the pose detector goes quiet on the unreadable frame and the motion detector in the SAME
        # bank keeps counting through it.
        class _ShortSkeletonPose:
            name = "short"

            def detect(self, pixels):
                return [PoseDetection(tuple(PoseKeypoint(0.5, 0.5) for _ in range(16)))]

        detector = PoseOpenness(provider=_ShortSkeletonPose())
        register_detector("spy-short-pose", lambda: detector, senses="s", when_to_use="w", output="o")
        bank = DetectorBank(["spy-short-pose", "motion-periodicity"])
        out = []
        for t, v in enumerate([0.0, 0.5, 1.0, 0.5, 0.0]):
            out.extend(bank.push(float(t), motion=v, frame=_DecodedPixels()))
        self.assertTrue(out and all(b.detector == "motion-periodicity" for b in out))

    def test_a_non_numeric_signal_raises_out_of_the_bank(self):
        self._spy("spy-numeric", "motion")
        with self.assertRaises(ValueError):
            DetectorBank(["spy-numeric"]).push(1.0, motion="loud")

    def test_an_unregistered_name_raises_at_bank_construction(self):
        with self.assertRaises(KeyError):
            DetectorBank(["motion-periodicity", "telepathy"])


if __name__ == "__main__":
    unittest.main()
