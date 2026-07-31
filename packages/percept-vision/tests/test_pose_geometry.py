"""The five pose geometry helpers, measured directly instead of through ``PoseChannels``.

``percept/harness/pose.py`` publishes a small geometry layer that everything frame-backed stands on:
``body_scale`` (the per-pose length unit), ``pose_channels`` / ``channel_coverage`` (the measurement
basis and its visibility gate), ``keypoints_bbox`` / ``bbox_xywh`` (the crop the re-id path reads),
and ``hip_centre`` / ``select_nearest_hip`` (the single-subject pick). Until this file they had **no
direct test** — only indirect exercise through the detector, which cannot reach any of the boundaries
below because it never authors a degenerate pose.

Three things here are load-bearing rather than incidental:

* **``body_scale`` is Euclidean on purpose.** Its docstring names the regression: a vertical ``|dy|``
  torso length COLLAPSES toward zero for LYING/horizontal poses (bench press, push-ups, sit-ups), so
  it hits ``_MIN_BODY_SCALE`` and every channel divided by it EXPLODES. ``LyingPoseTests`` measures a
  horizontal pose and asserts both halves of that: the true separation is returned, *and* the
  vertical-only surrogate would have been floored.
* **The floor is a clamp, not a repair.** Below ``_MIN_BODY_SCALE`` the channels are normalized by the
  floor rather than by the body, so their amplitudes stop being a property of the subject. That is the
  shipped behaviour and it is pinned here as such — a distant subject's channels are *not* comparable
  to a near one's. What changed is that the clamp is now OBSERVABLE: ``body_scale_measured`` returns
  ``(scale, floored)``, so a consumer applying an absolute amplitude gate can tell a measured 0.05
  torso from a clamped 0.015 one. The floor itself is unchanged and must stay — without it a collapsed
  torso makes every channel explode.
* **The arity check lives at the DTO boundary.** ``PoseDetection`` refuses a skeleton shorter than
  COCO-17 (``PoseSkeletonIncomplete``), so every consumer is protected by construction rather than by
  a guard at each of the six index sites — a guard there would have had to substitute a joint, which
  silently changes what the eye measures. The geometry helpers below still index a RAW tuple with no
  length check, and that is pinned too: the protection is the DTO, not the helpers.

Every expectation is hand-computed from binary-exact coordinates (eighths, sixteenths, sixty-fourths)
so the arithmetic can be asserted exactly rather than approximately.
"""

from __future__ import annotations

import os
import unittest
from unittest import mock

from percept.harness.pose import (
    CHANNEL_KEYPOINTS,
    COCO_LEFT_ANKLE,
    COCO_LEFT_HIP,
    COCO_LEFT_SHOULDER,
    COCO_LEFT_WRIST,
    COCO_NOSE,
    COCO_RIGHT_ANKLE,
    COCO_RIGHT_HIP,
    COCO_RIGHT_SHOULDER,
    COCO_RIGHT_WRIST,
    POSE_CHANNEL_NAMES,
    PoseDetection,
    PoseKeypoint,
    PoseSkeletonIncomplete,
    _MIN_BODY_SCALE,
    bbox_xywh,
    body_scale,
    body_scale_measured,
    channel_coverage,
    hip_centre,
    keypoints_bbox,
    pose_channels,
    select_nearest_hip,
)

#: Every keypoint the caller does not name sits here. Deliberately one shared point: an untouched
#: default is itself the "all keypoints collapsed" degenerate pose.
_DEFAULT_KEYPOINT = (0.5, 0.5, 1.0)


def make_keypoints(
    overrides: dict[int, tuple[float, float, float]] | None = None,
    *,
    count: int = 17,
    default: tuple[float, float, float] = _DEFAULT_KEYPOINT,
) -> tuple[PoseKeypoint, ...]:
    """A COCO-17 keypoint tuple: ``default`` everywhere, ``{index: (x, y, conf)}`` on top.

    ``count`` exists only so the <17-keypoint provider boundary can be authored; nothing shipped
    should ever emit a tuple of another length.
    """
    given = overrides or {}
    out = []
    for i in range(count):
        x, y, conf = given.get(i, default)
        out.append(PoseKeypoint(float(x), float(y), float(conf)))
    return tuple(out)


def _upright(torso: float, *, shoulder_y: float = 0.25) -> tuple[PoseKeypoint, ...]:
    """A vertical pose whose shoulder-to-hip distance is exactly ``torso`` on both sides."""
    return make_keypoints({
        COCO_LEFT_SHOULDER: (0.375, shoulder_y, 1.0),
        COCO_RIGHT_SHOULDER: (0.625, shoulder_y, 1.0),
        COCO_LEFT_HIP: (0.375, shoulder_y + torso, 1.0),
        COCO_RIGHT_HIP: (0.625, shoulder_y + torso, 1.0),
    })


def _detection(confidence: float, hip: tuple[float, float]) -> PoseDetection:
    """A detection whose hip centre is exactly ``hip`` (both hips on the same point)."""
    x, y = hip
    return PoseDetection(
        keypoints=make_keypoints({COCO_LEFT_HIP: (x, y, 1.0), COCO_RIGHT_HIP: (x, y, 1.0)}),
        confidence=confidence,
    )


class BodyScaleFloorTests(unittest.TestCase):
    """The clamp at ``_MIN_BODY_SCALE``, bracketed on both sides of the boundary."""

    def test_shipped_floor_is_five_hundredths_of_a_frame(self):
        # Module-private and unregistered: no config surface guards this value, so the test does.
        self.assertEqual(_MIN_BODY_SCALE, 0.05)

    def test_torso_exactly_at_the_floor_returns_the_floor_value_itself(self):
        # max(0.05, 0.05) -- the boundary is inclusive from both directions and must not drift.
        self.assertEqual(body_scale(_upright(0.05, shoulder_y=0.0)), 0.05)

    def test_torso_just_above_the_floor_is_the_true_measurement_not_the_floor(self):
        self.assertEqual(body_scale(_upright(0.0625, shoulder_y=0.0)), 0.0625)

    def test_torso_below_the_floor_is_clamped_up(self):
        self.assertEqual(body_scale(_upright(0.03125, shoulder_y=0.0)), 0.05)

    def test_all_keypoints_at_one_point_returns_the_floor_not_zero(self):
        collapsed = make_keypoints()  # every joint on _DEFAULT_KEYPOINT
        self.assertEqual({(k.x, k.y) for k in collapsed}, {(0.5, 0.5)})
        self.assertEqual(body_scale(collapsed), 0.05)

    def test_below_the_floor_every_channel_is_normalized_by_the_floor_not_the_body(self):
        """The pathology the floor buys: a distant subject's amplitudes stop being its own.

        Pinned as shipped behaviour, not endorsed -- two subjects at different distances produce
        channel amplitudes that are NOT comparable once either one is under the floor.
        """
        tiny = _upright(0.015625, shoulder_y=0.5)  # a true torso of 1/64 of the frame
        self.assertEqual(body_scale(tiny), 0.05)
        channels = pose_channels(tiny)
        # hip y is 0.515625; divided by the FLOOR (0.05), not by the real torso (0.015625).
        self.assertAlmostEqual(channels["hip-vertical"], 0.515625 / 0.05, places=12)
        # Had it been normalized by the body, it would have been ~3x larger -- the amplitude the
        # absolute `expected_rom` gate would have seen.
        self.assertNotAlmostEqual(channels["hip-vertical"], 0.515625 / 0.015625, places=6)


class BodyScaleObservabilityTests(unittest.TestCase):
    """``body_scale_measured`` -> ``(scale, floored)``: which branch of the clamp won.

    ``body_scale`` alone cannot answer that -- a returned 0.05 is either a real 0.05 torso or a
    clamped 0.015 one, and the two mean opposite things to a consumer applying an ABSOLUTE amplitude
    gate in these units. The contract this exists to serve: *either the clamp is observable to the
    consumer, or the absolute gate is not applied when the normalizer is not a real measurement.*
    """

    def test_a_real_measurement_above_the_floor_is_not_flagged(self):
        self.assertEqual(body_scale_measured(_upright(0.0625, shoulder_y=0.0)), (0.0625, False))

    def test_a_torso_below_the_floor_returns_the_floor_and_says_so(self):
        self.assertEqual(body_scale_measured(_upright(0.015625, shoulder_y=0.0)), (0.05, True))

    def test_a_torso_exactly_at_the_floor_is_a_measurement_not_a_clamp(self):
        # The boundary belongs to the measurement: `max()` returns the same number either way, but
        # 0.05 measured and 0.05 clamped are different facts and only one of them invalidates a gate.
        self.assertEqual(body_scale_measured(_upright(0.05, shoulder_y=0.0)), (0.05, False))

    def test_a_pose_collapsed_to_one_point_is_the_floor_and_is_flagged(self):
        self.assertEqual(body_scale_measured(make_keypoints()), (0.05, True))

    def test_the_two_indistinguishable_scales_are_now_distinguishable(self):
        # The whole defect in one assertion: same returned scale, opposite flags.
        measured, clamped = _upright(0.05, shoulder_y=0.0), _upright(0.015625, shoulder_y=0.0)
        self.assertEqual(body_scale(measured), body_scale(clamped))
        self.assertNotEqual(body_scale_measured(measured)[1], body_scale_measured(clamped)[1])

    def test_body_scale_is_exactly_the_first_element_of_the_measured_pair(self):
        # The wrapper must stay a wrapper: 40+ call sites and their tests read the scalar form.
        for torso in (0.5, 0.0625, 0.05, 0.015625, 0.0):
            with self.subTest(torso=torso):
                pose = _upright(torso, shoulder_y=0.0)
                self.assertEqual(body_scale(pose), body_scale_measured(pose)[0])


class LyingPoseTests(unittest.TestCase):
    """The case Euclidean was chosen for: shoulders and hips share a y, separated only in x."""

    def _lying(self) -> tuple[PoseKeypoint, ...]:
        # Head-to-the-right, feet-to-the-left: shoulders at x=0.75, hips at x=0.25, each side flat.
        return make_keypoints({
            COCO_LEFT_SHOULDER: (0.75, 0.5, 1.0),
            COCO_LEFT_HIP: (0.25, 0.5, 1.0),
            COCO_RIGHT_SHOULDER: (0.75, 0.5625, 1.0),
            COCO_RIGHT_HIP: (0.25, 0.5625, 1.0),
        })

    def test_horizontal_torso_returns_its_true_separation(self):
        self.assertEqual(body_scale(self._lying()), 0.5)

    def test_a_vertical_only_normalizer_would_have_collapsed_to_the_floor(self):
        """The regression itself, made explicit: |dy| is zero for this pose."""
        k = self._lying()
        vertical_only = 0.5 * (
            abs(k[COCO_LEFT_HIP].y - k[COCO_LEFT_SHOULDER].y)
            + abs(k[COCO_RIGHT_HIP].y - k[COCO_RIGHT_SHOULDER].y)
        )
        self.assertEqual(vertical_only, 0.0)
        self.assertLess(vertical_only, _MIN_BODY_SCALE)
        # 0.5 vs a floored 0.05: every channel would have read 10x its true amplitude.
        self.assertEqual(body_scale(self._lying()) / max(vertical_only, _MIN_BODY_SCALE), 10.0)

    def test_asymmetric_left_right_torso_returns_the_mean_of_the_two_sides(self):
        asymmetric = make_keypoints({
            COCO_LEFT_SHOULDER: (0.25, 0.0, 1.0),
            COCO_LEFT_HIP: (0.25, 0.10, 1.0),
            COCO_RIGHT_SHOULDER: (0.75, 0.0, 1.0),
            COCO_RIGHT_HIP: (0.75, 0.30, 1.0),
        })
        self.assertEqual(body_scale(asymmetric), 0.2)


class PoseChannelBasisTests(unittest.TestCase):
    """Each channel is a raw measurement divided by ``body_scale`` -- asserted arithmetically."""

    def _pose(self) -> tuple[PoseKeypoint, ...]:
        # Torso = 0.75 - 0.25 = 0.5 on both sides, so body_scale is exactly 0.5.
        return make_keypoints({
            COCO_LEFT_SHOULDER: (0.375, 0.25, 1.0),
            COCO_RIGHT_SHOULDER: (0.625, 0.25, 1.0),
            COCO_LEFT_HIP: (0.375, 0.75, 1.0),
            COCO_RIGHT_HIP: (0.625, 0.75, 1.0),
            COCO_LEFT_WRIST: (0.25, 0.125, 1.0),
            COCO_RIGHT_WRIST: (0.75, 0.0625, 1.0),
            COCO_LEFT_ANKLE: (0.25, 0.9375, 1.0),
            COCO_RIGHT_ANKLE: (0.625, 0.9375, 1.0),
        })

    def test_body_scale_of_the_fixture_is_exactly_a_half(self):
        self.assertEqual(body_scale(self._pose()), 0.5)

    def test_every_channel_is_its_raw_measure_divided_by_torso_length(self):
        channels = pose_channels(self._pose())
        # limb-vertical: mean(shoulder.y - wrist.y) = ((0.25-0.125) + (0.25-0.0625)) / 2 = 0.15625
        self.assertEqual(channels["limb-vertical"], 0.15625 / 0.5)
        self.assertEqual(channels["limb-vertical"], 0.3125)
        # ankle-spread: |0.25 - 0.625| = 0.375
        self.assertEqual(channels["ankle-spread"], 0.375 / 0.5)
        self.assertEqual(channels["ankle-spread"], 0.75)
        # hip-vertical: mean hip y = 0.75
        self.assertEqual(channels["hip-vertical"], 0.75 / 0.5)
        self.assertEqual(channels["hip-vertical"], 1.5)
        # torso-vertical: mean shoulder y = 0.25
        self.assertEqual(channels["torso-vertical"], 0.25 / 0.5)
        self.assertEqual(channels["torso-vertical"], 0.5)

    def test_ankle_spread_is_unsigned_so_crossed_ankles_read_the_same(self):
        left_first = pose_channels(self._pose())
        crossed = make_keypoints({
            COCO_LEFT_SHOULDER: (0.375, 0.25, 1.0),
            COCO_RIGHT_SHOULDER: (0.625, 0.25, 1.0),
            COCO_LEFT_HIP: (0.375, 0.75, 1.0),
            COCO_RIGHT_HIP: (0.625, 0.75, 1.0),
            COCO_LEFT_WRIST: (0.25, 0.125, 1.0),
            COCO_RIGHT_WRIST: (0.75, 0.0625, 1.0),
            COCO_LEFT_ANKLE: (0.625, 0.9375, 1.0),  # swapped with the right ankle
            COCO_RIGHT_ANKLE: (0.25, 0.9375, 1.0),
        })
        self.assertEqual(pose_channels(crossed)["ankle-spread"], left_first["ankle-spread"])

    def test_default_basis_is_exactly_the_four_published_names(self):
        self.assertEqual(tuple(pose_channels(self._pose())), POSE_CHANNEL_NAMES)
        self.assertEqual(set(CHANNEL_KEYPOINTS), set(POSE_CHANNEL_NAMES))

    def test_env_restricts_the_returned_keys_to_the_named_subset(self):
        with mock.patch.dict(os.environ, {"PERCEPT_POSE_CHANNELS": "hip-vertical, ankle-spread"}):
            channels = pose_channels(self._pose())
        self.assertEqual(set(channels), {"hip-vertical", "ankle-spread"})
        # The surviving values are unchanged by the restriction -- it is a filter, not a re-scaling.
        self.assertEqual(channels["hip-vertical"], 1.5)
        self.assertEqual(channels["ankle-spread"], 0.75)

    def test_env_selection_matching_nothing_falls_back_to_all_four(self):
        with mock.patch.dict(os.environ, {"PERCEPT_POSE_CHANNELS": "torso-openness"}):
            channels = pose_channels(self._pose())
        self.assertEqual(tuple(channels), POSE_CHANNEL_NAMES)

    def test_env_of_only_separators_falls_back_to_all_four(self):
        with mock.patch.dict(os.environ, {"PERCEPT_POSE_CHANNELS": " , , "}):
            channels = pose_channels(self._pose())
        self.assertEqual(tuple(channels), POSE_CHANNEL_NAMES)

    def test_env_is_read_per_call_so_it_does_not_leak_past_its_scope(self):
        with mock.patch.dict(os.environ, {"PERCEPT_POSE_CHANNELS": "hip-vertical"}):
            self.assertEqual(set(pose_channels(self._pose())), {"hip-vertical"})
        self.assertEqual(tuple(pose_channels(self._pose())), POSE_CHANNEL_NAMES)


class ChannelCoverageTests(unittest.TestCase):
    """Per-channel visibility: the mean confidence of the joints the channel is measured from."""

    def _pose(self) -> tuple[PoseKeypoint, ...]:
        return make_keypoints({
            COCO_LEFT_WRIST: (0.25, 0.125, 0.125),
            COCO_RIGHT_WRIST: (0.75, 0.125, 0.375),
            COCO_LEFT_SHOULDER: (0.375, 0.25, 0.5),
            COCO_RIGHT_SHOULDER: (0.625, 0.25, 0.75),
            COCO_LEFT_ANKLE: (0.25, 0.9375, 0.25),
            COCO_RIGHT_ANKLE: (0.75, 0.9375, 0.75),
            COCO_LEFT_HIP: (0.375, 0.75, 0.625),
            COCO_RIGHT_HIP: (0.625, 0.75, 0.875),
        })

    def test_each_channel_reports_the_mean_confidence_of_its_own_keypoints(self):
        coverage = channel_coverage(self._pose())
        # wrists + shoulders: (0.125 + 0.375 + 0.5 + 0.75) / 4
        self.assertEqual(coverage["limb-vertical"], 0.4375)
        # ankles: (0.25 + 0.75) / 2
        self.assertEqual(coverage["ankle-spread"], 0.5)
        # hips: (0.625 + 0.875) / 2
        self.assertEqual(coverage["hip-vertical"], 0.75)
        # shoulders only: (0.5 + 0.75) / 2
        self.assertEqual(coverage["torso-vertical"], 0.625)

    def test_coverage_covers_every_channel_the_basis_emits(self):
        pose = self._pose()
        self.assertEqual(set(channel_coverage(pose)), set(pose_channels(pose)))

    def test_a_fully_occluded_channel_reports_zero_rather_than_dropping_out(self):
        # The pommel-horse case: legs at confidence 0. The channel must still be reported so the
        # brain can route away from it; a missing key would read as "no such channel".
        occluded = make_keypoints({
            COCO_LEFT_ANKLE: (0.25, 0.9375, 0.0),
            COCO_RIGHT_ANKLE: (0.75, 0.9375, 0.0),
        })
        coverage = channel_coverage(occluded)
        self.assertIn("ankle-spread", coverage)
        self.assertEqual(coverage["ankle-spread"], 0.0)

    def test_coverage_ignores_position_entirely(self):
        near = make_keypoints({COCO_LEFT_HIP: (0.0, 0.0, 0.5), COCO_RIGHT_HIP: (0.0, 0.0, 0.25)})
        far = make_keypoints({COCO_LEFT_HIP: (0.9, 0.9, 0.5), COCO_RIGHT_HIP: (0.1, 0.1, 0.25)})
        self.assertEqual(channel_coverage(near)["hip-vertical"], channel_coverage(far)["hip-vertical"])


class ShortKeypointListTests(unittest.TestCase):
    """The <17-keypoint provider: refused at the DTO, still unguarded in the raw-tuple helpers.

    The six ``keypoints[COCO_*]`` sites in ``detectors.py`` and the three here read fixed indices with
    no length check, so a short skeleton used to surface as a bare ``IndexError`` from whichever site
    reached past the end first -- out of ``DetectorBank.push``, ending the run for every goal in the
    bank. The check now happens ONCE, where the detection is built, so every consumer is protected by
    construction. The helpers themselves are unchanged, and that is deliberate: they take a raw tuple,
    a guard there would have to invent a substitute joint, and inventing one silently changes what the
    eye measures.
    """

    def _short_torso(self, count: int = 16) -> tuple[PoseKeypoint, ...]:
        return make_keypoints(
            {
                COCO_LEFT_SHOULDER: (0.375, 0.25, 1.0),
                COCO_RIGHT_SHOULDER: (0.625, 0.25, 1.0),
                COCO_LEFT_HIP: (0.375, 0.75, 1.0),
                COCO_RIGHT_HIP: (0.625, 0.75, 1.0),
            },
            count=count,
        )

    def test_a_pose_detection_refuses_a_short_skeleton_at_construction(self):
        with self.assertRaises(PoseSkeletonIncomplete) as caught:
            PoseDetection(keypoints=self._short_torso(), confidence=1.0, provider="custom")
        self.assertIsInstance(caught.exception, ValueError)  # catchable as the stdlib type too
        self.assertEqual((caught.exception.got, caught.exception.provider), (16, "custom"))

    def test_exactly_seventeen_is_accepted_and_the_boundary_is_inclusive(self):
        self.assertEqual(len(PoseDetection(keypoints=make_keypoints()).keypoints), 17)
        with self.assertRaises(PoseSkeletonIncomplete):
            PoseDetection(keypoints=make_keypoints(count=16))
        with self.assertRaises(PoseSkeletonIncomplete):
            PoseDetection(keypoints=())

    def test_a_sixteen_keypoint_tuple_still_measures_body_scale(self):
        # body_scale only reads indices 5, 6, 11, 12 -- the raise was never at the entry point,
        # which is exactly why a per-site guard could not be the fix.
        short = self._short_torso()
        self.assertEqual(len(short), 16)
        self.assertEqual(body_scale(short), 0.5)
        self.assertEqual(body_scale_measured(short), (0.5, False))

    def test_pose_channels_raises_index_error_on_a_sixteen_keypoint_tuple(self):
        # COCO_RIGHT_ANKLE is index 16 -- one past the end of a 16-keypoint provider's output.
        # Reachable only by handing the helper a raw tuple; no PoseDetection can carry one.
        self.assertEqual(COCO_RIGHT_ANKLE, 16)
        with self.assertRaises(IndexError):
            pose_channels(make_keypoints(count=16))

    def test_channel_coverage_raises_index_error_on_a_sixteen_keypoint_tuple(self):
        with self.assertRaises(IndexError):
            channel_coverage(make_keypoints(count=16))

    def test_hip_centre_only_ever_sees_a_full_skeleton_because_it_takes_a_detection(self):
        # hip_centre reads indices 11 and 12, so a 16-point tuple would have satisfied it -- but it
        # takes a PoseDetection, and the DTO no longer lets one that short exist.
        detection = PoseDetection(
            keypoints=make_keypoints({COCO_LEFT_HIP: (0.25, 0.5, 1.0), COCO_RIGHT_HIP: (0.75, 0.5, 1.0)}),
            confidence=1.0,
        )
        self.assertEqual(hip_centre(detection), (0.5, 0.5))


class KeypointsBboxTests(unittest.TestCase):
    """The provider-less fallback crop: ``(min_x, min_y, max_x - min_x, max_y - min_y)``."""

    def test_bbox_is_the_min_corner_plus_the_extent(self):
        pose = make_keypoints({
            COCO_NOSE: (0.5, 0.125, 1.0),
            COCO_LEFT_WRIST: (0.125, 0.375, 1.0),
            COCO_RIGHT_WRIST: (0.875, 0.375, 1.0),
            COCO_LEFT_ANKLE: (0.375, 0.875, 1.0),
            COCO_RIGHT_ANKLE: (0.625, 0.875, 1.0),
        })
        self.assertEqual(keypoints_bbox(pose), (0.125, 0.125, 0.75, 0.75))

    def test_a_single_point_pose_degenerates_to_zero_width_and_height(self):
        # Not clamped, not padded: a collapsed pose yields a zero-area crop, which any consumer
        # that slices pixels out of it must handle.
        self.assertEqual(keypoints_bbox(make_keypoints()), (0.5, 0.5, 0.0, 0.0))

    def test_an_empty_keypoint_tuple_raises_rather_than_returning_a_box(self):
        # min() over an empty sequence -- the same unguarded shape as the short-list case above.
        with self.assertRaises(ValueError):
            keypoints_bbox(())

    def test_low_confidence_keypoints_still_widen_the_box(self):
        # keypoints_bbox reads x/y only; a confidence-0 joint at the frame edge still enlarges it.
        pose = make_keypoints({COCO_LEFT_ANKLE: (0.0, 1.0, 0.0)})
        self.assertEqual(keypoints_bbox(pose), (0.0, 0.5, 0.5, 0.5))


class BboxXywhTests(unittest.TestCase):
    """``(x1, y1, x2, y2)`` from a provider, converted to ``(x, y, w, h)``."""

    def test_corner_pair_becomes_origin_plus_extent(self):
        self.assertEqual(bbox_xywh((0.25, 0.125, 0.75, 0.625)), (0.25, 0.125, 0.5, 0.5))

    def test_an_inverted_box_clamps_to_zero_extent_and_keeps_its_origin(self):
        # x2 < x1 and y2 < y1: the width/height clamp to 0.0 rather than going negative, which
        # would otherwise flip the sign of every downstream crop.
        self.assertEqual(bbox_xywh((0.75, 0.5, 0.25, 0.125)), (0.75, 0.5, 0.0, 0.0))

    def test_a_zero_area_box_is_preserved_exactly(self):
        self.assertEqual(bbox_xywh((0.5, 0.5, 0.5, 0.5)), (0.5, 0.5, 0.0, 0.0))

    def test_only_the_inverted_axis_is_clamped(self):
        self.assertEqual(bbox_xywh((0.25, 0.75, 0.75, 0.25)), (0.25, 0.75, 0.5, 0.0))


class HipCentreTests(unittest.TestCase):
    def test_hip_centre_is_the_midpoint_of_the_two_hips(self):
        detection = PoseDetection(
            keypoints=make_keypoints(
                {COCO_LEFT_HIP: (0.25, 0.5, 1.0), COCO_RIGHT_HIP: (0.75, 0.25, 1.0)}
            ),
            confidence=1.0,
        )
        self.assertEqual(hip_centre(detection), (0.5, 0.375))

    def test_hip_centre_ignores_every_other_joint(self):
        hips = {COCO_LEFT_HIP: (0.25, 0.5, 1.0), COCO_RIGHT_HIP: (0.75, 0.25, 1.0)}
        plain = PoseDetection(keypoints=make_keypoints(hips), confidence=1.0)
        arms_up = PoseDetection(
            keypoints=make_keypoints({**hips, COCO_LEFT_WRIST: (0.0, 0.0, 1.0)}), confidence=1.0
        )
        self.assertEqual(hip_centre(plain), hip_centre(arms_up))


class SelectNearestHipTests(unittest.TestCase):
    """The single-subject pick: continuity first, confidence only as the cold-start tie-break."""

    def test_none_when_every_detection_is_below_min_conf(self):
        poses = [_detection(0.1, (0.25, 0.25)), _detection(0.49, (0.75, 0.75))]
        self.assertIsNone(select_nearest_hip(poses, None, 0.5))

    def test_none_even_with_an_anchor_when_everything_is_below_min_conf(self):
        # The confidence filter runs BEFORE the anchor branch -- a nearby ghost cannot be adopted.
        poses = [_detection(0.1, (0.25, 0.25))]
        self.assertIsNone(select_nearest_hip(poses, (0.25, 0.25), 0.5))

    def test_none_on_an_empty_detection_list(self):
        self.assertIsNone(select_nearest_hip([], (0.5, 0.5), 0.5))

    def test_confidence_exactly_at_min_conf_is_kept(self):
        # The comparison is >=, so the boundary detection survives.
        exact = _detection(0.5, (0.25, 0.25))
        self.assertIs(select_nearest_hip([exact], None, 0.5), exact)

    def test_without_an_anchor_the_most_confident_detection_wins(self):
        weak = _detection(0.6, (0.1, 0.1))
        strong = _detection(0.9, (0.9, 0.9))
        self.assertIs(select_nearest_hip([weak, strong], None, 0.5), strong)
        self.assertIs(select_nearest_hip([strong, weak], None, 0.5), strong)

    def test_with_an_anchor_the_nearest_hip_centre_wins_over_the_most_confident(self):
        # Continuity is the point: the less-confident but closer subject is the same person.
        near_but_weaker = _detection(0.6, (0.3, 0.3))
        far_but_stronger = _detection(0.95, (0.9, 0.9))
        picked = select_nearest_hip([far_but_stronger, near_but_weaker], (0.25, 0.25), 0.5)
        self.assertIs(picked, near_but_weaker)

    def test_the_anchor_distance_is_measured_on_both_axes(self):
        # A detection aligned in x but far in y must lose to one that is close in both.
        same_x_far_y = _detection(0.9, (0.25, 0.95))
        diagonal_close = _detection(0.9, (0.35, 0.35))
        picked = select_nearest_hip([same_x_far_y, diagonal_close], (0.25, 0.25), 0.5)
        self.assertIs(picked, diagonal_close)

    def test_a_sub_threshold_detection_is_never_picked_even_when_it_is_the_nearest(self):
        ghost = _detection(0.2, (0.25, 0.25))
        real = _detection(0.8, (0.8, 0.8))
        self.assertIs(select_nearest_hip([ghost, real], (0.25, 0.25), 0.5), real)

    def test_equal_confidence_without_an_anchor_resolves_to_the_first_deterministically(self):
        # Determinism matters more than which one: a coin-flip here swaps the counted subject
        # between frames on the cold-start frame.
        first = _detection(0.8, (0.2, 0.2))
        second = _detection(0.8, (0.8, 0.8))
        self.assertIs(select_nearest_hip([first, second], None, 0.5), first)
        self.assertIs(select_nearest_hip([first, second], None, 0.5), first)


if __name__ == "__main__":  # pragma: no cover - convenience runner
    unittest.main()
