"""event_capsule -> agent plan ranking: an authored ``event_capsule`` perception spec must never be
silently floored/downgraded by an active ``WatchProfile.plan_mode``.

``event_capsule`` is a composite SUPERSET evidence shape (pre-roll + episode boundary + early
snapshot, see :mod:`percept.capsule`), never a subset of what a plain strip/window gives the judge.
This module pins the rank rule added to :data:`percept.agent._PLAN_RANK`:

  1. ``event_capsule`` sits ABOVE ``window`` in ``_PLAN_RANK`` — the highest rung, since it is a
     superset shape, not a point on the single/strip/window ladder.
  2. an AUTHORED ``event_capsule`` spec survives every one of the four profile plan_modes
     (single/strip/window/sprite) — the floor comparison never reinterprets it downward.
  3. a profile can never UPGRADE a cheaper spec TO capsule: ``event_capsule`` stays out of
     :data:`percept.watchprofile.PLAN_MODES` (the vocabulary a resolved ``WatchProfile.plan_mode`` is
     validated against), and :func:`percept.agent._planned_perception` refuses it explicitly as an
     escalation target even if a raw string bypasses that validation.
  4. every non-capsule goal's overlay result stays byte-for-byte identical to before this rank rung
     was added.
"""

from __future__ import annotations

import unittest

from percept.agent import _PLAN_RANK, _planned_perception
from percept.behavior import PerceptionSpec
from percept.watchprofile import PLAN_MODES


class EventCapsuleRankTests(unittest.TestCase):
    """``event_capsule`` occupies the top rung of ``_PLAN_RANK``, above ``window``."""

    def test_event_capsule_is_ranked_above_window(self):
        self.assertIn("event_capsule", _PLAN_RANK)
        self.assertGreater(_PLAN_RANK["event_capsule"], _PLAN_RANK["window"])

    def test_event_capsule_ranks_above_every_other_rung(self):
        top = _PLAN_RANK["event_capsule"]
        for mode, rank in _PLAN_RANK.items():
            if mode == "event_capsule":
                continue
            self.assertLess(rank, top, f"{mode!r} must rank below event_capsule")


class AuthoredCapsuleSurvivesEveryProfilePlanModeTests(unittest.TestCase):
    """An authored ``event_capsule`` spec is never floored/downgraded by ANY active plan_mode."""

    def _authored(self, **overrides) -> PerceptionSpec:
        return PerceptionSpec(temporal="event_capsule", **overrides)

    def test_survives_single(self):
        authored = self._authored()
        self.assertIs(_planned_perception(authored, "single"), authored)

    def test_survives_strip(self):
        authored = self._authored(n_frames=4)
        self.assertIs(_planned_perception(authored, "strip"), authored)

    def test_survives_window(self):
        authored = self._authored(n_frames=5)
        self.assertIs(_planned_perception(authored, "window"), authored)

    def test_survives_sprite(self):
        # sprite normalizes to window before ranking — capsule still outranks it.
        authored = self._authored()
        self.assertIs(_planned_perception(authored, "sprite"), authored)

    def test_survives_every_profile_plan_mode(self):
        # PLAN_MODES is the full vocabulary a resolved WatchProfile.plan_mode can carry.
        for mode in PLAN_MODES:
            authored = self._authored(n_frames=3)
            self.assertIs(
                _planned_perception(authored, mode),
                authored,
                f"event_capsule must survive plan_mode={mode!r}",
            )

    def test_authored_capsule_other_axes_untouched(self):
        authored = self._authored(modality="fused", roi="full", sampling="uniform", audio_s=2.0,
                                   window_s=4.0, n_frames=3)
        spec = _planned_perception(authored, "window")
        self.assertIs(spec, authored)
        self.assertEqual(
            (spec.modality, spec.roi, spec.sampling, spec.audio_s, spec.window_s, spec.n_frames),
            ("fused", "full", "uniform", 2.0, 4.0, 3),
        )


class ProfileNeverUpgradesToCapsuleTests(unittest.TestCase):
    """A profile plan_mode can never raise a cheaper spec UP TO ``event_capsule``."""

    def test_event_capsule_is_not_a_valid_profile_plan_mode(self):
        self.assertNotIn("event_capsule", PLAN_MODES)

    def test_raw_event_capsule_plan_mode_is_refused_as_a_target(self):
        # Defense-in-depth: even a caller that bypasses WatchProfile validation and passes the raw
        # string directly must not have a cheap spec escalated to capsule.
        base = PerceptionSpec()
        self.assertIs(_planned_perception(base, "event_capsule"), base)

        strip = PerceptionSpec(temporal="strip", n_frames=4)
        self.assertIs(_planned_perception(strip, "event_capsule"), strip)


class NoCapsuleGoalsByteForByteTests(unittest.TestCase):
    """Every non-capsule overlay result is unchanged by the new rank rung."""

    def test_none_plan_mode_still_byte_for_byte(self):
        base = PerceptionSpec()
        self.assertIs(_planned_perception(base, None), base)
        self.assertIs(_planned_perception(base, ""), base)

    def test_window_still_raises_a_single_frame_spec(self):
        spec = _planned_perception(PerceptionSpec(), "window")
        self.assertEqual(spec.temporal, "window")
        self.assertGreaterEqual(spec.n_frames, 3)

    def test_window_still_raises_a_strip_spec(self):
        spec = _planned_perception(PerceptionSpec(temporal="strip", n_frames=4), "window")
        self.assertEqual((spec.temporal, spec.n_frames), ("window", 4))

    def test_authored_window_still_never_downgraded_by_single_or_strip(self):
        authored = PerceptionSpec(temporal="window", n_frames=5)
        self.assertIs(_planned_perception(authored, "single"), authored)
        self.assertIs(_planned_perception(authored, "strip"), authored)

    def test_unknown_plan_mode_still_ignored(self):
        base = PerceptionSpec()
        self.assertIs(_planned_perception(base, "telepathy"), base)


class PlanModesExcludesEventCapsuleTests(unittest.TestCase):
    """``PLAN_MODES`` — the vocabulary a resolved ``WatchProfile.plan_mode`` is validated against —
    still excludes ``event_capsule``; this suite covers the agent-side rank rung, not the profile fence."""

    def test_plan_modes_excludes_event_capsule(self):
        self.assertNotIn("event_capsule", PLAN_MODES)

    def test_plan_modes_still_has_the_four_live_modes(self):
        self.assertEqual(set(PLAN_MODES), {"single", "strip", "window", "sprite"})


if __name__ == "__main__":
    unittest.main()
