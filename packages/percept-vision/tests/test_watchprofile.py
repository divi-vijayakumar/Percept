"""WatchProfile schema validation and registry lookup.

Pure dataclass + dict-of-tuples logic: no model, no I/O. Covers (1) the schema rejects out-of-vocabulary /
out-of-range knobs, matching the real settings they mirror (percept.config.EdgeSettings/GateSettings,
percept.behavior.Temporal, percept.sampling.SensePolicy), and (2) the registry's exact + wildcard-fallback
lookup.
"""

from __future__ import annotations

import unittest

from percept.sampling import SensePolicy
from percept.watchprofile import (
    PLAN_MODES,
    SALIENCE_STATS,
    CadenceProfile,
    GateProfile,
    GoalDensityGuidance,
    ProfileNotFoundError,
    WatchProfile,
    WatchProfileError,
    WatchProfileRegistry,
)


def _profile(**overrides) -> WatchProfile:
    kwargs = dict(
        name="fixed-cam-localized",
        environment="fixed_camera",
        goal_type="stative",
        state="start_anchored",
        salience_stat="block_max",
        stat_thresholds={"block_max": 0.20},
        plan_mode="single",
    )
    kwargs.update(overrides)
    return WatchProfile(**kwargs)


# ---------------------------------------------------------------------------
# Vocabulary sanity — these mirror real settings; a change to either should show up here.
# ---------------------------------------------------------------------------


class VocabularyTests(unittest.TestCase):
    def test_salience_stats_match_edge_settings_vocabulary(self):
        self.assertEqual(set(SALIENCE_STATS), {"mean", "block_max", "p95"})

    def test_plan_modes_match_behavior_temporal_vocabulary(self):
        self.assertEqual(set(PLAN_MODES), {"single", "strip", "window", "sprite"})


# ---------------------------------------------------------------------------
# CadenceProfile
# ---------------------------------------------------------------------------


class CadenceProfileTests(unittest.TestCase):
    def test_defaults_are_valid(self):
        CadenceProfile()  # must not raise

    def test_round_trips_to_a_real_sense_policy(self):
        c = CadenceProfile(base_fps=2.0, max_fps=10.0, escalate_threshold=0.12, hold_s=0.75)
        sp = c.to_sense_policy()
        self.assertIsInstance(sp, SensePolicy)
        self.assertEqual((sp.base_fps, sp.max_fps, sp.escalate_threshold, sp.hold_s), (2.0, 10.0, 0.12, 0.75))

    def test_rejects_base_fps_above_max_fps(self):
        with self.assertRaises(WatchProfileError):
            CadenceProfile(base_fps=20.0, max_fps=10.0)

    def test_rejects_non_positive_fps(self):
        with self.assertRaises(WatchProfileError):
            CadenceProfile(base_fps=0.0)
        with self.assertRaises(WatchProfileError):
            CadenceProfile(max_fps=-1.0)

    def test_rejects_escalate_threshold_out_of_range(self):
        with self.assertRaises(WatchProfileError):
            CadenceProfile(escalate_threshold=1.5)
        with self.assertRaises(WatchProfileError):
            CadenceProfile(escalate_threshold=-0.01)

    def test_rejects_negative_hold_s(self):
        with self.assertRaises(WatchProfileError):
            CadenceProfile(hold_s=-0.1)


# ---------------------------------------------------------------------------
# GateProfile
# ---------------------------------------------------------------------------


class GateProfileTests(unittest.TestCase):
    def test_defaults_are_valid(self):
        GateProfile()  # must not raise

    def test_rejects_negative_refractory(self):
        with self.assertRaises(WatchProfileError):
            GateProfile(refractory_s=-1.0)

    def test_require_false_first_is_a_plain_bool_passthrough(self):
        self.assertTrue(GateProfile(require_false_first=True).require_false_first)
        self.assertFalse(GateProfile(require_false_first=False).require_false_first)


# ---------------------------------------------------------------------------
# GoalDensityGuidance
# ---------------------------------------------------------------------------


class GoalDensityGuidanceTests(unittest.TestCase):
    def test_default_matches_h0_operating_point(self):
        # H0's full-stack operating point held <=3 distinct goals (the documented implementation evidence ablation table).
        self.assertEqual(GoalDensityGuidance().max_recommended_goals, 3)

    def test_none_is_allowed_no_guidance(self):
        GoalDensityGuidance(max_recommended_goals=None)  # must not raise

    def test_rejects_non_positive_max_goals(self):
        with self.assertRaises(WatchProfileError):
            GoalDensityGuidance(max_recommended_goals=0)
        with self.assertRaises(WatchProfileError):
            GoalDensityGuidance(max_recommended_goals=-2)


# ---------------------------------------------------------------------------
# WatchProfile
# ---------------------------------------------------------------------------


class WatchProfileTests(unittest.TestCase):
    def test_valid_profile_constructs(self):
        p = _profile()
        self.assertEqual(p.key, ("fixed_camera", "stative", "start_anchored"))

    def test_defaults_are_the_legacy_byte_for_byte_path(self):
        # mean / single mirror EdgeSettings.salience_stat="mean" and PerceptionSpec's single-frame default
        # (percept/config.py, percept/behavior.py) — a profile built with no overrides changes nothing.
        p = WatchProfile(name="noop", environment="any", goal_type="any", state="any")
        self.assertEqual(p.salience_stat, "mean")
        self.assertEqual(p.plan_mode, "single")
        self.assertEqual(p.stat_thresholds, {})
        self.assertFalse(p.gate.require_false_first)

    def test_rejects_empty_identity_fields(self):
        for field_name in ("name", "environment", "goal_type", "state"):
            with self.assertRaises(WatchProfileError):
                _profile(**{field_name: ""})
            with self.assertRaises(WatchProfileError):
                _profile(**{field_name: "   "})

    def test_rejects_unknown_salience_stat(self):
        with self.assertRaises(WatchProfileError):
            _profile(salience_stat="max_pool")

    def test_rejects_unknown_plan_mode(self):
        with self.assertRaises(WatchProfileError):
            _profile(plan_mode="montage")

    def test_rejects_stat_thresholds_key_not_a_salience_stat(self):
        with self.assertRaises(WatchProfileError):
            _profile(stat_thresholds={"whole_frame_mean": 0.1})

    def test_rejects_stat_thresholds_value_out_of_range(self):
        with self.assertRaises(WatchProfileError):
            _profile(stat_thresholds={"block_max": 1.5})
        with self.assertRaises(WatchProfileError):
            _profile(stat_thresholds={"block_max": -0.1})

    def test_rejects_stat_thresholds_non_numeric_value(self):
        with self.assertRaises(WatchProfileError):
            _profile(stat_thresholds={"block_max": "high"})

    def test_nested_cadence_validation_propagates(self):
        with self.assertRaises(WatchProfileError):
            _profile(cadence=CadenceProfile(base_fps=99.0, max_fps=1.0))

    def test_nested_gate_validation_propagates(self):
        with self.assertRaises(WatchProfileError):
            _profile(gate=GateProfile(refractory_s=-5.0))

    def test_matches_h0_fixed_cam_localized_finding(self):
        # The H0 ablation win: fixed camera + localized action -> block_max @ 0.20 (the documented implementation evidence).
        p = _profile()
        self.assertEqual(p.salience_stat, "block_max")
        self.assertEqual(p.stat_thresholds["block_max"], 0.20)


# ---------------------------------------------------------------------------
# WatchProfileRegistry
# ---------------------------------------------------------------------------


class WatchProfileRegistryTests(unittest.TestCase):
    def setUp(self):
        self.registry = WatchProfileRegistry()

    def test_register_then_exact_get(self):
        p = _profile()
        self.registry.register(p)
        self.assertIs(self.registry.get("fixed_camera", "stative", "start_anchored"), p)

    def test_get_is_exact_only_no_fallback(self):
        p = _profile(environment="fixed_camera", goal_type="stative", state="start_anchored")
        self.registry.register(p)
        self.assertIsNone(self.registry.get("fixed_camera", "stative", "transition"))

    def test_registers_under_its_own_key_not_a_caller_supplied_one(self):
        p = _profile()
        self.registry.register(p)
        self.assertIn(p.key, self.registry)
        self.assertEqual(len(self.registry), 1)

    def test_duplicate_registration_without_overwrite_raises(self):
        self.registry.register(_profile(name="a"))
        with self.assertRaises(WatchProfileError):
            self.registry.register(_profile(name="b"))  # same (environment, goal_type, state) key

    def test_duplicate_registration_with_overwrite_replaces(self):
        self.registry.register(_profile(name="a"))
        self.registry.register(_profile(name="b"), overwrite=True)
        got = self.registry.get("fixed_camera", "stative", "start_anchored")
        self.assertEqual(got.name, "b")

    def test_reregistering_the_identical_object_is_a_noop(self):
        p = _profile()
        self.registry.register(p)
        self.registry.register(p)  # must not raise
        self.assertEqual(len(self.registry), 1)

    def test_unregister_removes_exact_key(self):
        self.registry.register(_profile())
        self.registry.unregister("fixed_camera", "stative", "start_anchored")
        self.assertEqual(len(self.registry), 0)

    def test_unregister_missing_key_is_a_noop(self):
        self.registry.unregister("nope", "nope", "nope")  # must not raise

    def test_resolve_exact_match(self):
        p = _profile()
        self.registry.register(p)
        self.assertIs(self.registry.resolve("fixed_camera", "stative", "start_anchored"), p)

    def test_resolve_falls_back_to_goal_type_wildcard(self):
        fallback = _profile(name="fixed-any-state", state="*")
        self.registry.register(fallback)
        got = self.registry.resolve("fixed_camera", "stative", "transition")
        self.assertEqual(got.name, "fixed-any-state")

    def test_resolve_falls_back_to_environment_wildcard(self):
        fallback = _profile(name="fixed-cam-anything", goal_type="*", state="*")
        self.registry.register(fallback)
        got = self.registry.resolve("fixed_camera", "dynamic", "transition")
        self.assertEqual(got.name, "fixed-cam-anything")

    def test_resolve_falls_back_to_global_default(self):
        default = _profile(name="global-default", environment="*", goal_type="*", state="*")
        self.registry.register(default)
        got = self.registry.resolve("handheld", "periodic", "transition")
        self.assertEqual(got.name, "global-default")

    def test_resolve_prefers_more_specific_match(self):
        specific = _profile(name="specific")
        broad = _profile(name="broad", goal_type="*", state="*")
        self.registry.register(specific)
        self.registry.register(broad)
        got = self.registry.resolve("fixed_camera", "stative", "start_anchored")
        self.assertEqual(got.name, "specific")

    def test_resolve_raises_when_nothing_matches(self):
        with self.assertRaises(ProfileNotFoundError):
            self.registry.resolve("fixed_camera", "stative", "start_anchored")

    def test_all_and_iteration(self):
        a, b = _profile(name="a"), _profile(name="b", state="transition")
        self.registry.register(a)
        self.registry.register(b)
        self.assertEqual({p.name for p in self.registry.all()}, {"a", "b"})
        self.assertEqual({p.name for p in self.registry}, {"a", "b"})

    def test_registries_are_independent_instances(self):
        # DEFAULT_REGISTRY is a shared process-wide instance; a fresh WatchProfileRegistry() must not
        # see anything registered on it (or on another fresh instance) — no accidental global state.
        other = WatchProfileRegistry()
        other.register(_profile())
        self.assertEqual(len(self.registry), 0)
        self.assertEqual(len(other), 1)


if __name__ == "__main__":
    unittest.main()
