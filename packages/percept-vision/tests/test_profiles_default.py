"""Seeded, H0-derived default WatchProfiles (card P3, impl plan §2/§3).

Verifies: importing ``percept.profiles_default`` seeds ``percept.watchprofile.DEFAULT_REGISTRY`` with the
three named H0-derived profiles plus a WILDCARD fallback; each profile's knobs match the H0-measured
values verbatim; ``resolve()`` reaches every profile -- including an unrecognized context, via the
wildcard; and every profile carries a provenance note tying it to the H0 charades-e2e ablation ladder.
"""

from __future__ import annotations

import unittest

from percept import profiles_default
from percept.watchprofile import DEFAULT_REGISTRY, WatchProfileRegistry


# ---------------------------------------------------------------------------
# Registry seeded at import
# ---------------------------------------------------------------------------


class RegistrySeededAtImportTests(unittest.TestCase):
    def test_default_registry_contains_all_four_profiles_after_import(self):
        # profiles_default is already imported (module import above) -- DEFAULT_REGISTRY must be seeded.
        self.assertEqual(len(DEFAULT_REGISTRY), 4)
        for profile in profiles_default.ALL_PROFILES:
            self.assertIn(profile.key, DEFAULT_REGISTRY)
            self.assertIs(DEFAULT_REGISTRY.get(*profile.key), profile)

    def test_reseeding_the_default_registry_is_a_noop(self):
        profiles_default.seed()  # same singleton profile objects -> no-op, must not raise
        self.assertEqual(len(DEFAULT_REGISTRY), 4)

    def test_seed_onto_a_fresh_registry_is_independent(self):
        registry = WatchProfileRegistry()
        profiles_default.seed(registry)
        self.assertEqual(len(registry), 4)
        # the process-wide default is untouched by seeding a fresh registry
        self.assertEqual(len(DEFAULT_REGISTRY), 4)


# ---------------------------------------------------------------------------
# fixed-cam-localized -- the proven 0.90-recall stack
# ---------------------------------------------------------------------------


class FixedCamLocalizedTests(unittest.TestCase):
    def test_key(self):
        p = profiles_default.FIXED_CAM_LOCALIZED
        self.assertEqual(p.key, ("fixed_camera", "stative", "start_anchored"))

    def test_knobs_match_h0(self):
        p = profiles_default.FIXED_CAM_LOCALIZED
        self.assertEqual(p.salience_stat, "block_max")
        self.assertEqual(p.stat_thresholds, {"block_max": 0.20})
        self.assertEqual(p.plan_mode, "window")
        self.assertFalse(p.gate.require_false_first)
        self.assertEqual(p.gate.refractory_s, 15.0)


# ---------------------------------------------------------------------------
# dynamic-action
# ---------------------------------------------------------------------------


class DynamicActionTests(unittest.TestCase):
    def test_environment_is_a_fixed_camera(self):
        self.assertEqual(profiles_default.DYNAMIC_ACTION.environment, "fixed_camera")

    def test_knobs_match_h0(self):
        p = profiles_default.DYNAMIC_ACTION
        self.assertEqual(p.salience_stat, "block_max")
        self.assertEqual(p.stat_thresholds, {"block_max": 0.20})
        self.assertEqual(p.plan_mode, "window")
        self.assertEqual(p.gate.refractory_s, 15.0)


# ---------------------------------------------------------------------------
# handheld -- ego-compensation ON
# ---------------------------------------------------------------------------


class HandheldTests(unittest.TestCase):
    def test_key_is_broad_over_goal_and_state(self):
        p = profiles_default.HANDHELD
        self.assertEqual(p.key, ("handheld", "*", "*"))

    def test_knobs_match_h0(self):
        p = profiles_default.HANDHELD
        self.assertEqual(p.salience_stat, "block_max")
        self.assertEqual(p.stat_thresholds, {"block_max": 0.20})
        self.assertEqual(p.plan_mode, "window")
        self.assertEqual(p.gate.refractory_s, 15.0)

    def test_ego_compensation_is_documented(self):
        # No typed ego_compensate field on WatchProfile yet (card P1 scope) -- the binding is the note.
        notes = profiles_default.HANDHELD.notes
        self.assertIn("ego_compensate", notes)
        self.assertIn("98%", notes)
        self.assertIn("30%", notes)


# ---------------------------------------------------------------------------
# WILDCARD fallback
# ---------------------------------------------------------------------------


class WildcardFallbackTests(unittest.TestCase):
    def test_wildcard_default_key_is_fully_broad(self):
        self.assertEqual(profiles_default.WILDCARD_DEFAULT.key, ("*", "*", "*"))

    def test_resolve_reaches_wildcard_default_for_an_unknown_context(self):
        got = DEFAULT_REGISTRY.resolve("drone", "unknown-goal-type", "unknown-state")
        self.assertIs(got, profiles_default.WILDCARD_DEFAULT)

    def test_resolve_reaches_handheld_for_an_unlisted_goal_and_state(self):
        got = DEFAULT_REGISTRY.resolve("handheld", "periodic", "transition")
        self.assertIs(got, profiles_default.HANDHELD)

    def test_resolve_prefers_the_named_profiles_over_the_wildcard(self):
        self.assertIs(
            DEFAULT_REGISTRY.resolve("fixed_camera", "stative", "start_anchored"),
            profiles_default.FIXED_CAM_LOCALIZED,
        )
        self.assertIs(
            DEFAULT_REGISTRY.resolve("fixed_camera", "dynamic", "transition"),
            profiles_default.DYNAMIC_ACTION,
        )

    def test_resolve_never_raises_for_any_context_once_seeded(self):
        registry = WatchProfileRegistry()
        profiles_default.seed(registry)
        for env, goal, state in (
            ("fixed_camera", "stative", "start_anchored"),
            ("fixed_camera", "dynamic", "transition"),
            ("handheld", "stative", "start_anchored"),
            ("wearable", "periodic", "transition"),  # not even "handheld" -- must still resolve
            ("", "", ""),
        ):
            registry.resolve(env or "anything", goal or "anything", state or "anything")  # must not raise


# ---------------------------------------------------------------------------
# Provenance -- every profile must be auditable, not a magic number.
# ---------------------------------------------------------------------------


class ProvenanceTests(unittest.TestCase):
    def test_every_profile_carries_a_nonempty_provenance_note(self):
        for p in profiles_default.ALL_PROFILES:
            self.assertTrue(p.notes.strip(), f"{p.name} has no provenance note")

    def test_every_profile_cites_the_h0_ablation_ladder(self):
        for p in profiles_default.ALL_PROFILES:
            self.assertIn("charades-e2e", p.notes, f"{p.name} missing eval-run citation")
            self.assertIn("0.39", p.notes, f"{p.name} missing baseline recall figure")
            self.assertIn("0.68", p.notes, f"{p.name} missing block_max recall figure")
            self.assertIn("0.73", p.notes, f"{p.name} missing +window recall figure")
            self.assertIn("0.90", p.notes, f"{p.name} missing +gate-relax recall figure")


if __name__ == "__main__":
    unittest.main()
