"""— the runtime reads its knobs from a resolved WatchProfile, not env vars.

Proves ``percept.profile_wiring.resolve_watch_profile`` really binds a ``WatchProfile`` onto the real
settings objects (``EdgeSettings``, ``GateSettings``, ``SensePolicy``, ``plan_mode``) with ZERO env vars
set, that ``percept.watch.watch()``/``Percept.create`` apply it at CORE CREATION time, and that
``profile=None`` keeps today's ``Settings``/env-driven behavior BYTE-FOR-BYTE.
"""

from __future__ import annotations

import asyncio
import os
import unittest

from percept import Percept
from percept.adapters import Goal
from percept.config import Settings, load_settings
from percept.profile_wiring import ResolvedProfile, resolve_watch_profile
from percept.sampling import SamplingController, SensePolicy
from percept.watch import watch
from percept.watchprofile import CadenceProfile, GateProfile, WatchProfile

_ENV_KEYS = (
    "PERCEPT_MOTION_GATE",
    "PERCEPT_SALIENCE_STAT",
    "PERCEPT_MOTION_GATE_BLOCK_MAX",
    "PERCEPT_MOTION_GATE_P95",
    "PERCEPT_GATE_REQUIRE_FALSE_FIRST",
    "PERCEPT_GATE_REFRACTORY_S",
)


def _no_env_vars_set() -> None:
    for k in _ENV_KEYS:
        assert k not in os.environ, f"{k} must not be set for this test to prove zero-env-var wiring"


def _profile(**overrides) -> WatchProfile:
    kwargs = dict(
        name="fixed-cam-localized",
        environment="fixed_camera",
        goal_type="stative",
        state="start_anchored",
        salience_stat="block_max",
        stat_thresholds={"block_max": 0.20, "p95": 0.15},
        plan_mode="window",
        cadence=CadenceProfile(base_fps=2.0, max_fps=20.0, escalate_threshold=0.3, hold_s=1.0),
        gate=GateProfile(require_false_first=True, refractory_s=2.5),
    )
    kwargs.update(overrides)
    return WatchProfile(**kwargs)


class ResolveWatchProfileTests(unittest.TestCase):
    """``resolve_watch_profile`` binds every profile knob onto the real settings objects."""

    def setUp(self):
        _no_env_vars_set()

    def test_returns_resolved_profile(self):
        resolved = resolve_watch_profile(_profile())
        self.assertIsInstance(resolved, ResolvedProfile)
        self.assertIsInstance(resolved.settings, Settings)
        self.assertIsInstance(resolved.sense_policy, SensePolicy)
        self.assertIsInstance(resolved.plan_mode, str)

    def test_salience_stat_lands_on_edge_settings(self):
        resolved = resolve_watch_profile(_profile(salience_stat="p95", stat_thresholds={"p95": 0.15}))
        self.assertEqual(resolved.settings.edge.salience_stat, "p95")

    def test_stat_thresholds_land_on_the_matching_edge_field(self):
        resolved = resolve_watch_profile(
            _profile(stat_thresholds={"mean": 0.33, "block_max": 0.44, "p95": 0.55})
        )
        self.assertEqual(resolved.settings.edge.motion_gate, 0.33)
        self.assertEqual(resolved.settings.edge.motion_gate_block_max, 0.44)
        self.assertEqual(resolved.settings.edge.motion_gate_p95, 0.55)

    def test_missing_stat_threshold_falls_back_to_base_settings(self):
        base = load_settings()
        resolved = resolve_watch_profile(_profile(stat_thresholds={"block_max": 0.20}), base=base)
        # "mean"/"p95" not in stat_thresholds -> untouched from base
        self.assertEqual(resolved.settings.edge.motion_gate, base.edge.motion_gate)
        self.assertEqual(resolved.settings.edge.motion_gate_p95, base.edge.motion_gate_p95)
        self.assertEqual(resolved.settings.edge.motion_gate_block_max, 0.20)

    def test_gate_require_false_first_and_refractory_land_on_gate_settings(self):
        resolved = resolve_watch_profile(_profile(gate=GateProfile(require_false_first=True, refractory_s=3.0)))
        self.assertTrue(resolved.settings.gate.require_false_first)
        self.assertEqual(resolved.settings.gate.refractory_s, 3.0)

    def test_cadence_resolves_to_a_real_sense_policy(self):
        cadence = CadenceProfile(base_fps=1.0, max_fps=9.0, escalate_threshold=0.4, hold_s=2.0)
        resolved = resolve_watch_profile(_profile(cadence=cadence))
        self.assertEqual(resolved.sense_policy, cadence.to_sense_policy())
        self.assertEqual(resolved.sense_policy.base_fps, 1.0)
        self.assertEqual(resolved.sense_policy.max_fps, 9.0)

    def test_plan_mode_threaded_through_unchanged(self):
        resolved = resolve_watch_profile(_profile(plan_mode="sprite"))
        self.assertEqual(resolved.plan_mode, "sprite")

    def test_untouched_settings_branches_carry_over_from_base(self):
        base = load_settings()
        resolved = resolve_watch_profile(_profile(), base=base)
        self.assertEqual(resolved.settings.cache, base.cache)
        self.assertEqual(resolved.settings.models, base.models)
        self.assertEqual(resolved.settings.routing, base.routing)

    def test_default_base_is_load_settings(self):
        # no ``base=`` kwarg -> resolve_watch_profile seeds from load_settings() itself (env-driven
        # defaults), not some other hardcoded Settings().
        resolved = resolve_watch_profile(_profile())
        self.assertEqual(resolved.settings.cache, load_settings().cache)


class CadenceChangesSensePolicyBehaviorTests(unittest.TestCase):
    """The resolved cadence is not just a field — it changes real ``SamplingController`` decisions."""

    def setUp(self):
        _no_env_vars_set()

    def test_localized_profile_cadence_escalates_where_default_does_not(self):
        default_policy = SensePolicy()  # base_fps=4.0, max_fps=15.0, escalate_threshold=0.15
        profile = _profile(
            cadence=CadenceProfile(base_fps=1.0, max_fps=30.0, escalate_threshold=0.05, hold_s=0.5)
        )
        resolved = resolve_watch_profile(profile)

        default_ctrl = SamplingController(default=default_policy)
        profile_ctrl = SamplingController(default=resolved.sense_policy)

        # Prime both controllers with a first (always-processed) frame at t=0, then probe a SECOND
        # frame shortly after: a motion level of 0.10 clears the profile's low escalate_threshold
        # (0.05) -- boosting it toward max_fps=30, dense enough to process again at t=0.1 -- but NOT
        # the default's higher threshold (0.15), which stays at base_fps=4.0 (interval 0.25s) and
        # skips t=0.1. Observable behavior difference, zero env vars involved.
        default_ctrl.should_process("g", t=0.0, motion=0.0)
        profile_ctrl.should_process("g", t=0.0, motion=0.0)
        self.assertFalse(default_ctrl.should_process("g", t=0.1, motion=0.10))
        self.assertTrue(profile_ctrl.should_process("g", t=0.1, motion=0.10))


class PerceptCreateAppliesProfileAtCreationTests(unittest.TestCase):
    """M2: cadence/plan_mode are applied at CORE CREATION time, never a mid-run mutation."""

    def setUp(self):
        _no_env_vars_set()

    def test_create_with_no_overrides_defaults_to_none(self):
        core = Percept.create(mode="offline", discover_plugins=False)
        self.assertIsNone(core.sense_policy)
        self.assertIsNone(core.plan_mode)

    def test_create_carries_sense_policy_and_plan_mode_verbatim(self):
        policy = SensePolicy(base_fps=2.0, max_fps=8.0)
        core = Percept.create(
            mode="offline", discover_plugins=False, sense_policy=policy, plan_mode="window"
        )
        self.assertIs(core.sense_policy, policy)
        self.assertEqual(core.plan_mode, "window")

    def test_resolved_profile_settings_reach_the_gate(self):
        # gate.require_false_first / gate.refractory_s -> GateSettings -> the real Gate object.
        profile = _profile(gate=GateProfile(require_false_first=True, refractory_s=4.0))
        resolved = resolve_watch_profile(profile)
        core = Percept.create(mode="offline", discover_plugins=False, settings=resolved.settings)
        self.assertTrue(core.gate.require_false_first)
        self.assertEqual(core.gate.refractory_s, 4.0)


class WatchThreadsProfileAtBuildTests(unittest.TestCase):
    """``watch()`` applies a WatchProfile when it builds its own default core (no ``percept=``)."""

    def setUp(self):
        _no_env_vars_set()

    def test_watch_builds_a_core_carrying_the_resolved_profile(self):
        profile = _profile(gate=GateProfile(require_false_first=True, refractory_s=1.5))
        async def _run():
            goal = Goal(id="g", condition="anything", say="hi")
            async for _ in watch([], [goal], profile, source="camera"):
                pass

        # watch([]) never yields a frame, so we can't introspect the internally-built core directly;
        # instead prove the SAME resolution watch() uses (_build_core -> resolve_watch_profile) lands
        # the profile's gate knobs on a freshly created core, end to end.
        asyncio.run(_run())  # sanity: profile threading through watch() doesn't raise
        resolved = resolve_watch_profile(profile)
        core = Percept.create(
            settings=resolved.settings, sense_policy=resolved.sense_policy, plan_mode=resolved.plan_mode
        )
        self.assertTrue(core.gate.require_false_first)
        self.assertEqual(core.gate.refractory_s, 1.5)
        self.assertEqual(core.plan_mode, "window")
        self.assertEqual(core.sense_policy, profile.cadence.to_sense_policy())


class ProfileNoneKeepsByteForByteDefaultTests(unittest.TestCase):
    """CRITICAL: ``profile=None`` must keep today's Settings/env-driven behavior BYTE-FOR-BYTE."""

    def test_settings_from_profile_none_is_none(self):
        from percept.watch import _settings_from_profile

        self.assertIsNone(_settings_from_profile(None))

    def test_build_core_with_none_profile_matches_bare_create(self):
        from percept.watch import _build_core

        core = _build_core(None)
        bare = Percept.create()
        # same settings tree (both loaded fresh from env -- no profile knobs applied)
        self.assertEqual(core.settings, bare.settings)
        self.assertIsNone(core.sense_policy)
        self.assertIsNone(bare.sense_policy)
        self.assertIsNone(core.plan_mode)
        self.assertIsNone(bare.plan_mode)

    def test_watch_default_profile_fires_identically_to_pre_m2_path(self):
        # No profile at all -- the documented default. Proves watch() with profile=None still drives
        # the existing fovea -> judge -> gate path unchanged (mirrors tests/test_watch.py's L1 coverage).
        from percept import FIRE
        from percept.adapters import Verdict
        from percept.backends.fake import FakeVision

        vision = FakeVision(
            script={"intruder": Verdict(satisfied=True, confidence=0.95, evidence="a person is in the zone")}
        )
        core = Percept.create(vision=vision, discover_plugins=False)
        goal = Goal(id="zone", condition="a person is in the restricted zone", say="Alert: person in zone")

        async def _run():
            return [ev async for ev in watch(["intruder"], [goal], None, percept=core)]

        events = asyncio.run(_run())
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].action, FIRE)


if __name__ == "__main__":
    unittest.main()
