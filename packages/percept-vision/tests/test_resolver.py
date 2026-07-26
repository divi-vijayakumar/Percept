"""ConfigResolver — the ADR-008 resolution core (percept/resolver.py).

Covers, per the card's own test list: precedence both directions (env beats a lower-precedence
default; an explicit WatchPlan override beats env); the loud diagnostic when env overrides an
explicitly-selected profile; an out-of-ownership write raising ConfigError; an unknown key
raising; ``explain()`` rendering every effective field with its owner + source; hashes stable
across identical-input resolve() calls; and profile_resolution.resolve_situation's contributed
values arriving provenance-stamped. Also covers the DomainPack/ExecutionPolicy/OptimizationLock
placeholder inputs and the OptimizationLock safe_to_optimize / hash-mismatch rules.

No runtime execution: every test constructs inputs and calls ``ConfigResolver.resolve(...)``,
never touching worker.py/agent.py/gate.py.
"""

from __future__ import annotations

import unittest
from unittest import mock

from percept.config import Settings
from percept.contracts import WatchPlan
from percept.resolver import (
    ConfigError,
    ConfigResolver,
    DomainPack,
    ExecutionPolicy,
    LAYER_DEPLOYMENT,
    LAYER_EXECUTION_POLICY,
    LAYER_GATE,
    LAYER_OPTIMIZATION_LOCK,
    LAYER_SDK,
    LAYER_SOURCE_PROFILE,
    LAYER_WATCH_PLAN,
    OptimizationLock,
)
from percept.sourceprofile import SourceProfile


def _plan(**kwargs) -> WatchPlan:
    kwargs.setdefault("condition", "a person is at the door")
    return WatchPlan(**kwargs)


class TestPrecedenceBothDirections(unittest.TestCase):
    """ADR-008 §Precedence's two named "tested consequences"."""

    def test_env_beats_sdk_default(self) -> None:
        resolver = ConfigResolver()
        with mock.patch.dict("os.environ", {"PERCEPT_SALIENCE_STAT": "block_max"}, clear=False):
            resolved = resolver.resolve(_plan())
        self.assertEqual(resolved.effective["Settings.edge.salience_stat"], "block_max")
        origin = resolved.provenance["Settings.edge.salience_stat"]
        self.assertEqual(origin.owner_layer, LAYER_DEPLOYMENT)
        self.assertEqual(origin.source_id, "env:PERCEPT_SALIENCE_STAT")

    def test_env_beats_situation_derived_default(self) -> None:
        """A situation-derived value sits at the SourceProfile precedence SLOT (even though its
        provenance owner label is often 'WatchPlan'/'Gate' per the registry — see resolver.py's
        module docstring); Deployment/env, positioned higher, still outranks it."""
        resolver = ConfigResolver()
        with mock.patch.dict("os.environ", {"PERCEPT_SALIENCE_STAT": "p95"}, clear=False):
            resolved = resolver.resolve(_plan(), situation=("dynamic", "fixed", "start_anchored"))
        # resolve_situation always picks "block_max"; env must still win.
        self.assertEqual(resolved.effective["Settings.edge.salience_stat"], "p95")
        self.assertEqual(resolved.provenance["Settings.edge.salience_stat"].owner_layer, LAYER_DEPLOYMENT)

    def test_explicit_watch_plan_beats_env(self) -> None:
        resolver = ConfigResolver()
        plan = _plan(explicit={"WatchProfile.salience_stat": "p95"})
        with mock.patch.dict("os.environ", {"PERCEPT_SALIENCE_STAT": "block_max"}, clear=False):
            resolved = resolver.resolve(plan)
        self.assertEqual(resolved.effective["Settings.edge.salience_stat"], "p95")
        origin = resolved.provenance["Settings.edge.salience_stat"]
        self.assertEqual(origin.owner_layer, LAYER_WATCH_PLAN)
        self.assertEqual(origin.source_id, "watch_plan.explicit")

    def test_unset_watch_plan_field_does_not_participate(self) -> None:
        """An unset WatchPlan field is not 'explicit' — the next value down the chain applies
        untouched (ADR-008 §Precedence)."""
        resolver = ConfigResolver()
        with mock.patch.dict("os.environ", {"PERCEPT_SALIENCE_STAT": "block_max"}, clear=False):
            resolved = resolver.resolve(_plan())  # no explicit override at all
        self.assertEqual(resolved.effective["Settings.edge.salience_stat"], "block_max")
        self.assertEqual(resolved.provenance["Settings.edge.salience_stat"].owner_layer, LAYER_DEPLOYMENT)

    def test_no_env_no_override_stays_sdk_default(self) -> None:
        resolver = ConfigResolver()
        with mock.patch.dict("os.environ", {}, clear=True):
            resolved = resolver.resolve(_plan())
        self.assertEqual(resolved.effective["Settings.edge.salience_stat"], "mean")
        self.assertEqual(resolved.provenance["Settings.edge.salience_stat"].owner_layer, LAYER_SDK)
        self.assertEqual(resolved.provenance["Settings.edge.salience_stat"].source_id, "sdk_default")

    def test_stat_thresholds_fan_out_to_three_motion_gate_fields(self) -> None:
        resolver = ConfigResolver()
        plan = _plan(explicit={"WatchProfile.stat_thresholds": {"block_max": 0.33, "p95": 0.44}})
        resolved = resolver.resolve(plan)
        self.assertEqual(resolved.effective["Settings.edge.motion_gate_block_max"], 0.33)
        self.assertEqual(resolved.effective["Settings.edge.motion_gate_p95"], 0.44)
        self.assertEqual(resolved.provenance["Settings.edge.motion_gate_block_max"].owner_layer, LAYER_WATCH_PLAN)


class TestLoudDiagnosticOnProfileOverride(unittest.TestCase):
    """"When a Deployment/env value overrides a field owned by an explicitly selected profile,
    resolution succeeds but emits a loud diagnostic" (ADR-008 §Precedence)."""

    def test_env_overriding_explicit_situation_emits_warning(self) -> None:
        resolver = ConfigResolver()
        with mock.patch.dict("os.environ", {"PERCEPT_EGO_COMPENSATE": "true"}, clear=False):
            resolved = resolver.resolve(_plan(), situation=("dynamic", "fixed", "start_anchored"))
        # resolve_situation picks ego_compensate=False for a "fixed" mount; env flips it to True.
        self.assertTrue(resolved.effective["Settings.edge.ego_compensate"])
        self.assertEqual(resolver.last_receipt.status, "applied_with_warnings")
        codes = [w.code for w in resolver.last_receipt.warnings]
        self.assertIn("env_overrides_explicit_profile", codes)
        warning = next(w for w in resolver.last_receipt.warnings if w.code == "env_overrides_explicit_profile")
        self.assertEqual(warning.path, "Settings.edge.ego_compensate")

    def test_env_agreeing_with_explicit_profile_no_warning(self) -> None:
        """No conflict, no diagnostic — the loud warning is only for a genuine override."""
        resolver = ConfigResolver()
        with mock.patch.dict("os.environ", {"PERCEPT_EGO_COMPENSATE": "false"}, clear=False):
            resolver.resolve(_plan(), situation=("dynamic", "fixed", "start_anchored"))
        self.assertEqual(resolver.last_receipt.status, "applied")
        self.assertEqual(resolver.last_receipt.warnings, ())

    def test_no_env_override_no_warning(self) -> None:
        resolver = ConfigResolver()
        with mock.patch.dict("os.environ", {}, clear=True):
            resolver.resolve(_plan(), situation=("dynamic", "fixed", "start_anchored"))
        self.assertEqual(resolver.last_receipt.status, "applied")

    def test_env_override_without_explicit_profile_is_silent(self) -> None:
        """Nothing 'explicit' was overridden (no source_profile, no situation passed) — env simply
        wins over the SDK default, no loud diagnostic warranted."""
        resolver = ConfigResolver()
        with mock.patch.dict("os.environ", {"PERCEPT_EGO_COMPENSATE": "true"}, clear=False):
            resolver.resolve(_plan())
        self.assertEqual(resolver.last_receipt.status, "applied")


class TestOwnershipConfigError(unittest.TestCase):
    def test_domain_pack_cannot_set_deployment_field(self) -> None:
        """ADR-008's own worked example: 'A DomainPack cannot set PERCEPT_VISION_BACKEND-shaped
        deployment fields no matter when it is applied in the precedence chain'."""
        resolver = ConfigResolver()
        pack = DomainPack(id="dp1", values={"Settings.models.vision": "claude"})
        with self.assertRaises(ConfigError) as ctx:
            resolver.resolve(_plan(), domain_pack=pack)
        self.assertIn("DomainPack", str(ctx.exception))
        self.assertIn("DeploymentConfig", str(ctx.exception))

    def test_execution_policy_cannot_set_source_profile_field(self) -> None:
        resolver = ConfigResolver()
        policy = ExecutionPolicy(id="ep1", values={"Settings.edge.motion_gate": 0.5})
        with self.assertRaises(ConfigError):
            resolver.resolve(_plan(), execution_policy=policy)

    def test_watch_plan_cannot_set_deployment_field(self) -> None:
        resolver = ConfigResolver()
        plan = _plan(explicit={"Settings.models.vision": "claude"})
        with self.assertRaises(ConfigError):
            resolver.resolve(plan)

    def test_watch_plan_cannot_set_the_default_tier_path_directly(self) -> None:
        """Settings.edge.salience_stat's registry owner is SourceProfile (the DEFAULT-tier
        registration), not WatchPlan — a WatchPlan must use the WatchProfile.* EXPLICIT-tier path
        instead (see resolver.py's _expand_explicit_path docstring)."""
        resolver = ConfigResolver()
        plan = _plan(explicit={"Settings.edge.salience_stat": "block_max"})
        with self.assertRaises(ConfigError):
            resolver.resolve(plan)

    def test_watch_plan_cannot_set_gate_field(self) -> None:
        """One Gate: no configuration object may carry gate semantics (Invariants)."""
        resolver = ConfigResolver()
        plan = _plan(explicit={"Settings.gate.threshold": 0.9})
        with self.assertRaises(ConfigError):
            resolver.resolve(plan)

    def test_execution_policy_can_set_its_own_owned_field(self) -> None:
        """A positive control: ExecutionPolicy legitimately owns Settings.cache.enabled."""
        resolver = ConfigResolver()
        policy = ExecutionPolicy(id="ep1", values={"Settings.cache.enabled": True})
        resolved = resolver.resolve(_plan(), execution_policy=policy)
        self.assertTrue(resolved.effective["Settings.cache.enabled"])
        self.assertEqual(resolved.provenance["Settings.cache.enabled"].owner_layer, LAYER_EXECUTION_POLICY)

    def test_unclassified_field_is_permissive(self) -> None:
        """A field genuinely still unclassified in the tunables registry — nothing gates against it.
        (SourceProfile.parameters.coverage_floor, ADR-008's own worked unresolved-ownership example,
        is resolved to owner='SourceProfile' by the implementation -- see TestCoverageFloorOwnership below --
        so this test now exercises a DIFFERENT still-open field.)"""
        resolver = ConfigResolver()
        pack = DomainPack(
            id="dp1", values={"WatchProfile.goal_density_guidance.max_recommended_goals": 5}
        )
        resolved = resolver.resolve(_plan(), domain_pack=pack)
        self.assertEqual(
            resolved.effective["WatchProfile.goal_density_guidance.max_recommended_goals"], 5
        )


class TestUnknownKeyRaises(unittest.TestCase):
    def test_unknown_key_in_watch_plan_explicit_raises(self) -> None:
        resolver = ConfigResolver()
        plan = _plan(explicit={"Totally.Bogus.Path": 1})
        with self.assertRaises(ConfigError) as ctx:
            resolver.resolve(plan)
        self.assertIn("unknown tunable path", str(ctx.exception))

    def test_unknown_key_in_domain_pack_raises(self) -> None:
        resolver = ConfigResolver()
        pack = DomainPack(id="dp1", values={"Nonexistent.Field": 1})
        with self.assertRaises(ConfigError):
            resolver.resolve(_plan(), domain_pack=pack)

    def test_unknown_key_in_optimization_lock_raises(self) -> None:
        resolver = ConfigResolver()
        lock = OptimizationLock(id="lock1", values={"Nonexistent.Field": 1})
        with self.assertRaises(ConfigError):
            resolver.resolve(_plan(), optimization_lock=lock)


class TestExplainRendersEveryEffectiveField(unittest.TestCase):
    def test_explain_covers_every_effective_field(self) -> None:
        resolver = ConfigResolver()
        with mock.patch.dict("os.environ", {"PERCEPT_SALIENCE_STAT": "block_max"}, clear=False):
            resolved = resolver.resolve(_plan())
        rendered = resolved.explain()
        for path in resolved.effective:
            self.assertIn(path, rendered)
        self.assertIn("Deployment", rendered)
        self.assertIn("env:PERCEPT_SALIENCE_STAT", rendered)

    def test_explain_shows_owner_and_source_for_each_line(self) -> None:
        resolver = ConfigResolver()
        resolved = resolver.resolve(_plan(explicit={"WatchProfile.salience_stat": "p95"}))
        rendered = resolved.explain()
        self.assertIn("Settings.edge.salience_stat = 'p95'", rendered)
        self.assertIn("WatchPlan", rendered)
        self.assertIn("watch_plan.explicit", rendered)


class TestHashesStable(unittest.TestCase):
    def test_plan_hash_stable_across_identical_resolves(self) -> None:
        resolver = ConfigResolver()
        plan = _plan(explicit={"WatchProfile.salience_stat": "p95"})
        with mock.patch.dict("os.environ", {}, clear=True):
            r1 = resolver.resolve(plan)
            r2 = resolver.resolve(plan)
        self.assertEqual(r1.plan_hash, r2.plan_hash)
        self.assertEqual(r1.component_snapshot_hash, r2.component_snapshot_hash)

    def test_plan_hash_changes_with_condition(self) -> None:
        resolver = ConfigResolver()
        r1 = resolver.resolve(_plan(condition="a person is at the door"))
        r2 = resolver.resolve(_plan(condition="a dog is at the door"))
        self.assertNotEqual(r1.plan_hash, r2.plan_hash)

    def test_component_snapshot_hash_unaffected_by_non_component_fields(self) -> None:
        """component_snapshot_hash is a content address over ResolvedWatch.components (the
        provider-name identities plus the catalog's own component-set hash) — NOT over every
        resolved Settings.* value. A gate/edge tuning knob like salience_stat moves `effective`
        but names no different provider, so the component snapshot is unaffected."""
        resolver = ConfigResolver()
        plan = _plan()
        with mock.patch.dict("os.environ", {}, clear=True):
            r1 = resolver.resolve(plan)
        with mock.patch.dict("os.environ", {"PERCEPT_SALIENCE_STAT": "block_max"}, clear=False):
            r2 = resolver.resolve(plan)
        self.assertNotEqual(r1.effective["Settings.edge.salience_stat"], r2.effective["Settings.edge.salience_stat"])
        self.assertEqual(r1.component_snapshot_hash, r2.component_snapshot_hash)
        # plan_hash is over the WatchPlan alone, so it is unaffected by the env change.
        self.assertEqual(r1.plan_hash, r2.plan_hash)

    def test_component_snapshot_hash_changes_with_provider_name(self) -> None:
        """A real provider-identity change (a different backend NAME for a seam) DOES move
        component_snapshot_hash — this is exactly the kind of change the field exists to catch."""
        resolver = ConfigResolver()
        plan = _plan()
        with mock.patch.dict("os.environ", {}, clear=True):
            r1 = resolver.resolve(plan)
        with mock.patch.dict("os.environ", {"PERCEPT_VISION_BACKEND": "fake"}, clear=False):
            r2 = resolver.resolve(plan)
        self.assertNotEqual(r1.component_snapshot_hash, r2.component_snapshot_hash)
        self.assertEqual(r1.plan_hash, r2.plan_hash)

    def test_hashes_are_hex_digests(self) -> None:
        resolver = ConfigResolver()
        resolved = resolver.resolve(_plan())
        int(resolved.plan_hash, 16)  # raises ValueError if not hex
        int(resolved.component_snapshot_hash, 16)


class TestSituationResolverProvenanceStamped(unittest.TestCase):
    """The situation resolver's contributed values arrive with real provenance, per the tunables
    registry's own ownership for each underlying field — not a single blanket label."""

    def test_plan_mode_stamped_watch_plan(self) -> None:
        resolver = ConfigResolver()
        resolved = resolver.resolve(_plan(), situation=("dynamic", "fixed", "start_anchored"))
        self.assertEqual(resolved.provenance["WatchProfile.plan_mode"].owner_layer, LAYER_WATCH_PLAN)
        self.assertEqual(resolved.effective["WatchProfile.plan_mode"], "window")

    def test_situation_alone_does_not_touch_salience_stat(self) -> None:
        """ADR-008 ownership transfer: situation axes no longer carry salience_stat — a
        caller must ALSO resolve a SourceProfile (or set it explicitly) to get a tuned statistic.
        The SDK-default layer still contributes the field (every Settings.* path always does), but
        its owner stays SDK and its value stays the legacy default."""
        resolver = ConfigResolver()
        resolved = resolver.resolve(_plan(), situation=("dynamic", "fixed", "start_anchored"))
        self.assertEqual(resolved.effective["Settings.edge.salience_stat"], "mean")
        self.assertEqual(resolved.provenance["Settings.edge.salience_stat"].owner_layer, "SDK")

    def test_gate_field_stamped_gate_layer(self) -> None:
        resolver = ConfigResolver()
        resolved = resolver.resolve(_plan(), situation=("dynamic", "fixed", "transition"))
        self.assertEqual(resolved.provenance["Settings.gate.require_false_first"].owner_layer, LAYER_GATE)
        self.assertTrue(resolved.effective["Settings.gate.require_false_first"])

    def test_ego_compensate_stamped_source_profile_layer(self) -> None:
        resolver = ConfigResolver()
        resolved = resolver.resolve(_plan(), situation=("dynamic", "handheld", "start_anchored"))
        self.assertEqual(resolved.provenance["Settings.edge.ego_compensate"].owner_layer, LAYER_SOURCE_PROFILE)
        self.assertTrue(resolved.effective["Settings.edge.ego_compensate"])

    def test_source_id_names_the_situation_resolver(self) -> None:
        resolver = ConfigResolver()
        resolved = resolver.resolve(_plan(), situation=("stative", "fixed", "start_anchored"))
        self.assertEqual(
            resolved.provenance["WatchProfile.plan_mode"].source_id,
            "profile_resolution.resolve_situation",
        )

    def test_no_situation_no_situation_derived_fields(self) -> None:
        resolver = ConfigResolver()
        resolved = resolver.resolve(_plan())
        self.assertNotIn("WatchProfile.plan_mode", resolved.effective)

    def test_never_selects_sprite(self) -> None:
        """Corrections(a): window beats sprite unconditionally; the situation resolver has no code
        path that can select 'sprite' — confirm that survives through the resolver too."""
        resolver = ConfigResolver()
        for temporality in ("stative", "dynamic", "unknown-garbage", None):
            resolved = resolver.resolve(_plan(), situation=(temporality, "fixed", "start_anchored"))
            self.assertNotEqual(resolved.effective.get("WatchProfile.plan_mode"), "sprite")


class TestOptimizationLock(unittest.TestCase):
    def test_safe_to_optimize_field_accepted(self) -> None:
        resolver = ConfigResolver()
        lock = OptimizationLock(id="lock1", values={"Settings.entity.iou_match": 0.7})
        resolved = resolver.resolve(_plan(), optimization_lock=lock)
        self.assertEqual(resolved.effective["Settings.entity.iou_match"], 0.7)
        self.assertEqual(resolved.provenance["Settings.entity.iou_match"].validated_by, "lock1")

    def test_unsafe_field_rejected(self) -> None:
        """Invariants: 'the optimizer tunes only declared-safe paths' — Settings.gate.threshold is
        safe_to_optimize=False (Gate-owned, high risk)."""
        resolver = ConfigResolver()
        lock = OptimizationLock(id="lock1", values={"Settings.gate.threshold": 0.9})
        with self.assertRaises(ConfigError):
            resolver.resolve(_plan(), optimization_lock=lock)

    def test_lock_beats_execution_policy_and_source_profile(self) -> None:
        resolver = ConfigResolver()
        policy = ExecutionPolicy(id="ep1", values={})
        lock = OptimizationLock(id="lock1", values={"Settings.entity.iou_match": 0.9})
        resolved = resolver.resolve(_plan(), execution_policy=policy, optimization_lock=lock)
        self.assertEqual(resolved.effective["Settings.entity.iou_match"], 0.9)
        self.assertEqual(resolved.provenance["Settings.entity.iou_match"].owner_layer, LAYER_OPTIMIZATION_LOCK)

    def test_env_still_beats_lock(self) -> None:
        """Deployment sits above OptimizationLock in the precedence chain."""
        resolver = ConfigResolver()
        lock = OptimizationLock(id="lock1", values={"Settings.entity.iou_match": 0.9})
        with mock.patch.dict("os.environ", {"PERCEPT_IOU_MATCH": "0.42"}, clear=False):
            resolved = resolver.resolve(_plan(), optimization_lock=lock)
        self.assertEqual(resolved.effective["Settings.entity.iou_match"], 0.42)

    def test_lock_hash_matches_own_values(self) -> None:
        lock = OptimizationLock(id="lock1", values={"Settings.entity.iou_match": 0.7})
        resolver = ConfigResolver()
        resolved = resolver.resolve(_plan(), optimization_lock=lock)
        self.assertEqual(resolved.optimization_lock_hash, lock.content_hash)

    def test_lock_hash_mismatch_emits_loud_diagnostic(self) -> None:
        """Corrections(d): a hash mismatch is reported, never silently trusted."""
        lock = OptimizationLock(
            id="lock1", values={"Settings.entity.iou_match": 0.7}, locked_hash="stale-hash-value",
        )
        resolver = ConfigResolver()
        resolver.resolve(_plan(), optimization_lock=lock)
        codes = [w.code for w in resolver.last_receipt.warnings]
        self.assertIn("optimization_lock_hash_mismatch", codes)

    def test_lock_hash_matching_locked_hash_no_diagnostic(self) -> None:
        values = {"Settings.entity.iou_match": 0.7}
        correct_hash = OptimizationLock(id="lock1", values=values).content_hash
        lock = OptimizationLock(id="lock1", values=values, locked_hash=correct_hash)
        resolver = ConfigResolver()
        resolver.resolve(_plan(), optimization_lock=lock)
        codes = [w.code for w in resolver.last_receipt.warnings]
        self.assertNotIn("optimization_lock_hash_mismatch", codes)

    def test_deployment_override_of_locked_field_emits_loud_diagnostic(self) -> None:
        """C5 / ADR-008 Corrections(d), second half: a Deployment/env change touching a field the
        lock VALIDATED must surface loudly, never silently outrank the lock."""
        resolver = ConfigResolver()
        lock = OptimizationLock(id="lock1", values={"Settings.entity.iou_match": 0.7})
        with mock.patch.dict("os.environ", {"PERCEPT_IOU_MATCH": "0.42"}, clear=False):
            resolved = resolver.resolve(_plan(), optimization_lock=lock)
        self.assertEqual(resolved.effective["Settings.entity.iou_match"], 0.42)
        codes = [w.code for w in resolver.last_receipt.warnings]
        self.assertIn("deployment_overrides_locked_field", codes)
        warning = next(
            w for w in resolver.last_receipt.warnings if w.code == "deployment_overrides_locked_field"
        )
        self.assertEqual(warning.path, "Settings.entity.iou_match")
        self.assertIn("lock1", warning.message)

    def test_deployment_agreeing_with_lock_no_diagnostic(self) -> None:
        """No conflict, no diagnostic — deployment merely confirms the locked value."""
        resolver = ConfigResolver()
        lock = OptimizationLock(id="lock1", values={"Settings.entity.iou_match": 0.7})
        with mock.patch.dict("os.environ", {"PERCEPT_IOU_MATCH": "0.7"}, clear=False):
            resolver.resolve(_plan(), optimization_lock=lock)
        codes = [w.code for w in resolver.last_receipt.warnings]
        self.assertNotIn("deployment_overrides_locked_field", codes)

    def test_no_lock_no_deployment_invalidation_diagnostic(self) -> None:
        resolver = ConfigResolver()
        with mock.patch.dict("os.environ", {"PERCEPT_IOU_MATCH": "0.42"}, clear=False):
            resolver.resolve(_plan())
        codes = [w.code for w in resolver.last_receipt.warnings]
        self.assertNotIn("deployment_overrides_locked_field", codes)

    def test_lock_graduated_provenance_fields_round_trip(self) -> None:
        """the implementation: base_plan_hash / catalog_component_hash / fixture_hash / held_out_metrics /
        created_at graduate the placeholder's id/version/values/locked_hash shape — round-trips
        through to_dict/from_dict, schema_version stamped."""
        lock = OptimizationLock(
            id="lock1",
            values={"Settings.entity.iou_match": 0.7},
            base_plan_hash="planhash123",
            catalog_component_hash="cataloghash456",
            fixture_hash="fixturehash789",
            held_out_metrics={"recall": 0.85, "precision": 0.9},
            created_at="2026-07-22T00:00:00+00:00",
        )
        payload = lock.to_dict()
        self.assertEqual(payload["schema_version"], 1)
        self.assertEqual(payload["base_plan_hash"], "planhash123")
        self.assertEqual(payload["catalog_component_hash"], "cataloghash456")
        self.assertEqual(payload["fixture_hash"], "fixturehash789")
        self.assertEqual(payload["held_out_metrics"], {"recall": 0.85, "precision": 0.9})
        round_tripped = OptimizationLock.from_dict(payload)
        self.assertEqual(round_tripped.base_plan_hash, lock.base_plan_hash)
        self.assertEqual(round_tripped.held_out_metrics, lock.held_out_metrics)
        self.assertEqual(round_tripped.content_hash, lock.content_hash)


class TestSourceProfileComposition(unittest.TestCase):
    def test_real_source_profile_fields_land_at_source_profile_layer(self) -> None:
        resolver = ConfigResolver()
        profile = SourceProfile(
            name="physical_scene",
            modalities=frozenset({"video"}),
            temporal_unit="frame",
            admission_strategy="motion",
            evidence_strategy="frame",
            identity_strategy="appearance",
        )
        resolved = resolver.resolve(_plan(), source_profile=profile)
        self.assertEqual(resolved.effective["SourceProfile.admission_strategy"], "motion")
        self.assertEqual(
            resolved.provenance["SourceProfile.admission_strategy"].owner_layer, LAYER_SOURCE_PROFILE
        )
        self.assertEqual(resolved.provenance["SourceProfile.admission_strategy"].source_id, "physical_scene")

    def test_source_profile_snapshot_on_resolved_watch(self) -> None:
        resolver = ConfigResolver()
        profile = SourceProfile(
            name="physical_scene", modalities=frozenset({"video"}), temporal_unit="frame",
            admission_strategy="motion", evidence_strategy="frame", identity_strategy="appearance",
        )
        resolved = resolver.resolve(_plan(), source_profile=profile)
        self.assertIsNotNone(resolved.source_profile)
        self.assertEqual(resolved.source_profile["SourceProfile.name"], "physical_scene")

    def test_no_source_profile_snapshot_is_none(self) -> None:
        resolver = ConfigResolver()
        resolved = resolver.resolve(_plan())
        self.assertIsNone(resolved.source_profile)


class TestDeploymentSettingsInjection(unittest.TestCase):
    """The `settings=` test-injection seam for what load_settings() would return, without
    mutating real process env."""

    def test_injected_settings_used_instead_of_load_settings(self) -> None:
        resolver = ConfigResolver()
        injected = Settings()
        object.__setattr__(injected, "edge", injected.edge.__class__(
            motion_gate=0.10, salience_stat="p95", ego_compensate=False,
            motion_gate_block_max=0.20, motion_gate_p95=0.15,
        ))
        with mock.patch.dict("os.environ", {"PERCEPT_SALIENCE_STAT": "block_max"}, clear=False):
            resolved = resolver.resolve(_plan(), settings=injected)
        # Real env is set to block_max but is IGNORED because an explicit settings= was injected.
        self.assertEqual(resolved.effective["Settings.edge.salience_stat"], "p95")


class TestReceiptAcceptedFields(unittest.TestCase):
    def test_accepted_fields_cover_effective(self) -> None:
        resolver = ConfigResolver()
        resolved = resolver.resolve(_plan())
        self.assertEqual(set(resolver.last_receipt.accepted_fields), set(resolved.effective))

    def test_status_applied_when_no_warnings(self) -> None:
        resolver = ConfigResolver()
        with mock.patch.dict("os.environ", {}, clear=True):
            resolver.resolve(_plan())
        self.assertEqual(resolver.last_receipt.status, "applied")


class TestOmittedObjectsContributeNothing(unittest.TestCase):
    """ADR-008 §Two-step binding: 'any [object] may be omitted; an omitted object contributes
    nothing, it does not error'."""

    def test_all_optional_inputs_omitted(self) -> None:
        resolver = ConfigResolver()
        resolved = resolver.resolve(_plan())
        self.assertIsNone(resolved.source_profile)
        self.assertIsNone(resolved.domain_pack)
        self.assertIsNone(resolved.execution_policy)
        self.assertIsNone(resolved.optimization_lock_hash)
        self.assertGreater(len(resolved.effective), 0)  # SDK defaults still populate it


if __name__ == "__main__":
    unittest.main()
