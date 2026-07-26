"""ExecutionPolicy (percept/execution_policy.py, percept/execution_policies_default.py) — the implementation.

Covers, per the card's own exit criteria: each of the shipped policies (balanced@1, burst@1,
low_cost@1, privacy_edge@1, strict_admission@1) resolves through ConfigResolver AND compiles through
percept.runtime_compiler.runtime_compile; balanced@1 proves byte-for-byte equality with a
no-execution-policy resolution; a policy writing outside ExecutionPolicy's ownership raises
ConfigError; and the shipped policies are visible via percept.catalog (the same surface
SourceProfiles/WatchProfiles already use).

No runtime execution beyond what ConfigResolver.resolve/runtime_compile already do internally —
this module never touches percept.runtime.worker/percept.agent's runtime paths.
"""

from __future__ import annotations

import unittest

from percept import catalog
from percept.contracts import WatchPlan
from percept.execution_policies_default import (
    BALANCED_V1,
    BURST_V1,
    LOW_COST_V1,
    PRIVACY_EDGE_V1,
)
from percept.execution_policy import (
    DEFAULT_EXECUTION_POLICY_REGISTRY,
    ExecutionPolicy,
    ExecutionPolicyNotFoundError,
    get_execution_policy,
)
from percept.resolver import ConfigError, ConfigResolver, LAYER_EXECUTION_POLICY
from percept.runtime_compiler import runtime_compile
from percept.tunables import REGISTRY

_SHIPPED_NAMES = ("balanced@1", "burst@1", "low_cost@1", "privacy_edge@1", "strict_admission@1")

_OWNER_BY_PATH = {spec.path: spec.owner for spec in REGISTRY}


def _plan() -> WatchPlan:
    return WatchPlan(condition="a person is at the door")


class TestFourPoliciesResolveAndCompile(unittest.TestCase):
    """Exit criterion: 'four named policies resolvable end-to-end with provenance'."""

    def test_all_four_names_registered(self) -> None:
        self.assertEqual(set(DEFAULT_EXECUTION_POLICY_REGISTRY.names()) & set(_SHIPPED_NAMES), set(_SHIPPED_NAMES))

    def test_every_shipped_value_is_execution_policy_owned(self) -> None:
        """A shipped policy touching a path outside ExecutionPolicy's 13-path ownership would be a
        bug in this card, not just a test failure — assert it structurally, not just via resolve()."""
        for name in _SHIPPED_NAMES:
            policy = get_execution_policy(name)
            for path in policy.values:
                owner = _OWNER_BY_PATH.get(path)
                self.assertEqual(
                    owner, "ExecutionPolicy",
                    f"{name} sets {path!r}, owned by {owner!r} not ExecutionPolicy",
                )

    def test_each_policy_resolves_with_provenance(self) -> None:
        for name in _SHIPPED_NAMES:
            policy = get_execution_policy(name)
            resolved = ConfigResolver().resolve(_plan(), execution_policy=policy)
            for path, value in policy.values.items():
                self.assertEqual(resolved.effective[path], value)
                self.assertEqual(resolved.provenance[path].owner_layer, LAYER_EXECUTION_POLICY)
                self.assertEqual(resolved.provenance[path].source_id, policy.id)

    def test_each_policy_compiles_through_runtime_compile(self) -> None:
        for name in _SHIPPED_NAMES:
            policy = get_execution_policy(name)
            resolved = ConfigResolver().resolve(_plan(), execution_policy=policy)
            compiled = runtime_compile(resolved)  # must not raise
            self.assertIsNotNone(compiled.goal)
            self.assertIsNotNone(compiled.settings)

    def test_burst_cadence_lands_on_the_compiled_sense_policy(self) -> None:
        resolved = ConfigResolver().resolve(_plan(), execution_policy=BURST_V1)
        compiled = runtime_compile(resolved)
        self.assertIsNotNone(compiled.sense_policy)
        self.assertEqual(compiled.sense_policy.base_fps, 8.0)
        self.assertEqual(compiled.sense_policy.max_fps, 30.0)

    def test_low_cost_wires_cache_and_routing_settings(self) -> None:
        resolved = ConfigResolver().resolve(_plan(), execution_policy=LOW_COST_V1)
        compiled = runtime_compile(resolved)
        self.assertTrue(compiled.settings.cache.enabled)
        self.assertEqual(compiled.settings.entity.max_entities, 32)
        self.assertEqual(compiled.settings.episodes.max_episodes, 32)

    def test_privacy_edge_turns_down_retention_ceilings(self) -> None:
        resolved = ConfigResolver().resolve(_plan(), execution_policy=PRIVACY_EDGE_V1)
        compiled = runtime_compile(resolved)
        self.assertFalse(compiled.settings.cache.enabled)
        self.assertEqual(compiled.settings.entity.max_entities, 16)
        self.assertEqual(compiled.settings.episodes.max_episodes, 16)
        self.assertEqual(compiled.settings.events.preroll_max_bytes, 16_777_216)


class TestBalancedIsAByteForByteNoOp(unittest.TestCase):
    """Exit criterion: 'balanced@1 proves byte-for-byte equality with no-policy resolution'."""

    def test_effective_matches(self) -> None:
        baseline = ConfigResolver().resolve(_plan())
        with_policy = ConfigResolver().resolve(_plan(), execution_policy=BALANCED_V1)
        self.assertEqual(with_policy.effective, baseline.effective)

    def test_provenance_matches(self) -> None:
        baseline = ConfigResolver().resolve(_plan())
        with_policy = ConfigResolver().resolve(_plan(), execution_policy=BALANCED_V1)
        self.assertEqual(with_policy.provenance, baseline.provenance)

    def test_hashes_match(self) -> None:
        baseline = ConfigResolver().resolve(_plan())
        with_policy = ConfigResolver().resolve(_plan(), execution_policy=BALANCED_V1)
        self.assertEqual(with_policy.plan_hash, baseline.plan_hash)
        self.assertEqual(with_policy.component_snapshot_hash, baseline.component_snapshot_hash)
        self.assertEqual(with_policy.components, baseline.components)

    def test_compiled_watch_matches(self) -> None:
        """The byte-for-byte claim survives all the way through RuntimeCompiler too."""
        baseline = runtime_compile(ConfigResolver().resolve(_plan()))
        with_policy = runtime_compile(ConfigResolver().resolve(_plan(), execution_policy=BALANCED_V1))
        self.assertEqual(with_policy.goal, baseline.goal)
        self.assertEqual(with_policy.settings, baseline.settings)
        self.assertEqual(with_policy.sense_policy, baseline.sense_policy)


class TestOwnershipViolation(unittest.TestCase):
    """Exit criterion: 'ownership violation raises' — tested against a Gate-owned field and a
    SourceProfile-owned field, the same two negative controls test_resolver.py already uses for
    the other placeholder layers."""

    def test_execution_policy_cannot_set_gate_field(self) -> None:
        bad = ExecutionPolicy(id="bad", version="1", values={"Settings.gate.threshold": 0.9})
        with self.assertRaises(ConfigError):
            ConfigResolver().resolve(_plan(), execution_policy=bad)

    def test_execution_policy_cannot_set_source_profile_field(self) -> None:
        bad = ExecutionPolicy(id="bad", version="1", values={"Settings.entity.iou_match": 0.9})
        with self.assertRaises(ConfigError):
            ConfigResolver().resolve(_plan(), execution_policy=bad)

    def test_execution_policy_cannot_set_deployment_field(self) -> None:
        bad = ExecutionPolicy(id="bad", version="1", values={"Settings.models.vision": "claude"})
        with self.assertRaises(ConfigError):
            ConfigResolver().resolve(_plan(), execution_policy=bad)

    def test_unknown_path_raises(self) -> None:
        bad = ExecutionPolicy(id="bad", version="1", values={"Settings.not_a_real_field": 1})
        with self.assertRaises(ConfigError):
            ConfigResolver().resolve(_plan(), execution_policy=bad)


class TestCatalogSurfacing(unittest.TestCase):
    """Exit criterion: 'policies visible via percept components/catalog... match how SourceProfiles
    surface' — same _descriptors()-builder pattern as ROLE_SENSE/ROLE_PROFILE."""

    def test_execution_policy_role_exists(self) -> None:
        self.assertIn(catalog.ROLE_EXECUTION_POLICY, catalog.ROLES)

    def test_all_four_shipped_policies_listed(self) -> None:
        names = {c.name for c in catalog.list_components(role=catalog.ROLE_EXECUTION_POLICY)}
        self.assertEqual(names, set(_SHIPPED_NAMES))

    def test_descriptor_carries_no_measured_field(self) -> None:
        """'Measured performance is never a descriptor field' (catalog.py module docstring)."""
        for c in catalog.list_components(role=catalog.ROLE_EXECUTION_POLICY):
            self.assertNotIn("metrics", c.metadata)
            self.assertNotIn("recall", c.metadata)
            self.assertNotIn("precision", c.metadata)

    def test_snapshot_includes_execution_policies(self) -> None:
        snap = catalog.snapshot()
        by_role = snap.by_role(catalog.ROLE_EXECUTION_POLICY)
        self.assertEqual(len(by_role), len(_SHIPPED_NAMES))


class TestRegistryLookup(unittest.TestCase):
    def test_unknown_name_raises_not_found(self) -> None:
        with self.assertRaises(ExecutionPolicyNotFoundError):
            get_execution_policy("not_a_real_policy@1")

    def test_qualified_name_round_trips(self) -> None:
        policy = ExecutionPolicy(id="foo", version="2")
        self.assertEqual(policy.qualified_name, "foo@2")


if __name__ == "__main__":
    unittest.main()
