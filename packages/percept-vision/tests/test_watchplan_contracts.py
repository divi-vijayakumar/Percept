"""ADR-008 DTOs — WatchPlan / SettingOrigin / ResolvedWatch / WatchReceipt / FieldDiagnostic.

Stdlib only; no resolver, no runtime execution (that is ``percept.resolver``, tested separately in
``tests/test_resolver.py``). This file only pins the DTO shapes: construction, immutability,
validation, ``to_dict()``/``explain()`` rendering, and re-export parity across
``percept.contracts`` / ``percept.adapters``.
"""

from __future__ import annotations

import unittest
from types import MappingProxyType

from percept.contracts import (
    RECEIPT_STATUSES,
    CONTRACT_VERSION,
    FieldDiagnostic,
    ResolvedWatch,
    SettingOrigin,
    WatchPlan,
    WatchReceipt,
)


class TestContractVersion(unittest.TestCase):
    def test_contract_version_bumped_minor(self) -> None:
        major, minor, _patch = (int(x) for x in CONTRACT_VERSION.split("."))
        self.assertEqual(major, 1)
        self.assertGreaterEqual(minor, 8)


class TestWatchPlan(unittest.TestCase):
    def test_condition_required_nonempty(self) -> None:
        with self.assertRaises(ValueError):
            WatchPlan(condition="")
        with self.assertRaises(ValueError):
            WatchPlan(condition="   ")

    def test_condition_carried_verbatim(self) -> None:
        plan = WatchPlan(condition="a person is at the door")
        self.assertEqual(plan.condition, "a person is at the door")

    def test_defaults_are_unset_so_nothing_participates(self) -> None:
        plan = WatchPlan(condition="x")
        self.assertIsNone(plan.sense)
        self.assertIsNone(plan.admission)
        self.assertEqual(plan.explicit, {})
        self.assertEqual(plan.detectors, ())
        self.assertEqual(plan.source, "camera")

    def test_explicit_is_frozen_mapping(self) -> None:
        plan = WatchPlan(condition="x", explicit={"Settings.edge.salience_stat": "block_max"})
        self.assertIsInstance(plan.explicit, MappingProxyType)
        with self.assertRaises(TypeError):
            plan.explicit["new_key"] = "value"  # type: ignore[index]

    def test_explicit_keys_must_be_nonempty_strings(self) -> None:
        with self.assertRaises(ValueError):
            WatchPlan(condition="x", explicit={"": "block_max"})

    def test_detectors_and_extractors_normalize_to_tuples(self) -> None:
        plan = WatchPlan(condition="x", detectors=["motion", "acoustic"], extractors=["ocr"])
        self.assertEqual(plan.detectors, ("motion", "acoustic"))
        self.assertEqual(plan.extractors, ("ocr",))

    def test_consequence_and_schedule_must_be_mapping_or_none(self) -> None:
        with self.assertRaises(ValueError):
            WatchPlan(condition="x", consequence="mild")  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            WatchPlan(condition="x", schedule="always")  # type: ignore[arg-type]
        plan = WatchPlan(condition="x", consequence={"miss_cost": "severe"}, schedule={"days": [1, 2]})
        self.assertEqual(plan.consequence, {"miss_cost": "severe"})

    def test_is_frozen(self) -> None:
        plan = WatchPlan(condition="x")
        with self.assertRaises(Exception):
            plan.condition = "y"  # type: ignore[misc]

    def test_to_dict_roundtrips_shape(self) -> None:
        plan = WatchPlan(
            condition="a cup appears on the desk",
            sense="vision",
            explicit={"Settings.edge.salience_stat": "block_max"},
            detectors=["motion"],
        )
        d = plan.to_dict()
        self.assertEqual(d["condition"], plan.condition)
        self.assertEqual(d["explicit"], {"Settings.edge.salience_stat": "block_max"})
        self.assertEqual(d["detectors"], ["motion"])
        self.assertIsInstance(d["detectors"], list)  # JSON-shaped, not a tuple


class TestFieldDiagnostic(unittest.TestCase):
    def test_code_required_never_prose_only(self) -> None:
        with self.assertRaises(ValueError):
            FieldDiagnostic(path="Settings.edge.salience_stat", code="")

    def test_path_required(self) -> None:
        with self.assertRaises(ValueError):
            FieldDiagnostic(path="", code="unknown_key")

    def test_minimal_construction(self) -> None:
        d = FieldDiagnostic(path="Settings.edge.salience_stat", code="coerced")
        self.assertIsNone(d.goal_id)
        self.assertEqual(d.message, "")
        self.assertEqual(
            d.to_dict(),
            {"path": "Settings.edge.salience_stat", "code": "coerced", "goal_id": None, "message": ""},
        )

    def test_goal_scoped(self) -> None:
        d = FieldDiagnostic(path="Goal.state.hold_s", code="dropped", goal_id="g1", message="not positive")
        self.assertEqual(d.goal_id, "g1")


class TestSettingOrigin(unittest.TestCase):
    def test_construction_and_to_dict(self) -> None:
        origin = SettingOrigin(
            value="block_max",
            owner_layer="Deployment",
            source_id="env:PERCEPT_SALIENCE_STAT",
            reason="env beats SourceProfile default",
        )
        self.assertIsNone(origin.validated_by)
        self.assertEqual(
            origin.to_dict(),
            {
                "value": "block_max",
                "owner_layer": "Deployment",
                "source_id": "env:PERCEPT_SALIENCE_STAT",
                "reason": "env beats SourceProfile default",
                "validated_by": None,
            },
        )

    def test_is_frozen(self) -> None:
        origin = SettingOrigin(value=1, owner_layer="SDK", source_id="sdk_default", reason="baseline")
        with self.assertRaises(Exception):
            origin.value = 2  # type: ignore[misc]


def _resolved_watch(**overrides) -> ResolvedWatch:
    plan = overrides.pop("plan", WatchPlan(condition="a person is at the door"))
    effective = overrides.pop("effective", {"Settings.edge.salience_stat": "block_max"})
    provenance = overrides.pop(
        "provenance",
        {
            "Settings.edge.salience_stat": SettingOrigin(
                value="block_max", owner_layer="SourceProfile", source_id="physical_scene",
                reason="profile default",
            )
        },
    )
    kwargs = dict(
        plan=plan,
        source_profile=None,
        domain_pack=None,
        execution_policy=None,
        components={},
        effective=effective,
        provenance=provenance,
        plan_hash="deadbeef" * 8,
        component_snapshot_hash="cafebabe" * 8,
    )
    kwargs.update(overrides)
    return ResolvedWatch(**kwargs)


class TestResolvedWatch(unittest.TestCase):
    def test_effective_and_provenance_keys_must_match(self) -> None:
        with self.assertRaises(ValueError):
            _resolved_watch(effective={"a": 1, "b": 2}, provenance={
                "a": SettingOrigin(value=1, owner_layer="SDK", source_id="sdk_default", reason="baseline")
            })

    def test_provenance_values_must_be_setting_origin(self) -> None:
        with self.assertRaises(ValueError):
            _resolved_watch(
                effective={"a": 1},
                provenance={"a": "not-an-origin"},  # type: ignore[dict-item]
            )

    def test_plan_hash_and_component_hash_required(self) -> None:
        with self.assertRaises(ValueError):
            _resolved_watch(plan_hash="")
        with self.assertRaises(ValueError):
            _resolved_watch(component_snapshot_hash="")

    def test_plan_must_be_watchplan(self) -> None:
        with self.assertRaises(ValueError):
            _resolved_watch(plan="not-a-plan")  # type: ignore[arg-type]

    def test_is_frozen_and_mappings_immutable(self) -> None:
        rw = _resolved_watch()
        with self.assertRaises(Exception):
            rw.plan_hash = "x"  # type: ignore[misc]
        with self.assertRaises(TypeError):
            rw.effective["new"] = 1  # type: ignore[index]
        with self.assertRaises(TypeError):
            rw.provenance["new"] = None  # type: ignore[index]

    def test_optimization_lock_hash_defaults_none(self) -> None:
        rw = _resolved_watch()
        self.assertIsNone(rw.optimization_lock_hash)

    def test_explain_renders_every_effective_field_with_owner_and_source(self) -> None:
        rw = _resolved_watch(
            effective={
                "Settings.edge.salience_stat": "block_max",
                "Settings.edge.motion_gate_block_max": 0.2,
            },
            provenance={
                "Settings.edge.salience_stat": SettingOrigin(
                    value="block_max", owner_layer="Deployment", source_id="env:PERCEPT_SALIENCE_STAT",
                    reason="env beats profile default",
                ),
                "Settings.edge.motion_gate_block_max": SettingOrigin(
                    value=0.2, owner_layer="SourceProfile", source_id="physical_scene",
                    reason="profile default",
                ),
            },
        )
        rendered = rw.explain()
        self.assertIn("Settings.edge.salience_stat = 'block_max'", rendered)
        self.assertIn("Deployment", rendered)
        self.assertIn("env:PERCEPT_SALIENCE_STAT", rendered)
        self.assertIn("Settings.edge.motion_gate_block_max = 0.2", rendered)
        self.assertIn("SourceProfile", rendered)
        # every effective field must appear
        for path in rw.effective:
            self.assertIn(path, rendered)

    def test_to_dict_shape(self) -> None:
        rw = _resolved_watch()
        d = rw.to_dict()
        self.assertEqual(d["plan"]["condition"], "a person is at the door")
        self.assertEqual(d["effective"], {"Settings.edge.salience_stat": "block_max"})
        self.assertIn("Settings.edge.salience_stat", d["provenance"])
        self.assertEqual(d["provenance"]["Settings.edge.salience_stat"]["owner_layer"], "SourceProfile")
        self.assertEqual(d["plan_hash"], "deadbeef" * 8)
        self.assertIsNone(d["optimization_lock_hash"])

    def test_snapshot_fields_are_frozen_when_provided(self) -> None:
        rw = _resolved_watch(source_profile={"name": "physical_scene"}, components={"vision": "claude"})
        self.assertIsInstance(rw.source_profile, MappingProxyType)
        with self.assertRaises(TypeError):
            rw.source_profile["x"] = 1  # type: ignore[index]
        self.assertEqual(dict(rw.source_profile), {"name": "physical_scene"})


class TestWatchReceipt(unittest.TestCase):
    def test_status_must_be_closed_vocab(self) -> None:
        with self.assertRaises(ValueError):
            WatchReceipt(status="ok")  # type: ignore[arg-type]
        for status in RECEIPT_STATUSES:
            WatchReceipt(status=status)  # type: ignore[arg-type]

    def test_defaults(self) -> None:
        r = WatchReceipt(status="applied")
        self.assertEqual(r.watch_ids, ())
        self.assertEqual(r.dropped_fields, ())
        self.assertIsNone(r.plan_revision)

    def test_diagnostic_groups_must_contain_only_field_diagnostics(self) -> None:
        with self.assertRaises(ValueError):
            WatchReceipt(status="partial", dropped_fields=("not-a-diagnostic",))  # type: ignore[arg-type]

    def test_applied_with_warnings_carries_warnings(self) -> None:
        warn = FieldDiagnostic(
            path="Settings.edge.salience_stat", code="env_overrides_explicit_profile",
            message="env PERCEPT_SALIENCE_STAT overrode explicitly-selected SourceProfile 'physical_scene'",
        )
        r = WatchReceipt(status="applied_with_warnings", warnings=(warn,))
        self.assertEqual(len(r.warnings), 1)
        self.assertEqual(r.warnings[0].code, "env_overrides_explicit_profile")

    def test_to_dict_shape(self) -> None:
        dropped = FieldDiagnostic(path="Bogus.field", code="unknown_key")
        r = WatchReceipt(status="partial", watch_ids=("g1",), dropped_fields=(dropped,))
        d = r.to_dict()
        self.assertEqual(d["watch_ids"], ["g1"])
        self.assertEqual(d["dropped_fields"], [dropped.to_dict()])

    def test_is_frozen(self) -> None:
        r = WatchReceipt(status="applied")
        with self.assertRaises(Exception):
            r.status = "rejected"  # type: ignore[misc]


class TestAdaptersReexport(unittest.TestCase):
    """percept.adapters is the backward-compatible re-export shim; every new DTO must round-trip
    through it identically to percept.contracts (same object, not a copy)."""

    def test_watchplan_family_reexported_identically(self) -> None:
        import percept.adapters as adapters
        import percept.contracts as contracts

        for name in (
            "WatchPlan", "SettingOrigin", "ResolvedWatch", "WatchReceipt", "FieldDiagnostic",
            "ReceiptStatus", "RECEIPT_STATUSES",
        ):
            self.assertIs(getattr(adapters, name), getattr(contracts, name), name)


if __name__ == "__main__":
    unittest.main()
