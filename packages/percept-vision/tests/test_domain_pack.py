"""DomainPack contract + facility-observation@1 (percept/domain_pack.py, percept/packs_default.py).

Covers the public exit criteria: install -> inspect (catalog/CLI) -> compose with
source+execution -> watch_template arm -> resolve end-to-end, with ZERO edits to the core runtime
modules (percept.behavior/percept.gate/percept.agent/percept.runtime_compiler/
percept.watchplan_apply) — this pack composes with them through their EXISTING public API only.

No runtime execution beyond the offline scripted seam (percept.agent.Percept.create(mode="offline")
+ percept.watchplan_apply.apply_watch_plan) — no network, no API key required.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

from percept import behavior as behavior_mod
from percept import catalog
from percept.agent import Percept
from percept.contracts.watchplan import WatchPlan
from percept.direct_judge import DirectRouting
from percept.domain_pack import (
    ComponentRecommendation,
    DEFAULT_DOMAIN_PACK_REGISTRY,
    DomainPack,
    DomainPackError,
    DomainPackNotFoundError,
    DomainPackRegistry,
    GoalTemplate,
    discover_domain_pack_entrypoints,
    get_domain_pack,
    watch_template,
)
from percept import domain_pack as domain_pack_mod
from percept.packs_default import (
    EGO_MOTION_STARTER_V1,
    FACILITY_OBSERVATION_V1,
    PHYSICAL_SCENE_STARTER_V1,
    register_ego_motion_entrypoint_pack,
    register_entrypoint_pack,
    register_physical_scene_entrypoint_pack,
)
from percept.resolver import ConfigError, ConfigResolver
from percept.watchplan_apply import apply_watch_plan


# ---------------------------------------------------------------------------
# ComponentRecommendation
# ---------------------------------------------------------------------------


class TestComponentRecommendation(unittest.TestCase):
    def test_valid_recommendation(self) -> None:
        rec = ComponentRecommendation(ref="sense:physical_scene", strength="preferred", reason="x")
        self.assertEqual(rec.to_dict()["ref"], "sense:physical_scene")

    def test_default_strength_is_preferred(self) -> None:
        self.assertEqual(
            ComponentRecommendation(ref="detector:motion-periodicity").strength, "preferred"
        )

    def test_empty_ref_raises(self) -> None:
        with self.assertRaises(DomainPackError):
            ComponentRecommendation(ref="")

    def test_bad_strength_raises(self) -> None:
        with self.assertRaises(DomainPackError):
            ComponentRecommendation(ref="x", strength="mandatory")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# GoalTemplate — closed-schema authoring validation
# ---------------------------------------------------------------------------


class TestGoalTemplateValidation(unittest.TestCase):
    def test_valid_transition_template(self) -> None:
        t = GoalTemplate(
            name="t1",
            condition="x appears",
            behavior={"type": "transition", "from": "absent", "to": "present"},
        )
        self.assertEqual(t.behavior["type"], "transition")

    def test_valid_absence_template(self) -> None:
        t = GoalTemplate(
            name="t2",
            condition="the door is open",
            parameters={"deadline_after_s": 60.0},
            behavior={"rearm": "deadline", "deadline_after_s": "{deadline_after_s}"},
        )
        self.assertIn("deadline_after_s", t.parameters)

    def test_valid_episode_count_template(self) -> None:
        t = GoalTemplate(name="t3", condition="a person enters", behavior={"type": "count"})
        self.assertEqual(t.behavior["type"], "count")

    def test_unknown_behavior_key_raises(self) -> None:
        """A key coerce_authored_behavior does not recognize is an AUTHORING error, raised at
        construction — never silently dropped the way an untrusted runtime submission would be."""
        with self.assertRaises(DomainPackError) as ctx:
            GoalTemplate(name="bad", condition="x", behavior={"bogus_field": 1})
        self.assertIn("bogus_field", str(ctx.exception))

    def test_malformed_transition_raises(self) -> None:
        """type=transition requires both from AND to (coerce_authored_behavior's own rule) — a
        template author who forgets `to` gets a loud construction-time failure."""
        with self.assertRaises(DomainPackError):
            GoalTemplate(name="bad", condition="x", behavior={"type": "transition", "from": "a"})

    def test_negative_deadline_raises(self) -> None:
        with self.assertRaises(DomainPackError):
            GoalTemplate(
                name="bad",
                condition="x",
                parameters={"deadline_after_s": -5.0},
                behavior={"rearm": "deadline", "deadline_after_s": "{deadline_after_s}"},
            )

    def test_unsafe_fire_when_raises(self) -> None:
        with self.assertRaises(DomainPackError):
            GoalTemplate(
                name="bad",
                condition="x",
                behavior={"accumulate": {"fire_when": "__import__('os').system('x')"}},
            )

    def test_bad_consequence_cost_raises(self) -> None:
        with self.assertRaises(DomainPackError):
            GoalTemplate(name="bad", condition="x", consequence={"miss_cost": "catastrophic"})

    def test_bad_episode_policy_raises(self) -> None:
        with self.assertRaises(DomainPackError):
            GoalTemplate(name="bad", condition="x", episode={"policy": "not_a_real_policy"})

    def test_valid_episode_policy(self) -> None:
        t = GoalTemplate(name="t4", condition="x", episode={"policy": "door_interval"})
        self.assertEqual(t.episode["policy"], "door_interval")

    def test_identity_policy_must_be_nonempty_when_declared(self) -> None:
        for invalid in ("", "  ", 7):
            with self.subTest(invalid=invalid):
                with self.assertRaises(DomainPackError):
                    GoalTemplate(
                        name="bad",
                        condition="x",
                        identity_policy=invalid,  # type: ignore[arg-type]
                    )

    def test_empty_condition_raises(self) -> None:
        with self.assertRaises(DomainPackError):
            GoalTemplate(name="t", condition="")

    def test_empty_name_raises(self) -> None:
        with self.assertRaises(DomainPackError):
            GoalTemplate(name="", condition="x")


class TestGoalTemplateRender(unittest.TestCase):
    def test_render_returns_watch_plan(self) -> None:
        t = GoalTemplate(name="t", condition="x is present")
        plan = t.render()
        self.assertIsInstance(plan, WatchPlan)

    def test_render_never_arms_anything(self) -> None:
        """Rendering builds a plain, portable WatchPlan — no Percept core, no goal registry, no
        global state is touched (ADR-008 Invariants: 'templates never auto-arm')."""
        t = GoalTemplate(name="t", condition="x is present")
        before = dict(DEFAULT_DOMAIN_PACK_REGISTRY._packs)
        t.render()
        t.render()
        after = dict(DEFAULT_DOMAIN_PACK_REGISTRY._packs)
        self.assertEqual(before, after)

    def test_typed_param_reference_preserves_type(self) -> None:
        t = GoalTemplate(
            name="t",
            condition="the door is open",
            parameters={"deadline_after_s": 300.0},
            behavior={"rearm": "deadline", "deadline_after_s": "{deadline_after_s}"},
        )
        plan = t.render()
        self.assertEqual(plan.explicit["Goal.state.deadline_after_s"], 300.0)
        self.assertIsInstance(plan.explicit["Goal.state.deadline_after_s"], float)

    def test_override_substitutes(self) -> None:
        t = GoalTemplate(
            name="t",
            condition="a package appears at {zone}",
            parameters={"zone": "doorstep"},
        )
        plan = t.render(zone="loading dock")
        self.assertEqual(plan.condition, "a package appears at loading dock")

    def test_unknown_override_raises(self) -> None:
        t = GoalTemplate(name="t", condition="x", parameters={"a": 1})
        with self.assertRaises(DomainPackError):
            t.render(b=2)

    def test_transition_explicit_paths(self) -> None:
        t = GoalTemplate(
            name="t",
            condition="x appears",
            behavior={"type": "transition", "from": "absent", "to": "present"},
        )
        plan = t.render()
        self.assertEqual(plan.explicit["Goal.state.type"], "transition")
        self.assertEqual(plan.explicit["Goal.state.from"], "absent")
        self.assertEqual(plan.explicit["Goal.state.to"], "present")

    def test_consequence_lands_on_watchplan_consequence(self) -> None:
        t = GoalTemplate(name="t", condition="x", consequence={"miss_cost": "severe"})
        plan = t.render()
        self.assertEqual(plan.consequence["miss_cost"], "severe")
        # consequence is NOT ownership-gated explicit -- it rides the WatchPlan.consequence field.
        self.assertNotIn("Goal.state.miss_cost", plan.explicit)

    def test_identity_policy_renders_to_typed_state_path_and_serializes(self) -> None:
        t = GoalTemplate(
            name="scene",
            condition="the zone is occupied",
            identity_policy=" scene ",
        )
        self.assertEqual(t.identity_policy, "scene")
        self.assertEqual(t.render().explicit["Goal.state.identity_policy"], "scene")
        self.assertEqual(t.to_dict()["identity_policy"], "scene")


# ---------------------------------------------------------------------------
# DomainPack construction
# ---------------------------------------------------------------------------


class TestDomainPackConstruction(unittest.TestCase):
    def test_qualified_name(self) -> None:
        self.assertEqual(DomainPack(id="p1", version="2").qualified_name, "p1@2")

    def test_default_version_is_v1(self) -> None:
        self.assertEqual(DomainPack(id="p1").version, "v1")

    def test_template_key_must_match_name(self) -> None:
        t = GoalTemplate(name="real_name", condition="x")
        with self.assertRaises(DomainPackError):
            DomainPack(id="p1", templates={"wrong_key": t})

    def test_non_template_value_raises(self) -> None:
        with self.assertRaises(DomainPackError):
            DomainPack(id="p1", templates={"x": object()})  # type: ignore[dict-item]

    def test_non_recommendation_component_raises(self) -> None:
        with self.assertRaises(DomainPackError):
            DomainPack(id="p1", components=(object(),))  # type: ignore[arg-type]

    def test_unknown_policy_clamp_key_raises(self) -> None:
        with self.assertRaises(DomainPackError) as ctx:
            DomainPack(id="p1", policy_clamps={"bogus_clamp": 1})
        self.assertIn("bogus_clamp", str(ctx.exception))

    def test_bad_false_negative_cost_min_raises(self) -> None:
        with self.assertRaises(DomainPackError):
            DomainPack(id="p1", policy_clamps={"false_negative_cost_min": "catastrophic"})

    def test_bad_retention_max_days_raises(self) -> None:
        with self.assertRaises(DomainPackError):
            DomainPack(id="p1", policy_clamps={"retention_max_days": -1})

    def test_valid_policy_clamps(self) -> None:
        pack = DomainPack(
            id="p1", policy_clamps={"false_negative_cost_min": "mild", "retention_max_days": 7}
        )
        self.assertEqual(pack.policy_clamps["false_negative_cost_min"], "mild")

    def test_non_direct_routing_raises(self) -> None:
        with self.assertRaises(DomainPackError):
            DomainPack(id="p1", direct_judge_routes=(object(),))  # type: ignore[arg-type]

    def test_direct_routing_accepted(self) -> None:
        pack = DomainPack(id="p1", direct_judge_routes=(DirectRouting(condition_namespace="x."),))
        self.assertEqual(len(pack.direct_judge_routes), 1)

    def test_template_lookup(self) -> None:
        t = GoalTemplate(name="a", condition="x")
        pack = DomainPack(id="p1", templates={"a": t})
        self.assertIs(pack.template("a"), t)

    def test_template_lookup_missing_raises(self) -> None:
        pack = DomainPack(id="p1")
        with self.assertRaises(DomainPackNotFoundError):
            pack.template("nope")

    def test_to_dict_roundtrip_shape(self) -> None:
        pack = DomainPack(
            id="p1",
            templates={"a": GoalTemplate(name="a", condition="x")},
            components=(ComponentRecommendation(ref="sense:physical_scene"),),
            policy_clamps={"false_negative_cost_min": "mild"},
        )
        d = pack.to_dict()
        self.assertEqual(d["id"], "p1")
        self.assertIn("a", d["templates"])
        self.assertEqual(d["components"][0]["ref"], "sense:physical_scene")


# ---------------------------------------------------------------------------
# Registry + entry-point discovery
# ---------------------------------------------------------------------------


class TestDomainPackRegistry(unittest.TestCase):
    def test_register_and_resolve(self) -> None:
        reg = DomainPackRegistry()
        pack = DomainPack(id="x")
        reg.register(pack)
        self.assertIs(reg.resolve("x@v1"), pack)

    def test_duplicate_register_raises(self) -> None:
        """Only a GENUINELY conflicting re-register raises (mirrors ExecutionPolicyRegistry): two
        byte-identical DomainPack(id="x") constructions are equal by value, so registering the
        second is a harmless no-op, not a conflict — use a differing pack to prove the guard."""
        reg = DomainPackRegistry()
        reg.register(DomainPack(id="x", notes="first"))
        with self.assertRaises(DomainPackError):
            reg.register(DomainPack(id="x", notes="different"))

    def test_duplicate_register_overwrite_allowed(self) -> None:
        reg = DomainPackRegistry()
        reg.register(DomainPack(id="x", notes="a"))
        reg.register(DomainPack(id="x", notes="b"), overwrite=True)
        self.assertEqual(reg.resolve("x@v1").notes, "b")

    def test_resolve_missing_raises_with_listing(self) -> None:
        reg = DomainPackRegistry()
        reg.register(DomainPack(id="x"))
        with self.assertRaises(DomainPackNotFoundError) as ctx:
            reg.resolve("y@v1")
        self.assertIn("x@v1", str(ctx.exception))

    def test_get_domain_pack_seeds_shipped_pack(self) -> None:
        pack = get_domain_pack("facility_observation@1")
        self.assertIs(pack, FACILITY_OBSERVATION_V1)

    def test_get_domain_pack_missing_raises(self) -> None:
        with self.assertRaises(DomainPackNotFoundError):
            get_domain_pack("no_such_pack@1")


class TestEntryPointDiscovery(unittest.TestCase):
    """percept.packs entry-point group — same opt-in shape as
    percept.sourceprofile.discover_source_profile_entrypoints: not auto-run by get_domain_pack."""

    def test_register_entrypoint_pack_target_callable(self) -> None:
        """The pyproject.toml-advertised entry-point TARGET, exercised directly against a fresh
        registry (proves the two registration paths -- in-code default and entry-point -- agree)."""
        reg = DomainPackRegistry()
        register_entrypoint_pack(reg.register)
        self.assertIn(FACILITY_OBSERVATION_V1.qualified_name, reg.names())

    def test_official_starter_entrypoint_targets_register_separately(self) -> None:
        reg = DomainPackRegistry()
        register_physical_scene_entrypoint_pack(reg.register)
        self.assertEqual(reg.names(), ["physical_scene_starter@1"])
        register_ego_motion_entrypoint_pack(reg.register)
        self.assertEqual(
            reg.names(),
            ["ego_motion_starter@1", "physical_scene_starter@1"],
        )

    def test_discover_domain_pack_entrypoints_loads_mocked_plugin(self) -> None:
        fake_pack = DomainPack(id="third_party")
        fake_ep = SimpleNamespace(load=lambda: lambda register: register(fake_pack))
        with mock.patch("importlib.metadata.entry_points", return_value=[fake_ep]):
            discover_domain_pack_entrypoints()
        self.assertIn("third_party@v1", DEFAULT_DOMAIN_PACK_REGISTRY.names())
        DEFAULT_DOMAIN_PACK_REGISTRY.unregister("third_party@v1")  # leave global state as found

    def test_bad_plugin_does_not_break_discovery(self) -> None:
        def _raises() -> None:
            raise RuntimeError("boom")

        bad_ep = SimpleNamespace(load=lambda: _raises())
        with mock.patch("importlib.metadata.entry_points", return_value=[bad_ep]):
            discover_domain_pack_entrypoints()  # must not raise


# ---------------------------------------------------------------------------
# watch_template()
# ---------------------------------------------------------------------------


class TestWatchTemplateAPI(unittest.TestCase):
    def test_watch_template_returns_watch_plan(self) -> None:
        plan = watch_template("facility_observation@1/door_left_open")
        self.assertIsInstance(plan, WatchPlan)

    def test_watch_template_bad_ref_raises(self) -> None:
        with self.assertRaises(DomainPackError):
            watch_template("no-slash-here")

    def test_watch_template_unknown_pack_raises(self) -> None:
        with self.assertRaises(DomainPackNotFoundError):
            watch_template("nonexistent@1/x")

    def test_watch_template_unknown_template_raises(self) -> None:
        with self.assertRaises(DomainPackNotFoundError):
            watch_template("facility_observation@1/nonexistent_template")

    def test_watch_template_with_parameters(self) -> None:
        plan = watch_template(
            "facility_observation@1/door_left_open",
            {"door_name": "loading dock door", "deadline_after_s": 120.0},
        )
        self.assertIn("loading dock door", plan.condition)
        self.assertEqual(plan.explicit["Goal.state.hold_s"], 120.0)
        self.assertEqual(plan.explicit["Goal.state.cadence.min_judge_interval"], 15.0)
        self.assertEqual(plan.explicit["Goal.state.cadence.gate_on"], "fixed")
        self.assertNotIn("Goal.state.deadline_after_s", plan.explicit)


# ---------------------------------------------------------------------------
# facility-observation@1 — the shipped pack
# ---------------------------------------------------------------------------


class TestFacilityObservationPack(unittest.TestCase):
    def test_four_templates_present(self) -> None:
        self.assertEqual(
            set(FACILITY_OBSERVATION_V1.templates),
            {"door_left_open", "package_appeared", "standing_water", "zone_occupied"},
        )

    def test_door_left_open_uses_sustained_scene_watch(self) -> None:
        plan = FACILITY_OBSERVATION_V1.template("door_left_open").render()
        self.assertEqual(plan.explicit["Goal.state.hold_s"], 300.0)
        self.assertEqual(plan.explicit["Goal.state.cadence.min_judge_interval"], 15.0)
        self.assertEqual(plan.explicit["Goal.state.cadence.gate_on"], "fixed")
        self.assertEqual(plan.explicit["Goal.state.identity_policy"], "scene")
        self.assertNotIn("Goal.state.rearm", plan.explicit)
        self.assertNotIn("Goal.state.deadline_after_s", plan.explicit)
        self.assertFalse(any(path.startswith("Goal.state.episode.") for path in plan.explicit))

    def test_package_appeared_uses_transition_sense(self) -> None:
        plan = FACILITY_OBSERVATION_V1.template("package_appeared").render()
        self.assertEqual(plan.explicit["Goal.state.type"], "transition")

    def test_standing_water_uses_transition_sense(self) -> None:
        plan = FACILITY_OBSERVATION_V1.template("standing_water").render()
        self.assertEqual(plan.explicit["Goal.state.type"], "transition")

    def test_zone_occupied_uses_contiguous_watch_semantics(self) -> None:
        plan = FACILITY_OBSERVATION_V1.template("zone_occupied").render()
        self.assertNotIn("Goal.state.type", plan.explicit)
        self.assertEqual(plan.explicit["Goal.state.identity_policy"], "scene")
        self.assertEqual(
            FACILITY_OBSERVATION_V1.template("zone_occupied").to_dict()["identity_policy"],
            "scene",
        )
        self.assertIn(
            "contiguous occupied interval", FACILITY_OBSERVATION_V1.template("zone_occupied").notes
        )
        self.assertIn("does not identify", FACILITY_OBSERVATION_V1.template("zone_occupied").notes)

    def test_all_components_are_preferred_not_required(self) -> None:
        """Recommended components remain optional; no ``security@1`` is implied."""
        self.assertTrue(FACILITY_OBSERVATION_V1.components)
        for rec in FACILITY_OBSERVATION_V1.components:
            self.assertEqual(rec.strength, "preferred")

    def test_no_security_pack_shipped(self) -> None:
        self.assertNotIn("security@1", DEFAULT_DOMAIN_PACK_REGISTRY.names())

    def test_policy_clamps_declared(self) -> None:
        self.assertIn("false_negative_cost_min", FACILITY_OBSERVATION_V1.policy_clamps)
        self.assertIn("retention_max_days", FACILITY_OBSERVATION_V1.policy_clamps)

    def test_legacy_candidate_declarations_are_preserved(self) -> None:
        self.assertTrue(FACILITY_OBSERVATION_V1.search_space)
        self.assertEqual(
            FACILITY_OBSERVATION_V1.fixtures,
            (
                "facility_observation/package_appeared",
                "facility_observation/standing_water",
                "facility_observation/zone_occupied",
            ),
        )
        self.assertEqual(
            dict(FACILITY_OBSERVATION_V1.quality_bars),
            {
                "package_appeared.recall_min": 0.60,
                "standing_water.recall_min": 0.55,
                "zone_occupied.recall_min": 0.60,
            },
        )
        self.assertIn("candidate qualification declarations", FACILITY_OBSERVATION_V1.notes)
        self.assertIn("not achieved results", FACILITY_OBSERVATION_V1.notes)

    def test_unmatched_duration_template_has_no_candidate_bar(self) -> None:
        self.assertNotIn(
            "facility_observation/door_left_open",
            FACILITY_OBSERVATION_V1.fixtures,
        )
        self.assertFalse(
            any(key.startswith("door_left_open.") for key in FACILITY_OBSERVATION_V1.quality_bars)
        )

    def test_search_space_paths_are_safe_to_optimize(self) -> None:
        from percept.tunables import REGISTRY

        by_path = {s.path: s for s in REGISTRY}
        for path in FACILITY_OBSERVATION_V1.search_space:
            self.assertIn(path, by_path)
            self.assertTrue(by_path[path].safe_to_optimize, f"{path} must be safe_to_optimize")


# ---------------------------------------------------------------------------
# Catalog surfacing — install -> inspect
# ---------------------------------------------------------------------------


class TestCatalogSurfacing(unittest.TestCase):
    def test_domain_pack_role_in_catalog_roles(self) -> None:
        self.assertIn("domain_pack", catalog.ROLES)

    def test_facility_observation_listed(self) -> None:
        names = [c.name for c in catalog.list_components(role="domain_pack")]
        self.assertIn("facility_observation@1", names)
        self.assertIn("physical_scene_starter@1", names)
        self.assertIn("ego_motion_starter@1", names)

    def test_describe_via_cli_grammar(self) -> None:
        matches = [
            c
            for c in catalog.list_components(role="domain_pack")
            if c.name == "facility_observation@1"
        ]
        self.assertEqual(len(matches), 1)
        self.assertIn("templates", matches[0].metadata)
        self.assertEqual(
            set(matches[0].metadata["templates"]), set(FACILITY_OBSERVATION_V1.templates)
        )

    def test_official_starters_expose_no_qualification_claims(self) -> None:
        for pack in (PHYSICAL_SCENE_STARTER_V1, EGO_MOTION_STARTER_V1):
            with self.subTest(pack=pack.qualified_name):
                payload = pack.to_dict()
                self.assertEqual(payload["fixtures"], [])
                self.assertEqual(payload["quality_bars"], {})
                self.assertIn("not qualification", pack.notes)

    def test_snapshot_includes_domain_pack(self) -> None:
        snap = catalog.snapshot()
        self.assertTrue(any(c.role == "domain_pack" for c in snap.components))


# ---------------------------------------------------------------------------
# Resolver: ownership gate (unchanged) + the NEW policy-clamp gate
# ---------------------------------------------------------------------------


class TestResolverDomainPackOwnership(unittest.TestCase):
    def test_domain_pack_values_still_ownership_gated(self) -> None:
        """Unchanged behavior from the earlier placeholder: a DomainPack cannot set a
        Deployment-owned field."""
        resolver = ConfigResolver()
        pack = DomainPack(id="dp1", values={"Settings.models.vision": "claude"})
        plan = WatchPlan(condition="x")
        with self.assertRaises(ConfigError):
            resolver.resolve(plan, domain_pack=pack)

    def test_domain_pack_snapshot_carries_more_than_values(self) -> None:
        resolver = ConfigResolver()
        pack = DomainPack(
            id="dp1",
            components=(ComponentRecommendation(ref="sense:physical_scene"),),
            policy_clamps={"false_negative_cost_min": "mild"},
        )
        resolved = resolver.resolve(WatchPlan(condition="x"), domain_pack=pack)
        self.assertEqual(resolved.domain_pack["id"], "dp1")
        self.assertEqual(len(resolved.domain_pack["components"]), 1)
        self.assertEqual(resolved.domain_pack["policy_clamps"]["false_negative_cost_min"], "mild")


class TestResolverPolicyClamps(unittest.TestCase):
    """Policy-clamp enforcement: a RESOLVE-TIME ConfigError, never a silent merge/clamp
    (ADR-008's 'enforced like the envelope' posture)."""

    def test_false_negative_cost_min_violation_raises(self) -> None:
        resolver = ConfigResolver()
        pack = DomainPack(id="dp1", policy_clamps={"false_negative_cost_min": "severe"})
        plan = WatchPlan(condition="x", consequence={"miss_cost": "mild"})
        with self.assertRaises(ConfigError) as ctx:
            resolver.resolve(plan, domain_pack=pack)
        self.assertIn("false_negative_cost_min", str(ctx.exception))

    def test_false_negative_cost_min_satisfied_passes(self) -> None:
        resolver = ConfigResolver()
        pack = DomainPack(id="dp1", policy_clamps={"false_negative_cost_min": "mild"})
        plan = WatchPlan(condition="x", consequence={"miss_cost": "severe"})
        resolved = resolver.resolve(plan, domain_pack=pack)
        self.assertIsNotNone(resolved)

    def test_false_negative_cost_min_no_consequence_does_not_raise(self) -> None:
        """An unset consequence does not participate in the clamp (nothing to violate) — the same
        'unset field does not participate' posture the rest of ADR-008 precedence takes."""
        resolver = ConfigResolver()
        pack = DomainPack(id="dp1", policy_clamps={"false_negative_cost_min": "severe"})
        resolved = resolver.resolve(WatchPlan(condition="x"), domain_pack=pack)
        self.assertIsNotNone(resolved)

    def test_retention_max_days_violation_raises(self) -> None:
        from percept.config import Settings
        import dataclasses as dc

        resolver = ConfigResolver()
        pack = DomainPack(id="dp1", policy_clamps={"retention_max_days": 1.0})
        settings = Settings()
        settings = dc.replace(
            settings, episodes=dc.replace(settings.episodes, window_s=10 * 86400.0)
        )
        with self.assertRaises(ConfigError) as ctx:
            resolver.resolve(WatchPlan(condition="x"), domain_pack=pack, settings=settings)
        self.assertIn("retention_max_days", str(ctx.exception))

    def test_retention_max_days_within_bound_passes(self) -> None:
        resolver = ConfigResolver()
        pack = DomainPack(id="dp1", policy_clamps={"retention_max_days": 30.0})
        resolved = resolver.resolve(WatchPlan(condition="x"), domain_pack=pack)
        self.assertIsNotNone(resolved)

    def test_no_domain_pack_no_clamp_check(self) -> None:
        resolver = ConfigResolver()
        resolved = resolver.resolve(WatchPlan(condition="x", consequence={"miss_cost": "trivial"}))
        self.assertIsNotNone(resolved)

    def test_facility_observation_pack_rejects_trivial_miss_cost(self) -> None:
        resolver = ConfigResolver()
        plan = watch_template("facility_observation@1/door_left_open")
        from dataclasses import replace as dc_replace

        plan = dc_replace(plan, consequence={"miss_cost": "trivial"})
        with self.assertRaises(ConfigError):
            resolver.resolve(plan, domain_pack=FACILITY_OBSERVATION_V1)

    def test_facility_observation_pack_accepts_own_template_defaults(self) -> None:
        """Every shipped template's own consequence default must satisfy the pack's own clamp —
        an authoring self-consistency check, not just a unit test of the clamp mechanism."""
        resolver = ConfigResolver()
        for name in FACILITY_OBSERVATION_V1.templates:
            plan = watch_template(f"facility_observation@1/{name}")
            resolved = resolver.resolve(plan, domain_pack=FACILITY_OBSERVATION_V1)
            self.assertIsNotNone(resolved)


# ---------------------------------------------------------------------------
# End-to-end: install -> inspect -> compose -> watch_template -> arm -> resolve
# ---------------------------------------------------------------------------


class TestEndToEndArm(unittest.TestCase):
    """apply_watch_plan(core, watch_template(...), domain_pack=..., execution_policy=...) — the
    full compose-with-source+execution -> arm path, over the real (offline, zero-spend) runtime."""

    def test_arm_door_left_open_on_real_core(self) -> None:
        core = Percept.create(mode="offline", discover_plugins=False)
        plan = watch_template(
            "facility_observation@1/door_left_open",
            {"door_name": "back door", "deadline_after_s": 45.0},
        )
        receipt = apply_watch_plan(core, plan, domain_pack=FACILITY_OBSERVATION_V1)
        self.assertEqual(receipt.status, "applied")
        self.assertEqual(len(receipt.watch_ids), 1)
        goal = core.goals.get(receipt.watch_ids[0])
        self.assertEqual(goal.state["hold_s"], 45.0)
        self.assertEqual(
            goal.state["cadence"],
            {"min_judge_interval": 15.0, "gate_on": "fixed"},
        )
        self.assertEqual(goal.state["identity_policy"], "scene")
        self.assertNotIn("rearm", goal.state)
        self.assertNotIn("deadline_after_s", goal.state)
        self.assertNotIn("episode", goal.state)
        self.assertIn("back door", goal.condition)

    def test_arm_package_appeared_on_real_core(self) -> None:
        core = Percept.create(mode="offline", discover_plugins=False)
        plan = watch_template("facility_observation@1/package_appeared")
        receipt = apply_watch_plan(core, plan, domain_pack=FACILITY_OBSERVATION_V1)
        self.assertEqual(receipt.status, "applied")
        goal = core.goals.get(receipt.watch_ids[0])
        self.assertEqual(goal.state["type"], "transition")
        self.assertEqual(goal.state["from"], "no package is present")

    def test_arm_zone_occupied_on_real_core(self) -> None:
        core = Percept.create(mode="offline", discover_plugins=False)
        plan = watch_template("facility_observation@1/zone_occupied", {"zone_name": "server room"})
        receipt = apply_watch_plan(core, plan, domain_pack=FACILITY_OBSERVATION_V1)
        self.assertEqual(receipt.status, "applied")
        goal = core.goals.get(receipt.watch_ids[0])
        self.assertEqual(goal.state["type"], "watch")
        self.assertIn("server room", goal.condition)

    def test_arm_rejected_when_clamp_violated(self) -> None:
        """apply_watch_plan surfaces a resolve-time ConfigError as a 'rejected' receipt — nothing
        arms (no goal_id), the FULL batch/plan semantics apply_watch_plan already gives every
        ConfigError, unmodified."""
        core = Percept.create(mode="offline", discover_plugins=False)
        from dataclasses import replace as dc_replace

        plan = watch_template("facility_observation@1/door_left_open")
        plan = dc_replace(plan, consequence={"miss_cost": "trivial"})
        receipt = apply_watch_plan(core, plan, domain_pack=FACILITY_OBSERVATION_V1)
        self.assertEqual(receipt.status, "rejected")
        self.assertEqual(receipt.watch_ids, ())
        self.assertEqual(len(core.goals.all()), 0)

    def test_execution_policy_composes_alongside_domain_pack(self) -> None:
        """DomainPack + ExecutionPolicy resolve together onto the SAME ResolvedWatch — the
        precedence chain (SourceProfile < DomainPack < ExecutionPolicy < ...) composes rather than
        one layer excluding the other. Core-scoped settings must be selected when constructing the
        core; applying a conflicting per-Watch value is rejected rather than silently inert."""
        from percept.execution_policies_default import LOW_COST_V1

        resolver = ConfigResolver()
        plan = watch_template("facility_observation@1/standing_water")
        resolved = resolver.resolve(
            plan, domain_pack=FACILITY_OBSERVATION_V1, execution_policy=LOW_COST_V1
        )
        self.assertTrue(
            resolved.effective["Settings.cache.enabled"]
        )  # LOW_COST_V1's own contribution
        self.assertIsNotNone(resolved.domain_pack)
        self.assertIsNotNone(resolved.execution_policy)

        core = Percept.create(mode="offline", discover_plugins=False)
        receipt = apply_watch_plan(
            core, plan, domain_pack=FACILITY_OBSERVATION_V1, execution_policy=LOW_COST_V1
        )
        self.assertEqual(receipt.status, "rejected")
        self.assertEqual(
            receipt.dropped_fields[0].code,
            "core_scoped_setting_requires_separate_runtime",
        )

        configured = Percept.configure(
            mode="offline",
            discover_plugins=False,
            execution_policy=LOW_COST_V1,
        )
        receipt = apply_watch_plan(
            configured,
            plan,
            domain_pack=FACILITY_OBSERVATION_V1,
            execution_policy=LOW_COST_V1,
        )
        self.assertEqual(receipt.status, "applied")
        self.assertTrue(configured.settings.cache.enabled)


# ---------------------------------------------------------------------------
# "Templates never auto-arm" — the ADR-008 invariant, checked directly.
# ---------------------------------------------------------------------------


class TestTemplatesNeverAutoArm(unittest.TestCase):
    def test_watch_template_alone_arms_nothing(self) -> None:
        core = Percept.create(mode="offline", discover_plugins=False)
        watch_template("facility_observation@1/door_left_open")
        watch_template("facility_observation@1/package_appeared")
        self.assertEqual(len(core.goals.all()), 0)

    def test_get_domain_pack_alone_arms_nothing(self) -> None:
        core = Percept.create(mode="offline", discover_plugins=False)
        get_domain_pack("facility_observation@1")
        self.assertEqual(len(core.goals.all()), 0)

    def test_resolve_alone_arms_nothing(self) -> None:
        """ConfigResolver.resolve() (step one of the two-step binder) never touches any Percept
        core — arming is always a separate, explicit apply_watch_plan/add_goal call."""
        core = Percept.create(mode="offline", discover_plugins=False)
        resolver = ConfigResolver()
        plan = watch_template("facility_observation@1/door_left_open")
        resolver.resolve(plan, domain_pack=FACILITY_OBSERVATION_V1)
        self.assertEqual(len(core.goals.all()), 0)

    def test_pack_construction_arms_nothing(self) -> None:
        core = Percept.create(mode="offline", discover_plugins=False)
        DomainPack(
            id="p",
            templates={"t": GoalTemplate(name="t", condition="x")},
        )
        self.assertEqual(len(core.goals.all()), 0)


# ---------------------------------------------------------------------------
# Zero-core-edits proof: this module composes with the EXISTING core coercers/senses by identity
# (never a reimplementation), and the full pipeline runs end-to-end through unmodified public API.
# ---------------------------------------------------------------------------


class TestZeroCoreEditsProof(unittest.TestCase):
    def test_domain_pack_reuses_the_real_behavior_coercers_by_identity(self) -> None:
        """percept.domain_pack imports (not reimplements) percept.behavior's own closed-vocabulary
        firewall — the SAME function objects, proven by identity, not just by matching output."""
        self.assertIs(
            domain_pack_mod.coerce_authored_behavior, behavior_mod.coerce_authored_behavior
        )
        self.assertIs(domain_pack_mod.coerce_authored_episode, behavior_mod.coerce_authored_episode)

    def test_watch_plan_type_is_the_real_frozen_contract(self) -> None:
        """GoalTemplate.render() returns THE real percept.contracts.watchplan.WatchPlan — not a
        pack-local shadow type — so it composes with ConfigResolver/apply_watch_plan unmodified."""
        from percept.contracts.watchplan import WatchPlan as RealWatchPlan

        plan = FACILITY_OBSERVATION_V1.template("door_left_open").render()
        self.assertIs(type(plan), RealWatchPlan)

    def test_direct_routing_type_is_the_real_direct_judge_type(self) -> None:
        from percept.direct_judge import DirectRouting as RealDirectRouting

        for route in FACILITY_OBSERVATION_V1.direct_judge_routes:
            self.assertIsInstance(route, RealDirectRouting)

    def test_senses_compile_via_the_unmodified_program_of(self) -> None:
        """The four templates' rendered Goal.state, run through percept.behavior.program_of
        (unmodified), select the sense the template's docstring claims — the load-bearing proof
        that this pack composes with the EXISTING sense vocabulary rather than inventing a new one."""
        core = Percept.create(mode="offline", discover_plugins=False)
        expected_sense = {
            "door_left_open": "judge",
            "package_appeared": "transition",
            "standing_water": "transition",
            "zone_occupied": "judge",
        }
        for name, sense in expected_sense.items():
            plan = watch_template(f"facility_observation@1/{name}")
            receipt = apply_watch_plan(core, plan, domain_pack=FACILITY_OBSERVATION_V1)
            goal = core.goals.get(receipt.watch_ids[0])
            program = behavior_mod.program_of(goal)
            self.assertEqual(program.sense, sense, f"{name} expected sense={sense!r}")


if __name__ == "__main__":
    unittest.main()
