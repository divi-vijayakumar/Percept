"""Pins the SDK's public export surface — ``percept.__all__`` plus the two submodules this card
de-exported dark seams from (``percept.sensorplan``, ``percept.domain_pack``) — so a name silently
added or removed fails CI instead of shipping unreviewed.

Why this exists: ``percept.__all__`` is the 1.0 API promise. Exporting a name commits the SDK to
supporting it; an SDK that grows its surface by accident (a new import quietly added to ``__all__``
in an unrelated PR) breaks that promise just as surely as one that removes a name without notice.
This test makes BOTH directions a deliberate, reviewed diff: bump the expected set here in the same
PR that changes the export, or the suite fails.

Every name is also checked to actually resolve on its module (``__all__`` cannot list a name the
module doesn't carry) and every list is checked for duplicates (a repeated name is never intentional
in a hand-maintained ``__all__``).
"""

from __future__ import annotations

import unittest

import percept
from percept import domain_pack, sensorplan

# The reviewed 1.0 export surface. A name added or removed here must be a deliberate line in the
# SAME diff that changes percept/__init__.py's __all__ — that is the whole point of pinning it.
EXPECTED_PERCEPT_ALL = frozenset({
    "ACTUATION_CLASSES", "ASK", "ActionLog", "ActionRecord", "AppearanceIdentity", "AttentionIntent",
    "Attribute", "AudioFrame", "BaselineAssumption", "BehaviorRoute", "Box", "BuildInfo", "BundleError",
    "BundlePolicy", "CONTRACT_VERSION", "CacheCascade", "CacheKey", "CacheResult",
    "CompiledWatchSnapshotUnsupportedError", "ComponentIdentity", "Concern", "ConcurrentCoreRunError",
    "ConsentDenied", "ConsentProfile", "Consequence", "ContinuationDecision", "ContinuityIdentity",
    "DEFAULT_FIXTURE", "DEFAULT_RULES", "DELIVERED_ACTIONS", "Decision", "DispositionLog",
    "EPISODE_POLICIES", "EVICTION_ORDER", "Enricher", "Entity", "EntityGraph", "EnvelopeViolation",
    "Episode", "EpisodeLog", "EpisodePolicy", "Event", "EventLog", "Evidence", "EvidenceAccessDenied",
    "EvidenceArtifact", "EvidenceCorrupt", "EvidenceError", "EvidenceExpired", "EvidenceItem",
    "EvidenceStore", "EvidenceUnavailable", "FIRE", "FUNNEL_STAGES", "FireEvent", "FixtureVision",
    "FrameHashLayer", "FrameWindow", "Funnel", "GATE_ACTIONS", "GATE_REASONS", "Gate", "GateAction",
    "GateOutcome", "GateReason", "Goal", "GoalOp", "GoalOrigin", "GoalRegistry", "INTEREST_ENTERED",
    "INTEREST_EXITED", "INTEREST_UPDATED", "InMemoryPersistence", "Incident", "IncidentSnippet",
    "InterestIntervalConfig", "InterestIntervalEngine", "InterestIntervalEvent", "InterestSample",
    "InterestSubscription", "LabelRecord", "LabelStore", "LiveInterestAdmission", "LiveInterestEvent",
    "MediaReference", "Metrics", "ModeError", "OTelExporter", "OUT_OF_ENVELOPE", "Observation",
    "OfflineFixture", "OfflineFixtureError", "PHYSICAL_ACTUATION", "Percept", "PlanIdentity",
    "ProviderError", "ProviderSpec", "REASON_ACTION", "RegexRouteLayer", "ReplayGoalConfig",
    "ReplayReport", "RetentionPolicy", "RouteEngine", "RouteLayer", "RouteMatch", "Rule",
    "RunTraceManifest", "RuntimeSession", "RuntimeStats", "SILENT", "STTEvent", "SUPPRESSION_REASONS",
    "Scheduler", "Settings", "SqlitePersistence", "SubjectResolver", "Tier0MotionInterest",
    "Tier0Signal", "Timer", "TraceBundle", "TraceEvent", "TracePolicy", "TraceRecord", "TraceRecorder",
    "TraceSummary", "TurnResult", "UndrivenBehaviorRouteWarning", "UnsupportedBehaviorRouteError",
    "Verdict", "VisualizeError", "VizOptions", "WATCH_VERSION", "WatchCondition", "WatchEvent",
    "WatchSpec", "__version__", "assert_live_or_offline", "build_info", "client_asserted_subject",
    "desugar_goal", "desugar_goals", "envelope_summary", "episode_policy_of", "export_bug_report",
    "funnel_key", "gate_config_of", "get_logger", "goal_id_for", "health", "identity_policy_of",
    "judge_usage", "load_fixture", "load_settings", "load_trace", "offline_core", "offline_frames",
    "offline_goals", "otel_exporter", "register_subject_resolver", "registry", "render_html",
    "replay_cognition", "resolve_evidence", "resolve_vision", "run", "select_identity_policy", "start",
    "verdict_ref", "visualize_trace", "watch",
})

# percept.sensorplan — de-exported attach_sensor_plan/coerce_authored_sensor_plan (zero callers
# outside the module itself; see tests/test_sensorplan.py and this card's report).
EXPECTED_SENSORPLAN_ALL = frozenset({
    "SensorPrimitive", "PRIMITIVES", "SensorPlan", "sensor_plan_for",
})

# percept.domain_pack — unchanged by this card at the __all__ level: DomainPack keeps real callers
# (resolver.py, packs_default.py, catalog.py, eval dataset loaders). Only its dead
# ``direct_judge_routes`` FIELD was removed, which is not a name in this list.
EXPECTED_DOMAIN_PACK_ALL = frozenset({
    "DomainPackError", "DomainPackNotFoundError", "ComponentStrength", "ComponentRecommendation",
    "GoalTemplate", "DomainPack", "DomainPackRegistry", "DEFAULT_DOMAIN_PACK_REGISTRY",
    "register_domain_pack", "get_domain_pack", "discover_domain_pack_entrypoints", "watch_template",
})


class TestPerceptPublicSurface(unittest.TestCase):
    def test_no_duplicate_names(self) -> None:
        self.assertEqual(len(percept.__all__), len(set(percept.__all__)))

    def test_every_name_resolves_on_the_module(self) -> None:
        missing = [n for n in percept.__all__ if not hasattr(percept, n)]
        self.assertEqual(missing, [], f"__all__ lists names percept does not carry: {missing}")

    def test_exported_surface_is_pinned(self) -> None:
        current = set(percept.__all__)
        added = current - EXPECTED_PERCEPT_ALL
        removed = EXPECTED_PERCEPT_ALL - current
        self.assertEqual(
            added, set(),
            f"percept.__all__ grew silently: {sorted(added)} — "
            "update EXPECTED_PERCEPT_ALL in this test in the SAME diff that adds the export.",
        )
        self.assertEqual(
            removed, set(),
            f"percept.__all__ lost names without this test being updated: {sorted(removed)} — "
            "if the removal is intentional, update EXPECTED_PERCEPT_ALL and report it as a breaking change.",
        )


class TestSensorplanPublicSurface(unittest.TestCase):
    def test_no_duplicate_names(self) -> None:
        self.assertEqual(len(sensorplan.__all__), len(set(sensorplan.__all__)))

    def test_every_name_resolves_on_the_module(self) -> None:
        missing = [n for n in sensorplan.__all__ if not hasattr(sensorplan, n)]
        self.assertEqual(missing, [])

    def test_exported_surface_is_pinned(self) -> None:
        self.assertEqual(set(sensorplan.__all__), EXPECTED_SENSORPLAN_ALL)


class TestDomainPackPublicSurface(unittest.TestCase):
    def test_no_duplicate_names(self) -> None:
        self.assertEqual(len(domain_pack.__all__), len(set(domain_pack.__all__)))

    def test_every_name_resolves_on_the_module(self) -> None:
        missing = [n for n in domain_pack.__all__ if not hasattr(domain_pack, n)]
        self.assertEqual(missing, [])

    def test_exported_surface_is_pinned(self) -> None:
        self.assertEqual(set(domain_pack.__all__), EXPECTED_DOMAIN_PACK_ALL)

    def test_direct_judge_routes_is_not_a_field_anymore(self) -> None:
        """The dark seam this card removed: DomainPack used to carry a direct_judge_routes field
        that was declared/validated/serialised but consumed by nothing on the routing path."""
        pack = domain_pack.DomainPack(id="p")
        self.assertFalse(hasattr(pack, "direct_judge_routes"))
        self.assertNotIn("direct_judge_routes", pack.to_dict())


if __name__ == "__main__":
    unittest.main()
