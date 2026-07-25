"""percept — the open-source cognition layer for goal-driven, proactive vision agents.

Public surface. The reactive watch loop, the three-state gate, the entity graph, and the goal
registry are wired (Phase 1); identity/episodes, events/scheduler, the edge, and persistence land
in later phases *beside* this core. See ``Spec/`` for the what/why and the implementation plan.
"""

from __future__ import annotations

from . import registry
from .adapters import (
    Attribute,
    Box,
    Concern,
    Consequence,
    Decision,
    Enricher,
    Evidence,
    Goal,
    GoalOrigin,
    Observation,
    STTEvent,
    Tier0Signal,
    TurnResult,
    Verdict,
    WatchCondition,
    WatchSpec,
)
from percept.contracts import (
    AttentionIntent,
    AudioFrame,
    CONTRACT_VERSION,
    Incident,
    IncidentSnippet,
    LabelRecord,
    WatchEvent,
)
from .labelstore import DispositionLog, LabelStore, verdict_ref
from .cache import CacheCascade, CacheKey, CacheResult, FrameHashLayer
from .config import ModeError, Settings, load_settings
from .offline import (
    DEFAULT_FIXTURE,
    FixtureVision,
    OfflineFixture,
    OfflineFixtureError,
    assert_live_or_offline,
    load_fixture,
    offline_core,
    offline_frames,
    offline_goals,
)
from .consent import ActionLog, ActionRecord, ConsentDenied, ConsentProfile
from .envelope import (
    ACTUATION_CLASSES,
    OUT_OF_ENVELOPE,
    PHYSICAL_ACTUATION,
    EnvelopeViolation,
    SubjectResolver,
    client_asserted_subject,
    envelope_summary,
    register_subject_resolver,
)
from .entity import Entity, EntityGraph
from .episodes import (
    EPISODE_POLICIES,
    ContinuationDecision,
    Episode,
    EpisodeLog,
    EpisodePolicy,
    episode_policy_of,
)
from .events import Event, EventLog, FrameWindow
from .evidence import (
    EVICTION_ORDER,
    EvidenceAccessDenied,
    EvidenceArtifact,
    EvidenceCorrupt,
    EvidenceError,
    EvidenceExpired,
    EvidenceStore,
    EvidenceUnavailable,
    RetentionPolicy,
    resolve_evidence,
)
from .identity_policy import identity_policy_of, select_identity_policy
from .gate import (
    ASK,
    DELIVERED_ACTIONS,
    FIRE,
    GATE_ACTIONS,
    GATE_REASONS,
    REASON_ACTION,
    SILENT,
    SUPPRESSION_REASONS,
    Gate,
    GateAction,
    GateOutcome,
    GateReason,
)
from .goals import GoalOp, GoalRegistry
from .identity import AppearanceIdentity, ContinuityIdentity
from .scheduler import Scheduler, Timer
from .observability import (
    FUNNEL_STAGES,
    Funnel,
    Metrics,
    OTelExporter,
    TraceEvent,
    funnel_key,
    get_logger,
    health,
    judge_usage,
    otel_exporter,
)
from .persistence import InMemoryPersistence, SqlitePersistence
from .routing import (
    DEFAULT_RULES,
    RegexRouteLayer,
    RouteEngine,
    RouteLayer,
    RouteMatch,
    Rule,
)

# Single-sourced from pyproject ``[project].version`` — never a hand-synced literal. Installed: read
# the dist metadata. Bare source checkout (no metadata): read pyproject directly (it isn't in the
# wheel, but a checkout always has it), so a version bump can't leave a stale fallback here.
def _resolve_version() -> str:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("percept-vision")
    except PackageNotFoundError:  # pragma: no cover - dev-only path (uninstalled source tree)
        import pathlib
        import re

        try:
            text = (pathlib.Path(__file__).resolve().parent.parent / "pyproject.toml").read_text()
        except OSError:
            return "0.0.0+unknown"
        m = re.search(r'(?m)^\s*version\s*=\s*"([^"]+)"', text)
        return m.group(1) if m else "0.0.0+unknown"


__version__ = _resolve_version()
del _resolve_version

# Build provenance (version + git commit + CONTRACT_VERSION + build timestamp) — stamped into the
# wheel/sdist at build time, "unreleased" fallback in a bare source checkout. Stdlib-pure.
from ._build import BuildInfo, build_info  # noqa: E402

# Imported after __version__ so agent.health() can read it lazily without a cycle.
from .agent import CompiledWatchSnapshotUnsupportedError, FireEvent, Percept  # noqa: E402
from .watch import WATCH_VERSION, watch  # noqa: E402
from .interest import (  # noqa: E402
    INTEREST_ENTERED,
    INTEREST_EXITED,
    INTEREST_UPDATED,
    InterestIntervalConfig,
    InterestIntervalEngine,
    InterestIntervalEvent,
    InterestSample,
    Tier0MotionInterest,
)

# String sugar for watch(): `watch(stream, "the milk is boiling over")`. Exported so callers can
# reproduce the desugared id (goal_id_for) without re-implementing the documented scheme.
from .goalsugar import desugar_goal, desugar_goals, goal_id_for  # noqa: E402
# Provider identity: a NAME resolves to a factory; the model id is configuration
# carried to it, never the adapter's identity. Stdlib-pure (registry + config), no vendor import.
from .provider import (  # noqa: E402
    ProviderError,
    ProviderSpec,
    resolve_vision,
)

# The runtime worker — the batteries-included "watch a source" front door. Surfaced at top level
# (PR: Wave 0) so `percept.run` / `percept.RuntimeSession` are the FIRST thing a builder reaches,
# instead of being two import-hops deep at `percept.runtime.worker.run`. Import is stdlib-pure
# (cv2/numpy are lazy inside run()), so the zero-vendor-import invariant holds.
from .runtime import RuntimeSession, RuntimeStats, run, start  # noqa: E402

# The diagnostic data model: percept doctor, percept eval and the visualizer all derive
# their numbers from THIS module, so three surfaces cannot invent three definitions of the funnel.
# Imported after __version__ (the manifest stamps it) and stdlib-pure.
from .tracing import (  # noqa: E402
    BaselineAssumption,
    ComponentIdentity,
    PlanIdentity,
    RunTraceManifest,
    TracePolicy,
    TraceRecord,
    TraceRecorder,
    TraceSummary,
    load_trace,
)

# The portable trace bundle + the deterministic cognition replay. Media inclusion is
# opt-in and OFF by default (see percept/trace_bundle.py); replay never calls a model.
from .trace_bundle import (  # noqa: E402
    BundleError,
    BundlePolicy,
    EvidenceItem,
    MediaReference,
    TraceBundle,
    export_bug_report,
)
from .replay import (  # noqa: E402
    ReplayGoalConfig,
    ReplayReport,
    gate_config_of,
    replay_cognition,
)

# The native visualizer: lanes over media time, click-through decision chains, the
# per-stat counterfactual. Reads the artifacts above; recomputes no metric. Media embedding is
# opt-in and OFF by default, and the page makes ZERO network requests.
from .visualize import (  # noqa: E402
    VisualizeError,
    VizOptions,
    render_html,
    visualize_trace,
)

__all__ = [
    "__version__",
    "build_info",
    "BuildInfo",
    "Percept",
    "FireEvent",
    "ModeError",
    "registry",
    # fixture-bound offline mode — the keyless first run
    "offline_core",
    "offline_frames",
    "offline_goals",
    "load_fixture",
    "OfflineFixture",
    "OfflineFixtureError",
    "FixtureVision",
    "DEFAULT_FIXTURE",
    "assert_live_or_offline",
    # the runtime front door (start here)
    "run",
    "start",
    "RuntimeSession",
    "RuntimeStats",
    # the public watch() entrypoint — the frozen seam every downstream piece depends on
    "watch",
    "WATCH_VERSION",
    "WatchEvent",
    # deterministic finite-replay interest lifecycle (admission signal, not semantic judgment)
    "INTEREST_ENTERED",
    "INTEREST_UPDATED",
    "INTEREST_EXITED",
    "InterestIntervalConfig",
    "InterestSample",
    "InterestIntervalEvent",
    "InterestIntervalEngine",
    "Tier0MotionInterest",
    # string sugar for watch() goals (pass-through, never compilation)
    "desugar_goal",
    "desugar_goals",
    "goal_id_for",
    # provider identity: name -> factory; model id is configuration, not identity
    "resolve_vision",
    "ProviderSpec",
    "ProviderError",
    # DTOs
    "Box",
    "Observation",
    "Verdict",
    "Attribute",
    "Goal",
    "GoalOrigin",
    "Concern",
    "Consequence",
    "Evidence",
    "Decision",
    "STTEvent",
    "TurnResult",
    "WatchCondition",
    "WatchSpec",
    "Tier0Signal",
    "Enricher",
    "AudioFrame",
    "Incident",
    "IncidentSnippet",
    "AttentionIntent",
    # evidence dereference (ADR-003) — the opaque handle, its resolver, typed failures, retention seam
    "EvidenceArtifact",
    "EvidenceStore",
    "resolve_evidence",
    "RetentionPolicy",
    "EVICTION_ORDER",
    "EvidenceError",
    "EvidenceExpired",
    "EvidenceUnavailable",
    "EvidenceAccessDenied",
    "EvidenceCorrupt",
    # feedback rail — local label + disposition stores
    "LabelRecord",
    "LabelStore",
    "DispositionLog",
    "verdict_ref",
    # contract version
    "CONTRACT_VERSION",
    # cognition core
    "Gate",
    "GateAction",
    "FIRE",
    "ASK",
    "SILENT",
    # The consequence gate's GA decision model — typed vocabulary + suppression codes.
    # SILENT is exported as a value of the MODEL, not as a deliverable action (see DELIVERED_ACTIONS).
    "GateOutcome",
    "GateReason",
    "GATE_ACTIONS",
    "DELIVERED_ACTIONS",
    "GATE_REASONS",
    "REASON_ACTION",
    "SUPPRESSION_REASONS",
    "EntityGraph",
    "Entity",
    "GoalRegistry",
    "GoalOp",
    "CompiledWatchSnapshotUnsupportedError",
    # identity + episodes (Phase 4 — re-id across gaps; the cup problem)
    "ContinuityIdentity",
    "AppearanceIdentity",
    "Episode",
    "EpisodeLog",
    "EpisodePolicy",
    "EPISODE_POLICIES",
    "ContinuationDecision",
    "episode_policy_of",
    # concern-level identity-resolver selection (concern SELECTS, profile DECLARES)
    "identity_policy_of",
    "select_identity_policy",
    # events + scheduler (Phase 5 — temporal perception; time-triggered fires)
    "Event",
    "EventLog",
    "FrameWindow",
    "Scheduler",
    "Timer",
    # persistence + consent (Phase 7 — endurance + safe act modality)
    "InMemoryPersistence",
    "SqlitePersistence",
    "ConsentProfile",
    "ConsentDenied",
    "ActionLog",
    "ActionRecord",
    # envelope (ADR-007 — the three OUT verdicts, in config not prose)
    "ACTUATION_CLASSES",
    "PHYSICAL_ACTUATION",
    "OUT_OF_ENVELOPE",
    "EnvelopeViolation",
    "SubjectResolver",
    "client_asserted_subject",
    "register_subject_resolver",
    "envelope_summary",
    # routing (Phase 3 — local fast-path; a future learned tier slots behind RouteLayer)
    "RouteEngine",
    "RouteLayer",
    "RouteMatch",
    "RegexRouteLayer",
    "Rule",
    "DEFAULT_RULES",
    # cache
    "CacheCascade",
    "CacheResult",
    "CacheKey",
    "FrameHashLayer",
    # infra
    "Settings",
    "load_settings",
    "Metrics",
    "Funnel",
    "FUNNEL_STAGES",
    "TraceEvent",
    "funnel_key",
    "judge_usage",
    "OTelExporter",
    "otel_exporter",
    "get_logger",
    "health",
    # diagnostics — the ONE definition of observed/admitted/judged/saved
    "RunTraceManifest",
    "TraceRecord",
    "TraceRecorder",
    "TraceSummary",
    "TracePolicy",
    "BaselineAssumption",
    "ComponentIdentity",
    "PlanIdentity",
    "load_trace",
    # portable trace bundle + deterministic cognition replay
    "TraceBundle",
    "VisualizeError",
    "VizOptions",
    "render_html",
    "visualize_trace",
    "BundlePolicy",
    "BundleError",
    "EvidenceItem",
    "MediaReference",
    "export_bug_report",
    "replay_cognition",
    "ReplayReport",
    "ReplayGoalConfig",
    "gate_config_of",
]
