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
    Observation,
    STTEvent,
    Tier0Signal,
    TurnResult,
    Verdict,
    WatchCondition,
    WatchSpec,
)
from percept_contracts import AudioFrame, CONTRACT_VERSION, Incident, IncidentSnippet
from .cache import CacheCascade, CacheKey, CacheResult, FrameHashLayer
from .config import Settings, load_settings
from .consent import ActionLog, ActionRecord, ConsentDenied, ConsentProfile
from .entity import Entity, EntityGraph
from .episodes import Episode, EpisodeLog
from .events import Event, EventLog, FrameWindow
from .gate import ASK, FIRE, SILENT, Gate, GateAction
from .goals import GoalOp, GoalRegistry
from .identity import AppearanceIdentity, ContinuityIdentity
from .scheduler import Scheduler, Timer
from .observability import (
    FUNNEL_STAGES,
    Funnel,
    Metrics,
    OTelExporter,
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

__version__ = "0.1.0"

# Imported after __version__ so agent.health() can read it lazily without a cycle.
from .agent import FireEvent, Percept  # noqa: E402

__all__ = [
    "__version__",
    "Percept",
    "FireEvent",
    "registry",
    # DTOs
    "Box",
    "Observation",
    "Verdict",
    "Attribute",
    "Goal",
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
    # contract version
    "CONTRACT_VERSION",
    # cognition core
    "Gate",
    "GateAction",
    "FIRE",
    "ASK",
    "SILENT",
    "EntityGraph",
    "Entity",
    "GoalRegistry",
    "GoalOp",
    # identity + episodes (Phase 4 — re-id across gaps; the cup problem)
    "ContinuityIdentity",
    "AppearanceIdentity",
    "Episode",
    "EpisodeLog",
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
    # routing (Phase 3 — local fast-path; learned tier slots behind RouteLayer in V0.1)
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
    "funnel_key",
    "judge_usage",
    "OTelExporter",
    "otel_exporter",
    "get_logger",
    "health",
]
