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
    Decision,
    Goal,
    Observation,
    STTEvent,
    Tier0Signal,
    TurnResult,
    Verdict,
    WatchCondition,
    WatchSpec,
)
from .cache import CacheCascade, CacheKey, CacheResult, FrameHashLayer
from .config import Settings, load_settings
from .entity import Entity, EntityGraph
from .gate import ASK, FIRE, SILENT, Gate, GateAction
from .goals import GoalOp, GoalRegistry
from .observability import Metrics, get_logger, health
from .routing import (
    DEFAULT_RULES,
    RegexRouteLayer,
    RouteEngine,
    RouteLayer,
    RouteMatch,
    Rule,
)

__version__ = "0.0.0.dev0"

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
    "Decision",
    "STTEvent",
    "TurnResult",
    "WatchCondition",
    "WatchSpec",
    "Tier0Signal",
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
    "get_logger",
    "health",
]
