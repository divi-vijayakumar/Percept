"""Local routing — the confidence-gated fast-path in front of the conductor (Phase 3).

The router fast-paths only confident, non-authoring turns; everything goal-authoring,
conditional, ambiguous, or fragmentary DEFERS (``route(...) -> None``) to the conductor, which
is the only tier that writes standing state. The V0.0 tier is a regex lexicon
(:class:`RegexRouteLayer` over :data:`DEFAULT_RULES`); a learned classifier slots in behind the
same :class:`RouteLayer` seam in V0.1 without touching the engine or the agent.
"""

from __future__ import annotations

from .engine import RegexRouteLayer, RouteEngine, RouteLayer, RouteMatch
from .rules import DEFAULT_RULES, DEFER_EXAMPLES, Rule

__all__ = [
    "RouteEngine",
    "RouteLayer",
    "RouteMatch",
    "RegexRouteLayer",
    "Rule",
    "DEFAULT_RULES",
    "DEFER_EXAMPLES",
]
