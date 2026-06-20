"""The local route engine — a confidence-gated, layered fast-path before the conductor.

Structure (the V0.1 seam): a :class:`RouteEngine` walks an ordered list of :class:`RouteLayer`
tiers. Each layer scores an utterance and returns a :class:`RouteMatch` (route + confidence)
or ``None``. The engine takes the **first match at or above the threshold** and turns it into
a :class:`~percept.adapters.Decision`; if no layer clears the bar it returns ``None`` —
**DEFER to the conductor**.

Defer-on-miss is therefore **structural**, not a regex special-case: a match whose confidence
falls below ``threshold`` is indistinguishable from no match at all. To make the router quieter
you raise the threshold; to special-case nothing you add no code. This is what lets the V0.1
``LearnedRouteLayer`` (an ONNX intent classifier) slot in front of — or instead of — the regex
lexicon with **zero changes here**: it is just another ``RouteLayer`` whose ``score`` returns a
calibrated confidence, and the same threshold gates it.

The router never authors a goal. It can only pick a confident **route**; standing-state writes
(goal ops) are the conductor's alone (the only tier that writes state). On any goal-authoring,
conditional, ambiguous, or fragmentary utterance the lexicon simply has no confident rule, so
the engine defers — exactly the "when in doubt, defer" invariant.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import re

from ..adapters import Decision
from .rules import DEFAULT_RULES, DEFER_GUARDS, Rule

# Routes the local router is allowed to emit must be valid Decision.route values. We don't
# import the Literal at runtime (it's a typing construct); we validate names against this set,
# which mirrors adapters.Decision.route exactly. If that frozen set changes, this list is the
# one place to reconcile.
_VALID_ROUTES = frozenset(
    {
        "respond_direct",
        "respond_with_vision",
        "task_with_tools",
        "act",
        "proactive_speak",
        "ignore",
    }
)


@dataclass(frozen=True)
class RouteMatch:
    """A layer's scored verdict for one utterance."""

    route: str
    confidence: float
    look: str = "none"
    rule: str = ""  # provenance: which rule/layer fired (for logging + the teacher log)


@runtime_checkable
class RouteLayer(Protocol):
    """A tier the engine can ask to score an utterance.

    Implementations: :class:`RegexRouteLayer` (V0.0 lexicon) and, in V0.1,
    ``LearnedRouteLayer`` (an embedding + linear/MLP classifier). Both return a
    :class:`RouteMatch` with a **calibrated confidence**, or ``None``; the engine's threshold —
    not the layer — decides whether that confidence is enough to fast-path.
    """

    def score(self, utterance: str, context: dict | None = None) -> RouteMatch | None: ...


class RegexRouteLayer:
    """The V0.0 lexicon tier: ordered regex rules, first matching rule wins.

    This is deliberately dumb pattern-matching. It exists to make the *common, confident*
    turns cheap and instant; everything it is unsure about it simply doesn't match, and the
    engine defers. It is replaced wholesale by ``LearnedRouteLayer`` in V0.1 — the regexes are
    not a contract, the ``RouteLayer`` seam is.
    """

    def __init__(
        self,
        rules: tuple[Rule, ...] = DEFAULT_RULES,
        guards: tuple[str, ...] = DEFER_GUARDS,
    ) -> None:
        self.rules = rules
        self._compiled = [(r, r.compiled()) for r in rules]
        # Veto lexicon: a match here forces a defer regardless of any rule match (the
        # structural "when in doubt, defer" — see rules.DEFER_GUARDS).
        self._guards = [re.compile(p, re.IGNORECASE) for p in guards]

    def _vetoed(self, text: str) -> bool:
        return any(g.search(text) for g in self._guards)

    def score(self, utterance: str, context: dict | None = None) -> RouteMatch | None:
        text = utterance or ""
        if self._vetoed(text):
            return None  # goal-authoring / conditional / ambiguous marker -> DEFER
        for rule, patterns in self._compiled:
            for pat in patterns:
                if pat.search(text):
                    return RouteMatch(
                        route=rule.route,
                        confidence=rule.confidence,
                        look=rule.look,
                        rule=rule.name,
                    )
        return None


class RouteEngine:
    """Walks ordered :class:`RouteLayer`s; first match >= threshold routes, else DEFER.

    ``route(utterance)`` returns a :class:`~percept.adapters.Decision` on a confident match, or
    ``None`` to signal the caller should defer to the conductor. The threshold is injected
    (it lives in ``config.RoutingSettings.confidence_threshold`` — no literal in logic).
    """

    def __init__(self, layers: list[RouteLayer], *, threshold: float) -> None:
        self.layers = layers
        self.threshold = threshold

    @classmethod
    def default(cls, *, threshold: float) -> "RouteEngine":
        """The V0.0 engine: a single regex lexicon layer. V0.1 prepends a LearnedRouteLayer."""
        return cls([RegexRouteLayer()], threshold=threshold)

    def best_match(self, utterance: str, context: dict | None = None) -> RouteMatch | None:
        """The first layer match at or above threshold, else ``None`` (structural defer)."""
        for layer in self.layers:
            match = layer.score(utterance, context)
            if match is None:
                continue
            if match.confidence < self.threshold:
                # Below the bar -> treat as no-match and keep walking. A learned layer that is
                # unsure defers here exactly like an unmatched regex does.
                continue
            if match.route not in _VALID_ROUTES:
                # A misconfigured layer can never silently mis-route; it defers.
                continue
            return match
        return None

    def route(self, utterance: str, context: dict | None = None) -> Decision | None:
        """Return a confident :class:`Decision`, or ``None`` to DEFER to the conductor."""
        match = self.best_match(utterance, context)
        if match is None:
            return None
        return Decision(route=match.route, look=match.look)  # type: ignore[arg-type]


__all__ = [
    "RouteMatch",
    "RouteLayer",
    "RegexRouteLayer",
    "RouteEngine",
]
