"""The three-state gate — the trust surface (invariant 1).

One firing path, one gate (invariant 3). A :class:`~percept.adapters.Verdict` plus a freshness
value maps to exactly one action:

| condition                                   | action   |
|---------------------------------------------|----------|
| confident yes, fresh, on a *rising* edge    | ``FIRE`` |
| confident yes, but already fired / not rising | ``SILENT`` |
| confident no                                | ``SILENT`` (re-arms the edge) |
| low-confidence **or** stale                 | ``ASK``  (never a confident-wrong nag) |

Rising-edge + refractory are implemented as a single armed/disarmed bit per key. A *key* is
``(goal_id, entity_or_episode_id)`` so a brand-new entity gets a fresh edge (a new cup fires even
though the old cup already did), while the same entity never double-fires until it goes
confident-false and re-arms.
"""

from __future__ import annotations

from typing import Hashable, Literal

from .adapters import Verdict

GateAction = Literal["fire", "ask", "silent"]
FIRE: GateAction = "fire"
ASK: GateAction = "ask"
SILENT: GateAction = "silent"

# verdict -> coarse truth, before edge logic
_Truth = Literal["yes", "no", "unknown"]


def classify(
    verdict: Verdict, freshness: float, threshold: float, fresh_min: float
) -> _Truth:
    """Map a verdict + freshness to yes / no / unknown (the pre-edge decision).

    Low confidence OR a stale supporting fact is ``unknown`` -> ASK. This is the refusal-to-guess
    rule: ambiguity never becomes a confident fire.
    """
    if verdict.confidence < threshold:
        return "unknown"
    if freshness < fresh_min:
        return "unknown"
    return "yes" if verdict.satisfied else "no"


class Gate:
    """Stateful three-state gate. ``evaluate`` is the only firing path."""

    def __init__(self, threshold: float = 0.7, fresh_min: float = 0.5) -> None:
        self.threshold = threshold
        self.fresh_min = fresh_min
        self._armed: dict[Hashable, bool] = {}  # key -> ready-to-fire

    def evaluate(
        self, key: Hashable, verdict: Verdict, *, freshness: float = 1.0
    ) -> GateAction:
        """Return FIRE / ASK / SILENT for ``key``, updating edge state."""
        truth = classify(verdict, freshness, self.threshold, self.fresh_min)
        if truth == "unknown":
            return ASK  # asking never changes the edge — the fact is still pending
        if truth == "no":
            self._armed[key] = True  # confident-false re-arms
            return SILENT
        # truth == "yes"
        if self._armed.get(key, True):  # new keys default to armed -> fresh edge
            self._armed[key] = False
            return FIRE
        return SILENT  # already true / already fired this episode

    def rearm(self, key: Hashable) -> None:
        """Force a key back to armed (e.g. after a correction clears the edge)."""
        self._armed[key] = True

    def reset_goal(self, goal_id: str) -> None:
        """Drop all edge state for a goal so its next evaluation starts fresh."""
        for key in [k for k in self._armed if isinstance(k, tuple) and k and k[0] == goal_id]:
            del self._armed[key]


__all__ = ["GateAction", "FIRE", "ASK", "SILENT", "classify", "Gate"]
