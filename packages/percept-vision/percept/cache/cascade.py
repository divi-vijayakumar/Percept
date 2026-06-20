"""The cache contract + cascade composition.

Caching ships a *contract, not a policy* (invariant 6). An **empty cascade always DEFERs**
— the safe default. A layer only short-circuits when it returns a HIT whose confidence clears
the cascade threshold; anything else falls through to the next layer, and finally to DEFER
(re-perceive). False-serve is the cache's analog of false-fire and is audited downstream.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Protocol, runtime_checkable

from ..adapters import Provenance, Verdict

CacheStatus = Literal["HIT", "DEFER"]


@dataclass(frozen=True)
class CacheResult:
    status: CacheStatus
    verdict: Verdict | None = None
    confidence: float = 0.0
    provenance: Provenance = "cached"
    source_layer: str | None = None

    @classmethod
    def defer(cls) -> "CacheResult":
        return cls(status="DEFER")


@runtime_checkable
class CacheLayer(Protocol):
    name: str

    def lookup(self, key: Any) -> CacheResult:
        """Return a HIT (with verdict + confidence) or ``CacheResult.defer()``."""
        ...

    def store(self, key: Any, verdict: Verdict, confidence: float) -> None: ...


@dataclass
class CacheCascade:
    """Ordered cache layers resolved front-to-back. First HIT >= threshold wins."""

    layers: list[CacheLayer]
    threshold: float = 0.0

    def resolve(self, key: Any) -> CacheResult:
        for layer in self.layers:
            result = layer.lookup(key)
            if result.status == "HIT" and result.confidence >= self.threshold:
                return result
        return CacheResult.defer()  # empty cascade, or no confident hit -> re-perceive

    def store(self, key: Any, verdict: Verdict, confidence: float) -> None:
        for layer in self.layers:
            layer.store(key, verdict, confidence)


__all__ = ["CacheStatus", "CacheResult", "CacheLayer", "CacheCascade"]
