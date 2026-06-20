"""percept — the open-source cognition layer for goal-driven, proactive vision agents.

Public surface. In Phase 0 :class:`Percept` is a thin shell that resolves backends behind the
frozen seams and does nothing intelligent yet — the gate, entity graph, and watch loop arrive
in Phase 1. See ``Spec/`` for the what/why and the implementation plan.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

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
from .cache import CacheCascade, CacheResult
from .config import Settings, load_settings
from .observability import Metrics, get_logger, health

__version__ = "0.0.0.dev0"


@dataclass
class Percept:
    """The cognition core, holding resolved backends behind the seams.

    Construct via :meth:`create`. Phase 0 wires the seams; later phases add the gate,
    entity graph, episodes, events, and the watch loop *beside* this shell.
    """

    vision: Any
    specialist: Any
    conductor: Any
    stt: Any
    tts: Any
    settings: Settings
    metrics: Metrics

    @classmethod
    def create(
        cls,
        *,
        vision: str = "fake",
        specialist: str = "fake",
        conductor: str = "fake",
        stt: str = "fake",
        tts: str = "fake",
        settings: Settings | None = None,
        discover_plugins: bool = True,
    ) -> "Percept":
        cfg = settings or load_settings()
        if discover_plugins:
            registry.discover_entrypoints()
        return cls(
            vision=registry.make_vision(vision, cfg),
            specialist=registry.make_specialist(specialist, cfg),
            conductor=registry.make_conductor(conductor, cfg),
            stt=registry.make_stt(stt, cfg),
            tts=registry.make_tts(tts, cfg),
            settings=cfg,
            metrics=Metrics(),
        )

    def health(self) -> dict:
        backends = {
            "vision": type(self.vision).__name__,
            "specialist": type(self.specialist).__name__,
            "conductor": type(self.conductor).__name__,
            "stt": type(self.stt).__name__,
            "tts": type(self.tts).__name__,
        }
        return health(__version__, backends, self.metrics)


__all__ = [
    "__version__",
    "Percept",
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
    # cache
    "CacheCascade",
    "CacheResult",
    # infra
    "Settings",
    "load_settings",
    "Metrics",
    "get_logger",
]
