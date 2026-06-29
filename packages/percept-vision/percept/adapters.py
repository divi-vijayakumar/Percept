"""The FROZEN membrane — vendor-neutral DTOs + Protocols.

This module is the contract between the cognition core and everything vendor-specific.
Per the implementation plan (§0, Phase 0) it is **immutable within a phase**: later phases
implement *behind* these interfaces and add *beside* them. If a phase needs to change a
frozen shape here, stop and flag it — do not edit silently.

Cardinal rule (the firing contract): the Vision seam judges a goal's plain-language
*condition* and returns ``Verdict{satisfied, confidence, evidence}``. It never compiles the
goal to a rigid ``attr_key=value`` and exact-matches. One firing path, one gate.

The contract now lives in the standalone, zero-dependency ``percept.contracts`` package so that
non-vision consumers (clients, the IRA, dashboards, the JS edge twin) can import it directly
instead of re-declaring it. This module is a backward-compatible **re-export shim**: every
``from percept.adapters import X`` (and ``from percept import X``) keeps working unchanged.
"""

from __future__ import annotations

from percept.contracts import (
    Attribute,
    AudioFrame,
    Box,
    Concern,
    Conductor,
    Consequence,
    Cost,
    Decision,
    Enricher,
    Evidence,
    FalseFireCost,
    Goal,
    Identity,
    Observation,
    Persistence,
    Priors,
    Provenance,
    Schedule,
    Specialist,
    STT,
    STTEvent,
    Subject,
    Tier0Signal,
    minute_of_day,
    TrackingBehavior,
    TTS,
    TurnResult,
    Verdict,
    Vision,
    WatchCondition,
    WatchSpec,
)

__all__ = [
    "Box",
    "Observation",
    "Verdict",
    "Provenance",
    "Attribute",
    "Goal",
    "Cost",
    "FalseFireCost",
    "Consequence",
    "Evidence",
    "TrackingBehavior",
    "Subject",
    "Priors",
    "Concern",
    "Decision",
    "STTEvent",
    "TurnResult",
    "Schedule",
    "minute_of_day",
    "WatchCondition",
    "WatchSpec",
    "Tier0Signal",
    "AudioFrame",
    "Vision",
    "Specialist",
    "Enricher",
    "Conductor",
    "STT",
    "TTS",
    "Identity",
    "Persistence",
]
