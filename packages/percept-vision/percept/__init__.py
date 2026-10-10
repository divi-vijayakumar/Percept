"""Percept: a decision layer for live video.

Percept watches a stream for plain-language conditions and emits one decision per real
occurrence, so an agent can act on events instead of frames.
"""

from percept.contracts import (
    Condition,
    Decision,
    Delivery,
    EvidenceRef,
    Frame,
    Health,
    Judge,
    JudgeRequest,
    JudgeResult,
    Observation,
    Sink,
    Source,
    Store,
    Trigger,
    Usage,
)

__version__ = "1.0.0.dev0"

__all__ = [
    "Condition",
    "Decision",
    "Delivery",
    "EvidenceRef",
    "Frame",
    "Health",
    "Judge",
    "JudgeRequest",
    "JudgeResult",
    "Observation",
    "Sink",
    "Source",
    "Store",
    "Trigger",
    "Usage",
]
