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
    JudgeItem,
    JudgeRequest,
    JudgeResult,
    Observation,
    PriceModel,
    RequestUsage,
    Sink,
    Source,
    Store,
    Trigger,
    Usage,
)
from percept import decide  # internal for now: the runtime drives it; not exported
from percept import judges
from percept.delivery import MemoryStore, SqliteStore
from percept.monitor import Monitor
from percept import trace  # TraceWriter, read, replay(path); the pure replay is percept.replay
from percept.sources import PushSource, open_source
from percept.triggers import every, on_change

__version__ = "1.0.0.dev1"

__all__ = [
    "Condition",
    "Decision",
    "Delivery",
    "EvidenceRef",
    "Frame",
    "Health",
    "Judge",
    "JudgeItem",
    "JudgeRequest",
    "JudgeResult",
    "MemoryStore",
    "Monitor",
    "Observation",
    "PriceModel",
    "PushSource",
    "RequestUsage",
    "Sink",
    "Source",
    "SqliteStore",
    "Store",
    "Trigger",
    "Usage",
    "every",
    "judges",
    "on_change",
    "open_source",
    "trace",
]
