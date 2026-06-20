"""Structured JSON logging + a tiny metrics registry + a /health payload.

Stdlib only. The harness/assistant can route these to wherever; the core just emits.
"""

from __future__ import annotations

import json
import logging
import sys
from collections import Counter
from typing import Any


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        extra = getattr(record, "fields", None)
        if isinstance(extra, dict):
            payload.update(extra)
        return json.dumps(payload, default=str)


def get_logger(name: str = "percept") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(_JsonFormatter())
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


def log(logger: logging.Logger, msg: str, **fields: Any) -> None:
    """Emit a structured line: ``log(lg, "fired", goal_id=g, conf=0.9)``."""
    logger.info(msg, extra={"fields": fields})


class Metrics:
    """A minimal in-process counter registry."""

    def __init__(self) -> None:
        self._counters: Counter[str] = Counter()

    def incr(self, name: str, n: int = 1) -> None:
        self._counters[name] += n

    def get(self, name: str) -> int:
        return self._counters[name]

    def snapshot(self) -> dict[str, int]:
        return dict(self._counters)


def health(version: str, backends: dict[str, str], metrics: Metrics | None = None) -> dict:
    """The /health payload — what's wired and how it's doing."""
    return {
        "status": "ok",
        "version": version,
        "backends": backends,
        "metrics": metrics.snapshot() if metrics else {},
    }


__all__ = ["get_logger", "log", "Metrics", "health"]
