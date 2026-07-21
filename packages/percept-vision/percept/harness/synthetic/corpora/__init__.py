"""Per-capability corpus generators.

Each module exposes ``NAME`` and ``build() -> (stream_doc, labels_doc)`` and is a PURE function of
``schema.SEED`` plus its own constants — no wall clock, no unseeded randomness, no I/O.
"""

from __future__ import annotations

from typing import Callable

from . import (
    continuation,
    episode_counting,
    idempotency,
    intervals,
    longhorizon,
    mutation,
    subject_identity,
)

MODULES = (
    subject_identity,
    continuation,
    mutation,
    episode_counting,
    intervals,
    idempotency,
    longhorizon,
)

#: corpus name -> builder
BUILDERS: dict[str, Callable[[], tuple[dict, dict]]] = {m.NAME: m.build for m in MODULES}

__all__ = ["MODULES", "BUILDERS"]
