"""percept-vision eval harness (L2) — the replay-and-align scorer + the CI-gate runner.

Distinct from ``percept.eval`` (the golden-set *loader*): this top-level ``eval`` package is the
*scorer* (spec §10.2). The Makefile drives it as ``python -m eval.run`` from the SDK root.

The **alignment + metric math** in :mod:`eval.replay` is pure and **L1-testable on synthetic
fire/edge lists** — no models, no frames. The model-driven replay (running a real clip through a
``Percept``) and the CI gates live in :mod:`eval.run` (L2; skips without keys).
"""

from .replay import (
    AlignmentResult,
    Edge,
    Fire,
    ReplayMetrics,
    align,
    pauc,
    score,
)

__all__ = [
    "Edge",
    "Fire",
    "AlignmentResult",
    "ReplayMetrics",
    "align",
    "pauc",
    "score",
]
