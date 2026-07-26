"""percept-vision eval harness (L2) — the replay-and-align scorer + the CI-gate runner.

Distinct from ``percept.eval`` (the golden-set *loader*): this top-level ``eval`` package is the
legacy replay scorer. The Makefile drives it as ``python -m eval.run`` from the SDK root.

The **alignment + metric math** in :mod:`eval.replay` is pure and **L1-testable on synthetic
fire/edge lists** — no models, no frames. The model-driven replay (running a real clip through a
``Percept``) and the CI gates live in :mod:`eval.run` (L2; skips without keys).

The **consumer-facing** path — per-concern precision / recall from a labelled-clip *manifest* that
runs fully offline on the fake backend — lives in :mod:`eval.benchmark` (drive your own benchmark;
see ``docs/eval.md``).
"""

from typing import Any

from .replay import (
    AlignmentResult,
    Edge,
    Fire,
    ReplayMetrics,
    align,
    pauc,
    score,
)

# ``eval.benchmark`` names are re-exported lazily: importing them eagerly here would pull the
# submodule into ``sys.modules`` before ``python -m eval.benchmark`` runs it, which trips runpy's
# "found in sys.modules ... prior to execution" warning. ``__getattr__`` (PEP 562) keeps the public
# ``from eval import Manifest`` surface without that import-order coupling.
_BENCHMARK_EXPORTS = frozenset(
    {
        "Manifest",
        "ManifestClip",
        "ConcernObservation",
        "ConcernReport",
        "BenchmarkGate",
        "score_concern",
        "per_concern_report",
        "run_manifest",
        "check_benchmark_gate",
    }
)


def __getattr__(name: str) -> Any:  # PEP 562 — lazy re-export of eval.benchmark
    if name in _BENCHMARK_EXPORTS:
        from . import benchmark

        return getattr(benchmark, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "Edge",
    "Fire",
    "AlignmentResult",
    "ReplayMetrics",
    "align",
    "pauc",
    "score",
    # consumer-facing per-concern benchmark (eval.benchmark; lazily re-exported)
    "Manifest",
    "ManifestClip",
    "ConcernObservation",
    "ConcernReport",
    "BenchmarkGate",
    "score_concern",
    "per_concern_report",
    "run_manifest",
    "check_benchmark_gate",
]
