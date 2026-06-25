"""Public benchmark CLI shim.

The implementation lives in ``eval.benchmark`` for repo continuity, but installed users should not
need to know about the historical top-level ``eval`` package. This module gives them the stable
surface:

    python -m percept.bench examples/benchmark/manifest.json
    percept-bench examples/benchmark/manifest.json
"""

from __future__ import annotations

import sys

from eval.benchmark import (  # noqa: F401
    BenchmarkGate,
    ConcernObservation,
    ConcernReport,
    Manifest,
    ManifestClip,
    check_benchmark_gate,
    edges_from_manifest,
    format_report,
    per_concern_report,
    run_manifest,
    score_concern,
)
from eval.benchmark import main as _benchmark_main
from eval.bench.run import main as _scorecard_main
from eval.bench.validate import ValidationError, validate_baselines, validate_golden_manifest


def main(argv: list[str] | None = None) -> int:
    """Run the public benchmark CLI.

    ``percept-bench manifest.json`` runs the per-concern P/R harness.
    ``percept-bench scorecard`` emits the multi-track snapshot card.
    """
    if argv is None:
        argv = sys.argv[1:]
    if argv and argv[0] == "scorecard":
        return _scorecard_main(argv[1:])
    return _benchmark_main(argv)


def scorecard_main(argv: list[str] | None = None) -> int:
    """Emit the public multi-track benchmark snapshot card."""
    return _scorecard_main(argv)


__all__ = [
    "BenchmarkGate",
    "ConcernObservation",
    "ConcernReport",
    "Manifest",
    "ManifestClip",
    "check_benchmark_gate",
    "edges_from_manifest",
    "format_report",
    "per_concern_report",
    "run_manifest",
    "score_concern",
    "ValidationError",
    "validate_baselines",
    "validate_golden_manifest",
    "main",
    "scorecard_main",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
