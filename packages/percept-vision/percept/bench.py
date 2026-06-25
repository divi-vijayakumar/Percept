"""Public benchmark CLI shim.

The implementation lives in ``eval.benchmark`` for repo continuity, but installed users should not
need to know about the historical top-level ``eval`` package. This module gives them the stable
surface:

    python -m percept.bench examples/benchmark/manifest.json
    percept-bench examples/benchmark/manifest.json
"""

from __future__ import annotations

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
from eval.benchmark import main as _main


def main(argv: list[str] | None = None) -> int:
    """Run the per-concern benchmark CLI."""
    return _main(argv)


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
    "main",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
