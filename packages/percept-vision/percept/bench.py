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
from eval.bench.validate import (
    ValidationError,
    audit_artifacts,
    format_audit,
    validate_baselines,
    validate_golden_manifest,
)


def _audit_main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(
        prog="percept-bench audit",
        description="Validate Percept benchmark manifest and baseline artifacts.",
    )
    ap.add_argument("--manifest", default="eval/golden-v1/MANIFEST.json")
    ap.add_argument("--baselines", default="packages/percept-vision/eval/baselines.json")
    ap.add_argument(
        "--release",
        action="store_true",
        help="require release-ready artifacts: non-empty hashed test split and release manifest status",
    )
    args = ap.parse_args(argv)

    try:
        audit = audit_artifacts(manifest=args.manifest, baselines=args.baselines, release=args.release)
    except ValidationError as exc:
        print(f"BENCHMARK ARTIFACT AUDIT FAILED: {exc}")
        return 1
    print(format_audit(audit))
    return 0


def main(argv: list[str] | None = None) -> int:
    """Run the public benchmark CLI.

    ``percept-bench manifest.json`` runs the per-concern P/R harness.
    ``percept-bench scorecard`` emits the multi-track snapshot card.
    """
    if argv is None:
        argv = sys.argv[1:]
    try:
        if argv and argv[0] == "audit":
            return _audit_main(argv[1:])
        if argv and argv[0] == "scorecard":
            return _scorecard_main(argv[1:])
        return _benchmark_main(argv)
    except FileNotFoundError as exc:
        # The bundled golden set is repo data, not packaged in the wheel — so the default-manifest
        # commands only work from a source checkout. Fail with guidance instead of a bare traceback.
        missing = getattr(exc, "filename", None) or exc
        print(
            f"percept-bench: input not found: {missing}\n"
            "The bundled golden benchmark set ships with the source repo, not the installed wheel.\n"
            "Run from a repo checkout, or pass your own manifest, e.g.:\n"
            "  percept-bench <manifest.json>                              # per-concern P/R on your clips\n"
            "  percept-bench scorecard --manifest <path/to/MANIFEST.json>\n"
            "Docs: https://github.com/divi-vijayakumar/Percept/blob/main/docs/how-to/eval.md",
            file=sys.stderr,
        )
        return 2


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
    "audit_artifacts",
    "format_audit",
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
