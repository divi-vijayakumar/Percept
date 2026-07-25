"""Public benchmark CLI shim.

The benchmark harness lives in the top-level ``eval`` package, which ships with the SOURCE REPO, not
the installed wheel (the golden data isn't packaged either). So this shim must import cleanly WITHOUT
``eval`` present — only actually *running* the CLI, or touching a re-exported harness symbol, needs it.
The public names are re-exported **lazily** (PEP 562 ``__getattr__``), and the CLI degrades to a
"run from a checkout" note instead of crashing at import on a bare wheel install.

    percept-bench eval/benchmark_ci_gate/manifest.json  # per-concern P/R
    percept-bench scorecard --manifest <path>          # multi-track card
    percept-bench audit                                # validate golden artifacts
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

_HARNESS_MISSING = (
    "percept-bench: the benchmark harness ships with the source repo, not the installed wheel.\n"
    "Run it from a repo checkout (the top-level `eval` package must be importable), e.g.:\n"
    "  git clone https://github.com/divi-vijayakumar/Percept && cd Percept\n"
    "  percept-bench <manifest.json>\n"
    "Docs: https://github.com/divi-vijayakumar/Percept/blob/main/docs/how-to/eval.md"
)

# Public re-exports, resolved lazily from the repo-only harness so ``import percept.bench`` never
# requires ``eval``. ``bench.Manifest`` etc. still resolve in a checkout; out-of-repo they raise a
# clear ImportError only when actually accessed.
_LAZY = {
    "BenchmarkGate": "eval.benchmark",
    "ConcernObservation": "eval.benchmark",
    "ConcernReport": "eval.benchmark",
    "Manifest": "eval.benchmark",
    "ManifestClip": "eval.benchmark",
    "check_benchmark_gate": "eval.benchmark",
    "edges_from_manifest": "eval.benchmark",
    "format_report": "eval.benchmark",
    "per_concern_report": "eval.benchmark",
    "run_manifest": "eval.benchmark",
    "score_concern": "eval.benchmark",
    "ValidationError": "eval.bench.validate",
    "audit_artifacts": "eval.bench.validate",
    "format_audit": "eval.bench.validate",
    "validate_baselines": "eval.bench.validate",
    "validate_golden_manifest": "eval.bench.validate",
}


def __getattr__(name: str):  # PEP 562 — lazy re-exports from the repo-only harness
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(importlib.import_module(module), name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY))


def _audit_main(argv: list[str] | None = None) -> int:
    import argparse

    from eval.bench.validate import ValidationError, audit_artifacts, format_audit  # lazy: repo-only

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


def _reproduce_main(argv: list[str] | None = None) -> int:
    """Audit the inputs, execute the offline scorecard, and require its artifact."""
    import argparse

    ap = argparse.ArgumentParser(
        prog="percept-bench reproduce",
        description="Reproduce the keyless OSS scorecard artifact (Cloud qualification is separate).",
    )
    ap.add_argument("--manifest", default="eval/golden-v1/MANIFEST.json")
    ap.add_argument("--baselines", default="packages/percept-vision/eval/baselines.json")
    ap.add_argument("--out", default="scorecard.txt")
    args = ap.parse_args(argv)

    audit_rc = _audit_main(["--manifest", args.manifest, "--baselines", args.baselines])
    if audit_rc:
        return audit_rc
    from eval.bench.run import main as _scorecard_main

    scorecard_rc = _scorecard_main(
        ["--manifest", args.manifest, "--out", args.out, "--offline"]
    )
    if scorecard_rc:
        return scorecard_rc
    artifact = Path(args.out)
    if not artifact.exists() or artifact.stat().st_size == 0:
        print(f"REPRODUCE FAILED: required scorecard artifact is missing or empty: {artifact}")
        return 1
    from percept.harness.bench_gating import main as _call_reduction_main

    reduction_rc = _call_reduction_main()
    if reduction_rc:
        return reduction_rc
    print(
        "REPRODUCTION: PASS — artifact audit, offline DRAFT scorecard, and call-reduction "
        "diagnostic executed."
    )
    print(
        "OSS LAUNCH QUALIFICATION: BLOCKED — the card is DRAFT with modeled/incomplete cells, "
        "missing coverage, and skipped tracks; call reduction is not matched-quality/same-archetype."
    )
    print(
        "Cloud provider cost qualification: BLOCKED — validated budget declaration is not hard "
        "spend enforcement."
    )
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
            from eval.bench.run import main as _scorecard_main  # lazy: repo-only harness

            return _scorecard_main(argv[1:])
        if argv and argv[0] == "reproduce":
            return _reproduce_main(argv[1:])
        from eval.benchmark import main as _benchmark_main  # lazy: repo-only harness

        return _benchmark_main(argv)
    except ImportError as exc:
        # The `eval` harness isn't importable — a wheel install without the source repo. This is why
        # the imports above are lazy: it protects the `percept.bench:main` entry point from crashing
        # at import, so we can print guidance instead of a bare traceback.
        print(_HARNESS_MISSING, file=sys.stderr)
        print(f"  (missing: {exc.name or exc})", file=sys.stderr)
        return 2
    except FileNotFoundError as exc:
        missing = getattr(exc, "filename", None) or str(exc)
        # Only the DEFAULT (bundled-golden) paths get the "ships with the repo" guidance — a user's own
        # --manifest/clip typo must surface the real missing file, not a misdiagnosis.
        defaults = ("eval/golden-v1/MANIFEST.json", "packages/percept-vision/eval/baselines.json")
        if any(str(missing).replace("\\", "/").endswith(d) for d in defaults):
            print(
                f"percept-bench: bundled input not found: {missing}\n"
                "The golden benchmark set ships with the source repo, not the installed wheel.\n"
                "Run from a repo checkout, or pass your own manifest, e.g.:\n"
                "  percept-bench <manifest.json>                              # per-concern P/R on your clips\n"
                "  percept-bench scorecard --manifest <path/to/MANIFEST.json>\n"
                "Docs: https://github.com/divi-vijayakumar/Percept/blob/main/docs/how-to/eval.md",
                file=sys.stderr,
            )
        else:
            print(f"percept-bench: input not found: {missing}", file=sys.stderr)
        return 2


def scorecard_main(argv: list[str] | None = None) -> int:
    """Emit the public multi-track benchmark snapshot card."""
    from eval.bench.run import main as _scorecard_main  # lazy: repo-only harness

    return _scorecard_main(argv)


if __name__ == "__main__":  # pragma: no cover - exercised through subprocess tests
    raise SystemExit(main())


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
