"""CI regression-gate template — wire a per-concern P/R threshold into your build.

This is the consumer analog of the SDK's own ``eval.run`` CI gate (which asserts
``false_fire == 0`` on the held-out test split). Here you point the harness at *your* benchmark
manifest, choose the floors your product needs, and let a non-zero exit code fail the build.

Run it (fully offline — fake backend, no keys):

    PYTHONPATH=packages/percept-vision python3 examples/benchmark/ci_gate.py

Wire it into CI as a step that must succeed, e.g. GitHub Actions::

    - name: Percept per-concern regression gate
      run: PYTHONPATH=packages/percept-vision python3 examples/benchmark/ci_gate.py

When a code/prompt change regresses a concern below its floor (or introduces *any* false-fire),
the step exits non-zero and the PR goes red — with the offending concern named in the output.
"""

from pathlib import Path

from eval.benchmark import (
    BenchmarkGate,
    Manifest,
    check_benchmark_gate,
    format_report,
    run_manifest,
)

MANIFEST = Path(__file__).with_name("manifest.json")

# The trust posture, made explicit:
#   * max_false_fire = 0  -> never confidently-wrong fire (the same hard gate as eval.run).
#   * min_recall      -> every concern must catch this fraction of the moments it should fire.
#   * min_precision   -> of the fires a concern emits, this fraction must be right.
# Tune the floors to your product's tolerance; loosen recall before you ever loosen false-fire.
# (Flip min_recall to 1.0 with a clip the model misses, or add a false-fire clip, to see it go red.)
GATE = BenchmarkGate(max_false_fire=0, min_precision=1.0, min_recall=0.9)


def main() -> int:
    manifest = Manifest.load(MANIFEST)
    report = run_manifest(manifest)  # offline replay on the deterministic fake backend
    print(format_report(report))

    passed, reasons = check_benchmark_gate(report, GATE)
    if not passed:
        print("REGRESSION GATE FAILED:")
        for r in reasons:
            print(f"  - {r}")
        return 1
    print("REGRESSION GATE PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
