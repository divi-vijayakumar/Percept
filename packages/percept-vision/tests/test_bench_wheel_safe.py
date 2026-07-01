"""Wheel-safety guard: ``percept.bench`` must import + degrade WITHOUT the repo-only ``eval`` harness.

The wheel ships only ``percept*`` (no top-level ``eval`` namespace squat). This mirrors
``test_core_no_vendor_imports``: a fresh interpreter with ``eval`` blocked from the import system,
asserting ``import percept.bench`` still succeeds and the ``percept.bench:main`` entry point returns
rc 2 with guidance instead of crashing at import. Stops a top-level ``from eval …`` from creeping back
in and silently re-breaking the wheel.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import unittest
from pathlib import Path

_PKG_ROOT = str(Path(__file__).resolve().parent.parent)  # packages/percept-vision (has percept/ + eval/)

_CODE = textwrap.dedent(
    """
    import sys

    class _BlockEval:
        def find_spec(self, name, path=None, target=None):
            if name == "eval" or name.startswith("eval."):
                raise ModuleNotFoundError(name)  # pretend the harness isn't installed (wheel case)
            return None

    sys.meta_path.insert(0, _BlockEval())
    for _m in [m for m in sys.modules if m == "eval" or m.startswith("eval.")]:
        del sys.modules[_m]

    import percept.bench                     # must import with NO top-level eval dependency
    rc = percept.bench.main(["audit"])       # must not crash — guidance + rc 2
    sys.exit(0 if rc == 2 else 3)
    """
)


class BenchWheelSafeTests(unittest.TestCase):
    def test_bench_imports_and_degrades_without_eval_harness(self):
        env = {**os.environ, "PYTHONPATH": _PKG_ROOT}
        r = subprocess.run([sys.executable, "-c", _CODE], capture_output=True, text=True, env=env)
        self.assertEqual(r.returncode, 0, f"import/degrade failed\nstdout={r.stdout!r}\nstderr={r.stderr!r}")
        self.assertIn("ships with the source repo", r.stderr)


if __name__ == "__main__":
    unittest.main()
