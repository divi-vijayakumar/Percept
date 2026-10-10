"""End-to-end validation: guardrails, the test suite and the offline quickstart.

Portable (no shell `timeout`): every step runs as a subprocess with a Python-side timeout,
and the quickstart's output is checked, not just its exit code.

    python tools/validate.py [--python PATH]
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
QUICKSTART_EXPECTS = ("met ", "cleared ", "uncertain ", "coverage:", "reproduced exactly")


def run(cmd: list[str], *, timeout: float, cwd: Path) -> subprocess.CompletedProcess[str]:
    print(f"$ {' '.join(cmd)}", flush=True)
    try:
        return subprocess.run(cmd, cwd=cwd, text=True, capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        raise SystemExit(f"FAILED: timed out after {timeout:.0f} s: {' '.join(cmd)}\n{exc.stdout or ''}") from exc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args(argv)
    # Absolute path: the quickstart runs from a temporary directory.
    py = shutil.which(args.python) or args.python
    py = str(Path(py).absolute())

    steps = [
        ([py, "tools/guardrails/guardrails.py", "all"], 120),
        ([py, "-m", "pytest", "-q", "packages/percept-vision/tests", "tools/guardrails/tests"], 900),
    ]
    for cmd, timeout in steps:
        result = run(cmd, timeout=timeout, cwd=ROOT)
        print(result.stdout[-2000:], end="")
        if result.returncode != 0:
            print(result.stderr[-2000:], file=sys.stderr)
            print(f"FAILED: {' '.join(cmd)} exited {result.returncode}")
            return 1

    with tempfile.TemporaryDirectory() as tmp:  # the quickstart writes its trace and store here
        result = run([py, str(ROOT / "examples" / "quickstart.py")], timeout=120, cwd=Path(tmp))
    print(result.stdout, end="")
    missing = [s for s in QUICKSTART_EXPECTS if s not in result.stdout]
    if result.returncode != 0 or missing:
        print(result.stderr[-2000:], file=sys.stderr)
        print(f"FAILED: quickstart exited {result.returncode}; missing output: {missing}")
        return 1
    print("validate: ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
