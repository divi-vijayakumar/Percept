"""End-to-end validation: guardrails, the test suite, the offline quickstart, and a clean install.

Portable (no shell `timeout`): every step runs as a subprocess with a Python-side timeout,
and the quickstart's output is checked, not just its exit code.

The clean-install step builds the wheel from a fresh copy of ``packages/percept-vision`` (no
stale ``build/``), checks that the wheel's ``percept/`` files are exactly the tracked ones
(``git ls-files``), installs it into an empty directory and runs the quickstart against that
install, not the source tree. Building needs ``pip`` or ``uv``, and the build backend
(setuptools) is fetched as for any wheel build.

    python tools/validate.py [--python PATH]
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "packages" / "percept-vision"
QUICKSTART_EXPECTS = ("met ", "cleared ", "uncertain ", "coverage:", "reproduced exactly")


def run(cmd: list[str], *, timeout: float, cwd: Path, env: dict[str, str] | None = None
        ) -> subprocess.CompletedProcess[str]:
    print(f"$ {' '.join(cmd)}", flush=True)
    try:
        return subprocess.run(cmd, cwd=cwd, text=True, capture_output=True, timeout=timeout, env=env)
    except subprocess.TimeoutExpired as exc:
        raise SystemExit(f"FAILED: timed out after {timeout:.0f} s: {' '.join(cmd)}\n{exc.stdout or ''}") from exc


def quickstart(py: str, *, env: dict[str, str] | None = None, label: str = "quickstart") -> bool:
    with tempfile.TemporaryDirectory() as tmp:  # the quickstart writes its trace and store here
        result = run([py, str(ROOT / "examples" / "quickstart.py")], timeout=120, cwd=Path(tmp), env=env)
    print(result.stdout, end="")
    missing = [s for s in QUICKSTART_EXPECTS if s not in result.stdout]
    if result.returncode != 0 or missing:
        print(result.stderr[-2000:], file=sys.stderr)
        print(f"FAILED: {label} exited {result.returncode}; missing output: {missing}")
        return False
    return True


def build_wheel(py: str, src: Path, out: Path) -> Path | None:
    """Build the wheel with pip when the interpreter has it, else with uv."""
    has_pip = run([py, "-m", "pip", "--version"], timeout=60, cwd=src).returncode == 0
    uv = shutil.which("uv")
    if has_pip:
        cmd = [py, "-m", "pip", "wheel", "--no-deps", "--quiet", "--wheel-dir", str(out), str(src)]
    elif uv is not None:
        cmd = [uv, "build", "--wheel", "--out-dir", str(out), "--python", py, str(src)]
    else:
        print("FAILED: building the wheel needs pip in the interpreter, or uv on PATH")
        return None
    result = run(cmd, timeout=600, cwd=src)
    wheels = sorted(out.glob("percept_vision-*.whl"))
    if result.returncode != 0 or len(wheels) != 1:
        print(result.stdout[-2000:], result.stderr[-2000:], file=sys.stderr)
        print(f"FAILED: wheel build exited {result.returncode}, produced {[w.name for w in wheels]}")
        return None
    return wheels[0]


def clean_install(py: str) -> bool:
    """Build from a clean copy, compare the wheel with git, install it and run the quickstart."""
    tracked = run(["git", "ls-files", "packages/percept-vision/percept"], timeout=60, cwd=ROOT)
    if tracked.returncode != 0:
        print(tracked.stderr[-2000:], file=sys.stderr)
        print("FAILED: git ls-files (the clean-install check compares the wheel with the tracked files)")
        return False
    expected = {line.split("packages/percept-vision/", 1)[1] for line in tracked.stdout.splitlines() if line}
    version = re.search(r'^version\s*=\s*"([^"]+)"', (PACKAGE / "pyproject.toml").read_text(), re.M)
    with tempfile.TemporaryDirectory() as tmp_name:
        tmp = Path(tmp_name)
        src = tmp / "percept-vision"
        shutil.copytree(PACKAGE, src, ignore=shutil.ignore_patterns(
            "build", "dist", "*.egg-info", "__pycache__", ".pytest_cache", "tests"))
        wheel = build_wheel(py, src, tmp / "dist")
        if wheel is None:
            return False
        with zipfile.ZipFile(wheel) as zf:
            names = zf.namelist()
            shipped = {n for n in names if n.startswith("percept/") and not n.endswith("/")}
            tops = {n.split("/", 1)[0] for n in names}
            others = sorted(t for t in tops if t != "percept" and not t.endswith(".dist-info"))
            meta = next((n for n in names if n.endswith(".dist-info/WHEEL")), None)
            purelib = meta is not None and "Root-Is-Purelib: true" in zf.read(meta).decode()
            site = tmp / "site"
            zf.extractall(site)    # a pure-Python wheel with no scripts installs by unpacking it
        print(f"wheel: {wheel.name}, {len(shipped)} files under percept/")
        if shipped != expected or others or not purelib:
            print(f"FAILED: the wheel does not match the tracked tree; not shipped: {sorted(expected - shipped)}; "
                  f"untracked but shipped: {sorted(shipped - expected)}; other top-level entries: {others}; "
                  f"pure-Python: {purelib}")
            return False
        env = {k: v for k, v in os.environ.items() if not k.endswith("_API_KEY")}
        env["PYTHONPATH"] = str(site)
        probe = run([py, "-c", "import percept; print(percept.__file__); print(percept.__version__)"],
                    timeout=60, cwd=tmp, env=env)
        where, _, got = probe.stdout.strip().partition("\n")
        if probe.returncode != 0 or not Path(where).resolve().is_relative_to(site.resolve()) \
                or version is None or got != version.group(1):
            print(probe.stderr[-2000:], file=sys.stderr)
            print(f"FAILED: the installed wheel is not what imports: {where!r} version {got!r}")
            return False
        print(f"installed: percept {got} from the wheel")
        return quickstart(py, env=env, label="quickstart against the installed wheel")


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

    if not quickstart(py):
        return 1
    if not clean_install(py):
        return 1
    print("validate: ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
