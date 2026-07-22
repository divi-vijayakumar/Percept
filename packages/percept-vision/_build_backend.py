"""PEP 517 build backend wrapper — stamps ``percept/_build_info.py`` before packaging.

Delegates every hook to ``setuptools.build_meta``; the only thing this wrapper adds is
(re)generating ``percept/_build_info.py`` immediately before ``build_sdist``/``build_wheel``/
``build_editable`` run, so a built artifact (wheel or sdist) can report exactly what it was cut
from: the package version, the git commit it was built at, the frozen wire ``CONTRACT_VERSION``
it shipped with, and a UTC build timestamp.

A plain source checkout that never runs a build (e.g. ``pip install -e .`` without going through
this backend, or just importing from a git clone) keeps the checked-in fallback stub in
``percept/_build_info.py``, which reports ``"unreleased"`` for everything — see
``percept.build_info()``.
"""

from __future__ import annotations

import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

# Re-export the whole setuptools backend surface, then override the three hooks that need to
# stamp first. Anything this module doesn't define (get_requires_for_build_*, prepare_metadata_*,
# ...) falls through to setuptools unchanged.
from setuptools.build_meta import *  # noqa: F401,F403
from setuptools.build_meta import build_editable as _orig_build_editable
from setuptools.build_meta import build_sdist as _orig_build_sdist
from setuptools.build_meta import build_wheel as _orig_build_wheel

_HERE = Path(__file__).resolve().parent
_STAMP_PATH = _HERE / "percept" / "_build_info.py"


def _git_commit() -> str:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=_HERE,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except Exception:  # pragma: no cover - defensive (no git binary, etc.)
        return "unknown"
    commit = proc.stdout.strip()
    return commit if proc.returncode == 0 and commit else "unknown"


def _project_version() -> str:
    text = (_HERE / "pyproject.toml").read_text(encoding="utf-8")
    m = re.search(r'(?m)^\s*version\s*=\s*"([^"]+)"', text)
    return m.group(1) if m else "0.0.0+unknown"


def _contract_version() -> str:
    text = (_HERE / "percept" / "contracts" / "__init__.py").read_text(encoding="utf-8")
    m = re.search(r'(?m)^CONTRACT_VERSION\s*=\s*"([^"]+)"', text)
    return m.group(1) if m else "0.0.0+unknown"


def _write_stamp() -> None:
    stamp = '''"""Build provenance — generated at build time by ``_build_backend.py``. Do not edit by hand.

Reports the provenance of THIS built artifact: the package version, the git commit it was built
from, the frozen wire ``CONTRACT_VERSION`` it shipped with, and a UTC build timestamp. Read this
through ``percept.build_info()``, never by importing this module directly.
"""

PACKAGE_VERSION = {package_version!r}
GIT_COMMIT = {git_commit!r}
CONTRACT_VERSION = {contract_version!r}
BUILD_TIMESTAMP = {build_timestamp!r}
'''.format(
        package_version=_project_version(),
        git_commit=_git_commit(),
        contract_version=_contract_version(),
        build_timestamp=datetime.now(timezone.utc).isoformat(),
    )
    _STAMP_PATH.write_text(stamp, encoding="utf-8")


def build_sdist(sdist_directory, config_settings=None):
    _write_stamp()
    return _orig_build_sdist(sdist_directory, config_settings)


def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    _write_stamp()
    return _orig_build_wheel(wheel_directory, config_settings, metadata_directory)


def build_editable(wheel_directory, config_settings=None, metadata_directory=None):
    _write_stamp()
    return _orig_build_editable(wheel_directory, config_settings, metadata_directory)
