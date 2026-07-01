"""__version__ is single-sourced from pyproject — the dev-fallback path must not drift."""

from __future__ import annotations

import re
import unittest
from pathlib import Path

import percept


def _pyproject_version() -> str:
    for parent in Path(__file__).resolve().parents:
        pp = parent / "pyproject.toml"
        if pp.is_file():
            m = re.search(r'(?m)^\s*version\s*=\s*"([^"]+)"', pp.read_text())
            if m:
                return m.group(1)
    raise unittest.SkipTest("pyproject.toml not found")


class VersionTests(unittest.TestCase):
    def test_version_matches_pyproject(self):
        # __version__ comes from installed dist metadata when available, else the pyproject fallback. In a
        # fresh install both equal pyproject. An editable checkout left stale after a version bump (metadata
        # captured at install time) is an environment artifact, not code drift — the fallback reads
        # pyproject — so skip that case rather than false-fail; CI (fresh install) still asserts equality.
        pyproject_v = _pyproject_version()
        try:
            from importlib.metadata import PackageNotFoundError, version

            installed = version("percept-vision")
        except PackageNotFoundError:
            installed = None
        if installed is not None and installed != pyproject_v:
            raise unittest.SkipTest(
                f"stale editable install ({installed} != pyproject {pyproject_v}); reinstall to sync"
            )
        self.assertEqual(percept.__version__, pyproject_v)


if __name__ == "__main__":
    unittest.main()
