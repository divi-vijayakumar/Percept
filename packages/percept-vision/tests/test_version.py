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
        self.assertEqual(percept.__version__, _pyproject_version())


if __name__ == "__main__":
    unittest.main()
