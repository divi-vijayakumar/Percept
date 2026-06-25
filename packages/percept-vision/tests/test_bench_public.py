"""Public benchmark import/CLI shim."""

from __future__ import annotations

import unittest

from percept import bench


class TestPublicBenchShim(unittest.TestCase):
    def test_exports_manifest_and_report_helpers(self):
        self.assertTrue(hasattr(bench, "Manifest"))
        self.assertTrue(hasattr(bench, "run_manifest"))
        self.assertTrue(callable(bench.main))


if __name__ == "__main__":
    unittest.main()
