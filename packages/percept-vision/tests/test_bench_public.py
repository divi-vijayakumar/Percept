"""Public benchmark import/CLI shim."""

from __future__ import annotations

import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory

from percept import bench


class TestPublicBenchShim(unittest.TestCase):
    def test_exports_manifest_and_report_helpers(self):
        self.assertTrue(hasattr(bench, "Manifest"))
        self.assertTrue(hasattr(bench, "run_manifest"))
        self.assertTrue(callable(bench.main))
        self.assertTrue(callable(bench.scorecard_main))

    def test_scorecard_subcommand_writes_card(self):
        with TemporaryDirectory() as d:
            out = Path(d) / "scorecard.txt"
            manifest = Path(d) / "MANIFEST.json"
            manifest.write_text('{"version":"test"}')
            with redirect_stdout(StringIO()):
                rc = bench.main(["scorecard", "--manifest", str(manifest), "--out", str(out)])
            self.assertEqual(rc, 0)
            text = out.read_text()
            self.assertIn("Percept Benchmark Snapshot", text)
            self.assertIn("sha256:", text)


if __name__ == "__main__":
    unittest.main()
