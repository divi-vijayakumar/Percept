"""Public benchmark import/CLI shim."""

from __future__ import annotations

import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory

from percept import bench


ROOT = Path(__file__).resolve().parents[3]


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

    def test_scorecard_require_release_rejects_draft_card(self):
        with TemporaryDirectory() as d:
            out = Path(d) / "scorecard.txt"
            manifest = Path(d) / "MANIFEST.json"
            manifest.write_text('{"version":"test"}')
            buf = StringIO()
            with redirect_stdout(buf):
                rc = bench.main(
                    ["scorecard", "--manifest", str(manifest), "--out", str(out), "--require-release"]
                )
            self.assertEqual(rc, 1)
            self.assertIn("SCORECARD RELEASE GATE FAILED", buf.getvalue())
            self.assertIn("stamp: DRAFT", out.read_text())

    def test_audit_subcommand_accepts_draft_artifacts(self):
        with TemporaryDirectory() as d:
            manifest = Path(d) / "MANIFEST.json"
            baselines = Path(d) / "baselines.json"
            manifest.write_text(
                """
                {
                  "version": "golden-v1-draft",
                  "status": "draft",
                  "tracks": [{"id": "relational", "source": "self", "status": "draft", "metrics": ["f1"]}],
                  "splits": {"dev": [], "test": []}
                }
                """
            )
            baselines.write_text(
                """
                {
                  "baselines": [{
                    "id": "ref",
                    "system": "Reference",
                    "kind": "reference_bar",
                    "controlled_ab": false,
                    "source": "https://example.com",
                    "metric": "f1",
                    "comparability": "context only"
                  }]
                }
                """
            )
            buf = StringIO()
            with redirect_stdout(buf):
                rc = bench.main(["audit", "--manifest", str(manifest), "--baselines", str(baselines)])
            self.assertEqual(rc, 0)
            self.assertIn("draft-ok", buf.getvalue())

    def test_audit_release_mode_rejects_draft_manifest(self):
        buf = StringIO()
        with redirect_stdout(buf):
            rc = bench.main(
                [
                    "audit",
                    "--release",
                    "--manifest",
                    str(ROOT / "eval/golden-v1/MANIFEST.json"),
                    "--baselines",
                    str(ROOT / "packages/percept-vision/eval/baselines.json"),
                ]
            )
        self.assertEqual(rc, 1)
        self.assertIn("AUDIT FAILED", buf.getvalue())


if __name__ == "__main__":
    unittest.main()
