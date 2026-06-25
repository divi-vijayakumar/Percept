"""Public benchmark artifact validation."""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from percept.bench import ValidationError, validate_baselines, validate_golden_manifest


ROOT = Path(__file__).resolve().parents[3]


class BenchmarkArtifactValidationTests(unittest.TestCase):
    def test_repo_baseline_registry_validates(self):
        data = validate_baselines(ROOT / "packages/percept-vision/eval/baselines.json")
        self.assertGreaterEqual(len(data["baselines"]), 1)

    def test_repo_golden_v1_draft_validates_as_draft(self):
        data = validate_golden_manifest(ROOT / "eval/golden-v1/MANIFEST.json")
        self.assertEqual(data["status"], "draft")

    def test_draft_manifest_fails_release_validation(self):
        with self.assertRaises(ValidationError):
            validate_golden_manifest(ROOT / "eval/golden-v1/MANIFEST.json", release=True)

    def test_release_manifest_requires_hashed_test_items(self):
        with TemporaryDirectory() as d:
            p = Path(d) / "MANIFEST.json"
            p.write_text(
                json.dumps(
                    {
                        "version": "golden-v1",
                        "status": "release",
                        "tracks": [
                            {"id": "relational", "source": "self-recorded", "status": "ready", "metrics": ["f1"]}
                        ],
                        "splits": {"dev": [], "test": [{"id": "case-1", "track": "relational"}]},
                    }
                )
            )
            with self.assertRaises(ValidationError):
                validate_golden_manifest(p, release=True)


if __name__ == "__main__":
    unittest.main()
