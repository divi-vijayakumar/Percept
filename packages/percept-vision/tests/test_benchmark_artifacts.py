"""Public benchmark artifact validation."""

from __future__ import annotations

import json
import hashlib
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from eval.bench import perception_test as pt
from percept.bench import ValidationError, validate_baselines, validate_golden_manifest


ROOT = Path(__file__).resolve().parents[3]


class BenchmarkArtifactValidationTests(unittest.TestCase):
    def test_repo_baseline_registry_validates(self):
        data = validate_baselines(ROOT / "packages/percept-vision/eval/baselines.json")
        self.assertGreaterEqual(len(data["baselines"]), 1)

    def test_repo_golden_v1_draft_validates_as_draft(self):
        data = validate_golden_manifest(ROOT / "eval/golden-v1/MANIFEST.json")
        self.assertEqual(data["status"], "draft")
        self.assertGreaterEqual(len(data["splits"]["test"]), 8)
        self.assertTrue(any(item["track"] == "perception-test-acoustic" for item in data["splits"]["test"]))
        self.assertTrue(any(item["track"] == "perception-test-action" for item in data["splits"]["test"]))

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

    def test_perception_test_golden_loader_extracts_events_without_media(self):
        with TemporaryDirectory() as d:
            root = Path(d)
            ann = root / "sample.json"
            ann.write_text(
                json.dumps(
                    {
                        "video_1": {
                            "metadata": {"video_id": "video_1", "frame_rate": 30.0},
                            "action_localisation": [
                                {"label": "Opening something", "timestamps": [1_000_000, 2_000_000]}
                            ],
                            "sound_localisation": [
                                {
                                    "label": "Interaction:hitting",
                                    "timestamps": [3_000_000, 4_000_000],
                                    "is_visible": 0,
                                }
                            ],
                        }
                    }
                )
            )
            digest = hashlib.sha256(ann.read_bytes()).hexdigest()
            manifest = root / "MANIFEST.json"
            manifest.write_text(
                json.dumps(
                    {
                        "version": "golden-test",
                        "status": "draft",
                        "tracks": [
                            {"id": "perception-test-acoustic", "source": "pt", "status": "candidate", "metrics": ["recall"]}
                        ],
                        "splits": {
                            "dev": [],
                            "test": [
                                {
                                    "id": "pt-video_1",
                                    "source": "pt sample",
                                    "license": "CC-BY-4.0",
                                    "track": "perception-test-acoustic",
                                    "task": "sound_localisation",
                                    "video_id": "video_1",
                                    "media_path": str(root / "missing.mp4"),
                                    "media_sha256": "not-present",
                                    "annotation_path": str(ann),
                                    "annotation_sha256": digest,
                                    "action_segments": 1,
                                    "sound_segments": 1,
                                }
                            ],
                        },
                    }
                )
            )
            rows = pt.verified_rows("perception-test-acoustic", manifest=manifest, require_media=False)
            self.assertEqual(len(rows), 1)
            self.assertEqual(pt.golden_rows("perception-test-acoustic", manifest=manifest)[0]["video_id"], "video_1")


if __name__ == "__main__":
    unittest.main()
