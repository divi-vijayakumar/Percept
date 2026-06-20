"""L1: golden loader + split discipline (disjoint splits; deterministic load)."""

import json
import tempfile
import unittest
from pathlib import Path

from percept import eval as golden


def _write_fixture(root: Path):
    (root / "splits.json").write_text(
        json.dumps({"train": ["s1", "s2"], "dev": ["s3"], "test": ["s4"]})
    )
    (root / "labels.jsonl").write_text(
        '{"scene_id": "s1", "entities": []}\n\n{"scene_id": "s2", "entities": []}\n'
    )
    (root / "gate_cases.jsonl").write_text(
        '{"scene_id": "s1", "expected_verdict": "fire"}\n'
    )
    (root / "utterances.jsonl").write_text("")


class TestGoldenLoader(unittest.TestCase):
    def test_loads_and_indexes(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            _write_fixture(root)
            gs = golden.load_golden(root)
            self.assertEqual(gs.scenes("train"), ["s1", "s2"])
            self.assertEqual(gs.split_of("s3"), "dev")
            self.assertIsNone(gs.split_of("unknown"))
            self.assertEqual(len(gs.labels), 2)  # blank line skipped
            self.assertEqual(len(gs.gate_cases), 1)

    def test_load_is_deterministic(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            _write_fixture(root)
            a = golden.load_golden(root)
            b = golden.load_golden(root)
            self.assertEqual(a.splits, b.splits)
            self.assertEqual(a.labels, b.labels)

    def test_overlapping_splits_raise(self):
        with self.assertRaises(golden.SplitError):
            golden.assert_disjoint({"train": ["s1"], "dev": ["s1"], "test": []})

    def test_disjoint_splits_pass(self):
        golden.assert_disjoint({"train": ["s1"], "dev": ["s2"], "test": ["s3"]})

    def test_real_fixture_set_has_disjoint_splits_if_present(self):
        # The shipped private fixtures must keep splits disjoint (test held out).
        root = golden.find_golden_root()
        if root is None:
            self.skipTest("private golden fixtures not present")
        gs = golden.load_golden(root)  # load_splits asserts disjointness internally
        all_scenes = [s for name in golden.SPLIT_NAMES for s in gs.scenes(name)]
        self.assertEqual(len(all_scenes), len(set(all_scenes)))


if __name__ == "__main__":
    unittest.main()
