"""Tests for the A16-facing GroundTruth-by-construction helper (card A6 scaffolding).

Run directly: `python3 eval/viability/test_ground_truth.py`
Or via pytest pointed at this file/dir: `pytest eval/viability/`
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

# Make the top-level `eval/viability` module importable without requiring a package install —
# this directory is not (yet) a Python package installed via any pyproject.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ground_truth import CORE_ARCHETYPE_ELIGIBLE_AFTER_S, ground_truth_for_archetype  # noqa: E402


class TestGroundTruthForArchetype(unittest.TestCase):
    def test_covers_all_six_core_archetypes(self):
        expected = {
            "state",
            "transition",
            "brief_action",
            "persistence",
            "absence",
            "repeated_episode",
        }
        self.assertEqual(set(CORE_ARCHETYPE_ELIGIBLE_AFTER_S), expected)

    def test_known_archetype_is_deterministic(self):
        a = ground_truth_for_archetype("transition", event_start_ts=10.0, fixture_id="fx-1")
        b = ground_truth_for_archetype("transition", event_start_ts=10.0, fixture_id="fx-1")
        self.assertEqual(a, b)
        self.assertEqual(a.decision_eligible_ts, 10.5)

    def test_explicit_offset_overrides_default(self):
        gt = ground_truth_for_archetype(
            "persistence", event_start_ts=0.0, fixture_id="fx-2", eligible_after_s=5.0
        )
        self.assertEqual(gt.decision_eligible_ts, 5.0)

    def test_unknown_archetype_without_explicit_offset_raises(self):
        with self.assertRaises(KeyError):
            ground_truth_for_archetype("not_a_real_archetype", event_start_ts=0.0, fixture_id="fx-3")


if __name__ == "__main__":
    unittest.main()
