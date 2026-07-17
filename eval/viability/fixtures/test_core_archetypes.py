"""Tests for the A16 neutral archetype golden fixtures (CORE gating set only).

Run directly: `python3 eval/viability/fixtures/test_core_archetypes.py`
Or via pytest pointed at this file/dir: `pytest eval/viability/fixtures/`
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

# Same "not (yet) an installed package" pattern as `eval/viability/test_ground_truth.py` — make
# this file's own directory importable so `from core_archetypes import ...` works regardless of how
# pytest/unittest discovers/imports this file.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from core_archetypes import (  # noqa: E402
    CORE_ARCHETYPES,
    EXCLUDED_SUPPLEMENTAL_ARCHETYPES,
    build_core_archetype_fixtures,
)

_EXPECTED_CORE_ARCHETYPES = {
    "state",
    "transition",
    "brief_action",
    "persistence",
    "absence",
    "repeated_episode",
}


class TestCoreArchetypeCoverage(unittest.TestCase):
    def test_all_six_core_archetypes_present(self):
        fixtures = build_core_archetype_fixtures()
        archetypes = {f.archetype for f in fixtures}
        self.assertEqual(archetypes, _EXPECTED_CORE_ARCHETYPES)

    def test_core_archetypes_constant_matches_expected_six(self):
        self.assertEqual(set(CORE_ARCHETYPES), _EXPECTED_CORE_ARCHETYPES)

    def test_supplemental_archetypes_are_excluded(self):
        fixtures = build_core_archetype_fixtures()
        archetypes = {f.archetype for f in fixtures}
        # The two supplemental (non-gating, audio/specialist) archetypes must never appear.
        self.assertFalse(archetypes & EXCLUDED_SUPPLEMENTAL_ARCHETYPES)
        self.assertEqual(
            EXCLUDED_SUPPLEMENTAL_ARCHETYPES,
            {"audio_visual_indicator_alarm", "specialist_verification"},
        )

    def test_every_archetype_has_a_positive_and_a_negative_variant(self):
        fixtures = build_core_archetype_fixtures()
        for archetype in _EXPECTED_CORE_ARCHETYPES:
            variants = [f for f in fixtures if f.archetype == archetype]
            self.assertTrue(
                any(f.positive for f in variants), f"{archetype} missing a positive fixture"
            )
            self.assertTrue(
                any(not f.positive for f in variants), f"{archetype} missing a negative fixture"
            )

    def test_fixture_ids_are_unique(self):
        fixtures = build_core_archetype_fixtures()
        ids = [f.fixture_id for f in fixtures]
        self.assertEqual(len(ids), len(set(ids)))


class TestGroundTruthByConstruction(unittest.TestCase):
    def test_positive_fixtures_carry_ground_truth_negative_do_not(self):
        for f in build_core_archetype_fixtures():
            if f.positive:
                self.assertIsNotNone(f.ground_truth, f"{f.fixture_id} should carry a GroundTruth")
            else:
                self.assertIsNone(f.ground_truth, f"{f.fixture_id} should NOT carry a GroundTruth")

    def test_decision_eligible_ts_is_set_by_construction_and_not_before_event_start(self):
        for f in build_core_archetype_fixtures():
            if f.ground_truth is None:
                continue
            self.assertGreaterEqual(f.ground_truth.decision_eligible_ts, f.ground_truth.event_start_ts)
            # by-construction records always carry a non-empty provenance/version stamp.
            self.assertTrue(f.ground_truth.annotation_version)
            self.assertEqual(f.ground_truth.annotator_or_fixture_id, f.fixture_id)

    def test_ground_truth_offsets_match_documented_archetype_defaults_or_explicit_override(self):
        expected_offset = {
            "state": 0.1,
            "transition": 0.5,
            "brief_action": 0.75,
            "absence": 2.0,
            "repeated_episode": 0.5,
        }
        for f in build_core_archetype_fixtures():
            if f.ground_truth is None or f.archetype not in expected_offset:
                continue
            actual_offset = f.ground_truth.decision_eligible_ts - f.ground_truth.event_start_ts
            self.assertAlmostEqual(actual_offset, expected_offset[f.archetype], places=9)

    def test_persistence_ground_truth_uses_the_fixture_supplied_hold_length(self):
        (persistence_positive,) = [
            f
            for f in build_core_archetype_fixtures()
            if f.archetype == "persistence" and f.positive
        ]
        gt = persistence_positive.ground_truth
        assert gt is not None
        # The persistence default in ground_truth.py (1.0s) is documented as a floor, not the real
        # value — this fixture supplies its actual 3.0s hold length explicitly.
        self.assertEqual(gt.decision_eligible_ts - gt.event_start_ts, 3.0)


class TestDeterminism(unittest.TestCase):
    def test_same_build_call_twice_yields_identical_fixtures(self):
        a = build_core_archetype_fixtures()
        b = build_core_archetype_fixtures()
        self.assertEqual(a, b)

    def test_timelines_are_frame_and_timestamp_identical_across_builds(self):
        a = {f.fixture_id: f.timeline for f in build_core_archetype_fixtures()}
        b = {f.fixture_id: f.timeline for f in build_core_archetype_fixtures()}
        self.assertEqual(set(a), set(b))
        for fixture_id in a:
            self.assertEqual(a[fixture_id], b[fixture_id], f"timeline drifted for {fixture_id}")

    def test_timelines_are_nonempty_and_monotonically_scheduled(self):
        for f in build_core_archetype_fixtures():
            self.assertGreater(len(f.timeline), 0)
            ts = [tf.t for tf in f.timeline]
            self.assertEqual(ts, sorted(ts))


if __name__ == "__main__":
    unittest.main()
