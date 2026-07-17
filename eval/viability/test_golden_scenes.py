"""Tests for eval/viability/golden_scenes.py (card A32, Viability §5a SECONDARY real-clip set).

Run directly: `python3 eval/viability/test_golden_scenes.py`
Or via pytest pointed at this file/dir: `pytest eval/viability/test_golden_scenes.py`
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

# Make the top-level `eval/viability` module importable without requiring a package install --
# mirrors eval/viability/test_ground_truth.py's convention.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from golden_scenes import (  # noqa: E402
    DEFAULT_LABELS_PATH,
    SECONDARY_SUBSTRATE_CAVEAT,
    RealClipEpisode,
    episodes_for_scene,
    load_real_clip_episodes,
    load_scene_records,
    score_real_clip_predictions,
)
from percept.clock import FakeClock  # noqa: E402
from percept.contracts import JudgmentEvent, JudgmentTiming  # noqa: E402
from percept.harness.golden_eval import MatchOutcome  # noqa: E402

TOTAL_SCENES = 48
TOTAL_RISING_EDGES = 72
APPROVED_SCENES = 17
APPROVED_SCENES_WITH_RISING_EDGES = 8
APPROVED_AND_PINNED_RISING_EDGES = 6


def _fire(decision_emitted_ts: float, key: str = "pred") -> JudgmentEvent:
    timing = JudgmentTiming(
        ingest_observed_ts=decision_emitted_ts,
        admitted_ts=decision_emitted_ts,
        judge_started_ts=decision_emitted_ts,
        judge_completed_ts=decision_emitted_ts,
        decision_emitted_ts=decision_emitted_ts,
    )
    return JudgmentEvent(key=key, evidence_seq=0, action="fire", timing=timing)


class TestLoadScenRecords(unittest.TestCase):
    def test_parses_all_48_scenes(self):
        records = load_scene_records()
        self.assertEqual(len(records), TOTAL_SCENES)
        self.assertEqual(len({r["scene_id"] for r in records}), TOTAL_SCENES)

    def test_default_labels_path_exists(self):
        self.assertTrue(DEFAULT_LABELS_PATH.is_file())


class TestLoadRealClipEpisodes(unittest.TestCase):
    def test_unfiltered_load_covers_every_rising_edge(self):
        episodes = load_real_clip_episodes(approved_only=False, pinned_only=False)
        self.assertEqual(len(episodes), TOTAL_RISING_EDGES)
        self.assertTrue(all(isinstance(ep, RealClipEpisode) for ep in episodes))

    def test_approved_only_filter(self):
        episodes = load_real_clip_episodes(approved_only=True, pinned_only=False)
        self.assertEqual(len(episodes), APPROVED_SCENES_WITH_RISING_EDGES)
        self.assertTrue(all(ep.review_status == "approved" for ep in episodes))

    def test_default_is_approved_and_pinned(self):
        episodes = load_real_clip_episodes()
        self.assertEqual(len(episodes), APPROVED_AND_PINNED_RISING_EDGES)
        self.assertTrue(all(ep.review_status == "approved" for ep in episodes))
        self.assertTrue(all(ep.verify == "pinned" for ep in episodes))

    def test_rising_edge_t_becomes_decision_eligible_ts(self):
        episodes = load_real_clip_episodes()
        by_scene = {ep.scene_id: ep for ep in episodes}
        chase_bank = by_scene["eval1_s02"]
        self.assertAlmostEqual(
            chase_bank.episode.ground_truth.decision_eligible_ts, 4.637
        )
        self.assertEqual(chase_bank.split, "dev")
        self.assertTrue(chase_bank.matched_event)
        self.assertFalse(chase_bank.label_inconsistent)

    def test_matched_event_supplies_episode_end_ts(self):
        episodes = load_real_clip_episodes()
        chase_bank = next(ep for ep in episodes if ep.scene_id == "eval1_s02")
        # events[].t_end for the matching "Chase bank" event is 10.0.
        self.assertAlmostEqual(chase_bank.episode.episode_end_ts, 10.0)

    def test_label_inconsistency_is_clamped_not_raised(self):
        # eval2_s01's pinned rising edge (t=130.0) lands after its own matched event's t_end
        # (94.0) -- a real single-reviewer label inconsistency. The loader must not raise, and
        # must flag it via label_inconsistent rather than silently hiding it.
        episodes = load_real_clip_episodes()
        cooking = next(ep for ep in episodes if ep.scene_id == "eval2_s01")
        self.assertTrue(cooking.label_inconsistent)
        # Both frozen-contract invariants still hold despite the inconsistency.
        self.assertGreaterEqual(
            cooking.episode.ground_truth.decision_eligible_ts,
            cooking.episode.ground_truth.event_start_ts,
        )
        self.assertGreaterEqual(
            cooking.episode.episode_end_ts,
            cooking.episode.ground_truth.decision_eligible_ts,
        )

    def test_unmatched_condition_falls_back_to_zero_width_window(self):
        # eval3_s27's "mail truck leaves the scene" rising edge has no events[] record sharing its
        # exact condition text -- widen the filters to reach it (it is not approved/pinned).
        episodes = load_real_clip_episodes(approved_only=False, pinned_only=False)
        leaves = next(
            ep
            for ep in episodes
            if ep.scene_id == "eval3_s27" and ep.condition == "The mail truck leaves the scene."
        )
        self.assertFalse(leaves.matched_event)
        self.assertEqual(
            leaves.episode.ground_truth.event_start_ts, leaves.episode.episode_end_ts
        )

    def test_episodes_for_scene_filters_and_preserves_order(self):
        episodes = load_real_clip_episodes(approved_only=False, pinned_only=False)
        subset = episodes_for_scene(episodes, "eval3_s27")
        self.assertEqual([ep.scene_id for ep in subset], ["eval3_s27"] * len(subset))
        self.assertGreaterEqual(len(subset), 3)

    def test_episode_ids_unique_within_one_load(self):
        episodes = load_real_clip_episodes(approved_only=False, pinned_only=False)
        ids = [ep.episode.episode_id for ep in episodes]
        self.assertEqual(len(ids), len(set(ids)))


class TestScoreRealClipPredictions(unittest.TestCase):
    """Scores a synthetic predicted-event set against the known eval1_s02 scene (single, clean
    approved+pinned episode: decision_eligible_ts=4.637, episode_end_ts=10.0,
    allowed_lateness_s=1.0 default -> window (4.637, 11.0))."""

    def setUp(self):
        episodes = load_real_clip_episodes()
        self.chase_bank = next(ep for ep in episodes if ep.scene_id == "eval1_s02")

    def test_prediction_inside_window_is_a_match(self):
        clock = FakeClock(start_s=5.0)  # inside (4.637, 11.0)
        result = score_real_clip_predictions([self.chase_bank], [_fire(clock())])
        self.assertEqual(result.recall, 1.0)
        self.assertEqual(result.precision, 1.0)
        self.assertEqual(result.true_positives, 1)
        self.assertEqual(result.misses, 0)
        self.assertEqual(result.false_fires, 0)
        self.assertEqual(result.prediction_scores[0].outcome, MatchOutcome.MATCH)

    def test_prediction_before_eligibility_is_a_false_fire_and_a_miss(self):
        clock = FakeClock(start_s=1.0)  # before decision_eligible_ts=4.637
        result = score_real_clip_predictions([self.chase_bank], [_fire(clock())])
        self.assertEqual(result.recall, 0.0)
        self.assertEqual(result.precision, 0.0)
        self.assertEqual(result.misses, 1)
        self.assertEqual(result.false_fires, 1)
        self.assertEqual(result.prediction_scores[0].outcome, MatchOutcome.FALSE_FIRE)

    def test_no_predictions_is_a_miss_with_vacuous_precision(self):
        result = score_real_clip_predictions([self.chase_bank], [])
        self.assertEqual(result.recall, 0.0)
        self.assertEqual(result.precision, 1.0)  # vacuous: no predictions scored at all
        self.assertEqual(result.misses, 1)

    def test_duplicate_prediction_in_same_episode_window(self):
        first = _fire(5.0, key="pred-a")
        duplicate = _fire(6.0, key="pred-b")
        result = score_real_clip_predictions([self.chase_bank], [first, duplicate])
        self.assertEqual(result.recall, 1.0)
        self.assertEqual(result.true_positives, 1)
        self.assertEqual(result.duplicate_false_fires, 1)
        self.assertAlmostEqual(result.precision, 0.5)


class TestSecondarySubstrateCaveat(unittest.TestCase):
    """The §5a caveat -- single-reviewer real-clip labels are a SECONDARY validation substrate,
    never the primary gate substrate -- must be documented both in prose and as data callers can
    assert on programmatically."""

    def test_caveat_constant_names_the_two_annotator_gap(self):
        self.assertIn("SECONDARY", SECONDARY_SUBSTRATE_CAVEAT)
        self.assertIn("TWO independent annotators", SECONDARY_SUBSTRATE_CAVEAT)
        self.assertIn("core PRE-MVP VIABILITY gating decision", SECONDARY_SUBSTRATE_CAVEAT)

    def test_module_docstring_documents_the_caveat(self):
        import golden_scenes

        doc = golden_scenes.__doc__ or ""
        self.assertIn("SECONDARY VALIDATION SUBSTRATE", doc)
        self.assertIn("PRIMARY", doc)

    def test_single_reviewer_provenance_is_structural_not_just_documented(self):
        # Every loaded GroundTruth's provenance id is prefixed "single-reviewer:" -- distinct,
        # by construction, from GroundTruth.from_two_annotator's "a+b" id shape -- so a caller
        # cannot mistake this set for a two-annotator-adjudicated one just by inspecting a record.
        episodes = load_real_clip_episodes()
        self.assertTrue(episodes, "expected at least one approved+pinned episode to check")
        for ep in episodes:
            self.assertTrue(
                ep.episode.ground_truth.annotator_or_fixture_id.startswith("single-reviewer:")
            )

    @unittest.skip(
        "Two-annotator + adjudication real-clip ground truth (Viability §5a secondary path, "
        "GroundTruth.from_two_annotator) is NOT implemented by this loader -- these 48 scenes "
        "carry only a single reviewer's approved/pinned label, with no second annotator and no "
        "reported inter-annotator agreement. A future for-record real-VLM validation run wanting "
        "full §5a rigor for real clips needs a second annotation pass over labels.jsonl (or an "
        "equivalent) before that gap can close; that is out of scope for this card (A32), which "
        "only builds the loader/scoring wiring for the SECONDARY substrate as-labeled today."
    )
    def test_two_annotator_adjudication_not_available_for_these_scenes(self):
        raise NotImplementedError


if __name__ == "__main__":
    unittest.main()
