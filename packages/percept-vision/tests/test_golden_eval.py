"""Card A14: probe-harness ``golden_eval`` (Viability §5a event-level matching rule).

Proves that :mod:`percept.harness.golden_eval`:

  * matches a predicted event to AT MOST ONE ground-truth episode, requiring
    ``decision_emitted_ts in [decision_eligible_ts, episode_end_ts + allowed_lateness]``.
  * scores a fire that lands before ``decision_eligible_ts`` or after
    ``episode_end_ts + allowed_lateness`` as unmatched (a miss for the episode, a false fire for
    the prediction).
  * scores a second prediction landing in an already-matched episode's window as a DUPLICATE false
    fire, not a second match.
  * scores an unmatched ground-truth episode as a miss.
  * computes event-level recall/precision arithmetic correctly on a small labeled set.
  * emits a deterministic golden timeline, independent of input list order.
"""

from __future__ import annotations

import unittest

from percept.contracts import GroundTruth, JudgmentEvent, JudgmentTiming
from percept.harness.golden_eval import (
    GoldenEvalResult,
    GroundTruthEpisode,
    MatchOutcome,
    build_golden_timeline,
    score,
)


def _gt(event_start_ts: float, decision_eligible_ts: float, fixture_id: str = "fx") -> GroundTruth:
    return GroundTruth(
        event_start_ts=event_start_ts,
        decision_eligible_ts=decision_eligible_ts,
        annotation_version="eligibility-rubric-v1",
        annotator_or_fixture_id=fixture_id,
    )


def _episode(
    episode_id: str,
    *,
    event_start_ts: float = 0.0,
    decision_eligible_ts: float = 1.0,
    episode_end_ts: float = 5.0,
    allowed_lateness_s: float = 0.0,
) -> GroundTruthEpisode:
    return GroundTruthEpisode(
        episode_id=episode_id,
        ground_truth=_gt(event_start_ts, decision_eligible_ts, fixture_id=episode_id),
        episode_end_ts=episode_end_ts,
        allowed_lateness_s=allowed_lateness_s,
    )


def _fire(key: str, decision_emitted_ts: float, evidence_seq: int = 0) -> JudgmentEvent:
    timing = JudgmentTiming(
        ingest_observed_ts=decision_emitted_ts,
        admitted_ts=decision_emitted_ts,
        judge_started_ts=decision_emitted_ts,
        judge_completed_ts=decision_emitted_ts,
        decision_emitted_ts=decision_emitted_ts,
    )
    return JudgmentEvent(key=key, evidence_seq=evidence_seq, action="fire", timing=timing)


class TestGroundTruthEpisodeValidation(unittest.TestCase):
    def test_window_is_eligible_to_end_plus_lateness(self):
        ep = _episode("e1", decision_eligible_ts=2.0, episode_end_ts=5.0, allowed_lateness_s=1.5)
        self.assertEqual(ep.window, (2.0, 6.5))

    def test_empty_episode_id_rejected(self):
        with self.assertRaises(ValueError):
            _episode("")

    def test_end_before_eligible_rejected(self):
        with self.assertRaises(ValueError):
            GroundTruthEpisode(
                episode_id="e1",
                ground_truth=_gt(0.0, 5.0),
                episode_end_ts=4.0,
            )

    def test_negative_lateness_rejected(self):
        with self.assertRaises(ValueError):
            _episode("e1", allowed_lateness_s=-0.1)


class TestExactMatchInsideWindow(unittest.TestCase):
    def test_fire_inside_window_matches(self):
        ep = _episode("e1", decision_eligible_ts=1.0, episode_end_ts=5.0, allowed_lateness_s=0.5)
        pred = _fire("e1", 5.3)  # inside [1.0, 5.5]
        result = score([ep], [pred])
        self.assertEqual(result.matched_episode_ids, ("e1",))
        self.assertEqual(result.missed_episode_ids, ())
        self.assertEqual(len(result.prediction_scores), 1)
        self.assertEqual(result.prediction_scores[0].outcome, MatchOutcome.MATCH)
        self.assertEqual(result.prediction_scores[0].matched_episode_id, "e1")
        self.assertEqual(result.recall, 1.0)
        self.assertEqual(result.precision, 1.0)

    def test_fire_at_exact_boundaries_matches(self):
        # Closed interval: exactly decision_eligible_ts and exactly episode_end_ts+lateness match.
        ep = _episode("e1", decision_eligible_ts=1.0, episode_end_ts=5.0, allowed_lateness_s=0.0)
        lo_result = score([ep], [_fire("e1", 1.0)])
        self.assertEqual(lo_result.prediction_scores[0].outcome, MatchOutcome.MATCH)
        hi_result = score([ep], [_fire("e1", 5.0)])
        self.assertEqual(hi_result.prediction_scores[0].outcome, MatchOutcome.MATCH)


class TestMissWhenOutsideWindow(unittest.TestCase):
    def test_fire_before_eligible_is_a_miss_and_false_fire(self):
        ep = _episode("e1", decision_eligible_ts=2.0, episode_end_ts=5.0)
        pred = _fire("e1", 1.999)  # just before decision_eligible_ts
        result = score([ep], [pred])
        self.assertEqual(result.matched_episode_ids, ())
        self.assertEqual(result.missed_episode_ids, ("e1",))
        self.assertEqual(result.prediction_scores[0].outcome, MatchOutcome.FALSE_FIRE)
        self.assertEqual(result.recall, 0.0)
        self.assertEqual(result.precision, 0.0)

    def test_fire_after_end_plus_lateness_is_a_miss_and_false_fire(self):
        ep = _episode("e1", decision_eligible_ts=1.0, episode_end_ts=5.0, allowed_lateness_s=1.0)
        pred = _fire("e1", 6.001)  # just past episode_end_ts + allowed_lateness_s = 6.0
        result = score([ep], [pred])
        self.assertEqual(result.missed_episode_ids, ("e1",))
        self.assertEqual(result.prediction_scores[0].outcome, MatchOutcome.FALSE_FIRE)


class TestDuplicateFalseFire(unittest.TestCase):
    def test_second_prediction_in_matched_episode_is_duplicate(self):
        ep = _episode("e1", decision_eligible_ts=1.0, episode_end_ts=5.0)
        first = _fire("e1", 2.0, evidence_seq=0)
        second = _fire("e1", 3.0, evidence_seq=1)  # same episode window, after the first match
        result = score([ep], [first, second])
        self.assertEqual(result.matched_episode_ids, ("e1",))
        self.assertEqual(len(result.prediction_scores), 2)
        outcomes = {p.decision_emitted_ts: p.outcome for p in result.prediction_scores}
        self.assertEqual(outcomes[2.0], MatchOutcome.MATCH)
        self.assertEqual(outcomes[3.0], MatchOutcome.DUPLICATE)
        # duplicate is tagged against the episode it duplicated.
        dup = next(p for p in result.prediction_scores if p.outcome is MatchOutcome.DUPLICATE)
        self.assertEqual(dup.matched_episode_id, "e1")
        self.assertEqual(result.duplicate_false_fires, 1)
        self.assertEqual(result.unmatched_false_fires, 0)
        self.assertEqual(result.false_fires, 1)
        # event-level precision counts the duplicate as a false positive: 1 match / 2 predictions.
        self.assertEqual(result.precision, 0.5)
        self.assertEqual(result.recall, 1.0)

    def test_earliest_prediction_wins_the_match_regardless_of_input_order(self):
        ep = _episode("e1", decision_eligible_ts=1.0, episode_end_ts=5.0)
        first = _fire("e1", 2.0, evidence_seq=0)
        second = _fire("e1", 3.0, evidence_seq=1)
        # Pass out of chronological order — the matcher must still sort by decision_emitted_ts.
        result = score([ep], [second, first])
        outcomes = {p.decision_emitted_ts: p.outcome for p in result.prediction_scores}
        self.assertEqual(outcomes[2.0], MatchOutcome.MATCH)
        self.assertEqual(outcomes[3.0], MatchOutcome.DUPLICATE)


class TestUnmatchedGroundTruthIsMiss(unittest.TestCase):
    def test_episode_with_no_prediction_is_a_miss(self):
        ep1 = _episode("e1", decision_eligible_ts=1.0, episode_end_ts=5.0)
        ep2 = _episode("e2", decision_eligible_ts=10.0, episode_end_ts=15.0)
        pred = _fire("e1", 2.0)
        result = score([ep1, ep2], [pred])
        self.assertEqual(result.matched_episode_ids, ("e1",))
        self.assertEqual(result.missed_episode_ids, ("e2",))
        self.assertEqual(result.misses, 1)


class TestRecallPrecisionArithmetic(unittest.TestCase):
    def test_small_labeled_set(self):
        # 3 episodes: e1 correctly matched, e2 missed entirely, e3 matched + 1 duplicate.
        # Plus 1 stray false fire that matches no episode window at all.
        ep1 = _episode("e1", decision_eligible_ts=1.0, episode_end_ts=3.0)
        ep2 = _episode("e2", decision_eligible_ts=10.0, episode_end_ts=12.0)
        ep3 = _episode("e3", decision_eligible_ts=20.0, episode_end_ts=22.0)

        preds = [
            _fire("e1", 1.5, evidence_seq=0),   # matches e1
            _fire("e3", 20.5, evidence_seq=1),  # matches e3
            _fire("e3", 21.0, evidence_seq=2),  # duplicate of e3
            _fire("stray", 100.0, evidence_seq=3),  # false fire, no episode window
        ]
        result = score([ep1, ep2, ep3], preds)

        self.assertEqual(set(result.matched_episode_ids), {"e1", "e3"})
        self.assertEqual(result.missed_episode_ids, ("e2",))
        self.assertEqual(result.true_positives, 2)
        self.assertEqual(result.duplicate_false_fires, 1)
        self.assertEqual(result.unmatched_false_fires, 1)
        self.assertEqual(result.false_fires, 2)
        self.assertEqual(result.misses, 1)

        # recall = TP / total episodes = 2/3
        self.assertAlmostEqual(result.recall, 2 / 3)
        # precision = TP / total scored predictions = 2/4
        self.assertAlmostEqual(result.precision, 0.5)

    def test_no_episodes_recall_is_vacuously_one(self):
        result = score([], [_fire("x", 1.0)])
        self.assertEqual(result.recall, 1.0)
        self.assertEqual(result.precision, 0.0)

    def test_no_predictions_precision_is_vacuously_one(self):
        ep = _episode("e1")
        result = score([ep], [])
        self.assertEqual(result.precision, 1.0)
        self.assertEqual(result.recall, 0.0)

    def test_only_predicted_actions_are_scored(self):
        ep = _episode("e1", decision_eligible_ts=1.0, episode_end_ts=5.0)
        silent = JudgmentEvent(
            key="e1",
            evidence_seq=0,
            action="silent",
            timing=JudgmentTiming(
                ingest_observed_ts=2.0,
                admitted_ts=2.0,
                judge_started_ts=2.0,
                judge_completed_ts=2.0,
                decision_emitted_ts=2.0,
            ),
        )
        result = score([ep], [silent])
        self.assertEqual(result.prediction_scores, ())
        self.assertEqual(result.missed_episode_ids, ("e1",))


class TestDuplicateEpisodeIdsRejected(unittest.TestCase):
    def test_score_rejects_duplicate_episode_ids(self):
        ep = _episode("e1")
        with self.assertRaises(ValueError):
            score([ep, ep], [])

    def test_build_golden_timeline_rejects_duplicate_episode_ids(self):
        ep = _episode("e1")
        with self.assertRaises(ValueError):
            build_golden_timeline([ep, ep])


class TestGoldenTimelineDeterminism(unittest.TestCase):
    def test_timeline_is_sorted_and_sequential(self):
        ep1 = _episode("e1", decision_eligible_ts=10.0, episode_end_ts=12.0)
        ep2 = _episode("e2", decision_eligible_ts=1.0, episode_end_ts=3.0)
        ep3 = _episode("e3", decision_eligible_ts=5.0, episode_end_ts=7.0)

        # Pass in an arbitrary, non-chronological order.
        timeline = build_golden_timeline([ep1, ep2, ep3])

        self.assertEqual([e.key for e in timeline], ["e2", "e3", "e1"])
        self.assertEqual([e.evidence_seq for e in timeline], [0, 1, 2])
        self.assertTrue(all(e.action == "fire" for e in timeline))
        self.assertEqual(
            [e.timing.decision_emitted_ts for e in timeline], [1.0, 5.0, 10.0]
        )

    def test_timeline_is_deterministic_across_calls_and_input_order(self):
        ep1 = _episode("e1", decision_eligible_ts=10.0, episode_end_ts=12.0)
        ep2 = _episode("e2", decision_eligible_ts=1.0, episode_end_ts=3.0)

        a = build_golden_timeline([ep1, ep2])
        b = build_golden_timeline([ep2, ep1])  # reversed input order
        self.assertEqual(a, b)

    def test_golden_event_matches_its_own_episode_when_scored(self):
        # Sanity: the golden/ideal event for an episode should score as a MATCH against that
        # episode's own window (it fires exactly at decision_eligible_ts).
        ep = _episode("e1", decision_eligible_ts=1.0, episode_end_ts=5.0)
        timeline = build_golden_timeline([ep])
        result = score([ep], list(timeline))
        self.assertEqual(result.matched_episode_ids, ("e1",))
        self.assertEqual(result.recall, 1.0)
        self.assertEqual(result.precision, 1.0)


if __name__ == "__main__":
    unittest.main()
