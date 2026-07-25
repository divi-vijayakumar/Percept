"""B3 — reliability/ECE metric core, L1 on synthetic pairs (no golden set, no models).

Pins the math AND the diagnostic: the overconfident-constant case (all confidence ~0.95, accuracy
< 1) must report ECE = the overconfidence gap, AUROC ≈ 0.5, and discriminative=False — i.e. it
reproduces "confidence does not discriminate; the threshold is the wrong lever" from data, not anecdote.
"""

import unittest

from eval.reliability import (
    auroc,
    below_threshold,
    discrimination,
    ece,
    reliability_report,
    silent_frame_pairs,
)


class TestECE(unittest.TestCase):
    def test_perfectly_calibrated_is_zero(self):
        pairs = [(0.9, True)] * 9 + [(0.9, False)] * 1  # bin conf 0.9, acc 0.9 -> |0.9-0.9|=0
        self.assertAlmostEqual(ece(pairs), 0.0)

    def test_overconfident_constant_equals_the_gap(self):
        pairs = [(0.95, True)] * 7 + [(0.95, False)] * 3  # conf 0.95, acc 0.70
        self.assertAlmostEqual(ece(pairs), 0.25)  # |0.70 - 0.95|

    def test_empty_is_zero(self):
        self.assertEqual(ece([]), 0.0)


class TestAUROC(unittest.TestCase):
    def test_separable_is_one(self):
        pairs = [(0.9, True), (0.85, True), (0.3, False), (0.2, False)]
        self.assertAlmostEqual(auroc(pairs), 1.0)

    def test_constant_confidence_is_half(self):
        pairs = [(0.95, True)] * 7 + [(0.95, False)] * 3  # all ties
        self.assertAlmostEqual(auroc(pairs), 0.5)

    def test_undefined_without_both_classes(self):
        self.assertIsNone(auroc([(0.9, True), (0.8, True)]))


class TestDiscrimination(unittest.TestCase):
    def test_constant_confidence_has_zero_gap(self):
        d = discrimination([(0.95, True)] * 7 + [(0.95, False)] * 3)
        self.assertAlmostEqual(d["gap"], 0.0)
        self.assertAlmostEqual(d["mean_conf_right"], 0.95)
        self.assertAlmostEqual(d["mean_conf_wrong"], 0.95)


class TestBelowThreshold(unittest.TestCase):
    def test_counts_confidences_below_default(self):
        cases = [{"verdicts": [[True, 0.95], [False, 0.6], [True, 0.69]]}]
        self.assertEqual(below_threshold(cases, 0.7), (2, 3))  # 0.6 and 0.69 are below


class TestSilentFramePairs(unittest.TestCase):
    def test_only_silent_scenes_truth_is_unsatisfied(self):
        cases = [
            {"expected": "silent", "verdicts": [[False, 0.9], [True, 0.95]]},  # correct, then false-fire
            {"expected": "fire", "verdicts": [[True, 0.9]]},  # excluded (no crisp per-frame truth)
            {"expected": "ask", "verdicts": [[True, 0.4]]},  # excluded
        ]
        pairs = silent_frame_pairs(cases)
        self.assertEqual(pairs, [(0.9, True), (0.95, False)])  # was_correct = not satisfied


class TestReport(unittest.TestCase):
    def test_overconfident_capture_flags_lever_dead(self):
        # A silent scene the model gets right at 0.95 and wrong at 0.95 -> no discrimination.
        cases = [
            {"expected": "silent", "verdicts": [[False, 0.95]] * 7 + [[True, 0.95]] * 3},
            {"expected": "fire", "verdicts": [[True, 0.9]]},
        ]
        r = reliability_report(cases)
        self.assertEqual(r["n_reliability_frames"], 10)  # only the silent-scene frames
        self.assertAlmostEqual(r["ece"], 0.25)
        self.assertAlmostEqual(r["discrimination"]["auroc"], 0.5)
        self.assertFalse(r["discriminative"])
        self.assertIn("wrong lever", r["verdict"])

    def test_discriminative_capture_greenlights_threshold(self):
        cases = [{"expected": "silent", "verdicts": [[False, 0.9]] * 5 + [[True, 0.2]] * 5}]
        r = reliability_report(cases)
        self.assertAlmostEqual(r["discrimination"]["auroc"], 1.0)
        self.assertTrue(r["discriminative"])


if __name__ == "__main__":
    unittest.main()
