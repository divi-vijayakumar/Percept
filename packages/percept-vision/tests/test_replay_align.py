"""L1: the replay-and-align scorer on synthetic fire/edge lists — no models (spec §9, §10.2).

Covers alignment (TP / FN / FP), the timing discipline (*a right answer at the wrong time scores
wrong* — early or past-omega fires don't satisfy an edge), count off-by-one, PAUC@omega
(promptness), and the CI gate wiring (false-fire == 0, false-serve <= bound).
"""

import unittest

from eval.replay import Edge, Fire, align, pauc, score
from eval.run import check_gates


class TestAlign(unittest.TestCase):
    def test_single_in_window_fire_is_a_tp(self):
        a = align([Fire("g1", 10.4)], [Edge("g1", 10.0)], omega=1.0)
        self.assertEqual((a.tp, a.fn, a.fp), (1, 0, 0))
        self.assertAlmostEqual(a.latencies[0], 0.4)

    def test_no_fire_is_a_fn(self):
        a = align([], [Edge("g1", 10.0)], omega=1.0)
        self.assertEqual((a.tp, a.fn, a.fp), (0, 1, 0))

    def test_unmatched_fire_is_a_fp(self):
        a = align([Fire("g1", 50.0)], [Edge("g1", 10.0)], omega=1.0)
        self.assertEqual((a.tp, a.fn, a.fp), (0, 1, 1))  # edge missed + a stray false fire

    def test_fire_before_the_edge_does_not_satisfy_it(self):
        # a right answer at the wrong time scores wrong: a fire BEFORE the edge is not a TP.
        a = align([Fire("g1", 9.5)], [Edge("g1", 10.0)], omega=1.0)
        self.assertEqual((a.tp, a.fn, a.fp), (0, 1, 1))

    def test_fire_past_omega_does_not_satisfy_it(self):
        a = align([Fire("g1", 12.0)], [Edge("g1", 10.0)], omega=1.0)  # 2s late, omega 1s
        self.assertEqual((a.tp, a.fn, a.fp), (0, 1, 1))

    def test_boundary_fires_are_inclusive(self):
        # exactly at the edge and exactly at edge+omega both count.
        at_edge = align([Fire("g1", 10.0)], [Edge("g1", 10.0)], omega=1.0)
        at_omega = align([Fire("g1", 11.0)], [Edge("g1", 10.0)], omega=1.0)
        self.assertEqual(at_edge.tp, 1)
        self.assertEqual(at_omega.tp, 1)

    def test_goal_id_must_match(self):
        a = align([Fire("g2", 10.2)], [Edge("g1", 10.0)], omega=1.0)
        self.assertEqual((a.tp, a.fn, a.fp), (0, 1, 1))

    def test_each_edge_consumes_at_most_one_fire(self):
        # two fires in one edge's window -> one TP, one FP (no double-credit).
        a = align([Fire("g1", 10.1), Fire("g1", 10.2)], [Edge("g1", 10.0)], omega=1.0)
        self.assertEqual((a.tp, a.fp), (1, 1))
        self.assertAlmostEqual(a.latencies[0], 0.1)  # the earliest qualifying fire is chosen

    def test_two_edges_two_fires_align_in_order(self):
        fires = [Fire("g1", 10.3), Fire("g1", 20.1)]
        edges = [Edge("g1", 10.0), Edge("g1", 20.0)]
        a = align(fires, edges, omega=1.0)
        self.assertEqual((a.tp, a.fn, a.fp), (2, 0, 0))

    def test_precision_recall(self):
        # 2 edges, 1 hit + 1 stray fire -> precision 1/2, recall 1/2.
        fires = [Fire("g1", 10.2), Fire("g1", 99.0)]
        edges = [Edge("g1", 10.0), Edge("g1", 20.0)]
        a = align(fires, edges, omega=1.0)
        self.assertAlmostEqual(a.precision, 0.5)
        self.assertAlmostEqual(a.recall, 0.5)
        self.assertEqual(a.false_fire, 1)


class TestPAUC(unittest.TestCase):
    def test_no_edges_is_one(self):
        self.assertEqual(pauc([], omega=2.0, n_edges=0), 1.0)

    def test_instant_hit_is_one(self):
        # every edge hit at latency 0 -> the curve is flat at 1.0 -> PAUC 1.0.
        self.assertAlmostEqual(pauc([0.0], omega=2.0, n_edges=1), 1.0)

    def test_late_hit_scores_below_instant(self):
        late = pauc([1.9], omega=2.0, n_edges=1)
        early = pauc([0.1], omega=2.0, n_edges=1)
        self.assertLess(late, early)
        self.assertGreater(late, 0.0)

    def test_miss_drags_pauc_down(self):
        # one of two edges missed (no latency recorded) caps the curve at 0.5.
        self.assertLessEqual(pauc([0.0], omega=2.0, n_edges=2), 0.5 + 1e-9)

    def test_latency_outside_window_does_not_count(self):
        self.assertAlmostEqual(pauc([5.0], omega=2.0, n_edges=1), 0.0)


class TestCountOffByOne(unittest.TestCase):
    def test_exact_count_is_zero(self):
        m = score([Fire("g1", 10.2)], [Edge("g1", 10.0)], omega=1.0)
        self.assertEqual(m.count_off_by_one, 0)

    def test_double_fire_is_off_by_one(self):
        m = score([Fire("g1", 10.1), Fire("g1", 10.2)], [Edge("g1", 10.0)], omega=1.0)
        self.assertEqual(m.count_off_by_one, 1)  # 2 fires vs 1 edge

    def test_missed_edge_is_off_by_one(self):
        m = score([], [Edge("g1", 10.0)], omega=1.0)
        self.assertEqual(m.count_off_by_one, 1)

    def test_per_goal_counts_sum(self):
        fires = [Fire("g1", 1.0), Fire("g1", 2.0), Fire("g2", 1.0)]
        edges = [Edge("g1", 1.0), Edge("g2", 1.0), Edge("g2", 2.0)]
        # g1: |2-1|=1 ; g2: |1-2|=1 -> 2
        m = score(fires, edges, omega=10.0)
        self.assertEqual(m.count_off_by_one, 2)


class TestScoreRollup(unittest.TestCase):
    def test_cost_and_tokens_aggregate(self):
        fires = [Fire("g1", 10.2, cost=0.3, tokens=100), Fire("g1", 99.0, cost=0.1, tokens=50)]
        edges = [Edge("g1", 10.0)]
        m = score(fires, edges, omega=1.0)
        self.assertAlmostEqual(m.total_cost, 0.4)
        self.assertEqual(m.total_tokens, 150)
        self.assertAlmostEqual(m.cost_per_fire, 0.2)
        self.assertEqual(m.false_fire, 1)
        self.assertTrue(0.0 <= m.pauc <= 1.0)

    def test_as_dict_is_json_shaped(self):
        m = score([Fire("g1", 10.2)], [Edge("g1", 10.0)], omega=1.0)
        d = m.as_dict()
        for k in ("precision", "recall", "false_fire", "pauc", "count_off_by_one"):
            self.assertIn(k, d)


class TestGates(unittest.TestCase):
    def test_clean_scorecard_passes(self):
        m = score([Fire("g1", 10.2)], [Edge("g1", 10.0)], omega=1.0)
        rep = check_gates(m, false_serve=0, false_serve_bound=0)
        self.assertTrue(rep.passed)
        self.assertEqual(rep.reasons, [])

    def test_false_fire_fails_the_gate(self):
        m = score([Fire("g1", 99.0)], [Edge("g1", 10.0)], omega=1.0)  # FP=1
        rep = check_gates(m, false_serve=0, false_serve_bound=0)
        self.assertFalse(rep.passed)
        self.assertTrue(any("false-fire" in r for r in rep.reasons))

    def test_false_serve_over_bound_fails_the_gate(self):
        m = score([Fire("g1", 10.2)], [Edge("g1", 10.0)], omega=1.0)  # clean fires
        rep = check_gates(m, false_serve=3, false_serve_bound=1)
        self.assertFalse(rep.passed)
        self.assertTrue(any("false-serve" in r for r in rep.reasons))

    def test_false_serve_within_bound_passes(self):
        m = score([Fire("g1", 10.2)], [Edge("g1", 10.0)], omega=1.0)
        rep = check_gates(m, false_serve=1, false_serve_bound=2)
        self.assertTrue(rep.passed)


if __name__ == "__main__":
    unittest.main()
