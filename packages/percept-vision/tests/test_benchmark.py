"""L1: the consumer-facing per-concern benchmark — no models, no keys (D2).

Two surfaces:

* the **pure scorer** (:func:`eval.benchmark.per_concern_report`, :func:`score_concern`) on
  synthetic ``(expected_edges, fires, asks)`` — asserts the precision / recall / false-fire /
  refuse math directly;
* the **manifest replay** (:func:`eval.benchmark.run_manifest`) driving a real ``Percept`` on the
  deterministic fake backend — asserts a fire clip scores a TP and a no-fire clip stays clean
  (no false-fire) end-to-end, offline.
"""

import json
import tempfile
import unittest
from pathlib import Path

from eval.benchmark import (
    BenchmarkGate,
    ConcernObservation,
    Manifest,
    check_benchmark_gate,
    edges_from_manifest,
    per_concern_report,
    run_manifest,
    score_concern,
)
from eval.replay import Edge, Fire

OMEGA = 2.0


# --- pure per-concern scorer (synthetic fire/edge lists) -----------------------------------------


class TestScoreConcern(unittest.TestCase):
    def test_perfect_concern(self):
        # one edge, one in-window fire -> precision 1, recall 1, no false-fire.
        obs = ConcernObservation("c", expected_edges=[Edge("c", 10.0)], fires=[Fire("c", 10.4)])
        r = score_concern(obs, OMEGA)
        self.assertEqual((r.tp, r.fn, r.fp), (1, 0, 0))
        self.assertAlmostEqual(r.precision, 1.0)
        self.assertAlmostEqual(r.recall, 1.0)
        self.assertEqual(r.false_fire, 0)
        self.assertAlmostEqual(r.refuse_rate, 0.0)

    def test_half_precision_half_recall(self):
        # 2 edges, 1 hit + 1 stray fire + 1 miss -> P 1/2, R 1/2, false-fire 1.
        obs = ConcernObservation(
            "c",
            expected_edges=[Edge("c", 10.0), Edge("c", 20.0)],
            fires=[Fire("c", 10.5), Fire("c", 99.0)],
        )
        r = score_concern(obs, OMEGA)
        self.assertAlmostEqual(r.precision, 0.5)
        self.assertAlmostEqual(r.recall, 0.5)
        self.assertEqual(r.false_fire, 1)
        self.assertAlmostEqual(r.false_fire_rate, 0.5)  # 1 of 2 fires was wrong

    def test_refuse_rate(self):
        # 1 fire + 3 asks -> refuse rate 3/4.
        obs = ConcernObservation(
            "c", expected_edges=[Edge("c", 10.0)], fires=[Fire("c", 10.2)], n_asks=3
        )
        r = score_concern(obs, OMEGA)
        self.assertAlmostEqual(r.refuse_rate, 0.75)

    def test_no_fires_is_a_recall_floor_not_a_false_fire(self):
        # an edge with no fire is a miss (recall 0), never a false-fire (precision vacuously 1).
        obs = ConcernObservation("c", expected_edges=[Edge("c", 10.0)], fires=[])
        r = score_concern(obs, OMEGA)
        self.assertEqual(r.false_fire, 0)
        self.assertAlmostEqual(r.recall, 0.0)
        self.assertAlmostEqual(r.precision, 1.0)


class TestPerConcernReport(unittest.TestCase):
    def test_two_concerns_micro_average(self):
        obs = [
            ConcernObservation("a", expected_edges=[Edge("a", 10.0)], fires=[Fire("a", 10.3)]),
            ConcernObservation(
                "b",
                expected_edges=[Edge("b", 10.0), Edge("b", 20.0)],
                fires=[Fire("b", 10.3)],  # 1 hit, 1 miss
            ),
        ]
        rep = per_concern_report(obs, OMEGA)
        self.assertAlmostEqual(rep["concerns"]["a"]["recall"], 1.0)
        self.assertAlmostEqual(rep["concerns"]["b"]["recall"], 0.5)
        # overall micro-avg: tp=2, fn=1 -> recall 2/3; fp=0 -> precision 1, false-fire 0.
        self.assertEqual(rep["overall"]["false_fire"], 0)
        self.assertAlmostEqual(rep["overall"]["recall"], 2 / 3)
        self.assertAlmostEqual(rep["overall"]["precision"], 1.0)

    def test_false_fire_rolls_up(self):
        obs = [
            ConcernObservation("a", expected_edges=[Edge("a", 10.0)], fires=[Fire("a", 99.0)]),  # FP
        ]
        rep = per_concern_report(obs, OMEGA)
        self.assertEqual(rep["overall"]["false_fire"], 1)


# --- manifest parsing + edge extraction ----------------------------------------------------------


class TestManifest(unittest.TestCase):
    def test_undeclared_concern_raises(self):
        with self.assertRaises(ValueError):
            Manifest.from_dict(
                {
                    "concerns": [{"id": "coffee", "condition": "x"}],
                    "clips": [{"clip_id": "c1", "frames": [], "expected": {"typo": [1.0]}}],
                }
            )

    def test_concern_needs_id_and_condition(self):
        with self.assertRaises(ValueError):
            Manifest.from_dict({"concerns": [{"id": "x"}], "clips": []})

    def test_edges_namespaced_by_clip(self):
        m = Manifest.from_dict(
            {
                "concerns": [{"id": "coffee", "condition": "x"}],
                "clips": [
                    {"clip_id": "c1", "frames": [], "expected": {"coffee": [1.0]}},
                    {"clip_id": "c2", "frames": [], "expected": {"coffee": [2.0]}},
                ],
            }
        )
        edges = edges_from_manifest(m)
        ids = sorted(e.goal_id for e in edges["coffee"])
        self.assertEqual(ids, ["c1::coffee", "c2::coffee"])


# --- end-to-end manifest replay through a real Percept on the fake backend (offline) -------------

# fake-backend scripted tokens (see percept.backends.fake): sip-coffee -> satisfied @0.9 (FIRE),
# sip-water/empty -> not satisfied (no fire), maybe-coffee -> satisfied @0.4 (below gate -> ASK).
_FIRE_AND_CLEAN = {
    "omega": OMEGA,
    "concerns": [{"id": "coffee", "condition": "the user is drinking coffee", "say": "caffeine!"}],
    "clips": [
        # a fire clip: empty then coffee -> a rising-edge FIRE at t=1.0 (the labelled edge).
        {"clip_id": "c1", "frames": [[0.0, "empty"], [1.0, "sip-coffee"]], "expected": {"coffee": [1.0]}},
        # a no-fire clip: only water -> any fire here would be a false-fire.
        {"clip_id": "c2", "frames": [[0.0, "empty"], [1.0, "sip-water"]], "expected": {"coffee": []}},
    ],
}


class TestRunManifestOffline(unittest.TestCase):
    def test_fire_clip_scores_tp_and_clean_clip_has_no_false_fire(self):
        rep = run_manifest(Manifest.from_dict(_FIRE_AND_CLEAN))
        coffee = rep["concerns"]["coffee"]
        self.assertEqual(coffee["tp"], 1)  # the coffee rising edge was hit
        self.assertEqual(coffee["false_fire"], 0)  # the water clip stayed silent
        self.assertAlmostEqual(coffee["precision"], 1.0)
        self.assertAlmostEqual(coffee["recall"], 1.0)
        self.assertEqual(rep["overall"]["false_fire"], 0)

    def test_ask_clip_refuses_instead_of_firing(self):
        # maybe-coffee is confidence 0.4 -> below the gate threshold -> ASK, not FIRE.
        manifest = Manifest.from_dict(
            {
                "omega": OMEGA,
                "concerns": [{"id": "coffee", "condition": "drinking coffee", "say": "x"}],
                "clips": [
                    {
                        "clip_id": "a1",
                        "frames": [[0.0, "empty"], [1.0, "maybe-coffee"]],
                        "expected": {"coffee": [1.0]},
                    }
                ],
            }
        )
        coffee = run_manifest(manifest)["concerns"]["coffee"]
        self.assertEqual(coffee["n_fires"], 0)
        self.assertEqual(coffee["n_asks"], 1)
        self.assertEqual(coffee["false_fire"], 0)
        self.assertAlmostEqual(coffee["refuse_rate"], 1.0)

    def test_loads_manifest_from_file(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "manifest.json"
            p.write_text(json.dumps(_FIRE_AND_CLEAN))
            rep = run_manifest(Manifest.load(p))
            self.assertEqual(rep["overall"]["false_fire"], 0)


# --- the consumer CI regression gate -------------------------------------------------------------


class TestBenchmarkGate(unittest.TestCase):
    def _clean_report(self):
        return run_manifest(Manifest.from_dict(_FIRE_AND_CLEAN))

    def test_clean_report_passes_strict_gate(self):
        passed, reasons = check_benchmark_gate(
            self._clean_report(), BenchmarkGate(min_precision=1.0, min_recall=1.0)
        )
        self.assertTrue(passed, reasons)
        self.assertEqual(reasons, [])

    def test_false_fire_fails_the_gate(self):
        rep = per_concern_report(
            [ConcernObservation("c", expected_edges=[Edge("c", 10.0)], fires=[Fire("c", 99.0)])],
            OMEGA,
        )
        passed, reasons = check_benchmark_gate(rep, BenchmarkGate())
        self.assertFalse(passed)
        self.assertTrue(any("false-fire" in r for r in reasons))

    def test_recall_floor_fails_per_concern(self):
        rep = per_concern_report(
            [
                ConcernObservation(
                    "c", expected_edges=[Edge("c", 10.0), Edge("c", 20.0)], fires=[Fire("c", 10.3)]
                )
            ],
            OMEGA,
        )  # recall 0.5
        passed, reasons = check_benchmark_gate(rep, BenchmarkGate(min_recall=0.8))
        self.assertFalse(passed)
        self.assertTrue(any("recall" in r and "'c'" in r for r in reasons))


if __name__ == "__main__":
    unittest.main()
