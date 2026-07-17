"""Smoke test for the Phase-A for-record runner (eval/viability/run_phase_a.py).

Exercises the runner end-to-end (fake judge, FakeClock — fast, deterministic) and asserts it
computes the Phase-A gate structure and refuses to run if bars are not locked. It does NOT assert
specific numbers (those are the for-record output) — only that the shape + honesty labels hold.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))  # import the sibling run_phase_a module

import run_phase_a as rpa  # noqa: E402


class PhaseARunnerSmoke(unittest.TestCase):
    def test_run_produces_gate_structure_and_honesty_labels(self):
        res = rpa.run()
        self.assertEqual(res["phase"], "A")
        self.assertIn("fake", res["substrate"].lower())
        g = res["gates"]
        # all three gate blocks present
        self.assertIn("G1_boundedness", g)
        self.assertIn("G2_econ_structure", g)
        self.assertIn("G3_latency_floor", g)
        # G1 has a real PASS/FAIL verdict
        self.assertIn(g["G1_boundedness"]["verdict"], ("PASS", "FAIL"))
        # G3 is explicitly labeled a fake floor, not a product number
        self.assertIn("FLOOR", g["G3_latency_floor"]["label"].upper())
        self.assertIn(g["G3_latency_floor"]["band"], ("GREEN", "YELLOW", "RED"))
        # G2 volume reduction is computed and > 1 (VID must beat the naive baseline structurally)
        self.assertGreater(g["G2_econ_structure"]["volume_reduction_factor"], 1.0)
        # the deferred-to-Phase-B list is non-empty and mentions recall/precision + provider
        deferred = " ".join(res["deferred_to_phase_b"]).lower()
        self.assertTrue(res["deferred_to_phase_b"])
        self.assertIn("recall", deferred)
        self.assertIn("provider", deferred)

    def test_refuses_when_bars_not_locked(self):
        original = rpa._load_locked_config
        rpa._load_locked_config = lambda: (_ for _ in ()).throw(SystemExit("locked check"))
        try:
            with self.assertRaises(SystemExit):
                rpa.run()
        finally:
            rpa._load_locked_config = original


if __name__ == "__main__":
    unittest.main()
