from __future__ import annotations

import unittest

from eval.bench import track_escalation_policy as track
from percept_harness.tier0 import gray_frame


class EscalationPolicyTrackTests(unittest.TestCase):
    def test_policy_stats_count_hits_false_calls_and_latency(self):
        original = track._frame_series
        try:
            track._frame_series = lambda _path: [
                (0.0, gray_frame(2, 2, 0)),
                (0.5, gray_frame(2, 2, 0)),
                (1.0, gray_frame(2, 2, 255)),
                (2.0, gray_frame(2, 2, 255)),
                (3.0, gray_frame(2, 2, 0)),
            ]
            policies = {"motion": track.Policy("motion", threshold=0.1)}
            stats = track._score_policies(
                [{"path": "synthetic.mp4", "onset_s": 1.0, "end_s": 2.0}],
                policies,
            )["motion"]
        finally:
            track._frame_series = original

        self.assertEqual(stats.hits, 1)
        self.assertEqual(stats.tp_calls, 1)
        self.assertEqual(stats.fp_calls, 1)
        self.assertEqual(stats.recall, 1.0)
        self.assertEqual(stats.precision_proxy, 0.5)
        self.assertEqual(stats.median_latency_s, 0.0)


if __name__ == "__main__":
    unittest.main()
