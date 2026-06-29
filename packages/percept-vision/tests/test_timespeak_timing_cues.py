"""L1 coverage for the timespeak timing-cue filter.

These are pure stdlib ``unittest`` tests (the package is run via ``python -m unittest discover``).
They pin the derivation lever added in ``track_timespeak._timing_cues``: cues that carry no real
timing window (a degenerate placeholder ``(lo, hi)`` with ``lo == hi``, as proactiveqa-e2e emits)
must NOT contribute to the lateness/latency derivation, because a zero-width window opens and closes
at the same instant and mechanically scores every fire as "late" against no ground truth.
"""

import unittest

from eval.bench.derived_interaction import DerivedCue, time_metrics
from eval.bench.track_timespeak import _timing_cues


class TimingCueFilterTests(unittest.TestCase):
    def test_drops_zero_width_placeholder_windows(self):
        cues = [
            DerivedCue("pt", "real onset", (10.0, 15.0), (11.0,)),
            DerivedCue("pvqa", "placeholder turn 1", (0.0, 0.0), (60.0,)),
            DerivedCue("pvqa", "placeholder turn 2", (0.0, 0.0), (60.0,)),
        ]

        kept = _timing_cues(cues)

        self.assertEqual([c.label for c in kept], ["real onset"])

    def test_keeps_all_genuine_windows(self):
        cues = [
            DerivedCue("a", "w1", (1.0, 3.0), (1.5,)),
            DerivedCue("b", "w2", (0.0, 20.9), (12.0,)),  # window may open at 0 but has real width
        ]

        self.assertEqual(_timing_cues(cues), cues)

    def test_placeholder_windows_no_longer_inflate_late_fire_rate(self):
        # One genuine on-time fire plus several placeholder turns that each fire at t=60 against a
        # (0,0) window. Before the filter, every placeholder counts as a late fire (0.75 here);
        # after filtering, late_fire_rate reflects only the genuine cue.
        cues = [
            DerivedCue("pt", "on time", (10.0, 15.0), (10.5,)),
            DerivedCue("pvqa", "turn 1", (0.0, 0.0), (60.0,)),
            DerivedCue("pvqa", "turn 2", (0.0, 0.0), (60.0,)),
            DerivedCue("pvqa", "turn 3", (0.0, 0.0), (60.0,)),
        ]

        before = time_metrics(cues, omega_s=2.0)
        after = time_metrics(_timing_cues(cues), omega_s=2.0)

        self.assertEqual(before["late_fire_rate"], 3 / 4)
        self.assertEqual(after["late_fire_rate"], 0.0)
        # The genuine cue's in-window latency is untouched (no precision-for-latency trade).
        self.assertEqual(after["latency_p50"], 0.5)
        self.assertEqual(after["timing_accuracy"], 1.0)

    def test_filter_does_not_lower_honest_latency(self):
        # Placeholder cues never produce an in-window (valid) latency, so dropping them must leave
        # the latency_p50 untouched while removing the false lateness.
        cues = [
            DerivedCue("c", "late but real", (10.0, 30.0), (18.0,)),  # valid latency 8.0
            DerivedCue("pvqa", "placeholder", (0.0, 0.0), (60.0,)),
        ]

        before = time_metrics(cues, omega_s=2.0)
        after = time_metrics(_timing_cues(cues), omega_s=2.0)

        self.assertEqual(before["latency_p50"], 8.0)
        self.assertEqual(after["latency_p50"], 8.0)


if __name__ == "__main__":
    unittest.main()
