"""Independent exhaustive oracle for the scorer's matching cardinality."""

import itertools
import pathlib
import sys
import unittest


ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from scorer import Incident, score  # noqa: E402
from zone_entry import Occurrence  # noqa: E402


def event(index: int, onset: float) -> Occurrence:
    return Occurrence(str(index), "oracle", index, 0, 0, onset, onset)


def brute_force_cardinality(onsets: tuple[float, ...], fires: tuple[float, ...], window: float) -> int:
    best = 0

    def search(fire_index: int, used: frozenset[int], matches: int) -> None:
        nonlocal best
        if fire_index == len(fires):
            best = max(best, matches)
            return
        search(fire_index + 1, used, matches)
        for occurrence_index, onset in enumerate(onsets):
            if occurrence_index not in used and onset <= fires[fire_index] <= onset + window:
                search(fire_index + 1, used | {occurrence_index}, matches + 1)

    search(0, frozenset(), 0)
    return best


class IndependentMatchingOracleTests(unittest.TestCase):
    def test_all_small_timeline_combinations_match_exhaustive_oracle(self):
        timeline = (0.0, 0.5, 1.0, 1.5, 2.0)
        checked = 0
        for onsets in itertools.combinations(timeline, 3):
            for fires in itertools.combinations(timeline, 3):
                expected = brute_force_cardinality(onsets, fires, 1.0)
                measured = score(
                    [event(index, onset) for index, onset in enumerate(onsets)],
                    [
                        Incident(str(index), "fire", timestamp, timestamp)
                        for index, timestamp in enumerate(fires)
                    ],
                    match_window_seconds=1.0,
                    timely_deadline_seconds=2.0,
                    source_hours=1.0,
                )
                self.assertEqual(expected, measured["tp"], (onsets, fires))
                checked += 1
        self.assertEqual(100, checked)


if __name__ == "__main__":
    unittest.main()
