import pathlib
import sys
import unittest


ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from scorer import Incident, RuntimeCounts, latency_summary, score  # noqa: E402
from zone_entry import (  # noqa: E402
    ObjectRow,
    Occurrence,
    parse_virat_object_rows,
    point_in_polygon,
    transform_zone_entries,
)


ZONE = ((0.5, 0.0), (1.0, 0.0), (1.0, 1.0), (0.5, 1.0))


def row(track: int, frame: int, foot_x: float, **changes: object) -> ObjectRow:
    values: dict[str, object] = {
        "track_id": track,
        "frame": frame,
        "x": foot_x * 100 - 5,
        "y": 70,
        "width": 10,
        "height": 20,
        "class_id": 1,
    }
    values.update(changes)
    return ObjectRow(**values)  # type: ignore[arg-type]


class TransformerTests(unittest.TestCase):
    def transform(self, rows: list[ObjectRow], dwell: float = 0.0, excluded=()):
        return transform_zone_entries(
            rows,
            source_id="synthetic",
            fps=4,
            frame_width=100,
            frame_height=100,
            polygon=ZONE,
            dwell_seconds=dwell,
            excluded_intervals=excluded,
        )

    def test_boundary_is_inside_and_clean_crossing_emits_once(self):
        self.assertTrue(point_in_polygon((0.5, 0.5), ZONE))
        events = self.transform([row(7, 0, 0.4), row(7, 1, 0.5), row(7, 2, 0.8)])
        self.assertEqual(1, len(events))
        self.assertEqual((1, 1, 0.25), (
            events[0].onset_frame,
            events[0].confirmation_frame,
            events[0].onset_seconds,
        ))

    def test_initial_inside_is_left_censored_then_reentry_counts(self):
        events = self.transform(
            [row(1, 0, 0.8), row(1, 1, 0.8), row(1, 2, 0.4), row(1, 3, 0.8)]
        )
        self.assertEqual([3], [event.onset_frame for event in events])

    def test_dwell_uses_first_inside_as_onset(self):
        events = self.transform(
            [
                row(2, 0, 0.4),
                row(2, 1, 0.8),
                row(2, 2, 0.8),
                row(2, 3, 0.8),
            ],
            dwell=0.5,
        )
        self.assertEqual((1, 3), (events[0].onset_frame, events[0].confirmation_frame))

    def test_official_legacy_row_parser_is_strict(self):
        parsed = parse_virat_object_rows(["7 9 11 10 20 30 40 1\n"])
        self.assertEqual((7, 11, "not_provided"), (
            parsed[0].track_id,
            parsed[0].frame,
            parsed[0].occlusion,
        ))
        with self.assertRaisesRegex(ValueError, "expected 8 fields"):
            parse_virat_object_rows(["7 9 11"])

    def test_gap_exclusion_and_ambiguous_rows_cannot_prove_transition(self):
        cases = [
            [row(3, 0, 0.4), row(3, 2, 0.8)],
            [row(3, 0, 0.4), row(3, 1, 0.8)],
            [row(3, 0, 0.4), row(3, 1, 0.8, occlusion="mostly"), row(3, 2, 0.8)],
            [row(3, 0, 0.4), row(3, 1, 0.8, height=2), row(3, 2, 0.8)],
        ]
        self.assertEqual([], self.transform(cases[0]))
        self.assertEqual([], self.transform(cases[1], excluded=((1, 1),)))
        self.assertEqual([], self.transform(cases[2]))
        self.assertEqual([], self.transform(cases[3]))


def occurrence(identity: str, onset: float) -> Occurrence:
    return Occurrence(identity, "synthetic", 1, 0, 0, onset, onset)


class ScorerTests(unittest.TestCase):
    def test_maximum_cardinality_matching_beats_greedy_order(self):
        result = score(
            [occurrence("a", 0.0), occurrence("b", 1.0)],
            [
                Incident("flexible", "fire", 1.1, 1.2),
                Incident("only-b", "fire", 2.5, 2.6),
            ],
            match_window_seconds=2,
            timely_deadline_seconds=2,
            source_hours=1,
        )
        self.assertEqual((2, 0, 0), (result["tp"], result["fp"], result["fn"]))

    def test_hand_calculated_quality_latency_and_reconciliation(self):
        result = score(
            [occurrence("a", 0), occurrence("b", 10), occurrence("c", 20)],
            [
                Incident("fa", "fire", 0.5, 1.0, {"decode": 0.1}),
                Incident("dup", "fire", 0.7, 0.8, {"decode": 0.2}),
                Incident("fb", "fire", 11.0, 13.0, {"decode": 0.3}),
                Incident("fp", "fire", 30.0, None),
                Incident("ask", "ask", 19.0),
                Incident("ignored", "fire", 40.0, 40.1),
            ],
            match_window_seconds=2,
            timely_deadline_seconds=2,
            source_hours=0.5,
            excluded_intervals_seconds=((39, 41),),
            runtime_counts=RuntimeCounts(4, 4, 4, 4, 3),
            expected_runtime_counts=RuntimeCounts(4, 4, 4, 4, 3),
        )
        self.assertEqual((2, 2, 1), (result["tp"], result["fp"], result["fn"]))
        self.assertEqual(0.5, result["precision"])
        self.assertEqual(2 / 3, result["recall"])
        self.assertEqual(1 / 3, result["timely_recall"])
        self.assertEqual((1, 1, 1), (result["ask"], result["duplicates"], result["misses"]))
        self.assertTrue(result["reconciliation"]["exact"])
        self.assertEqual({"p50": 2.0, "p95": 2.9, "p99": 2.98, "max": 3.0}, result["onset_to_receiver_ack_seconds"])

    def test_empty_latency_is_reported_as_absent(self):
        self.assertEqual(
            {"p50": None, "p95": None, "p99": None, "max": None},
            latency_summary([]),
        )


if __name__ == "__main__":
    unittest.main()
