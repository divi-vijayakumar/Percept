"""L1: identity matchers — continuity, appearance re-id across a gap, distinctness (stub embedder)."""

import unittest

from percept.adapters import Box, Observation
from percept.entity import EntityGraph
from percept.identity import AppearanceIdentity, ContinuityIdentity, cosine


class _Known:
    """A minimal stand-in for an entity the matcher inspects (id/label/box/appearance)."""

    def __init__(self, id, label, box, appearance=None):
        self.id, self.label, self.box, self.appearance = id, label, box, appearance


NEAR = Box(0.40, 0.40, 0.20, 0.20)
SAME = Box(0.41, 0.40, 0.20, 0.20)  # high IoU with NEAR
FAR = Box(0.05, 0.05, 0.10, 0.10)  # disjoint from NEAR


class TestContinuity(unittest.TestCase):
    def setUp(self):
        self.id = ContinuityIdentity(iou_match=0.5)

    def test_matches_overlapping_same_label(self):
        known = [_Known("e1", "cup", NEAR)]
        self.assertEqual(self.id.match(Observation("cup", SAME, 0.9), known), "e1")

    def test_no_match_when_disjoint(self):
        known = [_Known("e1", "cup", NEAR)]
        self.assertIsNone(self.id.match(Observation("cup", FAR, 0.9), known))

    def test_label_must_match(self):
        known = [_Known("e1", "mug", NEAR)]
        self.assertIsNone(self.id.match(Observation("cup", SAME, 0.9), known))


class TestAppearance(unittest.TestCase):
    def setUp(self):
        self.id = AppearanceIdentity(threshold=0.9, iou_match=0.5)

    def test_cosine_helper(self):
        self.assertAlmostEqual(cosine([1, 0, 0], [1, 0, 0]), 1.0)
        self.assertAlmostEqual(cosine([1, 0, 0], [0, 1, 0]), 0.0)

    def test_continuity_takes_precedence(self):
        known = [_Known("e1", "cup", NEAR, appearance=[0, 1, 0])]
        # overlapping box -> continuity wins even though signature differs
        self.assertEqual(self.id.match(Observation("cup", SAME, 0.9), known, signature=[1, 0, 0]), "e1")

    def test_reid_across_gap_by_appearance(self):
        known = [_Known("e1", "cup", NEAR, appearance=[1, 0, 0])]
        # disjoint box (left + re-entered) but matching signature -> same id
        self.assertEqual(self.id.match(Observation("cup", FAR, 0.9), known, signature=[1, 0, 0]), "e1")

    def test_distinct_appearance_is_new(self):
        known = [_Known("e1", "cup", NEAR, appearance=[1, 0, 0])]
        self.assertIsNone(self.id.match(Observation("cup", FAR, 0.9), known, signature=[0, 1, 0]))

    def test_no_signature_degrades_to_continuity(self):
        known = [_Known("e1", "cup", NEAR, appearance=[1, 0, 0])]
        self.assertIsNone(self.id.match(Observation("cup", FAR, 0.9), known, signature=None))


class TestEntityGraphReid(unittest.TestCase):
    """The payoff: appearance re-id beats pure IoU when an object leaves and re-enters tracking."""

    SIGS = {"mug": [1.0, 0.0, 0.0], "mug-moved": [1.0, 0.0, 0.0], "other": [0.0, 1.0, 0.0]}

    def _embedder(self, frame, box):
        return self.SIGS[str(frame)]

    def test_appearance_reids_across_gap(self):
        g = EntityGraph(
            iou_match=0.5,
            identity=AppearanceIdentity(threshold=0.9, iou_match=0.5),
            embedder=self._embedder,
        )
        [e1] = g.ingest([Observation("cup", NEAR, 0.9)], frame="mug", t=1.0)
        # same object, moved far (continuity would fail) but same appearance -> SAME id
        [e2] = g.ingest([Observation("cup", FAR, 0.9)], frame="mug-moved", t=2.0)
        self.assertEqual(e1.id, e2.id)
        self.assertEqual(len(g), 1)

    def test_pure_continuity_mints_new_across_gap(self):
        g = EntityGraph(iou_match=0.5)  # default ContinuityIdentity, no embedder
        [e1] = g.ingest([Observation("cup", NEAR, 0.9)], frame="mug", t=1.0)
        [e2] = g.ingest([Observation("cup", FAR, 0.9)], frame="mug-moved", t=2.0)
        self.assertNotEqual(e1.id, e2.id)  # the gap problem appearance re-id solves

    def test_distinct_object_is_new_entity(self):
        g = EntityGraph(
            identity=AppearanceIdentity(threshold=0.9, iou_match=0.5), embedder=self._embedder
        )
        g.ingest([Observation("cup", NEAR, 0.9)], frame="mug", t=1.0)
        [e2] = g.ingest([Observation("cup", FAR, 0.9)], frame="other", t=2.0)
        self.assertEqual(len(g), 2)
        self.assertEqual(e2.appearance, [0.0, 1.0, 0.0])


if __name__ == "__main__":
    unittest.main()
