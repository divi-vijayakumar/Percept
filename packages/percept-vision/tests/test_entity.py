"""L1: the entity graph — continuity, provenance, assertion-beats-observation, focus, eviction."""

import unittest

from percept.adapters import Box, Observation
from percept.entity import EntityGraph

A = Box(0.4, 0.4, 0.2, 0.2)
B = Box(0.1, 0.1, 0.2, 0.2)


def cup(box=A, conf=0.9, **attrs):
    return Observation(label="cup", box=box, confidence=conf, attrs=attrs)


class TestEntityGraph(unittest.TestCase):
    def setUp(self):
        self.g = EntityGraph(iou_match=0.5, max_entities=8)

    def test_mints_new_entity(self):
        [e] = self.g.ingest([cup()], t=1.0)
        self.assertEqual(e.label, "cup")
        self.assertEqual(len(self.g), 1)

    def test_continuity_same_box_same_id(self):
        [e1] = self.g.ingest([cup()], t=1.0)
        [e2] = self.g.ingest([cup()], t=2.0)  # same box+label -> same entity
        self.assertEqual(e1.id, e2.id)
        self.assertEqual(len(self.g), 1)

    def test_distinct_box_new_id(self):
        [e1] = self.g.ingest([cup(box=A)], t=1.0)
        [e2] = self.g.ingest([cup(box=B)], t=2.0)  # disjoint -> new entity
        self.assertNotEqual(e1.id, e2.id)
        self.assertEqual(len(self.g), 2)

    def test_observed_attrs_carry_provenance(self):
        [e] = self.g.ingest([cup(state="full")], t=1.0)
        self.assertEqual(e.attributes["state"].source, "observed")

    def test_assertion_beats_observation(self):
        [e] = self.g.ingest([cup(contents="coffee")], t=1.0)
        self.g.assert_attribute(e.id, "contents", "tea", t=1.5)
        self.assertEqual(e.attributes["contents"].value, "tea")
        self.assertEqual(e.attributes["contents"].source, "asserted")
        # a later observation must NOT clobber the asserted fact
        self.g.ingest([cup(contents="coffee")], t=2.0)
        self.assertEqual(e.attributes["contents"].value, "tea")
        self.assertEqual(e.asserted(), {"contents": "tea"})

    def test_in_focus_is_largest_in_last_frame(self):
        big = Observation("cup", Box(0.1, 0.1, 0.5, 0.5), 0.9)
        small = Observation("cup", Box(0.8, 0.8, 0.1, 0.1), 0.9)
        self.g.ingest([small, big], t=1.0)
        self.assertEqual(self.g.in_focus().box, big.box)

    def test_in_focus_none_when_no_frame(self):
        self.assertIsNone(self.g.in_focus())

    def test_eviction_respects_max(self):
        g = EntityGraph(iou_match=0.5, max_entities=2)
        g.ingest([Observation("a", Box(0, 0, 0.1, 0.1), 0.9)], t=1.0)
        g.ingest([Observation("b", Box(0.3, 0.3, 0.1, 0.1), 0.9)], t=2.0)
        g.ingest([Observation("c", Box(0.6, 0.6, 0.1, 0.1), 0.9)], t=3.0)
        self.assertEqual(len(g), 2)  # oldest ('a') evicted

    def test_note_mention_binds_to_focus(self):
        self.g.ingest([cup()], t=1.0)
        bound = self.g.note_mention("this is my travel mug", t=1.2)
        self.assertIsNotNone(bound)
        self.assertEqual(self.g.in_focus().attributes["mention"].source, "asserted")


if __name__ == "__main__":
    unittest.main()
