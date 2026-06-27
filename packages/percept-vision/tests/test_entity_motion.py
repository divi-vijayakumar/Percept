"""Predictive continuity — the EntityGraph holds a fast-moving subject's id across a displacement.

The base matcher is memoryless IoU (match the stale last box), so a subject that displaces faster than its
own box-overlap breaks → a new id is minted → a moving hero fragments. The graph adds a constant-velocity
recovery: when normal continuity+appearance fails, it matches against where the entity is PREDICTED to be.
It AUGMENTS (only fires on a miss) and self-disables on slow/static objects, so it costs nothing until a
subject moves fast — hence it is always on, not a mode.
"""

from __future__ import annotations

import unittest

from percept.adapters import Box, Observation
from percept.entity import EntityGraph
from percept.identity import ContinuityIdentity


def _ids(graph, boxes_at_times):
    return [graph.ingest([Observation("person", b, 0.9)], t=t)[0].id for t, b in boxes_at_times]


# A subject moving steadily right at ~6 fps; the first frames overlap (so a velocity is learned), then a
# DROPPED frame makes the next box jump beyond IoU overlap with the last box — the memoryless break. The
# gap (0.3 s) is within the coast window (coast_s 1.0).
TRACK = [
    (0.0, Box(0.00, 0.4, 0.10, 0.2)),
    (0.2, Box(0.04, 0.4, 0.10, 0.2)),
    (0.4, Box(0.08, 0.4, 0.10, 0.2)),
    (0.7, Box(0.16, 0.4, 0.10, 0.2)),  # a frame dropped → IoU vs last box < 0.5
]


class PredictiveContinuityTests(unittest.TestCase):
    def test_memoryless_matcher_alone_would_fragment(self):
        # The underlying break the graph repairs: raw IoU continuity can't match the post-jump box.
        m = ContinuityIdentity()
        last = type("E", (), {"id": "e1", "label": "person", "box": TRACK[2][1]})()
        self.assertIsNone(m.match(Observation("person", TRACK[3][1], 0.9), [last]))

    def test_motion_holds_the_same_id_across_the_jump(self):
        ids = _ids(EntityGraph(identity=ContinuityIdentity()), TRACK)
        self.assertEqual(len(set(ids)), 1)           # prediction re-links to the same entity

    def test_coast_bound_rejects_a_stale_track(self):
        # after the coast window, a prediction must NOT reach out and claim a far-away passer-by.
        g = EntityGraph(identity=ContinuityIdentity(), coast_s=1.0)
        _ids(g, TRACK[:3])                            # build a velocity, last seen at t=0.4
        late = g.ingest([Observation("person", Box(0.5, 0.4, 0.1, 0.2), 0.9)], t=10.0)[0].id
        self.assertNotIn(late, {e.id for e in g.active() if e.id != late})  # a fresh id, not the stale one

    def test_slow_subject_is_unaffected(self):
        # a near-stationary subject keeps a single id (the recovery path is a no-op when nothing moves).
        still = [(i * 0.2, Box(0.30, 0.30, 0.10, 0.2)) for i in range(5)]
        self.assertEqual(len(set(_ids(EntityGraph(identity=ContinuityIdentity()), still))), 1)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
