"""Activating the AppearanceIdentity seam — re-acquire identity across a tracking GAP via an appearance
embedder (the validated hero-probe mechanism, now in the core graph).

The seam was built but dormant (no embedder injected). Here we inject the harness colour-histogram embedder
and prove the graph re-acquires the same id when a subject leaves tracking and re-enters at a new location —
where pure continuity (geometry) would mint a new id. This is the foundation the entity-scoped rhythm hero
will stand on. numpy/opencv live behind the harness embedder; the core only sees the float signature.
"""

from __future__ import annotations

import unittest

from percept.adapters import Box, Observation
from percept.entity import EntityGraph
from percept.identity import AppearanceIdentity, ContinuityIdentity

try:
    import numpy as np
    from percept.harness.appearance import appearance_signature
    _HAVE = True
except Exception:  # pragma: no cover - opencv/numpy not installed
    _HAVE = False

RED, BLUE = (0, 0, 255), (255, 0, 0)  # BGR


def _frame(color, box, size=100):
    """A size×size BGR frame with ``color`` filling ``box``'s region (rest black)."""
    f = np.zeros((size, size, 3), dtype=np.uint8)
    x0, y0 = int(box.x * size), int(box.y * size)
    x1, y1 = int((box.x + box.w) * size), int((box.y + box.h) * size)
    f[y0:y1, x0:x1] = color
    return f


@unittest.skipUnless(_HAVE, "needs numpy + opencv for the appearance embedder")
class AppearanceReidTests(unittest.TestCase):
    def test_signature_discriminates_colour(self):
        b = Box(0.1, 0.1, 0.3, 0.5)
        from percept.identity import cosine
        red, blue = appearance_signature(_frame(RED, b), b), appearance_signature(_frame(BLUE, b), b)
        self.assertGreater(cosine(red, appearance_signature(_frame(RED, b), b)), 0.95)  # same colour ≈ identical
        self.assertLess(cosine(red, blue), 0.2)                                         # red vs blue ≈ orthogonal

    def test_reacquires_identity_after_a_gap(self):
        g = EntityGraph(identity=AppearanceIdentity(threshold=0.5), embedder=appearance_signature)
        b1, b2, b3 = Box(0.05, 0.1, 0.2, 0.4), Box(0.7, 0.1, 0.2, 0.4), Box(0.4, 0.5, 0.2, 0.4)
        a_id = g.ingest([Observation("person", b1, 0.9)], frame=_frame(RED, b1), t=0.0)[0].id
        g.ingest([Observation("person", b2, 0.9)], frame=_frame(BLUE, b2), t=1.0)  # a blue bystander elsewhere
        # the red subject re-enters at a NEW location (no IoU overlap with b1) — continuity fails, appearance wins
        out = g.ingest([Observation("person", b3, 0.9)], frame=_frame(RED, b3), t=2.0)
        self.assertEqual(out[0].id, a_id)  # re-acquired the SAME identity across the gap

    def test_continuity_alone_loses_identity_across_the_jump(self):
        g = EntityGraph(identity=ContinuityIdentity())  # The continuity-only default drifts here.
        b1, b3 = Box(0.05, 0.1, 0.2, 0.4), Box(0.4, 0.5, 0.2, 0.4)
        a_id = g.ingest([Observation("person", b1, 0.9)], t=0.0)[0].id
        moved = g.ingest([Observation("person", b3, 0.9)], t=1.0)[0].id
        self.assertNotEqual(a_id, moved)  # geometry alone mints a new id on the jump (the failure re-id fixes)


if __name__ == "__main__":
    unittest.main()
