"""L1: the perception cache (``FrameHashLayer``) — synthetic frames, no models (spec §6.2).

Covers near-match (identical -> HIT, structural change -> DEFER), the confidence math
(``min(verdict.conf, sim*age_decay)``: stale decays, low-conf never served confidently), the TTL
hard ceiling, and the invalidation order (goal-change drop, assertion-override drop, TTL backstop).

Frames are synthetic 2D luminance grids / bytes — the layer is pure stdlib, no numpy required.
"""

import unittest

from percept.adapters import Box, Verdict
from percept.cache.framehash import CacheKey, FrameHashLayer, dhash, hamming
from percept.config import CacheSettings


def smooth_grid(n: int = 10) -> list[list[float]]:
    """A left-to-right luminance ramp (every horizontal-adjacency bit set)."""
    return [[float(x) for x in range(n)] for _ in range(n)]


def reversed_grid(n: int = 10) -> list[list[float]]:
    """The structural opposite of ``smooth_grid`` (max Hamming distance)."""
    return [[float(n - 1 - x) for x in range(n)] for _ in range(n)]


class _Clock:
    """A hand-cranked monotonic clock so TTL/decay tests are deterministic (no real time)."""

    def __init__(self, t: float = 0.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


class TestDHash(unittest.TestCase):
    def test_identical_frame_zero_distance(self):
        g = smooth_grid()
        self.assertEqual(hamming(dhash(g), dhash(g)), 0)

    def test_structural_change_large_distance(self):
        self.assertEqual(hamming(dhash(smooth_grid()), dhash(reversed_grid())), 64)

    def test_bytes_frame_is_hashable(self):
        self.assertIsNotNone(dhash(bytes(range(64))))

    def test_unhashable_frame_returns_none(self):
        # a bare token (string) carries no visual signal -> not griddable.
        self.assertIsNone(dhash("sip-coffee"))


class TestNearMatch(unittest.TestCase):
    def setUp(self):
        self.clock = _Clock(0.0)
        self.cfg = CacheSettings(ttl_s=10.0, tau_bits=6)
        self.layer = FrameHashLayer(self.cfg, clock=self.clock)
        self.v = Verdict(satisfied=True, confidence=0.9, evidence="cached coffee")

    def test_cold_lookup_defers(self):
        r = self.layer.lookup(CacheKey("g1", smooth_grid(), None, "e1"))
        self.assertEqual(r.status, "DEFER")

    def test_identical_frame_hits(self):
        key = CacheKey("g1", smooth_grid(), None, "e1")
        self.layer.store(key, self.v, 0.9)
        r = self.layer.lookup(CacheKey("g1", smooth_grid(), None, "e1"))
        self.assertEqual(r.status, "HIT")
        self.assertIs(r.verdict, self.v)
        self.assertEqual(r.source_layer, "framehash")
        self.assertEqual(r.provenance, "cached")

    def test_structural_change_defers(self):
        self.layer.store(CacheKey("g1", smooth_grid(), None, "e1"), self.v, 0.9)
        r = self.layer.lookup(CacheKey("g1", reversed_grid(), None, "e1"))
        self.assertEqual(r.status, "DEFER")  # beyond tau_bits -> re-perceive

    def test_tau_bits_zero_requires_exact(self):
        cfg = CacheSettings(ttl_s=10.0, tau_bits=0)
        layer = FrameHashLayer(cfg, clock=self.clock)
        g = smooth_grid()
        layer.store(CacheKey("g1", g, None, "e1"), self.v, 0.9)
        # a one-pixel structural nudge that flips >=1 gradient bit must now DEFER.
        nudged = [row[:] for row in g]
        nudged[0][1] = -100.0  # flips the (0,0) adjacency bit
        self.assertEqual(layer.lookup(CacheKey("g1", nudged, None, "e1")).status, "DEFER")


class TestConfidenceMath(unittest.TestCase):
    def setUp(self):
        self.clock = _Clock(0.0)
        self.cfg = CacheSettings(ttl_s=10.0, tau_bits=6)
        self.layer = FrameHashLayer(self.cfg, clock=self.clock)

    def test_fresh_identical_confidence_is_verdict_conf(self):
        key = CacheKey("g1", smooth_grid(), None, "e1")
        self.layer.store(key, Verdict(True, 0.9), 0.9)
        r = self.layer.lookup(CacheKey("g1", smooth_grid(), None, "e1"))
        # sim=1.0 (dist 0), decay=1.0 (age 0) -> min(0.9, 1.0) = 0.9
        self.assertAlmostEqual(r.confidence, 0.9)

    def test_stale_entry_decays_confidence(self):
        self.layer.store(CacheKey("g1", smooth_grid(), None, "e1"), Verdict(True, 0.9), 0.9)
        self.clock.t = 5.0  # half the TTL -> decay 0.5
        r = self.layer.lookup(CacheKey("g1", smooth_grid(), None, "e1"))
        self.assertEqual(r.status, "HIT")
        self.assertAlmostEqual(r.confidence, 0.5)  # min(0.9, 1.0*0.5)

    def test_low_conf_verdict_never_served_confidently(self):
        # a verdict that was itself low-confidence is capped by min() -> can't be laundered to high.
        self.layer.store(CacheKey("g1", smooth_grid(), None, "e1"), Verdict(True, 0.2), 0.2)
        r = self.layer.lookup(CacheKey("g1", smooth_grid(), None, "e1"))
        self.assertEqual(r.status, "HIT")
        self.assertAlmostEqual(r.confidence, 0.2)

    def test_near_but_not_identical_lowers_confidence(self):
        g = smooth_grid()
        self.layer.store(CacheKey("g1", g, None, "e1"), Verdict(True, 0.9), 0.9)
        # flip exactly a couple of gradient bits -> sim < 1 -> confidence drops below 0.9.
        nudged = [row[:] for row in g]
        nudged[0][1] = -100.0
        nudged[1][1] = -100.0
        r = self.layer.lookup(CacheKey("g1", nudged, None, "e1"))
        if r.status == "HIT":  # within tau_bits
            self.assertLess(r.confidence, 0.9)


class TestTTL(unittest.TestCase):
    def test_ttl_is_a_hard_ceiling(self):
        clock = _Clock(0.0)
        layer = FrameHashLayer(CacheSettings(ttl_s=10.0, tau_bits=6), clock=clock)
        layer.store(CacheKey("g1", smooth_grid(), None, "e1"), Verdict(True, 0.99), 0.99)
        clock.t = 10.0  # exactly at the TTL -> always DEFER, even on an identical frame
        self.assertEqual(layer.lookup(CacheKey("g1", smooth_grid(), None, "e1")).status, "DEFER")

    def test_zero_ttl_never_serves(self):
        layer = FrameHashLayer(CacheSettings(ttl_s=0.0, tau_bits=6), clock=_Clock(0.0))
        layer.store(CacheKey("g1", smooth_grid(), None, "e1"), Verdict(True, 0.9), 0.9)
        self.assertEqual(layer.lookup(CacheKey("g1", smooth_grid(), None, "e1")).status, "DEFER")


class TestInvalidation(unittest.TestCase):
    def setUp(self):
        self.clock = _Clock(0.0)
        self.layer = FrameHashLayer(CacheSettings(ttl_s=10.0, tau_bits=6), clock=self.clock)
        self.v = Verdict(True, 0.9)
        # two goals, two entities cached.
        self.layer.store(CacheKey("g1", smooth_grid(), None, "e1"), self.v, 0.9)
        self.layer.store(CacheKey("g2", smooth_grid(), None, "e1"), self.v, 0.9)
        self.layer.store(CacheKey("g1", smooth_grid(), None, "e2"), self.v, 0.9)

    def test_goal_change_drops_only_that_goal(self):
        dropped = self.layer.invalidate("g1")
        self.assertEqual(dropped, 2)  # (g1,e1) and (g1,e2)
        self.assertEqual(self.layer.lookup(CacheKey("g1", smooth_grid(), None, "e1")).status, "DEFER")
        self.assertEqual(self.layer.lookup(CacheKey("g2", smooth_grid(), None, "e1")).status, "HIT")

    def test_invalidate_all_clears_everything(self):
        self.assertEqual(self.layer.invalidate(None), 3)
        self.assertEqual(len(self.layer), 0)

    def test_assertion_override_drops_entity_across_goals(self):
        # a correction binds to an entity -> every cached verdict resting on e1 is suspect.
        dropped = self.layer.invalidate_entity("e1")
        self.assertEqual(dropped, 2)  # (g1,e1) and (g2,e1)
        self.assertEqual(self.layer.lookup(CacheKey("g1", smooth_grid(), None, "e2")).status, "HIT")

    def test_assertion_override_scoped_to_goal(self):
        dropped = self.layer.invalidate_entity("e1", goal_id="g1")
        self.assertEqual(dropped, 1)
        self.assertEqual(self.layer.lookup(CacheKey("g2", smooth_grid(), None, "e1")).status, "HIT")


class TestRoiScoping(unittest.TestCase):
    def test_roi_crops_to_a_subregion(self):
        # two grids that differ only OUTSIDE the ROI should still near-match within it.
        clock = _Clock(0.0)
        layer = FrameHashLayer(CacheSettings(ttl_s=10.0, tau_bits=2), clock=clock)
        base = smooth_grid(12)
        roi = Box(0.0, 0.0, 0.5, 0.5)  # top-left quadrant
        layer.store(CacheKey("g1", base, roi, "e1"), Verdict(True, 0.9), 0.9)
        # perturb the bottom-right (outside the ROI).
        other = [row[:] for row in base]
        for y in range(6, 12):
            for x in range(6, 12):
                other[y][x] = -50.0
        r = layer.lookup(CacheKey("g1", other, roi, "e1"))
        self.assertEqual(r.status, "HIT")  # the watched ROI is unchanged


if __name__ == "__main__":
    unittest.main()
