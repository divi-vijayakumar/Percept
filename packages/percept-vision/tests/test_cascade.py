"""L1: cache cascade composition — empty defers, HIT>=threshold short-circuits."""

import unittest

from percept.adapters import Verdict
from percept.cache import CacheCascade, CacheResult


class _StubLayer:
    """A scripted cache layer for testing cascade control flow (not intelligence).

    ``conf=None`` -> DEFER; otherwise a HIT stamped with this layer's own name (mirroring how a
    real layer tags ``source_layer`` on its own result).
    """

    def __init__(self, name, conf):
        self.name = name
        self._conf = conf
        self.lookups = 0

    def lookup(self, key):
        self.lookups += 1
        if self._conf is None:
            return CacheResult.defer()
        return CacheResult(
            status="HIT",
            verdict=Verdict(satisfied=True, confidence=self._conf),
            confidence=self._conf,
            source_layer=self.name,
        )

    def store(self, key, verdict, confidence):
        pass


class TestCascade(unittest.TestCase):
    def test_empty_cascade_defers(self):
        cascade = CacheCascade(layers=[], threshold=0.5)
        self.assertEqual(cascade.resolve("k").status, "DEFER")

    def test_hit_above_threshold_short_circuits(self):
        first = _StubLayer("a", 0.9)
        second = _StubLayer("b", 0.95)
        cascade = CacheCascade(layers=[first, second], threshold=0.5)
        result = cascade.resolve("k")
        self.assertEqual(result.status, "HIT")
        self.assertEqual(result.source_layer, "a")
        self.assertEqual(second.lookups, 0)  # short-circuited

    def test_hit_below_threshold_falls_through_to_defer(self):
        only = _StubLayer("a", 0.3)
        cascade = CacheCascade(layers=[only], threshold=0.5)
        self.assertEqual(cascade.resolve("k").status, "DEFER")

    def test_below_threshold_then_above_uses_second(self):
        first = _StubLayer("a", 0.3)
        second = _StubLayer("b", 0.8)
        cascade = CacheCascade(layers=[first, second], threshold=0.5)
        result = cascade.resolve("k")
        self.assertEqual(result.status, "HIT")
        self.assertEqual(result.source_layer, "b")
        self.assertEqual(first.lookups, 1)

    def test_all_defer_defers(self):
        a = _StubLayer("a", None)
        b = _StubLayer("b", None)
        cascade = CacheCascade(layers=[a, b], threshold=0.0)
        self.assertEqual(cascade.resolve("k").status, "DEFER")


if __name__ == "__main__":
    unittest.main()
