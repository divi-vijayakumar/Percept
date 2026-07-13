"""AGENT-D / D3a — re-arming a goal id must not serve a verdict cached under its OLD meaning.

Two holes, one symptom (a stale cached verdict served for a NEW condition):
  * ``RuntimeSession.arm() -> core.add_goal() -> goals.create()`` did NO cache invalidation (only
    ``apply_goal_ops`` did) — so a re-armed id kept its old cache entries;
  * ``CacheKey`` did not include the goal's CONDITION — so even a fresh key would collide with the old
    entry's slot.

Latent today (``CacheSettings.enabled=False`` by default), a live correctness bug once the cache is on;
these tests ENABLE the cache to exercise it.
"""
from __future__ import annotations

import asyncio
import dataclasses
import unittest

import percept
from percept.adapters import Goal, Verdict
from percept.backends.fake import FakeVision
from percept.cache.framehash import CacheKey, FrameHashLayer
from percept.config import CacheSettings, load_settings

# A griddable frame (bytes) so the FrameHashLayer actually hashes + caches (a bare Frame/token doesn't).
FRAME = bytes(range(64))


def _grid(n=10):
    return [[float(x) for x in range(n)] for _ in range(n)]


class D3aCacheKeyConditionTests(unittest.TestCase):
    """Layer-level: a verdict cached under condition A must NOT be served for condition B."""

    def test_condition_change_is_a_cache_miss(self):
        clock = lambda: 0.0  # noqa: E731 - hand-cranked, within TTL
        layer = FrameHashLayer(CacheSettings(ttl_s=100.0, tau_bits=6), clock=clock)
        v = Verdict(satisfied=True, confidence=0.9, evidence="under condition A")
        layer.store(CacheKey("g", _grid(), None, "e", condition="the door is OPEN"), v, 0.9)
        # same goal id + same entity + same frame, but the CONDITION changed -> must re-perceive.
        r = layer.lookup(CacheKey("g", _grid(), None, "e", condition="the door is CLOSED"))
        self.assertEqual(r.status, "DEFER", "a verdict cached under the old condition must not be served")
        # sanity: the SAME condition still hits (the cache still works for an unchanged goal).
        same = layer.lookup(CacheKey("g", _grid(), None, "e", condition="the door is OPEN"))
        self.assertEqual(same.status, "HIT")


class D3aAddGoalInvalidatesTests(unittest.TestCase):
    """Integration: ``add_goal`` (the arm() path) must invalidate the cache for that id."""

    def _core(self):
        cfg = dataclasses.replace(load_settings(), cache=CacheSettings(enabled=True, ttl_s=100.0, tau_bits=6))
        vision = FakeVision()
        return percept.Percept.create(vision=vision, settings=cfg, discover_plugins=False)

    def test_rearming_an_id_with_a_new_condition_forces_a_fresh_judge(self):
        async def go():
            core = self._core()
            core.add_goal(Goal(id="g", condition="the door is open", say="!"))
            await core.perceive_judged(FRAME, "camera", t=0.0)   # cold -> judge + store
            await core.perceive_judged(FRAME, "camera", t=1.0)   # warm -> HIT (cache works)
            hits_before = core.metrics.get("cache.hit")
            # Re-arm the SAME id with a DIFFERENT condition (voice re-arm). Must drop the stale entry.
            core.add_goal(Goal(id="g", condition="the door is closed", say="!"))
            await core.perceive_judged(FRAME, "camera", t=2.0)   # must be a FRESH read, not a HIT
            hits_after = core.metrics.get("cache.hit")
            return hits_before, hits_after

        hits_before, hits_after = asyncio.run(go())
        self.assertGreaterEqual(hits_before, 1, "sanity: the cache must actually be serving hits when enabled")
        self.assertEqual(hits_after, hits_before,
                         "re-arming with a new condition must NOT serve a cached verdict from the old one")


if __name__ == "__main__":
    unittest.main()
