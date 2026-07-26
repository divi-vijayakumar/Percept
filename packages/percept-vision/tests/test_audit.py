"""L1: the false-serve audit hook and cache-enabled wiring.

False-serve is the cache's analog of false-fire: on a cache HIT, re-perceive an ``audit_rate``
fraction through real vision and compare; a divergence is a **false-serve** and the fresh read
wins. The audit decision is **deterministic** (a counter, no RNG) so this lane is reproducible.

Frames here are synthetic luminance grids so the ``FrameHashLayer`` can actually hash + near-match
them; the vision backend is a counting stub (never asserts intelligence — it returns scripted
verdicts and records its call count so we can prove the VLM was skipped on a HIT).
"""

import asyncio
import unittest

from percept.adapters import Box, Goal, Observation, Verdict
from percept.agent import DeterministicAuditor, Percept
from percept.config import CacheSettings, GateSettings, Settings


def grid(level: float, n: int = 10) -> list[list[float]]:
    """A flat-ish ramp grid; distinct ``level`` shifts brightness but not structure."""
    return [[float(x) + level for x in range(n)] for _ in range(n)]


# one stable entity in the centre across the whole replay -> a single cache slot.
BOX = Box(0.3, 0.3, 0.4, 0.4)


class CountingVision:
    """A stub ``Vision`` that records ``judge`` calls and returns a scripted verdict.

    Not an intelligence test (rule 2): it maps a fixed verdict and counts invocations so a test
    can assert the cache skipped the VLM on a HIT and that the auditor re-invoked it on the
    audited fraction.
    """

    def __init__(self, verdict: Verdict, *, flip_after: int | None = None,
                 flipped: Verdict | None = None) -> None:
        self._verdict = verdict
        self._flip_after = flip_after
        self._flipped = flipped
        self.calls = 0

    async def judge(self, condition, frame, *, roi=None, exemplar=None, corrections=None):
        self.calls += 1
        if self._flip_after is not None and self.calls > self._flip_after:
            return self._flipped
        return self._verdict

    async def detect(self, frame, goals):
        return [Observation("cup", BOX, 0.9)]

    async def detect_events(self, frames, goals):
        return []


def build(vision, *, audit=None, audit_rate=0.0, ttl_s=100.0):
    settings = Settings(
        gate=GateSettings(threshold=0.5, validity_window_s=100.0),
        cache=CacheSettings(enabled=True, threshold=0.0, ttl_s=ttl_s, tau_bits=6,
                            audit_rate=audit_rate),
    )
    a = Percept.create(vision=vision, settings=settings, discover_plugins=False,
                       clock=lambda: build.t, audit=audit)
    a.add_goal(Goal(id="g1", condition="the user is drinking coffee", say="nudge"))
    return a


build.t = 0.0  # shared hand-cranked clock


class TestCacheWiring(unittest.TestCase):
    def setUp(self):
        build.t = 0.0

    def test_cache_disabled_by_default(self):
        a = Percept.create(vision=CountingVision(Verdict(True, 0.9)), discover_plugins=False)
        self.assertEqual(a.cache.layers, [])  # empty cascade -> always DEFER (contract, not policy)

    def test_cache_enabled_inserts_framehash_layer(self):
        a = build(CountingVision(Verdict(True, 0.9)))
        self.assertEqual([type(layer).__name__ for layer in a.cache.layers], ["FrameHashLayer"])

    def test_hit_skips_the_vlm(self):
        async def run():
            vision = CountingVision(Verdict(False, 0.9, "no coffee"))  # silent verdict, no fire
            a = build(vision, audit_rate=0.0)
            build.t = 0.0
            await a.perceive_judged(grid(0.0), t=0.0)
            calls_after_first = vision.calls
            # an identical frame within TTL -> cache HIT -> the VLM must NOT be called again.
            build.t = 1.0
            await a.perceive_judged(grid(0.0), t=1.0)
            self.assertEqual(vision.calls, calls_after_first)  # skipped
            self.assertEqual(a.metrics.get("cache.hit"), 1)

        asyncio.run(run())


class TestDeterministicAuditor(unittest.TestCase):
    def test_zero_rate_never_audits(self):
        aud = DeterministicAuditor(0.0)
        self.assertFalse(any(aud("k") for _ in range(20)))

    def test_full_rate_audits_every_hit(self):
        aud = DeterministicAuditor(1.0)
        self.assertTrue(all(aud("k") for _ in range(20)))

    def test_fractional_rate_is_stratified_and_deterministic(self):
        # rate 0.25 -> ~1 in 4; deterministic across runs (no RNG).
        aud1 = DeterministicAuditor(0.25)
        aud2 = DeterministicAuditor(0.25)
        seq1 = [aud1("k") for _ in range(8)]
        seq2 = [aud2("k") for _ in range(8)]
        self.assertEqual(seq1, seq2)  # reproducible
        self.assertEqual(sum(seq1), 2)  # 8 * 0.25 = 2 audits


class TestFalseServe(unittest.TestCase):
    def setUp(self):
        build.t = 0.0

    def test_audit_reperceives_and_agreeing_read_is_no_false_serve(self):
        async def run():
            vision = CountingVision(Verdict(False, 0.9, "no coffee"))
            a = build(vision, audit_rate=1.0)  # audit every HIT
            build.t = 0.0
            await a.perceive_judged(grid(0.0), t=0.0)
            calls = vision.calls
            build.t = 1.0
            await a.perceive_judged(grid(0.0), t=1.0)  # HIT -> audited (re-judged), agrees
            self.assertEqual(vision.calls, calls + 1)  # the audit re-perceived
            self.assertEqual(a.metrics.get("cache.audit"), 1)
            self.assertEqual(a.metrics.get("cache.false_serve"), 0)

        asyncio.run(run())

    def test_divergence_is_a_false_serve_and_fresh_read_wins(self):
        async def run():
            # first judge says NOT coffee (cached), then the audit re-judge flips to coffee.
            vision = CountingVision(
                Verdict(False, 0.9, "no coffee"),
                flip_after=1,
                flipped=Verdict(True, 0.9, "now it's coffee"),
            )
            a = build(vision, audit_rate=1.0)
            build.t = 0.0
            ev0 = await a.perceive_judged(grid(0.0), t=0.0)  # silent (not coffee)
            self.assertEqual(ev0, [])
            build.t = 1.0
            ev1 = await a.perceive_judged(grid(0.0), t=1.0)  # HIT -> audit flips -> false-serve
            self.assertEqual(a.metrics.get("cache.false_serve"), 1)
            # the fresh read won: the corrected verdict (coffee) flows to the gate -> a FIRE.
            self.assertEqual([e.action for e in ev1], ["fire"])

        asyncio.run(run())


if __name__ == "__main__":
    unittest.main()
