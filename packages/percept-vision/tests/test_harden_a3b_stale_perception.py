"""A-3b: the STALENESS arm of the three-state gate — perception recency feeds freshness.

Invariant 2 is "low confidence OR stale freshness -> ASK". Before this change ``_freshness`` derived
staleness ONLY from user-asserted attributes (``a.source == "asserted"``); PERCEPTION recency never fed
it. For any goal a human never manually corrected — essentially all of them — freshness was permanently
1.0, so ``classify``'s stale->ASK arm could never trigger from perception going stale. Half of invariant
2 was dead.

The fix adds a perception-recency term to ``_freshness`` behind ``GateSettings.stale_ask`` (default OFF):
a goal not confidently read within ``validity_window_s`` contributes freshness 0.0 -> ASK.

This is the ONLY change in the hardening pass that alters WHAT FIRES, so it lands INERT:
* OFF (default): byte-for-byte the asserted-only path (the full suite is unchanged — proved elsewhere).
* ON: the perception term is active.

These tests fail on the pre-fix tree because the perception term does not exist (freshness stays 1.0
regardless of ``last_supported`` age, so the stale reads below classify as "yes"/FIRE, never "unknown").
"""

import asyncio
import unittest

from percept.adapters import Box, Goal, Observation, Verdict
from percept.agent import Percept
from percept.backends.fake import FakeVision
from percept.config import GateSettings, Settings
from percept.gate import ASK, FIRE, classify

CUP = Box(0.40, 0.40, 0.20, 0.20)
JUDGE_SCRIPT = {"cup": Verdict(True, 0.9, "a fresh cup of coffee")}
DETECT_SCRIPT = {"cup": [Observation("cup", CUP, 0.9)]}
WINDOW = 10.0


def build(stale_ask: bool) -> Percept:
    vision = FakeVision(script=JUDGE_SCRIPT, detect_script=DETECT_SCRIPT)
    settings = Settings(
        gate=GateSettings(threshold=0.7, validity_window_s=WINDOW, stale_ask=stale_ask)
    )
    a = Percept.create(vision=vision, settings=settings, discover_plugins=False)
    a.add_goal(Goal(id="g1", condition="the user is drinking coffee", say="Stay off caffeine."))
    return a


class TestStalePerceptionFreshness(unittest.TestCase):
    def test_default_setting_is_off(self):
        """Provable inertness at the source: the switch ships OFF."""
        self.assertFalse(GateSettings().stale_ask)

    def test_off_freshness_is_asserted_only(self):
        """OFF: a long-idle goal (last_supported way in the past) is still freshness 1.0 — byte-for-byte
        the asserted-only path. No perception term consulted."""
        a = build(stale_ask=False)
        a.last_supported["g1"] = 0.0  # confidently read long ago
        # focus is None here (no graph ingest) -> asserted term is 1.0; OFF must ignore perception entirely
        self.assertEqual(a._freshness(None, now=10_000.0, goal_id="g1"), 1.0)

    def test_on_stale_perception_is_stale(self):
        """ON: a goal read at t, then not read for > validity_window, is freshness 0.0 -> classify ASK.

        Fails pre-fix: without the perception term _freshness returns 1.0, so classify(...) == 'yes'."""
        a = build(stale_ask=True)
        a.last_supported["g1"] = 100.0
        fresh = a._freshness(None, now=100.0 + WINDOW + 1.0, goal_id="g1")
        self.assertEqual(fresh, 0.0)
        self.assertEqual(classify(JUDGE_SCRIPT["cup"], fresh, 0.7, 0.5), "unknown")

    def test_on_recent_perception_stays_fresh(self):
        """ON: a goal read WITHIN the window stays freshness 1.0 -> no spurious ASK."""
        a = build(stale_ask=True)
        a.last_supported["g1"] = 100.0
        fresh = a._freshness(None, now=100.0 + WINDOW - 1.0, goal_id="g1")
        self.assertEqual(fresh, 1.0)
        self.assertEqual(classify(JUDGE_SCRIPT["cup"], fresh, 0.7, 0.5), "yes")

    def test_on_first_read_is_fresh(self):
        """ON: a goal with NO prior confident read (missing last_supported) is fresh — classify runs
        before the stamp, so the current judgment IS the fresh perception; a first sighting must not ASK."""
        a = build(stale_ask=True)
        self.assertEqual(a._freshness(None, now=5.0, goal_id="g1"), 1.0)

    def test_on_end_to_end_stale_gap_asks(self):
        """ON, end-to-end through perceive_judged: fire on first sight, then a > window gap with no
        perception makes the next confident read classify stale -> ASK instead of SILENT (refractory).

        Pre-fix this second read is SILENT (edge already fired, freshness stuck at 1.0)."""
        async def run():
            a = build(stale_ask=True)
            ev = await a.perceive_judged("cup", t=1.0)
            self.assertEqual([e.action for e in ev], [FIRE])  # rising edge on first sight
            # No perception for > validity_window (last_supported stamped at t=1.0). Next read is stale.
            ev = await a.perceive_judged("cup", t=1.0 + WINDOW + 5.0)
            self.assertEqual([e.action for e in ev], [ASK])  # refuse to trust a long-unseen goal

        asyncio.run(run())

    def test_off_end_to_end_no_spurious_ask(self):
        """OFF, same trace: the second (long-gap) read stays SILENT (refractory) — perception recency is
        never consulted, so the pre-fix disposition is preserved exactly."""
        async def run():
            a = build(stale_ask=False)
            ev = await a.perceive_judged("cup", t=1.0)
            self.assertEqual([e.action for e in ev], [FIRE])
            ev = await a.perceive_judged("cup", t=1.0 + WINDOW + 5.0)
            self.assertEqual(ev, [])  # SILENT — refractory, no stale-ASK

        asyncio.run(run())


if __name__ == "__main__":
    unittest.main()
