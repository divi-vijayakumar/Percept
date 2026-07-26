"""L1 integration: the caffeine-restraint trace end-to-end on fakes.

Goal: "keep me off caffeine — nudge me when I drink coffee." This exercises the whole Phase-1
firing path — detect -> entity continuity -> grounded judge (with corrections) -> three-state
gate -> rising-edge fire — and proves the trust property: the system **refuses to guess** and
**never confidently-wrong fires**.

| t  | event            | expected gate action                         |
|----|------------------|----------------------------------------------|
| t1 | cup1, unsure     | ASK ("is that coffee?")                       |
|    | user: "tea"      | assert contents=tea on cup1                   |
| t2 | cup1 again       | SILENT (known-false via the asserted fact)    |
| t3 | cup1, fact stale | ASK ("still tea?")                            |
| t4 | cup2 (new cup)   | FIRE ("a new cup — stay off caffeine")        |
| t4'| cup2 again       | SILENT (refractory — no double-fire)          |
"""

import asyncio
import unittest

from percept.adapters import Box, Goal, Observation, Verdict
from percept.agent import Percept
from percept.backends.fake import FakeVision
from percept.config import GateSettings, Settings
from percept.gate import ASK, FIRE

CUP1 = Box(0.40, 0.40, 0.20, 0.20)
CUP2 = Box(0.10, 0.10, 0.20, 0.20)  # disjoint from CUP1 -> a distinct entity

JUDGE_SCRIPT = {
    "cup1-unsure": Verdict(True, 0.4, "looks like coffee, unsure"),
    "cup1": Verdict(True, 0.9, "looks like coffee"),
    "cup1|corrected": Verdict(False, 0.9, "user said it's tea"),  # correction flows in
    "cup2": Verdict(True, 0.9, "a fresh cup of coffee"),
}
DETECT_SCRIPT = {
    "cup1-unsure": [Observation("cup", CUP1, 0.9)],
    "cup1": [Observation("cup", CUP1, 0.9)],
    "cup2": [Observation("cup", CUP2, 0.9)],
}


def build():
    vision = FakeVision(script=JUDGE_SCRIPT, detect_script=DETECT_SCRIPT)
    settings = Settings(gate=GateSettings(threshold=0.7, validity_window_s=10.0))
    a = Percept.create(vision=vision, settings=settings, discover_plugins=False)
    a.add_goal(
        Goal(id="g1", condition="the user is drinking coffee", say="Heads up — stay off caffeine.")
    )
    return a


class TestCaffeineTrace(unittest.TestCase):
    def test_full_trace(self):
        async def run():
            a = build()

            # t1 — unsure -> ASK, then the user corrects it to tea
            ev = await a.perceive_judged("cup1-unsure", t=1.0)
            self.assertEqual([e.action for e in ev], [ASK])
            a.assert_attribute("contents", "tea", t=1.0)

            # t2 — same cup, now known-false via the asserted fact -> SILENT
            ev = await a.perceive_judged("cup1", t=2.0)
            self.assertEqual(ev, [])

            # t3 — the asserted fact has gone stale (age 20 > validity 10) -> ASK again
            ev = await a.perceive_judged("cup1", t=20.0)
            self.assertEqual([e.action for e in ev], [ASK])

            # t4 — a genuinely new cup -> FIRE the nudge (rising edge on a fresh entity)
            ev = await a.perceive_judged("cup2", t=21.0)
            self.assertEqual([e.action for e in ev], [FIRE])
            self.assertEqual(ev[0].text, "Heads up — stay off caffeine.")

            # t4' — same new cup again -> refractory, no double-fire
            ev = await a.perceive_judged("cup2", t=22.0)
            self.assertEqual(ev, [])

        asyncio.run(run())

    def test_never_confidently_wrong_fires(self):
        # The trust gate: across the whole trace, the only FIRE is the genuine new-cup event.
        async def run():
            a = build()
            fires = 0
            await a.perceive_judged("cup1-unsure", t=1.0)
            a.assert_attribute("contents", "tea", t=1.0)
            for frame, t in [("cup1", 2.0), ("cup1", 20.0)]:
                fires += sum(e.action == FIRE for e in await a.perceive_judged(frame, t=t))
            self.assertEqual(fires, 0)  # tea never fires the caffeine nudge

        asyncio.run(run())


if __name__ == "__main__":
    unittest.main()
