"""L1: state-transition goals — fire on A->B only after A was witnessed (the wake-up fix).

Uses a stub vision that scripts distinct (from, to) verdicts per frame — exercising OUR transition
logic, not model intelligence.
"""

import asyncio
import unittest

from percept.adapters import Goal, Verdict
from percept.agent import Percept
from percept.gate import FIRE

FROM = "the user is asleep"
TO = "the user is awake and moving"


class StubVision:
    """Maps (condition, frame_token) -> (satisfied, confidence). Detect yields no entity."""

    def __init__(self, script):
        self._script = script

    async def judge(self, condition, frame, *, roi=None, exemplar=None, corrections=None):
        sat, conf = self._script.get((condition, str(frame)), (False, 0.9))
        return Verdict(satisfied=sat, confidence=conf)

    async def detect(self, frame, goals):
        return []

    async def detect_events(self, frames, goals):
        return []


def _wake_goal():
    return Goal(
        id="wake",
        condition="when I wake up",
        say="good morning",
        state={"type": "transition", "from": FROM, "to": TO, "count": 0},
    )


def _run(frames):
    """frames: list of (token, (from_sat, from_conf), (to_sat, to_conf)). Returns (fires, agent)."""
    script = {}
    for tok, fv, tv in frames:
        script[(FROM, tok)] = fv
        script[(TO, tok)] = tv
    a = Percept.create(vision=StubVision(script), discover_plugins=False)
    a.add_goal(_wake_goal())
    fires = []

    async def go():
        for tok, _, _ in frames:
            evs = await a.perceive_judged(tok, t=0.0)
            fires.extend(e for e in evs if e.action == FIRE)

    asyncio.run(go())
    return fires, a


class TestTransition(unittest.TestCase):
    def test_fires_on_from_then_to(self):
        fires, a = _run(
            [
                ("f1", (True, 0.9), (False, 0.9)),  # asleep
                ("f2", (True, 0.9), (False, 0.9)),  # asleep
                ("f3", (False, 0.9), (True, 0.9)),  # awake -> the transition
            ]
        )
        self.assertEqual(len(fires), 1)
        self.assertEqual(a.goals.get("wake").state["count"], 1)

    def test_no_fire_if_from_never_seen(self):
        # the eval4_s04 case: awake from the start, never observed asleep -> never fires
        fires, _ = _run([(f"f{i}", (False, 0.95), (True, 0.95)) for i in range(5)])
        self.assertEqual(fires, [])

    def test_refractory_single_fire_while_to_persists(self):
        fires, _ = _run(
            [
                ("f1", (True, 0.9), (False, 0.9)),
                ("f2", (False, 0.9), (True, 0.9)),  # fire
                ("f3", (False, 0.9), (True, 0.9)),  # still awake -> no second fire
            ]
        )
        self.assertEqual(len(fires), 1)

    def test_rearm_on_return_to_from(self):
        # asleep->awake (fire), back asleep, awake again -> a second genuine transition fires
        fires, _ = _run(
            [
                ("f1", (True, 0.9), (False, 0.9)),
                ("f2", (False, 0.9), (True, 0.9)),  # fire 1
                ("f3", (True, 0.9), (False, 0.9)),  # back to 'from' -> re-arm
                ("f4", (False, 0.9), (True, 0.9)),  # fire 2
            ]
        )
        self.assertEqual(len(fires), 2)

    def test_low_confidence_to_does_not_fire(self):
        fires, _ = _run(
            [
                ("f1", (True, 0.9), (False, 0.9)),  # asleep, arm
                ("f2", (False, 0.9), (True, 0.5)),  # "awake" but low conf -> refuse to guess
            ]
        )
        self.assertEqual(fires, [])


if __name__ == "__main__":
    unittest.main()
