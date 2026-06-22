"""Held-out NOVEL-TASK eval — the generalization regression target (L1, end-to-end).

Each task below is one no code path was written for. It is authored exactly as the conductor would —
a natural-language request -> a guarded ``behavior`` dict run through the REAL coerce_authored_behavior
-> goal.state -> the unchanged dispatch -> the one executor. The assertion is the user-facing fire
pattern. Passing this suite is the operational definition of "as close as possible to an LLM": diverse
novel tasks handled by a general loop with ZERO new code per task. If a new task shape needs code, it
fails here first.
"""

import asyncio
import unittest

from percept.adapters import Box, Goal, Observation, Verdict
from percept.agent import Percept
from percept.behavior import coerce_authored_behavior
from percept.config import GateSettings, Settings
from percept.gate import FIRE

YES = Verdict(True, 0.95, "yes")
NO = Verdict(False, 0.95, "no")
BOX = Box(0.4, 0.4, 0.2, 0.2)  # one stable subject -> one gate key (rising-edge distinctness)


class _EvalVision:
    """Condition-aware fake (keys by frame AND condition) so transition's two conditions/frame work."""

    def __init__(self, judge_fn, detect):
        self._judge, self._detect = judge_fn, detect

    async def judge(self, condition, frame, *, roi=None, exemplar=None, corrections=None):
        return self._judge(str(frame), condition)

    async def detect(self, frame, goals):
        return list(self._detect.get(str(frame), []))


def _agent(judge_fn, detect):
    return Percept.create(
        vision=_EvalVision(judge_fn, detect),
        settings=Settings(gate=GateSettings(threshold=0.7)),
        discover_plugins=False,
    )


def _author(goal, nl_behavior: dict):
    """Simulate the conductor: a behavior dict -> guarded goal.state (the REAL coercion firewall)."""
    goal.state.update(coerce_authored_behavior(nl_behavior))
    return goal


def _drive(agent, frames, times=None):
    times = times or [float(i) for i in range(len(frames))]

    async def run():
        return [[e.action for e in await agent.perceive_judged(f, t=t)] for f, t in zip(frames, times)]

    return asyncio.run(run())


class TestNovelTasks(unittest.TestCase):
    def test_alert_me_the_2nd_distinct_time(self):
        # NL: "alert me the 2nd time the cat gets on the table"  -> watch + accumulate(n==2)
        detect = {"on": [Observation("cat", BOX, 0.95)], "off": [Observation("cat", BOX, 0.95)]}
        a = _agent(lambda f, c: YES if f == "on" else NO, detect)
        a.add_goal(_author(Goal(id="g", condition="the cat is on the table", say="2nd time!"),
                           {"accumulate": {"fire_when": "n == 2"}}))
        fires = _drive(a, ["on", "off", "on", "off", "on"])  # rising edges at 0,2,4
        self.assertEqual(fires, [[], [], [FIRE], [], []])  # only the 2nd distinct time alerts

    def test_tell_me_once_then_never(self):
        # NL: "tell me once when the door opens, then stop"  -> watch + rearm=never
        detect = {"open": [Observation("door", BOX, 0.95)], "shut": [Observation("door", BOX, 0.95)]}
        a = _agent(lambda f, c: YES if f == "open" else NO, detect)
        a.add_goal(_author(Goal(id="g", condition="the door is open", say="door opened"),
                           {"rearm": "never"}))
        fires = _drive(a, ["open", "shut", "open"])
        self.assertEqual(fires, [[FIRE], [], []])  # fires once, never re-arms

    def test_remind_me_every_10_seconds(self):
        # NL: "remind me every 10 seconds while I'm at my desk"  -> watch + rearm=interval(10)
        a = _agent(lambda f, c: YES, {"desk": [Observation("me", BOX, 0.95)]})
        a.add_goal(_author(Goal(id="g", condition="I am at my desk", say="stretch break"),
                           {"rearm": "interval", "cadence_s": 10}))
        fires = _drive(a, ["desk", "desk", "desk"], times=[0.0, 5.0, 10.0])
        self.assertEqual(fires, [[FIRE], [], [FIRE]])  # one per 10s bucket, silent within a bucket

    def test_warn_me_if_stove_not_off_in_time(self):
        # NL: "warn me if the stove isn't turned off within 30 seconds"  -> absence(deadline) + severe
        a = _agent(lambda f, c: YES, {})
        a.add_goal(_author(Goal(id="g", condition="the stove is off", say="stove still on!"),
                           {"rearm": "deadline", "deadline_s": 30, "consequence": {"miss_cost": "severe"}}))
        self.assertEqual(a.check_deadlines(t=10.0), [])  # within the deadline -> waiting
        self.assertEqual([e.action for e in a.check_deadlines(t=31.0)], [FIRE])  # overdue -> warn

    def test_tell_me_when_the_baby_wakes_up(self):
        # NL: "tell me when the baby wakes up"  -> transition(asleep -> awake)
        def judge(f, c):
            asleep = f == "asleep"
            return (YES if asleep else NO) if "asleep" in c else (NO if asleep else YES)

        detect = {"asleep": [Observation("baby", BOX, 0.95)], "awake": [Observation("baby", BOX, 0.95)]}
        a = _agent(judge, detect)
        a.add_goal(_author(Goal(id="g", condition="the baby is awake", say="baby's awake"),
                           {"type": "transition", "from": "the baby is asleep", "to": "the baby is awake"}))
        fires = _drive(a, ["awake", "asleep", "awake"])  # join in awake (no fire), see asleep, then wake
        self.assertEqual(fires, [[], [], [FIRE]])  # fires only on awake AFTER asleep was witnessed


if __name__ == "__main__":
    unittest.main()
