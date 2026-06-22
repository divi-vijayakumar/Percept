"""Posture-B deliberative escalation — the consequence×novelty dial (L1, fake conductor).

Recurrent ambiguity is the trigger; consequence sets how soon; a per-concern cooldown is the
deliberative refractory. When it fires, the agent re-invokes the conductor to REFINE the program
instead of re-asking the human.
"""

import asyncio
import unittest

from percept.adapters import Box, Consequence, Decision, Goal, Observation, Verdict
from percept.agent import Percept
from percept.backends.fake import FakeConductor, FakeVision
from percept.behavior import ConcernContext
from percept.config import GateSettings, Settings
from percept.escalation import escalation_threshold, should_escalate
from percept.goals import GoalOp

SEVERE = Consequence(miss_cost="severe")
MILD = Consequence(miss_cost="mild")
TRIVIAL = Consequence(miss_cost="trivial")


class TestThresholdByStakes(unittest.TestCase):
    def test_higher_stakes_escalate_sooner(self):
        self.assertEqual(escalation_threshold(SEVERE), 2)  # severe: after 2 ambiguous reads
        self.assertEqual(escalation_threshold(MILD), 4)  # mild: after 4
        self.assertIsNone(escalation_threshold(TRIVIAL))  # trivial: never worth an LLM call


class TestShouldEscalate(unittest.TestCase):
    def _ctx(self, unknowns, last=float("-inf")):
        c = ConcernContext()
        c.confusion.consecutive_unknown = unknowns
        c.last_escalated_at = last
        return c

    def test_severe_escalates_at_its_threshold(self):
        self.assertFalse(should_escalate(self._ctx(1), SEVERE, 100.0, cooldown_s=30.0))
        self.assertTrue(should_escalate(self._ctx(2), SEVERE, 100.0, cooldown_s=30.0))

    def test_mild_needs_more_ambiguity(self):
        self.assertFalse(should_escalate(self._ctx(2), MILD, 100.0, cooldown_s=30.0))  # needs 4
        self.assertTrue(should_escalate(self._ctx(4), MILD, 100.0, cooldown_s=30.0))

    def test_trivial_never_escalates(self):
        self.assertFalse(should_escalate(self._ctx(10), TRIVIAL, 100.0, cooldown_s=30.0))

    def test_cooldown_is_the_deliberative_refractory(self):
        self.assertFalse(should_escalate(self._ctx(3, last=95.0), SEVERE, 100.0, cooldown_s=30.0))  # 5s ago
        self.assertTrue(should_escalate(self._ctx(3, last=60.0), SEVERE, 100.0, cooldown_s=30.0))  # 40s ago


class TestDeliberateWiring(unittest.TestCase):
    def _agent(self, decision):
        a = Percept.create(vision=FakeVision(script={}), discover_plugins=False)
        a.conductor = FakeConductor(decision)
        return a

    def _confused_severe(self, a, gid="g"):
        g = Goal(id=gid, condition="is that suspicious?", say="heads up")
        g.state["miss_cost"] = "severe"
        a.add_goal(g)
        a.executor.ctx(gid).confusion.consecutive_unknown = 3  # recurrent ambiguity
        return g

    def test_confused_severe_reinvokes_and_applies_refinement(self):
        refine = Decision(
            route="respond_direct",
            goal_ops=[GoalOp(op="create", goal=Goal(id="g-ref", condition="refined", say="s"))],
        )
        a = self._agent(refine)
        g = self._confused_severe(a)
        dec = asyncio.run(a.deliberate(g, t=100.0))
        self.assertIsNotNone(dec)
        self.assertEqual(a.metrics.get("deliberate.escalate"), 1)
        self.assertEqual(a.metrics.get("deliberate.refined"), 1)
        self.assertIsNotNone(a.goals.get("g-ref"))  # the LLM's refinement was applied

    def test_refractory_blocks_a_second_reinvocation(self):
        a = self._agent(Decision(route="ignore"))
        g = self._confused_severe(a)
        self.assertIsNotNone(asyncio.run(a.deliberate(g, t=100.0)))  # first escalates
        self.assertIsNone(asyncio.run(a.deliberate(g, t=110.0)))  # within cooldown -> blocked

    def test_mild_concern_below_threshold_does_not_escalate(self):
        a = self._agent(Decision(route="ignore"))
        g = Goal(id="g", condition="c", say="s")  # mild default
        a.add_goal(g)
        a.executor.ctx("g").confusion.consecutive_unknown = 3  # mild needs 4
        self.assertIsNone(asyncio.run(a.deliberate(g, t=100.0)))
        self.assertEqual(a.metrics.get("deliberate.escalate"), 0)


class TestAutoDeliberateInPerceiveLoop(unittest.TestCase):
    def test_recurrent_ambiguity_climbs_perception_then_the_llm(self):
        # The escalation LADDER: a severe concern that keeps reading AMBIGUOUS first escalates PERCEPTION
        # (cheap, better-looking: single->strip->+audio) and ONLY once that is maxed re-invokes the LLM
        # (more-thinking). No explicit deliberate() call — both rungs fire in-loop.
        refine = Decision(
            route="respond_direct",
            goal_ops=[GoalOp(op="create", goal=Goal(id="g-refined", condition="x", say="s"))],
        )
        vision = FakeVision(
            script={"f": Verdict(True, 0.4, "unsure")},  # always ambiguous -> confusion never resolves
            detect_script={"f": [Observation("x", Box(0.4, 0.4, 0.2, 0.2), 0.9)]},
        )
        a = Percept.create(
            vision=vision, settings=Settings(gate=GateSettings(threshold=0.7)), discover_plugins=False
        )
        a.conductor = FakeConductor(refine)
        a.perception_cooldown_s = 0.0  # let the ladder climb every tick for the test
        g = Goal(id="g", condition="is that suspicious?", say="heads up")
        g.state["miss_cost"] = "severe"  # severe -> threshold of 2 ambiguous reads
        a.add_goal(g)

        async def run():
            for t in (0.0, 1.0, 2.0, 3.0):
                await a.perceive_judged("f", t=t)

        asyncio.run(run())
        # perception escalated TWICE (single->strip->+audio) before the LLM, and is now maxed ...
        self.assertEqual(a.metrics.get("perception.escalate"), 2)
        self.assertEqual(a.executor.ctx("g").perception_level, 2)
        # ... then, perception maxed and STILL ambiguous, the LLM rung fired once and was applied in-loop
        self.assertEqual(a.metrics.get("deliberate.escalate"), 1)
        self.assertIsNotNone(a.goals.get("g-refined"))

    def test_confident_read_decays_the_perception_ladder(self):
        # once a confident read resolves the ambiguity, the rung decays back to the cheap single-frame default
        vision = FakeVision(
            script={"hard": Verdict(True, 0.4, "unsure"), "easy": Verdict(True, 0.95, "yes")},
            detect_script={"hard": [Observation("x", Box(0.4, 0.4, 0.2, 0.2), 0.9)]},
        )
        a = Percept.create(
            vision=vision, settings=Settings(gate=GateSettings(threshold=0.7)), discover_plugins=False
        )
        a.perception_cooldown_s = 0.0
        g = Goal(id="g", condition="is that suspicious?", say="heads up")
        g.state["miss_cost"] = "severe"
        a.add_goal(g)

        async def run():
            for t in (0.0, 1.0, 2.0):  # ambiguous -> climbs the ladder
                await a.perceive_judged("hard", t=t)
            await a.perceive_judged("easy", t=3.0)  # confident read -> decays

        asyncio.run(run())
        self.assertEqual(a.executor.ctx("g").perception_level, 0)  # back to single-frame

    def test_calm_concern_never_triggers_a_conductor_call(self):
        # A mild, confident concern perceiving normally must NOT trigger deliberation (byte-for-byte).
        vision = FakeVision(
            script={"f": Verdict(True, 0.95, "yes")},
            detect_script={"f": [Observation("x", Box(0.4, 0.4, 0.2, 0.2), 0.9)]},
        )
        a = Percept.create(vision=vision, discover_plugins=False)
        a.conductor = FakeConductor(Decision(route="respond_direct", goal_ops=[
            GoalOp(op="create", goal=Goal(id="should-not-exist", condition="x", say="s"))]))
        a.add_goal(Goal(id="g", condition="the dog is here", say="dog"))
        asyncio.run(a.perceive_judged("f", t=0.0))
        self.assertEqual(a.metrics.get("deliberate.escalate"), 0)  # no escalation
        self.assertIsNone(a.goals.get("should-not-exist"))  # conductor never invoked


if __name__ == "__main__":
    unittest.main()
