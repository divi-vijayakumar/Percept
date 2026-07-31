"""DirectJudge — route a condition to a cheap answer, without ever becoming a second gate.

All offline (scripted vision that RAISES if called proves the confident path spends zero vision
calls — standing rule 2: wiring, not intelligence). Covers:

* :class:`~percept.direct_judge.DirectRouting` matching by ``condition_namespace`` / ``role``.
* confident -> the judge is never called; the SAME unmodified Gate still classifies the verdict.
* uncertain -> facts merge into the judge's context under the extractor reserved key, and the
  normal judge path still runs.
* degraded (no rule matches, or a rule raises) -> falls through to the normal judge path,
  completely unchanged.
* stakes-derived verify cadence forces a real judge call every N confident answers.
"""

from __future__ import annotations

import asyncio
import unittest

import pytest

from percept.agent import Percept
from percept.contracts import Goal, Verdict
from percept.direct_judge import (
    DirectJudge,
    DirectResult,
    DirectRouting,
)
from percept.extraction import CONTEXT_KEY
from percept.harness.scripted import ScriptedJudge


class _RaisingVision:
    """A Vision double that fails the test the moment it is called — the zero-vision-call proof."""

    def __init__(self) -> None:
        self.calls = 0

    async def judge(self, *args, **kwargs):
        self.calls += 1
        raise AssertionError("vision.judge must not be called on the confident DirectJudge path")

    async def detect(self, frame, goals):
        return []

    async def detect_events(self, frames, goals):
        return []


class _ConfidentRule:
    applies_to = DirectRouting(condition_namespace="screen.ocr.")

    def evaluate(self, condition, frame, roi, facts):
        return DirectResult(outcome="confident", verdict={"satisfied": True, "confidence": 0.97, "evidence": "direct"})


class _UncertainRule:
    applies_to = DirectRouting(condition_namespace="screen.ocr.")

    def evaluate(self, condition, frame, roi, facts):
        return DirectResult(outcome="uncertain", facts={"partial": "reading"})


class _RaisingRule:
    applies_to = DirectRouting(condition_namespace="screen.ocr.")

    def evaluate(self, condition, frame, roi, facts):
        raise RuntimeError("boom")


def _drive(agent: Percept, source: str, n: int) -> list:
    async def go():
        out = []
        for i in range(n):
            out += await agent.perceive_judged(source, t=float(i + 1))
        return out

    return asyncio.run(go())


class DirectRoutingTests(unittest.TestCase):
    def test_condition_namespace_prefix_match(self):
        routing = DirectRouting(condition_namespace="screen.ocr.")
        self.assertTrue(routing.matches(condition="screen.ocr.total_exceeds"))
        self.assertFalse(routing.matches(condition="camera.motion"))

    def test_role_match(self):
        routing = DirectRouting(role="detector")
        self.assertTrue(routing.matches(condition="anything", role="detector"))
        self.assertFalse(routing.matches(condition="anything", role="backend"))

    def test_unset_axes_match_anything(self):
        routing = DirectRouting()
        self.assertTrue(routing.matches(condition="anything", role="anything"))


class DirectResultTests(unittest.TestCase):
    def test_confident_requires_a_verdict(self):
        with pytest.raises(ValueError):
            DirectResult(outcome="confident")

    def test_confident_normalizes_a_dict_verdict(self):
        r = DirectResult(outcome="confident", verdict={"satisfied": True, "confidence": 0.5})
        self.assertIsInstance(r.verdict, Verdict)

    def test_unknown_outcome_rejected(self):
        with pytest.raises(ValueError):
            DirectResult(outcome="sure-why-not")


class DirectHitCountsAsJudgedTests(unittest.TestCase):
    """A confident direct hit is a FRESH answer for this frame, so the fold must count it as judged.

    ``_observation_was_judged`` gates whether ``Episode.judged_observations`` advances, and its
    docstring names this exact case: "a caller that reached a fold without going through ``_judge``,
    e.g. a DirectJudge read - counts as judged". The flag it reads is written by ``_judge``, which a
    confident direct hit skips entirely, so on the inline path the goal used to keep whatever the
    PREVIOUS tick left behind. After a cache serve that value is True, and a fresh rule answer was
    then folded as "not judged".

    The split path has always set the flag unconditionally before its fold. These two paths are meant
    to be twins; this pins that they agree.
    """

    def test_a_direct_hit_is_judged_even_after_a_cache_serve_left_the_flag_set(self):
        vision = _RaisingVision()
        dj = DirectJudge([_ConfidentRule()])
        core = Percept.create(vision=vision, direct_judge=dj, discover_plugins=False)
        core.add_goal(Goal(id="g1", condition="screen.ocr.total_exceeds", say="alert"))
        # Stand in for the state a cache serve leaves behind on the previous tick.
        core._verdict_from_cache["g1"] = True

        _drive(core, "frame1", 1)

        self.assertIs(
            core._observation_was_judged("g1"), True,
            "a confident direct hit formed a fresh answer for this frame and must fold as judged",
        )
        self.assertEqual(vision.calls, 0, "the premise: no provider was consulted")

    def test_the_flag_is_cleared_on_every_direct_hit_not_just_the_first(self):
        vision = _RaisingVision()
        dj = DirectJudge([_ConfidentRule()])
        core = Percept.create(vision=vision, direct_judge=dj, discover_plugins=False)
        core.add_goal(Goal(id="g1", condition="screen.ocr.total_exceeds", say="alert"))

        for tick in range(3):
            core._verdict_from_cache["g1"] = True  # re-dirty it between ticks
            _drive(core, f"frame{tick}", 1)
            with self.subTest(tick=tick):
                self.assertIs(core._observation_was_judged("g1"), True)


class DirectJudgeRoutingTests(unittest.TestCase):
    def test_route_confident_zero_vision_calls(self):
        vision = _RaisingVision()
        dj = DirectJudge([_ConfidentRule()])
        core = Percept.create(vision=vision, direct_judge=dj, discover_plugins=False)
        core.add_goal(Goal(id="g1", condition="screen.ocr.total_exceeds", say="alert"))

        events = _drive(core, "frame1", 1)

        self.assertEqual(vision.calls, 0)
        fires = [e for e in events if e.action == "fire"]
        self.assertEqual(len(fires), 1)
        self.assertEqual(fires[0].verdict.confidence, 0.97)

    def test_route_uncertain_merges_facts_and_still_judges(self):
        judge = ScriptedJudge(default_verdict=Verdict(satisfied=True, confidence=0.8, evidence="ok"),
                              on_exhausted="repeat")
        dj = DirectJudge([_UncertainRule()])
        core = Percept.create(vision=judge, direct_judge=dj, discover_plugins=False)
        core.add_goal(Goal(id="g1", condition="screen.ocr.total_exceeds", say="alert"))

        _drive(core, "frame1", 1)

        self.assertEqual(len(judge.calls), 1)
        self.assertEqual(judge.calls[0]["context"][CONTEXT_KEY]["facts"], {"partial": "reading"})

    def test_route_degraded_when_no_rule_matches(self):
        judge = ScriptedJudge(default_verdict=Verdict(satisfied=True, confidence=0.8, evidence="ok"),
                              on_exhausted="repeat")
        dj = DirectJudge([_ConfidentRule()])  # only matches "screen.ocr." namespace
        core = Percept.create(vision=judge, direct_judge=dj, discover_plugins=False)
        core.add_goal(Goal(id="g1", condition="camera.motion", say="alert"))

        _drive(core, "frame1", 1)

        self.assertEqual(len(judge.calls), 1)  # fell through to the normal judge path

    def test_route_degraded_when_rule_raises(self):
        judge = ScriptedJudge(default_verdict=Verdict(satisfied=True, confidence=0.8, evidence="ok"),
                              on_exhausted="repeat")
        dj = DirectJudge([_RaisingRule()])
        core = Percept.create(vision=judge, direct_judge=dj, discover_plugins=False)
        core.add_goal(Goal(id="g1", condition="screen.ocr.total_exceeds", say="alert"))

        _drive(core, "frame1", 1)

        self.assertEqual(len(judge.calls), 1)  # the raising rule degraded, never crashed the tick

    def test_no_direct_judge_configured_is_byte_for_byte(self):
        judge = ScriptedJudge(default_verdict=Verdict(satisfied=True, confidence=0.8, evidence="ok"),
                              on_exhausted="repeat")
        core = Percept.create(vision=judge, discover_plugins=False)
        core.add_goal(Goal(id="g1", condition="screen.ocr.total_exceeds", say="alert"))

        _drive(core, "frame1", 1)

        self.assertEqual(len(judge.calls), 1)


class StakesVerifyCadenceTests(unittest.TestCase):
    def test_confident_streak_forces_a_real_judge_call_at_cadence(self):
        judge = ScriptedJudge(default_verdict=Verdict(satisfied=True, confidence=0.6, evidence="verified"),
                              on_exhausted="repeat")
        dj = DirectJudge([_ConfidentRule()], verify_cadence={"medium": 3})
        core = Percept.create(vision=judge, direct_judge=dj, discover_plugins=False)
        core.add_goal(Goal(id="g1", condition="screen.ocr.total_exceeds", say="alert"))

        _drive(core, "frame1", 3)  # 1st, 2nd confident; 3rd forced to a real judge call

        self.assertEqual(len(judge.calls), 1)

    def test_zero_cadence_never_forces_a_verify(self):
        judge = ScriptedJudge(default_verdict=Verdict(satisfied=True, confidence=0.6, evidence="verified"),
                              on_exhausted="repeat")
        dj = DirectJudge([_ConfidentRule()], verify_cadence={"medium": 0})
        core = Percept.create(vision=judge, direct_judge=dj, discover_plugins=False)
        core.add_goal(Goal(id="g1", condition="screen.ocr.total_exceeds", say="alert"))

        _drive(core, "frame1", 10)

        self.assertEqual(len(judge.calls), 0)


if __name__ == "__main__":
    unittest.main()
