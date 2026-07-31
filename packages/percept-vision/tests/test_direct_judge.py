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
    DEFAULT_VERIFY_CADENCE,
    DirectJudge,
    DirectResult,
    DirectRouting,
)
from percept.extraction import CONTEXT_KEY
from percept.harness.scripted import ScriptedJudge
from percept.harness.tier0 import gray_frame


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


class _NamedRule:
    """A confident rule that signs its verdict, so an assertion can NAME which rule answered.

    ``_ConfidentRule`` above always says ``evidence="direct"``; two of those are
    indistinguishable, which is exactly what an ordering test must be able to tell apart.
    """

    def __init__(self, name: str, applies_to: DirectRouting) -> None:
        self.name = name
        self.applies_to = applies_to
        self.calls = 0

    def evaluate(self, condition, frame, roi, facts):
        self.calls += 1
        return DirectResult(
            outcome="confident",
            verdict={"satisfied": True, "confidence": 0.9, "evidence": f"answered by {self.name}"},
        )


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


class UnregisteredStakesStringTests(unittest.TestCase):
    """``stakes`` is a free string, and an unregistered one turns the verify cadence OFF.

    ``route`` resolves the cadence with ``self.verify_cadence.get(stakes, 0)``
    (``direct_judge.py:193``) against a map that ships exactly three keys
    (``DEFAULT_VERIFY_CADENCE``, ``:67``). Any other label — a new severity name, a typo, a
    different casing — misses the map and gets ``0``, which ``:196`` reads as "never force".
    There is no error, no warning and no attribute on the result that says so: the caller asked
    for the TIGHTEST supervision they could name and silently received none at all.
    """

    CONDITION = "screen.ocr.total_exceeds"

    def _drive_confident(self, dj: DirectJudge, stakes: str, n: int) -> list:
        frame = gray_frame(48, 36)
        return [dj.route(self.CONDITION, frame, stakes=stakes) for _ in range(n)]

    def test_the_shipped_cadence_map_has_exactly_three_keys(self):
        # The premise every assertion below rests on. If someone adds a key, these tests must be
        # re-read rather than silently keep passing for a different reason.
        self.assertEqual(dict(DEFAULT_VERIFY_CADENCE), {"low": 0, "medium": 20, "high": 5})

    def test_an_unregistered_stakes_string_never_forces_a_verify(self):
        dj = DirectJudge([_ConfidentRule()])
        self.assertNotIn("critical", DEFAULT_VERIFY_CADENCE, "premise: nobody registered this label")

        # 25 > 20 (medium) > 5 (high): this run passes EVERY cadence the map ships, twice over for
        # the tightest one. Under any registered label at least one of these would be downgraded.
        results = self._drive_confident(dj, "critical", 25)

        self.assertEqual(
            [r.outcome for r in results], ["confident"] * 25,
            "an unregistered stakes label resolves to cadence 0, so no confident answer is ever "
            "downgraded — the rule governs indefinitely, unverified",
        )
        self.assertEqual(dj.verify_cadence.get("critical", 0), 0, "the mechanism: .get's default")
        self.assertGreater(25, max(DEFAULT_VERIFY_CADENCE.values()), "the run outlasted every shipped cadence")

    def test_a_casing_typo_on_a_real_label_silently_disables_it_too(self):
        # The realistic version of the same defect: the caller MEANT the tightest supervision.
        dj = DirectJudge([_ConfidentRule()])
        results = self._drive_confident(dj, "HIGH", 25)
        self.assertEqual(
            [r.outcome for r in results], ["confident"] * 25,
            "lookup is exact-match, so 'HIGH' is as unregistered as any invented label",
        )

    def test_the_same_drive_under_registered_high_does_force_a_verify(self):
        # The contrast that proves the 25-confident run above is the stakes string's doing and not
        # a dead cadence mechanism.
        dj = DirectJudge([_ConfidentRule()])
        results = self._drive_confident(dj, "high", 5)

        self.assertEqual([r.outcome for r in results[:4]], ["confident"] * 4)
        self.assertEqual(results[4].outcome, "uncertain", "the 5th confident answer is downgraded")
        self.assertIn("verify cadence reached", results[4].reason)
        self.assertIn("stakes='high'", results[4].reason)
        self.assertIsNone(results[4].verdict, "the rule's verdict is discarded on the forced call")

    def test_unregistered_and_registered_low_are_indistinguishable(self):
        # ``low`` is registered AS 0. So the caller cannot tell "I asked for no supervision" from
        # "I asked for supervision and the label did not exist" by observing route's output.
        frame = gray_frame(48, 36)
        typo = [DirectJudge([_ConfidentRule()]).route("screen.ocr.a", frame, stakes="critical")]
        low = [DirectJudge([_ConfidentRule()]).route("screen.ocr.a", frame, stakes="low")]
        self.assertEqual(typo[0].outcome, low[0].outcome)
        self.assertEqual(typo[0].reason, low[0].reason)


class _ScriptedOutcomeRule:
    """Returns a caller-supplied outcome sequence, so a test can interleave non-confident answers."""

    applies_to = DirectRouting(condition_namespace="screen.ocr.")

    def __init__(self, outcomes) -> None:
        self._outcomes = list(outcomes)
        self._i = 0

    def evaluate(self, condition, frame, roi, facts):
        outcome = self._outcomes[self._i]
        self._i += 1
        if outcome == "confident":
            return DirectResult(outcome="confident", verdict={"satisfied": True, "confidence": 0.9})
        return DirectResult(outcome=outcome, facts={"partial": "reading"})


class CadenceStreakIsCumulativeNotConsecutiveTests(unittest.TestCase):
    """The counter is never reset, so "every Nth confident answer EVER" — not "N in a row".

    ``route`` increments ``_streak`` only on the confident branch and returns a non-confident
    result before ever touching it (``direct_judge.py:190-195``); nothing zeroes it except the
    explicit :meth:`DirectJudge.reset`. The in-code comment at ``:204-206`` states this outright.
    Prose that calls it a "consecutive" streak describes a mechanism that does not exist here.
    """

    def test_an_intervening_uncertain_does_not_reset_the_streak(self):
        rule = _ScriptedOutcomeRule(["confident", "uncertain", "confident", "confident"])
        dj = DirectJudge([rule], verify_cadence={"medium": 3})
        frame = gray_frame(48, 36)

        results = [dj.route("screen.ocr.total_exceeds", frame) for _ in range(4)]

        # Under "3 CONSECUTIVE confident" this sequence would never reach a forced verify: the
        # longest run is 2. It fires on the 3rd confident answer overall, at call 4.
        self.assertEqual([r.outcome for r in results], ["confident", "uncertain", "confident", "uncertain"])
        self.assertIn("verify cadence reached", results[3].reason)
        self.assertNotIn("verify cadence", results[1].reason, "call 2 is the rule's own uncertain")

    def test_reset_is_the_only_thing_that_clears_the_counter(self):
        rule = _ScriptedOutcomeRule(["confident"] * 6)
        dj = DirectJudge([rule], verify_cadence={"medium": 3})
        frame = gray_frame(48, 36)

        first = [dj.route("screen.ocr.total_exceeds", frame).outcome for _ in range(2)]
        dj.reset()
        after = [dj.route("screen.ocr.total_exceeds", frame).outcome for _ in range(3)]

        self.assertEqual(first, ["confident", "confident"])
        self.assertEqual(
            after, ["confident", "confident", "uncertain"],
            "reset() zeroed the streak, so the forced verify moved out to the 3rd post-reset answer",
        )


class RegistrationOrderIsPriorityTests(unittest.TestCase):
    """First match wins, so registration order is priority — NOT specificity.

    ``_rule_for`` (``direct_judge.py:161-165``) returns the first rule whose ``applies_to``
    matches and stops. A ``DirectRouting()`` with neither axis set matches every call
    unconditionally (``:89-94``), so registering one before a narrower rule shadows the narrower
    rule permanently. Nothing raises, nothing warns, and the narrow rule simply never runs.
    """

    CONDITION = "screen.ocr.total_exceeds"

    def _catch_all(self) -> _NamedRule:
        return _NamedRule("catch-all", DirectRouting())

    def _narrow(self) -> _NamedRule:
        return _NamedRule("narrow", DirectRouting(condition_namespace="screen.ocr."))

    def test_both_rules_really_do_match_the_condition(self):
        # Without this the ordering assertions below would be satisfied by the narrow rule simply
        # not matching, which proves nothing about priority.
        catch_all, narrow = self._catch_all(), self._narrow()
        self.assertTrue(catch_all.applies_to.matches(condition=self.CONDITION))
        self.assertTrue(narrow.applies_to.matches(condition=self.CONDITION))

    def test_a_catch_all_registered_first_shadows_the_narrower_rule(self):
        catch_all, narrow = self._catch_all(), self._narrow()
        dj = DirectJudge([catch_all, narrow])

        result = dj.route(self.CONDITION, gray_frame(48, 36))

        self.assertEqual(result.verdict.evidence, "answered by catch-all")
        self.assertEqual(narrow.calls, 0, "the narrower rule never ran — it is unreachable")

    def test_the_reverse_order_routes_the_same_condition_to_the_narrow_rule(self):
        # Same two rules, same condition, only the order differs. If matching were resolved by
        # specificity the narrow rule would win in BOTH orders; it wins in exactly one.
        catch_all, narrow = self._catch_all(), self._narrow()
        dj = DirectJudge([narrow, catch_all])

        result = dj.route(self.CONDITION, gray_frame(48, 36))

        self.assertEqual(result.verdict.evidence, "answered by narrow")
        self.assertEqual(catch_all.calls, 0)

    def test_order_decides_even_when_the_narrow_rule_is_strictly_more_specific(self):
        # A rule narrowed on BOTH axes still loses to an earlier catch-all.
        catch_all = self._catch_all()
        narrowest = _NamedRule("narrowest", DirectRouting(condition_namespace="screen.ocr.", role="detector"))
        dj = DirectJudge([catch_all, narrowest])

        result = dj.route(self.CONDITION, gray_frame(48, 36), role="detector")

        self.assertTrue(narrowest.applies_to.matches(condition=self.CONDITION, role="detector"))
        self.assertEqual(result.verdict.evidence, "answered by catch-all")

    def test_a_catch_all_also_swallows_conditions_no_other_rule_claims(self):
        # Why the class docstring says at most one catch-all should be registered: it is not a
        # fallback, it is a front-of-queue rule for every condition in the process.
        catch_all, narrow = self._catch_all(), self._narrow()
        dj = DirectJudge([catch_all, narrow])

        result = dj.route("camera.motion.crossed_line", gray_frame(48, 36))

        self.assertEqual(result.verdict.evidence, "answered by catch-all")
        self.assertNotEqual(
            result.outcome, "degraded",
            "with a catch-all present nothing ever reaches the normal judge path via 'no rule matched'",
        )

    def test_register_appends_so_it_loses_to_a_constructor_rule(self):
        catch_all, narrow = self._catch_all(), self._narrow()
        dj = DirectJudge([catch_all])
        dj.register(narrow)

        result = dj.route(self.CONDITION, gray_frame(48, 36))

        self.assertEqual(result.verdict.evidence, "answered by catch-all")
        self.assertEqual(narrow.calls, 0)

    def test_a_constructor_rule_registered_first_still_wins_when_the_catch_all_is_appended(self):
        catch_all, narrow = self._catch_all(), self._narrow()
        dj = DirectJudge([narrow])
        dj.register(catch_all)

        result = dj.route(self.CONDITION, gray_frame(48, 36))

        self.assertEqual(result.verdict.evidence, "answered by narrow")

    def test_all_register_agrees_with_the_equivalent_constructor_list(self):
        # The two ways to get rules in must not disagree, or "registration order" would mean two
        # different things depending on which seam the caller used.
        for order in (("catch-all", "narrow"), ("narrow", "catch-all")):
            with self.subTest(order=order):
                built = {"catch-all": self._catch_all, "narrow": self._narrow}
                via_ctor = DirectJudge([built[order[0]](), built[order[1]]()])
                via_register = DirectJudge()
                via_register.register(built[order[0]]())
                via_register.register(built[order[1]]())

                frame = gray_frame(48, 36)
                self.assertEqual(
                    via_ctor.route(self.CONDITION, frame).verdict.evidence,
                    via_register.route(self.CONDITION, frame).verdict.evidence,
                )
                self.assertEqual(
                    via_ctor.route(self.CONDITION, frame).verdict.evidence,
                    f"answered by {order[0]}",
                )


if __name__ == "__main__":
    unittest.main()
