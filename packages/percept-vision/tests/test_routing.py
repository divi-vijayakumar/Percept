"""L1: the routing rule-table + the structural defer-on-miss.

These prove **routing-table logic and the defer contract**, not model accuracy (that's L2 with
a real classifier). Cases are auto-generated from each rule's ``examples`` (must HIT the rule)
and ``defers`` (must DEFER), plus the table-wide ``DEFER_EXAMPLES`` covering the four classes
the spec calls out: goal-authoring, conditional, ambiguous, fragment. Invariant 2 — when in
doubt, defer — is the property under test: the router only fast-paths confident, non-authoring
intents; everything else returns ``None`` so the conductor (the only tier that writes state)
gets the turn.
"""

import unittest

from percept.config import load_settings
from percept.routing import DEFAULT_RULES, RegexRouteLayer, RouteEngine
from percept.routing.rules import DEFER_EXAMPLES


def _engine() -> RouteEngine:
    threshold = load_settings().routing.confidence_threshold
    return RouteEngine.default(threshold=threshold)


class TestRuleExamplesHit(unittest.TestCase):
    """Every rule's ``examples`` must match THAT rule (first-match-wins, confident route)."""

    def test_examples_route_to_their_rule(self):
        layer = RegexRouteLayer(DEFAULT_RULES)
        engine = _engine()
        for rule in DEFAULT_RULES:
            for utterance in rule.examples:
                with self.subTest(rule=rule.name, utterance=utterance):
                    match = layer.score(utterance)
                    self.assertIsNotNone(
                        match, f"{utterance!r} should HIT a rule, got no match"
                    )
                    self.assertEqual(
                        match.rule,
                        rule.name,
                        f"{utterance!r} hit {match.rule!r}, expected {rule.name!r}",
                    )
                    self.assertEqual(match.route, rule.route)
                    # And the engine must surface it as a confident Decision (not a defer).
                    decision = engine.route(utterance)
                    self.assertIsNotNone(decision)
                    self.assertEqual(decision.route, rule.route)


class TestRuleDefers(unittest.TestCase):
    """Each rule's ``defers`` look superficially similar but must DEFER to the conductor."""

    def test_rule_local_defers(self):
        engine = _engine()
        for rule in DEFAULT_RULES:
            for utterance in rule.defers:
                with self.subTest(rule=rule.name, utterance=utterance):
                    self.assertIsNone(
                        engine.route(utterance),
                        f"{utterance!r} must DEFER (no confident route), but routed",
                    )


class TestTableWideDefers(unittest.TestCase):
    """Goal-authoring / conditional / ambiguous / fragment utterances all DEFER."""

    def test_defer_examples_defer(self):
        engine = _engine()
        for utterance in DEFER_EXAMPLES:
            with self.subTest(utterance=utterance):
                self.assertIsNone(
                    engine.route(utterance),
                    f"{utterance!r} must DEFER to the conductor (router must not fast-path it)",
                )

    def test_goal_authoring_never_fast_paths(self):
        # The load-bearing case: a watch/remind request must reach the conductor so a goal can
        # be authored. A false fast-path here would skip goal authoring (invariant 2).
        engine = _engine()
        for utterance in ("watch my coffee", "remind me when the kettle boils"):
            self.assertIsNone(engine.route(utterance))


class TestDeferIsStructural(unittest.TestCase):
    """Defer-on-miss is a confidence threshold, not a hardcoded special-case."""

    def test_threshold_gates_a_low_confidence_match(self):
        # A layer that matches but with low confidence must be gated exactly like a miss.
        class LowConfLayer:
            def score(self, utterance, context=None):
                from percept.routing import RouteMatch

                return RouteMatch(route="respond_direct", confidence=0.3, rule="stub")

        engine = RouteEngine([LowConfLayer()], threshold=0.6)
        self.assertIsNone(engine.route("anything"))  # 0.3 < 0.6 -> structural defer

    def test_raising_threshold_turns_a_hit_into_a_defer(self):
        # Same utterance, higher bar -> defer. Proves the threshold (config), not the regex,
        # is what decides; this is the seam the V0.1 LearnedRouteLayer reuses.
        utterance = "do you know the time?"  # direct_question @ 0.8
        self.assertIsNotNone(RouteEngine.default(threshold=0.6).route(utterance))
        self.assertIsNone(RouteEngine.default(threshold=0.95).route(utterance))

    def test_unknown_route_from_a_layer_defers_not_misroutes(self):
        class BadLayer:
            def score(self, utterance, context=None):
                from percept.routing import RouteMatch

                return RouteMatch(route="not_a_real_route", confidence=0.99, rule="bad")

        engine = RouteEngine([BadLayer()], threshold=0.6)
        self.assertIsNone(engine.route("hello"))  # invalid route can't silently mis-route


class TestRouterAuthorsNoGoals(unittest.TestCase):
    """The router can only pick a route; goal ops are the conductor's alone."""

    def test_routed_decision_has_no_goal_ops(self):
        engine = _engine()
        decision = engine.route("where is my phone?")
        self.assertIsNotNone(decision)
        self.assertEqual(decision.goal_ops, [])


if __name__ == "__main__":
    unittest.main()
