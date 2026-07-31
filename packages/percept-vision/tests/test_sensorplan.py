"""SensorPlan compiler (Build 5) — the deterministic goal->plan compiler.

Proves: (1) every goal type compiles to a typed plan that reflects what it will actually run (incl. the
opt-in Build 2/3 knobs); (2) compiling never changes firing (program_of is read, not rewritten).

The two boundary classes at the bottom cover the compiler's edges and the brief-window Nyquist
derivation ``_sensitive_cadence``, which ships with no direct caller in the suite. Where the shipped
code does not meet the bar its own docstring sets, the test is a STRICT xfail naming its logged
defect — never a weakened assertion.

``CadenceFirewallDriftTests`` is the guard on the one thing this module now depends on that it does
not own: ``percept.behavior.coerce_authored_cadence`` and ``percept.harness.tier0._coerce_cadence``
are two copies of one firewall (the harness cannot be imported from the pure-stdlib compiler, and
tier0 could not be edited in the pass that split them), so the suite holds them equal.
"""

from __future__ import annotations

import copy
import inspect
import math
import unittest
from typing import get_args
from unittest import mock

from percept import Goal
from percept.behavior import (
    _VALID_GATE_ON,
    BehaviorProgram,
    Sense,
    coerce_authored_cadence,
    program_of,
)
from percept.harness.tier0 import _GATE_ON_VOCAB, _coerce_cadence
from percept.sensorplan import (
    PRIMITIVES,
    SensorPlan,
    _BRIEF_ESCALATE_THRESHOLD,
    _BRIEF_MAX_HEADROOM,
    _BRIEF_NYQUIST_SAMPLES,
    _sensitive_cadence,
    sensor_plan_for,
)


def goal(condition="x", **state):
    return Goal(id="g", condition=condition, say="!", state=state)


class CompilerTests(unittest.TestCase):
    def test_rhythm_goal_compiles_to_periodic_plan(self):
        plan = sensor_plan_for(goal("a completed squat repetition", type="rhythm"))
        self.assertIsInstance(plan, SensorPlan)
        self.assertEqual(plan.primitives, ("rhythm", "single"))
        self.assertEqual(plan.event_model["kind"], "periodic")
        # opt-in individuation knobs surface at their defaults (off)
        self.assertEqual(plan.event_model["smooth_s"], 0.0)
        self.assertFalse(plan.event_model["canonical_select"])

    def test_rhythm_plan_surfaces_authored_knobs(self):
        g = goal("rep", type="rhythm", rhythm={"channel": "limb-vertical", "smooth_s": 0.15,
                                               "canonical_select": True, "cadence_s": 2.0})
        plan = sensor_plan_for(g)
        self.assertEqual(plan.event_model["smooth_s"], 0.15)
        self.assertTrue(plan.event_model["canonical_select"])
        self.assertEqual(plan.event_model["period_hint_s"], 2.0)
        self.assertEqual(plan.event_model["channel"], "limb-vertical")

    def test_watch_goal_is_edge_judge_plan(self):
        plan = sensor_plan_for(goal("the door is open"))  # default watch -> judge
        self.assertEqual(plan.primitives[0], "judge")
        self.assertEqual(plan.event_model["kind"], "edge")

    def test_count_and_transition_and_absence_plans(self):
        self.assertEqual(sensor_plan_for(goal("a cup", type="count")).event_model["kind"], "distinct")
        tr = sensor_plan_for(goal("x", type="transition", **{"from": "raw", "to": "cooked"}))
        self.assertEqual(tr.event_model["kind"], "transition")
        ab = sensor_plan_for(goal("x", rearm="deadline", deadline_after_s=30.0))
        self.assertEqual(ab.event_model["kind"], "absence")
        self.assertEqual(ab.event_model["deadline_after_s"], 30.0)

    def test_as_dict_carries_the_compiled_shape(self):
        g = goal("rep", type="rhythm")
        plan = sensor_plan_for(g)
        d = plan.as_dict()
        self.assertEqual(d["primitives"], ["rhythm", "single"])
        self.assertEqual(d["event_model"]["kind"], "periodic")
        self.assertIn("cadence", d)
        self.assertIn("perception", d)

    def test_compiling_does_not_change_program(self):
        g = goal("rep", type="rhythm")
        before = program_of(g)
        sensor_plan_for(g)
        after = program_of(g)  # compiling a plan must not alter the compiled behavior program
        self.assertEqual((before.sense, before.perception), (after.sense, after.perception))

    def test_registry_primitives_are_self_describing(self):
        for name, prim in PRIMITIVES.items():
            self.assertEqual(prim.name, name)
            self.assertIn(prim.kind, {"sense", "perception"})
            self.assertTrue(prim.when_to_use)  # the cue sensor_plan_for reads to compose it


class CompilerBoundaryTests(unittest.TestCase):
    """The compiler's edges: absent/malformed state, the unguarded ``PRIMITIVES[sense]`` index, the
    hardcoded ``gate_on``, and the promise that compiling is a read."""

    def test_a_goal_with_no_authored_state_compiles_an_empty_judge_cadence(self):
        # Both "no state" shapes a caller can produce: the dataclass default, and an empty dict.
        for g in (Goal(id="g", condition="the door is open", say="!"), goal()):
            with self.subTest(state=g.state):
                plan = sensor_plan_for(g)
                self.assertEqual(plan.primitives, ("judge", "single"))
                # Nothing authored, no brief prior -> every cadence knob is None and the fovea keeps
                # its own SensePolicy defaults. This is the "byte-for-byte the default" claim at :153.
                self.assertEqual(
                    plan.cadence,
                    {"gate_on": "motion", "base_fps": None, "max_fps": None, "escalate_threshold": None},
                )

    def test_a_goal_whose_state_is_not_a_dict_raises_before_the_defensive_getattr(self):
        # sensor_plan_for's `getattr(goal, "state", {}) or {}` (:139) reads as tolerant of a missing or
        # None state — but program_of (:136) has already done a bare `goal.state.get(...)`, so that
        # defensive default is unreachable. Pinned so the tolerance is not mistaken for a contract.
        class _NoState:
            pass

        with self.assertRaises(AttributeError):
            sensor_plan_for(Goal(id="g", condition="x", say="!", state=None))
        with self.assertRaises(AttributeError):
            sensor_plan_for(_NoState())

    def test_a_sense_block_that_is_not_a_dict_is_ignored_rather_than_read(self):
        for bad in ("brief", [("brief_window_s", 1.0)], 0.5, None, True):
            with self.subTest(sense=bad):
                plan = sensor_plan_for(goal("the door is open", sense=bad))
                self.assertEqual(plan.cadence["base_fps"], None)
                self.assertEqual(plan.cadence["escalate_threshold"], None)
                self.assertNotIn("brief_window_s", plan.cadence)

    def test_every_shipped_sense_has_a_registered_primitive(self):
        # `PRIMITIVES[sense]` (:138) is an unguarded index. It is total TODAY only because every value
        # of the Sense literal is registered; this is the guard that keeps it total. A sixth sense added
        # without a manifest makes the KeyError below reachable from an ordinary Goal.
        for name in get_args(Sense):
            with self.subTest(sense=name):
                self.assertIn(name, PRIMITIVES)
                self.assertEqual(PRIMITIVES[name].kind, "sense")

    def test_a_sense_outside_the_registry_raises_a_bare_keyerror(self):
        # Not reachable from a Goal today (see the totality test above), so the program has to be
        # substituted to reach the seam at all. What it pins: the index does not degrade, it raises —
        # and _capture_plan's bare `except Exception: continue` (runtime/worker.py:3303) would swallow
        # it, dropping that goal's cadence floor with no diagnostic.
        with mock.patch("percept.sensorplan.program_of",
                        return_value=BehaviorProgram(sense="telepathy", predicate="x")):
            with self.assertRaises(KeyError):
                sensor_plan_for(goal())

    def test_compiling_twice_mutates_neither_the_goal_state_nor_the_program(self):
        for g in (goal("rep", type="rhythm", rhythm={"cadence_s": 2.0, "smooth_s": 0.15}),
                  goal("a hand touches the rail", sense={"brief_window_s": 0.5})):
            with self.subTest(state=g.state):
                state_before = copy.deepcopy(g.state)
                prog_before = program_of(g)
                first = sensor_plan_for(g)
                second = sensor_plan_for(g)
                self.assertEqual(first, second)  # deterministic: same goal, same plan
                self.assertEqual(g.state, state_before)  # nothing written back into goal.state
                self.assertNotIn("sensor_plan", g.state)
                self.assertEqual(program_of(g), prog_before)

    def test_the_compiled_gate_on_agrees_with_the_tier0_cadence_firewall(self):
        # BUG-A-04-03, fixed. This was the one divergence: tier-0's cadence firewall accepts
        # gate_on="fixed" (with an interval, as it requires) while the plan that claims to describe
        # what the goal will run reported the hardcoded "motion". Both now run the same firewall.
        g = goal("the pan is smoking", cadence={"gate_on": "fixed", "min_judge_interval": 2.0})
        self.assertEqual(_coerce_cadence(g.state["cadence"]), (2.0, "fixed"))
        self.assertEqual(sensor_plan_for(g).cadence["gate_on"], "fixed")

    def test_the_plan_reports_the_gate_the_goal_will_actually_run_on(self):
        g = goal("the pan is smoking", cadence={"gate_on": "fixed", "min_judge_interval": 2.0})
        self.assertEqual(sensor_plan_for(g).cadence["gate_on"], "fixed")

    def test_a_malformed_or_period_less_cadence_block_compiles_the_motion_default(self):
        # The plan reports the FALLBACK the gate will really apply, not the authored word: an
        # out-of-vocab gate, a "fixed" with no period, and a non-mapping cadence all arm on motion.
        for bad in ({"gate_on": "fixed"},  # fixed with no interval is not fixed
                    {"gate_on": "bogus", "min_judge_interval": 2.0},
                    {"gate_on": "fixed", "min_judge_interval": 0},
                    {}, None, "fixed", ["fixed"], 2.0):
            with self.subTest(cadence=bad):
                plan = sensor_plan_for(goal("the pan is smoking", cadence=bad))
                self.assertEqual(plan.cadence["gate_on"], "motion")

    def test_the_structural_and_semantic_gates_reach_the_plan_too(self):
        for gate in ("structural", "semantic"):
            with self.subTest(gate_on=gate):
                g = goal("the pan is smoking", cadence={"gate_on": gate})
                # these two need no interval to stand — only "fixed" does
                self.assertEqual(sensor_plan_for(g).cadence["gate_on"], gate)

    def test_a_derived_gate_does_not_disturb_the_rest_of_the_cadence(self):
        # The other three keys stay exactly what an unauthored judge goal compiles, so the only
        # thing BUG-A-04-03's fix changed in the plan is the field that was lying.
        plan = sensor_plan_for(goal("the pan is smoking",
                                    cadence={"gate_on": "fixed", "min_judge_interval": 2.0}))
        self.assertEqual(
            plan.cadence,
            {"gate_on": "fixed", "base_fps": None, "max_fps": None, "escalate_threshold": None},
        )
        # The interval the firewall also decides is NOT surfaced — the plan under-describes the
        # judge throttle by design (see sensor_plan_for). Pinned so adding it is a deliberate act.
        self.assertNotIn("min_judge_interval", plan.cadence)


class CadenceFirewallDriftTests(unittest.TestCase):
    """``percept.behavior.coerce_authored_cadence`` and ``percept.harness.tier0._coerce_cadence``.

    BUG-A-04-03 moved the cadence firewall into ``behavior`` so the pure-stdlib plan compiler could
    run the SAME coercion the gate runs without importing the harness (which would invert the
    layering), leaving ``harness.tier0`` holding a byte-identical COPY that this class held equal.

    **The copies are collapsed.** ``tier0._coerce_cadence`` IS ``behavior.coerce_authored_cadence``
    and ``tier0._GATE_ON_VOCAB`` IS ``behavior._VALID_GATE_ON`` — imported, not duplicated. This
    class is now the guard that they stay the same OBJECT rather than drifting back into two:
    identity is what makes ``fused`` impossible to add to one authoring surface and not the other.
    The behavioural corpus below survives because it is what pins the quirks THEMSELVES, which two
    copies drifting together would always have passed anyway.
    """

    #: Every input the two copies must agree on: the full gate_on vocabulary crossed with the
    #: interval cases, plus the malformed shapes that reach the fallback.
    def _corpus(self) -> list:
        cases: list = [
            None, {}, [], (), "fixed", "2s", 2.0, 0, True, False, object(),
            [("gate_on", "fixed")],  # a mapping-shaped list is not a mapping
            {"gate_on": "bogus"}, {"gate_on": None}, {"gate_on": 1}, {"gate_on": "FIXED"},
            {"min_judge_interval": 0}, {"min_judge_interval": -2.0}, {"min_judge_interval": "2"},
            {"min_judge_interval": float("nan")}, {"min_judge_interval": float("inf")},
            {"min_judge_interval": True}, {"min_judge_interval": None},
            {"unknown_key": 1},
        ]
        for gate in sorted(_VALID_GATE_ON) + ["bogus", None]:
            for interval in (None, 0, -1, 2.0, 2, float("inf"), float("nan"), "2", True):
                raw: dict = {"gate_on": gate}
                if interval is not None:
                    raw["min_judge_interval"] = interval
                cases.append(raw)
        return cases

    def test_there_is_one_vocabulary_object_and_it_carries_fused(self):
        # Identity, not equality: two equal-but-separate sets are exactly the state this collapse
        # removed, and the next channel added to one would silently not reach the other.
        self.assertIs(_VALID_GATE_ON, _GATE_ON_VOCAB)
        self.assertEqual(
            _VALID_GATE_ON, {"motion", "structural", "semantic", "fixed", "fused"}
        )

    def test_there_is_one_coercer_object(self):
        self.assertIs(coerce_authored_cadence, _coerce_cadence)

    def test_the_one_coercer_decides_the_whole_corpus_inside_the_vocabulary(self):
        for raw in self._corpus():
            with self.subTest(raw=repr(raw)):
                decided = coerce_authored_cadence(raw)
                self.assertIn(decided[1], _VALID_GATE_ON)
                self.assertTrue(decided[0] is None or isinstance(decided[0], float))

    def test_a_goal_can_author_the_fused_channel(self):
        # The A-01-04 half: a wake policy no longer overrides every goal's channel, so a goal that
        # wants the noisy-OR has to be able to SAY so through this firewall.
        self.assertEqual(coerce_authored_cadence({"gate_on": "fused"}), (None, "fused"))
        self.assertEqual(
            coerce_authored_cadence({"gate_on": "fused", "min_judge_interval": 2}), (2.0, "fused")
        )

    def test_the_shared_copy_keeps_the_documented_quirks(self):
        # Pinned literally: the quirks are the contract, and they are what a rewrite of the single
        # implementation would otherwise be free to "improve" unnoticed.
        self.assertEqual(coerce_authored_cadence({"min_judge_interval": True}), (1.0, "motion"))
        self.assertEqual(coerce_authored_cadence({"min_judge_interval": "2"}), (None, "motion"))
        self.assertEqual(
            coerce_authored_cadence({"min_judge_interval": float("nan")}), (None, "motion")
        )
        self.assertEqual(
            coerce_authored_cadence({"min_judge_interval": float("inf")}), (float("inf"), "motion")
        )
        self.assertEqual(coerce_authored_cadence({"gate_on": "fixed"}), (None, "motion"))
        self.assertEqual(
            coerce_authored_cadence({"gate_on": "fixed", "min_judge_interval": 2}), (2.0, "fixed")
        )
        self.assertEqual(coerce_authored_cadence("nope"), (None, "motion"))

    def test_the_plan_compiler_never_imports_the_harness(self):
        # The reason the firewall had to MOVE rather than be imported: sensorplan is pure stdlib +
        # percept.behavior. A harness import here inverts the layering (the harness imports down).
        import percept.sensorplan as sp

        source = inspect.getsource(sp)
        self.assertNotIn("percept.harness", source)
        self.assertNotIn("from .harness", source)


class SensitiveCadenceTests(unittest.TestCase):
    """``_sensitive_cadence`` — the brief-window Nyquist derivation, reached directly. Its three
    constants are module-private and unregistered, so the derived numbers are asserted literally AND
    against the constants: either drifting is a real change to the coverage guarantee."""

    def _base(self, **authored):
        """The exact dict shape sensor_plan_for hands it (all four keys present, unset ones None)."""
        cad = {"gate_on": "motion", "base_fps": None, "max_fps": None, "escalate_threshold": None}
        cad.update(authored)
        return cad

    def test_a_one_second_window_derives_two_fps_and_the_documented_headroom(self):
        base = self._base()
        out = _sensitive_cadence(1.0, base)
        self.assertEqual(out["base_fps"], 2.0)
        self.assertEqual(out["base_fps"], _BRIEF_NYQUIST_SAMPLES / 1.0)
        self.assertEqual(out["max_fps"], 6.0)
        self.assertEqual(out["max_fps"], 2.0 * _BRIEF_MAX_HEADROOM)
        self.assertEqual(out["escalate_threshold"], 0.05)
        self.assertEqual(out["escalate_threshold"], _BRIEF_ESCALATE_THRESHOLD)
        self.assertEqual(out["brief_window_s"], 1.0)  # surfaced so the adapter can read the prior back
        self.assertEqual(out["gate_on"], "motion")  # untouched keys carry through
        self.assertEqual(base, self._base())  # the caller's dict is copied, not mutated

    def test_an_absent_or_non_positive_or_non_numeric_prior_returns_the_cadence_unchanged(self):
        base = self._base()
        for prior in (None, 0, 0.0, -0.5, -1, "1.0", "", True, False, [], {}, object()):
            with self.subTest(brief_window_s=prior):
                out = _sensitive_cadence(prior, base)
                self.assertIs(out, base)  # same object: a non-brief judge goal is untouched
                self.assertNotIn("brief_window_s", out)

    def test_an_authored_cadence_wins_over_the_derived_nyquist(self):
        out = _sensitive_cadence(0.5, self._base(base_fps=1.0, max_fps=2.0, escalate_threshold=0.3))
        self.assertEqual(out["base_fps"], 1.0)  # not 4.0 (= 2.0/0.5)
        self.assertEqual(out["max_fps"], 2.0)  # not 3.0 (= authored base x headroom)
        self.assertEqual(out["escalate_threshold"], 0.3)
        self.assertEqual(out["brief_window_s"], 0.5)

    def test_an_authored_zero_is_kept_for_every_knob_alike(self):
        """BUG-A-04-01, fixed. All three knobs are now tested with ``is None``: an authored 0 is a
        VALUE, never a stand-in for an absent key. The derivation is a pure fill and validates
        nothing — a 0 fps envelope is refused downstream by the one authority that owns cadence, not
        quietly papered over here by a fifth partial validator."""
        out = _sensitive_cadence(1.0, self._base(base_fps=0, max_fps=0, escalate_threshold=0))
        self.assertEqual(out["escalate_threshold"], 0)
        self.assertEqual(out["base_fps"], 0)
        self.assertEqual(out["max_fps"], 0)
        # Same through the public compiler, so it is not an artifact of calling the private fn. Note
        # max_fps is UNSET here, so it is derived from the authored base — 0 x the headroom.
        plan = sensor_plan_for(goal("a hand touches the rail",
                                    sense={"brief_window_s": 1.0, "base_fps": 0, "escalate_threshold": 0}))
        self.assertEqual(plan.cadence["base_fps"], 0)
        self.assertEqual(plan.cadence["max_fps"], 0)
        self.assertEqual(plan.cadence["escalate_threshold"], 0)

    def test_an_authored_fps_of_zero_survives_the_brief_window_derivation(self):
        out = _sensitive_cadence(1.0, self._base(base_fps=0, max_fps=0))
        self.assertEqual(out["base_fps"], 0)
        self.assertEqual(out["max_fps"], 0)

    def test_a_nan_prior_is_refused_like_every_other_unusable_prior(self):
        """BUG-A-04-02, fixed. ``nan <= 0`` is False, so the non-positive guard used to miss NaN and
        derive ``2.0/nan == nan`` as the cadence floor. NaN is TRUTHY, so unlike 0 it then walked
        past ``_capture_plan``'s ``if not base`` drop, blinded the coast after one frame, AND
        suppressed the ``capture_plan_unmet`` report (``max(0.0, nan)`` is 0.0)."""
        base = self._base()
        for prior in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(brief_window_s=prior):
                self.assertIs(_sensitive_cadence(prior, base), base)

    def test_a_sub_frame_window_derives_a_floor_far_above_any_real_capture_rate(self):
        # The registered bound is Goal.state.sense.brief_window_s in [1/30, 30] s, so the low end of
        # the SHIPPED range already asks for 60 fps. The values only; the capture clamp is worker's.
        out = _sensitive_cadence(0.01, self._base())
        self.assertEqual(out["base_fps"], 200.0)
        self.assertEqual(out["max_fps"], 600.0)
        at_registered_floor = _sensitive_cadence(1.0 / 30.0, self._base())
        self.assertEqual(at_registered_floor["base_fps"], 60.0)
        self.assertEqual(at_registered_floor["max_fps"], 180.0)
        self.assertFalse(math.isinf(out["max_fps"]))

    def test_only_the_rising_edge_sense_compiles_a_sensitive_cadence(self):
        # The prior is read behind `prim.event_model["kind"] == "edge"` (:154). Every other sense
        # carries the same authored prior and compiles no floor at all — silently, which is why the
        # brief-window knob is judge-only in practice.
        brief = {"brief_window_s": 0.5}
        self.assertEqual(sensor_plan_for(goal("x", sense=brief)).cadence["base_fps"], 4.0)
        for state in ({"type": "rhythm"}, {"type": "count"},
                      {"type": "transition", "from": "raw", "to": "cooked"},
                      {"rearm": "deadline", "deadline_after_s": 30.0}):
            with self.subTest(**state):
                plan = sensor_plan_for(goal("x", sense=brief, **state))
                self.assertIsNone(plan.cadence["base_fps"])
                self.assertNotIn("brief_window_s", plan.cadence)


if __name__ == "__main__":
    unittest.main()
