"""Phase-0 equivalence spike (the go/no-go): ONE executor reproduces the legacy firing behaviors
byte-for-byte, and honors the three review constraints. No agent.py, no gate.py edits.

If this is green, "generality without behavior change" holds and the executor is the seed for the
real refactor. If a single trace diverges, we stop before touching agent.py.
"""

import unittest

from percept.adapters import Goal, Verdict
from percept.behavior import (
    Accumulator,
    BehaviorExecutor,
    BehaviorProgram,
    ConcernContext,
    UnsafeExpression,
    compile_fire_when,
    program_of,
)
from percept.gate import ASK, FIRE, Gate

YES = Verdict(True, 0.95, "yes")
NO = Verdict(False, 0.95, "no")
UNSURE = Verdict(True, 0.4, "unsure")


def _ex():
    g = Gate()
    return g, BehaviorExecutor(g)


def _drive(ex, goal, program, verdicts, base_key=("g", "e")):
    out = []
    for i, v in enumerate(verdicts):
        f = ex.fire_judged(goal, v, base_key, program, freshness=1.0, t=float(i))
        out.append(f.action if f is not None else None)
    return out


# --------------------------------------------------------------------------------------------------
# 1. judge sense reproduces test_firing_cr's rearm traces byte-for-byte
# --------------------------------------------------------------------------------------------------
class TestJudgeRearmEquivalence(unittest.TestCase):
    def test_on_false_default(self):
        _, ex = _ex()
        g = Goal(id="g", condition="c", say="s")
        self.assertEqual(_drive(ex, g, program_of(g), [YES, YES, NO, YES]), [FIRE, None, None, FIRE])

    def test_unsure_asks_without_touching_edge(self):
        _, ex = _ex()
        g = Goal(id="g", condition="c", say="s")
        self.assertEqual(_drive(ex, g, program_of(g), [UNSURE, YES]), [ASK, FIRE])

    def test_never_one_shot(self):
        _, ex = _ex()
        g = Goal(id="g", condition="c", say="s")
        g.state["rearm"] = "never"
        self.assertEqual(
            _drive(ex, g, program_of(g), [YES, YES, NO, YES, NO, YES]),
            [FIRE, None, None, None, None, None],
        )

    def test_interval_per_bucket(self):
        _, ex = _ex()
        g = Goal(id="g", condition="c", say="s")
        g.state["rearm"] = "interval"
        g.state["cadence_s"] = 10.0
        out = _drive(ex, g, program_of(g), [YES] * 21)  # t = 0..20
        self.assertEqual([i for i, a in enumerate(out) if a == FIRE], [0, 10, 20])

    def test_fire_event_carries_say_and_entity(self):
        _, ex = _ex()
        g = Goal(id="g", condition="c", say="stay off caffeine")
        f = ex.fire_judged(g, YES, ("g", "cup2"), program_of(g))
        self.assertEqual((f.action, f.text, f.entity_id), (FIRE, "stay off caffeine", "cup2"))

    def test_hard_quadrant_fire_escalates_to_ask(self):
        _, ex = _ex()
        g = Goal(id="g", condition="the stove is unattended", say="turn off the stove")
        g.state.update({"miss_cost": "severe", "false_fire_cost": "irreversible"})
        f = ex.fire_judged(g, YES, ("g", "e"), program_of(g))
        self.assertEqual(f.action, ASK)
        self.assertIn("confirm", f.text)


# --------------------------------------------------------------------------------------------------
# 2. transition sense reproduces _perceive_transition (fire on B only after A)
# --------------------------------------------------------------------------------------------------
class TestTransitionEquivalence(unittest.TestCase):
    def test_fires_on_b_after_a_only(self):
        _, ex = _ex()
        g = Goal(id="wake", condition="awake", say="good morning")
        # (from, to) per tick: join in B (no A) -> arm on A -> fire on B after A -> no re-fire
        script = [(NO, YES), (YES, NO), (NO, YES), (NO, YES)]
        out = [ex.fire_transition(g, fr, to) for fr, to in script]
        self.assertEqual([f.action if f else None for f in out], [None, None, FIRE, None])
        self.assertEqual(out[2].text, "good morning")

    def test_never_fires_if_a_never_seen(self):
        _, ex = _ex()
        g = Goal(id="wake", condition="awake", say="up")
        out = [ex.fire_transition(g, NO, YES) for _ in range(5)]  # always in B, never saw A
        self.assertEqual([f for f in out], [None] * 5)

    def test_both_true_is_not_an_ordering_edge_and_preserves_latches(self):
        _, ex = _ex()
        g = Goal(id="wake", condition="awake", say="up")

        first = ex.transition_decide(g, YES, YES)
        self.assertIsNone(first.fired)
        self.assertEqual((first.armed, first.fired_latch, first.outcome), (False, False, "idle"))

        arm = ex.transition_decide(g, YES, NO)
        self.assertEqual((arm.armed, arm.fired_latch, arm.outcome), (True, False, "armed"))
        fire = ex.transition_decide(g, NO, YES)
        self.assertIsNotNone(fire.fired)

        contradictory_after_fire = ex.transition_decide(g, YES, YES)
        self.assertIsNone(contradictory_after_fire.fired)
        self.assertEqual(
            (
                contradictory_after_fire.armed,
                contradictory_after_fire.fired_latch,
                contradictory_after_fire.outcome,
            ),
            (True, True, "suppressed_latched"),
        )
        self.assertIsNone(ex.fire_transition(g, NO, YES))

        rearm = ex.transition_decide(g, YES, NO)
        self.assertEqual((rearm.armed, rearm.fired_latch, rearm.outcome), (True, False, "rearmed"))
        self.assertIsNotNone(ex.fire_transition(g, NO, YES))

    def test_transition_ambiguity_does_not_invent_state_order(self):
        _, ex = _ex()
        g = Goal(id="wake", condition="awake", say="up")

        self.assertIsNone(ex.fire_transition(g, UNSURE, YES))
        armed = ex.transition_decide(g, YES, UNSURE)
        self.assertEqual((armed.armed, armed.fired_latch, armed.outcome), (True, False, "armed"))
        self.assertIsNotNone(ex.fire_transition(g, UNSURE, YES))


# --------------------------------------------------------------------------------------------------
# 3. absence sense reproduces check_deadlines (overdue fires once)
# --------------------------------------------------------------------------------------------------
class TestAbsenceEquivalence(unittest.TestCase):
    def test_overdue_fires_once(self):
        _, ex = _ex()
        g = Goal(id="salt", condition="salt has been added", say="...")
        p = BehaviorProgram(sense="absence", deadline_at=10.0)
        self.assertIsNone(ex.fire_absence(g, p, now=5.0, satisfied=False))  # before deadline
        f = ex.fire_absence(g, p, now=11.0, satisfied=False)  # overdue
        self.assertEqual(f.action, FIRE)
        self.assertIn("Still waiting", f.text)
        self.assertIsNone(ex.fire_absence(g, p, now=12.0, satisfied=False))  # refractory

    def test_satisfied_in_time_never_fires(self):
        _, ex = _ex()
        g = Goal(id="salt", condition="salt has been added", say="...")
        p = BehaviorProgram(sense="absence", deadline_at=10.0)
        self.assertIsNone(ex.fire_absence(g, p, now=11.0, satisfied=True))


# --------------------------------------------------------------------------------------------------
# 4. constraint 3 — fire_when is AST-whitelisted; eval() is never used
# --------------------------------------------------------------------------------------------------
class TestFireWhenSafety(unittest.TestCase):
    def test_valid_expressions(self):
        self.assertTrue(compile_fire_when("n >= 3")({"n": 3}))
        self.assertFalse(compile_fire_when("n >= 3")({"n": 2}))
        self.assertTrue(compile_fire_when("n == 3 and t < 600")({"n": 3, "t": 100}))
        self.assertFalse(compile_fire_when("n == 3 and t < 600")({"n": 3, "t": 700}))

    def test_rejects_injection_and_out_of_grammar(self):
        for bad in (
            "__import__('os').system('x')",  # call + foreign name
            "n.real",  # attribute access
            "open('x')",  # call to foreign name
            "foo > 1",  # name outside {n, t}
            "n() > 1",  # call
            "[].append(1)",  # subscript/list/call
            "True",  # non-numeric constant
            "n + 1",  # arithmetic (BinOp not whitelisted)
        ):
            with self.assertRaises(UnsafeExpression, msg=bad):
                compile_fire_when(bad)


# --------------------------------------------------------------------------------------------------
# 5. the NOVEL task — "alert me the 3rd distinct time" — composes with ZERO new code
# --------------------------------------------------------------------------------------------------
class TestNovelTaskComposes(unittest.TestCase):
    def test_third_distinct_only(self):
        _, ex = _ex()
        g = Goal(id="nth", condition="the thing happened", say="that's the 3rd")
        p = BehaviorProgram(
            sense="episode", accumulate=Accumulator(kind="count_distinct", window_s=600, fire_when="n == 3")
        )
        actions = [
            (ex.fire_episode(g, p, n=n, is_new=True, t=float(n)) or _N).action for n in (1, 2, 3, 4)
        ]
        self.assertEqual(actions, [None, None, FIRE, None])  # fires only at the 3rd distinct

    def test_fire_each_default(self):
        _, ex = _ex()
        g = Goal(id="ev", condition="rep done", say="rep")
        p = BehaviorProgram(sense="episode")  # no fire_when -> fire each new episode
        out = [ex.fire_episode(g, p, n=n, is_new=True) for n in (1, 2, 3)]
        self.assertEqual([f.action for f in out], [FIRE, FIRE, FIRE])


class _NoFire:
    action = None


_N = _NoFire()


# --------------------------------------------------------------------------------------------------
# 6. constraint 1 — stateful-primitive memory is ISOLATED from the gate
# --------------------------------------------------------------------------------------------------
class TestStateIsolation(unittest.TestCase):
    def test_transition_state_never_pollutes_the_gate(self):
        """Constraint 1, sharpened: the transition sense's STATEFUL PRIMITIVE memory stays in the
        isolated per-concern ctx, and what it puts in the gate is ONE key holding ONE bit.

        This assertion used to read ``gate._armed == {}`` — "transition never calls the gate at all".
        That was not isolation, it was the BYPASS: it is precisely why the transition sense had no
        rising edge of its own, no refractory, no dwell and no delivery merge, and why its fires
        carried ``reason=None``. Constraint 1 was never "this sense may not use the gate"; it is "the
        gate stays one bit per key and stateful primitives keep their own memory elsewhere" (see the
        ConcernContext re-arm-discriminator note in behavior.py). Both halves are asserted below, and
        the key-count assertion is what would catch a regression to per-tick / per-bucket gate keys.
        """
        gate, ex = _ex()
        g = Goal(id="wake", condition="awake", say="up")
        for fr, to in [(NO, YES), (YES, NO), (NO, YES)]:
            ex.fire_transition(g, fr, to)
        # The ORDERING WITNESS (has A been seen?) is executor-isolated ctx state — the gate has no
        # vocabulary for it and never sees it. ``fired`` is now only a mirror of the gate's edge.
        self.assertEqual(ex.ctx("wake").transitions["scene"], {"seen_from": True, "fired": True})
        # ...and the gate holds EXACTLY the one mechanical armed/disarmed bit for this concern's own
        # latch key — no per-tick, per-focus or per-bucket key leak, and nothing for any other goal.
        self.assertEqual(gate._armed, {("wake", "scene"): False})
        # The two ctx bits and the one gate bit agree: the edge is spent, and only a re-established
        # 'from' (a confident-false to the gate) can re-open it.
        self.assertIs(gate._armed[("wake", "scene")], False)

    def test_each_concern_has_its_own_context(self):
        _, ex = _ex()
        self.assertIsNot(ex.ctx("a"), ex.ctx("b"))
        self.assertIsInstance(ex.ctx("a"), ConcernContext)


# --------------------------------------------------------------------------------------------------
# 7. constraint 2 — the confusion substrate records, but does NOT alter the firing trace
# --------------------------------------------------------------------------------------------------
class TestConfusionSubstrate(unittest.TestCase):
    def test_run_of_unknowns_flags_confused_without_changing_trace(self):
        _, ex = _ex()
        g = Goal(id="g", condition="c", say="s")
        actions = _drive(ex, g, program_of(g), [UNSURE, UNSURE, UNSURE])
        self.assertEqual(actions, [ASK, ASK, ASK])  # trace is exactly the legacy ASK,ASK,ASK
        self.assertTrue(ex.ctx("g").confusion.confused())  # ...and confusion is now flagged
        self.assertEqual(ex.ctx("g").confusion.consecutive_unknown, 3)

    def test_confident_verdict_resets_the_counter(self):
        _, ex = _ex()
        g = Goal(id="g", condition="c", say="s")
        _drive(ex, g, program_of(g), [UNSURE, UNSURE, YES])
        self.assertEqual(ex.ctx("g").confusion.consecutive_unknown, 0)
        self.assertFalse(ex.ctx("g").confusion.confused())


# --------------------------------------------------------------------------------------------------
# 8. presets — legacy goal.state compiles to the right program (enum values become constructors)
# --------------------------------------------------------------------------------------------------
class TestPresetCompile(unittest.TestCase):
    def test_default_is_judge_on_false(self):
        p = program_of(Goal(id="g", condition="c", say="s"))
        self.assertEqual((p.sense, p.key.suppress_false), ("judge", False))

    def test_never_sets_suppress_false(self):
        g = Goal(id="g", condition="c", say="s")
        g.state["rearm"] = "never"
        self.assertTrue(program_of(g).key.suppress_false)

    def test_interval_sets_time_bucket(self):
        g = Goal(id="g", condition="c", say="s")
        g.state.update({"rearm": "interval", "cadence_s": 10.0})
        p = program_of(g)
        self.assertIn("time_bucket", p.key.by)
        self.assertEqual(p.key.bucket_s, 10.0)

    def test_deadline_is_absence(self):
        g = Goal(id="g", condition="c", say="s")
        # program_of reads the ARMED absolute stamp (deadline_at), not the authored duration.
        g.state.update({"rearm": "deadline", "deadline_at": 30.0})
        self.assertEqual((program_of(g).sense, program_of(g).deadline_at), ("absence", 30.0))

    def test_transition_and_count(self):
        gt = Goal(id="t", condition="c", say="s")
        gt.state.update({"type": "transition", "from": "A", "to": "B"})
        self.assertEqual(program_of(gt).sense, "transition")
        gc = Goal(id="c", condition="c", say="s")
        gc.state["type"] = "count"
        self.assertEqual(program_of(gc).sense, "episode")


if __name__ == "__main__":
    unittest.main()
