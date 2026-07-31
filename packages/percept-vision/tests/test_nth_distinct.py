"""The novel task — "alert me the 3rd distinct time" — fires end-to-end with ZERO new agent code.

It is a plain watch goal + an authored ``fire_when``: each rising edge (off->on) is a distinct
occurrence (the gate already gives one fire per rising edge), and the executor's accumulate gates the
user-facing alert on ``fire_when(n)``. So it routes through the UNCHANGED dispatch (watch -> _fire_concern
-> executor.fire_judged). This is the payoff of the generalization: a task no code path was written for.
"""

import ast
import asyncio
import builtins
import contextlib
import unittest

from percept.adapters import Box, Goal, Observation, Verdict
from percept.agent import Percept
from percept.backends.fake import FakeVision
from percept.behavior import (
    BehaviorExecutor,
    UnsafeExpression,
    _eval_node,
    coerce_authored_behavior,
    compile_fire_when,
    program_of,
)
from percept.config import GateSettings, Settings
from percept.gate import FIRE, Gate

YES = Verdict(True, 0.95, "yes")
NO = Verdict(False, 0.95, "no")
BOX = Box(0.4, 0.4, 0.2, 0.2)  # one stable entity across frames -> one gate key


class TestAccumulateUnit(unittest.TestCase):
    def test_fires_only_on_nth_rising_edge(self):
        ex = BehaviorExecutor(Gate())
        g = Goal(id="g", condition="c", say="3rd time")
        g.state["fire_when"] = "n == 3"
        out = []
        for v in [YES, NO, YES, NO, YES]:  # 3 rising edges
            f = ex.fire_judged(g, v, ("g", "e"), program_of(g))
            out.append(f.action if f is not None else None)
        self.assertEqual(out, [None, None, None, None, FIRE])  # alert only on the 3rd

    def test_no_fire_when_is_byte_for_byte(self):
        ex = BehaviorExecutor(Gate())
        g = Goal(id="g", condition="c", say="s")  # no fire_when
        out = []
        for v in [YES, NO, YES]:
            f = ex.fire_judged(g, v, ("g", "e"), program_of(g))
            out.append(f.action if f is not None else None)
        self.assertEqual(out, [FIRE, None, FIRE])  # every rising edge fires (unchanged)


class TestNthDistinctEndToEnd(unittest.TestCase):
    def _build(self):
        vision = FakeVision(
            script={"on": Verdict(True, 0.95, "on couch"), "off": Verdict(False, 0.95, "off couch")},
            detect_script={"on": [Observation("dog", BOX, 0.95)], "off": [Observation("dog", BOX, 0.95)]},
        )
        a = Percept.create(
            vision=vision, settings=Settings(gate=GateSettings(threshold=0.7)), discover_plugins=False
        )
        g = Goal(id="g-dog", condition="the dog is on the couch", say="that's the 3rd time!")
        g.state["fire_when"] = "n == 3"  # what the conductor authors from "alert me the 3rd distinct time"
        a.add_goal(g)
        return a

    def test_third_rising_edge_alerts_through_unchanged_dispatch(self):
        async def run():
            a = self._build()
            out = []
            for i, frame in enumerate(["on", "off", "on", "off", "on"]):  # rising edges at 0,2,4
                evs = await a.perceive_judged(frame, t=float(i))
                out.append([e.action for e in evs])
            return out

        out = asyncio.run(run())
        self.assertEqual(out, [[], [], [], [], [FIRE]])  # only the 3rd distinct time alerts
        # the firing event carries the user's say
        a = self._build()
        last = asyncio.run(self._final_fire(a))
        self.assertEqual(last.text, "that's the 3rd time!")

    async def _final_fire(self, a):
        ev = None
        for i, frame in enumerate(["on", "off", "on", "off", "on"]):
            evs = await a.perceive_judged(frame, t=float(i))
            if evs:
                ev = evs[0]
        return ev


# --------------------------------------------------------------------------------------------------
# The full hostile node-class table for ``fire_when``.
#
# ``compile_fire_when`` behavior.py:186; the up-front validator loop :196-208; the interpreter
# ``_eval_node`` :159. The stake is not style: ``fire_when`` arrives as LLM-authored text, so any
# escape from the whitelist is arbitrary code execution. ``eval``/``exec`` are never invoked and
# ``compile`` is only ever reached through ``ast.parse`` in parse-only mode.
#
# test_program_equivalence.py::TestFireWhenSafety already pins eight inputs — ``__import__(...)``,
# ``n.real``, ``open('x')``, ``foo > 1``, ``n() > 1``, ``[].append(1)``, ``True``, ``n + 1``. This
# table deliberately covers only the node classes that one misses.
# --------------------------------------------------------------------------------------------------

# (node class the input smuggles in, expression). Every one must raise UnsafeExpression.
HOSTILE_EXPRESSIONS: list[tuple[str, str]] = [
    ("Subscript", "n[0] > 1"),
    ("Subscript-slice", "n[0:1] > 1"),
    ("Lambda", "n > (lambda: 1)"),
    ("ListComp", "[x for x in (1, 2)] > n"),
    ("SetComp", "{x for x in (1, 2)} > n"),
    ("DictComp", "{x: 1 for x in (1, 2)} > n"),
    ("GeneratorExp", "n > (x for x in (1, 2))"),
    ("JoinedStr", 'f"{n}" > n'),
    ("NamedExpr", "(m := n) > 1"),
    ("UnaryOp-USub", "-n > 1"),
    ("UnaryOp-Not", "not n"),
    ("UnaryOp-Invert", "~n > 1"),
    ("IfExp", "n if t else 0"),
    ("In", "n in (1, 2)"),
    ("NotIn", "n not in (1, 2)"),
    ("Is", "n is t"),
    ("IsNot", "n is not None"),
    ("Tuple", "n > (1, 2)"),
    ("List", "n > [1, 2]"),
    ("Set", "n > {1, 2}"),
    ("Dict", "n > {1: 2}"),
    ("Constant-str", 'n > "3"'),
    ("Constant-bytes", "n > b'3'"),
    ("Constant-None", "None"),
    ("Constant-False", "False"),
    ("Constant-complex", "n > 1j"),
    ("Constant-Ellipsis", "n > ..."),
    # A foreign name the INTERPRETER alone never reaches: the chain short-circuits at ``n < 0``.
    # Only the up-front walk at :196 sees it. Pinned structurally below as well.
    ("Name-outside-whitelist-behind-a-short-circuit", "n < 0 < evil"),
    # The two branches a reader assumes are dead. The whole tree is validated before anything runs.
    ("Call-on-a-branch-python-would-short-circuit", "n == 3 or __import__('os').system('x')"),
    ("Attribute-on-a-branch-python-would-short-circuit", "n == 999 and n.real"),
]

# Not expressions at all, so ``ast.parse(..., mode="eval")`` refuses them before the whitelist runs.
# BUG-D-05-01, closed: that refusal used to escape as the RAW ``SyntaxError``, a second refusal type
# for one job. UnsafeExpression is a ValueError and SyntaxError is not, so every guard in the tree —
# all of them written as ``except UnsafeExpression`` — was blind to exactly half of what
# compile_fire_when could raise. ``compile_fire_when`` converts at the parse now, so these raise the
# same single refusal the whitelist does.
NON_EXPRESSION_INPUTS: list[tuple[str, str]] = [
    ("import-statement", "import os"),
    ("assignment-statement", "n = 3"),
    ("function-definition", "def f(): pass"),
    ("two-statements", "n == 3\nn == 4"),
    ("trailing-semicolon", "n == 3;"),
    ("empty-string", ""),
    ("whitespace-only", "   "),
]


@contextlib.contextmanager
def _no_code_execution():
    """Poison ``eval``/``exec`` and allow ``compile`` only in ``PyCF_ONLY_AST`` (parse-only) mode.

    ``ast.parse`` resolves ``compile`` through builtins, so this really does observe the one call
    the module makes. Yields the list of ``flags`` every ``compile`` call was made with.
    """
    real_eval, real_exec, real_compile = builtins.eval, builtins.exec, builtins.compile
    flags_seen: list[int] = []

    def _guard_eval(*a, **k):
        raise AssertionError("eval() was invoked while compiling a fire_when expression")

    def _guard_exec(*a, **k):
        raise AssertionError("exec() was invoked while compiling a fire_when expression")

    def _guard_compile(source, filename="<unknown>", mode="exec", flags=0, *a, **k):
        flags_seen.append(flags)
        if not flags & ast.PyCF_ONLY_AST:
            raise AssertionError("compile() was invoked to produce an executable code object")
        return real_compile(source, filename, mode, flags, *a, **k)

    builtins.eval, builtins.exec, builtins.compile = _guard_eval, _guard_exec, _guard_compile
    try:
        yield flags_seen
    finally:
        builtins.eval, builtins.exec, builtins.compile = real_eval, real_exec, real_compile


class TestFireWhenHostileNodeMatrix(unittest.TestCase):
    def test_the_grammar_it_does_accept(self):
        """Positive control — without this the rejection table could pass by rejecting everything."""
        self.assertTrue(compile_fire_when("n == 3")({"n": 3, "t": 0.0}))
        self.assertFalse(compile_fire_when("n == 3")({"n": 4, "t": 0.0}))
        self.assertTrue(compile_fire_when("n > 1 and t < 5")({"n": 2, "t": 1.0}))
        self.assertFalse(compile_fire_when("n > 1 and t < 5")({"n": 2, "t": 9.0}))
        self.assertTrue(compile_fire_when("n == 1 or n == 2")({"n": 2, "t": 0.0}))
        self.assertTrue(compile_fire_when("1 < n < 5")({"n": 3, "t": 0.0}))  # a chained comparison
        self.assertFalse(compile_fire_when("1 < n < 5")({"n": 9, "t": 0.0}))
        self.assertTrue(compile_fire_when("t >= 1.5")({"n": 0, "t": 1.5}))  # a float constant

    def test_every_out_of_grammar_node_class_is_rejected(self):
        for node_class, expr in HOSTILE_EXPRESSIONS:
            with self.subTest(node=node_class, expr=expr):
                with self.assertRaises(UnsafeExpression, msg=f"{node_class} was ACCEPTED: {expr!r}"):
                    compile_fire_when(expr)

    def test_a_non_expression_is_refused_as_the_same_single_exception_type(self):
        """One refusal, not two. ``ast.parse``'s SyntaxError is converted at the authority.

        The type is the whole point. A caller guards with ``except UnsafeExpression`` (or the
        ``ValueError`` it subclasses); a SyntaxError leaking out of the same function is a refusal
        no correctly-written guard can see, and it took down the authoring firewall and both
        firing-time call sites (neither of which has any handler at all).
        """
        for label, src in NON_EXPRESSION_INPUTS:
            with self.subTest(input=label, src=src):
                with self.assertRaises(UnsafeExpression, msg=f"{label} was ACCEPTED: {src!r}"):
                    compile_fire_when(src)

    def test_no_refusal_from_compile_fire_when_escapes_as_anything_but_unsafe_expression(self):
        """The structural version of the above: sweep BOTH tables and pin the one exception type.

        A SyntaxError is not a ValueError, so ``except ValueError`` — the broadest guard a caller
        would reasonably write — must still be exhaustive over everything this function refuses.
        """
        for _label, src in HOSTILE_EXPRESSIONS + NON_EXPRESSION_INPUTS:
            with self.subTest(src=src):
                try:
                    compile_fire_when(src)
                except ValueError as exc:  # UnsafeExpression subclasses ValueError
                    self.assertIsInstance(exc, UnsafeExpression, f"{src!r} raised {type(exc).__name__}")
                except BaseException as exc:  # noqa: BLE001 - the point is that nothing else escapes
                    self.fail(f"{src!r} escaped as {type(exc).__name__}, not UnsafeExpression: {exc}")
                else:
                    self.fail(f"{src!r} was ACCEPTED")

    def test_a_non_str_fire_when_is_the_same_refusal_not_a_type_error(self):
        """The programmatic door: ``goal.state['fire_when'] = 3`` never passes the authoring
        firewall's ``isinstance(..., str)`` check, so it can only arrive at firing time — where
        there is no handler. It refuses as UnsafeExpression like everything else.

        ``bytes`` is deliberately absent: ``compile`` accepts it and produces the SAME tree, which
        the same whitelist then validates, so it is a different encoding of a legal input rather
        than a hole.
        """
        for bad in (None, 3, 3.5, ["n == 3"], {"n": 3}):
            with self.subTest(value=bad):
                with self.assertRaises(UnsafeExpression):
                    compile_fire_when(bad)
        self.assertTrue(compile_fire_when(b"n == 3")({"n": 3, "t": 0.0}))  # …and bytes still works

    def test_unsafe_expression_is_a_value_error(self):
        """Callers guard with ``except ValueError``; the class declaration at :155 must hold."""
        self.assertTrue(issubclass(UnsafeExpression, ValueError))

    def test_a_short_circuited_comparison_chain_is_still_validated_up_front(self):
        """The interpreter alone would let ``evil`` through; only the walk at :196 catches it.

        ``_eval_node``'s Compare loop returns as soon as one comparison is false, so with ``n = 5``
        it never evaluates the third operand. That is exactly why the validator walks the WHOLE tree
        before any evaluation — and why this must fail at compile time, not at some later ``n``.
        """
        tree = ast.parse("n < 0 < evil", mode="eval")
        self.assertIs(_eval_node(tree.body, {"n": 5}), False)  # the interpreter never sees `evil`
        with self.assertRaises(UnsafeExpression):
            compile_fire_when("n < 0 < evil")

    def test_a_dead_boolean_branch_is_still_validated_up_front(self):
        """``n == 999 and n.real`` — python would short-circuit; the validator does not care."""
        with self.assertRaises(UnsafeExpression):
            compile_fire_when("n == 999 and n.real")
        with self.assertRaises(UnsafeExpression):
            compile_fire_when("n == 3 or __import__('os').system('x')")
        # And the interpreter is not short-circuiting either: BoolOp evaluates every operand (:160)
        # before folding, so even a hand-built tree cannot hide a node in a dead branch.
        dead = ast.parse("n == 999 and n.real", mode="eval")
        with self.assertRaises(UnsafeExpression):
            _eval_node(dead.body, {"n": 1})

    def test_eval_and_exec_are_never_invoked_and_compile_only_parses(self):
        """The security claim in the docstring at :187, asserted rather than trusted."""
        with _no_code_execution() as flags_seen:
            pred = compile_fire_when("n == 3 and t < 600")
            self.assertTrue(pred({"n": 3, "t": 1.0}))  # the predicate itself must not eval either
            self.assertFalse(pred({"n": 2, "t": 1.0}))
            for _, expr in HOSTILE_EXPRESSIONS:
                with self.assertRaises(UnsafeExpression):
                    compile_fire_when(expr)
        self.assertTrue(flags_seen, "ast.parse never reached builtins.compile — the guard was blind")
        self.assertTrue(all(f & ast.PyCF_ONLY_AST for f in flags_seen))


class TestAuthoringFirewallDropsHostileFireWhen(unittest.TestCase):
    """``coerce_authored_behavior`` :1972-1978 — the LLM-facing door onto compile_fire_when."""

    def test_a_valid_authored_fire_when_reaches_goal_state(self):
        """Positive control for the drop tests below."""
        out = coerce_authored_behavior({"accumulate": {"fire_when": "n == 3", "window_s": 600}})
        self.assertEqual(out["fire_when"], "n == 3")
        self.assertEqual(out["window_s"], 600.0)

    def test_every_hostile_authored_fire_when_is_dropped(self):
        for node_class, expr in HOSTILE_EXPRESSIONS:
            with self.subTest(node=node_class, expr=expr):
                out = coerce_authored_behavior({"accumulate": {"fire_when": expr, "window_s": 600}})
                self.assertNotIn("fire_when", out, f"{node_class} reached goal.state: {expr!r}")
                self.assertNotIn("window_s", out)  # the window rides along only with a valid expr

    def test_a_non_expression_authored_fire_when_is_dropped_not_raised(self):
        """BUG-D-05-01, closed. The firewall's docstring says anything malformed is DROPPED.

        The whole point of the firewall is that untrusted conductor output can never crash the
        caller: it is validated, and on anything it does not like the goal falls back to a plain
        watch. A `fire_when` of "import os" is exactly the LLM output this is here to absorb — and
        it used to abort the whole authoring call instead, so the user's watch was never created.

        Fixed at the authority (``compile_fire_when`` converts the SyntaxError), not by widening
        this ``except``: the two FIRING-TIME callers reach the same function with no handler at all,
        and only fixing the authority covers them.
        """
        for label, src in NON_EXPRESSION_INPUTS:
            with self.subTest(input=label):
                out = coerce_authored_behavior({"accumulate": {"fire_when": src}})
                self.assertNotIn("fire_when", out, label)

    def test_a_non_expression_fire_when_degrades_the_live_tick_instead_of_crashing_it(self):
        """The half the firewall never covered: ``goal.state['fire_when']`` set programmatically.

        ``BehaviorExecutor._fire_when`` calls ``compile_fire_when`` with no try/except on the firing
        path, so a fire_when that reached ``goal.state`` by any door other than the firewall used to
        raise SyntaxError straight out of a live tick. It is now the module's documented refusal, so
        a caller that wants to degrade can actually catch it.
        """
        ex = BehaviorExecutor(Gate())
        g = Goal(id="g", condition="c", say="s")
        g.state["fire_when"] = "import os"
        with self.assertRaises(UnsafeExpression):
            ex.fire_judged(g, YES, ("g", "e"), program_of(g))


if __name__ == "__main__":
    unittest.main()
