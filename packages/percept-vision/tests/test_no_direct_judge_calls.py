"""P06-002 — structural invariant: ``_timed_judge`` is the ONE site every judge call passes through.

``Percept._timed_judge``'s docstring claims it is "the only site EVERY judge call passes through".
That is load-bearing: the cost/latency wire and (via ``_safe_timed_judge``) the fail-closed boundary
only cover a judge call if the call actually goes through this wrapper. Before P06-002 the transition
path called the adapter directly, silently bypassing both.

This test makes the claim STRUCTURALLY true by PARSING ``agent.py`` with :mod:`ast` (not a regex — so
a ``.judge(`` inside a string, comment or docstring can never match, and a call under ANY receiver
name is caught, not just the two ``adapter`` / ``self.vision`` spellings). It asserts that every
``<expr>.judge(...)`` call node in the module lives inside the ``_timed_judge`` function body.
"""
from __future__ import annotations

import ast
import unittest
from pathlib import Path

import percept.agent as agent_mod

_SOURCE = Path(agent_mod.__file__).read_text()
_TREE = ast.parse(_SOURCE)


def _timed_judge_span() -> tuple[int, int]:
    """The 1-based [start, end] line range of the ``_timed_judge`` def (inclusive)."""
    for node in ast.walk(_TREE):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "_timed_judge":
            return node.lineno, (node.end_lineno or node.lineno)
    raise AssertionError("could not locate _timed_judge in agent.py")


def _judge_call_lines() -> list[int]:
    """Line of every ``<expr>.judge(...)`` CALL in agent.py — attribute name exactly ``judge``, so
    ``_timed_judge(`` / ``_safe_timed_judge(`` / ``judge_clip(`` / ``self._judge(`` are NOT matched
    (different attr names), and a ``.judge(`` appearing in a docstring is not a Call node at all."""
    return [
        node.lineno
        for node in ast.walk(_TREE)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "judge"
    ]


class NoDirectJudgeCallsTest(unittest.TestCase):
    def test_only_timed_judge_calls_the_adapter(self):
        lo, hi = _timed_judge_span()
        offenders = [ln for ln in _judge_call_lines() if not (lo <= ln <= hi)]
        self.assertEqual(
            offenders, [],
            msg=(
                "Every live judge call must route through _timed_judge (directly or via "
                "_safe_timed_judge). A `<expr>.judge(...)` call was found OUTSIDE the _timed_judge "
                f"body (lines {lo}-{hi}) at agent.py line(s): {offenders}"
            ),
        )

    def test_timed_judge_body_does_contain_the_one_call(self):
        # Guard the guard: if the wrapper stopped calling the adapter the test above would pass
        # vacuously. Pin that the sanctioned call really lives inside the span.
        lo, hi = _timed_judge_span()
        inside = [ln for ln in _judge_call_lines() if lo <= ln <= hi]
        self.assertTrue(inside, "expected a `<expr>.judge(` call inside the _timed_judge body")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
