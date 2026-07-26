"""Structural invariant: every judge call crosses the provider execution boundary.

``Percept._timed_judge`` remains the metering/fail-closed wrapper, but provider code now executes in
``_invoke_provider_call`` so the live runtime Task is outside provider reach and process teardown
finishes before the result is returned.  No ``<expr>.judge(...)`` call may remain in the parent.

This test parses ``agent.py`` with :mod:`ast`: it rejects every direct ``.judge(...)`` call and pins
that ``_timed_judge`` passes an authenticated operation name plus adapter identity to
``_invoke_provider_call``. The parent must not even retrieve an untrusted ``adapter.judge`` bound
method before the process boundary.
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


def _provider_boundary_calls() -> list[ast.Call]:
    return [
        node
        for node in ast.walk(_TREE)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "_invoke_provider_call"
    ]


class NoDirectJudgeCallsTest(unittest.TestCase):
    def test_no_parent_code_calls_the_adapter_directly(self):
        self.assertEqual(
            _judge_call_lines(),
            [],
            msg=(
                "Every live judge call must route through _invoke_provider_call; "
                "the parent must contain no direct `<expr>.judge(...)` call."
            ),
        )

    def test_timed_judge_routes_authenticated_operation_through_boundary(self):
        lo, hi = _timed_judge_span()
        boundary_calls = [
            node
            for node in _provider_boundary_calls()
            if lo <= node.lineno <= hi
        ]
        judge_references = [
            node
            for node in ast.walk(_TREE)
            if isinstance(node, ast.Attribute)
            and node.attr == "judge"
            and lo <= node.lineno <= hi
        ]
        operation_keywords = [
            keyword
            for call in boundary_calls
            for keyword in call.keywords
            if keyword.arg == "operation"
            and isinstance(keyword.value, ast.Constant)
            and keyword.value.value == "judge"
        ]
        adapter_keywords = [
            keyword
            for call in boundary_calls
            for keyword in call.keywords
            if keyword.arg == "adapter"
            and isinstance(keyword.value, ast.Name)
            and keyword.value.id == "adapter"
        ]
        self.assertTrue(
            boundary_calls and operation_keywords and adapter_keywords,
            "expected _timed_judge to pass operation='judge' and the "
            "adapter identity to _invoke_provider_call",
        )
        self.assertEqual(
            judge_references,
            [],
            "the parent must not retrieve adapter.judge before containment",
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
