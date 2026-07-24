"""P06-002 — structural invariant: ``_timed_judge`` is the ONE site every judge call passes through.

``Percept._timed_judge``'s docstring claims it is "the only site EVERY judge call passes through".
That is a load-bearing claim: the cost/latency wire, and (via ``_safe_timed_judge``) the fail-closed
boundary, only cover a judge call if the call actually goes through this wrapper. Before P06-002 the
transition path called ``adapter.judge`` directly, silently bypassing both. This test makes the
claim STRUCTURALLY true by reading ``agent.py`` source text and asserting no ``adapter.judge(`` /
``self.vision.judge(`` call exists anywhere in the module EXCEPT inside the ``_timed_judge`` body.
"""
from __future__ import annotations

import re
import unittest
from pathlib import Path

import percept.agent as agent_mod

_SOURCE = Path(agent_mod.__file__).read_text()
# The two ways a live judge call can be spelled at a dispatch site (the wrapper itself uses
# ``adapter.judge`` internally). A docstring reference like ``Vision.judge(`` matches neither.
_DIRECT_CALL = re.compile(r"\b(?:adapter|self\.vision)\.judge\(")
_METHOD_DEF = re.compile(r"^    (?:async )?def (\w+)\(")


def _timed_judge_span(source: str) -> tuple[int, int]:
    """The 1-based [start, end] line range of the ``_timed_judge`` method body."""
    lines = source.splitlines()
    start = None
    for i, line in enumerate(lines):
        m = _METHOD_DEF.match(line)
        if m and m.group(1) == "_timed_judge":
            start = i
            continue
        if start is not None and m:  # the next method at method indentation ends the body
            return start + 1, i  # 1-based, exclusive of the next def line
    assert start is not None, "could not locate _timed_judge in agent.py"
    return start + 1, len(lines)


class NoDirectJudgeCallsTest(unittest.TestCase):
    def test_only_timed_judge_calls_adapter_judge(self):
        lo, hi = _timed_judge_span(_SOURCE)
        offenders = []
        for lineno, line in enumerate(_SOURCE.splitlines(), start=1):
            if _DIRECT_CALL.search(line) and not (lo <= lineno <= hi):
                offenders.append((lineno, line.strip()))
        self.assertEqual(
            offenders, [],
            msg=(
                "Every live judge call must route through _timed_judge (directly or via "
                "_safe_timed_judge). Direct adapter.judge/self.vision.judge calls found OUTSIDE "
                f"the _timed_judge body (lines {lo}-{hi}):\n"
                + "\n".join(f"  line {n}: {t}" for n, t in offenders)
            ),
        )

    def test_timed_judge_body_does_contain_the_one_call(self):
        # Guard the guard: if the wrapper stopped calling adapter.judge the test above would pass
        # vacuously. Pin that the single sanctioned call really lives inside the span.
        lo, hi = _timed_judge_span(_SOURCE)
        body = "\n".join(_SOURCE.splitlines()[lo - 1:hi])
        self.assertRegex(body, r"\badapter\.judge\(")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
