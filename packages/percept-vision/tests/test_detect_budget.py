"""The ``detect`` leg has ONE output budget, and an unreadable reply is never silently empty.

Two things are pinned here, and they are two halves of one change.

THE BUDGET. The judge leg has had a shared, measured policy (``_judge_max_tokens``) since reflex
landed. ``detect`` never did: each adapter that makes a real call carried the number inline, and they
had drifted — gemini and anthropic at 512 against openrouter at 1024, so the same watch bought a 2x
different worst case depending on which backend resolved. :func:`_detect_max_tokens` is now the one
place, and the tests below fail if an adapter goes back to an inline literal.

The VALUE is deliberately unchanged at the incumbent 512 and is not tuned here. Latency tracks
output tokens (about ``0.85s + tokens/26``, validated within 8%), so the ceiling is a real lever —
but detect's measured p50 implies a typical reply near 74 tokens, so it binds only the tail, and
nothing has been re-measured since the scene-scope detect elision landed. Lowering it now would be
picking a number to suit a metric.

WHY THE PARSER HAD TO CHANGE FIRST. ``detect`` returns ``list[Observation]``, so a failure has one
shape available to it: the empty list — the same value an empty scene produces. A reply cut off at
the token cap is unparseable JSON, lands in that path, and is indistinguishable downstream from
"nothing was there". That is the exact collapse this project has already fixed twice (the deaf
``AcousticOnset`` bar; ``zero_admission_coverage``), and detect was the remaining instance. Lowering
the cap without fixing it first would have manufactured the bug rather than found it.

The parser still returns the empty list — the contract requires it — but it now says so out loud.
No provider or network calls: the parser is pure and is driven directly.
"""

from __future__ import annotations

import logging
import re
import unittest
from pathlib import Path

from percept.backends.anthropic import _detect_max_tokens, _parse_observations

_BACKENDS = Path(__file__).resolve().parents[1] / "percept" / "backends"


def _detect_body(module: str) -> str:
    """The source of ``module``'s ``async def detect``, up to the next SIBLING method.

    The sibling boundary is anchored to exactly four spaces of indent. ``detect`` nests its own
    ``async def _call()`` at eight, and a looser scan stops there — before the provider call that
    carries the token budget, which is the whole thing these tests read.
    """
    src = (_BACKENDS / f"{module}.py").read_text()
    start = src.index("    async def detect(self")
    nxt = re.search(r"\n {4}(?:async )?def |\n(?:class |@|\S)", src[start + 10:])
    return src[start:start + 10 + nxt.start()] if nxt else src[start:]


class TestOneSharedBudget(unittest.TestCase):
    """Every adapter that really calls a provider reads the SAME policy."""

    #: The three that make a real request. ``fake``/``fallback``/``judge_pool`` delegate or return
    #: ``[]``; ``gemini_live``'s detect returns ``[]`` unconditionally (filed as D-6) and so has no
    #: budget to share.
    CALLERS = ("anthropic", "gemini", "openrouter")

    def test_every_calling_adapter_uses_the_policy(self):
        for module in self.CALLERS:
            with self.subTest(module=module):
                self.assertIn("_detect_max_tokens()", _detect_body(module))

    def test_no_adapter_carries_an_inline_token_literal(self):
        # The regression this file exists for: openrouter shipped 1024 while the others shipped 512,
        # so the worst-case detect reply depended on which backend happened to resolve.
        for module in self.CALLERS:
            with self.subTest(module=module):
                inline = re.findall(r"max_(?:output_)?tokens\s*=\s*(\d+)", _detect_body(module))
                self.assertEqual(inline, [], f"{module}.detect re-inlined a token literal: {inline}")

    def test_the_budget_is_the_incumbent_value(self):
        # Pinned so a change to the number is a deliberate, reviewed line rather than a drift.
        self.assertEqual(_detect_max_tokens(), 512)


class TestAnUnreadableReplyIsNotSilentlyEmpty(unittest.TestCase):
    """The half that had to land before the cap could ever be lowered."""

    TRUNCATED = '[{"label":"person","box":[0.1,0.1,0.2,0.2]},{"label":"cu'

    def test_a_truncated_reply_returns_empty_but_warns(self):
        with self.assertLogs("percept.backends", level=logging.WARNING) as caught:
            self.assertEqual(_parse_observations(self.TRUNCATED), [])
        self.assertTrue(
            any("detect.unreadable" in line for line in caught.output),
            "a reply cut off mid-JSON must be distinguishable from an empty scene",
        )

    def test_an_empty_reply_stays_silent(self):
        # A model saying "nothing" is not a failure and must not cry wolf, or the warning above
        # becomes noise nobody reads -- which is how the deaf audio gate survived so long.
        for reply in ("", "   ", "\n"):
            with self.subTest(reply=reply):
                with self.assertNoLogs("percept.backends", level=logging.WARNING):
                    self.assertEqual(_parse_observations(reply), [])

    def test_a_partial_list_keeps_what_parsed_and_says_what_it_dropped(self):
        both = '[{"label":"a","box":[0,0,1,1]},{"label":"b"}]'  # second item has no box
        with self.assertLogs("percept.backends", level=logging.WARNING) as caught:
            out = _parse_observations(both)
        self.assertEqual([o.label for o in out], ["a"])
        self.assertTrue(any("detect.partial" in line for line in caught.output))
        self.assertTrue(
            any("dropped 1" in line and "kept 1" in line for line in caught.output),
            "the counts are the point: they say how much of the scene was lost",
        )

    def test_a_wholly_valid_reply_is_silent(self):
        good = '[{"label":"person","box":[0.1,0.2,0.3,0.4],"confidence":0.9}]'
        with self.assertNoLogs("percept.backends", level=logging.WARNING):
            out = _parse_observations(good)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].label, "person")
        self.assertAlmostEqual(out[0].confidence, 0.9)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
