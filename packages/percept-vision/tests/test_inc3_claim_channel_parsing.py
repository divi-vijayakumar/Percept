"""— the claim-mode correction channel in the three REAL judge backends.

The gap this closes: the claim/supersession machinery is fully
built and wired unconditionally in ``agent.perceive_judged`` — it reads ``verdict.operation``/``value``
— but NO shipping judge backend ever emitted those fields, so a live correction never mutated a claim
and no revision chain formed. This module proves the backend half:

* **prompt-snapshot** — a non-claim goal's assembled prompt is BYTE-FOR-BYTE the pre-1.5 prompt on
  every backend (the claim contract is appended ONLY under ``claim_mode``); and ``claim_mode`` appends
  the correction contract.
* **parser strictness** — ``claim_mode`` folds ``operation``/``value``/``normalized_value`` onto the
  verdict; an unknown operation degrades to ``"assert"`` (never invents a mutation); a missing operation
  defaults to ``"assert"``; ``asserted_by`` is never fabricated; malformed JSON takes the SHARED,
  unchanged salvage path; and a non-claim parse sets none of the claim fields (byte-for-byte).
"""
from __future__ import annotations

import unittest

from percept import Percept
from percept.backends import anthropic as an
from percept.backends import gemini as gm
from percept.backends import openrouter as orr
from percept.backends.fake import FakeVision
from percept.contracts.watchplan import WatchPlan
from percept.watchplan_apply import apply_watch_plan

FRAME = b"\x89PNG\r\n\x1a\n-fake-bytes"
COND = "a buyer states the approved budget"

# The EXACT single-frame, no-context, no-correction question the pre-1.5 prompt produced (the golden
# snapshot). If a future edit perturbs the non-claim prompt this string breaks — that is the point.
_PLAIN_SUFFIX = 'Answer as STRICT JSON {"satisfied", "confidence", "evidence"}.'
_ANTHROPIC_PLAIN = (
    f'Looking only at the region of interest: is this true right now — "{COND}"? ' + _PLAIN_SUFFIX
)
_OPENROUTER_PLAIN = f'Looking only at the image: is this true right now — "{COND}"? ' + _PLAIN_SUFFIX


def _anthropic_text(context=None, claim_mode=False):
    v = an.AnthropicVision(client=object())
    content = v._build_content(COND, FRAME, roi=None, exemplar=None, corrections=None,
                               frames=None, context=context, claim_mode=claim_mode)
    return content[-1]["text"]


def _openrouter_text(context=None):
    v = orr.OpenRouterVision(client=object())
    content = v._build_content(COND, FRAME, exemplar=None, corrections=None, frames=None, context=context)
    return content[-1]["text"]


# --------------------------------------------------------------------------------------------------
# Prompt snapshot — non-claim goals are byte-for-byte; claim_mode appends the contract.
# --------------------------------------------------------------------------------------------------


class TestPromptSnapshotByteForByte(unittest.TestCase):
    def test_anthropic_non_claim_is_byte_for_byte(self):
        self.assertEqual(_anthropic_text(context=None, claim_mode=False), _ANTHROPIC_PLAIN)

    def test_anthropic_non_claim_with_context_is_byte_for_byte(self):
        # A real (non-claim) specialist context still renders exactly as before — the reserved-key
        # filter is a no-op when no "claim_mode" key is present.
        text = _anthropic_text(context={"plate": "7XYZ123"}, claim_mode=False)
        self.assertIn("plate=7XYZ123", text)
        self.assertNotIn("claim_mode", text)
        self.assertTrue(text.endswith(_PLAIN_SUFFIX))

    def test_anthropic_claim_mode_appends_contract(self):
        text = _anthropic_text(context={"claim_mode": True}, claim_mode=True)
        # the plain prompt is still fully present (contract is APPENDED, not a replacement)...
        self.assertIn(_ANTHROPIC_PLAIN, text)
        # ...the correction contract is now advertised...
        self.assertIn('"operation"', text)
        self.assertIn('"replace"', text)
        self.assertIn('"retract"', text)
        # ...and the reserved control key never leaks in as a "specialist fact".
        self.assertNotIn("claim_mode=True", text)

    def test_openrouter_non_claim_is_byte_for_byte(self):
        self.assertEqual(_openrouter_text(context=None), _OPENROUTER_PLAIN)

    def test_gemini_reuses_the_shared_contract_constant(self):
        # Gemini builds its question inline but MUST share the identical correction contract, so the
        # parse/prompt semantics match across backends.
        self.assertIs(gm._CLAIM_MODE_CONTRACT, an._CLAIM_MODE_CONTRACT)


# --------------------------------------------------------------------------------------------------
# Parser strictness — claim fields fold only in claim_mode, unknown ops degrade to assert.
# --------------------------------------------------------------------------------------------------


class TestParserStrictness(unittest.TestCase):
    def test_non_claim_parse_sets_no_claim_fields(self):
        v = an._parse_verdict('{"satisfied": true, "confidence": 0.9, "evidence": "e"}')
        self.assertTrue(v.satisfied)
        self.assertEqual(v.operation, "assert")  # the dataclass default, untouched
        self.assertIsNone(v.value)
        self.assertIsNone(v.normalized_value)
        self.assertIsNone(v.asserted_by)

    def test_non_claim_ignores_operation_in_the_json(self):
        # Even if a reply carries operation/value, a NON-claim parse must ignore them (byte-for-byte):
        # the mutation channel only exists for claim-slot concerns.
        v = an._parse_verdict(
            '{"satisfied": true, "confidence": 0.9, "evidence": "e", "operation": "replace", "value": "$25K"}'
        )
        self.assertEqual(v.operation, "assert")
        self.assertIsNone(v.value)

    def test_claim_mode_folds_replace_and_value(self):
        v = an._parse_verdict(
            '{"satisfied": true, "confidence": 0.9, "evidence": "actually $25K", '
            '"operation": "replace", "value": "$25K", "normalized_value": 25000}',
            claim_mode=True,
        )
        self.assertEqual(v.operation, "replace")
        self.assertEqual(v.value, "$25K")
        self.assertEqual(v.normalized_value, 25000)
        self.assertIsNone(v.asserted_by)  # never fabricated

    def test_claim_mode_unknown_operation_degrades_to_assert(self):
        v = an._parse_verdict(
            '{"satisfied": true, "confidence": 0.8, "evidence": "e", "operation": "frobnicate", "value": "$9"}',
            claim_mode=True,
        )
        self.assertEqual(v.operation, "assert")  # never invents a mutation
        self.assertEqual(v.value, "$9")

    def test_claim_mode_missing_operation_defaults_assert(self):
        v = an._parse_verdict(
            '{"satisfied": true, "confidence": 0.8, "evidence": "budget is $40K", "value": "$40K"}',
            claim_mode=True,
        )
        self.assertEqual(v.operation, "assert")
        self.assertEqual(v.value, "$40K")

    def test_claim_mode_retract_carries_no_value(self):
        v = an._parse_verdict(
            '{"satisfied": true, "confidence": 0.7, "evidence": "never mind", "operation": "retract"}',
            claim_mode=True,
        )
        self.assertEqual(v.operation, "retract")
        self.assertIsNone(v.value)

    def test_claim_mode_operation_is_case_and_whitespace_tolerant(self):
        v = an._parse_verdict(
            '{"satisfied": true, "confidence": 0.9, "evidence": "e", "operation": " Replace "}',
            claim_mode=True,
        )
        self.assertEqual(v.operation, "replace")

    def test_malformed_json_takes_shared_salvage_path_unchanged(self):
        # A truncated reply (evidence string cut) still recovers satisfied/confidence via the SHARED
        # salvage path; the claim fields simply stay at their defaults (operation "assert").
        truncated = '{"satisfied": true, "confidence": 0.85, "evidence": "the budget is'
        v = an._parse_verdict(truncated, claim_mode=True)
        self.assertTrue(v.satisfied)
        self.assertAlmostEqual(v.confidence, 0.85)
        self.assertEqual(v.operation, "assert")

    def test_fully_unparsed_reply_is_low_confidence_unknown(self):
        v = an._parse_verdict("the model rambled and never produced json", claim_mode=True)
        self.assertFalse(v.satisfied)
        self.assertEqual(v.confidence, 0.0)
        self.assertEqual(v.operation, "assert")


# --------------------------------------------------------------------------------------------------
# Receipt capability warning — a claim_slot plan on a non-parsing backend warns (never rejects).
# --------------------------------------------------------------------------------------------------


class _NonParsingVision:
    """A minimal vision backend WITHOUT the ``PARSES_CLAIM_CHANNEL`` marker (a client's custom judge
    that does not emit the correction channel)."""

    async def judge(self, *a, **k):  # pragma: no cover - never called in these apply-time tests
        from percept.adapters import Verdict

        return Verdict(satisfied=False, confidence=0.0, evidence="")

    async def detect(self, *a, **k):
        return []

    async def detect_events(self, *a, **k):
        return []


_CLAIM_PLAN = WatchPlan(
    condition="a buyer states the approved budget",
    explicit={"Goal.state.identity_policy": "claim_slot"},
)


class TestReceiptCapabilityWarning(unittest.TestCase):
    def test_non_parsing_backend_warns_but_arms(self):
        core = Percept.create(vision=_NonParsingVision(), discover_plugins=False)
        receipt = apply_watch_plan(core, _CLAIM_PLAN)
        # The plan ARMS (a warning, not a rejection) ...
        self.assertEqual(len(receipt.watch_ids), 1)
        self.assertEqual(receipt.status, "applied_with_warnings")
        # ... and the machine-actionable warning code is present, scoped to the goal.
        codes = [w.code for w in receipt.warnings]
        self.assertIn("claim_mode_backend_unverified", codes)
        warn = next(w for w in receipt.warnings if w.code == "claim_mode_backend_unverified")
        self.assertEqual(warn.path, "Goal.state.identity_policy")
        self.assertEqual(warn.goal_id, receipt.watch_ids[0])

    def test_fake_backend_is_parsing_capable_no_warning(self):
        core = Percept.create(vision=FakeVision(), discover_plugins=False)
        receipt = apply_watch_plan(core, _CLAIM_PLAN)
        self.assertEqual(len(receipt.watch_ids), 1)
        self.assertNotIn(
            "claim_mode_backend_unverified", [w.code for w in receipt.warnings]
        )

    def test_non_claim_plan_never_warns_even_on_non_parsing_backend(self):
        core = Percept.create(vision=_NonParsingVision(), discover_plugins=False)
        receipt = apply_watch_plan(core, WatchPlan(condition="a person is in the zone"))
        self.assertNotIn(
            "claim_mode_backend_unverified", [w.code for w in receipt.warnings]
        )


if __name__ == "__main__":
    unittest.main()
