"""The reflex judge request is TERSE and BOUNDED; the deliberate one is unchanged.

Streaming decomposition of the sub-1.5s reflex gate showed the miss was in the GENERATION half, not
in ingestion: a shipped ``max_tokens=1024`` "headroom" plus a prose ``evidence`` field let a route
spend seconds emitting text *after* it already knew the answer. These tests pin the fix as a
property of the ROUTE — ``JudgePool(mode='reflex')`` stamps a reserved context key, and a
verbosity-aware backend asks only for what the gate reads.

No provider or network calls: every completion is an injected fake, exactly like the existing
openrouter/gemini backend tests.
"""

from __future__ import annotations

import asyncio
import json
import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from percept.adapters import Verdict
from percept.backends.anthropic import (
    _JUDGE_SYSTEM,
    _JUDGE_SYSTEM_REFLEX,
    _REFLEX_MAX_TOKENS,
    _REFLEX_STRUCTURED_MAX_TOKENS,
    _parse_verdict,
)
from percept.backends.gemini import GeminiVision
from percept.backends.judge_pool import (
    LATENCY_MODE_CONTEXT_KEY,
    JudgeCapability,
    JudgePool,
    JudgeRoute,
)
from percept.backends.openrouter import OpenRouterVision
from percept.gate import ASK, FIRE, Gate

_DELIBERATE_MAX_TOKENS = 1024


# --------------------------------------------------------------------------------------------
# fakes
# --------------------------------------------------------------------------------------------


class _FakeCompletions:
    def __init__(self, content: str) -> None:
        self._content = content
        self.calls: list[dict] = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self._content))]
        )


class _FakeClient:
    def __init__(self, content: str) -> None:
        self.chat = SimpleNamespace(completions=_FakeCompletions(content))


def _settings(model: str = "qwen/qwen3-vl-8b-instruct"):
    return SimpleNamespace(models=SimpleNamespace(vision_model=model))


def _openrouter(content: str) -> tuple[OpenRouterVision, _FakeClient]:
    client = _FakeClient(content)
    return OpenRouterVision(_settings(), client=client), client


def _route(adapter, *, shape: str = "single_image") -> JudgeRoute:
    return JudgeRoute(
        "primary",
        adapter,
        JudgeCapability(
            evidence_shapes=frozenset({shape}),
            latency_slo_s=1.5,
            modes=frozenset({"reflex", "deliberate"}),
        ),
    )


def _pool(adapter, *, mode: str, shape: str = "single_image") -> JudgePool:
    return JudgePool(
        [_route(adapter, shape=shape)],
        mode=mode,
        deadline_s=1.5,
        max_parallel=1,
    )


def _sent(client: _FakeClient) -> dict:
    assert len(client.chat.completions.calls) == 1
    return client.chat.completions.calls[0]


def _system_of(sent: dict) -> str:
    return next(m["content"] for m in sent["messages"] if m["role"] == "system")


def _prompt_of(sent: dict) -> str:
    user = next(m for m in sent["messages"] if m["role"] == "user")
    return " ".join(
        block["text"] for block in user["content"] if block.get("type") == "text"
    )


TERSE_REPLY = json.dumps({"satisfied": True, "confidence": 0.87})
VERBOSE_REPLY = json.dumps(
    {"satisfied": True, "confidence": 0.87, "evidence": "a hand closes the lid"}
)


# --------------------------------------------------------------------------------------------
# the pool publishes the route's latency mode
# --------------------------------------------------------------------------------------------


class _RecordingJudge:
    def __init__(self) -> None:
        self.contexts: list[dict | None] = []

    async def judge(self, _condition, _frame, **kwargs):
        self.contexts.append(kwargs.get("context"))
        return Verdict(satisfied=True, confidence=0.9, evidence="ok")


class JudgePoolPublishesLatencyMode(unittest.TestCase):
    def test_reflex_pool_stamps_the_mode_without_mutating_the_caller_context(self):
        judge = _RecordingJudge()
        caller_context = {"extractor": {"plate": "AB-12"}}
        asyncio.run(
            _pool(judge, mode="reflex").judge("x?", b"jpeg", context=caller_context)
        )
        self.assertEqual(
            judge.contexts[0][LATENCY_MODE_CONTEXT_KEY],
            "reflex",
        )
        # the specialist fact still rides along, and the caller's dict is untouched
        self.assertEqual(judge.contexts[0]["extractor"], {"plate": "AB-12"})
        self.assertEqual(caller_context, {"extractor": {"plate": "AB-12"}})

    def test_reflex_pool_stamps_the_mode_when_the_caller_passed_no_context(self):
        judge = _RecordingJudge()
        asyncio.run(_pool(judge, mode="reflex").judge("x?", b"jpeg"))
        self.assertEqual(judge.contexts[0], {LATENCY_MODE_CONTEXT_KEY: "reflex"})

    def test_deliberate_pool_leaves_the_context_exactly_as_the_caller_sent_it(self):
        judge = _RecordingJudge()
        asyncio.run(_pool(judge, mode="deliberate").judge("x?", b"jpeg"))
        self.assertIsNone(judge.contexts[0])

        judge = _RecordingJudge()
        caller_context = {"event_chain": True}
        asyncio.run(
            _pool(judge, mode="deliberate", shape="video_clip").judge(
                "x?", b"jpeg", context=caller_context
            )
        )
        self.assertEqual(judge.contexts[0], {"event_chain": True})
        self.assertNotIn(LATENCY_MODE_CONTEXT_KEY, judge.contexts[0])


# --------------------------------------------------------------------------------------------
# OpenRouter: the shipped judge request
# --------------------------------------------------------------------------------------------


class OpenRouterReflexRequest(unittest.TestCase):
    def test_reflex_request_is_bounded_and_asks_for_no_prose(self):
        adapter, client = _openrouter(TERSE_REPLY)
        asyncio.run(_pool(adapter, mode="reflex").judge("the lid is closed", b"jpeg"))

        sent = _sent(client)
        self.assertEqual(sent["max_tokens"], _REFLEX_MAX_TOKENS)
        self.assertLessEqual(sent["max_tokens"], 32)
        self.assertEqual(_system_of(sent), _JUDGE_SYSTEM_REFLEX)

        prompt = _prompt_of(sent)
        self.assertIn('STRICT JSON {"satisfied", "confidence"}', prompt)
        self.assertNotIn('"evidence"}', prompt)
        # neither half of the request may still solicit a written reason
        self.assertNotIn("<short reason>", _system_of(sent))
        self.assertIn("no prose", prompt)
        # ...but the grounded question itself is unchanged
        self.assertIn("the lid is closed", prompt)

    def test_reflex_request_still_demands_confidence(self):
        # the Gate is the sole firing authority and compares confidence to its threshold; a
        # request that made confidence optional would silently turn every verdict into an ASK.
        adapter, client = _openrouter(TERSE_REPLY)
        asyncio.run(_pool(adapter, mode="reflex").judge("x?", b"jpeg"))
        sent = _sent(client)
        self.assertIn('"confidence"', _prompt_of(sent))
        self.assertIn('"confidence": <0..1>', _system_of(sent))
        self.assertIn("REQUIRED", _system_of(sent))

    def test_deliberate_request_is_byte_for_byte_the_direct_request(self):
        pooled_adapter, pooled_client = _openrouter(VERBOSE_REPLY)
        asyncio.run(_pool(pooled_adapter, mode="deliberate").judge("the lid is closed", b"jpeg"))

        direct_adapter, direct_client = _openrouter(VERBOSE_REPLY)
        asyncio.run(direct_adapter.judge("the lid is closed", b"jpeg"))

        self.assertEqual(_sent(pooled_client), _sent(direct_client))
        sent = _sent(pooled_client)
        self.assertEqual(sent["max_tokens"], _DELIBERATE_MAX_TOKENS)
        self.assertEqual(_system_of(sent), _JUDGE_SYSTEM)
        self.assertIn('{"satisfied", "confidence", "evidence"}', _prompt_of(sent))

    def test_reflex_occurrence_request_keeps_every_field_cognition_reads(self):
        # phase/action_present/events are NOT prose: behavior.normalize_event_chain_verdict
        # abstains on an inconsistent tuple and the occurrence ledger counts events. A reflex
        # occurrence route therefore drops only the free-text evidence field.
        adapter, client = _openrouter(
            json.dumps(
                {
                    "satisfied": True,
                    "confidence": 0.93,
                    "action_present": True,
                    "phase": "during",
                    "events": [{"t_start": 0.1, "t_end": 0.4, "label": "lid closes"}],
                }
            )
        )
        verdict = asyncio.run(
            _pool(adapter, mode="reflex", shape="video_clip").judge(
                "one discrete closing",
                b"latest",
                frames=[b"earlier"],
                context={"event_chain": True},
            )
        )

        sent = _sent(client)
        self.assertEqual(sent["max_tokens"], _REFLEX_STRUCTURED_MAX_TOKENS)
        self.assertLess(sent["max_tokens"], _DELIBERATE_MAX_TOKENS)
        prompt = _prompt_of(sent)
        self.assertIn('"action_present", "phase", "events"', prompt)
        self.assertIn("List every grounded occurrence", prompt)
        self.assertIn('omit "evidence"', prompt)
        self.assertNotIn('"confidence", "evidence"', prompt)
        # and the structured contract still round-trips into the verdict
        self.assertEqual((verdict.phase, verdict.action_present), ("during", True))
        self.assertEqual(len(verdict.events), 1)

    def test_reflex_claim_mode_keeps_the_correction_channel_and_its_budget(self):
        # A claim watch exists to catch a CORRECTION; the ledger reads operation/value off the
        # reply. Trimming that to hit a deadline would silently drop the thing being watched for,
        # so claim mode opts out of terseness even on a reflex route.
        adapter, client = _openrouter(
            json.dumps(
                {
                    "satisfied": True,
                    "confidence": 0.9,
                    "evidence": "buyer restates the price",
                    "operation": "replace",
                    "value": "$25K",
                }
            )
        )
        verdict = asyncio.run(
            _pool(adapter, mode="reflex").judge(
                "the price is stated",
                b"jpeg",
                context={"claim_mode": True},
            )
        )

        sent = _sent(client)
        self.assertEqual(sent["max_tokens"], _DELIBERATE_MAX_TOKENS)
        self.assertEqual(_system_of(sent), _JUDGE_SYSTEM)
        prompt = _prompt_of(sent)
        self.assertIn('{"satisfied", "confidence", "evidence"}', prompt)
        self.assertIn("CLAIMED VALUE", prompt)
        self.assertEqual((verdict.operation, verdict.value), ("replace", "$25K"))

    def test_a_directly_constructed_adapter_is_untouched_by_the_mechanism(self):
        # No pool, no stamp: an SDK user calling the adapter straight keeps the verbose contract.
        adapter, client = _openrouter(VERBOSE_REPLY)
        asyncio.run(adapter.judge("x?", b"jpeg"))
        self.assertEqual(_sent(client)["max_tokens"], _DELIBERATE_MAX_TOKENS)
        self.assertEqual(_system_of(_sent(client)), _JUDGE_SYSTEM)


# --------------------------------------------------------------------------------------------
# Gemini: the same policy, so the two providers stay coherent
# --------------------------------------------------------------------------------------------


class _GeminiResponse:
    def __init__(self, text: str) -> None:
        self.text = text
        self.usage_metadata = None


class _GeminiModels:
    def __init__(self, text: str) -> None:
        self._text = text
        self.calls: list[dict] = []

    async def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        return _GeminiResponse(self._text)


class _GeminiClient:
    def __init__(self, text: str) -> None:
        self.models = _GeminiModels(text)
        self.aio = SimpleNamespace(models=self.models)


def _lazy_google() -> dict[str, types.ModuleType]:
    class Part:
        @staticmethod
        def from_bytes(**kwargs: object) -> object:
            return ("part", kwargs)

    class GenerateContentConfig:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

    class HttpOptions:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

    google = types.ModuleType("google")
    genai = types.ModuleType("google.genai")
    google_types = types.ModuleType("google.genai.types")
    genai.types = google_types
    google.genai = genai
    google_types.Part = Part
    google_types.GenerateContentConfig = GenerateContentConfig
    google_types.HttpOptions = HttpOptions
    return {
        "google": google,
        "google.genai": genai,
        "google.genai.types": google_types,
    }


def _gemini_config(mode: str, reply: str) -> dict:
    client = _GeminiClient(reply)
    adapter = GeminiVision(client=client)
    with patch.dict(sys.modules, _lazy_google()):
        asyncio.run(_pool(adapter, mode=mode).judge("the lid is closed", b"jpeg"))
    assert len(client.models.calls) == 1
    return client.models.calls[0]["config"].kwargs


class GeminiMatchesTheOpenRouterPolicy(unittest.TestCase):
    def test_reflex_request_is_bounded_and_asks_for_no_prose(self):
        config = _gemini_config("reflex", TERSE_REPLY)
        self.assertEqual(config["max_output_tokens"], _REFLEX_MAX_TOKENS)
        self.assertEqual(config["system_instruction"], _JUDGE_SYSTEM_REFLEX)

    def test_deliberate_request_keeps_the_historical_headroom_and_prose(self):
        config = _gemini_config("deliberate", VERBOSE_REPLY)
        self.assertEqual(config["max_output_tokens"], _DELIBERATE_MAX_TOKENS)
        self.assertEqual(config["system_instruction"], _JUDGE_SYSTEM)


# --------------------------------------------------------------------------------------------
# the parser accepts both shapes and still fails CLOSED
# --------------------------------------------------------------------------------------------


class ParserAcceptsTerseAndVerbose(unittest.TestCase):
    def test_terse_shape_parses_into_a_usable_verdict(self):
        verdict = _parse_verdict('{"satisfied":false,"confidence":0.9}')
        self.assertFalse(verdict.satisfied)
        self.assertAlmostEqual(verdict.confidence, 0.9)
        # evidence is absent, not invalid: the contract says Verdict.evidence is a str
        self.assertEqual(verdict.evidence, "")
        self.assertIsInstance(verdict.evidence, str)

    def test_verbose_shape_still_parses_unchanged(self):
        verdict = _parse_verdict(
            '{"satisfied":true,"confidence":0.82,"evidence":"the lid is down"}'
        )
        self.assertTrue(verdict.satisfied)
        self.assertAlmostEqual(verdict.confidence, 0.82)
        self.assertEqual(verdict.evidence, "the lid is down")

    def test_a_provider_that_ignores_the_terse_ask_is_still_parsed(self):
        # the ask is an instruction, not a guarantee — prose must never become a parse miss
        verdict = _parse_verdict(
            'Sure! ```json\n{"satisfied": true, "confidence": 0.75, '
            '"evidence": "the hand withdraws"}\n```'
        )
        self.assertTrue(verdict.satisfied)
        self.assertAlmostEqual(verdict.confidence, 0.75)

    def test_a_reply_truncated_by_the_small_budget_salvages_both_decisive_fields(self):
        verdict = _parse_verdict('{"satisfied": true, "confidence": 0.88, "evidence": "a hand wa')
        self.assertTrue(verdict.satisfied)
        self.assertAlmostEqual(verdict.confidence, 0.88)

    def test_malformed_replies_fail_closed_to_unknown_not_to_a_verdict(self):
        # ASK is the only safe landing: it neither fires nor touches the gate's rising edge.
        for text in (
            "",
            "not json at all",
            "{",
            "[]",
            '{"satisfied": true}',  # no confidence -> undecidable, never a confident yes
            '{"confidence": 0.99}',  # no answer    -> must never read as a confident NO
            '{"confidence": 0.99, "evidence": "the lid is up"}',
        ):
            with self.subTest(text=text):
                verdict = _parse_verdict(text)
                self.assertEqual(verdict.confidence, 0.0)
                self.assertEqual(
                    Gate(threshold=0.7).decide("k", verdict).action,
                    ASK,
                )

    def test_an_answerless_reply_never_re_arms_the_gate(self):
        # Regression: a stray {"confidence": ...} used to parse as a CONFIDENT NO, which re-arms
        # the rising edge — so the next true read could mint a duplicate fire off a parse miss.
        gate = Gate(threshold=0.7, require_false_first=True)
        self.assertEqual(
            gate.decide("lid", _parse_verdict('{"confidence": 0.99}')).action,
            ASK,
        )
        outcome = gate.decide(
            "lid", _parse_verdict('{"satisfied": true, "confidence": 0.95}')
        )
        self.assertNotEqual(outcome.action, FIRE)


# --------------------------------------------------------------------------------------------
# confidence survives the reflex path all the way to the firing authority
# --------------------------------------------------------------------------------------------


class ConfidenceReachesTheGate(unittest.TestCase):
    def _reflex_verdict(self, reply: str) -> Verdict:
        adapter, _ = _openrouter(reply)
        return asyncio.run(_pool(adapter, mode="reflex").judge("the lid is closed", b"jpeg"))

    def test_a_confident_terse_yes_fires(self):
        verdict = self._reflex_verdict(TERSE_REPLY)
        self.assertAlmostEqual(verdict.confidence, 0.87)
        gate = Gate(threshold=0.7)
        gate.decide("lid", Verdict(satisfied=False, confidence=0.99, evidence=""))
        self.assertEqual(gate.decide("lid", verdict).action, FIRE)

    def test_an_unconfident_terse_yes_asks_instead_of_firing(self):
        verdict = self._reflex_verdict(json.dumps({"satisfied": True, "confidence": 0.55}))
        self.assertAlmostEqual(verdict.confidence, 0.55)
        outcome = Gate(threshold=0.7).decide("lid", verdict)
        self.assertEqual(outcome.action, ASK)
        self.assertEqual(outcome.reason, "low_confidence")

    def test_a_terse_verdict_is_a_valid_pool_verdict_not_a_malformed_one(self):
        # JudgePool._valid_verdict requires evidence to be a str; the terse shape leaves it "",
        # so a reflex reply must route as decisive rather than degrade to the UNKNOWN fallback.
        adapter, _ = _openrouter(TERSE_REPLY)
        pool = _pool(adapter, mode="reflex")
        asyncio.run(pool.judge("the lid is closed", b"jpeg"))
        receipt = pool.receipts()[-1]
        self.assertEqual(receipt.outcome, "decisive")
        self.assertAlmostEqual(receipt.confidence, 0.87)


if __name__ == "__main__":
    unittest.main()
