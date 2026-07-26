"""Judge metering reconciliation — the OpenRouter seam of the existing metering path.

HEAD already wires metering end-to-end: ``Verdict.usage`` is a real dataclass FIELD (default
``{}``, never a dynamically-added attribute — do NOT ``hasattr`` it away), and
``observability.judge_usage`` + ``Funnel.record_judge`` read it (covered by
``tests/test_judge_usage_metering.py``, not duplicated here). This file covers what a prior attempt
got wrong and this suite fixes for real:

1. an OpenRouter judge call now REQUESTS the vendor's opt-in usage-accounting extension
   (``extra_body={"usage": {"include": True}}``) — without it OpenRouter never returns ``usage`` at
   all, so cost data silently stayed absent forever.
2. ``_usage_of`` also reads OpenRouter's ``cost`` field (present only when that extension is on) into
   ``Verdict.usage["cost_usd"]``.
3. the OpenRouter model id resolves an explicit ``vendor/model`` id from settings, and otherwise falls
   back to OpenRouter's own default — never a leaked Claude/Gemini id from the shared
   ``vision_model`` setting.
"""
from __future__ import annotations

import asyncio
import unittest
from types import SimpleNamespace as NS

from percept.adapters import Verdict
from percept.backends.openrouter import OpenRouterVision, _resolve_model, _usage_of


class _FakeCompletions:
    def __init__(self, content: str, usage: object | None = None) -> None:
        self._content = content
        self._usage = usage
        self.calls: list[dict] = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        resp = NS(choices=[NS(message=NS(content=self._content))])
        if self._usage is not None:
            resp.usage = self._usage
        return resp


class _FakeClient:
    def __init__(self, content: str, usage: object | None = None) -> None:
        self.chat = NS(completions=_FakeCompletions(content, usage))


def _settings(model: str | None = None):
    return NS(models=NS(vision_model=model))


class VerdictUsageIsAFieldTests(unittest.TestCase):
    """Pin HEAD's reality: ``usage`` is a real, always-present, defaulted field — not an optional
    dynamic attribute. A stale prior assumption (``assertFalse(hasattr(verdict, "usage"))``) is WRONG
    against this contract and must never come back."""

    def test_usage_is_a_real_field_present_on_every_verdict(self):
        v = Verdict(satisfied=True, confidence=0.5)
        self.assertTrue(hasattr(v, "usage"))  # deliberately opposite the stale assertion
        self.assertIn("usage", Verdict.__dataclass_fields__)

    def test_default_usage_is_empty_and_falsy(self):
        v = Verdict(satisfied=True, confidence=0.5)
        self.assertEqual(v.usage, {})
        self.assertFalse(v.usage)  # falsy default -> judge_usage() naturally yields {}

    def test_usage_is_independent_per_instance(self):
        # a mutable default must not be shared across instances (dataclass field(default_factory=dict))
        a, b = Verdict(satisfied=True, confidence=0.5), Verdict(satisfied=False, confidence=0.1)
        a.usage["input_tokens"] = 10
        self.assertEqual(b.usage, {})


class OpenRouterUsageExtensionTests(unittest.TestCase):
    """The genuinely-additive metering fix: request + parse OpenRouter's usage-accounting extension."""

    def test_judge_call_requests_usage_accounting_extension(self):
        client = _FakeClient('{"satisfied": true, "confidence": 0.9, "evidence": "seen"}')
        v = OpenRouterVision(_settings("qwen/qwen3-vl-32b-instruct"), client=client)
        asyncio.run(v.judge("x?", b"frame"))
        sent = client.chat.completions.calls[0]
        self.assertEqual(sent.get("extra_body"), {"usage": {"include": True}})

    def test_process_qa_call_also_requests_usage_accounting_extension(self):
        client = _FakeClient('{"updated_summary": "s", "satisfied": false, "confidence": 0.5}')
        v = OpenRouterVision(_settings("qwen/qwen3-vl-32b-instruct"), client=client)
        asyncio.run(v.judge("x?", b"frame", context={"process_qa": True, "running_answer": ""}))
        sent = client.chat.completions.calls[0]
        self.assertEqual(sent.get("extra_body"), {"usage": {"include": True}})

    def test_usage_of_reads_cost_when_extension_present(self):
        resp = NS(usage=NS(prompt_tokens=500, completion_tokens=15, cost=0.00042))
        self.assertEqual(_usage_of(resp), {"input_tokens": 500, "output_tokens": 15, "cost_usd": 0.00042})

    def test_usage_of_omits_cost_when_absent(self):
        # unchanged shape when the extension wasn't requested/echoed (e.g. an older test double)
        resp = NS(usage=NS(prompt_tokens=500, completion_tokens=15))
        self.assertEqual(_usage_of(resp), {"input_tokens": 500, "output_tokens": 15})

    def test_judge_populates_verdict_usage_including_cost_end_to_end(self):
        usage = NS(prompt_tokens=1200, completion_tokens=40, cost=0.0021)
        client = _FakeClient('{"satisfied": true, "confidence": 0.9, "evidence": "seen"}', usage=usage)
        v = OpenRouterVision(_settings("qwen/qwen3-vl-32b-instruct"), client=client)
        verdict = asyncio.run(v.judge("x?", b"frame"))
        self.assertEqual(verdict.usage, {"input_tokens": 1200, "output_tokens": 40, "cost_usd": 0.0021})

    def test_process_qa_verdict_also_carries_usage(self):
        usage = NS(prompt_tokens=300, completion_tokens=20)
        client = _FakeClient('{"updated_summary": "s", "satisfied": false, "confidence": 0.5}', usage=usage)
        v = OpenRouterVision(_settings("qwen/qwen3-vl-32b-instruct"), client=client)
        verdict = asyncio.run(v.judge("x?", b"frame", context={"process_qa": True, "running_answer": ""}))
        self.assertEqual(verdict.usage, {"input_tokens": 300, "output_tokens": 20})


class OpenRouterModelResolutionTests(unittest.TestCase):
    """The real bug: a shared ``vision_model`` setting leaking a non-OpenRouter id into OpenRouter."""

    def test_explicit_vendor_model_id_is_respected(self):
        self.assertEqual(_resolve_model(_settings("qwen/qwen2.5-vl-72b-instruct")),
                         "qwen/qwen2.5-vl-72b-instruct")
        self.assertEqual(_resolve_model(_settings("anthropic/claude-3-opus")), "anthropic/claude-3-opus")

    def test_leaked_bare_claude_model_id_falls_back_to_openrouter_default(self):
        # "claude-opus-4-8" is the SHARED cross-backend default (ModelSettings.vision_model) — it has no
        # "/" and is not an OpenRouter id. This is the actual production bug: it must NOT be sent as-is.
        self.assertEqual(_resolve_model(_settings("claude-opus-4-8")), "qwen/qwen3-vl-32b-instruct")

    def test_missing_vision_model_falls_back_to_openrouter_default(self):
        self.assertEqual(_resolve_model(NS(models=NS())), "qwen/qwen3-vl-32b-instruct")
        self.assertEqual(_resolve_model(NS()), "qwen/qwen3-vl-32b-instruct")

    def test_constructor_uses_resolved_model(self):
        v = OpenRouterVision(_settings("claude-opus-4-8"), client=object())
        self.assertEqual(v._model, "qwen/qwen3-vl-32b-instruct")
        v2 = OpenRouterVision(_settings("openai/gpt-5-mini"), client=object())
        self.assertEqual(v2._model, "openai/gpt-5-mini")


if __name__ == "__main__":
    unittest.main()
