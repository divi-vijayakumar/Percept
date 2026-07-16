"""PR 1.1 — token/cost metering must stay wired end-to-end.

Regression guard for a bug that shipped: the consumer side (``_timed_judge`` →
``judge_usage`` → ``Funnel.record_judge`` → OTel) was fully built, but NO backend annotated
``Verdict.usage``, so the token/cost counters read 0 forever. These tests assert both halves:
each backend's usage extractor reads its SDK's response shape, and a usage-annotated verdict
actually lands in the funnel's token counters through the agent's real judge path.
"""

import asyncio
from types import SimpleNamespace as NS

import pytest

import percept
from percept.adapters import Goal, Verdict
from percept.backends.anthropic import _usage_of as anthropic_usage
from percept.backends.gemini import _usage_of as gemini_usage
from percept.backends.openrouter import _usage_of as openrouter_usage
from percept.observability import judge_usage


def test_gemini_usage_extractor():
    resp = NS(usage_metadata=NS(prompt_token_count=1200, candidates_token_count=40))
    assert gemini_usage(resp) == {"input_tokens": 1200, "output_tokens": 40}


def test_anthropic_usage_extractor():
    assert anthropic_usage(NS(usage=NS(input_tokens=800, output_tokens=25))) == {
        "input_tokens": 800, "output_tokens": 25,
    }


def test_openrouter_usage_extractor():
    assert openrouter_usage(NS(usage=NS(prompt_tokens=500, completion_tokens=15))) == {
        "input_tokens": 500, "output_tokens": 15,
    }


@pytest.mark.parametrize("extractor", [gemini_usage, anthropic_usage, openrouter_usage])
def test_usage_extractors_degrade_to_empty(extractor):
    # A response without usage (fakes/tests / older SDK) must yield {} — the hook stays a no-op.
    assert extractor(NS()) == {}


class _UsageVision:
    """Minimal Vision adapter that annotates usage exactly as the real backends now do."""

    async def judge(self, condition, frame, *, roi=None, exemplar=None, corrections=None,
                    frames=None, audio=None, context=None):
        v = Verdict(satisfied=True, confidence=0.95, evidence="seen")
        v.usage = {"input_tokens": 1200, "output_tokens": 40}
        return v

    async def detect(self, frame, goals):
        return []

    async def detect_events(self, frames, goals):
        return []


def test_usage_flows_to_funnel_end_to_end():
    async def run():
        agent = percept.Percept.create(vision=_UsageVision(), discover_plugins=False)
        agent.add_goal(Goal(id="coffee", condition="the user is drinking coffee", say="hi"))
        await agent.perceive_judged(("image/jpeg", b"x"), source="camera", t=1.0)
        return agent.metrics.snapshot()

    snap = asyncio.run(run())
    assert snap.get("judge.tokens.input") == 1200.0
    assert snap.get("judge.tokens.output") == 40.0
    assert snap.get("judge.latency.count") == 1  # latency was already wired; assert it still is


def test_judge_usage_reads_annotated_verdict():
    v = Verdict(satisfied=True, confidence=0.9)
    v.usage = {"input_tokens": 10, "output_tokens": 2, "cost_usd": 0.001}
    assert judge_usage(v) == {"input_tokens": 10.0, "output_tokens": 2.0, "cost_usd": 0.001}
