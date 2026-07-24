"""P06-001 — the TRANSITION dispatch path (``Percept._perceive_transition``) must fail closed on the
per-goal judge binding exactly like the watch path (``Percept._judge``).

A transition goal spends TWO judge calls per tick (``from`` and ``to``). Before this fix that path
called ``self.vision.judge`` DIRECTLY, so a transition goal MARKED judge-requesting whose binding was
missing would silently fall back to the deployment-default judge — the exact failure class this
incident forbids. This module pins all three branches on the transition path:

* bound goal     -> both from/to calls reach the BOUND adapter (self.vision spends zero) and the
                    verified static context is injected under the reserved key on BOTH calls;
* marked+unbound -> :class:`JudgeBindingInvariantError` (zero judge calls, zero fallback);
* unmarked goal  -> self.vision, byte-for-byte (no reserved key ever injected).

Uses the same local doubles as ``test_judge_dispatch.py`` (no real ``judge_registry`` needed).
"""
from __future__ import annotations

import asyncio
import unittest
from dataclasses import dataclass
from typing import Any

from percept.adapters import Goal, Verdict
from percept.agent import JudgeBindingInvariantError, Percept
from percept.contracts.watchplan import ResolvedJudgeSpec
from percept.judge_context import JUDGE_STATIC_CONTEXT_KEY, ResolvedJudgeContext

FROM = "the form is pending approval"
TO = "the form shows approved"


class SpyJudge:
    """Judge-seam double: records every ``judge`` call (condition + received context). Serves as
    either ``self.vision`` or a bound adapter."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.judge_calls: list[dict[str, Any]] = []

    async def judge(self, condition: str, frame: Any, *, roi: Any = None, exemplar: Any = None,
                    corrections: dict | None = None, frames: list | None = None, audio: Any = None,
                    context: dict | None = None) -> Verdict:
        self.judge_calls.append({"condition": condition, "context": context})
        return Verdict(True, 0.95, f"{self.name}:{condition}")

    async def detect(self, frame: Any, goals: list) -> list:
        return []

    async def detect_events(self, frames: list, goals: list) -> list:
        return []


@dataclass(frozen=True)
class FakeBound:
    """Local stand-in for the pinned ``judge_registry.BoundJudge`` shape."""

    spec: ResolvedJudgeSpec
    adapter: Any
    static_context: Any = None


def _spec(resolved_ref: str, *, context_ref: str | None = None,
          context_sha256: str | None = None) -> ResolvedJudgeSpec:
    return ResolvedJudgeSpec(
        requested_ref=resolved_ref, resolved_ref=resolved_ref, factory_ref="factory@1",
        provider_ref="prov", model_requested="m", model_resolved="m",
        prompt_envelope_sha256=None, context_ref=context_ref, context_sha256=context_sha256,
    )


def _static_ctx(text: str, sha: str = "a" * 64) -> ResolvedJudgeContext:
    return ResolvedJudgeContext(
        ref="rubric.txt", sha256=sha, content_bytes=text.encode("utf-8"),
        decoded_text=text, media_type="text/plain; charset=utf-8",
    )


def _transition_goal(gid: str = "approval") -> Goal:
    return Goal(
        id=gid, condition="the form was approved", say="approved", source="camera",
        state={"type": "transition", "from": FROM, "to": TO, "count": 0},
    )


def _perceive(core: Percept, goal: Goal, frame: str = "frame"):
    return asyncio.run(core._perceive_transition(
        goal, frame, None, now=1.0, t=1.0, ingest_observed_at=1.0,
        trace_id="trace", tracing=False, source="camera",
    ))


class TransitionDispatchFailClosedTests(unittest.TestCase):
    def test_bound_transition_routes_both_calls_to_adapter_and_injects_static_context(self):
        vision = SpyJudge("vision")
        adapter = SpyJudge("bound")
        core = Percept.create(vision=vision, discover_plugins=False)
        g = _transition_goal()
        core.add_goal(g)
        ctx = _static_ctx("RUBRIC: approved means the green banner", sha="b" * 64)
        core._register_judge_binding(
            g.id, FakeBound(spec=_spec("audit@1", context_ref="rubric.txt", context_sha256="b" * 64),
                            adapter=adapter, static_context=ctx),
        )

        _perceive(core, g)

        # BOTH the 'from' and the 'to' judge calls reached the BOUND adapter; self.vision spent zero.
        self.assertEqual([c["condition"] for c in adapter.judge_calls], [FROM, TO])
        self.assertEqual(vision.judge_calls, [])
        # The verified static context was injected under the reserved key on BOTH calls.
        for call in adapter.judge_calls:
            self.assertEqual(
                call["context"][JUDGE_STATIC_CONTEXT_KEY], "RUBRIC: approved means the green banner"
            )

    def test_marked_but_unbound_transition_raises_and_spends_zero_judge_calls(self):
        vision = SpyJudge("vision")
        core = Percept.create(vision=vision, discover_plugins=False)
        g = _transition_goal()
        core.add_goal(g)
        # The invariant violation: MARKED judge-requesting but the binding is gone at dispatch.
        core._register_judge_binding(g.id, FakeBound(spec=_spec("audit@1"), adapter=SpyJudge("b")))
        core._judge_bindings.pop(g.id)  # binding lost, mark retained

        with self.assertRaises(JudgeBindingInvariantError):
            _perceive(core, g)
        self.assertEqual(vision.judge_calls, [])  # fail-closed: NO fallback to the generalist

    def test_unmarked_transition_uses_vision_and_never_injects_reserved_key(self):
        vision = SpyJudge("vision")
        core = Percept.create(vision=vision, discover_plugins=False)
        g = _transition_goal()
        core.add_goal(g)

        _perceive(core, g)

        # Byte-for-byte the pre-fix path: both calls go to self.vision, no reserved key ever injected.
        self.assertEqual([c["condition"] for c in vision.judge_calls], [FROM, TO])
        for call in vision.judge_calls:
            self.assertIsNone(call["context"])

    def test_bound_transition_static_context_and_extractor_facts_coexist(self):
        # A bound static context and the extractor enrichment ride the SAME judge_context on the
        # transition path: the reserved key holds the static text, the extractor key holds the facts.
        # Injection COPIES the extractor dict (never mutates it) before adding the reserved key.
        vision = SpyJudge("vision")
        adapter = SpyJudge("bound")

        class _FactsExtractor:
            def extract(self, condition, frame, roi=None):
                from percept.extraction import Enrichment
                return Enrichment(facts={"invoice_number": "INV-2291"})

        core = Percept.create(vision=vision, discover_plugins=False, extractor=_FactsExtractor())
        g = _transition_goal()
        core.add_goal(g)
        core._register_judge_binding(
            g.id, FakeBound(spec=_spec("audit@1", context_ref="r.txt", context_sha256="c" * 64),
                            adapter=adapter, static_context=_static_ctx("R", sha="c" * 64)),
        )
        _perceive(core, g)
        for call in adapter.judge_calls:
            self.assertEqual(call["context"][JUDGE_STATIC_CONTEXT_KEY], "R")
            self.assertEqual(call["context"]["extractor"]["facts"]["invoice_number"], "INV-2291")
        self.assertEqual(vision.judge_calls, [])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
