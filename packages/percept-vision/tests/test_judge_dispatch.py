"""P06-001 per-goal judge dispatch — fail-closed routing, static-context injection, and runtime-
stamped trace provenance, all on LOCAL test doubles (no real ``percept.judge_registry`` needed).

The dispatch site under test is ``Percept._judge`` / ``Percept._timed_judge`` (the single place a
judge adapter is awaited) plus the per-tick trace emission in ``perceive_judged``. The doubles below
match the pinned ``BoundJudge`` shape (``.spec`` / ``.adapter`` / ``.static_context``); cross-module
integration with the real registry is verified in the next merge wave.
"""
from __future__ import annotations

import asyncio
import unittest
from dataclasses import dataclass
from typing import Any

from percept.adapters import Goal, Verdict
from percept.agent import JudgeBindingInvariantError, Percept
from percept.contracts.watchplan import ResolvedJudgeSpec
from percept.judge_context import (
    JUDGE_STATIC_CONTEXT_KEY,
    ReservedContextKeyError,
    ResolvedJudgeContext,
)


def _usage_verdict(evidence: str) -> Verdict:
    """A well-formed verdict annotated with usage so a ``judge.usage`` trace event is emitted."""
    return Verdict(True, 0.9, evidence, usage={"input_tokens": 10, "output_tokens": 5, "cost_usd": 0.01})


class SpyJudge:
    """A judge-seam double: records every ``judge`` call (condition + the context it received) and
    returns a scripted, usage-annotated verdict. Doubles as a vision backend (``detect`` returns
    nothing) so it can serve as either ``self.vision`` or a bound adapter."""

    PARSES_CLAIM_CHANNEL = True

    def __init__(self, name: str, *, model: str | None = None) -> None:
        self.name = name
        self.model = model
        self.judge_calls: list[dict[str, Any]] = []

    async def judge(self, condition: str, frame: Any, *, roi: Any = None, exemplar: Any = None,
                    corrections: dict | None = None, frames: list | None = None, audio: Any = None,
                    context: dict | None = None) -> Verdict:
        self.judge_calls.append({"condition": condition, "frame": frame, "context": context})
        return _usage_verdict(f"{self.name}:{condition}")

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


def _spec(resolved_ref: str, *, context_ref: str | None = None, context_sha256: str | None = None) -> ResolvedJudgeSpec:
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


def _core(vision: SpyJudge) -> Percept:
    return Percept.create(vision=vision, discover_plugins=False)


def _goal(gid: str) -> Goal:
    return Goal(id=gid, condition=f"condition for {gid}", say="x", source="camera")


def _run(coro):
    return asyncio.run(coro)


class JudgeRoutingTests(unittest.TestCase):
    def test_bound_goal_routes_to_adapter_and_vision_gets_zero_calls(self):
        vision = SpyJudge("vision")
        adapter = SpyJudge("bound")
        core = _core(vision)
        g = _goal("g1")
        core.add_goal(g)
        core._register_judge_binding(g.id, FakeBound(spec=_spec("audit@1"), adapter=adapter))

        verdict = _run(core._judge(g, "frame", None, {}))
        self.assertTrue(verdict.satisfied)
        self.assertEqual(len(adapter.judge_calls), 1)          # the bound adapter received it
        self.assertEqual(vision.judge_calls, [])               # self.vision spent ZERO calls

    def test_marked_but_unbound_raises_and_never_calls_vision(self):
        vision = SpyJudge("vision")
        core = _core(vision)
        g = _goal("g1")
        core.add_goal(g)
        # Model the invariant violation: the goal is MARKED judge-requesting but the binding is gone.
        core._register_judge_binding(g.id, FakeBound(spec=_spec("audit@1"), adapter=SpyJudge("b")))
        core._judge_bindings.pop(g.id)  # binding lost, mark retained

        with self.assertRaises(JudgeBindingInvariantError):
            _run(core._judge(g, "frame", None, {}))
        self.assertEqual(vision.judge_calls, [])  # fail-closed: no fallback to self.vision

    def test_unmarked_goal_uses_vision_and_leaves_context_untouched(self):
        vision = SpyJudge("vision")
        core = _core(vision)
        g = _goal("g1")
        core.add_goal(g)

        caller_ctx = {"process_qa": True}  # forwarded to the adapter (proves the reserved key is absent)
        _run(core._judge(g, "frame", None, {}, context=caller_ctx))

        self.assertEqual(len(vision.judge_calls), 1)
        received = vision.judge_calls[0]["context"]
        self.assertNotIn(JUDGE_STATIC_CONTEXT_KEY, received)
        self.assertNotIn(JUDGE_STATIC_CONTEXT_KEY, caller_ctx)  # caller's dict never mutated

    def test_two_goals_with_different_bindings_do_not_cross_route(self):
        vision = SpyJudge("vision")
        core = _core(vision)
        a_adapter, b_adapter = SpyJudge("A"), SpyJudge("B")
        ga, gb = _goal("ga"), _goal("gb")
        core.add_goal(ga)
        core.add_goal(gb)
        core._register_judge_binding(ga.id, FakeBound(spec=_spec("a@1"), adapter=a_adapter))
        core._register_judge_binding(gb.id, FakeBound(spec=_spec("b@1"), adapter=b_adapter))

        _run(core._judge(ga, "fa", None, {}))
        _run(core._judge(gb, "fb", None, {}))

        self.assertEqual([c["condition"] for c in a_adapter.judge_calls], [ga.condition])
        self.assertEqual([c["condition"] for c in b_adapter.judge_calls], [gb.condition])
        self.assertEqual(vision.judge_calls, [])


class StaticContextInjectionTests(unittest.TestCase):
    def test_decoded_text_arrives_under_reserved_key_without_mutating_caller(self):
        vision = SpyJudge("vision")
        adapter = SpyJudge("bound")
        core = _core(vision)
        g = _goal("g1")
        core.add_goal(g)
        ctx = _static_ctx("RUBRIC: fire only on X", sha="b" * 64)
        core._register_judge_binding(
            g.id, FakeBound(spec=_spec("j@1", context_ref="rubric.txt", context_sha256="b" * 64),
                            adapter=adapter, static_context=ctx),
        )

        caller_ctx = {"foo": "bar"}
        _run(core._judge(g, "frame", None, {}, context=caller_ctx))

        received = adapter.judge_calls[0]["context"]
        self.assertEqual(received[JUDGE_STATIC_CONTEXT_KEY], "RUBRIC: fire only on X")
        self.assertEqual(received["foo"], "bar")
        self.assertNotIn(JUDGE_STATIC_CONTEXT_KEY, caller_ctx)  # caller's dict is COPIED, never mutated

    def test_preexisting_reserved_key_raises(self):
        vision = SpyJudge("vision")
        adapter = SpyJudge("bound")
        core = _core(vision)
        g = _goal("g1")
        core.add_goal(g)
        core._register_judge_binding(
            g.id, FakeBound(spec=_spec("j@1", context_ref="r.txt", context_sha256="c" * 64),
                            adapter=adapter, static_context=_static_ctx("R", sha="c" * 64)),
        )

        with self.assertRaises(ReservedContextKeyError):
            _run(core._judge(g, "frame", None, {}, context={JUDGE_STATIC_CONTEXT_KEY: "smuggled"}))
        self.assertEqual(adapter.judge_calls, [])  # nothing dispatched on a reserved-key collision

    def test_shared_adapter_two_goals_no_static_context_leakage(self):
        vision = SpyJudge("vision")
        shared = SpyJudge("shared")  # ONE adapter, two goals, two different static contexts
        core = _core(vision)
        ga, gb = _goal("ga"), _goal("gb")
        core.add_goal(ga)
        core.add_goal(gb)
        core._register_judge_binding(
            ga.id, FakeBound(spec=_spec("a@1", context_ref="a.txt", context_sha256="d" * 64),
                             adapter=shared, static_context=_static_ctx("CONTEXT-A", sha="d" * 64)),
        )
        core._register_judge_binding(
            gb.id, FakeBound(spec=_spec("b@1", context_ref="b.txt", context_sha256="e" * 64),
                             adapter=shared, static_context=_static_ctx("CONTEXT-B", sha="e" * 64)),
        )

        _run(core._judge(ga, "fa", None, {}))
        _run(core._judge(gb, "fb", None, {}))

        self.assertEqual(shared.judge_calls[0]["context"][JUDGE_STATIC_CONTEXT_KEY], "CONTEXT-A")
        self.assertEqual(shared.judge_calls[1]["context"][JUDGE_STATIC_CONTEXT_KEY], "CONTEXT-B")


class JudgeTraceProvenanceTests(unittest.TestCase):
    def test_bound_goal_trace_events_carry_provenance_unbound_do_not(self):
        vision = SpyJudge("vision")
        adapter = SpyJudge("bound")
        events: list = []
        core = Percept.create(vision=vision, discover_plugins=False, on_trace=events.append)
        bound_goal, plain_goal = _goal("bound"), _goal("plain")
        core.add_goal(bound_goal)
        core.add_goal(plain_goal)
        core._register_judge_binding(
            bound_goal.id, FakeBound(spec=_spec("audit@2", context_sha256="f" * 64), adapter=adapter),
        )

        _run(core.perceive_judged("frame", t=1.0))

        def _for(stage: str, goal_id: str) -> dict:
            matches = [e.data for e in events if e.stage == stage and e.goal_id == goal_id]
            self.assertEqual(len(matches), 1, f"expected one {stage} for {goal_id}")
            return matches[0]

        for stage in ("judge.started", "judge.finished"):
            bound_data = _for(stage, bound_goal.id)
            self.assertEqual(bound_data["judge_resolved"], "audit@2")
            self.assertEqual(bound_data["judge_context_sha256"], "f" * 64)
            self.assertIs(bound_data["fallback_used"], False)

            plain_data = _for(stage, plain_goal.id)
            self.assertNotIn("judge_resolved", plain_data)
            self.assertNotIn("fallback_used", plain_data)

        # judge.usage carries no goal_id; distinguish by the additive provenance keys. Exactly the
        # bound goal's usage event is stamped, and the adapter — not the Verdict — set the values.
        usage_with_prov = [e.data for e in events if e.stage == "judge.usage" and "judge_resolved" in e.data]
        usage_without = [e.data for e in events if e.stage == "judge.usage" and "judge_resolved" not in e.data]
        self.assertEqual(len(usage_with_prov), 1)
        self.assertEqual(len(usage_without), 1)
        self.assertEqual(usage_with_prov[0]["judge_resolved"], "audit@2")
        self.assertIs(usage_with_prov[0]["fallback_used"], False)

        # Routing held under the full tick: the bound adapter judged its goal, self.vision judged the plain one.
        self.assertEqual([c["condition"] for c in adapter.judge_calls], [bound_goal.condition])
        self.assertEqual([c["condition"] for c in vision.judge_calls], [plain_goal.condition])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
