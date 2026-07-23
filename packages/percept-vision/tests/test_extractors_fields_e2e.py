"""End-to-end: FieldNormalizer/ScriptedFields facts reach the judge and the trace through the real
perceive_judged path, and prove the incident this card exists to close: a correctly-instructed judge
sees "25,000 dollars" (spoken) and "threshold: 25,000 dollars" (visible) as EQUAL, not a mismatch.

Mirrors ``tests/test_extractors_ocr_e2e.py`` — drives the registry-resolved extractor through
``Percept.create(extractor=...)`` -> ``perceive_judged`` -> judge ``context["extractor"]`` + the
``extractor.enriched`` trace stage. All offline against ``percept.harness.scripted.ScriptedJudge`` —
zero API spend (standing rule 2).
"""

from __future__ import annotations

import asyncio
import unittest

from percept import registry
from percept.adapters import Goal, Verdict
from percept.agent import Percept
from percept.extraction import CONTEXT_KEY
from percept.harness.scripted import ScriptedJudge


def _drive(agent: Percept, source: str, n: int) -> list:
    async def go():
        out = []
        for i in range(n):
            out += await agent.perceive_judged(source, t=float(i + 1))
        return out

    return asyncio.run(go())


class ScriptedFieldsEndToEndTests(unittest.TestCase):
    def test_fields_facts_reach_judge_context_via_the_registry_seam(self):
        extractor = registry.make_extractor(
            "fields-scripted", {"doc_frame": {"text": "threshold: 25,000 dollars"}},
        )
        judge = ScriptedJudge(default_verdict=Verdict(satisfied=True, confidence=0.9, evidence="ok"),
                              on_exhausted="repeat")
        core = Percept.create(vision=judge, extractor=extractor, discover_plugins=False)
        core.add_goal(Goal(id="g1", condition="threshold field visible", say="alert"))

        _drive(core, "doc_frame", 1)

        self.assertEqual(len(judge.calls), 1)
        ctx = judge.calls[0]["context"]
        fields = ctx[CONTEXT_KEY]["facts"]["fields"]
        self.assertEqual(fields[0]["field"], "threshold")
        self.assertEqual(fields[0]["normalized"], "USD 25000.00")

    def test_fields_extractor_enriched_trace_stage_carries_the_same_facts(self):
        extractor = registry.make_extractor(
            "fields-scripted", {"doc_frame": {"text": "monthly"}},
        )
        judge = ScriptedJudge(default_verdict=Verdict(satisfied=True, confidence=0.9, evidence="ok"),
                              on_exhausted="repeat")
        trace: list = []
        core = Percept.create(vision=judge, extractor=extractor, discover_plugins=False,
                              on_trace=trace.append)
        core.add_goal(Goal(id="g1", condition="frequency field visible", say="alert"))

        _drive(core, "doc_frame", 1)

        enriched = [ev for ev in trace if getattr(ev, "stage", None) == "extractor.enriched"]
        self.assertEqual(len(enriched), 1)
        self.assertEqual(enriched[0].data["facts"]["fields"][0]["normalized"], "1/month")
        stages = [ev.stage for ev in trace]
        self.assertLess(stages.index("extractor.enriched"), stages.index("judge.started"))

    def test_unscripted_frame_yields_empty_facts_and_no_reserved_context_key(self):
        extractor = registry.make_extractor("fields-scripted", {})  # never matches
        judge = ScriptedJudge(default_verdict=Verdict(satisfied=True, confidence=0.9, evidence="ok"),
                              on_exhausted="repeat")
        trace: list = []
        core = Percept.create(vision=judge, extractor=extractor, discover_plugins=False,
                              on_trace=trace.append)
        core.add_goal(Goal(id="g1", condition="field visible", say="alert"))

        _drive(core, "doc_frame", 1)

        self.assertIsNone(judge.calls[0]["context"])
        enriched = [ev for ev in trace if getattr(ev, "stage", None) == "extractor.enriched"]
        self.assertEqual(enriched[0].data["facts"], {})


class FusedCompareIncidentRegressionTests(unittest.TestCase):
    """The exact client incident: a fused compare over ('25,000 dollars' spoken, 'threshold:
    25,000 dollars' visible) must not false-fire a mismatch at confidence 1.0 -- a correctly-
    instructed judge reading BOTH sides' normalized facts sees them as equal."""

    def test_spoken_and_visible_frames_normalize_to_the_same_canonical_value_via_the_real_path(self):
        # Two frames (one per modality's admitted moment), same registry-resolved extractor.
        extractor = registry.make_extractor(
            "fields-scripted",
            {
                "spoken_frame": {"text": "25,000 dollars"},
                "visible_frame": {"text": "threshold: 25,000 dollars"},
            },
        )
        judge = ScriptedJudge(default_verdict=Verdict(satisfied=True, confidence=0.9, evidence="ok"),
                              on_exhausted="repeat")
        core = Percept.create(vision=judge, extractor=extractor, discover_plugins=False)
        core.add_goal(Goal(
            id="g1",
            condition=(
                "compare context.extractor.facts.fields[0].normalized across the spoken and visible "
                "reads; fire only on a confirmed VALUE mismatch, never on differing raw text; ASK "
                "if either side's normalized is None"
            ),
            say="mismatch",
        ))

        async def go():
            spoken = await core.perceive_judged("spoken_frame", t=1.0)
            visible = await core.perceive_judged("visible_frame", t=2.0)
            return spoken, visible

        asyncio.run(go())

        self.assertEqual(len(judge.calls), 2)
        spoken_ctx = judge.calls[0]["context"][CONTEXT_KEY]["facts"]["fields"][0]
        visible_ctx = judge.calls[1]["context"][CONTEXT_KEY]["facts"]["fields"][0]
        # This is the assertion the incident violated: a judge that reads THESE facts (instead of
        # raw text) is handed IDENTICAL normalized values -- "same value" is now a fact, not an
        # inference the judge has to get right on its own.
        self.assertEqual(spoken_ctx["normalized"], visible_ctx["normalized"])
        self.assertEqual(spoken_ctx["normalized"], "USD 25000.00")
        self.assertNotEqual(spoken_ctx["raw"], visible_ctx.get("field"))  # surface forms do differ

    def test_ocr_fields_scripted_composition_carries_both_sides_through_one_extractor_name(self):
        extractor = registry.make_extractor(
            "ocr-fields-scripted",
            {
                "spoken_frame": {"text": "25,000 dollars"},
                "visible_frame": {"text": "threshold: 25,000 dollars"},
            },
        )
        judge = ScriptedJudge(default_verdict=Verdict(satisfied=True, confidence=0.9, evidence="ok"),
                              on_exhausted="repeat")
        core = Percept.create(vision=judge, extractor=extractor, discover_plugins=False)
        core.add_goal(Goal(id="g1", condition="threshold compare", say="mismatch"))

        async def go():
            await core.perceive_judged("spoken_frame", t=1.0)
            await core.perceive_judged("visible_frame", t=2.0)

        asyncio.run(go())

        spoken_facts = judge.calls[0]["context"][CONTEXT_KEY]["facts"]
        visible_facts = judge.calls[1]["context"][CONTEXT_KEY]["facts"]
        self.assertEqual(spoken_facts["fields"][0]["normalized"], "USD 25000.00")
        self.assertEqual(visible_facts["fields"][0]["normalized"], "USD 25000.00")
        self.assertEqual(visible_facts["text"], "threshold: 25,000 dollars")


if __name__ == "__main__":
    unittest.main()
