"""End-to-end: ScriptedOCR facts reach the judge and the trace through the real perceive_judged path.

Mirrors ``tests/test_extraction.py`` (the generic seam-wiring proof) but drives the OCR-specific
extractor registered under the seam (``registry.make_extractor("ocr-scripted", ...)``), so this
proves the FULL path an app actually configures: registry name -> constructed
``percept.extractors.ocr.ScriptedOCR`` -> ``Percept.create(extractor=...)`` ->
``perceive_judged`` -> judge ``context["extractor"]`` + the ``extractor.enriched`` trace stage.
All offline against ``percept.harness.scripted.ScriptedJudge`` — zero API spend (standing rule 2).
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


class ScriptedOCREndToEndTests(unittest.TestCase):
    def test_ocr_facts_reach_judge_context_via_the_registry_seam(self):
        extractor = registry.make_extractor(
            "ocr-scripted",
            {"doc_frame": {"text": "INVOICE 2291", "regions": [{"text": "INVOICE"}], "confidence": 0.92}},
        )
        judge = ScriptedJudge(default_verdict=Verdict(satisfied=True, confidence=0.9, evidence="ok"),
                              on_exhausted="repeat")
        core = Percept.create(vision=judge, extractor=extractor, discover_plugins=False)
        core.add_goal(Goal(id="g1", condition="invoice document visible", say="alert"))

        _drive(core, "doc_frame", 1)

        self.assertEqual(len(judge.calls), 1)
        ctx = judge.calls[0]["context"]
        self.assertEqual(ctx[CONTEXT_KEY]["facts"]["text"], "INVOICE 2291")
        self.assertEqual(ctx[CONTEXT_KEY]["confidence"], 0.92)

    def test_ocr_extractor_enriched_trace_stage_carries_the_same_facts(self):
        extractor = registry.make_extractor(
            "ocr-scripted", {"doc_frame": {"text": "RECEIPT 004", "confidence": 0.5}},
        )
        judge = ScriptedJudge(default_verdict=Verdict(satisfied=True, confidence=0.9, evidence="ok"),
                              on_exhausted="repeat")
        trace: list = []
        core = Percept.create(vision=judge, extractor=extractor, discover_plugins=False,
                              on_trace=trace.append)
        core.add_goal(Goal(id="g1", condition="receipt document visible", say="alert"))

        _drive(core, "doc_frame", 1)

        enriched = [ev for ev in trace if getattr(ev, "stage", None) == "extractor.enriched"]
        self.assertEqual(len(enriched), 1)
        self.assertEqual(enriched[0].data["facts"]["text"], "RECEIPT 004")
        # extractor.enriched must fire before judge.started — the facts must exist before dispatch.
        stages = [ev.stage for ev in trace]
        self.assertLess(stages.index("extractor.enriched"), stages.index("judge.started"))

    def test_unscripted_frame_yields_empty_facts_and_no_reserved_context_key(self):
        extractor = registry.make_extractor("ocr-scripted", {})  # never matches
        judge = ScriptedJudge(default_verdict=Verdict(satisfied=True, confidence=0.9, evidence="ok"),
                              on_exhausted="repeat")
        trace: list = []
        core = Percept.create(vision=judge, extractor=extractor, discover_plugins=False,
                              on_trace=trace.append)
        core.add_goal(Goal(id="g1", condition="document visible", say="alert"))

        _drive(core, "doc_frame", 1)

        self.assertIsNone(judge.calls[0]["context"])
        enriched = [ev for ev in trace if getattr(ev, "stage", None) == "extractor.enriched"]
        self.assertEqual(enriched[0].data["facts"], {})


if __name__ == "__main__":
    unittest.main()
