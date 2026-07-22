"""The EvidenceExtractor seam — scripted facts visibly reaching judge context= and the trace.

All offline against ``percept.harness.scripted.ScriptedJudge`` — zero API spend, wiring not
intelligence (standing rule 2). Covers:

* :class:`~percept.extraction.Enrichment` shape/validation.
* the ``registry.py`` "extractor" seam (``available``/``make_extractor``/``seams``).
* ``Percept.perceive_judged`` calling a configured extractor, merging its facts into
  ``context["extractor"]`` (the reserved sub-dict), and emitting an ``extractor.enriched`` trace
  stage carrying the SAME facts.
* the absent-extractor default stays byte-for-byte (no stage, no context).
"""

from __future__ import annotations

import asyncio
import unittest

import pytest

from percept import registry
from percept.adapters import Goal, Verdict
from percept.agent import Percept
from percept.backends.fake import ScriptedExtractor
from percept.extraction import CONTEXT_KEY, Enrichment
from percept.harness.scripted import ScriptedJudge


def _drive(agent: Percept, source: str, n: int) -> list:
    async def go():
        out = []
        for i in range(n):
            out += await agent.perceive_judged(source, t=float(i + 1))
        return out

    return asyncio.run(go())


class EnrichmentTests(unittest.TestCase):
    def test_defaults_are_empty_and_is_empty_true(self):
        e = Enrichment()
        self.assertEqual(e.facts, {})
        self.assertEqual(e.evidence_refs, ())
        self.assertIsNone(e.ts_range)
        self.assertIsNone(e.confidence)
        self.assertTrue(e.is_empty)

    def test_non_empty_facts_flip_is_empty(self):
        e = Enrichment(facts={"plate": "7XYZ123"})
        self.assertFalse(e.is_empty)

    def test_to_dict_is_json_shaped(self):
        e = Enrichment(facts={"a": 1}, evidence_refs=("ev_1",), ts_range=(1.0, 2.0), confidence=0.5)
        self.assertEqual(
            e.to_dict(),
            {"facts": {"a": 1}, "evidence_refs": ["ev_1"], "ts_range": [1.0, 2.0], "confidence": 0.5},
        )

    def test_confidence_out_of_range_rejected(self):
        with pytest.raises(ValueError):
            Enrichment(confidence=1.5)

    def test_ts_range_end_before_start_rejected(self):
        with pytest.raises(ValueError):
            Enrichment(ts_range=(2.0, 1.0))

    def test_facts_are_frozen_copy_not_the_caller_dict(self):
        src = {"a": 1}
        e = Enrichment(facts=src)
        src["a"] = 2  # mutating the caller's dict must not reach the Enrichment
        self.assertEqual(e.facts, {"a": 1})


class RegistrySeamTests(unittest.TestCase):
    def test_extractor_seam_is_registered(self):
        self.assertIn("extractor", registry.seams())

    def test_scripted_and_fake_extractor_resolve(self):
        for name in ("scripted", "fake"):
            adapter = registry.make_extractor(name)
            self.assertIsInstance(adapter, ScriptedExtractor)

    def test_scripted_extractor_extract_returns_enrichment(self):
        e = registry.make_extractor("scripted").extract("cond", "unscripted-token")
        self.assertIsInstance(e, Enrichment)
        self.assertTrue(e.is_empty)  # no script entry for this token -> empty, never raises


class PerceiveJudgedExtractorWiringTests(unittest.TestCase):
    def test_facts_reach_judge_context_under_the_reserved_key(self):
        judge = ScriptedJudge(default_verdict=Verdict(satisfied=True, confidence=0.9, evidence="ok"),
                              on_exhausted="repeat")
        extractor = ScriptedExtractor({"frame1": {"facts": {"invoice_number": "INV-2291"}}})
        core = Percept.create(vision=judge, extractor=extractor, discover_plugins=False)
        core.add_goal(Goal(id="g1", condition="invoice visible", say="alert"))

        _drive(core, "frame1", 1)

        self.assertEqual(len(judge.calls), 1)
        ctx = judge.calls[0]["context"]
        self.assertEqual(ctx[CONTEXT_KEY]["facts"], {"invoice_number": "INV-2291"})

    def test_trace_carries_extractor_enriched_stage_with_same_facts(self):
        judge = ScriptedJudge(default_verdict=Verdict(satisfied=True, confidence=0.9, evidence="ok"),
                              on_exhausted="repeat")
        extractor = ScriptedExtractor({"frame1": {"facts": {"invoice_number": "INV-2291"}}})
        trace: list = []
        core = Percept.create(vision=judge, extractor=extractor, discover_plugins=False,
                              on_trace=trace.append)
        core.add_goal(Goal(id="g1", condition="invoice visible", say="alert"))

        _drive(core, "frame1", 1)

        enriched = [ev for ev in trace if getattr(ev, "stage", None) == "extractor.enriched"]
        self.assertEqual(len(enriched), 1)
        self.assertEqual(enriched[0].data["facts"], {"invoice_number": "INV-2291"})
        # the stage fires strictly before judge.started (facts must exist before dispatch)
        stages = [ev.stage for ev in trace]
        self.assertLess(stages.index("extractor.enriched"), stages.index("judge.started"))

    def test_empty_enrichment_still_emits_the_stage_but_no_context_key(self):
        judge = ScriptedJudge(default_verdict=Verdict(satisfied=True, confidence=0.9, evidence="ok"),
                              on_exhausted="repeat")
        extractor = ScriptedExtractor({})  # never matches -> always empty Enrichment
        trace: list = []
        core = Percept.create(vision=judge, extractor=extractor, discover_plugins=False,
                              on_trace=trace.append)
        core.add_goal(Goal(id="g1", condition="invoice visible", say="alert"))

        _drive(core, "frame1", 1)

        enriched = [ev for ev in trace if getattr(ev, "stage", None) == "extractor.enriched"]
        self.assertEqual(len(enriched), 1)
        self.assertEqual(enriched[0].data["facts"], {})
        self.assertIsNone(judge.calls[0]["context"])  # no reserved key merged for an empty read

    def test_no_extractor_configured_is_byte_for_byte(self):
        judge = ScriptedJudge(default_verdict=Verdict(satisfied=True, confidence=0.9, evidence="ok"),
                              on_exhausted="repeat")
        trace: list = []
        core = Percept.create(vision=judge, discover_plugins=False, on_trace=trace.append)
        core.add_goal(Goal(id="g1", condition="invoice visible", say="alert"))

        _drive(core, "frame1", 1)

        self.assertIsNone(judge.calls[0]["context"])
        self.assertNotIn("extractor.enriched", [ev.stage for ev in trace])

    def test_extractor_runs_every_call_not_cached_away(self):
        # An extractor-enriched call must bypass the single-frame cache (its facts are per-call) —
        # mirrors the process_qa cache-bypass contract. Prove it by scripting the SAME frame twice
        # and asserting the extractor ran (and the judge was called) both times.
        judge = ScriptedJudge(default_verdict=Verdict(satisfied=True, confidence=0.9, evidence="ok"),
                              on_exhausted="repeat")
        seen: list = []

        class _CountingExtractor:
            def extract(self, condition, frame, roi=None):
                seen.append(frame)
                return Enrichment(facts={"n": len(seen)})

        core = Percept.create(vision=judge, extractor=_CountingExtractor(), discover_plugins=False)
        core.add_goal(Goal(id="g1", condition="invoice visible", say="alert"))

        _drive(core, "frame1", 2)

        self.assertEqual(seen, ["frame1", "frame1"])
        self.assertEqual(len(judge.calls), 2)
        self.assertEqual(judge.calls[0]["context"][CONTEXT_KEY]["facts"], {"n": 1})
        self.assertEqual(judge.calls[1]["context"][CONTEXT_KEY]["facts"], {"n": 2})


if __name__ == "__main__":
    unittest.main()
