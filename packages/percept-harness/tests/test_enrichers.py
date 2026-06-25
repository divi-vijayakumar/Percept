"""F1 — the specialist-model enrichment seam (signal -> specialist -> judge context).

Deterministic stdlib tests (no model, no keys). They prove the SEAM, never model intelligence:

  * register_enricher / teardown — a specialist registers under a signal name; lifecycle mirrors detectors.
  * the driver (enrich_signal)    — runs the specialist, splits structured context from a short-circuit
                                     verdict, and degrades a raising/garbage enricher to a no-op.
  * structured output reaches the judge — enrich a `plate-presence` cue, feed its facts to Vision.judge as
                                     context=, and show the judge's verdict changes (the facts flowed in).
  * a SECOND specialist adds with ZERO core changes — a second register_enricher call, nothing else.
"""

from __future__ import annotations

import asyncio
import unittest

from percept.adapters import Enricher, Verdict
from percept.backends.fake import FakeVision

from percept_harness.enrichers import (
    Enrichment,
    FakeEnricher,
    NullEnricher,
    clear_enrichers,
    enrich_signal,
    enricher_for,
    register_enricher,
    registered_enrichers,
    unregister_enricher,
)


def _run(coro):
    return asyncio.run(coro)


class TestEnricherRegistry(unittest.TestCase):
    """register_enricher binds a specialist to a signal name; resolution + teardown mirror detectors."""

    def setUp(self):
        clear_enrichers()

    def tearDown(self):
        clear_enrichers()

    def test_register_and_resolve(self):
        e = FakeEnricher({"plate": "7XYZ123"})
        self.assertIs(register_enricher("plate-presence", e), e)  # returns the enricher
        self.assertEqual(registered_enrichers(), ["plate-presence"])
        self.assertIs(enricher_for("plate-presence"), e)

    def test_unregistered_signal_resolves_to_none(self):
        self.assertIsNone(enricher_for("no-specialist-here"))

    def test_reregister_same_signal_replaces(self):
        register_enricher("plate-presence", FakeEnricher({"plate": "OLD"}))
        register_enricher("plate-presence", FakeEnricher({"plate": "NEW"}))
        self.assertEqual(registered_enrichers(), ["plate-presence"])  # last-writer-wins, one entry
        self.assertEqual(enrich_signal("plate-presence").context, {"plate": "NEW"})

    def test_teardown_drops_specialist(self):
        register_enricher("plate-presence", FakeEnricher({"plate": "X"}))
        self.assertTrue(unregister_enricher("plate-presence"))
        self.assertIsNone(enricher_for("plate-presence"))
        self.assertFalse(unregister_enricher("plate-presence"))  # idempotent: gone -> False

    def test_protocol_satisfied_by_fakes(self):
        # the fakes ARE Enrichers (runtime_checkable protocol) — the registry's accepted shape
        self.assertIsInstance(FakeEnricher(), Enricher)
        self.assertIsInstance(NullEnricher(), Enricher)


class TestEnrichDriver(unittest.TestCase):
    """enrich_signal: resolve -> run -> split context / short-circuit verdict; failures degrade to no-op."""

    def setUp(self):
        clear_enrichers()

    def tearDown(self):
        clear_enrichers()

    def test_unregistered_signal_is_noop(self):
        out = enrich_signal("nothing-registered")
        self.assertIsInstance(out, Enrichment)
        self.assertTrue(out.is_noop)
        self.assertEqual(out.context, {})
        self.assertIsNone(out.verdict)

    def test_null_enricher_is_noop(self):
        register_enricher("plate-presence", NullEnricher())
        self.assertTrue(enrich_signal("plate-presence").is_noop)

    def test_structured_facts_become_context(self):
        register_enricher("plate-presence", FakeEnricher({"plate": "7XYZ123", "region": "US-CA"}))
        out = enrich_signal("plate-presence", signal="cue", frame="frame")
        self.assertEqual(out.context, {"plate": "7XYZ123", "region": "US-CA"})
        self.assertIsNone(out.verdict)  # no short-circuit
        self.assertFalse(out.is_noop)

    def test_verdict_key_short_circuits(self):
        v = Verdict(satisfied=True, confidence=0.99, evidence="LPR: watched plate matched")
        register_enricher("plate-presence", FakeEnricher({"plate": "7XYZ123"}, verdict=v))
        out = enrich_signal("plate-presence")
        self.assertIs(out.verdict, v)  # short-circuit verdict surfaced
        self.assertEqual(out.context, {"plate": "7XYZ123"})  # the reserved key is split OUT of context
        self.assertNotIn("verdict", out.context)

    def test_raising_enricher_degrades_to_noop(self):
        # a flaky specialist must never break the judge path
        register_enricher("plate-presence", FakeEnricher({"plate": "X"}, raises=True))
        self.assertTrue(enrich_signal("plate-presence").is_noop)


class TestEnrichmentReachesJudge(unittest.TestCase):
    """The payoff: a specialist's STRUCTURED output flows into Vision.judge(context=...) and changes it."""

    def setUp(self):
        clear_enrichers()

    def tearDown(self):
        clear_enrichers()

    def test_enricher_output_reaches_judge_context(self):
        # FakeVision routes a frame token to a distinct verdict WHEN context is present ("<token>|enriched"),
        # so a changed verdict PROVES the enricher's facts actually reached judge (wiring, not intelligence).
        vision = FakeVision(
            script={
                "plate-frame": Verdict(satisfied=False, confidence=0.3, evidence="no read"),
                "plate-frame|enriched": Verdict(satisfied=True, confidence=0.95, evidence="LPR matched"),
            }
        )
        # baseline: no enricher registered -> no context -> the un-enriched verdict
        bare = _run(vision.judge("the watched plate is present", "plate-frame"))
        self.assertFalse(bare.satisfied)

        # register the specialist for the cue, run the driver, feed its context to the SAME judge call
        register_enricher("plate-presence", FakeEnricher({"plate": "7XYZ123"}))
        enr = enrich_signal("plate-presence", signal="cue", frame="plate-frame")
        enriched = _run(
            vision.judge("the watched plate is present", "plate-frame", context=enr.context)
        )
        self.assertTrue(enriched.satisfied)  # the specialist's facts flowed into the judge -> different read
        self.assertEqual(enriched.evidence, "LPR matched")

    def test_short_circuit_verdict_bypasses_judge(self):
        # when the specialist short-circuits, the brain uses its verdict and never calls the judge at all
        v = Verdict(satisfied=True, confidence=0.99, evidence="LPR: exact plate match")
        register_enricher("plate-presence", FakeEnricher(verdict=v))
        enr = enrich_signal("plate-presence")
        self.assertIsNotNone(enr.verdict)
        # the brain's rule: a short-circuit verdict is used directly; no judge call is spent
        verdict = enr.verdict if enr.verdict is not None else _run(
            FakeVision().judge("unused", "plate-frame")
        )
        self.assertIs(verdict, v)


class TestSecondSpecialistZeroCoreChange(unittest.TestCase):
    """Pluggability: adding a SECOND specialist is one more register_enricher — no core/judge/contract edit."""

    def setUp(self):
        clear_enrichers()

    def tearDown(self):
        clear_enrichers()

    def test_two_independent_specialists_coexist(self):
        # specialist #1: an LPR model for a plate cue
        register_enricher("plate-presence", FakeEnricher({"plate": "7XYZ123"}))
        # specialist #2: a fire/smoke classifier for a hot-surface cue — added with ZERO changes to
        # Percept core, the judge, or the contract; just a second registration.
        register_enricher("smoke-cue", FakeEnricher({"smoke": "true", "severity": "high"}))

        self.assertEqual(registered_enrichers(), ["plate-presence", "smoke-cue"])
        # each resolves to ITS OWN structured output, keyed only by signal name
        self.assertEqual(enrich_signal("plate-presence").context, {"plate": "7XYZ123"})
        self.assertEqual(
            enrich_signal("smoke-cue").context, {"smoke": "true", "severity": "high"}
        )

    def test_a_third_party_class_satisfies_the_protocol(self):
        # a brand-new specialist defined OUTSIDE the package (e.g. a third-party LPR) just implements
        # enrich() — no base class to inherit, no core registration table to edit.
        class ThirdPartyLpr:
            def enrich(self, signal, frame=None):
                return {"plate": "ABC789", "confidence": "0.92"}

        register_enricher("plate-presence", ThirdPartyLpr())
        self.assertIsInstance(enricher_for("plate-presence"), Enricher)
        self.assertEqual(enrich_signal("plate-presence").context["plate"], "ABC789")


if __name__ == "__main__":
    unittest.main()
