"""D6(c) — the dead enricher seam is GONE, and stays gone.

`percept.harness.enrichers` (`register_enricher` / `enrich_signal` / `enricher_for`) was documented as
"the way to add an OCR specialist without editing core" and had ZERO callers anywhere outside its own
module — registering a specialist changed nothing about any watch. It was removed rather than left
standing as an architecture-shaped seam a client could wire up and get silence from.

What must still be true after the removal:
  * the registry names are unimportable (a wrong import fails loudly at import time, not silently at 2am),
  * `Vision.judge(context=...)` — the LIVE way to hand the judge structured facts — is untouched.
"""
from __future__ import annotations

import unittest


class EnricherSeamRemovedTests(unittest.TestCase):
    def test_the_enricher_registry_module_is_gone(self):
        with self.assertRaises(ModuleNotFoundError):
            __import__("percept.harness.enrichers")
        with self.assertRaises(ModuleNotFoundError):
            __import__("percept.edge.enrichers")

    def test_the_registry_names_are_not_re_exported(self):
        import percept.edge as edge
        import percept.harness as harness

        for name in (
            "register_enricher",
            "enricher_for",
            "registered_enrichers",
            "unregister_enricher",
            "clear_enrichers",
            "enrich_signal",
            "Enrichment",
            "NullEnricher",
            "FakeEnricher",
        ):
            self.assertNotIn(name, harness.__all__, f"{name} still exported from percept.harness")
            self.assertNotIn(name, edge.__all__, f"{name} still exported from percept.edge")
            self.assertFalse(hasattr(harness, name), f"percept.harness.{name} still resolves")

    def test_the_live_replacement_is_the_judge_context_parameter(self):
        """Facts reach the judge because the CALLER passes them — no registry in between."""
        import asyncio
        import inspect

        from percept.adapters import Verdict
        from percept.backends.fake import FakeVision

        self.assertIn("context", inspect.signature(FakeVision.judge).parameters)
        v = FakeVision(script={"x": Verdict(satisfied=True, confidence=0.9, evidence="ok")})
        out = asyncio.run(v.judge("the invoice changed", "x", context={"invoice_number": "INV-2291"}))
        self.assertIsInstance(out, Verdict)

    def test_the_deprecated_protocol_says_so(self):
        from percept import Enricher

        self.assertIn("DEPRECATED", (Enricher.__doc__ or ""))


if __name__ == "__main__":
    unittest.main()
