"""percept.extractors.ocr — ScriptedOCR (offline double) + TesseractOCR (optional, lazy) unit tests.

All offline / zero API spend (standing rule 2): ScriptedOCR is exercised directly, and
TesseractOCR's typed-error path is exercised by forcing its lazy import to miss — this SDK's CI
never installs pytesseract/Pillow, so this test suite must pass identically whether or not they
happen to be present in a given environment.
"""

from __future__ import annotations

import unittest

import pytest

from percept import registry
from percept.contracts import Box
from percept.extraction import Enrichment
from percept.extractors import ocr as ocr_mod
from percept.extractors.ocr import ScriptedOCR, TesseractOCR, _frame_key


class ScriptedOCRTests(unittest.TestCase):
    def test_scripted_token_hit_returns_text_regions_confidence(self):
        backend = ScriptedOCR({
            "frame1": {
                "text": "INVOICE 2291",
                "regions": [{"text": "INVOICE", "box": {"x": 0.0, "y": 0.0, "w": 0.5, "h": 0.2}}],
                "confidence": 0.87,
            }
        })
        e = backend.extract("invoice visible", "frame1")
        self.assertEqual(e.facts["text"], "INVOICE 2291")
        self.assertEqual(len(e.facts["regions"]), 1)
        self.assertEqual(e.confidence, 0.87)

    def test_unscripted_token_returns_empty_enrichment_never_raises(self):
        backend = ScriptedOCR({"frame1": {"text": "hit"}})
        e = backend.extract("cond", "totally-unscripted-token")
        self.assertIsInstance(e, Enrichment)
        self.assertTrue(e.is_empty)

    def test_script_entry_may_be_a_ready_made_enrichment(self):
        made = Enrichment(facts={"text": "prebuilt"}, confidence=0.5)
        backend = ScriptedOCR({"frame1": made})
        e = backend.extract("cond", "frame1")
        self.assertIs(e, made)

    def test_non_string_frame_keys_off_a_stable_content_hash(self):
        class Frame:
            def __repr__(self) -> str:
                return "FRAME-CONTENT-XYZ"

        key = _frame_key(Frame())
        self.assertEqual(key, _frame_key(Frame()))  # same repr -> same key, deterministic
        backend = ScriptedOCR({key: {"text": "hashed hit"}})
        e = backend.extract("cond", Frame())
        self.assertEqual(e.facts["text"], "hashed hit")

    def test_default_missing_regions_and_confidence_degrade_to_empty_list_and_none(self):
        backend = ScriptedOCR({"frame1": {"text": "just text"}})
        e = backend.extract("cond", "frame1")
        self.assertEqual(e.facts["regions"], [])
        self.assertIsNone(e.confidence)

    def test_no_script_defaults_to_empty_dict_not_a_crash(self):
        backend = ScriptedOCR()
        e = backend.extract("cond", "anything")
        self.assertTrue(e.is_empty)


class TesseractOCRLazyImportTests(unittest.TestCase):
    """Exercises the typed-error path deterministically by forcing the lazy import to miss —
    independent of whether pytesseract/Pillow actually happen to be installed."""

    def test_missing_pytesseract_raises_typed_runtime_error(self):
        def _boom(name: str):
            raise ModuleNotFoundError(name)

        original = ocr_mod.import_module
        ocr_mod.import_module = _boom
        try:
            backend = TesseractOCR()
            with pytest.raises(RuntimeError) as exc_info:
                backend.extract("cond", "some/path.png")
            msg = str(exc_info.value)
            self.assertIn("pytesseract", msg)
            self.assertIn("percept-vision[ocr]", msg)
            self.assertIn("ocr-scripted", msg)
        finally:
            ocr_mod.import_module = original

    def test_construction_never_imports_the_optional_backend(self):
        # Constructing TesseractOCR must succeed even when pytesseract/Pillow are unavailable —
        # only .extract() may fail.
        def _boom(name: str):
            raise ModuleNotFoundError(name)

        original = ocr_mod.import_module
        ocr_mod.import_module = _boom
        try:
            TesseractOCR()  # must not raise
        finally:
            ocr_mod.import_module = original


class RegistrySeamTests(unittest.TestCase):
    def test_ocr_and_ocr_scripted_are_registered(self):
        available = registry.available("extractor")
        self.assertIn("ocr", available)
        self.assertIn("ocr-scripted", available)

    def test_make_extractor_ocr_scripted_resolves_to_scripted_ocr(self):
        backend = registry.make_extractor("ocr-scripted", {"frame1": {"text": "hi"}})
        self.assertIsInstance(backend, ScriptedOCR)
        self.assertEqual(backend.extract("cond", "frame1").facts["text"], "hi")

    def test_make_extractor_ocr_resolves_to_tesseract_ocr_without_the_optional_deps(self):
        # Construction must succeed regardless of whether pytesseract/Pillow are installed in
        # this environment — the seam never requires the optional extra just to resolve a name.
        backend = registry.make_extractor("ocr")
        self.assertIsInstance(backend, TesseractOCR)


if __name__ == "__main__":
    unittest.main()
