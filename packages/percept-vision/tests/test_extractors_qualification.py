"""The OCR qualification runner — a fixture-hashed QualificationRecord, never a descriptor field.

All offline against ``percept.extractors.ocr.ScriptedOCR`` — zero API spend, wiring not
intelligence (standing rule 2). Covers:

* :func:`~percept.extractors.qualification.qualify_extractor` scores recall/miss_rate/mean_confidence
  correctly against a labeled fixture set.
* ``fixture_hash`` changes when the fixture set changes (a different expected text, an added
  fixture) — content-addressed provenance, not a floating claim.
* the component identity (``component_role``/``component_name``) stays fixed across fixture
  changes, and ``percept.catalog.ComponentDescriptor`` never carries a measured number — identity
  and measurement stay in separate types.
"""

from __future__ import annotations

import unittest

import pytest

from percept import catalog
from percept.catalog import ROLE_BACKEND, QualificationRecord
from percept.extractors.ocr import ScriptedOCR
from percept.extractors.qualification import OCRFixture, qualify_extractor


def _extractor() -> ScriptedOCR:
    return ScriptedOCR({
        "doc1": {"text": "INVOICE 2291", "confidence": 0.9},
        "doc2": {"text": "RECEIPT 004", "confidence": 0.7},
        "doc3": {"text": "no useful text here", "confidence": 0.4},
        # "doc4" deliberately unscripted -> always an empty Enrichment (a miss).
    })


def _fixtures() -> list[OCRFixture]:
    return [
        OCRFixture(frame="doc1", expected_text="INVOICE"),
        OCRFixture(frame="doc2", expected_text="RECEIPT"),
        OCRFixture(frame="doc3", expected_text="MISMATCH"),  # scripted text present, wrong content
        OCRFixture(frame="doc4", expected_text="anything"),   # unscripted -> empty Enrichment
    ]


class QualifyExtractorTests(unittest.TestCase):
    def test_empty_fixtures_rejected(self):
        with pytest.raises(ValueError):
            qualify_extractor(_extractor(), [])

    def test_metrics_scored_against_labeled_fixtures(self):
        record = qualify_extractor(_extractor(), _fixtures(), component_name="ocr-scripted")
        # doc1, doc2 hit; doc3 scripted but text mismatched; doc4 unscripted (empty) -> 1 miss.
        self.assertAlmostEqual(record.metrics["recall"], 2 / 4)
        self.assertAlmostEqual(record.metrics["miss_rate"], 1 / 4)
        # mean_confidence averages over doc1/doc2/doc3 (all non-empty, self-reported confidence)
        self.assertAlmostEqual(record.metrics["mean_confidence"], (0.9 + 0.7 + 0.4) / 3)

    def test_record_is_a_qualification_record_stamped_to_backend_role(self):
        record = qualify_extractor(_extractor(), _fixtures(), component_name="ocr-scripted")
        self.assertIsInstance(record, QualificationRecord)
        self.assertEqual(record.component_role, ROLE_BACKEND)
        self.assertEqual(record.component_name, "ocr-scripted")

    def test_fixture_hash_changes_when_a_fixture_changes(self):
        base = qualify_extractor(_extractor(), _fixtures(), component_name="ocr-scripted")

        changed = list(_fixtures())
        changed[0] = OCRFixture(frame="doc1", expected_text="DIFFERENT")
        after_edit = qualify_extractor(_extractor(), changed, component_name="ocr-scripted")
        self.assertNotEqual(base.fixture_hash, after_edit.fixture_hash)

    def test_fixture_hash_changes_when_a_fixture_is_added(self):
        base = qualify_extractor(_extractor(), _fixtures(), component_name="ocr-scripted")

        appended = list(_fixtures()) + [OCRFixture(frame="doc2", expected_text="RECEIPT")]
        after_add = qualify_extractor(_extractor(), appended, component_name="ocr-scripted")
        self.assertNotEqual(base.fixture_hash, after_add.fixture_hash)

    def test_fixture_hash_is_stable_for_the_same_fixture_set(self):
        first = qualify_extractor(_extractor(), _fixtures(), component_name="ocr-scripted")
        second = qualify_extractor(_extractor(), _fixtures(), component_name="ocr-scripted")
        self.assertEqual(first.fixture_hash, second.fixture_hash)

    def test_component_identity_is_unaffected_by_fixture_changes(self):
        base = qualify_extractor(_extractor(), _fixtures(), component_name="ocr-scripted")
        changed = list(_fixtures())
        changed[0] = OCRFixture(frame="doc1", expected_text="DIFFERENT")
        after_edit = qualify_extractor(_extractor(), changed, component_name="ocr-scripted")
        # the fixture hash moved, but WHICH component this measured did not.
        self.assertEqual(base.component_role, after_edit.component_role)
        self.assertEqual(base.component_name, after_edit.component_name)

    def test_catalog_descriptor_for_ocr_extractor_carries_no_measured_number(self):
        # The descriptor the catalog surfaces for the registered "ocr-scripted" backend must stay
        # identity-only — QualificationRecord.metrics never leaks back onto it.
        components = catalog.list_components(role=catalog.ROLE_BACKEND)
        ocr_descriptor = next(c for c in components if c.seam == "extractor" and c.name == "ocr-scripted")
        payload = ocr_descriptor.to_dict()
        self.assertNotIn("metrics", payload)
        self.assertNotIn("recall", str(payload))
        self.assertNotIn("fixture_hash", payload)


if __name__ == "__main__":
    unittest.main()
