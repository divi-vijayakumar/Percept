"""percept.extractors.fields — normalization table, ScriptedFields/FieldNormalizer, registry/catalog
surfacing, and the composition helper.

Table-driven per standing rule: every normalization RULE this module documents gets one row here,
including the incident that motivated the module — "25,000 dollars" (spoken, unlabeled) vs
"threshold: 25,000 dollars" (visible, labeled) must normalize to the SAME canonical string despite
differing surface form. All offline / zero API spend (standing rule 2).
"""

from __future__ import annotations

import unittest

from percept import catalog, registry
from percept.extraction import Enrichment
from percept.extractors.fields import (
    FieldNormalizer,
    ScriptedFields,
    chain_ocr_then_fields,
    classify_value,
    extract_fields,
)
from percept.extractors.ocr import ScriptedOCR


# ---- normalization table -----------------------------------------------------------------------
# (raw value, expected kind, expected normalized)  -- normalized=None encodes "never guess".
NORMALIZATION_TABLE: tuple[tuple[str, str, "str | None"], ...] = (
    # currency: symbol / word / ISO-code all converge on the SAME canonical string
    ("25,000 dollars", "currency", "USD 25000.00"),
    ("$25,000", "currency", "USD 25000.00"),
    ("USD 25000", "currency", "USD 25000.00"),
    ("$25,000.00", "currency", "USD 25000.00"),
    ("100 euros", "currency", "EUR 100.00"),
    ("€100", "currency", "EUR 100.00"),
    ("£50", "currency", "GBP 50.00"),
    ("EUR 100", "currency", "EUR 100.00"),
    # currency: ambiguous word -> shape recognized, value NOT invented
    ("25,000 kr", "currency", None),
    ("100 krona", "currency", None),
    # dates: unambiguous forms -> ISO-8601
    ("2026-03-05", "date", "2026-03-05"),
    ("March 5, 2026", "date", "2026-03-05"),
    ("March 5 2026", "date", "2026-03-05"),
    ("5 March 2026", "date", "2026-03-05"),
    # dates: numeric MM/DD vs DD/MM -> genuinely ambiguous, never guessed
    ("03/05/2026", "date", None),
    ("3.5.2026", "date", None),
    # dates: syntactically shaped but not a real calendar date -> never guessed
    ("2026-13-40", "date", None),
    # frequencies: canonical per-period form
    ("monthly", "frequency", "1/month"),
    ("per month", "frequency", "1/month"),
    ("a month", "frequency", "1/month"),
    ("weekly", "frequency", "1/week"),
    ("daily", "frequency", "1/day"),
    ("annually", "frequency", "1/year"),
    ("quarterly", "frequency", "1/quarter"),
    # plain numbers
    ("42", "number", "42"),
    ("3.14", "number", "3.14"),
    ("1,234", "number", "1234"),
    # unparseable -> unknown, never a guess
    ("banana", "unknown", None),
    ("", "unknown", None),
)


class NormalizationTableTests(unittest.TestCase):
    def test_normalization_table(self):
        for raw, expected_kind, expected_normalized in NORMALIZATION_TABLE:
            with self.subTest(raw=raw):
                kind, normalized = classify_value(raw)
                self.assertEqual(kind, expected_kind, f"{raw!r} -> kind {kind!r}, want {expected_kind!r}")
                self.assertEqual(
                    normalized, expected_normalized,
                    f"{raw!r} -> normalized {normalized!r}, want {expected_normalized!r}",
                )

    def test_the_client_incident_spoken_vs_visible_normalize_identically(self):
        """The exact motivating case: unlabeled spoken mention vs labeled visible mention of the
        SAME value must normalize to the SAME canonical string, even though field/raw differ."""
        spoken = extract_fields("25,000 dollars")
        visible = extract_fields("threshold: 25,000 dollars")

        self.assertEqual(len(spoken), 1)
        self.assertEqual(len(visible), 1)
        self.assertIsNone(spoken[0]["field"])
        self.assertEqual(visible[0]["field"], "threshold")
        self.assertEqual(spoken[0]["raw"], "25,000 dollars")
        self.assertEqual(visible[0]["raw"], "25,000 dollars")
        self.assertEqual(spoken[0]["kind"], "currency")
        self.assertEqual(visible[0]["kind"], "currency")
        # THE assertion: normalized values agree despite differing surface text / labeling.
        self.assertEqual(spoken[0]["normalized"], visible[0]["normalized"])
        self.assertEqual(spoken[0]["normalized"], "USD 25000.00")

    def test_ambiguous_currency_keeps_raw_and_kind_but_never_invents_a_code(self):
        fields = extract_fields("25,000 kr")
        self.assertEqual(fields[0]["kind"], "currency")
        self.assertIsNone(fields[0]["normalized"])
        self.assertEqual(fields[0]["raw"], "25,000 kr")

    def test_ambiguous_numeric_date_keeps_raw_and_kind_but_never_guesses_locale(self):
        fields = extract_fields("03/05/2026")
        self.assertEqual(fields[0]["kind"], "date")
        self.assertIsNone(fields[0]["normalized"])

    def test_multi_field_text_splits_on_newline_not_comma(self):
        fields = extract_fields("threshold: 25,000 dollars\ndue: 2026-03-05")
        self.assertEqual(len(fields), 2)
        self.assertEqual(fields[0]["field"], "threshold")
        self.assertEqual(fields[0]["normalized"], "USD 25000.00")
        self.assertEqual(fields[1]["field"], "due")
        self.assertEqual(fields[1]["normalized"], "2026-03-05")

    def test_empty_text_yields_no_fields(self):
        self.assertEqual(extract_fields(""), [])
        self.assertEqual(extract_fields("   "), [])


class FieldNormalizerTests(unittest.TestCase):
    def test_plain_string_frame_is_read_as_text(self):
        backend = FieldNormalizer()
        e = backend.extract("cond", "threshold: 25,000 dollars")
        self.assertEqual(e.facts["fields"][0]["normalized"], "USD 25000.00")

    def test_mapping_frame_reads_the_text_key_the_ocr_composition_point(self):
        backend = FieldNormalizer()
        e = backend.extract("cond", {"text": "threshold: 25,000 dollars", "regions": []})
        self.assertEqual(e.facts["fields"][0]["normalized"], "USD 25000.00")

    def test_no_text_never_raises_returns_empty_enrichment(self):
        backend = FieldNormalizer()
        self.assertTrue(backend.extract("cond", {"regions": []}).is_empty)
        self.assertTrue(backend.extract("cond", 12345).is_empty)
        self.assertTrue(backend.extract("cond", "").is_empty)

    def test_text_with_no_field_shaped_mentions_is_empty(self):
        backend = FieldNormalizer()
        self.assertTrue(backend.extract("cond", "hello there, nothing to see").is_empty)


class ScriptedFieldsDeterminismTests(unittest.TestCase):
    def test_scripted_by_token_is_deterministic_across_calls(self):
        backend = ScriptedFields({"frame1": {"text": "threshold: 25,000 dollars"}})
        e1 = backend.extract("cond", "frame1")
        e2 = backend.extract("cond", "frame1")
        self.assertEqual(e1.to_dict(), e2.to_dict())
        self.assertEqual(e1.facts["fields"][0]["normalized"], "USD 25000.00")

    def test_scripted_prebuilt_fields_used_as_is_no_reparse(self):
        prebuilt = [{"field": "x", "raw": "weird raw", "normalized": "NOT-REAL", "kind": "unknown"}]
        backend = ScriptedFields({"frame1": {"fields": prebuilt}})
        e = backend.extract("cond", "frame1")
        self.assertEqual(e.facts["fields"], prebuilt)

    def test_scripted_ready_made_enrichment_returned_verbatim(self):
        made = Enrichment(facts={"fields": [{"field": "a", "raw": "1", "normalized": "1", "kind": "number"}]})
        backend = ScriptedFields({"frame1": made})
        self.assertIs(backend.extract("cond", "frame1"), made)

    def test_unscripted_frame_returns_empty_never_raises(self):
        backend = ScriptedFields({"frame1": {"text": "hit"}})
        self.assertTrue(backend.extract("cond", "totally-unscripted").is_empty)

    def test_non_string_frame_keys_off_stable_content_hash_like_ocr(self):
        class Frame:
            def __repr__(self) -> str:
                return "FIELDS-FRAME-XYZ"

        from percept.extractors.ocr import _frame_key

        key = _frame_key(Frame())
        backend = ScriptedFields({key: {"text": "threshold: 25,000 dollars"}})
        e = backend.extract("cond", Frame())
        self.assertEqual(e.facts["fields"][0]["normalized"], "USD 25000.00")


class CompositionHelperTests(unittest.TestCase):
    def test_chain_ocr_then_fields_merges_text_regions_and_fields(self):
        ocr = ScriptedOCR({"doc_frame": {"text": "threshold: 25,000 dollars", "regions": [{"text": "x"}]}})
        chained = chain_ocr_then_fields(ocr, FieldNormalizer())
        e = chained.extract("cond", "doc_frame")
        self.assertEqual(e.facts["text"], "threshold: 25,000 dollars")
        self.assertEqual(len(e.facts["regions"]), 1)
        self.assertEqual(e.facts["fields"][0]["normalized"], "USD 25000.00")

    def test_chain_defaults_to_a_field_normalizer_when_none_given(self):
        ocr = ScriptedOCR({"doc_frame": {"text": "42 dollars"}})
        chained = chain_ocr_then_fields(ocr)
        e = chained.extract("cond", "doc_frame")
        self.assertEqual(e.facts["fields"][0]["normalized"], "USD 42.00")

    def test_chain_passes_through_empty_ocr_enrichment_unchanged(self):
        ocr = ScriptedOCR({})  # never matches
        chained = chain_ocr_then_fields(ocr)
        e = chained.extract("cond", "doc_frame")
        self.assertTrue(e.is_empty)


class RegistrySeamAndCatalogTests(unittest.TestCase):
    def test_fields_and_fields_scripted_and_ocr_fields_scripted_are_registered(self):
        available = registry.available("extractor")
        self.assertIn("fields", available)
        self.assertIn("fields-scripted", available)
        self.assertIn("ocr-fields-scripted", available)

    def test_make_extractor_fields_scripted_resolves_to_scripted_fields(self):
        backend = registry.make_extractor("fields-scripted", {"frame1": {"text": "42 dollars"}})
        self.assertIsInstance(backend, ScriptedFields)
        self.assertEqual(backend.extract("cond", "frame1").facts["fields"][0]["normalized"], "USD 42.00")

    def test_make_extractor_fields_resolves_to_field_normalizer(self):
        backend = registry.make_extractor("fields")
        self.assertIsInstance(backend, FieldNormalizer)

    def test_make_extractor_ocr_fields_scripted_resolves_to_a_chained_extractor(self):
        backend = registry.make_extractor(
            "ocr-fields-scripted", {"doc_frame": {"text": "threshold: 25,000 dollars"}},
        )
        e = backend.extract("cond", "doc_frame")
        self.assertEqual(e.facts["text"], "threshold: 25,000 dollars")
        self.assertEqual(e.facts["fields"][0]["normalized"], "USD 25000.00")

    def test_catalog_surfaces_fields_backends_under_the_extractor_seam(self):
        snap = catalog.snapshot()
        extractor_names = {c.name for c in snap.by_role(catalog.ROLE_BACKEND) if c.seam == "extractor"}
        self.assertIn("fields", extractor_names)
        self.assertIn("fields-scripted", extractor_names)
        self.assertIn("ocr-fields-scripted", extractor_names)


if __name__ == "__main__":
    unittest.main()
