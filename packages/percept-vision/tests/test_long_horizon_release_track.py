"""Manifest-derived deterministic checks for session continuity.

The fixtures use scripted verdicts and speaker labels, so this module gates state-machine semantics
only. It must never be read as model, person re-identification, or speaker-diarization accuracy.
"""

from __future__ import annotations

import copy
import unittest

from percept.harness.synthetic.corpora import BUILDERS
from percept.harness.synthetic.long_horizon_track import (
    ALLOWED_UNSUPPORTED_SEMANTIC_CAPABILITIES,
    REQUIRED_SEMANTIC_CAPABILITY_CONTRACT,
    REQUIRED_SEMANTIC_CAPABILITIES,
    load_long_horizon_manifest,
    run_long_horizon_track,
)


class LongHorizonManifestTests(unittest.TestCase):
    def test_manifest_declares_scope_and_quality_limit(self):
        manifest = load_long_horizon_manifest()
        self.assertTrue(manifest["synthetic"])
        self.assertTrue(manifest["session_scope"])
        self.assertFalse(manifest["quality_claim"])

    def test_manifest_declares_the_independent_semantic_capability_contract(self):
        manifest = load_long_horizon_manifest()
        self.assertEqual(
            set(manifest["required_semantic_capabilities"]),
            REQUIRED_SEMANTIC_CAPABILITIES,
        )
        owned = [
            requirement
            for entry in manifest["cases"]
            for requirement in entry["requirements"]
        ]
        self.assertEqual(set(owned), REQUIRED_SEMANTIC_CAPABILITIES)
        self.assertEqual(len(owned), len(REQUIRED_SEMANTIC_CAPABILITIES))
        contract_unsupported = {
            requirement
            for requirement, binding in REQUIRED_SEMANTIC_CAPABILITY_CONTRACT.items()
            if binding.expected_status == "unsupported"
        }
        self.assertEqual(
            contract_unsupported, ALLOWED_UNSUPPORTED_SEMANTIC_CAPABILITIES
        )
        for entry in manifest["cases"]:
            for requirement in entry["requirements"]:
                binding = REQUIRED_SEMANTIC_CAPABILITY_CONTRACT[requirement]
                with self.subTest(requirement=requirement):
                    self.assertEqual(entry["expected_status"], binding.expected_status)
                    self.assertEqual(entry.get("handler"), binding.handler)
                    self.assertEqual(entry.get("corpus"), binding.corpus)

    def test_manifest_case_selection_resolves_without_frozen_counts(self):
        manifest = load_long_horizon_manifest()
        report = run_long_horizon_track(manifest)
        by_id = {result.id: result for result in report.results}

        for entry in manifest["cases"]:
            if entry.get("expected_status") == "unsupported":
                continue
            result = by_id[entry["id"]]
            stream, _labels = BUILDERS[entry["corpus"]]()
            expected_ids = (
                tuple(case["id"] for case in stream["cases"])
                if entry.get("case_selector") == "all"
                else tuple(entry["case_ids"])
            )
            with self.subTest(entry=entry["id"]):
                self.assertEqual(result.fixture_cases, expected_ids)


class LongHorizonStateMachineTrackTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = load_long_horizon_manifest()
        cls.report = run_long_horizon_track(cls.manifest)

    def test_every_implemented_manifest_entry_executes_successfully(self):
        failures = [result for result in self.report.results if result.status == "fail"]
        self.assertEqual(
            failures,
            [],
            "\n".join(f"{result.id}: {result.detail}" for result in failures),
        )
        self.assertTrue(self.report.structurally_complete)
        self.assertTrue(self.report.execution_succeeded)

    def test_unsupported_requirements_block_release_readiness(self):
        expected = tuple(
            requirement
            for entry in self.manifest["cases"]
            if entry.get("expected_status") == "unsupported"
            for requirement in entry["requirements"]
        )
        self.assertEqual(self.report.unsupported_requirements, expected)
        self.assertIn("restart_continuity", expected)
        self.assertIn("person_reidentification_accuracy", expected)
        self.assertIn("speaker_diarization_accuracy", expected)
        self.assertEqual(set(expected), ALLOWED_UNSUPPORTED_SEMANTIC_CAPABILITIES)
        self.assertFalse(self.report.release_ready)
        self.assertFalse(hasattr(self.report, "passed"))

    def test_report_cannot_be_misread_as_a_quality_claim(self):
        self.assertTrue(self.report.synthetic)
        self.assertTrue(self.report.session_scope)
        self.assertFalse(self.report.quality_claim)

    def test_missing_fixture_fails_closed(self):
        broken = copy.deepcopy(self.manifest)
        entry = next(item for item in broken["cases"] if item.get("case_selector") == "all")
        entry.pop("case_selector")
        entry["case_ids"] = ["fixture-that-does-not-exist"]

        report = run_long_horizon_track(broken)
        result = next(item for item in report.results if item.id == entry["id"])
        self.assertEqual(result.status, "fail")
        self.assertFalse(report.execution_succeeded)
        self.assertFalse(report.release_ready)

    def test_deleting_a_manifest_case_fails_closed(self):
        broken = copy.deepcopy(self.manifest)
        broken["cases"] = broken["cases"][1:]

        with self.assertRaisesRegex(
            AssertionError, "do not own required semantic capabilities"
        ):
            run_long_horizon_track(broken)

    def test_deleting_an_unsupported_declaration_fails_closed(self):
        broken = copy.deepcopy(self.manifest)
        broken["cases"] = [
            entry
            for entry in broken["cases"]
            if entry["id"] != "real_person_and_speaker_accuracy"
        ]

        with self.assertRaisesRegex(
            AssertionError, "do not own required semantic capabilities"
        ):
            run_long_horizon_track(broken)

    def test_deleting_a_case_requirement_fails_closed(self):
        broken = copy.deepcopy(self.manifest)
        broken["cases"][0]["requirements"].pop()

        with self.assertRaisesRegex(
            AssertionError, "do not own required semantic capabilities"
        ):
            run_long_horizon_track(broken)

    def test_deleting_a_capability_declaration_fails_closed(self):
        broken = copy.deepcopy(self.manifest)
        broken["required_semantic_capabilities"].pop()

        with self.assertRaisesRegex(
            AssertionError, "must exactly match the code-owned contract"
        ):
            run_long_horizon_track(broken)

    def test_swapping_registered_handlers_between_requirements_fails_closed(self):
        broken = copy.deepcopy(self.manifest)
        first, second = broken["cases"][:2]
        first["handler"], second["handler"] = second["handler"], first["handler"]

        with self.assertRaisesRegex(AssertionError, "must bind handler"):
            run_long_horizon_track(broken)

    def test_swapping_registered_corpora_between_requirements_fails_closed(self):
        broken = copy.deepcopy(self.manifest)
        first, second = broken["cases"][:2]
        first["corpus"], second["corpus"] = second["corpus"], first["corpus"]

        with self.assertRaisesRegex(AssertionError, "must bind corpus"):
            run_long_horizon_track(broken)

    def test_deleting_an_implemented_handler_fails_closed(self):
        broken = copy.deepcopy(self.manifest)
        broken["cases"][0].pop("handler")

        with self.assertRaisesRegex(AssertionError, "must bind handler"):
            run_long_horizon_track(broken)

    def test_deleting_an_implemented_corpus_fails_closed(self):
        broken = copy.deepcopy(self.manifest)
        broken["cases"][0].pop("corpus")

        with self.assertRaisesRegex(AssertionError, "must bind corpus"):
            run_long_horizon_track(broken)

    def test_reclassifying_an_implemented_capability_as_unsupported_fails_closed(self):
        broken = copy.deepcopy(self.manifest)
        entry = broken["cases"][0]
        entry["expected_status"] = "unsupported"
        entry["reason"] = "weakened by manifest mutation"
        for field in ("handler", "corpus", "case_selector"):
            entry.pop(field)

        with self.assertRaisesRegex(AssertionError, "status must be 'implemented'"):
            run_long_horizon_track(broken)

    def test_changing_an_unsupported_capability_status_fails_closed(self):
        broken = copy.deepcopy(self.manifest)
        entry = next(
            item
            for item in broken["cases"]
            if "restart_continuity" in item["requirements"]
        )
        entry["expected_status"] = "implemented"
        entry.pop("reason")
        entry["handler"] = "episode_boundaries"
        entry["corpus"] = "episode_counting"
        entry["case_selector"] = "all"

        with self.assertRaisesRegex(AssertionError, "status must be 'unsupported'"):
            run_long_horizon_track(broken)


if __name__ == "__main__":
    unittest.main()
