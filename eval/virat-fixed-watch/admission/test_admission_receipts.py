import json
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parent


class AdmissionReceiptTests(unittest.TestCase):
    def test_scope_and_feasibility_stop_at_first_failed_gate(self):
        b0 = json.loads((ROOT / "b0-scope-freeze.json").read_text())
        b1 = json.loads((ROOT / "b1-feasibility.json").read_text())
        self.assertEqual("pass", b0["status"])
        self.assertEqual("fail", b1["status"])
        self.assertFalse(b1["next_milestone_admitted"])
        self.assertFalse(b1["paid_activity_performed"])
        self.assertFalse(
            b1["annotation_semantics"]["whole_frame_negative_intervals_exhaustive"]
        )

    def test_projection_is_exact_and_fits_but_rights_block_download(self):
        b1 = json.loads((ROOT / "b1-feasibility.json").read_text())
        corpus = b1["official_corpus"]
        self.assertEqual(
            corpus["zip_package_bytes"] + corpus["diva_archive_bytes"],
            corpus["exact_projected_acquisition_bytes"],
        )
        self.assertGreater(
            b1["preflight"]["evidence_volume"]["available_bytes"],
            corpus["exact_projected_acquisition_bytes"],
        )
        self.assertFalse(b1["rights"]["signed_acceptance_receipt_verified"])
        self.assertFalse(b1["rights"]["protected_download_admitted"])

    def test_exposure_registry_is_ordered_and_disjoint_from_holdout(self):
        records = [
            json.loads(line)
            for line in (ROOT / "exposure-registry.jsonl").read_text().splitlines()
        ]
        self.assertEqual(list(range(len(records))), [r["sequence"] for r in records])
        exposed = {
            r["source_group"]
            for r in records
            if r["event"] == "source_group_exposed"
        }
        self.assertEqual({"0002", "0400", "0401"}, exposed)
        b1 = json.loads((ROOT / "b1-feasibility.json").read_text())
        partitions = b1["partition_freeze"]
        self.assertTrue(exposed.isdisjoint(partitions["sealed_unassigned_groups"]))
        self.assertEqual([], partitions["validation_groups"])
        self.assertEqual([], partitions["qualification_groups"])


if __name__ == "__main__":
    unittest.main()
