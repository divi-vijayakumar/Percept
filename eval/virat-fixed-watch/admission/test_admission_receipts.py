import hashlib
import json
import math
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parent


class AdmissionReceiptTests(unittest.TestCase):
    def test_effective_scope_promotes_zone_entry_and_demotes_whole_frame(self):
        amendment = json.loads((ROOT / "b0-scope-amendment-1.json").read_text())
        self.assertEqual("pass", amendment["status"])
        self.assertEqual("F2-Z1", amendment["effective_launch_watch"]["id"])
        self.assertFalse(amendment["whole_frame_f1"]["launch_blocking"])
        self.assertFalse(amendment["paid_provider_calls_authorized"])
        self.assertFalse(amendment["paid_benchmarks_authorized"])
        self.assertFalse(amendment["soak_72h_authorized"])
        plan = (ROOT / amendment["plan_amendment"]["path"]).resolve()
        self.assertEqual(amendment["plan_amendment"]["bytes"], plan.stat().st_size)
        self.assertEqual(
            amendment["plan_amendment"]["sha256"],
            hashlib.sha256(plan.read_bytes()).hexdigest(),
        )

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

    def test_projection_is_exact_and_fits_with_rights_admitted(self):
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
        self.assertTrue(b1["rights"]["signed_acceptance_founder_confirmed"])
        self.assertTrue(b1["rights"]["protected_acquisition_rights_admitted"])
        self.assertFalse(b1["rights"]["protected_download_performed"])

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
        self.assertEqual({"0002", "0400", "0401", "0502"}, exposed)
        b1 = json.loads((ROOT / "b1-feasibility.json").read_text())
        partitions = b1["partition_freeze"]
        self.assertTrue(exposed.isdisjoint(partitions["sealed_unassigned_groups"]))
        self.assertEqual([], partitions["validation_groups"])
        self.assertEqual([], partitions["qualification_groups"])

    def test_amended_zone_partitions_are_disjoint_and_sized(self):
        manifest = json.loads(
            (ROOT / "f2-z1-zones-and-partitions.v1.json").read_text()
        )
        parts = manifest["partitions"]
        named = [
            set(parts["calibration_exposed"]),
            set(parts["validation"]),
            set(parts["qualification"]),
            set(parts["reserve"]),
        ]
        for index, left in enumerate(named):
            for right in named[index + 1 :]:
                self.assertTrue(left.isdisjoint(right))
        survey = manifest["aggregate_seed_rate_survey"]
        self.assertEqual(
            sum(survey["qualification"].values()), survey["qualification_total"]
        )
        self.assertEqual(276, survey["qualification_total"])
        self.assertEqual(
            299,
            survey[
                "minimum_zero_failure_n_for_one_sided_95pct_lower_bound_above_0_99"
            ],
        )
        self.assertEqual(23, survey["event_shortfall"])
        self.assertFalse(survey["sample_size_admitted"])
        self.assertEqual(set(parts["qualification"]), set(survey["qualification"]))

    def test_zone_entry_b1_stops_before_b2_on_measured_shortfall(self):
        receipt = json.loads((ROOT / "b1-zone-entry-readmission.json").read_text())
        self.assertEqual("fail", receipt["status"])
        self.assertEqual("F2-Z1", receipt["effective_launch_watch"])
        self.assertFalse(receipt["next_milestone_admitted"])
        self.assertFalse(receipt["paid_activity_performed"])
        annotations = receipt["official_annotation_package"]
        self.assertEqual(315, annotations["proper_object_files"])
        self.assertEqual(6, annotations["malformed_object_files"])
        self.assertEqual(12, annotations["malformed_rows"])
        survey = receipt["rate_survey"]
        self.assertEqual(
            math.ceil(math.log(0.05) / math.log(0.99)),
            survey["minimum_zero_failure_n_for_one_sided_95pct_lower_bound_above_0_99"],
        )
        self.assertEqual(
            survey["minimum_zero_failure_n_for_one_sided_95pct_lower_bound_above_0_99"]
            - survey["qualification_seed_occurrences"],
            survey["shortfall"],
        )
        self.assertFalse(survey["sample_size_admitted"])
        manifest_ref = receipt["media_manifest"]
        manifest_path = ROOT / manifest_ref["path"]
        self.assertEqual(manifest_ref["bytes"], manifest_path.stat().st_size)
        self.assertEqual(
            manifest_ref["sha256"], hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        )
        manifest = json.loads(manifest_path.read_text())
        self.assertEqual(manifest_ref["selected_files"], manifest["summary"]["files"])
        self.assertEqual(manifest_ref["selected_bytes"], manifest["summary"]["bytes"])
        self.assertTrue(
            all(
                row["annotation"]["malformed_rows"] == 0
                for row in manifest["media"]
            )
        )


if __name__ == "__main__":
    unittest.main()
