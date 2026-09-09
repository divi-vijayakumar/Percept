import hashlib
import json
import pathlib


ROOT = pathlib.Path(__file__).resolve().parent


def sha256(name: str) -> str:
    return hashlib.sha256((ROOT / name).read_bytes()).hexdigest()


def test_failure_receipt_binds_code_and_preserves_seals() -> None:
    receipt = json.loads((ROOT / "b4-calibration-result.json").read_text())
    assert receipt["status"] == "fail"
    assert receipt["search"]["configurations_evaluated"] == 135
    assert receipt["search"]["configurations_meeting_precision_and_recall_bar"] == 0
    assert receipt["best_measured_result"]["timely_recall"] is None
    assert receipt["best_measured_result"]["receiver_ack_metrics_measured"] is False
    assert receipt["validation_opened"] is False
    assert receipt["qualification_labels_opened"] is False
    assert receipt["pilot_276_opened"] is False
    assert receipt["paid_activity_performed"] is False
    identities = receipt["implementation_identities"]
    assert identities["zone_entry_watch_sha256"] == sha256("zone_entry_watch.py")
    assert identities["cache_builder_sha256"] == sha256("build_detection_cache.py")
    assert identities["cache_evaluator_sha256"] == sha256("evaluate_cache.py")
    assert identities["sweep_runner_sha256"] == sha256("sweep_cache.py")
    assert identities["search_space_sha256"] == sha256("b4-search-space.json")


def test_exposure_registry_is_append_only_sequence() -> None:
    registry = ROOT.parent / "admission" / "exposure-registry.jsonl"
    rows = [json.loads(line) for line in registry.read_text().splitlines()]
    assert [row["sequence"] for row in rows] == list(range(len(rows)))
    assert any(row["event"] == "b4_calibration_evaluated" for row in rows)
    assert rows[-1]["event"] == "b4_v2_calibration_evaluated"
    assert rows[-1]["qualification_labels_opened"] is False
