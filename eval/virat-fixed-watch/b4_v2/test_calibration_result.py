import json
import pathlib


ROOT = pathlib.Path(__file__).resolve().parent


def test_complete_calibration_failure_is_internally_consistent_and_sealed() -> None:
    result = json.loads((ROOT / "b4-v2-calibration-result.json").read_text())
    assert result["status"] == "fail"
    assert result["search"]["configurations_evaluated"] == 972
    assert result["search"]["configurations_passing"] == 0
    assert result["search"]["configurations_with_full_recall"] == 0
    assert result["search"]["configurations_with_zero_fp"] == 0
    rows = result["source_distribution"]
    assert len(rows) == 21
    assert sum(row[1] for row in rows) == 16
    assert sum(row[2] for row in rows) == 34
    assert sum(row[3] for row in rows) == result["best"]["tp"] == 12
    assert sum(row[4] for row in rows) == result["best"]["fp_against_provisional_gt"] == 22
    assert sum(row[5] for row in rows) == result["best"]["fn_against_official_track_seeds"] == 4
    assert sum(row[6] for row in rows) == result["best"]["ask"] == 5
    assert result["best"]["duplicates"] == 0
    assert result["best"]["timely_recall"] is None
    assert result["reconciliation"]["exact"] is True
    assert result["validation_opened"] is False
    assert result["qualification_labels_opened"] is False
    assert result["pilot_276_opened"] is False
    assert result["paid_activity_performed"] is False
