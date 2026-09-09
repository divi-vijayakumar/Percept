import hashlib
import json
import pathlib


ROOT = pathlib.Path(__file__).resolve().parent


def test_candidate_freeze_binds_implementation_and_seals_later_splits() -> None:
    receipt = json.loads((ROOT / "b4-v2-candidate-freeze.json").read_text())
    names = {
        "tracker_sha256": "persistent_zone_entry_watch.py",
        "evaluator_sha256": "evaluate_v2.py",
        "sweep_runner_sha256": "sweep_v2.py",
        "search_space_sha256": "b4-v2-search-space.json",
    }
    for field, name in names.items():
        assert receipt["identities"][field] == hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
    assert receipt["search_configurations"] == 972
    assert receipt["validation_opened"] is False
    assert receipt["qualification_labels_opened"] is False
    assert receipt["pilot_276_opened"] is False
    assert receipt["paid_activity_performed"] is False
