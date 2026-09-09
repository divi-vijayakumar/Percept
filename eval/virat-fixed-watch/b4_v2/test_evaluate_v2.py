import json
import pathlib
import zipfile

from evaluate_v2 import evaluate, load_calibration


def test_synthetic_cache_loads_and_reconciles(tmp_path: pathlib.Path) -> None:
    cache_dir = tmp_path / "calibration"
    cache_dir.mkdir()
    source = "VIRAT_S_050201_00_000000_000003"
    cache = {
        "partition": "calibration_exposed",
        "source": {
            "name": f"{source}.mp4",
            "fps": 1.0,
            "frame_count": 3,
            "width": 1000,
            "height": 1000,
        },
        "samples": [
            {"source_seconds": 0.0, "inference_seconds": 0.1, "person_detections": [{"bbox": [50, 700, 150, 800], "score": 0.9}]},
            {"source_seconds": 1.0, "inference_seconds": 0.1, "person_detections": [{"bbox": [400, 700, 500, 800], "score": 0.9}]},
            {"source_seconds": 2.0, "inference_seconds": 0.1, "person_detections": []},
        ],
    }
    (cache_dir / f"{source}.detections.json").write_text(json.dumps(cache))
    zones = {
        "partitions": {"calibration_exposed": ["0502"]},
        "zones": {"0502": {"polygon": [[0.2, 0], [0.85, 0], [0.82, 0.95], [0.25, 0.95]]}},
    }
    zone_path = tmp_path / "zones.json"
    zone_path.write_text(json.dumps(zones))
    annotations = tmp_path / "annotations.zip"
    with zipfile.ZipFile(annotations, "w") as archive:
        archive.writestr(
            f"annotations/{source}.viratdata.objects.txt",
            "1 2 0 50 700 100 100 1\n1 2 1 400 700 100 100 1\n",
        )
    data = load_calibration(cache_dir, annotations, zone_path)
    result = evaluate(
        data,
        {
            "score_threshold": 0.1,
            "maximum_association_distance": 0.5,
            "maximum_track_gap_seconds": 2.1,
            "minimum_outside_observations": 1,
            "boundary_margin_normalized": 0.0,
            "duplicate_footpoint_distance": 0.0,
            "refractory_seconds": 0.0,
        },
    )
    assert result["reconciliation"]["exact"] is True
    assert result["occurrences"] == 1
    assert result["tp"] == 1
    assert result["fp"] == 0
    assert result["fn"] == 0
    assert result["timely_recall"] is None
