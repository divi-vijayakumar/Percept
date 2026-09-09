#!/usr/bin/env python3
"""Score B4 detection caches against provisional calibration/validation GT."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import zipfile

B2 = pathlib.Path(__file__).resolve().parents[1] / "b2"
sys.path.insert(0, str(B2))

from scorer import Incident, latency_summary, score  # noqa: E402
from zone_entry import parse_virat_object_rows, transform_zone_entries  # noqa: E402
from zone_entry_watch import Detection, ZoneEntryWatch  # noqa: E402


def evaluate(
    *,
    cache_dir: pathlib.Path,
    annotations_zip: pathlib.Path,
    zone_manifest: pathlib.Path,
    score_threshold: float,
    association_distance: float,
    track_gap_seconds: float,
    dwell_seconds: float,
) -> dict[str, object]:
    caches = [
        path
        for path in sorted(cache_dir.glob("*.detections.json"))
        if not path.name.startswith("._")
    ]
    if not caches:
        raise ValueError("no detection caches")
    zones = json.loads(zone_manifest.read_text())
    occurrences = []
    incidents = []
    source_hours = 0.0
    inference_values = []
    with zipfile.ZipFile(annotations_zip) as archive:
        members = {
            pathlib.PurePosixPath(info.filename).name: info
            for info in archive.infolist()
            if not info.is_dir()
        }
        for cache_path in caches:
            cache = json.loads(cache_path.read_text())
            partition = cache["partition"]
            if partition not in ("calibration_exposed", "validation"):
                raise ValueError("qualification cache is forbidden in B4")
            source = cache["source"]
            group = source["name"].split("_")[2][:4]
            if group not in zones["partitions"][partition]:
                raise ValueError("cache source group is not frozen for its partition")
            annotation_name = f"{pathlib.Path(source['name']).stem}.viratdata.objects.txt"
            info = members.get(annotation_name)
            if info is None:
                continue
            try:
                rows = parse_virat_object_rows(
                    archive.read(info).decode("utf-8").splitlines()
                )
            except ValueError:
                continue
            occurrences.extend(
                transform_zone_entries(
                    rows,
                    source_id=pathlib.Path(source["name"]).stem,
                    fps=source["fps"],
                    frame_width=source["width"],
                    frame_height=source["height"],
                    polygon=zones["zones"][group]["polygon"],
                    dwell_seconds=dwell_seconds,
                )
            )
            watch = ZoneEntryWatch(
                polygon=zones["zones"][group]["polygon"],
                frame_width=source["width"],
                frame_height=source["height"],
                score_threshold=score_threshold,
                maximum_association_distance=association_distance,
                maximum_track_gap_seconds=track_gap_seconds,
                dwell_seconds=dwell_seconds,
            )
            for sample in cache["samples"]:
                detections = [
                    Detection(tuple(item["bbox"]), item["score"])
                    for item in sample["person_detections"]
                ]
                inference_values.append(sample["inference_seconds"])
                for track_id in watch.observe(detections, sample["source_seconds"]):
                    incidents.append(
                        Incident(
                            incident_id=(
                                f"{pathlib.Path(source['name']).stem}:"
                                f"{sample['source_seconds']}:{track_id}"
                            ),
                            action="fire",
                            source_seconds=sample["source_seconds"],
                            stage_latency_seconds={
                                "detector_inference": sample["inference_seconds"]
                            },
                            source_id=pathlib.Path(source["name"]).stem,
                        )
                    )
            source_hours += ((source["frame_count"] - 1) / source["fps"]) / 3600
    result = score(
        occurrences,
        incidents,
        match_window_seconds=2.0,
        timely_deadline_seconds=2.0,
        source_hours=source_hours,
    )
    # Detection-cache evaluation has no receiver or ACK path. A numeric zero here would look like a
    # measured timely-recall failure, so preserve the distinction required by the reporting contract.
    result["timely_recall"] = None
    result["unacknowledged"] = None
    result.update(
        {
            "schema": "percept.benchmark.b4-cache-score.v1",
            "partition": json.loads(caches[0].read_text())["partition"],
            "configuration": {
                "score_threshold": score_threshold,
                "association_distance": association_distance,
                "track_gap_seconds": track_gap_seconds,
                "dwell_seconds": dwell_seconds,
            },
            "provisional_gt": True,
            "quality_metrics_provisional": True,
            "occurrences": len(occurrences),
            "incidents": len(incidents),
            "source_hours": source_hours,
            "all_detector_inference_seconds": latency_summary(inference_values),
            "receiver_ack_metrics_measured": False,
            "timely_recall_interpretation": "not measured in offline calibration cache",
            "qualification_labels_opened": False,
            "remote_provider_calls": 0,
            "paid_activity_performed": False,
        }
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=pathlib.Path, required=True)
    parser.add_argument("--annotations-zip", type=pathlib.Path, required=True)
    parser.add_argument("--zone-manifest", type=pathlib.Path, required=True)
    parser.add_argument("--score-threshold", type=float, required=True)
    parser.add_argument("--association-distance", type=float, required=True)
    parser.add_argument("--track-gap-seconds", type=float, required=True)
    parser.add_argument("--dwell-seconds", type=float, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    arguments = parser.parse_args()
    payload = evaluate(
        cache_dir=arguments.cache_dir,
        annotations_zip=arguments.annotations_zip,
        zone_manifest=arguments.zone_manifest,
        score_threshold=arguments.score_threshold,
        association_distance=arguments.association_distance,
        track_gap_seconds=arguments.track_gap_seconds,
        dwell_seconds=arguments.dwell_seconds,
    )
    arguments.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
