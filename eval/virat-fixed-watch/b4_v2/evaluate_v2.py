"""Load immutable B4 caches once and evaluate persistent-tracking configurations."""

from __future__ import annotations

from dataclasses import dataclass
import json
import pathlib
import sys
import time
from typing import Any
import zipfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "b2")]

from scorer import Incident, latency_summary, score  # noqa: E402
from zone_entry import Occurrence, parse_virat_object_rows, transform_zone_entries  # noqa: E402
from persistent_zone_entry_watch import (  # noqa: E402
    Detection,
    PersistentZoneEntryWatch,
)


@dataclass(frozen=True)
class SourceData:
    source_id: str
    width: int
    height: int
    polygon: list[list[float]]
    duration_seconds: float
    occurrences: list[Occurrence]
    samples: list[dict[str, Any]]


@dataclass(frozen=True)
class CalibrationData:
    sources: list[SourceData]
    detector_inference_seconds: list[float]
    cached_files: int
    malformed_files_excluded: int


def load_calibration(
    cache_dir: pathlib.Path,
    annotations_zip: pathlib.Path,
    zone_manifest: pathlib.Path,
) -> CalibrationData:
    caches = [
        path
        for path in sorted(cache_dir.glob("*.detections.json"))
        if not path.name.startswith("._")
    ]
    zones = json.loads(zone_manifest.read_text())
    sources = []
    inference_values = []
    malformed = 0
    with zipfile.ZipFile(annotations_zip) as archive:
        members = {
            pathlib.PurePosixPath(info.filename).name: info
            for info in archive.infolist()
            if not info.is_dir()
        }
        for path in caches:
            cache = json.loads(path.read_text())
            if cache["partition"] != "calibration_exposed":
                raise ValueError("v2 evaluator accepts exposed calibration caches only")
            source = cache["source"]
            source_id = pathlib.Path(source["name"]).stem
            group = source["name"].split("_")[2][:4]
            if group not in zones["partitions"]["calibration_exposed"]:
                raise ValueError("source group is not frozen calibration-exposed")
            info = members.get(f"{source_id}.viratdata.objects.txt")
            if info is None:
                malformed += 1
                continue
            try:
                rows = parse_virat_object_rows(
                    archive.read(info).decode("utf-8").splitlines()
                )
            except ValueError:
                malformed += 1
                continue
            occurrences = transform_zone_entries(
                rows,
                source_id=source_id,
                fps=source["fps"],
                frame_width=source["width"],
                frame_height=source["height"],
                polygon=zones["zones"][group]["polygon"],
                dwell_seconds=0.0,
            )
            inference_values.extend(item["inference_seconds"] for item in cache["samples"])
            sources.append(
                SourceData(
                    source_id=source_id,
                    width=source["width"],
                    height=source["height"],
                    polygon=zones["zones"][group]["polygon"],
                    duration_seconds=(source["frame_count"] - 1) / source["fps"],
                    occurrences=occurrences,
                    samples=cache["samples"],
                )
            )
    return CalibrationData(sources, inference_values, len(caches), malformed)


def evaluate(data: CalibrationData, configuration: dict[str, float | int]) -> dict[str, Any]:
    occurrences = [occurrence for source in data.sources for occurrence in source.occurrences]
    incidents = []
    tracker_latencies = []
    suppressed_refractory = 0
    suppressed_duplicate_detections = 0
    suppressed_rising_edge = 0
    observed_samples = 0
    source_hours = sum(source.duration_seconds for source in data.sources) / 3600
    for source in data.sources:
        gate_armed: bool | None = None
        watch = PersistentZoneEntryWatch(
            polygon=source.polygon,
            frame_width=source.width,
            frame_height=source.height,
            score_threshold=float(configuration["score_threshold"]),
            maximum_association_distance=float(configuration["maximum_association_distance"]),
            maximum_track_gap_seconds=float(configuration["maximum_track_gap_seconds"]),
            minimum_outside_observations=int(configuration["minimum_outside_observations"]),
            boundary_margin_normalized=float(configuration["boundary_margin_normalized"]),
            duplicate_footpoint_distance=float(configuration["duplicate_footpoint_distance"]),
            refractory_seconds=float(configuration["refractory_seconds"]),
        )
        for sample in source.samples:
            detections = [
                Detection(tuple(item["bbox"]), item["score"])
                for item in sample["person_detections"]
            ]
            started = time.perf_counter()
            outcomes = watch.observe(detections, sample["source_seconds"])
            tracker_latency = time.perf_counter() - started
            tracker_latencies.append(tracker_latency)
            observed_samples += 1
            fire_outcomes = [outcome for outcome in outcomes if outcome.action == "fire"]
            ask_outcomes = [outcome for outcome in outcomes if outcome.action == "ask"]
            logical_outcomes = []
            if fire_outcomes:
                if gate_armed is True:
                    logical_outcomes = [fire_outcomes[0]]
                else:
                    suppressed_rising_edge += 1
                suppressed_rising_edge += max(0, len(fire_outcomes) - 1)
                gate_armed = False
            elif ask_outcomes:
                logical_outcomes = [ask_outcomes[0]]
            else:
                gate_armed = True
            for outcome in logical_outcomes:
                incidents.append(
                    Incident(
                        incident_id=(
                            f"{source.source_id}:{sample['source_seconds']}:"
                            f"{outcome.track_id}:{outcome.action}"
                        ),
                        action=outcome.action,
                        source_seconds=sample["source_seconds"],
                        stage_latency_seconds={"tracker_state": tracker_latency},
                        source_id=source.source_id,
                    )
                )
        suppressed_refractory += watch.suppressed_refractory
        suppressed_duplicate_detections += watch.suppressed_duplicate_detections

    result = score(
        occurrences,
        incidents,
        match_window_seconds=2.0,
        timely_deadline_seconds=2.0,
        source_hours=source_hours,
    )
    result["timely_recall"] = None
    result["unacknowledged"] = None
    matched_incidents = {row["incident_id"] for row in result["matches"]}
    matched_occurrences = {row["occurrence_id"] for row in result["matches"]}
    by_source = []
    for source in data.sources:
        source_incidents = [incident for incident in incidents if incident.source_id == source.source_id]
        source_occurrences = source.occurrences
        fires = [incident for incident in source_incidents if incident.action == "fire"]
        asks = [incident for incident in source_incidents if incident.action == "ask"]
        tp = sum(incident.incident_id in matched_incidents for incident in fires)
        fn = sum(
            occurrence.occurrence_id not in matched_occurrences for occurrence in source_occurrences
        )
        by_source.append(
            {
                "source_id": source.source_id,
                "occurrences": len(source_occurrences),
                "fires": len(fires),
                "tp": tp,
                "fp": len(fires) - tp,
                "fn": fn,
                "ask": len(asks),
            }
        )
    result.update(
        {
            "schema": "percept.benchmark.b4-v2-score.v1",
            "configuration": configuration,
            "occurrences": len(occurrences),
            "fires": sum(incident.action == "fire" for incident in incidents),
            "quality_metrics_provisional": True,
            "receiver_ack_metrics_measured": False,
            "onset_to_receiver_ack_seconds": None,
            "all_detector_inference_seconds": latency_summary(data.detector_inference_seconds),
            "all_tracker_stage_seconds": latency_summary(tracker_latencies),
            "suppressed_refractory": suppressed_refractory,
            "suppressed_duplicate_detections": suppressed_duplicate_detections,
            "suppressed_rising_edge": suppressed_rising_edge,
            "source_distribution": by_source,
            "reconciliation": {
                "expected_sources": len(data.sources),
                "observed_sources": len(by_source),
                "expected_samples": sum(len(source.samples) for source in data.sources),
                "observed_samples": observed_samples,
                "exact": len(data.sources) == len(by_source)
                and sum(len(source.samples) for source in data.sources) == observed_samples,
            },
            "source_hours": source_hours,
            "validation_opened": False,
            "qualification_labels_opened": False,
            "pilot_276_opened": False,
            "remote_provider_calls": 0,
            "paid_activity_performed": False,
        }
    )
    return result
