"""Reconstruct the exact unmatched fires in zero-provisional-GT calibration sources."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT / "b2"), str(ROOT / "b4_v2")]

from evaluate_v2 import load_calibration  # noqa: E402
from persistent_zone_entry_watch import Detection, PersistentZoneEntryWatch  # noqa: E402
from scorer import Incident, score  # noqa: E402

DEFAULT_EVIDENCE_ROOT = pathlib.Path(
    "/Volumes/Percept/Percept-Eternal/Percept-Benchmarks/virat-fixed-watch"
)


def reconstruct(evidence_root: pathlib.Path) -> dict:
    result = json.loads((ROOT / "b4_v2/b4-v2-calibration-result.json").read_text())
    config = result["best"]["configuration"]
    data = load_calibration(
        evidence_root / "evidence/b4/calibration-fasterrcnn-320-1hz",
        evidence_root / "annotations-and-docs.zip",
        ROOT / "admission/f2-z1-zones-and-partitions.v1.json",
    )
    incidents: list[Incident] = []
    details: dict[str, dict] = {}
    for source in data.sources:
        watch = PersistentZoneEntryWatch(
            polygon=source.polygon,
            frame_width=source.width,
            frame_height=source.height,
            score_threshold=config["score_threshold"],
            maximum_association_distance=config["maximum_association_distance"],
            maximum_track_gap_seconds=config["maximum_track_gap_seconds"],
            minimum_outside_observations=config["minimum_outside_observations"],
            boundary_margin_normalized=config["boundary_margin_normalized"],
            duplicate_footpoint_distance=config["duplicate_footpoint_distance"],
            refractory_seconds=config["refractory_seconds"],
        )
        gate_armed = None
        for sample in source.samples:
            detections = [
                Detection(tuple(item["bbox"]), item["score"])
                for item in sample["person_detections"]
            ]
            outcomes = watch.observe(detections, sample["source_seconds"])
            fires = [item for item in outcomes if item.action == "fire"]
            asks = [item for item in outcomes if item.action == "ask"]
            logical = []
            if fires:
                if gate_armed is True:
                    logical = [fires[0]]
                gate_armed = False
            elif asks:
                logical = [asks[0]]
            else:
                gate_armed = True
            for outcome in logical:
                identity = (
                    f"{source.source_id}:{sample['source_seconds']}:"
                    f"{outcome.track_id}:{outcome.action}"
                )
                incidents.append(
                    Incident(identity, outcome.action, sample["source_seconds"], source_id=source.source_id)
                )
                details[identity] = {
                    "case_id": "",
                    "incident_id": identity,
                    "source_id": source.source_id,
                    "source_seconds": sample["source_seconds"],
                    "track_id": outcome.track_id,
                    "reason": outcome.reason,
                    "footpoint_normalized": list(outcome.footpoint),
                    "polygon_normalized": [list(point) for point in source.polygon],
                    "media_path": str(
                        evidence_root
                        / "media/calibration-exposed-0502"
                        / f"{source.source_id}.mp4"
                    ),
                }

    occurrences = [item for source in data.sources for item in source.occurrences]
    scored = score(
        occurrences,
        incidents,
        match_window_seconds=2,
        timely_deadline_seconds=2,
        source_hours=result["source_hours"],
    )
    matched = {item["incident_id"] for item in scored["matches"]}
    zero_gt_sources = {source.source_id for source in data.sources if not source.occurrences}
    review = [
        details[item.incident_id]
        for item in incidents
        if item.action == "fire"
        and item.incident_id not in matched
        and item.source_id in zero_gt_sources
    ]
    for index, item in enumerate(review, start=1):
        item["case_id"] = f"B4V2-ZGT-{index:02d}"
    if len(review) != 15:
        raise RuntimeError(f"expected exactly 15 review cases, reconstructed {len(review)}")
    return {
        "schema": "percept.benchmark.local-vlm-review-cases.v1",
        "evidentiary_status": "calibration-only model-assisted triage; not ground truth",
        "candidate_family": result["candidate_family"],
        "configuration": config,
        "case_count": len(review),
        "cases": review,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence-root", type=pathlib.Path, default=DEFAULT_EVIDENCE_ROOT)
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args()
    payload = reconstruct(args.evidence_root)
    encoded = json.dumps(payload, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded)
    else:
        print(encoded, end="")


if __name__ == "__main__":
    main()

