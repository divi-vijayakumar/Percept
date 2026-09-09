#!/usr/bin/env python3
"""Run the frozen persistent-tracking B4 v2 search."""

from __future__ import annotations

import argparse
import itertools
import json
import pathlib

from evaluate_v2 import evaluate, load_calibration


def passes(result, acceptance) -> bool:
    detector = result["all_detector_inference_seconds"]
    tracker = result["all_tracker_stage_seconds"]
    return (
        (result["precision"] or 0) >= acceptance["precision_minimum_provisional"]
        and (result["recall"] or 0) >= acceptance["recall_minimum_provisional"]
        and result["duplicates"] <= acceptance["duplicates_maximum"]
        and detector["p99"] <= acceptance["detector_inference_p99_seconds_maximum"]
        and detector["max"] <= acceptance["detector_inference_max_seconds_maximum"]
        and tracker["p99"] <= acceptance["tracker_stage_p99_seconds_maximum"]
        and tracker["max"] <= acceptance["tracker_stage_max_seconds_maximum"]
        and result["reconciliation"]["exact"] is acceptance["reconciliation_exact"]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=pathlib.Path, required=True)
    parser.add_argument("--annotations-zip", type=pathlib.Path, required=True)
    parser.add_argument("--zone-manifest", type=pathlib.Path, required=True)
    parser.add_argument("--search-space", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    arguments = parser.parse_args()
    search = json.loads(arguments.search_space.read_text())
    data = load_calibration(arguments.cache_dir, arguments.annotations_zip, arguments.zone_manifest)
    names = list(search["tunables"])
    results = []
    for values in itertools.product(*(search["tunables"][name] for name in names)):
        configuration = dict(zip(names, values))
        result = evaluate(data, configuration)
        precision, recall = result["precision"], result["recall"]
        result["f1"] = (
            2 * precision * recall / (precision + recall)
            if precision is not None and recall is not None and precision + recall
            else None
        )
        result["passes_calibration_gate"] = passes(result, search["calibration_acceptance"])
        results.append(result)
    ranked = sorted(
        results,
        key=lambda row: (
            -int(row["passes_calibration_gate"]),
            -(row["f1"] or 0),
            -(row["precision"] or 0),
            -(row["recall"] or 0),
            row["ask"],
            row["configuration"]["refractory_seconds"],
            row["configuration"]["score_threshold"],
        ),
    )
    payload = {
        "schema": "percept.benchmark.b4-v2-calibration-sweep.v1",
        "candidate_family": search["candidate_family"],
        "search_space": search,
        "configurations_evaluated": len(results),
        "configurations_passing": sum(row["passes_calibration_gate"] for row in results),
        "best": ranked[0],
        "ranked_results": ranked,
        "validation_opened": False,
        "qualification_labels_opened": False,
        "pilot_276_opened": False,
        "remote_provider_calls": 0,
        "paid_activity_performed": False,
    }
    arguments.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
