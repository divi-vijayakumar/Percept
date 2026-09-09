#!/usr/bin/env python3
"""Evaluate the frozen B4 calibration search space."""

from __future__ import annotations

import argparse
import itertools
import json
import pathlib

from evaluate_cache import evaluate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=pathlib.Path, required=True)
    parser.add_argument("--annotations-zip", type=pathlib.Path, required=True)
    parser.add_argument("--zone-manifest", type=pathlib.Path, required=True)
    parser.add_argument("--search-space", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    arguments = parser.parse_args()
    search = json.loads(arguments.search_space.read_text())
    if search["partition"] != "calibration_exposed":
        raise ValueError("B4 sweep is calibration-only")
    tunables = search["tunables"]
    results = []
    for threshold, distance, gap, dwell in itertools.product(
        tunables["score_threshold"],
        tunables["maximum_association_distance"],
        tunables["maximum_track_gap_seconds"],
        tunables["dwell_seconds"],
    ):
        result = evaluate(
            cache_dir=arguments.cache_dir,
            annotations_zip=arguments.annotations_zip,
            zone_manifest=arguments.zone_manifest,
            score_threshold=threshold,
            association_distance=distance,
            track_gap_seconds=gap,
            dwell_seconds=dwell,
        )
        precision = result["precision"]
        recall = result["recall"]
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision is not None and recall is not None and precision + recall
            else None
        )
        results.append({**result, "f1": f1})
    ranked = sorted(
        results,
        key=lambda row: (
            -int(
                (row["precision"] or 0) >= search["quality_gate"]["precision_minimum"]
                and (row["recall"] or 0) >= search["quality_gate"]["recall_minimum"]
            ),
            -(row["f1"] or 0),
            -(row["precision"] or 0),
            -(row["recall"] or 0),
            row["configuration"]["score_threshold"],
            row["configuration"]["dwell_seconds"],
        ),
    )
    payload = {
        "schema": "percept.benchmark.b4-calibration-sweep.v1",
        "search_space": search,
        "configurations_evaluated": len(results),
        "configurations_meeting_quality_bar": sum(
            (row["precision"] or 0) >= search["quality_gate"]["precision_minimum"]
            and (row["recall"] or 0) >= search["quality_gate"]["recall_minimum"]
            for row in results
        ),
        "best": ranked[0],
        "ranked_results": ranked,
        "qualification_labels_opened": False,
        "remote_provider_calls": 0,
        "paid_activity_performed": False,
    }
    arguments.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
