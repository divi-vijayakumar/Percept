#!/usr/bin/env python3
"""Reproduce the B4 RTMO-S calibration feasibility gate on exposed VIRAT data."""

from __future__ import annotations

import argparse
import hashlib
from importlib import metadata
import json
import pathlib
import sys
import time
import zipfile


B2 = pathlib.Path(__file__).resolve().parents[1] / "b2"
sys.path.insert(0, str(B2))

from zone_entry import parse_virat_object_rows, transform_zone_entries  # noqa: E402


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iou(left: list[float], right: list[float]) -> float:
    x1 = max(left[0], right[0])
    y1 = max(left[1], right[1])
    x2 = min(left[2], right[2])
    y2 = min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union else 0.0


def _predictions(inferencer, image, threshold: float) -> list[dict[str, object]]:
    result = next(
        inferencer(image, return_vis=False, bbox_thr=threshold), None
    ) or {}
    predictions = result.get("predictions", [])
    if predictions and isinstance(predictions[0], list):
        predictions = predictions[0]
    normalized = []
    for prediction in predictions:
        raw_box = prediction.get("bbox")
        if (
            isinstance(raw_box, (list, tuple))
            and len(raw_box) == 1
            and isinstance(raw_box[0], (list, tuple))
        ):
            raw_box = raw_box[0]
        normalized.append(
            {
                "bbox": [float(value) for value in raw_box],
                "score": float(prediction.get("bbox_score", prediction.get("score", 0))),
            }
        )
    return normalized


def probe(
    *,
    media: pathlib.Path,
    annotations_zip: pathlib.Path,
    zone_manifest: pathlib.Path,
    model_config: pathlib.Path,
    model_weights: pathlib.Path,
) -> dict[str, object]:
    if "calibration-exposed" not in str(media.parent):
        raise ValueError("B4 feasibility probe accepts calibration-exposed media only")
    import cv2
    import numpy as np
    import torch
    from mmpose.apis import MMPoseInferencer

    zones = json.loads(zone_manifest.read_text())
    group = media.name.split("_")[2][:4]
    if group not in zones["partitions"]["calibration_exposed"]:
        raise ValueError("source group is not frozen as calibration-exposed")
    member_name = f"{media.stem}.viratdata.objects.txt"
    with zipfile.ZipFile(annotations_zip) as archive:
        members = {
            pathlib.PurePosixPath(info.filename).name: info
            for info in archive.infolist()
            if not info.is_dir()
        }
        rows = parse_virat_object_rows(
            archive.read(members[member_name]).decode("utf-8").splitlines()
        )

    capture = cv2.VideoCapture(str(media))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    occurrences = transform_zone_entries(
        rows,
        source_id=media.stem,
        fps=fps,
        frame_width=width,
        frame_height=height,
        polygon=zones["zones"][group]["polygon"],
        dwell_seconds=0.0,
    )
    if not occurrences:
        raise RuntimeError("calibration source has no provisional occurrence")
    selected = min(occurrences, key=lambda occurrence: occurrence.onset_frame)
    target = next(
        row
        for row in rows
        if row.track_id == selected.track_id and row.frame == selected.onset_frame
    )
    target_box = [target.x, target.y, target.x + target.width, target.y + target.height]
    calibration_frames = (0, selected.onset_frame)
    images = {}
    for frame in calibration_frames:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ok, image = capture.read()
        if not ok:
            raise RuntimeError(f"could not decode calibration frame {frame}")
        images[frame] = image
    capture.release()
    frame_zero_person_rows = sum(
        row.frame == 0 and row.class_id == 1 for row in rows
    )

    allowed = [
        (np.core.multiarray._reconstruct, "numpy.core.multiarray._reconstruct"),
        np.ndarray,
        np.dtype,
    ]
    allowed.extend(
        value for value in vars(np.dtypes).values() if isinstance(value, type)
    )
    started = time.monotonic()
    with torch.serialization.safe_globals(allowed):
        inferencer = MMPoseInferencer(
            pose2d=str(model_config),
            pose2d_weights=str(model_weights),
            device="cpu",
            show_progress=False,
        )
    model_init_seconds = time.monotonic() - started
    thresholds: dict[str, object] = {}
    for threshold in (0.3, 0.1, 0.03):
        rows_by_frame = {}
        for frame in calibration_frames:
            predictions = _predictions(inferencer, images[frame], threshold)
            overlaps = [
                iou(prediction["bbox"], target_box)  # type: ignore[arg-type]
                for prediction in predictions
            ] if frame == selected.onset_frame else []
            rows_by_frame[str(frame)] = {
                "detections": len(predictions),
                "max_score": max(
                    (float(prediction["score"]) for prediction in predictions),
                    default=None,
                ),
                "maximum_iou_to_entering_person": max(overlaps, default=None),
                "detections_iou_at_least_0_5": sum(value >= 0.5 for value in overlaps),
            }
        thresholds[str(threshold)] = rows_by_frame

    return {
        "schema": "percept.benchmark.b4-rtmo-s-feasibility.v1",
        "partition": "calibration_exposed",
        "source_group": group,
        "source": {
            "name": media.name,
            "sha256": sha256(media),
            "width": width,
            "height": height,
            "fps": fps,
        },
        "provisional_gt_probe": {
            "entry_frame": selected.onset_frame,
            "entry_seconds": selected.onset_seconds,
            "track_id": selected.track_id,
            "bbox_xyxy": target_box,
            "frame_zero_person_rows": frame_zero_person_rows,
            "complete_zone_audit_pending": True,
        },
        "model": {
            "name": "rtmo-s_8xb32-600e_coco-640x640",
            "weights_bytes": model_weights.stat().st_size,
            "weights_sha256": sha256(model_weights),
            "config_sha256": sha256(model_config),
            "device": "cpu",
            "initialization_seconds": model_init_seconds,
            "runtime_versions": {
                name: metadata.version(name)
                for name in ("mmcv", "mmengine", "mmpose", "numpy", "torch")
            },
        },
        "threshold_results": thresholds,
        "candidate_disposition": "reject" if not any(
            result["detections_iou_at_least_0_5"] > 0
            for by_frame in thresholds.values()
            for frame, result in by_frame.items()
            if frame == str(selected.onset_frame)
        ) else "retain",
        "disposition_scope": "RTMO-S candidate only; this is not the overall B4 gate",
        "qualification_labels_opened": False,
        "remote_provider_calls": 0,
        "paid_activity_performed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--media", type=pathlib.Path, required=True)
    parser.add_argument("--annotations-zip", type=pathlib.Path, required=True)
    parser.add_argument("--zone-manifest", type=pathlib.Path, required=True)
    parser.add_argument("--model-config", type=pathlib.Path, required=True)
    parser.add_argument("--model-weights", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    arguments = parser.parse_args()
    result = probe(
        media=arguments.media,
        annotations_zip=arguments.annotations_zip,
        zone_manifest=arguments.zone_manifest,
        model_config=arguments.model_config,
        model_weights=arguments.model_weights,
    )
    arguments.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
