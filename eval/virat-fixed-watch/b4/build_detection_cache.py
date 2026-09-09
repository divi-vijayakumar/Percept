#!/usr/bin/env python3
"""Build a zero-paid Faster R-CNN detection cache for an allowed B4 partition."""

from __future__ import annotations

import argparse
import hashlib
from importlib import metadata
import json
import pathlib
import time


ALLOWED_DIRECTORY_PREFIXES = ("calibration-exposed-", "validation-")


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_group(media: pathlib.Path) -> str:
    return media.name.split("_")[2][:4]


def build(
    *,
    media_dir: pathlib.Path,
    weights: pathlib.Path,
    zone_manifest: pathlib.Path,
    output_dir: pathlib.Path,
    sample_hz: float,
    minimum_size: int,
    threads: int,
) -> dict[str, object]:
    if not media_dir.name.startswith(ALLOWED_DIRECTORY_PREFIXES):
        raise ValueError("B4 cache accepts calibration-exposed or validation directories only")
    partition = (
        "calibration_exposed"
        if media_dir.name.startswith("calibration-exposed-")
        else "validation"
    )
    zones = json.loads(zone_manifest.read_text())
    media_files = [
        path
        for path in sorted(media_dir.glob("*.mp4"))
        if not path.name.startswith("._")
    ]
    for media in media_files:
        if source_group(media) not in zones["partitions"][partition]:
            raise ValueError(f"{media.name} is not frozen in partition {partition}")

    import cv2
    import torch
    from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2

    torch.set_num_threads(threads)
    state = torch.load(weights, map_location="cpu", weights_only=True)
    model = fasterrcnn_resnet50_fpn_v2(
        weights=None,
        weights_backbone=None,
        min_size=minimum_size,
        max_size=round(minimum_size * 1.6675),
        box_detections_per_img=100,
    )
    model.load_state_dict(state)
    model.eval()
    output_dir.mkdir(parents=True, exist_ok=True)
    source_summaries = []
    for media in media_files:
        capture = cv2.VideoCapture(str(media))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if fps <= 0 or frame_count <= 0:
            raise RuntimeError(f"invalid clock for {media}")
        duration = (frame_count - 1) / fps
        rows = []
        inference_values = []
        source_second = 0.0
        while source_second <= duration + 1e-9:
            capture.set(cv2.CAP_PROP_POS_MSEC, source_second * 1000)
            ok, bgr = capture.read()
            if not ok:
                break
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            image = torch.from_numpy(rgb.copy()).permute(2, 0, 1).float().div_(255)
            started = time.perf_counter()
            with torch.inference_mode():
                output = model([image])[0]
            inference = time.perf_counter() - started
            inference_values.append(inference)
            detections = [
                {
                    "bbox": [round(float(value), 4) for value in box],
                    "score": round(float(score), 8),
                }
                for label, score, box in zip(
                    output["labels"], output["scores"], output["boxes"]
                )
                if int(label) == 1
            ]
            rows.append(
                {
                    "source_seconds": round(source_second, 6),
                    "inference_seconds": inference,
                    "person_detections": detections,
                }
            )
            source_second += 1 / sample_hz
        capture.release()
        payload = {
            "schema": "percept.benchmark.b4-detection-cache.v1",
            "partition": partition,
            "source": {
                "name": media.name,
                "sha256": sha256(media),
                "fps": fps,
                "frame_count": frame_count,
                "width": width,
                "height": height,
            },
            "model": {
                "name": "torchvision/fasterrcnn_resnet50_fpn_v2",
                "weights_sha256": sha256(weights),
                "minimum_size": minimum_size,
                "sample_hz": sample_hz,
                "device": "cpu",
            },
            "samples": rows,
            "remote_provider_calls": 0,
            "paid_activity_performed": False,
        }
        destination = output_dir / f"{media.stem}.detections.json"
        temporary = destination.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, separators=(",", ":")) + "\n")
        temporary.replace(destination)
        source_summaries.append(
            {
                "name": media.name,
                "samples": len(rows),
                "source_seconds": duration,
                "inference_seconds_total": sum(inference_values),
            }
        )
        print(json.dumps(source_summaries[-1]), flush=True)
    return {
        "schema": "percept.benchmark.b4-detection-cache-manifest.v1",
        "partition": partition,
        "media_directory": str(media_dir),
        "weights_bytes": weights.stat().st_size,
        "weights_sha256": sha256(weights),
        "zone_manifest_sha256": sha256(zone_manifest),
        "configuration": {
            "sample_hz": sample_hz,
            "minimum_size": minimum_size,
            "threads": threads,
        },
        "runtime_versions": {
            name: metadata.version(name)
            for name in ("numpy", "opencv-python", "torch", "torchvision")
        },
        "sources": source_summaries,
        "qualification_labels_opened": False,
        "remote_provider_calls": 0,
        "paid_activity_performed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--media-dir", type=pathlib.Path, required=True)
    parser.add_argument("--weights", type=pathlib.Path, required=True)
    parser.add_argument("--zone-manifest", type=pathlib.Path, required=True)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--output-manifest", type=pathlib.Path, required=True)
    parser.add_argument("--sample-hz", type=float, default=1.0)
    parser.add_argument("--minimum-size", type=int, default=320)
    parser.add_argument("--threads", type=int, default=8)
    arguments = parser.parse_args()
    result = build(
        media_dir=arguments.media_dir,
        weights=arguments.weights,
        zone_manifest=arguments.zone_manifest,
        output_dir=arguments.output_dir,
        sample_hz=arguments.sample_hz,
        minimum_size=arguments.minimum_size,
        threads=arguments.threads,
    )
    arguments.output_manifest.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
