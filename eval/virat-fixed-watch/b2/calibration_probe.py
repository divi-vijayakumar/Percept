#!/usr/bin/env python3
"""Run the GT transformer on an explicitly exposed calibration directory."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import subprocess
import zipfile

from zone_entry import parse_virat_object_rows, transform_zone_entries


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def probe_video(path: pathlib.Path) -> tuple[int, int, float]:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,avg_frame_rate",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    stream = json.loads(result.stdout)["streams"][0]
    numerator, denominator = map(int, stream["avg_frame_rate"].split("/"))
    return int(stream["width"]), int(stream["height"]), numerator / denominator


def run(
    media_dir: pathlib.Path,
    annotations_zip: pathlib.Path,
    zone_manifest: pathlib.Path,
    source_group: str,
) -> dict[str, object]:
    if "calibration" not in media_dir.name or "exposed" not in media_dir.name:
        raise ValueError("probe accepts only a directory explicitly named calibration-exposed")
    zones = json.loads(zone_manifest.read_text())
    if source_group not in zones["partitions"]["calibration_exposed"]:
        raise ValueError("source group is not registered calibration-exposed")
    polygon = zones["zones"][source_group]["polygon"]
    dwell_values = (0.0, 0.25, 0.5)
    counts = {str(dwell): 0 for dwell in dwell_values}
    admitted_files = 0
    malformed_files = 0
    with zipfile.ZipFile(annotations_zip) as archive:
        members = {
            pathlib.PurePosixPath(info.filename).name: info
            for info in archive.infolist()
            if not info.is_dir()
        }
        for media in sorted(media_dir.glob("*.mp4")):
            if media.name.startswith("._"):
                continue
            annotation_name = f"{media.stem}.viratdata.objects.txt"
            info = members.get(annotation_name)
            if info is None:
                continue
            try:
                rows = parse_virat_object_rows(
                    archive.read(info).decode("utf-8").splitlines()
                )
            except ValueError:
                malformed_files += 1
                continue
            width, height, fps = probe_video(media)
            admitted_files += 1
            for dwell in dwell_values:
                counts[str(dwell)] += len(
                    transform_zone_entries(
                        rows,
                        source_id=media.stem,
                        fps=fps,
                        frame_width=width,
                        frame_height=height,
                        polygon=polygon,
                        dwell_seconds=dwell,
                    )
                )
    return {
        "schema": "percept.benchmark.b2-calibration-probe.v1",
        "source_group": source_group,
        "partition": "calibration_exposed",
        "media_files_admitted": admitted_files,
        "malformed_annotation_files_rejected": malformed_files,
        "provisional_occurrences_by_dwell_seconds": counts,
        "legacy_occlusion_field_present": False,
        "occlusion_interpretation": "not_provided; provisional engineering only",
        "qualification_labels_opened": False,
        "annotations_zip_sha256": sha256(annotations_zip),
        "zone_manifest_sha256": sha256(zone_manifest),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--media-dir", type=pathlib.Path, required=True)
    parser.add_argument("--annotations-zip", type=pathlib.Path, required=True)
    parser.add_argument("--zone-manifest", type=pathlib.Path, required=True)
    parser.add_argument("--source-group", required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    arguments = parser.parse_args()
    payload = run(
        arguments.media_dir,
        arguments.annotations_zip,
        arguments.zone_manifest,
        arguments.source_group,
    )
    arguments.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
