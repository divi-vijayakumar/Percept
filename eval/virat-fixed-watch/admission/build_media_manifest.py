#!/usr/bin/env python3
"""Build a deterministic, non-redistributive manifest for admitted VIRAT media."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import re
import subprocess
import tempfile
import zipfile


LISTING_ROW = re.compile(
    r"^\s*(?P<size>\d+)\s+Stored\s+\d+\s+\d+%\s+\S+\s+\S+\s+\S+\s+"
    r"videos_original/(?P<name>VIRAT_S_\S+\.mp4)$"
)
SOURCE_GROUP = re.compile(r"^VIRAT_S_(\d{4})")
PARTITIONS = {
    "validation-0101": "validation",
    "qualification-0000": "qualification",
    "qualification-0001": "qualification",
    "qualification-0100": "qualification",
    "qualification-0500": "qualification",
    "qualification-0503": "qualification",
}


def parse_official_listings(paths: list[pathlib.Path]) -> dict[str, int]:
    sizes: dict[str, int] = {}
    for path in paths:
        for line in path.read_text().splitlines():
            match = LISTING_ROW.match(line)
            if match:
                sizes[match.group("name")] = int(match.group("size"))
    return sizes


def annotation_inventory(path: pathlib.Path) -> dict[str, dict[str, object]]:
    suffix = ".viratdata.objects.txt"
    with zipfile.ZipFile(path) as archive:
        inventory: dict[str, dict[str, object]] = {}
        for info in archive.infolist():
            if info.is_dir() or not info.filename.endswith(suffix):
                continue
            stem = pathlib.PurePosixPath(info.filename).name.removesuffix(suffix)
            frames: list[int] = []
            person_tracks: set[int] = set()
            person_rows = 0
            malformed_rows = 0
            for line in archive.read(info).decode().splitlines():
                fields = line.split()
                if len(fields) != 8:
                    malformed_rows += 1
                    continue
                frames.append(int(fields[2]))
                if fields[7] == "1":
                    person_rows += 1
                    person_tracks.add(int(fields[0]))
            inventory[stem] = {
                "rows": len(frames),
                "minimum_frame": min(frames) if frames else None,
                "maximum_frame": max(frames) if frames else None,
                "person_rows": person_rows,
                "person_tracks": len(person_tracks),
                "malformed_rows": malformed_rows,
            }
        return inventory


def annotation_stems(path: pathlib.Path) -> set[str]:
    return set(annotation_inventory(path))


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def probe(path: pathlib.Path) -> dict[str, object]:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,width,height,r_frame_rate,avg_frame_rate,nb_frames,duration",
            "-show_entries",
            "format=duration,size",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    stream = payload["streams"][0]
    container = payload["format"]
    return {
        "codec": stream["codec_name"],
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "r_frame_rate": stream["r_frame_rate"],
        "avg_frame_rate": stream["avg_frame_rate"],
        "frames": int(stream["nb_frames"]),
        "duration_seconds": float(container["duration"]),
    }


def build(
    media_root: pathlib.Path,
    annotations_zip: pathlib.Path,
    listings: list[pathlib.Path],
    recorded_at: str,
) -> dict[str, object]:
    official_sizes = parse_official_listings(listings)
    labels = annotation_inventory(annotations_zip)
    rows: list[dict[str, object]] = []
    for directory, partition in PARTITIONS.items():
        source_dir = media_root / directory
        for path in sorted(source_dir.glob("*.mp4")):
            if path.name.startswith("._"):
                continue
            match = SOURCE_GROUP.match(path.name)
            if not match:
                raise ValueError(f"unrecognized VIRAT source name: {path.name}")
            if path.stem not in labels:
                raise ValueError(f"missing official object annotation: {path.name}")
            if path.name not in official_sizes:
                raise ValueError(f"missing official package listing: {path.name}")
            actual_size = path.stat().st_size
            if actual_size != official_sizes[path.name]:
                raise ValueError(
                    f"official size mismatch for {path.name}: "
                    f"{actual_size} != {official_sizes[path.name]}"
                )
            video = probe(path)
            annotation = labels[path.stem]
            if annotation["malformed_rows"]:
                raise ValueError(f"malformed official object annotation: {path.name}")
            maximum_frame = annotation["maximum_frame"]
            if maximum_frame is not None and int(maximum_frame) > int(video["frames"]):
                raise ValueError(
                    f"annotation frame outside decoded media for {path.name}: "
                    f"{maximum_frame} > {video['frames']}"
                )
            rows.append(
                {
                    "partition": partition,
                    "source_group": match.group(1),
                    "name": path.name,
                    "relative_path": f"{directory}/{path.name}",
                    "bytes": actual_size,
                    "sha256": sha256(path),
                    "video": video,
                    "annotation": annotation,
                    "annotation_frames_within_media": maximum_frame is None
                    or int(maximum_frame) < int(video["frames"]),
                    "annotation_max_equals_decoded_frame_count": maximum_frame is not None
                    and int(maximum_frame) == int(video["frames"]),
                    "official_object_annotation_present": True,
                    "official_package_size_match": True,
                }
            )
    by_partition = {
        partition: {
            "files": sum(row["partition"] == partition for row in rows),
            "bytes": sum(
                int(row["bytes"]) for row in rows if row["partition"] == partition
            ),
            "source_groups": sorted(
                {row["source_group"] for row in rows if row["partition"] == partition}
            ),
        }
        for partition in ("validation", "qualification")
    }
    return {
        "schema": "percept.benchmark.virat-media-manifest.v1",
        "recorded_at": recorded_at,
        "media_root": str(media_root),
        "protected_media_redistributed": False,
        "summary": {
            "files": len(rows),
            "bytes": sum(int(row["bytes"]) for row in rows),
            "files_with_annotation_max_equal_to_decoded_frame_count": sum(
                bool(row["annotation_max_equals_decoded_frame_count"]) for row in rows
            ),
            "partitions": by_partition,
        },
        "media": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--media-root", type=pathlib.Path, required=True)
    parser.add_argument("--annotations-zip", type=pathlib.Path, required=True)
    parser.add_argument("--listing", type=pathlib.Path, action="append", required=True)
    parser.add_argument("--recorded-at", required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    payload = build(
        args.media_root, args.annotations_zip, args.listing, args.recorded_at
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=args.output.parent, delete=False
    ) as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        temporary = pathlib.Path(stream.name)
    temporary.replace(args.output)


if __name__ == "__main__":
    main()
