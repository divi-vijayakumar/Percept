"""Validation helpers for public benchmark release artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class ValidationError(ValueError):
    """Raised when a benchmark artifact is malformed or not release-ready."""


def _load(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def validate_baselines(path: str | Path) -> dict[str, Any]:
    data = _load(path)
    ids: set[str] = set()
    for row in data.get("baselines", []):
        for key in ("id", "system", "kind", "controlled_ab", "source", "metric", "comparability"):
            if key not in row:
                raise ValidationError(f"baseline {row.get('id', '<unknown>')} missing {key}")
        if row["id"] in ids:
            raise ValidationError(f"duplicate baseline id: {row['id']}")
        ids.add(row["id"])
        if row["kind"] not in {"reference_bar", "controlled_ab", "dataset"}:
            raise ValidationError(f"baseline {row['id']} has unknown kind {row['kind']!r}")
        if row["kind"] == "controlled_ab" and row["controlled_ab"] is not True:
            raise ValidationError(f"controlled baseline {row['id']} must set controlled_ab=true")
    if not ids:
        raise ValidationError("baseline registry is empty")
    return data


def validate_golden_manifest(path: str | Path, *, release: bool = False) -> dict[str, Any]:
    data = _load(path)
    for key in ("version", "status", "tracks", "splits"):
        if key not in data:
            raise ValidationError(f"golden manifest missing {key}")
    track_ids: set[str] = set()
    for track in data["tracks"]:
        for key in ("id", "source", "status", "metrics"):
            if key not in track:
                raise ValidationError(f"track {track.get('id', '<unknown>')} missing {key}")
        if track["id"] in track_ids:
            raise ValidationError(f"duplicate track id: {track['id']}")
        track_ids.add(track["id"])
        if not track["metrics"]:
            raise ValidationError(f"track {track['id']} has no metrics")

    splits = data["splits"]
    if not isinstance(splits, dict):
        raise ValidationError("splits must be an object")
    for split in ("dev", "test"):
        if split not in splits:
            raise ValidationError(f"splits missing {split}")

    if release:
        if data.get("status") != "release":
            raise ValidationError("release validation requires status='release'")
        if not splits.get("test"):
            raise ValidationError("release manifest needs non-empty test split")
        for item in splits["test"]:
            for key in ("id", "source", "license", "media_sha256", "track"):
                if key not in item:
                    raise ValidationError(f"release test item missing {key}")
            if item["track"] not in track_ids:
                raise ValidationError(f"test item {item['id']} references unknown track {item['track']}")
    return data


__all__ = ["ValidationError", "validate_baselines", "validate_golden_manifest"]
