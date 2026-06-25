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


def audit_artifacts(
    *,
    manifest: str | Path,
    baselines: str | Path,
    release: bool = False,
) -> dict[str, Any]:
    """Validate benchmark release artifacts and return a small machine-readable audit.

    ``release=False`` is the OSS/default path: the artifacts must be well-formed, but a draft
    manifest is allowed. ``release=True`` is intentionally stricter and fails until the golden set has
    non-empty, license-cleared, content-hashed test rows.
    """
    golden = validate_golden_manifest(manifest, release=release)
    baseline_data = validate_baselines(baselines)
    status = "release-ready" if release else "draft-ok"
    return {
        "status": status,
        "release": release,
        "manifest": {
            "version": golden["version"],
            "status": golden["status"],
            "tracks": len(golden["tracks"]),
            "dev_items": len(golden["splits"].get("dev", [])),
            "test_items": len(golden["splits"].get("test", [])),
        },
        "baselines": {
            "version": baseline_data.get("version"),
            "rows": len(baseline_data["baselines"]),
            "controlled_ab": sum(1 for row in baseline_data["baselines"] if row.get("controlled_ab") is True),
        },
    }


def format_audit(audit: dict[str, Any]) -> str:
    manifest = audit["manifest"]
    baselines = audit["baselines"]
    lines = [
        "",
        "=" * 72,
        "  PERCEPT BENCHMARK ARTIFACT AUDIT",
        "=" * 72,
        f"  status       : {audit['status']}",
        f"  release mode : {audit['release']}",
        f"  manifest     : {manifest['version']} ({manifest['status']})",
        f"  tracks       : {manifest['tracks']}",
        f"  split items  : dev={manifest['dev_items']} test={manifest['test_items']}",
        f"  baselines    : {baselines['rows']} rows ({baselines['controlled_ab']} controlled A/B)",
        "=" * 72,
        "",
    ]
    return "\n".join(lines)


__all__ = [
    "ValidationError",
    "audit_artifacts",
    "format_audit",
    "validate_baselines",
    "validate_golden_manifest",
]
