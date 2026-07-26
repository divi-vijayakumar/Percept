"""Validation helpers for public benchmark release artifacts."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

PT_ACTION_ACCEPTANCE_METRICS = frozenset(
    {
        "occurrence_recall",
        "notification_precision",
        "timely_occurrence_recall_at_2s",
        "physical_call_reduction",
        "max_action_onset_to_incident",
    }
)
ACCEPTANCE_OPERATORS = frozenset({">=", "<="})


class ValidationError(ValueError):
    """Raised when a benchmark artifact is malformed or not release-ready."""


def manifest_runner_map(
    data: dict[str, Any],
    *,
    require_all_tracks: bool = False,
) -> dict[str, dict[str, Any]]:
    """Build the explicit runner -> logical track/result identity mapping."""
    out: dict[str, dict[str, str]] = {}
    result_tracks: set[str] = set()
    for track in data.get("tracks", []):
        runners = track.get("runners")
        if runners is None:
            if require_all_tracks:
                raise ValidationError(f"track {track['id']} missing non-empty runners mapping")
            continue
        if not isinstance(runners, list) or not runners:
            raise ValidationError(f"track {track['id']} runners must be a non-empty list")
        for row in runners:
            if not isinstance(row, dict):
                raise ValidationError(f"track {track['id']} runner mapping must be an object")
            runner = row.get("id")
            result_track = row.get("result_track")
            if not isinstance(runner, str) or not runner:
                raise ValidationError(f"track {track['id']} runner mapping missing non-empty id")
            if not isinstance(result_track, str) or not result_track:
                raise ValidationError(
                    f"track {track['id']} runner {runner!r} missing non-empty result_track"
                )
            if runner in out:
                raise ValidationError(f"duplicate manifest runner id: {runner}")
            if result_track in result_tracks:
                raise ValidationError(f"duplicate manifest result_track: {result_track}")
            result_tracks.add(result_track)
            out[runner] = {
                "manifest_track": str(track["id"]),
                "result_track": result_track,
                "required": track.get("required") is True,
                "acceptance": track.get("acceptance"),
            }
    return out


def _validate_acceptance(track: dict[str, Any]) -> None:
    acceptance = track.get("acceptance")
    if not isinstance(acceptance, dict) or not acceptance:
        raise ValidationError(
            f"required track {track['id']} needs non-empty acceptance rules"
        )
    for metric, rule in acceptance.items():
        if not isinstance(metric, str) or not metric:
            raise ValidationError(
                f"required track {track['id']} has an invalid acceptance metric"
            )
        if not isinstance(rule, dict):
            raise ValidationError(
                f"required track {track['id']} acceptance {metric} must be an object"
            )
        operator = rule.get("operator")
        threshold = rule.get("threshold")
        basis = rule.get("basis")
        if operator not in ACCEPTANCE_OPERATORS:
            raise ValidationError(
                f"required track {track['id']} acceptance {metric} has "
                f"unsupported operator {operator!r}"
            )
        if (
            isinstance(threshold, bool)
            or not isinstance(threshold, (int, float))
            or not math.isfinite(float(threshold))
        ):
            raise ValidationError(
                f"required track {track['id']} acceptance {metric} needs a "
                "finite numeric threshold"
            )
        if not isinstance(basis, str) or not basis.strip():
            raise ValidationError(
                f"required track {track['id']} acceptance {metric} needs a "
                "non-empty evidence basis"
            )
    result_tracks = {
        str(row.get("result_track"))
        for row in track.get("runners", [])
        if isinstance(row, dict)
    }
    if "pt-action-e2e" in result_tracks:
        missing = sorted(PT_ACTION_ACCEPTANCE_METRICS - set(acceptance))
        if missing:
            raise ValidationError(
                "required pt-action-e2e track acceptance is missing: "
                + ", ".join(missing)
            )


def _load(path: str | Path) -> dict[str, Any]:
    artifact = Path(path)
    if not artifact.exists():
        raise ValidationError(f"required artifact does not exist: {artifact}")
    if artifact.stat().st_size == 0:
        raise ValidationError(f"required artifact is empty: {artifact}")
    try:
        data = json.loads(artifact.read_text())
    except json.JSONDecodeError as exc:
        raise ValidationError(f"invalid JSON in {artifact}: {exc}") from exc
    if not isinstance(data, dict) or not data:
        raise ValidationError(f"required artifact must be a non-empty JSON object: {artifact}")
    return data


def validate_baselines(path: str | Path) -> dict[str, Any]:
    data = _load(path)
    if not isinstance(data.get("baselines"), list):
        raise ValidationError("baseline registry needs a baselines list")
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
    if not isinstance(data["tracks"], list) or not data["tracks"]:
        raise ValidationError("golden manifest needs at least one benchmark track")
    track_ids: set[str] = set()
    for track in data["tracks"]:
        for key in ("id", "source", "status", "metrics"):
            if key not in track:
                raise ValidationError(f"track {track.get('id', '<unknown>')} missing {key}")
        if track["id"] in track_ids:
            raise ValidationError(f"duplicate track id: {track['id']}")
        track_ids.add(track["id"])
        if not isinstance(track["metrics"], list) or not track["metrics"]:
            raise ValidationError(f"track {track['id']} has no metrics")
    manifest_runner_map(data, require_all_tracks=release)

    splits = data["splits"]
    if not isinstance(splits, dict):
        raise ValidationError("splits must be an object")
    for split in ("dev", "test"):
        if split not in splits:
            raise ValidationError(f"splits missing {split}")
        if not isinstance(splits[split], list):
            raise ValidationError(f"splits.{split} must be a list")

    # The OSS artifact gate is allowed to audit a draft corpus, but not an empty placeholder.
    # At least one labelled scenario must connect a declared track to an auditable source.
    if not splits["test"]:
        raise ValidationError("benchmark manifest needs a non-empty test split")
    ids: set[str] = set()
    for item in splits["test"]:
        for key in ("id", "source", "track"):
            if not item.get(key):
                raise ValidationError(f"test item missing non-empty {key}")
        if item["id"] in ids:
            raise ValidationError(f"duplicate test item id: {item['id']}")
        ids.add(item["id"])
        if item["track"] not in track_ids:
            raise ValidationError(f"test item {item['id']} references unknown track {item['track']}")

    if release:
        if data.get("status") != "release":
            raise ValidationError("release validation requires status='release'")
        required_tracks = [
            track for track in data["tracks"] if track.get("required") is True
        ]
        if not required_tracks:
            raise ValidationError(
                "release validation requires at least one required track"
            )
        for track in required_tracks:
            _validate_acceptance(track)
        for item in splits["test"]:
            for key in ("license", "media_sha256"):
                if not item.get(key):
                    raise ValidationError(f"release test item {item['id']} missing non-empty {key}")
            digest = str(item["media_sha256"])
            if len(digest) != 64 or any(ch not in "0123456789abcdefABCDEF" for ch in digest):
                raise ValidationError(f"release test item {item['id']} has invalid media_sha256")
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
    status = "ARTIFACT-AUDIT PASS"
    cloud_status = "NOT-RUN"
    return {
        "status": status,
        "cloud_status": cloud_status,
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
        "  artifact audit : PASS (non-empty manifest + baseline registry)",
        f"  Cloud cost gate : {audit['cloud_status']} (artifact validation is not economics)",
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
    "manifest_runner_map",
    "validate_baselines",
    "validate_golden_manifest",
]
