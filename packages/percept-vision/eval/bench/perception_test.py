"""Perception Test fetcher (CC-BY-4.0) — cheap annotations + range-fetched sampled videos.

The bulk video/audio zips are 70GB/33GB with NO per-video URLs, so we never download them whole: the
annotations are tiny (~550KB), and GCS supports HTTP range requests, so `remotezip` reads the zip's
central directory (~2s) and extracts only the individual ~2MB videos we sampled. Videos carry embedded
audio, so one fetch per event gives frames (cv2) + audio (PyAV).

Deterministic sampling (no RNG): one event per video, videos sorted by id — so a manifest hash pins the
exact sample. Raw data lands under eval/bench-staging/perception_test/ (gitignored).
"""

from __future__ import annotations

import hashlib
import json
import os
import zipfile
from functools import lru_cache
from pathlib import Path
from typing import Any

GCS = "https://storage.googleapis.com/dm-perception-test/zip_data"
ANN_ZIP = {
    "sound": "challenge_sound_localisation_valid_annotations.zip",
    "action": "challenge_action_localisation_valid_annotations.zip",
}
TASK_KEY = {"sound": "sound_localisation", "action": "action_localisation"}
REPO_ROOT = Path(__file__).resolve().parents[4]
VIDEOS_ZIP = f"{GCS}/valid_videos.zip"

_DEFAULT_STAGE_ROOT = Path(__file__).resolve().parent.parent / "bench-staging"
_STAGE_ROOT = Path(os.environ.get("PERCEPT_BENCH_DATA_DIR", _DEFAULT_STAGE_ROOT)).expanduser()
STAGE = (_STAGE_ROOT if _STAGE_ROOT.is_absolute() else REPO_ROOT / _STAGE_ROOT) / "perception_test" / "raw"
VIDEO_DIR = STAGE / "videos"
GOLDEN_MANIFEST = REPO_ROOT / "eval" / "golden-v1" / "MANIFEST.json"
SAMPLE_SEED = os.environ.get("PERCEPT_BENCH_SAMPLE_SEED", "percept-golden-v1")
_NON_GROUNDED_ACTION_LABELS = {"", "other", "unknown", "none", "n/a"}


class GoldenSetCapacityError(RuntimeError):
    """The pinned licensed corpus cannot satisfy the requested benchmark size."""


def _download(url: str, dest: Path) -> Path:
    if not dest.exists():
        import requests  # proper SSL via certifi (macOS framework urllib lacks the cert bundle)
        dest.parent.mkdir(parents=True, exist_ok=True)
        r = requests.get(url, timeout=180)
        r.raise_for_status()
        dest.write_bytes(r.content)
    return dest


@lru_cache(maxsize=2)
def load_annotations(task: str = "sound") -> dict:
    """The full valid-split annotations for a task (downloaded + cached; ~550KB, bzip2 inside the zip)."""
    z = _download(f"{GCS}/{ANN_ZIP[task]}", STAGE / ANN_ZIP[task])
    with zipfile.ZipFile(z) as zf:
        return json.loads(zf.read(zf.namelist()[0]))


def sample_events(task: str, n: int, *, is_visible: int | None = None) -> list[dict]:
    """Deterministic: the FIRST matching segment per video, for the first ``n`` videos (sorted by id).

    Returns dicts: {video_id, label, onset_s, end_s, fps, is_visible}. ``is_visible`` filters sound events
    to (un)seen sources — is_visible=0 is the cleanest fusion test (the source isn't visible)."""
    data = load_annotations(task)
    key = TASK_KEY[task]
    out: list[dict] = []
    for vid in sorted(data):
        v = data[vid]
        for s in v.get(key, []):
            if is_visible is not None and s.get("is_visible") != is_visible:
                continue
            ts = s["timestamps"]
            out.append({
                "video_id": v["metadata"]["video_id"], "label": s["label"],
                "onset_s": ts[0] / 1e6, "end_s": ts[1] / 1e6,
                "fps": v["metadata"]["frame_rate"], "is_visible": s.get("is_visible"),
            })
            break  # one event per video -> diverse sample
    out.sort(key=lambda e: hashlib.sha256(f"{SAMPLE_SEED}|pt|{task}|{e['video_id']}".encode("utf-8")).hexdigest())
    return out[:n]


@lru_cache(maxsize=1)
def _remote():
    from remotezip import RemoteZip
    return RemoteZip(VIDEOS_ZIP)


def fetch_video(video_id: str) -> Path:
    """Range-fetch one ~2MB video from the 70GB GCS zip (cached to disk)."""
    dest = VIDEO_DIR / f"{video_id}.mp4"
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(_remote().read(f"videos/{video_id}.mp4"))
    return dest


def manifest_hash(events: list[dict]) -> str:
    """Pin the exact sample used by the benchmark card."""
    key = json.dumps(sorted((e["video_id"], round(e["onset_s"], 3)) for e in events))
    return "pt-" + hashlib.sha256(key.encode()).hexdigest()[:12]


def label_phrase(label: str) -> str:
    """Turn a PT label into a plain-language sound condition. 'Interaction:hitting' -> 'audible hitting
    sound'; 'Animal:sound' -> 'audible animal sound' (the ':sound' detail falls back to the category)."""
    cat, _, detail = label.partition(":")
    subject = detail.strip().lower()
    if subject in ("", "sound"):
        subject = cat.strip().lower()
    return f"right now there is an audible {subject} sound or event"


def repo_path(path: str | Path) -> Path:
    """Resolve a manifest-relative path from the repository root."""
    p = Path(path)
    if p.is_absolute():
        return p
    if p.parts[:2] == ("eval", "bench-staging"):
        configured = os.environ.get("PERCEPT_BENCH_DATA_DIR")
        if configured:
            root = Path(configured).expanduser()
            if not root.is_absolute():
                root = REPO_ROOT / root
            return root.joinpath(*p.parts[2:])
    return REPO_ROOT / p


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


@lru_cache(maxsize=4)
def load_golden_manifest(path: str | Path = GOLDEN_MANIFEST) -> dict[str, Any]:
    """Load the public golden manifest that pins PT sample rows by content hash."""
    return json.loads(Path(path).read_text())


def golden_rows(
    track: str | None = None,
    *,
    manifest: str | Path = GOLDEN_MANIFEST,
) -> list[dict[str, Any]]:
    rows = list(load_golden_manifest(manifest).get("splits", {}).get("test", []))
    if track is not None:
        rows = [r for r in rows if r.get("track") == track]
    return rows


def verify_row(row: dict[str, Any], *, require_media: bool = True) -> bool:
    """Return true when a manifest row's local files exist and match their pinned hashes."""
    ann = repo_path(row["annotation_path"])
    if not ann.exists() or sha256_file(ann) != row["annotation_sha256"]:
        return False
    media = repo_path(row["media_path"])
    if not require_media:
        return (not media.exists()) or sha256_file(media) == row["media_sha256"]
    return media.exists() and sha256_file(media) == row["media_sha256"]


def verified_rows(
    track: str | None = None,
    *,
    manifest: str | Path = GOLDEN_MANIFEST,
    require_media: bool = True,
) -> list[dict[str, Any]]:
    return [r for r in golden_rows(track, manifest=manifest) if verify_row(r, require_media=require_media)]


@lru_cache(maxsize=4)
def load_sample_annotations(path: str | Path) -> dict[str, Any]:
    annotation_path = repo_path(path)
    if annotation_path.suffix == ".zip":
        with zipfile.ZipFile(annotation_path) as archive:
            json_members = sorted(
                name for name in archive.namelist() if name.endswith(".json")
            )
            if len(json_members) != 1:
                raise ValueError(
                    f"annotation archive must contain exactly one JSON member: {annotation_path}"
                )
            return json.loads(archive.read(json_members[0]))
    return json.loads(annotation_path.read_text())


def _event_from_row(row: dict[str, Any], key: str) -> dict[str, Any] | None:
    data = load_sample_annotations(row["annotation_path"])
    video = data.get(row["video_id"])
    if not video:
        return None
    segments = video.get(key, [])
    if not segments:
        return None
    if key == "action_localisation":
        seg = next(
            (
                candidate
                for candidate in segments
                if str(candidate.get("label", "")).strip().casefold()
                not in _NON_GROUNDED_ACTION_LABELS
            ),
            None,
        )
        if seg is None:
            return None
    else:
        seg = segments[0]
    ts = seg["timestamps"]
    label = seg["label"]
    positive_windows = [
        [same_label["timestamps"][0] / 1e6, same_label["timestamps"][1] / 1e6]
        for same_label in segments
        if same_label.get("label") == label
    ]
    return {
        "video_id": row["video_id"],
        "label": label,
        "onset_s": ts[0] / 1e6,
        "end_s": ts[1] / 1e6,
        "positive_windows": positive_windows,
        "fps": video["metadata"]["frame_rate"],
        "is_visible": seg.get("is_visible"),
        "path": repo_path(row["media_path"]),
        "row_id": row["id"],
    }


def golden_action_rows(
    *,
    manifest: str | Path = GOLDEN_MANIFEST,
) -> list[dict[str, Any]]:
    """Licensed action rows in pinned-manifest order, with no sampling outside the manifest."""
    return [
        row
        for row in golden_rows("perception-test-action", manifest=manifest)
        if row.get("license") and row.get("action_segments", 0) > 0
    ]


def golden_action_capacity(
    *,
    manifest: str | Path = GOLDEN_MANIFEST,
) -> dict[str, int]:
    """Return declared licensed and locally verified action-case capacity."""
    rows = golden_action_rows(manifest=manifest)
    return {
        "licensed": len(rows),
        "verified_local": sum(verify_row(row, require_media=True) for row in rows),
    }


def golden_action_events(
    *,
    manifest: str | Path = GOLDEN_MANIFEST,
    require_media: bool = True,
) -> list[dict[str, Any]]:
    """One stable action case per row, retaining every same-label positive window."""
    rows = golden_action_rows(manifest=manifest)
    rows = [r for r in rows if verify_row(r, require_media=require_media)]
    return [ev for r in rows if (ev := _event_from_row(r, "action_localisation")) is not None]


def select_golden_action_events(
    n: int,
    *,
    manifest: str | Path = GOLDEN_MANIFEST,
    require_media: bool = True,
) -> list[dict[str, Any]]:
    """Select exactly ``n`` pinned cases or explain why the corpus cannot supply them."""
    if n <= 0:
        raise GoldenSetCapacityError("pt-action-e2e requires N to be positive")
    capacity = golden_action_capacity(manifest=manifest)
    if n > capacity["licensed"]:
        raise GoldenSetCapacityError(
            f"pt-action-e2e requested N={n}, but the pinned manifest has only "
            f"{capacity['licensed']} licensed action cases"
        )
    events = golden_action_events(manifest=manifest, require_media=require_media)
    if n > len(events):
        raise GoldenSetCapacityError(
            f"pt-action-e2e requested N={n}, but only {len(events)} of "
            f"{capacity['licensed']} licensed pinned cases have verified local annotations/media"
        )
    return events[:n]


def golden_sound_events(
    *,
    manifest: str | Path = GOLDEN_MANIFEST,
    require_media: bool = True,
) -> list[dict[str, Any]]:
    """One sound onset per seeded Perception Test sample video with sound annotations."""
    rows = [
        r
        for r in golden_rows("perception-test-acoustic", manifest=manifest)
        if r.get("sound_segments", 0) > 0
    ]
    rows = [r for r in rows if verify_row(r, require_media=require_media)]
    return [ev for r in rows if (ev := _event_from_row(r, "sound_localisation")) is not None]
