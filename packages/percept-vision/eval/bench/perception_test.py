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
VIDEOS_ZIP = f"{GCS}/valid_videos.zip"

STAGE = Path(__file__).resolve().parent.parent / "bench-staging" / "perception_test"
VIDEO_DIR = STAGE / "videos"
REPO_ROOT = Path(__file__).resolve().parents[4]
GOLDEN_MANIFEST = REPO_ROOT / "eval" / "golden-v1" / "MANIFEST.json"


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
        if len(out) >= n:
            break
    return out


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
    """Pin the exact sample (data-manifest for the card)."""
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
    return p if p.is_absolute() else REPO_ROOT / p


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
    return json.loads(repo_path(path).read_text())


def _event_from_row(row: dict[str, Any], key: str) -> dict[str, Any] | None:
    data = load_sample_annotations(row["annotation_path"])
    video = data.get(row["video_id"])
    if not video:
        return None
    segments = video.get(key, [])
    if not segments:
        return None
    seg = segments[0]
    ts = seg["timestamps"]
    return {
        "video_id": row["video_id"],
        "label": seg["label"],
        "onset_s": ts[0] / 1e6,
        "end_s": ts[1] / 1e6,
        "fps": video["metadata"]["frame_rate"],
        "is_visible": seg.get("is_visible"),
        "path": repo_path(row["media_path"]),
        "row_id": row["id"],
    }


def golden_action_events(*, require_media: bool = True) -> list[dict[str, Any]]:
    """One action onset per seeded Perception Test sample video."""
    rows = [r for r in golden_rows() if r.get("action_segments", 0) > 0]
    rows = [r for r in rows if verify_row(r, require_media=require_media)]
    return [ev for r in rows if (ev := _event_from_row(r, "action_localisation")) is not None]


def golden_sound_events(*, require_media: bool = True) -> list[dict[str, Any]]:
    """One sound onset per seeded Perception Test sample video with sound annotations."""
    rows = [r for r in golden_rows("perception-test-acoustic") if r.get("sound_segments", 0) > 0]
    rows = [r for r in rows if verify_row(r, require_media=require_media)]
    return [ev for r in rows if (ev := _event_from_row(r, "sound_localisation")) is not None]
