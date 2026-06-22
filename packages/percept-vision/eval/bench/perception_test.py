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

GCS = "https://storage.googleapis.com/dm-perception-test/zip_data"
ANN_ZIP = {
    "sound": "challenge_sound_localisation_valid_annotations.zip",
    "action": "challenge_action_localisation_valid_annotations.zip",
}
TASK_KEY = {"sound": "sound_localisation", "action": "action_localisation"}
VIDEOS_ZIP = f"{GCS}/valid_videos.zip"

STAGE = Path(__file__).resolve().parent.parent / "bench-staging" / "perception_test"
VIDEO_DIR = STAGE / "videos"


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
