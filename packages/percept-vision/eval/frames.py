"""Frame extraction from golden clips — the drop-in the live replay path needs.

The golden corpus stores each clip as ``scenes/<id>/clip.json`` with
``frames: [[t_seconds, base64_jpeg], ...]``. This module decodes those into the
``(ts, frame)`` tuples the replay scorer and the live ``AnthropicVision`` consume — where a
``frame`` is a ``(media_type, bytes)`` pair that ``backends/anthropic.py:_image_block`` accepts
directly. Pure stdlib; no image library required.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Iterator

# A decoded frame as handed to the vision adapter: (media_type, raw_bytes).
Frame = tuple[str, bytes]


def _media_type(raw: bytes) -> str:
    if raw[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if raw[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    return "image/jpeg"  # golden clips are JPEG; default accordingly


def frames_for_scene(scene_dir: str | Path) -> list[tuple[float, Frame]]:
    """Decode ``<scene_dir>/clip.json`` into ``(ts, (media_type, bytes))`` tuples, ts-ordered."""
    clip = json.loads((Path(scene_dir) / "clip.json").read_text())
    out: list[tuple[float, Frame]] = []
    for entry in clip.get("frames", []):
        ts, b64 = entry[0], entry[1]
        raw = base64.b64decode(b64)
        out.append((float(ts), (_media_type(raw), raw)))
    out.sort(key=lambda x: x[0])
    return out


def first_frame(scene_dir: str | Path) -> tuple[float, Frame] | None:
    """The earliest frame of a scene (handy for a cheap single-call smoke), or None if empty."""
    frames = frames_for_scene(scene_dir)
    return frames[0] if frames else None


def iter_scene_frames(
    golden_root: str | Path, scene_ids: list[str]
) -> Iterator[tuple[str, list[tuple[float, Frame]]]]:
    """Yield ``(scene_id, frames)`` for each scene that has a ``clip.json`` on disk.

    Scenes whose pixels aren't present locally (the binaries are git-ignored) are skipped, so a
    checkout without the heavy data degrades cleanly instead of erroring.
    """
    root = Path(golden_root)
    for sid in scene_ids:
        scene_dir = root / "scenes" / sid
        if (scene_dir / "clip.json").exists():
            yield sid, frames_for_scene(scene_dir)


__all__ = ["Frame", "frames_for_scene", "first_frame", "iter_scene_frames"]
