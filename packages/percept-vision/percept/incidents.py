"""Incident producer — turn a fire (a :class:`FireEvent`) into a durable :class:`Incident` record.

The `Incident`/`IncidentSnippet` DTOs live in `percept_contracts` but had no producer; this is it. The
runtime worker calls :func:`make_incident` on every fire and hands the result to its `on_incident` sink.
`fire` is duck-typed (``.goal_id`` / ``.text`` / ``.verdict``) so this module never imports the agent
(no cycle). Pure stdlib.
"""
from __future__ import annotations

import base64
import json
import uuid
from typing import Any

from percept_contracts import Incident, IncidentSnippet, Verdict


def make_incident(fire: Any, *, ts: float, snippet: IncidentSnippet | None = None, id: str | None = None) -> Incident:
    """Map a fire to an :class:`Incident`. ``id`` defaults to a deterministic, sortable handle.

    The fire's ``verdict`` carries the VLM's evidence; ``text`` (the goal's ``say``) becomes the
    human-facing ``message``. ``label`` stays ``None`` until a human / the calibration loop classifies it.
    """
    verdict = getattr(fire, "verdict", None) or Verdict(satisfied=True, confidence=0.0, evidence="")
    goal_id = getattr(fire, "goal_id", "")
    return Incident(
        id=id or f"inc-{goal_id}-{ts:.3f}-{uuid.uuid4().hex[:8]}",
        concern_id=goal_id,
        verdict=verdict,
        ts=float(ts),
        snippet=snippet,
        message=getattr(fire, "text", "") or "",
    )


def _encode_frame_bundle(frames: list, *, t_start: float, t_end: float) -> tuple[str, bytes]:
    bundle = {"version": 1, "t_start": t_start, "t_end": t_end, "frames": []}
    for frame in frames:
        if (
            isinstance(frame, tuple)
            and len(frame) == 2
            and isinstance(frame[0], str)
            and isinstance(frame[1], (bytes, bytearray))
        ):
            bundle["frames"].append(
                {"mime": frame[0], "data": base64.b64encode(bytes(frame[1])).decode("ascii")}
            )
        else:
            bundle["frames"].append({"repr": repr(frame)})
    return "application/x-percept-frame-bundle+json", json.dumps(bundle).encode("utf-8")


def snippet_from_window(window: Any, *, kind: str = "frame", pre_s: float | None = None) -> IncidentSnippet | None:
    """Assemble an :class:`IncidentSnippet` from a rolling :class:`~percept.events.FrameWindow`.

    Grabs the buffered ``(t, frame)`` pairs as the pre-roll (the lead-up to the fire). ``pre_s`` bounds it
    by time (last ``pre_s`` seconds); ``None`` takes the whole window. ``kind="frame"`` keeps the decoded
    frames in ``snippet.frames`` (no encoding). ``kind="clip"`` stores a portable frame-bundle in
    ``snippet.media`` so webhook/storage sinks have one bytes payload without requiring ffmpeg.
    Returns ``None`` for an empty window.
    """
    frames = window.frames() if hasattr(window, "frames") else list(window)
    if not frames:
        return None
    t_end = frames[-1][0]
    if pre_s is not None:
        cutoff = t_end - pre_s
        frames = [(t, f) for (t, f) in frames if t >= cutoff]
    if not frames:
        return None
    payload_frames = [f for (_t, f) in frames]
    t_start, t_end = frames[0][0], frames[-1][0]
    if kind == "clip":
        mime, media = _encode_frame_bundle(payload_frames, t_start=t_start, t_end=t_end)
        return IncidentSnippet(kind="clip", media=media, mime=mime, t_start=t_start, t_end=t_end)
    return IncidentSnippet(kind=kind, frames=payload_frames, t_start=t_start, t_end=t_end)
