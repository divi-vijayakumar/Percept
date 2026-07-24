"""A minimal reader for MEVA KPF *activity* annotations.

KPF (Kitware Packet Format) is the YAML annotation format MEVA ships. Three files describe one
clip: ``<clip>.activities.yml`` (the ``act`` records), ``<clip>.geom.yml`` (per-frame boxes) and
``<clip>.types.yml`` (actor id -> object type). This transform only needs the ACTIVITIES file: an
activity's *type* and its BEGIN frame are what become a Percept ground-truth episode's condition and
fire timestamp.

Canonical KPF activity record (flow style, exactly as MEVA writes it)::

    - { meta: "activity annotations ..." }
    - { act: { act2: { person_opens_facility_door: 1.0 }, id2: 6, src: truth,
               timespan: [ { tsr0: [ 300, 480 ] } ],
               actors: [ { id1: 12, timespan: [ { tsr0: [ 300, 480 ] } ] } ] } }

* ``act2`` maps the activity type name -> confidence (a dict). Older/DIVA-style files instead carry
  a plain ``activity: <name>`` string; both are accepted.
* ``id2`` is the activity id (unique within a clip).
* ``timespan`` is a list of ``{ tsr0: [begin_frame, end_frame] }`` spans; the activity's begin is
  the earliest ``begin_frame`` and its end the latest ``end_frame``.
* ``actors`` lists the participating track ids (``id1``).

Frames -> seconds uses a fixed clip frame rate (MEVA is 30 fps); the reader carries it so callers
work in seconds on the same timeline the fixtures use.

This module is stdlib + PyYAML only (PyYAML is a dev/test dep of the ``eval`` tree, never of
``percept`` core). It performs no network I/O.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

__all__ = [
    "KpfError",
    "MevaActivity",
    "MevaClip",
    "MEVA_DEFAULT_FPS",
    "parse_clip_id",
    "load_kpf_activities",
    "load_kpf_file",
    "iter_kpf_dir",
]

#: MEVA is recorded at 30 fps; the frame index of an activity's begin divided by this is its second.
MEVA_DEFAULT_FPS = 30.0

#: A MEVA camera-location code, e.g. ``G330`` / ``G420`` — a leading letter then two-plus digits.
#: A clip stem is ``<date>.<start>.<end>.<site>.<CAMERA>[.<extra>...]``; the camera token is where
#: the SCENE (everything before it — the multiview recording session) ends and the VIEW begins.
_CAMERA_RE = re.compile(r"^[A-Z]\d{2,}$")

#: KPF sibling suffixes stripped to recover the bare clip stem.
_KPF_SUFFIXES = (".activities.yml", ".geom.yml", ".types.yml", ".activities.yaml", ".yml", ".yaml", ".avi")


class KpfError(ValueError):
    """A KPF activities document (or a clip filename) was malformed or unreadable."""


@dataclass(frozen=True)
class MevaActivity:
    """One MEVA activity: its type, its frame span, and the track ids that took part.

    ``begin_frame`` is the load-bearing field for the transform — it becomes the ground-truth
    episode's fire timestamp (``begin_frame / fps`` seconds).
    """

    activity_id: int
    activity_type: str
    begin_frame: int
    end_frame: int
    actor_ids: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if not self.activity_type:
            raise KpfError("MevaActivity.activity_type is required (non-empty)")
        if self.end_frame < self.begin_frame:
            raise KpfError(
                f"activity {self.activity_id} ({self.activity_type!r}): end_frame "
                f"{self.end_frame} < begin_frame {self.begin_frame}"
            )

    def begin_ts(self, fps: float = MEVA_DEFAULT_FPS) -> float:
        """The activity BEGIN, in seconds — the fire timestamp a Percept episode is anchored to."""
        return self.begin_frame / fps

    def end_ts(self, fps: float = MEVA_DEFAULT_FPS) -> float:
        """The activity END, in seconds (the labeled episode's close)."""
        return self.end_frame / fps


@dataclass(frozen=True)
class MevaClip:
    """One MEVA clip (one camera view of one recording session) and its activities.

    ``scene_id`` groups every camera VIEW of the SAME recording session (``<date>.<start>.<end>.
    <site>``). It is the unit the held-out split MUST partition on, because the same physical event
    is annotated once per camera and a clip-level split would leak it across train/test.
    """

    clip_id: str
    scene_id: str
    site: str
    camera: str
    date: str
    fps: float
    activities: tuple[MevaActivity, ...] = ()


def parse_clip_id(name: str) -> tuple[str, str, str, str]:
    """Parse a MEVA clip filename/stem into ``(clip_id, scene_id, site, camera)``.

    ``2018-03-05.11-25-00.11-30-00.school.G330.activities.yml`` ->
    ``("2018-03-05.11-25-00.11-30-00.school.G330", "2018-03-05.11-25-00.11-30-00.school",
       "school", "G330")``.

    The SCENE is everything up to (not including) the first camera-code token; the camera code is
    the view. A stem with no recognizable camera token is rejected loudly rather than silently
    treated as its own scene — that would risk the exact multiview leak this split exists to prevent.
    """
    stem = name
    for suffix in _KPF_SUFFIXES:
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    parts = stem.split(".")
    cam_index = next((i for i, tok in enumerate(parts) if _CAMERA_RE.match(tok)), None)
    if cam_index is None or cam_index == 0:
        raise KpfError(
            f"cannot locate a camera-location token (e.g. 'G330') in clip name {name!r}; a MEVA "
            f"clip is '<date>.<start>.<end>.<site>.<CAMERA>' and the camera token is required to "
            f"derive the scene id (grouping all camera views of one recording session)"
        )
    scene_id = ".".join(parts[:cam_index])
    site = parts[cam_index - 1]
    camera = parts[cam_index]
    return stem, scene_id, site, camera


def _activity_type(record: Mapping[str, Any]) -> str:
    """Extract the activity type name from a KPF ``act`` record (``act2`` dict or ``activity`` str)."""
    act2 = record.get("act2")
    if isinstance(act2, Mapping) and act2:
        # {type_name: confidence, ...} — pick the highest-confidence name, ties broken by name so
        # the read is deterministic.
        return max(sorted(act2), key=lambda k: (float(act2[k]), k)) if len(act2) > 1 else next(iter(act2))
    if isinstance(act2, str) and act2:
        return act2
    activity = record.get("activity")
    if isinstance(activity, str) and activity:
        return activity
    raise KpfError(f"act record has no 'act2' type map or 'activity' name: {record!r}")


def _frame_bounds(timespan: Any, *, context: str) -> tuple[int, int]:
    """Fold a KPF ``timespan`` (list of ``{tsr0: [begin, end]}``) into ``(min begin, max end)``."""
    begins: list[int] = []
    ends: list[int] = []
    for span in timespan or ():
        if not isinstance(span, Mapping):
            continue
        tsr = span.get("tsr0")
        if isinstance(tsr, Sequence) and len(tsr) >= 2:
            begins.append(int(tsr[0]))
            ends.append(int(tsr[1]))
    if not begins:
        raise KpfError(f"{context}: no usable 'tsr0' frame span in timespan {timespan!r}")
    return min(begins), max(ends)


def load_kpf_activities(text: str, *, clip_id: str, scene_id: str, site: str, camera: str,
                        date: str, fps: float = MEVA_DEFAULT_FPS) -> MevaClip:
    """Parse KPF activities YAML ``text`` into a :class:`MevaClip`.

    Non-``act`` records (``meta`` headers, blank docs) are skipped. Each ``act`` record yields one
    :class:`MevaActivity`; a record missing a type or a frame span raises :class:`KpfError` rather
    than being silently dropped.
    """
    try:
        import yaml  # noqa: PLC0415 — dev/test dep of the eval tree, not of percept core
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise KpfError(
            "reading KPF requires PyYAML (a dev dependency of the eval tree). Install it with "
            "`pip install pyyaml`."
        ) from exc

    data = yaml.safe_load(text) or []
    if not isinstance(data, list):
        raise KpfError(f"{clip_id}: KPF activities file must be a YAML list, got {type(data).__name__}")

    activities: list[MevaActivity] = []
    for index, item in enumerate(data):
        if not isinstance(item, Mapping) or "act" not in item:
            continue
        record = item["act"]
        if not isinstance(record, Mapping):
            raise KpfError(f"{clip_id}: item {index} 'act' is not a mapping: {record!r}")
        activity_type = _activity_type(record)
        activity_id = int(record.get("id2", index))
        begin, end = _frame_bounds(
            record.get("timespan"), context=f"{clip_id} act id2={activity_id} ({activity_type})"
        )
        actor_ids = tuple(
            int(a["id1"])
            for a in record.get("actors", ())
            if isinstance(a, Mapping) and "id1" in a
        )
        activities.append(
            MevaActivity(
                activity_id=activity_id,
                activity_type=activity_type,
                begin_frame=begin,
                end_frame=end,
                actor_ids=actor_ids,
            )
        )

    return MevaClip(
        clip_id=clip_id,
        scene_id=scene_id,
        site=site,
        camera=camera,
        date=date,
        fps=fps,
        activities=tuple(sorted(activities, key=lambda a: (a.begin_frame, a.activity_id))),
    )


def load_kpf_file(path: str | Path, *, fps: float = MEVA_DEFAULT_FPS) -> MevaClip:
    """Load one ``<clip>.activities.yml`` file into a :class:`MevaClip` (clip id parsed from name)."""
    p = Path(path)
    if not p.exists():
        raise KpfError(f"KPF activities file not found: {p}")
    clip_id, scene_id, site, camera = parse_clip_id(p.name)
    date = clip_id.split(".", 1)[0]
    return load_kpf_activities(
        p.read_text(), clip_id=clip_id, scene_id=scene_id, site=site, camera=camera,
        date=date, fps=fps,
    )


def iter_kpf_dir(directory: str | Path, *, fps: float = MEVA_DEFAULT_FPS) -> tuple[MevaClip, ...]:
    """Load every ``*.activities.yml`` under ``directory`` into :class:`MevaClip` objects (sorted)."""
    base = Path(directory)
    if not base.is_dir():
        raise KpfError(f"KPF directory not found: {base}")
    files = sorted(base.glob("*.activities.yml")) + sorted(base.glob("*.activities.yaml"))
    if not files:
        raise KpfError(f"no *.activities.yml KPF files under {base}")
    return tuple(load_kpf_file(f, fps=fps) for f in files)
