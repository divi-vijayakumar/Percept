"""Frozen F2-Z1 ground-truth transformation primitives.

This module is deliberately independent of Percept runtime code. It consumes normalized
annotation rows and emits source-clock occurrences suitable for a separately tested scorer.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Sequence


@dataclass(frozen=True)
class ObjectRow:
    track_id: int
    frame: int
    x: float
    y: float
    width: float
    height: float
    class_id: int
    occlusion: str = "none"


@dataclass(frozen=True)
class Occurrence:
    occurrence_id: str
    source_id: str
    track_id: int
    onset_frame: int
    confirmation_frame: int
    onset_seconds: float
    confirmation_seconds: float


def parse_virat_object_rows(lines: Iterable[str]) -> list[ObjectRow]:
    """Parse the official eight-column legacy VIRAT object-track representation."""
    parsed: list[ObjectRow] = []
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 8:
            raise ValueError(f"line {line_number}: expected 8 fields, found {len(fields)}")
        track_id, _duration, frame, x, y, width, height, class_id = map(int, fields)
        parsed.append(
            ObjectRow(
                track_id=track_id,
                frame=frame,
                x=x,
                y=y,
                width=width,
                height=height,
                class_id=class_id,
                occlusion="not_provided",
            )
        )
    return parsed


def point_in_polygon(point: tuple[float, float], polygon: Sequence[Sequence[float]]) -> bool:
    """Return True for points inside or on the boundary of a non-self-intersecting polygon."""
    px, py = point
    inside = False
    count = len(polygon)
    if count < 3:
        raise ValueError("polygon must contain at least three vertices")
    for index in range(count):
        x1, y1 = polygon[index]
        x2, y2 = polygon[(index + 1) % count]
        cross = (px - x1) * (y2 - y1) - (py - y1) * (x2 - x1)
        if abs(cross) <= 1e-12 and min(x1, x2) <= px <= max(x1, x2) and min(
            y1, y2
        ) <= py <= max(y1, y2):
            return True
        if (y1 > py) != (y2 > py):
            intersection_x = (x2 - x1) * (py - y1) / (y2 - y1) + x1
            if px < intersection_x:
                inside = not inside
    return inside


def _excluded(frame: int, intervals: Sequence[tuple[int, int]]) -> bool:
    return any(start <= frame <= end for start, end in intervals)


def transform_zone_entries(
    rows: Iterable[ObjectRow],
    *,
    source_id: str,
    fps: float,
    frame_width: int,
    frame_height: int,
    polygon: Sequence[Sequence[float]],
    dwell_seconds: float,
    excluded_intervals: Sequence[tuple[int, int]] = (),
    person_class_id: int = 1,
    minimum_height_fraction: float = 0.025,
) -> list[Occurrence]:
    """Transform person tracks into clean outside-to-inside entry occurrences.

    A gap, excluded frame, ineligible row, or duplicate frame makes state unknown. An inside
    observation qualifies only after a clean, immediately preceding outside observation and the
    requested continuous dwell. A track first observed inside is therefore left-censored.
    """
    if fps <= 0 or frame_width <= 0 or frame_height <= 0:
        raise ValueError("fps and frame dimensions must be positive")
    if dwell_seconds < 0:
        raise ValueError("dwell_seconds must be non-negative")
    if not 0 <= minimum_height_fraction <= 1:
        raise ValueError("minimum_height_fraction must be in [0, 1]")

    dwell_frame_delta = math.ceil(dwell_seconds * fps)
    by_track: dict[int, list[ObjectRow]] = {}
    for row in rows:
        by_track.setdefault(row.track_id, []).append(row)

    occurrences: list[Occurrence] = []
    for track_id, track_rows in sorted(by_track.items()):
        previous_frame: int | None = None
        clean_outside = False
        pending_onset: int | None = None
        entry_number = 0
        for row in sorted(track_rows, key=lambda item: item.frame):
            consecutive = previous_frame is not None and row.frame == previous_frame + 1
            duplicate_frame = previous_frame == row.frame
            previous_frame = row.frame
            eligible = (
                row.class_id == person_class_id
                and row.height / frame_height >= minimum_height_fraction
                and row.occlusion.lower() != "mostly"
                and not _excluded(row.frame, excluded_intervals)
                and not duplicate_frame
            )
            if not eligible:
                clean_outside = False
                pending_onset = None
                continue

            footpoint = (
                (row.x + row.width / 2) / frame_width,
                (row.y + row.height) / frame_height,
            )
            inside = point_in_polygon(footpoint, polygon)
            if not consecutive:
                clean_outside = not inside
                pending_onset = None
                continue
            if not inside:
                clean_outside = True
                pending_onset = None
                continue
            if pending_onset is None:
                if not clean_outside:
                    continue
                pending_onset = row.frame
            if row.frame - pending_onset >= dwell_frame_delta:
                entry_number += 1
                occurrences.append(
                    Occurrence(
                        occurrence_id=f"{source_id}:{track_id}:{entry_number}",
                        source_id=source_id,
                        track_id=track_id,
                        onset_frame=pending_onset,
                        confirmation_frame=row.frame,
                        onset_seconds=pending_onset / fps,
                        confirmation_seconds=row.frame / fps,
                    )
                )
                clean_outside = False
                pending_onset = None
    return occurrences
