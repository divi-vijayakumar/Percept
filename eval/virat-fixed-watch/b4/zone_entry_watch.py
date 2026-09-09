"""Authored fixed-camera zone-entry Watch primitives.

The detector is intentionally separate from this state machine.  The Watch consumes person
detections, associates them across samples, and emits a pulse only for an observed outside-to-inside
footpoint transition.  A track first seen inside is left-censored.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence


@dataclass(frozen=True)
class Detection:
    bbox: tuple[float, float, float, float]
    score: float

    @property
    def footpoint(self) -> tuple[float, float]:
        return ((self.bbox[0] + self.bbox[2]) / 2, self.bbox[3])


@dataclass
class _Track:
    identifier: int
    footpoint: tuple[float, float]
    inside: bool
    last_seconds: float
    pending_entry_seconds: float | None = None


def point_in_polygon(point: tuple[float, float], polygon: Sequence[Sequence[float]]) -> bool:
    px, py = point
    inside = False
    for index in range(len(polygon)):
        x1, y1 = polygon[index]
        x2, y2 = polygon[(index + 1) % len(polygon)]
        cross = (px - x1) * (y2 - y1) - (py - y1) * (x2 - x1)
        if abs(cross) <= 1e-12 and min(x1, x2) <= px <= max(x1, x2) and min(
            y1, y2
        ) <= py <= max(y1, y2):
            return True
        if (y1 > py) != (y2 > py):
            crossing_x = (x2 - x1) * (py - y1) / (y2 - y1) + x1
            if px < crossing_x:
                inside = not inside
    return inside


class ZoneEntryWatch:
    """Small deterministic footpoint tracker for one fixed camera source."""

    def __init__(
        self,
        *,
        polygon: Sequence[Sequence[float]],
        frame_width: int,
        frame_height: int,
        score_threshold: float = 0.10,
        minimum_height_fraction: float = 0.025,
        maximum_association_distance: float = 0.12,
        maximum_track_gap_seconds: float = 2.1,
        dwell_seconds: float = 0.0,
    ) -> None:
        self.polygon = polygon
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.score_threshold = score_threshold
        self.minimum_height_fraction = minimum_height_fraction
        self.maximum_association_distance = maximum_association_distance
        self.maximum_track_gap_seconds = maximum_track_gap_seconds
        self.dwell_seconds = dwell_seconds
        self._tracks: dict[int, _Track] = {}
        self._next_identifier = 1

    def reset(self) -> None:
        self._tracks.clear()
        self._next_identifier = 1

    def observe(self, detections: Sequence[Detection], source_seconds: float) -> list[int]:
        self._tracks = {
            key: track
            for key, track in self._tracks.items()
            if source_seconds - track.last_seconds <= self.maximum_track_gap_seconds
        }
        eligible = [
            detection
            for detection in detections
            if detection.score >= self.score_threshold
            and (detection.bbox[3] - detection.bbox[1]) / self.frame_height
            >= self.minimum_height_fraction
        ]
        normalized = [
            (
                detection.footpoint[0] / self.frame_width,
                detection.footpoint[1] / self.frame_height,
            )
            for detection in eligible
        ]
        pairs = sorted(
            (
                math.dist(point, track.footpoint),
                detection_index,
                track_id,
            )
            for detection_index, point in enumerate(normalized)
            for track_id, track in self._tracks.items()
        )
        assigned_detections: set[int] = set()
        assigned_tracks: set[int] = set()
        assignments: dict[int, int] = {}
        for distance, detection_index, track_id in pairs:
            if distance > self.maximum_association_distance:
                break
            if detection_index in assigned_detections or track_id in assigned_tracks:
                continue
            assigned_detections.add(detection_index)
            assigned_tracks.add(track_id)
            assignments[detection_index] = track_id

        entries: list[int] = []
        for index, point in enumerate(normalized):
            inside = point_in_polygon(point, self.polygon)
            track_id = assignments.get(index)
            if track_id is None:
                track_id = self._next_identifier
                self._next_identifier += 1
                self._tracks[track_id] = _Track(track_id, point, inside, source_seconds)
                continue
            track = self._tracks[track_id]
            if not track.inside and inside:
                track.pending_entry_seconds = source_seconds
            elif not inside:
                track.pending_entry_seconds = None
            if (
                inside
                and track.pending_entry_seconds is not None
                and source_seconds - track.pending_entry_seconds >= self.dwell_seconds
            ):
                entries.append(track_id)
                track.pending_entry_seconds = None
            track.footpoint = point
            track.inside = inside
            track.last_seconds = source_seconds
        return entries
