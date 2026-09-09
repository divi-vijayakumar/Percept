"""Persistent motion-aware authored Watch for fixed-camera zone entry."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass(frozen=True)
class Detection:
    bbox: tuple[float, float, float, float]
    score: float

    @property
    def footpoint(self) -> tuple[float, float]:
        return ((self.bbox[0] + self.bbox[2]) / 2, self.bbox[3])


@dataclass(frozen=True)
class WatchOutcome:
    action: str
    track_id: int
    reason: str
    footpoint: tuple[float, float]


@dataclass
class _Track:
    identifier: int
    bbox: tuple[float, float, float, float]
    footpoint: tuple[float, float]
    velocity: tuple[float, float]
    last_seconds: float
    outside_observations: int
    armed: bool
    ambiguous_since: float | None
    missed_since_observation: bool = False


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


def _point_segment_distance(
    point: tuple[float, float], start: Sequence[float], end: Sequence[float]
) -> float:
    px, py = point
    x1, y1 = start
    x2, y2 = end
    dx, dy = x2 - x1, y2 - y1
    denominator = dx * dx + dy * dy
    if denominator == 0:
        return math.dist(point, (x1, y1))
    position = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / denominator))
    return math.dist(point, (x1 + position * dx, y1 + position * dy))


def boundary_distance(point: tuple[float, float], polygon: Sequence[Sequence[float]]) -> float:
    return min(
        _point_segment_distance(point, polygon[index], polygon[(index + 1) % len(polygon)])
        for index in range(len(polygon))
    )


def bbox_iou(left: Sequence[float], right: Sequence[float]) -> float:
    x1, y1 = max(left[0], right[0]), max(left[1], right[1])
    x2, y2 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union else 0.0


class PersistentZoneEntryWatch:
    """Track-backed one-pulse zone-entry state machine for one physical source."""

    def __init__(
        self,
        *,
        polygon: Sequence[Sequence[float]],
        frame_width: int,
        frame_height: int,
        score_threshold: float,
        maximum_association_distance: float,
        maximum_track_gap_seconds: float,
        minimum_outside_observations: int,
        boundary_margin_normalized: float,
        duplicate_footpoint_distance: float,
        refractory_seconds: float,
        minimum_height_fraction: float = 0.025,
        duplicate_iou_threshold: float = 0.5,
        refractory_distance: float = 0.03,
        maximum_ambiguous_span_seconds: float = 2.1,
    ) -> None:
        self.polygon = polygon
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.score_threshold = score_threshold
        self.maximum_association_distance = maximum_association_distance
        self.maximum_track_gap_seconds = maximum_track_gap_seconds
        self.minimum_outside_observations = minimum_outside_observations
        self.boundary_margin_normalized = boundary_margin_normalized
        self.duplicate_footpoint_distance = duplicate_footpoint_distance
        self.refractory_seconds = refractory_seconds
        self.minimum_height_fraction = minimum_height_fraction
        self.duplicate_iou_threshold = duplicate_iou_threshold
        self.refractory_distance = refractory_distance
        self.maximum_ambiguous_span_seconds = maximum_ambiguous_span_seconds
        self._tracks: dict[int, _Track] = {}
        self._next_identifier = 1
        self._recent_fires: list[tuple[float, tuple[float, float]]] = []
        self.suppressed_refractory = 0
        self.suppressed_duplicate_detections = 0

    def _normalize(self, detection: Detection) -> tuple[float, float]:
        x, y = detection.footpoint
        return (x / self.frame_width, y / self.frame_height)

    def _side(self, point: tuple[float, float]) -> str:
        if boundary_distance(point, self.polygon) <= self.boundary_margin_normalized:
            return "ambiguous"
        return "inside" if point_in_polygon(point, self.polygon) else "outside"

    def _deduplicate(self, detections: Sequence[Detection]) -> list[Detection]:
        kept: list[Detection] = []
        for detection in sorted(detections, key=lambda item: item.score, reverse=True):
            point = self._normalize(detection)
            duplicate = any(
                bbox_iou(detection.bbox, other.bbox) >= self.duplicate_iou_threshold
                or (
                    self.duplicate_footpoint_distance > 0
                    and math.dist(point, self._normalize(other))
                    <= self.duplicate_footpoint_distance
                )
                for other in kept
            )
            if duplicate:
                self.suppressed_duplicate_detections += 1
            else:
                kept.append(detection)
        return kept

    def _expire(self, source_seconds: float) -> list[WatchOutcome]:
        asks = []
        for track_id, track in list(self._tracks.items()):
            if source_seconds - track.last_seconds <= self.maximum_track_gap_seconds:
                continue
            if track.armed and track.ambiguous_since is not None:
                asks.append(
                    WatchOutcome(
                        "ask",
                        track_id,
                        "armed track expired after unresolved boundary evidence",
                        track.footpoint,
                    )
                )
            del self._tracks[track_id]
        return asks

    def _assign(
        self, detections: Sequence[Detection], source_seconds: float
    ) -> dict[int, int]:
        track_ids = sorted(self._tracks)
        if not track_ids or not detections:
            return {}
        costs = np.full((len(track_ids), len(detections)), 1_000_000.0)
        for row, track_id in enumerate(track_ids):
            track = self._tracks[track_id]
            elapsed = source_seconds - track.last_seconds
            predicted = (
                track.footpoint[0] + track.velocity[0] * elapsed,
                track.footpoint[1] + track.velocity[1] * elapsed,
            )
            for column, detection in enumerate(detections):
                distance = math.dist(predicted, self._normalize(detection))
                if distance <= self.maximum_association_distance:
                    costs[row, column] = distance + 0.02 * (1 - bbox_iou(track.bbox, detection.bbox))
        rows, columns = linear_sum_assignment(costs)
        return {
            int(column): track_ids[int(row)]
            for row, column in zip(rows, columns)
            if costs[row, column] < 1_000_000.0
        }

    def _refractory(self, point: tuple[float, float], source_seconds: float) -> bool:
        self._recent_fires = [
            item for item in self._recent_fires if source_seconds - item[0] <= self.refractory_seconds
        ]
        if self.refractory_seconds <= 0:
            return False
        return any(
            math.dist(point, prior_point) <= self.refractory_distance
            for _, prior_point in self._recent_fires
        )

    def observe(self, detections: Sequence[Detection], source_seconds: float) -> list[WatchOutcome]:
        outcomes = self._expire(source_seconds)
        eligible = [
            detection
            for detection in detections
            if detection.score >= self.score_threshold
            and (detection.bbox[3] - detection.bbox[1]) / self.frame_height
            >= self.minimum_height_fraction
        ]
        eligible = self._deduplicate(eligible)
        assignments = self._assign(eligible, source_seconds)
        observed_track_ids = set(assignments.values())
        for track_id, track in self._tracks.items():
            if track_id not in observed_track_ids:
                track.missed_since_observation = True

        for index, detection in enumerate(eligible):
            point = self._normalize(detection)
            side = self._side(point)
            track_id = assignments.get(index)
            if track_id is None:
                track_id = self._next_identifier
                self._next_identifier += 1
                outside = 1 if side == "outside" else 0
                self._tracks[track_id] = _Track(
                    track_id,
                    detection.bbox,
                    point,
                    (0.0, 0.0),
                    source_seconds,
                    outside,
                    outside >= self.minimum_outside_observations,
                    source_seconds if side == "ambiguous" else None,
                )
                continue

            track = self._tracks[track_id]
            elapsed = max(source_seconds - track.last_seconds, 1e-9)
            measured_velocity = (
                (point[0] - track.footpoint[0]) / elapsed,
                (point[1] - track.footpoint[1]) / elapsed,
            )
            track.velocity = (
                0.5 * track.velocity[0] + 0.5 * measured_velocity[0],
                0.5 * track.velocity[1] + 0.5 * measured_velocity[1],
            )
            had_missing = track.missed_since_observation
            track.bbox = detection.bbox
            track.footpoint = point
            track.last_seconds = source_seconds
            track.missed_since_observation = False

            if side == "outside":
                track.outside_observations += 1
                track.armed = track.outside_observations >= self.minimum_outside_observations
                track.ambiguous_since = None
                continue
            if side == "ambiguous":
                if track.armed and track.ambiguous_since is None:
                    track.ambiguous_since = source_seconds
                continue
            if not track.armed:
                track.ambiguous_since = None
                continue
            unresolved_too_long = (
                track.ambiguous_since is not None
                and source_seconds - track.ambiguous_since > self.maximum_ambiguous_span_seconds
            )
            if had_missing or unresolved_too_long:
                outcomes.append(
                    WatchOutcome(
                        "ask",
                        track_id,
                        "entry evidence lost continuity",
                        point,
                    )
                )
            elif self._refractory(point, source_seconds):
                self.suppressed_refractory += 1
            else:
                outcomes.append(WatchOutcome("fire", track_id, "observable outside-to-inside", point))
                self._recent_fires.append((source_seconds, point))
            track.armed = False
            track.outside_observations = 0
            track.ambiguous_since = None
        return outcomes
