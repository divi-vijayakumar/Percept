"""One-to-one event scorer for the F2-Z1 Watch."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Iterable, Mapping, Sequence

from zone_entry import Occurrence


@dataclass(frozen=True)
class Incident:
    incident_id: str
    action: str
    source_seconds: float
    receiver_ack_seconds: float | None = None
    stage_latency_seconds: Mapping[str, float] | None = None
    source_id: str | None = None


@dataclass(frozen=True)
class RuntimeCounts:
    episodes: int
    context_ledger_entries: int
    outbox_rows: int
    deliveries: int
    receiver_acks: int


def _in_intervals(value: float, intervals: Sequence[tuple[float, float]]) -> bool:
    return any(start <= value <= end for start, end in intervals)


def _percentile(values: Sequence[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def latency_summary(values: Iterable[float]) -> dict[str, float | None]:
    materialized = list(values)
    return {
        "p50": _percentile(materialized, 0.50),
        "p95": _percentile(materialized, 0.95),
        "p99": _percentile(materialized, 0.99),
        "max": max(materialized) if materialized else None,
    }


def score(
    occurrences: Sequence[Occurrence],
    incidents: Sequence[Incident],
    *,
    match_window_seconds: float,
    timely_deadline_seconds: float,
    source_hours: float,
    excluded_intervals_seconds: Sequence[tuple[float, float]] = (),
    runtime_counts: RuntimeCounts | None = None,
    expected_runtime_counts: RuntimeCounts | None = None,
) -> dict[str, object]:
    """Maximum-cardinality one-to-one score with explicit duplicate and ACK accounting."""
    if match_window_seconds < 0 or timely_deadline_seconds < 0 or source_hours <= 0:
        raise ValueError("windows must be non-negative and source_hours must be positive")
    fires = sorted(
        (
            incident
            for incident in incidents
            if incident.action == "fire"
            and not _in_intervals(incident.source_seconds, excluded_intervals_seconds)
        ),
        key=lambda incident: (incident.source_seconds, incident.incident_id),
    )
    asks = [
        incident
        for incident in incidents
        if incident.action == "ask"
        and not _in_intervals(incident.source_seconds, excluded_intervals_seconds)
    ]

    eligible: dict[int, list[int]] = {}
    for fire_index, fire in enumerate(fires):
        eligible[fire_index] = sorted(
            (
                occurrence_index
                for occurrence_index, occurrence in enumerate(occurrences)
                if (fire.source_id is None or occurrence.source_id == fire.source_id)
                and
                occurrence.onset_seconds
                <= fire.source_seconds
                <= occurrence.onset_seconds + match_window_seconds
            ),
            key=lambda index: (
                abs(fire.source_seconds - occurrences[index].onset_seconds),
                occurrences[index].occurrence_id,
            ),
        )

    occurrence_to_fire: dict[int, int] = {}

    def augment(fire_index: int, visited: set[int]) -> bool:
        for occurrence_index in eligible[fire_index]:
            if occurrence_index in visited:
                continue
            visited.add(occurrence_index)
            incumbent = occurrence_to_fire.get(occurrence_index)
            if incumbent is None or augment(incumbent, visited):
                occurrence_to_fire[occurrence_index] = fire_index
                return True
        return False

    for fire_index in range(len(fires)):
        augment(fire_index, set())

    matched_fire_indexes = set(occurrence_to_fire.values())
    unmatched_fire_indexes = set(range(len(fires))) - matched_fire_indexes
    matched_occurrence_indexes = set(occurrence_to_fire)
    duplicate_indexes = {
        fire_index
        for fire_index in unmatched_fire_indexes
        if any(
            occurrence.onset_seconds
            <= fires[fire_index].source_seconds
            <= occurrence.onset_seconds + match_window_seconds
            for occurrence_index, occurrence in enumerate(occurrences)
            if occurrence_index in matched_occurrence_indexes
            and (fires[fire_index].source_id is None or occurrence.source_id == fires[fire_index].source_id)
        )
    }

    tp = len(matched_occurrence_indexes)
    fp = len(unmatched_fire_indexes)
    fn = len(occurrences) - tp
    timely = 0
    onset_to_ack: list[float] = []
    stage_values: dict[str, list[float]] = {}
    unacknowledged = 0
    for occurrence_index, fire_index in occurrence_to_fire.items():
        occurrence = occurrences[occurrence_index]
        fire = fires[fire_index]
        if fire.receiver_ack_seconds is None:
            unacknowledged += 1
        else:
            latency = fire.receiver_ack_seconds - occurrence.onset_seconds
            onset_to_ack.append(latency)
            if 0 <= latency <= timely_deadline_seconds:
                timely += 1
        for name, value in (fire.stage_latency_seconds or {}).items():
            stage_values.setdefault(name, []).append(value)

    reconciliation = None
    if (runtime_counts is None) != (expected_runtime_counts is None):
        raise ValueError("observed and expected runtime counts must be supplied together")
    if runtime_counts is not None and expected_runtime_counts is not None:
        reconciliation = {
            "expected": asdict(expected_runtime_counts),
            "observed": asdict(runtime_counts),
            "exact": runtime_counts == expected_runtime_counts,
        }

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "timely_recall": timely / len(occurrences) if occurrences else None,
        "ask": len(asks),
        "ask_per_source_hour": len(asks) / source_hours,
        "duplicates": len(duplicate_indexes),
        "misses": fn,
        "unacknowledged": unacknowledged,
        "false_fires_per_source_hour": fp / source_hours,
        "onset_to_receiver_ack_seconds": latency_summary(onset_to_ack),
        "stage_latency_seconds": {
            name: latency_summary(values) for name, values in sorted(stage_values.items())
        },
        "matches": [
            {
                "occurrence_id": occurrences[occurrence_index].occurrence_id,
                "incident_id": fires[fire_index].incident_id,
            }
            for occurrence_index, fire_index in sorted(occurrence_to_fire.items())
        ],
        "reconciliation": reconciliation,
    }
