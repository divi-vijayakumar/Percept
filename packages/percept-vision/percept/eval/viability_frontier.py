"""Matched recall-versus-call-reduction frontiers for programmable Watches.

This evaluator measures the perception runtime over live multimodal streams. It compares a
baseline arm with adaptive operating points on the exact same cases, candidate windows, and event
windows. It does not choose a universal threshold: every archetype's policy and operating points
must be preregistered in the input manifest.

The evaluator reports rates and their exact integer denominators. It does not estimate monetary
cost. This module is diagnostic-only: release qualification fails closed until a separately
verified, provenance-preserving measured-evidence adapter is implemented.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from percept._safe_json import SafeJSONInputError, load_json_file, load_json_or_jsonl_file

__all__ = [
    "ARMS",
    "EVIDENCE_KINDS",
    "REQUIRED_MEASURED_ADAPTER",
    "ViabilityInputError",
    "Ratio",
    "ConditionalRatio",
    "CallIntensity",
    "Precision",
    "CallReduction",
    "ArchetypePolicy",
    "ArmMetrics",
    "OperatingPointResult",
    "ArchetypeResult",
    "ViabilityFrontierReport",
    "load_viability_manifest",
    "load_viability_evidence",
    "evaluate_viability_frontier",
    "evaluate_viability_files",
    "render_text",
    "main",
]

ARMS = ("baseline", "adaptive")
EVIDENCE_KINDS = ("measured", "synthetic")
EVENT_OUTCOMES = ("fire", "ask", "silent", "missing")
SCOPE = "programmable Watches over live multimodal streams"
REQUIRED_MEASURED_ADAPTER = "percept.eval.viability_frontier.measured_evidence/v1"
_SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")

_MANIFEST_KEYS = {
    "schema_version",
    "track",
    "scope",
    "evidence_kind",
    "preregistered",
    "archetypes",
}
_ARCHETYPE_KEYS = {"id", "cases", "operating_points", "policy"}
_CASE_KEYS = {"id", "candidate_windows_total", "events"}
_MANIFEST_EVENT_KEYS = {"id", "candidate_windows"}
_OPERATING_POINT_KEYS = {"id", "configuration", "configuration_sha256"}
_POLICY_KEYS = {
    "min_candidate_recall",
    "min_event_recall",
    "min_precision",
    "min_call_reduction",
    "max_event_abstention_rate",
    "max_missing_rate",
}
_ROW_KEYS = {
    "case_id",
    "archetype",
    "operating_point",
    "configuration_sha256",
    "arm",
    "evidence_kind",
    "candidate_windows_total",
    "candidate_windows_admitted",
    "runtime_dispatches",
    "judge_calls",
    "decisions",
    "event_windows",
    "fire_event_ids",
}
_DECISION_KEYS = {"fire", "ask", "silent", "missing"}
_EVENT_KEYS = {"id", "candidate_windows", "admitted_candidates", "outcome"}


class ViabilityInputError(ValueError):
    """The manifest or matched evidence is incomplete, ambiguous, or unsupported."""


def _fail(message: str) -> None:
    raise ViabilityInputError(message)


def _require(condition: bool, message: str) -> None:
    if not condition:
        _fail(message)


def _object(value: Any, context: str) -> dict[str, Any]:
    _require(isinstance(value, Mapping), f"{context} must be an object")
    return dict(value)


def _exact_keys(value: Mapping[str, Any], expected: set[str], context: str) -> None:
    actual = set(value)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    _require(not missing, f"{context} is missing required fields {missing!r}")
    _require(not extra, f"{context} has unsupported fields {extra!r}")


def _identifier(value: Any, context: str) -> str:
    _require(
        isinstance(value, str) and bool(value.strip()),
        f"{context} must be a non-empty string",
    )
    return value.strip()


def _count(value: Any, context: str, *, positive: bool = False) -> int:
    _require(
        isinstance(value, int) and not isinstance(value, bool),
        f"{context} must be an integer",
    )
    minimum = 1 if positive else 0
    _require(value >= minimum, f"{context} must be >= {minimum}")
    return value


def _rate(value: Any, context: str) -> float:
    _require(
        isinstance(value, (int, float)) and not isinstance(value, bool),
        f"{context} must be numeric",
    )
    result = float(value)
    _require(0.0 <= result <= 1.0, f"{context} must be in [0, 1]")
    return result


def _unique_ids(values: Any, context: str, *, minimum: int = 1) -> tuple[str, ...]:
    _require(isinstance(values, list), f"{context} must be a list")
    result = tuple(_identifier(value, f"{context} item") for value in values)
    _require(len(result) >= minimum, f"{context} must contain at least {minimum} item(s)")
    _require(len(result) == len(set(result)), f"{context} contains duplicate ids")
    return result


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return f"sha256:{hashlib.sha256(payload.encode('utf-8')).hexdigest()}"


@dataclass(frozen=True)
class Ratio:
    """An exact numerator and denominator, with no hidden or inferred denominator."""

    numerator: int
    denominator: int

    def __post_init__(self) -> None:
        if self.numerator < 0 or self.denominator <= 0 or self.numerator > self.denominator:
            raise ViabilityInputError(
                f"invalid ratio {self.numerator}/{self.denominator}; require "
                "0 <= numerator <= denominator and denominator > 0"
            )

    @property
    def value(self) -> float:
        return self.numerator / self.denominator

    def to_dict(self) -> dict[str, Any]:
        return {
            "numerator": self.numerator,
            "denominator": self.denominator,
            "value": self.value,
        }


@dataclass(frozen=True)
class ConditionalRatio:
    """An exact ratio whose natural denominator can be zero.

    Undefined rates preserve their exact ``0/0`` shape, expose ``defined=False``, and serialize
    their value as ``null``. Policy code must check ``defined`` rather than treating them as zero.
    """

    numerator: int
    denominator: int

    def __post_init__(self) -> None:
        if self.numerator < 0 or self.denominator < 0 or self.numerator > self.denominator:
            raise ViabilityInputError(
                f"invalid conditional ratio {self.numerator}/{self.denominator}; require "
                "0 <= numerator <= denominator"
            )

    @property
    def defined(self) -> bool:
        return self.denominator > 0

    @property
    def value(self) -> float | None:
        return self.numerator / self.denominator if self.defined else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "numerator": self.numerator,
            "denominator": self.denominator,
            "value": self.value,
            "defined": self.defined,
        }


# Backwards-compatible descriptive name for the most familiar conditional ratio.
Precision = ConditionalRatio


@dataclass(frozen=True)
class CallReduction:
    """Exact ``(baseline calls - adaptive calls) / baseline calls``.

    The numerator may be negative: an adaptive point that makes more calls is evidence too, and
    must be reported as a regression rather than rejected or clipped to zero.
    """

    numerator: int
    denominator: int

    def __post_init__(self) -> None:
        if self.denominator <= 0:
            raise ViabilityInputError("call-reduction denominator must be > 0")

    @property
    def value(self) -> float:
        return self.numerator / self.denominator

    def to_dict(self) -> dict[str, Any]:
        return {
            "numerator": self.numerator,
            "denominator": self.denominator,
            "value": self.value,
        }


@dataclass(frozen=True)
class CallIntensity:
    """Exact provider-call intensity.

    One runtime dispatch can make multiple provider calls (for example, a transition Watch judges
    both its ``from`` and ``to`` predicates). Unlike :class:`Ratio`, the numerator is therefore not
    bounded by the candidate denominator.
    """

    numerator: int
    denominator: int

    def __post_init__(self) -> None:
        if self.numerator < 0 or self.denominator <= 0:
            raise ViabilityInputError(
                f"invalid call intensity {self.numerator}/{self.denominator}; require "
                "numerator >= 0 and denominator > 0"
            )

    @property
    def value(self) -> float:
        return self.numerator / self.denominator

    def to_dict(self) -> dict[str, Any]:
        return {
            "numerator": self.numerator,
            "denominator": self.denominator,
            "value": self.value,
        }


@dataclass(frozen=True)
class ArchetypePolicy:
    """One archetype's preregistered viability policy; no field has a default."""

    min_candidate_recall: float
    min_event_recall: float
    min_precision: float
    min_call_reduction: float
    max_event_abstention_rate: float
    max_missing_rate: float

    @classmethod
    def from_mapping(cls, value: Any, context: str) -> "ArchetypePolicy":
        doc = _object(value, context)
        _exact_keys(doc, _POLICY_KEYS, context)
        return cls(**{key: _rate(doc[key], f"{context}.{key}") for key in _POLICY_KEYS})

    def to_dict(self) -> dict[str, float]:
        return {
            "min_candidate_recall": self.min_candidate_recall,
            "min_event_recall": self.min_event_recall,
            "min_precision": self.min_precision,
            "min_call_reduction": self.min_call_reduction,
            "max_event_abstention_rate": self.max_event_abstention_rate,
            "max_missing_rate": self.max_missing_rate,
        }


@dataclass(frozen=True)
class _EventWindow:
    id: str
    candidate_windows: int
    admitted_candidates: int
    outcome: str

    @classmethod
    def from_mapping(cls, value: Any, context: str) -> "_EventWindow":
        doc = _object(value, context)
        _exact_keys(doc, _EVENT_KEYS, context)
        event_id = _identifier(doc["id"], f"{context}.id")
        candidates = _count(doc["candidate_windows"], f"{context}.candidate_windows", positive=True)
        admitted = _count(doc["admitted_candidates"], f"{context}.admitted_candidates")
        _require(
            admitted <= candidates,
            f"{context}.admitted_candidates exceeds candidate_windows",
        )
        outcome = _identifier(doc["outcome"], f"{context}.outcome")
        _require(outcome in EVENT_OUTCOMES, f"{context}.outcome must be one of {EVENT_OUTCOMES!r}")
        _require(
            outcome in ("silent", "missing") or admitted > 0,
            f"{context}: outcome {outcome!r} requires at least one admitted candidate",
        )
        return cls(event_id, candidates, admitted, outcome)

    def match_signature(self) -> tuple[str, int]:
        return (self.id, self.candidate_windows)


@dataclass(frozen=True)
class _ArmRow:
    case_id: str
    archetype: str
    operating_point: str
    configuration_sha256: str
    arm: str
    evidence_kind: str
    candidate_windows_total: int
    candidate_windows_admitted: int
    runtime_dispatches: int
    judge_calls: int
    decisions: dict[str, int]
    event_windows: tuple[_EventWindow, ...]
    fire_event_ids: tuple[str, ...]

    @classmethod
    def from_mapping(cls, value: Any, index: int) -> "_ArmRow":
        context = f"evidence row {index}"
        doc = _object(value, context)
        _exact_keys(doc, _ROW_KEYS, context)
        case_id = _identifier(doc["case_id"], f"{context}.case_id")
        archetype = _identifier(doc["archetype"], f"{context}.archetype")
        operating_point = _identifier(doc["operating_point"], f"{context}.operating_point")
        configuration_sha256 = _identifier(
            doc["configuration_sha256"],
            f"{context}.configuration_sha256",
        )
        _require(
            bool(_SHA256_RE.fullmatch(configuration_sha256)),
            f"{context}.configuration_sha256 must be sha256:<64 lowercase hex>",
        )
        arm = _identifier(doc["arm"], f"{context}.arm")
        _require(arm in ARMS, f"{context}.arm must be one of {ARMS!r}")
        evidence_kind = _identifier(doc["evidence_kind"], f"{context}.evidence_kind").lower()
        _require(
            evidence_kind in EVIDENCE_KINDS,
            f"{context}.evidence_kind {evidence_kind!r} is unsupported; modeled and stub data "
            f"cannot be evaluated",
        )
        total = _count(
            doc["candidate_windows_total"],
            f"{context}.candidate_windows_total",
            positive=True,
        )
        admitted = _count(
            doc["candidate_windows_admitted"],
            f"{context}.candidate_windows_admitted",
        )
        dispatches = _count(doc["runtime_dispatches"], f"{context}.runtime_dispatches")
        calls = _count(doc["judge_calls"], f"{context}.judge_calls")
        _require(
            admitted <= total,
            f"{context}: admitted candidates exceed the candidate denominator",
        )
        _require(
            dispatches <= admitted,
            f"{context}: runtime dispatches exceed admitted candidates",
        )

        decisions_doc = _object(doc["decisions"], f"{context}.decisions")
        _exact_keys(decisions_doc, _DECISION_KEYS, f"{context}.decisions")
        decisions = {
            key: _count(decisions_doc[key], f"{context}.decisions.{key}")
            for key in _DECISION_KEYS
        }
        _require(
            sum(decisions.values()) == dispatches,
            f"{context}: decision counts sum to {sum(decisions.values())}, not "
            f"runtime_dispatches={dispatches}",
        )

        raw_events = doc["event_windows"]
        _require(
            isinstance(raw_events, list) and raw_events,
            f"{context}.event_windows must be non-empty",
        )
        events = tuple(
            _EventWindow.from_mapping(event, f"{context}.event_windows[{event_index}]")
            for event_index, event in enumerate(raw_events)
        )
        event_ids = [event.id for event in events]
        _require(len(event_ids) == len(set(event_ids)), f"{context} has duplicate event-window ids")
        _require(
            sum(event.candidate_windows for event in events) <= total,
            f"{context}: event candidate windows exceed the case candidate denominator",
        )
        _require(
            sum(event.admitted_candidates for event in events) <= admitted,
            f"{context}: admitted event candidates exceed all admitted candidates",
        )
        _require(
            sum(event.outcome == "fire" for event in events) <= decisions["fire"],
            f"{context}: event-window FIRE outcomes exceed FIRE decisions",
        )
        _require(
            sum(event.outcome == "ask" for event in events) <= decisions["ask"],
            f"{context}: event-window ASK outcomes exceed ASK decisions",
        )
        fire_event_ids = _unique_ids(
            doc["fire_event_ids"],
            f"{context}.fire_event_ids",
            minimum=0,
        )
        _require(
            len(fire_event_ids) == decisions["fire"],
            f"{context}: fire_event_ids count does not match FIRE decisions",
        )
        event_outcomes = {event.id: event.outcome for event in events}
        known_fires = {event_id for event_id in fire_event_ids if event_id in event_outcomes}
        expected_fires = {
            event.id
            for event in events
            if event.outcome == "fire"
        }
        _require(
            known_fires == expected_fires,
            f"{context}: FIRE decisions must be tied exactly to event-window ids with outcome 'fire'",
        )
        false_positive_fires = len(fire_event_ids) - len(known_fires)
        non_event_candidates = total - sum(event.candidate_windows for event in events)
        _require(
            false_positive_fires <= non_event_candidates,
            f"{context}: false-positive FIRE decisions exceed non-event candidate windows",
        )
        return cls(
            case_id,
            archetype,
            operating_point,
            configuration_sha256,
            arm,
            evidence_kind,
            total,
            admitted,
            dispatches,
            calls,
            decisions,
            events,
            fire_event_ids,
        )

    def match_signature(self) -> tuple[int, tuple[tuple[str, int], ...]]:
        return (
            self.candidate_windows_total,
            tuple(event.match_signature() for event in self.event_windows),
        )

    def baseline_signature(self) -> tuple[Any, ...]:
        return (
            self.match_signature(),
            self.candidate_windows_admitted,
            self.runtime_dispatches,
            self.judge_calls,
            tuple(sorted(self.decisions.items())),
            tuple(
                (event.id, event.admitted_candidates, event.outcome)
                for event in self.event_windows
            ),
            self.fire_event_ids,
        )


@dataclass(frozen=True)
class ArmMetrics:
    """Aggregated metrics for one matched arm at one archetype operating point."""

    case_count: int
    candidate_recall: Ratio
    event_recall: Ratio
    precision: ConditionalRatio
    false_positive_rate: Ratio
    admitted_rate: Ratio
    judge_call_intensity: CallIntensity
    decision_abstention_rate: ConditionalRatio
    event_abstention_rate: Ratio
    event_missing_rate: Ratio
    execution_missing_rate: ConditionalRatio
    candidate_windows: int
    admitted_candidates: int
    runtime_dispatches: int
    judge_calls: int
    decisions: dict[str, int]
    event_windows: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_count": self.case_count,
            "candidate_recall": self.candidate_recall.to_dict(),
            "event_recall": self.event_recall.to_dict(),
            "precision": self.precision.to_dict(),
            "false_positive_rate": self.false_positive_rate.to_dict(),
            "admitted_rate": self.admitted_rate.to_dict(),
            "judge_call_intensity": self.judge_call_intensity.to_dict(),
            "decision_abstention_rate": self.decision_abstention_rate.to_dict(),
            "event_abstention_rate": self.event_abstention_rate.to_dict(),
            "event_missing_rate": self.event_missing_rate.to_dict(),
            "execution_missing_rate": self.execution_missing_rate.to_dict(),
            "counts": {
                "candidate_windows": self.candidate_windows,
                "admitted_candidates": self.admitted_candidates,
                "runtime_dispatches": self.runtime_dispatches,
                "judge_calls": self.judge_calls,
                "decisions": dict(self.decisions),
                "event_windows": self.event_windows,
                "true_positive_fires": self.precision.numerator,
                "false_positive_fires": self.precision.denominator - self.precision.numerator,
            },
        }


def _aggregate(rows: Sequence[_ArmRow]) -> ArmMetrics:
    _require(bool(rows), "cannot aggregate an empty arm")
    candidate_windows = sum(row.candidate_windows_total for row in rows)
    admitted = sum(row.candidate_windows_admitted for row in rows)
    dispatches = sum(row.runtime_dispatches for row in rows)
    calls = sum(row.judge_calls for row in rows)
    decisions = {
        key: sum(row.decisions[key] for row in rows)
        for key in _DECISION_KEYS
    }
    event_windows = [event for row in rows for event in row.event_windows]
    event_candidates = sum(event.candidate_windows for event in event_windows)
    admitted_event_candidates = sum(event.admitted_candidates for event in event_windows)
    event_fires = sum(event.outcome == "fire" for event in event_windows)
    fire_decisions = [event_id for row in rows for event_id in row.fire_event_ids]
    true_positive_fires = sum(
        event_id in {event.id for event in row.event_windows}
        for row in rows
        for event_id in row.fire_event_ids
    )
    false_positive_fires = len(fire_decisions) - true_positive_fires
    non_event_candidates = candidate_windows - event_candidates
    event_abstentions = sum(event.outcome in ("ask", "silent") for event in event_windows)
    event_missing = sum(event.outcome == "missing" for event in event_windows)
    missing_units = sum(row.candidate_windows_admitted - row.runtime_dispatches for row in rows)
    missing_units += decisions["missing"]

    return ArmMetrics(
        case_count=len(rows),
        candidate_recall=Ratio(admitted_event_candidates, event_candidates),
        event_recall=Ratio(event_fires, len(event_windows)),
        precision=ConditionalRatio(true_positive_fires, len(fire_decisions)),
        false_positive_rate=Ratio(false_positive_fires, non_event_candidates),
        admitted_rate=Ratio(admitted, candidate_windows),
        judge_call_intensity=CallIntensity(calls, candidate_windows),
        decision_abstention_rate=ConditionalRatio(
            decisions["ask"] + decisions["silent"],
            dispatches,
        ),
        event_abstention_rate=Ratio(event_abstentions, len(event_windows)),
        event_missing_rate=Ratio(event_missing, len(event_windows)),
        execution_missing_rate=ConditionalRatio(missing_units, admitted),
        candidate_windows=candidate_windows,
        admitted_candidates=admitted,
        runtime_dispatches=dispatches,
        judge_calls=calls,
        decisions=decisions,
        event_windows=len(event_windows),
    )


@dataclass(frozen=True)
class OperatingPointResult:
    id: str
    configuration_sha256: str
    baseline: ArmMetrics
    adaptive: ArmMetrics
    call_reduction: CallReduction
    meets_policy_thresholds: bool
    clears_policy: bool
    policy_failures: tuple[str, ...]
    pareto_frontier: bool = False
    policy_feasible_pareto: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "configuration_sha256": self.configuration_sha256,
            "baseline": self.baseline.to_dict(),
            "adaptive": self.adaptive.to_dict(),
            "call_reduction": self.call_reduction.to_dict(),
            "meets_policy_thresholds": self.meets_policy_thresholds,
            "clears_policy": self.clears_policy,
            "policy_failures": list(self.policy_failures),
            "pareto_frontier": self.pareto_frontier,
            "policy_feasible_pareto": self.policy_feasible_pareto,
        }


@dataclass(frozen=True)
class ArchetypeResult:
    id: str
    case_ids: tuple[str, ...]
    policy: ArchetypePolicy
    operating_points: tuple[OperatingPointResult, ...]
    clears_policy: bool

    @property
    def frontier(self) -> tuple[OperatingPointResult, ...]:
        return tuple(point for point in self.operating_points if point.pareto_frontier)

    @property
    def policy_feasible_frontier(self) -> tuple[OperatingPointResult, ...]:
        return tuple(
            point for point in self.operating_points if point.policy_feasible_pareto
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "case_ids": list(self.case_ids),
            "case_count": len(self.case_ids),
            "policy": self.policy.to_dict(),
            "operating_points": [point.to_dict() for point in self.operating_points],
            "frontier_operating_points": [point.id for point in self.frontier],
            "policy_feasible_frontier_operating_points": [
                point.id for point in self.policy_feasible_frontier
            ],
            "clears_policy": self.clears_policy,
        }


@dataclass(frozen=True)
class ViabilityFrontierReport:
    track: str
    scope: str
    evidence_kind: str
    preregistered: bool
    archetypes: tuple[ArchetypeResult, ...]
    launch_qualified: bool
    qualification_reason: str
    qualification_blockers: tuple[str, ...]
    required_measured_adapter: str

    @property
    def policy_cleared(self) -> bool:
        return bool(self.archetypes) and all(
            archetype.clears_policy for archetype in self.archetypes
        )

    def archetype(self, archetype_id: str) -> ArchetypeResult:
        for archetype in self.archetypes:
            if archetype.id == archetype_id:
                return archetype
        raise KeyError(archetype_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "percept.eval.viability_frontier/3",
            "track": self.track,
            "scope": self.scope,
            "declared_evidence_kind": self.evidence_kind,
            "declared_preregistered": self.preregistered,
            "archetype_count": len(self.archetypes),
            "case_count": sum(len(archetype.case_ids) for archetype in self.archetypes),
            "operating_point_count": sum(
                len(archetype.operating_points) for archetype in self.archetypes
            ),
            "policy_cleared": self.policy_cleared,
            "launch_qualified": self.launch_qualified,
            "qualification_reason": self.qualification_reason,
            "qualification_blockers": list(self.qualification_blockers),
            "required_measured_adapter": self.required_measured_adapter,
            "archetypes": [archetype.to_dict() for archetype in self.archetypes],
        }

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)


@dataclass(frozen=True)
class _CaseSpec:
    id: str
    candidate_windows_total: int
    events: tuple[tuple[str, int], ...]

    def match_signature(self) -> tuple[int, tuple[tuple[str, int], ...]]:
        return (self.candidate_windows_total, self.events)


@dataclass(frozen=True)
class _OperatingPointSpec:
    id: str
    configuration_sha256: str


@dataclass(frozen=True)
class _ArchetypeSpec:
    id: str
    cases: tuple[_CaseSpec, ...]
    operating_points: tuple[_OperatingPointSpec, ...]
    policy: ArchetypePolicy


@dataclass(frozen=True)
class _Manifest:
    track: str
    scope: str
    evidence_kind: str
    preregistered: bool
    archetypes: tuple[_ArchetypeSpec, ...]


def _validate_manifest(value: Any) -> _Manifest:
    doc = _object(value, "manifest")
    _exact_keys(doc, _MANIFEST_KEYS, "manifest")
    _require(doc["schema_version"] == "3", "manifest.schema_version must be '3'")
    _require(doc["track"] == "viability_frontier", "manifest.track must be 'viability_frontier'")
    _require(doc["scope"] == SCOPE, f"manifest.scope must be {SCOPE!r}")
    evidence_kind = _identifier(doc["evidence_kind"], "manifest.evidence_kind").lower()
    _require(
        evidence_kind in EVIDENCE_KINDS,
        "manifest.evidence_kind must be 'measured' or 'synthetic'; "
        "modeled and stub data are refused",
    )
    _require(
        isinstance(doc["preregistered"], bool),
        "manifest.preregistered must be a caller-declared boolean",
    )
    raw_archetypes = doc["archetypes"]
    _require(
        isinstance(raw_archetypes, list) and len(raw_archetypes) >= 2,
        "manifest.archetypes must contain at least 2 archetypes",
    )

    archetypes: list[_ArchetypeSpec] = []
    archetype_ids: set[str] = set()
    global_cases: set[str] = set()
    global_events: set[str] = set()
    for index, raw in enumerate(raw_archetypes):
        context = f"manifest.archetypes[{index}]"
        archetype = _object(raw, context)
        _exact_keys(archetype, _ARCHETYPE_KEYS, context)
        archetype_id = _identifier(archetype["id"], f"{context}.id")
        _require(archetype_id not in archetype_ids, f"duplicate archetype id {archetype_id!r}")
        archetype_ids.add(archetype_id)
        raw_cases = archetype["cases"]
        _require(
            isinstance(raw_cases, list) and len(raw_cases) >= 2,
            f"{context}.cases must contain at least 2 manifest-anchored cases",
        )
        cases: list[_CaseSpec] = []
        for case_index, raw_case in enumerate(raw_cases):
            case_context = f"{context}.cases[{case_index}]"
            case = _object(raw_case, case_context)
            _exact_keys(case, _CASE_KEYS, case_context)
            case_id = _identifier(case["id"], f"{case_context}.id")
            _require(case_id not in global_cases, f"duplicate case id {case_id!r}")
            global_cases.add(case_id)
            total = _count(
                case["candidate_windows_total"],
                f"{case_context}.candidate_windows_total",
                positive=True,
            )
            raw_events = case["events"]
            _require(
                isinstance(raw_events, list) and len(raw_events) >= 2,
                f"{case_context}.events must contain at least 2 manifest-anchored events",
            )
            events: list[tuple[str, int]] = []
            for event_index, raw_event in enumerate(raw_events):
                event_context = f"{case_context}.events[{event_index}]"
                event = _object(raw_event, event_context)
                _exact_keys(event, _MANIFEST_EVENT_KEYS, event_context)
                event_id = _identifier(event["id"], f"{event_context}.id")
                _require(event_id not in global_events, f"duplicate event id {event_id!r}")
                global_events.add(event_id)
                event_candidates = _count(
                    event["candidate_windows"],
                    f"{event_context}.candidate_windows",
                    positive=True,
                )
                events.append((event_id, event_candidates))
            anchored_event_candidates = sum(count for _, count in events)
            _require(
                anchored_event_candidates < total,
                f"{case_context}: event candidate windows must be less than the exact case "
                "candidate denominator so false-positive rate has a denominator",
            )
            cases.append(_CaseSpec(case_id, total, tuple(events)))

        raw_points = archetype["operating_points"]
        _require(
            isinstance(raw_points, list) and len(raw_points) >= 2,
            f"{context}.operating_points must contain at least 2 configured points",
        )
        operating_points: list[_OperatingPointSpec] = []
        point_ids: set[str] = set()
        point_fingerprints: set[str] = set()
        for point_index, raw_point in enumerate(raw_points):
            point_context = f"{context}.operating_points[{point_index}]"
            point = _object(raw_point, point_context)
            _exact_keys(point, _OPERATING_POINT_KEYS, point_context)
            point_id = _identifier(point["id"], f"{point_context}.id")
            _require(point_id not in point_ids, f"{context}.operating_points contains duplicate ids")
            point_ids.add(point_id)
            configuration = _object(point["configuration"], f"{point_context}.configuration")
            _require(bool(configuration), f"{point_context}.configuration must be non-empty")
            fingerprint = _identifier(
                point["configuration_sha256"],
                f"{point_context}.configuration_sha256",
            )
            _require(
                bool(_SHA256_RE.fullmatch(fingerprint)),
                f"{point_context}.configuration_sha256 must be sha256:<64 lowercase hex>",
            )
            computed = _canonical_sha256(configuration)
            _require(
                fingerprint == computed,
                f"{point_context}.configuration_sha256 does not match canonical configuration",
            )
            _require(
                fingerprint not in point_fingerprints,
                f"{context}.operating_points must have distinct configuration fingerprints",
            )
            point_fingerprints.add(fingerprint)
            operating_points.append(_OperatingPointSpec(point_id, fingerprint))
        policy = ArchetypePolicy.from_mapping(archetype["policy"], f"{context}.policy")
        archetypes.append(
            _ArchetypeSpec(archetype_id, tuple(cases), tuple(operating_points), policy)
        )
    return _Manifest(
        track="viability_frontier",
        scope=SCOPE,
        evidence_kind=evidence_kind,
        preregistered=doc["preregistered"],
        archetypes=tuple(archetypes),
    )


def load_viability_manifest(path: str | Path) -> dict[str, Any]:
    """Load and validate a preregistered viability manifest."""
    source = _path_label(path)
    try:
        doc = load_json_file(path).value
    except SafeJSONInputError as exc:
        raise ViabilityInputError(f"cannot read manifest {source}: {exc}") from exc
    _validate_manifest(doc)
    return doc


def _path_label(path: object) -> str:
    # Do not invoke arbitrary ``__str__``/``__fspath__`` caller code before the safe reader's
    # single protected coercion. Strings are already inert labels; every other input gets a stable
    # placeholder until a verified receipt exists.
    return path if isinstance(path, str) else "<path input>"


def load_viability_evidence(path: str | Path) -> list[dict[str, Any]]:
    """Load JSON or JSONL evidence. Empty files and malformed rows fail closed."""
    source = _path_label(path)
    try:
        result, is_jsonl = load_json_or_jsonl_file(path)
        payload = result.value
        rows = payload if is_jsonl or isinstance(payload, list) else payload.get("rows")
    except SafeJSONInputError as exc:
        if exc.code == "empty":
            raise ViabilityInputError(f"evidence is empty: {source}") from exc
        action = "parse" if exc.phase == "parse" else "read"
        raise ViabilityInputError(f"cannot {action} evidence {source}: {exc}") from exc
    except AttributeError as exc:
        raise ViabilityInputError(f"cannot parse evidence {source}: {exc}") from exc
    if is_jsonl:
        _require(bool(rows), f"evidence is empty: {source}")
    _require(
        isinstance(rows, list) and rows,
        f"evidence must contain a non-empty row list: {source}",
    )
    return rows


def _policy_failures(
    policy: ArchetypePolicy,
    adaptive: ArmMetrics,
    call_reduction: CallReduction,
) -> tuple[str, ...]:
    failures: list[str] = []
    conditional_metrics = (
        ("precision", adaptive.precision),
        ("decision_abstention_rate", adaptive.decision_abstention_rate),
        ("execution_missing_rate", adaptive.execution_missing_rate),
    )
    for name, metric in conditional_metrics:
        if not metric.defined:
            failures.append(f"{name} is undefined ({metric.numerator}/{metric.denominator})")

    checks = (
        (
            adaptive.candidate_recall.value >= policy.min_candidate_recall,
            "candidate_recall",
            adaptive.candidate_recall.value,
            ">=",
            policy.min_candidate_recall,
        ),
        (
            adaptive.event_recall.value >= policy.min_event_recall,
            "event_recall",
            adaptive.event_recall.value,
            ">=",
            policy.min_event_recall,
        ),
        (
            call_reduction.value >= policy.min_call_reduction,
            "call_reduction",
            call_reduction.value,
            ">=",
            policy.min_call_reduction,
        ),
        (
            adaptive.event_abstention_rate.value <= policy.max_event_abstention_rate,
            "event_abstention_rate",
            adaptive.event_abstention_rate.value,
            "<=",
            policy.max_event_abstention_rate,
        ),
        (
            adaptive.event_missing_rate.value <= policy.max_missing_rate,
            "event_missing_rate",
            adaptive.event_missing_rate.value,
            "<=",
            policy.max_missing_rate,
        ),
    )
    for passed, name, value, operator, threshold in checks:
        if not passed:
            failures.append(f"{name}={value:.6g} must be {operator} {threshold:.6g}")
    if adaptive.precision.defined:
        precision = adaptive.precision.value
        assert precision is not None
        if precision < policy.min_precision:
            failures.append(
                f"precision={precision:.6g} must be >= {policy.min_precision:.6g}"
            )
    if adaptive.execution_missing_rate.defined:
        execution_missing = adaptive.execution_missing_rate.value
        assert execution_missing is not None
        if execution_missing > policy.max_missing_rate:
            failures.append(
                f"execution_missing_rate={execution_missing:.6g} must be "
                f"<= {policy.max_missing_rate:.6g}"
            )
    return tuple(failures)


def _mark_pareto(points: Sequence[OperatingPointResult]) -> tuple[OperatingPointResult, ...]:
    def dominates(other: OperatingPointResult, candidate: OperatingPointResult) -> bool:
        return (
            other.id != candidate.id
            and other.adaptive.event_recall.value >= candidate.adaptive.event_recall.value
            and other.call_reduction.value >= candidate.call_reduction.value
            and (
                other.adaptive.event_recall.value > candidate.adaptive.event_recall.value
                or other.call_reduction.value > candidate.call_reduction.value
            )
        )

    marked: list[OperatingPointResult] = []
    for candidate in points:
        raw_dominated = any(dominates(other, candidate) for other in points)
        feasible_dominated = candidate.meets_policy_thresholds and any(
            other.meets_policy_thresholds and dominates(other, candidate)
            for other in points
        )
        policy_feasible_pareto = (
            candidate.meets_policy_thresholds and not feasible_dominated
        )
        failures = candidate.policy_failures
        if feasible_dominated:
            failures = (*failures, "operating point is dominated and not policy-feasible Pareto")
        marked.append(
            replace(
                candidate,
                pareto_frontier=not raw_dominated,
                policy_feasible_pareto=policy_feasible_pareto,
                clears_policy=policy_feasible_pareto,
                policy_failures=failures,
            )
        )
    return tuple(marked)


def evaluate_viability_frontier(
    manifest: Mapping[str, Any],
    evidence_rows: Sequence[Mapping[str, Any]],
) -> ViabilityFrontierReport:
    """Validate matched arms and compute every manifest-declared archetype frontier."""
    spec = _validate_manifest(manifest)
    _require(bool(evidence_rows), "evidence rows must be non-empty")
    rows = tuple(_ArmRow.from_mapping(row, index) for index, row in enumerate(evidence_rows))

    row_index: dict[tuple[str, str, str], _ArmRow] = {}
    for row in rows:
        key = (row.case_id, row.operating_point, row.arm)
        _require(key not in row_index, f"duplicate evidence row for {key!r}")
        row_index[key] = row

    expected: set[tuple[str, str, str]] = set()
    case_owner: dict[str, str] = {}
    case_specs: dict[str, _CaseSpec] = {}
    point_fingerprints: dict[tuple[str, str], str] = {}
    for archetype in spec.archetypes:
        for point in archetype.operating_points:
            point_fingerprints[(archetype.id, point.id)] = point.configuration_sha256
        for case in archetype.cases:
            case_owner[case.id] = archetype.id
            case_specs[case.id] = case
            for point in archetype.operating_points:
                expected.update((case.id, point.id, arm) for arm in ARMS)
    actual = set(row_index)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    _require(not missing, f"missing baseline/adaptive evidence rows {missing!r}")
    _require(not extra, f"evidence case/operating-point mismatch; unexpected rows {extra!r}")

    for row in rows:
        owner = case_owner[row.case_id]
        _require(
            row.archetype == owner,
            f"case {row.case_id!r} belongs to archetype {owner!r}, not {row.archetype!r}",
        )
        _require(
            row.evidence_kind == spec.evidence_kind,
            f"row {row.case_id!r}/{row.operating_point!r}/{row.arm!r} evidence_kind "
            f"{row.evidence_kind!r} does not match manifest {spec.evidence_kind!r}",
        )
        expected_fingerprint = point_fingerprints[(owner, row.operating_point)]
        _require(
            row.configuration_sha256 == expected_fingerprint,
            f"row {row.case_id!r}/{row.operating_point!r}/{row.arm!r} configuration "
            "fingerprint does not match its manifest operating point",
        )
        anchored_case = case_specs[row.case_id]
        _require(
            row.match_signature() == anchored_case.match_signature(),
            f"{row.archetype}/{row.operating_point}/{row.case_id}: evidence does not match "
            "manifest-anchored exact case/event/candidate denominators",
        )

    archetype_results: list[ArchetypeResult] = []
    for archetype in spec.archetypes:
        canonical_baselines: dict[str, tuple[Any, ...]] = {}
        point_results: list[OperatingPointResult] = []
        for point in archetype.operating_points:
            baseline_rows: list[_ArmRow] = []
            adaptive_rows: list[_ArmRow] = []
            for case in archetype.cases:
                baseline = row_index[(case.id, point.id, "baseline")]
                adaptive = row_index[(case.id, point.id, "adaptive")]
                _require(
                    baseline.match_signature() == adaptive.match_signature(),
                    f"{archetype.id}/{point.id}/{case.id}: baseline and adaptive arms do not share "
                    "the exact candidate/event-window denominators",
                )
                _require(
                    baseline.candidate_windows_admitted == baseline.candidate_windows_total
                    and baseline.runtime_dispatches == baseline.candidate_windows_total,
                    f"{archetype.id}/{point.id}/{case.id}: baseline must be exhaustive "
                    "(admitted == runtime dispatches == candidate total)",
                )
                _require(
                    baseline.decisions["missing"] == 0
                    and all(event.outcome != "missing" for event in baseline.event_windows),
                    f"{archetype.id}/{point.id}/{case.id}: baseline must have no missing work",
                )
                _require(
                    all(
                        event.admitted_candidates == event.candidate_windows
                        for event in baseline.event_windows
                    ),
                    f"{archetype.id}/{point.id}/{case.id}: baseline must admit every "
                    "manifest-anchored event candidate",
                )
                canonical = canonical_baselines.setdefault(
                    case.id,
                    baseline.baseline_signature(),
                )
                _require(
                    canonical == baseline.baseline_signature(),
                    f"{archetype.id}/{case.id}: baseline arm changed across operating points",
                )
                baseline_rows.append(baseline)
                adaptive_rows.append(adaptive)
            baseline_metrics = _aggregate(baseline_rows)
            adaptive_metrics = _aggregate(adaptive_rows)
            _require(
                baseline_metrics.judge_calls > 0,
                f"{archetype.id}/{point}: baseline judge-call denominator is zero",
            )
            reduction = CallReduction(
                baseline_metrics.judge_calls - adaptive_metrics.judge_calls,
                baseline_metrics.judge_calls,
            )
            failures = _policy_failures(archetype.policy, adaptive_metrics, reduction)
            point_results.append(
                OperatingPointResult(
                    id=point.id,
                    configuration_sha256=point.configuration_sha256,
                    baseline=baseline_metrics,
                    adaptive=adaptive_metrics,
                    call_reduction=reduction,
                    meets_policy_thresholds=not failures,
                    clears_policy=False,
                    policy_failures=failures,
                )
            )
        points = _mark_pareto(point_results)
        archetype_results.append(
            ArchetypeResult(
                id=archetype.id,
                case_ids=tuple(case.id for case in archetype.cases),
                policy=archetype.policy,
                operating_points=points,
                clears_policy=any(point.clears_policy for point in points),
            )
        )

    policy_cleared = all(archetype.clears_policy for archetype in archetype_results)
    blockers = [
        (
            f"release qualification is disabled until a separately verified "
            f"{REQUIRED_MEASURED_ADAPTER} adapter supplies provenance-preserving matched evidence"
        ),
        (
            "manifest evidence_kind and preregistered fields are caller-declared metadata, "
            "not independent attestation"
        ),
    ]
    if spec.evidence_kind == "synthetic":
        blockers.append("declared synthetic evidence validates diagnostic mechanics only")
    if not spec.preregistered:
        blockers.append("manifest declares that its policies were not preregistered")
    if not policy_cleared:
        blockers.append(
            "at least one archetype has no policy-feasible point on the Pareto frontier"
        )
    launch_qualified = False
    reason = "; ".join(blockers)
    return ViabilityFrontierReport(
        track=spec.track,
        scope=spec.scope,
        evidence_kind=spec.evidence_kind,
        preregistered=spec.preregistered,
        archetypes=tuple(archetype_results),
        launch_qualified=launch_qualified,
        qualification_reason=reason,
        qualification_blockers=tuple(blockers),
        required_measured_adapter=REQUIRED_MEASURED_ADAPTER,
    )


def evaluate_viability_files(
    manifest_path: str | Path,
    evidence_path: str | Path,
) -> ViabilityFrontierReport:
    return evaluate_viability_frontier(
        load_viability_manifest(manifest_path),
        load_viability_evidence(evidence_path),
    )


def _ratio_text(ratio: Ratio | ConditionalRatio | CallReduction) -> str:
    if isinstance(ratio, ConditionalRatio) and not ratio.defined:
        return f"undefined ({ratio.numerator}/{ratio.denominator})"
    value = ratio.value
    assert value is not None
    return f"{value:.3f} ({ratio.numerator}/{ratio.denominator})"


def render_text(report: ViabilityFrontierReport) -> str:
    """Render a compact report that keeps every denominator and refusal visible."""
    lines = [
        "VIABILITY FRONTIER — RECALL VS JUDGE-CALL REDUCTION",
        f"Scope: {report.scope}",
        (
            f"Caller-declared metadata (not attested): evidence_kind={report.evidence_kind}; "
            f"preregistered={str(report.preregistered).lower()}"
        ),
        (
            f"Manifest-derived coverage: {len(report.archetypes)} archetype(s), "
            f"{sum(len(a.case_ids) for a in report.archetypes)} case(s), "
            f"{sum(len(a.operating_points) for a in report.archetypes)} operating point(s)"
        ),
        "",
    ]
    for archetype in report.archetypes:
        lines.append(
            f"[{archetype.id}] cases={len(archetype.case_ids)} "
            f"policy_cleared={str(archetype.clears_policy).lower()}"
        )
        lines.append(f"  policy={json.dumps(archetype.policy.to_dict(), sort_keys=True)}")
        for point in archetype.operating_points:
            baseline = point.baseline
            adaptive = point.adaptive
            marker = "PARETO" if point.pareto_frontier else "dominated"
            feasible_marker = (
                "policy-feasible-Pareto"
                if point.policy_feasible_pareto
                else "not-policy-feasible-Pareto"
            )
            lines.append(
                f"  {point.id} [{marker}; {feasible_marker}] "
                f"config={point.configuration_sha256} "
                f"clear={str(point.clears_policy).lower()}: "
                f"baseline_event_recall={_ratio_text(baseline.event_recall)}, "
                f"baseline_judge_call_intensity={_ratio_text(baseline.judge_call_intensity)}; "
                f"candidate_recall={_ratio_text(adaptive.candidate_recall)}, "
                f"event_recall={_ratio_text(adaptive.event_recall)}, "
                f"precision={_ratio_text(adaptive.precision)}, "
                f"false_positive_rate={_ratio_text(adaptive.false_positive_rate)}, "
                f"admitted_rate={_ratio_text(adaptive.admitted_rate)}, "
                f"judge_call_intensity={_ratio_text(adaptive.judge_call_intensity)}, "
                f"call_reduction={_ratio_text(point.call_reduction)}, "
                f"decision_abstention={_ratio_text(adaptive.decision_abstention_rate)}, "
                f"event_abstention={_ratio_text(adaptive.event_abstention_rate)}, "
                f"event_missing={_ratio_text(adaptive.event_missing_rate)}, "
                f"execution_missing={_ratio_text(adaptive.execution_missing_rate)}"
            )
            if point.policy_failures:
                lines.append(f"    policy failures: {'; '.join(point.policy_failures)}")
        lines.append("")
    lines.extend(
        [
            f"POLICY CLEARED: {str(report.policy_cleared).upper()}",
            f"LAUNCH QUALIFIED: {str(report.launch_qualified).upper()}",
            f"Reason: {report.qualification_reason}",
            f"Required measured adapter/schema: {report.required_measured_adapter}",
            "Qualification blockers:",
            *(f"- {blocker}" for blocker in report.qualification_blockers),
        ]
    )
    return "\n".join(lines) + "\n"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate matched recall-versus-judge-call-reduction frontiers for programmable "
            "Watches over live multimodal streams."
        )
    )
    parser.add_argument("--manifest", required=True, help="preregistered viability manifest")
    parser.add_argument("--evidence", required=True, help="matched JSON or JSONL evidence rows")
    parser.add_argument("--json", action="store_true", help="emit JSON instead of text")
    parser.add_argument(
        "--require-launch-qualification",
        action="store_true",
        help=(
            "return status 1; release qualification is unavailable until the separately verified "
            "measured-evidence adapter is implemented"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = evaluate_viability_files(args.manifest, args.evidence)
    except ViabilityInputError as exc:
        print(f"VIABILITY INPUT ERROR: {exc}", file=sys.stderr)
        return 2
    print(report.to_json() if args.json else render_text(report), end="" if not args.json else "\n")
    if args.require_launch_qualification and not report.launch_qualified:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
