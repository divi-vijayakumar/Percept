"""The track contract — every track scorer returns a TrackResult; the benchmark emitter consumes a list of them.

A track scorer is a module exposing::

    def score(*, live: bool, n: int) -> TrackResult

- ``live=False``  -> a fast OFFLINE result so the benchmark card renders with no API.
- ``live=True``   -> the real numbers (Gemini), reusing the existing per-track tool.

Cells are keyed ``(lane, metric)`` with ``lane in CONFIGS``. Lanes match the benchmark methodology:

``dense-modeled``
    A non-executed baseline estimate, usually call-count/cost only. No dense live VLM loops.
``edge``
    The conductor/edge layer only: wake decisions, recall, precision proxy, and call economics.
``e2e``
    Product path: goal -> edge candidates -> backend judges candidates -> core emits fire/ask/silent.
``core-replay``
    The cognition core replayed over already-collected observations/verdicts.

Numbers that are *modeled* (not executed) must be listed in ``.modeled`` so the benchmark card flags them with a
``~``. Non-dense modeled result cells refuse a RELEASE stamp; ``dense-modeled`` cost references are
allowed because they are explicitly part of the methodology. Tracks that are useful diagnostics but not
publishable product benchmarks must set ``diagnostic=True`` and explain why in ``excluded``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

CONFIGS = ("dense-modeled", "edge", "e2e", "core-replay")
TRACK_RESULT_SCHEMA_V1 = "percept.benchmark.track_result/1"
TRACK_RESULT_SCHEMA_V2 = "percept.benchmark.track_result/2"
TRACK_RESULT_SCHEMAS = (TRACK_RESULT_SCHEMA_V1, TRACK_RESULT_SCHEMA_V2)


def bench_vision(settings=None, *, request_budget=None):
    """Build the judge backend for a live track from ``PERCEPT_VISION_BACKEND`` (default ``gemini``).

    Resolves through the registry, so one env var runs the same track on any registered vision backend
    (``gemini`` · ``gemini-live`` · ``openrouter`` · ``claude``) — the seam the model sweep needs, replacing
    the hardcoded ``GeminiVision`` import in each adapter."""
    from percept import registry

    backend = os.environ.get("PERCEPT_VISION_BACKEND", "gemini")
    vision = registry.make_vision(backend, settings)
    if request_budget is not None:
        configure = getattr(vision, "configure_request_budget", None)
        if not callable(configure):
            raise RuntimeError(
                f"request-budget enforcement is not available for vision backend {backend!r}"
            )
        configure(request_budget)
    return vision


@dataclass
class Coverage:
    """Per-track coverage/provenance gate for a publishable benchmark card."""

    dataset: str
    expected: int
    observed: int
    failed: int = 0
    skipped: int = 0
    notes: str = ""
    case_ids: list[str] = field(default_factory=list)
    positive_events: int | None = None

    @property
    def passed(self) -> bool:
        return (
            self.expected > 0
            and self.observed >= self.expected
            and self.failed == 0
            and self.skipped == 0
            and len(self.case_ids) == self.expected
            and len(set(self.case_ids)) == len(self.case_ids)
        )


@dataclass(frozen=True)
class Measurement:
    """One typed benchmark measurement with its denominator and provenance.

    ``TrackResult.cells`` remains the compact rendering interface.  Version-2
    qualification artifacts additionally carry these records so values cannot be
    separated from their units, conditioning population, or measurement source.
    """

    value: int | float | str | None
    unit: str
    conditioning: str
    source: str = "measured"  # measured | modeled
    numerator: int | float | None = None
    denominator: int | float | None = None
    provenance: dict[str, Any] = field(default_factory=dict)
    valid: bool = True

    def invalid_reasons(self, label: str) -> list[str]:
        reasons: list[str] = []
        if self.value is None:
            reasons.append(f"{label}: value is missing")
        if not self.unit.strip():
            reasons.append(f"{label}: unit is missing")
        if not self.conditioning.strip():
            reasons.append(f"{label}: conditioning population is missing")
        if self.source not in {"measured", "modeled"}:
            reasons.append(f"{label}: unsupported source {self.source!r}")
        if self.denominator is not None and self.denominator <= 0:
            reasons.append(f"{label}: denominator must be positive")
        if not self.valid:
            reasons.append(f"{label}: measurement is marked invalid")
        return reasons


@dataclass
class TrackResult:
    track: str  # "relational" | "counting" | "acoustic" | ...
    cells: dict = field(default_factory=dict)  # {(lane, metric): value}  e.g. ("e2e","precision"): 0.83
    fmts: dict = field(default_factory=dict)  # {metric: "{:.2f}"} optional per-metric format
    headline: dict = field(default_factory=dict)  # {label: str} -> benchmark headline vector
    measured: list = field(default_factory=list)  # provenance: measured cell labels
    modeled: list = field(default_factory=list)  # provenance: modeled cells (rendered with ~)
    excluded: list = field(default_factory=list)  # invalid items + reason
    reference: dict = field(default_factory=dict)  # {metric: {"value": v, "by": "TransRAC CVPR'22"}} SOTA ref
    calls: int = 0  # live vision calls spent (cost)
    notes: str = ""
    # Exact effective knobs needed to reconstruct the track.  Unlike ``notes`` this stays structured
    # in the saved JSON, so a result never has to infer launch-critical defaults from a source commit.
    configuration: dict = field(default_factory=dict)
    coverage: list[Coverage | dict] = field(default_factory=list)
    diagnostic: bool = False
    status: str = "pass"  # pass | skipped | blocked | failed
    qualifying_events: int | None = None  # fires/answers that make an E2E result non-vacuous
    schema_version: str = TRACK_RESULT_SCHEMA_V1
    # Version-2 qualification envelope.  Legacy diagnostic tracks intentionally
    # remain readable as v1 until they are migrated to the compiled runtime path.
    contract: dict[str, Any] = field(default_factory=dict)
    measurements: dict[str, Measurement | dict[str, Any]] = field(default_factory=dict)
    reconciliation: dict[str, Any] = field(default_factory=dict)
    quality_gates: list[dict[str, Any]] = field(default_factory=list)

    def metrics(self) -> list[str]:
        """Distinct metric names in cell-insertion order."""
        seen: list[str] = []
        for _config, metric in self.cells:
            if metric not in seen:
                seen.append(metric)
        return seen

    def fmt(self, metric: str) -> str:
        return self.fmts.get(metric, "{}")

    def normalized_coverage(self) -> list[Coverage]:
        out: list[Coverage] = []
        for item in self.coverage:
            out.append(item if isinstance(item, Coverage) else Coverage(**item))
        return out

    def normalized_measurements(self) -> dict[str, Measurement]:
        return {
            label: (
                item
                if isinstance(item, Measurement)
                else Measurement(**item)
            )
            for label, item in self.measurements.items()
        }

    def invalid_reasons(self) -> list[str]:
        reasons: list[str] = []
        if self.schema_version not in TRACK_RESULT_SCHEMAS:
            reasons.append(
                f"{self.track}: unsupported result schema {self.schema_version!r}"
            )
        if self.track == "pt-action-e2e" and self.schema_version != TRACK_RESULT_SCHEMA_V2:
            reasons.append(
                f"{self.track}: OSS qualification requires {TRACK_RESULT_SCHEMA_V2}"
            )
        if self.schema_version == TRACK_RESULT_SCHEMA_V2:
            if not self.contract:
                reasons.append(f"{self.track}: v2 contract is missing")
            if not self.measurements:
                reasons.append(f"{self.track}: v2 measurements are missing")
            if not self.reconciliation:
                reasons.append(f"{self.track}: v2 reconciliation is missing")
            if not self.quality_gates:
                reasons.append(f"{self.track}: v2 quality gates are missing")
            for label, item in self.normalized_measurements().items():
                reasons.extend(
                    f"{self.track}: {reason}"
                    for reason in item.invalid_reasons(label)
                )
            for gate in self.quality_gates:
                gate_id = str(gate.get("id") or "(unnamed)")
                gate_status = gate.get("status")
                if gate_status not in {"pass", "fail", "blocked"}:
                    reasons.append(
                        f"{self.track}: quality gate {gate_id} has invalid status "
                        f"{gate_status!r}"
                    )
                elif gate_status != "pass":
                    reasons.append(
                        f"{self.track}: quality gate {gate_id}={gate_status}"
                    )
        if self.status != "pass":
            reasons.append(f"{self.track}: status={self.status}")
        if self.diagnostic:
            reasons.append(f"{self.track}: diagnostic-only")
        if not self.cells:
            reasons.append(f"{self.track}: no result cells")
        if not self.measured:
            reasons.append(f"{self.track}: no measured cells")
        if not self.coverage:
            reasons.append(f"{self.track}: no coverage evidence")
        event_metric = any(
            lane == "e2e" and metric in {"notification_recall", "interruptions"}
            for lane, metric in self.cells
        )
        if event_metric and self.qualifying_events is None:
            reasons.append(f"{self.track}: qualifying E2E event count not recorded")
        if self.qualifying_events == 0:
            reasons.append(f"{self.track}: no qualifying E2E events")
        reasons.extend(str(e) for e in self.excluded)
        reasons.extend(f"{self.track}: modeled {m}" for m in self.modeled if not str(m).startswith("dense-modeled:"))
        measured = set(str(item) for item in self.measured)
        modeled = set(str(item) for item in self.modeled)
        for (lane, metric), value in self.cells.items():
            label = f"{lane}:{metric}"
            if value is None:
                reasons.append(f"{self.track}: incomplete cell {label}")
            elif label not in measured and label not in modeled:
                reasons.append(f"{self.track}: cell lacks measured/modeled provenance {label}")
        known = {f"{lane}:{metric}" for lane, metric in self.cells}
        for label in measured | modeled:
            if label not in known:
                reasons.append(f"{self.track}: provenance references missing cell {label}")
        for item in self.normalized_coverage():
            if not item.passed:
                reasons.append(
                    f"{self.track}: coverage failed for {item.dataset} "
                    f"(expected={item.expected}, observed={item.observed}, failed={item.failed}, skipped={item.skipped})"
                )
        for lane, metric in self.cells:
            if lane not in CONFIGS:
                reasons.append(f"{self.track}: unsupported lane {lane!r} for metric {metric!r}")
        return reasons

    def release_contract_reasons(self) -> list[str]:
        """Strict v2 runtime proof required for an OSS release stamp."""
        if self.schema_version != TRACK_RESULT_SCHEMA_V2:
            return [
                f"{self.track}: release requires {TRACK_RESULT_SCHEMA_V2}"
            ]
        reasons: list[str] = []
        required_contract = (
            "runtime_path",
            "qualification_scope",
            "plan_hashes",
            "component_snapshot_hashes",
            "evidence",
            "decision_output",
            "dense_baseline",
        )
        for key in required_contract:
            if not self.contract.get(key):
                reasons.append(
                    f"{self.track}: release contract missing {key}"
                )
        runtime_path = self.contract.get("runtime_path")
        required_runtime_stages = (
            "WatchPlan",
            "ConfigResolver",
            "ResolvedWatch",
            "RuntimeCompiler",
            "JudgePool",
            "Incident",
        )
        if not isinstance(runtime_path, str) or not all(
            stage in runtime_path for stage in required_runtime_stages
        ):
            reasons.append(
                f"{self.track}: release contract omits canonical runtime stages"
            )
        if self.contract.get("decision_output") != "Incident":
            reasons.append(
                f"{self.track}: release contract decision_output must be Incident"
            )
        for key in ("plan_hashes", "component_snapshot_hashes"):
            hashes = self.contract.get(key)
            if (
                not isinstance(hashes, list)
                or not hashes
                or any(
                    not isinstance(value, str)
                    or len(value) != 64
                    or any(
                        char not in "0123456789abcdef"
                        for char in value.lower()
                    )
                    for value in hashes
                )
            ):
                reasons.append(
                    f"{self.track}: release contract has invalid {key}"
                )
        dense = self.contract.get("dense_baseline")
        if not isinstance(dense, dict):
            reasons.append(
                f"{self.track}: release contract dense_baseline must be an object"
            )
        else:
            if dense.get("unit") != "watch opportunities":
                reasons.append(
                    f"{self.track}: dense baseline unit must be watch opportunities"
                )
            if not isinstance(dense.get("formula"), str) or not dense[
                "formula"
            ].strip():
                reasons.append(
                    f"{self.track}: dense baseline formula is missing"
                )
            for key in ("capture_fps", "active_watches"):
                value = dense.get(key)
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or value <= 0
                ):
                    reasons.append(
                        f"{self.track}: dense baseline {key} must be positive"
                    )
        checks = self.reconciliation.get("checks")
        if not isinstance(checks, dict) or not checks:
            reasons.append(
                f"{self.track}: release reconciliation checks are missing"
            )
        elif not all(value is True for value in checks.values()):
            reasons.append(
                f"{self.track}: release reconciliation check failed"
            )
        zero_fields = (
            "source_dropped_frames",
            "forbidden_evidence_fallbacks",
            "superseded_before_launch",
            "revision_dropped_before_launch",
            "deadline_dropped_before_launch",
            "deadline_expired_after_launch",
            "final_provider_inflight",
            "final_pool_inflight",
        )
        for key in zero_fields:
            if self.reconciliation.get(key) != 0:
                reasons.append(
                    f"{self.track}: release reconciliation requires {key}=0"
                )
        return reasons


def apply_acceptance_gates(
    result: TrackResult,
    acceptance: dict[str, Any],
) -> None:
    """Evaluate manifest-preregistered rules against typed measurements."""
    measurements = result.normalized_measurements()
    for label, rule in acceptance.items():
        measurement = measurements.get(label)
        operator = rule["operator"]
        threshold = float(rule["threshold"])
        value = measurement.value if measurement is not None else None
        numeric = (
            not isinstance(value, bool)
            and isinstance(value, (int, float))
        )
        passed = False
        if numeric:
            observed = float(value)
            passed = (
                observed >= threshold
                if operator == ">="
                else observed <= threshold
            )
        result.quality_gates.append(
            {
                "id": f"acceptance:{label}",
                "status": "pass" if passed else "fail",
                "operator": operator,
                "threshold": threshold,
                "observed": value,
                "basis": rule["basis"],
            }
        )
