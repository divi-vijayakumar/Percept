"""Receipt-derived diagnostic evidence for programmable Watches over live multimodal streams.

This adapter accepts only descriptor-backed persisted matched-Watch receipts. It reconstructs the
runner DTOs and derives evaluator rows from their raw execution records; caller-authored row JSON
is not an input. The resulting report proves internal consistency and byte provenance, not corpus,
reviewer, license, preregistration, provider, or release authority.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, ClassVar, Mapping, Sequence

from percept._safe_json import SafeInputReceipt, SafeJSONInputError, load_json_file
from percept.eval.matched_watch import (
    LoadedMatchedWatchReceipt,
    MatchedWatchInputError,
    load_matched_watch_receipt,
)
from percept.eval.viability_frontier import (
    REQUIRED_MEASURED_ADAPTER,
    ViabilityFrontierReport,
    ViabilityInputError,
    evaluate_viability_frontier,
)

__all__ = [
    "MeasuredEvidenceInputError",
    "MeasuredEvidenceArtifact",
    "MeasuredEvidenceCoverage",
    "MeasuredEvidenceBundleReport",
    "derive_measured_evidence_bundle",
    "write_measured_evidence_bundle",
]

_SCHEMA = REQUIRED_MEASURED_ADAPTER
_METHOD = "validated matched-Watch receipt derivation (diagnostic)"
_AUTHORITY_BLOCKERS = (
    "external corpus/source/split authority is unavailable",
    "independent label-review authority is unavailable",
    "media license and provider-transmission-rights authority is unavailable",
    "preregistration authority is unavailable",
    "provider execution and usage attestation is unavailable",
    "scripted/admission evidence is diagnostic only",
)


class MeasuredEvidenceInputError(ValueError):
    """Persisted execution evidence is unsafe, incomplete, duplicated, or drifted."""


def _fail(message: str) -> None:
    raise MeasuredEvidenceInputError(message)


def _require(condition: bool, message: str) -> None:
    if not condition:
        _fail(message)


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return f"sha256:{hashlib.sha256(payload.encode('utf-8')).hexdigest()}"


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


def _safe_receipt_dict(receipt: SafeInputReceipt) -> dict[str, Any]:
    return {
        "path": receipt.path,
        "canonical_path": receipt.canonical_path,
        "byte_count": receipt.byte_count,
        "sha256": receipt.sha256,
        "device": receipt.device,
        "inode": receipt.inode,
        "modified_time_ns": receipt.modified_time_ns,
    }


@dataclass(frozen=True)
class MeasuredEvidenceArtifact:
    """Content and descriptor identity for one safely snapshotted input artifact."""

    role: str
    artifact_id: str
    content_sha256: str
    raw_sha256: str
    byte_count: int
    input_receipt: SafeInputReceipt

    def content_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "artifact_id": self.artifact_id,
            "content_sha256": self.content_sha256,
        }

    def evidence_dict(self) -> dict[str, Any]:
        result = self.content_dict()
        result["raw_sha256"] = self.raw_sha256
        result["byte_count"] = self.byte_count
        result["input_receipt"] = _safe_receipt_dict(self.input_receipt)
        return result


@dataclass(frozen=True)
class MeasuredEvidenceCoverage:
    """One exact case × operating-point receipt cell."""

    case_id: str
    operating_point: str
    receipt_artifact_id: str
    receipt_content_sha256: str
    receipt_evidence_sha256: str
    watch_plan_sha256: str
    configuration_sha256: str
    baseline_row_sha256: str
    adaptive_row_sha256: str

    def content_dict(self) -> dict[str, str]:
        return {
            "case_id": self.case_id,
            "operating_point": self.operating_point,
            "receipt_artifact_id": self.receipt_artifact_id,
            "receipt_content_sha256": self.receipt_content_sha256,
            "watch_plan_sha256": self.watch_plan_sha256,
            "configuration_sha256": self.configuration_sha256,
            "baseline_row_sha256": self.baseline_row_sha256,
            "adaptive_row_sha256": self.adaptive_row_sha256,
        }

    def evidence_dict(self) -> dict[str, str]:
        result = self.content_dict()
        result["receipt_evidence_sha256"] = self.receipt_evidence_sha256
        return result

    def to_dict(self) -> dict[str, str]:
        return self.evidence_dict()


@dataclass(frozen=True)
class MeasuredEvidenceBundleReport:
    """Append-only adapter receipt for strictly reconstructed diagnostic execution evidence."""

    artifacts: tuple[MeasuredEvidenceArtifact, ...]
    coverage: tuple[MeasuredEvidenceCoverage, ...]
    derived_rows: tuple[Mapping[str, Any], ...]
    derived_row_sha256: tuple[str, ...]
    build_provenance: Mapping[str, str]
    judge_identity: Mapping[str, str]
    watch_plan_matrix: tuple[tuple[str, str], ...]
    receipt_evidence_seals: tuple[str, ...]
    frontier: ViabilityFrontierReport
    _METHOD: ClassVar[str] = _METHOD
    _AUTHORITY_BLOCKERS: ClassVar[tuple[str, ...]] = _AUTHORITY_BLOCKERS

    @property
    def method(self) -> str:
        return self._METHOD

    @property
    def authority_verified(self) -> bool:
        return False

    @property
    def authority_blockers(self) -> tuple[str, ...]:
        return self._AUTHORITY_BLOCKERS

    @property
    def launch_qualified(self) -> bool:
        return False

    @property
    def rows(self) -> tuple[Mapping[str, Any], ...]:
        return self.derived_rows

    @property
    def blockers(self) -> tuple[str, ...]:
        return self.authority_blockers

    @property
    def diagnostic_frontier(self) -> ViabilityFrontierReport:
        return self.frontier

    @property
    def frontier_report(self) -> ViabilityFrontierReport:
        return self.frontier

    def content_dict(self) -> dict[str, Any]:
        return {
            "schema": _SCHEMA,
            "method": self.method,
            "artifacts": [item.content_dict() for item in self.artifacts],
            "coverage": [item.content_dict() for item in self.coverage],
            "derived_rows": [_thaw(item) for item in self.derived_rows],
            "derived_row_sha256": list(self.derived_row_sha256),
            "build_provenance": _thaw(self.build_provenance),
            "judge_identity": _thaw(self.judge_identity),
            "watch_plan_matrix": [
                {"case_id": case_id, "watch_plan_sha256": plan_sha256}
                for case_id, plan_sha256 in self.watch_plan_matrix
            ],
            "diagnostic_frontier": self.frontier.to_dict(),
            "authority_verified": False,
            "authority_blockers": list(self.authority_blockers),
            "launch_qualified": False,
            "qualification_reason": "; ".join(self.authority_blockers),
        }

    def evidence_dict(self) -> dict[str, Any]:
        result = self.content_dict()
        result["artifacts"] = [item.evidence_dict() for item in self.artifacts]
        result["coverage"] = [item.evidence_dict() for item in self.coverage]
        result["receipt_evidence_seals"] = list(self.receipt_evidence_seals)
        return result

    @property
    def content_sha256(self) -> str:
        """Path- and run-time-neutral canonical hash of the bundle's semantic content."""

        return _canonical_sha256(self.content_dict())

    @property
    def evidence_sha256(self) -> str:
        """Exact hash additionally binding every descriptor-backed local input snapshot."""

        return _canonical_sha256(self.evidence_dict())

    def to_dict(self) -> dict[str, Any]:
        result = self.evidence_dict()
        result["content_sha256"] = self.content_sha256
        result["evidence_sha256"] = self.evidence_sha256
        return result

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)


def _artifact(
    role: str,
    content_sha256: str,
    input_receipt: SafeInputReceipt,
) -> MeasuredEvidenceArtifact:
    artifact_id = f"{role}/{content_sha256[7:]}"
    return MeasuredEvidenceArtifact(
        role=role,
        artifact_id=artifact_id,
        content_sha256=content_sha256,
        raw_sha256=input_receipt.sha256,
        byte_count=input_receipt.byte_count,
        input_receipt=input_receipt,
    )


def _load_opaque_artifact(path: str | Path, role: str) -> MeasuredEvidenceArtifact:
    try:
        loaded = load_json_file(path)
    except SafeJSONInputError as exc:
        raise MeasuredEvidenceInputError(f"unsafe {role} artifact: {exc}") from exc
    return _artifact(role, _canonical_sha256(loaded.value), loaded.receipt)


def _receipt_case_signature(loaded: LoadedMatchedWatchReceipt) -> str:
    receipt = loaded.receipt
    return _canonical_sha256(
        {
            "archetype": receipt.manifest["archetype"],
            "split": receipt.manifest["split"],
            "media": {
                "sha256": receipt.media_input_receipt["sha256"],
                "byte_count": receipt.media_input_receipt["byte_count"],
            },
            "replay": _thaw(receipt.manifest["replay"]),
            "events": _thaw(receipt.event_windows),
            "watch_plan_sha256": receipt.watch_plan_sha256,
        }
    )


def _baseline_signature(loaded: LoadedMatchedWatchReceipt) -> str:
    baseline = loaded.receipt.hash_content_dict()["baseline"]
    return _canonical_sha256(baseline)


def derive_measured_evidence_bundle(
    *,
    viability_manifest_path: str | Path,
    receipt_paths: Sequence[str | Path],
    corpus_report_path: str | Path | None = None,
    preregistration_path: str | Path | None = None,
) -> MeasuredEvidenceBundleReport:
    """Derive a fail-closed diagnostic report exclusively from validated persisted receipts."""

    _require(
        isinstance(receipt_paths, Sequence)
        and not isinstance(receipt_paths, (str, bytes, bytearray))
        and bool(receipt_paths),
        "receipt_paths must be a non-empty sequence of paths",
    )
    try:
        manifest_loaded = load_json_file(viability_manifest_path)
    except SafeJSONInputError as exc:
        raise MeasuredEvidenceInputError(f"unsafe viability manifest: {exc}") from exc
    manifest = manifest_loaded.value
    manifest_content_sha256 = _canonical_sha256(manifest)
    artifacts: list[MeasuredEvidenceArtifact] = [
        _artifact("viability_manifest", manifest_content_sha256, manifest_loaded.receipt)
    ]
    if corpus_report_path is not None:
        artifacts.append(_load_opaque_artifact(corpus_report_path, "corpus_report"))
    if preregistration_path is not None:
        artifacts.append(_load_opaque_artifact(preregistration_path, "preregistration"))

    loaded_receipts: list[LoadedMatchedWatchReceipt] = []
    for index, path in enumerate(receipt_paths):
        try:
            loaded = load_matched_watch_receipt(path)
        except MatchedWatchInputError as exc:
            raise MeasuredEvidenceInputError(
                f"invalid matched-Watch receipt at index {index}: {exc}"
            ) from exc
        loaded_receipts.append(loaded)
        artifacts.append(
            _artifact(
                "matched_watch_receipt",
                loaded.receipt.content_sha256,
                loaded.input_receipt,
            )
        )

    cells: dict[tuple[str, str], LoadedMatchedWatchReceipt] = {}
    used_raw: dict[tuple[str, int], tuple[str, str]] = {}
    used_files: dict[tuple[int, int], tuple[str, str]] = {}
    used_content: dict[str, tuple[str, str]] = {}
    used_evidence: dict[str, tuple[str, str]] = {}
    common_build: dict[str, Any] | None = None
    common_judge: dict[str, Any] | None = None
    case_signatures: dict[str, str] = {}
    case_watch_plans: dict[str, str] = {}
    baseline_signatures: dict[str, str] = {}

    for loaded in loaded_receipts:
        receipt = loaded.receipt
        case_id = str(receipt.manifest["case_id"])
        point = str(receipt.manifest["adaptive"]["operating_point"])
        cell = (case_id, point)
        _require(cell not in cells, f"duplicate receipt cell {cell!r}")
        cells[cell] = loaded
        raw_key = (loaded.input_receipt.sha256, loaded.input_receipt.byte_count)
        file_key = (loaded.input_receipt.device, loaded.input_receipt.inode)
        for registry, key, label in (
            (used_raw, raw_key, "raw receipt bytes"),
            (used_files, file_key, "receipt file"),
            (used_content, receipt.content_sha256, "receipt semantic content"),
            (used_evidence, receipt.evidence_sha256, "receipt evidence seal"),
        ):
            prior = registry.get(key)
            _require(prior is None, f"{label} reused for cells {prior!r} and {cell!r}")
            registry[key] = cell

        build = _thaw(receipt.build_provenance)
        judge = _thaw(receipt.judge_identity)
        if common_build is None:
            common_build = build
            common_judge = judge
        else:
            _require(build == common_build, "receipt build provenance drift")
            _require(judge == common_judge, "receipt judge adapter/model/prompt drift")
        prior_plan = case_watch_plans.setdefault(case_id, receipt.watch_plan_sha256)
        _require(
            prior_plan == receipt.watch_plan_sha256,
            f"WatchPlan identity drift across operating points for {case_id!r}",
        )

        case_signature = _receipt_case_signature(loaded)
        prior_case_signature = case_signatures.setdefault(case_id, case_signature)
        _require(
            prior_case_signature == case_signature,
            f"case media/replay/event identity drift for {case_id!r}",
        )
        baseline_signature = _baseline_signature(loaded)
        prior_baseline = baseline_signatures.setdefault(case_id, baseline_signature)
        _require(
            prior_baseline == baseline_signature,
            f"baseline execution drift across operating points for {case_id!r}",
        )

    ordered_cells = sorted(cells.items())
    rows: list[Mapping[str, Any]] = []
    coverage: list[MeasuredEvidenceCoverage] = []
    receipt_artifacts = {
        artifact.content_sha256: artifact
        for artifact in artifacts
        if artifact.role == "matched_watch_receipt"
    }
    for (case_id, point), loaded in ordered_cells:
        baseline, adaptive = loaded.viability_rows
        rows.extend((_thaw(baseline), _thaw(adaptive)))
        receipt = loaded.receipt
        coverage.append(
            MeasuredEvidenceCoverage(
                case_id=case_id,
                operating_point=point,
                receipt_artifact_id=receipt_artifacts[receipt.content_sha256].artifact_id,
                receipt_content_sha256=receipt.content_sha256,
                receipt_evidence_sha256=receipt.evidence_sha256,
                watch_plan_sha256=receipt.watch_plan_sha256,
                configuration_sha256=receipt.adaptive.configuration_sha256,
                baseline_row_sha256=_canonical_sha256(_thaw(baseline)),
                adaptive_row_sha256=_canonical_sha256(_thaw(adaptive)),
            )
        )

    try:
        frontier = evaluate_viability_frontier(manifest, rows)
    except ViabilityInputError as exc:
        raise MeasuredEvidenceInputError(f"receipt-derived frontier is invalid: {exc}") from exc
    _require(not frontier.launch_qualified, "diagnostic frontier unexpectedly qualified launch")
    assert common_build is not None
    assert common_judge is not None
    frozen_rows = tuple(_freeze(_thaw(row)) for row in rows)
    row_hashes = tuple(_canonical_sha256(_thaw(row)) for row in frozen_rows)
    return MeasuredEvidenceBundleReport(
        artifacts=tuple(sorted(artifacts, key=lambda item: (item.role, item.artifact_id))),
        coverage=tuple(coverage),
        derived_rows=frozen_rows,
        derived_row_sha256=row_hashes,
        build_provenance=_freeze(common_build),
        judge_identity=_freeze(common_judge),
        watch_plan_matrix=tuple(sorted(case_watch_plans.items())),
        receipt_evidence_seals=tuple(
            loaded.receipt.evidence_sha256
            for _, loaded in ordered_cells
        ),
        frontier=frontier,
    )


def write_measured_evidence_bundle(
    report: MeasuredEvidenceBundleReport,
    path: str | Path,
) -> None:
    """Write one adapter receipt with exclusive creation; existing evidence is never replaced."""

    _require(
        isinstance(report, MeasuredEvidenceBundleReport),
        "report must be a MeasuredEvidenceBundleReport",
    )
    payload = report.to_json() + "\n"
    try:
        with Path(path).open("x", encoding="utf-8") as stream:
            stream.write(payload)
    except FileExistsError as exc:
        raise MeasuredEvidenceInputError(
            "measured-evidence output already exists; append-only receipts cannot be overwritten"
        ) from exc
    except (OSError, TypeError, ValueError) as exc:
        raise MeasuredEvidenceInputError(f"cannot create measured-evidence output: {exc}") from exc
