"""Receipt-derived diagnostic evidence for programmable Watches over live multimodal streams.

This adapter accepts only descriptor-backed persisted matched-Watch receipts. It reconstructs the
runner DTOs and derives evaluator rows from their raw execution records; caller-authored row JSON
is not an input. The resulting report proves internal consistency and byte provenance, not corpus,
reviewer, license, preregistration, provider, or release authority.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import tempfile
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
_MAX_OUTPUT_BYTES = 8 * 1024 * 1024


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


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 71
        and value.startswith("sha256:")
        and all(character in "0123456789abcdef" for character in value[7:])
    )


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
    viability_manifest: Mapping[str, Any]
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

    def __post_init__(self) -> None:
        artifacts = tuple(self.artifacts)
        coverage = tuple(self.coverage)
        rows = tuple(self.derived_rows)
        row_hashes = tuple(self.derived_row_sha256)
        seals = tuple(self.receipt_evidence_seals)
        plan_matrix = tuple(self.watch_plan_matrix)
        _require(bool(artifacts), "measured-evidence report artifacts must be non-empty")
        _require(bool(coverage), "measured-evidence report coverage must be non-empty")
        _require(bool(rows), "measured-evidence report derived rows must be non-empty")
        _require(
            all(isinstance(item, MeasuredEvidenceArtifact) for item in artifacts),
            "measured-evidence artifacts must contain only MeasuredEvidenceArtifact values",
        )
        _require(
            all(
                isinstance(item.role, str)
                and bool(item.role)
                and isinstance(item.artifact_id, str)
                and bool(item.artifact_id)
                for item in artifacts
            ),
            "report artifacts have an invalid role or artifact id",
        )
        _require(
            all(isinstance(item, MeasuredEvidenceCoverage) for item in coverage),
            "measured-evidence coverage must contain only MeasuredEvidenceCoverage values",
        )
        _require(
            all(
                isinstance(item.case_id, str)
                and bool(item.case_id)
                and isinstance(item.operating_point, str)
                and bool(item.operating_point)
                for item in coverage
            ),
            "report coverage has an invalid case or operating-point id",
        )
        _require(
            artifacts == tuple(sorted(artifacts, key=lambda item: (item.role, item.artifact_id))),
            "measured-evidence artifacts must be in canonical role/artifact order",
        )
        _require(
            coverage == tuple(sorted(coverage, key=lambda item: (item.case_id, item.operating_point))),
            "measured-evidence coverage must be in canonical case/operating-point order",
        )
        known_roles = {
            "viability_manifest",
            "matched_watch_receipt",
            "corpus_report",
            "preregistration",
        }
        roles: dict[str, list[MeasuredEvidenceArtifact]] = {}
        artifact_ids: set[str] = set()
        for index, artifact in enumerate(artifacts):
            _require(artifact.role in known_roles, f"artifact {index} has unsupported role")
            _require(
                isinstance(artifact.input_receipt, SafeInputReceipt),
                f"artifact {index} is missing its SafeInputReceipt",
            )
            _require(
                _is_sha256(artifact.content_sha256)
                and _is_sha256(artifact.raw_sha256)
                and isinstance(artifact.byte_count, int)
                and not isinstance(artifact.byte_count, bool)
                and artifact.byte_count >= 0,
                f"artifact {index} has an invalid content or raw identity",
            )
            _require(
                artifact.raw_sha256 == artifact.input_receipt.sha256
                and artifact.byte_count == artifact.input_receipt.byte_count,
                f"artifact {index} raw identity disagrees with its SafeInputReceipt",
            )
            _require(
                artifact.artifact_id == f"{artifact.role}/{artifact.content_sha256[7:]}",
                f"artifact {index} id disagrees with its semantic content hash",
            )
            _require(artifact.artifact_id not in artifact_ids, "duplicate artifact id")
            artifact_ids.add(artifact.artifact_id)
            roles.setdefault(artifact.role, []).append(artifact)
        _require(
            len(roles.get("viability_manifest", ())) == 1,
            "report must contain exactly one viability manifest artifact",
        )
        _require(
            len(roles.get("matched_watch_receipt", ())) == len(coverage),
            "receipt artifact count must equal exact case/operating-point coverage",
        )
        for optional_role in ("corpus_report", "preregistration"):
            _require(
                len(roles.get(optional_role, ())) <= 1,
                f"report may contain at most one {optional_role} artifact",
            )

        manifest = _thaw(self.viability_manifest)
        _require(isinstance(manifest, dict), "report viability_manifest must be an object")
        manifest_artifact = roles["viability_manifest"][0]
        _require(
            _canonical_sha256(manifest) == manifest_artifact.content_sha256,
            "report viability manifest disagrees with its semantic artifact hash",
        )

        receipt_artifacts = {
            item.artifact_id: item for item in roles["matched_watch_receipt"]
        }
        _require(
            len({(item.raw_sha256, item.byte_count) for item in receipt_artifacts.values()})
            == len(receipt_artifacts),
            "report reuses raw receipt bytes",
        )
        _require(
            len({(item.input_receipt.device, item.input_receipt.inode) for item in receipt_artifacts.values()})
            == len(receipt_artifacts),
            "report reuses a receipt file",
        )
        cell_keys = [(item.case_id, item.operating_point) for item in coverage]
        _require(len(cell_keys) == len(set(cell_keys)), "report contains duplicate coverage cells")
        _require(
            len(rows) == 2 * len(coverage)
            and len(row_hashes) == len(rows)
            and len(seals) == len(coverage),
            "report rows, row hashes, coverage, and evidence seals are misaligned",
        )
        _require(
            all(_is_sha256(value) for value in (*row_hashes, *seals)),
            "report row hashes and receipt evidence seals must be canonical sha256 values",
        )
        _require(len(seals) == len(set(seals)), "report reuses a receipt evidence seal")

        case_plans: dict[str, str] = {}
        for cell_index, cell in enumerate(coverage):
            _require(
                isinstance(cell.case_id, str)
                and bool(cell.case_id)
                and isinstance(cell.operating_point, str)
                and bool(cell.operating_point),
                f"coverage {cell_index} has an invalid case or operating-point id",
            )
            _require(
                isinstance(cell.receipt_artifact_id, str) and bool(cell.receipt_artifact_id),
                f"coverage {cell_index} has an invalid receipt artifact id",
            )
            _require(
                all(
                    _is_sha256(value)
                    for value in (
                        cell.receipt_content_sha256,
                        cell.receipt_evidence_sha256,
                        cell.watch_plan_sha256,
                        cell.configuration_sha256,
                        cell.baseline_row_sha256,
                        cell.adaptive_row_sha256,
                    )
                ),
                f"coverage {cell_index} has an invalid hash",
            )
            artifact = receipt_artifacts.get(cell.receipt_artifact_id)
            _require(artifact is not None, f"coverage {cell_index} references an unknown receipt")
            _require(
                artifact.content_sha256 == cell.receipt_content_sha256,
                f"coverage {cell_index} receipt content hash mismatch",
            )
            _require(
                seals[cell_index] == cell.receipt_evidence_sha256,
                f"coverage {cell_index} receipt evidence seal mismatch",
            )
            prior_plan = case_plans.setdefault(cell.case_id, cell.watch_plan_sha256)
            _require(
                prior_plan == cell.watch_plan_sha256,
                f"coverage WatchPlan drift for case {cell.case_id!r}",
            )
            for arm_offset, arm in enumerate(("baseline", "adaptive")):
                row_index = 2 * cell_index + arm_offset
                row = rows[row_index]
                _require(isinstance(row, Mapping), f"derived row {row_index} must be an object")
                row_doc = _thaw(row)
                computed_row_hash = _canonical_sha256(row_doc)
                _require(
                    row_hashes[row_index] == computed_row_hash,
                    f"derived row {row_index} hash mismatch",
                )
                _require(
                    row_doc.get("case_id") == cell.case_id
                    and row_doc.get("operating_point") == cell.operating_point
                    and row_doc.get("arm") == arm
                    and row_doc.get("configuration_sha256") == cell.configuration_sha256,
                    f"derived row {row_index} does not match its coverage cell",
                )
                expected_hash = (
                    cell.baseline_row_sha256 if arm == "baseline" else cell.adaptive_row_sha256
                )
                _require(
                    expected_hash == computed_row_hash,
                    f"coverage row hash mismatch for {cell.case_id!r}/{cell.operating_point!r}/{arm}",
                )
        _require(
            plan_matrix == tuple(sorted(case_plans.items())),
            "report WatchPlan matrix disagrees with coverage",
        )
        try:
            recomputed_frontier = evaluate_viability_frontier(
                manifest,
                [_thaw(row) for row in rows],
            )
        except ViabilityInputError as exc:
            raise MeasuredEvidenceInputError(
                f"report rows/manifest cannot reconstruct the diagnostic frontier: {exc}"
            ) from exc
        _require(
            isinstance(self.frontier, ViabilityFrontierReport)
            and self.frontier.to_dict() == recomputed_frontier.to_dict(),
            "report diagnostic frontier disagrees with its manifest-derived rows",
        )
        _require(not self.frontier.launch_qualified, "report frontier must remain launch-unqualified")
        _require(
            isinstance(self.build_provenance, Mapping)
            and set(self.build_provenance)
            == {"package_version", "git_commit", "contract_version", "build_timestamp"}
            and all(
                isinstance(value, str) and bool(value)
                for value in self.build_provenance.values()
            ),
            "report build provenance must have the exact persisted receipt shape",
        )
        judge_keys = set(self.judge_identity) if isinstance(self.judge_identity, Mapping) else set()
        _require(
            isinstance(self.judge_identity, Mapping)
            and judge_keys
            in (
                {"adapter", "model", "prompt_sha256"},
                {"adapter", "model", "prompt_sha256", "judge_mode"},
            )
            and isinstance(self.judge_identity.get("adapter"), str)
            and bool(self.judge_identity.get("adapter"))
            and isinstance(self.judge_identity.get("model"), str)
            and bool(self.judge_identity.get("model"))
            and _is_sha256(self.judge_identity.get("prompt_sha256")),
            "report judge identity must have the exact persisted receipt shape",
        )
        if "judge_mode" in judge_keys:
            _require(
                self.judge_identity.get("judge_mode") == "offline_oracle",
                "report v2 judge identity must declare judge_mode='offline_oracle'",
            )
        object.__setattr__(self, "artifacts", artifacts)
        object.__setattr__(self, "viability_manifest", _freeze(manifest))
        object.__setattr__(self, "coverage", coverage)
        object.__setattr__(self, "derived_rows", tuple(_freeze(_thaw(row)) for row in rows))
        object.__setattr__(self, "derived_row_sha256", row_hashes)
        object.__setattr__(self, "build_provenance", _freeze(_thaw(self.build_provenance)))
        object.__setattr__(self, "judge_identity", _freeze(_thaw(self.judge_identity)))
        object.__setattr__(self, "watch_plan_matrix", plan_matrix)
        object.__setattr__(self, "receipt_evidence_seals", seals)

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

    def _unchecked_content_dict(self) -> dict[str, Any]:
        return {
            "schema": _SCHEMA,
            "method": self.method,
            "artifacts": [item.content_dict() for item in self.artifacts],
            "viability_manifest": _thaw(self.viability_manifest),
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

    def _unchecked_evidence_dict(self) -> dict[str, Any]:
        result = self._unchecked_content_dict()
        result["artifacts"] = [item.evidence_dict() for item in self.artifacts]
        result["coverage"] = [item.evidence_dict() for item in self.coverage]
        result["receipt_evidence_seals"] = list(self.receipt_evidence_seals)
        return result

    def _validated_rederived(self) -> MeasuredEvidenceBundleReport:
        return _rederive_measured_evidence_bundle(self)

    def content_dict(self) -> dict[str, Any]:
        """Return semantic content only after revalidating every descriptor-backed input."""

        return self._validated_rederived()._unchecked_content_dict()

    def evidence_dict(self) -> dict[str, Any]:
        """Return exact evidence only after revalidating every descriptor-backed input."""

        return self._validated_rederived()._unchecked_evidence_dict()

    def _unchecked_content_sha256(self) -> str:
        return _canonical_sha256(self._unchecked_content_dict())

    def _unchecked_evidence_sha256(self) -> str:
        return _canonical_sha256(self._unchecked_evidence_dict())

    @property
    def content_sha256(self) -> str:
        """Path- and run-time-neutral canonical hash of the bundle's semantic content."""

        return self._validated_rederived()._unchecked_content_sha256()

    @property
    def evidence_sha256(self) -> str:
        """Exact hash additionally binding every descriptor-backed local input snapshot."""

        return self._validated_rederived()._unchecked_evidence_sha256()

    def _unchecked_to_dict(self) -> dict[str, Any]:
        result = self._unchecked_evidence_dict()
        result["content_sha256"] = self._unchecked_content_sha256()
        result["evidence_sha256"] = self._unchecked_evidence_sha256()
        return result

    def to_dict(self) -> dict[str, Any]:
        return self._validated_rederived()._unchecked_to_dict()

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
        loaded = load_json_file(_stable_input_path(path))
    except SafeJSONInputError as exc:
        raise MeasuredEvidenceInputError(f"unsafe {role} artifact: {exc}") from exc
    return _artifact(role, _canonical_sha256(loaded.value), loaded.receipt)


def _stable_input_path(path: str | Path) -> str:
    """Capture a cwd-independent display path while retaining canonical descriptor identity."""

    try:
        raw = os.fspath(path)
        _require(isinstance(raw, str), "input path must resolve to text")
        return os.path.abspath(raw)
    except MeasuredEvidenceInputError:
        raise
    except Exception as exc:
        raise MeasuredEvidenceInputError(f"invalid evidence input path: {exc}") from exc


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


def _receipt_within_case_fingerprint(loaded: LoadedMatchedWatchReceipt) -> str:
    """Bind replay/timeline/window geometry across operating points for one declared case."""

    receipt = loaded.receipt
    geometry = sorted(
        (
            float(event["eligible_at_s"]),
            float(event["window_start_s"]),
            float(event["window_end_s"]),
        )
        for event in receipt.event_windows
    )
    return _canonical_sha256(
        {
            "archetype": receipt.manifest["archetype"],
            "media_sha256": receipt.media_input_receipt["sha256"],
            "media_byte_count": receipt.media_input_receipt["byte_count"],
            "replay": {
                "fps": receipt.manifest["replay"]["fps"],
                "timebase": receipt.manifest["replay"]["timebase"],
                "source_id": receipt.manifest["replay"]["source_id"],
            },
            "candidate_timeline": list(receipt.baseline.candidate_timestamps),
            "event_boundary_geometry": geometry,
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
        manifest_loaded = load_json_file(_stable_input_path(viability_manifest_path))
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
            loaded = load_matched_watch_receipt(_stable_input_path(path))
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
    case_operating_point_fingerprints: dict[str, str] = {}
    media_artifact_owners: dict[tuple[str, str, int], str] = {}
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
        source_fingerprint = _receipt_within_case_fingerprint(loaded)
        prior_source_fingerprint = case_operating_point_fingerprints.setdefault(
            case_id,
            source_fingerprint,
        )
        _require(
            prior_source_fingerprint == source_fingerprint,
            f"case source/timeline/window geometry drift for {case_id!r}",
        )
        media_key = (
            str(receipt.manifest["archetype"]),
            str(receipt.media_input_receipt["sha256"]),
            int(receipt.media_input_receipt["byte_count"]),
        )
        source_owner = media_artifact_owners.setdefault(media_key, case_id)
        _require(
            source_owner == case_id,
            "duplicate underlying case evidence: media artifact reused across case ids "
            f"{source_owner!r} and {case_id!r}",
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
        viability_manifest=_freeze(manifest),
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


def _rederive_measured_evidence_bundle(
    report: MeasuredEvidenceBundleReport,
) -> MeasuredEvidenceBundleReport:
    """Reconstruct a report from its recorded inputs and require an exact DTO match."""

    try:
        artifacts_by_role: dict[str, list[MeasuredEvidenceArtifact]] = {}
        artifacts_by_id = {item.artifact_id: item for item in report.artifacts}
        for artifact in report.artifacts:
            artifacts_by_role.setdefault(artifact.role, []).append(artifact)
        manifest_artifact = artifacts_by_role["viability_manifest"][0]
        receipt_paths = [
            artifacts_by_id[cell.receipt_artifact_id].input_receipt.path for cell in report.coverage
        ]
        optional_paths = {
            role: (
                artifacts_by_role[role][0].input_receipt.path
                if artifacts_by_role.get(role)
                else None
            )
            for role in ("corpus_report", "preregistration")
        }
    except (AttributeError, IndexError, KeyError, TypeError) as exc:
        raise MeasuredEvidenceInputError(
            f"measured-evidence report cannot identify its descriptor-backed inputs: {exc}"
        ) from exc
    rederived = derive_measured_evidence_bundle(
        viability_manifest_path=manifest_artifact.input_receipt.path,
        receipt_paths=receipt_paths,
        corpus_report_path=optional_paths["corpus_report"],
        preregistration_path=optional_paths["preregistration"],
    )
    _require(
        rederived._unchecked_content_dict() == report._unchecked_content_dict()
        and rederived._unchecked_evidence_dict() == report._unchecked_evidence_dict(),
        "measured-evidence report no longer exactly matches its descriptor-backed inputs",
    )
    return rederived


def write_measured_evidence_bundle(
    report: MeasuredEvidenceBundleReport,
    path: str | Path,
) -> None:
    """Write one adapter receipt with exclusive creation; existing evidence is never replaced."""

    _require(
        isinstance(report, MeasuredEvidenceBundleReport),
        "report must be a MeasuredEvidenceBundleReport",
    )
    rederived = _rederive_measured_evidence_bundle(report)
    payload = (json.dumps(rederived._unchecked_to_dict(), indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    _require(
        len(payload) <= _MAX_OUTPUT_BYTES,
        f"measured-evidence output exceeds the {_MAX_OUTPUT_BYTES}-byte limit",
    )
    try:
        destination = os.path.abspath(os.fspath(path))
    except (OSError, TypeError, ValueError) as exc:
        raise MeasuredEvidenceInputError(f"invalid measured-evidence output path: {exc}") from exc
    parent = os.path.dirname(destination)
    name = os.path.basename(destination)
    _require(bool(name), "measured-evidence output path must name a file")
    try:
        existing = os.lstat(destination)
    except FileNotFoundError:
        existing = None
    except OSError as exc:
        raise MeasuredEvidenceInputError(f"cannot inspect measured-evidence output: {exc}") from exc
    if existing is not None:
        if stat.S_ISLNK(existing.st_mode):
            _fail("measured-evidence output target is a symlink")
        if not stat.S_ISREG(existing.st_mode):
            _fail("measured-evidence output target is not a regular file")
        _fail("measured-evidence output already exists; append-only receipts cannot be overwritten")

    temporary_path: str | None = None
    fd: int | None = None
    published = False
    temporary_identity: tuple[int, int] | None = None
    try:
        fd, temporary_path = tempfile.mkstemp(
            prefix=f".{name}.",
            suffix=".tmp",
            dir=parent,
        )
        temporary_info = os.fstat(fd)
        temporary_identity = (temporary_info.st_dev, temporary_info.st_ino)
        if not stat.S_ISREG(temporary_info.st_mode) or temporary_info.st_nlink != 1:
            raise OSError("temporary measured-evidence output is not a single-link regular file")
        offset = 0
        while offset < len(payload):
            written = os.write(fd, payload[offset:])
            if written <= 0:
                raise OSError("short write while persisting measured-evidence output")
            offset += written
        os.fsync(fd)
        os.close(fd)
        fd = None
        try:
            os.link(temporary_path, destination, follow_symlinks=False)
        except FileExistsError as exc:
            raise MeasuredEvidenceInputError(
                "measured-evidence output already exists; append-only receipts cannot be overwritten"
            ) from exc
        published = True
        linked_info = os.lstat(destination)
        if (linked_info.st_dev, linked_info.st_ino) != temporary_identity:
            raise OSError("published measured-evidence output identity changed during creation")
        os.unlink(temporary_path)
        temporary_path = None
        final_info = os.lstat(destination)
        if (
            (final_info.st_dev, final_info.st_ino) != temporary_identity
            or not stat.S_ISREG(final_info.st_mode)
            or final_info.st_nlink != 1
        ):
            raise OSError("published measured-evidence output is not a single-link regular file")
        directory_flags = os.O_RDONLY | int(getattr(os, "O_DIRECTORY", 0))
        directory_fd = os.open(parent, directory_flags)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except MeasuredEvidenceInputError:
        raise
    except (OSError, TypeError, ValueError) as exc:
        if published and temporary_identity is not None:
            try:
                destination_info = os.lstat(destination)
                if (destination_info.st_dev, destination_info.st_ino) == temporary_identity:
                    os.unlink(destination)
            except OSError:
                pass
            published = False
        raise MeasuredEvidenceInputError(
            f"cannot atomically create measured-evidence output: {exc}"
        ) from exc
    finally:
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass
        if temporary_path is not None:
            try:
                os.unlink(temporary_path)
            except OSError:
                pass
