"""Independent adversarial oracle for the measured-evidence v1 boundary.

These tests intentionally exercise only the public persisted-receipt and receipt-bundle APIs.
They do not import implementation helpers from ``percept.eval.measured_evidence`` and they build
four genuinely executed case/operating-point receipts rather than copying or relabelling rows.

The authority assertions are deliberately stronger than hash-integrity assertions: an internally
coherent, caller-resealed receipt is allowed to remain internally consistent, but it can never make
external review/license/preregistration/provider authority available or qualify a launch.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import FrozenInstanceError, dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import pytest

from percept.eval.matched_watch import (
    MatchedWatchInputError,
    MatchedWatchReceipt,
    derive_viability_rows,
    load_matched_watch_receipt,
    run_matched_watch,
)
from percept.eval.measured_evidence import (
    MeasuredEvidenceInputError,
    derive_measured_evidence_bundle,
)


_MEASURED_SCHEMA = "percept.eval.viability_frontier.measured_evidence/v1"
_SHA_ZERO = "sha256:" + "0" * 64
_POLICY = {
    "min_candidate_recall": 0.0,
    "min_event_recall": 0.0,
    "min_precision": 0.0,
    "min_call_reduction": 0.0,
    "max_event_abstention_rate": 1.0,
    "max_missing_rate": 1.0,
}


def _sha(payload: bytes) -> str:
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _json_bytes(value: Any, *, trailing_newline: bool = False) -> bytes:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return payload + (b"\n" if trailing_newline else b"")


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


def _verdict(satisfied: bool, evidence: str, *, error: str | None = None) -> dict[str, Any]:
    return {
        "satisfied": satisfied,
        "confidence": 0.97,
        "evidence": evidence,
        "error": error,
    }


def _media_document(
    *,
    case_number: int,
    still: bool = False,
    fail_to_at: int | None = None,
) -> dict[str, Any]:
    # The cases have genuinely different pixels/tokens/evidence.  Across operating points the same
    # case retains exactly the same media and labels, as a matched frontier requires.
    dark = 17 + case_number
    light = 235 - case_number
    # Baseline admission uses a zero threshold but still requires a nonzero observed delta.  Every
    # normal consecutive frame therefore changes by more than the sparse 0.75 threshold.  The
    # zero-admission control retains a small nonzero delta at every step: baseline stays exhaustive
    # while each normalized delta remains far below the adaptive 0.25 threshold.
    fills = (
        (dark, dark + 3, dark + 1, dark + 4, dark + 2, dark + 5)
        if still
        else (dark, light, dark, light, dark, light)
    )
    states = (False, False, True, False, True, True)
    frames = []
    for index, (fill, present) in enumerate(zip(fills, states)):
        frames.append(
            {
                "t": float(index),
                "fill": fill,
                "token": f"independent-case-{case_number}-candidate-{index}",
                "from": _verdict(
                    not present,
                    f"case {case_number}: package absent at candidate {index}",
                ),
                "to": _verdict(
                    present,
                    f"case {case_number}: package visible at candidate {index}",
                    error=("independent scripted timeout" if fail_to_at == index else None),
                ),
            }
        )
    return {
        "schema": "percept.eval.matched_watch.synthetic_video/v1",
        "width": 8,
        "height": 8,
        "frames": frames,
    }


def _events(
    case_id: str,
    case_number: int,
    *,
    authority: str = "reviewed/licensed/preregistered/provider_measured/approved",
) -> list[dict]:
    return [
        {
            "id": f"{case_id}/arrival",
            "eligible_at_s": 2.0,
            "window_start_s": 2.0,
            "window_end_s": 2.0,
            "review_authority": authority,
            "label_digest": _sha(f"case {case_number}: arrival at t=2".encode()),
        },
        {
            "id": f"{case_id}/return",
            "eligible_at_s": 4.0,
            "window_start_s": 4.0,
            "window_end_s": 5.0,
            "review_authority": authority,
            "label_digest": _sha(f"case {case_number}: return at t=4".encode()),
        },
    ]


@dataclass(frozen=True)
class _Cell:
    case_id: str
    point_id: str
    threshold: float
    directory: Path
    manifest_path: Path
    receipt_path: Path
    receipt: MatchedWatchReceipt


@dataclass(frozen=True)
class _ExecutedMatrix:
    root: Path
    manifest_path: Path
    cells: tuple[_Cell, ...]

    @property
    def receipt_paths(self) -> tuple[Path, ...]:
        return tuple(cell.receipt_path for cell in self.cells)

    def cell(self, case_id: str, point_id: str) -> _Cell:
        return next(
            cell
            for cell in self.cells
            if (cell.case_id, cell.point_id) == (case_id, point_id)
        )


def _write_cell(
    directory: Path,
    *,
    case_number: int,
    point_id: str,
    threshold: float,
    still: bool = False,
    fail_to_at: int | None = None,
    case_id: str | None = None,
    authority: str = "reviewed/licensed/preregistered/provider_measured/approved",
    split: str = "licensed-reviewed-preregistered-provider_measured-held-out",
) -> _Cell:
    directory.mkdir(parents=True)
    resolved_case = case_id or f"independent-package-{case_number:02d}"
    media = _media_document(case_number=case_number, still=still, fail_to_at=fail_to_at)
    media_bytes = _json_bytes(media)
    media_path = directory / "media.json"
    media_path.write_bytes(media_bytes)
    manifest = {
        "schema": "percept.eval.matched_watch/v1",
        "case_id": resolved_case,
        "archetype": "physical_scene/package_appeared",
        "split": split,
        "media": {"path": "media.json", "sha256": _sha(media_bytes)},
        "replay": {
            "fps": 4.0,
            "timebase": "media",
            # The shipped package-appearance Watch is scoped to the canonical ``camera`` source.
            # Case independence comes from media/events, not by silently moving the replay to a
            # different source that the compiled Watch correctly ignores.
            "source_id": "camera",
        },
        "watch": {
            "template_ref": "facility_observation@1/package_appeared",
            # A frontier operating point has one exact executable configuration across cases.
            # Cases vary in media and event labels while exercising the same Watch parameters.
            "parameters": {"zone_name": "receiving-zone"},
            "domain_pack": "facility_observation@1",
        },
        "judge": {
            "adapter": "scripted_transition/v1",
            "model": "independent-scripted-contract",
            "prompt_sha256": _sha(b"independent fixed package transition prompt"),
        },
        "adaptive": {
            "operating_point": point_id,
            "salience_stat": "block_max",
            "threshold": threshold,
        },
        "events": _events(resolved_case, case_number, authority=authority),
        "evidence_classification": "scripted_contract",
    }
    manifest_path = directory / "runner-manifest.json"
    manifest_path.write_bytes(_json_bytes(manifest, trailing_newline=True))
    receipt = run_matched_watch(manifest_path)
    receipt_path = directory / "runner-receipt.json"
    receipt_path.write_bytes(receipt.to_json(indent=2).encode("utf-8") + b"\n")
    return _Cell(
        resolved_case,
        point_id,
        threshold,
        directory,
        manifest_path,
        receipt_path,
        receipt,
    )


def _configuration(cell: _Cell) -> dict[str, Any]:
    arm = cell.receipt.adaptive.to_dict()
    return {
        "effective": arm["resolved_watch"]["effective"],
        "source_profile": arm["source_profile"],
        "compiled_binding": arm["compiled_binding"],
    }


def _write_viability_manifest(root: Path, cells: Sequence[_Cell]) -> Path:
    cases: list[dict[str, Any]] = []
    case_ids: set[str] = set()
    for cell in cells:
        if cell.case_id in case_ids:
            continue
        case_ids.add(cell.case_id)
        cases.append(
            {
                "id": cell.case_id,
                "candidate_windows_total": 6,
                "events": [
                    {"id": f"{cell.case_id}/arrival", "candidate_windows": 1},
                    {"id": f"{cell.case_id}/return", "candidate_windows": 2},
                ],
            }
        )
    points: list[dict[str, Any]] = []
    point_ids: set[str] = set()
    for cell in cells:
        if cell.point_id in point_ids:
            continue
        point_ids.add(cell.point_id)
        points.append(
            {
                "id": cell.point_id,
                "configuration": _configuration(cell),
                "configuration_sha256": cell.receipt.adaptive.configuration_sha256,
            }
        )
    manifest = {
        "schema_version": "3",
        "track": "viability_frontier",
        "scope": "programmable Watches over live multimodal streams",
        "evidence_kind": "synthetic",
        "preregistered": True,
        "archetypes": [
            {
                "id": "physical_scene/package_appeared",
                "provider_call_contract": {
                    "completed_dispatch_calls": 2,
                    "missing_dispatch_min_calls": 1,
                    "missing_dispatch_max_calls": 2,
                },
                "cases": cases,
                "operating_points": points,
                "policy": _POLICY,
            }
        ],
    }
    path = root / "viability-manifest.json"
    path.write_bytes(_json_bytes(manifest, trailing_newline=True))
    return path


@pytest.fixture(scope="module")
def executed_matrix(tmp_path_factory: pytest.TempPathFactory) -> _ExecutedMatrix:
    root = tmp_path_factory.mktemp("measured-evidence-independent-matrix")
    cells: list[_Cell] = []
    for case_number in (1, 2):
        for point_id, threshold in (("observe", 0.25), ("sparse", 0.75)):
            cells.append(
                _write_cell(
                    root / f"case-{case_number}-{point_id}",
                    case_number=case_number,
                    point_id=point_id,
                    threshold=threshold,
                )
            )
    manifest_path = _write_viability_manifest(root, cells)
    return _ExecutedMatrix(root, manifest_path, tuple(cells))


def _derive_bundle(
    matrix: _ExecutedMatrix,
    *,
    receipt_paths: Sequence[Path] | None = None,
    viability_manifest_path: Path | None = None,
):
    return derive_measured_evidence_bundle(
        viability_manifest_path=viability_manifest_path or matrix.manifest_path,
        receipt_paths=tuple(receipt_paths or matrix.receipt_paths),
    )


def _rows(report: Any) -> tuple[Any, ...]:
    rows = getattr(report, "derived_rows", None)
    if rows is None:
        rows = report.rows
    return tuple(rows)


def _frontier(report: Any) -> Any:
    frontier = getattr(report, "frontier_report", None)
    if frontier is not None:
        return frontier
    evaluator = getattr(report, "evaluate_frontier", None)
    assert callable(evaluator), "report must expose its diagnostic frontier or an exact evaluator"
    return evaluator()


def _artifacts(report: Any) -> tuple[Any, ...]:
    artifacts = getattr(report, "artifacts", None)
    if artifacts is None:
        artifacts = report.input_artifacts
    return tuple(artifacts)


def _field(value: Any, name: str) -> Any:
    return value[name] if isinstance(value, Mapping) else getattr(value, name)


def _write_receipt_doc(path: Path, document: Any, *, raw: bytes | None = None) -> Path:
    path.write_bytes(raw if raw is not None else _json_bytes(document, trailing_newline=True))
    return path


def _find_descriptor(value: Any, sha256: str) -> Mapping[str, Any] | None:
    if isinstance(value, Mapping):
        if value.get("sha256") == sha256 and "byte_count" in value:
            return value
        for item in value.values():
            found = _find_descriptor(item, sha256)
            if found is not None:
                return found
    elif isinstance(value, (list, tuple)):
        for item in value:
            found = _find_descriptor(item, sha256)
            if found is not None:
                return found
    return None


def _set_path(document: dict[str, Any], path: Sequence[Any], value: Any) -> None:
    target: Any = document
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value


def _del_path(document: dict[str, Any], path: Sequence[Any]) -> None:
    target: Any = document
    for key in path[:-1]:
        target = target[key]
    del target[path[-1]]


def test_persisted_receipt_roundtrip_rebuilds_identical_rows_hashes_and_raw_receipt(
    executed_matrix: _ExecutedMatrix,
):
    cell = executed_matrix.cells[0]
    raw = cell.receipt_path.read_bytes()
    loaded = load_matched_watch_receipt(cell.receipt_path)

    assert loaded.receipt is not cell.receipt
    assert loaded.receipt.to_dict() == cell.receipt.to_dict()
    assert loaded.receipt.content_sha256 == cell.receipt.content_sha256
    assert loaded.receipt.evidence_sha256 == cell.receipt.evidence_sha256
    assert derive_viability_rows(loaded.receipt) == derive_viability_rows(cell.receipt)
    assert loaded.input_receipt.byte_count == len(raw)
    assert loaded.input_receipt.sha256 == _sha(raw)
    assert loaded.input_receipt.canonical_path == str(cell.receipt_path.resolve())


def test_loader_result_is_deeply_immutable(executed_matrix: _ExecutedMatrix):
    loaded = load_matched_watch_receipt(executed_matrix.cells[0].receipt_path)

    with pytest.raises(FrozenInstanceError):
        loaded.receipt = loaded.receipt  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        loaded.receipt.baseline = loaded.receipt.baseline  # type: ignore[misc]
    with pytest.raises(TypeError):
        loaded.receipt.manifest["case_id"] = "mutated"
    with pytest.raises(TypeError):
        loaded.receipt.baseline.runtime_stats["frames"] = 0
    assert isinstance(loaded.receipt.baseline.candidates, tuple)
    assert isinstance(loaded.receipt.baseline.candidates[0].provider_calls, tuple)
    assert isinstance(loaded.receipt.event_windows, tuple)


@pytest.mark.parametrize(
    ("mutation", "path", "replacement"),
    [
        ("missing", ("schema",), None),
        ("missing", ("manifest", "case_id"), None),
        ("missing", ("manifest", "replay", "source_id"), None),
        ("missing", ("event_windows", 0, "label_digest"), None),
        ("missing", ("baseline", "runtime_stats", "judge_calls"), None),
        ("missing", ("baseline", "candidates", 0, "edge_decision"), None),
        ("missing", ("baseline", "candidates", 0, "provider_calls", 0, "status"), None),
        ("extra", ("manifest",), None),
        ("extra", ("baseline", "runtime_stats"), None),
        ("extra", ("adaptive", "candidates", 0), None),
        ("wrong", ("manifest", "case_id"), 7),
        ("wrong", ("manifest_input_receipt", "byte_count"), True),
        ("wrong", ("event_windows", 0, "window_start_s"), "2.0"),
        ("wrong", ("baseline", "runtime_dispatches"), 1.0),
        ("wrong", ("baseline", "candidates"), {}),
        ("wrong", ("baseline", "candidates", 0, "admitted"), 1),
        ("wrong", ("baseline", "candidates", 0, "provider_calls", 0, "satisfied"), 1),
    ],
)
def test_loader_rejects_missing_extra_and_wrong_typed_nested_receipt_fields(
    executed_matrix: _ExecutedMatrix,
    tmp_path: Path,
    mutation: str,
    path: tuple[Any, ...],
    replacement: Any,
):
    document = json.loads(executed_matrix.cells[0].receipt_path.read_text(encoding="utf-8"))
    if mutation == "missing":
        _del_path(document, path)
    elif mutation == "wrong":
        _set_path(document, path, replacement)
    else:
        target: Any = document
        for key in path:
            target = target[key]
        target["oracle_unsupported_field"] = "must fail"

    with pytest.raises(MatchedWatchInputError):
        load_matched_watch_receipt(_write_receipt_doc(tmp_path / "mutated.json", document))


@pytest.mark.parametrize(
    "raw_mutator",
    [
        lambda raw: raw.replace(b'"schema":', b'"schema":"duplicate","schema":', 1),
        lambda raw: raw.replace(b'"runtime_stats": {', b'"runtime_stats":{"frames":1,"frames":2},'
                            b'"discarded_runtime_stats": {', 1),
        lambda raw: raw.replace(b'"candidate_t": 0.0', b'"candidate_t": NaN', 1),
        lambda raw: raw.replace(b'"confidence": 0.97', b'"confidence": Infinity', 1),
    ],
)
def test_loader_rejects_duplicate_json_keys_and_nonfinite_numbers(
    executed_matrix: _ExecutedMatrix,
    tmp_path: Path,
    raw_mutator,
):
    raw = executed_matrix.cells[0].receipt_path.read_bytes()
    attacked = raw_mutator(raw)
    assert attacked != raw
    path = _write_receipt_doc(tmp_path / "unsafe-json.json", {}, raw=attacked)

    with pytest.raises(MatchedWatchInputError):
        load_matched_watch_receipt(path)


def test_loader_rejects_symlink_and_hardlink_receipt_inputs(
    executed_matrix: _ExecutedMatrix,
    tmp_path: Path,
):
    source = executed_matrix.cells[0].receipt_path
    symlink = tmp_path / "receipt-symlink.json"
    symlink.symlink_to(source)
    hardlink_source = tmp_path / "hardlink-source.json"
    shutil.copyfile(source, hardlink_source)
    hardlink = tmp_path / "receipt-hardlink.json"
    os.link(hardlink_source, hardlink)

    for path in (symlink, hardlink):
        with pytest.raises(MatchedWatchInputError):
            load_matched_watch_receipt(path)


def test_loader_rejects_path_retargeted_after_descriptor_open(
    executed_matrix: _ExecutedMatrix,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    import percept._safe_json as safe_json

    path = tmp_path / "retarget.json"
    raw = executed_matrix.cells[0].receipt_path.read_bytes()
    path.write_bytes(raw)
    displaced = tmp_path / "opened-original.json"
    real_read = safe_json.os.read
    replaced = False

    def retargeting_read(fd: int, size: int) -> bytes:
        nonlocal replaced
        chunk = real_read(fd, size)
        if not replaced:
            replaced = True
            path.rename(displaced)
            path.write_bytes(raw)
        return chunk

    monkeypatch.setattr(safe_json.os, "read", retargeting_read)
    with pytest.raises(MatchedWatchInputError):
        load_matched_watch_receipt(path)


def test_truncation_missing_origin_and_active_build_tamper_fail_closed(
    executed_matrix: _ExecutedMatrix,
    tmp_path: Path,
):
    original = json.loads(executed_matrix.cells[0].receipt_path.read_text(encoding="utf-8"))
    missing_origin = json.loads(json.dumps(original))
    missing_origin["origin_evidence_sha256"] = None
    changed_build = json.loads(json.dumps(original))
    changed_build["build_provenance"]["git_commit"] = "caller-forged-build"

    attacks = (
        (tmp_path / "truncated.json", b'{"schema":"percept.eval.matched_watch.receipt/v1"'),
        (tmp_path / "missing-origin.json", _json_bytes(missing_origin)),
        (tmp_path / "changed-build.json", _json_bytes(changed_build)),
    )
    for path, raw in attacks:
        path.write_bytes(raw)
        with pytest.raises(MatchedWatchInputError):
            load_matched_watch_receipt(path)


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (("run_timestamp",), "2099-12-31T23:59:59+00:00"),
        (("manifest_input_receipt", "canonical_path"), "/caller/retargeted/manifest.json"),
        (("manifest_input_receipt", "device"), 999999999),
        (("manifest_input_receipt", "inode"), 999999999),
        (("manifest_input_receipt", "modified_time_ns"), 999999999),
        (("adaptive", "runtime_stats", "last_wall_t"), 999999999.0),
        (("adaptive", "health", 0, "ts"), 999999999.0),
        (("adaptive", "traces", 0, "trace_id"), "caller-correlation-id"),
    ],
)
def test_loader_recomputes_exact_evidence_and_rejects_volatile_or_provenance_tamper(
    executed_matrix: _ExecutedMatrix,
    tmp_path: Path,
    path: tuple[Any, ...],
    replacement: Any,
):
    document = json.loads(executed_matrix.cells[0].receipt_path.read_text(encoding="utf-8"))
    _set_path(document, path, replacement)

    with pytest.raises(MatchedWatchInputError):
        load_matched_watch_receipt(_write_receipt_doc(tmp_path / "evidence-tamper.json", document))


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (("adaptive", "runtime_dispatches"), 0),
        (("adaptive", "provider_call_attempts"), 0),
        (("adaptive", "provider_call_successes"), 0),
        (("adaptive", "runtime_stats", "fires"), 0),
        (("adaptive", "candidates", 2, "decision"), "silent"),
        (("adaptive", "candidates", 2, "incident_ids"), []),
        (("adaptive", "candidates", 2, "provider_calls", 0, "sequence"), 999),
        (("adaptive", "incidents"), []),
        (("adaptive", "traces"), []),
        (("adaptive", "health"), []),
        (("adaptive", "valid"), False),
        (("comparable",), False),
    ],
)
def test_loader_recomputes_candidates_calls_events_health_and_cached_state(
    executed_matrix: _ExecutedMatrix,
    tmp_path: Path,
    path: tuple[Any, ...],
    replacement: Any,
):
    document = json.loads(executed_matrix.cells[0].receipt_path.read_text(encoding="utf-8"))
    _set_path(document, path, replacement)

    with pytest.raises(MatchedWatchInputError):
        load_matched_watch_receipt(_write_receipt_doc(tmp_path / "recomputed-state.json", document))


def test_bundle_derives_exactly_eight_rows_exclusively_from_four_genuine_receipts(
    executed_matrix: _ExecutedMatrix,
):
    report = _derive_bundle(executed_matrix)
    expected = []
    for cell in executed_matrix.cells:
        expected.extend(derive_viability_rows(cell.receipt))

    assert len(executed_matrix.cells) == 4
    assert len({cell.receipt.evidence_sha256 for cell in executed_matrix.cells}) == 4
    assert len(_rows(report)) == 8
    actual_by_key = {
        (row["case_id"], row["operating_point"], row["arm"]): _thaw(row)
        for row in _rows(report)
    }
    expected_by_key = {
        (row["case_id"], row["operating_point"], row["arm"]): row for row in expected
    }
    assert actual_by_key == expected_by_key
    assert report.to_dict()["schema"] == _MEASURED_SCHEMA


def test_bundle_api_has_no_direct_row_json_ingress(
    executed_matrix: _ExecutedMatrix,
    tmp_path: Path,
):
    row_path = tmp_path / "caller-authored-row.json"
    row_path.write_bytes(_json_bytes(derive_viability_rows(executed_matrix.cells[0].receipt)[0]))

    with pytest.raises((MeasuredEvidenceInputError, MatchedWatchInputError)):
        _derive_bundle(
            executed_matrix,
            receipt_paths=(*executed_matrix.receipt_paths[:-1], row_path),
        )
    with pytest.raises(TypeError):
        derive_measured_evidence_bundle(  # type: ignore[call-arg]
            viability_manifest_path=executed_matrix.manifest_path,
            receipt_paths=executed_matrix.receipt_paths,
            rows=[json.loads(row_path.read_text(encoding="utf-8"))],
        )


def test_bundle_rejects_missing_extra_reused_and_duplicate_cells(
    executed_matrix: _ExecutedMatrix,
    tmp_path: Path,
):
    duplicate_cell = _write_cell(
        tmp_path / "second-genuine-execution-same-cell",
        case_number=1,
        point_id="observe",
        threshold=0.25,
        case_id="independent-package-01",
    )
    extra_cell = _write_cell(
        tmp_path / "genuine-extra-cell",
        case_number=9,
        point_id="observe",
        threshold=0.25,
        case_id="not-in-frozen-manifest",
    )
    attacks = (
        executed_matrix.receipt_paths[:-1],
        (*executed_matrix.receipt_paths, extra_cell.receipt_path),
        (*executed_matrix.receipt_paths, executed_matrix.receipt_paths[0]),
        (*executed_matrix.receipt_paths, duplicate_cell.receipt_path),
    )

    for paths in attacks:
        with pytest.raises(MeasuredEvidenceInputError):
            _derive_bundle(executed_matrix, receipt_paths=paths)


def test_bundle_rejects_case_point_identity_drift_and_manifest_configuration_drift(
    executed_matrix: _ExecutedMatrix,
    tmp_path: Path,
):
    drifted_cell = _write_cell(
        tmp_path / "drifted-point",
        case_number=1,
        point_id="unregistered-point",
        threshold=0.25,
        case_id="independent-package-01",
    )
    paths = tuple(
        drifted_cell.receipt_path if index == 0 else path
        for index, path in enumerate(executed_matrix.receipt_paths)
    )
    with pytest.raises(MeasuredEvidenceInputError):
        _derive_bundle(executed_matrix, receipt_paths=paths)

    manifest = json.loads(executed_matrix.manifest_path.read_text(encoding="utf-8"))
    manifest["archetypes"][0]["operating_points"][0]["configuration_sha256"] = _SHA_ZERO
    drifted_manifest = tmp_path / "drifted-configuration.json"
    drifted_manifest.write_bytes(_json_bytes(manifest))
    with pytest.raises(MeasuredEvidenceInputError):
        _derive_bundle(executed_matrix, viability_manifest_path=drifted_manifest)


def test_bundle_order_is_deterministic_and_content_hash_path_neutral_but_evidence_exact(
    executed_matrix: _ExecutedMatrix,
    tmp_path: Path,
):
    original = _derive_bundle(executed_matrix)
    reversed_order = _derive_bundle(
        executed_matrix,
        receipt_paths=tuple(reversed(executed_matrix.receipt_paths)),
    )
    assert reversed_order.content_sha256 == original.content_sha256
    assert [_thaw(row) for row in _rows(reversed_order)] == [
        _thaw(row) for row in _rows(original)
    ]

    relocated = tmp_path / "relocated-independent-artifacts"
    relocated.mkdir()
    relocated_manifest = relocated / "renamed-viability.json"
    shutil.copyfile(executed_matrix.manifest_path, relocated_manifest)
    relocated_receipts = []
    for index, source in enumerate(reversed(executed_matrix.receipt_paths)):
        destination = relocated / f"opaque-artifact-{index}.json"
        shutil.copyfile(source, destination)
        relocated_receipts.append(destination)
    moved = _derive_bundle(
        executed_matrix,
        viability_manifest_path=relocated_manifest,
        receipt_paths=relocated_receipts,
    )

    assert moved.content_sha256 == original.content_sha256
    assert moved.evidence_sha256 != original.evidence_sha256
    assert [_thaw(row) for row in _rows(moved)] == [_thaw(row) for row in _rows(original)]


def test_bundle_preserves_raw_receipt_bytes_digest_and_size_without_canonical_rewrite(
    executed_matrix: _ExecutedMatrix,
    tmp_path: Path,
):
    source = executed_matrix.cells[0].receipt_path
    raw = source.read_bytes()
    noncanonical = raw.rstrip() + b"\n\n  \n"
    rewritten = tmp_path / "same-json-different-raw-bytes.json"
    rewritten.write_bytes(noncanonical)
    paths = (rewritten, *executed_matrix.receipt_paths[1:])

    report = _derive_bundle(executed_matrix, receipt_paths=paths)
    loaded = load_matched_watch_receipt(rewritten)

    assert loaded.input_receipt.byte_count == len(noncanonical)
    assert loaded.input_receipt.sha256 == _sha(noncanonical)
    assert loaded.input_receipt.sha256 != _sha(_json_bytes(json.loads(noncanonical)))
    payload = report.to_dict()
    descriptor = _find_descriptor(payload, loaded.input_receipt.sha256)
    assert descriptor is not None
    assert descriptor["byte_count"] == len(noncanonical)


def _coherently_reseal_review_claims(
    receipt: MatchedWatchReceipt,
    *,
    authority: str,
    split: str,
) -> MatchedWatchReceipt:
    manifest = _thaw(receipt.manifest)
    manifest["split"] = split
    events = _thaw(receipt.event_windows)
    for index, event in enumerate(events):
        event["review_authority"] = authority
        event["label_digest"] = _sha(
            f"caller-resealed {event['id']} as reviewed/licensed/provider-measured {index}".encode()
        )
    manifest["events"] = events
    forged = replace(
        receipt,
        manifest=manifest,
        event_windows=tuple(events),
        origin_evidence_sha256="",
    )
    return replace(forged, origin_evidence_sha256=forged.evidence_sha256)


@pytest.mark.parametrize(
    "authority",
    [
        "reviewed",
        "licensed",
        "preregistered",
        "provider_measured",
        "reviewed/licensed/preregistered/provider_measured/approved",
    ],
)
def test_coherent_caller_reseal_and_authority_strings_never_remove_blockers_or_unlock_launch(
    executed_matrix: _ExecutedMatrix,
    tmp_path: Path,
    authority: str,
):
    cell = executed_matrix.cells[0]
    forged = _coherently_reseal_review_claims(
        cell.receipt,
        authority=authority,
        split=f"{authority}-held-out",
    )
    authority_hash = hashlib.sha256(authority.encode()).hexdigest()
    forged_path = tmp_path / f"coherently-resealed-{authority_hash}.json"
    forged_path.write_bytes(forged.to_json().encode("utf-8") + b"\n")

    paths = tuple(
        forged_path if path == cell.receipt_path else path
        for path in executed_matrix.receipt_paths
    )
    try:
        # This is the exact authenticity boundary: an implementation may reject a caller-resealed
        # origin outright.  If it can only establish internal consistency, it must still refuse
        # authority and launch below.
        loaded = load_matched_watch_receipt(forged_path)
        assert loaded.receipt.origin_evidence_sha256 == loaded.receipt.evidence_sha256
        derive_viability_rows(loaded.receipt)
        report = _derive_bundle(executed_matrix, receipt_paths=paths)
    except (MatchedWatchInputError, MeasuredEvidenceInputError):
        return
    blockers = " ".join(report.authority_blockers).lower()
    assert report.authority_verified is False
    assert report.launch_qualified is False
    for missing_authority in ("review", "license", "preregistration", "provider"):
        assert missing_authority in blockers


def test_authority_and_launch_are_unconditionally_false_and_evaluator_cannot_unlock(
    executed_matrix: _ExecutedMatrix,
):
    report = _derive_bundle(executed_matrix)
    frontier = _frontier(report)

    assert report.authority_verified is False
    assert report.authority_blockers
    assert report.launch_qualified is False
    assert frontier.policy_cleared is True
    assert frontier.launch_qualified is False
    blockers = " ".join(report.authority_blockers).lower()
    for missing_authority in ("review", "license", "preregistration", "provider"):
        assert missing_authority in blockers


def test_opaque_corpus_and_preregistration_declarations_are_snapshotted_but_have_no_authority(
    executed_matrix: _ExecutedMatrix,
    tmp_path: Path,
):
    corpus_claims = {
        "reviewed": True,
        "license": "approved",
        "provider_measured": "yes",
        "review_authority": "independent-approved",
    }
    preregistration_claims = {
        "preregistered": True,
        "approved": True,
        "authority": "trusted-by-caller",
    }
    corpus_path = tmp_path / "opaque-corpus-report.json"
    preregistration_path = tmp_path / "opaque-preregistration.json"
    corpus_raw = _json_bytes(corpus_claims, trailing_newline=True)
    preregistration_raw = _json_bytes(preregistration_claims, trailing_newline=True)
    corpus_path.write_bytes(corpus_raw)
    preregistration_path.write_bytes(preregistration_raw)

    report = derive_measured_evidence_bundle(
        viability_manifest_path=executed_matrix.manifest_path,
        receipt_paths=executed_matrix.receipt_paths,
        corpus_report_path=corpus_path,
        preregistration_path=preregistration_path,
    )

    assert report.authority_verified is False
    assert report.launch_qualified is False
    blockers = " ".join(report.authority_blockers).lower()
    for missing_authority in ("review", "license", "preregistration", "provider"):
        assert missing_authority in blockers
    artifacts = _artifacts(report)
    assert isinstance(
        getattr(report, "artifacts", getattr(report, "input_artifacts", None)),
        tuple,
    )
    ordering = [(_field(item, "role"), _field(item, "artifact_id")) for item in artifacts]
    assert ordering == sorted(ordering)
    payload = report.to_dict()
    corpus_descriptor = _find_descriptor(payload, _sha(corpus_raw))
    preregistration_descriptor = _find_descriptor(payload, _sha(preregistration_raw))
    assert corpus_descriptor is not None
    assert corpus_descriptor["byte_count"] == len(corpus_raw)
    assert preregistration_descriptor is not None
    assert preregistration_descriptor["byte_count"] == len(preregistration_raw)


@pytest.mark.parametrize("kind", ["symlink", "hardlink", "nonfinite"])
def test_optional_opaque_artifacts_use_the_same_safe_json_boundary(
    executed_matrix: _ExecutedMatrix,
    tmp_path: Path,
    kind: str,
):
    source = tmp_path / "source.json"
    source.write_bytes(b'{"provider_measured":"caller claim"}\n')
    attacked = tmp_path / f"{kind}.json"
    if kind == "symlink":
        attacked.symlink_to(source)
    elif kind == "hardlink":
        os.link(source, attacked)
    else:
        attacked.write_bytes(b'{"provider_measured":NaN}\n')

    with pytest.raises(MeasuredEvidenceInputError):
        derive_measured_evidence_bundle(
            viability_manifest_path=executed_matrix.manifest_path,
            receipt_paths=executed_matrix.receipt_paths,
            corpus_report_path=attacked,
        )


def test_zero_admission_is_preserved_as_zero_not_dropped_or_inferred(
    executed_matrix: _ExecutedMatrix,
    tmp_path: Path,
):
    zero_cells = {
        point_id: _write_cell(
            tmp_path / f"genuine-zero-admission-{point_id}",
            case_number=1,
            point_id=point_id,
            threshold=threshold,
            still=True,
            case_id="independent-package-01",
        )
        for point_id, threshold in (("observe", 0.25), ("sparse", 0.75))
    }
    _, independently_derived = derive_viability_rows(zero_cells["observe"].receipt)
    assert independently_derived["candidate_windows_admitted"] == 0
    assert independently_derived["runtime_dispatches"] == 0
    assert independently_derived["judge_calls"] == 0

    paths = tuple(
        zero_cells[cell.point_id].receipt_path
        if cell.case_id == "independent-package-01"
        else cell.receipt_path
        for cell in executed_matrix.cells
    )
    report = _derive_bundle(executed_matrix, receipt_paths=paths)
    row = next(
        row
        for row in _rows(report)
        if (
            row["case_id"],
            row["operating_point"],
            row["arm"],
        )
        == ("independent-package-01", "observe", "adaptive")
    )
    assert row["candidate_windows_admitted"] == 0
    assert row["runtime_dispatches"] == 0
    assert row["judge_calls"] == 0
    assert row["decisions"] == {"fire": 0, "ask": 0, "silent": 0, "missing": 0}
    assert report.launch_qualified is False


def test_missing_and_failed_evidence_are_refused_not_silently_erased(
    executed_matrix: _ExecutedMatrix,
    tmp_path: Path,
):
    with pytest.raises(MeasuredEvidenceInputError):
        _derive_bundle(executed_matrix, receipt_paths=executed_matrix.receipt_paths[:-1])

    failed = _write_cell(
        tmp_path / "genuine-failed-provider-call",
        case_number=1,
        point_id="observe",
        threshold=0.25,
        fail_to_at=2,
        case_id="independent-package-01",
    )
    assert failed.receipt.adaptive.provider_call_failures == 1
    assert any(candidate.decision == "missing" for candidate in failed.receipt.adaptive.candidates)
    paths = tuple(
        failed.receipt_path if index == 0 else path
        for index, path in enumerate(executed_matrix.receipt_paths)
    )
    with pytest.raises(MeasuredEvidenceInputError):
        _derive_bundle(executed_matrix, receipt_paths=paths)
