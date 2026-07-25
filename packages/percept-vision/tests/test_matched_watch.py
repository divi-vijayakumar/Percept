"""L1 contract tests for the real compiled matched-Watch replay path."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
from dataclasses import replace
from pathlib import Path

import pytest

from percept.eval.matched_watch import (
    MatchedWatchInputError,
    derive_viability_rows,
    load_matched_watch_manifest,
    load_matched_watch_receipt,
    main,
    run_matched_watch,
    validate_matched_arms,
)
from percept.eval.measured_evidence import (
    MeasuredEvidenceInputError,
    derive_measured_evidence_bundle,
    write_measured_evidence_bundle,
)
import percept.eval.measured_evidence as measured_evidence_module
from percept.eval.viability_frontier import evaluate_viability_frontier
from percept.tracing import content_hash


def _sha(payload: bytes) -> str:
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _verdict(satisfied: bool, evidence: str, *, error: str | None = None) -> dict:
    return {
        "satisfied": satisfied,
        "confidence": 0.99,
        "evidence": evidence,
        "error": error,
    }


def _documents(
    *,
    fills=(0, 255, 0, 255),
    states: tuple[bool, ...] | None = None,
    fail_to_at: int | None = None,
):
    if states is not None:
        assert len(states) == len(fills)
    frames = []
    for index, fill in enumerate(fills):
        present = states[index] if states is not None else index >= 2
        frames.append(
            {
                "t": float(index),
                "fill": fill,
                "token": f"candidate-{index}",
                "from": _verdict(not present, "scripted: package absent"),
                "to": _verdict(
                    present,
                    "scripted: package visible",
                    error=("scripted timeout" if fail_to_at == index else None),
                ),
            }
        )
    media = {
        "schema": "percept.eval.matched_watch.synthetic_video/v1",
        "width": 8,
        "height": 8,
        "frames": frames,
    }
    return media


def _write_case(
    tmp_path,
    *,
    media=None,
    threshold=0.5,
    media_sha=None,
    event_start=2.0,
    event_end=3.0,
    case_id="physical-package-001",
    events=None,
    operating_point="block-max-0.5",
    source_id="camera",
):
    media_doc = media or _documents()
    media_bytes = json.dumps(media_doc, sort_keys=True, separators=(",", ":")).encode()
    media_path = tmp_path / "media.json"
    media_path.write_bytes(media_bytes)
    manifest = {
        "schema": "percept.eval.matched_watch/v1",
        "case_id": case_id,
        "archetype": "physical_scene/package_appeared",
        "split": "dev",
        "media": {
            "path": "media.json",
            "sha256": media_sha or _sha(media_bytes),
        },
        "replay": {"fps": 4.0, "timebase": "media", "source_id": source_id},
        "watch": {
            "template_ref": "facility_observation@1/package_appeared",
            "parameters": {"zone_name": "doorstep"},
            "domain_pack": "facility_observation@1",
        },
        "judge": {
            "adapter": "scripted_transition/v1",
            "model": "scripted-contract",
            "prompt_sha256": _sha(b"fixed transition prompt"),
        },
        "adaptive": {
            "operating_point": operating_point,
            "salience_stat": "block_max",
            "threshold": threshold,
        },
        "events": events or [
            {
                "id": f"{case_id}/e1",
                "eligible_at_s": 2.0,
                "window_start_s": event_start,
                "window_end_s": event_end,
                "review_authority": "fixture-authority/v1",
                "label_digest": _sha(b"package visible at t=2"),
            }
        ],
        "evidence_classification": "scripted_contract",
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path


def test_full_compiled_transition_path_records_dispatches_and_subcalls(tmp_path):
    receipt = run_matched_watch(_write_case(tmp_path))
    baseline, adaptive = derive_viability_rows(receipt)

    assert receipt.comparable is True
    assert receipt.launch_qualified is False
    assert (
        receipt.baseline.compiled_binding["watchspec_kwargs"]["stat_thresholds"]["block_max"]
        == 0.0
    )
    assert receipt.adaptive.compiled_binding["watchspec_kwargs"]["salience_stat"] == "block_max"
    assert (
        receipt.adaptive.compiled_binding["watchspec_kwargs"]["stat_thresholds"]["block_max"]
        == 0.5
    )
    assert baseline["candidate_windows_admitted"] == 4
    assert baseline["runtime_dispatches"] == 4
    assert baseline["judge_calls"] == 8
    assert adaptive["candidate_windows_admitted"] == 3
    assert adaptive["runtime_dispatches"] == 3
    assert adaptive["judge_calls"] == 6
    assert adaptive["decisions"] == {"fire": 1, "ask": 0, "silent": 2, "missing": 0}
    assert adaptive["fire_event_ids"] == ["physical-package-001/e1"]
    assert len(receipt.adaptive.incidents) == 1
    assert [item["stage"] for item in receipt.adaptive.traces].count("edge.decision") == 4
    assert [item["stage"] for item in receipt.adaptive.traces].count("transition.decision") == 3
    assert all(
        [call.predicate for call in candidate.provider_calls] == ["from", "to"]
        for candidate in receipt.adaptive.candidates
        if candidate.runtime_dispatched
    )


def test_sustained_true_fires_once_and_boundary_candidates_are_single_owned(tmp_path):
    receipt = run_matched_watch(_write_case(tmp_path, event_start=2.0, event_end=3.0))
    _, adaptive = derive_viability_rows(receipt)

    assert len(receipt.adaptive.incidents) == 1
    assert receipt.adaptive.incidents[0]["ts"] == 2.0
    assert adaptive["event_windows"] == [
        {
            "id": "physical-package-001/e1",
            "candidate_windows": 2,
            "admitted_candidates": 2,
            "outcome": "fire",
        }
    ]


def test_adaptive_zero_admission_and_miss_are_preserved(tmp_path):
    media = _documents(fills=(0, 0, 0, 0))
    receipt = run_matched_watch(_write_case(tmp_path, media=media, threshold=0.5))
    baseline, adaptive = derive_viability_rows(receipt)

    assert baseline["candidate_windows_admitted"] == 4
    assert adaptive["candidate_windows_admitted"] == 0
    assert adaptive["runtime_dispatches"] == 0
    assert adaptive["judge_calls"] == 0
    assert adaptive["decisions"] == {"fire": 0, "ask": 0, "silent": 0, "missing": 0}
    assert adaptive["event_windows"][0]["outcome"] == "silent"
    zero = next(
        item
        for item in receipt.adaptive.health
        if item["kind"] == "zero_admission_coverage"
    )
    assert zero["data"]["stat"] == "block_max"
    assert zero["data"]["threshold"] == 0.5
    assert zero["data"]["floor_unmet"] is False
    path = tmp_path / "zero-receipt.json"
    path.write_text(receipt.to_json(), encoding="utf-8")
    reloaded = load_matched_watch_receipt(path)
    assert reloaded.viability_rows[1]["candidate_windows_admitted"] == 0


def test_second_transition_subcall_failure_is_costed_and_never_partial_success(tmp_path):
    receipt = run_matched_watch(
        _write_case(tmp_path, media=_documents(fail_to_at=2))
    )
    candidate = receipt.adaptive.candidates[2]

    assert [call.status for call in candidate.provider_calls] == ["success", "error"]
    assert candidate.decision == "missing"
    assert candidate.incident_ids == ()
    assert receipt.adaptive.provider_call_attempts == 6
    assert receipt.adaptive.provider_call_failures == 1
    assert receipt.comparable is False
    with pytest.raises(MatchedWatchInputError, match="not comparable"):
        derive_viability_rows(receipt)


def test_comparability_rejects_timestamp_and_unapproved_config_mismatches(tmp_path):
    receipt = run_matched_watch(_write_case(tmp_path))
    changed_timestamps = replace(
        receipt.adaptive,
        candidate_timestamps=(0.0, 1.0, 2.0, 4.0),
    )
    assert "candidate timestamp sequences differ" in validate_matched_arms(
        receipt.baseline, changed_timestamps
    )

    changed_resolved = dict(receipt.adaptive.resolved_watch)
    changed_effective = dict(changed_resolved["effective"])
    changed_effective["Settings.gate.threshold"] = 0.123
    changed_resolved["effective"] = changed_effective
    changed_config = replace(receipt.adaptive, resolved_watch=changed_resolved)
    assert "arms differ outside approved admission configuration" in validate_matched_arms(
        receipt.baseline, changed_config
    )


def test_canonical_content_excludes_volatile_timestamp(tmp_path):
    manifest = _write_case(tmp_path)
    first = run_matched_watch(manifest)
    second = run_matched_watch(manifest)

    assert first.content_sha256 == second.content_sha256
    assert first.hash_content_dict() == second.hash_content_dict()
    assert "run_timestamp" not in first.content_dict()


def test_manifest_and_media_fail_closed_at_safe_input_boundary(tmp_path):
    missing = tmp_path / "missing.json"
    with pytest.raises(MatchedWatchInputError, match="unsafe matched-Watch manifest"):
        load_matched_watch_manifest(missing)

    manifest = _write_case(tmp_path, media_sha="sha256:" + "0" * 64)
    with pytest.raises(MatchedWatchInputError, match="media digest"):
        load_matched_watch_manifest(manifest)


def test_nonmonotonic_media_and_overlapping_event_ownership_fail_closed(tmp_path):
    media = _documents()
    media["frames"][2]["t"] = 1.0
    with pytest.raises(MatchedWatchInputError, match="strictly increasing"):
        load_matched_watch_manifest(_write_case(tmp_path, media=media))


def test_cli_output_is_create_exclusive_and_json_serializable(tmp_path):
    manifest = _write_case(tmp_path)
    output = tmp_path / "receipt.json"

    assert main(["--manifest", str(manifest), "--out", str(output)]) == 0
    assert json.loads(output.read_text(encoding="utf-8"))["launch_qualified"] is False
    with pytest.raises(SystemExit):
        main(["--manifest", str(manifest), "--out", str(output)])


def test_synthetic_dimensions_and_aggregate_allocation_are_bounded(tmp_path):
    fractional = _documents()
    fractional["width"] = 8.5
    with pytest.raises(MatchedWatchInputError, match="exact integer"):
        load_matched_watch_manifest(_write_case(tmp_path, media=fractional))

    oversized = _documents(fills=tuple(0 for _ in range(16)))
    oversized["width"] = 512
    oversized["height"] = 512
    with pytest.raises(MatchedWatchInputError, match="total-pixel"):
        load_matched_watch_manifest(_write_case(tmp_path, media=oversized))


def test_huge_numeric_input_is_normalized_to_contract_error(tmp_path):
    media = _documents()
    media["frames"][1]["t"] = 10**1000
    with pytest.raises(MatchedWatchInputError, match="finite number"):
        load_matched_watch_manifest(_write_case(tmp_path, media=media))


def test_derived_rows_are_accepted_by_a_matching_v3_frontier_manifest(tmp_path):
    policy = {
        "min_candidate_recall": 0.0,
        "min_event_recall": 0.0,
        "min_precision": 0.0,
        "min_call_reduction": 0.0,
        "max_event_abstention_rate": 1.0,
        "max_missing_rate": 1.0,
    }
    point_thresholds = {"block-max-0.25": 0.25, "block-max-0.75": 0.75}
    cases = []
    operating_points = {}
    rows = []
    for case_index in range(2):
        case_id = f"physical-package-{case_index + 1:03d}"
        events = [
            {
                "id": f"{case_id}/e1",
                "eligible_at_s": 2.0,
                "window_start_s": 2.0,
                "window_end_s": 2.0,
                "review_authority": "fixture-authority/v1",
                "label_digest": _sha(f"{case_id}: package visible at t=2".encode()),
            },
            {
                "id": f"{case_id}/e2",
                "eligible_at_s": 4.0,
                "window_start_s": 4.0,
                "window_end_s": 5.0,
                "review_authority": "fixture-authority/v1",
                "label_digest": _sha(f"{case_id}: package visible at t=4".encode()),
            },
        ]
        cases.append(
            {
                "id": case_id,
                "candidate_windows_total": 6,
                "events": [
                    {"id": events[0]["id"], "candidate_windows": 1},
                    {"id": events[1]["id"], "candidate_windows": 2},
                ],
            }
        )
        for point, threshold in point_thresholds.items():
            case_dir = tmp_path / f"{case_id}-{point}"
            case_dir.mkdir()
            receipt = run_matched_watch(
                    _write_case(
                        case_dir,
                        media=_documents(
                            fills=(0, 255, 0, 255, 0, 0),
                            states=(False, False, True, False, True, True),
                        ),
                    threshold=threshold,
                    case_id=case_id,
                    events=events,
                    operating_point=point,
                )
            )
            baseline, adaptive = derive_viability_rows(receipt)
            rows.extend((baseline, adaptive))
            if point not in operating_points:
                adaptive_arm = receipt.adaptive.to_dict()
                operating_points[point] = {
                    "id": point,
                    "configuration": {
                        "effective": adaptive_arm["resolved_watch"]["effective"],
                        "source_profile": adaptive_arm["source_profile"],
                        "compiled_binding": adaptive_arm["compiled_binding"],
                    },
                    "configuration_sha256": adaptive["configuration_sha256"],
                }
    archetypes = [
        {
            "id": "physical_scene/package_appeared",
            "provider_call_contract": {
                "completed_dispatch_calls": 2,
                "missing_dispatch_min_calls": 1,
                "missing_dispatch_max_calls": 2,
            },
            "cases": cases,
            "operating_points": list(operating_points.values()),
            "policy": policy,
        }
    ]
    report = evaluate_viability_frontier(
        {
            "schema_version": "3",
            "track": "viability_frontier",
            "scope": "programmable Watches over live multimodal streams",
            "evidence_kind": "synthetic",
            "preregistered": True,
            "archetypes": archetypes,
        },
        rows,
    )
    assert report.launch_qualified is False


def test_unlabelled_real_transition_is_preserved_as_false_positive_fire_id(tmp_path):
    receipt = run_matched_watch(
        _write_case(
            tmp_path,
            media=_documents(
                fills=(0, 255, 0, 255, 0, 0),
                states=(False, False, True, False, True, True),
            ),
            event_start=2.0,
            event_end=2.0,
        )
    )
    _, adaptive = derive_viability_rows(receipt)
    unaligned = next(
        str(incident["id"])
        for incident in receipt.adaptive.incidents
        if incident["ts"] == 4.0
    )

    assert adaptive["fire_event_ids"] == ["physical-package-001/e1", unaligned]


def test_coherent_nonzero_baseline_configuration_cannot_bypass_exhaustive_gate(tmp_path):
    receipt = run_matched_watch(_write_case(tmp_path))
    baseline = receipt.baseline.to_dict()
    profile = baseline["source_profile"]
    profile["parameters"]["salience_threshold"] = 0.25
    resolved = baseline["resolved_watch"]
    resolved["effective"]["Settings.edge.motion_gate_block_max"] = 0.25
    binding = baseline["compiled_binding"]
    binding["watchspec_kwargs"]["stat_thresholds"]["block_max"] = 0.25
    configuration = {
        "effective": resolved["effective"],
        "source_profile": profile,
        "compiled_binding": binding,
    }
    configuration_sha256 = f"sha256:{content_hash(configuration)}"
    coherent_baseline = replace(
        receipt.baseline,
        source_profile=profile,
        resolved_watch=resolved,
        compiled_binding=binding,
        configuration_sha256=configuration_sha256,
    )
    tampered = replace(
        receipt,
        baseline=coherent_baseline,
        comparable=True,
        comparability_blockers=(),
    )

    with pytest.raises(
        MatchedWatchInputError,
        match="baseline compiled admission threshold|cached receipt comparable",
    ):
        derive_viability_rows(tampered)


def test_evidence_origin_seal_binds_safe_input_descriptor_provenance(tmp_path):
    receipt = run_matched_watch(_write_case(tmp_path))
    manifest_input = dict(receipt.manifest_input_receipt)
    manifest_input.update(
        path="/forged/manifest.json",
        canonical_path="/forged/manifest.json",
        device=int(manifest_input["device"]) + 1,
        inode=int(manifest_input["inode"]) + 1,
        modified_time_ns=int(manifest_input["modified_time_ns"]) + 1,
    )
    forged = replace(receipt, manifest_input_receipt=manifest_input)

    assert forged.content_sha256 == receipt.content_sha256
    assert forged.evidence_sha256 != receipt.evidence_sha256
    with pytest.raises(MatchedWatchInputError, match="origin seal"):
        derive_viability_rows(forged)


def test_runtime_started_receipt_identity_is_compact_and_manifest_bound(tmp_path):
    receipt = run_matched_watch(_write_case(tmp_path))
    arm = receipt.adaptive.to_dict()
    data = next(item["data"] for item in arm["health"] if item["kind"] == "runtime_started")

    assert data["source_kind"] == "synthetic_video"
    assert data["candidate_count"] == len(receipt.adaptive.candidates)
    assert data["media"] == arm["media_identity"]
    assert data["replay"] == arm["replay_identity"]
    encoded = json.dumps(arm["health"], sort_keys=True)
    assert len(json.dumps(data, sort_keys=True)) < 1_024
    assert "pixels=" not in encoded
    assert "_JudgePayload" not in encoded


def test_evidence_origin_seal_binds_runtime_started_and_last_wall_time(tmp_path):
    receipt = run_matched_watch(_write_case(tmp_path))
    health = list(receipt.adaptive.health)
    started_index = next(
        index for index, item in enumerate(health) if item["kind"] == "runtime_started"
    )
    changed_started = dict(health[started_index])
    changed_data = dict(changed_started["data"])
    changed_data["candidate_count"] = int(changed_data["candidate_count"]) + 1
    changed_started["data"] = changed_data
    health[started_index] = changed_started
    changed_health_receipt = replace(
        receipt,
        adaptive=replace(receipt.adaptive, health=tuple(health)),
    )
    stats = dict(receipt.adaptive.runtime_stats)
    stats["last_wall_t"] = float(stats["last_wall_t"]) + 1.0
    changed_wall_receipt = replace(
        receipt,
        adaptive=replace(receipt.adaptive, runtime_stats=stats),
    )

    for forged in (changed_health_receipt, changed_wall_receipt):
        assert forged.content_sha256 == receipt.content_sha256
        assert forged.evidence_sha256 != receipt.evidence_sha256
        with pytest.raises(MatchedWatchInputError):
            derive_viability_rows(forged)


def test_evidence_origin_seal_binds_run_timestamp(tmp_path):
    receipt = run_matched_watch(_write_case(tmp_path))
    forged = replace(receipt, run_timestamp="2099-01-01T00:00:00+00:00")

    assert forged.content_sha256 == receipt.content_sha256
    assert forged.evidence_sha256 != receipt.evidence_sha256
    with pytest.raises(MatchedWatchInputError, match="origin seal"):
        derive_viability_rows(forged)


def test_none_origin_cannot_reseal_forged_build_provenance(tmp_path):
    receipt = run_matched_watch(_write_case(tmp_path))
    provenance = dict(receipt.build_provenance)
    provenance["git_commit"] = "forged"
    forged = replace(
        receipt,
        build_provenance=provenance,
        origin_evidence_sha256=None,
    )

    assert forged.origin_evidence_sha256 is None
    with pytest.raises(MatchedWatchInputError, match="build provenance|origin seal"):
        derive_viability_rows(forged)


def test_none_origin_cannot_reseal_cached_validity_or_comparability_mismatch(tmp_path):
    receipt = run_matched_watch(_write_case(tmp_path))
    invalid_arm = replace(
        receipt,
        adaptive=replace(receipt.adaptive, valid=False),
        origin_evidence_sha256=None,
    )
    stale_comparability = replace(
        receipt,
        comparable=False,
        comparability_blockers=(),
        origin_evidence_sha256=None,
    )

    assert invalid_arm.origin_evidence_sha256 is None
    assert stale_comparability.origin_evidence_sha256 is None
    with pytest.raises(MatchedWatchInputError, match="cached validity"):
        derive_viability_rows(invalid_arm)
    with pytest.raises(MatchedWatchInputError, match="cached receipt comparable"):
        derive_viability_rows(stale_comparability)


def test_persisted_receipt_reload_preserves_raw_snapshot_and_frozen_dtos(tmp_path):
    receipt = run_matched_watch(_write_case(tmp_path))
    path = tmp_path / "receipt.json"
    path.write_text(receipt.to_json(), encoding="utf-8")

    loaded = load_matched_watch_receipt(path)

    assert loaded.receipt.to_dict() == receipt.to_dict()
    assert loaded.input_receipt.sha256 == _sha(path.read_bytes())
    assert loaded.input_receipt.byte_count == len(path.read_bytes())
    expected_rows = derive_viability_rows(receipt)
    assert [row["case_id"] for row in loaded.viability_rows] == [
        row["case_id"] for row in expected_rows
    ]
    assert [row["judge_calls"] for row in loaded.viability_rows] == [
        row["judge_calls"] for row in expected_rows
    ]
    with pytest.raises(TypeError):
        loaded.receipt.manifest["case_id"] = "forged"
    with pytest.raises(TypeError):
        loaded.viability_rows[0]["case_id"] = "forged"


def test_persisted_receipt_rejects_extra_fields_and_cached_hash_drift(tmp_path):
    receipt = run_matched_watch(_write_case(tmp_path))
    document = receipt.to_dict()
    document["baseline"]["candidates"][0]["provider_calls"].append({"extra": True})
    path = tmp_path / "receipt.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(MatchedWatchInputError):
        load_matched_watch_receipt(path)

    document = receipt.to_dict()
    document["content_sha256"] = "sha256:" + "0" * 64
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(MatchedWatchInputError, match="cached content hash"):
        load_matched_watch_receipt(path)

    for mutation in ("extra", "missing", "wrong_type"):
        document = receipt.to_dict()
        if mutation == "extra":
            document["watch_plan"]["caller_field"] = "ignored"
        elif mutation == "missing":
            document["watch_plan"].pop("condition")
        else:
            document["watch_plan"]["detectors"] = "not-a-list"
        path.write_text(json.dumps(document), encoding="utf-8")
        with pytest.raises(MatchedWatchInputError, match="watch_plan"):
            load_matched_watch_receipt(path)


def test_persisted_receipt_rejects_noncanonical_resolved_provenance(tmp_path):
    receipt = run_matched_watch(_write_case(tmp_path))
    mutations = ("extra_only", "mirrored_unknown", "missing", "wrong_type", "value_drift")
    for mutation in mutations:
        baseline = receipt.baseline.to_dict()
        resolved = baseline["resolved_watch"]
        provenance = resolved["provenance"]
        effective = resolved["effective"]
        known_path = next(iter(provenance))
        if mutation == "extra_only":
            provenance["oracle_unsupported_nested_key"] = {"caller": True}
        elif mutation == "mirrored_unknown":
            effective["oracle_unsupported_nested_key"] = True
            provenance["oracle_unsupported_nested_key"] = {
                "value": True,
                "owner_layer": "SDK",
                "source_id": "caller",
                "reason": "caller",
                "validated_by": None,
            }
        elif mutation == "missing":
            provenance[known_path].pop("reason")
        elif mutation == "wrong_type":
            provenance[known_path]["owner_layer"] = True
        else:
            provenance[known_path]["value"] = "caller-drift"
        forged_arm = replace(receipt.baseline, resolved_watch=resolved)
        forged = replace(
            receipt,
            baseline=forged_arm,
            origin_evidence_sha256="",
        )
        sealed = replace(forged, origin_evidence_sha256=forged.evidence_sha256)
        path = tmp_path / f"resolved-{mutation}.json"
        path.write_text(sealed.to_json(), encoding="utf-8")

        with pytest.raises(MatchedWatchInputError, match="resolved_watch|provenance"):
            load_matched_watch_receipt(path)


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        ("source_parameters", "source_profile.parameters"),
        ("compiled_kwargs", "compiled_binding.watchspec_kwargs"),
        ("compiled_components", "compiled_binding.components"),
        ("apply_receipt", "apply_receipt"),
        ("domain_values", "domain_pack.values"),
        ("resolved_providers", "components.providers"),
    ],
)
def test_persisted_receipt_rejects_extra_fields_in_closed_nested_maps(
    tmp_path,
    mutation,
    expected,
):
    case_dir = tmp_path / mutation
    case_dir.mkdir()
    receipt = run_matched_watch(_write_case(case_dir))
    arm = receipt.baseline.to_dict()
    if mutation == "source_parameters":
        arm["source_profile"]["parameters"]["oracle_extra"] = True
    elif mutation == "compiled_kwargs":
        arm["compiled_binding"]["watchspec_kwargs"]["oracle_extra"] = True
    elif mutation == "compiled_components":
        arm["compiled_binding"]["components"]["oracle_extra"] = None
    elif mutation == "apply_receipt":
        arm["apply_receipt"]["oracle_extra"] = True
    elif mutation == "domain_values":
        arm["resolved_watch"]["domain_pack"]["values"]["oracle_extra"] = True
    else:
        arm["resolved_watch"]["components"]["providers"]["oracle_extra"] = None
    forged = replace(
        receipt,
        baseline=replace(
            receipt.baseline,
            source_profile=arm["source_profile"],
            resolved_watch=arm["resolved_watch"],
            apply_receipt=arm["apply_receipt"],
            compiled_binding=arm["compiled_binding"],
        ),
        origin_evidence_sha256="",
    )
    sealed = replace(forged, origin_evidence_sha256=forged.evidence_sha256)
    path = tmp_path / f"closed-map-{mutation}.json"
    path.write_text(sealed.to_json(), encoding="utf-8")

    with pytest.raises(MatchedWatchInputError, match=expected):
        load_matched_watch_receipt(path)


def _write_measured_bundle(tmp_path, *, duplicate_source_cases=False):
    tmp_path.mkdir(parents=True, exist_ok=True)
    policy = {
        "min_candidate_recall": 0.0,
        "min_event_recall": 0.0,
        "min_precision": 0.0,
        "min_call_reduction": 0.0,
        "max_event_abstention_rate": 1.0,
        "max_missing_rate": 1.0,
    }
    cases = []
    operating_points = {}
    receipt_paths = []
    for case_index in range(2):
        case_id = f"physical-package-{case_index + 1:03d}"
        events = [
            {
                "id": f"{case_id}/e1",
                "eligible_at_s": 2.0,
                "window_start_s": 2.0,
                "window_end_s": 2.0,
                "review_authority": "caller-declared-review/v1",
                "label_digest": _sha(f"{case_id}/e1".encode()),
            },
            {
                "id": f"{case_id}/e2",
                "eligible_at_s": 4.0,
                "window_start_s": 4.0,
                "window_end_s": 5.0,
                "review_authority": "caller-declared-review/v1",
                "label_digest": _sha(f"{case_id}/e2".encode()),
            },
        ]
        cases.append(
            {
                "id": case_id,
                "candidate_windows_total": 6,
                "events": [
                    {"id": events[0]["id"], "candidate_windows": 1},
                    {"id": events[1]["id"], "candidate_windows": 2},
                ],
            }
        )
        for point, threshold in (("block-max-0.25", 0.25), ("block-max-0.75", 0.75)):
            case_dir = tmp_path / f"{case_id}-{point}"
            case_dir.mkdir()
            case_fill = 0 if duplicate_source_cases else case_index
            manifest = _write_case(
                case_dir,
                media=_documents(
                    fills=(
                        case_fill,
                        255 - case_fill,
                        case_fill,
                        255 - case_fill,
                        case_fill,
                        case_fill,
                    ),
                    states=(False, False, True, False, True, True),
                ),
                threshold=threshold,
                case_id=case_id,
                events=events,
                operating_point=point,
                source_id="camera",
            )
            receipt = run_matched_watch(manifest)
            receipt_path = case_dir / "receipt.json"
            receipt_path.write_text(receipt.to_json(), encoding="utf-8")
            receipt_paths.append(receipt_path)
            if point not in operating_points:
                adaptive = receipt.adaptive.to_dict()
                configuration = {
                    "effective": adaptive["resolved_watch"]["effective"],
                    "source_profile": adaptive["source_profile"],
                    "compiled_binding": adaptive["compiled_binding"],
                }
                operating_points[point] = {
                    "id": point,
                    "configuration": configuration,
                    "configuration_sha256": f"sha256:{content_hash(configuration)}",
                }
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
                "operating_points": list(operating_points.values()),
                "policy": policy,
            }
        ],
    }
    manifest_path = tmp_path / "viability.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path, receipt_paths


def test_measured_bundle_derives_exact_rows_and_keeps_authority_fail_closed(tmp_path):
    manifest_path, receipt_paths = _write_measured_bundle(tmp_path)

    report = derive_measured_evidence_bundle(
        viability_manifest_path=manifest_path,
        receipt_paths=list(reversed(receipt_paths)),
    )

    assert len(report.derived_rows) == 8
    assert len(report.coverage) == 4
    assert report.authority_verified is False
    assert report.launch_qualified is False
    assert report.frontier.launch_qualified is False
    assert "independent label-review authority is unavailable" in report.blockers
    assert all(row["evidence_kind"] == "synthetic" for row in report.rows)
    assert json.loads(report.to_json())["launch_qualified"] is False
    with pytest.raises(TypeError):
        replace(report, authority_verified=True)
    with pytest.raises(TypeError):
        replace(report, launch_qualified=True)
    with pytest.raises(TypeError):
        replace(report, method="caller-approved")
    with pytest.raises(MeasuredEvidenceInputError):
        replace(report, artifacts=())
    with pytest.raises(MeasuredEvidenceInputError):
        replace(report, artifacts=tuple(reversed(report.artifacts)))
    with pytest.raises(MeasuredEvidenceInputError):
        replace(report, coverage=())
    with pytest.raises(MeasuredEvidenceInputError):
        replace(report, coverage=tuple(reversed(report.coverage)))
    with pytest.raises(MeasuredEvidenceInputError):
        replace(report, derived_rows=(), derived_row_sha256=())
    with pytest.raises(MeasuredEvidenceInputError):
        replace(report, derived_rows=tuple(reversed(report.derived_rows)))
    with pytest.raises(MeasuredEvidenceInputError):
        replace(
            report,
            derived_row_sha256=(
                "sha256:" + "0" * 64,
                *report.derived_row_sha256[1:],
            ),
        )
    with pytest.raises(MeasuredEvidenceInputError):
        replace(report, receipt_evidence_seals=())
    with pytest.raises(MeasuredEvidenceInputError):
        replace(report, receipt_evidence_seals=tuple(reversed(report.receipt_evidence_seals)))
    changed_manifest = report.content_dict()["viability_manifest"]
    changed_manifest["scope"] = "caller-expanded scope"
    with pytest.raises(MeasuredEvidenceInputError):
        replace(report, viability_manifest=changed_manifest)
    with pytest.raises(MeasuredEvidenceInputError):
        replace(report, build_provenance={"package_version": "caller-only"})
    changed_judge = dict(report.judge_identity)
    changed_judge["prompt_sha256"] = "sha256:" + "0" * 64
    # Structurally valid caller changes remain blocked by the descriptor re-derivation at write.
    structurally_valid_tamper = replace(report, judge_identity=changed_judge)
    with pytest.raises(MeasuredEvidenceInputError, match="exactly matches"):
        write_measured_evidence_bundle(
            structurally_valid_tamper,
            tmp_path / "tampered-report.json",
        )
    changed_coverage = replace(
        report.coverage[0],
        configuration_sha256="sha256:" + "0" * 64,
    )
    with pytest.raises(MeasuredEvidenceInputError):
        replace(report, coverage=(changed_coverage, *report.coverage[1:]))
    forged_frontier = replace(
        report.frontier,
        qualification_reason="caller-approved",
    )
    with pytest.raises(MeasuredEvidenceInputError):
        replace(report, frontier=forged_frontier)
    output = tmp_path / "measured-evidence.json"
    write_measured_evidence_bundle(report, output)
    original = output.read_bytes()
    output_info = os.lstat(output)
    assert stat.S_ISREG(output_info.st_mode)
    assert output_info.st_nlink == 1
    with pytest.raises(MeasuredEvidenceInputError, match="already exists"):
        write_measured_evidence_bundle(report, output)
    assert output.read_bytes() == original


def test_measured_bundle_rejects_relabelled_duplicate_underlying_cases(tmp_path):
    manifest_path, receipt_paths = _write_measured_bundle(
        tmp_path,
        duplicate_source_cases=True,
    )

    with pytest.raises(MeasuredEvidenceInputError, match="duplicate underlying case evidence"):
        derive_measured_evidence_bundle(
            viability_manifest_path=manifest_path,
            receipt_paths=receipt_paths,
        )


def test_writer_reloads_descriptor_backed_inputs_before_creation(tmp_path):
    manifest_path, receipt_paths = _write_measured_bundle(tmp_path / "source")
    report = derive_measured_evidence_bundle(
        viability_manifest_path=manifest_path,
        receipt_paths=receipt_paths,
    )
    receipt_paths[0].write_text(
        receipt_paths[0].read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "must-not-exist.json"

    with pytest.raises(MeasuredEvidenceInputError, match="exactly matches"):
        write_measured_evidence_bundle(report, output)
    assert not output.exists()


@pytest.mark.parametrize("failure", ["partial_write", "fsync", "directory_fsync", "link"])
def test_writer_failures_leave_no_partial_destination(tmp_path, monkeypatch, failure):
    manifest_path, receipt_paths = _write_measured_bundle(tmp_path / "source")
    report = derive_measured_evidence_bundle(
        viability_manifest_path=manifest_path,
        receipt_paths=receipt_paths,
    )
    output = tmp_path / f"failed-{failure}.json"
    if failure == "partial_write":
        real_write = os.write
        calls = 0

        def fail_after_partial(fd, payload):
            nonlocal calls
            calls += 1
            if calls == 1:
                return real_write(fd, payload[:37])
            raise OSError("simulated partial stream failure")

        monkeypatch.setattr(measured_evidence_module.os, "write", fail_after_partial)
    elif failure == "fsync":
        monkeypatch.setattr(
            measured_evidence_module.os,
            "fsync",
            lambda _fd: (_ for _ in ()).throw(OSError("simulated fsync failure")),
        )
    elif failure == "directory_fsync":
        real_fsync = os.fsync
        calls = 0

        def fail_directory_fsync(fd):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError("simulated directory fsync failure")
            return real_fsync(fd)

        monkeypatch.setattr(measured_evidence_module.os, "fsync", fail_directory_fsync)
    else:
        monkeypatch.setattr(
            measured_evidence_module.os,
            "link",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                OSError("simulated link failure")
            ),
        )

    with pytest.raises(MeasuredEvidenceInputError, match="atomically"):
        write_measured_evidence_bundle(report, output)
    assert not output.exists()
    assert not list(tmp_path.glob(f".{output.name}.*.tmp"))


def test_writer_path_swap_cleanup_preserves_external_target(tmp_path, monkeypatch):
    manifest_path, receipt_paths = _write_measured_bundle(tmp_path / "source")
    report = derive_measured_evidence_bundle(
        viability_manifest_path=manifest_path,
        receipt_paths=receipt_paths,
    )
    output = tmp_path / "externally-swapped.json"
    real_link = os.link

    def swap_after_link(source, destination, **kwargs):
        real_link(source, destination, **kwargs)
        os.unlink(destination)
        destination_path = Path(destination)
        destination_path.write_bytes(b"external replacement")

    monkeypatch.setattr(measured_evidence_module.os, "link", swap_after_link)

    with pytest.raises(MeasuredEvidenceInputError, match="identity changed"):
        write_measured_evidence_bundle(report, output)
    assert output.read_bytes() == b"external replacement"
    assert not list(tmp_path.glob(f".{output.name}.*.tmp"))


def test_writer_rejects_symlink_and_non_regular_targets(tmp_path):
    manifest_path, receipt_paths = _write_measured_bundle(tmp_path / "source")
    report = derive_measured_evidence_bundle(
        viability_manifest_path=manifest_path,
        receipt_paths=receipt_paths,
    )
    existing = tmp_path / "existing.json"
    existing.write_text("untouched", encoding="utf-8")
    symlink = tmp_path / "output-link.json"
    symlink.symlink_to(existing)
    directory = tmp_path / "output-directory"
    directory.mkdir()

    with pytest.raises(MeasuredEvidenceInputError, match="symlink"):
        write_measured_evidence_bundle(report, symlink)
    with pytest.raises(MeasuredEvidenceInputError, match="not a regular file"):
        write_measured_evidence_bundle(report, directory)
    assert existing.read_text(encoding="utf-8") == "untouched"


def test_measured_bundle_is_path_neutral_but_evidence_binds_descriptors(tmp_path):
    manifest_path, receipt_paths = _write_measured_bundle(tmp_path / "source")
    first = derive_measured_evidence_bundle(
        viability_manifest_path=manifest_path,
        receipt_paths=receipt_paths,
    )
    copied = tmp_path / "copied"
    copied.mkdir()
    copied_manifest = copied / "viability.json"
    shutil.copyfile(manifest_path, copied_manifest)
    copied_receipts = []
    for index, source in enumerate(receipt_paths):
        destination = copied / f"receipt-{index}.json"
        shutil.copyfile(source, destination)
        copied_receipts.append(destination)
    second = derive_measured_evidence_bundle(
        viability_manifest_path=copied_manifest,
        receipt_paths=copied_receipts,
    )

    assert first.content_sha256 == second.content_sha256
    assert first.evidence_sha256 != second.evidence_sha256
    rerun_manifest, rerun_receipts = _write_measured_bundle(tmp_path / "rerun")
    rerun = derive_measured_evidence_bundle(
        viability_manifest_path=rerun_manifest,
        receipt_paths=list(reversed(rerun_receipts)),
    )
    assert first.content_sha256 == rerun.content_sha256
    assert first.evidence_sha256 != rerun.evidence_sha256
    with pytest.raises(MeasuredEvidenceInputError, match="duplicate|reused"):
        derive_measured_evidence_bundle(
            viability_manifest_path=manifest_path,
            receipt_paths=[*receipt_paths, receipt_paths[0]],
        )
