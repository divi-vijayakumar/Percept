from __future__ import annotations

import json
from dataclasses import replace

import pytest

from percept.eval.matched_watch import (
    MatchedWatchInputError,
    derive_viability_rows,
    load_matched_watch_manifest,
    run_matched_watch,
    validate_matched_arms,
)
from tests.test_matched_watch import _write_case


def test_receipt_json_is_serializable(tmp_path):
    receipt = run_matched_watch(_write_case(tmp_path))

    payload = json.loads(receipt.to_json())

    assert payload["content_sha256"] == receipt.content_sha256
    assert payload["launch_qualified"] is False


def test_raw_evidence_hash_binds_traces_health_and_correlation_ids(tmp_path):
    receipt = run_matched_watch(_write_case(tmp_path))
    baseline_without_traces = replace(receipt.baseline, traces=())
    no_traces = replace(receipt, baseline=baseline_without_traces)
    adaptive_without_health = replace(receipt.adaptive, health=())
    no_health = replace(receipt, adaptive=adaptive_without_health)
    fire_candidate = next(
        candidate for candidate in receipt.adaptive.candidates if candidate.incident_ids
    )
    changed_candidate = replace(fire_candidate, incident_ids=("different-incident",))
    changed_candidates = tuple(
        changed_candidate if candidate.index == fire_candidate.index else candidate
        for candidate in receipt.adaptive.candidates
    )
    changed_links = replace(
        receipt,
        adaptive=replace(receipt.adaptive, candidates=changed_candidates),
    )

    assert no_traces.evidence_sha256 != receipt.evidence_sha256
    assert no_health.evidence_sha256 != receipt.evidence_sha256
    assert changed_links.evidence_sha256 != receipt.evidence_sha256


def test_zero_candidate_event_window_fails_before_row_derivation(tmp_path):
    manifest_path = _write_case(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["events"][0].update(
        {
            "eligible_at_s": 10.0,
            "window_start_s": 10.0,
            "window_end_s": 11.0,
        }
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(MatchedWatchInputError, match="candidate"):
        load_matched_watch_manifest(manifest_path)


def test_unselected_threshold_drift_is_not_an_allowed_arm_difference(tmp_path):
    receipt = run_matched_watch(_write_case(tmp_path))
    resolved = dict(receipt.adaptive.resolved_watch)
    effective = dict(resolved["effective"])
    effective["Settings.edge.motion_gate_p95"] = 0.987
    resolved["effective"] = effective
    binding = dict(receipt.adaptive.compiled_binding)
    kwargs = dict(binding["watchspec_kwargs"])
    thresholds = dict(kwargs["stat_thresholds"])
    thresholds["p95"] = 0.987
    kwargs["stat_thresholds"] = thresholds
    binding["watchspec_kwargs"] = kwargs
    adaptive = replace(
        receipt.adaptive,
        resolved_watch=resolved,
        compiled_binding=binding,
    )

    assert validate_matched_arms(receipt.baseline, adaptive)


def test_derivation_rejects_trace_and_terminal_health_deletion(tmp_path):
    receipt = run_matched_watch(_write_case(tmp_path))
    no_traces = replace(
        receipt,
        adaptive=replace(receipt.adaptive, traces=()),
        comparable=True,
        comparability_blockers=(),
    )
    health = tuple(
        item for item in receipt.adaptive.health if item["kind"] != "runtime_ended"
    )
    no_terminal = replace(
        receipt,
        adaptive=replace(receipt.adaptive, health=health),
        comparable=True,
        comparability_blockers=(),
    )

    with pytest.raises(MatchedWatchInputError, match="trace"):
        derive_viability_rows(no_traces)
    with pytest.raises(MatchedWatchInputError, match="runtime_ended|terminal"):
        derive_viability_rows(no_terminal)


def test_derivation_requires_bijective_candidate_incident_links(tmp_path):
    receipt = run_matched_watch(_write_case(tmp_path))
    fire_candidate = next(
        candidate for candidate in receipt.adaptive.candidates if candidate.incident_ids
    )
    silent_candidate = next(
        candidate
        for candidate in receipt.adaptive.candidates
        if candidate.runtime_dispatched and not candidate.incident_ids
    )
    duplicate = replace(
        silent_candidate,
        decision="fire",
        incident_ids=fire_candidate.incident_ids,
    )
    candidates = tuple(
        duplicate if candidate.index == silent_candidate.index else candidate
        for candidate in receipt.adaptive.candidates
    )
    adaptive = replace(receipt.adaptive, candidates=candidates)
    tampered = replace(
        receipt,
        adaptive=adaptive,
        comparable=True,
        comparability_blockers=(),
    )

    with pytest.raises(MatchedWatchInputError, match="incident|FIRE|bijective"):
        derive_viability_rows(tampered)


def test_derivation_binds_manifest_replay_to_each_arm(tmp_path):
    receipt = run_matched_watch(_write_case(tmp_path))
    manifest = dict(receipt.manifest)
    replay = dict(manifest["replay"])
    replay["fps"] = 999.0
    replay["source_id"] = "different-source"
    manifest["replay"] = replay
    tampered = replace(receipt, manifest=manifest)

    with pytest.raises(MatchedWatchInputError, match="replay|source|fps"):
        derive_viability_rows(tampered)
